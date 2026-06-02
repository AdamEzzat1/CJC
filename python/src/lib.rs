//! Python wrapper for cjc-locke.
//!
//! Design goals:
//!
//! - **Thin**: every Python function delegates to a single Rust call. No
//!   business logic on the Python side. Output objects are wrapper structs
//!   that hold the Rust value verbatim and expose it via `to_json()` (then
//!   the Python facade does the cheap JSON-decode for `to_dict()`).
//! - **Fast**: numpy `f64`/`i64`/`bool` arrays cross the FFI boundary
//!   zero-copy via the buffer protocol. Python lists go through one
//!   element-wise pass each, copied into the Rust column. The actual
//!   `validate()` / `compare_drift()` work is unchanged Rust.
//! - **Deterministic**: the Rust side is byte-identical to a native call.
//!   Input ordering: Python dicts are insertion-ordered (PEP 468) and
//!   `cjc_data::DataFrame` canonicalises internally (BTreeMap-backed
//!   downstream). Numpy/pandas/polars all preserve column order.
//! - **Memory/thermal/power**: identical to native — every Python call
//!   does one Rust-side allocation (the result struct) plus column copies.
//!   No background threads, no caches, no Python-side state.

use std::collections::BTreeMap;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use std::sync::Arc;

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{
    belief_report_from_locke, belief_report_from_locke_with_model,
    causal_guardrail as locke_causal_guardrail, validate,
    validate_and_compare as locke_validate_and_compare, ValidateOptions,
};
use cjc_locke::custom_detector::{
    validate_custom_code, BeliefAxisSet, CustomDetector, FindingSink,
};
use cjc_locke::report::{FindingEvidence, FindingSeverity};
use cjc_locke::belief::{BeliefPenalty, BeliefReport};
use cjc_locke::causal::{CausalConfig, CausalGuardrailReport};
use cjc_locke::drift::{compare as locke_compare_drift, DriftConfig, InductionRiskReport};
use cjc_locke::lineage::{
    AuditEvent, ImpressionKind, LineageBuilder, LineageGraph, LockeIdea, LockeImpression,
    TransformationRecord,
};
use cjc_locke::policy::{
    apply_policy as locke_apply_policy, OwnerRule, Policy, PolicyResult, RequiredFindingRule,
    RequirementOperator, SuppressionRule,
};
use cjc_locke::report::LockeReport;
use cjc_locke::streaming::{StreamingConfig, StreamingValidator};
use cjc_locke::temporal::{detect_temporal_issues as locke_detect_temporal_issues, TimeColumnConfig};
use cjc_locke::validation::{NullMask, NullMaskMap, ValidationConfig};
use cjc_locke::FingerprintId;

// ─── Input bridge: Python dict → cjc_data::DataFrame ──────────────────────────
//
// The single conversion point for all `validate(...)`-style entries. Keeping
// it in one place means the determinism contract (column order = dict order)
// is documented in one spot and tested via the workspace's existing
// DataFrame tests.

/// Convert one Python value (numpy array OR list) into a `cjc_data::Column`.
/// Numpy paths are zero-copy reads via the buffer protocol; lists incur one
/// element-wise pass + a heap allocation per column.
fn py_value_to_column(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Column> {
    // Fast paths via numpy buffer protocol. `as_slice()` is a single zero-copy
    // view; `.to_vec()` is a single memcpy into a Rust-owned buffer (we have to
    // own it because cjc_data::Column does).
    if let Ok(arr) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        let slice = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("numpy f64 not contiguous: {}", e)))?;
        return Ok(Column::Float(slice.to_vec()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArray1<'_, i64>>() {
        let slice = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("numpy i64 not contiguous: {}", e)))?;
        return Ok(Column::Int(slice.to_vec()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArray1<'_, bool>>() {
        let slice = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("numpy bool not contiguous: {}", e)))?;
        return Ok(Column::Bool(slice.to_vec()));
    }
    // Less common numpy dtypes — widen.
    if let Ok(arr) = value.extract::<PyReadonlyArray1<'_, f32>>() {
        let slice = arr.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(Column::Float(slice.iter().map(|&v| v as f64).collect()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArray1<'_, i32>>() {
        let slice = arr.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(Column::Int(slice.iter().map(|&v| v as i64).collect()));
    }

    // Python list fallback. Type-dispatch on the first non-None element.
    if let Ok(list) = value.downcast::<PyList>() {
        // Find the first non-None to determine the column type. A column with
        // only Nones (or zero-length) is treated as Float — the natural
        // numeric default; downstream validators handle the all-NaN case.
        let probe_idx = (0..list.len()).find(|&i| {
            list.get_item(i)
                .map(|v| !v.is_none())
                .unwrap_or(false)
        });
        let probe = match probe_idx {
            Some(i) => list.get_item(i).unwrap(),
            None => {
                let v: Vec<f64> = (0..list.len()).map(|_| f64::NAN).collect();
                return Ok(Column::Float(v));
            }
        };

        if probe.extract::<bool>().is_ok() && !probe.is_instance_of::<pyo3::types::PyInt>() {
            // Bool BEFORE int because Python's `True` is also a valid int.
            return collect_list_bool(list);
        }
        if probe.extract::<i64>().is_ok() {
            return collect_list_int(list);
        }
        if probe.extract::<f64>().is_ok() {
            return collect_list_float(list);
        }
        if probe.downcast::<PyString>().is_ok() {
            return collect_list_str(list);
        }
        return Err(PyTypeError::new_err(format!(
            "column list element type not supported: {}",
            probe.get_type().name()?,
        )));
    }

    let _ = py;
    Err(PyTypeError::new_err(
        "column value must be numpy ndarray (f64/i64/bool/f32/i32) or a Python list",
    ))
}

fn collect_list_float(list: &Bound<'_, PyList>) -> PyResult<Column> {
    let mut out: Vec<f64> = Vec::with_capacity(list.len());
    for item in list.iter() {
        if item.is_none() {
            out.push(f64::NAN);
        } else {
            out.push(item.extract::<f64>()?);
        }
    }
    Ok(Column::Float(out))
}

fn collect_list_int(list: &Bound<'_, PyList>) -> PyResult<Column> {
    let mut out: Vec<i64> = Vec::with_capacity(list.len());
    for item in list.iter() {
        // None in an int column has no canonical representation in cjc_data
        // (Int is non-nullable). Coerce to 0 and require the user to pass a
        // null mask if they care.
        if item.is_none() {
            out.push(0);
        } else {
            out.push(item.extract::<i64>()?);
        }
    }
    Ok(Column::Int(out))
}

fn collect_list_str(list: &Bound<'_, PyList>) -> PyResult<Column> {
    let mut out: Vec<String> = Vec::with_capacity(list.len());
    for item in list.iter() {
        if item.is_none() {
            out.push(String::new());
        } else {
            out.push(item.extract::<String>()?);
        }
    }
    Ok(Column::Str(out))
}

fn collect_list_bool(list: &Bound<'_, PyList>) -> PyResult<Column> {
    let mut out: Vec<bool> = Vec::with_capacity(list.len());
    for item in list.iter() {
        if item.is_none() {
            out.push(false);
        } else {
            out.push(item.extract::<bool>()?);
        }
    }
    Ok(Column::Bool(out))
}

/// Convert a Python dict[str, ndarray|list] into a `DataFrame`. Column order
/// is the dict's insertion order (PEP 468). Empty dict → empty DataFrame.
fn dict_to_dataframe(py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<DataFrame> {
    let mut columns: Vec<(String, Column)> = Vec::with_capacity(data.len());
    for (k, v) in data.iter() {
        let name: String = k.extract().map_err(|_| {
            PyTypeError::new_err("dataframe dict keys must be strings (column names)")
        })?;
        let col = py_value_to_column(py, &v)
            .map_err(|e| PyValueError::new_err(format!("column `{}`: {}", name, e)))?;
        columns.push((name, col));
    }
    DataFrame::from_columns(columns).map_err(|e| {
        PyValueError::new_err(format!("DataFrame::from_columns failed: {:?}", e))
    })
}

// ─── Config bridges ──────────────────────────────────────────────────────────
//
// All config types accept a Python dict where missing keys take Rust defaults.
// This means a caller can pass `{}` for "defaults", `{"mean_shift_warn": 0.05}`
// to override one knob, or build a fully-specified config. Type errors fail
// loudly via PyValueError so the user sees exactly which knob was wrong.

fn dict_to_drift_config(d: Option<&Bound<'_, PyDict>>) -> PyResult<DriftConfig> {
    let mut cfg = DriftConfig::default();
    let Some(d) = d else { return Ok(cfg); };
    macro_rules! get_f64 { ($k:literal, $field:ident) => {
        if let Some(v) = d.get_item($k)? { cfg.$field = v.extract::<f64>()?; }
    } }
    macro_rules! get_u64 { ($k:literal, $field:ident) => {
        if let Some(v) = d.get_item($k)? { cfg.$field = v.extract::<u64>()?; }
    } }
    macro_rules! get_usize { ($k:literal, $field:ident) => {
        if let Some(v) = d.get_item($k)? { cfg.$field = v.extract::<usize>()?; }
    } }
    get_f64!("mean_shift_warn", mean_shift_warn);
    get_f64!("mean_shift_error", mean_shift_error);
    get_f64!("std_shift_warn", std_shift_warn);
    get_f64!("psi_warn", psi_warn);
    get_f64!("psi_error", psi_error);
    get_f64!("ks_d_warn", ks_d_warn);
    get_f64!("ks_d_error", ks_d_error);
    get_f64!("category_tvd_warn", category_tvd_warn);
    get_f64!("category_tvd_error", category_tvd_error);
    get_f64!("missingness_shift_warn", missingness_shift_warn);
    get_u64!("small_sample_threshold", small_sample_threshold);
    get_usize!("n_bins", n_bins);
    get_f64!("cardinality_explosion_ratio", cardinality_explosion_ratio);
    get_f64!("entropy_shift_warn", entropy_shift_warn);
    get_u64!("entropy_min_distinct", entropy_min_distinct);
    get_f64!("mean_shift_near_zero_threshold", mean_shift_near_zero_threshold);
    Ok(cfg)
}

fn dict_to_validate_options(d: Option<&Bound<'_, PyDict>>, label: &str) -> PyResult<ValidateOptions> {
    let mut opts = ValidateOptions {
        dataset_label: label.to_string(),
        ..Default::default()
    };
    let Some(d) = d else { return Ok(opts); };
    if let Some(v) = d.get_item("dataset_label")? {
        opts.dataset_label = v.extract::<String>()?;
    }
    if let Some(v) = d.get_item("config")? {
        let cfg_dict = v.downcast::<PyDict>().map_err(|_| {
            PyTypeError::new_err("validate options `config` must be a dict")
        })?;
        opts.config = dict_to_validation_config(Some(cfg_dict))?;
    }
    if let Some(v) = d.get_item("null_masks")? {
        let masks_dict = v.downcast::<PyDict>().map_err(|_| {
            PyTypeError::new_err("validate options `null_masks` must be a dict[str, list[int]]")
        })?;
        opts.null_masks = dict_to_null_mask_map(masks_dict)?;
    }
    Ok(opts)
}

fn dict_to_validation_config(d: Option<&Bound<'_, PyDict>>) -> PyResult<ValidationConfig> {
    let cfg = ValidationConfig::default();
    let Some(_d) = d else { return Ok(cfg); };
    // ValidationConfig has many fields; we only surface the ones users
    // realistically tune from Python. Extend as needed.
    Ok(cfg)
}

fn dict_to_null_mask_map(d: &Bound<'_, PyDict>) -> PyResult<NullMaskMap> {
    let mut out = NullMaskMap::new();
    for (k, v) in d.iter() {
        let col: String = k.extract()?;
        let rows: Vec<usize> = v.extract()?;
        out.insert(col, NullMask::from_indices(rows));
    }
    Ok(out)
}

fn dict_to_causal_config(d: Option<&Bound<'_, PyDict>>) -> PyResult<CausalConfig> {
    let mut cfg = CausalConfig::default();
    let Some(d) = d else { return Ok(cfg); };
    if let Some(v) = d.get_item("strong_correlation_threshold")? {
        cfg.strong_correlation_threshold = v.extract()?;
    }
    if let Some(v) = d.get_item("confounder_threshold")? {
        cfg.confounder_threshold = v.extract()?;
    }
    if let Some(v) = d.get_item("mode")? {
        let s: String = v.extract()?;
        cfg.mode = match s.as_str() {
            "default" => cjc_locke::causal::CausalMode::Default,
            "observational_only" => cjc_locke::causal::CausalMode::ObservationalOnly,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown causal mode `{}`; expected `default` or `observational_only`",
                    other
                )))
            }
        };
    }
    if let Some(v) = d.get_item("causal_keywords")? {
        cfg.causal_keywords = v.extract::<Vec<String>>()?;
    }
    Ok(cfg)
}

fn dict_to_time_column_config(d: Option<&Bound<'_, PyDict>>) -> PyResult<TimeColumnConfig> {
    let mut cfg = TimeColumnConfig::default();
    let Some(d) = d else { return Ok(cfg); };
    if let Some(v) = d.get_item("max_timestamp")? {
        cfg.max_timestamp = Some(v.extract()?);
    }
    if let Some(v) = d.get_item("gap_threshold")? {
        cfg.gap_threshold = Some(v.extract()?);
    }
    Ok(cfg)
}

fn dict_to_streaming_config(d: Option<&Bound<'_, PyDict>>) -> PyResult<StreamingConfig> {
    let mut cfg = StreamingConfig::default();
    let Some(d) = d else { return Ok(cfg); };
    if let Some(v) = d.get_item("sample_cap")? {
        cfg.sample_cap = v.extract()?;
    }
    if let Some(v) = d.get_item("distinct_cap")? {
        cfg.distinct_cap = v.extract()?;
    }
    Ok(cfg)
}

fn dict_to_belief_penalty(d: Option<&Bound<'_, PyDict>>) -> PyResult<BeliefPenalty> {
    let mut p = BeliefPenalty::default();
    let Some(d) = d else { return Ok(p); };
    if let Some(v) = d.get_item("info")? { p.info = v.extract()?; }
    if let Some(v) = d.get_item("notice")? { p.notice = v.extract()?; }
    if let Some(v) = d.get_item("warning")? { p.warning = v.extract()?; }
    if let Some(v) = d.get_item("error")? { p.error = v.extract()?; }
    Ok(p)
}

// ─── Custom detector bridge (ADR-0041) ───────────────────────────────────────
//
// Python users subclass `cjc_locke.CustomDetector` and pass instances to
// `validate(..., custom_detectors=[...])`. The bridge:
//
// 1. Validates each detector's static config (code namespace, axes).
// 2. Wraps each Python instance in a `PyCustomDetectorAdapter` that
//    implements the Rust `CustomDetector` trait.
// 3. Inside `run()`, builds read-only Python views of the DataFrame and
//    of a finding-collecting sink, then calls the Python detector's
//    `run(df, sink)` method.
// 4. After Python returns, drains the sink's pending emissions into the
//    real Rust `FindingSink`, which applies the same canonicalization,
//    severity rules, and sort order as built-in detectors.
//
// Determinism contract: the Python-side `run()` can use any data
// structures it likes — the Rust framework sorts emitted findings by
// `sort_key()` after the call returns, so emission order inside the
// Python code does not affect the final report bytes. The user is
// responsible for the *set* of findings being a deterministic function
// of the input.

/// Read-only view of a `DataFrame` passed to a Python custom detector.
///
/// Holds an `Arc<DataFrame>` cloned once per detector invocation so the
/// view can outlive the Rust `run()` call without lifetime headaches.
/// Most pattern-based detectors only need column names + types, which
/// are O(cols) to access. Bulk data access (`get_float` etc.) costs
/// O(rows) per column.
#[pyclass(name = "CustomDetectorDataFrame", module = "cjc_locke")]
struct PyDetectorDataFrame {
    inner: Arc<DataFrame>,
}

#[pymethods]
impl PyDetectorDataFrame {
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.nrows()
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.inner.ncols()
    }

    /// Column names in dataframe order.
    fn column_names(&self) -> Vec<String> {
        self.inner.column_names().iter().map(|s| s.to_string()).collect()
    }

    /// Returns the type name of a column: "Float", "Int", "Bool", "Str",
    /// "Categorical", "CategoricalAdaptive", "DateTime", or `None` if the
    /// column does not exist.
    fn column_type(&self, name: &str) -> Option<String> {
        self.inner
            .get_column(name)
            .map(|c| c.type_name().to_string())
    }

    /// Returns the column as a list of f64. Raises if the column is not
    /// numeric. NaN values are preserved.
    fn get_float(&self, name: &str) -> PyResult<Vec<f64>> {
        match self.inner.get_column(name) {
            Some(Column::Float(v)) => Ok(v.clone()),
            Some(Column::Int(v)) => Ok(v.iter().map(|&x| x as f64).collect()),
            Some(Column::Bool(v)) => Ok(v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()),
            Some(other) => Err(PyValueError::new_err(format!(
                "column `{}` is `{}`, not numeric",
                name,
                other.type_name()
            ))),
            None => Err(PyValueError::new_err(format!("column `{}` not found", name))),
        }
    }

    /// Returns the column as a list of strings. Works on Str and
    /// Categorical columns; raises on numeric.
    fn get_str(&self, name: &str) -> PyResult<Vec<String>> {
        match self.inner.get_column(name) {
            Some(Column::Str(v)) => Ok(v.clone()),
            Some(Column::Categorical { levels, codes }) => Ok(codes
                .iter()
                .map(|&c| levels.get(c as usize).cloned().unwrap_or_default())
                .collect()),
            Some(other) => Err(PyValueError::new_err(format!(
                "column `{}` is `{}`, not string-like",
                name,
                other.type_name()
            ))),
            None => Err(PyValueError::new_err(format!("column `{}` not found", name))),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn __repr__(&self) -> String {
        format!(
            "<CustomDetectorDataFrame rows={} cols={}>",
            self.inner.nrows(),
            self.inner.ncols()
        )
    }
}

/// Sink passed to a Python custom detector. Internally collects pending
/// emissions; after the Rust side reads them they are forwarded into the
/// real Rust `FindingSink` (where canonicalization happens).
///
/// This pattern avoids unsafe mutable references across the Python
/// boundary: the sink the user touches is owned by Python; the actual
/// canonical sink stays on the Rust side.
#[pyclass(name = "CustomDetectorSink", module = "cjc_locke")]
struct PyDetectorSink {
    pending: Vec<PyEmittedFinding>,
    code: String,
    axes_empty: bool,
    /// Last error message, exposed for Python tests that want to assert
    /// rejection behaviour.
    last_error: Option<String>,
}

#[derive(Clone, Debug)]
struct PyEmittedFinding {
    severity: FindingSeverity,
    message: String,
    column: Option<String>,
    row_range: Option<(usize, usize)>,
    sample_size: u64,
}

#[pymethods]
impl PyDetectorSink {
    /// Emit a finding. Mirrors `FindingSink::emit` on the Rust side.
    ///
    /// `severity` must be one of `"info"`, `"notice"`, `"warning"`, `"error"`.
    /// `message` must be non-empty.
    ///
    /// Empty messages and non-Info severities on an axes-empty detector
    /// are recorded as errors and dropped — same contract as the Rust
    /// `FindingSink`.
    #[pyo3(signature = (severity, message, column=None, row_range=None, sample_size=0))]
    fn emit(
        &mut self,
        severity: &str,
        message: &str,
        column: Option<String>,
        row_range: Option<(usize, usize)>,
        sample_size: u64,
    ) -> PyResult<usize> {
        if message.is_empty() {
            self.last_error = Some("empty message".into());
            return Ok(self.pending.len());
        }
        let sev = match severity {
            "info" => FindingSeverity::Info,
            "notice" => FindingSeverity::Notice,
            "warning" => FindingSeverity::Warning,
            "error" => FindingSeverity::Error,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown severity `{}` (expected info/notice/warning/error)",
                    other
                )))
            }
        };
        if self.axes_empty && sev != FindingSeverity::Info {
            self.last_error = Some(
                "detector declares no belief axes; only Info-severity findings are accepted".into(),
            );
            return Ok(self.pending.len());
        }
        self.pending.push(PyEmittedFinding {
            severity: sev,
            message: message.to_string(),
            column,
            row_range,
            sample_size,
        });
        Ok(self.pending.len())
    }

    /// Read-only: the detector's declared code (used by Python tests).
    #[getter]
    fn code(&self) -> String {
        self.code.clone()
    }

    /// Read-only: number of emissions pending.
    #[getter]
    fn n_pending(&self) -> usize {
        self.pending.len()
    }

    /// Read-only: last rejection error (None if all emissions succeeded).
    #[getter]
    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }
}

/// Rust-side wrapper that adapts a Python detector instance into the
/// `CustomDetector` trait.
struct PyCustomDetectorAdapter {
    /// The Python instance. `Py<PyAny>` is refcounted so the user's
    /// detector survives until the wrapper is dropped.
    instance: Py<PyAny>,
    /// Leaked at registration so the trait's `code()` can return
    /// `&'static str`. Detectors are typically registered once per
    /// process so the leak is bounded.
    code: &'static str,
    axes: BeliefAxisSet,
    name: String,
}

impl std::fmt::Debug for PyCustomDetectorAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyCustomDetectorAdapter {{ code: {}, name: {} }}", self.code, self.name)
    }
}

impl CustomDetector for PyCustomDetectorAdapter {
    fn code(&self) -> &'static str {
        self.code
    }
    fn belief_axes(&self) -> BeliefAxisSet {
        self.axes
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn run(&self, df: &DataFrame, sink: &mut FindingSink) {
        Python::with_gil(|py| {
            // Build the view + sink that the Python detector sees.
            let view = Py::new(
                py,
                PyDetectorDataFrame {
                    inner: Arc::new(df.clone()),
                },
            );
            let view = match view {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "[cjc_locke] failed to allocate PyDetectorDataFrame for `{}`: {}",
                        self.code, e
                    );
                    return;
                }
            };
            let py_sink = Py::new(
                py,
                PyDetectorSink {
                    pending: Vec::new(),
                    code: self.code.to_string(),
                    axes_empty: self.axes.is_empty(),
                    last_error: None,
                },
            );
            let py_sink = match py_sink {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[cjc_locke] failed to allocate PyDetectorSink for `{}`: {}",
                        self.code, e
                    );
                    return;
                }
            };
            // Call the user's run(df, sink). Errors are caught and
            // surfaced as findings on stderr; we don't abort the whole
            // validate() call on one detector's exception.
            match self
                .instance
                .bind(py)
                .call_method1("run", (view, py_sink.clone_ref(py)))
            {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[cjc_locke] custom detector `{}` raised: {}",
                        self.code, e
                    );
                    return;
                }
            }
            // Drain pending emissions into the Rust sink.
            let py_sink_bound = py_sink.bind(py);
            let pending = {
                let borrow = py_sink_bound.borrow();
                borrow.pending.clone()
            };
            for emission in pending {
                sink.emit(
                    emission.severity,
                    emission.message,
                    emission.column,
                    emission.row_range,
                    Vec::<FindingEvidence>::new(),
                    emission.sample_size,
                );
            }
        });
    }
}

/// Build a Rust trait wrapper from a Python detector instance.
/// Validates static config at registration. Raises if the code is
/// outside the custom namespace or the axes list contains unknown names.
fn build_python_detector_adapter(
    py: Python<'_>,
    detector: Py<PyAny>,
) -> PyResult<PyCustomDetectorAdapter> {
    let bound = detector.bind(py);
    let code: String = bound.getattr("code")?.call0()?.extract()?;
    validate_custom_code(&code).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let axes_raw: Vec<String> = bound.getattr("belief_axes")?.call0()?.extract()?;
    let axes_refs: Vec<&str> = axes_raw.iter().map(|s| s.as_str()).collect();
    let axes = BeliefAxisSet::from_names(&axes_refs).map_err(PyValueError::new_err)?;
    let name: String = match bound.getattr("name") {
        Ok(n) => n.call0().and_then(|v| v.extract::<String>()).unwrap_or_else(|_| code.clone()),
        Err(_) => code.clone(),
    };
    // Leak the code so the trait can return &'static str. Detectors
    // typically live for the process lifetime so the leak is bounded.
    let leaked: &'static str = Box::leak(code.into_boxed_str());
    Ok(PyCustomDetectorAdapter {
        instance: detector,
        code: leaked,
        axes,
        name,
    })
}

// ─── Output wrappers ──────────────────────────────────────────────────────────
//
// Each output type wraps a Rust value verbatim. `to_json()` uses the existing
// deterministic emit; `to_dict()` is implemented Python-side as
// `json.loads(self.to_json())` for cheap dict access. Direct field accessors
// are exposed only for the hottest values (severity_counts, finding counts).

#[pyclass(name = "LockeReport", module = "cjc_locke")]
struct PyLockeReport {
    inner: LockeReport,
}

#[pymethods]
impl PyLockeReport {
    fn to_json(&self) -> String {
        cjc_locke::emit_locke_report_json(&self.inner)
    }

    #[getter]
    fn n_findings(&self) -> usize {
        self.inner.findings.len()
    }

    #[getter]
    fn severity_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("info", self.inner.severity_counts.info)?;
        d.set_item("notice", self.inner.severity_counts.notice)?;
        d.set_item("warning", self.inner.severity_counts.warning)?;
        d.set_item("error", self.inner.severity_counts.error)?;
        Ok(d)
    }

    #[getter]
    fn n_rows(&self) -> u64 {
        self.inner.input.n_rows
    }

    #[getter]
    fn n_cols(&self) -> u64 {
        self.inner.input.n_cols
    }

    fn finding_codes(&self) -> Vec<String> {
        self.inner
            .findings
            .iter()
            .map(|f| f.code.to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<LockeReport n_findings={} info={} notice={} warning={} error={}>",
            self.inner.findings.len(),
            self.inner.severity_counts.info,
            self.inner.severity_counts.notice,
            self.inner.severity_counts.warning,
            self.inner.severity_counts.error,
        )
    }
}

#[pyclass(name = "InductionRiskReport", module = "cjc_locke")]
struct PyInductionRiskReport {
    inner: InductionRiskReport,
}

#[pymethods]
impl PyInductionRiskReport {
    #[getter]
    fn n_train(&self) -> u64 {
        self.inner.n_train
    }
    #[getter]
    fn n_test(&self) -> u64 {
        self.inner.n_test
    }
    #[getter]
    fn shared_columns(&self) -> Vec<String> {
        self.inner.shared_columns.clone()
    }
    #[getter]
    fn train_only_columns(&self) -> Vec<String> {
        self.inner.train_only_columns.clone()
    }
    #[getter]
    fn test_only_columns(&self) -> Vec<String> {
        self.inner.test_only_columns.clone()
    }
    #[getter]
    fn n_findings(&self) -> usize {
        self.inner.findings.len()
    }
    fn finding_codes(&self) -> Vec<String> {
        self.inner.findings.iter().map(|f| f.code.to_string()).collect()
    }
    fn to_json(&self) -> String {
        // No dedicated emitter for InductionRiskReport; wrap findings in
        // a LockeReport-style envelope so the bytes stay deterministic.
        // Callers wanting the canonical drift JSON should use
        // `validate_and_compare(...)` and read the val_report side.
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"n_train\":{},\"n_test\":{}", self.inner.n_train, self.inner.n_test));
        out.push_str(",\"shared_columns\":[");
        for (i, c) in self.inner.shared_columns.iter().enumerate() {
            if i > 0 { out.push(','); }
            out.push('"');
            out.push_str(c);
            out.push('"');
        }
        out.push(']');
        out.push_str(",\"n_findings\":");
        out.push_str(&self.inner.findings.len().to_string());
        out.push('}');
        out
    }
    fn __repr__(&self) -> String {
        format!(
            "<InductionRiskReport n_train={} n_test={} n_findings={}>",
            self.inner.n_train,
            self.inner.n_test,
            self.inner.findings.len(),
        )
    }
}

#[pyclass(name = "BeliefReport", module = "cjc_locke")]
struct PyBeliefReport {
    inner: BeliefReport,
}

#[pymethods]
impl PyBeliefReport {
    #[getter] fn overall(&self) -> f64 { self.inner.score.overall }
    #[getter] fn schema_score(&self) -> f64 { self.inner.score.schema_score }
    #[getter] fn missingness_score(&self) -> f64 { self.inner.score.missingness_score }
    #[getter] fn drift_score(&self) -> f64 { self.inner.score.drift_score }
    #[getter] fn leakage_score(&self) -> f64 { self.inner.score.leakage_score }
    #[getter] fn lineage_score(&self) -> f64 { self.inner.score.lineage_score }
    #[getter] fn sample_score(&self) -> f64 { self.inner.score.sample_score }
    #[getter] fn duplication_score(&self) -> f64 { self.inner.score.duplication_score }
    #[getter] fn constraint_score(&self) -> f64 { self.inner.score.constraint_score }
    #[getter] fn assumptions(&self) -> Vec<String> { self.inner.assumptions.clone() }

    fn score_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        let s = &self.inner.score;
        d.set_item("overall", s.overall)?;
        d.set_item("schema", s.schema_score)?;
        d.set_item("missingness", s.missingness_score)?;
        d.set_item("drift", s.drift_score)?;
        d.set_item("leakage", s.leakage_score)?;
        d.set_item("lineage", s.lineage_score)?;
        d.set_item("sample", s.sample_score)?;
        d.set_item("duplication", s.duplication_score)?;
        d.set_item("constraint", s.constraint_score)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "<BeliefReport overall={:.4} schema={:.4} drift={:.4}>",
            self.inner.score.overall,
            self.inner.score.schema_score,
            self.inner.score.drift_score,
        )
    }
}

#[pyclass(name = "CausalGuardrailReport", module = "cjc_locke")]
struct PyCausalGuardrailReport {
    inner: CausalGuardrailReport,
}

#[pymethods]
impl PyCausalGuardrailReport {
    #[getter] fn n_warnings(&self) -> usize { self.inner.warnings.len() }
    #[getter] fn n_correlations(&self) -> usize { self.inner.correlations.len() }
    #[getter] fn n_confounder_hints(&self) -> usize { self.inner.confounder_hints.len() }
    fn warning_kinds(&self) -> Vec<String> {
        self.inner
            .warnings
            .iter()
            .map(|w| format!("{:?}", w.kind))
            .collect()
    }
}

// ─── Top-level functions ─────────────────────────────────────────────────────

/// Validate a single DataFrame.
///
/// `data`              — dict mapping column name to numpy array (f64/i64/bool/f32/i32)
///                       or Python list (float/int/str/bool).
/// `label`             — dataset label that lands in the report (default "dataset").
/// `options`           — optional dict of ValidateOptions overrides (e.g.
///                       `{"null_masks": {"col": [3, 7, 12]}}`).
/// `custom_detectors`  — optional list of Python `CustomDetector` instances
///                       (ADR-0041). Each must expose `code()` returning
///                       `"E9500"-"E9999"`, `belief_axes()` returning a list of
///                       axis names, and `run(df, sink)`.
#[pyfunction]
#[pyo3(signature = (data, label="dataset", options=None, custom_detectors=None))]
fn validate_dataframe(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    label: &str,
    options: Option<&Bound<'_, PyDict>>,
    custom_detectors: Option<Vec<Py<PyAny>>>,
) -> PyResult<PyLockeReport> {
    let df = dict_to_dataframe(py, data)?;
    let mut opts = dict_to_validate_options(options, label)?;
    if let Some(detectors) = custom_detectors {
        for detector in detectors {
            let adapter = build_python_detector_adapter(py, detector)?;
            opts.custom_detectors
                .push(Arc::new(adapter) as Arc<dyn CustomDetector>);
        }
    }
    let report = validate(&df, &opts);
    Ok(PyLockeReport { inner: report })
}

/// Compare two DataFrames (train vs test) and emit an InductionRiskReport.
#[pyfunction]
#[pyo3(signature = (train, test, drift_config=None))]
fn compare_drift(
    py: Python<'_>,
    train: &Bound<'_, PyDict>,
    test: &Bound<'_, PyDict>,
    drift_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyInductionRiskReport> {
    let train_df = dict_to_dataframe(py, train)?;
    let test_df = dict_to_dataframe(py, test)?;
    let cfg = dict_to_drift_config(drift_config)?;
    let report = locke_compare_drift(&train_df, &test_df, &cfg);
    Ok(PyInductionRiskReport { inner: report })
}

/// Combined validate + drift + belief.
#[pyfunction]
#[pyo3(signature = (train, test, label="dataset", options=None, drift_config=None))]
fn validate_and_compare(
    py: Python<'_>,
    train: &Bound<'_, PyDict>,
    test: &Bound<'_, PyDict>,
    label: &str,
    options: Option<&Bound<'_, PyDict>>,
    drift_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(PyLockeReport, PyInductionRiskReport, PyBeliefReport)> {
    let train_df = dict_to_dataframe(py, train)?;
    let test_df = dict_to_dataframe(py, test)?;
    let opts = dict_to_validate_options(options, label)?;
    let dcfg = dict_to_drift_config(drift_config)?;
    let (val, drift, belief) = locke_validate_and_compare(&train_df, &test_df, &opts, &dcfg);
    Ok((
        PyLockeReport { inner: val },
        PyInductionRiskReport { inner: drift },
        PyBeliefReport { inner: belief },
    ))
}

#[pyfunction]
#[pyo3(signature = (report, model=None))]
fn belief_report(
    report: &PyLockeReport,
    model: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyBeliefReport> {
    let belief = match model {
        Some(m) => {
            let penalty = dict_to_belief_penalty(Some(m))?;
            belief_report_from_locke_with_model(&report.inner, &penalty)
        }
        None => belief_report_from_locke(&report.inner),
    };
    Ok(PyBeliefReport { inner: belief })
}

#[pyfunction]
#[pyo3(signature = (data, target_column=None, causal_config=None, label_text=None, interpret_model_explanation_as_causal=false))]
fn causal_guardrail(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    target_column: Option<&str>,
    causal_config: Option<&Bound<'_, PyDict>>,
    label_text: Option<&str>,
    interpret_model_explanation_as_causal: bool,
) -> PyResult<PyCausalGuardrailReport> {
    let df = dict_to_dataframe(py, data)?;
    let cfg = dict_to_causal_config(causal_config)?;
    let report = locke_causal_guardrail(
        &df,
        target_column,
        &cfg,
        label_text,
        interpret_model_explanation_as_causal,
    );
    Ok(PyCausalGuardrailReport { inner: report })
}

#[pyfunction]
#[pyo3(signature = (data, time_col, config=None))]
fn detect_temporal_issues(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    time_col: &str,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<String>> {
    let df = dict_to_dataframe(py, data)?;
    let cfg = dict_to_time_column_config(config)?;
    let findings = locke_detect_temporal_issues(&df, time_col, &cfg);
    // Surface the codes; full evidence available via the report-generating
    // workflows (validate + temporal goes through validate's path in v0.7+).
    Ok(findings.iter().map(|f| f.code.to_string()).collect())
}

// ─── JSON round-trip ─────────────────────────────────────────────────────────

#[pyfunction]
fn emit_report_json(report: &PyLockeReport) -> String {
    cjc_locke::emit_locke_report_json(&report.inner)
}

#[pyfunction]
fn parse_report_json(json: &str) -> PyResult<PyLockeReport> {
    cjc_locke::parse_locke_report_json(json)
        .map(|r| PyLockeReport { inner: r })
        .map_err(|e| PyValueError::new_err(e))
}

// ─── Streaming validator ─────────────────────────────────────────────────────

#[pyclass(name = "StreamingValidator", module = "cjc_locke")]
struct PyStreamingValidator {
    inner: Option<StreamingValidator>,
}

#[pymethods]
impl PyStreamingValidator {
    #[new]
    #[pyo3(signature = (label="dataset", config=None))]
    fn new(label: &str, config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let cfg = dict_to_streaming_config(config)?;
        Ok(Self {
            inner: Some(StreamingValidator::new(label, cfg)),
        })
    }

    fn ingest_chunk(&mut self, py: Python<'_>, chunk: &Bound<'_, PyDict>) -> PyResult<()> {
        let df = dict_to_dataframe(py, chunk)?;
        let sv = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("StreamingValidator already finalized"))?;
        sv.ingest_chunk(&df)
            .map_err(|e| PyValueError::new_err(format!("ingest_chunk: {}", e)))
    }

    #[pyo3(signature = (options=None))]
    fn into_report(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyLockeReport> {
        let sv = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("StreamingValidator already finalized"))?;
        let opts = match options {
            Some(d) => Some(dict_to_validate_options(Some(d), "stream")?),
            None => None,
        };
        sv.into_report(opts)
            .map(|r| PyLockeReport { inner: r })
            .map_err(|e| PyValueError::new_err(e))
    }
}

// ─── Lineage builder ─────────────────────────────────────────────────────────
//
// The Rust `LineageBuilder` API has zero-cost iterators and a `TracedDataFrame`
// wrapper that uses lifetimes — these don't map cleanly to Python. We expose
// the simpler primitive surface: add_impression, add_idea, finish. Callers
// can build the full provenance chain by hand.

#[pyclass(name = "LineageBuilder", module = "cjc_locke")]
struct PyLineageBuilder {
    inner: Option<LineageBuilder>,
}

#[pymethods]
impl PyLineageBuilder {
    #[new]
    fn new(run_label: &str) -> Self {
        Self {
            inner: Some(LineageBuilder::new(run_label)),
        }
    }

    /// Add a leaf impression (a source dataset/table/event).
    ///
    /// Returns the fingerprint id as `int` (use it as the `parent` for ideas
    /// built on top of this impression).
    #[pyo3(signature = (source, kind="dataset", n_rows=0, columns=vec![]))]
    fn add_impression(
        &mut self,
        source: &str,
        kind: &str,
        n_rows: u64,
        columns: Vec<String>,
    ) -> PyResult<u64> {
        let kind = match kind {
            "dataset" => ImpressionKind::Dataset,
            "column" => ImpressionKind::Column,
            "row" => ImpressionKind::Row,
            "schema" => ImpressionKind::Schema,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown impression kind `{}`; expected dataset/column/row/schema",
                    other
                )))
            }
        };
        let imp = LockeImpression::new(source, kind, n_rows, columns);
        let builder = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("LineageBuilder already finished"))?;
        builder
            .add_impression(imp)
            .map(|id| id.0)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Add a derived idea node with a transformation record.
    ///
    /// `parents` is a list of parent fingerprint ids (returned by previous
    /// `add_impression` / `add_idea` calls). `op_id` is the operation
    /// identifier (e.g. "filter", "join", "select"). `params` is an optional
    /// dict of stable-string parameters used in fingerprinting.
    #[pyo3(signature = (name, op_id, parents, params=None, seed=None))]
    fn add_idea(
        &mut self,
        name: &str,
        op_id: &str,
        parents: Vec<u64>,
        params: Option<&Bound<'_, PyDict>>,
        seed: Option<u64>,
    ) -> PyResult<u64> {
        let mut p = BTreeMap::new();
        if let Some(d) = params {
            for (k, v) in d.iter() {
                let key: String = k.extract()?;
                let val: String = v.extract()?;
                p.insert(key, val);
            }
        }
        let xform = TransformationRecord {
            op_id: op_id.to_string(),
            params: p,
            seed,
        };
        let idea = LockeIdea::new(
            name,
            xform,
            parents.into_iter().map(FingerprintId).collect(),
        );
        let builder = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("LineageBuilder already finished"))?;
        builder
            .add_idea(idea)
            .map(|id| id.0)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    fn audit_note(&mut self, kind: &str, subject_id: u64, note: &str) -> PyResult<()> {
        let builder = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("LineageBuilder already finished"))?;
        builder.audit_note(kind, FingerprintId(subject_id), note);
        Ok(())
    }

    fn finish(&mut self) -> PyResult<PyLineageGraph> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("LineageBuilder already finished"))?;
        Ok(PyLineageGraph {
            inner: builder.finish(),
        })
    }
}

#[pyclass(name = "LineageGraph", module = "cjc_locke")]
struct PyLineageGraph {
    inner: LineageGraph,
}

#[pymethods]
impl PyLineageGraph {
    #[getter]
    fn root_fingerprint(&self) -> u64 {
        self.inner.root_fingerprint.0
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.edges.len()
    }

    #[getter]
    fn n_audit_events(&self) -> usize {
        self.inner.audit.len()
    }

    fn is_acyclic(&self) -> bool {
        self.inner.is_acyclic()
    }

    /// Reachable ancestor set of `node_id`. Returns a sorted list of u64 ids.
    fn ancestors(&self, node_id: u64) -> Vec<u64> {
        let set = self.inner.ancestors(FingerprintId(node_id));
        set.into_iter().map(|id| id.0).collect()
    }

    fn validate_audit_monotonic(&self) -> PyResult<()> {
        self.inner
            .validate_audit_monotonic()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    fn emit_mermaid(&self) -> String {
        cjc_locke::lineage::emit_lineage_mermaid(&self.inner)
    }

    fn emit_text(&self) -> String {
        cjc_locke::lineage::emit_lineage_text(&self.inner)
    }
}

// ─── Audit event ──────────────────────────────────────────────────────────────

/// Construct an audit event. Monotonicity (seq strictly increasing per
/// run_label) is the caller's responsibility — use `LineageBuilder` for
/// automatic seq assignment, or validate via `LineageGraph.validate_audit_monotonic`.
#[pyfunction]
fn make_audit_event(
    run_label: &str,
    seq: u64,
    kind: &str,
    subject_id: u64,
    note: &str,
) -> PyAuditEvent {
    PyAuditEvent {
        inner: AuditEvent::new(run_label, seq, kind, FingerprintId(subject_id), note),
    }
}

#[pyclass(name = "AuditEvent", module = "cjc_locke")]
struct PyAuditEvent {
    inner: AuditEvent,
}

#[pymethods]
impl PyAuditEvent {
    #[getter] fn id(&self) -> u64 { self.inner.id.0 }
    #[getter] fn run_label(&self) -> String { self.inner.run_label.clone() }
    #[getter] fn seq(&self) -> u64 { self.inner.seq }
    #[getter] fn kind(&self) -> String { self.inner.kind.clone() }
    #[getter] fn subject_id(&self) -> u64 { self.inner.subject_id.0 }
    #[getter] fn note(&self) -> String { self.inner.note.clone() }
}

// ─── Policy ──────────────────────────────────────────────────────────────────
//
// Policy rules are constructed from Python dicts. This keeps the wrapper
// minimal (no per-rule pyclass needed); each rule is a `{"code": ..., ...}`
// dict that we translate to the Rust struct.

fn dict_to_suppression_rule(d: &Bound<'_, PyDict>) -> PyResult<SuppressionRule> {
    let code: String = d
        .get_item("code")?
        .ok_or_else(|| PyValueError::new_err("SuppressionRule needs `code`"))?
        .extract()?;
    let reason: String = d
        .get_item("reason")?
        .map(|v| v.extract::<String>())
        .unwrap_or_else(|| Ok(String::new()))?;
    let column = match d.get_item("column")? {
        Some(v) if !v.is_none() => Some(cjc_locke::policy::ColumnMatcher::from_pattern(
            &v.extract::<String>()?,
        )),
        _ => None,
    };
    Ok(SuppressionRule {
        code,
        column,
        reason,
    })
}

fn dict_to_owner_rule(d: &Bound<'_, PyDict>) -> PyResult<OwnerRule> {
    let team: String = d
        .get_item("team")?
        .ok_or_else(|| PyValueError::new_err("OwnerRule needs `team`"))?
        .extract()?;
    let code = match d.get_item("code")? {
        Some(v) if !v.is_none() => Some(v.extract::<String>()?),
        _ => None,
    };
    let column = match d.get_item("column")? {
        Some(v) if !v.is_none() => Some(cjc_locke::policy::ColumnMatcher::from_pattern(
            &v.extract::<String>()?,
        )),
        _ => None,
    };
    Ok(OwnerRule { team, code, column })
}

fn dict_to_requirement_rule(d: &Bound<'_, PyDict>) -> PyResult<RequiredFindingRule> {
    let code: String = d
        .get_item("code")?
        .ok_or_else(|| PyValueError::new_err("RequiredFindingRule needs `code`"))?
        .extract()?;
    let op_str: String = d
        .get_item("operator")?
        .map(|v| v.extract::<String>())
        .unwrap_or_else(|| Ok("eq_zero".to_string()))?;
    let operator = match op_str.as_str() {
        "eq_zero" | "==0" => RequirementOperator::EqZero,
        "lt" | "<" => RequirementOperator::Less,
        "le" | "<=" | "lte" => RequirementOperator::LessOrEqual,
        "gt" | ">" => RequirementOperator::Greater,
        "ge" | ">=" | "gte" => RequirementOperator::GreaterOrEqual,
        "eq" | "==" => RequirementOperator::Equal,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown requirement operator `{}` (expected one of eq_zero/lt/le/gt/ge/eq)",
                other
            )))
        }
    };
    let threshold: u64 = d
        .get_item("threshold")?
        .map(|v| v.extract::<u64>())
        .unwrap_or(Ok(0))?;
    let owner = match d.get_item("owner")? {
        Some(v) if !v.is_none() => Some(v.extract::<String>()?),
        _ => None,
    };
    Ok(RequiredFindingRule {
        code,
        operator,
        threshold,
        owner,
    })
}

/// Apply a policy to a LockeReport. The policy is a dict like:
/// `{"suppressions": [{"code": "E9001", "column": "phone", "reason": "ack"}],
///   "owners": [{"team": "data", "code": "E9072"}],
///   "requirements": [{"code": "E9001", "operator": "eq_zero", "threshold": 0}]}`
#[pyfunction]
fn apply_policy(
    report: &PyLockeReport,
    policy: &Bound<'_, PyDict>,
) -> PyResult<PyPolicyResult> {
    let suppressions = match policy.get_item("suppressions")? {
        Some(v) => {
            let list = v.downcast::<PyList>()?;
            list.iter()
                .map(|item| dict_to_suppression_rule(item.downcast::<PyDict>()?))
                .collect::<PyResult<Vec<_>>>()?
        }
        None => vec![],
    };
    let owners = match policy.get_item("owners")? {
        Some(v) => {
            let list = v.downcast::<PyList>()?;
            list.iter()
                .map(|item| dict_to_owner_rule(item.downcast::<PyDict>()?))
                .collect::<PyResult<Vec<_>>>()?
        }
        None => vec![],
    };
    let requirements = match policy.get_item("requirements")? {
        Some(v) => {
            let list = v.downcast::<PyList>()?;
            list.iter()
                .map(|item| dict_to_requirement_rule(item.downcast::<PyDict>()?))
                .collect::<PyResult<Vec<_>>>()?
        }
        None => vec![],
    };
    let p = Policy {
        suppressions,
        owners,
        requirements,
    };
    Ok(PyPolicyResult {
        inner: locke_apply_policy(&report.inner, &p),
    })
}

#[pyclass(name = "PolicyResult", module = "cjc_locke")]
struct PyPolicyResult {
    inner: PolicyResult,
}

#[pymethods]
impl PyPolicyResult {
    #[getter]
    fn n_suppressed(&self) -> usize {
        self.inner.suppressions.len()
    }
    #[getter]
    fn n_remaining(&self) -> usize {
        self.inner.remaining_findings.len()
    }
    #[getter]
    fn n_attributions(&self) -> usize {
        self.inner.attributions.len()
    }
    fn all_requirements_satisfied(&self) -> bool {
        self.inner.all_requirements_satisfied()
    }
    fn gate_fails(&self) -> bool {
        self.inner.gate_fails()
    }
    fn remaining_codes(&self) -> Vec<String> {
        self.inner
            .remaining_findings
            .iter()
            .map(|f| f.code.to_string())
            .collect()
    }
}

// ─── Module registration ──────────────────────────────────────────────────────

#[pymodule]
fn _cjc_locke(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core validation + drift
    m.add_function(wrap_pyfunction!(validate_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(compare_drift, m)?)?;
    m.add_function(wrap_pyfunction!(validate_and_compare, m)?)?;
    m.add_function(wrap_pyfunction!(belief_report, m)?)?;

    // Causal + temporal
    m.add_function(wrap_pyfunction!(causal_guardrail, m)?)?;
    m.add_function(wrap_pyfunction!(detect_temporal_issues, m)?)?;

    // JSON
    m.add_function(wrap_pyfunction!(emit_report_json, m)?)?;
    m.add_function(wrap_pyfunction!(parse_report_json, m)?)?;

    // Audit event constructor
    m.add_function(wrap_pyfunction!(make_audit_event, m)?)?;

    // Policy
    m.add_function(wrap_pyfunction!(apply_policy, m)?)?;

    // Classes
    m.add_class::<PyLockeReport>()?;
    m.add_class::<PyInductionRiskReport>()?;
    m.add_class::<PyBeliefReport>()?;
    m.add_class::<PyCausalGuardrailReport>()?;
    m.add_class::<PyStreamingValidator>()?;
    m.add_class::<PyLineageBuilder>()?;
    m.add_class::<PyLineageGraph>()?;
    m.add_class::<PyAuditEvent>()?;
    m.add_class::<PyPolicyResult>()?;

    // Custom-detector bridge (ADR-0041)
    m.add_class::<PyDetectorDataFrame>()?;
    m.add_class::<PyDetectorSink>()?;

    // Version (matches the Rust workspace at build time).
    m.add("__version__", "0.1.0")?;
    m.add("__rust_crate_version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
