"""cjc_locke — deterministic dataset validation for Python.

Thin wrapper over the Rust `cjc-locke` crate via PyO3.

Design contract:
  * Every public function delegates to one Rust call. No business logic
    on the Python side.
  * Input dicts are insertion-ordered (PEP 468), so column order in →
    column order out, reproducible across runs.
  * Numpy arrays cross the FFI boundary zero-copy via the buffer protocol;
    Python lists incur one allocation per column.
  * The Rust side is byte-identical to a native call — the wrapper does
    not multithread, cache, or reorder anything.

Quick start:

    >>> import numpy as np
    >>> import cjc_locke
    >>> data = {"age": np.array([20.0, 30.0, np.nan, 40.0]),
    ...         "city": ["NY", "LA", "NY", "SF"]}
    >>> report = cjc_locke.validate(data, label="users")
    >>> report.severity_counts
    {'info': 0, 'notice': 0, 'warning': 1, 'error': 0}
    >>> report.finding_codes()
    ['E9001']

The full output (every finding with full evidence) is one JSON-decode away:

    >>> import json
    >>> report.to_dict()['findings'][0]['evidence']
    [{'kind': 'count', 'label': 'n_missing', 'value': 1}, ...]
"""

from __future__ import annotations

import json as _json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import _cjc_locke

__version__ = _cjc_locke.__version__
__rust_crate_version__ = _cjc_locke.__rust_crate_version__

# ── Type aliases ──────────────────────────────────────────────────────────────
#
# We keep these as `Any` rather than depending on numpy so the package has
# zero hard runtime deps. Users with numpy installed get the zero-copy fast
# path automatically.
DataDict = Dict[str, Any]


# ── Output classes ────────────────────────────────────────────────────────────
#
# These are imported from the native module but re-exported here so users
# can do `from cjc_locke import LockeReport` for type hinting.

LockeReport = _cjc_locke.LockeReport
InductionRiskReport = _cjc_locke.InductionRiskReport
BeliefReport = _cjc_locke.BeliefReport
CausalGuardrailReport = _cjc_locke.CausalGuardrailReport
StreamingValidator = _cjc_locke.StreamingValidator
LineageBuilder = _cjc_locke.LineageBuilder
LineageGraph = _cjc_locke.LineageGraph
AuditEvent = _cjc_locke.AuditEvent
PolicyResult = _cjc_locke.PolicyResult

# Custom-detector bridge (ADR-0041). The DataFrame view + sink classes
# are passed BY THE FRAMEWORK to a user's CustomDetector.run(df, sink) —
# users don't construct them directly, but they're re-exported here for
# type hinting.
CustomDetectorDataFrame = _cjc_locke.CustomDetectorDataFrame
CustomDetectorSink = _cjc_locke.CustomDetectorSink


# ── CustomDetector ABC ───────────────────────────────────────────────────────
#
# User detectors subclass this. The framework calls `code()`,
# `belief_axes()`, and `run(df, sink)` directly — no metaclass magic, no
# registry, just three method calls.


class CustomDetector:
    """Base class for user-defined Locke detectors (ADR-0041).

    Subclass and override:

    - ``code()`` → str — E-code in ``E9500..=E9999``. Built-in codes
      ``E9001..=E9112`` are reserved.
    - ``belief_axes()`` → list[str] — list of axis names this detector's
      findings affect. Valid names: ``schema``, ``missingness``, ``drift``,
      ``leakage``, ``lineage``, ``sample``, ``duplication``, ``constraint``.
      Empty list = the detector contributes findings to the report but
      does NOT affect any belief score (advisory-only). In that case,
      only ``info``-severity findings are accepted by the sink.
    - ``run(df, sink)`` — detection logic. ``df`` is a read-only
      ``CustomDetectorDataFrame`` (column_names, n_rows, get_float,
      get_str, ...). ``sink`` is a ``CustomDetectorSink`` with one method:
      ``emit(severity, message, column=None, row_range=None, sample_size=0)``.

    The framework guarantees:

    - Detectors are invoked in canonical (sort-by-code) order.
    - Emitted findings are sorted by ``sort_key`` after ``run()`` returns,
      so emission order inside ``run()`` does not affect report bytes.
    - Findings outside the ``E9500..=E9999`` namespace are rejected at
      registration.

    Example::

        class PostOriginationByName(cjc_locke.CustomDetector):
            def code(self) -> str:
                return "E9500"

            def belief_axes(self) -> list[str]:
                return ["leakage"]

            def run(self, df, sink) -> None:
                for col in df.column_names():
                    if col.startswith(("total_", "last_pymnt_", "recoveries")):
                        sink.emit(
                            "error",
                            f"`{col}` matches a post-origination naming pattern",
                            column=col,
                            sample_size=df.n_rows,
                        )

        report = cjc_locke.validate(
            data,
            label="lc",
            custom_detectors=[PostOriginationByName()],
        )
    """

    def code(self) -> str:
        raise NotImplementedError("override CustomDetector.code() to return an E-code string")

    def belief_axes(self) -> List[str]:
        raise NotImplementedError(
            "override CustomDetector.belief_axes() to return a list of axis names"
        )

    def name(self) -> str:
        """Human-readable label used in error messages. Defaults to ``code()``."""
        return self.code()

    def run(self, df: Any, sink: Any) -> None:
        raise NotImplementedError(
            "override CustomDetector.run(df, sink) to emit findings via sink.emit(...)"
        )


def _pandas_column_payload(series: Any) -> Any:
    """Pick the cheapest representation that the Rust side can accept.

    Numeric columns go through `.to_numpy()` for the zero-copy buffer-
    protocol path. Object/string/category dtypes go through `.tolist()`
    because numpy object arrays aren't supported by the wrapper (the
    Rust side would have to per-element extract anyway).
    """
    dtype = getattr(series, "dtype", None)
    kind = getattr(dtype, "kind", None)
    # 'i'/'u'/'f'/'b' = numeric / bool; pass via numpy buffer protocol.
    if kind in ("i", "u", "f", "b"):
        return series.to_numpy()
    # Everything else (object, string, category, datetime, etc.) → list.
    # We could special-case category to extract codes + levels, but the
    # list path is correct and ~ms-level overhead for typical sizes.
    return series.tolist()


def _polars_column_payload(series: Any) -> Any:
    """Same idea for polars: try `.to_numpy()` first; if the dtype is
    a string/object/category that numpy can't represent zero-copy,
    fall back to `.to_list()`."""
    # polars Utf8 / Categorical etc. can be detected via `.dtype`. We
    # don't import polars (zero hard deps) so we duck-type by trying
    # to_numpy and catching the failure case.
    try:
        arr = series.to_numpy()
        # Object-dtype arrays will fail downstream — convert to list.
        if hasattr(arr, "dtype") and arr.dtype == object:
            return series.to_list()
        return arr
    except Exception:
        return series.to_list()


def _ensure_dict(data: Any) -> DataDict:
    """Coerce a pandas / polars DataFrame into a plain dict[str, array|list].

    Pandas and polars are optional. We duck-type on attribute presence
    so both libraries remain optional dependencies. Plain dicts pass
    through unchanged.

    Polars is checked BEFORE pandas because polars DataFrames also have
    `.to_dict()` (so the pandas duck-typing branch would catch them
    incorrectly). `.to_pandas()` is polars-specific.
    """
    if isinstance(data, dict):
        return data
    # polars DataFrame — check first because pandas check would match too
    if hasattr(data, "columns") and hasattr(data, "to_pandas"):
        return {c: _polars_column_payload(data[c]) for c in data.columns}
    # pandas DataFrame
    if hasattr(data, "columns") and hasattr(data, "to_dict"):
        return {c: _pandas_column_payload(data[c]) for c in data.columns}
    raise TypeError(
        f"unsupported data type {type(data).__name__}: expected dict, "
        f"pandas.DataFrame, or polars.DataFrame"
    )


# ── Public functions ──────────────────────────────────────────────────────────


def validate(
    data: Any,
    label: str = "dataset",
    options: Optional[Dict[str, Any]] = None,
    custom_detectors: Optional[Sequence[CustomDetector]] = None,
) -> LockeReport:
    """Validate a single DataFrame.

    Args:
        data: dict[str, np.ndarray | list] (or pandas/polars DataFrame).
        label: dataset label that lands in the report.
        options: optional dict of `ValidateOptions` overrides.
        custom_detectors: optional list of `CustomDetector` instances
            (ADR-0041). Each detector's findings are merged into the
            report under the determinism contract documented on
            `CustomDetector`.

    Returns:
        A `LockeReport`. Inspect it via `.severity_counts`, `.finding_codes()`,
        `.to_json()`, or `report.to_dict()` for the full content.
    """
    detectors = list(custom_detectors) if custom_detectors is not None else None
    return _cjc_locke.validate_dataframe(_ensure_dict(data), label, options, detectors)


def compare_drift(
    train: Any,
    test: Any,
    drift_config: Optional[Dict[str, Any]] = None,
) -> InductionRiskReport:
    """Compare train and test DataFrames for distribution drift."""
    return _cjc_locke.compare_drift(_ensure_dict(train), _ensure_dict(test), drift_config)


def validate_and_compare(
    train: Any,
    test: Any,
    label: str = "dataset",
    options: Optional[Dict[str, Any]] = None,
    drift_config: Optional[Dict[str, Any]] = None,
) -> Tuple[LockeReport, InductionRiskReport, BeliefReport]:
    """Combined validate + drift + belief in one call.

    Returns the three reports as a tuple. The belief score's drift axis is
    composed from the drift report under the meet-semilattice algebra
    (see the Rust-side docs).
    """
    return _cjc_locke.validate_and_compare(
        _ensure_dict(train),
        _ensure_dict(test),
        label,
        options,
        drift_config,
    )


def belief_report(
    report: LockeReport,
    model: Optional[Dict[str, float]] = None,
) -> BeliefReport:
    """Derive a BeliefReport from a LockeReport.

    `model` is an optional `BeliefPenalty` model: `{"info": float, "notice":
    float, "warning": float, "error": float}`. Defaults match the Rust-side
    `BeliefPenalty::default()`.
    """
    return _cjc_locke.belief_report(report, model)


def causal_guardrail(
    data: Any,
    target_column: Optional[str] = None,
    causal_config: Optional[Dict[str, Any]] = None,
    label_text: Optional[str] = None,
    interpret_model_explanation_as_causal: bool = False,
) -> CausalGuardrailReport:
    """Run the causal guardrail audit on a DataFrame."""
    return _cjc_locke.causal_guardrail(
        _ensure_dict(data),
        target_column,
        causal_config,
        label_text,
        interpret_model_explanation_as_causal,
    )


def detect_temporal_issues(
    data: Any,
    time_col: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run temporal sanity checks on a column. Returns the list of finding codes."""
    return _cjc_locke.detect_temporal_issues(_ensure_dict(data), time_col, config)


def apply_policy(report: LockeReport, policy: Dict[str, Any]) -> PolicyResult:
    """Apply a policy (suppressions/owners/requirements) to a LockeReport.

    Policy shape:

        {
            "suppressions": [{"code": "E9001", "column": "phone", "reason": "ack"}],
            "owners":       [{"team": "data", "code": "E9072"}],
            "requirements": [{"code": "E9001", "operator": "eq_zero", "threshold": 0}],
        }
    """
    return _cjc_locke.apply_policy(report, policy)


def make_audit_event(
    run_label: str,
    seq: int,
    kind: str,
    subject_id: int,
    note: str,
) -> AuditEvent:
    """Construct an AuditEvent. Monotonicity is the caller's responsibility —
    prefer `LineageBuilder` for automatic seq assignment."""
    return _cjc_locke.make_audit_event(run_label, seq, kind, subject_id, note)


# ── JSON round-trip ───────────────────────────────────────────────────────────


def emit_report_json(report: LockeReport) -> str:
    """Serialize a LockeReport to canonical JSON. Byte-identical across runs."""
    return _cjc_locke.emit_report_json(report)


def parse_report_json(json_str: str) -> LockeReport:
    """Parse a canonical-JSON-serialized LockeReport back into an object."""
    return _cjc_locke.parse_report_json(json_str)


# ── Convenience: .to_dict() on output objects ─────────────────────────────────
#
# We attach this to the output classes from Python so the Rust wrapper stays
# free of Python-dict-construction machinery. It's a cheap JSON round-trip —
# the canonical bytes are byte-identical to a native Rust call, then
# `json.loads` produces the dict.


def _locke_report_to_dict(self: LockeReport) -> Dict[str, Any]:
    return _json.loads(self.to_json())


def _induction_to_dict(self: InductionRiskReport) -> Dict[str, Any]:
    return _json.loads(self.to_json())


LockeReport.to_dict = _locke_report_to_dict  # type: ignore[attr-defined]
InductionRiskReport.to_dict = _induction_to_dict  # type: ignore[attr-defined]


# ── Module exports ────────────────────────────────────────────────────────────

__all__ = [
    # Functions
    "validate",
    "compare_drift",
    "validate_and_compare",
    "belief_report",
    "causal_guardrail",
    "detect_temporal_issues",
    "apply_policy",
    "make_audit_event",
    "emit_report_json",
    "parse_report_json",
    # Classes
    "LockeReport",
    "InductionRiskReport",
    "BeliefReport",
    "CausalGuardrailReport",
    "StreamingValidator",
    "LineageBuilder",
    "LineageGraph",
    "AuditEvent",
    "PolicyResult",
    "CustomDetector",
    "CustomDetectorDataFrame",
    "CustomDetectorSink",
    # Metadata
    "__version__",
    "__rust_crate_version__",
]
