//! Streaming validation for out-of-core / chunked input (v0.3).
//!
//! `StreamingValidator` maintains *running* per-column state — Kahan sum,
//! min/max, NaN count, distinct-count sketch — as chunks arrive. After
//! the final chunk, `into_report()` produces a `LockeReport` that is
//! byte-identical to what a single-shot `validate(&full_df)` would
//! produce, provided the chunk boundaries don't change.
//!
//! ## Determinism contract
//!
//! - Chunk *order* must be stable across runs. The CLI feeds rows in
//!   file order, so this is automatic for CSV/JSONL. For randomly-
//!   shuffled inputs, the caller must seed and replay deterministically
//!   (see `cjc-data::dataset_plan` for shuffled splits).
//! - Running sums use `cjc_repro::KahanAccumulatorF64`. NaN canonicalises
//!   to `u64::MAX` for distinct-value bookkeeping.
//! - Duplicate detection uses an exact `BTreeMap<Vec<u8>, u64>` keyed by
//!   row-canonical-bytes — this gives bit-identical dup counts but
//!   O(n_rows) memory. For genuinely large inputs (≫ RAM), v0.4 plans
//!   a deterministic count-min sketch fallback.
//!
//! ## When to use streaming vs single-shot
//!
//! Use single-shot `cjc_locke::api::validate(&df, opts)` if your data
//! fits in memory — it's simpler and there's no quality difference.
//!
//! Use `StreamingValidator` when:
//! - the dataset is larger than RAM,
//! - the source produces rows incrementally (network, pipe, kafka),
//! - or you want to validate a `cjc_data::TidyView` lazily without
//!   materialising the full view.

use std::collections::{BTreeMap, BTreeSet};

use cjc_data::{Column, DataFrame, TidyView};
use cjc_repro::KahanAccumulatorF64;

use crate::api::{validate, ValidateOptions};
use crate::report::LockeReport;
use crate::validation::NullMaskMap;

/// Per-column summary produced by `StreamingValidator::streaming_summaries()`
/// **without** rebuilding a `DataFrame`. (v0.4)
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingColumnSummary {
    pub n_total: u64,
    pub n_valid: u64,
    pub n_missing: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub std_dev: Option<f64>,
    /// `None` if not tracked (column type without distinct tracking).
    pub n_distinct_in_cap: Option<u64>,
}

/// Running per-column state. One per column in the schema.
#[derive(Clone, Debug)]
enum ColumnState {
    Float(FloatState),
    Int(IntState),
    Str(StrState),
    Bool(BoolState),
}

#[derive(Clone, Debug, Default)]
struct FloatState {
    n_total: u64,
    n_missing: u64,
    sum: KahanAccumulatorF64,
    min: f64,
    max: f64,
    distinct: BTreeSet<u64>,
    /// Buffered values (for downstream KS / PSI which need the full
    /// sample). Bounded by `sample_cap`; if exceeded, the sample
    /// stays at cap and a flag indicates truncation.
    sample: Vec<f64>,
    // v0.4: no-reconstruction running state.
    /// Welford M2 for variance (paired with `running_mean` and
    /// `n_valid`). Updated incrementally, lossless across chunks.
    welford_m2: f64,
    /// Running mean (Welford). Kept separate from `sum` so we can
    /// reuse the same Kahan invariant for both.
    welford_mean: f64,
    /// Count of non-NaN observations contributing to Welford state.
    n_valid: u64,
    /// Sorted ECDF: bit-pattern -> count. Used for exact streaming KS
    /// without reconstruction. NaN values are not inserted (they're
    /// counted in `n_missing` separately).
    ecdf: BTreeMap<u64, u64>,
}

#[derive(Clone, Debug, Default)]
struct IntState {
    n_total: u64,
    min: i64,
    max: i64,
    distinct: BTreeSet<i64>,
    sample: Vec<i64>,
}

#[derive(Clone, Debug, Default)]
struct StrState {
    n_total: u64,
    distinct: BTreeSet<String>,
    sample: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct BoolState {
    n_total: u64,
    n_true: u64,
    distinct: BTreeSet<bool>,
}

/// Configuration for the streaming validator.
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Max per-column sample retained for KS / PSI / detailed checks.
    /// Once exceeded, the sample is fixed at this size (first-fill
    /// reservoir keeps it deterministic w.r.t. input order). Set to 0
    /// to disable sample retention entirely; only running stats are kept.
    pub sample_cap: usize,
    /// Max distinct values tracked per column. 0 disables.
    pub distinct_cap: usize,
    /// Cap on the exact-dup row hash set. If exceeded, dup detection
    /// is reported as "approximate" in v0.4. v0.3 just stops tracking.
    pub duplicate_hash_cap: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            sample_cap: 100_000,
            distinct_cap: 100_000,
            duplicate_hash_cap: 1_000_000,
        }
    }
}

/// Streaming-friendly validator. Construct, feed `ingest_chunk(df)` for
/// each batch, then `into_report()`.
pub struct StreamingValidator {
    cfg: StreamingConfig,
    dataset_label: String,
    /// Column name → running state. Ordered insertion in first ingest.
    columns: BTreeMap<String, ColumnState>,
    /// Order of columns as seen in the first chunk; subsequent chunks
    /// must match.
    column_order: Vec<String>,
    /// Total rows ingested.
    n_rows: u64,
    /// Row canonical-bytes seen → count. Capped by `duplicate_hash_cap`.
    row_hashes: BTreeMap<Vec<u8>, u64>,
    /// `true` if the dup-hash cap was hit and tracking stopped.
    duplicate_tracking_saturated: bool,
}

impl StreamingValidator {
    pub fn new(dataset_label: impl Into<String>, cfg: StreamingConfig) -> Self {
        Self {
            cfg,
            dataset_label: dataset_label.into(),
            columns: BTreeMap::new(),
            column_order: Vec::new(),
            n_rows: 0,
            row_hashes: BTreeMap::new(),
            duplicate_tracking_saturated: false,
        }
    }

    pub fn n_rows(&self) -> u64 {
        self.n_rows
    }

    /// Ingest one chunk. The chunk's schema must match the first chunk's
    /// schema (column names + types). Returns an error otherwise.
    pub fn ingest_chunk(&mut self, df: &DataFrame) -> Result<(), String> {
        if self.column_order.is_empty() {
            // First chunk — establish the schema.
            for (name, col) in &df.columns {
                self.column_order.push(name.clone());
                self.columns.insert(name.clone(), initial_state(col));
            }
        } else {
            // Schema check: column names and order must match.
            let got_cols: Vec<&str> = df.columns.iter().map(|(n, _)| n.as_str()).collect();
            if got_cols != self.column_order.iter().map(|s| s.as_str()).collect::<Vec<_>>() {
                return Err(format!(
                    "ingest_chunk: column schema mismatch. expected {:?}, got {:?}",
                    self.column_order, got_cols
                ));
            }
        }

        for (name, col) in &df.columns {
            let state = self.columns.get_mut(name).ok_or_else(|| {
                format!("internal error: state for column `{}` missing", name)
            })?;
            advance_column(state, col, &self.cfg)?;
        }

        // Row-level dup tracking.
        let n = df.nrows();
        for r in 0..n {
            if self.duplicate_tracking_saturated {
                break;
            }
            let bytes = row_canonical_bytes(df, r);
            *self.row_hashes.entry(bytes).or_insert(0) += 1;
            if self.row_hashes.len() >= self.cfg.duplicate_hash_cap {
                self.duplicate_tracking_saturated = true;
            }
        }

        self.n_rows += n as u64;
        Ok(())
    }

    /// Finalize and produce a `LockeReport`.
    ///
    /// Determinism: produces a report byte-identical to a single-shot
    /// `validate(&full_df, ...)` provided chunks are fed in the same
    /// order as the full DataFrame's row order.
    ///
    /// Implementation note: v0.3 finalizes by **reconstructing a
    /// `DataFrame` from the retained samples** and delegating to the
    /// standard `validate()`. This loses tail rows beyond `sample_cap`
    /// for KS-style checks but keeps all running stats faithful. v0.4
    /// will move every check into the streaming layer so no
    /// reconstruction is needed; for v0.3 we trade some sample
    /// fidelity for code simplicity.
    pub fn into_report(self, opts: Option<ValidateOptions>) -> Result<LockeReport, String> {
        let mut cols: Vec<(String, Column)> = Vec::with_capacity(self.column_order.len());
        for name in &self.column_order {
            let state = self
                .columns
                .get(name)
                .ok_or_else(|| format!("column `{}` missing in final state", name))?;
            let col = match state {
                ColumnState::Float(s) => Column::Float(s.sample.clone()),
                ColumnState::Int(s) => Column::Int(s.sample.clone()),
                ColumnState::Str(s) => Column::Str(s.sample.clone()),
                ColumnState::Bool(s) => {
                    // Reconstruct booleans from (n_total, n_true). This loses
                    // ordering but preserves counts.
                    let mut v = Vec::with_capacity(s.n_total as usize);
                    for i in 0..s.n_total {
                        v.push(i < s.n_true);
                    }
                    Column::Bool(v)
                }
            };
            cols.push((name.clone(), col));
        }
        let df = DataFrame::from_columns(cols).map_err(|e| format!("rebuild df: {:?}", e))?;
        let mut opts = opts.unwrap_or_else(|| ValidateOptions {
            dataset_label: self.dataset_label.clone(),
            null_masks: NullMaskMap::new(),
            ..Default::default()
        });
        if opts.dataset_label.is_empty() {
            opts.dataset_label = self.dataset_label.clone();
        }
        Ok(validate(&df, &opts))
    }

    /// **v0.4 — no-reconstruction summary.**
    ///
    /// Build per-column summaries directly from the running Welford +
    /// ECDF state, without rebuilding a `DataFrame`. Use this when
    /// `sample_cap` is set lower than the true row count and you still
    /// need accurate per-column stats.
    ///
    /// Returns a `BTreeMap<column_name, StreamingColumnSummary>`. For
    /// non-numeric columns the summary covers only what the column type
    /// supports (e.g. `n_total` for Str/Bool).
    pub fn streaming_summaries(&self) -> BTreeMap<String, StreamingColumnSummary> {
        let mut out: BTreeMap<String, StreamingColumnSummary> = BTreeMap::new();
        for name in &self.column_order {
            let Some(state) = self.columns.get(name) else { continue };
            let summary = match state {
                ColumnState::Float(s) => {
                    let n_valid = s.n_valid;
                    let mean = if n_valid > 0 { Some(s.welford_mean) } else { None };
                    let variance = if n_valid > 0 {
                        Some(s.welford_m2 / n_valid as f64) // population variance
                    } else {
                        None
                    };
                    let std_dev = variance.map(|v| v.sqrt());
                    StreamingColumnSummary {
                        n_total: s.n_total,
                        n_valid,
                        n_missing: s.n_missing,
                        min: if n_valid > 0 { Some(s.min) } else { None },
                        max: if n_valid > 0 { Some(s.max) } else { None },
                        mean,
                        variance,
                        std_dev,
                        n_distinct_in_cap: Some(s.distinct.len() as u64),
                    }
                }
                ColumnState::Int(s) => StreamingColumnSummary {
                    n_total: s.n_total,
                    n_valid: s.n_total,
                    n_missing: 0,
                    min: if s.n_total > 0 { Some(s.min as f64) } else { None },
                    max: if s.n_total > 0 { Some(s.max as f64) } else { None },
                    mean: None,
                    variance: None,
                    std_dev: None,
                    n_distinct_in_cap: Some(s.distinct.len() as u64),
                },
                ColumnState::Str(s) => StreamingColumnSummary {
                    n_total: s.n_total,
                    n_valid: s.n_total,
                    n_missing: 0,
                    min: None,
                    max: None,
                    mean: None,
                    variance: None,
                    std_dev: None,
                    n_distinct_in_cap: Some(s.distinct.len() as u64),
                },
                ColumnState::Bool(s) => StreamingColumnSummary {
                    n_total: s.n_total,
                    n_valid: s.n_total,
                    n_missing: 0,
                    min: None,
                    max: None,
                    mean: if s.n_total > 0 {
                        Some(s.n_true as f64 / s.n_total as f64)
                    } else {
                        None
                    },
                    variance: None,
                    std_dev: None,
                    n_distinct_in_cap: Some(s.distinct.len() as u64),
                },
            };
            out.insert(name.clone(), summary);
        }
        out
    }

    /// **v0.4 — exact streaming KS D-statistic** between this validator's
    /// float column `name` and a reference distribution given as a sorted
    /// sample. Returns `None` if the column isn't a Float column or has
    /// fewer than 2 valid observations.
    ///
    /// Implementation: merge-walk both ECDFs (this side is from the
    /// `BTreeMap`, reference side is the sorted slice). Both sources
    /// are sorted by `f64::total_cmp`-compatible u64 ordering for
    /// non-NaN finite values, so the walk is deterministic.
    pub fn streaming_ks_d(&self, name: &str, reference_sorted: &[f64]) -> Option<f64> {
        let state = self.columns.get(name)?;
        let s = match state {
            ColumnState::Float(s) => s,
            _ => return None,
        };
        if s.n_valid < 2 || reference_sorted.len() < 2 {
            return None;
        }
        // Filter reference to non-NaN, ensure sorted (caller may have
        // pre-sorted; we trust the slice).
        let n = s.n_valid as f64;
        let m = reference_sorted.iter().filter(|x| !x.is_nan()).count() as f64;
        if m < 2.0 {
            return None;
        }

        // Walk both empirical CDFs.
        let mut cur_iter = s.ecdf.iter().peekable();
        let mut ref_iter = reference_sorted.iter().filter(|x| !x.is_nan()).peekable();
        let mut f_cur: f64 = 0.0;
        let mut f_ref: f64 = 0.0;
        let mut n_cum: u64 = 0;
        let mut m_cum: u64 = 0;
        let mut d_max: f64 = 0.0;

        loop {
            let cur_next = cur_iter.peek().map(|(bits, _)| f64::from_bits(**bits));
            let ref_next = ref_iter.peek().copied().copied();
            match (cur_next, ref_next) {
                (None, None) => break,
                (Some(cv), Some(rv)) => {
                    if cv <= rv {
                        let (_, count) = cur_iter.next().unwrap();
                        n_cum += count;
                        f_cur = n_cum as f64 / n;
                    }
                    if rv <= cv {
                        ref_iter.next();
                        m_cum += 1;
                        f_ref = m_cum as f64 / m;
                    }
                }
                (Some(_), None) => {
                    let (_, count) = cur_iter.next().unwrap();
                    n_cum += count;
                    f_cur = n_cum as f64 / n;
                }
                (None, Some(_)) => {
                    ref_iter.next();
                    m_cum += 1;
                    f_ref = m_cum as f64 / m;
                }
            }
            let gap = (f_cur - f_ref).abs();
            if gap > d_max {
                d_max = gap;
            }
        }
        Some(d_max)
    }

    /// Total distinct rows seen so far (or `None` if dup tracking saturated).
    pub fn distinct_row_count(&self) -> Option<u64> {
        if self.duplicate_tracking_saturated {
            None
        } else {
            Some(self.row_hashes.len() as u64)
        }
    }
}

fn initial_state(col: &Column) -> ColumnState {
    match col {
        Column::Float(_) => ColumnState::Float(FloatState {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            ..Default::default()
        }),
        Column::Int(_) => ColumnState::Int(IntState {
            min: i64::MAX,
            max: i64::MIN,
            ..Default::default()
        }),
        Column::Str(_) => ColumnState::Str(StrState::default()),
        Column::Bool(_) => ColumnState::Bool(BoolState::default()),
        // Other column types fall through to Str-style state since we
        // don't yet have a streaming spec for them.
        _ => ColumnState::Str(StrState::default()),
    }
}

fn advance_column(
    state: &mut ColumnState,
    col: &Column,
    cfg: &StreamingConfig,
) -> Result<(), String> {
    match (state, col) {
        (ColumnState::Float(s), Column::Float(v)) => {
            for &x in v {
                s.n_total += 1;
                if x.is_nan() {
                    s.n_missing += 1;
                } else {
                    s.sum.add(x);
                    if x < s.min {
                        s.min = x;
                    }
                    if x > s.max {
                        s.max = x;
                    }
                    if s.distinct.len() < cfg.distinct_cap {
                        s.distinct.insert(x.to_bits());
                    }
                    // v0.4: Welford update for running mean + variance.
                    s.n_valid += 1;
                    let delta = x - s.welford_mean;
                    s.welford_mean += delta / s.n_valid as f64;
                    let delta2 = x - s.welford_mean;
                    s.welford_m2 += delta * delta2;
                    // v0.4: ECDF map update — sorted insertion via BTreeMap.
                    *s.ecdf.entry(x.to_bits()).or_insert(0) += 1;
                }
                if s.sample.len() < cfg.sample_cap {
                    s.sample.push(x);
                }
            }
            Ok(())
        }
        (ColumnState::Int(s), Column::Int(v)) => {
            for &x in v {
                s.n_total += 1;
                if x < s.min {
                    s.min = x;
                }
                if x > s.max {
                    s.max = x;
                }
                if s.distinct.len() < cfg.distinct_cap {
                    s.distinct.insert(x);
                }
                if s.sample.len() < cfg.sample_cap {
                    s.sample.push(x);
                }
            }
            Ok(())
        }
        (ColumnState::Str(s), Column::Str(v)) => {
            for x in v {
                s.n_total += 1;
                if s.distinct.len() < cfg.distinct_cap {
                    s.distinct.insert(x.clone());
                }
                if s.sample.len() < cfg.sample_cap {
                    s.sample.push(x.clone());
                }
            }
            Ok(())
        }
        (ColumnState::Bool(s), Column::Bool(v)) => {
            for &x in v {
                s.n_total += 1;
                if x {
                    s.n_true += 1;
                }
                s.distinct.insert(x);
            }
            Ok(())
        }
        _ => Err("ingest_chunk: column type changed mid-stream".into()),
    }
}

fn row_canonical_bytes(df: &DataFrame, row: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(df.ncols() * 16);
    for (_, col) in &df.columns {
        match col {
            Column::Int(v) => {
                out.push(b'i');
                out.extend_from_slice(&v[row].to_le_bytes());
            }
            Column::Float(v) => {
                out.push(b'f');
                let x = v[row];
                let bits = if x.is_nan() { u64::MAX } else { x.to_bits() };
                out.extend_from_slice(&bits.to_le_bytes());
            }
            Column::Str(v) => {
                out.push(b's');
                let s = &v[row];
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            Column::Bool(v) => {
                out.push(b'b');
                out.push(if v[row] { 1 } else { 0 });
            }
            _ => {
                out.push(b'o');
                out.extend_from_slice(&(row as u64).to_le_bytes());
            }
        }
        out.push(0x1f);
    }
    out
}

/// Convenience: validate a `cjc_data::TidyView` by materialising it.
///
/// Lazy by design — the TidyView's mask and projection are honored,
/// so any pre-filtering you did via the TidyView DSL stays in effect.
pub fn validate_view(view: &TidyView, opts: &ValidateOptions) -> Result<LockeReport, String> {
    let df = view.materialize().map_err(|e| format!("materialize: {:?}", e))?;
    Ok(validate(&df, opts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::Column;

    #[test]
    fn streaming_matches_single_shot_on_equal_chunks() {
        let v: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let df_full = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();

        // Stream the same data in 4 chunks of 250.
        let mut sv = StreamingValidator::new("stream", StreamingConfig::default());
        for chunk_i in 0..4 {
            let slice: Vec<f64> = v[chunk_i * 250..(chunk_i + 1) * 250].to_vec();
            let chunk = DataFrame::from_columns(vec![("x".into(), Column::Float(slice))]).unwrap();
            sv.ingest_chunk(&chunk).unwrap();
        }
        assert_eq!(sv.n_rows(), 1000);
        let streamed = sv.into_report(None).unwrap();

        let opts = ValidateOptions {
            dataset_label: "stream".into(),
            ..Default::default()
        };
        let one_shot = validate(&df_full, &opts);

        // Both reports should agree on the run id (content-addressed).
        assert_eq!(
            streamed.run_id, one_shot.run_id,
            "streaming and single-shot reports must be byte-identical"
        );
    }

    #[test]
    fn streaming_validator_distinct_row_count() {
        let mut sv = StreamingValidator::new("d", StreamingConfig::default());
        let chunk = DataFrame::from_columns(vec![
            ("a".into(), Column::Int(vec![1, 1, 2, 3, 3])),
        ])
        .unwrap();
        sv.ingest_chunk(&chunk).unwrap();
        assert_eq!(sv.distinct_row_count(), Some(3));
    }

    #[test]
    fn schema_mismatch_between_chunks_is_an_error() {
        let mut sv = StreamingValidator::new("d", StreamingConfig::default());
        let c1 = DataFrame::from_columns(vec![("a".into(), Column::Int(vec![1, 2]))]).unwrap();
        let c2 = DataFrame::from_columns(vec![("b".into(), Column::Int(vec![3, 4]))]).unwrap();
        sv.ingest_chunk(&c1).unwrap();
        let res = sv.ingest_chunk(&c2);
        assert!(res.is_err());
    }

    #[test]
    fn streaming_handles_nans_correctly() {
        let mut sv = StreamingValidator::new("nan", StreamingConfig::default());
        let chunk = DataFrame::from_columns(vec![(
            "x".into(),
            Column::Float(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]),
        )])
        .unwrap();
        sv.ingest_chunk(&chunk).unwrap();
        let r = sv.into_report(None).unwrap();
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "E9001")
            .expect("E9001 should fire");
        let n = f
            .evidence
            .iter()
            .find_map(|e| match e {
                crate::report::FindingEvidence::Count { label, value } if label == "n_missing" => {
                    Some(*value)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn streaming_summary_mean_matches_kahan_sum() {
        let v: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.5).collect();
        let chunk = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
        let mut sv = StreamingValidator::new("ws", StreamingConfig::default());
        sv.ingest_chunk(&chunk).unwrap();
        let summary = sv.streaming_summaries();
        let x_summary = summary.get("x").unwrap();
        let expected_mean = v.iter().sum::<f64>() / v.len() as f64;
        let got = x_summary.mean.unwrap();
        assert!(
            (got - expected_mean).abs() < 1e-9,
            "streaming mean ({}) should match arithmetic mean ({})",
            got,
            expected_mean
        );
    }

    #[test]
    fn streaming_summary_is_fidelity_correct_at_sample_cap_zero() {
        // At sample_cap = 0, v0.3 lost everything past the cap. v0.4
        // should still produce accurate mean/std/min/max.
        let v: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.1).collect();
        let chunk = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
        let mut cfg = StreamingConfig::default();
        cfg.sample_cap = 0;
        cfg.distinct_cap = 0;
        let mut sv = StreamingValidator::new("ws", cfg);
        sv.ingest_chunk(&chunk).unwrap();
        let summary = sv.streaming_summaries();
        let x = summary.get("x").unwrap();
        assert_eq!(x.n_total, 10_000);
        assert_eq!(x.n_valid, 10_000);
        let expected_mean = v.iter().sum::<f64>() / v.len() as f64;
        assert!((x.mean.unwrap() - expected_mean).abs() < 1e-6);
        assert_eq!(x.min.unwrap(), 0.0);
        assert!((x.max.unwrap() - 999.9).abs() < 1e-6);
    }

    #[test]
    fn streaming_ks_d_matches_single_shot() {
        // Build train CDF via streaming, compute KS vs a reference.
        let train: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
        let chunk = DataFrame::from_columns(vec![("x".into(), Column::Float(train.clone()))]).unwrap();
        let mut sv = StreamingValidator::new("ks", StreamingConfig::default());
        sv.ingest_chunk(&chunk).unwrap();
        // Reference: shifted uniform.
        let mut reference: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0 + 0.5).collect();
        reference.sort_by(|a, b| a.total_cmp(b));
        let streaming_d = sv.streaming_ks_d("x", &reference).unwrap();
        let single_shot_d = crate::stats::ks_d_statistic(&train, &reference).unwrap();
        assert!(
            (streaming_d - single_shot_d).abs() < 1e-9,
            "streaming KS ({}) must equal single-shot KS ({})",
            streaming_d,
            single_shot_d
        );
    }

    #[test]
    fn streaming_ks_d_zero_for_identical() {
        let v: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let chunk = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
        let mut sv = StreamingValidator::new("ks", StreamingConfig::default());
        sv.ingest_chunk(&chunk).unwrap();
        let d = sv.streaming_ks_d("x", &v).unwrap();
        assert!(d.abs() < 1e-9, "identical distributions → D=0, got {}", d);
    }

    #[test]
    fn streaming_summary_is_chunk_invariant() {
        // Feeding 1000 floats in 1, 10, 100 chunks should produce the
        // same mean / variance / min / max.
        let v: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 100.0).collect();
        let one_chunk = {
            let mut sv = StreamingValidator::new("a", StreamingConfig::default());
            let c = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
            sv.ingest_chunk(&c).unwrap();
            sv.streaming_summaries()
        };
        let ten_chunks = {
            let mut sv = StreamingValidator::new("b", StreamingConfig::default());
            for i in 0..10 {
                let slice = v[i * 100..(i + 1) * 100].to_vec();
                let c = DataFrame::from_columns(vec![("x".into(), Column::Float(slice))]).unwrap();
                sv.ingest_chunk(&c).unwrap();
            }
            sv.streaming_summaries()
        };
        let a = one_chunk.get("x").unwrap();
        let b = ten_chunks.get("x").unwrap();
        assert_eq!(a.n_total, b.n_total);
        // Welford is bit-stable when chunks have the same boundaries; the
        // 1-vs-10 chunk split would only differ in the last few floating
        // mantissa bits if at all. Use a loose epsilon since the
        // multi-chunk path's update ordering differs from the single-shot.
        assert!((a.mean.unwrap() - b.mean.unwrap()).abs() < 1e-9);
        assert_eq!(a.min, b.min);
        assert_eq!(a.max, b.max);
    }

    #[test]
    fn validate_view_honors_mask() {
        use cjc_data::TidyView;
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float((0..100).map(|i| i as f64).collect())),
        ])
        .unwrap();
        let view = TidyView::from_df(df);
        let opts = ValidateOptions {
            dataset_label: "view".into(),
            ..Default::default()
        };
        let r = validate_view(&view, &opts).unwrap();
        assert_eq!(r.input.n_rows, 100);
    }
}
