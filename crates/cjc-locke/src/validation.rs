//! Data-skepticism validators.
//!
//! Each validator reads a `cjc_data::DataFrame` and emits a `Vec<ValidationFinding>`.
//! Validators never mutate input. They are deterministic and produce
//! findings whose IDs depend only on (code, column, evidence) — repeated
//! runs over the same data produce byte-identical IDs.
//!
//! ## Supported
//!
//! * `detect_missingness` — NaN-as-missing for floats; user-supplied
//!   `NullMask` (added v0.2) for non-float types. When no mask is supplied
//!   for a non-float column, a low-severity informational finding (E9002)
//!   acknowledges that missingness could not be inferred.
//! * `detect_duplicates_full_row` — full-row duplicates via canonical byte hashing.
//! * `detect_duplicate_keys` — duplicates restricted to a named key column.
//! * `detect_constant_and_near_constant` — `n_distinct == 1` and top-freq dominance.
//! * `detect_impossible_values` — user-supplied constraints (`ImpossibleValueRule`).
//! * `detect_high_cardinality_categorical` — heuristic categorical anomalies.
//! * `detect_schema_mismatch` — column-name and type comparison vs an expected schema.

use std::collections::{BTreeMap, BTreeSet};

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

// ─── Configuration ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ValidationConfig {
    pub near_constant_threshold: f64,
    pub high_cardinality_ratio: f64,
    pub duplicate_sample_limit: usize,
    /// **v0.6.4** — auto-detect common missing-value sentinels in `Str`
    /// columns (`?`, `NA`, `N/A`, `NULL`, `null`, `nan`, `NaN`, `None`,
    /// `-`, and empty string). When `true` (the default), every `Str`
    /// column is pre-scanned and matching rows are folded into the
    /// effective null mask before `detect_missingness` runs. Each
    /// affected column emits one `E9008` info finding naming the
    /// sentinel(s) and the missingness rate. Disable to restore v0.6.3
    /// behaviour (NaN-only for Float, mask-only for everything else).
    ///
    /// Motivated by the Phase 0.10 §4.D Part 1 finding: diabetes-130's
    /// `weight` column is 96.9% `?` but Locke's default missingness
    /// detector reported `missingness_score = 1.0000` ("perfect") until
    /// a `NullMaskMap` was hand-built.
    pub auto_detect_sentinels: bool,
    /// Custom sentinels added on top of the built-in list. Exact match
    /// after no transformation — case-sensitive, whitespace-significant.
    /// Use this to register dataset-specific conventions (e.g., `"."`
    /// in older SAS exports, `"unknown"` in spreadsheet data).
    pub additional_sentinels: Vec<String>,
    /// **v0.7+ (A2-by-default)** — when `true`, run
    /// [`crate::per_value_lineage::build_per_value_lineage`] during
    /// [`crate::api::validate`] and attach the result to
    /// [`crate::report::LockeReport::per_value_lineage`]. Default
    /// `false` so existing reports stay byte-identical to v0.7.
    ///
    /// Enable when investigating *which canonical form a value would
    /// take* under Locke's normalisation pipeline — without having to
    /// invoke `cjcl locke trace-value` per value. Cheap on small
    /// datasets (O(distinct_categorical_values)); use the
    /// `max_distinct_per_column` knob in `PerValueLineageConfig` if
    /// you want to bound a wide column.
    pub collect_per_value_lineage: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            near_constant_threshold: 0.99,
            high_cardinality_ratio: 0.5,
            duplicate_sample_limit: 5,
            // v0.6.4 — opt-in by default; the false-positive cost is
            // small (one info finding the user can read and ignore),
            // the false-negative cost is what §4.D Part 1 documented
            // (a 97%-missing column reported as perfectly clean).
            auto_detect_sentinels: true,
            additional_sentinels: Vec::new(),
            // v0.7+ — opt-in, default false to preserve byte-identical
            // reports for existing CI gates. CLI exposes `--with-trace`.
            collect_per_value_lineage: false,
        }
    }
}

/// **v0.6.4** — built-in list of recognised string sentinels. Frozen.
/// Add custom values via `ValidationConfig.additional_sentinels`.
/// Order is significant for canonicalisation of the audit message.
pub const BUILTIN_STRING_SENTINELS: &[&str] = &[
    "?",       // UCI convention (diabetes-130, etc.)
    "NA",      // R convention
    "N/A",     // forms / spreadsheets
    "NULL",    // SQL convention (upper)
    "null",    // SQL convention (lower)
    "nan",     // pandas / numpy stringified
    "NaN",     // ditto, capitalised
    "None",    // Python stringified
    "-",       // dash placeholder
    "",        // empty string
];

// ─── Null masks (v0.2) ──────────────────────────────────────────────────────

/// A sparse set of row indices that should be treated as null for a column.
///
/// Used for `Column::Int`, `Column::Bool`, `Column::Str`, `Column::Categorical`,
/// `Column::DateTime` — types where the column itself has no native null
/// sentinel. For `Column::Float`, NaN is the canonical null and a mask is
/// usually unnecessary; if a float column has both NaN values *and* a mask,
/// the union is treated as missing.
///
/// Determinism: `BTreeSet` iterates in sorted order, so the resulting
/// findings have stable IDs regardless of how the caller built the mask.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NullMask {
    pub null_rows: BTreeSet<usize>,
}

impl NullMask {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_indices<I: IntoIterator<Item = usize>>(rows: I) -> Self {
        Self {
            null_rows: rows.into_iter().collect(),
        }
    }
    pub fn count(&self) -> u64 {
        self.null_rows.len() as u64
    }
    pub fn contains(&self, row: usize) -> bool {
        self.null_rows.contains(&row)
    }
}

/// Convenience alias used in [`ValidateOptions`].
pub type NullMaskMap = BTreeMap<String, NullMask>;

/// **v0.6.4** — pre-scan `Str` columns for common missing-value
/// sentinels and build a `NullMaskMap` keyed by column name. Returns
/// the mask alongside a `Vec<ValidationFinding>` of [`E9008`]
/// info-severity findings — one per `Str` column where at least one
/// sentinel matched.
///
/// Sentinels recognised: [`BUILTIN_STRING_SENTINELS`] plus any custom
/// values supplied in `cfg.additional_sentinels`. Matching is exact:
/// no trimming, no case folding. Use `cfg.additional_sentinels` to
/// register dataset conventions.
///
/// When `cfg.auto_detect_sentinels` is `false`, returns an empty mask
/// and an empty findings vector — backward-compatible with v0.6.3
/// behaviour.
///
/// # Composition
///
/// The intent is for the caller to union this mask with any
/// user-supplied [`NullMaskMap`] before invoking
/// [`detect_missingness`] or [`detect_conditional_missingness`]. The
/// canonical caller is [`crate::api::validate`], which performs that
/// union once and passes the combined map throughout the validator
/// pipeline.
///
/// # Determinism
///
/// Iteration is in `BTreeMap` order (`df.columns` is sorted; output
/// `NullMask.null_rows` is a `BTreeSet`). Findings are appended in
/// column-name order. Two runs over the same `DataFrame` produce
/// byte-identical `ValidationFinding` IDs.
pub fn detect_string_sentinels(
    df: &DataFrame,
    cfg: &ValidationConfig,
) -> (NullMaskMap, Vec<ValidationFinding>) {
    let mut masks = NullMaskMap::new();
    let mut findings = Vec::new();
    if !cfg.auto_detect_sentinels {
        return (masks, findings);
    }

    let n_rows = df.nrows() as u64;
    for (name, col) in &df.columns {
        let Column::Str(values) = col else { continue };

        // Per-sentinel hit indices, kept in a BTreeMap for deterministic
        // sample ordering in the audit message.
        let mut hits: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for (i, v) in values.iter().enumerate() {
            let s = v.as_str();
            let is_builtin = BUILTIN_STRING_SENTINELS.iter().any(|b| *b == s);
            let is_custom = cfg.additional_sentinels.iter().any(|c| c == s);
            if is_builtin || is_custom {
                hits.entry(s.to_string()).or_default().insert(i);
            }
        }
        if hits.is_empty() {
            continue;
        }

        // Union the per-sentinel sets into one mask. BTreeSet keeps
        // null_rows sorted; the resulting NullMask is byte-canonical.
        let union: BTreeSet<usize> = hits
            .values()
            .flat_map(|s| s.iter().copied())
            .collect();
        let n_missing = union.len() as u64;
        let rate = if n_rows == 0 {
            0.0
        } else {
            n_missing as f64 / n_rows as f64
        };

        // Deterministic, human-readable breakdown showing up to 3
        // sentinels and their counts.
        let breakdown = hits
            .iter()
            .take(3)
            .map(|(s, idx)| format!("{:?}: {}", s, idx.len()))
            .collect::<Vec<_>>()
            .join(", ");

        masks.insert(name.clone(), NullMask { null_rows: union });
        findings.push(ValidationFinding::new(
            "E9008",
            FindingSeverity::Info,
            format!(
                "auto-detected {} string-sentinel value(s) in `{}` ({:.1}% of rows); folded into the effective null mask",
                n_missing, name, rate * 100.0
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_sentinel_rows".into(),
                    value: n_missing,
                },
                FindingEvidence::Ratio {
                    label: "sentinel_rate".into(),
                    value: rate,
                },
                FindingEvidence::Sample {
                    label: "sentinels_seen".into(),
                    value: breakdown,
                },
            ],
            n_rows,
            vec![
                "Built-in sentinels: ?, NA, N/A, NULL, null, nan, NaN, None, -, ``".into(),
                "Set ValidationConfig.auto_detect_sentinels = false to disable".into(),
                "Use ValidationConfig.additional_sentinels for dataset-specific conventions".into(),
            ],
            vec![
                "Confirm the sentinel encodes 'missing' here — some columns legitimately use `-` or `NA`".into(),
                "If false positive, opt out and supply a manual NullMaskMap".into(),
            ],
        ));
    }
    (masks, findings)
}

/// **v0.6.4** — union two `NullMaskMap`s, key by key. For columns
/// present in both, the resulting `NullMask.null_rows` is the
/// set-union of both inputs.
pub fn merge_null_mask_maps(a: &NullMaskMap, b: &NullMaskMap) -> NullMaskMap {
    let mut out: NullMaskMap = a.clone();
    for (k, mask_b) in b {
        let entry = out.entry(k.clone()).or_default();
        for r in &mask_b.null_rows {
            entry.null_rows.insert(*r);
        }
    }
    out
}

// ─── Impossible-value constraint DSL ────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum ImpossibleValueRule {
    /// f64 value < min or > max is impossible.
    NumericRange { column: String, min: f64, max: f64 },
    /// f64 value not in the listed set is impossible.
    AllowedFloats { column: String, allowed: Vec<f64> },
    /// i64 value not in the listed set is impossible.
    AllowedInts { column: String, allowed: Vec<i64> },
    /// String value not in the listed set is impossible.
    AllowedStrings { column: String, allowed: BTreeSet<String> },
    /// f64 value must be non-negative.
    NonNegative { column: String },
    /// f64 / i64 value must be strictly positive.
    StrictlyPositive { column: String },
}

impl ImpossibleValueRule {
    pub fn column(&self) -> &str {
        match self {
            ImpossibleValueRule::NumericRange { column, .. }
            | ImpossibleValueRule::AllowedFloats { column, .. }
            | ImpossibleValueRule::AllowedInts { column, .. }
            | ImpossibleValueRule::AllowedStrings { column, .. }
            | ImpossibleValueRule::NonNegative { column }
            | ImpossibleValueRule::StrictlyPositive { column } => column,
        }
    }
}

// ─── Schema expectation ─────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct ExpectedSchema {
    /// Required column names with expected type tags. Type tag uses
    /// `Column::type_name()` strings: "Int", "Float", "Str", "Bool",
    /// "Categorical", "CategoricalAdaptive", "DateTime".
    pub columns: BTreeMap<String, String>,
    /// If `true`, extra columns in the actual frame are an error;
    /// otherwise they're a notice.
    pub strict_extra: bool,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn nan_count(values: &[f64]) -> u64 {
    values.iter().filter(|v| v.is_nan()).count() as u64
}

/// Canonical byte rep of one column cell. Used to hash rows for duplicate
/// detection. NaN is canonicalised so two NaNs hash the same.
fn cell_bytes(col: &Column, row: usize, out: &mut Vec<u8>) {
    match col {
        Column::Int(v) => {
            out.push(b'i');
            out.extend_from_slice(&v[row].to_le_bytes());
        }
        Column::Float(v) => {
            out.push(b'f');
            let x = v[row];
            // Canonicalise NaN: any NaN bit pattern becomes the same value.
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
        Column::Categorical { levels, codes } => {
            out.push(b'c');
            let code = codes[row] as usize;
            let level = levels.get(code).map(|s| s.as_str()).unwrap_or("");
            out.extend_from_slice(&(level.len() as u64).to_le_bytes());
            out.extend_from_slice(level.as_bytes());
        }
        Column::CategoricalAdaptive(cc) => {
            // v0.7+ deep-dive bug-fix (was CRITICAL): previously this branch
            // wrote `(b'a', row_index_bytes)` so every row produced a unique
            // canonical hash. As a result, `detect_duplicates_full_row` and
            // `detect_duplicate_keys` *silently never fired E9003/E9004* on
            // any DataFrame containing a CategoricalAdaptive column — a 100%
            // false-negative rate on duplicate detection for the storage
            // variant used by the diabetes-130 high-cardinality columns.
            //
            // The fix dereferences the dictionary to the same byte stream the
            // plain Categorical branch uses. The `b'c'` tag is shared with
            // Column::Categorical so equal content produces equal fingerprints
            // across the two storage variants — the user shouldn't get
            // different finding IDs just because they chose adaptive encoding.
            out.push(b'c');
            let code = cc.codes().get(row);
            let bytes = cc.dictionary().get(code).unwrap_or(&[]);
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        Column::DateTime(v) => {
            out.push(b'd');
            out.extend_from_slice(&v[row].to_le_bytes());
        }
    }
}

fn row_canonical_bytes(df: &DataFrame, row: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(df.ncols() * 16);
    for (_, col) in &df.columns {
        cell_bytes(col, row, &mut out);
        out.push(0x1f);
    }
    out
}

// ─── Missingness ────────────────────────────────────────────────────────────

/// Detect missingness column-by-column, honoring an optional user-supplied
/// null mask for non-float columns (v0.2).
///
/// For `Column::Float`, NaN is missing. If a null mask is also supplied
/// for a float column, the **union** of NaN positions and mask positions
/// counts as missing.
///
/// For other column types, missingness is **only** detected if the caller
/// supplies a `NullMask` in `null_masks` for that column. Without a mask,
/// Locke emits an Info-level E9002 acknowledging that missingness could
/// not be inferred.
///
/// Out-of-bounds mask indices (≥ column length) are surfaced as E9006
/// (Warning) and the offending indices are skipped.
pub fn detect_missingness(
    df: &DataFrame,
    _cfg: &ValidationConfig,
    null_masks: &NullMaskMap,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;

    for (name, col) in &df.columns {
        let col_len = col.len();
        let mask = null_masks.get(name);

        // Validate the mask bounds, if present.
        let (clean_mask_count, oob_count) = if let Some(m) = mask {
            let oob: Vec<usize> = m
                .null_rows
                .iter()
                .copied()
                .filter(|i| *i >= col_len)
                .collect();
            if !oob.is_empty() {
                let sample = oob
                    .iter()
                    .take(3)
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                out.push(ValidationFinding::new(
                    "E9006",
                    FindingSeverity::Warning,
                    format!(
                        "null mask for `{}` contains {} out-of-bounds index(es); these will be ignored",
                        name,
                        oob.len()
                    ),
                    Some(name.clone()),
                    None,
                    vec![
                        FindingEvidence::Count {
                            label: "out_of_bounds_indices".into(),
                            value: oob.len() as u64,
                        },
                        FindingEvidence::Count {
                            label: "column_length".into(),
                            value: col_len as u64,
                        },
                        FindingEvidence::Sample {
                            label: "sample_oob".into(),
                            value: sample,
                        },
                    ],
                    col_len as u64,
                    vec![
                        "out-of-bounds indices indicate a caller bug — mask was built against a different column length".into(),
                    ],
                    vec![
                        "verify the mask is built against the same DataFrame snapshot used for validation".into(),
                    ],
                ));
            }
            let clean = m.null_rows.iter().filter(|i| **i < col_len).count() as u64;
            (clean, oob.len() as u64)
        } else {
            (0, 0)
        };
        let _ = oob_count; // already surfaced above

        match col {
            Column::Float(v) => {
                let nan_n = nan_count(v);
                // v0.7+ deep-dive perf-fix: previously this branch built a
                // BTreeSet<usize> of every NaN row index just to count its
                // size — O(n log n) inserts + O(n) memory for a single
                // u64 result. For a 90%-NaN column with masking, that was
                // 0.9n usize entries allocated and dropped immediately.
                //
                // The mask `null_rows` is a BTreeSet<usize>, so we can
                // compute the union count with a single column walk +
                // mask lookup, no intermediate set allocation.
                let n_miss = if let Some(m) = mask {
                    let mut count = 0u64;
                    for (i, x) in v.iter().enumerate() {
                        // Count row i as missing if either NaN or
                        // present in the mask (mask lookup is O(log m)).
                        if x.is_nan() || m.null_rows.contains(&i) {
                            count += 1;
                        }
                    }
                    count
                } else {
                    nan_n
                };

                if n_miss > 0 {
                    let rate = n_miss as f64 / n_rows.max(1) as f64;
                    let severity = if rate >= 0.5 {
                        FindingSeverity::Error
                    } else if rate >= 0.1 {
                        FindingSeverity::Warning
                    } else {
                        FindingSeverity::Notice
                    };
                    let mut assumptions = vec!["NaN values treated as missing".into()];
                    if mask.is_some() {
                        assumptions
                            .push("user-supplied null mask combined with NaN positions (union)".into());
                    }
                    out.push(ValidationFinding::new(
                        "E9001",
                        severity,
                        format!("{} of {} values in `{}` are missing", n_miss, n_rows, name),
                        Some(name.clone()),
                        None,
                        vec![
                            FindingEvidence::Count {
                                label: "n_missing".into(),
                                value: n_miss,
                            },
                            FindingEvidence::Ratio {
                                label: "missingness_rate".into(),
                                value: rate,
                            },
                        ],
                        n_rows,
                        assumptions,
                        vec![
                            "inspect a sample of missing rows".into(),
                            "decide imputation strategy or drop policy".into(),
                        ],
                    ));
                }
            }
            Column::Int(_) | Column::Bool(_) | Column::Str(_) | Column::DateTime(_)
            | Column::Categorical { .. } | Column::CategoricalAdaptive(_) => {
                if let Some(_m) = mask {
                    let n_miss = clean_mask_count;
                    if n_miss > 0 {
                        let rate = n_miss as f64 / n_rows.max(1) as f64;
                        let severity = if rate >= 0.5 {
                            FindingSeverity::Error
                        } else if rate >= 0.1 {
                            FindingSeverity::Warning
                        } else {
                            FindingSeverity::Notice
                        };
                        out.push(ValidationFinding::new(
                            "E9001",
                            severity,
                            format!(
                                "{} of {} values in `{}` are missing (via null mask)",
                                n_miss, n_rows, name
                            ),
                            Some(name.clone()),
                            None,
                            vec![
                                FindingEvidence::Count {
                                    label: "n_missing".into(),
                                    value: n_miss,
                                },
                                FindingEvidence::Ratio {
                                    label: "missingness_rate".into(),
                                    value: rate,
                                },
                                FindingEvidence::Sample {
                                    label: "type".into(),
                                    value: col.type_name().into(),
                                },
                            ],
                            n_rows,
                            vec![
                                "missingness derived from caller-supplied NullMask".into(),
                            ],
                            vec![
                                "decide imputation strategy or drop policy".into(),
                            ],
                        ));
                    }
                } else {
                    // No mask supplied: emit the legacy E9002 acknowledgement
                    // so missingness is still surfaced (at Info severity).
                    out.push(ValidationFinding::new(
                        "E9002",
                        FindingSeverity::Info,
                        format!(
                            "column `{}` is of type `{}`; no null mask supplied so missingness cannot be inferred",
                            name,
                            col.type_name()
                        ),
                        Some(name.clone()),
                        None,
                        vec![FindingEvidence::Sample {
                            label: "type".into(),
                            value: col.type_name().into(),
                        }],
                        n_rows,
                        vec![
                            "Locke treats only Float NaN as missing without a null mask".into(),
                            "supply ValidateOptions.null_masks to mark null rows explicitly".into(),
                        ],
                        vec![
                            "if missingness matters here, build a NullMask::from_indices(...) for this column".into(),
                        ],
                    ));
                }
            }
        }
    }

    out
}

// ─── Duplicates ─────────────────────────────────────────────────────────────

pub fn detect_duplicates_full_row(df: &DataFrame, cfg: &ValidationConfig) -> Vec<ValidationFinding> {
    let n_rows = df.nrows();
    if n_rows == 0 {
        return vec![];
    }
    // Bucket rows by canonical-byte fingerprint into a BTreeMap so output
    // ordering is deterministic.
    let mut buckets: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for r in 0..n_rows {
        buckets.entry(row_canonical_bytes(df, r)).or_default().push(r);
    }
    let mut total_dupes: u64 = 0;
    let mut sample_rows: Vec<u64> = Vec::new();
    let mut groups: u64 = 0;
    for (_k, rows) in &buckets {
        if rows.len() > 1 {
            groups += 1;
            total_dupes += (rows.len() - 1) as u64;
            for r in rows.iter().take(cfg.duplicate_sample_limit) {
                sample_rows.push(*r as u64);
            }
        }
    }
    if total_dupes == 0 {
        return vec![];
    }
    let rate = total_dupes as f64 / n_rows as f64;
    let mut sample_str = String::new();
    for (i, r) in sample_rows.iter().take(cfg.duplicate_sample_limit).enumerate() {
        if i > 0 {
            sample_str.push(',');
        }
        sample_str.push_str(&r.to_string());
    }
    let severity = if rate >= 0.2 {
        FindingSeverity::Error
    } else if rate >= 0.05 {
        FindingSeverity::Warning
    } else {
        FindingSeverity::Notice
    };
    vec![ValidationFinding::new(
        "E9003",
        severity,
        format!("{} duplicate rows across {} groups", total_dupes, groups),
        None,
        None,
        vec![
            FindingEvidence::Count {
                label: "duplicate_rows".into(),
                value: total_dupes,
            },
            FindingEvidence::Ratio {
                label: "duplicate_rate".into(),
                value: rate,
            },
            FindingEvidence::Sample {
                label: "sample_row_indices".into(),
                value: sample_str,
            },
        ],
        n_rows as u64,
        vec!["row equality is byte-canonical; NaN equals NaN under this comparison".into()],
        vec![
            "decide whether duplicates are erroneous re-ingest or legitimate repeated observations".into(),
            "consider df.distinct() or grouping with a primary key".into(),
        ],
    )]
}

pub fn detect_duplicate_keys(df: &DataFrame, key_column: &str) -> Vec<ValidationFinding> {
    let col = match df.get_column(key_column) {
        Some(c) => c,
        None => {
            return vec![ValidationFinding::new(
                "E9005",
                FindingSeverity::Error,
                format!("key column `{}` not found in dataframe", key_column),
                Some(key_column.into()),
                None,
                vec![],
                df.nrows() as u64,
                vec![],
                vec!["verify the column name and re-run".into()],
            )];
        }
    };
    let n = col.len();
    let mut buckets: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for r in 0..n {
        let mut bytes = Vec::new();
        cell_bytes(col, r, &mut bytes);
        buckets.entry(bytes).or_default().push(r);
    }
    let mut dups: u64 = 0;
    let mut groups: u64 = 0;
    for (_, rows) in &buckets {
        if rows.len() > 1 {
            groups += 1;
            dups += (rows.len() - 1) as u64;
        }
    }
    if dups == 0 {
        return vec![];
    }
    vec![ValidationFinding::new(
        "E9004",
        FindingSeverity::Error,
        format!(
            "key column `{}` has {} duplicate values across {} groups",
            key_column, dups, groups
        ),
        Some(key_column.into()),
        None,
        vec![
            FindingEvidence::Count {
                label: "duplicate_keys".into(),
                value: dups,
            },
            FindingEvidence::Count {
                label: "duplicate_groups".into(),
                value: groups,
            },
        ],
        n as u64,
        vec!["key uniqueness assumed; primary-key constraint not enforced by cjc-data".into()],
        vec!["check ingest pipeline; primary keys should be unique".into()],
    )]
}

// ─── Constant / near-constant ───────────────────────────────────────────────

pub(crate) fn distinct_count(col: &Column) -> u64 {
    match col {
        Column::Int(v) => {
            let set: BTreeSet<i64> = v.iter().copied().collect();
            set.len() as u64
        }
        Column::Float(v) => {
            // Bit-pattern dedup, with NaN canonicalised so all NaNs count once.
            let set: BTreeSet<u64> = v
                .iter()
                .map(|x| if x.is_nan() { u64::MAX } else { x.to_bits() })
                .collect();
            set.len() as u64
        }
        Column::Str(v) => {
            let set: BTreeSet<&String> = v.iter().collect();
            set.len() as u64
        }
        Column::Bool(v) => {
            let set: BTreeSet<bool> = v.iter().copied().collect();
            set.len() as u64
        }
        Column::Categorical { codes, .. } => {
            let set: BTreeSet<u32> = codes.iter().copied().collect();
            set.len() as u64
        }
        // v0.7+ deep-dive bug-fix (was CRITICAL): previously this returned
        // `cc.len()` (the row count, commented "conservative upper bound").
        // That made the `distinct <= 1` check in E9010 unreachable for
        // adaptive-categorical columns, and made `detect_id_like_columns`
        // (E9072 in leakage.rs) compute `distinct/n_rows = 1.0` for EVERY
        // such column — flagging every adaptive-categorical column as an
        // ID-like leakage candidate. Replaced with the actual distinct
        // level count via the dictionary API.
        Column::CategoricalAdaptive(cc) => cc.dictionary().len() as u64,
        Column::DateTime(v) => {
            let set: BTreeSet<i64> = v.iter().copied().collect();
            set.len() as u64
        }
    }
}

/// Top-value frequency for a column (max count of any single value).
pub(crate) fn top_value_freq(col: &Column) -> u64 {
    match col {
        Column::Int(v) => count_top(v.iter().copied()),
        Column::Float(v) => count_top(
            v.iter()
                .map(|x| if x.is_nan() { u64::MAX } else { x.to_bits() }),
        ),
        // v0.7+ deep-dive perf-fix: was cloning every String into a Vec
        // before counting. For a 1M-row Str column with 10 distinct values
        // that's 1M wasted clones. Counting by `&str` keeps the lifetime
        // tied to the column and zero-allocates the keys.
        Column::Str(v) => {
            let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
            for s in v {
                *counts.entry(s.as_str()).or_insert(0) += 1;
            }
            counts.values().copied().max().unwrap_or(0)
        }
        Column::Bool(v) => count_top(v.iter().copied()),
        Column::Categorical { codes, .. } => count_top(codes.iter().copied()),
        Column::DateTime(v) => count_top(v.iter().copied()),
        // v0.7+ deep-dive bug-fix (was HIGH): previously returned 0
        // unconditionally for adaptive-categorical, which made
        // `detect_constant_and_near_constant` always report 0 frequency on
        // the top value → no near-constant findings could ever fire on
        // these columns. Now properly counts via the code stream.
        Column::CategoricalAdaptive(cc) => {
            let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
            for code in cc.codes().iter() {
                *counts.entry(code).or_insert(0) += 1;
            }
            counts.values().copied().max().unwrap_or(0)
        }
    }
}

fn count_top<T: Ord, I: IntoIterator<Item = T>>(values: I) -> u64 {
    let mut counts: BTreeMap<T, u64> = BTreeMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    counts.values().copied().max().unwrap_or(0)
}

pub fn detect_constant_and_near_constant(
    df: &DataFrame,
    cfg: &ValidationConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    for (name, col) in &df.columns {
        let n = col.len() as u64;
        if n == 0 {
            continue;
        }
        let distinct = distinct_count(col);
        if distinct <= 1 {
            out.push(ValidationFinding::new(
                "E9010",
                FindingSeverity::Warning,
                format!("column `{}` is constant", name),
                Some(name.clone()),
                None,
                vec![FindingEvidence::Count {
                    label: "distinct".into(),
                    value: distinct,
                }],
                n_rows,
                vec!["constant features carry no predictive signal".into()],
                vec!["drop the column or check whether the ingest is correct".into()],
            ));
            continue;
        }
        let top = top_value_freq(col);
        let ratio = top as f64 / n as f64;
        if ratio >= cfg.near_constant_threshold {
            out.push(ValidationFinding::new(
                "E9011",
                FindingSeverity::Notice,
                format!(
                    "column `{}` is near-constant: top value occupies {:.1}% of rows",
                    name,
                    ratio * 100.0
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Ratio {
                        label: "top_freq_ratio".into(),
                        value: ratio,
                    },
                    FindingEvidence::Count {
                        label: "distinct".into(),
                        value: distinct,
                    },
                ],
                n_rows,
                vec![
                    format!("near-constant threshold {:.2}", cfg.near_constant_threshold)
                ],
                vec![
                    "consider whether this column carries useful signal".into(),
                    "examine the rare-value rows manually".into(),
                ],
            ));
        }
    }
    out
}

// ─── Impossible values ──────────────────────────────────────────────────────

pub fn detect_impossible_values(
    df: &DataFrame,
    rules: &[ImpossibleValueRule],
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    for rule in rules {
        let col = match df.get_column(rule.column()) {
            Some(c) => c,
            None => {
                out.push(ValidationFinding::new(
                    "E9012",
                    FindingSeverity::Error,
                    format!(
                        "impossible-value rule references missing column `{}`",
                        rule.column()
                    ),
                    Some(rule.column().into()),
                    None,
                    vec![],
                    0,
                    vec![],
                    vec!["correct the rule or the schema".into()],
                ));
                continue;
            }
        };
        let n = col.len() as u64;
        let mut violations: u64 = 0;
        let mut sample = String::new();
        let mut samples_added = 0;
        match (rule, col) {
            (ImpossibleValueRule::NumericRange { min, max, .. }, Column::Float(v)) => {
                for (i, x) in v.iter().enumerate() {
                    if x.is_nan() {
                        continue;
                    }
                    if x < min || x > max {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::NumericRange { min, max, .. }, Column::Int(v)) => {
                let (mn, mx) = (*min, *max);
                for (i, x) in v.iter().enumerate() {
                    let f = *x as f64;
                    if f < mn || f > mx {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::NonNegative { .. }, Column::Float(v)) => {
                for (i, x) in v.iter().enumerate() {
                    if x.is_nan() {
                        continue;
                    }
                    if *x < 0.0 {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::NonNegative { .. }, Column::Int(v)) => {
                for (i, x) in v.iter().enumerate() {
                    if *x < 0 {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::StrictlyPositive { .. }, Column::Float(v)) => {
                for (i, x) in v.iter().enumerate() {
                    if x.is_nan() {
                        continue;
                    }
                    if *x <= 0.0 {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::StrictlyPositive { .. }, Column::Int(v)) => {
                for (i, x) in v.iter().enumerate() {
                    if *x <= 0 {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::AllowedFloats { allowed, .. }, Column::Float(v)) => {
                let mut allowed_bits: BTreeSet<u64> = BTreeSet::new();
                for &a in allowed {
                    let bits = if a.is_nan() { u64::MAX } else { a.to_bits() };
                    allowed_bits.insert(bits);
                }
                for (i, x) in v.iter().enumerate() {
                    if x.is_nan() {
                        continue;
                    }
                    if !allowed_bits.contains(&x.to_bits()) {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::AllowedInts { allowed, .. }, Column::Int(v)) => {
                let allowed_set: BTreeSet<i64> = allowed.iter().copied().collect();
                for (i, x) in v.iter().enumerate() {
                    if !allowed_set.contains(x) {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={}", i, x));
                            samples_added += 1;
                        }
                    }
                }
            }
            (ImpossibleValueRule::AllowedStrings { allowed, .. }, Column::Str(v)) => {
                for (i, s) in v.iter().enumerate() {
                    if !allowed.contains(s) {
                        violations += 1;
                        if samples_added < 3 {
                            if samples_added > 0 {
                                sample.push(',');
                            }
                            sample.push_str(&format!("row{}={:?}", i, s));
                            samples_added += 1;
                        }
                    }
                }
            }
            (rule, col) => {
                out.push(ValidationFinding::new(
                    "E9013",
                    FindingSeverity::Warning,
                    format!(
                        "rule `{:?}` is not applicable to column `{}` of type `{}` (skipped)",
                        std::mem::discriminant(rule),
                        rule.column(),
                        col.type_name()
                    ),
                    Some(rule.column().into()),
                    None,
                    vec![],
                    n,
                    vec!["rule-column type mismatch".into()],
                    vec!["choose a rule appropriate for the column type".into()],
                ));
                continue;
            }
        }

        if violations > 0 {
            let rate = violations as f64 / n.max(1) as f64;
            let severity = if rate >= 0.10 {
                FindingSeverity::Error
            } else if rate >= 0.01 {
                FindingSeverity::Warning
            } else {
                FindingSeverity::Notice
            };
            out.push(ValidationFinding::new(
                "E9014",
                severity,
                format!(
                    "{} impossible values in `{}` ({:.2}% of rows)",
                    violations,
                    rule.column(),
                    rate * 100.0
                ),
                Some(rule.column().into()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "violations".into(),
                        value: violations,
                    },
                    FindingEvidence::Ratio {
                        label: "violation_rate".into(),
                        value: rate,
                    },
                    FindingEvidence::Sample {
                        label: "sample".into(),
                        value: sample,
                    },
                ],
                n,
                vec!["constraint defined by caller; Locke does not infer constraints".into()],
                vec![
                    "fix at source if possible".into(),
                    "decide whether to filter or impute these rows".into(),
                ],
            ));
        }
    }
    out
}

// ─── High-cardinality categorical ───────────────────────────────────────────

pub fn detect_high_cardinality_categorical(
    df: &DataFrame,
    cfg: &ValidationConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    for (name, col) in &df.columns {
        let (level_count, kind_tag) = match col {
            Column::Categorical { levels, .. } => (levels.len() as u64, "Categorical"),
            Column::Str(v) => {
                let s: BTreeSet<&String> = v.iter().collect();
                (s.len() as u64, "Str")
            }
            // v0.6.3: CategoricalAdaptive — read dictionary size directly.
            Column::CategoricalAdaptive(cc) => (cc.dictionary().len() as u64, "CategoricalAdaptive"),
            _ => continue,
        };
        let ratio = level_count as f64 / n_rows.max(1) as f64;
        if ratio >= cfg.high_cardinality_ratio && n_rows >= 10 {
            out.push(ValidationFinding::new(
                "E9015",
                FindingSeverity::Notice,
                format!(
                    "column `{}` has {} distinct {} values out of {} rows ({:.0}%)",
                    name,
                    level_count,
                    kind_tag,
                    n_rows,
                    ratio * 100.0
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_distinct".into(),
                        value: level_count,
                    },
                    FindingEvidence::Ratio {
                        label: "cardinality_ratio".into(),
                        value: ratio,
                    },
                ],
                n_rows,
                vec!["high cardinality may indicate IDs leaking as features".into()],
                vec![
                    "verify this column is intended to be high-cardinality".into(),
                    "if not, consider grouping or hashing".into(),
                ],
            ));
        }
    }
    out
}

// ─── Schema mismatch ────────────────────────────────────────────────────────

pub fn detect_schema_mismatch(df: &DataFrame, expected: &ExpectedSchema) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let actual: BTreeMap<String, String> = df
        .columns
        .iter()
        .map(|(n, c)| (n.clone(), c.type_name().to_string()))
        .collect();

    // Missing columns.
    for (name, exp_ty) in &expected.columns {
        match actual.get(name) {
            None => {
                out.push(ValidationFinding::new(
                    "E9020",
                    FindingSeverity::Error,
                    format!("expected column `{}` is missing", name),
                    Some(name.clone()),
                    None,
                    vec![FindingEvidence::Sample {
                        label: "expected_type".into(),
                        value: exp_ty.clone(),
                    }],
                    df.nrows() as u64,
                    vec![],
                    vec!["check ingest path; ensure schema is enforced upstream".into()],
                ));
            }
            Some(got_ty) if got_ty != exp_ty => {
                out.push(ValidationFinding::new(
                    "E9021",
                    FindingSeverity::Error,
                    format!(
                        "column `{}` has type `{}`, expected `{}`",
                        name, got_ty, exp_ty
                    ),
                    Some(name.clone()),
                    None,
                    vec![
                        FindingEvidence::Sample {
                            label: "actual_type".into(),
                            value: got_ty.clone(),
                        },
                        FindingEvidence::Sample {
                            label: "expected_type".into(),
                            value: exp_ty.clone(),
                        },
                    ],
                    df.nrows() as u64,
                    vec!["type mismatch likely from upstream serialisation drift".into()],
                    vec!["align the upstream schema or update the expected schema".into()],
                ));
            }
            _ => {}
        }
    }
    // Extra columns.
    for name in actual.keys() {
        if !expected.columns.contains_key(name) {
            let sev = if expected.strict_extra {
                FindingSeverity::Error
            } else {
                FindingSeverity::Notice
            };
            out.push(ValidationFinding::new(
                "E9022",
                sev,
                format!("column `{}` is not in the expected schema", name),
                Some(name.clone()),
                None,
                vec![],
                df.nrows() as u64,
                vec![
                    "strict_extra controls whether this is an error or a notice".into(),
                ],
                vec!["either widen the schema or drop the column".into()],
            ));
        }
    }
    out
}

// ─── Conditional missingness (v0.5) ────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ConditionalMissingnessConfig {
    /// Minimum count of missing rows in A for the check to fire.
    pub min_missing_in_a: u64,
    /// Threshold on `P(missing(B) | missing(A))` — if ≥ this, fire.
    pub implication_threshold: f64,
}

impl Default for ConditionalMissingnessConfig {
    fn default() -> Self {
        Self {
            min_missing_in_a: 5,
            implication_threshold: 0.95,
        }
    }
}

/// Detect "missing(A) implies missing(B)" patterns — when A is missing,
/// B is *also* missing in ≥ threshold of those rows. Common churn-
/// pipeline bug: a join failed and a whole feature family is jointly
/// null.
///
/// Only fires on Float columns in v0.5 (NaN-based missingness). For
/// columns whose missingness is declared via NullMask, the caller can
/// build their own conditional-missingness check via the same shape.
///
/// Emits **E9070** (Notice) per (A, B) pair that crosses the threshold.
pub fn detect_conditional_missingness(
    df: &DataFrame,
    cfg: &ConditionalMissingnessConfig,
    null_masks: &NullMaskMap,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows();
    if n_rows == 0 {
        return out;
    }

    // **v0.6.4 fix** — the v0.5 version was Float-only (it skipped
    // `Str` / `Int` / etc. columns entirely, so a `?`-sentinel-bearing
    // `Str` column was invisible to the pairwise implication check
    // even when the caller had passed a `NullMaskMap`). We now build
    // the per-column missing-row set by *unioning* Float NaN positions
    // with the caller-supplied `null_masks`, for every column type.
    let mut miss_sets: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for (name, col) in &df.columns {
        let mut s: BTreeSet<usize> = BTreeSet::new();
        // Float NaN positions (no-op for non-Float).
        if let Column::Float(v) = col {
            for (i, x) in v.iter().enumerate() {
                if x.is_nan() {
                    s.insert(i);
                }
            }
        }
        // Mask-driven null positions for ALL types (Str, Int, Bool,
        // Categorical, DateTime), bounded to col length.
        if let Some(mask) = null_masks.get(name) {
            let col_len = col.len();
            for r in &mask.null_rows {
                if *r < col_len {
                    s.insert(*r);
                }
            }
        }
        if (s.len() as u64) >= cfg.min_missing_in_a {
            miss_sets.insert(name.clone(), s);
        }
    }

    // Pairwise check.
    let names: Vec<&String> = miss_sets.keys().collect();
    for i in 0..names.len() {
        for j in 0..names.len() {
            if i == j {
                continue;
            }
            let a = names[i];
            let b = names[j];
            let set_a = &miss_sets[a];
            let set_b = &miss_sets[b];
            let n_a = set_a.len() as u64;
            let intersection = set_a.iter().filter(|r| set_b.contains(r)).count() as u64;
            let p = intersection as f64 / n_a.max(1) as f64;
            if p >= cfg.implication_threshold {
                out.push(ValidationFinding::new(
                    "E9070",
                    FindingSeverity::Notice,
                    format!(
                        "missing(`{}`) implies missing(`{}`) in {:.1}% of rows ({}/{})",
                        a, b, p * 100.0, intersection, n_a
                    ),
                    Some(a.clone()),
                    None,
                    vec![
                        FindingEvidence::Ratio {
                            label: "implication_strength".into(),
                            value: p,
                        },
                        FindingEvidence::Count {
                            label: "n_jointly_missing".into(),
                            value: intersection,
                        },
                        FindingEvidence::Count {
                            label: "n_missing_in_a".into(),
                            value: n_a,
                        },
                        FindingEvidence::Sample {
                            label: "implied_column".into(),
                            value: b.clone(),
                        },
                    ],
                    n_rows as u64,
                    vec![
                        "joint missingness often signals a failed join or a shared upstream pipeline failure".into(),
                    ],
                    vec![
                        format!("inspect the rows where both `{}` and `{}` are NaN", a, b),
                        "verify whether the upstream join keys are correct".into(),
                    ],
                ));
            }
        }
    }
    out
}

// ─── Imbalanced-class warning (v0.5) ───────────────────────────────────────

/// For a caller-declared *binary* target column, flag if the minority
/// class is below `min_minority_rate` of the column.
///
/// Code: **E9071** (Notice if minority ≥ 1%, Warning if minority < 1%).
pub fn detect_imbalanced_target(
    df: &DataFrame,
    target_col: &str,
    min_minority_rate: f64,
) -> Vec<ValidationFinding> {
    let Some(col) = df.get_column(target_col) else {
        return vec![ValidationFinding::new(
            "E9075",
            FindingSeverity::Error,
            format!("target column `{}` not found", target_col),
            Some(target_col.into()),
            None,
            vec![],
            df.nrows() as u64,
            vec![],
            vec!["verify the column name".into()],
        )];
    };
    let n_rows = df.nrows() as u64;
    let (n_pos, n_neg) = match col {
        Column::Bool(v) => {
            let p = v.iter().filter(|b| **b).count() as u64;
            (p, n_rows - p)
        }
        Column::Int(v) => {
            let mut distinct: BTreeSet<i64> = BTreeSet::new();
            for x in v {
                distinct.insert(*x);
                if distinct.len() > 2 {
                    return vec![];
                }
            }
            if distinct.len() != 2 {
                return vec![];
            }
            let mut it = distinct.iter();
            let lo = *it.next().unwrap();
            let hi = *it.next().unwrap();
            let p = v.iter().filter(|x| **x == hi).count() as u64;
            let _ = lo;
            (p, n_rows - p)
        }
        _ => return vec![],
    };
    if n_rows == 0 {
        return vec![];
    }
    let minority = n_pos.min(n_neg);
    let minority_rate = minority as f64 / n_rows as f64;
    if minority_rate >= min_minority_rate {
        return vec![];
    }
    let severity = if minority_rate < 0.01 {
        FindingSeverity::Warning
    } else {
        FindingSeverity::Notice
    };
    vec![ValidationFinding::new(
        "E9071",
        severity,
        format!(
            "target `{}` is heavily imbalanced: minority class is {:.2}% ({} of {} rows)",
            target_col, minority_rate * 100.0, minority, n_rows
        ),
        Some(target_col.into()),
        None,
        vec![
            FindingEvidence::Ratio {
                label: "minority_rate".into(),
                value: minority_rate,
            },
            FindingEvidence::Count {
                label: "n_pos".into(),
                value: n_pos,
            },
            FindingEvidence::Count {
                label: "n_neg".into(),
                value: n_neg,
            },
        ],
        n_rows,
        vec![
            "models trained on imbalanced data tend to predict the majority class regardless of input".into(),
        ],
        vec![
            "use class_weight='balanced' or stratified sampling at training time".into(),
            "report precision/recall (not accuracy) when evaluating".into(),
            "consider whether undersampling, oversampling, or SMOTE is appropriate".into(),
        ],
    )]
}

// ─── Duplicate-key conditioning (v0.5) ─────────────────────────────────────

/// Augments `detect_duplicate_keys` with information about WHICH other
/// columns differ within each duplicate-key group. Helps debug bad
/// joins where the same primary key has been ingested with different
/// values in some columns.
///
/// Emits **E9073** (Notice) describing each column whose values vary
/// within at least one duplicate-key group.
pub fn detect_duplicate_key_conditioning(
    df: &DataFrame,
    key_column: &str,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let Some(key_col) = df.get_column(key_column) else {
        return out;
    };
    let n = key_col.len();

    // Bucket rows by key.
    let mut buckets: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for r in 0..n {
        let mut bytes = Vec::new();
        cell_bytes(key_col, r, &mut bytes);
        buckets.entry(bytes).or_default().push(r);
    }
    let dup_groups: Vec<&Vec<usize>> = buckets.values().filter(|v| v.len() > 1).collect();
    if dup_groups.is_empty() {
        return out;
    }

    // For each non-key column, count how many duplicate groups contain
    // rows that differ in that column.
    for (col_name, col) in &df.columns {
        if col_name == key_column {
            continue;
        }
        let mut n_groups_with_disagreement = 0u64;
        let mut sample_group_size = 0usize;
        for group in &dup_groups {
            let mut first_bytes: Option<Vec<u8>> = None;
            let mut disagrees = false;
            for &r in *group {
                let mut bytes = Vec::new();
                cell_bytes(col, r, &mut bytes);
                if let Some(fb) = &first_bytes {
                    if fb != &bytes {
                        disagrees = true;
                        break;
                    }
                } else {
                    first_bytes = Some(bytes);
                }
            }
            if disagrees {
                n_groups_with_disagreement += 1;
                if sample_group_size == 0 {
                    sample_group_size = group.len();
                }
            }
        }
        if n_groups_with_disagreement > 0 {
            out.push(ValidationFinding::new(
                "E9073",
                FindingSeverity::Notice,
                format!(
                    "duplicate-key groups disagree on column `{}` in {} group(s)",
                    col_name, n_groups_with_disagreement
                ),
                Some(col_name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_groups_with_disagreement".into(),
                        value: n_groups_with_disagreement,
                    },
                    FindingEvidence::Sample {
                        label: "key_column".into(),
                        value: key_column.into(),
                    },
                ],
                n as u64,
                vec![
                    "rows sharing the same primary key but disagreeing on this column typically signal a join bug or stale data".into(),
                ],
                vec![
                    format!("group by `{}` and inspect the rows that disagree on `{}`", key_column, col_name),
                    "decide which value is authoritative".into(),
                ],
            ));
        }
    }
    out
}

// ─── Sentinel-value detection (v0.4) ───────────────────────────────────────

/// Configuration for sentinel-value detection.
#[derive(Clone, Debug)]
pub struct SentinelConfig {
    /// Numeric sentinel candidates. Default: `-1`, `-9999`, `-99`, `999`, `9999`.
    pub numeric_sentinels: Vec<f64>,
    /// String sentinel candidates (case-insensitive). Default: empty, "NA",
    /// "null", "N/A", "unknown", "missing", "-".
    pub string_sentinels: Vec<String>,
    /// Minimum count of the sentinel in the column to fire.
    pub min_count: u64,
    /// Minimum fraction of column the sentinel must occupy to fire.
    pub min_rate: f64,
}

impl Default for SentinelConfig {
    fn default() -> Self {
        Self {
            numeric_sentinels: vec![-1.0, -99.0, -999.0, -9999.0, 999.0, 9999.0],
            string_sentinels: vec![
                "".into(),
                "NA".into(),
                "N/A".into(),
                "null".into(),
                "NULL".into(),
                "Null".into(),
                "None".into(),
                "unknown".into(),
                "Unknown".into(),
                "missing".into(),
                "MISSING".into(),
                "-".into(),
                "?".into(),
            ],
            min_count: 3,
            min_rate: 0.01,
        }
    }
}

/// Detect common "magic missing" sentinel values per column.
///
/// Emits **E9007** (Info) per (column, sentinel) pair that crosses both
/// `min_count` and `min_rate` thresholds. The message is intentionally
/// hedged ("may be a sentinel missing value") — Locke can't know
/// without domain context.
pub fn detect_sentinel_values(df: &DataFrame, cfg: &SentinelConfig) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows == 0 {
        return out;
    }

    for (name, col) in &df.columns {
        match col {
            Column::Float(v) => {
                for &sentinel in &cfg.numeric_sentinels {
                    let count = v
                        .iter()
                        .filter(|x| !x.is_nan() && (*x - sentinel).abs() < 1e-12)
                        .count() as u64;
                    if count >= cfg.min_count
                        && (count as f64 / n_rows as f64) >= cfg.min_rate
                    {
                        out.push(make_sentinel_finding(
                            name, &format!("{}", sentinel), count, n_rows, "numeric"
                        ));
                    }
                }
            }
            Column::Int(v) => {
                for &sentinel in &cfg.numeric_sentinels {
                    let s = sentinel as i64;
                    if (s as f64 - sentinel).abs() > 1e-12 {
                        continue; // non-integer sentinel can't match Int column
                    }
                    let count = v.iter().filter(|x| **x == s).count() as u64;
                    if count >= cfg.min_count
                        && (count as f64 / n_rows as f64) >= cfg.min_rate
                    {
                        out.push(make_sentinel_finding(
                            name, &format!("{}", s), count, n_rows, "integer"
                        ));
                    }
                }
            }
            Column::Str(v) => {
                for sentinel in &cfg.string_sentinels {
                    let count = v.iter().filter(|s| s.as_str() == sentinel.as_str()).count() as u64;
                    if count >= cfg.min_count
                        && (count as f64 / n_rows as f64) >= cfg.min_rate
                    {
                        out.push(make_sentinel_finding(
                            name,
                            &format!("{:?}", sentinel),
                            count,
                            n_rows,
                            "string",
                        ));
                    }
                }
            }
            _ => {} // Bool / DateTime / Categorical / CategoricalAdaptive: skip
        }
    }
    out
}

fn make_sentinel_finding(
    column: &str,
    sentinel_repr: &str,
    count: u64,
    n_rows: u64,
    kind: &str,
) -> ValidationFinding {
    let rate = count as f64 / n_rows as f64;
    ValidationFinding::new(
        "E9007",
        FindingSeverity::Info,
        format!(
            "column `{}` has {} occurrences of {} ({:.1}%); may be a sentinel missing value",
            column, count, sentinel_repr, rate * 100.0
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Count {
                label: "occurrences".into(),
                value: count,
            },
            FindingEvidence::Ratio {
                label: "rate".into(),
                value: rate,
            },
            FindingEvidence::Sample {
                label: "sentinel".into(),
                value: sentinel_repr.to_string(),
            },
            FindingEvidence::Sample {
                label: "kind".into(),
                value: kind.into(),
            },
        ],
        n_rows,
        vec![
            "sentinel detection is heuristic — `-1` may be real data in a delta column".into(),
            "Locke only surfaces candidates; the user decides whether to treat as null".into(),
        ],
        vec![
            "if this is missing, declare it via NullMask::from_indices(...) in v0.2+".into(),
            "or filter / impute at source".into(),
        ],
    )
}

// ─── Outliers (v0.4) ────────────────────────────────────────────────────────

/// Configuration for outlier detection.
#[derive(Clone, Debug)]
pub struct OutlierConfig {
    /// 1.5×IQR is the textbook "mild" cutoff.
    pub mild_iqr_multiplier: f64,
    /// 3×IQR is the textbook "extreme" cutoff.
    pub extreme_iqr_multiplier: f64,
    /// |modified Z| ≥ 3.5 is Iglewicz & Hoaglin's mild cutoff.
    pub mild_mod_z: f64,
    /// |modified Z| ≥ 5 is the extreme cutoff.
    pub extreme_mod_z: f64,
    /// Don't fire any outlier finding if the column has fewer than this many
    /// valid (non-NaN) values — outlier detection is unreliable below ~20.
    pub min_n_for_outliers: u64,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self {
            mild_iqr_multiplier: 1.5,
            extreme_iqr_multiplier: 3.0,
            mild_mod_z: 3.5,
            extreme_mod_z: 5.0,
            min_n_for_outliers: 20,
        }
    }
}

/// Per-column outlier counts, surfaced as findings.
///
/// Two finding codes:
/// - **E9040** (Notice) — column has `n_mild` mild outliers
/// - **E9041** (Warning) — column has `n_extreme` extreme outliers
pub fn detect_outliers(df: &DataFrame, cfg: &OutlierConfig) -> Vec<ValidationFinding> {
    use crate::stats::outlier_baselines;
    let mut out = Vec::new();
    for (name, col) in &df.columns {
        let values: Option<Vec<f64>> = match col {
            Column::Float(v) => Some(v.clone()),
            Column::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
            _ => None,
        };
        let Some(values) = values else { continue };
        let n_valid = values.iter().filter(|x| !x.is_nan()).count() as u64;
        if n_valid < cfg.min_n_for_outliers {
            continue;
        }
        let Some(b) = outlier_baselines(&values) else { continue };
        if b.iqr <= 0.0 && b.mad <= 0.0 {
            continue;
        }
        // Count mild + extreme by either method.
        let mut n_mild: u64 = 0;
        let mut n_extreme: u64 = 0;
        let mut sample_extreme = String::new();
        let mut samples_added = 0;
        for (i, &x) in values.iter().enumerate() {
            if x.is_nan() {
                continue;
            }
            // IQR test.
            let iqr_extreme = b.iqr > 0.0
                && (x < b.q1 - cfg.extreme_iqr_multiplier * b.iqr
                    || x > b.q3 + cfg.extreme_iqr_multiplier * b.iqr);
            let iqr_mild = b.iqr > 0.0
                && (x < b.q1 - cfg.mild_iqr_multiplier * b.iqr
                    || x > b.q3 + cfg.mild_iqr_multiplier * b.iqr);
            // Modified Z test.
            let mod_z = (x - b.median) * b.mod_z_inv;
            let mod_z_extreme = b.mad > 0.0 && mod_z.abs() >= cfg.extreme_mod_z;
            let mod_z_mild = b.mad > 0.0 && mod_z.abs() >= cfg.mild_mod_z;

            let is_extreme = iqr_extreme || mod_z_extreme;
            let is_mild = !is_extreme && (iqr_mild || mod_z_mild);

            if is_extreme {
                n_extreme += 1;
                if samples_added < 3 {
                    if samples_added > 0 {
                        sample_extreme.push(',');
                    }
                    sample_extreme.push_str(&format!("row{}={}", i, x));
                    samples_added += 1;
                }
            } else if is_mild {
                n_mild += 1;
            }
        }

        if n_mild > 0 {
            out.push(ValidationFinding::new(
                "E9040",
                FindingSeverity::Notice,
                format!(
                    "column `{}` has {} mild outliers (1.5×IQR or |mod-Z| ≥ {:.1})",
                    name, n_mild, cfg.mild_mod_z
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_mild".into(),
                        value: n_mild,
                    },
                    FindingEvidence::Range {
                        label: "iqr_q1_q3".into(),
                        min: b.q1,
                        max: b.q3,
                    },
                    FindingEvidence::Metric {
                        label: "iqr".into(),
                        value: b.iqr,
                    },
                    FindingEvidence::Metric {
                        label: "mad".into(),
                        value: b.mad,
                    },
                ],
                n_valid,
                vec![
                    "outliers are flagged by IQR (1.5×) or modified-Z (≥3.5)".into(),
                    "not all outliers are errors — check whether they're plausible at source".into(),
                ],
                vec![
                    "inspect the sample rows".into(),
                    "decide imputation, clipping, or 'this is a real tail' policy".into(),
                ],
            ));
        }

        if n_extreme > 0 {
            out.push(ValidationFinding::new(
                "E9041",
                FindingSeverity::Warning,
                format!(
                    "column `{}` has {} extreme outliers (3×IQR or |mod-Z| ≥ {:.1})",
                    name, n_extreme, cfg.extreme_mod_z
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_extreme".into(),
                        value: n_extreme,
                    },
                    FindingEvidence::Sample {
                        label: "sample".into(),
                        value: sample_extreme,
                    },
                    FindingEvidence::Range {
                        label: "iqr_q1_q3".into(),
                        min: b.q1,
                        max: b.q3,
                    },
                ],
                n_valid,
                vec![
                    "extreme outliers usually indicate sensor errors, sentinel values, or unit-mix-ups".into(),
                ],
                vec![
                    "verify physical plausibility (e.g. age = 999, temperature = -9999)".into(),
                    "consider declaring sentinel values as nulls via NullMask".into(),
                ],
            ));
        }
    }
    out
}

// ─── Top-level orchestrator ─────────────────────────────────────────────────

/// Run every v0 validator and concatenate findings.
pub fn validate_dataframe(
    df: &DataFrame,
    cfg: &ValidationConfig,
    impossible: &[ImpossibleValueRule],
    expected_schema: Option<&ExpectedSchema>,
    primary_key: Option<&str>,
    null_masks: &NullMaskMap,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    // **v0.6.4** — auto-detect common string sentinels (?, NA, NULL,
    // ...) before the missingness pipeline runs. The detected mask is
    // unioned with the caller's mask so both `detect_missingness` and
    // `detect_conditional_missingness` see the full set of missing
    // rows. Opt out via `cfg.auto_detect_sentinels = false`.
    let (auto_masks, auto_findings) = detect_string_sentinels(df, cfg);
    out.extend(auto_findings);
    let effective_masks = merge_null_mask_maps(null_masks, &auto_masks);
    out.extend(detect_missingness(df, cfg, &effective_masks));
    out.extend(detect_duplicates_full_row(df, cfg));
    if let Some(pk) = primary_key {
        out.extend(detect_duplicate_keys(df, pk));
    }
    out.extend(detect_constant_and_near_constant(df, cfg));
    out.extend(detect_impossible_values(df, impossible));
    out.extend(detect_high_cardinality_categorical(df, cfg));
    // v0.4: outlier detection on numeric columns.
    out.extend(detect_outliers(df, &OutlierConfig::default()));
    // v0.4: sentinel-value detection (heuristic, Info severity).
    out.extend(detect_sentinel_values(df, &SentinelConfig::default()));
    // v0.6: categorical / string semantic quality (E9010 rare,
    // E9011 encoding-risk, E9080 case-fold, E9081 whitespace,
    // E9082 near-duplicate).
    out.extend(crate::categorical::detect_all_categorical_quality(
        df,
        &crate::categorical::CategoricalQualityConfig::default(),
    ));
    // v0.6 batch 2: Int columns that look nominal-not-ordinal.
    out.extend(detect_label_encoding_risk(df, &LabelEncodingRiskConfig::default()));
    // v0.6 batch 2: PII pattern scan (E9090-E9093).
    out.extend(crate::pii::detect_all_pii(df, &crate::pii::PiiConfig::default()));
    // v0.6.3: distribution-shape diagnostics (E9024).
    out.extend(crate::shape::detect_distribution_shape(
        df,
        &crate::shape::ShapeConfig::default(),
    ));
    if let Some(sch) = expected_schema {
        out.extend(detect_schema_mismatch(df, sch));
    }
    out
}

// ─── Label-encoding risk (v0.6 batch 2) ─────────────────────────────────────

/// Config for `detect_label_encoding_risk`.
#[derive(Clone, Debug)]
pub struct LabelEncodingRiskConfig {
    /// Fire E9023 only when the Int column has <= this many distinct
    /// values. Default 30 — above this it's probably a real count, not
    /// an encoded category.
    pub max_distinct_for_risk: u64,
    /// And requires at least this many rows to avoid noise on tiny data.
    pub min_rows: u64,
    /// Suppress E9023 when the distinct values span a "wide enough"
    /// range relative to their count, suggesting a true ordinal /
    /// numeric quantity (e.g. ages 0..120). Specifically: fire only when
    /// `(max - min + 1) <= distinct * span_tightness` — i.e. the values
    /// densely pack into a small contiguous range.
    pub span_tightness: f64,
}

impl Default for LabelEncodingRiskConfig {
    fn default() -> Self {
        Self {
            max_distinct_for_risk: 30,
            min_rows: 30,
            span_tightness: 1.5,
        }
    }
}

/// Fire E9023 (Notice) when an `Int` column has a small bounded set of
/// distinct values arranged like a label-encoding (e.g. `0..N-1` or
/// `1..N`). Treating such a column as ordinal-numeric (e.g. in a linear
/// model) introduces a spurious ordering that wasn't in the source data.
pub fn detect_label_encoding_risk(
    df: &DataFrame,
    cfg: &LabelEncodingRiskConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.min_rows {
        return out;
    }
    for (name, col) in &df.columns {
        let Column::Int(values) = col else { continue };
        if values.is_empty() {
            continue;
        }
        let mut distinct: BTreeSet<i64> = BTreeSet::new();
        for &v in values {
            distinct.insert(v);
        }
        let n_distinct = distinct.len() as u64;
        if n_distinct < 2 || n_distinct > cfg.max_distinct_for_risk {
            continue;
        }
        // Range packing: how tight is the value spread?
        let (lo, hi) = (*distinct.iter().next().unwrap(), *distinct.iter().next_back().unwrap());
        // v0.6.4 — guard against i64 overflow when (hi - lo + 1) would
        // exceed i64::MAX. A column spanning ~the full i64 range is
        // definitionally not a label encoding; skip it.
        let span = match hi.checked_sub(lo).and_then(|d| d.checked_add(1)) {
            Some(s) => s as f64,
            None => continue,
        };
        let density = span / n_distinct as f64;
        if density > cfg.span_tightness {
            // Sparse spread — probably a real numeric quantity (e.g. ages
            // 1, 18, 67, 99). Don't fire.
            continue;
        }
        let sample_vals: Vec<String> = distinct.iter().take(8).map(|v| v.to_string()).collect();
        out.push(ValidationFinding::new(
            "E9023",
            FindingSeverity::Notice,
            format!(
                "column `{}` has {} distinct Int values densely packed in [{}, {}] — looks like a label-encoded nominal, not an ordinal numeric",
                name, n_distinct, lo, hi
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_distinct".into(),
                    value: n_distinct,
                },
                FindingEvidence::Range {
                    label: "value_range".into(),
                    min: lo as f64,
                    max: hi as f64,
                },
                FindingEvidence::Metric {
                    label: "range_density".into(),
                    value: density,
                },
                FindingEvidence::Sample {
                    label: "distinct_values".into(),
                    value: sample_vals.join(","),
                },
            ],
            n_rows,
            vec![
                "label-encoding risk is heuristic; the column may be a true ordinal (age, rating) — Locke surfaces a hint, not a verdict".into(),
                "tunable via LabelEncodingRiskConfig::max_distinct_for_risk and span_tightness".into(),
            ],
            vec![
                "if the values are categorical (e.g. discharge_disposition_id codes), one-hot encode or use embeddings".into(),
                "if the values are ordinal, document the ordering and keep the Int representation".into(),
            ],
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn df_with_nans() -> DataFrame {
        DataFrame::from_columns(vec![
            (
                "age".into(),
                Column::Float(vec![1.0, 2.0, f64::NAN, 4.0, f64::NAN]),
            ),
            ("flag".into(), Column::Bool(vec![true, true, true, true, true])),
        ])
        .unwrap()
    }

    #[test]
    fn missingness_detects_nan_in_float() {
        let df = df_with_nans();
        let cfg = ValidationConfig::default();
        let findings = detect_missingness(&df, &cfg, &NullMaskMap::new());
        let miss = findings
            .iter()
            .find(|f| f.code == "E9001" && f.column.as_deref() == Some("age"))
            .expect("expected missingness finding");
        let n = miss
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Count { label, value } if label == "n_missing" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn missingness_emits_limitation_for_non_float() {
        let df = df_with_nans();
        let cfg = ValidationConfig::default();
        let findings = detect_missingness(&df, &cfg, &NullMaskMap::new());
        // bool column gets a limitation note (E9002)
        assert!(findings
            .iter()
            .any(|f| f.code == "E9002" && f.column.as_deref() == Some("flag")));
    }

    #[test]
    fn detect_duplicates_full_row_finds_dupes() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Int(vec![1, 1, 2, 2, 3])),
            ("y".into(), Column::Int(vec![10, 10, 20, 20, 30])),
        ])
        .unwrap();
        let cfg = ValidationConfig::default();
        let findings = detect_duplicates_full_row(&df, &cfg);
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        let n = f
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Count { label, value } if label == "duplicate_rows" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn duplicate_keys_detects_repeated_values() {
        let df = DataFrame::from_columns(vec![(
            "id".into(),
            Column::Int(vec![1, 2, 2, 3, 3, 3]),
        )])
        .unwrap();
        let findings = detect_duplicate_keys(&df, "id");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "E9004");
    }

    #[test]
    fn duplicate_keys_missing_column_is_error() {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Int(vec![1]))]).unwrap();
        let findings = detect_duplicate_keys(&df, "id");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "E9005");
        assert_eq!(findings[0].severity, FindingSeverity::Error);
    }

    #[test]
    fn constant_column_warns() {
        let df = DataFrame::from_columns(vec![("c".into(), Column::Int(vec![5, 5, 5, 5]))]).unwrap();
        let cfg = ValidationConfig::default();
        let findings = detect_constant_and_near_constant(&df, &cfg);
        assert!(findings.iter().any(|f| f.code == "E9010"));
    }

    #[test]
    fn near_constant_column_notices() {
        let mut vals = vec![1i64; 100];
        vals[0] = 2;
        let df = DataFrame::from_columns(vec![("c".into(), Column::Int(vals))]).unwrap();
        let cfg = ValidationConfig::default();
        let findings = detect_constant_and_near_constant(&df, &cfg);
        assert!(findings.iter().any(|f| f.code == "E9011"));
    }

    #[test]
    fn impossible_numeric_range_flags() {
        let df = DataFrame::from_columns(vec![(
            "x".into(),
            Column::Float(vec![1.0, -5.0, 3.0, 99.0]),
        )])
        .unwrap();
        let rules = vec![ImpossibleValueRule::NumericRange {
            column: "x".into(),
            min: 0.0,
            max: 10.0,
        }];
        let findings = detect_impossible_values(&df, &rules);
        assert!(findings.iter().any(|f| f.code == "E9014"));
    }

    #[test]
    fn impossible_allowed_strings_flags() {
        let df = DataFrame::from_columns(vec![(
            "country".into(),
            Column::Str(vec!["us".into(), "uk".into(), "atlantis".into()]),
        )])
        .unwrap();
        let mut allowed = BTreeSet::new();
        allowed.insert("us".into());
        allowed.insert("uk".into());
        let rules = vec![ImpossibleValueRule::AllowedStrings {
            column: "country".into(),
            allowed,
        }];
        let findings = detect_impossible_values(&df, &rules);
        assert!(findings.iter().any(|f| f.code == "E9014"));
    }

    #[test]
    fn schema_mismatch_detects_missing_and_typed() {
        let df = DataFrame::from_columns(vec![
            ("age".into(), Column::Int(vec![1, 2, 3])),
            ("extra".into(), Column::Int(vec![9, 9, 9])),
        ])
        .unwrap();
        let mut cols = BTreeMap::new();
        cols.insert("age".into(), "Float".into()); // type mismatch
        cols.insert("missing".into(), "Int".into()); // missing column
        let sch = ExpectedSchema {
            columns: cols,
            strict_extra: false,
        };
        let findings = detect_schema_mismatch(&df, &sch);
        assert!(findings.iter().any(|f| f.code == "E9020")); // missing
        assert!(findings.iter().any(|f| f.code == "E9021")); // type
        assert!(findings.iter().any(|f| f.code == "E9022")); // extra (notice)
    }

    #[test]
    fn validators_are_deterministic_across_runs() {
        let df = df_with_nans();
        let cfg = ValidationConfig::default();
        let masks = NullMaskMap::new();
        let a = detect_missingness(&df, &cfg, &masks);
        let b = detect_missingness(&df, &cfg, &masks);
        assert_eq!(a, b);
    }

    #[test]
    fn conditional_missingness_detects_joint_nans() {
        // Build a DF where rows 0..10 are NaN in BOTH columns, the rest are valid.
        let mut a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();
        for i in 0..10 {
            a[i] = f64::NAN;
            b[i] = f64::NAN;
        }
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Float(a)),
            ("b".into(), Column::Float(b)),
        ])
        .unwrap();
        let r = detect_conditional_missingness(&df, &ConditionalMissingnessConfig::default(), &NullMaskMap::new());
        // Should see both directions: a→b and b→a, both 100%.
        assert!(r.iter().any(|f| f.code == "E9070" && f.column.as_deref() == Some("a")));
        assert!(r.iter().any(|f| f.code == "E9070" && f.column.as_deref() == Some("b")));
    }

    #[test]
    fn conditional_missingness_no_finding_for_independent_nans() {
        // a is missing at rows 0..10, b is missing at rows 50..60 — disjoint.
        let mut a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();
        for i in 0..10 {
            a[i] = f64::NAN;
        }
        for i in 50..60 {
            b[i] = f64::NAN;
        }
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Float(a)),
            ("b".into(), Column::Float(b)),
        ])
        .unwrap();
        let r = detect_conditional_missingness(&df, &ConditionalMissingnessConfig::default(), &NullMaskMap::new());
        assert!(r.is_empty());
    }

    #[test]
    fn imbalanced_target_warning_fires_below_one_percent() {
        let mut y = vec![0i64; 1000];
        for i in 0..5 {
            y[i] = 1; // 0.5% positive — clearly under the Warning threshold
        }
        let df = DataFrame::from_columns(vec![("churned".into(), Column::Int(y))]).unwrap();
        let r = detect_imbalanced_target(&df, "churned", 0.05);
        let f = r.iter().find(|f| f.code == "E9071").expect("E9071 expected");
        assert_eq!(f.severity, FindingSeverity::Warning);
    }

    #[test]
    fn imbalanced_target_no_warning_for_50_50() {
        let y: Vec<i64> = (0..100).map(|i| (i % 2) as i64).collect();
        let df = DataFrame::from_columns(vec![("churned".into(), Column::Int(y))]).unwrap();
        let r = detect_imbalanced_target(&df, "churned", 0.05);
        assert!(r.is_empty());
    }

    #[test]
    fn duplicate_key_conditioning_finds_disagreement() {
        // user_id 7 appears twice with different last_login values
        let ids = vec![1i64, 2, 7, 7, 5];
        let logins = vec![100.0f64, 200.0, 300.0, 999.0, 500.0]; // row 2 & 3 disagree
        let df = DataFrame::from_columns(vec![
            ("user_id".into(), Column::Int(ids)),
            ("last_login".into(), Column::Float(logins)),
        ])
        .unwrap();
        let r = detect_duplicate_key_conditioning(&df, "user_id");
        assert!(r.iter().any(|f| f.code == "E9073" && f.column.as_deref() == Some("last_login")));
    }

    #[test]
    fn duplicate_key_conditioning_no_finding_when_dups_agree() {
        // user_id 7 appears twice with the SAME last_login
        let ids = vec![1i64, 2, 7, 7, 5];
        let logins = vec![100.0f64, 200.0, 300.0, 300.0, 500.0];
        let df = DataFrame::from_columns(vec![
            ("user_id".into(), Column::Int(ids)),
            ("last_login".into(), Column::Float(logins)),
        ])
        .unwrap();
        let r = detect_duplicate_key_conditioning(&df, "user_id");
        assert!(r.is_empty());
    }

    #[test]
    fn sentinel_detection_flags_minus_nine_thousand_nine_hundred_ninety_nine() {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        for i in 0..10 {
            v[i] = -9999.0;
        }
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let findings = detect_sentinel_values(&df, &SentinelConfig::default());
        assert!(findings.iter().any(|f| f.code == "E9007"));
    }

    #[test]
    fn sentinel_detection_below_threshold_does_not_fire() {
        // Only 2 occurrences out of 100 → below min_count (3).
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        v[0] = -9999.0;
        v[1] = -9999.0;
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let findings = detect_sentinel_values(&df, &SentinelConfig::default());
        assert!(!findings.iter().any(|f| f.code == "E9007"));
    }

    #[test]
    fn sentinel_detection_string_na_fires() {
        let mut v: Vec<String> = (0..100).map(|i| format!("v{}", i)).collect();
        for i in 0..5 {
            v[i] = "NA".into();
        }
        let df = DataFrame::from_columns(vec![("country".into(), Column::Str(v))]).unwrap();
        let findings = detect_sentinel_values(&df, &SentinelConfig::default());
        assert!(findings.iter().any(|f| f.code == "E9007" && f.message.contains("\\\"NA\\\"") || f.message.contains("NA")));
    }

    #[test]
    fn sentinel_detection_is_deterministic() {
        let v: Vec<f64> = (0..100).map(|i| if i < 5 { -9999.0 } else { i as f64 }).collect();
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let cfg = SentinelConfig::default();
        let a = detect_sentinel_values(&df, &cfg);
        let b = detect_sentinel_values(&df, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn outlier_detection_emits_e9041_for_extreme() {
        // 50 normal-ish values + 1 wildly extreme one.
        let mut v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        v.push(10_000.0);
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let cfg = OutlierConfig::default();
        let findings = detect_outliers(&df, &cfg);
        assert!(findings.iter().any(|f| f.code == "E9041"));
    }

    #[test]
    fn outlier_detection_does_not_fire_on_small_samples() {
        // Only 10 rows — below `min_n_for_outliers = 20`. No finding even
        // if a wild value is present.
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10_000.0];
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let findings = detect_outliers(&df, &OutlierConfig::default());
        assert!(findings.is_empty());
    }

    #[test]
    fn outlier_detection_skips_non_numeric_columns() {
        let df = DataFrame::from_columns(vec![(
            "name".into(),
            Column::Str(vec!["a".into(); 30]),
        )])
        .unwrap();
        let findings = detect_outliers(&df, &OutlierConfig::default());
        assert!(findings.is_empty());
    }

    #[test]
    fn outlier_detection_is_deterministic() {
        let mut v: Vec<f64> = (0..50).map(|i| (i as f64).sin() * 10.0).collect();
        v.push(1_000.0);
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let cfg = OutlierConfig::default();
        let a = detect_outliers(&df, &cfg);
        let b = detect_outliers(&df, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn null_mask_for_int_column_emits_e9001() {
        let df = DataFrame::from_columns(vec![
            ("age".into(), Column::Int(vec![10, 20, 30, 40, 50])),
        ])
        .unwrap();
        let cfg = ValidationConfig::default();
        let mut masks = NullMaskMap::new();
        masks.insert("age".into(), NullMask::from_indices([1, 3]));
        let findings = detect_missingness(&df, &cfg, &masks);
        // E9001 should fire with n_missing=2; E9002 must NOT fire.
        assert!(findings.iter().any(|f| f.code == "E9001"));
        assert!(!findings.iter().any(|f| f.code == "E9002"));
    }

    #[test]
    fn null_mask_out_of_bounds_emits_e9006() {
        let df = DataFrame::from_columns(vec![
            ("age".into(), Column::Int(vec![10, 20, 30])),
        ])
        .unwrap();
        let cfg = ValidationConfig::default();
        let mut masks = NullMaskMap::new();
        masks.insert("age".into(), NullMask::from_indices([1, 99]));
        let findings = detect_missingness(&df, &cfg, &masks);
        assert!(findings.iter().any(|f| f.code == "E9006"));
        // Out-of-bounds index ignored; in-bounds index 1 still counts.
        let miss = findings.iter().find(|f| f.code == "E9001").unwrap();
        for ev in &miss.evidence {
            if let FindingEvidence::Count { label, value } = ev {
                if label == "n_missing" {
                    assert_eq!(*value, 1);
                }
            }
        }
    }

    #[test]
    fn null_mask_union_with_nan_for_float_columns() {
        let df = DataFrame::from_columns(vec![(
            "x".into(),
            Column::Float(vec![1.0, f64::NAN, 3.0, 4.0, 5.0]),
        )])
        .unwrap();
        let cfg = ValidationConfig::default();
        let mut masks = NullMaskMap::new();
        // Mark rows 1 (already NaN) and 4 (clean). Union should be {1, 4} = 2.
        masks.insert("x".into(), NullMask::from_indices([1, 4]));
        let findings = detect_missingness(&df, &cfg, &masks);
        let miss = findings.iter().find(|f| f.code == "E9001").unwrap();
        let n = miss
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Count { label, value } if label == "n_missing" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 2);
    }

    // ─── v0.6.4 — auto string-sentinel detection (E9008) ───────────────────

    fn df_with_sentinels() -> DataFrame {
        DataFrame::from_columns(vec![
            (
                "weight".into(),
                Column::Str(vec![
                    "?".into(), "?".into(), "?".into(),
                    "[0-25)".into(), "[25-50)".into(),
                ]),
            ),
            (
                "race".into(),
                Column::Str(vec![
                    "Caucasian".into(), "AfricanAmerican".into(),
                    "?".into(), "Asian".into(), "Other".into(),
                ]),
            ),
            (
                "clean".into(),
                Column::Str(vec![
                    "x".into(), "y".into(), "z".into(), "x".into(), "y".into(),
                ]),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn string_sentinel_detection_finds_question_mark_columns() {
        let df = df_with_sentinels();
        let cfg = ValidationConfig::default();
        let (masks, findings) = detect_string_sentinels(&df, &cfg);
        // weight (3 of 5) + race (1 of 5) — clean has none.
        assert_eq!(masks.len(), 2);
        assert_eq!(masks["weight"].null_rows, [0, 1, 2].iter().copied().collect());
        assert_eq!(masks["race"].null_rows, [2].iter().copied().collect());
        assert!(!masks.contains_key("clean"));
        // One E9008 finding per affected column.
        assert_eq!(findings.iter().filter(|f| f.code == "E9008").count(), 2);
        assert!(findings
            .iter()
            .any(|f| f.code == "E9008" && f.column.as_deref() == Some("weight")));
    }

    #[test]
    fn string_sentinel_detection_opt_out_is_backward_compatible() {
        let df = df_with_sentinels();
        let cfg = ValidationConfig {
            auto_detect_sentinels: false,
            ..Default::default()
        };
        let (masks, findings) = detect_string_sentinels(&df, &cfg);
        assert!(masks.is_empty());
        assert!(findings.is_empty());
    }

    #[test]
    fn string_sentinel_detection_handles_custom_sentinel() {
        let df = DataFrame::from_columns(vec![(
            "discharge".into(),
            Column::Str(vec![
                "DECEASED".into(), "Home".into(), "DECEASED".into(), "Home".into(),
            ]),
        )])
        .unwrap();
        let cfg = ValidationConfig {
            additional_sentinels: vec!["DECEASED".into()],
            ..Default::default()
        };
        let (masks, _findings) = detect_string_sentinels(&df, &cfg);
        assert_eq!(masks["discharge"].null_rows, [0, 2].iter().copied().collect());
    }

    #[test]
    fn string_sentinel_detection_finds_all_builtins() {
        let mut cols = Vec::new();
        for (i, sentinel) in BUILTIN_STRING_SENTINELS.iter().enumerate() {
            cols.push((
                format!("col_{}", i),
                Column::Str(vec![(*sentinel).into(), "real_value".into()]),
            ));
        }
        let df = DataFrame::from_columns(cols).unwrap();
        let cfg = ValidationConfig::default();
        let (masks, _findings) = detect_string_sentinels(&df, &cfg);
        // Every builtin should match its row 0.
        assert_eq!(masks.len(), BUILTIN_STRING_SENTINELS.len());
    }

    #[test]
    fn string_sentinel_detection_skips_non_str_columns() {
        let df = DataFrame::from_columns(vec![
            ("score".into(), Column::Float(vec![1.0, 2.0, 3.0])),
            ("n".into(), Column::Int(vec![1, 2, 3])),
        ])
        .unwrap();
        let cfg = ValidationConfig::default();
        let (masks, findings) = detect_string_sentinels(&df, &cfg);
        assert!(masks.is_empty());
        assert!(findings.is_empty());
    }

    #[test]
    fn string_sentinel_detection_is_deterministic() {
        let df = df_with_sentinels();
        let cfg = ValidationConfig::default();
        let a = detect_string_sentinels(&df, &cfg);
        let b = detect_string_sentinels(&df, &cfg);
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn merge_null_mask_maps_unions_row_sets() {
        let mut a = NullMaskMap::new();
        a.insert("x".into(), NullMask::from_indices([1, 3]));
        a.insert("y".into(), NullMask::from_indices([0]));
        let mut b = NullMaskMap::new();
        b.insert("x".into(), NullMask::from_indices([2, 3]));
        b.insert("z".into(), NullMask::from_indices([4]));
        let merged = merge_null_mask_maps(&a, &b);
        assert_eq!(merged["x"].null_rows, [1, 2, 3].iter().copied().collect());
        assert_eq!(merged["y"].null_rows, [0].iter().copied().collect());
        assert_eq!(merged["z"].null_rows, [4].iter().copied().collect());
    }

    #[test]
    fn validate_dataframe_pipeline_picks_up_sentinels_automatically() {
        let df = df_with_sentinels();
        let cfg = ValidationConfig::default();
        let findings = validate_dataframe(
            &df, &cfg, &[], None, None, &NullMaskMap::new(),
        );
        // E9008 (sentinels) AND E9001 (missingness for weight, since
        // 60% > 10% Warning) should both fire.
        assert!(findings.iter().any(|f| f.code == "E9008" && f.column.as_deref() == Some("weight")));
        assert!(findings.iter().any(|f| f.code == "E9001" && f.column.as_deref() == Some("weight")));
    }

    #[test]
    fn conditional_missingness_sees_str_sentinels_via_mask() {
        // Two Str columns both 100% `?` — implication should fire
        // 100% in both directions when the mask covers them.
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Str(vec!["?".into(); 20])),
            ("b".into(), Column::Str(vec!["?".into(); 20])),
        ])
        .unwrap();
        let mut masks = NullMaskMap::new();
        masks.insert("a".into(), NullMask::from_indices(0..20));
        masks.insert("b".into(), NullMask::from_indices(0..20));
        let r = detect_conditional_missingness(
            &df,
            &ConditionalMissingnessConfig::default(),
            &masks,
        );
        // E9070 should fire in both directions: a→b and b→a.
        assert!(r.iter().any(|f| f.code == "E9070" && f.column.as_deref() == Some("a")));
        assert!(r.iter().any(|f| f.code == "E9070" && f.column.as_deref() == Some("b")));
    }
}
