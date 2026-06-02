//! Auto-promote mostly-numeric `Str` columns to `Float` with NaN
//! replacing sentinel and unparseable values.
//!
//! # The gap this fixes (ADR-0042)
//!
//! CSV readers (`cjc-data::CsvReader` and most others) infer column types
//! from the first data row. A numeric column whose first row is empty
//! (`""`), `"?"`, `"NA"`, or any other sentinel value gets typed as `Str`
//! — and stays `Str` even when most subsequent rows hold real numbers.
//!
//! Several Locke detectors are gated on `Float` typing:
//!
//! - **E9039** (numeric drift, `compare()` path) — only runs on
//!   `Column::Float`
//! - **E9070** (conditional missingness) — pairs only fire between
//!   `Float` columns (NaN is the link)
//! - **E9050..=E9054** (temporal sanity) — Float / DateTime only
//!
//! The LendingClub demo (2026-06-01) made this concrete: `annual_inc_joint`
//! is logically numeric but stays `Str` because ~85% of rows are empty
//! (non-joint applications). E9070 silently skipped the joint columns;
//! E9039 would have done the same on a drift compare. The user gets a
//! partial audit and the report doesn't say so.
//!
//! # The fix
//!
//! After E9008 detects sentinels on `Str` columns, scan each `Str` column
//! and classify every value as one of:
//!
//! - **Sentinel** (matches the [`BUILTIN_STRING_SENTINELS`] set or
//!   `cfg.additional_sentinels`)
//! - **Parseable** (`str::parse::<f64>()` succeeds)
//! - **Other** (text, malformed numbers)
//!
//! If the column has at least `min_non_sentinel_rows_for_promotion`
//! non-sentinel rows AND the parseable fraction of those non-sentinel
//! rows is at least `min_parseable_fraction_for_promotion`, the column
//! is **promoted**: rebuilt as `Column::Float` where sentinels and
//! unparseable values become NaN.
//!
//! Each promoted column emits one E9009 `Info` finding documenting the
//! promotion: how many rows were sentinels, how many parseable, the
//! median parseable fraction, etc.
//!
//! # What we explicitly do NOT do
//!
//! - **Str → Int promotion** — `Int` columns can't represent NaN. Use
//!   Float; downstream consumers can `as i64` if they need integer
//!   semantics.
//! - **Str → Bool promotion** — string `"true"`/`"false"` columns are
//!   rare in production datasets. Excluded for scope.
//! - **Categorical promotion** — Categorical columns are explicitly
//!   categorical by construction.
//! - **Numeric-with-suffix parsing** (`"15.0%"`, `"45000$"`) — would
//!   require domain-specific format hints and is a rabbit hole. The
//!   `str::parse::<f64>` standard is selective enough to avoid false
//!   positives on text columns.
//!
//! # Determinism
//!
//! The promoted DataFrame is a deterministic function of the input
//! frame and config. The scan is single-pass over each column;
//! emission order matches the DataFrame's column order. Two calls over
//! the same input produce byte-identical output. Proptest-locked.

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};
use crate::validation::{ValidationConfig, BUILTIN_STRING_SENTINELS};

/// E-code for promotion notices. Built-in namespace, picked to slot
/// next to E9008 (sentinel detection) in the auditor mental model.
pub const PROMOTION_FINDING_CODE: &str = "E9009";

/// Per-column scan result. Public so users can inspect promotion
/// decisions without running the full mutation pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnScan {
    pub column: String,
    pub n_total: usize,
    pub n_sentinel: usize,
    pub n_parseable: usize,
    pub n_other: usize,
    /// `n_parseable / (n_total - n_sentinel)` clamped to 0.0 when the
    /// denominator is zero. Comparable directly against
    /// `cfg.min_parseable_fraction_for_promotion`.
    pub parseable_fraction: f64,
    pub will_promote: bool,
}

/// Classify a single value. The `additional_sentinels` parameter is
/// borrowed as a slice of `String` to mirror `ValidationConfig`'s
/// storage and avoid per-call allocations.
fn classify_value(s: &str, additional_sentinels: &[String]) -> ValueClass {
    if BUILTIN_STRING_SENTINELS.iter().any(|b| *b == s) {
        return ValueClass::Sentinel;
    }
    if additional_sentinels.iter().any(|c| c == s) {
        return ValueClass::Sentinel;
    }
    if s.parse::<f64>().is_ok() {
        return ValueClass::Parseable;
    }
    ValueClass::Other
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueClass {
    Sentinel,
    Parseable,
    Other,
}

/// Scan one `Str` column and decide whether it qualifies for promotion.
/// Does NOT mutate; safe to call repeatedly for inspection.
fn scan_str_column(
    name: &str,
    values: &[String],
    cfg: &ValidationConfig,
) -> ColumnScan {
    let mut n_sentinel = 0usize;
    let mut n_parseable = 0usize;
    let mut n_other = 0usize;
    for v in values {
        match classify_value(v, &cfg.additional_sentinels) {
            ValueClass::Sentinel => n_sentinel += 1,
            ValueClass::Parseable => n_parseable += 1,
            ValueClass::Other => n_other += 1,
        }
    }
    let n_total = values.len();
    let non_sentinel = n_total - n_sentinel;
    let parseable_fraction = if non_sentinel == 0 {
        0.0
    } else {
        n_parseable as f64 / non_sentinel as f64
    };
    let meets_count = non_sentinel >= cfg.min_non_sentinel_rows_for_promotion;
    let meets_fraction = parseable_fraction >= cfg.min_parseable_fraction_for_promotion;
    let will_promote = meets_count && meets_fraction;
    ColumnScan {
        column: name.to_string(),
        n_total,
        n_sentinel,
        n_parseable,
        n_other,
        parseable_fraction,
        will_promote,
    }
}

/// Build the promoted `Vec<f64>` for a scanned column: sentinels and
/// unparseable values become NaN; parseable values are parsed.
fn build_promoted_float(values: &[String], cfg: &ValidationConfig) -> Vec<f64> {
    values
        .iter()
        .map(|v| match classify_value(v, &cfg.additional_sentinels) {
            ValueClass::Parseable => v.parse::<f64>().unwrap_or(f64::NAN),
            _ => f64::NAN,
        })
        .collect()
}

/// Scan every `Str` column in the dataframe and promote those that
/// qualify. Returns:
///
/// - `(Some(new_df), findings)` if any column was promoted. The new
///   dataframe shares column order with the input; promoted columns
///   are `Column::Float`, others are cloned verbatim. Each promoted
///   column emits one E9009 finding.
/// - `(None, vec![])` if no `Str` column qualified — callers should
///   continue with the original dataframe (zero allocation cost).
///
/// Determinism: column iteration order matches `df.columns`; finding
/// emission order matches the promotion order; the parser is the
/// standard `f64::from_str`. Two calls over the same input produce
/// byte-identical output.
pub fn auto_promote_str_columns(
    df: &DataFrame,
    cfg: &ValidationConfig,
) -> (Option<DataFrame>, Vec<ValidationFinding>) {
    if !cfg.auto_promote_str_to_float {
        return (None, Vec::new());
    }

    // First pass: scan every Str column. If none qualify, return early.
    let mut scans: Vec<(usize, ColumnScan)> = Vec::new();
    for (idx, (name, col)) in df.columns.iter().enumerate() {
        let Column::Str(values) = col else { continue };
        let scan = scan_str_column(name, values, cfg);
        if scan.will_promote {
            scans.push((idx, scan));
        }
    }
    if scans.is_empty() {
        return (None, Vec::new());
    }

    // Second pass: build the new column vector. Cheap; we only re-parse
    // the columns that will actually be promoted.
    let n_rows = df.nrows() as u64;
    let mut findings: Vec<ValidationFinding> = Vec::with_capacity(scans.len());
    let mut new_columns: Vec<(String, Column)> = df.columns.clone();
    for (idx, scan) in scans {
        let Column::Str(values) = &df.columns[idx].1 else {
            unreachable!("scan only collects Str columns")
        };
        let promoted = build_promoted_float(values, cfg);
        new_columns[idx] = (df.columns[idx].0.clone(), Column::Float(promoted));

        findings.push(ValidationFinding::new(
            PROMOTION_FINDING_CODE,
            FindingSeverity::Info,
            format!(
                "auto-promoted `{}` from Str to Float ({}/{} non-sentinel values parseable as f64, fraction = {:.3}); sentinel and unparseable cells become NaN",
                scan.column,
                scan.n_parseable,
                scan.n_total - scan.n_sentinel,
                scan.parseable_fraction,
            ),
            Some(scan.column.clone()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_sentinel_rows".into(),
                    value: scan.n_sentinel as u64,
                },
                FindingEvidence::Count {
                    label: "n_parseable_rows".into(),
                    value: scan.n_parseable as u64,
                },
                FindingEvidence::Count {
                    label: "n_other_rows".into(),
                    value: scan.n_other as u64,
                },
                FindingEvidence::Ratio {
                    label: "parseable_fraction".into(),
                    value: scan.parseable_fraction,
                },
            ],
            n_rows,
            vec![
                "promotion uses str::parse::<f64> (accepts 1.5e-3 scientific notation, rejects suffix units)".into(),
                "downstream Float-only detectors (E9070 conditional missingness, E9039 drift, E9050+ temporal) will now see this column".into(),
            ],
            vec![
                "if the column should stay Str, set ValidationConfig::auto_promote_str_to_float = false".into(),
                "if the parse rule is too strict (e.g. you need '$45000' → 45000), pre-clean the column or call auto_promote with a wider sentinel set".into(),
            ],
        ));
    }

    let new_df = DataFrame::from_columns(new_columns).expect(
        "from_columns preserves row count by construction; column lengths unchanged",
    );
    (Some(new_df), findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn cfg_default() -> ValidationConfig {
        ValidationConfig::default()
    }

    #[test]
    fn classify_value_recognises_sentinels_and_numbers() {
        let no_extra: Vec<String> = vec![];
        assert_eq!(classify_value("", &no_extra), ValueClass::Sentinel);
        assert_eq!(classify_value("?", &no_extra), ValueClass::Sentinel);
        assert_eq!(classify_value("NA", &no_extra), ValueClass::Sentinel);
        assert_eq!(classify_value("123", &no_extra), ValueClass::Parseable);
        assert_eq!(classify_value("-1.5e-3", &no_extra), ValueClass::Parseable);
        assert_eq!(classify_value("hello", &no_extra), ValueClass::Other);
        assert_eq!(classify_value(" 36 months", &no_extra), ValueClass::Other);
        assert_eq!(classify_value("10.5%", &no_extra), ValueClass::Other);
        // additional sentinel
        let extra = vec!["n/a".to_string()];
        assert_eq!(classify_value("n/a", &extra), ValueClass::Sentinel);
    }

    #[test]
    fn scan_promotes_mostly_numeric_with_sentinels() {
        // 11 sentinels ("" × 10 + "?" × 1), 11 parseable. non_sentinel
        // = 11 >= min 10; parseable_fraction = 1.0 >= 0.80 → promote.
        let mut values: Vec<String> = Vec::new();
        for _ in 0..10 {
            values.push("".into());
        }
        values.push("?".into());
        for i in 0..11 {
            values.push(format!("{}", 45000 + i * 5000));
        }
        let cfg = cfg_default();
        let scan = scan_str_column("annual_inc_joint", &values, &cfg);
        assert_eq!(scan.n_sentinel, 11);
        assert_eq!(scan.n_parseable, 11);
        assert_eq!(scan.n_other, 0);
        assert!((scan.parseable_fraction - 1.0).abs() < 1e-9);
        assert!(scan.will_promote);
    }

    #[test]
    fn scan_skips_text_columns() {
        let values: Vec<String> = vec![
            "NY".into(),
            "CA".into(),
            "TX".into(),
            "WA".into(),
            "FL".into(),
        ];
        let cfg = cfg_default();
        let scan = scan_str_column("addr_state", &values, &cfg);
        assert_eq!(scan.n_other, 5);
        assert!(!scan.will_promote);
    }

    #[test]
    fn scan_skips_lc_term_column() {
        // The classic LC `term` column: trims and digit+text, NOT parseable.
        let values: Vec<String> = vec![
            " 36 months".into(),
            " 60 months".into(),
            " 36 months".into(),
        ];
        let cfg = cfg_default();
        let scan = scan_str_column("term", &values, &cfg);
        assert_eq!(scan.n_other, 3);
        assert!(!scan.will_promote);
    }

    #[test]
    fn scan_respects_min_non_sentinel_threshold() {
        // 100 sentinels, 9 parseable → non_sentinel = 9 < default 10
        let mut values: Vec<String> = vec!["".into(); 100];
        for i in 0..9 {
            values.push(format!("{}", i));
        }
        let cfg = cfg_default();
        assert_eq!(cfg.min_non_sentinel_rows_for_promotion, 10);
        let scan = scan_str_column("sparse", &values, &cfg);
        assert!(!scan.will_promote);
        // Bumping to 10 parseable → promotes
        values.push("9".into());
        let scan = scan_str_column("sparse", &values, &cfg);
        assert!(scan.will_promote);
    }

    #[test]
    fn scan_respects_parseable_fraction_threshold() {
        // 9 parseable, 1 text → fraction = 9/10 = 0.9 ≥ default 0.80, promote
        let mut values: Vec<String> = vec![];
        for i in 0..9 {
            values.push(format!("{}", i));
        }
        values.push("garbage".into());
        let cfg = cfg_default();
        let scan = scan_str_column("mixed", &values, &cfg);
        assert!(scan.will_promote);
        assert!((scan.parseable_fraction - 0.9).abs() < 1e-9);
        // Add 2 more text → fraction = 9/12 = 0.75 < 0.80, skip
        values.push("more".into());
        values.push("text".into());
        let scan = scan_str_column("mixed", &values, &cfg);
        assert!(!scan.will_promote);
    }

    #[test]
    fn promote_skips_when_disabled() {
        let cfg = ValidationConfig {
            auto_promote_str_to_float: false,
            ..ValidationConfig::default()
        };
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Str(vec!["1".into(), "2".into(), "3".into()].iter().cloned().chain(vec!["4".into(); 20]).collect())),
        ])
        .unwrap();
        let (maybe, findings) = auto_promote_str_columns(&df, &cfg);
        assert!(maybe.is_none());
        assert!(findings.is_empty());
    }

    #[test]
    fn promote_rebuilds_column_as_float_with_nan() {
        // Build a 25-row column: 5 "" + 1 "?" + 1 "garbage" + 18 parseable.
        // non_sentinel = 19 ≥ default min 10; parseable_fraction = 18/19 ≈ 0.947
        // ≥ default 0.80 → promotes.
        let mut values: Vec<String> = vec![
            "".into(),
            "45000".into(),
            "".into(),
            "60000".into(),
            "".into(),
            "75000".into(),
            "?".into(),
            "100000".into(),
            "garbage".into(),
            "120000".into(),
            "".into(),
            "150000".into(),
            "".into(),
        ];
        for i in 0..12 {
            values.push(format!("{}", 200000 + i * 1000));
        }
        let n = values.len();
        let df = DataFrame::from_columns(vec![
            ("annual_inc_joint".into(), Column::Str(values)),
            ("loan_amnt".into(), Column::Float(vec![0.0; n])),
        ])
        .unwrap();
        let cfg = cfg_default();
        let (Some(new_df), findings) = auto_promote_str_columns(&df, &cfg) else {
            panic!("expected promotion");
        };
        assert_eq!(new_df.nrows(), n);
        let promoted = new_df.get_column("annual_inc_joint").unwrap();
        let Column::Float(v) = promoted else {
            panic!("expected Float");
        };
        assert!(v[0].is_nan());
        assert_eq!(v[1], 45000.0);
        assert!(v[2].is_nan());
        assert_eq!(v[3], 60000.0);
        assert!(v[6].is_nan()); // "?"
        assert!(v[8].is_nan()); // "garbage"
        assert_eq!(v[11], 150000.0);
        // Other columns untouched.
        let Column::Float(_) = new_df.get_column("loan_amnt").unwrap() else {
            panic!("loan_amnt should remain Float");
        };
        // One E9009 finding.
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "E9009");
        assert_eq!(findings[0].severity, FindingSeverity::Info);
        assert_eq!(findings[0].column.as_deref(), Some("annual_inc_joint"));
    }

    #[test]
    fn promote_does_not_touch_non_str_columns() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float(vec![1.0, 2.0, 3.0])),
            ("y".into(), Column::Int(vec![1, 2, 3])),
            ("z".into(), Column::Bool(vec![true, false, true])),
        ])
        .unwrap();
        let cfg = cfg_default();
        let (maybe, findings) = auto_promote_str_columns(&df, &cfg);
        assert!(maybe.is_none());
        assert!(findings.is_empty());
    }

    #[test]
    fn promote_idempotent_on_already_promoted_frame() {
        let values: Vec<String> = (0..20).map(|i| format!("{}", i)).collect();
        let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
        let cfg = cfg_default();
        let (Some(once), _) = auto_promote_str_columns(&df, &cfg) else {
            panic!()
        };
        // Run promotion again on the already-promoted frame.
        let (maybe_twice, findings_twice) = auto_promote_str_columns(&once, &cfg);
        assert!(maybe_twice.is_none(), "Float columns should not promote");
        assert!(findings_twice.is_empty());
    }

    #[test]
    fn promote_is_deterministic_across_runs() {
        let values: Vec<String> = vec![
            "".into(),
            "1".into(),
            "2".into(),
            "?".into(),
            "3".into(),
            "4".into(),
            "5".into(),
            "6".into(),
            "7".into(),
            "8".into(),
            "9".into(),
            "10".into(),
        ];
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Str(values.clone())),
            ("b".into(), Column::Str(values.clone())),
        ])
        .unwrap();
        let cfg = cfg_default();
        let (Some(df1), fnd1) = auto_promote_str_columns(&df, &cfg) else {
            panic!()
        };
        let (Some(df2), fnd2) = auto_promote_str_columns(&df, &cfg) else {
            panic!()
        };
        assert_eq!(df1.columns.len(), df2.columns.len());
        for ((n1, c1), (n2, c2)) in df1.columns.iter().zip(df2.columns.iter()) {
            assert_eq!(n1, n2);
            match (c1, c2) {
                (Column::Float(v1), Column::Float(v2)) => {
                    assert_eq!(v1.len(), v2.len());
                    for (x, y) in v1.iter().zip(v2.iter()) {
                        // bit-identical including NaN
                        assert_eq!(x.to_bits(), y.to_bits());
                    }
                }
                _ => panic!("both columns should be Float"),
            }
        }
        assert_eq!(fnd1.len(), fnd2.len());
        for (a, b) in fnd1.iter().zip(fnd2.iter()) {
            assert_eq!(a.sort_key(), b.sort_key());
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn promote_emits_finding_evidence_with_counts() {
        let values: Vec<String> = vec!["".into(); 5]
            .into_iter()
            .chain((0..15).map(|i| format!("{}", i)))
            .collect();
        let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
        let cfg = cfg_default();
        let (Some(_), findings) = auto_promote_str_columns(&df, &cfg) else {
            panic!()
        };
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        // Check the evidence payload includes the expected labels.
        let labels: Vec<&str> = f
            .evidence
            .iter()
            .map(|e| match e {
                FindingEvidence::Count { label, .. } => label.as_str(),
                FindingEvidence::Ratio { label, .. } => label.as_str(),
                FindingEvidence::Metric { label, .. } => label.as_str(),
                FindingEvidence::Sample { label, .. } => label.as_str(),
                FindingEvidence::Range { label, .. } => label.as_str(),
            })
            .collect();
        assert!(labels.contains(&"n_sentinel_rows"));
        assert!(labels.contains(&"n_parseable_rows"));
        assert!(labels.contains(&"parseable_fraction"));
    }
}
