//! Library half of the LendingClub Locke demo.
//!
//! The binary in `main.rs` is a thin CLI shim around the functions here.
//! Tests in `tests/expected_findings.rs` exercise the same surface.
//!
//! ## Pipeline
//!
//! 1. `load_csv_gz(path)` — open `.csv.gz`, gunzip, parse with `cjc-data::csv`
//! 2. `binarize_loan_status(df)` — derive `target_default` (Bool) from
//!    multi-class `loan_status`; drop rows whose status is not terminal
//!    (`Charged Off` / `Default` / `Fully Paid`). The two retained
//!    distinct values become `cjc-locke`'s required binary target.
//! 3. `run_locke_audit(df)` — single-shot `validate()` + `detect_target_leakage`
//!    + `detect_id_like_columns`. Returns a merged `LockeReport`.
//!
//! ## Why we filter loan_status
//!
//! `cjc-locke::leakage::detect_target_leakage` requires a binary target
//! (`Bool` or `Int` with exactly two distinct values). LendingClub's
//! `loan_status` has 7+ values including `Current`, `Late (16-30 days)`,
//! `In Grace Period`. For a credit-risk leakage audit, only the rows that
//! reached a terminal outcome (`Charged Off` / `Default` → 1; `Fully Paid`
//! → 0) carry a usable label. Mid-life loans have no ground-truth answer
//! to the question "did this loan default?", so including them would
//! contaminate the AUC computation.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use cjc_data::{Column, CsvConfig, CsvReader, DataFrame};
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::leakage::{detect_id_like_columns, detect_target_leakage, LeakageConfig};
use cjc_locke::report::{LockeReport, SeverityCounts, ValidationFinding};
use cjc_locke::validation::{ImpossibleValueRule, ValidationConfig};
use flate2::read::GzDecoder;

pub const TARGET_COLUMN: &str = "target_default";
pub const SOURCE_TARGET_COLUMN: &str = "loan_status";

/// Map a raw `loan_status` value to `Some(true)` (defaulted), `Some(false)`
/// (paid), or `None` (mid-life / unknown — row gets dropped).
///
/// Matches LendingClub's status taxonomy as of 2018-Q4. New statuses added
/// post-acquisition fall through to `None`.
pub fn classify_status(raw: &str) -> Option<bool> {
    match raw.trim() {
        "Charged Off" | "Default" => Some(true),
        "Fully Paid" => Some(false),
        // Does not meet credit-policy variants kept ambiguous: the row
        // is dropped because LC's own dataset doesn't carry the post-policy
        // resolution as a separate field.
        _ => None,
    }
}

/// Open a `.csv.gz` file, fully decompress into memory, and parse to a
/// `DataFrame`. Returns the parsed frame and the decompressed byte count
/// (for sanity logging).
///
/// `max_rows = None` parses every row. For the LC dataset that's ~2.5 M
/// rows × ~150 cols → ~3-4 GB peak. Callers on a 4 GB budget may pass
/// `Some(N)` to subsample; the leakage signal stays unambiguous even at
/// 100 K rows because the post-origination columns have AUC > 0.99.
pub fn load_csv_gz(
    path: &Path,
    max_rows: Option<usize>,
) -> Result<(DataFrame, usize), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut gz = GzDecoder::new(f);
    let mut bytes = Vec::new();
    gz.read_to_end(&mut bytes)
        .map_err(|e| format!("gunzip {}: {}", path.display(), e))?;
    let n_bytes = bytes.len();

    let cfg = CsvConfig {
        max_rows,
        ..CsvConfig::default()
    };
    let df = CsvReader::new(cfg)
        .parse(&bytes)
        .map_err(|e| format!("csv parse: {:?}", e))?;
    Ok((df, n_bytes))
}

/// Derive a binary `target_default` column from `loan_status` and filter
/// the dataframe to terminal-outcome rows only. Returns the filtered
/// dataframe with the new column appended.
pub fn binarize_loan_status(df: DataFrame) -> Result<DataFrame, String> {
    let status_col = df
        .get_column(SOURCE_TARGET_COLUMN)
        .ok_or_else(|| format!("column `{}` not found", SOURCE_TARGET_COLUMN))?;
    let status_values: Vec<String> = match status_col {
        Column::Str(v) => v.clone(),
        other => {
            return Err(format!(
                "`{}` is not a Str column (got {})",
                SOURCE_TARGET_COLUMN,
                other.type_name()
            ))
        }
    };

    let mut keep_mask: Vec<bool> = Vec::with_capacity(status_values.len());
    let mut target: Vec<bool> = Vec::new();
    for raw in &status_values {
        match classify_status(raw) {
            Some(b) => {
                keep_mask.push(true);
                target.push(b);
            }
            None => keep_mask.push(false),
        }
    }

    // Filter every existing column by keep_mask.
    let DataFrame { columns } = df;
    let mut filtered_columns: Vec<(String, Column)> = Vec::with_capacity(columns.len() + 1);
    for (name, col) in columns {
        let filtered = filter_column(&col, &keep_mask);
        filtered_columns.push((name, filtered));
    }
    filtered_columns.push((TARGET_COLUMN.to_string(), Column::Bool(target)));

    DataFrame::from_columns(filtered_columns).map_err(|e| format!("rebuild df: {:?}", e))
}

fn filter_column(col: &Column, mask: &[bool]) -> Column {
    match col {
        Column::Float(v) => Column::Float(
            v.iter()
                .zip(mask)
                .filter_map(|(x, k)| if *k { Some(*x) } else { None })
                .collect(),
        ),
        Column::Int(v) => Column::Int(
            v.iter()
                .zip(mask)
                .filter_map(|(x, k)| if *k { Some(*x) } else { None })
                .collect(),
        ),
        Column::Bool(v) => Column::Bool(
            v.iter()
                .zip(mask)
                .filter_map(|(x, k)| if *k { Some(*x) } else { None })
                .collect(),
        ),
        Column::Str(v) => Column::Str(
            v.iter()
                .zip(mask)
                .filter_map(|(x, k)| if *k { Some(x.clone()) } else { None })
                .collect(),
        ),
        Column::DateTime(v) => Column::DateTime(
            v.iter()
                .zip(mask)
                .filter_map(|(x, k)| if *k { Some(*x) } else { None })
                .collect(),
        ),
        Column::Categorical { levels, codes } => {
            let kept_codes: Vec<u32> = codes
                .iter()
                .zip(mask)
                .filter_map(|(c, k)| if *k { Some(*c) } else { None })
                .collect();
            Column::Categorical {
                levels: levels.clone(),
                codes: kept_codes,
            }
        }
        // CsvReader cannot emit CategoricalAdaptive, so this arm only
        // fires if a caller pre-processes the dataframe upstream. The
        // demo binary never does; we hard-fail with a clear message
        // rather than silently materialising the column as Str.
        Column::CategoricalAdaptive(_) => {
            panic!("filter_column: CategoricalAdaptive not supported in lendingclub-demo path")
        }
    }
}

/// Build the canonical `ValidateOptions` for this demo. Mirrors handoff
/// §3.2 exactly so the cross-validation file's expected E-codes are
/// reproducible.
pub fn lendingclub_validate_options() -> ValidateOptions {
    ValidateOptions {
        dataset_label: "lendingclub-2007-2018".into(),
        config: ValidationConfig {
            auto_detect_sentinels: true,
            additional_sentinels: vec!["n/a".into(), "NONE".into()],
            near_constant_threshold: 0.99,
            high_cardinality_ratio: 0.5,
            duplicate_sample_limit: 5,
            collect_per_value_lineage: false,
        },
        impossible_rules: vec![
            ImpossibleValueRule::NumericRange {
                column: "loan_amnt".into(),
                min: 0.0,
                max: 1_000_000.0,
            },
            ImpossibleValueRule::NumericRange {
                column: "int_rate".into(),
                min: 0.0,
                max: 50.0,
            },
            ImpossibleValueRule::NumericRange {
                column: "dti".into(),
                min: 0.0,
                max: 100.0,
            },
            ImpossibleValueRule::NumericRange {
                column: "annual_inc".into(),
                min: 0.0,
                max: 100_000_000.0,
            },
        ],
        expected_schema: None,
        primary_key: Some("id".into()),
        null_masks: Default::default(),
    }
}

/// The merged audit: `validate()` + `detect_target_leakage` + `detect_id_like_columns`.
///
/// We splice leakage and ID-like findings into the report's top-level
/// `findings` vector so a single JSON emit covers everything. The
/// severity_counts are recomputed from the merged set.
pub fn run_locke_audit(df: &DataFrame) -> LockeReport {
    let opts = lendingclub_validate_options();
    let mut report = validate(df, &opts);

    let leakage_cfg = LeakageConfig::default();
    let leakage_findings = detect_target_leakage(df, TARGET_COLUMN, &leakage_cfg);
    let id_findings = detect_id_like_columns(df, &leakage_cfg);

    let mut all: Vec<ValidationFinding> = report.findings.clone();
    all.extend(leakage_findings);
    all.extend(id_findings);
    all.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    report.severity_counts = SeverityCounts::from_findings(&all);
    report.findings = all;
    report
}

/// Tally findings by `(code, severity)` — the exact shape the regression
/// gate in `expected_findings.json` compares against.
pub fn finding_counts_by_code(
    findings: &[ValidationFinding],
) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for f in findings {
        *counts.entry(f.code.to_string()).or_insert(0) += 1;
    }
    counts
}

// ─── Honest-model AUC harness (cross_validate.md §3) ────────────────────────

/// Columns Locke flagged at E9061 (|AUC| ≥ 0.85) on the full LC dataset,
/// 2026-06-01 run. These are the *empirical* leakage flags, not the
/// handoff's a-priori prediction list.
pub const LOCKE_E9061_COLUMNS: &[&str] = &[
    "last_fico_range_high",
    "last_fico_range_low",
    "total_rec_prncp",
];

/// Domain-knowledge canonical post-origination columns from the LC data
/// dictionary. Used to build the "domain-honest" model. Strict superset of
/// `LOCKE_E9061_COLUMNS` — these are columns that *cannot* be safely used
/// as features because they are recorded after the outcome, regardless of
/// what Locke's |AUC| heuristic does or does not flag.
pub const DOMAIN_POST_ORIGINATION_COLUMNS: &[&str] = &[
    "collection_recovery_fee",
    "last_fico_range_high",
    "last_fico_range_low",
    "last_pymnt_amnt",
    "last_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "recoveries",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_int",
    "total_rec_late_fee",
    "total_rec_prncp",
];

/// Always-exclude columns: the target itself and ID-like keys.
pub const ALWAYS_EXCLUDED_COLUMNS: &[&str] = &[TARGET_COLUMN, "id", "member_id"];

/// Numeric column extraction. Returns column names and per-column f64
/// vectors. Skips columns whose names appear in `exclude`. Bool and Int
/// columns are coerced to f64; Str/Categorical/DateTime are dropped.
///
/// Order is deterministic: same as the input DataFrame's column order.
pub fn select_numeric_columns(
    df: &DataFrame,
    exclude: &[&str],
) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut names = Vec::new();
    let mut data = Vec::new();
    for (name, col) in &df.columns {
        if exclude.contains(&name.as_str()) {
            continue;
        }
        let col_data: Vec<f64> = match col {
            Column::Float(v) => v.clone(),
            Column::Int(v) => v.iter().map(|x| *x as f64).collect(),
            Column::Bool(v) => v.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect(),
            _ => continue,
        };
        names.push(name.clone());
        data.push(col_data);
    }
    (names, data)
}

/// Filter out columns that would harm IRLS convergence: all-NaN columns
/// (no signal) and zero-variance-after-NaN-removal columns (constant or
/// near-constant).
///
/// Returns the surviving (name, column) pairs in input order.
pub fn drop_useless_columns(
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut out_names = Vec::new();
    let mut out_columns = Vec::new();
    for (n, c) in names.into_iter().zip(columns.into_iter()) {
        let valid: Vec<f64> = c.iter().copied().filter(|x| x.is_finite()).collect();
        if valid.is_empty() {
            continue;
        }
        let v_min = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let v_max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (v_max - v_min).abs() < f64::EPSILON {
            continue;
        }
        out_names.push(n);
        out_columns.push(c);
    }
    (out_names, out_columns)
}

/// Per-column training fit: returns the (mean, std) tuple computed on the
/// non-NaN values of the training subset. NaN is treated as missing.
///
/// `mean` is the Kahan-summed arithmetic mean of finite values; if every
/// value is NaN the mean defaults to 0.0. `std` is the population standard
/// deviation; if every value is the same the std defaults to 1.0 (so the
/// downstream z-score yields 0.0 instead of NaN). This is honest because
/// a zero-variance column has no information either way.
pub fn fit_column_stats(values: &[f64]) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    let mut n = 0usize;
    for &x in values {
        if !x.is_finite() {
            continue;
        }
        let y = x - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
        n += 1;
    }
    if n == 0 {
        return (0.0, 1.0);
    }
    let mean = sum / n as f64;

    let mut sq_sum = 0.0_f64;
    let mut sq_comp = 0.0_f64;
    for &x in values {
        if !x.is_finite() {
            continue;
        }
        let d = x - mean;
        let y = d * d - sq_comp;
        let t = sq_sum + y;
        sq_comp = (t - sq_sum) - y;
        sq_sum = t;
    }
    let var = sq_sum / n as f64;
    let std = if var > 0.0 { var.sqrt() } else { 1.0 };
    (mean, std)
}

/// Apply train-set (mean, std) to a column: replace NaN with mean, then
/// z-score. Pure function — does not mutate `values` and does not look
/// at the column's own statistics.
pub fn apply_column_transform(values: &[f64], mean: f64, std: f64) -> Vec<f64> {
    let divisor = if std == 0.0 { 1.0 } else { std };
    values
        .iter()
        .map(|&x| {
            let cleaned = if x.is_finite() { x } else { mean };
            (cleaned - mean) / divisor
        })
        .collect()
}

/// Flatten per-column data at the given row indices into a row-major
/// `Vec<f64>` of shape `(indices.len(), n_cols)`. The output is what
/// `cjc_runtime::hypothesis::logistic_regression` expects for `x_flat`.
pub fn flatten_subset_row_major(
    columns: &[Vec<f64>],
    indices: &[usize],
) -> Vec<f64> {
    let n_rows = indices.len();
    let n_cols = columns.len();
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for &row_idx in indices {
        for col in columns.iter() {
            out.push(col[row_idx]);
        }
    }
    out
}

/// Given a trained logistic regression's full coefficient vector (intercept
/// at index 0, then p feature coefficients), score a row-major test matrix
/// of shape `(n, p)` and return the raw linear predictors `eta = x @ beta`.
///
/// For AUC we don't need to apply sigmoid — rank order is preserved by the
/// monotonic logistic transform.
pub fn score_logistic(
    coefficients: &[f64],
    x_flat: &[f64],
    n: usize,
    p: usize,
) -> Vec<f64> {
    debug_assert_eq!(coefficients.len(), p + 1);
    debug_assert_eq!(x_flat.len(), n * p);
    let intercept = coefficients[0];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut eta = intercept;
        for j in 0..p {
            eta += coefficients[j + 1] * x_flat[i * p + j];
        }
        out.push(eta);
    }
    out
}

/// Extract the binarized target column as a `Vec<bool>` for `auc_roc`.
pub fn extract_target_bool(df: &DataFrame) -> Result<Vec<bool>, String> {
    match df.get_column(TARGET_COLUMN) {
        Some(Column::Bool(v)) => Ok(v.clone()),
        Some(other) => Err(format!(
            "{} expected Bool, got {}",
            TARGET_COLUMN,
            other.type_name()
        )),
        None => Err(format!("{} column not present", TARGET_COLUMN)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_status_terminal_only() {
        assert_eq!(classify_status("Charged Off"), Some(true));
        assert_eq!(classify_status("Default"), Some(true));
        assert_eq!(classify_status("Fully Paid"), Some(false));
        assert_eq!(classify_status("Current"), None);
        assert_eq!(classify_status("Late (31-120 days)"), None);
        assert_eq!(classify_status("In Grace Period"), None);
        // whitespace tolerance
        assert_eq!(classify_status(" Fully Paid "), Some(false));
    }

    #[test]
    fn binarize_filters_and_appends() {
        // Hand-built mini frame: 5 rows, 3 columns, 2 terminal + 3 non-terminal.
        let status = vec![
            "Charged Off".to_string(),
            "Current".to_string(),
            "Fully Paid".to_string(),
            "Late (16-30 days)".to_string(),
            "Charged Off".to_string(),
        ];
        let loan_amnt = vec![10000.0, 5000.0, 7500.0, 12000.0, 25000.0];
        let id_col = vec![1i64, 2, 3, 4, 5];
        let df = DataFrame::from_columns(vec![
            ("loan_status".into(), Column::Str(status)),
            ("loan_amnt".into(), Column::Float(loan_amnt)),
            ("id".into(), Column::Int(id_col)),
        ])
        .unwrap();

        let out = binarize_loan_status(df).unwrap();
        assert_eq!(out.nrows(), 3); // 2 Charged Off + 1 Fully Paid
        assert_eq!(out.ncols(), 4); // original 3 + target_default

        let target = out.get_column(TARGET_COLUMN).unwrap();
        match target {
            Column::Bool(v) => assert_eq!(v, &vec![true, false, true]),
            _ => panic!("target_default should be Bool"),
        }

        // Filtered loan_amnt order corresponds to rows 0, 2, 4.
        match out.get_column("loan_amnt").unwrap() {
            Column::Float(v) => assert_eq!(v, &vec![10000.0, 7500.0, 25000.0]),
            _ => panic!(),
        }
    }

    fn make_demo_df() -> DataFrame {
        DataFrame::from_columns(vec![
            ("target_default".into(), Column::Bool(vec![true, false, true, false])),
            ("loan_amnt".into(), Column::Float(vec![1000.0, 2000.0, 3000.0, 4000.0])),
            ("term_int".into(), Column::Int(vec![36, 60, 36, 60])),
            ("verified".into(), Column::Bool(vec![true, true, false, false])),
            // Always all-NaN: should be dropped by drop_useless_columns.
            ("never_present".into(), Column::Float(vec![f64::NAN; 4])),
            // Always constant: should also be dropped.
            ("zero_var".into(), Column::Float(vec![5.0; 4])),
            // Str: should be skipped by select_numeric_columns.
            ("addr_state".into(), Column::Str(vec!["NY".into(), "CA".into(), "TX".into(), "WA".into()])),
            ("id".into(), Column::Int(vec![1, 2, 3, 4])),
        ])
        .unwrap()
    }

    #[test]
    fn select_numeric_columns_skips_str_and_excluded() {
        let df = make_demo_df();
        let (names, _data) = select_numeric_columns(&df, ALWAYS_EXCLUDED_COLUMNS);
        // Excludes target_default + id, keeps numeric others, drops Str.
        assert!(names.contains(&"loan_amnt".to_string()));
        assert!(names.contains(&"term_int".to_string()));
        assert!(names.contains(&"verified".to_string()));
        assert!(names.contains(&"never_present".to_string())); // not dropped yet
        assert!(!names.contains(&"target_default".to_string()));
        assert!(!names.contains(&"id".to_string()));
        assert!(!names.contains(&"addr_state".to_string()));
    }

    #[test]
    fn drop_useless_columns_removes_all_nan_and_zero_variance() {
        let df = make_demo_df();
        let (names, cols) = select_numeric_columns(&df, ALWAYS_EXCLUDED_COLUMNS);
        let (kept_names, _) = drop_useless_columns(names, cols);
        assert!(kept_names.contains(&"loan_amnt".to_string()));
        assert!(!kept_names.contains(&"never_present".to_string()));
        assert!(!kept_names.contains(&"zero_var".to_string()));
    }

    #[test]
    fn fit_column_stats_handles_nan_and_constant() {
        // Mixed finite + NaN.
        let v = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
        let (mean, std) = fit_column_stats(&v);
        assert!((mean - 2.5).abs() < 1e-9);
        assert!(std > 0.0);
        // All-NaN should default to (0.0, 1.0) so apply_column_transform
        // doesn't blow up.
        let all_nan = vec![f64::NAN; 5];
        assert_eq!(fit_column_stats(&all_nan), (0.0, 1.0));
        // Constant non-NaN should default std to 1.0 so z-score = 0.
        let const_col = vec![7.0; 5];
        let (m, s) = fit_column_stats(&const_col);
        assert!((m - 7.0).abs() < 1e-9);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn apply_column_transform_imputes_then_zscores() {
        let stats = (5.0, 2.0);
        let raw = vec![5.0, 7.0, f64::NAN, 3.0];
        let out = apply_column_transform(&raw, stats.0, stats.1);
        // 5 → (5-5)/2 = 0; 7 → (7-5)/2 = 1; NaN → mean → 0; 3 → (3-5)/2 = -1.
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 1.0).abs() < 1e-9);
        assert!((out[2] - 0.0).abs() < 1e-9);
        assert!((out[3] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn flatten_subset_row_major_uses_provided_indices() {
        let columns = vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ];
        // Select rows 0 and 2 in order.
        let out = flatten_subset_row_major(&columns, &[0, 2]);
        assert_eq!(out, vec![10.0, 1.0, 30.0, 3.0]);
    }

    #[test]
    fn score_logistic_includes_intercept() {
        // beta = [intercept=1.0, w_0=2.0, w_1=-1.0]
        let coeffs = vec![1.0, 2.0, -1.0];
        // 2 rows × 2 features
        let x = vec![1.0, 1.0, 0.5, -0.5];
        let eta = score_logistic(&coeffs, &x, 2, 2);
        // row 0: 1 + 2*1 + (-1)*1 = 2
        // row 1: 1 + 2*0.5 + (-1)*(-0.5) = 1 + 1 + 0.5 = 2.5
        assert!((eta[0] - 2.0).abs() < 1e-9);
        assert!((eta[1] - 2.5).abs() < 1e-9);
    }
}
