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
}
