//! Integration tests for Locke validators over real `cjc-data` DataFrames.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{validate, ValidateOptions},
    detect_duplicate_keys, detect_impossible_values, detect_missingness,
    ExpectedSchema, FindingSeverity, ImpossibleValueRule, NullMask, NullMaskMap,
    ValidationConfig,
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn validate_end_to_end_emits_findings_and_run_id() {
    let df = DataFrame::from_columns(vec![
        ("age".into(), Column::Float(vec![25.0, 30.0, f64::NAN, 30.0])),
        ("country".into(), Column::Str(vec!["us".into(), "uk".into(), "us".into(), "us".into()])),
    ])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "tests/integration".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    assert!(!report.findings.is_empty());
    assert_ne!(report.run_id.0, 0);
    assert_eq!(report.input.n_rows, 4);
    assert_eq!(report.input.n_cols, 2);
}

#[test]
fn integration_picks_up_duplicates_and_missingness_simultaneously() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![1.0, 1.0, f64::NAN, 2.0])),
        ("y".into(), Column::Int(vec![10, 10, 20, 20])),
    ])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "dup+miss".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // Should see at least one duplicate finding (E9003) and one missingness (E9001).
    assert!(report.findings.iter().any(|f| f.code == "E9003"));
    assert!(report.findings.iter().any(|f| f.code == "E9001"));
}

#[test]
fn validate_with_schema_mismatch_emits_typed_findings() {
    let df = DataFrame::from_columns(vec![
        ("age".into(), Column::Int(vec![1, 2, 3])),
        ("extra".into(), Column::Int(vec![9, 9, 9])),
    ])
    .unwrap();
    let mut cols = BTreeMap::new();
    cols.insert("age".into(), "Float".into());
    cols.insert("missing".into(), "Int".into());
    let sch = ExpectedSchema {
        columns: cols,
        strict_extra: false,
    };
    let opts = ValidateOptions {
        dataset_label: "schema".into(),
        config: ValidationConfig::default(),
        expected_schema: Some(sch),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    assert!(r.findings.iter().any(|f| f.code == "E9020")); // missing column
    assert!(r.findings.iter().any(|f| f.code == "E9021")); // type mismatch
}

#[test]
fn impossible_rule_for_allowed_strings_finds_violation() {
    let df = DataFrame::from_columns(vec![(
        "country".into(),
        Column::Str(vec!["us".into(), "atlantis".into()]),
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
fn duplicate_keys_can_combine_with_other_validators() {
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 2])),
        ("v".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap();
    let dup = detect_duplicate_keys(&df, "id");
    assert!(dup.iter().any(|f| f.code == "E9004" && f.severity == FindingSeverity::Error));
}

#[test]
fn worst_severity_reflects_max() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![1.0; 100])),
        ("y".into(), Column::Int(vec![5; 100])),
    ])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "constants".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    // Constant + 99% duplicate rows → at least Warning. The duplicate
    // detector (E9003) sees 99/100 rows as duplicates → Error severity,
    // so worst should be Error here.
    assert_eq!(r.worst_severity(), FindingSeverity::Error);
    // And at least one constant finding (E9010) should be present.
    assert!(r.findings.iter().any(|f| f.code == "E9010"));
}

#[test]
fn empty_dataframe_does_not_panic() {
    let df = DataFrame::new();
    let opts = ValidateOptions {
        dataset_label: "empty".into(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    assert_eq!(r.input.n_rows, 0);
    assert_eq!(r.input.n_cols, 0);
}

#[test]
fn null_mask_for_str_column_emits_missingness_finding() {
    let df = DataFrame::from_columns(vec![(
        "country".into(),
        Column::Str(vec!["us".into(), "uk".into(), "atlantis".into(), "us".into()]),
    )])
    .unwrap();
    let mut masks = NullMaskMap::new();
    // Mark rows 1 and 3 as null.
    masks.insert("country".into(), NullMask::from_indices([1, 3]));
    let opts = ValidateOptions {
        dataset_label: "null-mask-str".into(),
        null_masks: masks,
        ..Default::default()
    };
    let r = validate(&df, &opts);
    let miss = r
        .findings
        .iter()
        .find(|f| f.code == "E9001" && f.column.as_deref() == Some("country"))
        .expect("E9001 should fire from null mask on Str");
    // and E9002 (limitation note) must NOT fire for country anymore.
    assert!(!r
        .findings
        .iter()
        .any(|f| f.code == "E9002" && f.column.as_deref() == Some("country")));
    // n_missing must be 2.
    let n = miss
        .evidence
        .iter()
        .find_map(|e| match e {
            cjc_locke::FindingEvidence::Count { label, value } if label == "n_missing" => {
                Some(*value)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(n, 2);
}

#[test]
fn missingness_severity_scales_with_rate() {
    // 60% missing -> Error
    let mut v = vec![1.0; 100];
    for x in v.iter_mut().take(60) {
        *x = f64::NAN;
    }
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
    let cfg = ValidationConfig::default();
    let findings = detect_missingness(&df, &cfg, &NullMaskMap::new());
    let miss = findings.iter().find(|f| f.code == "E9001").unwrap();
    assert_eq!(miss.severity, FindingSeverity::Error);
}
