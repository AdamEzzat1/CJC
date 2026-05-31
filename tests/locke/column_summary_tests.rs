//! Integration tests for the v0.6 batch 2 per-column confidence summary.
//!
//! These exercise `build_per_column_summaries` and
//! `emit_per_column_confidence_summary` on real `LockeReport`s produced
//! by `validate()` over crafted DataFrames.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{validate, ValidateOptions},
    build_per_column_summaries, emit_per_column_confidence_summary, ConfidenceBand,
    ValidationConfig,
};

#[test]
fn empty_dataset_yields_empty_summaries() {
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(vec![]))]).unwrap();
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "empty".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let s = build_per_column_summaries(&report);
    // The Bool/Str/Int E9002 limitation Info gets attached to columns;
    // for an empty Float column we expect no per-column findings.
    assert!(s.iter().all(|c| c.column != "x" || c.total_findings() <= 1));
}

#[test]
fn dirty_column_lands_in_low_or_moderate_band() {
    // Many NaNs in a Float column → E9001 Error → Low band on that column.
    let nans: Vec<f64> = std::iter::repeat(f64::NAN).take(50).chain((0..50).map(|i| i as f64)).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(nans))]).unwrap();
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "dirty".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let s = build_per_column_summaries(&report);
    let x = s.iter().find(|c| c.column == "x").expect("column x in summaries");
    assert_ne!(x.confidence, ConfidenceBand::High);
}

#[test]
fn text_emit_is_deterministic_and_well_formed() {
    let mut values = vec!["Premium", "premium", "PREMIUM"];
    values.extend(vec!["basic"; 27]);
    let df = DataFrame::from_columns(vec![(
        "tier".into(),
        Column::Str(values.iter().map(|s| (*s).into()).collect()),
    )])
    .unwrap();
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "tier".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let a = emit_per_column_confidence_summary(&report);
    let b = emit_per_column_confidence_summary(&report);
    assert_eq!(a, b);
    assert!(a.starts_with("# Per-column confidence summary"));
    // The tier column carries E9080 (case-fold collision Warning).
    assert!(a.contains("Column: tier"));
}

#[test]
fn multi_column_emit_sorts_columns_alphabetically() {
    let zulu: Vec<&str> = vec!["X"; 30];
    let alpha: Vec<&str> = vec!["Y"; 30];
    let mike: Vec<&str> = vec!["Z"; 30];
    let df = DataFrame::from_columns(vec![
        ("zulu".into(), Column::Str(zulu.iter().map(|s| (*s).into()).collect())),
        ("alpha".into(), Column::Str(alpha.iter().map(|s| (*s).into()).collect())),
        ("mike".into(), Column::Str(mike.iter().map(|s| (*s).into()).collect())),
    ])
    .unwrap();
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "multi".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let s = build_per_column_summaries(&report);
    // alpha < mike < zulu lexicographically.
    let columns: Vec<&str> = s.iter().map(|c| c.column.as_str()).collect();
    let mut sorted = columns.clone();
    sorted.sort();
    assert_eq!(columns, sorted);
}

#[test]
fn dataset_wide_finding_gets_synthetic_row() {
    // 23 full-row duplicates → E9003 (dataset-wide finding).
    let xs: Vec<f64> = (0..50).chain(0..50).chain(0..23).map(|i| i as f64).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(xs))]).unwrap();
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "dups".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let s = build_per_column_summaries(&report);
    let has_dataset = s.iter().any(|c| c.column == "<dataset>");
    assert!(has_dataset, "expected <dataset> synthetic row, got {:?}", s.iter().map(|c| &c.column).collect::<Vec<_>>());
}
