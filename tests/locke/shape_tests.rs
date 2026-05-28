//! Integration tests for v0.6.3 distribution-shape diagnostics (E9024).

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{validate, ValidateOptions},
    detect_distribution_shape, skew_and_kurtosis, top_k_modes, FindingSeverity, ShapeConfig,
    ValidationConfig,
};

fn mk_float(name: &str, v: Vec<f64>) -> DataFrame {
    DataFrame::from_columns(vec![(name.into(), Column::Float(v))]).unwrap()
}

#[test]
fn e9024_fires_via_validate_on_skewed_column() {
    let mut v: Vec<f64> = vec![0.0; 100];
    v.extend(vec![100.0; 10]);
    let df = mk_float("x", v);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "skewed".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let f = report
        .findings
        .iter()
        .find(|f| f.code == "E9024")
        .expect("E9024 expected");
    assert_eq!(f.severity, FindingSeverity::Notice);
}

#[test]
fn e9024_quiet_on_normal_data() {
    let v: Vec<f64> = (-50..50).map(|i| i as f64).collect();
    let df = mk_float("x", v);
    let f = detect_distribution_shape(&df, &ShapeConfig::default());
    assert!(f.is_empty());
}

#[test]
fn shape_is_deterministic_across_runs() {
    let mut v: Vec<f64> = vec![0.0; 100];
    v.extend(vec![100.0; 10]);
    let df = mk_float("x", v);
    let cfg = ShapeConfig::default();
    let a = detect_distribution_shape(&df, &cfg);
    let b = detect_distribution_shape(&df, &cfg);
    assert_eq!(a, b);
}

#[test]
fn skew_and_kurtosis_handles_nan() {
    let v: Vec<f64> = vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0];
    let (s, k) = skew_and_kurtosis(&v).unwrap();
    assert!(s.is_finite());
    assert!(k.is_finite());
}

#[test]
fn top_k_modes_returns_sorted_results() {
    let df = mk_float("x", vec![3.0, 3.0, 3.0, 2.0, 2.0, 1.0]);
    let col = df.get_column("x").unwrap();
    let modes = top_k_modes(col, 3);
    assert_eq!(modes[0].0, "3");
    assert_eq!(modes[0].1, 3);
    assert_eq!(modes[1].0, "2");
    assert_eq!(modes[2].0, "1");
}

#[test]
fn e9024_handles_int_columns() {
    let mut v: Vec<i64> = vec![0; 100];
    v.extend(vec![100; 10]);
    let df = DataFrame::from_columns(vec![("counts".into(), Column::Int(v))]).unwrap();
    let f = detect_distribution_shape(&df, &ShapeConfig::default());
    assert!(f.iter().any(|x| x.code == "E9024"));
}

#[test]
fn e9024_skips_short_columns() {
    let v: Vec<f64> = vec![0.0, 100.0]; // n_valid=2 < min_n_valid=20
    let df = mk_float("x", v);
    let f = detect_distribution_shape(&df, &ShapeConfig::default());
    assert!(f.is_empty());
}
