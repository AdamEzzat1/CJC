//! Integration tests for v0.6.3 multi-class target-leakage AUC (E9063).

use cjc_data::{Column, DataFrame};
use cjc_locke::leakage::{
    detect_target_leakage_multiclass, multiclass_max_one_vs_rest_auc, LeakageConfig,
};
use cjc_locke::FindingSeverity;

#[test]
fn multiclass_leakage_fires_when_feature_perfectly_predicts_one_class() {
    // Target: 3 classes (0, 1, 2). Feature: 1.0 iff class==2, else 0.0
    // → AUC for class 2 is 1.0 → E9063 Error.
    let n_per_class = 30;
    let target: Vec<i64> = (0..3)
        .flat_map(|c| std::iter::repeat(c).take(n_per_class))
        .collect();
    let feat: Vec<f64> = target
        .iter()
        .map(|&t| if t == 2 { 1.0 } else { 0.0 })
        .collect();
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Float(feat)),
        ("target".into(), Column::Int(target)),
    ])
    .unwrap();
    let f = detect_target_leakage_multiclass(&df, "target", &LeakageConfig::default());
    let e9063 = f
        .iter()
        .find(|x| x.code == "E9063" && x.column.as_deref() == Some("feat"))
        .expect("E9063 expected");
    assert_eq!(e9063.severity, FindingSeverity::Error);
}

#[test]
fn multiclass_leakage_quiet_on_noise_feature() {
    let n_per_class = 30;
    let target: Vec<i64> = (0..3)
        .flat_map(|c| std::iter::repeat(c).take(n_per_class))
        .collect();
    // Constant feature → AUC ≈ 0.5 for every class → no E9063.
    let feat: Vec<f64> = vec![0.5; target.len()];
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Float(feat)),
        ("target".into(), Column::Int(target)),
    ])
    .unwrap();
    let f = detect_target_leakage_multiclass(&df, "target", &LeakageConfig::default());
    assert!(f.iter().all(|x| x.code != "E9063"));
}

#[test]
fn multiclass_leakage_skips_binary_target() {
    // Binary target → multi-class detector returns no findings (E9060/E9061
    // owns the binary case).
    let target: Vec<i64> = vec![0i64; 30].into_iter().chain(vec![1; 30]).collect();
    let feat: Vec<f64> = target.iter().map(|&t| t as f64).collect();
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Float(feat)),
        ("target".into(), Column::Int(target)),
    ])
    .unwrap();
    let f = detect_target_leakage_multiclass(&df, "target", &LeakageConfig::default());
    assert!(f.is_empty());
}

#[test]
fn multiclass_leakage_skips_continuous_target() {
    // 100 distinct target values → above default max_classes=20 → skip.
    let target: Vec<i64> = (0..100).collect();
    let feat: Vec<f64> = target.iter().map(|t| *t as f64).collect();
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Float(feat)),
        ("target".into(), Column::Int(target)),
    ])
    .unwrap();
    let f = detect_target_leakage_multiclass(&df, "target", &LeakageConfig::default());
    assert!(f.is_empty());
}

#[test]
fn multiclass_max_ovr_auc_returns_max_across_classes() {
    let target: Vec<u32> = vec![0; 30].into_iter().chain(vec![1; 30]).chain(vec![2; 30]).collect();
    let feat: Vec<f64> = target.iter().map(|&t| if t == 1 { 1.0 } else { 0.0 }).collect();
    let auc = multiclass_max_one_vs_rest_auc(&feat, &target, 3, 10).unwrap();
    // Class 1 is perfectly identified → max OVR AUC = 1.0
    assert!((auc - 1.0).abs() < 1e-9, "expected ~1.0, got {}", auc);
}

#[test]
fn multiclass_leakage_is_deterministic() {
    let n_per_class = 30;
    let target: Vec<i64> = (0..3)
        .flat_map(|c| std::iter::repeat(c).take(n_per_class))
        .collect();
    let feat: Vec<f64> = target.iter().map(|&t| if t == 2 { 1.0 } else { 0.0 }).collect();
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Float(feat)),
        ("target".into(), Column::Int(target)),
    ])
    .unwrap();
    let cfg = LeakageConfig::default();
    let a = detect_target_leakage_multiclass(&df, "target", &cfg);
    let b = detect_target_leakage_multiclass(&df, "target", &cfg);
    assert_eq!(a, b);
}
