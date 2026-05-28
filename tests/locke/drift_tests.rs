//! Integration tests for the Locke drift / induction-risk module.

use cjc_data::{Column, DataFrame};
use cjc_locke::drift::{compare, DriftConfig};

fn mk(name: &str, v: Vec<f64>) -> (String, Column) {
    (name.into(), Column::Float(v))
}

#[test]
fn identical_train_and_test_report_no_significant_drift() {
    let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let train = DataFrame::from_columns(vec![mk("x", v.clone())]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", v)]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    // v0.2: KS D (E9039) replaces PSI (E9033) as the default numeric drift signal.
    let drift_codes = ["E9030", "E9031", "E9034", "E9039"];
    let drift_findings = r
        .findings
        .iter()
        .filter(|f| drift_codes.contains(&f.code))
        .count();
    assert_eq!(drift_findings, 0, "no drift expected, got: {:?}", r.findings);
}

#[test]
fn large_mean_shift_is_error() {
    let train = DataFrame::from_columns(vec![mk("x", vec![1.0; 200])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![10.0; 200])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "E9030")
        .expect("mean shift finding");
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Error);
}

#[test]
fn psi_detects_distribution_shift() {
    // Train uniform 0..1; test concentrated near 1.
    let train: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
    let test: Vec<f64> = vec![0.95; 1000];
    let train_df = DataFrame::from_columns(vec![mk("x", train)]).unwrap();
    let test_df = DataFrame::from_columns(vec![mk("x", test)]).unwrap();
    let r = compare(&train_df, &test_df, &DriftConfig::default());
    assert!(r.findings.iter().any(|f| f.code == "E9039"));
}

#[test]
fn schema_train_only_test_only_columns_listed() {
    let train = DataFrame::from_columns(vec![mk("a", vec![1.0]), mk("b", vec![2.0])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("a", vec![1.0]), mk("c", vec![3.0])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert_eq!(r.train_only_columns, vec!["b".to_string()]);
    assert_eq!(r.test_only_columns, vec!["c".to_string()]);
    assert_eq!(r.shared_columns, vec!["a".to_string()]);
}

#[test]
fn repeated_runs_produce_identical_reports() {
    let v: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
    let train = DataFrame::from_columns(vec![mk("x", v.clone())]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", v.iter().map(|x| x + 0.05).collect())]).unwrap();
    let a = compare(&train, &test, &DriftConfig::default());
    let b = compare(&train, &test, &DriftConfig::default());
    assert_eq!(a, b);
}

#[test]
fn ks_d_evidence_is_reported_when_drift_present() {
    // Two disjoint supports → KS D = 1.0 → Error severity.
    let train: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let test: Vec<f64> = (1000..1100).map(|i| i as f64).collect();
    let train_df = DataFrame::from_columns(vec![mk("x", train)]).unwrap();
    let test_df = DataFrame::from_columns(vec![mk("x", test)]).unwrap();
    let r = compare(&train_df, &test_df, &DriftConfig::default());
    let ks = r
        .findings
        .iter()
        .find(|f| f.code == "E9039")
        .expect("E9039 KS finding");
    assert_eq!(ks.severity, cjc_locke::FindingSeverity::Error);
    let d = ks
        .evidence
        .iter()
        .find_map(|e| match e {
            cjc_locke::FindingEvidence::Metric { label, value } if label == "ks_d" => {
                Some(*value)
            }
            _ => None,
        })
        .unwrap();
    assert!((d - 1.0).abs() < 1e-9, "KS D should be 1.0 for disjoint supports, got {}", d);
}

#[test]
fn ks_d_is_deterministic_in_drift_report() {
    let train: Vec<f64> = (0..200).map(|i| (i as f64).sin()).collect();
    let test: Vec<f64> = (0..200).map(|i| (i as f64).cos()).collect();
    let train_df = DataFrame::from_columns(vec![mk("x", train)]).unwrap();
    let test_df = DataFrame::from_columns(vec![mk("x", test)]).unwrap();
    let a = compare(&train_df, &test_df, &DriftConfig::default());
    let b = compare(&train_df, &test_df, &DriftConfig::default());
    assert_eq!(a, b);
}

#[test]
fn small_test_set_triggers_low_power_warning() {
    let train = DataFrame::from_columns(vec![mk("x", vec![0.0; 1000])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![0.0; 5])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert!(r.findings.iter().any(|f| f.code == "E9036"));
}
