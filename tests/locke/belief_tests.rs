//! Integration tests for belief score + report.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::belief::{sample_score_from_n, BeliefScore};
use cjc_locke::validation::ValidationConfig;

#[test]
fn perfect_dataset_yields_high_belief() {
    let v: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
    let opts = ValidateOptions {
        dataset_label: "perfect".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    let b = belief_report_from_locke(&r);
    assert!(b.score.overall > 0.95, "expected >0.95, got {}", b.score.overall);
    assert!(b.score.missingness_score >= 0.99);
    assert!(b.score.duplication_score >= 0.99);
}

#[test]
fn missingness_drags_score() {
    let mut v = vec![1.0; 100];
    for x in v.iter_mut().take(50) {
        *x = f64::NAN;
    }
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
    let opts = ValidateOptions {
        dataset_label: "missingness".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    let b = belief_report_from_locke(&r);
    assert!(b.score.missingness_score < 0.6, "expected <0.6, got {}", b.score.missingness_score);
}

#[test]
fn duplication_drags_score() {
    let df = DataFrame::from_columns(vec![("x".into(), Column::Int(vec![1; 100]))]).unwrap();
    let opts = ValidateOptions {
        dataset_label: "dup".into(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    let b = belief_report_from_locke(&r);
    // 99 duplicate rows -> Error severity penalty -> low duplication score.
    assert!(b.score.duplication_score < 0.9, "expected <0.9, got {}", b.score.duplication_score);
}

#[test]
fn explain_is_stable_text() {
    let s = BeliefScore::from_dimensions(1.0, 0.5, 0.7, 1.0, 0.3, 0.9, 1.0, 1.0);
    let a = s.explain();
    let b = s.explain();
    assert_eq!(a, b);
    assert!(a.contains("overall="));
    assert!(a.contains("schema      = 1.000"));
    assert!(a.contains("missingness = 0.500"));
}

#[test]
fn weighted_belief_score_emphasises_user_priority() {
    use cjc_locke::BeliefWeights;
    // Suppose the user cares mostly about missingness — boost its weight 10x.
    let mut w = BeliefWeights::default();
    w.missingness = 10.0;
    let bad_missingness = BeliefScore::from_dimensions_weighted(
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &w,
    );
    let bad_drift = BeliefScore::from_dimensions_weighted(
        1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, &w,
    );
    // With missingness × 10, bad missingness should drag overall MUCH lower
    // than bad drift.
    assert!(bad_missingness.overall < bad_drift.overall - 0.2);
}

#[test]
fn default_weights_are_bit_identical_to_unweighted() {
    use cjc_locke::BeliefWeights;
    let s1 = BeliefScore::from_dimensions(0.42, 0.31, 0.99, 0.7, 0.5, 0.8, 0.6, 0.4);
    let s2 = BeliefScore::from_dimensions_weighted(
        0.42, 0.31, 0.99, 0.7, 0.5, 0.8, 0.6, 0.4,
        &BeliefWeights::default(),
    );
    assert_eq!(s1.overall.to_bits(), s2.overall.to_bits());
}

#[test]
fn sample_score_curve_is_monotonic_in_n() {
    let xs: Vec<u64> = (0..1000).step_by(10).collect();
    let mut last = -1.0;
    for n in xs {
        let s = sample_score_from_n(n);
        assert!(s + 1e-12 >= last);
        last = s;
    }
}
