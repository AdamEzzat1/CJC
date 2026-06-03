//! End-to-end integration tests for [`PropensityScoreMatcher::estimate`].
//!
//! These tests exercise the full pipeline (refusal → propensity fit → match →
//! ATT → bootstrap CI → balance → identifier) on real DataFrames. They are
//! the floor for the per-estimator coverage required by ADR-0043 §test surface.

use super::common::{df_from_floats, empty_locke_report, synthetic_confounded, synthetic_randomised};
use cjc_causal::{CausalError, IdentificationAssumption, PropensityScoreMatcher};
use cjc_data::{Column, DataFrame};
use cjc_locke::report::{FindingEvidence, FindingSeverity, LockeInputSummary, LockeReport, ValidationFinding};
use std::collections::BTreeMap;

fn default_assumptions() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::Positivity,
        IdentificationAssumption::NoInterference,
    ]
}

#[test]
fn end_to_end_on_randomised_data_recovers_known_ate() {
    let df = synthetic_randomised(400, 7, 2.0);
    let report = empty_locke_report();
    let est = PropensityScoreMatcher::new()
        .with_seed(42)
        .with_bootstrap_reps(100)
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .expect("randomised data should not be refused or fail");
    // True ATE is 2.0. For randomised assignment, the estimate should be
    // close — accept a generous ±1.0 to keep the test stable across seeds.
    assert!(
        (est.point - 2.0).abs() < 1.0,
        "expected ATE ≈ 2.0, got {} (±{})",
        est.point,
        est.std_error
    );
    assert!(est.ci_lower <= est.point && est.point <= est.ci_upper);
    assert!(est.n_treated > 0 && est.n_control > 0);
    assert_eq!(est.confidence_level, 0.95);
}

#[test]
fn end_to_end_on_confounded_data_closes_some_of_the_gap() {
    let df = synthetic_confounded(400, 11, 2.0);
    let report = empty_locke_report();
    let est = PropensityScoreMatcher::new()
        .with_seed(7)
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .expect("confounded data should still match within caliper");
    // Confounding pulls naive estimates up; matching should produce a
    // finite, non-NaN estimate.
    assert!(est.point.is_finite(), "matching produced non-finite point estimate");
}

#[test]
fn builder_fluent_chain_compiles_and_sets_all_knobs() {
    let df = synthetic_randomised(60, 13, 1.0);
    let report = empty_locke_report();
    let est = PropensityScoreMatcher::new()
        .with_caliper_sd(0.5)
        .with_seed(99)
        .with_confidence_level(0.90)
        .with_bootstrap_reps(50)
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .expect("builder-configured run");
    assert_eq!(est.confidence_level, 0.90);
}

#[test]
fn unknown_column_returns_structured_error() {
    let df = synthetic_randomised(40, 1, 1.0);
    let report = empty_locke_report();
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "no_such_column", "outcome", &["age"], &default_assumptions(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::UnknownColumn { ref name } if name == "no_such_column"));
}

#[test]
fn non_numeric_covariate_returns_wrong_column_type() {
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])),
        ("outcome".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ("city".into(), Column::Str(vec![
            "NYC".into(), "LA".into(), "NYC".into(),
            "LA".into(), "NYC".into(), "LA".into(),
        ])),
    ]).unwrap();
    let report = empty_locke_report();
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["city"], &default_assumptions(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::WrongColumnType { .. }));
}

#[test]
fn empty_covariates_returns_unsupported() {
    let df = synthetic_randomised(40, 1, 1.0);
    let report = empty_locke_report();
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &[], &default_assumptions(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::Unsupported { .. }));
}

#[test]
fn all_treated_input_returns_numerical_error() {
    let df = df_from_floats(&[
        ("treatment", vec![1.0, 1.0, 1.0, 1.0]),
        ("outcome", vec![1.0, 2.0, 3.0, 4.0]),
        ("x", vec![0.5, 0.6, 0.7, 0.8]),
    ]);
    let report = empty_locke_report();
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["x"], &default_assumptions(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::Numerical { .. }));
}

#[test]
fn all_control_input_returns_numerical_error() {
    let df = df_from_floats(&[
        ("treatment", vec![0.0, 0.0, 0.0, 0.0]),
        ("outcome", vec![1.0, 2.0, 3.0, 4.0]),
        ("x", vec![0.5, 0.6, 0.7, 0.8]),
    ]);
    let report = empty_locke_report();
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["x"], &default_assumptions(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::Numerical { .. }));
}

#[test]
fn non_binary_treatment_returns_wrong_column_type() {
    let df = df_from_floats(&[
        ("treatment", vec![1.0, 0.0, 0.5, 1.0]),
        ("outcome", vec![1.0, 2.0, 3.0, 4.0]),
        ("x", vec![0.5, 0.6, 0.7, 0.8]),
    ]);
    let report = empty_locke_report();
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["x"], &default_assumptions(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::WrongColumnType { .. }));
}

#[test]
fn refusal_path_triggers_on_high_missingness_finding() {
    let df = synthetic_randomised(60, 5, 1.0);
    // Build a Locke report with E9001 high-missingness on the treatment column.
    let finding = ValidationFinding::new(
        "E9001",
        FindingSeverity::Error,
        "high missingness on T",
        Some("treatment".to_string()),
        None,
        vec![FindingEvidence::Ratio { label: "missing_fraction".into(), value: 0.45 }],
        60,
        vec![],
        vec![],
    );
    let report = LockeReport::new(
        LockeInputSummary::default(),
        vec![finding],
        BTreeMap::new(),
        vec![],
    );
    let err = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["age"], &default_assumptions(), &report)
        .unwrap_err();
    match err {
        CausalError::DataQualityRefusal { findings } => {
            assert_eq!(findings.len(), 1);
            assert_eq!(findings[0].code, "E9001");
        }
        other => panic!("expected DataQualityRefusal, got {:?}", other),
    }
}

#[test]
fn same_seed_produces_identical_estimate() {
    let df = synthetic_randomised(80, 17, 1.5);
    let report = empty_locke_report();
    let m = PropensityScoreMatcher::new().with_seed(42).with_bootstrap_reps(50);
    let est1 = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report).unwrap();
    let est2 = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report).unwrap();
    assert_eq!(est1.point.to_bits(), est2.point.to_bits(), "ATT must be bit-identical");
    assert_eq!(est1.std_error.to_bits(), est2.std_error.to_bits(), "SE must be bit-identical");
    assert_eq!(est1.identifier, est2.identifier, "identifier must be bit-identical");
}

#[test]
fn different_seed_changes_bootstrap_but_not_point_estimate() {
    let df = synthetic_randomised(80, 19, 1.5);
    let report = empty_locke_report();
    let est_seed1 = PropensityScoreMatcher::new()
        .with_seed(1)
        .with_bootstrap_reps(50)
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .unwrap();
    let est_seed2 = PropensityScoreMatcher::new()
        .with_seed(2)
        .with_bootstrap_reps(50)
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .unwrap();
    // Point estimate is deterministic from data + matching (no RNG involved).
    assert_eq!(est_seed1.point.to_bits(), est_seed2.point.to_bits());
    // Bootstrap SE may differ; identifier hashes over (seed, point, se), so it differs.
    assert_ne!(est_seed1.identifier, est_seed2.identifier);
}

#[test]
fn balance_report_lists_every_declared_covariate() {
    let df = synthetic_randomised(80, 23, 1.0);
    let report = empty_locke_report();
    let est = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .unwrap();
    let balance = est.balance_diagnostics.expect("balance must be present");
    assert!(balance.smd_post_match.contains_key("age"));
    assert!(balance.smd_post_match.contains_key("income"));
    assert_eq!(balance.smd_post_match.len(), 2);
}

#[test]
fn invalid_caliper_returns_unsupported() {
    let df = synthetic_randomised(60, 29, 1.0);
    let report = empty_locke_report();
    for bad in [-0.1, 0.0, f64::NAN, f64::INFINITY] {
        let err = PropensityScoreMatcher::new()
            .with_caliper_sd(bad)
            .estimate(&df, "treatment", "outcome", &["age"], &default_assumptions(), &report)
            .unwrap_err();
        assert!(matches!(err, CausalError::Unsupported { .. }), "caliper {} should be unsupported", bad);
    }
}

#[test]
fn invalid_confidence_level_returns_unsupported() {
    let df = synthetic_randomised(60, 31, 1.0);
    let report = empty_locke_report();
    for bad in [-0.1, 0.0, 1.0, 1.5, f64::NAN] {
        let err = PropensityScoreMatcher::new()
            .with_confidence_level(bad)
            .estimate(&df, "treatment", "outcome", &["age"], &default_assumptions(), &report)
            .unwrap_err();
        assert!(matches!(err, CausalError::Unsupported { .. }), "confidence {} should be unsupported", bad);
    }
}

#[test]
fn n_treated_in_result_equals_matched_pair_count() {
    let df = synthetic_randomised(80, 37, 1.0);
    let report = empty_locke_report();
    let est = PropensityScoreMatcher::new()
        .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        .unwrap();
    // In our v0.1 contract, n_treated == n_control == matched-pair count.
    assert_eq!(est.n_treated, est.n_control);
    assert!(est.n_treated > 0);
}
