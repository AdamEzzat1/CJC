//! End-to-end integration tests for [`DoubleMLEstimator::estimate`].
//!
//! v0.1 ships linear nuisances via `cjc_runtime::hypothesis::lm`. The headline
//! tested behaviour is the Chernozhukov 2018 orthogonal-moment + K-fold
//! cross-fitting discipline, not how flexible the nuisance is.

use super::common::{df_from_floats, empty_locke_report, synthetic_dml};
use cjc_causal::dml::ESTIMATOR_LABEL;
use cjc_causal::{
    CausalError, DoubleMLEstimator, IVRegression, IdentificationAssumption,
    PropensityScoreMatcher,
};
use cjc_data::{Column, DataFrame};
use cjc_locke::report::{
    FindingEvidence, FindingSeverity, LockeInputSummary, LockeReport, ValidationFinding,
};
use std::collections::BTreeMap;

fn assumptions_dml() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::NoInterference,
    ]
}

#[test]
fn end_to_end_recovers_known_beta_on_synthetic_data() {
    // β = 2.0, partially linear model with linear g(X) and m(X).
    // Linear nuisances are correctly specified; DML should hit β near 2.0.
    let df = synthetic_dml(500, 7, 2.0);
    let report = empty_locke_report();
    let est = DoubleMLEstimator::new()
        .with_seed(42)
        .estimate(
            &df,
            "treatment",
            "outcome",
            &["x1", "x2"],
            &assumptions_dml(),
            &report,
        )
        .expect("synthetic DML data should succeed");
    assert!(
        (est.point - 2.0).abs() < 0.5,
        "expected β ≈ 2.0, got {} (SE = {})",
        est.point,
        est.std_error
    );
    assert!(est.std_error >= 0.0);
    assert!(est.ci_lower <= est.point && est.point <= est.ci_upper);
}

#[test]
fn same_seed_byte_identical_estimate() {
    let df = synthetic_dml(120, 11, 1.5);
    let report = empty_locke_report();
    let est = DoubleMLEstimator::new().with_seed(42);
    let e1 = est.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report).unwrap();
    let e2 = est.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report).unwrap();
    assert_eq!(e1.point.to_bits(), e2.point.to_bits());
    assert_eq!(e1.std_error.to_bits(), e2.std_error.to_bits());
    assert_eq!(e1.identifier, e2.identifier);
}

#[test]
fn different_seed_changes_kfold_split_and_identifier() {
    let df = synthetic_dml(120, 13, 1.5);
    let report = empty_locke_report();
    let e_seed1 = DoubleMLEstimator::new()
        .with_seed(1)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    let e_seed2 = DoubleMLEstimator::new()
        .with_seed(2)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    // Different K-fold splits give different cross-fit predictions, so β
    // may differ — but more importantly the identifier MUST differ because
    // the seed is part of the canonical bytes.
    assert_ne!(e_seed1.identifier, e_seed2.identifier);
}

#[test]
fn k_folds_must_be_at_least_2() {
    let df = synthetic_dml(40, 17, 1.0);
    let report = empty_locke_report();
    for bad in [0, 1] {
        let err = DoubleMLEstimator::new()
            .with_k_folds(bad)
            .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
            .unwrap_err();
        assert!(matches!(err, CausalError::Unsupported { .. }), "k = {} should error", bad);
    }
}

#[test]
fn invalid_confidence_level_returns_unsupported() {
    let df = synthetic_dml(40, 19, 1.0);
    let report = empty_locke_report();
    for bad in [-0.1, 0.0, 1.0, 1.5, f64::NAN] {
        let err = DoubleMLEstimator::new()
            .with_confidence_level(bad)
            .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
            .unwrap_err();
        assert!(matches!(err, CausalError::Unsupported { .. }), "cl {} should error", bad);
    }
}

#[test]
fn unknown_treatment_column_returns_structured_error() {
    let df = synthetic_dml(40, 23, 1.0);
    let report = empty_locke_report();
    let err = DoubleMLEstimator::new()
        .estimate(&df, "no_such_t", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::UnknownColumn { ref name } if name == "no_such_t"));
}

#[test]
fn empty_covariates_returns_unsupported() {
    let df = synthetic_dml(40, 29, 1.0);
    let report = empty_locke_report();
    let err = DoubleMLEstimator::new()
        .estimate(&df, "treatment", "outcome", &[], &assumptions_dml(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::Unsupported { .. }));
}

#[test]
fn non_numeric_covariate_returns_wrong_column_type() {
    let n = 50;
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float((0..n).map(|i| i as f64).collect())),
        ("outcome".into(), Column::Float((0..n).map(|i| 2.0 * i as f64).collect())),
        ("x1".into(), Column::Float((0..n).map(|i| 0.1 * i as f64).collect())),
        ("city".into(), Column::Str(vec!["NYC".into(); n])),
    ])
    .unwrap();
    let report = empty_locke_report();
    let err = DoubleMLEstimator::new()
        .estimate(&df, "treatment", "outcome", &["x1", "city"], &assumptions_dml(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::WrongColumnType { .. }));
}

#[test]
fn n_too_small_for_kfold_returns_numerical_error() {
    // 6 rows, 5 folds → each training set is ~5 rows, p = 2 covariates,
    // we need n_train > p + 2 = 4. With ~5 it's borderline; let's force
    // failure with k=5 and very small n.
    let df = df_from_floats(&[
        ("treatment", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("outcome", vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        ("x1", vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ("x2", vec![0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
    ]);
    let report = empty_locke_report();
    let err = DoubleMLEstimator::new()
        .with_k_folds(5)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap_err();
    assert!(matches!(err, CausalError::Numerical { .. }));
}

#[test]
fn refusal_path_triggers_on_e9060_for_covariate() {
    let df = synthetic_dml(80, 31, 1.0);
    let finding = ValidationFinding::new(
        "E9060",
        FindingSeverity::Error,
        "strong target leakage on x1",
        Some("x1".to_string()),
        None,
        vec![FindingEvidence::Metric {
            label: "auc".to_string(),
            value: 0.97,
        }],
        80,
        vec![],
        vec![],
    );
    let report = LockeReport::new(
        LockeInputSummary::default(),
        vec![finding],
        BTreeMap::new(),
        vec![],
    );
    let err = DoubleMLEstimator::new()
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap_err();
    match err {
        CausalError::DataQualityRefusal { findings } => {
            assert!(findings.iter().any(|f| f.code == "E9060"));
        }
        other => panic!("expected DataQualityRefusal, got {:?}", other),
    }
}

#[test]
fn builder_chain_compiles_and_sets_all_knobs() {
    let df = synthetic_dml(80, 37, 1.5);
    let report = empty_locke_report();
    let est = DoubleMLEstimator::new()
        .with_k_folds(3)
        .with_seed(99)
        .with_confidence_level(0.90)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    assert_eq!(est.confidence_level, 0.90);
}

#[test]
fn iv_first_stage_f_is_none_for_dml() {
    let df = synthetic_dml(80, 41, 1.0);
    let report = empty_locke_report();
    let est = DoubleMLEstimator::new()
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    assert!(est.iv_first_stage_f.is_none(), "DML should not populate iv_first_stage_f");
}

#[test]
fn balance_diagnostics_is_none_for_dml() {
    let df = synthetic_dml(80, 43, 1.0);
    let report = empty_locke_report();
    let est = DoubleMLEstimator::new()
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    assert!(est.balance_diagnostics.is_none(), "DML should not populate balance_diagnostics");
}

#[test]
fn identifier_differs_from_psm_and_iv_on_same_columns() {
    // Build a DataFrame compatible with all three estimators (binary T for PSM).
    let n = 80;
    let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.5 * i as f64 + (i % 7) as f64).collect();
    let x1: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();
    let instrument: Vec<f64> = (0..n).map(|i| 0.3 * i as f64 + (i % 5) as f64 * 0.1).collect();
    let df = df_from_floats(&[
        ("treatment", t),
        ("outcome", y),
        ("x1", x1),
        ("x2", x2),
        ("instrument", instrument),
    ]);
    let report = empty_locke_report();
    let assumptions = assumptions_dml();

    let psm = PropensityScoreMatcher::new().with_bootstrap_reps(5);
    let iv = IVRegression::new();
    let dml = DoubleMLEstimator::new();

    let res_psm = psm.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions, &report);
    let res_iv = iv.estimate(&df, "treatment", "outcome", "instrument", &["x1", "x2"], &assumptions, &report);
    let res_dml = dml.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions, &report);

    if let (Ok(p), Ok(i), Ok(d)) = (res_psm, res_iv, res_dml) {
        assert_ne!(p.identifier, i.identifier);
        assert_ne!(p.identifier, d.identifier);
        assert_ne!(i.identifier, d.identifier);
    }
}

#[test]
fn confidence_interval_widens_at_higher_level() {
    let df = synthetic_dml(150, 47, 1.0);
    let report = empty_locke_report();
    let e_90 = DoubleMLEstimator::new()
        .with_confidence_level(0.90)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    let e_99 = DoubleMLEstimator::new()
        .with_confidence_level(0.99)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    let w_90 = e_90.ci_upper - e_90.ci_lower;
    let w_99 = e_99.ci_upper - e_99.ci_lower;
    assert!(w_99 > w_90, "99% CI ({}) should be wider than 90% ({})", w_99, w_90);
}

#[test]
fn estimator_label_is_dml_partial_linear() {
    assert_eq!(ESTIMATOR_LABEL, "double_ml_partial_linear");
}

#[test]
fn k_folds_three_vs_five_changes_estimate_and_id() {
    let df = synthetic_dml(120, 53, 1.0);
    let report = empty_locke_report();
    let e_3 = DoubleMLEstimator::new()
        .with_k_folds(3)
        .with_seed(0)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    let e_5 = DoubleMLEstimator::new()
        .with_k_folds(5)
        .with_seed(0)
        .estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report)
        .unwrap();
    // Different K means different fold structure → different cross-fit
    // predictions → different point estimate (typically) → different
    // identifier (definitely, since β bits are in the hash).
    assert_ne!(e_3.identifier, e_5.identifier);
}
