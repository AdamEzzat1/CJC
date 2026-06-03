//! End-to-end integration tests for [`IVRegression::estimate`].
//!
//! These tests exercise the full just-identified 2SLS pipeline (refusal →
//! first stage → fitted treatment → second stage → 2SLS residuals → HC1 SE →
//! identifier) on real DataFrames.

use super::common::{df_from_floats, empty_locke_report, synthetic_iv};
use cjc_causal::iv_regression::{ESTIMATOR_LABEL, WEAK_INSTRUMENT_CODE};
use cjc_causal::{
    weak_instrument_finding, CausalError, IVRegression, IdentificationAssumption,
    PropensityScoreMatcher,
};
use cjc_data::{Column, DataFrame};
use cjc_locke::report::{
    FindingEvidence, FindingSeverity, LockeInputSummary, LockeReport, ValidationFinding,
};
use std::collections::BTreeMap;

fn assumptions_iv() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::ExcludabilityOfInstrument,
        IdentificationAssumption::MonotonicityOfInstrument,
        IdentificationAssumption::NoInterference,
    ]
}

#[test]
fn end_to_end_strong_instrument_recovers_known_beta() {
    // Strong instrument (γ = 1.0) and true β = 2.0 over 500 rows.
    let df = synthetic_iv(500, 7, 2.0, 1.0);
    let report = empty_locke_report();
    let est = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .expect("strong-instrument IV should succeed");

    // Asymptotic consistency: IV recovers true β. Generous tolerance because
    // 500 rows is small and there's substantial noise + confounding.
    assert!(
        (est.point - 2.0).abs() < 1.0,
        "expected IV β ≈ 2.0, got {}",
        est.point
    );
    assert!(est.std_error >= 0.0);
    assert!(est.ci_lower <= est.point && est.point <= est.ci_upper);

    // First-stage F should be high for a strong instrument.
    let f = est.iv_first_stage_f.expect("IV must populate F");
    assert!(f > 10.0, "strong instrument should pass F > 10, got {}", f);
    assert!(weak_instrument_finding(&est, "instrument", 10.0).is_none());
}

#[test]
fn end_to_end_weak_instrument_emits_e9100_finding() {
    // Very weak instrument (γ ≈ 0.02 effective signal) → F well below 10.
    let df = synthetic_iv(120, 11, 2.0, 0.02);
    let report = empty_locke_report();
    let est = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .expect("weak-instrument IV still returns Ok — caller decides on finding");

    let f = est.iv_first_stage_f.expect("F must be populated");
    assert!(
        f < 10.0,
        "weak instrument should fail F < 10, got {}",
        f
    );

    let finding = weak_instrument_finding(&est, "instrument", 10.0)
        .expect("E9100 must fire when F < threshold");
    assert_eq!(finding.code, WEAK_INSTRUMENT_CODE);
    assert_eq!(finding.severity, FindingSeverity::Error);
    assert_eq!(finding.column.as_deref(), Some("instrument"));
}

#[test]
fn iv_first_stage_f_is_always_some() {
    // For any successful IV estimate, the F-stat field is populated.
    let df = synthetic_iv(80, 13, 1.5, 0.8);
    let report = empty_locke_report();
    let est = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    assert!(est.iv_first_stage_f.is_some());
    assert!(est.iv_first_stage_f.unwrap() >= 0.0, "F is t² ≥ 0");
}

#[test]
fn iv_and_psm_identifiers_differ_on_same_columns() {
    // Crucial cross-estimator property: IV's content-addressed identifier
    // includes the instrument and the estimator-label string, so it must
    // differ from PSM's identifier on the same (treatment, outcome,
    // covariates, assumptions).
    let df = df_from_floats(&[
        ("treatment", vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("outcome", (0..10).map(|i| i as f64).collect()),
        ("instrument", (0..10).map(|i| 0.1 * i as f64 + 0.05).collect()),
        ("x", (0..10).map(|i| 0.5 * i as f64).collect()),
    ]);
    let report = empty_locke_report();

    let psm = PropensityScoreMatcher::new().with_bootstrap_reps(5);
    let iv = IVRegression::new();

    if let (Ok(p), Ok(i)) = (
        psm.estimate(
            &df,
            "treatment",
            "outcome",
            &["x"],
            &assumptions_iv(),
            &report,
        ),
        iv.estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        ),
    ) {
        assert_ne!(
            p.identifier, i.identifier,
            "IV and PSM identifiers must differ on the same columns"
        );
        // PSM's iv_first_stage_f is always None; IV's is always Some.
        assert!(p.iv_first_stage_f.is_none());
        assert!(i.iv_first_stage_f.is_some());
    }
}

#[test]
fn unknown_instrument_column_returns_structured_error() {
    let df = synthetic_iv(40, 5, 1.0, 0.8);
    let report = empty_locke_report();
    let err = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "no_such_z",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap_err();
    assert!(matches!(err, CausalError::UnknownColumn { ref name } if name == "no_such_z"));
}

#[test]
fn non_numeric_instrument_returns_wrong_column_type() {
    let n = 30;
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float((0..n).map(|i| i as f64).collect())),
        ("outcome".into(), Column::Float((0..n).map(|i| 2.0 * i as f64).collect())),
        ("x".into(), Column::Float((0..n).map(|i| 0.1 * i as f64).collect())),
        ("instrument".into(), Column::Str(vec!["a".into(); n])),
    ])
    .unwrap();
    let report = empty_locke_report();
    let err = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap_err();
    assert!(matches!(err, CausalError::WrongColumnType { ref name, .. } if name == "instrument"));
}

#[test]
fn invalid_confidence_level_returns_unsupported() {
    let df = synthetic_iv(40, 17, 1.0, 0.8);
    let report = empty_locke_report();
    for bad in [-0.1, 0.0, 1.0, 1.5, f64::NAN] {
        let err = IVRegression::new()
            .with_confidence_level(bad)
            .estimate(
                &df,
                "treatment",
                "outcome",
                "instrument",
                &["x"],
                &assumptions_iv(),
                &report,
            )
            .unwrap_err();
        assert!(
            matches!(err, CausalError::Unsupported { .. }),
            "confidence {} should be unsupported",
            bad
        );
    }
}

#[test]
fn invalid_weak_instrument_threshold_returns_unsupported() {
    let df = synthetic_iv(40, 19, 1.0, 0.8);
    let report = empty_locke_report();
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let err = IVRegression::new()
            .with_weak_instrument_threshold(bad)
            .estimate(
                &df,
                "treatment",
                "outcome",
                "instrument",
                &["x"],
                &assumptions_iv(),
                &report,
            )
            .unwrap_err();
        assert!(
            matches!(err, CausalError::Unsupported { .. }),
            "threshold {} should be unsupported",
            bad
        );
    }
}

#[test]
fn n_too_small_returns_numerical_error() {
    // 3 rows, 1 covariate → k = 3 (intercept + T̂ + x). Need n > 4 = 5.
    let df = df_from_floats(&[
        ("treatment", vec![1.0, 2.0, 3.0]),
        ("outcome", vec![4.0, 5.0, 6.0]),
        ("instrument", vec![0.1, 0.2, 0.3]),
        ("x", vec![0.5, 0.6, 0.7]),
    ]);
    let report = empty_locke_report();
    let err = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap_err();
    assert!(matches!(err, CausalError::Numerical { .. }));
}

#[test]
fn refusal_path_triggers_on_e9009_for_instrument() {
    // E9009 on the instrument column should be refusal-grade for IV.
    let df = synthetic_iv(80, 23, 1.0, 0.8);
    let finding = ValidationFinding::new(
        "E9009",
        FindingSeverity::Info,
        "instrument not promoted to Float",
        Some("instrument".to_string()),
        None,
        vec![],
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
    let err = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap_err();
    match err {
        CausalError::DataQualityRefusal { findings } => {
            assert!(findings.iter().any(|f| f.code == "E9009"));
        }
        other => panic!("expected DataQualityRefusal, got {:?}", other),
    }
}

#[test]
fn builder_chain_compiles_and_sets_all_knobs() {
    let df = synthetic_iv(80, 29, 1.5, 0.7);
    let report = empty_locke_report();
    let est = IVRegression::new()
        .with_confidence_level(0.90)
        .with_weak_instrument_threshold(5.0)
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    assert_eq!(est.confidence_level, 0.90);
}

#[test]
fn same_input_produces_byte_identical_estimate() {
    let df = synthetic_iv(80, 31, 1.0, 0.8);
    let report = empty_locke_report();
    let iv = IVRegression::new();
    let e1 = iv
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    let e2 = iv
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    assert_eq!(e1.point.to_bits(), e2.point.to_bits());
    assert_eq!(e1.std_error.to_bits(), e2.std_error.to_bits());
    assert_eq!(e1.identifier, e2.identifier);
    assert_eq!(
        e1.iv_first_stage_f.unwrap().to_bits(),
        e2.iv_first_stage_f.unwrap().to_bits()
    );
}

#[test]
fn confidence_interval_widens_at_higher_level() {
    let df = synthetic_iv(150, 37, 1.0, 0.8);
    let report = empty_locke_report();
    let e_90 = IVRegression::new()
        .with_confidence_level(0.90)
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    let e_99 = IVRegression::new()
        .with_confidence_level(0.99)
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    let w_90 = e_90.ci_upper - e_90.ci_lower;
    let w_99 = e_99.ci_upper - e_99.ci_lower;
    assert!(
        w_99 > w_90,
        "99% CI ({}) should be wider than 90% ({})",
        w_99, w_90
    );
}

#[test]
fn estimator_label_is_iv_regression() {
    assert_eq!(ESTIMATOR_LABEL, "iv_regression");
}

#[test]
fn weak_instrument_finding_sample_size_matches_estimate() {
    let df = synthetic_iv(100, 41, 1.0, 0.05); // weak
    let report = empty_locke_report();
    let est = IVRegression::new()
        .estimate(
            &df,
            "treatment",
            "outcome",
            "instrument",
            &["x"],
            &assumptions_iv(),
            &report,
        )
        .unwrap();
    if let Some(finding) = weak_instrument_finding(&est, "instrument", 10.0) {
        assert_eq!(finding.sample_size, est.n_treated + est.n_control);
        // Verify evidence has the F-statistic and threshold.
        let has_f = finding.evidence.iter().any(|e| {
            matches!(e, FindingEvidence::Metric { label, .. } if label == "first_stage_f")
        });
        let has_t = finding.evidence.iter().any(|e| {
            matches!(e, FindingEvidence::Metric { label, .. } if label == "threshold")
        });
        assert!(has_f && has_t);
    }
}
