//! Bolero fuzz targets for DoubleMLEstimator.
//!
//! Invariant: DML's `estimate()` never panics on arbitrary input. Every
//! degenerate case surfaces as a structured `CausalError`.

use super::common::empty_locke_report;
use cjc_causal::{DoubleMLEstimator, IdentificationAssumption};
use cjc_data::{Column, DataFrame};

fn assumptions_dml() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::NoInterference,
    ]
}

#[test]
fn fuzz_arbitrary_covariate_values_never_panic() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|cov: &Vec<f64>| {
            let n = cov.len();
            if n < 20 {
                return;
            }
            let treatment: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.01).collect();
            let outcome: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.05).collect();
            let x2: Vec<f64> = (0..n).map(|i| 0.5 * (i as f64).powi(2)).collect();
            let df = DataFrame::from_columns(vec![
                ("treatment".into(), Column::Float(treatment)),
                ("outcome".into(), Column::Float(outcome)),
                ("x1".into(), Column::Float(cov.clone())),
                ("x2".into(), Column::Float(x2)),
            ])
            .unwrap();
            let report = empty_locke_report();
            let _ = DoubleMLEstimator::new().estimate(
                &df, "treatment", "outcome", &["x1", "x2"],
                &assumptions_dml(), &report,
            );
        });
}

#[test]
fn fuzz_arbitrary_k_folds_never_panic() {
    let n = 100;
    let treatment: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.01).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.05).collect();
    let x1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64).powi(2) * 0.01).collect();
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float(treatment)),
        ("outcome".into(), Column::Float(outcome)),
        ("x1".into(), Column::Float(x1)),
        ("x2".into(), Column::Float(x2)),
    ])
    .unwrap();
    let report = empty_locke_report();
    let assumptions = assumptions_dml();

    bolero::check!()
        .with_type::<u8>() // small k range
        .for_each(|k: &u8| {
            let _ = DoubleMLEstimator::new()
                .with_k_folds(*k as usize)
                .estimate(
                    &df, "treatment", "outcome", &["x1", "x2"],
                    &assumptions, &report,
                );
        });
}

#[test]
fn fuzz_arbitrary_confidence_level_never_panics() {
    let n = 100;
    let treatment: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.01).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.05).collect();
    let x1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64).powi(2) * 0.01).collect();
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float(treatment)),
        ("outcome".into(), Column::Float(outcome)),
        ("x1".into(), Column::Float(x1)),
        ("x2".into(), Column::Float(x2)),
    ])
    .unwrap();
    let report = empty_locke_report();
    let assumptions = assumptions_dml();

    bolero::check!()
        .with_type::<f64>()
        .for_each(|cl: &f64| {
            let _ = DoubleMLEstimator::new()
                .with_confidence_level(*cl)
                .estimate(
                    &df, "treatment", "outcome", &["x1", "x2"],
                    &assumptions, &report,
                );
        });
}
