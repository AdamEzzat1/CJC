//! Bolero fuzz targets for IVRegression.
//!
//! Invariant: IV's `estimate()` never panics on arbitrary user input. Every
//! degenerate case (NaN instrument, zero-variance, threshold = NaN, etc.)
//! surfaces as a structured `CausalError` variant.

use super::common::empty_locke_report;
use cjc_causal::{IVRegression, IdentificationAssumption};
use cjc_data::{Column, DataFrame};

fn assumptions_iv() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::ExcludabilityOfInstrument,
        IdentificationAssumption::NoInterference,
    ]
}

#[test]
fn fuzz_arbitrary_instrument_values_never_panic() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|z: &Vec<f64>| {
            let n = z.len();
            if n < 6 {
                return;
            }
            // Fixed treatment, outcome, and one covariate.
            let treatment: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let outcome: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
            let x: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
            let df = DataFrame::from_columns(vec![
                ("treatment".into(), Column::Float(treatment)),
                ("outcome".into(), Column::Float(outcome)),
                ("instrument".into(), Column::Float(z.clone())),
                ("x".into(), Column::Float(x)),
            ])
            .unwrap();
            let report = empty_locke_report();
            let _ = IVRegression::new().estimate(
                &df, "treatment", "outcome", "instrument", &["x"],
                &assumptions_iv(), &report,
            );
        });
}

#[test]
fn fuzz_arbitrary_confidence_level_never_panics() {
    let n = 60;
    let treatment: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
    let instrument: Vec<f64> = (0..n).map(|i| 0.3 * i as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float(treatment)),
        ("outcome".into(), Column::Float(outcome)),
        ("instrument".into(), Column::Float(instrument)),
        ("x".into(), Column::Float(x)),
    ])
    .unwrap();
    let report = empty_locke_report();
    let assumptions = assumptions_iv();

    bolero::check!()
        .with_type::<f64>()
        .for_each(|cl: &f64| {
            let _ = IVRegression::new()
                .with_confidence_level(*cl)
                .estimate(
                    &df, "treatment", "outcome", "instrument", &["x"],
                    &assumptions, &report,
                );
        });
}

#[test]
fn fuzz_arbitrary_weak_instrument_threshold_never_panics() {
    let n = 60;
    let treatment: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
    let instrument: Vec<f64> = (0..n).map(|i| 0.3 * i as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float(treatment)),
        ("outcome".into(), Column::Float(outcome)),
        ("instrument".into(), Column::Float(instrument)),
        ("x".into(), Column::Float(x)),
    ])
    .unwrap();
    let report = empty_locke_report();
    let assumptions = assumptions_iv();

    bolero::check!()
        .with_type::<f64>()
        .for_each(|thr: &f64| {
            let _ = IVRegression::new()
                .with_weak_instrument_threshold(*thr)
                .estimate(
                    &df, "treatment", "outcome", "instrument", &["x"],
                    &assumptions, &report,
                );
        });
}
