//! Bolero structural fuzz tests for cjc-causal.
//!
//! Bolero falls back to proptest on Windows/macOS and uses libfuzzer on
//! Linux CI. The invariant for all targets is the same: cjc-causal
//! estimators **never panic** on user-supplied data. Every failure mode
//! must surface as a structured `CausalError` variant.

use super::common::empty_locke_report;
use cjc_causal::{IdentificationAssumption, PropensityScoreMatcher};
use cjc_data::{Column, DataFrame};

fn default_assumptions() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::Positivity,
        IdentificationAssumption::NoInterference,
    ]
}

#[test]
fn fuzz_arbitrary_covariate_values_never_panic() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|cov: &Vec<f64>| {
            let n = cov.len();
            if n < 4 {
                return;
            }
            // Fixed treatment + outcome; the fuzzed input is just the covariate.
            // Half the rows are treated, half control, by parity of index.
            let treatment: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
            let outcome: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let df = DataFrame::from_columns(vec![
                ("treatment".into(), Column::Float(treatment)),
                ("outcome".into(), Column::Float(outcome)),
                ("x".into(), Column::Float(cov.clone())),
            ]).unwrap();
            let report = empty_locke_report();
            // Must return either Ok or a structured Err — never panic.
            let _ = PropensityScoreMatcher::new()
                .with_bootstrap_reps(5)
                .estimate(&df, "treatment", "outcome", &["x"], &default_assumptions(), &report);
        });
}

#[test]
fn fuzz_arbitrary_treatment_vectors_never_panic() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|treat: &Vec<f64>| {
            let n = treat.len();
            if n < 4 {
                return;
            }
            let outcome: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let cov: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
            let df = DataFrame::from_columns(vec![
                ("treatment".into(), Column::Float(treat.clone())),
                ("outcome".into(), Column::Float(outcome)),
                ("x".into(), Column::Float(cov)),
            ]).unwrap();
            let report = empty_locke_report();
            // Must return either Ok or a structured Err — never panic.
            // Most arbitrary treatment vectors will fail the binary-0/1 check
            // and return CausalError::WrongColumnType, which is correct.
            let _ = PropensityScoreMatcher::new()
                .with_bootstrap_reps(5)
                .estimate(&df, "treatment", "outcome", &["x"], &default_assumptions(), &report);
        });
}

#[test]
fn fuzz_arbitrary_caliper_widths_never_panic() {
    // Fixed dataset; only the caliper varies.
    let treatment: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let outcome: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let df = DataFrame::from_columns(vec![
        ("treatment".into(), Column::Float(treatment)),
        ("outcome".into(), Column::Float(outcome)),
        ("x".into(), Column::Float(x)),
    ]).unwrap();
    let report = empty_locke_report();
    let assumptions = default_assumptions();

    bolero::check!()
        .with_type::<f64>()
        .for_each(|caliper: &f64| {
            // Caliper of any value (negative, zero, infinite, NaN, very small,
            // very large) must not panic — invalid configurations surface as
            // CausalError::Unsupported.
            let _ = PropensityScoreMatcher::new()
                .with_caliper_sd(*caliper)
                .with_bootstrap_reps(5)
                .estimate(&df, "treatment", "outcome", &["x"], &assumptions, &report);
        });
}
