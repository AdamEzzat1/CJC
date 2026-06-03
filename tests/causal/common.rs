//! Shared test fixtures for cjc-causal integration tests.
//!
//! These helpers exist so each test file doesn't have to repeat the
//! `DataFrame::from_columns(vec![...])` boilerplate or hand-build a Locke
//! report.

#![allow(dead_code)] // Some helpers are used only by subset of test files.

use cjc_data::{Column, DataFrame};
use cjc_locke::report::{LockeInputSummary, LockeReport};
use cjc_repro::Rng;
use std::collections::BTreeMap;

/// An empty `LockeReport` — passes the refusal check for any column set.
pub fn empty_locke_report() -> LockeReport {
    LockeReport::new(
        LockeInputSummary::default(),
        vec![],
        BTreeMap::new(),
        vec![],
    )
}

/// Build a `DataFrame` from named `Vec<f64>` columns.
///
/// Panics on column-length mismatch (test-time fixture, not user-facing).
pub fn df_from_floats(named: &[(&str, Vec<f64>)]) -> DataFrame {
    let cols: Vec<(String, Column)> = named
        .iter()
        .map(|(n, v)| (n.to_string(), Column::Float(v.clone())))
        .collect();
    DataFrame::from_columns(cols).expect("test DataFrame column lengths must align")
}

/// Synthetic data generator: confounded observational study.
///
/// Treatment depends on `age` + `income` (positive coefficients) so the
/// naive treated-minus-control mean overstates the true ATE. Propensity-
/// score matching should close most of the gap.
///
/// Outcome model: `y = α·age + β·income + τ·T + ε`. The true ATE is `τ`.
pub fn synthetic_confounded(n: usize, seed: u64, true_ate: f64) -> DataFrame {
    let mut rng = Rng::seeded(seed);
    let mut age = Vec::with_capacity(n);
    let mut income = Vec::with_capacity(n);
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);

    for _ in 0..n {
        let a = (rng.next_f64() - 0.5) * 4.0;
        let i = (rng.next_f64() - 0.5) * 4.0;
        let logit = 0.5 * a + 0.3 * i;
        let p = 1.0 / (1.0 + (-logit).exp());
        let t = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = (rng.next_f64() - 0.5) * 0.5;
        let y = 1.0 * a + 0.5 * i + (if t == 1.0 { true_ate } else { 0.0 }) + noise;
        age.push(a);
        income.push(i);
        treatment.push(t);
        outcome.push(y);
    }
    df_from_floats(&[
        ("age", age),
        ("income", income),
        ("treatment", treatment),
        ("outcome", outcome),
    ])
}

/// Synthetic data generator: truly randomised assignment.
///
/// Treatment is a coin flip independent of covariates, so the naive
/// difference-in-means is already an unbiased ATE estimator. Matching
/// should produce an estimate close to `true_ate` and a CI that brackets
/// it for sufficiently large `n`.
pub fn synthetic_randomised(n: usize, seed: u64, true_ate: f64) -> DataFrame {
    let mut rng = Rng::seeded(seed);
    let mut age = Vec::with_capacity(n);
    let mut income = Vec::with_capacity(n);
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);

    for _ in 0..n {
        let a = (rng.next_f64() - 0.5) * 4.0;
        let i = (rng.next_f64() - 0.5) * 4.0;
        let t = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
        let noise = (rng.next_f64() - 0.5) * 0.5;
        let y = 1.0 * a + 0.5 * i + (if t == 1.0 { true_ate } else { 0.0 }) + noise;
        age.push(a);
        income.push(i);
        treatment.push(t);
        outcome.push(y);
    }
    df_from_floats(&[
        ("age", age),
        ("income", income),
        ("treatment", treatment),
        ("outcome", outcome),
    ])
}
