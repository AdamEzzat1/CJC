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

/// Synthetic data generator for the partially linear DML model:
/// `y = β·T + g(X) + ε`, `T = m(X) + V`.
///
/// Two non-collinear covariates `x1`, `x2`. Both `T` and `Y` depend on `X`,
/// so naive OLS of `Y ~ T` would be biased; DML's cross-fit residualisation
/// recovers `β` asymptotically.
pub fn synthetic_dml(n: usize, seed: u64, true_beta: f64) -> DataFrame {
    let mut rng = Rng::seeded(seed);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for _ in 0..n {
        let a = (rng.next_f64() - 0.5) * 4.0;
        // Non-collinear second covariate: a² minus a linear shift so it has
        // its own variance independent of x1.
        let b = (rng.next_f64() - 0.5) * 4.0;
        // m(X) = 0.5·x1 + 0.3·x2 (treatment depends on covariates)
        let t_noise = (rng.next_f64() - 0.5) * 0.5;
        let t_i = 0.5 * a + 0.3 * b + t_noise;
        // y = β·T + g(X) + ε with g(X) = 0.8·x1 + 0.4·x2
        let y_noise = (rng.next_f64() - 0.5) * 0.5;
        let y_i = true_beta * t_i + 0.8 * a + 0.4 * b + y_noise;
        x1.push(a);
        x2.push(b);
        t.push(t_i);
        y.push(y_i);
    }
    df_from_floats(&[
        ("treatment", t),
        ("outcome", y),
        ("x1", x1),
        ("x2", x2),
    ])
}

/// Synthetic data generator for IV regression: instrument-driven treatment
/// + unobserved confounder.
///
/// Model:
/// - `Z` (instrument) — exogenous, independent of confounder.
/// - `x` (covariate) — exogenous, included in both stages.
/// - `U` (confounder) — unobserved by the analyst; correlated with both T and Y.
/// - `T = instrument_strength·Z + 0.3·x + 0.7·U + noise_t` (endogenous).
/// - `y = true_beta·T + 0.5·x + 0.6·U + noise_y` (structural equation).
///
/// IV correctly recovers `true_beta` (asymptotically); naive OLS of y on T
/// is biased upward because `cov(T, U) ≠ 0`.
pub fn synthetic_iv(n: usize, seed: u64, true_beta: f64, instrument_strength: f64) -> DataFrame {
    let mut rng = Rng::seeded(seed);
    let mut z = Vec::with_capacity(n);
    let mut x_cov = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for _ in 0..n {
        let z_i = (rng.next_f64() - 0.5) * 4.0;
        let x_i = (rng.next_f64() - 0.5) * 4.0;
        let u = (rng.next_f64() - 0.5) * 2.0;
        let t_noise = (rng.next_f64() - 0.5) * 0.5;
        let t_i = instrument_strength * z_i + 0.3 * x_i + 0.7 * u + t_noise;
        let y_noise = (rng.next_f64() - 0.5) * 0.5;
        let y_i = true_beta * t_i + 0.5 * x_i + 0.6 * u + y_noise;
        z.push(z_i);
        x_cov.push(x_i);
        t.push(t_i);
        y.push(y_i);
    }
    df_from_floats(&[
        ("treatment", t),
        ("outcome", y),
        ("instrument", z),
        ("x", x_cov),
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
