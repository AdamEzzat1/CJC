//! Integration tests for [`MetropolisHastings::run`].

use cjc_tempest::{MetropolisHastings, TempestError};

fn log_p_gaussian_1d(x: &[f64]) -> f64 {
    -0.5 * x[0] * x[0]
}

fn log_p_gaussian_2d(x: &[f64]) -> f64 {
    -0.5 * (x[0] * x[0] + x[1] * x[1])
}

/// Banana density (Rosenbrock-shaped). Common MCMC test target.
fn log_p_banana_2d(x: &[f64]) -> f64 {
    // -log p ∝ (1 - x[0])² + 100·(x[1] - x[0]²)²
    let a = (1.0 - x[0]).powi(2);
    let b = 100.0 * (x[1] - x[0] * x[0]).powi(2);
    -(a + b)
}

#[test]
fn end_to_end_recovers_unit_gaussian_mean() {
    let mh = MetropolisHastings::new();
    let r = mh
        .run(log_p_gaussian_1d, &[0.0], 1, 200, 2000, 42)
        .expect("MH on unit Gaussian should succeed");
    let mean: f64 = r.chains[0].iter().map(|s| s[0]).sum::<f64>() / 2000.0;
    assert!(mean.abs() < 0.4, "sample mean = {}", mean);
}

#[test]
fn end_to_end_on_banana_runs_to_completion() {
    let mh = MetropolisHastings::new().with_init_sigma(0.3);
    let r = mh
        .run(log_p_banana_2d, &[0.0, 0.0], 1, 500, 1000, 7)
        .expect("MH on banana should succeed");
    assert_eq!(r.chains[0].len(), 1000);
    // Samples should remain finite (banana density doesn't blow up).
    for s in &r.chains[0] {
        for &v in s {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn multi_chain_run_produces_n_independent_chains() {
    let mh = MetropolisHastings::new();
    let r = mh
        .run(log_p_gaussian_2d, &[0.0, 0.0], 4, 100, 200, 99)
        .unwrap();
    assert_eq!(r.chains.len(), 4);
    assert_eq!(r.n_chains, 4);
    // Chains should differ from each other (per-chain seed stretch worked).
    for c1 in 0..4 {
        for c2 in (c1 + 1)..4 {
            // At least one sample bit pattern differs.
            let any_diff = (0..r.chains[c1].len()).any(|s| {
                (0..2).any(|d| r.chains[c1][s][d].to_bits() != r.chains[c2][s][d].to_bits())
            });
            assert!(any_diff, "chains {} and {} are byte-identical (per-chain stretch failed)", c1, c2);
        }
    }
}

#[test]
fn init_sigma_invalid_returns_unsupported() {
    let mh = MetropolisHastings::new().with_init_sigma(-1.0);
    let err = mh
        .run(log_p_gaussian_1d, &[0.0], 1, 100, 100, 0)
        .unwrap_err();
    assert!(matches!(err, TempestError::Unsupported { .. }));
}

#[test]
fn n_dim_in_posterior_matches_initial_state_length() {
    let mh = MetropolisHastings::new();
    let r = mh
        .run(|x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>(), &[0.0, 0.0, 0.0], 1, 50, 100, 17)
        .unwrap();
    assert_eq!(r.n_dim, 3);
    for s in &r.chains[0] {
        assert_eq!(s.len(), 3);
    }
}
