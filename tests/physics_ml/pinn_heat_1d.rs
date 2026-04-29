//! 1D Heat Equation PINN — accuracy, determinism, and sanity tests.
//!
//! Equation:    u_t = α · u_xx                on [0,1] × [0,1]
//! IC:          u(x, 0) = sin(π x)
//! BCs:         u(0, t) = u(1, t) = 0
//! Analytical:  u(x, t) = exp(-α π² t) · sin(π x)
//!
//! The trainer (`cjc_ad::pinn::pinn_heat_1d_nn_train`) computes
//! `l2_error` and `max_error` against this analytical solution
//! internally, so the tests below assert thresholds on those metrics
//! rather than re-deriving them.

use cjc_ad::pinn::{HeatConfig, pinn_heat_1d_nn_train};

use super::common::{assert_all_finite, bit_hash_f64, heat_1d_thresholds};

/// Smoke configuration — small enough to run in seconds, large enough
/// for the trainer to demonstrate downward loss trend and finite metrics.
fn smoke_config(seed: u64) -> HeatConfig {
    HeatConfig {
        epochs: 500,
        n_collocation: 64,
        n_ic: 50,
        n_bc: 25,
        seed,
        ..HeatConfig::default()
    }
}

/// Tiny configuration for fast determinism tests (50 epochs is enough
/// to amplify any source of nondeterminism into a divergent param vector).
fn determinism_config(seed: u64) -> HeatConfig {
    HeatConfig {
        epochs: 50,
        n_collocation: 32,
        n_ic: 20,
        n_bc: 10,
        seed,
        ..HeatConfig::default()
    }
}

#[test]
fn heat_1d_metrics_finite() {
    let result = pinn_heat_1d_nn_train(&smoke_config(42));
    assert!(result.l2_error.is_some(), "trainer must report l2_error");
    assert!(result.max_error.is_some(), "trainer must report max_error");
    let l2 = result.l2_error.unwrap();
    let max_e = result.max_error.unwrap();
    assert!(l2.is_finite(), "L2 error not finite: {l2}");
    assert!(max_e.is_finite(), "max error not finite: {max_e}");
    assert!(result.mean_residual.is_finite(),
        "mean_residual not finite: {}", result.mean_residual);
    assert_all_finite(&result.final_params, "final_params");
}

#[test]
fn heat_1d_l2_below_smoke_threshold() {
    let result = pinn_heat_1d_nn_train(&smoke_config(42));
    let l2 = result.l2_error.unwrap();
    assert!(
        l2 < heat_1d_thresholds::L2_SMOKE,
        "L2 error {l2} exceeds smoke threshold {}",
        heat_1d_thresholds::L2_SMOKE,
    );
}

#[test]
fn heat_1d_max_error_below_smoke_threshold() {
    let result = pinn_heat_1d_nn_train(&smoke_config(42));
    let max_e = result.max_error.unwrap();
    assert!(
        max_e < heat_1d_thresholds::MAX_ABS_SMOKE,
        "max abs error {max_e} exceeds smoke threshold {}",
        heat_1d_thresholds::MAX_ABS_SMOKE,
    );
}

#[test]
fn heat_1d_loss_history_trend_downward() {
    let result = pinn_heat_1d_nn_train(&smoke_config(42));
    let n = result.history.len();
    assert!(n >= 100, "expected ≥100 epochs, got {n}");
    // Compare last 10% mean to first 10% mean. Allows transient spikes
    // from the boundary-weight ramp to register early without breaking.
    let bin = (n / 10).max(1);
    let head_mean: f64 = result.history[..bin]
        .iter().map(|h| h.total_loss).sum::<f64>() / bin as f64;
    let tail_mean: f64 = result.history[n - bin..]
        .iter().map(|h| h.total_loss).sum::<f64>() / bin as f64;
    assert!(
        tail_mean < head_mean,
        "loss did not decrease: head_mean={head_mean}, tail_mean={tail_mean}",
    );
}

#[test]
fn heat_1d_double_run_bit_identical() {
    // Determinism gate: two runs with identical seed must produce
    // bit-identical final_params. Any HashMap iteration leak, parallel
    // reduction instability, or unseeded RNG would break this.
    let cfg = determinism_config(123);
    let r1 = pinn_heat_1d_nn_train(&cfg);
    let r2 = pinn_heat_1d_nn_train(&cfg);

    assert_eq!(
        r1.final_params.len(),
        r2.final_params.len(),
        "param count differs between runs",
    );

    let h1 = bit_hash_f64(&r1.final_params);
    let h2 = bit_hash_f64(&r2.final_params);
    assert_eq!(
        h1, h2,
        "bit-hash of final_params diverged: {h1:#x} vs {h2:#x}",
    );

    // Also verify reported metrics match exactly.
    assert_eq!(r1.l2_error.unwrap().to_bits(), r2.l2_error.unwrap().to_bits(),
        "l2_error not bit-identical");
    assert_eq!(r1.max_error.unwrap().to_bits(), r2.max_error.unwrap().to_bits(),
        "max_error not bit-identical");
    assert_eq!(r1.mean_residual.to_bits(), r2.mean_residual.to_bits(),
        "mean_residual not bit-identical");
}

#[test]
fn heat_1d_different_seeds_diverge() {
    // Sanity check: different seeds must produce different final params
    // (otherwise the seed isn't actually being threaded into init/sampling).
    let r1 = pinn_heat_1d_nn_train(&determinism_config(1));
    let r2 = pinn_heat_1d_nn_train(&determinism_config(2));
    let h1 = bit_hash_f64(&r1.final_params);
    let h2 = bit_hash_f64(&r2.final_params);
    assert_ne!(h1, h2, "different seeds produced identical params (RNG leak?)");
}

#[test]
fn heat_1d_no_nan_inf_in_history() {
    let result = pinn_heat_1d_nn_train(&smoke_config(42));
    for (i, h) in result.history.iter().enumerate() {
        assert!(h.total_loss.is_finite(), "epoch {i}: total_loss = {}", h.total_loss);
        assert!(h.physics_loss.is_finite(), "epoch {i}: physics_loss = {}", h.physics_loss);
        assert!(h.data_loss.is_finite(), "epoch {i}: data_loss = {}", h.data_loss);
        assert!(h.boundary_loss.is_finite(), "epoch {i}: boundary_loss = {}", h.boundary_loss);
        assert!(h.grad_norm.is_finite(), "epoch {i}: grad_norm = {}", h.grad_norm);
    }
}

/// Long-converge accuracy test. Run with `cargo test -- --ignored` to
/// validate that the post-hardening physics stack hits realistic PINN
/// accuracy thresholds at a serious training budget.
#[test]
#[ignore = "long-running convergence test; run with --ignored"]
fn heat_1d_long_converge_tight_thresholds() {
    let cfg = HeatConfig {
        epochs: 5_000,
        n_collocation: 128,
        n_ic: 100,
        n_bc: 50,
        seed: 42,
        ..HeatConfig::default()
    };
    let result = pinn_heat_1d_nn_train(&cfg);
    let l2 = result.l2_error.unwrap();
    let max_e = result.max_error.unwrap();
    // Tight thresholds — these are the real "post-hardening credible" gates.
    assert!(l2 < 0.10, "long-converge L2 {l2} above tight threshold 0.10");
    assert!(max_e < 0.20, "long-converge max {max_e} above tight threshold 0.20");
}
