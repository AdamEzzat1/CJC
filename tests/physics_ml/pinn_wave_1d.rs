//! 1D Wave Equation PINN — accuracy, determinism, and sanity tests.
//!
//! Equation:    u_tt = c² · u_xx                on [0,1] × [0,1]
//! IC:          u(x, 0) = sin(π x),  u_t(x, 0) = 0
//! BCs:         u(0, t) = u(1, t) = 0
//! Analytical:  u(x, t) = sin(π x) · cos(c π t)
//!
//! The trainer evaluates `l2_error` and `max_error` at the mid-time
//! slice t=0.5 (50 spatial points) against the analytical expression.
//! `mean_residual` is the RMS of the PDE residual at those same points.

use cjc_ad::pinn::{WaveConfig, pinn_wave_train};

use super::common::{assert_all_finite, bit_hash_f64, wave_1d_thresholds};

fn smoke_config(seed: u64) -> WaveConfig {
    WaveConfig {
        epochs: 500,
        n_collocation: 64,
        n_ic: 50,
        n_bc: 25,
        seed,
        ..WaveConfig::default()
    }
}

fn determinism_config(seed: u64) -> WaveConfig {
    WaveConfig {
        epochs: 50,
        n_collocation: 32,
        n_ic: 20,
        n_bc: 10,
        seed,
        ..WaveConfig::default()
    }
}

#[test]
fn wave_1d_metrics_finite() {
    let result = pinn_wave_train(&smoke_config(42));
    assert!(result.l2_error.is_some());
    assert!(result.max_error.is_some());
    let l2 = result.l2_error.unwrap();
    let max_e = result.max_error.unwrap();
    assert!(l2.is_finite(), "wave L2 not finite: {l2}");
    assert!(max_e.is_finite(), "wave max not finite: {max_e}");
    assert!(result.mean_residual.is_finite(),
        "wave mean_residual not finite: {}", result.mean_residual);
    assert_all_finite(&result.final_params, "wave final_params");
}

#[test]
fn wave_1d_l2_below_smoke_threshold() {
    let result = pinn_wave_train(&smoke_config(42));
    let l2 = result.l2_error.unwrap();
    assert!(
        l2 < wave_1d_thresholds::L2_SMOKE,
        "wave L2 {l2} exceeds smoke threshold {}",
        wave_1d_thresholds::L2_SMOKE,
    );
}

#[test]
fn wave_1d_max_error_below_smoke_threshold() {
    let result = pinn_wave_train(&smoke_config(42));
    let max_e = result.max_error.unwrap();
    assert!(
        max_e < wave_1d_thresholds::MAX_ABS_SMOKE,
        "wave max abs {max_e} exceeds smoke threshold {}",
        wave_1d_thresholds::MAX_ABS_SMOKE,
    );
}

// NOTE: A head-vs-tail or initial-vs-min total_loss gate does not work
// for wave (or burgers) because the boundary_weight=10 ramp pushes
// total_loss up before training drives it back down — initial values
// can be lower than late-training values even when the network is
// learning correctly. The accuracy thresholds + history NaN/Inf check
// + bit-identical replay collectively prove training worked, so a
// separate trend gate is redundant. Heat 1D keeps its trend gate
// because heat does converge monotonically under the same schedule.

#[test]
fn wave_1d_double_run_bit_identical() {
    let cfg = determinism_config(123);
    let r1 = pinn_wave_train(&cfg);
    let r2 = pinn_wave_train(&cfg);
    assert_eq!(r1.final_params.len(), r2.final_params.len());
    assert_eq!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "wave: bit-hash diverged across identical-seed runs",
    );
    assert_eq!(r1.l2_error.unwrap().to_bits(), r2.l2_error.unwrap().to_bits());
    assert_eq!(r1.max_error.unwrap().to_bits(), r2.max_error.unwrap().to_bits());
    assert_eq!(r1.mean_residual.to_bits(), r2.mean_residual.to_bits());
}

#[test]
fn wave_1d_different_seeds_diverge() {
    let r1 = pinn_wave_train(&determinism_config(1));
    let r2 = pinn_wave_train(&determinism_config(2));
    assert_ne!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "wave: different seeds produced identical params",
    );
}

#[test]
fn wave_1d_no_nan_inf_in_history() {
    let result = pinn_wave_train(&smoke_config(42));
    for (i, h) in result.history.iter().enumerate() {
        assert!(h.total_loss.is_finite(), "wave epoch {i}: total_loss = {}", h.total_loss);
        assert!(h.physics_loss.is_finite(), "wave epoch {i}: physics_loss = {}", h.physics_loss);
        assert!(h.data_loss.is_finite(), "wave epoch {i}: data_loss = {}", h.data_loss);
        assert!(h.boundary_loss.is_finite(), "wave epoch {i}: boundary_loss = {}", h.boundary_loss);
        assert!(h.grad_norm.is_finite(), "wave epoch {i}: grad_norm = {}", h.grad_norm);
    }
}

#[test]
#[ignore = "long-running convergence test; run with --ignored"]
fn wave_1d_long_converge_tight_thresholds() {
    let cfg = WaveConfig {
        epochs: 5_000,
        n_collocation: 128,
        n_ic: 100,
        n_bc: 50,
        seed: 42,
        ..WaveConfig::default()
    };
    let result = pinn_wave_train(&cfg);
    let l2 = result.l2_error.unwrap();
    let max_e = result.max_error.unwrap();
    assert!(l2 < 0.10, "wave long-converge L2 {l2} above tight threshold 0.10");
    assert!(max_e < 0.25, "wave long-converge max {max_e} above tight threshold 0.25");
}
