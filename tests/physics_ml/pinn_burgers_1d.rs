//! 1D Viscous Burgers' Equation PINN — convergence, determinism, sanity.
//!
//! Equation:    u_t + u·u_x = ν · u_xx          on [-1,1] × [0,1]
//! IC:          u(x, 0) = -sin(π x)
//! BCs:         u(-1, t) = u(1, t) = 0
//! Reference:   No closed-form analytical solution. Trainer reports
//!              `l2_error` / `max_error` as IC reproduction at t=0
//!              against `-sin(π x)` (NOT a full space-time L2).
//!
//! The most meaningful global convergence signal for Burgers is
//! `mean_residual` (RMS of u_t + u·u_x − ν·u_xx at final params).
//! Full-domain accuracy gating against a Cole-Hopf reference is
//! deferred to Phase 3.

use cjc_ad::pinn::{BurgersConfig, pinn_burgers_train};

use super::common::{assert_all_finite, bit_hash_f64, burgers_1d_thresholds};

fn smoke_config(seed: u64) -> BurgersConfig {
    BurgersConfig {
        epochs: 500,
        n_collocation: 64,
        n_ic: 50,
        n_bc: 25,
        seed,
        ..BurgersConfig::default()
    }
}

fn determinism_config(seed: u64) -> BurgersConfig {
    BurgersConfig {
        epochs: 50,
        n_collocation: 32,
        n_ic: 20,
        n_bc: 10,
        seed,
        ..BurgersConfig::default()
    }
}

#[test]
fn burgers_1d_metrics_finite() {
    let result = pinn_burgers_train(&smoke_config(42));
    assert!(result.l2_error.is_some());
    assert!(result.max_error.is_some());
    let l2 = result.l2_error.unwrap();
    let max_e = result.max_error.unwrap();
    assert!(l2.is_finite(), "burgers IC L2 not finite: {l2}");
    assert!(max_e.is_finite(), "burgers IC max not finite: {max_e}");
    assert!(result.mean_residual.is_finite(),
        "burgers mean_residual not finite: {}", result.mean_residual);
    assert_all_finite(&result.final_params, "burgers final_params");
}

#[test]
fn burgers_1d_ic_l2_below_smoke_threshold() {
    let result = pinn_burgers_train(&smoke_config(42));
    let l2 = result.l2_error.unwrap();
    assert!(
        l2 < burgers_1d_thresholds::IC_L2_SMOKE,
        "burgers IC L2 {l2} exceeds smoke threshold {}",
        burgers_1d_thresholds::IC_L2_SMOKE,
    );
}

#[test]
fn burgers_1d_ic_max_below_smoke_threshold() {
    let result = pinn_burgers_train(&smoke_config(42));
    let max_e = result.max_error.unwrap();
    assert!(
        max_e < burgers_1d_thresholds::IC_MAX_ABS_SMOKE,
        "burgers IC max {max_e} exceeds smoke threshold {}",
        burgers_1d_thresholds::IC_MAX_ABS_SMOKE,
    );
}

#[test]
fn burgers_1d_residual_below_smoke_threshold() {
    let result = pinn_burgers_train(&smoke_config(42));
    assert!(
        result.mean_residual < burgers_1d_thresholds::RESIDUAL_SMOKE,
        "burgers residual {} exceeds smoke threshold {}",
        result.mean_residual, burgers_1d_thresholds::RESIDUAL_SMOKE,
    );
}

// NOTE: see pinn_wave_1d.rs — same reasoning. Burgers is additionally
// stiff (shock formation by t≈1/π causes physics_loss oscillation), so
// a trend gate is even less reliable here than for wave. Accuracy
// thresholds + NaN/Inf check + bit-hash determinism are sufficient.

#[test]
fn burgers_1d_double_run_bit_identical() {
    let cfg = determinism_config(123);
    let r1 = pinn_burgers_train(&cfg);
    let r2 = pinn_burgers_train(&cfg);
    assert_eq!(r1.final_params.len(), r2.final_params.len());
    assert_eq!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "burgers: bit-hash diverged across identical-seed runs",
    );
    assert_eq!(r1.l2_error.unwrap().to_bits(), r2.l2_error.unwrap().to_bits());
    assert_eq!(r1.max_error.unwrap().to_bits(), r2.max_error.unwrap().to_bits());
    assert_eq!(r1.mean_residual.to_bits(), r2.mean_residual.to_bits());
}

#[test]
fn burgers_1d_different_seeds_diverge() {
    let r1 = pinn_burgers_train(&determinism_config(1));
    let r2 = pinn_burgers_train(&determinism_config(2));
    assert_ne!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "burgers: different seeds produced identical params",
    );
}

#[test]
fn burgers_1d_no_nan_inf_in_history() {
    let result = pinn_burgers_train(&smoke_config(42));
    for (i, h) in result.history.iter().enumerate() {
        assert!(h.total_loss.is_finite(), "burgers epoch {i}: total_loss = {}", h.total_loss);
        assert!(h.physics_loss.is_finite(), "burgers epoch {i}: physics_loss = {}", h.physics_loss);
        assert!(h.data_loss.is_finite(), "burgers epoch {i}: data_loss = {}", h.data_loss);
        assert!(h.boundary_loss.is_finite(), "burgers epoch {i}: boundary_loss = {}", h.boundary_loss);
        assert!(h.grad_norm.is_finite(), "burgers epoch {i}: grad_norm = {}", h.grad_norm);
    }
}
