//! 1D KdV Equation PINN — convergence, determinism, sanity.
//!
//! Equation:    u_t + 6·u·u_x + u_xxx = 0     on [-5,5] × [0,1]
//! IC:          u(x, 0) = 0.5·sech²(x/2)      (single soliton)
//! Reference:   Analytical 1-soliton `u(x,t) = 0.5·sech²(0.5·(x-t))` (c=1).
//!              The IC matches this expression at t=0 exactly, so the
//!              network is trained to learn a known traveling wave.
//!
//! Tests gate on full-domain RMSE/max over a 50×11 grid in (x,t) ∈
//! [-5,5] × [0,1] — evaluated externally via `pinn_mlp_eval_grid` against
//! `kdv_soliton_reference`. KdV stresses high-order dispersion (the
//! `u_xxx` term, computed via 4-point central FD on the network).

use cjc_ad::pinn::{
    KdvConfig, kdv_reference_grid, pinn_kdv_train,
    pinn_l2_max_errors, pinn_mlp_eval_grid,
};

use super::common::{assert_all_finite, bit_hash_f64, kdv_1d_thresholds};

fn smoke_config(seed: u64) -> KdvConfig {
    KdvConfig {
        epochs: 500,
        n_collocation: 64,
        n_ic: 50,
        n_bc: 25,
        seed,
        ..KdvConfig::default()
    }
}

fn determinism_config(seed: u64) -> KdvConfig {
    KdvConfig {
        epochs: 50,
        n_collocation: 32,
        n_ic: 20,
        n_bc: 10,
        seed,
        ..KdvConfig::default()
    }
}

fn full_domain_errors(cfg: &KdvConfig, params: &[f64]) -> (f64, f64) {
    // 50 spatial × 11 temporal points; c=1.
    let (inputs, targets) = kdv_reference_grid(-5.0, 5.0, 50, 0.0, 1.0, 11, 1.0);
    let pred = pinn_mlp_eval_grid(&cfg.layer_sizes, params, &inputs);
    pinn_l2_max_errors(&pred, &targets)
}

#[test]
fn kdv_1d_metrics_finite() {
    let cfg = smoke_config(42);
    let result = pinn_kdv_train(&cfg);
    assert!(result.mean_residual.is_finite(),
        "kdv mean_residual not finite: {}", result.mean_residual);
    assert_all_finite(&result.final_params, "kdv final_params");
    let (l2, max_e) = full_domain_errors(&cfg, &result.final_params);
    assert!(l2.is_finite(), "kdv full-domain L2 not finite: {l2}");
    assert!(max_e.is_finite(), "kdv full-domain max not finite: {max_e}");
}

#[test]
fn kdv_1d_l2_below_smoke_threshold() {
    let cfg = smoke_config(42);
    let result = pinn_kdv_train(&cfg);
    let (l2, _) = full_domain_errors(&cfg, &result.final_params);
    assert!(
        l2 < kdv_1d_thresholds::L2_SMOKE,
        "kdv full-domain L2 {l2} exceeds smoke threshold {}",
        kdv_1d_thresholds::L2_SMOKE,
    );
}

#[test]
fn kdv_1d_max_error_below_smoke_threshold() {
    let cfg = smoke_config(42);
    let result = pinn_kdv_train(&cfg);
    let (_, max_e) = full_domain_errors(&cfg, &result.final_params);
    assert!(
        max_e < kdv_1d_thresholds::MAX_ABS_SMOKE,
        "kdv full-domain max {max_e} exceeds smoke threshold {}",
        kdv_1d_thresholds::MAX_ABS_SMOKE,
    );
}

#[test]
fn kdv_1d_residual_below_smoke_threshold() {
    let result = pinn_kdv_train(&smoke_config(42));
    assert!(
        result.mean_residual < kdv_1d_thresholds::RESIDUAL_SMOKE,
        "kdv residual {} exceeds smoke threshold {}",
        result.mean_residual, kdv_1d_thresholds::RESIDUAL_SMOKE,
    );
}

// No head-vs-tail trend gate; same reasoning as wave/burgers.

#[test]
fn kdv_1d_double_run_bit_identical() {
    let cfg = determinism_config(123);
    let r1 = pinn_kdv_train(&cfg);
    let r2 = pinn_kdv_train(&cfg);
    assert_eq!(r1.final_params.len(), r2.final_params.len());
    assert_eq!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "kdv: bit-hash diverged across identical-seed runs",
    );
    assert_eq!(r1.mean_residual.to_bits(), r2.mean_residual.to_bits());
    let (l2_1, max_1) = full_domain_errors(&cfg, &r1.final_params);
    let (l2_2, max_2) = full_domain_errors(&cfg, &r2.final_params);
    assert_eq!(l2_1.to_bits(), l2_2.to_bits(), "kdv external L2 not bit-identical");
    assert_eq!(max_1.to_bits(), max_2.to_bits(), "kdv external max not bit-identical");
}

#[test]
fn kdv_1d_different_seeds_diverge() {
    let r1 = pinn_kdv_train(&determinism_config(1));
    let r2 = pinn_kdv_train(&determinism_config(2));
    assert_ne!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "kdv: different seeds produced identical params",
    );
}

#[test]
fn kdv_1d_no_nan_inf_in_history() {
    let result = pinn_kdv_train(&smoke_config(42));
    for (i, h) in result.history.iter().enumerate() {
        assert!(h.total_loss.is_finite(), "kdv epoch {i}: total_loss = {}", h.total_loss);
        assert!(h.physics_loss.is_finite(), "kdv epoch {i}: physics_loss = {}", h.physics_loss);
        assert!(h.data_loss.is_finite(), "kdv epoch {i}: data_loss = {}", h.data_loss);
        // boundary_loss is not used for KdV (only IC + physics).
        assert!(h.grad_norm.is_finite(), "kdv epoch {i}: grad_norm = {}", h.grad_norm);
    }
}
