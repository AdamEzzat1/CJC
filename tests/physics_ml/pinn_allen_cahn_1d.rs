//! 1D Allen-Cahn Equation PINN — convergence, determinism, sanity.
//!
//! Equation:    u_t = ε²·u_xx + u - u³        on [-1,1] × [0,1]
//! IC:          u(x, 0) = x²·cos(π·x)
//! BCs:         periodic, u(-1, t) = u(1, t)
//! Reference:   No closed-form analytical solution. Tests gate on
//!              (a) IC reproduction at t=0 vs `x²·cos(π·x)` evaluated
//!              externally via `pinn_mlp_eval_grid`, and (b) the trainer's
//!              `mean_residual` (RMS of u_t − ε²·u_xx − u + u³).
//!
//! A high-resolution implicit-FD reference for full-domain L2 is deferred
//! to Phase 3b — Allen-Cahn at ε=0.01 has stiff diffusion plus a cubic
//! reaction term, so a lightweight explicit solver would not be a fair
//! reference at the budgets used here.

use cjc_ad::pinn::{
    AllenCahnConfig, allen_cahn_ic_grid, pinn_allen_cahn_train,
    pinn_l2_max_errors, pinn_mlp_eval_grid,
};

use super::common::{allen_cahn_1d_thresholds, assert_all_finite, bit_hash_f64};

fn smoke_config(seed: u64) -> AllenCahnConfig {
    AllenCahnConfig {
        epochs: 500,
        n_collocation: 64,
        n_ic: 50,
        n_bc: 25,
        seed,
        ..AllenCahnConfig::default()
    }
}

fn determinism_config(seed: u64) -> AllenCahnConfig {
    AllenCahnConfig {
        epochs: 50,
        n_collocation: 32,
        n_ic: 20,
        n_bc: 10,
        seed,
        ..AllenCahnConfig::default()
    }
}

fn ic_errors(cfg: &AllenCahnConfig, params: &[f64]) -> (f64, f64) {
    let (inputs, targets) = allen_cahn_ic_grid(50);
    let pred = pinn_mlp_eval_grid(&cfg.layer_sizes, params, &inputs);
    pinn_l2_max_errors(&pred, &targets)
}

#[test]
fn allen_cahn_1d_metrics_finite() {
    let cfg = smoke_config(42);
    let result = pinn_allen_cahn_train(&cfg);
    assert!(result.mean_residual.is_finite(),
        "allen-cahn mean_residual not finite: {}", result.mean_residual);
    assert_all_finite(&result.final_params, "allen-cahn final_params");
    let (l2, max_e) = ic_errors(&cfg, &result.final_params);
    assert!(l2.is_finite(), "allen-cahn IC L2 not finite: {l2}");
    assert!(max_e.is_finite(), "allen-cahn IC max not finite: {max_e}");
}

#[test]
fn allen_cahn_1d_ic_l2_below_smoke_threshold() {
    let cfg = smoke_config(42);
    let result = pinn_allen_cahn_train(&cfg);
    let (l2, _) = ic_errors(&cfg, &result.final_params);
    assert!(
        l2 < allen_cahn_1d_thresholds::IC_L2_SMOKE,
        "allen-cahn IC L2 {l2} exceeds smoke threshold {}",
        allen_cahn_1d_thresholds::IC_L2_SMOKE,
    );
}

#[test]
fn allen_cahn_1d_ic_max_below_smoke_threshold() {
    let cfg = smoke_config(42);
    let result = pinn_allen_cahn_train(&cfg);
    let (_, max_e) = ic_errors(&cfg, &result.final_params);
    assert!(
        max_e < allen_cahn_1d_thresholds::IC_MAX_ABS_SMOKE,
        "allen-cahn IC max {max_e} exceeds smoke threshold {}",
        allen_cahn_1d_thresholds::IC_MAX_ABS_SMOKE,
    );
}

#[test]
fn allen_cahn_1d_residual_below_smoke_threshold() {
    let result = pinn_allen_cahn_train(&smoke_config(42));
    assert!(
        result.mean_residual < allen_cahn_1d_thresholds::RESIDUAL_SMOKE,
        "allen-cahn residual {} exceeds smoke threshold {}",
        result.mean_residual, allen_cahn_1d_thresholds::RESIDUAL_SMOKE,
    );
}

// NOTE: like wave/burgers, no head-vs-tail trend gate. Allen-Cahn's
// boundary_weight ramp + stiff cubic reaction makes total_loss
// non-monotone early. Accuracy + residual + replay gates are sufficient.

#[test]
fn allen_cahn_1d_double_run_bit_identical() {
    let cfg = determinism_config(123);
    let r1 = pinn_allen_cahn_train(&cfg);
    let r2 = pinn_allen_cahn_train(&cfg);
    assert_eq!(r1.final_params.len(), r2.final_params.len());
    assert_eq!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "allen-cahn: bit-hash diverged across identical-seed runs",
    );
    assert_eq!(r1.mean_residual.to_bits(), r2.mean_residual.to_bits());
    let (l2_1, max_1) = ic_errors(&cfg, &r1.final_params);
    let (l2_2, max_2) = ic_errors(&cfg, &r2.final_params);
    assert_eq!(l2_1.to_bits(), l2_2.to_bits(), "allen-cahn external IC L2 not bit-identical");
    assert_eq!(max_1.to_bits(), max_2.to_bits(), "allen-cahn external IC max not bit-identical");
}

#[test]
fn allen_cahn_1d_different_seeds_diverge() {
    let r1 = pinn_allen_cahn_train(&determinism_config(1));
    let r2 = pinn_allen_cahn_train(&determinism_config(2));
    assert_ne!(
        bit_hash_f64(&r1.final_params),
        bit_hash_f64(&r2.final_params),
        "allen-cahn: different seeds produced identical params",
    );
}

#[test]
fn allen_cahn_1d_no_nan_inf_in_history() {
    let result = pinn_allen_cahn_train(&smoke_config(42));
    for (i, h) in result.history.iter().enumerate() {
        assert!(h.total_loss.is_finite(), "allen-cahn epoch {i}: total_loss = {}", h.total_loss);
        assert!(h.physics_loss.is_finite(), "allen-cahn epoch {i}: physics_loss = {}", h.physics_loss);
        assert!(h.data_loss.is_finite(), "allen-cahn epoch {i}: data_loss = {}", h.data_loss);
        assert!(h.boundary_loss.is_finite(), "allen-cahn epoch {i}: boundary_loss = {}", h.boundary_loss);
        assert!(h.grad_norm.is_finite(), "allen-cahn epoch {i}: grad_norm = {}", h.grad_norm);
    }
}
