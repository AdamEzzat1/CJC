//! Shared helpers for the physics_ml benchmark suite.
//!
//! Determinism hashing: we compare `Vec<f64>` outputs *bitwise* by converting
//! each f64 to its IEEE-754 bit pattern (`f64::to_bits`) and folding into a
//! 64-bit accumulator using the SplitMix64 mix function (the same one used
//! by `cjc_repro::Rng`). This guarantees that any single-bit difference in
//! any element produces a different hash — no floating-point comparison
//! fuzziness.

/// SplitMix64 mix step — matches `cjc_repro::Rng` so determinism hashes
/// align with the rest of the codebase's RNG semantics.
#[inline]
fn splitmix64_mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Bitwise-deterministic hash over a slice of `f64` (treats each value as
/// its raw IEEE-754 bit pattern). Sensitive to ULP-level differences.
pub fn bit_hash_f64(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325; // FNV offset basis as seed
    for &x in xs {
        h ^= splitmix64_mix(x.to_bits());
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

/// Assert that every element is finite (no NaN, no ±∞).
pub fn assert_all_finite(xs: &[f64], label: &str) {
    for (i, &x) in xs.iter().enumerate() {
        assert!(
            x.is_finite(),
            "{label}[{i}] is not finite: {x}",
        );
    }
}

/// Default accuracy thresholds for the heat 1D benchmark at the
/// "smoke" (fast) budget. The long-converge `#[ignore]`d test
/// uses tighter thresholds defined inline.
pub mod heat_1d_thresholds {
    /// L2 relative error against analytical solution at the smoke
    /// budget (500 epochs, lr=1e-3, 2-20-20-1 Tanh net). Calibrated
    /// to ~3× headroom over the observed value (~0.07) so the gate
    /// catches real regressions without flaking on minor variance.
    pub const L2_SMOKE: f64 = 0.20;
    /// Max absolute error against analytical solution. Observed ~0.13
    /// at the smoke budget; 3× headroom for stability.
    pub const MAX_ABS_SMOKE: f64 = 0.40;
}

/// Wave 1D thresholds. Trainer evaluates against the analytical
/// `sin(πx)·cos(cπt)` at the mid-time slice t=0.5 (50 spatial points).
/// Defaults: 500 epochs, lr=1e-3, c=1.0, 2-20-20-20-1 Tanh net.
/// Calibrated post-run: observed L2≈0.236, max≈0.362 at the smoke budget;
/// thresholds carry ~2× headroom (wave converges slower than heat under
/// the same epoch budget, so headroom is intentionally smaller than
/// heat 1D's 3× — tightening further requires more epochs).
pub mod wave_1d_thresholds {
    pub const L2_SMOKE: f64 = 0.50;
    pub const MAX_ABS_SMOKE: f64 = 1.00;
}

/// Burgers 1D thresholds. Trainer reports IC reproduction error at t=0
/// against `-sin(πx)` (no closed-form analytical solution exists).
/// `mean_residual` is the most meaningful global convergence signal here.
/// Defaults: 500 epochs, lr=1e-3, ν=0.01/π, 2-20-20-20-1 Tanh net.
/// Calibrated post-run: observed IC L2≈0.095, IC max≈0.173, residual≈0.252
/// at the smoke budget; thresholds carry ~3× headroom.
pub mod burgers_1d_thresholds {
    /// IC reproduction L2 at t=0.
    pub const IC_L2_SMOKE: f64 = 0.30;
    /// IC reproduction max abs error at t=0.
    pub const IC_MAX_ABS_SMOKE: f64 = 0.55;
    /// Mean physics residual at final params.
    pub const RESIDUAL_SMOKE: f64 = 0.80;
}

/// Allen-Cahn 1D thresholds. No closed-form solution exists. The harness
/// gates on (a) IC reproduction at t=0 against `x²·cos(π·x)` evaluated
/// externally via `cjc_ad::pinn::pinn_mlp_eval_grid`, and (b) `mean_residual`
/// of the PDE residual at final params. Defaults: 500 epochs, lr=1e-3,
/// ε=0.01, 2-20-20-20-1 Tanh net. Calibrated post-run: observed IC L2≈0.129,
/// IC max≈0.280, residual≈0.0265 at the smoke budget; ~3× headroom.
pub mod allen_cahn_1d_thresholds {
    /// IC reproduction RMSE at t=0 over 50 uniform points in x∈[-1,1].
    pub const IC_L2_SMOKE: f64 = 0.40;
    /// IC reproduction max abs error at t=0.
    pub const IC_MAX_ABS_SMOKE: f64 = 0.85;
    /// Mean physics residual at final params.
    pub const RESIDUAL_SMOKE: f64 = 0.10;
}

/// KdV 1D thresholds. The trainer's IC `0.5·sech²(x/2)` matches the
/// analytical 1-soliton `0.5·sech²(0.5·(x-t))` with `c=1`, so the harness
/// gates on full-domain RMSE/max over a 50×11 grid in (x,t)∈[-5,5]×[0,1].
/// Defaults: 500 epochs, lr=1e-3, 2-20-20-20-1 Tanh net. Calibrated post-run:
/// observed L2≈0.038, max≈0.118, residual≈3.1e-3 at the smoke budget; ~3-6×
/// headroom (residual headroom is wider since KdV residual is unusually low).
pub mod kdv_1d_thresholds {
    /// Full-domain RMSE vs analytical 1-soliton over the 50×11 grid.
    pub const L2_SMOKE: f64 = 0.15;
    /// Full-domain max abs error vs analytical 1-soliton.
    pub const MAX_ABS_SMOKE: f64 = 0.40;
    /// Mean physics residual at final params.
    pub const RESIDUAL_SMOKE: f64 = 0.02;
}
