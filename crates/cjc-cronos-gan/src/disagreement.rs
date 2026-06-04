//! Phase 3 (minimal): [`TemporalDisagreement`] — the inspectable artifact
//! the Cronos GAN brief asks for.
//!
//! The disagreement is the SCALAR ARTIFACT not the LOSS. Where most GANs
//! train *toward* minimal disagreement (the discriminator can't tell
//! generated from real), Cronos GAN treats persistent calibrated
//! disagreement as the signal — a large gap between the SSM's smooth
//! prediction and the Liquid net's adaptive prediction is interpreted
//! as **the data has just transitioned between regimes**.
//!
//! This module is intentionally side-effect-free: it consumes two
//! prediction trajectories and returns a structured score. The Phase 3
//! `TemporalGan` struct chains rollout + disagreement together; this
//! module is the pure computation.
//!
//! ## Score definitions
//!
//! Given per-step output trajectories `ssm_out[t][d]` and `liq_out[t][d]`
//! and a target trajectory `target[t][d]`:
//!
//! - `ssm_score`        = mean over t of √(Σ_d (ssm_out − target)²) — per-
//!                        step RMSE of the SSM. Kahan-summed across t.
//! - `liquid_score`     = mean over t of √(Σ_d (liq_out − target)²).
//! - `absolute_gap`     = mean over t of √(Σ_d (ssm_out − liq_out)²) — the
//!                        symmetric disagreement, independent of the
//!                        target.
//! - `regime_shift_score` = max over t of √(Σ_d (ssm_out − liq_out)²) /
//!                          (1 + mean over t of same). The trailing `+ 1`
//!                          keeps the denominator bounded away from zero
//!                          when the networks agree perfectly. The score
//!                          is large when *one* step's gap is much bigger
//!                          than the average — exactly the regime-shift
//!                          signature.
//!
//! All reductions use `cjc_repro::KahanAccumulatorF64`, so cross-platform
//! byte-identity holds.

use crate::error::CronosGanError;
use cjc_repro::KahanAccumulatorF64;

/// Structured disagreement signal between two prediction trajectories.
///
/// All fields are non-negative by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalDisagreement {
    /// Mean per-step RMSE of the SSM against the target.
    pub ssm_score: f64,
    /// Mean per-step RMSE of the Liquid network against the target.
    pub liquid_score: f64,
    /// Mean per-step RMSE of SSM-vs-Liquid (target-free).
    pub absolute_gap: f64,
    /// Peak-to-mean ratio of the per-step SSM-vs-Liquid gap. Large values
    /// indicate a localised disagreement (regime shift) rather than a
    /// uniform mismatch.
    pub regime_shift_score: f64,
}

/// Compute [`TemporalDisagreement`] from per-step output trajectories.
///
/// `ssm_out`, `liq_out`, and `target` are row-major `[n_steps,
/// output_dim]`. All three must have the same shape. NaN / ±∞ in any
/// trajectory returns [`CronosGanError::NonFiniteInput`].
pub fn compute_disagreement(
    ssm_out: &[f64],
    liq_out: &[f64],
    target: &[f64],
    n_steps: usize,
    output_dim: usize,
) -> Result<TemporalDisagreement, CronosGanError> {
    let expected = n_steps * output_dim;
    if ssm_out.len() != expected || liq_out.len() != expected || target.len() != expected {
        return Err(CronosGanError::DimensionMismatch {
            detail: format!(
                "compute_disagreement: expected {} elements each, got ssm={} liq={} target={}",
                expected,
                ssm_out.len(),
                liq_out.len(),
                target.len()
            ),
        });
    }
    for (name, arr) in &[("ssm_out", ssm_out), ("liq_out", liq_out), ("target", target)] {
        for (i, &v) in arr.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosGanError::NonFiniteInput {
                    detail: format!("compute_disagreement: {}[{}] is non-finite", name, i),
                });
            }
        }
    }

    // Per-step Euclidean distances, Kahan-summed across steps then divided.
    let mut ssm_acc = KahanAccumulatorF64::new();
    let mut liq_acc = KahanAccumulatorF64::new();
    let mut gap_acc = KahanAccumulatorF64::new();
    let mut peak_gap = 0.0_f64;

    for t in 0..n_steps {
        // Per-dim Kahan inner products for each leg.
        let mut sq_ssm = KahanAccumulatorF64::new();
        let mut sq_liq = KahanAccumulatorF64::new();
        let mut sq_gap = KahanAccumulatorF64::new();
        for d in 0..output_dim {
            let i = t * output_dim + d;
            let e_ssm = ssm_out[i] - target[i];
            let e_liq = liq_out[i] - target[i];
            let e_gap = ssm_out[i] - liq_out[i];
            sq_ssm.add(e_ssm * e_ssm);
            sq_liq.add(e_liq * e_liq);
            sq_gap.add(e_gap * e_gap);
        }
        let step_rmse_ssm = sq_ssm.finalize().sqrt();
        let step_rmse_liq = sq_liq.finalize().sqrt();
        let step_rmse_gap = sq_gap.finalize().sqrt();
        ssm_acc.add(step_rmse_ssm);
        liq_acc.add(step_rmse_liq);
        gap_acc.add(step_rmse_gap);
        if step_rmse_gap > peak_gap {
            peak_gap = step_rmse_gap;
        }
    }
    let n = n_steps as f64;
    let ssm_score = ssm_acc.finalize() / n;
    let liquid_score = liq_acc.finalize() / n;
    let absolute_gap = gap_acc.finalize() / n;
    // +1 keeps the denominator from going to zero when both nets agree
    // perfectly (a real and useful corner case). The score equals
    // peak_gap exactly in the limit absolute_gap → 0; equals roughly
    // peak / mean when absolute_gap ≫ 1.
    let regime_shift_score = peak_gap / (1.0 + absolute_gap);

    Ok(TemporalDisagreement {
        ssm_score,
        liquid_score,
        absolute_gap,
        regime_shift_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_returns_error() {
        let err = compute_disagreement(&[0.0], &[0.0, 0.0], &[0.0], 1, 1).unwrap_err();
        assert!(matches!(err, CronosGanError::DimensionMismatch { .. }));
    }

    #[test]
    fn nan_input_returns_error() {
        let err = compute_disagreement(&[f64::NAN], &[0.0], &[0.0], 1, 1).unwrap_err();
        assert!(matches!(err, CronosGanError::NonFiniteInput { .. }));
    }

    #[test]
    fn perfect_agreement_with_zero_target_gap_returns_zero_scores() {
        // Both nets predict exactly the target ⇒ all four scores are 0,
        // except regime_shift = 0 / (1 + 0) = 0.
        let target = vec![1.0, 2.0, 3.0];
        let d = compute_disagreement(&target, &target, &target, 3, 1).unwrap();
        assert_eq!(d.ssm_score, 0.0);
        assert_eq!(d.liquid_score, 0.0);
        assert_eq!(d.absolute_gap, 0.0);
        assert_eq!(d.regime_shift_score, 0.0);
    }

    #[test]
    fn all_scores_non_negative_by_construction() {
        // Arbitrary numbers; sign flips and large magnitudes shouldn't
        // push any score negative.
        let ssm = vec![-3.0, 5.0, -7.0, 11.0];
        let liq = vec![1.0, -2.0, 8.0, -4.0];
        let tgt = vec![0.0, 0.0, 0.0, 0.0];
        let d = compute_disagreement(&ssm, &liq, &tgt, 4, 1).unwrap();
        assert!(d.ssm_score >= 0.0);
        assert!(d.liquid_score >= 0.0);
        assert!(d.absolute_gap >= 0.0);
        assert!(d.regime_shift_score >= 0.0);
    }

    #[test]
    fn regime_shift_score_fires_on_localised_gap() {
        // Three steps of agreement, one step of large gap. The
        // regime_shift_score should be large compared to the absolute_gap.
        let target = vec![0.0; 8];
        let ssm = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let liq = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        let d = compute_disagreement(&ssm, &liq, &target, 8, 1).unwrap();
        assert_eq!(d.ssm_score, 0.0);
        assert!(
            d.liquid_score > 0.0,
            "expected liquid_score > 0 for non-zero liquid prediction; got {}",
            d.liquid_score
        );
        // Peak gap = 10, mean gap = 10/8 = 1.25, regime = 10 / (1+1.25) ≈ 4.44.
        assert!(
            d.regime_shift_score > 3.0,
            "regime_shift_score {} below expected ~4.44 for localised gap",
            d.regime_shift_score
        );
    }

    #[test]
    fn determinism_byte_identical_across_calls() {
        let ssm = vec![0.1, 0.2, 0.3, 0.4];
        let liq = vec![0.05, 0.25, 0.31, 0.39];
        let tgt = vec![0.0, 0.3, 0.3, 0.5];
        let d1 = compute_disagreement(&ssm, &liq, &tgt, 4, 1).unwrap();
        let d2 = compute_disagreement(&ssm, &liq, &tgt, 4, 1).unwrap();
        assert_eq!(d1.ssm_score.to_bits(), d2.ssm_score.to_bits());
        assert_eq!(d1.liquid_score.to_bits(), d2.liquid_score.to_bits());
        assert_eq!(d1.absolute_gap.to_bits(), d2.absolute_gap.to_bits());
        assert_eq!(
            d1.regime_shift_score.to_bits(),
            d2.regime_shift_score.to_bits()
        );
    }
}
