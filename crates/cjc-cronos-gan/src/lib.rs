//! # cjc-cronos-gan — experimental deterministic temporal adversarial modeling
//!
//! **Status:** Phase 1 SCAFFOLDING. Ships the core temporal primitives, the
//! State Space Model (SSM) and Liquid Neural Network adversaries' forward
//! steps + rollouts, and the deterministic seeding contract. The
//! adversarial training layer, the synthetic-dataset experiment harness,
//! the property + Bolero fuzz suites, and the Obsidian vault deep-docs
//! ship in subsequent phases.
//!
//! ## ⚠️ Experimental crate
//!
//! Cronos-GAN is currently an experimental deterministic temporal modeling
//! crate. The first goal is **correctness, auditability, and
//! reproducibility**, not state-of-the-art forecasting accuracy. Do not
//! benchmark this crate against statsmodels / Prophet / N-BEATS until at
//! least Phase 4 (experiment harness + training loop) lands.
//!
//! ## Architecture
//!
//! The Temporal GAN opposes two structurally different temporal models:
//!
//! - [`StateSpaceModel`] — stable long-range latent dynamics. Linear,
//!   time-invariant. Spectral norm of the transition matrix is `≤ α < 1`
//!   *by construction*, so the network forgets perturbations exponentially.
//! - [`LiquidNetwork`] — adaptive nonlinear local dynamics. Discrete liquid
//!   time-constant network; the per-dimension time constant is gated by
//!   the current input *and* current state via clipped softplus, so the
//!   network can be slow (memory-like) or fast (reactive) depending on
//!   what's happening at the current step.
//!
//! These are deliberately opposite inductive biases. The GAN's
//! disagreement score (Phase 3) reads the gap between them as a
//! **regime-shift signal**: large gap ⇒ the data has just transitioned
//! between "smoothly evolving" and "locally volatile" regimes.
//!
//! ## Determinism contract
//!
//! Every random draw in the crate routes through [`CronosSeed::substream`]
//! with a domain-named salt. Two distinct domains (e.g. `"ssm.A"` and
//! `"liquid.W_h"`) never share a stream and therefore can't accidentally
//! couple — the same seed with the same configuration produces
//! byte-identical parameters across runs, machines, and platforms.
//!
//! Specifically:
//! 1. All RNG via `cjc_repro::Rng` (SplitMix64), seeded from
//!    [`CronosSeed::substream`].
//! 2. All reductions (matrix-vector products, row L² norms, loss sums) use
//!    `cjc_repro::KahanAccumulatorF64`.
//! 3. Liquid `tau` is clipped into `[tau_min, tau_max]` by construction;
//!    `softplus` is overflow-safe at every finite f64 input.
//! 4. SSM transition matrix `A` satisfies `||A||_F = α` exactly (and
//!    therefore `||A||_2 ≤ α < 1`) — stability is a property of the
//!    construction, not of training.
//! 5. No `HashMap` iteration anywhere. No FMA. No thread-parallel
//!    reductions that would alter accumulation order.
//!
//! ## What ships in Phase 1
//!
//! - Temporal primitives: [`TimeStep`], [`TimeSeries`], [`TemporalBatch`],
//!   [`SequenceMask`], [`ForecastWindow`], [`TemporalLoss`].
//! - State trait: [`TemporalState`], [`TemporalTransition`],
//!   [`TemporalRollout`].
//! - SSM: [`StateSpaceConfig`], [`StateSpaceParams`], [`StateSpaceState`],
//!   [`StateSpaceModel`], [`StateSpaceStepResult`],
//!   [`StateSpaceRolloutResult`].
//! - Liquid net: [`LiquidConfig`], [`LiquidParams`], [`LiquidState`],
//!   [`LiquidNetwork`], [`LiquidTimeConstant`], [`LiquidGate`],
//!   [`LiquidStepResult`], [`LiquidRolloutResult`].
//! - Determinism: [`CronosSeed`], [`CronosRunId`].
//! - Errors: [`CronosGanError`].
//!
//! ## What is deferred to later phases
//!
//! - Phase 2: training-loss helpers, autodiff integration via
//!   `cjc_ad::GradGraph`, per-network training loop.
//! - Phase 3: `TemporalGan`, `TemporalGanTrainer`, `TemporalDisagreement`
//!   inspector, three GAN modes (SSM-as-generator, Liquid-as-generator,
//!   symmetric).
//! - Phase 4: synthetic-dataset experiment harness (smooth sine, noisy
//!   sine, regime shift, step-change anomaly, chaotic spike) + result
//!   reporting.
//! - Phase 5: property tests (proptest) + Bolero fuzz targets, the full
//!   `tests/cronos/{unit,integration,prop,fuzz}/` workspace-level layout,
//!   Obsidian vault deep-docs (Architecture, SSM Primitive, Liquid
//!   Primitive, Adversarial Training, Experiment Results, Verification
//!   Report).
//! - Phase 6 (optional): cjc-locke composition (E9500+ custom-detector
//!   for regime-shift anomaly findings), Python bridge.
//!
//! ## Quick start
//!
//! ```
//! use cjc_cronos_gan::{
//!     CronosSeed, LiquidConfig, LiquidNetwork, LiquidState,
//!     StateSpaceConfig, StateSpaceModel, StateSpaceState,
//! };
//!
//! let seed = CronosSeed(42);
//!
//! // Both networks share input/output shapes so their outputs are
//! // directly comparable for disagreement scoring in Phase 3.
//! let ssm = StateSpaceModel::from_seed(
//!     StateSpaceConfig::new(8, 4, 2),
//!     seed,
//! ).unwrap();
//! let liquid = LiquidNetwork::from_seed(
//!     LiquidConfig::new(8, 4, 2),
//!     seed,
//! ).unwrap();
//!
//! // Roll both forward across the same input sequence.
//! let n_steps = 10;
//! let inputs: Vec<f64> = (0..n_steps * 4).map(|i| (i as f64 * 0.1).sin()).collect();
//! let ssm_out = ssm.rollout(&StateSpaceState::zeros(8), &inputs).unwrap();
//! let liq_out = liquid.rollout(&LiquidState::zeros(8), &inputs).unwrap();
//!
//! assert_eq!(ssm_out.n_steps(), liq_out.n_steps());
//! // Phase 3 will read (ssm_out.outputs, liq_out.outputs, liq_out.time_constants)
//! // to compute the TemporalDisagreement score.
//! ```

pub mod error;
pub mod liquid;
pub mod seed;
pub mod ssm;
pub mod temporal_state;
pub mod time_series;

pub use error::CronosGanError;
pub use liquid::{
    LiquidConfig, LiquidGate, LiquidNetwork, LiquidParams, LiquidRolloutResult, LiquidState,
    LiquidStepResult, LiquidTimeConstant,
};
pub use seed::{CronosRunId, CronosSeed};
pub use ssm::{
    StateSpaceConfig, StateSpaceModel, StateSpaceParams, StateSpaceRolloutResult,
    StateSpaceState, StateSpaceStepResult,
};
pub use temporal_state::{TemporalRollout, TemporalState, TemporalTransition};
pub use time_series::{
    ForecastWindow, SequenceMask, TemporalBatch, TemporalLoss, TimeSeries, TimeStep,
};

/// Re-export of `cjc_locke::id::FingerprintId` so callers don't need a
/// direct dep on cjc-locke just to spell content-addressed IDs.
pub use cjc_locke::id::FingerprintId;

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build both networks from the same seed, roll them
    /// forward across the same input sequence, confirm output shapes
    /// agree. This is the structural precondition for the Phase 3
    /// disagreement score.
    #[test]
    fn ssm_and_liquid_share_io_shape() {
        let seed = CronosSeed(42);
        let cfg_ssm = StateSpaceConfig::new(8, 4, 2);
        let cfg_liq = LiquidConfig::new(8, 4, 2);
        let ssm = StateSpaceModel::from_seed(cfg_ssm, seed).unwrap();
        let liq = LiquidNetwork::from_seed(cfg_liq, seed).unwrap();
        let n_steps = 12;
        let inputs: Vec<f64> = (0..n_steps * 4).map(|i| (i as f64 * 0.05).sin()).collect();
        let r_ssm = ssm.rollout(&StateSpaceState::zeros(8), &inputs).unwrap();
        let r_liq = liq.rollout(&LiquidState::zeros(8), &inputs).unwrap();
        assert_eq!(r_ssm.n_steps(), r_liq.n_steps());
        for (a, b) in r_ssm.outputs.iter().zip(r_liq.outputs.iter()) {
            assert_eq!(a.len(), b.len());
        }
    }

    /// Confirm cross-network independence: same seed but different domain
    /// salts produce *different* parameters. (If they didn't, SSM and
    /// Liquid would draw from the same RNG stream and disagree only on
    /// the deterministic part of their update rules — that would defeat
    /// the entire point of the adversarial setup.)
    #[test]
    fn ssm_and_liquid_use_independent_rng_substreams() {
        let seed = CronosSeed(42);
        let ssm = StateSpaceModel::from_seed(StateSpaceConfig::new(8, 4, 2), seed).unwrap();
        let liq = LiquidNetwork::from_seed(LiquidConfig::new(8, 4, 2), seed).unwrap();
        // Compare the SSM's transition matrix against the Liquid's
        // recurrent matrix — both `state_dim × state_dim`, both drawn
        // from N(0, init_scale²)-ish init, both seeded from the same
        // CronosSeed but using distinct salts ("ssm.A" vs "liquid.W_h").
        // They must differ.
        assert_eq!(ssm.params().a.len(), liq.params().w_h.len());
        assert_ne!(ssm.params().a, liq.params().w_h);
    }
}
