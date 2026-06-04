//! Phase 3 (minimal): [`TemporalGan`] container for the SSM+Liquid pair.
//!
//! Symmetric mode only. Asymmetric modes + alternating-update training
//! ship in Phase 3b — the brief's "disagreement IS the artifact" framing
//! needs a careful design pass on which network's loss adds vs subtracts
//! the disagreement, and that question deserves its own session.

use crate::disagreement::{compute_disagreement, TemporalDisagreement};
use crate::error::CronosGanError;
use crate::liquid::{LiquidConfig, LiquidNetwork, LiquidState};
use crate::seed::{CronosRunId, CronosSeed};
use crate::ssm::{StateSpaceConfig, StateSpaceModel, StateSpaceState};

/// Which adversarial mode the GAN runs.
///
/// Only `Symmetric` ships in Phase 3 minimal; `SsmAsGenerator` and
/// `LiquidAsGenerator` are reserved for Phase 3b.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemporalGanMode {
    /// Both networks predict the same target; disagreement is symmetric.
    Symmetric,
}

/// Configuration for a [`TemporalGan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalGanConfig {
    pub ssm: StateSpaceConfig,
    pub liquid: LiquidConfig,
    pub mode: TemporalGanMode,
}

impl TemporalGanConfig {
    /// Construct with both networks sharing the same `(state_dim,
    /// input_dim, output_dim)` triple — required for their outputs to be
    /// directly comparable.
    pub fn symmetric(state_dim: usize, input_dim: usize, output_dim: usize) -> Self {
        Self {
            ssm: StateSpaceConfig::new(state_dim, input_dim, output_dim),
            liquid: LiquidConfig::new(state_dim, input_dim, output_dim),
            mode: TemporalGanMode::Symmetric,
        }
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.ssm.canonical_bytes());
        bytes.extend(self.liquid.canonical_bytes());
        bytes.extend_from_slice(&(self.mode as u32 as u64).to_le_bytes());
        bytes
    }
}

/// Result of running [`TemporalGan::rollout_and_disagreement`].
#[derive(Clone, Debug)]
pub struct TemporalGanRolloutResult {
    /// Per-step SSM outputs, row-major `[n_steps, output_dim]`.
    pub ssm_outputs: Vec<f64>,
    /// Per-step Liquid outputs, row-major.
    pub liquid_outputs: Vec<f64>,
    /// Per-step Liquid τ trajectory (flattened, `[n_steps, state_dim]`).
    /// Useful for attributing regime-shift events to gate behaviour.
    pub liquid_tau_trace: Vec<f64>,
    /// Disagreement scores.
    pub disagreement: TemporalDisagreement,
    /// Content-addressed run ID of this rollout.
    pub run_id: CronosRunId,
}

/// A Temporal GAN pairing one [`StateSpaceModel`] and one [`LiquidNetwork`].
#[derive(Clone, Debug)]
pub struct TemporalGan {
    config: TemporalGanConfig,
    ssm: StateSpaceModel,
    liquid: LiquidNetwork,
    seed: CronosSeed,
}

impl TemporalGan {
    /// Construct both networks from the same `CronosSeed`. Per Phase 1's
    /// `independent_rng_substreams` invariant, the SSM and Liquid are
    /// initialised from disjoint SplitMix64 streams even though they
    /// share the same seed.
    pub fn from_seed(config: TemporalGanConfig, seed: CronosSeed) -> Result<Self, CronosGanError> {
        let ssm = StateSpaceModel::from_seed(config.ssm, seed)?;
        let liquid = LiquidNetwork::from_seed(config.liquid, seed)?;
        Ok(Self {
            config,
            ssm,
            liquid,
            seed,
        })
    }

    pub fn config(&self) -> &TemporalGanConfig {
        &self.config
    }

    pub fn ssm(&self) -> &StateSpaceModel {
        &self.ssm
    }

    pub fn liquid(&self) -> &LiquidNetwork {
        &self.liquid
    }

    pub fn ssm_mut(&mut self) -> &mut StateSpaceModel {
        &mut self.ssm
    }

    pub fn liquid_mut(&mut self) -> &mut LiquidNetwork {
        &mut self.liquid
    }

    /// Run both networks on the same input sequence, evaluate against the
    /// given target, and compute the structured disagreement.
    pub fn rollout_and_disagreement(
        &self,
        inputs: &[f64],
        target: &[f64],
    ) -> Result<TemporalGanRolloutResult, CronosGanError> {
        let cfg = &self.config;
        let id = cfg.ssm.input_dim;
        let od = cfg.ssm.output_dim;
        if cfg.liquid.input_dim != id || cfg.liquid.output_dim != od {
            return Err(CronosGanError::InvalidConfig {
                detail: format!(
                    "TemporalGan: SSM dims ({},{}) and Liquid dims ({},{}) must match",
                    id, od, cfg.liquid.input_dim, cfg.liquid.output_dim,
                ),
            });
        }
        if inputs.is_empty() || inputs.len() % id != 0 {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "TemporalGan rollout: inputs.len()={} not positive multiple of input_dim={}",
                    inputs.len(),
                    id
                ),
            });
        }
        let n_steps = inputs.len() / id;
        if target.len() != n_steps * od {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "TemporalGan rollout: target.len()={} expected {}",
                    target.len(),
                    n_steps * od
                ),
            });
        }

        let ssm_r = self
            .ssm
            .rollout(&StateSpaceState::zeros(cfg.ssm.state_dim), inputs)?;
        let mut ssm_outputs = Vec::with_capacity(n_steps * od);
        for step_out in &ssm_r.outputs {
            ssm_outputs.extend_from_slice(step_out);
        }

        let liq_r = self
            .liquid
            .rollout(&LiquidState::zeros(cfg.liquid.state_dim), inputs)?;
        let mut liquid_outputs = Vec::with_capacity(n_steps * od);
        let mut liquid_tau_trace = Vec::with_capacity(n_steps * cfg.liquid.state_dim);
        for step_out in &liq_r.outputs {
            liquid_outputs.extend_from_slice(step_out);
        }
        for tc in &liq_r.time_constants {
            liquid_tau_trace.extend_from_slice(&tc.tau);
        }

        let disagreement =
            compute_disagreement(&ssm_outputs, &liquid_outputs, target, n_steps, od)?;

        let run_id = CronosRunId::build(
            self.seed,
            "temporal_gan_symmetric_rollout",
            &cfg.canonical_bytes(),
        );

        Ok(TemporalGanRolloutResult {
            ssm_outputs,
            liquid_outputs,
            liquid_tau_trace,
            disagreement,
            run_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_gan_byte_identical_across_runs() {
        let cfg = TemporalGanConfig::symmetric(4, 1, 1);
        let g1 = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let g2 = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();

        let n_steps = 10;
        let inputs: Vec<f64> = (0..n_steps).map(|i| (i as f64 * 0.1).sin()).collect();
        let target: Vec<f64> = (0..n_steps).map(|i| (i as f64 * 0.11).cos()).collect();

        let r1 = g1.rollout_and_disagreement(&inputs, &target).unwrap();
        let r2 = g2.rollout_and_disagreement(&inputs, &target).unwrap();

        assert_eq!(r1.run_id, r2.run_id);
        assert_eq!(r1.disagreement.ssm_score.to_bits(), r2.disagreement.ssm_score.to_bits());
        assert_eq!(r1.disagreement.liquid_score.to_bits(), r2.disagreement.liquid_score.to_bits());
        assert_eq!(r1.disagreement.absolute_gap.to_bits(), r2.disagreement.absolute_gap.to_bits());
        assert_eq!(
            r1.disagreement.regime_shift_score.to_bits(),
            r2.disagreement.regime_shift_score.to_bits()
        );
    }

    #[test]
    fn run_id_differs_when_seed_changes() {
        let cfg = TemporalGanConfig::symmetric(4, 1, 1);
        let g1 = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let g2 = TemporalGan::from_seed(cfg, CronosSeed(43)).unwrap();
        let inputs = vec![0.0_f64; 4];
        let target = vec![0.0_f64; 4];
        let r1 = g1.rollout_and_disagreement(&inputs, &target).unwrap();
        let r2 = g2.rollout_and_disagreement(&inputs, &target).unwrap();
        assert_ne!(r1.run_id, r2.run_id);
    }

    #[test]
    fn liquid_tau_trace_dims_match_config() {
        let cfg = TemporalGanConfig::symmetric(6, 1, 1);
        let g = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let n_steps = 8;
        let inputs: Vec<f64> = (0..n_steps).map(|i| (i as f64 * 0.05).sin()).collect();
        let target = vec![0.0; n_steps];
        let r = g.rollout_and_disagreement(&inputs, &target).unwrap();
        assert_eq!(r.liquid_tau_trace.len(), n_steps * 6);
        for &t in &r.liquid_tau_trace {
            assert!(t >= cfg.liquid.tau_min && t <= cfg.liquid.tau_max);
        }
    }
}
