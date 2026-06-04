//! Phase 3b: [`TemporalGanTrainer`] — alternating-update training for
//! the asymmetric Cronos GAN modes.
//!
//! ## The predictor/challenger framing
//!
//! Standard GANs train one network to fool the other. Cronos GAN does not:
//! disagreement is the **artifact** the brief asks for, not the loss to
//! minimise. The asymmetric modes encode this differently:
//!
//! - In `SsmAsGenerator`: SSM is the **predictor** — a vanilla supervised
//!   MSE learner. Liquid is the **challenger** — a supervised MSE learner
//!   with an *additional* `−λ · MSE-vs-SSM` term that *rewards* divergence
//!   from the SSM's prediction. The Liquid is held accountable to the
//!   target via the positive MSE term, but it's also incentivised to
//!   find alternative prediction paths.
//! - In `LiquidAsGenerator`: roles flip.
//!
//! Per step the trainer:
//! 1. Updates the predictor with one Adam step on standard supervised MSE.
//! 2. Runs the (now-updated) predictor's forward to get its per-step
//!    outputs — these become a fixed `ChallengerSpec` for the next stage.
//! 3. Updates the challenger with one Adam step on the asymmetric loss.
//! 4. Computes the resulting [`crate::TemporalDisagreement`] for
//!    inspection.
//!
//! Step ordering matters for determinism: same `(seed, config, inputs,
//! targets, initial Adam state)` ⇒ byte-identical predictor update ⇒
//! byte-identical predictor outputs ⇒ byte-identical challenger update.
//! The replay invariant is the entire reason Cronos GAN exists.
//!
//! In `Symmetric` mode the trainer reduces to two independent supervised
//! Adam updates with no inter-network coupling.

use crate::disagreement::TemporalDisagreement;
use crate::error::CronosGanError;
use crate::gan::{TemporalGan, TemporalGanConfig, TemporalGanMode};
use crate::liquid::{LiquidNetwork, LiquidState};
use crate::ssm::{StateSpaceModel, StateSpaceState};
use crate::training::{
    ChallengerSpec, LossAggregation, RolloutLossKind, SupervisedTrainer, Trainable,
};

/// Per-network role at a given training step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Trained with pure supervised MSE. In symmetric mode, both networks
    /// are predictors.
    Predictor,
    /// Trained with supervised MSE − λ · MSE-vs-predictor. Reads the
    /// predictor's current outputs as a fixed reference.
    Challenger,
}

/// Result of one [`TemporalGanTrainer::step`].
#[derive(Clone, Copy, Debug)]
pub struct TemporalGanTrainStep {
    /// SSM loss BEFORE its Adam update on this step.
    pub ssm_loss: f64,
    /// Liquid loss BEFORE its Adam update on this step.
    pub liquid_loss: f64,
    /// SSM's role on this step (`Predictor` in symmetric and
    /// SsmAsGenerator modes, `Challenger` in LiquidAsGenerator mode).
    pub ssm_role: Role,
    pub liquid_role: Role,
    /// Disagreement computed AFTER both Adam updates on this step.
    pub disagreement: TemporalDisagreement,
}

/// Alternating-update trainer for the asymmetric Cronos GAN modes.
///
/// Holds one [`SupervisedTrainer`] per network so each carries its own
/// Adam moment buffers. Stateless across calls except for those buffers
/// and the step counter.
pub struct TemporalGanTrainer {
    ssm_trainer: SupervisedTrainer,
    liquid_trainer: SupervisedTrainer,
    config: TemporalGanConfig,
    step_count: u64,
}

impl TemporalGanTrainer {
    /// Construct from a configured [`TemporalGan`] and learning rate.
    pub fn new(config: TemporalGanConfig, gan: &TemporalGan, lr: f64) -> Self {
        Self {
            ssm_trainer: SupervisedTrainer::new(gan.ssm().n_params(), lr),
            liquid_trainer: SupervisedTrainer::new(gan.liquid().n_params(), lr),
            config,
            step_count: 0,
        }
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn config(&self) -> &TemporalGanConfig {
        &self.config
    }

    /// Run one alternating-update training step.
    ///
    /// **Determinism**: with the same `(self, gan, inputs, targets)`,
    /// the returned `TemporalGanTrainStep` and the new state of both
    /// networks are byte-identical across runs and platforms.
    pub fn step(
        &mut self,
        gan: &mut TemporalGan,
        inputs: &[f64],
        targets: &[f64],
    ) -> Result<TemporalGanTrainStep, CronosGanError> {
        validate_lambda_if_asymmetric(&self.config)?;

        let (ssm_loss, liquid_loss, ssm_role, liquid_role) = match self.config.mode {
            TemporalGanMode::Symmetric => {
                let s = self.ssm_trainer.step(gan.ssm_mut(), inputs, targets)?;
                let l = self.liquid_trainer.step(gan.liquid_mut(), inputs, targets)?;
                (s, l, Role::Predictor, Role::Predictor)
            }
            TemporalGanMode::SsmAsGenerator => {
                // SSM is the predictor → supervised step.
                let s = self.ssm_trainer.step(gan.ssm_mut(), inputs, targets)?;
                // Read SSM outputs with the updated weights.
                let predictor_outputs = forward_ssm_outputs(gan.ssm(), inputs)?;
                let spec = ChallengerSpec {
                    predictor_outputs: &predictor_outputs,
                    lambda: self.config.lambda_disagreement,
                };
                let l = self
                    .liquid_trainer
                    .step_with(gan.liquid_mut(), inputs, targets, Some(&spec))?;
                (s, l, Role::Predictor, Role::Challenger)
            }
            TemporalGanMode::LiquidAsGenerator => {
                // Liquid is the predictor → supervised step.
                let l = self.liquid_trainer.step(gan.liquid_mut(), inputs, targets)?;
                let predictor_outputs = forward_liquid_outputs(gan.liquid(), inputs)?;
                let spec = ChallengerSpec {
                    predictor_outputs: &predictor_outputs,
                    lambda: self.config.lambda_disagreement,
                };
                let s = self
                    .ssm_trainer
                    .step_with(gan.ssm_mut(), inputs, targets, Some(&spec))?;
                (s, l, Role::Challenger, Role::Predictor)
            }
        };

        // Disagreement is computed AFTER both updates. The user sees the
        // gap *at the end of this training step*, which is the quantity
        // they'd reason about for regime-shift inspection.
        let rollout = gan.rollout_and_disagreement(inputs, targets)?;

        self.step_count += 1;
        Ok(TemporalGanTrainStep {
            ssm_loss,
            liquid_loss,
            ssm_role,
            liquid_role,
            disagreement: rollout.disagreement,
        })
    }
}

fn validate_lambda_if_asymmetric(cfg: &TemporalGanConfig) -> Result<(), CronosGanError> {
    if cfg.mode.is_asymmetric()
        && (!cfg.lambda_disagreement.is_finite() || cfg.lambda_disagreement < 0.0)
    {
        return Err(CronosGanError::InvalidConfig {
            detail: format!(
                "TemporalGanConfig.lambda_disagreement must be finite and >= 0 in asymmetric mode, got {}",
                cfg.lambda_disagreement
            ),
        });
    }
    Ok(())
}

/// Run the SSM forward over `inputs` and return its per-step outputs
/// concatenated row-major `[n_steps * output_dim]`. Does NOT use the
/// autograd graph — this is a plain inference call.
fn forward_ssm_outputs(model: &StateSpaceModel, inputs: &[f64]) -> Result<Vec<f64>, CronosGanError> {
    let cfg = model.config();
    let s0 = StateSpaceState::zeros(cfg.state_dim);
    let r = model.rollout(&s0, inputs)?;
    let mut out = Vec::with_capacity(r.outputs.len() * cfg.output_dim);
    for step_out in &r.outputs {
        out.extend_from_slice(step_out);
    }
    Ok(out)
}

fn forward_liquid_outputs(model: &LiquidNetwork, inputs: &[f64]) -> Result<Vec<f64>, CronosGanError> {
    let cfg = model.config();
    let s0 = LiquidState::zeros(cfg.state_dim);
    let r = model.rollout(&s0, inputs)?;
    let mut out = Vec::with_capacity(r.outputs.len() * cfg.output_dim);
    for step_out in &r.outputs {
        out.extend_from_slice(step_out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::CronosSeed;

    fn sine_io(n_steps: usize) -> (Vec<f64>, Vec<f64>) {
        let inputs: Vec<f64> = (0..n_steps).map(|t| (t as f64 * 0.4).sin()).collect();
        let targets: Vec<f64> = (0..n_steps).map(|t| ((t + 1) as f64 * 0.4).sin()).collect();
        (inputs, targets)
    }

    #[test]
    fn validate_lambda_rejects_negative_in_asymmetric() {
        let cfg = TemporalGanConfig::symmetric(4, 1, 1)
            .with_mode(TemporalGanMode::SsmAsGenerator)
            .with_lambda_disagreement(-0.1);
        let gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let (inputs, targets) = sine_io(10);
        let mut gan = gan;
        let err = trainer.step(&mut gan, &inputs, &targets).unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }

    #[test]
    fn symmetric_mode_records_both_as_predictor() {
        let cfg = TemporalGanConfig::symmetric(4, 1, 1);
        let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let (inputs, targets) = sine_io(15);
        let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
        assert_eq!(step.ssm_role, Role::Predictor);
        assert_eq!(step.liquid_role, Role::Predictor);
    }

    #[test]
    fn ssm_as_generator_records_roles_correctly() {
        let cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
        let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let (inputs, targets) = sine_io(15);
        let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
        assert_eq!(step.ssm_role, Role::Predictor);
        assert_eq!(step.liquid_role, Role::Challenger);
    }

    #[test]
    fn liquid_as_generator_records_roles_correctly() {
        let cfg = TemporalGanConfig::liquid_as_generator(4, 1, 1, 0.1);
        let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let (inputs, targets) = sine_io(15);
        let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
        assert_eq!(step.ssm_role, Role::Challenger);
        assert_eq!(step.liquid_role, Role::Predictor);
    }
}
