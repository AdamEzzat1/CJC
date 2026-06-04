//! Phase 4: synthetic-dataset experiment harness.
//!
//! Wires together datasets ([`crate::datasets`]), per-network supervised
//! training ([`crate::training::SupervisedTrainer`]), and disagreement
//! scoring ([`crate::disagreement::compute_disagreement`]) into a single
//! [`run_experiment`] entry point that returns a structured
//! [`ExperimentReport`] plus a deterministic replay hash.
//!
//! The harness is intentionally **minimal**: no parallel training,
//! no cross-fitting, no early stopping. Phase 4's job is to prove the
//! end-to-end pipeline holds determinism + correctly reports the
//! disagreement statistics on five distinct dataset shapes. Hyperparameter
//! tuning per-dataset is deferred to Phase 4b.

use crate::datasets::CronosDataset;
use crate::disagreement::TemporalDisagreement;
use crate::error::CronosGanError;
use crate::gan::{TemporalGan, TemporalGanConfig};
use crate::seed::{CronosRunId, CronosSeed};
use crate::training::{LossAggregation, RolloutLossKind, SupervisedTrainer, Trainable};
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, IdDomain};

/// Configuration for one Cronos GAN experiment run.
#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    pub gan: TemporalGanConfig,
    pub dataset: CronosDataset,
    pub n_steps: usize,
    pub n_train_steps: usize,
    pub lr: f64,
}

impl ExperimentConfig {
    /// Default `n_train_steps = 150`, `lr = 1e-2`.
    pub fn new(gan: TemporalGanConfig, dataset: CronosDataset, n_steps: usize) -> Self {
        Self {
            gan,
            dataset,
            n_steps,
            n_train_steps: 150,
            lr: 1e-2,
        }
    }

    pub fn with_n_train_steps(mut self, n: usize) -> Self {
        self.n_train_steps = n;
        self
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.gan.ssm.canonical_bytes());
        bytes.extend(self.gan.liquid.canonical_bytes());
        bytes.extend_from_slice(self.dataset.label().as_bytes());
        bytes.extend_from_slice(&(self.n_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.n_train_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&self.lr.to_bits().to_le_bytes());
        bytes
    }
}

/// Per-experiment training trajectory: one loss value per training step,
/// per network.
#[derive(Clone, Debug)]
pub struct TrainingTrajectory {
    pub ssm_losses: Vec<f64>,
    pub liquid_losses: Vec<f64>,
}

/// Structured experiment result.
#[derive(Clone, Debug)]
pub struct ExperimentReport {
    pub dataset_label: &'static str,
    pub initial_loss_ssm: f64,
    pub final_loss_ssm: f64,
    pub initial_loss_liquid: f64,
    pub final_loss_liquid: f64,
    pub training_trajectory: TrainingTrajectory,
    pub final_disagreement: TemporalDisagreement,
    /// Content-addressed hash over `(experiment config, final SSM params,
    /// final Liquid params, training-trajectory bit pattern)`. Two runs of
    /// the same `(config, seed)` produce the same hash.
    pub replay_hash: CronosRunId,
    /// Convenience copy of the experiment's `CronosSeed`.
    pub seed: CronosSeed,
}

/// Run one full experiment: train both networks on the given dataset,
/// evaluate disagreement, return a structured report.
///
/// **Determinism guarantee**: two runs with the same
/// `(ExperimentConfig, CronosSeed)` produce a
/// byte-identical `ExperimentReport` (including the `replay_hash`).
pub fn run_experiment(
    config: &ExperimentConfig,
    seed: CronosSeed,
) -> Result<ExperimentReport, CronosGanError> {
    // Build dataset.
    let (inputs, targets) = config.dataset.generate(seed, config.n_steps)?;

    // Construct GAN (both networks from same seed via independent
    // substreams).
    let mut gan = TemporalGan::from_seed(config.gan, seed)?;

    // Train SSM.
    let n_ssm = gan.ssm().n_params();
    let mut ssm_trainer = SupervisedTrainer::new(n_ssm, config.lr);
    let initial_loss_ssm = ssm_trainer
        .step(gan.ssm_mut(), &inputs, &targets)
        .map_err(prepend_context("SSM initial"))?;
    let mut ssm_losses = Vec::with_capacity(config.n_train_steps);
    ssm_losses.push(initial_loss_ssm);
    let mut last_ssm = initial_loss_ssm;
    for _ in 0..config.n_train_steps {
        last_ssm = ssm_trainer
            .step(gan.ssm_mut(), &inputs, &targets)
            .map_err(prepend_context("SSM train step"))?;
        ssm_losses.push(last_ssm);
    }
    let final_loss_ssm = last_ssm;

    // Train Liquid.
    let n_liq = gan.liquid().n_params();
    let mut liq_trainer = SupervisedTrainer::new(n_liq, config.lr);
    let initial_loss_liquid = liq_trainer
        .step(gan.liquid_mut(), &inputs, &targets)
        .map_err(prepend_context("Liquid initial"))?;
    let mut liquid_losses = Vec::with_capacity(config.n_train_steps);
    liquid_losses.push(initial_loss_liquid);
    let mut last_liq = initial_loss_liquid;
    for _ in 0..config.n_train_steps {
        last_liq = liq_trainer
            .step(gan.liquid_mut(), &inputs, &targets)
            .map_err(prepend_context("Liquid train step"))?;
        liquid_losses.push(last_liq);
    }
    let final_loss_liquid = last_liq;

    // Evaluate disagreement on the same sequence (training+eval split is
    // a Phase 4b refinement).
    let rollout = gan.rollout_and_disagreement(&inputs, &targets)?;

    // Replay hash: combines (config bytes, seed, final SSM params, final
    // Liquid params, every training-trajectory loss bit pattern).
    let mut parts = Vec::new();
    parts.push(fingerprint_str(IdDomain::CausalClaim, "cronos_gan_experiment"));
    parts.push(fingerprint(
        IdDomain::CausalClaim,
        &config.canonical_bytes(),
    ));
    parts.push(fingerprint(IdDomain::CausalClaim, &seed.0.to_le_bytes()));
    let ssm_params_bytes = float_vec_bytes(&gan.ssm().params_flat());
    parts.push(fingerprint(IdDomain::CausalClaim, &ssm_params_bytes));
    let liq_params_bytes = float_vec_bytes(&gan.liquid().params_flat());
    parts.push(fingerprint(IdDomain::CausalClaim, &liq_params_bytes));
    let ssm_trace_bytes = float_vec_bytes(&ssm_losses);
    parts.push(fingerprint(IdDomain::CausalClaim, &ssm_trace_bytes));
    let liq_trace_bytes = float_vec_bytes(&liquid_losses);
    parts.push(fingerprint(IdDomain::CausalClaim, &liq_trace_bytes));
    let replay_hash = CronosRunId(fingerprint_compose(
        IdDomain::CausalClaim,
        "cronos_experiment_replay_hash",
        &parts,
    ));

    Ok(ExperimentReport {
        dataset_label: config.dataset.label(),
        initial_loss_ssm,
        final_loss_ssm,
        initial_loss_liquid,
        final_loss_liquid,
        training_trajectory: TrainingTrajectory {
            ssm_losses,
            liquid_losses,
        },
        final_disagreement: rollout.disagreement,
        replay_hash,
        seed,
    })
}

fn float_vec_bytes(v: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 8);
    for &f in v {
        bytes.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    bytes
}

fn prepend_context(stage: &'static str) -> impl Fn(CronosGanError) -> CronosGanError {
    move |e| match e {
        CronosGanError::DimensionMismatch { detail } => CronosGanError::DimensionMismatch {
            detail: format!("[{}] {}", stage, detail),
        },
        CronosGanError::InvalidConfig { detail } => CronosGanError::InvalidConfig {
            detail: format!("[{}] {}", stage, detail),
        },
        CronosGanError::NonFiniteInput { detail } => CronosGanError::NonFiniteInput {
            detail: format!("[{}] {}", stage, detail),
        },
        other => other,
    }
}

// Tests are intentionally light here — full coverage lives in the
// workspace-level integration suite at `tests/cronos/integration/`.
// These two confirm the two top-level invariants.
#[cfg(test)]
mod tests {
    use super::*;

    fn small_config(ds: CronosDataset) -> ExperimentConfig {
        let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
        ExperimentConfig::new(gan_cfg, ds, 20)
            .with_n_train_steps(30)
            .with_lr(1e-2)
    }

    #[test]
    fn experiment_runs_end_to_end_on_smooth_sine() {
        let cfg = small_config(CronosDataset::SmoothSine);
        let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
        assert_eq!(report.dataset_label, "smooth_sine");
        assert!(report.final_loss_ssm < report.initial_loss_ssm);
        assert!(report.final_loss_liquid < report.initial_loss_liquid);
        assert_eq!(
            report.training_trajectory.ssm_losses.len(),
            cfg.n_train_steps + 1
        );
        assert!(report.final_disagreement.ssm_score.is_finite());
        assert!(report.final_disagreement.absolute_gap >= 0.0);
    }

    #[test]
    fn experiment_replay_hash_byte_identical_across_runs() {
        let cfg = small_config(CronosDataset::RegimeShift);
        let r1 = run_experiment(&cfg, CronosSeed(42)).unwrap();
        let r2 = run_experiment(&cfg, CronosSeed(42)).unwrap();
        assert_eq!(r1.replay_hash, r2.replay_hash);
        // And the trajectory itself is bit-identical.
        for (a, b) in r1
            .training_trajectory
            .ssm_losses
            .iter()
            .zip(r2.training_trajectory.ssm_losses.iter())
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
