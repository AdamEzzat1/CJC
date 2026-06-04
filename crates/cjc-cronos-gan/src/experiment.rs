//! Phase 4 + 4b: synthetic-dataset experiment harness and 15-cell sweep.
//!
//! Phase 4 wired one [`run_experiment`] entry point that trained both
//! networks separately. Phase 4b reroutes that through the
//! [`crate::TemporalGanTrainer`] alternating-update path so all three
//! [`TemporalGanMode`](crate::TemporalGanMode) variants share one code
//! path AND adds the structured [`run_experiment_sweep`] driver covering
//! all 5 synthetic datasets × 3 GAN modes in a single call.
//!
//! ## What Phase 4b changes vs Phase 4
//!
//! - `run_experiment` now routes through `TemporalGanTrainer.step()`
//!   instead of two independent `SupervisedTrainer`s. For `Symmetric`
//!   mode the end-state parameters are mathematically equivalent
//!   (no inter-network coupling), but the bytes of the
//!   `training_trajectory` and `replay_hash` *shift* because the
//!   alternating update interleaves the Adam moments differently. This
//!   is a deliberate one-time rebaseline — Phase 4's hash values are
//!   no longer compared against.
//! - `ExperimentReport` gains `mode` and `disagreement_trajectory: Vec<TemporalDisagreement>`
//!   so callers can inspect the gap trajectory step-by-step.
//! - `SweepBaseConfig` carries per-dataset learning rates and shared
//!   dimensions/lambda.
//! - `ExperimentSweepReport` carries 15 cells in canonical iteration
//!   order — datasets in declaration order, modes in declaration order
//!   — plus a content-addressed `sweep_hash` for the entire sweep.

use crate::datasets::CronosDataset;
use crate::disagreement::TemporalDisagreement;
use crate::error::CronosGanError;
use crate::gan::{TemporalGan, TemporalGanConfig, TemporalGanMode};
use crate::gan_trainer::TemporalGanTrainer;
use crate::seed::{CronosRunId, CronosSeed};
use crate::training::Trainable;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, IdDomain};
use std::collections::BTreeMap;

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
    pub mode: TemporalGanMode,
    pub initial_loss_ssm: f64,
    pub final_loss_ssm: f64,
    pub initial_loss_liquid: f64,
    pub final_loss_liquid: f64,
    pub training_trajectory: TrainingTrajectory,
    /// Per-step disagreement during training — `disagreement_trajectory[t]`
    /// is the disagreement after the `t`-th training step (matching
    /// `training_trajectory.ssm_losses[t]`).
    pub disagreement_trajectory: Vec<TemporalDisagreement>,
    pub final_disagreement: TemporalDisagreement,
    /// Mean of `disagreement_trajectory[*].absolute_gap` across the
    /// post-initial training portion. Useful summary for the sweep
    /// table.
    pub mean_absolute_gap: f64,
    /// Maximum of `disagreement_trajectory[*].regime_shift_score`
    /// across training. Sentinel for whether the GAN saw a localised
    /// disagreement spike.
    pub max_regime_shift_score: f64,
    /// Content-addressed hash over `(experiment config, final SSM params,
    /// final Liquid params, training-trajectory bit pattern,
    /// disagreement-trajectory bit pattern)`. Two runs of the same
    /// `(config, seed)` produce the same hash.
    pub replay_hash: CronosRunId,
    /// Convenience copy of the experiment's `CronosSeed`.
    pub seed: CronosSeed,
}

/// Run one full experiment: train both networks via the unified
/// [`TemporalGanTrainer`] path (Phase 4b — same pipeline drives all
/// three [`TemporalGanMode`] variants), evaluate disagreement, return
/// a structured report.
///
/// **Determinism guarantee**: two runs with the same
/// `(ExperimentConfig, CronosSeed)` produce a byte-identical
/// `ExperimentReport` (including the `replay_hash`).
pub fn run_experiment(
    config: &ExperimentConfig,
    seed: CronosSeed,
) -> Result<ExperimentReport, CronosGanError> {
    let (inputs, targets) = config.dataset.generate(seed, config.n_steps)?;
    let mut gan = TemporalGan::from_seed(config.gan, seed)?;

    let mut trainer = TemporalGanTrainer::new(config.gan, &gan, config.lr);

    // Capacity = n_train_steps + 1 (initial + n more).
    let cap = config.n_train_steps + 1;
    let mut ssm_losses: Vec<f64> = Vec::with_capacity(cap);
    let mut liquid_losses: Vec<f64> = Vec::with_capacity(cap);
    let mut disagreement_trajectory: Vec<TemporalDisagreement> = Vec::with_capacity(cap);

    // Initial training step — equivalent to Phase 4's "initial loss" reading.
    let initial = trainer
        .step(&mut gan, &inputs, &targets)
        .map_err(prepend_context("initial training step"))?;
    let initial_loss_ssm = initial.ssm_loss;
    let initial_loss_liquid = initial.liquid_loss;
    ssm_losses.push(initial.ssm_loss);
    liquid_losses.push(initial.liquid_loss);
    disagreement_trajectory.push(initial.disagreement);

    for i in 0..config.n_train_steps {
        let step = trainer
            .step(&mut gan, &inputs, &targets)
            .map_err(prepend_context_step(i))?;
        ssm_losses.push(step.ssm_loss);
        liquid_losses.push(step.liquid_loss);
        disagreement_trajectory.push(step.disagreement);
    }

    let final_loss_ssm = *ssm_losses.last().expect("at least one entry");
    let final_loss_liquid = *liquid_losses.last().expect("at least one entry");

    // Re-run disagreement on the final-param state for canonical reporting.
    let rollout = gan.rollout_and_disagreement(&inputs, &targets)?;

    // Sweep-summary statistics computed over the FULL trajectory.
    let mean_absolute_gap = mean_absolute_gap(&disagreement_trajectory);
    let max_regime_shift_score = max_regime_shift(&disagreement_trajectory);

    // Replay hash: combines (config bytes, seed, final SSM params, final
    // Liquid params, training-trajectory bit pattern, disagreement-trajectory
    // bit pattern).
    let mut parts = Vec::new();
    parts.push(fingerprint_str(IdDomain::CausalClaim, "cronos_gan_experiment_v4b"));
    parts.push(fingerprint(IdDomain::CausalClaim, &config.canonical_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &seed.0.to_le_bytes()));
    let ssm_params_bytes = float_vec_bytes(&gan.ssm().params_flat());
    parts.push(fingerprint(IdDomain::CausalClaim, &ssm_params_bytes));
    let liq_params_bytes = float_vec_bytes(&gan.liquid().params_flat());
    parts.push(fingerprint(IdDomain::CausalClaim, &liq_params_bytes));
    let ssm_trace_bytes = float_vec_bytes(&ssm_losses);
    parts.push(fingerprint(IdDomain::CausalClaim, &ssm_trace_bytes));
    let liq_trace_bytes = float_vec_bytes(&liquid_losses);
    parts.push(fingerprint(IdDomain::CausalClaim, &liq_trace_bytes));
    let dis_bytes = disagreement_trajectory_bytes(&disagreement_trajectory);
    parts.push(fingerprint(IdDomain::CausalClaim, &dis_bytes));
    let replay_hash = CronosRunId(fingerprint_compose(
        IdDomain::CausalClaim,
        "cronos_experiment_replay_hash_v4b",
        &parts,
    ));

    Ok(ExperimentReport {
        dataset_label: config.dataset.label(),
        mode: config.gan.mode,
        initial_loss_ssm,
        final_loss_ssm,
        initial_loss_liquid,
        final_loss_liquid,
        training_trajectory: TrainingTrajectory {
            ssm_losses,
            liquid_losses,
        },
        disagreement_trajectory,
        final_disagreement: rollout.disagreement,
        mean_absolute_gap,
        max_regime_shift_score,
        replay_hash,
        seed,
    })
}

fn mean_absolute_gap(trajectory: &[TemporalDisagreement]) -> f64 {
    if trajectory.is_empty() {
        return 0.0;
    }
    let mut acc = cjc_repro::KahanAccumulatorF64::new();
    for d in trajectory {
        acc.add(d.absolute_gap);
    }
    acc.finalize() / trajectory.len() as f64
}

fn max_regime_shift(trajectory: &[TemporalDisagreement]) -> f64 {
    trajectory
        .iter()
        .map(|d| d.regime_shift_score)
        .fold(0.0_f64, |a, b| if b > a { b } else { a })
}

fn disagreement_trajectory_bytes(trajectory: &[TemporalDisagreement]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(trajectory.len() * 32);
    for d in trajectory {
        bytes.extend_from_slice(&d.ssm_score.to_bits().to_le_bytes());
        bytes.extend_from_slice(&d.liquid_score.to_bits().to_le_bytes());
        bytes.extend_from_slice(&d.absolute_gap.to_bits().to_le_bytes());
        bytes.extend_from_slice(&d.regime_shift_score.to_bits().to_le_bytes());
    }
    bytes
}

fn prepend_context_step(i: usize) -> impl Fn(CronosGanError) -> CronosGanError {
    move |e| match e {
        CronosGanError::DimensionMismatch { detail } => CronosGanError::DimensionMismatch {
            detail: format!("[training step {}] {}", i, detail),
        },
        CronosGanError::InvalidConfig { detail } => CronosGanError::InvalidConfig {
            detail: format!("[training step {}] {}", i, detail),
        },
        CronosGanError::NonFiniteInput { detail } => CronosGanError::NonFiniteInput {
            detail: format!("[training step {}] {}", i, detail),
        },
        other => other,
    }
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

// ───────────────────────────────────────────────────────────────────────
// Phase 4b: 15-cell sweep across 5 datasets × 3 modes
// ───────────────────────────────────────────────────────────────────────

/// Canonical iteration order for sweep datasets.
pub const SWEEP_DATASETS: [CronosDataset; 5] = [
    CronosDataset::SmoothSine,
    CronosDataset::NoisySine,
    CronosDataset::RegimeShift,
    CronosDataset::StepChangeAnomaly,
    CronosDataset::ChaoticSpike,
];

/// Canonical iteration order for sweep modes.
pub const SWEEP_MODES: [TemporalGanMode; 3] = [
    TemporalGanMode::Symmetric,
    TemporalGanMode::SsmAsGenerator,
    TemporalGanMode::LiquidAsGenerator,
];

/// Configuration shared across the 15 cells of a sweep.
///
/// All cells share `(state_dim, input_dim, output_dim, n_steps,
/// n_train_steps, lambda_disagreement)`. Per-dataset learning rates can
/// be overridden via [`with_lr_for`](Self::with_lr_for); cells without
/// an override use `default_lr`.
#[derive(Clone, Debug)]
pub struct SweepBaseConfig {
    pub state_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub n_steps: usize,
    pub n_train_steps: usize,
    /// Weight on the challenger network's `-λ · MSE-vs-predictor` term.
    /// Applied to both asymmetric modes; ignored for `Symmetric`.
    pub lambda_disagreement: f64,
    /// Per-dataset learning-rate overrides, keyed by dataset enum.
    pub per_dataset_lr: BTreeMap<CronosDataset, f64>,
    /// Fallback learning rate for datasets without an override.
    pub default_lr: f64,
}

impl SweepBaseConfig {
    /// Construct with reasonable defaults: `lambda = 0.1`, `default_lr = 1e-2`,
    /// no per-dataset overrides.
    pub fn new(
        state_dim: usize,
        input_dim: usize,
        output_dim: usize,
        n_steps: usize,
        n_train_steps: usize,
    ) -> Self {
        Self {
            state_dim,
            input_dim,
            output_dim,
            n_steps,
            n_train_steps,
            lambda_disagreement: 0.1,
            per_dataset_lr: BTreeMap::new(),
            default_lr: 1e-2,
        }
    }

    pub fn with_lambda_disagreement(mut self, lambda: f64) -> Self {
        self.lambda_disagreement = lambda;
        self
    }

    pub fn with_default_lr(mut self, lr: f64) -> Self {
        self.default_lr = lr;
        self
    }

    pub fn with_lr_for(mut self, dataset: CronosDataset, lr: f64) -> Self {
        self.per_dataset_lr.insert(dataset, lr);
        self
    }

    /// Construct a per-cell [`ExperimentConfig`] for `(dataset, mode)`.
    pub fn experiment_config_for(
        &self,
        dataset: CronosDataset,
        mode: TemporalGanMode,
    ) -> ExperimentConfig {
        let gan = TemporalGanConfig::symmetric(self.state_dim, self.input_dim, self.output_dim)
            .with_mode(mode)
            .with_lambda_disagreement(self.lambda_disagreement);
        let lr = self
            .per_dataset_lr
            .get(&dataset)
            .copied()
            .unwrap_or(self.default_lr);
        ExperimentConfig::new(gan, dataset, self.n_steps)
            .with_n_train_steps(self.n_train_steps)
            .with_lr(lr)
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.state_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.output_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.n_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.n_train_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&self.lambda_disagreement.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.default_lr.to_bits().to_le_bytes());
        // BTreeMap iteration is deterministic in key order — exactly the
        // property the determinism contract depends on.
        for (ds, lr) in &self.per_dataset_lr {
            bytes.extend_from_slice(ds.label().as_bytes());
            bytes.extend_from_slice(&lr.to_bits().to_le_bytes());
        }
        bytes
    }
}

/// One cell of an [`ExperimentSweepReport`].
#[derive(Clone, Debug)]
pub struct SweepCell {
    pub dataset: CronosDataset,
    pub mode: TemporalGanMode,
    pub report: ExperimentReport,
}

/// Structured report from a 15-cell sweep.
#[derive(Clone, Debug)]
pub struct ExperimentSweepReport {
    pub seed: CronosSeed,
    /// 15 cells in canonical order — datasets outer ([`SWEEP_DATASETS`]),
    /// modes inner ([`SWEEP_MODES`]).
    pub cells: Vec<SweepCell>,
    /// Content-addressed hash over `(base_config, seed, every cell's
    /// replay_hash)`. Two sweeps with the same `(base_config, seed)`
    /// produce the same hash.
    pub sweep_hash: CronosRunId,
}

impl ExperimentSweepReport {
    /// Fetch the report for a specific `(dataset, mode)` combination.
    pub fn cell(&self, dataset: CronosDataset, mode: TemporalGanMode) -> Option<&ExperimentReport> {
        self.cells
            .iter()
            .find(|c| c.dataset == dataset && c.mode == mode)
            .map(|c| &c.report)
    }

    /// Format a fixed-width 15-row table for stdout / a file. Columns:
    /// dataset, mode, final SSM loss, final Liquid loss,
    /// mean |gap|, max regime-shift, replay_hash.
    pub fn format_table(&self) -> String {
        let mut out = String::new();
        out.push_str("┌────────────────────────┬──────────────────────┬──────────────┬──────────────┬─────────────┬──────────────┬──────────────────┐\n");
        out.push_str("│ dataset                │ mode                 │ ssm_loss     │ liq_loss     │ mean |gap|  │ max regime   │ replay_hash      │\n");
        out.push_str("├────────────────────────┼──────────────────────┼──────────────┼──────────────┼─────────────┼──────────────┼──────────────────┤\n");
        for c in &self.cells {
            out.push_str(&format!(
                "│ {:<22} │ {:<20} │ {:>12.6e} │ {:>12.6e} │ {:>11.4e} │ {:>12.4e} │ {:>16} │\n",
                c.dataset.label(),
                c.mode.label(),
                c.report.final_loss_ssm,
                c.report.final_loss_liquid,
                c.report.mean_absolute_gap,
                c.report.max_regime_shift_score,
                format!("{}", c.report.replay_hash),
            ));
        }
        out.push_str("└────────────────────────┴──────────────────────┴──────────────┴──────────────┴─────────────┴──────────────┴──────────────────┘\n");
        out.push_str(&format!("sweep_hash = {}\n", self.sweep_hash));
        out.push_str(&format!("seed       = {}\n", self.seed.0));
        out
    }
}

/// Run the 15-cell sweep: 5 datasets × 3 modes. Cells are evaluated in
/// canonical order ([`SWEEP_DATASETS`] outer, [`SWEEP_MODES`] inner) so
/// the resulting [`ExperimentSweepReport.cells`] index is stable across
/// runs and platforms.
///
/// **Determinism guarantee**: two sweeps with the same `(base_config,
/// seed)` produce byte-identical `ExperimentSweepReport` (including the
/// `sweep_hash`).
pub fn run_experiment_sweep(
    base_config: &SweepBaseConfig,
    seed: CronosSeed,
) -> Result<ExperimentSweepReport, CronosGanError> {
    let mut cells: Vec<SweepCell> = Vec::with_capacity(SWEEP_DATASETS.len() * SWEEP_MODES.len());

    for &dataset in &SWEEP_DATASETS {
        for &mode in &SWEEP_MODES {
            let cfg = base_config.experiment_config_for(dataset, mode);
            let report = run_experiment(&cfg, seed).map_err(prepend_context_sweep(dataset, mode))?;
            cells.push(SweepCell { dataset, mode, report });
        }
    }

    // sweep_hash: combines (base_config bytes, seed, every cell's
    // replay_hash bytes).
    let mut parts = Vec::with_capacity(2 + cells.len());
    parts.push(fingerprint_str(IdDomain::CausalClaim, "cronos_sweep_v4b"));
    parts.push(fingerprint(IdDomain::CausalClaim, &base_config.canonical_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &seed.0.to_le_bytes()));
    for c in &cells {
        parts.push(fingerprint(
            IdDomain::CausalClaim,
            &c.report.replay_hash.0 .0.to_le_bytes(),
        ));
    }
    let sweep_hash = CronosRunId(fingerprint_compose(
        IdDomain::CausalClaim,
        "cronos_sweep_replay_hash",
        &parts,
    ));

    Ok(ExperimentSweepReport {
        seed,
        cells,
        sweep_hash,
    })
}

fn prepend_context_sweep(
    dataset: CronosDataset,
    mode: TemporalGanMode,
) -> impl Fn(CronosGanError) -> CronosGanError {
    move |e| match e {
        CronosGanError::DimensionMismatch { detail } => CronosGanError::DimensionMismatch {
            detail: format!("[sweep cell ({}, {})] {}", dataset.label(), mode.label(), detail),
        },
        CronosGanError::InvalidConfig { detail } => CronosGanError::InvalidConfig {
            detail: format!("[sweep cell ({}, {})] {}", dataset.label(), mode.label(), detail),
        },
        CronosGanError::NonFiniteInput { detail } => CronosGanError::NonFiniteInput {
            detail: format!("[sweep cell ({}, {})] {}", dataset.label(), mode.label(), detail),
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
