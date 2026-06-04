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
use crate::liquid::LiquidState;
use crate::seed::{CronosRunId, CronosSeed};
use crate::ssm::StateSpaceState;
use crate::training::Trainable;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, IdDomain};
use std::collections::BTreeMap;

/// Configuration for one Cronos GAN experiment run.
#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    pub gan: TemporalGanConfig,
    pub dataset: CronosDataset,
    /// Number of training steps the model sees. Training rollouts use
    /// `inputs[0..n_steps]` and `targets[0..n_steps]`.
    pub n_steps: usize,
    /// Phase 4c: number of held-out evaluation steps AFTER `n_steps`.
    /// The dataset generator is asked for `n_steps + eval_steps`
    /// samples. After training on the first `n_steps`, each network's
    /// rollout continues from its final-train state on
    /// `inputs[n_steps..n_steps + eval_steps]`, producing predictions
    /// compared against `targets[n_steps..n_steps + eval_steps]`. When
    /// `eval_steps == 0` (the default), behaviour matches Phase 4b
    /// exactly.
    pub eval_steps: usize,
    pub n_train_steps: usize,
    pub lr: f64,
}

impl ExperimentConfig {
    /// Default `n_train_steps = 150`, `lr = 1e-2`, `eval_steps = 0`
    /// (no held-out eval).
    pub fn new(gan: TemporalGanConfig, dataset: CronosDataset, n_steps: usize) -> Self {
        Self {
            gan,
            dataset,
            n_steps,
            eval_steps: 0,
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

    /// Phase 4c: set the held-out eval horizon. When > 0, the dataset
    /// generator produces `n_steps + eval_steps` total samples; eval
    /// is performed on the trailing window.
    pub fn with_eval_steps(mut self, eval_steps: usize) -> Self {
        self.eval_steps = eval_steps;
        self
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.gan.ssm.canonical_bytes());
        bytes.extend(self.gan.liquid.canonical_bytes());
        bytes.extend_from_slice(self.dataset.label().as_bytes());
        bytes.extend_from_slice(&(self.n_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.eval_steps as u64).to_le_bytes());
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

/// Phase 4c eval-window summary. `None` in [`ExperimentReport`] when
/// `eval_steps == 0`.
#[derive(Clone, Copy, Debug)]
pub struct EvalReport {
    /// MSE of the SSM's predictions on the held-out window.
    pub ssm_loss: f64,
    /// MSE of the Liquid network's predictions on the held-out window.
    pub liquid_loss: f64,
    /// Structured disagreement between SSM and Liquid on the held-out
    /// window. This is the true forecastability disagreement — neither
    /// network saw these timesteps during training.
    pub disagreement: TemporalDisagreement,
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
    /// Phase 4c: held-out eval results. `None` when
    /// `ExperimentConfig.eval_steps == 0`.
    pub eval: Option<EvalReport>,
    /// Content-addressed hash over `(experiment config, final SSM params,
    /// final Liquid params, training-trajectory bit pattern,
    /// disagreement-trajectory bit pattern, eval-report bit pattern)`.
    /// Two runs of the same `(config, seed)` produce the same hash.
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
    // Phase 4c: generate train+eval as one sequence then split. Datasets
    // get the FULL length so their internal structure (regime midpoint,
    // spike positions, noise stream) is determined by the total length,
    // not the train length — this means a (train=50, eval=20) run sees
    // the regime_shift dataset's midpoint at step 35 (= 70/2), with the
    // first 50 steps as train and the last 20 as held-out forecast.
    let total_steps = config.n_steps + config.eval_steps;
    let (full_inputs, full_targets) = config.dataset.generate(seed, total_steps)?;
    let id = config.gan.ssm.input_dim;
    let od = config.gan.ssm.output_dim;
    // Datasets generate 1-D series (input_dim = 1, output_dim = 1 by
    // construction in datasets.rs). Validate here so a future
    // higher-dim dataset doesn't silently misalign.
    if id != 1 || od != 1 {
        return Err(CronosGanError::Unsupported {
            detail: format!(
                "run_experiment currently supports input_dim=output_dim=1 datasets only (got {}, {})",
                id, od
            ),
        });
    }
    let train_input_end = config.n_steps * id;
    let train_target_end = config.n_steps * od;
    let inputs: Vec<f64> = full_inputs[..train_input_end].to_vec();
    let targets: Vec<f64> = full_targets[..train_target_end].to_vec();
    // Eval slice indexes are correct for both id and od.
    let eval_inputs: Vec<f64> = full_inputs[train_input_end..].to_vec();
    let eval_targets: Vec<f64> = full_targets[train_target_end..].to_vec();

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

    // Re-run disagreement on the final-param state for canonical reporting
    // (in-sample / training portion).
    let rollout = gan.rollout_and_disagreement(&inputs, &targets)?;

    // Phase 4c: held-out eval. Run each network's rollout through the
    // train portion to acquire its final hidden state, then continue on
    // the eval inputs from that state. The resulting predictions are
    // compared to the held-out eval_targets — true forecastability
    // metrics, no information leakage.
    let eval = if config.eval_steps > 0 {
        Some(run_eval(&gan, &inputs, &eval_inputs, &eval_targets)?)
    } else {
        None
    };

    // Sweep-summary statistics computed over the FULL trajectory.
    let mean_absolute_gap = mean_absolute_gap(&disagreement_trajectory);
    let max_regime_shift_score = max_regime_shift(&disagreement_trajectory);

    // Replay hash: combines (config bytes, seed, final SSM params, final
    // Liquid params, training-trajectory bit pattern, disagreement-
    // trajectory bit pattern, eval-bit pattern when present).
    let mut parts = Vec::new();
    parts.push(fingerprint_str(IdDomain::CausalClaim, "cronos_gan_experiment_v4c"));
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
    if let Some(e) = &eval {
        parts.push(fingerprint(IdDomain::CausalClaim, &eval_report_bytes(e)));
    }
    let replay_hash = CronosRunId(fingerprint_compose(
        IdDomain::CausalClaim,
        "cronos_experiment_replay_hash_v4c",
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
        eval,
        replay_hash,
        seed,
    })
}

/// Phase 4c eval rollout: each network's state is rewound to the END of
/// the train portion (via a re-run of the train rollout), then the
/// network continues from that state on `eval_inputs`. Predictions are
/// compared against `eval_targets`.
///
/// Returns `EvalReport` with per-network MSE and the
/// SSM-vs-Liquid-vs-target structured disagreement on the held-out window.
fn run_eval(
    gan: &TemporalGan,
    train_inputs: &[f64],
    eval_inputs: &[f64],
    eval_targets: &[f64],
) -> Result<EvalReport, CronosGanError> {
    let cfg = gan.config();
    let od = cfg.ssm.output_dim;
    let n_eval = eval_inputs.len() / cfg.ssm.input_dim;

    // Run the trained SSM through the train portion to acquire its final
    // state, then continue on the eval portion.
    let ssm_train_rollout = gan
        .ssm()
        .rollout(&StateSpaceState::zeros(cfg.ssm.state_dim), train_inputs)?;
    let ssm_final_train_state = ssm_train_rollout.final_state().clone();
    let ssm_eval_rollout = gan.ssm().rollout(&ssm_final_train_state, eval_inputs)?;
    let mut ssm_eval_outputs: Vec<f64> = Vec::with_capacity(n_eval * od);
    for step_out in &ssm_eval_rollout.outputs {
        ssm_eval_outputs.extend_from_slice(step_out);
    }

    let liq_train_rollout = gan
        .liquid()
        .rollout(&LiquidState::zeros(cfg.liquid.state_dim), train_inputs)?;
    let liq_final_train_state = liq_train_rollout.final_state().clone();
    let liq_eval_rollout = gan.liquid().rollout(&liq_final_train_state, eval_inputs)?;
    let mut liq_eval_outputs: Vec<f64> = Vec::with_capacity(n_eval * od);
    for step_out in &liq_eval_rollout.outputs {
        liq_eval_outputs.extend_from_slice(step_out);
    }

    // Eval-window MSE via the existing TemporalLoss enum (already
    // Kahan-summed).
    let ssm_loss = crate::time_series::TemporalLoss::Mse
        .evaluate(&ssm_eval_outputs, eval_targets)?;
    let liquid_loss = crate::time_series::TemporalLoss::Mse
        .evaluate(&liq_eval_outputs, eval_targets)?;

    // Structured disagreement on the held-out window.
    let disagreement = crate::compute_disagreement(
        &ssm_eval_outputs,
        &liq_eval_outputs,
        eval_targets,
        n_eval,
        od,
    )?;

    Ok(EvalReport {
        ssm_loss,
        liquid_loss,
        disagreement,
    })
}

fn eval_report_bytes(e: &EvalReport) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(48);
    bytes.extend_from_slice(&e.ssm_loss.to_bits().to_le_bytes());
    bytes.extend_from_slice(&e.liquid_loss.to_bits().to_le_bytes());
    bytes.extend_from_slice(&e.disagreement.ssm_score.to_bits().to_le_bytes());
    bytes.extend_from_slice(&e.disagreement.liquid_score.to_bits().to_le_bytes());
    bytes.extend_from_slice(&e.disagreement.absolute_gap.to_bits().to_le_bytes());
    bytes.extend_from_slice(&e.disagreement.regime_shift_score.to_bits().to_le_bytes());
    bytes
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
/// eval_steps, n_train_steps)`. Per-dataset learning rates and per-mode
/// challenger weights can be overridden — cells without an override use
/// the fallback `default_lr` / `lambda_disagreement`.
#[derive(Clone, Debug)]
pub struct SweepBaseConfig {
    pub state_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub n_steps: usize,
    /// Phase 4c: held-out evaluation horizon (default 0 → no eval).
    /// Cells with `eval_steps > 0` produce an
    /// [`ExperimentReport::eval`] populated with the forecastability
    /// metrics for that mode.
    pub eval_steps: usize,
    pub n_train_steps: usize,
    /// Fallback challenger weight applied to any
    /// [`TemporalGanMode`] not in `per_mode_lambda`. Ignored for
    /// `Symmetric` (its λ has no effect on the loss).
    pub lambda_disagreement: f64,
    /// Phase 4c: per-mode overrides for the challenger weight.
    /// `Symmetric` should normally stay at `0.0` (or be absent);
    /// `SsmAsGenerator` and `LiquidAsGenerator` may use different λ.
    pub per_mode_lambda: BTreeMap<TemporalGanMode, f64>,
    /// Per-dataset learning-rate overrides, keyed by dataset enum.
    pub per_dataset_lr: BTreeMap<CronosDataset, f64>,
    /// Fallback learning rate for datasets without an override.
    pub default_lr: f64,
}

impl SweepBaseConfig {
    /// Construct with reasonable defaults: `lambda = 0.1`, `default_lr =
    /// 1e-2`, `eval_steps = 0`, no per-mode or per-dataset overrides.
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
            eval_steps: 0,
            n_train_steps,
            lambda_disagreement: 0.1,
            per_mode_lambda: BTreeMap::new(),
            per_dataset_lr: BTreeMap::new(),
            default_lr: 1e-2,
        }
    }

    /// Set the *fallback* λ. Modes present in `per_mode_lambda`
    /// override this.
    pub fn with_lambda_disagreement(mut self, lambda: f64) -> Self {
        self.lambda_disagreement = lambda;
        self
    }

    /// Phase 4c: set a per-mode λ override.
    pub fn with_lambda_for(mut self, mode: TemporalGanMode, lambda: f64) -> Self {
        self.per_mode_lambda.insert(mode, lambda);
        self
    }

    /// Phase 4c: set the held-out eval horizon.
    pub fn with_eval_steps(mut self, eval_steps: usize) -> Self {
        self.eval_steps = eval_steps;
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

    /// Look up the effective λ for `mode` — falls back to
    /// `lambda_disagreement` if no per-mode override is set.
    pub fn lambda_for(&self, mode: TemporalGanMode) -> f64 {
        self.per_mode_lambda
            .get(&mode)
            .copied()
            .unwrap_or(self.lambda_disagreement)
    }

    /// Construct a per-cell [`ExperimentConfig`] for `(dataset, mode)`.
    pub fn experiment_config_for(
        &self,
        dataset: CronosDataset,
        mode: TemporalGanMode,
    ) -> ExperimentConfig {
        let gan = TemporalGanConfig::symmetric(self.state_dim, self.input_dim, self.output_dim)
            .with_mode(mode)
            .with_lambda_disagreement(self.lambda_for(mode));
        let lr = self
            .per_dataset_lr
            .get(&dataset)
            .copied()
            .unwrap_or(self.default_lr);
        ExperimentConfig::new(gan, dataset, self.n_steps)
            .with_n_train_steps(self.n_train_steps)
            .with_lr(lr)
            .with_eval_steps(self.eval_steps)
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.state_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.output_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.n_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.eval_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.n_train_steps as u64).to_le_bytes());
        bytes.extend_from_slice(&self.lambda_disagreement.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.default_lr.to_bits().to_le_bytes());
        // BTreeMap iteration is key-ordered → deterministic across runs
        // and platforms.
        for (mode, lambda) in &self.per_mode_lambda {
            bytes.extend_from_slice(mode.label().as_bytes());
            bytes.extend_from_slice(&lambda.to_bits().to_le_bytes());
        }
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
    ///
    /// **Phase 4c**: when any cell has a populated
    /// [`ExperimentReport::eval`], the table grows three additional
    /// columns covering held-out eval (eval_ssm_loss, eval_liq_loss,
    /// eval_mean_gap). Cells with no eval show `—`.
    pub fn format_table(&self) -> String {
        let any_eval = self.cells.iter().any(|c| c.report.eval.is_some());

        let mut out = String::new();
        if any_eval {
            out.push_str("┌────────────────────────┬──────────────────────┬──────────────┬──────────────┬─────────────┬──────────────┬──────────────┬──────────────┬─────────────┬──────────────────┐\n");
            out.push_str("│ dataset                │ mode                 │ ssm_loss     │ liq_loss     │ mean |gap|  │ max regime   │ eval ssm     │ eval liq     │ eval |gap|  │ replay_hash      │\n");
            out.push_str("├────────────────────────┼──────────────────────┼──────────────┼──────────────┼─────────────┼──────────────┼──────────────┼──────────────┼─────────────┼──────────────────┤\n");
        } else {
            out.push_str("┌────────────────────────┬──────────────────────┬──────────────┬──────────────┬─────────────┬──────────────┬──────────────────┐\n");
            out.push_str("│ dataset                │ mode                 │ ssm_loss     │ liq_loss     │ mean |gap|  │ max regime   │ replay_hash      │\n");
            out.push_str("├────────────────────────┼──────────────────────┼──────────────┼──────────────┼─────────────┼──────────────┼──────────────────┤\n");
        }

        for c in &self.cells {
            if any_eval {
                let (eval_ssm, eval_liq, eval_gap) = match &c.report.eval {
                    Some(e) => (
                        format!("{:>12.6e}", e.ssm_loss),
                        format!("{:>12.6e}", e.liquid_loss),
                        format!("{:>11.4e}", e.disagreement.absolute_gap),
                    ),
                    None => (
                        "           — ".to_string(),
                        "           — ".to_string(),
                        "          — ".to_string(),
                    ),
                };
                out.push_str(&format!(
                    "│ {:<22} │ {:<20} │ {:>12.6e} │ {:>12.6e} │ {:>11.4e} │ {:>12.4e} │ {} │ {} │ {} │ {:>16} │\n",
                    c.dataset.label(),
                    c.mode.label(),
                    c.report.final_loss_ssm,
                    c.report.final_loss_liquid,
                    c.report.mean_absolute_gap,
                    c.report.max_regime_shift_score,
                    eval_ssm,
                    eval_liq,
                    eval_gap,
                    format!("{}", c.report.replay_hash),
                ));
            } else {
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
        }

        if any_eval {
            out.push_str("└────────────────────────┴──────────────────────┴──────────────┴──────────────┴─────────────┴──────────────┴──────────────┴──────────────┴─────────────┴──────────────────┘\n");
        } else {
            out.push_str("└────────────────────────┴──────────────────────┴──────────────┴──────────────┴─────────────┴──────────────┴──────────────────┘\n");
        }
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
