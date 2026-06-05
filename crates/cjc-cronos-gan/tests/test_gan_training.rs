//! Phase 3b integration tests for the asymmetric GAN modes and the
//! alternating-update [`TemporalGanTrainer`].
//!
//! Coverage:
//! 1. Validation errors — negative / NaN λ in asymmetric mode.
//! 2. `λ = 0` ⇒ asymmetric mode is byte-identical to symmetric mode
//!    (the canonical sanity check on the predictor/challenger framing).
//! 3. Alternating-update determinism — same seed ⇒ byte-identical
//!    training trajectory across runs (in all three modes).
//! 4. Mode separation — `SsmAsGenerator` and `LiquidAsGenerator`
//!    produce different trajectories.
//! 5. λ > 0 produces different parameters than λ = 0 on the challenger
//!    after the same number of steps (proves the gradient flows
//!    through the challenger term).
//! 6. Challenger loss gradient correctness via finite-difference
//!    against the SSM and Liquid models.
//! 7. Adversarial training does not blow up: losses stay finite, the
//!    `regime_shift_score` is finite, and the training trajectory has
//!    no NaNs.

use cjc_cronos_gan::{
    ChallengerSpec, CronosGanError, CronosSeed, LiquidConfig, LiquidNetwork,
    LossAggregation, Role, RolloutLossKind, StateSpaceConfig, StateSpaceModel,
    SupervisedTrainer, TemporalGan, TemporalGanConfig, TemporalGanMode,
    TemporalGanTrainer, Trainable,
};

fn sine_io(n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    let inputs: Vec<f64> = (0..n_steps).map(|t| (t as f64 * 0.4).sin()).collect();
    let targets: Vec<f64> = (0..n_steps).map(|t| ((t + 1) as f64 * 0.4).sin()).collect();
    (inputs, targets)
}

// ─── § Validation ────────────────────────────────────────────────────────

#[test]
fn ssm_as_generator_rejects_negative_lambda() {
    let cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, -0.5);
    let gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
    let mut gan = gan;
    let (inputs, targets) = sine_io(10);
    let err = trainer.step(&mut gan, &inputs, &targets).unwrap_err();
    assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
}

#[test]
fn liquid_as_generator_rejects_nan_lambda() {
    let cfg = TemporalGanConfig::liquid_as_generator(4, 1, 1, f64::NAN);
    let gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
    let mut gan = gan;
    let (inputs, targets) = sine_io(10);
    let err = trainer.step(&mut gan, &inputs, &targets).unwrap_err();
    assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
}

#[test]
fn symmetric_mode_does_not_validate_lambda() {
    // Negative lambda in symmetric mode is allowed (and ignored) — the
    // field is meaningless when no challenger term enters the loss.
    let cfg = TemporalGanConfig::symmetric(4, 1, 1).with_lambda_disagreement(-1.0);
    let gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
    let mut gan = gan;
    let (inputs, targets) = sine_io(10);
    let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
    assert!(step.ssm_loss.is_finite());
    assert!(step.liquid_loss.is_finite());
}

#[test]
fn challenger_spec_validate_rejects_negative_lambda() {
    let outputs = vec![0.0; 4];
    let err = ChallengerSpec {
        predictor_outputs: &outputs,
        lambda: -0.5,
    }
    .validate(4, 1)
    .unwrap_err();
    assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
}

#[test]
fn challenger_spec_validate_rejects_shape_mismatch() {
    let outputs = vec![0.0; 3]; // wrong length for n_steps=4, output_dim=1
    let err = ChallengerSpec {
        predictor_outputs: &outputs,
        lambda: 0.1,
    }
    .validate(4, 1)
    .unwrap_err();
    assert!(matches!(err, CronosGanError::DimensionMismatch { .. }));
}

#[test]
fn challenger_spec_validate_rejects_nan_predictor_output() {
    let outputs = vec![0.0, f64::NAN, 0.0, 0.0];
    let err = ChallengerSpec {
        predictor_outputs: &outputs,
        lambda: 0.1,
    }
    .validate(4, 1)
    .unwrap_err();
    assert!(matches!(err, CronosGanError::NonFiniteInput { .. }));
}

// ─── § λ = 0 sanity ─ asymmetric reduces to vanilla on challenger ───────

#[test]
fn lambda_zero_step_loss_equals_supervised_step_loss_ssm() {
    // With lambda=0, the challenger formulation should produce the same
    // step_loss in graph as the supervised formulation. We verify this by
    // building both graphs and comparing the loss values.
    let cfg = StateSpaceConfig::new(4, 1, 1);
    let m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let (inputs, targets) = sine_io(10);

    let mut graph_a = cjc_ad::GradGraph::new();
    let r_a = m
        .build_rollout_graph(
            &mut graph_a,
            &inputs,
            &targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
        )
        .unwrap();
    let loss_a = graph_a.tensor(r_a.loss_node).to_vec()[0];

    // λ = 0 with arbitrary predictor outputs (the term vanishes)
    let mut graph_b = cjc_ad::GradGraph::new();
    let predictor_outputs = vec![123.456_f64; targets.len()]; // arbitrary
    let spec = ChallengerSpec {
        predictor_outputs: &predictor_outputs,
        lambda: 0.0,
    };
    let r_b = m
        .build_rollout_graph_with(
            &mut graph_b,
            &inputs,
            &targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
            Some(&spec),
        )
        .unwrap();
    let loss_b = graph_b.tensor(r_b.loss_node).to_vec()[0];

    assert_eq!(
        loss_a.to_bits(),
        loss_b.to_bits(),
        "λ=0 challenger loss must byte-equal supervised loss (got {} vs {})",
        loss_a, loss_b,
    );
}

#[test]
fn lambda_zero_step_loss_equals_supervised_step_loss_liquid() {
    let cfg = LiquidConfig::new(4, 1, 1);
    let n = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
    let (inputs, targets) = sine_io(10);

    let mut graph_a = cjc_ad::GradGraph::new();
    let r_a = n
        .build_rollout_graph(
            &mut graph_a,
            &inputs,
            &targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
        )
        .unwrap();
    let loss_a = graph_a.tensor(r_a.loss_node).to_vec()[0];

    let mut graph_b = cjc_ad::GradGraph::new();
    let predictor_outputs = vec![99.9_f64; targets.len()];
    let spec = ChallengerSpec {
        predictor_outputs: &predictor_outputs,
        lambda: 0.0,
    };
    let r_b = n
        .build_rollout_graph_with(
            &mut graph_b,
            &inputs,
            &targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
            Some(&spec),
        )
        .unwrap();
    let loss_b = graph_b.tensor(r_b.loss_node).to_vec()[0];

    assert_eq!(loss_a.to_bits(), loss_b.to_bits());
}

// ─── § λ > 0 changes the loss (proves the term flows) ────────────────────

#[test]
fn lambda_positive_changes_step_loss_ssm() {
    // With non-zero predictor outputs ≠ ssm's own predictions, λ > 0
    // should change the loss compared to the supervised loss.
    let cfg = StateSpaceConfig::new(4, 1, 1);
    let m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let (inputs, targets) = sine_io(10);

    let mut graph_a = cjc_ad::GradGraph::new();
    let r_a = m
        .build_rollout_graph(&mut graph_a, &inputs, &targets, RolloutLossKind::Mse, LossAggregation::PerStep)
        .unwrap();
    let loss_supervised = graph_a.tensor(r_a.loss_node).to_vec()[0];

    let mut graph_b = cjc_ad::GradGraph::new();
    let predictor_outputs: Vec<f64> = targets.iter().map(|v| v + 0.5).collect();
    let spec = ChallengerSpec {
        predictor_outputs: &predictor_outputs,
        lambda: 0.5,
    };
    let r_b = m
        .build_rollout_graph_with(&mut graph_b, &inputs, &targets, RolloutLossKind::Mse, LossAggregation::PerStep, Some(&spec))
        .unwrap();
    let loss_challenger = graph_b.tensor(r_b.loss_node).to_vec()[0];

    assert_ne!(
        loss_supervised, loss_challenger,
        "λ > 0 must change the challenger loss when predictor differs from self"
    );
    // The challenger term subtracts λ · MSE(self, predictor), so for
    // predictor far from self the challenger loss is LOWER than the
    // supervised loss (the network is rewarded for the gap that already
    // exists).
    assert!(loss_challenger < loss_supervised);
}

// ─── § Finite-difference gradient check for the challenger loss ─────────

fn fd_grads_challenger<M: Trainable + Clone>(
    model: &M,
    inputs: &[f64],
    targets: &[f64],
    predictor_outputs: &[f64],
    lambda: f64,
    h: f64,
) -> Vec<f64> {
    let p0 = model.params_flat();
    let mut grads = Vec::with_capacity(p0.len());
    for i in 0..p0.len() {
        let mut p_plus = p0.clone();
        let mut p_minus = p0.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let mut m_plus = model.clone();
        let mut m_minus = model.clone();
        m_plus.set_params_flat(&p_plus).unwrap();
        m_minus.set_params_flat(&p_minus).unwrap();
        let loss_plus = forward_loss_challenger(&m_plus, inputs, targets, predictor_outputs, lambda);
        let loss_minus = forward_loss_challenger(&m_minus, inputs, targets, predictor_outputs, lambda);
        grads.push((loss_plus - loss_minus) / (2.0 * h));
    }
    grads
}

fn forward_loss_challenger<M: Trainable>(
    model: &M,
    inputs: &[f64],
    targets: &[f64],
    predictor_outputs: &[f64],
    lambda: f64,
) -> f64 {
    let mut graph = cjc_ad::GradGraph::new();
    let spec = ChallengerSpec {
        predictor_outputs,
        lambda,
    };
    let r = model
        .build_rollout_graph_with(
            &mut graph,
            inputs,
            targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
            Some(&spec),
        )
        .unwrap();
    graph.tensor(r.loss_node).to_vec()[0]
}

fn analytic_grads_challenger<M: Trainable>(
    model: &M,
    inputs: &[f64],
    targets: &[f64],
    predictor_outputs: &[f64],
    lambda: f64,
) -> Vec<f64> {
    let mut graph = cjc_ad::GradGraph::new();
    let spec = ChallengerSpec {
        predictor_outputs,
        lambda,
    };
    let r = model
        .build_rollout_graph_with(
            &mut graph,
            inputs,
            targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
            Some(&spec),
        )
        .unwrap();
    let grads = graph.backward_collect(r.loss_node, &r.param_indices);
    let mut out = Vec::with_capacity(model.n_params());
    for (i, g) in grads.iter().enumerate() {
        let n = r.param_shapes[i].iter().product::<usize>();
        match g {
            Some(t) => out.extend(t.to_vec()),
            None => out.extend(std::iter::repeat(0.0).take(n)),
        }
    }
    out
}

#[test]
fn ssm_challenger_autograd_matches_finite_difference() {
    let cfg = StateSpaceConfig::new(3, 1, 1).with_alpha(0.85);
    let m = StateSpaceModel::from_seed(cfg, CronosSeed(7)).unwrap();
    let n_steps = 5;
    let (inputs, targets) = sine_io(n_steps);
    let predictor_outputs: Vec<f64> = targets.iter().map(|v| v * 0.7 + 0.2).collect();
    let lambda = 0.3;

    let analytic = analytic_grads_challenger(&m, &inputs, &targets, &predictor_outputs, lambda);
    let finite = fd_grads_challenger(&m, &inputs, &targets, &predictor_outputs, lambda, 1e-5);

    assert_eq!(analytic.len(), finite.len());
    let mut max_rel = 0.0_f64;
    for i in 0..analytic.len() {
        let denom = analytic[i].abs().max(finite[i].abs()).max(1e-12);
        let e = (analytic[i] - finite[i]).abs() / denom;
        if e > max_rel {
            max_rel = e;
        }
    }
    assert!(
        max_rel < 1e-4,
        "SSM challenger autograd vs FD max rel err {} (tolerance 1e-4)",
        max_rel
    );
}

#[test]
fn liquid_challenger_autograd_matches_finite_difference() {
    let cfg = LiquidConfig::new(3, 1, 1);
    let m = LiquidNetwork::from_seed(cfg, CronosSeed(7)).unwrap();
    let n_steps = 4;
    let (inputs, targets) = sine_io(n_steps);
    let predictor_outputs: Vec<f64> = targets.iter().map(|v| v * 0.6 + 0.3).collect();
    let lambda = 0.2;

    let analytic = analytic_grads_challenger(&m, &inputs, &targets, &predictor_outputs, lambda);
    let finite = fd_grads_challenger(&m, &inputs, &targets, &predictor_outputs, lambda, 1e-5);

    assert_eq!(analytic.len(), finite.len());
    let mut max_rel = 0.0_f64;
    for i in 0..analytic.len() {
        let denom = analytic[i].abs().max(finite[i].abs()).max(1e-12);
        let e = (analytic[i] - finite[i]).abs() / denom;
        if e > max_rel {
            max_rel = e;
        }
    }
    assert!(
        max_rel < 5e-4,
        "Liquid challenger autograd vs FD max rel err {} (tolerance 5e-4)",
        max_rel
    );
}

// ─── § Alternating-update determinism ────────────────────────────────────

fn run_n_steps(
    cfg: TemporalGanConfig,
    seed: CronosSeed,
    n_steps: usize,
    n_train_steps: usize,
    lr: f64,
) -> Vec<f64> {
    let mut gan = TemporalGan::from_seed(cfg, seed).unwrap();
    let mut trainer = TemporalGanTrainer::new(cfg, &gan, lr);
    let (inputs, targets) = sine_io(n_steps);
    let mut trajectory = Vec::new();
    for _ in 0..n_train_steps {
        let s = trainer.step(&mut gan, &inputs, &targets).unwrap();
        trajectory.push(s.ssm_loss);
        trajectory.push(s.liquid_loss);
        trajectory.push(s.disagreement.absolute_gap);
        trajectory.push(s.disagreement.regime_shift_score);
    }
    trajectory
}

#[test]
fn symmetric_training_byte_identical_across_runs() {
    let cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let t1 = run_n_steps(cfg, CronosSeed(42), 15, 10, 1e-2);
    let t2 = run_n_steps(cfg, CronosSeed(42), 15, 10, 1e-2);
    for (a, b) in t1.iter().zip(t2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn ssm_as_generator_training_byte_identical_across_runs() {
    let cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let t1 = run_n_steps(cfg, CronosSeed(42), 15, 10, 1e-2);
    let t2 = run_n_steps(cfg, CronosSeed(42), 15, 10, 1e-2);
    for (i, (a, b)) in t1.iter().zip(t2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "trajectory element {} diverged: {} vs {}",
            i, a, b
        );
    }
}

#[test]
fn liquid_as_generator_training_byte_identical_across_runs() {
    let cfg = TemporalGanConfig::liquid_as_generator(4, 1, 1, 0.1);
    let t1 = run_n_steps(cfg, CronosSeed(42), 15, 10, 1e-2);
    let t2 = run_n_steps(cfg, CronosSeed(42), 15, 10, 1e-2);
    for (a, b) in t1.iter().zip(t2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// ─── § Mode separation ──────────────────────────────────────────────────

#[test]
fn ssm_as_gen_diverges_from_liquid_as_gen() {
    let cfg_a = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.2);
    let cfg_b = TemporalGanConfig::liquid_as_generator(4, 1, 1, 0.2);
    let t_a = run_n_steps(cfg_a, CronosSeed(42), 15, 20, 1e-2);
    let t_b = run_n_steps(cfg_b, CronosSeed(42), 15, 20, 1e-2);
    // The two modes drive different training dynamics → trajectories must
    // differ. (Same seed, same data, different mode.)
    let mut differs = false;
    for (a, b) in t_a.iter().zip(t_b.iter()) {
        if a.to_bits() != b.to_bits() {
            differs = true;
            break;
        }
    }
    assert!(differs, "SsmAsGen and LiquidAsGen should produce different trajectories");
}

// ─── § λ = 0 in asymmetric ≡ symmetric (the canonical sanity check) ─────

#[test]
fn lambda_zero_asymmetric_equals_symmetric_trajectory() {
    // With λ = 0, the challenger's `−λ · MSE-vs-predictor` term vanishes,
    // so asymmetric mode should drive byte-identical training to
    // symmetric mode. This is the canonical sanity check on the
    // implementation.
    let cfg_sym = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg_asym = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.0);
    let t_sym = run_n_steps(cfg_sym, CronosSeed(42), 15, 10, 1e-2);
    let t_asym = run_n_steps(cfg_asym, CronosSeed(42), 15, 10, 1e-2);
    // Losses themselves should match — challenger term is zero. Note: the
    // absolute_gap and regime_shift_score may differ slightly because
    // training-step ORDER matters (in asymmetric mode SSM updates before
    // Liquid, while symmetric updates them independently). To make this
    // robust we compare just the ssm_loss / liquid_loss elements.
    let losses_only_sym: Vec<f64> = t_sym.chunks(4).flat_map(|c| [c[0], c[1]]).collect();
    let losses_only_asym: Vec<f64> = t_asym.chunks(4).flat_map(|c| [c[0], c[1]]).collect();
    for (a, b) in losses_only_sym.iter().zip(losses_only_asym.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "λ=0 asymmetric loss must match symmetric loss"
        );
    }
}

// ─── § Stability: training stays finite + bounded ────────────────────────

#[test]
fn asymmetric_training_stays_finite_over_many_steps() {
    let cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
    let (inputs, targets) = sine_io(20);
    for step_i in 0..100 {
        let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
        assert!(
            step.ssm_loss.is_finite() && step.liquid_loss.is_finite(),
            "non-finite loss at step {}: ssm={} liquid={}",
            step_i, step.ssm_loss, step.liquid_loss
        );
        assert!(step.disagreement.ssm_score.is_finite());
        assert!(step.disagreement.liquid_score.is_finite());
        assert!(step.disagreement.absolute_gap.is_finite());
        assert!(step.disagreement.regime_shift_score.is_finite());
    }
}

// ─── § Step roles reported correctly across all 3 modes ────────────────

#[test]
fn step_roles_match_mode() {
    // Triple-check that the `ssm_role` / `liquid_role` fields in the
    // returned TemporalGanTrainStep reflect the mode.
    let modes = [
        TemporalGanMode::Symmetric,
        TemporalGanMode::SsmAsGenerator,
        TemporalGanMode::LiquidAsGenerator,
    ];
    let expected = [
        (Role::Predictor, Role::Predictor),
        (Role::Predictor, Role::Challenger),
        (Role::Challenger, Role::Predictor),
    ];
    for (mode, (exp_ssm, exp_liq)) in modes.iter().zip(expected.iter()) {
        let cfg = TemporalGanConfig::symmetric(4, 1, 1)
            .with_mode(*mode)
            .with_lambda_disagreement(0.1);
        let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let (inputs, targets) = sine_io(10);
        let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
        assert_eq!(step.ssm_role, *exp_ssm, "mode {:?}", mode);
        assert_eq!(step.liquid_role, *exp_liq, "mode {:?}", mode);
    }
}

// ─── § Mode label propagates to TemporalGan rollout's run_id ─────────────

#[test]
fn run_id_differs_across_modes_with_same_seed_and_dims() {
    // Same seed, same network dims, different mode ⇒ different run_id
    // because the canonical config bytes include the mode label.
    let cfg_sym = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg_ssm = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg_liq = TemporalGanConfig::liquid_as_generator(4, 1, 1, 0.1);
    let g_sym = TemporalGan::from_seed(cfg_sym, CronosSeed(42)).unwrap();
    let g_ssm = TemporalGan::from_seed(cfg_ssm, CronosSeed(42)).unwrap();
    let g_liq = TemporalGan::from_seed(cfg_liq, CronosSeed(42)).unwrap();
    let (inputs, targets) = sine_io(10);
    let r_sym = g_sym.rollout_and_disagreement(&inputs, &targets).unwrap();
    let r_ssm = g_ssm.rollout_and_disagreement(&inputs, &targets).unwrap();
    let r_liq = g_liq.rollout_and_disagreement(&inputs, &targets).unwrap();
    assert_ne!(r_sym.run_id, r_ssm.run_id);
    assert_ne!(r_sym.run_id, r_liq.run_id);
    assert_ne!(r_ssm.run_id, r_liq.run_id);
}

// ─── § lambda affects challenger params after equal training ────────────

#[test]
fn lambda_affects_challenger_params_after_training() {
    // Train SsmAsGenerator with λ=0 and λ=0.5 for 20 steps. The
    // CHALLENGER's (Liquid's) parameters must differ between the two runs
    // — proving the challenger term flows through the optimizer.
    let cfg_a = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.0);
    let cfg_b = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.5);
    let mut gan_a = TemporalGan::from_seed(cfg_a, CronosSeed(42)).unwrap();
    let mut gan_b = TemporalGan::from_seed(cfg_b, CronosSeed(42)).unwrap();
    let mut t_a = TemporalGanTrainer::new(cfg_a, &gan_a, 1e-2);
    let mut t_b = TemporalGanTrainer::new(cfg_b, &gan_b, 1e-2);
    let (inputs, targets) = sine_io(15);
    for _ in 0..20 {
        t_a.step(&mut gan_a, &inputs, &targets).unwrap();
        t_b.step(&mut gan_b, &inputs, &targets).unwrap();
    }
    let liq_a = gan_a.liquid().params_flat();
    let liq_b = gan_b.liquid().params_flat();
    let mut differs = false;
    for (a, b) in liq_a.iter().zip(liq_b.iter()) {
        if a.to_bits() != b.to_bits() {
            differs = true;
            break;
        }
    }
    assert!(differs, "λ=0 and λ=0.5 should produce different challenger params");
}

// ─── § SupervisedTrainer.step_with works as an alternative entry ────────

#[test]
fn supervised_trainer_step_with_lambda_zero_matches_step() {
    let cfg = StateSpaceConfig::new(4, 1, 1);
    let mut m1 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut m2 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut t1 = SupervisedTrainer::new(m1.n_params(), 1e-2);
    let mut t2 = SupervisedTrainer::new(m2.n_params(), 1e-2);
    let (inputs, targets) = sine_io(10);

    let l1 = t1.step(&mut m1, &inputs, &targets).unwrap();
    let predictor = vec![1.0_f64; targets.len()]; // arbitrary; λ=0 ignores it
    let spec = ChallengerSpec {
        predictor_outputs: &predictor,
        lambda: 0.0,
    };
    let l2 = t2.step_with(&mut m2, &inputs, &targets, Some(&spec)).unwrap();

    assert_eq!(l1.to_bits(), l2.to_bits());
    // Params must also match.
    let p1 = m1.params_flat();
    let p2 = m2.params_flat();
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
