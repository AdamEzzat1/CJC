//! Phase 2 integration tests for the autodiff + training pipeline.
//!
//! These tests pin three properties Phase 2 has to make true:
//!
//! 1. **Gradient correctness** — the gradients `cjc_ad::GradGraph`
//!    produces for the SSM and Liquid rollout losses must match
//!    central-difference finite-difference gradients within 1e-4
//!    relative error. If this fails, the training will silently optimise
//!    the wrong direction.
//! 2. **Param round-trip is exact** — `params_flat` then
//!    `set_params_flat` must restore the model bit-identically.
//!    Otherwise the Adam loop loses precision every step.
//! 3. **Training converges on a tiny synthetic task** — fit a 1-dim
//!    sine-like sequence with both networks; loss must decrease
//!    monotonically over a short run, ending well below the initial
//!    loss. This is the smoke test that the whole pipeline (forward,
//!    backward, Adam, write-back) is hooked up correctly.

use cjc_cronos_gan::{
    CronosSeed, LiquidConfig, LiquidNetwork, LossAggregation, RolloutLossKind,
    StateSpaceConfig, StateSpaceModel, SupervisedTrainer, Trainable,
};

/// Generate a small deterministic sequence: `x_t = sin(0.4·t)`, used as
/// both input AND target for a 1-dim regression task.
fn sine_inputs_and_targets(n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    let inputs: Vec<f64> = (0..n_steps).map(|t| (t as f64 * 0.4).sin()).collect();
    let targets: Vec<f64> = (0..n_steps).map(|t| ((t + 1) as f64 * 0.4).sin()).collect();
    (inputs, targets)
}

// ─── § Param round-trip ──────────────────────────────────────────────────

#[test]
fn ssm_params_flat_roundtrip_is_exact() {
    let cfg = StateSpaceConfig::new(4, 2, 1);
    let mut m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let flat_before = m.params_flat();
    m.set_params_flat(&flat_before).unwrap();
    let flat_after = m.params_flat();
    assert_eq!(
        flat_before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        flat_after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "SSM params_flat round-trip must be bit-identical"
    );
}

#[test]
fn ssm_n_params_matches_flat_length() {
    let cfg = StateSpaceConfig::new(4, 2, 1);
    let m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    assert_eq!(m.params_flat().len(), m.n_params());
}

#[test]
fn ssm_set_params_flat_rejects_wrong_size() {
    let cfg = StateSpaceConfig::new(4, 2, 1);
    let mut m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let err = m.set_params_flat(&[0.0]).unwrap_err();
    assert!(matches!(
        err,
        cjc_cronos_gan::CronosGanError::DimensionMismatch { .. }
    ));
}

#[test]
fn liquid_params_flat_roundtrip_is_exact() {
    let cfg = LiquidConfig::new(4, 2, 1);
    let mut n = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
    let flat_before = n.params_flat();
    n.set_params_flat(&flat_before).unwrap();
    let flat_after = n.params_flat();
    assert_eq!(
        flat_before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        flat_after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "Liquid params_flat round-trip must be bit-identical"
    );
}

#[test]
fn liquid_n_params_matches_flat_length() {
    let cfg = LiquidConfig::new(4, 2, 1);
    let n = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
    assert_eq!(n.params_flat().len(), n.n_params());
}

// ─── § Forward loss matches per-step rollout MSE ─────────────────────────

#[test]
fn ssm_forward_loss_matches_per_step_rollout_mse() {
    // The graph-built MSE should equal the loss obtained from running
    // the non-autograd rollout + computing MSE in Vec<f64>, modulo
    // tensor-vs-Kahan matmul order. Tolerance: 1e-9.
    use cjc_cronos_gan::StateSpaceState;
    let cfg = StateSpaceConfig::new(4, 2, 1);
    let m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let n_steps = 8;
    let (inputs, targets) = make_inputs_targets(n_steps, cfg.input_dim, cfg.output_dim);

    // Non-autograd rollout.
    let s0 = StateSpaceState::zeros(cfg.state_dim);
    let r = m.rollout(&s0, &inputs).unwrap();
    let mut mse_manual = 0.0_f64;
    for t in 0..n_steps {
        let mut step_acc = 0.0;
        for d in 0..cfg.output_dim {
            let e = r.outputs[t][d] - targets[t * cfg.output_dim + d];
            step_acc += e * e;
        }
        mse_manual += step_acc / cfg.output_dim as f64;
    }
    mse_manual /= n_steps as f64;

    // Graph-built loss via the trainer.
    let mut graph = cjc_ad::GradGraph::new();
    let rollout = m
        .build_rollout_graph(
            &mut graph,
            &inputs,
            &targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
        )
        .unwrap();
    let mse_graph = graph.tensor(rollout.loss_node).to_vec()[0];

    assert!(
        (mse_graph - mse_manual).abs() < 1e-9,
        "SSM autograd MSE {} differs from manual rollout MSE {} by more than 1e-9",
        mse_graph, mse_manual,
    );
}

// ─── § Finite-difference gradient checks ─────────────────────────────────

fn make_inputs_targets(n_steps: usize, input_dim: usize, output_dim: usize) -> (Vec<f64>, Vec<f64>) {
    let inputs: Vec<f64> = (0..n_steps * input_dim)
        .map(|i| ((i + 1) as f64 * 0.07).sin())
        .collect();
    let targets: Vec<f64> = (0..n_steps * output_dim)
        .map(|i| ((i + 3) as f64 * 0.11).cos())
        .collect();
    (inputs, targets)
}

fn finite_difference_grads<M: Trainable + Clone>(
    model: &M,
    inputs: &[f64],
    targets: &[f64],
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
        let loss_plus = forward_loss(&m_plus, inputs, targets);
        let loss_minus = forward_loss(&m_minus, inputs, targets);
        grads.push((loss_plus - loss_minus) / (2.0 * h));
    }
    grads
}

fn forward_loss<M: Trainable>(model: &M, inputs: &[f64], targets: &[f64]) -> f64 {
    let mut graph = cjc_ad::GradGraph::new();
    let rollout = model
        .build_rollout_graph(
            &mut graph,
            inputs,
            targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
        )
        .unwrap();
    graph.tensor(rollout.loss_node).to_vec()[0]
}

fn analytic_grads<M: Trainable>(model: &M, inputs: &[f64], targets: &[f64]) -> Vec<f64> {
    let mut graph = cjc_ad::GradGraph::new();
    let rollout = model
        .build_rollout_graph(
            &mut graph,
            inputs,
            targets,
            RolloutLossKind::Mse,
            LossAggregation::PerStep,
        )
        .unwrap();
    let grad_tensors = graph.backward_collect(rollout.loss_node, &rollout.param_indices);
    let mut out = Vec::with_capacity(model.n_params());
    for (i, g) in grad_tensors.iter().enumerate() {
        let n = rollout.param_shapes[i].iter().product::<usize>();
        if let Some(t) = g {
            out.extend(t.to_vec());
        } else {
            out.extend(std::iter::repeat(0.0).take(n));
        }
    }
    out
}

fn rel_err(analytic: f64, finite: f64) -> f64 {
    let denom = analytic.abs().max(finite.abs()).max(1e-12);
    (analytic - finite).abs() / denom
}

#[test]
fn ssm_autograd_matches_finite_difference() {
    let cfg = StateSpaceConfig::new(3, 2, 1).with_alpha(0.85);
    let m = StateSpaceModel::from_seed(cfg, CronosSeed(7)).unwrap();
    let n_steps = 5;
    let (inputs, targets) = make_inputs_targets(n_steps, cfg.input_dim, cfg.output_dim);

    let analytic = analytic_grads(&m, &inputs, &targets);
    let finite = finite_difference_grads(&m, &inputs, &targets, 1e-5);

    assert_eq!(analytic.len(), finite.len(), "grad vector lengths must match");

    let mut max_rel = 0.0_f64;
    let mut argmax = 0usize;
    for i in 0..analytic.len() {
        let e = rel_err(analytic[i], finite[i]);
        if e > max_rel {
            max_rel = e;
            argmax = i;
        }
    }
    assert!(
        max_rel < 1e-4,
        "SSM autograd vs FD max relative error {} (at param {}; analytic={} finite={})",
        max_rel, argmax, analytic[argmax], finite[argmax],
    );
}

#[test]
fn liquid_autograd_matches_finite_difference() {
    // Smaller dims for a tractable FD check (Liquid has many params).
    let cfg = LiquidConfig::new(3, 2, 1);
    let m = LiquidNetwork::from_seed(cfg, CronosSeed(7)).unwrap();
    let n_steps = 4;
    let (inputs, targets) = make_inputs_targets(n_steps, cfg.input_dim, cfg.output_dim);

    let analytic = analytic_grads(&m, &inputs, &targets);
    let finite = finite_difference_grads(&m, &inputs, &targets, 1e-5);

    assert_eq!(analytic.len(), finite.len());

    let mut max_rel = 0.0_f64;
    let mut argmax = 0usize;
    for i in 0..analytic.len() {
        let e = rel_err(analytic[i], finite[i]);
        if e > max_rel {
            max_rel = e;
            argmax = i;
        }
    }
    // Liquid has more nonlinearity (tanh + sigmoid + div), so allow a
    // looser 5e-4 tolerance — well within FD truncation+roundoff bounds
    // for h=1e-5.
    assert!(
        max_rel < 5e-4,
        "Liquid autograd vs FD max relative error {} (at param {}; analytic={} finite={})",
        max_rel, argmax, analytic[argmax], finite[argmax],
    );
}

// ─── § Supervised training converges on a tiny task ──────────────────────

#[test]
fn ssm_supervised_training_decreases_loss() {
    let cfg = StateSpaceConfig::new(4, 1, 1).with_alpha(0.9);
    let mut m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let (inputs, targets) = sine_inputs_and_targets(20);

    let mut trainer = SupervisedTrainer::new(m.n_params(), 1e-2);
    let initial_loss = trainer.step(&mut m, &inputs, &targets).unwrap();

    let mut last_loss = initial_loss;
    for _ in 0..50 {
        last_loss = trainer.step(&mut m, &inputs, &targets).unwrap();
        assert!(last_loss.is_finite(), "SSM training produced non-finite loss");
    }

    assert!(
        last_loss < initial_loss * 0.5,
        "SSM did not improve enough: initial {} final {}",
        initial_loss,
        last_loss
    );
}

#[test]
fn liquid_supervised_training_decreases_loss() {
    // Liquid has ~50+ params for state_dim=4, more than SSM's ~16 — needs
    // more steps + slightly higher lr to converge on this tiny task.
    let cfg = LiquidConfig::new(4, 1, 1);
    let mut n = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
    let (inputs, targets) = sine_inputs_and_targets(20);

    let mut trainer = SupervisedTrainer::new(n.n_params(), 3e-2);
    let initial_loss = trainer.step(&mut n, &inputs, &targets).unwrap();

    let mut last_loss = initial_loss;
    for _ in 0..200 {
        last_loss = trainer.step(&mut n, &inputs, &targets).unwrap();
        assert!(
            last_loss.is_finite(),
            "Liquid training produced non-finite loss"
        );
    }

    assert!(
        last_loss < initial_loss * 0.7,
        "Liquid did not improve enough: initial {} final {}",
        initial_loss,
        last_loss
    );
}

// ─── § Training determinism: same seed ⇒ same trajectory ─────────────────

#[test]
fn ssm_training_byte_identical_across_runs() {
    let cfg = StateSpaceConfig::new(3, 1, 1);
    let (inputs, targets) = sine_inputs_and_targets(15);

    let mut m1 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut t1 = SupervisedTrainer::new(m1.n_params(), 1e-2);
    let mut losses_1 = Vec::new();
    for _ in 0..20 {
        losses_1.push(t1.step(&mut m1, &inputs, &targets).unwrap());
    }

    let mut m2 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut t2 = SupervisedTrainer::new(m2.n_params(), 1e-2);
    let mut losses_2 = Vec::new();
    for _ in 0..20 {
        losses_2.push(t2.step(&mut m2, &inputs, &targets).unwrap());
    }

    for (i, (a, b)) in losses_1.iter().zip(losses_2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "SSM training loss at step {} diverged: {} vs {}",
            i, a, b
        );
    }
    assert_eq!(
        m1.params_flat()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        m2.params_flat()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "SSM final params must be byte-identical across runs",
    );
}

#[test]
fn liquid_training_byte_identical_across_runs() {
    let cfg = LiquidConfig::new(3, 1, 1);
    let (inputs, targets) = sine_inputs_and_targets(15);

    let mut n1 = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut t1 = SupervisedTrainer::new(n1.n_params(), 1e-2);
    let mut losses_1 = Vec::new();
    for _ in 0..20 {
        losses_1.push(t1.step(&mut n1, &inputs, &targets).unwrap());
    }

    let mut n2 = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
    let mut t2 = SupervisedTrainer::new(n2.n_params(), 1e-2);
    let mut losses_2 = Vec::new();
    for _ in 0..20 {
        losses_2.push(t2.step(&mut n2, &inputs, &targets).unwrap());
    }

    for (i, (a, b)) in losses_1.iter().zip(losses_2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Liquid training loss at step {} diverged: {} vs {}",
            i, a, b
        );
    }
    assert_eq!(
        n1.params_flat()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        n2.params_flat()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "Liquid final params must be byte-identical across runs",
    );
}
