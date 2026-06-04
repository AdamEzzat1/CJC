//! Phase 2: supervised training infrastructure.
//!
//! Defines [`Trainable`] — the trait every cjc-cronos-gan model implements
//! so that [`SupervisedTrainer`] can drive end-to-end backprop-through-time
//! (BPTT) training. The trait deliberately stays minimal: read params as a
//! flat `Vec<f64>`, write them back, and build a `cjc_ad::GradGraph` for
//! one rollout. Everything else — Adam state, gradient flattening, loss
//! aggregation — lives in this module.
//!
//! ## Determinism contract (Phase 2 extensions)
//!
//! 1. **Param-flattening order is canonical and stable**. Each model
//!    documents the exact concatenation order of its parameter tensors;
//!    [`params_flat`](Trainable::params_flat) /
//!    [`set_params_flat`](Trainable::set_params_flat) must match
//!    [`build_rollout_graph`](Trainable::build_rollout_graph)'s
//!    `param_indices` order *exactly*. Mismatch → Adam will update the
//!    wrong parameter and bit-identical replay across runs breaks.
//! 2. **Constant inputs are bit-stable**. Tau bounds, dt, and zero states
//!    enter the graph as `Tensor::from_vec(vec![c; n], …)`. Same `c` and
//!    `n` ⇒ same bytes.
//! 3. **Adam uses cjc_runtime::ml::adam_step** which is the same
//!    deterministic implementation chess RL v2 trained against — already
//!    verified bit-identical on the 9.790915694115341 weight-hash gate.
//! 4. **Backward determinism**: after the cjc-ad A1 patch lands on master,
//!    gradient accumulation in `backward_collect` is Kahan-compensated.
//!    Currently `cjc-cronos-gan` builds against pre-A1 master; the
//!    structural correctness of the gradients holds either way, but
//!    cross-platform byte-identity strengthens once A1 merges.

use crate::error::CronosGanError;
use cjc_ad::GradGraph;
use cjc_runtime::ml::{adam_step, AdamState};

/// Per-step output loss kind. The Phase 2 implementation supports MSE
/// natively in-graph (the autodiff path); MAE and Huber are exposed via
/// [`crate::TemporalLoss`] for inference-time evaluation but are not
/// wired through the graph yet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RolloutLossKind {
    /// Mean squared error.
    Mse,
}

/// How to aggregate per-step losses into a single scalar trained against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossAggregation {
    /// Mean over all timesteps (canonical for length-invariant training).
    PerStep,
    /// Sum over all timesteps (useful when sequence length itself encodes
    /// importance).
    PerSequence,
}

/// Descriptor returned by [`Trainable::build_rollout_graph`].
///
/// `param_indices[i]` is the `GradGraph` node index of the i-th parameter
/// in canonical order; `param_shapes[i]` is its tensor shape. Both must
/// agree with the model's [`Trainable::params_flat`] enumeration so the
/// flattened gradient order matches the flattened parameter order.
#[derive(Clone, Debug)]
pub struct RolloutGraph {
    pub loss_node: usize,
    pub param_indices: Vec<usize>,
    pub param_shapes: Vec<Vec<usize>>,
}

/// A temporal model that can be trained via supervised BPTT.
pub trait Trainable {
    /// Total scalar parameter count across all trainable tensors.
    fn n_params(&self) -> usize;

    /// Read parameters as a flat `Vec<f64>` in canonical concatenation
    /// order. Same order every call.
    fn params_flat(&self) -> Vec<f64>;

    /// Write parameters from a flat `Vec<f64>` in the same canonical
    /// order. Returns `DimensionMismatch` if `params.len() != n_params()`.
    fn set_params_flat(&mut self, params: &[f64]) -> Result<(), CronosGanError>;

    /// Build a BPTT graph for one rollout. Inputs are row-major
    /// `[n_steps, input_dim]`; targets are row-major
    /// `[n_steps, output_dim]`.
    fn build_rollout_graph(
        &self,
        graph: &mut GradGraph,
        inputs: &[f64],
        targets: &[f64],
        loss_kind: RolloutLossKind,
        aggregation: LossAggregation,
    ) -> Result<RolloutGraph, CronosGanError>;
}

/// Supervised trainer with Adam.
///
/// Each call to [`step`](Self::step) builds a fresh `GradGraph` for the
/// supplied rollout, computes the loss, runs backward, flattens the
/// gradient tensors, runs one Adam update, and writes the new parameters
/// back. The trainer is **stateless across calls** except for the Adam
/// momentum buffers — replaying the same `(model, inputs, targets,
/// initial Adam state)` sequence produces byte-identical parameter
/// trajectories.
pub struct SupervisedTrainer {
    adam: AdamState,
    loss_kind: RolloutLossKind,
    aggregation: LossAggregation,
}

impl SupervisedTrainer {
    /// Construct with `n_params` total scalar parameters (must equal the
    /// trained model's [`Trainable::n_params`]) and Adam learning rate.
    /// Defaults: `loss_kind = Mse`, `aggregation = PerStep`,
    /// `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8` (per Adam canonical
    /// values).
    pub fn new(n_params: usize, lr: f64) -> Self {
        Self {
            adam: AdamState::new(n_params, lr),
            loss_kind: RolloutLossKind::Mse,
            aggregation: LossAggregation::PerStep,
        }
    }

    pub fn with_aggregation(mut self, agg: LossAggregation) -> Self {
        self.aggregation = agg;
        self
    }

    /// Number of training steps completed (Adam iteration counter).
    pub fn step_count(&self) -> u64 {
        self.adam.t
    }

    /// Run one supervised BPTT step. Returns the scalar loss BEFORE the
    /// Adam update (so the user sees the loss the gradient was computed
    /// against, not the loss after the step which would require a second
    /// forward).
    pub fn step<M: Trainable>(
        &mut self,
        model: &mut M,
        inputs: &[f64],
        targets: &[f64],
    ) -> Result<f64, CronosGanError> {
        debug_assert_eq!(
            model.n_params(),
            self.adam.n_params(),
            "Trainable::n_params must match SupervisedTrainer's AdamState size"
        );

        let mut graph = GradGraph::new();
        let rollout = model.build_rollout_graph(
            &mut graph,
            inputs,
            targets,
            self.loss_kind,
            self.aggregation,
        )?;

        // Forward pass is eager in cjc-ad — graph.tensor(loss_node) is the
        // loss already-computed at construction time.
        let loss_value = graph.tensor(rollout.loss_node).to_vec()[0];
        if !loss_value.is_finite() {
            return Err(CronosGanError::NonFiniteInput {
                detail: format!("SupervisedTrainer::step: loss = {} is non-finite", loss_value),
            });
        }

        // Backward + per-param gradient collection.
        let grad_tensors = graph.backward_collect(rollout.loss_node, &rollout.param_indices);

        // Flatten gradients in the canonical order of the model.
        let mut grads_flat: Vec<f64> = Vec::with_capacity(self.adam.n_params());
        for (i, g_opt) in grad_tensors.iter().enumerate() {
            let expected = rollout.param_shapes[i].iter().product::<usize>();
            match g_opt {
                Some(g) => {
                    let data = g.to_vec();
                    if data.len() != expected {
                        return Err(CronosGanError::DimensionMismatch {
                            detail: format!(
                                "SupervisedTrainer::step: param {} gradient has {} elements, expected {}",
                                i,
                                data.len(),
                                expected
                            ),
                        });
                    }
                    grads_flat.extend(data);
                }
                None => {
                    grads_flat.extend(std::iter::repeat(0.0).take(expected));
                }
            }
        }

        // Adam update.
        let mut params_flat = model.params_flat();
        adam_step(&mut params_flat, &grads_flat, &mut self.adam);
        model.set_params_flat(&params_flat)?;

        Ok(loss_value)
    }
}
