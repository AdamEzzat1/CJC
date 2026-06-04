//! Phase 2 autodiff adapter for [`StateSpaceModel`].
//!
//! Implements the [`Trainable`] trait by:
//! 1. Reading SSM params (A, B, C — D is held at zero by Phase 1 design
//!    convention and is therefore *not* trainable) in canonical order
//!    `[A | B | C]`.
//! 2. Building a single `cjc_ad::GradGraph` that unrolls the rollout over
//!    `n_steps`, producing the same forward outputs as
//!    [`StateSpaceModel::rollout`] (modulo bit-level differences in the
//!    underlying tensor matmul order vs. the `Vec<f64>`-based
//!    `matvec_kahan`).
//! 3. Returning `loss_node` and the parameter node indices so the trainer
//!    can backprop.
//!
//! Tensors used in the graph are 2-D throughout (`[rows, 1]` for vectors)
//! because `cjc_ad::GradGraph::matmul` requires both operands to be 2-D.
//! The SSM uses no transcendental ops, no elementwise nonlinearities —
//! every node is `parameter`, `input`, `matmul`, `add`, `sub`, `mul`, or
//! `mean` — so the gradient flow is dense and clean.

use crate::error::CronosGanError;
use crate::ssm::{StateSpaceModel, StateSpaceParams};
use crate::training::{LossAggregation, RolloutGraph, RolloutLossKind, Trainable};
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

/// Canonical parameter order for [`StateSpaceModel`].
///
/// Trainable params: `A`, `B`, `C`. `D` is held at zero by Phase 1
/// convention (no direct feedthrough) and is therefore *not* in the
/// trainable set.
const SSM_PARAM_NAMES: &[&str] = &["A", "B", "C"];

impl Trainable for StateSpaceModel {
    fn n_params(&self) -> usize {
        let cfg = self.config();
        cfg.state_dim * cfg.state_dim       // A
            + cfg.state_dim * cfg.input_dim // B
            + cfg.output_dim * cfg.state_dim // C
    }

    fn params_flat(&self) -> Vec<f64> {
        let p = self.params();
        let mut out = Vec::with_capacity(self.n_params());
        out.extend_from_slice(&p.a);
        out.extend_from_slice(&p.b);
        out.extend_from_slice(&p.c);
        out
    }

    fn set_params_flat(&mut self, params: &[f64]) -> Result<(), CronosGanError> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "StateSpaceModel::set_params_flat: got {} params, expected {}",
                    params.len(),
                    expected
                ),
            });
        }
        let cfg = self.config();
        let sd = cfg.state_dim;
        let id = cfg.input_dim;
        let od = cfg.output_dim;
        let n_a = sd * sd;
        let n_b = sd * id;
        let n_c = od * sd;
        let new_params = StateSpaceParams {
            a: params[0..n_a].to_vec(),
            b: params[n_a..n_a + n_b].to_vec(),
            c: params[n_a + n_b..n_a + n_b + n_c].to_vec(),
            d: self.params().d.clone(), // unchanged (zeros)
        };
        // Replace the params field — uses a small helper to avoid leaking
        // a public mutator that would let callers bypass canonical order.
        replace_params(self, new_params);
        Ok(())
    }

    fn build_rollout_graph(
        &self,
        graph: &mut GradGraph,
        inputs: &[f64],
        targets: &[f64],
        loss_kind: RolloutLossKind,
        aggregation: LossAggregation,
    ) -> Result<RolloutGraph, CronosGanError> {
        let cfg = self.config();
        let sd = cfg.state_dim;
        let id = cfg.input_dim;
        let od = cfg.output_dim;

        if inputs.is_empty() || inputs.len() % id != 0 {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "SSM build_rollout_graph: inputs.len()={} not a positive multiple of input_dim={}",
                    inputs.len(),
                    id
                ),
            });
        }
        let n_steps = inputs.len() / id;
        if targets.len() != n_steps * od {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "SSM build_rollout_graph: targets.len()={} but expected {} (n_steps={}, output_dim={})",
                    targets.len(),
                    n_steps * od,
                    n_steps,
                    od
                ),
            });
        }
        // Phase 2 forward-graph supports MSE only — caller is responsible
        // for picking that variant.
        match loss_kind {
            RolloutLossKind::Mse => {}
        }

        let p = self.params();

        // Parameter nodes (canonical order A, B, C).
        let a_node = graph.parameter(make_tensor(&p.a, &[sd, sd])?);
        let b_node = graph.parameter(make_tensor(&p.b, &[sd, id])?);
        let c_node = graph.parameter(make_tensor(&p.c, &[od, sd])?);

        // Initial state x_0 = 0 as a non-trainable input.
        let mut state_node = graph.input(Tensor::zeros(&[sd, 1]));

        // Build per-step graph, accumulate step losses.
        let mut step_losses: Vec<usize> = Vec::with_capacity(n_steps);
        for t in 0..n_steps {
            let u_slice = &inputs[t * id..(t + 1) * id];
            let target_slice = &targets[t * od..(t + 1) * od];
            for (i, &v) in u_slice.iter().enumerate() {
                if !v.is_finite() {
                    return Err(CronosGanError::NonFiniteInput {
                        detail: format!("SSM rollout inputs[t={},i={}] is non-finite", t, i),
                    });
                }
            }
            for (i, &v) in target_slice.iter().enumerate() {
                if !v.is_finite() {
                    return Err(CronosGanError::NonFiniteInput {
                        detail: format!("SSM rollout targets[t={},i={}] is non-finite", t, i),
                    });
                }
            }
            let u_node = graph.input(make_tensor(u_slice, &[id, 1])?);
            let target_node = graph.input(make_tensor(target_slice, &[od, 1])?);

            // x_{t+1} = A x_t + B u_t
            let ax = graph.matmul(a_node, state_node);
            let bu = graph.matmul(b_node, u_node);
            let next_state = graph.add(ax, bu);

            // y_t = C x_t   (D = 0 omitted)
            let y = graph.matmul(c_node, state_node);

            // step_loss = mean((y - target)²)
            let diff = graph.sub(y, target_node);
            let sq = graph.mul(diff, diff);
            let step_loss = graph.mean(sq);
            step_losses.push(step_loss);

            state_node = next_state;
        }

        // Aggregate step losses.
        let total = step_losses
            .iter()
            .copied()
            .reduce(|a, b| graph.add(a, b))
            .expect("n_steps >= 1 by earlier validation");
        let loss_node = match aggregation {
            LossAggregation::PerSequence => total,
            LossAggregation::PerStep => graph.scalar_mul(total, 1.0 / n_steps as f64),
        };

        Ok(RolloutGraph {
            loss_node,
            param_indices: vec![a_node, b_node, c_node],
            param_shapes: vec![vec![sd, sd], vec![sd, id], vec![od, sd]],
        })
    }
}

/// Documented parameter ordering for SSM, exposed for tooling and tests
/// that need to introspect which segment of a flat parameter vector
/// corresponds to which matrix.
pub fn ssm_param_names() -> &'static [&'static str] {
    SSM_PARAM_NAMES
}

/// Helper that builds a `cjc_runtime::Tensor` from a `&[f64]` + shape,
/// translating tensor-creation errors into our crate's error type.
fn make_tensor(data: &[f64], shape: &[usize]) -> Result<Tensor, CronosGanError> {
    Tensor::from_vec(data.to_vec(), shape).map_err(|e| CronosGanError::DimensionMismatch {
        detail: format!(
            "SSM autograd: Tensor::from_vec({:?}) failed: {}",
            shape, e
        ),
    })
}

/// Replace the parameters of `model` with `new_params`.
///
/// `StateSpaceModel` keeps `params: StateSpaceParams` private. Rather than
/// expose a public setter that would let callers bypass the canonical
/// parameter order Phase 2 depends on, we use an internal helper here
/// gated by `pub(crate)`. The Trainable impl is the only legitimate
/// caller.
pub(crate) fn replace_params(model: &mut StateSpaceModel, new_params: StateSpaceParams) {
    crate::ssm::set_params_internal(model, new_params);
}
