//! Phase 2 autodiff adapter for [`LiquidNetwork`].
//!
//! Implements the [`Trainable`] trait by:
//! 1. Reading Liquid params in canonical order
//!    `[W_h | W_x | bias | W_tau_u | W_tau_h | bias_tau | W_out | bias_out]`.
//! 2. Building a `cjc_ad::GradGraph` that unrolls the rollout, mirroring
//!    [`LiquidNetwork::step`]'s sigmoid-scaled-tau formulation node for
//!    node — this is exactly why Phase 2 refactored the Liquid forward
//!    away from `softplus.clamp(...)` (non-differentiable at the
//!    boundary) toward `tau_min + (tau_max − tau_min)·sigmoid(...)`
//!    (smooth everywhere).
//! 3. Returning `loss_node` and parameter indices for the trainer.
//!
//! Constants entering the graph (`tau_min · ones`, `(tau_max − tau_min)
//! · ones`, `dt · ones`) are added as `input` nodes — they participate in
//! forward computation but do not accumulate gradients. This is the
//! cleanest workaround for `cjc_ad::GradGraph` having no "constant
//! tensor" op; reusing `input` keeps the graph honest about what gets
//! optimized.

use crate::error::CronosGanError;
use crate::liquid::{LiquidNetwork, LiquidParams};
use crate::training::{LossAggregation, RolloutGraph, RolloutLossKind, Trainable};
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

/// Canonical parameter order for [`LiquidNetwork`].
const LIQUID_PARAM_NAMES: &[&str] = &[
    "W_h", "W_x", "bias", "W_tau_u", "W_tau_h", "bias_tau", "W_out", "bias_out",
];

impl Trainable for LiquidNetwork {
    fn n_params(&self) -> usize {
        let cfg = self.config();
        let sd = cfg.state_dim;
        let id = cfg.input_dim;
        let od = cfg.output_dim;
        sd * sd     // W_h
            + sd * id // W_x
            + sd      // bias
            + sd * id // W_tau_u
            + sd * sd // W_tau_h
            + sd      // bias_tau
            + od * sd // W_out
            + od      // bias_out
    }

    fn params_flat(&self) -> Vec<f64> {
        let p = self.params();
        let mut out = Vec::with_capacity(self.n_params());
        out.extend_from_slice(&p.w_h);
        out.extend_from_slice(&p.w_x);
        out.extend_from_slice(&p.bias);
        out.extend_from_slice(&p.w_tau_u);
        out.extend_from_slice(&p.w_tau_h);
        out.extend_from_slice(&p.bias_tau);
        out.extend_from_slice(&p.w_out);
        out.extend_from_slice(&p.bias_out);
        out
    }

    fn set_params_flat(&mut self, params: &[f64]) -> Result<(), CronosGanError> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "LiquidNetwork::set_params_flat: got {} params, expected {}",
                    params.len(),
                    expected
                ),
            });
        }
        let cfg = self.config();
        let sd = cfg.state_dim;
        let id = cfg.input_dim;
        let od = cfg.output_dim;
        let sizes = [
            sd * sd, // W_h
            sd * id, // W_x
            sd,      // bias
            sd * id, // W_tau_u
            sd * sd, // W_tau_h
            sd,      // bias_tau
            od * sd, // W_out
            od,      // bias_out
        ];
        let mut offset = 0;
        let mut take = |n: usize| {
            let s = &params[offset..offset + n];
            offset += n;
            s.to_vec()
        };
        let new_params = LiquidParams {
            w_h: take(sizes[0]),
            w_x: take(sizes[1]),
            bias: take(sizes[2]),
            w_tau_u: take(sizes[3]),
            w_tau_h: take(sizes[4]),
            bias_tau: take(sizes[5]),
            w_out: take(sizes[6]),
            bias_out: take(sizes[7]),
        };
        debug_assert_eq!(offset, expected);
        crate::liquid::set_params_internal(self, new_params);
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
                    "Liquid build_rollout_graph: inputs.len()={} not a positive multiple of input_dim={}",
                    inputs.len(),
                    id
                ),
            });
        }
        let n_steps = inputs.len() / id;
        if targets.len() != n_steps * od {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "Liquid build_rollout_graph: targets.len()={} but expected {}",
                    targets.len(),
                    n_steps * od
                ),
            });
        }
        match loss_kind {
            RolloutLossKind::Mse => {}
        }

        let p = self.params();

        // Parameter nodes in canonical order.
        let wh_n = graph.parameter(make_tensor(&p.w_h, &[sd, sd])?);
        let wx_n = graph.parameter(make_tensor(&p.w_x, &[sd, id])?);
        let bias_n = graph.parameter(make_tensor(&p.bias, &[sd, 1])?);
        let wtu_n = graph.parameter(make_tensor(&p.w_tau_u, &[sd, id])?);
        let wth_n = graph.parameter(make_tensor(&p.w_tau_h, &[sd, sd])?);
        let btau_n = graph.parameter(make_tensor(&p.bias_tau, &[sd, 1])?);
        let wout_n = graph.parameter(make_tensor(&p.w_out, &[od, sd])?);
        let bout_n = graph.parameter(make_tensor(&p.bias_out, &[od, 1])?);

        // Constants as input nodes. tau_min + (tau_max-tau_min)·s with
        // each tensor broadcast to [sd, 1].
        let tau_min_n =
            graph.input(Tensor::from_vec(vec![cfg.tau_min; sd], &[sd, 1]).unwrap());
        let tau_range_n = graph
            .input(Tensor::from_vec(vec![cfg.tau_max - cfg.tau_min; sd], &[sd, 1]).unwrap());
        let dt_n = graph.input(Tensor::from_vec(vec![cfg.dt; sd], &[sd, 1]).unwrap());

        // h_0 = 0
        let mut h_node = graph.input(Tensor::zeros(&[sd, 1]));

        let mut step_losses: Vec<usize> = Vec::with_capacity(n_steps);

        for t in 0..n_steps {
            let u_slice = &inputs[t * id..(t + 1) * id];
            let target_slice = &targets[t * od..(t + 1) * od];
            for (i, &v) in u_slice.iter().enumerate() {
                if !v.is_finite() {
                    return Err(CronosGanError::NonFiniteInput {
                        detail: format!(
                            "Liquid rollout inputs[t={},i={}] is non-finite",
                            t, i
                        ),
                    });
                }
            }
            for (i, &v) in target_slice.iter().enumerate() {
                if !v.is_finite() {
                    return Err(CronosGanError::NonFiniteInput {
                        detail: format!(
                            "Liquid rollout targets[t={},i={}] is non-finite",
                            t, i
                        ),
                    });
                }
            }
            let u_n = graph.input(make_tensor(u_slice, &[id, 1])?);
            let target_n = graph.input(make_tensor(target_slice, &[od, 1])?);

            // y_t = W_out · h_t + bias_out      (matches LiquidNetwork::step)
            let wh_out_n = graph.matmul(wout_n, h_node);
            let y = graph.add(wh_out_n, bout_n);

            // pre = W_h·h + W_x·u + bias
            let whh = graph.matmul(wh_n, h_node);
            let wxu = graph.matmul(wx_n, u_n);
            let pre1 = graph.add(whh, wxu);
            let pre = graph.add(pre1, bias_n);

            // act = tanh(pre)
            let act = graph.tanh_act(pre);

            // s = sigmoid(W_tau_u·u + W_tau_h·h + bias_tau)
            let wtuu = graph.matmul(wtu_n, u_n);
            let wthh = graph.matmul(wth_n, h_node);
            let tau_pre1 = graph.add(wtuu, wthh);
            let tau_pre = graph.add(tau_pre1, btau_n);
            let s = graph.sigmoid(tau_pre);

            // tau = tau_min + tau_range · s     (bounded in (tau_min, tau_max))
            let scaled = graph.mul(s, tau_range_n);
            let tau = graph.add(tau_min_n, scaled);

            // gate = dt / tau
            let gate = graph.div(dt_n, tau);

            // h_new = h + gate · (-h + act)
            let neg_h = graph.neg(h_node);
            let diff_inner = graph.add(neg_h, act);
            let delta = graph.mul(gate, diff_inner);
            let h_new = graph.add(h_node, delta);

            // step loss = mean((y - target)²)
            let err = graph.sub(y, target_n);
            let sq = graph.mul(err, err);
            let step_loss = graph.mean(sq);
            step_losses.push(step_loss);

            h_node = h_new;
        }

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
            param_indices: vec![
                wh_n, wx_n, bias_n, wtu_n, wth_n, btau_n, wout_n, bout_n,
            ],
            param_shapes: vec![
                vec![sd, sd], // W_h
                vec![sd, id], // W_x
                vec![sd, 1],  // bias
                vec![sd, id], // W_tau_u
                vec![sd, sd], // W_tau_h
                vec![sd, 1],  // bias_tau
                vec![od, sd], // W_out
                vec![od, 1],  // bias_out
            ],
        })
    }
}

/// Documented parameter ordering for Liquid, exposed for tooling and tests.
pub fn liquid_param_names() -> &'static [&'static str] {
    LIQUID_PARAM_NAMES
}

fn make_tensor(data: &[f64], shape: &[usize]) -> Result<Tensor, CronosGanError> {
    Tensor::from_vec(data.to_vec(), shape).map_err(|e| CronosGanError::DimensionMismatch {
        detail: format!(
            "Liquid autograd: Tensor::from_vec({:?}) failed: {}",
            shape, e
        ),
    })
}
