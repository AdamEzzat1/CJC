//! Phase 2d — GradGraph-based training for the cluster failure head.
//!
//! Replaces the Phase 2 count-based calibration with **actual gradient
//! descent** through [`cjc_ad::GradGraph`]. Architectural decisions:
//!
//! 1. **Train only the cluster head.** The encoder + temporal-state
//!    engine remain at their seeded init. Training the whole stack
//!    end-to-end would require either backprop through `tanh` activations
//!    in the encoder + temporal layers (doable with GradGraph) or
//!    pre-trained components. For Phase 2d the head is the discriminative
//!    layer; training it captures most of the signal, and the encoder
//!    is intentionally interpretable (one affine + tanh, no semantics
//!    learned).
//! 2. **Single forward graph, reused across batches via `set_tensor`.**
//!    We build the GradGraph once with placeholder tensors, then per
//!    batch we `set_tensor` the inputs, run forward, backward, and
//!    apply Adam to the parameter tensors. This is the same pattern
//!    the Phase 3c PINN demos use (see ADR-0016).
//! 3. **Adam optimizer.** Standard `(β1=0.9, β2=0.999, ε=1e-8)`. Same
//!    deterministic guarantees as the rest of the crate — the only
//!    "randomness" is the seeded weight init.
//! 4. **BCE loss vs the cluster-failure label.** Two binary tasks:
//!    `is_collapse_next` and `is_degraded_next`. Both heads trained
//!    jointly.
//!
//! ## Determinism contract carried forward
//!
//! - Weight init via `seed.substream("grad.collapse.W")` etc. — same
//!   seed produces identical training trajectories.
//! - Batch ordering is canonical: trajectory windows visited in tick
//!   order, deterministic chunking across epochs.
//! - No randomness inside the training loop.

use crate::cluster::ClusterTrajectory;
use crate::cluster_nss::{
    ClusterFailurePredictionHead, ClusterNeuralSystemsSimulator, ClusterNssConfig,
};
use crate::error::NssError;
use crate::failure::FailureKind;
use crate::seed::NssSeed;
use cjc_ad::GradGraph;
use cjc_runtime::Tensor;

/// Optimizer choice for cluster-head training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Optimizer {
    /// Phase 2's count-based calibration: walks the trajectory once,
    /// adds per-feature `(mean_collapse - mean_other) * gain` to the
    /// head weights. Cheap, deterministic, interpretable; doesn't
    /// minimise any loss function explicitly.
    CountBased,
    /// Phase 2d — Adam-based gradient descent through `cjc_ad::GradGraph`.
    /// `lr`, `epochs`, and `batch_size` are the standard knobs.
    Adam {
        /// Learning rate. Must be > 0.
        lr: f64,
        /// Number of full passes over the training data. Must be ≥ 1.
        epochs: u32,
        /// Per-batch sample count. Must be ≥ 1. Larger = smoother
        /// gradient estimates but slower per-step.
        batch_size: u32,
    },
}

impl Optimizer {
    /// Canonical bytes for inclusion in `ClusterNssConfig::canonical_bytes`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        match self {
            Optimizer::CountBased => {
                bytes.extend_from_slice(b"count_based");
            }
            Optimizer::Adam { lr, epochs, batch_size } => {
                bytes.extend_from_slice(b"adam|");
                bytes.extend_from_slice(&lr.to_bits().to_le_bytes());
                bytes.extend_from_slice(&(*epochs as u64).to_le_bytes());
                bytes.extend_from_slice(&(*batch_size as u64).to_le_bytes());
            }
        }
        bytes
    }

    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        match self {
            Optimizer::CountBased => Ok(()),
            Optimizer::Adam {
                lr,
                epochs,
                batch_size,
            } => {
                if !lr.is_finite() || *lr <= 0.0 {
                    return Err(NssError::InvalidConfig {
                        detail: format!("Adam lr must be > 0 and finite, got {}", lr),
                    });
                }
                if *epochs == 0 {
                    return Err(NssError::InvalidConfig {
                        detail: "Adam epochs must be >= 1".into(),
                    });
                }
                if *batch_size == 0 {
                    return Err(NssError::InvalidConfig {
                        detail: "Adam batch_size must be >= 1".into(),
                    });
                }
                Ok(())
            }
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::CountBased
    }
}

/// Loss history record emitted per epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpochLoss {
    /// Epoch index (0-based).
    pub epoch: u32,
    /// Mean BCE loss for the collapse head this epoch.
    pub collapse_loss: f64,
    /// Mean BCE loss for the degraded head this epoch.
    pub degraded_loss: f64,
}

/// Result of a gradient-trained fit: epoch-by-epoch loss curve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingHistory {
    /// Per-epoch losses, in epoch order.
    pub losses: Vec<EpochLoss>,
}

impl TrainingHistory {
    /// True if loss strictly decreased from first to last epoch (a
    /// minimal sanity check; not a convergence guarantee).
    pub fn loss_decreased(&self) -> bool {
        if self.losses.len() < 2 {
            return false;
        }
        let first = &self.losses[0];
        let last = &self.losses[self.losses.len() - 1];
        last.collapse_loss + last.degraded_loss < first.collapse_loss + first.degraded_loss
    }
}

/// Train the cluster failure head with Adam over a trajectory.
///
/// Returns the per-epoch loss curve. Mutates `nss` in place — its
/// internal head is replaced with the gradient-trained weights at the
/// end. The encoder + temporal engine are unchanged.
pub fn fit_with_adam(
    nss: &mut ClusterNeuralSystemsSimulator,
    traj: &ClusterTrajectory,
    lr: f64,
    epochs: u32,
    batch_size: u32,
) -> Result<TrainingHistory, NssError> {
    if traj.len() < 2 {
        return Err(NssError::InvalidTrajectory {
            detail: "Adam fit requires trajectory of length >= 2".into(),
        });
    }
    let cfg = *nss.config();
    let seed = nss.seed();

    // 1. Build the dataset: for each window (now, next), compute
    //    pooled latent + cluster-summary + binary labels.
    //    We do the encoder + temporal forward in plain f64 (frozen
    //    components) since we're only training the head.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(traj.len() - 1);
    let mut labels_collapse: Vec<f64> = Vec::with_capacity(traj.len() - 1);
    let mut labels_degraded: Vec<f64> = Vec::with_capacity(traj.len() - 1);
    for w in traj.as_slice().windows(2) {
        let now = &w[0];
        let next = &w[1];
        let cluster_latent = encode_cluster_latent(nss, &now.state);
        let summary = ClusterNeuralSystemsSimulator::build_cluster_summary(&now.state);
        // Concatenate [h_after_temporal | summary] — the head input.
        let h_after = step_temporal(nss, &cluster_latent);
        let mut feat = Vec::with_capacity(cfg.cluster_head_input());
        feat.extend_from_slice(&h_after);
        feat.extend_from_slice(&summary);
        features.push(feat);
        labels_collapse.push(if next.cluster_failure.kind == FailureKind::Collapse { 1.0 } else { 0.0 });
        labels_degraded.push(if next.cluster_failure.kind == FailureKind::Degraded { 1.0 } else { 0.0 });
    }
    let n = features.len();
    let in_dim = cfg.cluster_head_input();

    // 2. Build the GradGraph once. The graph is:
    //   logit_c = X · w_c + b_c   (shape [batch, 1])
    //   logit_d = X · w_d + b_d
    //   loss_c  = mean( -y_c·log(σ(logit_c)) - (1-y_c)·log(1-σ(logit_c)) )
    //   loss_d  = mean( -y_d·log(σ(logit_d)) - (1-y_d)·log(1-σ(logit_d)) )
    //   loss    = loss_c + loss_d
    //
    // To avoid a `cat` / batched matmul we use a flattened
    // dot-product approach: train one row at a time with
    // accumulation across the batch. Per-sample forward + per-sample
    // backward, then average gradients before applying Adam. This is
    // pedagogically the cleanest design and gives identical gradients
    // to a batched matmul.

    // Initialise parameter tensors from the seeded head — so the
    // gradient-trained run *continues from* the same seeded init the
    // count-based path uses.
    let head_seed = ClusterFailurePredictionHead::from_seed(cfg, seed)?;
    let w_c_init: Vec<f64> = head_seed.collapse_weights().to_vec();
    let b_c_init: f64 = bias_of_seeded_head(cfg, seed, /*collapse=*/ true);
    let w_d_init: Vec<f64> = degraded_weights_of_seeded_head(cfg, seed);
    let b_d_init: f64 = bias_of_seeded_head(cfg, seed, /*collapse=*/ false);

    // Plain Adam state per-parameter.
    let mut w_c = w_c_init;
    let mut b_c = b_c_init;
    let mut w_d = w_d_init;
    let mut b_d = b_d_init;
    let mut m_wc = vec![0.0; in_dim];
    let mut v_wc = vec![0.0; in_dim];
    let mut m_bc = 0.0;
    let mut v_bc = 0.0;
    let mut m_wd = vec![0.0; in_dim];
    let mut v_wd = vec![0.0; in_dim];
    let mut m_bd = 0.0;
    let mut v_bd = 0.0;
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps = 1e-8_f64;

    let mut history = TrainingHistory::default();
    let mut adam_step: u32 = 0;

    for epoch in 0..epochs {
        let mut epoch_collapse_loss = 0.0_f64;
        let mut epoch_degraded_loss = 0.0_f64;
        let mut count = 0_usize;

        // Iterate in batch-sized windows over `features` in canonical
        // (tick) order — deterministic chunking.
        for batch_start in (0..n).step_by(batch_size as usize) {
            let batch_end = (batch_start + batch_size as usize).min(n);
            let bs = batch_end - batch_start;
            if bs == 0 {
                continue;
            }

            // Build per-batch tiny GradGraph. We construct a fresh
            // graph each step to keep the implementation simple — the
            // graph is tiny (in_dim ≤ 32) and Adam's cost is dominated
            // by the param updates anyway.
            let mut graph = GradGraph::new();
            let w_c_idx = graph.parameter(
                Tensor::from_vec(w_c.clone(), &[in_dim]).expect("valid tensor"),
            );
            let b_c_idx = graph.parameter(
                Tensor::from_vec(vec![b_c], &[1]).expect("valid scalar"),
            );
            let w_d_idx = graph.parameter(
                Tensor::from_vec(w_d.clone(), &[in_dim]).expect("valid tensor"),
            );
            let b_d_idx = graph.parameter(
                Tensor::from_vec(vec![b_d], &[1]).expect("valid scalar"),
            );

            // Accumulate per-sample BCE losses (each is a scalar
            // node) and sum into a single batch loss. We compute
            // collapse + degraded losses jointly so one backward
            // updates both heads.
            //
            // Algorithm per sample (dim = in_dim):
            //   xw = sum_i(x_i * w_i)
            //   logit = xw + b
            //   sig   = 1 / (1 + exp(-logit))    (GradGraph has sigmoid)
            //   loss  = -(y log sig + (1 - y) log(1 - sig))
            //         = -y log sig - (1 - y) log(1 - sig)
            // To avoid log(0), we use the cross-entropy identity:
            //   bce(logit, y) = max(logit, 0) - logit * y + log(1 + exp(-|logit|))
            // But GradGraph doesn't expose `max` for a scalar; instead
            // we use the numerically-safer formulation:
            //   loss = log(1 + exp(-logit)) + (1 - y) * logit
            // when y is in {0, 1}. That's equivalent and uses only
            // ln, exp, add, mul, scalar_mul.
            //
            // We use `sigmoid` + `ln` for clarity here — the head is
            // small so the log(0) risk is negligible after the
            // graph's stable sigmoid clamps to ≈ 1e-12..1-1e-12.

            // Pre-compute "x" inputs per sample as input nodes.
            let mut batch_collapse_loss_idx: Option<usize> = None;
            let mut batch_degraded_loss_idx: Option<usize> = None;

            for sample_idx in batch_start..batch_end {
                let x = &features[sample_idx];
                let yc = labels_collapse[sample_idx];
                let yd = labels_degraded[sample_idx];

                // Tiny dot product via per-element mul + sum.
                // For graph size we do it as one input vector + one
                // matmul-equivalent — but cheaper: input is a 1×D
                // tensor, weight is a D×1 tensor, sum reduces it.
                let x_t = Tensor::from_vec(x.clone(), &[in_dim]).expect("valid input tensor");
                let x_idx = graph.input(x_t);
                // Reuse w_c_idx / w_d_idx (the parameter nodes).
                let mul_c = graph.mul(x_idx, w_c_idx);
                let dot_c = graph.sum(mul_c);
                let logit_c = graph.add(dot_c, b_c_idx);
                let mul_d = graph.mul(x_idx, w_d_idx);
                let dot_d = graph.sum(mul_d);
                let logit_d = graph.add(dot_d, b_d_idx);

                // sigmoid + BCE.
                let sig_c = graph.sigmoid(logit_c);
                let sig_d = graph.sigmoid(logit_d);
                let log_sig_c = graph.ln(sig_c);
                let log_sig_d = graph.ln(sig_d);
                // 1 - sigmoid (via neg + scalar_mul); we'll use
                // `ln(1 - sig)` directly via a small constant tensor.
                let one_t = Tensor::from_vec(vec![1.0; 1], &[1]).expect("valid one");
                let one_idx = graph.input(one_t);
                let one_minus_sig_c = graph.sub(one_idx, sig_c);
                let one_minus_sig_d = graph.sub(one_idx, sig_d);
                let log_omsig_c = graph.ln(one_minus_sig_c);
                let log_omsig_d = graph.ln(one_minus_sig_d);

                // loss = -(y log sig + (1 - y) log(1 - sig))
                // We bake yc / yd as constants in the sample-loss
                // closed form: since they're 0 or 1, the loss is
                // -log(sig) if yc==1 else -log(1-sig). We can express
                // this with scalar_mul and add.
                let term_c_a = graph.scalar_mul(log_sig_c, yc);
                let term_c_b = graph.scalar_mul(log_omsig_c, 1.0 - yc);
                let sum_c_terms = graph.add(term_c_a, term_c_b);
                let sample_loss_c = graph.neg(sum_c_terms);

                let term_d_a = graph.scalar_mul(log_sig_d, yd);
                let term_d_b = graph.scalar_mul(log_omsig_d, 1.0 - yd);
                let sum_d_terms = graph.add(term_d_a, term_d_b);
                let sample_loss_d = graph.neg(sum_d_terms);

                batch_collapse_loss_idx = Some(match batch_collapse_loss_idx {
                    Some(prev) => graph.add(prev, sample_loss_c),
                    None => sample_loss_c,
                });
                batch_degraded_loss_idx = Some(match batch_degraded_loss_idx {
                    Some(prev) => graph.add(prev, sample_loss_d),
                    None => sample_loss_d,
                });
            }

            // Mean over the batch (scalar_mul by 1/bs).
            let inv_bs = 1.0 / bs as f64;
            let loss_c_idx =
                graph.scalar_mul(batch_collapse_loss_idx.expect("at least 1 sample"), inv_bs);
            let loss_d_idx =
                graph.scalar_mul(batch_degraded_loss_idx.expect("at least 1 sample"), inv_bs);
            let total_loss = graph.add(loss_c_idx, loss_d_idx);

            // Track loss for the epoch (read scalar from the graph).
            let lc_val = scalar_of(&graph, loss_c_idx);
            let ld_val = scalar_of(&graph, loss_d_idx);
            epoch_collapse_loss += lc_val * bs as f64;
            epoch_degraded_loss += ld_val * bs as f64;
            count += bs;

            // Backward + collect gradients.
            let grads = graph.backward_collect(total_loss, &[w_c_idx, b_c_idx, w_d_idx, b_d_idx]);
            let g_wc = grads[0].clone().expect("w_c grad").to_vec();
            let g_bc = grads[1].clone().expect("b_c grad").to_vec()[0];
            let g_wd = grads[2].clone().expect("w_d grad").to_vec();
            let g_bd = grads[3].clone().expect("b_d grad").to_vec()[0];

            // Adam step.
            adam_step += 1;
            let bc1 = 1.0 - beta1.powi(adam_step as i32);
            let bc2 = 1.0 - beta2.powi(adam_step as i32);
            for i in 0..in_dim {
                m_wc[i] = beta1 * m_wc[i] + (1.0 - beta1) * g_wc[i];
                v_wc[i] = beta2 * v_wc[i] + (1.0 - beta2) * g_wc[i].powi(2);
                let m_hat = m_wc[i] / bc1;
                let v_hat = v_wc[i] / bc2;
                w_c[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                m_wd[i] = beta1 * m_wd[i] + (1.0 - beta1) * g_wd[i];
                v_wd[i] = beta2 * v_wd[i] + (1.0 - beta2) * g_wd[i].powi(2);
                let m_hat_d = m_wd[i] / bc1;
                let v_hat_d = v_wd[i] / bc2;
                w_d[i] -= lr * m_hat_d / (v_hat_d.sqrt() + eps);
            }
            m_bc = beta1 * m_bc + (1.0 - beta1) * g_bc;
            v_bc = beta2 * v_bc + (1.0 - beta2) * g_bc.powi(2);
            b_c -= lr * (m_bc / bc1) / ((v_bc / bc2).sqrt() + eps);
            m_bd = beta1 * m_bd + (1.0 - beta1) * g_bd;
            v_bd = beta2 * v_bd + (1.0 - beta2) * g_bd.powi(2);
            b_d -= lr * (m_bd / bc1) / ((v_bd / bc2).sqrt() + eps);
        }

        let denom = count.max(1) as f64;
        history.losses.push(EpochLoss {
            epoch,
            collapse_loss: epoch_collapse_loss / denom,
            degraded_loss: epoch_degraded_loss / denom,
        });
    }

    // 3. Write the trained weights back into the NSS's head. We
    //    expose collapse-weight setting; degraded-weight setting we
    //    add as a module-internal helper alongside.
    crate::cluster_nss::__set_cluster_head_weights(nss, w_c, b_c, w_d, b_d);
    Ok(history)
}

/// Helper: run the cluster encoder + sum-pool, returning the cluster
/// latent BEFORE the temporal step. Used by `fit_with_adam` to
/// pre-compute features.
fn encode_cluster_latent(
    nss: &ClusterNeuralSystemsSimulator,
    state: &crate::cluster::ClusterSystemState,
) -> Vec<f64> {
    crate::cluster_nss::__cluster_latent(nss, state)
}

/// Helper: step the temporal engine once from zero state. Used to
/// match the prediction-time forward pass.
fn step_temporal(nss: &ClusterNeuralSystemsSimulator, latent: &[f64]) -> Vec<f64> {
    crate::cluster_nss::__step_temporal_from_zero(nss, latent)
}

/// Read the seeded head's degraded-weights vector. Re-derives the
/// vector deterministically from `(cfg, seed)` so Phase 2d training
/// initialises from the same point as Phase 2's count-based path.
fn degraded_weights_of_seeded_head(cfg: ClusterNssConfig, seed: NssSeed) -> Vec<f64> {
    let head = ClusterFailurePredictionHead::from_seed(cfg, seed).expect("seeded head");
    crate::cluster_nss::__degraded_weights(&head).to_vec()
}

/// Read the seeded head's bias for either the collapse or degraded
/// logit.
fn bias_of_seeded_head(cfg: ClusterNssConfig, seed: NssSeed, collapse: bool) -> f64 {
    let head = ClusterFailurePredictionHead::from_seed(cfg, seed).expect("seeded head");
    if collapse {
        crate::cluster_nss::__collapse_bias(&head)
    } else {
        crate::cluster_nss::__degraded_bias(&head)
    }
}

/// Read a scalar (single-element tensor) out of the GradGraph.
fn scalar_of(graph: &GradGraph, idx: usize) -> f64 {
    let t = graph.tensor(idx);
    let v = t.to_vec();
    v[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterTopology;
    use crate::cluster_simulator::{ClusterConfig, ClusterSimulator};

    fn small_traj() -> ClusterTrajectory {
        let cfg = ClusterConfig {
            cluster_arrival_rate: 8.0,
            ..ClusterConfig::default()
        };
        let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
        sim.run(64).unwrap()
    }

    #[test]
    fn optimizer_validates() {
        assert!(Optimizer::CountBased.validate().is_ok());
        assert!(Optimizer::Adam {
            lr: 0.01,
            epochs: 5,
            batch_size: 8
        }
        .validate()
        .is_ok());
        assert!(Optimizer::Adam {
            lr: 0.0,
            epochs: 5,
            batch_size: 8
        }
        .validate()
        .is_err());
        assert!(Optimizer::Adam {
            lr: 0.01,
            epochs: 0,
            batch_size: 8
        }
        .validate()
        .is_err());
        assert!(Optimizer::Adam {
            lr: 0.01,
            epochs: 5,
            batch_size: 0
        }
        .validate()
        .is_err());
    }

    #[test]
    fn adam_fit_reduces_loss_over_epochs() {
        let cfg = ClusterNssConfig::default();
        let mut nss = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let traj = small_traj();
        let history = fit_with_adam(&mut nss, &traj, 0.05, 30, 8).unwrap();
        assert!(history.losses.len() == 30);
        assert!(
            history.loss_decreased(),
            "Adam fit must reduce loss: first={:?}, last={:?}",
            history.losses.first(),
            history.losses.last()
        );
    }

    #[test]
    fn adam_fit_is_deterministic_for_same_seed_and_traj() {
        let cfg = ClusterNssConfig::default();
        let mut nss_a = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let mut nss_b = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let traj = small_traj();
        let history_a = fit_with_adam(&mut nss_a, &traj, 0.05, 10, 8).unwrap();
        let history_b = fit_with_adam(&mut nss_b, &traj, 0.05, 10, 8).unwrap();
        for (a, b) in history_a.losses.iter().zip(history_b.losses.iter()) {
            assert!((a.collapse_loss - b.collapse_loss).abs() < 1e-12);
            assert!((a.degraded_loss - b.degraded_loss).abs() < 1e-12);
        }
        // Both must produce identical predictions on the same state.
        let last = traj.last_state().unwrap();
        let pa = nss_a.predict_next(last).unwrap();
        let pb = nss_b.predict_next(last).unwrap();
        assert_eq!(pa.run_id, pb.run_id);
        assert_eq!(
            pa.failure.collapse_probability.to_bits(),
            pb.failure.collapse_probability.to_bits()
        );
    }

    #[test]
    fn adam_fit_rejects_short_trajectory() {
        let cfg = ClusterNssConfig::default();
        let mut nss = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let top = ClusterTopology::complete(2, 8, 0.5).unwrap();
        let init = crate::cluster::ClusterSystemState::initial(&top);
        let ev = crate::cluster::ClusterEvent {
            state: init,
            actions: std::collections::BTreeMap::new(),
            failures: std::collections::BTreeMap::new(),
            cluster_failure: crate::failure::FailureState::nominal(),
        };
        let traj = ClusterTrajectory::from_events(vec![ev]).unwrap();
        assert!(fit_with_adam(&mut nss, &traj, 0.05, 5, 4).is_err());
    }
}
