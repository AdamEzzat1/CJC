//! Phase 0.4 Track C-2.3.6 — `abng_leaf_set_params_batch` builtin and
//! the corresponding `LeafParamsUpdatedBatch` audit kind (tag `0x19`).
//!
//! Background: each per-tensor `leaf_set_param(node, k, t)` call appends
//! one `LeafParamsUpdated` event. A 2-layer MLP head (`2(L+1) = 6`
//! tensors) costs 6 events per optimizer step under that pattern; a
//! 100-epoch / 10-leaf training loop emits ~6,000 events.
//!
//! `abng_leaf_set_params_batch` collapses the writeback into a single
//! atomic operation: validate every tensor's shape up-front, write
//! atomically, emit one `LeafParamsUpdatedBatch` event with one hash
//! witness for the whole post-update vector.
//!
//! These tests pin the contract:
//! - Atomicity: validation failure leaves no partial writes.
//! - Equivalence: the batch and per-tensor paths produce the same final
//!   `params_hash`.
//! - Audit-volume reduction: one event vs `n` events.
//! - Round-trip: rebuilt-from-events graphs match.
//! - Interleaving: batch and individual writebacks coexist in one log.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::leaf_head::LeafHeadError;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;

/// 2-feature input → 1-unit hidden → 1-output head. 4 param tensors:
/// `[W_h (1×2), b_h (1), W_o (1×1), b_o (1)]`.
fn graph_with_head() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![1], 1, Activation::Tanh).unwrap();
    g
}

/// Construct a fresh-Xavier-shaped param vector but with predictable
/// values so equivalence comparisons are stable.
fn replacement_params(g: &AdaptiveBeliefGraph) -> Vec<Tensor> {
    let n = g.leaf_param_count(0).unwrap();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let live = g.leaf_param(0, k as u32).unwrap();
        // Replace each tensor with same shape filled with a marker.
        let shape = live.shape().to_vec();
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = (0..numel).map(|i| 0.1 * ((k + 1) as f64) + 0.001 * (i as f64)).collect();
        out.push(Tensor::from_vec(data, &shape).unwrap());
    }
    out
}

#[test]
fn batch_writeback_emits_single_event() {
    let mut g = graph_with_head();
    let pre_audit = g.audit_len() as usize;
    let n_params = g.leaf_param_count(0).unwrap();
    assert!(n_params >= 4, "head should have ≥4 params");

    let new_params = replacement_params(&g);
    g.leaf_set_params_batch(0, new_params).unwrap();

    // Exactly one new event — `LeafParamsUpdatedBatch`.
    assert_eq!(g.audit.len(), pre_audit + 1);
    assert!(matches!(
        g.audit.get(pre_audit).unwrap().kind,
        AuditKind::LeafParamsUpdatedBatch { .. }
    ));
}

#[test]
fn per_tensor_writeback_emits_n_events() {
    // Pins the *baseline* the batch builtin is improving on. If this
    // count ever changes (e.g. someone adds a witness on
    // `leaf_set_param` for a different reason) the batch test should
    // be updated to keep tracking the delta.
    let mut g = graph_with_head();
    let pre_audit = g.audit_len() as usize;
    let n_params = g.leaf_param_count(0).unwrap();

    let new_params = replacement_params(&g);
    for (k, t) in new_params.into_iter().enumerate() {
        g.leaf_set_param(0, k as u32, t).unwrap();
    }

    // Exactly `n_params` new events, one per `leaf_set_param`.
    assert_eq!(g.audit.len(), pre_audit + n_params);
    for offset in 0..n_params {
        assert!(matches!(
            g.audit.get(pre_audit + offset).unwrap().kind,
            AuditKind::LeafParamsUpdated { .. }
        ));
    }
}

#[test]
fn batch_and_individual_produce_same_params_hash() {
    // Two graphs at the same seed; one writes via batch, one via
    // individual. Their `leaf_params_hash` must match — the SHA-256
    // covers the full param vector either way.
    let mut g_batch = graph_with_head();
    let mut g_indiv = graph_with_head();

    let new_params = replacement_params(&g_batch);
    g_batch.leaf_set_params_batch(0, new_params.clone()).unwrap();
    for (k, t) in new_params.into_iter().enumerate() {
        g_indiv.leaf_set_param(0, k as u32, t).unwrap();
    }

    let h_batch = g_batch.leaf_params_hash(0).unwrap();
    let h_indiv = g_indiv.leaf_params_hash(0).unwrap();
    assert_eq!(h_batch, h_indiv);
}

#[test]
fn batch_writeback_validates_param_count_too_few() {
    let mut g = graph_with_head();
    let n_params = g.leaf_param_count(0).unwrap();
    let pre_audit = g.audit_len();

    // Pass too few tensors. Must error and emit no audit event.
    let mut params = replacement_params(&g);
    params.pop();
    let err = g.leaf_set_params_batch(0, params).unwrap_err();
    assert!(
        matches!(
            err,
            GraphError::LeafHead(LeafHeadError::ParamIndexOutOfRange { .. })
        ),
        "expected ParamIndexOutOfRange, got {err:?}"
    );
    assert_eq!(g.audit_len(), pre_audit, "no audit append on count mismatch");
    let _ = n_params;
}

#[test]
fn batch_writeback_validates_param_count_too_many() {
    let mut g = graph_with_head();
    let pre_audit = g.audit_len();

    let mut params = replacement_params(&g);
    let extra = params[0].clone();
    params.push(extra);
    let err = g.leaf_set_params_batch(0, params).unwrap_err();
    assert!(matches!(
        err,
        GraphError::LeafHead(LeafHeadError::ParamIndexOutOfRange { .. })
    ));
    assert_eq!(g.audit_len(), pre_audit);
}

#[test]
fn batch_writeback_validates_each_tensor_shape() {
    let mut g = graph_with_head();
    let pre_audit = g.audit_len();

    let mut params = replacement_params(&g);
    // Replace tensor 1 with a wrong-shaped tensor.
    params[1] = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let err = g.leaf_set_params_batch(0, params).unwrap_err();
    assert!(matches!(
        err,
        GraphError::LeafHead(LeafHeadError::ShapeMismatch { param_index: 1, .. })
    ));
    assert_eq!(g.audit_len(), pre_audit);
}

#[test]
fn batch_writeback_no_partial_writes_on_validation_failure() {
    // Pin the all-or-nothing contract: if any tensor in the batch
    // fails validation, the node's params are unchanged from before
    // the call.
    let mut g = graph_with_head();
    let pre_hash = g.leaf_params_hash(0).unwrap();

    let mut params = replacement_params(&g);
    // Make tensor 2 the wrong shape; tensors 0 and 1 are still valid.
    params[2] = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
    let _ = g.leaf_set_params_batch(0, params).unwrap_err();

    let post_hash = g.leaf_params_hash(0).unwrap();
    assert_eq!(post_hash, pre_hash, "node params must be untouched");
}

#[test]
fn batch_writeback_node_out_of_range_errs() {
    let mut g = graph_with_head();
    let pre_audit = g.audit_len();
    let params = replacement_params(&g);
    let err = g.leaf_set_params_batch(99, params).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
    assert_eq!(g.audit_len(), pre_audit);
}

#[test]
fn batch_writeback_no_head_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.leaf_set_params_batch(0, vec![]).unwrap_err();
    assert!(matches!(err, GraphError::LeafHead(LeafHeadError::NoLeafHead)));
}

#[test]
fn batch_writeback_round_trip_replay_byte_identical() {
    let mut g = graph_with_head();
    let new_params = replacement_params(&g);
    g.leaf_set_params_batch(0, new_params).unwrap();
    let head_before = g.chain_head;
    let audit_before = g.audit_len();

    let blob1 = serialize(&g);
    let g2 = replay(&blob1).expect("replay accepts batch event");
    assert_eq!(g2.chain_head, head_before);
    assert_eq!(g2.audit_len(), audit_before);

    // Re-serialize must be byte-identical.
    let blob2 = serialize(&g2);
    assert_eq!(blob1, blob2);

    // Replayed graph contains exactly one batch event.
    let n_batch = g2
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::LeafParamsUpdatedBatch { .. }))
        .count();
    assert_eq!(n_batch, 1);
}

#[test]
fn batch_then_individual_interleaved_replays() {
    // Mix both event types in one graph and confirm replay handles
    // them in order. The end-of-replay `params_hash` matcher walks
    // the audit log in reverse and finds the most-recent
    // params-touching event — must recognize both kinds.
    let mut g = graph_with_head();
    let new_params = replacement_params(&g);

    // Step 1: batch writeback.
    g.leaf_set_params_batch(0, new_params).unwrap();

    // Step 2: individual writeback to tensor index 0 only — overwrites
    // a slot the batch had set.
    let live0 = g.leaf_param(0, 0).unwrap();
    let shape = live0.shape().to_vec();
    let numel: usize = shape.iter().product();
    let data: Vec<f64> = (0..numel).map(|i| 0.99 - 0.001 * (i as f64)).collect();
    let new_t0 = Tensor::from_vec(data, &shape).unwrap();
    g.leaf_set_param(0, 0, new_t0).unwrap();

    // Both event kinds present, in chronological order.
    let n_batch = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::LeafParamsUpdatedBatch { .. }))
        .count();
    let n_indiv = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::LeafParamsUpdated { .. }))
        .count();
    assert_eq!(n_batch, 1);
    assert_eq!(n_indiv, 1);

    // Replay round-trip must succeed — the post-replay matcher picks
    // up `LeafParamsUpdated` (the latest event) as the witness, and
    // its hash must match the live params.
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("interleaved log replays");
    assert_eq!(g2.chain_head, g.chain_head);
}
