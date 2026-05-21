//! Phase 0.4 Track C-2.3.5 — `BlrState.feature_version_hash` contract.
//!
//! Background: prior to 0.4 there was no contract enforcing that the
//! BLR posterior train on a stable feature space. If a user called
//! `leaf_set_param` to update the MLP weights between BLR install and
//! the next `blr_update`, the existing posterior — conditioned on the
//! OLD MLP's penultimate features — would silently absorb evidence
//! computed against the NEW MLP, leaving the posterior mathematically
//! inconsistent.
//!
//! `feature_version_hash` is a per-node 32-byte SHA-256 of the MLP
//! params at the moment the BLR state was last initialized or reset.
//! `blr_update` rejects with `BlrError::FeatureVersionStale` whenever
//! the current params hash differs. Recovery is `abng_reset_blr`,
//! which re-primes the posterior on the new feature space.
//!
//! These tests pin the contract end-to-end: install, drift, reject,
//! reset, succeed, snapshot round-trip.

use cjc_abng::audit::AuditKind;
use cjc_abng::blr::BlrError;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::leaf_head::params_hash;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;

/// 2-input → 2-hidden-unit → 1-output head; 4 param tensors total.
/// `blr_feature_dim` = last hidden width = 2, so BLR features rows
/// are 2-element vectors.
fn graph_with_head_and_blr() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g
}

#[test]
fn feature_version_hash_set_at_blr_install() {
    let g = graph_with_head_and_blr();
    let blr = g.nodes[0].blr.as_ref().expect("BLR installed");
    let expected = params_hash(&g.nodes[0].params);
    assert_eq!(
        blr.feature_version_hash, expected,
        "set_blr_prior must stamp fvh from current params"
    );
}

#[test]
fn feature_version_hash_propagates_to_child_node() {
    let mut g = graph_with_head_and_blr();
    let child_id = g.add_node(0, 1).unwrap();
    let child_blr = g.nodes[child_id as usize]
        .blr
        .as_ref()
        .expect("child BLR initialized");
    let expected = params_hash(&g.nodes[child_id as usize].params);
    assert_eq!(
        child_blr.feature_version_hash, expected,
        "add_node must stamp fvh on child's BLR from child's params"
    );
}

#[test]
fn child_and_root_have_distinct_fvh() {
    // Different node_ids → different Xavier seeds → different params →
    // different fvh. Pins the per-node nature of the hash.
    let mut g = graph_with_head_and_blr();
    let child_id = g.add_node(0, 1).unwrap();
    let root_fvh = g.nodes[0].blr.as_ref().unwrap().feature_version_hash;
    let child_fvh = g.nodes[child_id as usize]
        .blr
        .as_ref()
        .unwrap()
        .feature_version_hash;
    assert_ne!(
        root_fvh, child_fvh,
        "root and child have independent Xavier-init params; fvh should differ"
    );
}

#[test]
fn blr_update_succeeds_when_features_unchanged() {
    let mut g = graph_with_head_and_blr();
    // No leaf_set_param between install and update — fvh unchanged.
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
}

#[test]
fn blr_update_rejects_when_mlp_modified() {
    let mut g = graph_with_head_and_blr();
    let stored_fvh = g.nodes[0].blr.as_ref().unwrap().feature_version_hash;

    // Modify a single MLP param. The tensor's shape is determined by
    // the head; pull the live shape and replace with marker values.
    let original = g.leaf_param(0, 0).unwrap();
    let shape = original.shape().to_vec();
    let numel: usize = shape.iter().product();
    let new_w = Tensor::from_vec(vec![0.42; numel], &shape).unwrap();
    g.leaf_set_param(0, 0, new_w).unwrap();

    // BLR's stored fvh now diverges from current params hash.
    let current = params_hash(&g.nodes[0].params);
    assert_ne!(stored_fvh, current);

    let err = g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap_err();
    match err {
        GraphError::Blr(BlrError::FeatureVersionStale {
            stored,
            current: got,
        }) => {
            assert_eq!(stored, stored_fvh);
            assert_eq!(got, current);
        }
        other => panic!("expected FeatureVersionStale, got {other:?}"),
    }
}

#[test]
fn blr_update_rejection_leaves_posterior_unchanged() {
    let mut g = graph_with_head_and_blr();
    let pre_state_hash = g.nodes[0].blr.as_ref().unwrap().state_hash();
    let pre_chain = g.chain_head;

    // Drift the MLP, attempt blr_update — expect rejection without
    // any side effects.
    let original = g.leaf_param(0, 0).unwrap();
    let new_w = Tensor::from_vec(
        vec![0.99; original.shape().iter().product()],
        original.shape(),
    )
    .unwrap();
    g.leaf_set_param(0, 0, new_w).unwrap();
    let after_set_chain = g.chain_head; // bumped by LeafParamsUpdated
    assert!(g.blr_update(0, &[1.0, 0.5], &[1.0]).is_err());

    let post_state_hash = g.nodes[0].blr.as_ref().unwrap().state_hash();
    assert_eq!(pre_state_hash, post_state_hash, "BLR state untouched");
    assert_eq!(
        g.chain_head, after_set_chain,
        "chain advances only by the leaf_set_param event, not the rejected blr_update"
    );
    let _ = pre_chain;
}

#[test]
fn reset_blr_restores_prior_and_refreshes_fvh() {
    let mut g = graph_with_head_and_blr();

    // Train BLR (succeeds: fvh matches).
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let n_seen_before_reset = g.nodes[0].blr.as_ref().unwrap().n_seen;
    assert!(n_seen_before_reset > 0);

    // Drift the MLP.
    let original = g.leaf_param(0, 0).unwrap();
    let new_w = Tensor::from_vec(
        vec![0.5; original.shape().iter().product()],
        original.shape(),
    )
    .unwrap();
    g.leaf_set_param(0, 0, new_w).unwrap();

    // Reset clears posterior to prior + sets fvh to current params hash.
    g.reset_blr(0).unwrap();
    let blr = g.nodes[0].blr.as_ref().unwrap();
    assert_eq!(blr.n_seen, 0, "n_seen reset to 0");
    assert_eq!(
        blr.feature_version_hash,
        params_hash(&g.nodes[0].params),
        "fvh refreshed to current MLP"
    );
}

#[test]
fn reset_blr_emits_blr_initialized_event() {
    let mut g = graph_with_head_and_blr();
    let pre_audit = g.audit_len() as usize;
    g.reset_blr(0).unwrap();
    assert_eq!(g.audit.len(), pre_audit + 1);
    assert!(matches!(
        g.audit.get(pre_audit).unwrap().kind,
        AuditKind::BlrInitialized { .. }
    ));
}

#[test]
fn blr_update_succeeds_after_reset_with_new_features() {
    // Full recovery flow: install → train → drift MLP → blr_update
    // rejected → reset_blr → blr_update succeeds.
    let mut g = graph_with_head_and_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();

    let original = g.leaf_param(0, 0).unwrap();
    let new_w = Tensor::from_vec(
        vec![0.3; original.shape().iter().product()],
        original.shape(),
    )
    .unwrap();
    g.leaf_set_param(0, 0, new_w).unwrap();

    assert!(g.blr_update(0, &[2.0, 0.5], &[2.0]).is_err());
    g.reset_blr(0).unwrap();
    g.blr_update(0, &[2.0, 0.5], &[2.0]).unwrap();
    assert!(g.nodes[0].blr.as_ref().unwrap().n_seen > 0);
}

#[test]
fn reset_blr_node_out_of_range_errs() {
    let mut g = graph_with_head_and_blr();
    let err = g.reset_blr(99).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
}

#[test]
fn reset_blr_no_prior_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![1], 1, Activation::Tanh).unwrap();
    // No set_blr_prior — reset must error.
    let err = g.reset_blr(0).unwrap_err();
    assert!(matches!(err, GraphError::Blr(BlrError::NoBlrPrior)));
}

#[test]
fn snapshot_round_trip_preserves_fvh() {
    let mut g = graph_with_head_and_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let pre_fvh = g.nodes[0].blr.as_ref().unwrap().feature_version_hash;
    let pre_chain = g.chain_head;

    let blob = serialize(&g);
    let g2 = replay(&blob).expect("replays cleanly");

    assert_eq!(g2.chain_head, pre_chain);
    let post_fvh = g2.nodes[0].blr.as_ref().unwrap().feature_version_hash;
    assert_eq!(post_fvh, pre_fvh);
}

#[test]
fn snapshot_round_trip_after_reset_preserves_fvh() {
    let mut g = graph_with_head_and_blr();
    // Drift + reset so fvh is refreshed mid-history.
    let original = g.leaf_param(0, 0).unwrap();
    let new_w = Tensor::from_vec(
        vec![0.7; original.shape().iter().product()],
        original.shape(),
    )
    .unwrap();
    g.leaf_set_param(0, 0, new_w).unwrap();
    g.reset_blr(0).unwrap();
    g.blr_update(0, &[1.0, 0.5], &[2.0]).unwrap();

    let pre_fvh = g.nodes[0].blr.as_ref().unwrap().feature_version_hash;
    // Phase 0.9.5 R0-3 (Tier 2 Option C) — flush periodic-checkpoint
    // BLR witnesses before serialize so replay's end-of-replay verifier
    // finds a real state hash for the mid-interval node.
    g.checkpoint_blr();
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("replays cleanly");
    let post_fvh = g2.nodes[0].blr.as_ref().unwrap().feature_version_hash;
    assert_eq!(post_fvh, pre_fvh);
    assert_eq!(g2.chain_head, g.chain_head);
}
