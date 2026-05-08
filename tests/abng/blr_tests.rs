//! Phase 0.3b — pure-Rust integration tests of the per-leaf BLR head.

use cjc_abng::audit::AuditKind;
use cjc_abng::blr::BlrError;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

#[test]
fn set_blr_prior_initializes_root_and_emits_two_events() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    assert!(g.blr_prior.is_some());
    assert!(g.nodes[0].blr.is_some());
    // Final two audit events should be BlrPriorConfigured + BlrInitialized.
    let last_two: Vec<_> = g.audit.iter().rev().take(2).collect();
    assert!(matches!(last_two[0].kind, AuditKind::BlrInitialized { .. }));
    assert!(matches!(last_two[1].kind, AuditKind::BlrPriorConfigured { .. }));
}

#[test]
fn set_blr_prior_without_head_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.set_blr_prior(1.0, 1.5, 1.0).unwrap_err();
    assert_eq!(err, GraphError::Blr(BlrError::NoLeafHead));
}

#[test]
fn set_blr_prior_after_add_node_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.add_node(0, 1).unwrap();
    let err = g.set_blr_prior(1.0, 1.5, 1.0).unwrap_err();
    assert!(matches!(err, GraphError::Blr(BlrError::NotEmptyGraph { .. })));
}

#[test]
fn add_node_after_blr_initializes_child_blr() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    let n1 = g.add_node(0, 5).unwrap();
    assert!(g.nodes[n1 as usize].blr.is_some());
    // BlrInitialized event for n1 must appear after its NodeAdded event.
    let initialized_count = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BlrInitialized { .. }))
        .count();
    assert_eq!(initialized_count, 2); // root + n1
}

#[test]
fn blr_update_emits_audit_with_state_witness() {
    let mut g = AdaptiveBeliefGraph::new(0);
    // No hidden layers → d = input_dim = 2.
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    let xs = vec![1.0, 2.0, 0.5, 1.5];
    let ys = vec![1.0, 1.5];
    g.blr_update(0, &xs, &ys).unwrap();
    let last = g.audit.last().unwrap();
    match &last.kind {
        AuditKind::BlrUpdated { state_hash } => {
            let live_hash = g.blr_state_hash(0).unwrap();
            assert_eq!(*state_hash, live_hash);
        }
        other => panic!("expected BlrUpdated, got {other:?}"),
    }
    assert_eq!(g.blr_n_seen(0).unwrap(), 2);
}

#[test]
fn blr_predict_returns_three_values() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 2.0, 1.0).unwrap();
    let (mean, epi, ale) = g.blr_predict(0, &[0.1, 0.2, 0.3, 0.4]).unwrap();
    assert!(mean.is_finite());
    assert!(epi >= 0.0);
    assert!(ale > 0.0); // a=2.0 > 1, aleatoric is finite
}

#[test]
fn blr_features_with_no_hidden_returns_input_idx() {
    let mut g = AdaptiveBeliefGraph::new(0);
    // No hidden layers — degenerate case.
    g.set_leaf_head(3, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();

    cjc_ad::dispatch::reset_ambient();
    let x_idx = cjc_ad::dispatch::with_ambient(|gg| {
        gg.input(
            cjc_runtime::tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        )
    });
    let phi_idx = g.blr_features(0, x_idx).unwrap();
    assert_eq!(phi_idx, x_idx); // identity for empty hidden_dims
}

#[test]
fn blr_features_with_hidden_runs_layers_minus_last() {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();

    cjc_ad::dispatch::reset_ambient();
    let x_idx = cjc_ad::dispatch::with_ambient(|gg| {
        gg.input(
            cjc_runtime::tensor::Tensor::from_vec(vec![0.5, -0.5], &[1, 2]).unwrap(),
        )
    });
    let phi_idx = g.blr_features(0, x_idx).unwrap();
    let phi_shape =
        cjc_ad::dispatch::with_ambient(|gg| gg.tensor(phi_idx).shape().to_vec());
    // Penultimate features after the only hidden layer: shape [1, 4].
    assert_eq!(phi_shape, vec![1, 4]);
}

#[test]
fn full_blr_train_predict_round_trip() {
    // Train a tiny BLR posterior on a clean linear relationship and
    // confirm predictions converge to truth.
    let mut g = AdaptiveBeliefGraph::new(0);
    // No hidden layers — features are raw input.
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    // Vague prior: low precision, broad noise prior.
    g.set_blr_prior(0.001, 1.0, 1.0).unwrap();

    // Truth: y = 2 x_0 + 3 x_1.
    let mut xs = Vec::with_capacity(400);
    let mut ys = Vec::with_capacity(200);
    for i in 0..200 {
        let x0 = (i as f64) * 0.01;
        let x1 = ((i + 7) as f64) * 0.013;
        xs.push(x0);
        xs.push(x1);
        ys.push(2.0 * x0 + 3.0 * x1);
    }
    g.blr_update(0, &xs, &ys).unwrap();

    // Predict at a fresh point (1, 1) → expected ≈ 5.
    let (mean, _epi, _ale) = g.blr_predict(0, &[1.0, 1.0]).unwrap();
    assert!((mean - 5.0).abs() < 0.05, "mean drifted: got {mean}");
}

#[test]
fn blr_round_trip_byte_identical() {
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        let _n1 = g.add_node(0, 5).unwrap();
        g.blr_update(0, &[1.0, 2.0, 0.5, 1.5], &[1.0, 1.5]).unwrap();
        g
    };
    let g1 = mk();
    let g2 = mk();
    assert_eq!(serialize(&g1), serialize(&g2));

    let blob = serialize(&g1);
    let g3 = replay(&blob).unwrap();
    assert_eq!(g1.chain_head, g3.chain_head);
    assert_eq!(
        g1.blr_state_hash(0).unwrap(),
        g3.blr_state_hash(0).unwrap()
    );
    assert_eq!(g1.blr_n_seen(0).unwrap(), g3.blr_n_seen(0).unwrap());
}

#[test]
fn epistemic_uncertainty_decreases_with_data() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(0.01, 1.0, 1.0).unwrap();
    let phi = [1.0, 0.5];
    let (_m0, epi0, _a0) = g.blr_predict(0, &phi).unwrap();
    // Update with informative samples.
    let xs: Vec<f64> = (0..50).flat_map(|i| [(i as f64).sin(), (i as f64).cos()]).collect();
    let ys: Vec<f64> = (0..50).map(|i| (i as f64).sin() * 2.0).collect();
    g.blr_update(0, &xs, &ys).unwrap();
    let (_m1, epi1, _a1) = g.blr_predict(0, &phi).unwrap();
    assert!(
        epi1 < epi0,
        "epistemic var should decrease with data: {epi0} → {epi1}"
    );
}

#[test]
fn chain_verifies_after_blr_train_and_add_nodes() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    for k in 0u8..10 {
        g.add_node(0, k).unwrap();
    }
    for _ in 0..3 {
        g.blr_update(0, &[1.0, 2.0, 0.5, 1.5], &[1.0, 1.5]).unwrap();
    }
    assert!(g.verify_chain().is_ok());
}
