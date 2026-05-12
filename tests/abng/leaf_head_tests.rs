//! Phase 0.3a — pure-Rust tests of the per-leaf MLP head subsystem.
//!
//! Exercises the API path that does *not* go through the dispatch
//! layer: graph methods (`set_leaf_head`, `leaf_param`, `leaf_set_param`,
//! `leaf_forward`), Xavier-init determinism, audit-event ordering,
//! snapshot v3 round-trip with params, and the `set_leaf_head`
//! before-add_node ordering constraint.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::leaf_head::{params_hash, LeafHeadError};
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;

#[test]
fn set_leaf_head_initializes_root_and_emits_two_events() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    // Root should now have params populated.
    assert_eq!(g.nodes[0].params.len(), 4); // 2 layers × {W, b}
    assert!(g.head.is_some());
    // Three audit events: Created, LeafHeadConfigured, LeafParamsInitialized.
    assert_eq!(g.audit_len(), 3);
    assert!(matches!(
        g.audit.get(1).unwrap().kind,
        AuditKind::LeafHeadConfigured { .. }
    ));
    assert!(matches!(
        g.audit.get(2).unwrap().kind,
        AuditKind::LeafParamsInitialized { .. }
    ));
}

#[test]
fn set_leaf_head_after_add_node_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.add_node(0, 1).unwrap();
    let err = g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap_err();
    assert_eq!(
        err,
        GraphError::LeafHead(LeafHeadError::NotEmptyGraph { n_nodes: 2 })
    );
}

#[test]
fn set_leaf_head_twice_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    let err = g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap_err();
    assert_eq!(err, GraphError::LeafHead(LeafHeadError::AlreadyFrozen));
}

#[test]
fn set_leaf_head_zero_dim_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.set_leaf_head(0, vec![4], 1, Activation::Tanh).unwrap_err();
    assert_eq!(err, GraphError::LeafHead(LeafHeadError::ZeroDim));
}

#[test]
fn add_node_after_head_initializes_child_params_deterministically() {
    let mut g1 = AdaptiveBeliefGraph::new(7);
    g1.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    let n1 = g1.add_node(0, 5).unwrap();

    // Same setup → bit-identical child params.
    let mut g2 = AdaptiveBeliefGraph::new(7);
    g2.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    let n2 = g2.add_node(0, 5).unwrap();
    assert_eq!(n1, n2);

    for k in 0..g1.leaf_param_count(n1).unwrap() {
        let p1 = g1.leaf_param(n1, k as u32).unwrap();
        let p2 = g2.leaf_param(n2, k as u32).unwrap();
        assert_eq!(p1.to_vec(), p2.to_vec());
    }
}

#[test]
fn leaf_param_without_head_errs() {
    let g = AdaptiveBeliefGraph::new(0);
    let err = g.leaf_param(0, 0).unwrap_err();
    assert_eq!(err, GraphError::LeafHead(LeafHeadError::NoLeafHead));
}

#[test]
fn leaf_set_param_shape_mismatch_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    let bad = Tensor::from_vec(vec![0.0; 7], &[7]).unwrap();
    let err = g.leaf_set_param(0, 0, bad).unwrap_err();
    assert!(matches!(err, GraphError::LeafHead(LeafHeadError::ShapeMismatch { .. })));
}

#[test]
fn leaf_set_param_emits_updated_event_with_new_hash() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    let pre_hash = g.leaf_params_hash(0).unwrap();

    let new_w0 = Tensor::from_vec(vec![0.42; 8], &[4, 2]).unwrap();
    g.leaf_set_param(0, 0, new_w0).unwrap();

    // Last event must be LeafParamsUpdated with the post-update hash.
    let last = g.audit.last().unwrap();
    match &last.kind {
        AuditKind::LeafParamsUpdated { params_hash } => {
            let post_hash = g.leaf_params_hash(0).unwrap();
            assert_eq!(*params_hash, post_hash);
            assert_ne!(*params_hash, pre_hash);
        }
        other => panic!("expected LeafParamsUpdated, got {other:?}"),
    }
}

#[test]
fn leaf_forward_returns_correct_shape() {
    // 2 → [4] → 1, tanh: forward produces a Tensor of shape [1, 1] in the
    // ambient GradGraph (mlp_layer's typical batch=1 path) — we check via
    // grad graph state.
    use cjc_ad::dispatch::with_ambient;
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();

    // Reset the ambient graph and add an input.
    cjc_ad::dispatch::reset_ambient();
    let x_idx = with_ambient(|gg| gg.input(Tensor::from_vec(vec![0.5, -0.5], &[1, 2]).unwrap()));

    let (y_idx, params) = g.leaf_forward(0, x_idx).unwrap();
    let y_shape = with_ambient(|gg| gg.tensor(y_idx).shape().to_vec());
    assert_eq!(params.len(), 4); // 2 W + 2 b
    // Output is [1, 1] for batch=1, output_dim=1.
    assert_eq!(y_shape, vec![1, 1]);
}

#[test]
fn leaf_forward_is_deterministic() {
    use cjc_ad::dispatch::with_ambient;
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(11);
        g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
        cjc_ad::dispatch::reset_ambient();
        let x_idx = with_ambient(|gg| {
            gg.input(Tensor::from_vec(vec![0.3, -0.7], &[1, 2]).unwrap())
        });
        let (y_idx, _params) = g.leaf_forward(0, x_idx).unwrap();
        with_ambient(|gg| gg.tensor(y_idx).to_vec())
    };
    let a = mk();
    let b = mk();
    assert_eq!(a, b);
}

#[test]
fn full_train_step_chain_verifies() {
    // Full forward + backward + Adam-style update + writeback.
    use cjc_ad::dispatch::with_ambient;
    let mut g = AdaptiveBeliefGraph::new(42);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();

    cjc_ad::dispatch::reset_ambient();
    let x_idx = with_ambient(|gg| {
        gg.input(Tensor::from_vec(vec![0.5, 0.3], &[1, 2]).unwrap())
    });
    let (y_idx, params) = g.leaf_forward(0, x_idx).unwrap();

    // target = 1.0 → loss = (y - 1.0)^2
    let target_idx = with_ambient(|gg| {
        gg.input(Tensor::from_vec(vec![1.0], &[1, 1]).unwrap())
    });
    let diff = with_ambient(|gg| gg.sub(y_idx, target_idx));
    let sq = with_ambient(|gg| gg.mul(diff, diff));
    let loss = with_ambient(|gg| gg.sum(sq));
    let grads = with_ambient(|gg| gg.backward_collect(loss, &params));

    // Plain SGD: w' = w - lr * grad
    let lr = 0.01;
    for (k, grad_opt) in grads.into_iter().enumerate() {
        let grad = grad_opt.expect("missing gradient");
        let w = g.leaf_param(0, k as u32).unwrap();
        let w_data = w.to_vec();
        let g_data = grad.to_vec();
        let new_data: Vec<f64> = w_data
            .iter()
            .zip(g_data.iter())
            .map(|(&w, &g)| w - lr * g)
            .collect();
        let shape = w.shape().to_vec();
        let new_t = Tensor::from_vec(new_data, &shape).unwrap();
        g.leaf_set_param(0, k as u32, new_t).unwrap();
    }
    assert!(g.verify_chain().is_ok());
}

#[test]
fn round_trip_with_head_and_updates() {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    let n1 = g.add_node(0, 5).unwrap();
    let n2 = g.add_node(0, 9).unwrap();

    // Update n1's first weight tensor with a known value so we can
    // check the round-trip preserved post-update params.
    let custom = Tensor::from_vec(vec![1.5; 8], &[4, 2]).unwrap();
    g.leaf_set_param(n1, 0, custom).unwrap();

    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();

    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(
        g.head.as_ref().unwrap().config_hash,
        g2.head.as_ref().unwrap().config_hash
    );
    // n1's first weight tensor should match what we wrote.
    let p_orig = g.leaf_param(n1, 0).unwrap();
    let p_replayed = g2.leaf_param(n1, 0).unwrap();
    assert_eq!(p_orig.to_vec(), p_replayed.to_vec());
    // Determinism gate: every node's params hash matches.
    for nid in 0..g.node_count() {
        assert_eq!(
            params_hash(&g.nodes[nid as usize].params),
            params_hash(&g2.nodes[nid as usize].params),
        );
    }
    assert_eq!(n1, 1);
    assert_eq!(n2, 2);
}

#[test]
fn double_run_with_head_byte_identical_blob() {
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(99);
        g.set_leaf_head(3, vec![6, 6], 2, Activation::Relu).unwrap();
        for k in 0u8..5 {
            g.add_node(0, k).unwrap();
        }
        g
    };
    let a = mk();
    let b = mk();
    assert_eq!(serialize(&a), serialize(&b));
}
