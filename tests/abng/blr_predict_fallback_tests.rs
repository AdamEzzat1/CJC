//! Phase 0.4 Track C-2.3.8 — `abng_blr_predict_with_fallback` parent-walk
//! semantics.
//!
//! Background: each node's BLR is trained independently on observations
//! that flow into that specific node. A freshly-grown leaf has
//! `n_seen == 0` — predicting from it returns prior moments
//! (uninformative posterior mean, unbounded epistemic leverage). The
//! fallback variant walks up the parent chain to the nearest ancestor
//! with `n_seen >= 1` and predicts there instead, returning the source
//! node id alongside the prediction tuple.
//!
//! These tests pin the walk semantics, error cases, and determinism.

use cjc_abng::blr::BlrError;
use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

/// 2-input → 2-hidden → 1-output head; BLR feature dim = 2.
fn graph_with_head_and_blr() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g
}

#[test]
fn predict_at_self_when_n_seen_positive() {
    let mut g = graph_with_head_and_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let (mean_self, lev_self, ale_self) = g.blr_predict(0, &[1.0, 0.0]).unwrap();
    let (mean_fb, lev_fb, ale_fb, source) = g
        .blr_predict_with_fallback(0, &[1.0, 0.0])
        .unwrap();
    assert_eq!(source, 0, "self has n_seen >= 1; source must be self");
    assert_eq!(mean_fb, mean_self);
    assert_eq!(lev_fb, lev_self);
    assert_eq!(ale_fb, ale_self);
}

#[test]
fn predict_falls_back_to_parent_when_self_empty() {
    let mut g = graph_with_head_and_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let child = g.add_node(0, 1).unwrap();
    assert_eq!(g.blr_n_seen(child).unwrap(), 0, "fresh child untrained");
    assert!(g.blr_n_seen(0).unwrap() >= 1, "root trained");

    let (mean_root, lev_root, ale_root) = g.blr_predict(0, &[1.0, 0.0]).unwrap();
    let (mean_fb, lev_fb, ale_fb, source) = g
        .blr_predict_with_fallback(child, &[1.0, 0.0])
        .unwrap();
    assert_eq!(source, 0, "fallback walks one hop to root");
    assert_eq!(mean_fb, mean_root);
    assert_eq!(lev_fb, lev_root);
    assert_eq!(ale_fb, ale_root);
}

#[test]
fn predict_walks_up_to_grandparent() {
    let mut g = graph_with_head_and_blr();
    // Root trained; child + grandchild added but untrained.
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let child = g.add_node(0, 1).unwrap();
    let grandchild = g.add_node(child, 2).unwrap();

    assert_eq!(g.blr_n_seen(child).unwrap(), 0);
    assert_eq!(g.blr_n_seen(grandchild).unwrap(), 0);

    let (_, _, _, source) = g
        .blr_predict_with_fallback(grandchild, &[0.5, 0.5])
        .unwrap();
    assert_eq!(source, 0, "fallback walks two hops to root");
}

#[test]
fn no_evidence_when_all_ancestors_empty() {
    let mut g = graph_with_head_and_blr();
    // Root has BLR installed but never observed; child added — both
    // untrained.
    let child = g.add_node(0, 1).unwrap();
    assert_eq!(g.blr_n_seen(0).unwrap(), 0);
    assert_eq!(g.blr_n_seen(child).unwrap(), 0);

    let err = g
        .blr_predict_with_fallback(child, &[1.0, 0.0])
        .unwrap_err();
    match err {
        GraphError::Blr(BlrError::NoEvidence { walked }) => {
            assert_eq!(walked, 2, "child + root both visited");
        }
        other => panic!("expected BlrError::NoEvidence, got {other:?}"),
    }
}

#[test]
fn no_evidence_at_root_only_graph() {
    let g = graph_with_head_and_blr();
    let err = g.blr_predict_with_fallback(0, &[1.0, 0.0]).unwrap_err();
    match err {
        GraphError::Blr(BlrError::NoEvidence { walked }) => {
            assert_eq!(walked, 1, "only root visited");
        }
        other => panic!("expected BlrError::NoEvidence, got {other:?}"),
    }
}

#[test]
fn node_out_of_range_propagates() {
    let g = graph_with_head_and_blr();
    let err = g
        .blr_predict_with_fallback(99, &[1.0, 0.0])
        .unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { node_id: 99, .. }));
}

#[test]
fn no_blr_prior_propagates() {
    // Graph with leaf head but no set_blr_prior — predict_with_fallback
    // must error with NoBlrPrior at the first node it visits.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    let err = g.blr_predict_with_fallback(0, &[1.0, 0.0]).unwrap_err();
    assert!(matches!(err, GraphError::Blr(BlrError::NoBlrPrior)));
}

#[test]
fn feature_dim_mismatch_propagates() {
    // Train root, then call with wrong-dim phi. Predict-side errors
    // should propagate as-is (not be swallowed as "lack of evidence").
    let mut g = graph_with_head_and_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let err = g
        .blr_predict_with_fallback(0, &[1.0, 0.5, 0.5])
        .unwrap_err();
    assert!(matches!(
        err,
        GraphError::Blr(BlrError::FeatureDimMismatch { .. })
    ));
}

#[test]
fn determinism_double_run() {
    // Same graph constructed + queried twice must give bit-identical
    // (mean, leverage, aleatoric_var, source).
    fn run() -> (f64, f64, f64, u32) {
        let mut g = graph_with_head_and_blr();
        g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        let child = g.add_node(0, 1).unwrap();
        g.blr_predict_with_fallback(child, &[1.0, 0.0]).unwrap()
    }
    let (m1, l1, a1, s1) = run();
    let (m2, l2, a2, s2) = run();
    assert_eq!(m1.to_bits(), m2.to_bits());
    assert_eq!(l1.to_bits(), l2.to_bits());
    assert_eq!(a1.to_bits(), a2.to_bits());
    assert_eq!(s1, s2);
}

#[test]
fn read_only_no_audit_event() {
    // Fallback predict must not append any audit events. Pin this so
    // future "log every predict" instrumentation doesn't accidentally
    // flow through this path.
    let mut g = graph_with_head_and_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    let pre_audit_len = g.audit_len();
    let pre_chain = g.chain_head;
    let _ = g.blr_predict_with_fallback(0, &[1.0, 0.0]).unwrap();
    assert_eq!(g.audit_len(), pre_audit_len);
    assert_eq!(g.chain_head, pre_chain);
}

// ── Dispatch-layer tests ──────────────────────────────────────────────

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

fn try_call(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    dispatch_abng(name, args)
}

fn new_graph_with_head_and_blr_via_dispatch(seed: i64) -> i64 {
    let g = match call("abng_new", &[Value::Int(seed)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new"),
    };
    let hidden = Tensor::from_vec(vec![2.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    let _ = call(
        "abng_set_blr_prior",
        &[
            Value::Int(g),
            Value::Float(1.0),
            Value::Float(1.5),
            Value::Float(1.0),
        ],
    );
    g
}

#[test]
fn dispatch_returns_tensor4_with_source_id() {
    reset_arena();
    let g = new_graph_with_head_and_blr_via_dispatch(7);
    // Train root.
    let features = Tensor::from_vec(vec![1.0, 0.5, 0.5, 1.0], &[2, 2]).unwrap();
    let y = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    let _ = call(
        "abng_blr_update",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(features),
            Value::Tensor(y),
        ],
    );
    // Add a child (untrained) and predict at it via fallback.
    let child = match call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    ) {
        Value::Int(i) => i,
        _ => panic!("abng_add_node"),
    };
    let phi = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let result = call(
        "abng_blr_predict_with_fallback",
        &[Value::Int(g), Value::Int(child), Value::Tensor(phi)],
    );
    let t = match result {
        Value::Tensor(t) => t,
        _ => panic!("expected Tensor[4]"),
    };
    assert_eq!(t.shape(), &[4], "Tensor[4] = [mean, lev, ale, source_id]");
    let data = t.to_vec();
    // source_id must equal root (0) since child has no observations.
    assert_eq!(data[3], 0.0);
    // Tensor[4] values must equal direct predict at root.
    let direct = call(
        "abng_blr_predict",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap()),
        ],
    );
    let direct_vec = match direct {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    assert_eq!(data[0].to_bits(), direct_vec[0].to_bits());
    assert_eq!(data[1].to_bits(), direct_vec[1].to_bits());
    assert_eq!(data[2].to_bits(), direct_vec[2].to_bits());
}

#[test]
fn dispatch_no_evidence_errors_with_clean_message() {
    reset_arena();
    let g = new_graph_with_head_and_blr_via_dispatch(11);
    // No blr_update anywhere — root is at prior, no evidence.
    let phi = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let err = try_call(
        "abng_blr_predict_with_fallback",
        &[Value::Int(g), Value::Int(0), Value::Tensor(phi)],
    )
    .unwrap_err();
    assert!(
        err.contains("no ancestor"),
        "expected NoEvidence message, got {err:?}"
    );
}

#[test]
fn dispatch_node_out_of_range_errors() {
    reset_arena();
    let g = new_graph_with_head_and_blr_via_dispatch(13);
    let phi = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let err = try_call(
        "abng_blr_predict_with_fallback",
        &[Value::Int(g), Value::Int(99), Value::Tensor(phi)],
    )
    .unwrap_err();
    assert!(err.to_lowercase().contains("out of range") || err.to_lowercase().contains("nodeoutofrange"));
}
