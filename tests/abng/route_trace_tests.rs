//! Phase 0.4 Track A — `descend_traced` + Routed audit kind +
//! `abng_predict_snap` integration tests.

use cjc_abng::audit::AuditKind;
use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::predict_snap;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn graph_with_codebook() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    let _ = g.add_node(0, 1).unwrap();
    g
}

#[test]
fn descend_traced_returns_same_evidence_as_descend() {
    // Walk semantics must be identical between traced and untraced
    // descend; only the audit-event side effect differs.
    let g_untraced = graph_with_codebook();
    let mut g_traced = graph_with_codebook();
    let prefix = vec![1u8];
    let e_un = g_untraced.descend(&prefix);
    let e_tr = g_traced.descend_traced(&prefix);
    assert_eq!(e_un.leaf_id, e_tr.leaf_id);
    assert_eq!(e_un.matched_prefix, e_tr.matched_prefix);
    assert_eq!(e_un.path, e_tr.path);
}

#[test]
fn descend_traced_appends_routed_event() {
    let mut g = graph_with_codebook();
    let pre = g.audit.len();
    let _ = g.descend_traced(&[1u8]);
    assert_eq!(g.audit.len(), pre + 1);
    assert!(matches!(
        g.audit.last().unwrap().kind,
        AuditKind::Routed { .. }
    ));
}

#[test]
fn untraced_descend_emits_no_event() {
    let g = graph_with_codebook();
    let pre = g.audit.len();
    let _ = g.descend(&[1u8]);
    assert_eq!(g.audit.len(), pre, "descend must not append events");
}

#[test]
fn routed_event_records_correct_leaf_and_matched_prefix() {
    let mut g = graph_with_codebook();
    let evidence = g.descend_traced(&[1u8]);
    let last = g.audit.last().unwrap();
    if let AuditKind::Routed { leaf, matched_prefix } = last.kind {
        assert_eq!(leaf, evidence.leaf_id);
        assert_eq!(matched_prefix, evidence.matched_prefix);
    } else {
        panic!("expected Routed");
    }
    // The event's bound node is the resolved leaf.
    assert_eq!(last.node_id, evidence.leaf_id);
}

#[test]
fn descend_traced_advances_chain_head() {
    let mut g = graph_with_codebook();
    let pre = g.chain_head;
    let _ = g.descend_traced(&[1u8]);
    assert_ne!(pre, g.chain_head);
    assert!(g.verify_chain().is_ok());
}

#[test]
fn descend_traced_round_trips_through_replay() {
    let mut g = graph_with_codebook();
    let _ = g.descend_traced(&[1u8]);
    let _ = g.descend_traced(&[]);  // empty prefix still emits
    let pre_chain = g.chain_head;
    let pre_len = g.audit.len();
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("replay must accept v10 with Routed events");
    assert_eq!(g2.chain_head, pre_chain);
    assert_eq!(g2.audit.len(), pre_len);
    let routed_count_a = g.audit.iter().filter(|e| matches!(e.kind, AuditKind::Routed { .. })).count();
    let routed_count_b = g2.audit.iter().filter(|e| matches!(e.kind, AuditKind::Routed { .. })).count();
    assert_eq!(routed_count_a, routed_count_b);
}

#[test]
fn empty_prefix_descend_traced_records_root_with_zero_match() {
    let mut g = graph_with_codebook();
    let evidence = g.descend_traced(&[]);
    assert_eq!(evidence.leaf_id, 0);
    assert_eq!(evidence.matched_prefix, 0);
}

#[test]
fn descend_traced_two_calls_advance_distinct_chain_heads() {
    let mut g = graph_with_codebook();
    let h0 = g.chain_head;
    let _ = g.descend_traced(&[1u8]);
    let h1 = g.chain_head;
    let _ = g.descend_traced(&[1u8]);
    let h2 = g.chain_head;
    // Each traced descend MUST advance the chain — the seq number
    // alone changes, so even bit-identical-payload events produce
    // distinct hashes.
    assert_ne!(h0, h1);
    assert_ne!(h1, h2);
    // Two Routed events fire (the prior Created/CodebookFrozen/
    // ChildrenPromoted/NodeAdded setup events are not counted here).
    let routed: Vec<_> = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::Routed { .. }))
        .collect();
    assert_eq!(routed.len(), 2);
}

// ── Dispatch-layer tests ──────────────────────────────────────────────

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

/// Build a codebook tensor in the dispatch shape: 2-D Tensor[n_dims,
/// n_bins-1] of f64 boundaries. For 1 dim with 4 bins → boundaries
/// `[-1, 0, 1]` packed as [1,3].
fn codebook_tensor() -> Tensor {
    Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[1, 3]).unwrap()
}

#[test]
fn dispatch_abng_descend_traced_returns_tensor2() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(0)]) {
        Value::Int(i) => i,
        _ => panic!(),
    };
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(codebook_tensor())],
    );
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    );
    let prefix = Tensor::from_vec(vec![1.0], &[1]).unwrap();
    let result = call(
        "abng_descend_traced",
        &[Value::Int(g), Value::Tensor(prefix)],
    );
    let t = match result {
        Value::Tensor(t) => t,
        _ => panic!(),
    };
    assert_eq!(t.shape(), &[2]);
}

// ── abng_predict_snap (pack) ──────────────────────────────────────────

fn graph_with_blr() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    g
}

#[test]
fn abng_predict_snap_dispatch_returns_bytes_with_pred_magic() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(7)]) {
        Value::Int(i) => i,
        _ => panic!(),
    };
    let hidden = Tensor::from_vec(vec![2.0], &[1]).unwrap();
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(codebook_tensor())],
    );
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
        &[Value::Int(g), Value::Float(1.0), Value::Float(1.5), Value::Float(1.0)],
    );
    let features = Tensor::from_vec(vec![1.0, 0.5, 0.5, 1.0], &[2, 2]).unwrap();
    let y = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    let _ = call(
        "abng_blr_update",
        &[Value::Int(g), Value::Int(0), Value::Tensor(features), Value::Tensor(y)],
    );
    let phi = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let result = call(
        "abng_predict_snap",
        &[Value::Int(g), Value::Int(0), Value::Tensor(phi)],
    );
    let bytes: Vec<u8> = match result {
        Value::Bytes(b) => b.borrow().clone(),
        _ => panic!("expected Bytes"),
    };
    assert!(bytes.starts_with(predict_snap::PRED_MAGIC));
    let snap = predict_snap::unpack(&bytes).unwrap();
    assert_eq!(snap.node_id, 0);
    assert_eq!(snap.phi, vec![1.0, 0.0]);
}

#[test]
fn predict_snap_pack_carries_chain_head_and_lineage_hashes() {
    let g = graph_with_blr();
    let bytes = predict_snap::pack(&g, 0, &[1.0, 0.0]).unwrap();
    let snap = predict_snap::unpack(&bytes).unwrap();
    assert_eq!(snap.model_chain_head, g.chain_head);
    assert_eq!(snap.codebook_hash, g.codebook.as_ref().unwrap().frozen_hash);
    assert_eq!(snap.leaf_head_hash, g.head.as_ref().unwrap().config_hash);
    assert_eq!(snap.blr_state_hash, g.nodes[0].blr.as_ref().unwrap().state_hash());
    assert_eq!(snap.blr_n_seen, g.nodes[0].blr.as_ref().unwrap().n_seen);
}

#[test]
fn predict_snap_uncaptured_expected_epistemic_is_nan() {
    let g = graph_with_blr();
    let bytes = predict_snap::pack(&g, 0, &[1.0, 0.0]).unwrap();
    let snap = predict_snap::unpack(&bytes).unwrap();
    assert!(snap.expected_epistemic.is_nan());
}

#[test]
fn predict_snap_captured_expected_epistemic_round_trips() {
    let mut g = graph_with_blr();
    let v = g.force_recapture_expected_epistemic(0).unwrap();
    let bytes = predict_snap::pack(&g, 0, &[1.0, 0.0]).unwrap();
    let snap = predict_snap::unpack(&bytes).unwrap();
    assert_eq!(snap.expected_epistemic.to_bits(), v.to_bits());
}
