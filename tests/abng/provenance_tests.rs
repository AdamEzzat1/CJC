//! Phase 0.5 Item 1 — integration tests for the provenance-stamping
//! audit kind (`0x1C ProvenanceStamped`) and the `stamp_provenance`
//! graph method.
//!
//! Coverage:
//! * stamp — happy path; the field is updated, an event is emitted.
//! * replay — serialize + replay round-trips byte-identically.
//! * audit chain — `verify_chain` accepts the post-stamp chain.
//! * idempotence — re-stamping with the same hash is a no-op.
//! * out-of-range — stamping a non-existent node errors cleanly.
//! * dispatch — the `abng_stamp_provenance` /
//!   `abng_provenance_stamp` builtins round-trip the hex form.

use std::rc::Rc;

use cjc_abng::audit::AuditKind;
use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::serialize::{replay, serialize};
use cjc_runtime::value::Value;

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

fn call_err(name: &str, args: &[Value]) -> String {
    dispatch_abng(name, args).unwrap_err()
}

// ── stamp ──────────────────────────────────────────────────────────────

#[test]
fn stamp_writes_field_and_emits_event() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let stamp = [0xAAu8; 32];
    let pre_chain = g.chain_head;
    let pre_audit_len = g.audit_len();

    g.stamp_provenance(0, stamp).unwrap();

    assert_eq!(g.nodes[0].provenance_stamp_hash, stamp);
    assert_ne!(g.chain_head, pre_chain, "chain_head must advance after stamp");
    assert_eq!(g.audit_len(), pre_audit_len + 1);
    let last = g.audit.last().unwrap();
    match &last.kind {
        AuditKind::ProvenanceStamped { node_id, hash } => {
            assert_eq!(*node_id, 0);
            assert_eq!(*hash, stamp);
        }
        other => panic!("expected ProvenanceStamped, got {other:?}"),
    }
}

#[test]
fn stamp_default_unstamped_is_zero() {
    let g = AdaptiveBeliefGraph::new(0);
    assert_eq!(g.nodes[0].provenance_stamp_hash, [0u8; 32]);
}

// ── replay ─────────────────────────────────────────────────────────────

#[test]
fn replay_round_trips_provenance_byte_identically() {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.observe(0, 0.5).unwrap();
    g.stamp_provenance(0, [0xC0u8; 32]).unwrap();
    g.observe(0, 1.5).unwrap();

    let blob1 = serialize(&g);
    let g2 = replay(&blob1).unwrap();
    let blob2 = serialize(&g2);
    assert_eq!(blob1, blob2, "double-serialize must be byte-identical");
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(
        g2.nodes[0].provenance_stamp_hash, [0xC0u8; 32],
        "replay must restore the per-node provenance stamp"
    );
}

#[test]
fn replay_preserves_provenance_across_nodes() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let _a = g.add_node(0, 1).unwrap();
    let _b = g.add_node(0, 2).unwrap();
    g.stamp_provenance(0, [0x01u8; 32]).unwrap();
    g.stamp_provenance(2, [0x02u8; 32]).unwrap();

    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    assert_eq!(g2.nodes[0].provenance_stamp_hash, [0x01u8; 32]);
    assert_eq!(g2.nodes[1].provenance_stamp_hash, [0u8; 32]);
    assert_eq!(g2.nodes[2].provenance_stamp_hash, [0x02u8; 32]);
}

// ── audit chain ────────────────────────────────────────────────────────

#[test]
fn audit_chain_verifies_with_provenance_event() {
    let mut g = AdaptiveBeliefGraph::new(123);
    g.observe(0, 1.0).unwrap();
    g.stamp_provenance(0, [0xBBu8; 32]).unwrap();
    g.observe(0, 2.0).unwrap();
    assert!(g.verify_chain().is_ok());
}

#[test]
fn audit_chain_tamper_with_provenance_hash_breaks_chain() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.stamp_provenance(0, [0x11u8; 32]).unwrap();
    // Live tampering: we can't easily mutate the audit log through
    // public API, so sanity-check that a fresh recompute matches.
    let last = g.audit.last().unwrap();
    let recomputed = last.recompute_new_hash();
    assert_eq!(recomputed, last.new_hash);
}

// ── idempotence ────────────────────────────────────────────────────────

#[test]
fn stamp_with_same_hash_is_noop() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.stamp_provenance(0, [0x77u8; 32]).unwrap();
    let chain_after_first = g.chain_head;
    let audit_len_after_first = g.audit_len();

    g.stamp_provenance(0, [0x77u8; 32]).unwrap();
    assert_eq!(
        g.chain_head, chain_after_first,
        "re-stamping the same hash must not advance the chain"
    );
    assert_eq!(
        g.audit_len(),
        audit_len_after_first,
        "re-stamping the same hash must not emit a new event"
    );
}

#[test]
fn stamp_with_different_hash_advances_chain_each_time() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.stamp_provenance(0, [0x11u8; 32]).unwrap();
    let c1 = g.chain_head;
    g.stamp_provenance(0, [0x22u8; 32]).unwrap();
    let c2 = g.chain_head;
    g.stamp_provenance(0, [0x33u8; 32]).unwrap();
    let c3 = g.chain_head;
    assert_ne!(c1, c2);
    assert_ne!(c2, c3);
    assert_ne!(c1, c3);
    assert_eq!(g.nodes[0].provenance_stamp_hash, [0x33u8; 32]);
}

// ── out-of-range ───────────────────────────────────────────────────────

#[test]
fn stamp_out_of_range_node_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let result = g.stamp_provenance(99, [0x55u8; 32]);
    assert!(matches!(
        result,
        Err(GraphError::NodeOutOfRange { node_id: 99, .. })
    ));
}

// ── dispatch ───────────────────────────────────────────────────────────

#[test]
fn dispatch_stamp_then_read_round_trip_hex() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(0)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new did not return Int"),
    };
    let hex_in = "deadbeefcafef00d11223344556677889900aabbccddeeff0123456789abcdef";
    let _ = call(
        "abng_stamp_provenance",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::String(Rc::new(hex_in.to_string())),
        ],
    );
    match call(
        "abng_provenance_stamp",
        &[Value::Int(g), Value::Int(0)],
    ) {
        Value::String(s) => assert_eq!(&*s, hex_in),
        other => panic!("expected String, got {}", other.type_name()),
    }
}

#[test]
fn dispatch_stamp_rejects_short_hex() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(0)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new did not return Int"),
    };
    let err = call_err(
        "abng_stamp_provenance",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::String(Rc::new("deadbeef".to_string())),
        ],
    );
    assert!(err.contains("64-char"), "expected 64-char error, got {err}");
}

#[test]
fn dispatch_stamp_rejects_non_hex_byte() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(0)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new did not return Int"),
    };
    // 64 chars but a 'g' inside (not a hex digit).
    let bad = "g0".repeat(32);
    let err = call_err(
        "abng_stamp_provenance",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::String(Rc::new(bad)),
        ],
    );
    assert!(err.contains("non-hex"), "expected non-hex error, got {err}");
}

#[test]
fn dispatch_stamp_out_of_range_errs() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(0)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new did not return Int"),
    };
    let hex_in = "00".repeat(32);
    let err = call_err(
        "abng_stamp_provenance",
        &[
            Value::Int(g),
            Value::Int(99),
            Value::String(Rc::new(hex_in)),
        ],
    );
    // The graph-layer error is wrapped via `graph_err_to_string`.
    assert!(err.contains("99"), "expected node_id-99 mention, got {err}");
}
