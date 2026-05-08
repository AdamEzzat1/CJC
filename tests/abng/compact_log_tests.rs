//! Phase 0.4 Track A — `compact_log` + StatsSnapshot audit kind
//! integration tests.

use cjc_abng::audit::AuditKind;
use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize};
use cjc_runtime::value::Value;

fn graph_with_two_nodes_and_observations() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    let _ = g.add_node(0, 1).unwrap();
    g.observe(0, 0.10).unwrap();
    g.observe(0, 0.20).unwrap();
    g.observe(1, 0.30).unwrap();
    g
}

#[test]
fn compact_log_emits_one_event_per_touched_node() {
    let mut g = graph_with_two_nodes_and_observations();
    let pre_len = g.audit.len();
    let until = pre_len as u64;
    let emitted = g.compact_log(until);
    // Both root and child were observed → two StatsSnapshot events.
    assert_eq!(emitted, 2);
    assert_eq!(g.audit.len(), pre_len + 2);
    let last_two: Vec<_> = g.audit.iter().rev().take(2).collect();
    for ev in last_two {
        assert!(matches!(ev.kind, AuditKind::StatsSnapshot { .. }));
    }
}

#[test]
fn compact_log_visits_nodes_in_ascending_order() {
    let mut g = graph_with_two_nodes_and_observations();
    let pre = g.audit.len();
    let _ = g.compact_log(pre as u64);
    // The two new events should reference node 0 then node 1 in
    // arena order — required for deterministic chain advancement.
    let n0 = match &g.audit[pre].kind {
        AuditKind::StatsSnapshot { node_id, .. } => *node_id,
        _ => panic!(),
    };
    let n1 = match &g.audit[pre + 1].kind {
        AuditKind::StatsSnapshot { node_id, .. } => *node_id,
        _ => panic!(),
    };
    assert_eq!(n0, 0);
    assert_eq!(n1, 1);
}

#[test]
fn compact_log_with_until_zero_emits_nothing() {
    let mut g = graph_with_two_nodes_and_observations();
    let pre = g.audit.len();
    let emitted = g.compact_log(0);
    // Only the genesis Created event has seq < 0... actually seq starts
    // at 0, so until_seq = 0 means "events in [0, 0)" which is empty.
    assert_eq!(emitted, 0);
    assert_eq!(g.audit.len(), pre);
}

#[test]
fn compact_log_clamps_until_seq_to_audit_len() {
    let mut g = graph_with_two_nodes_and_observations();
    let pre = g.audit.len();
    // Pass a much larger until_seq — should still operate on the full
    // log without panic.
    let emitted = g.compact_log(u64::MAX);
    assert!(emitted >= 1);
    assert_eq!(g.audit.len(), pre + emitted as usize);
}

#[test]
fn compact_log_advances_chain_head() {
    let mut g = graph_with_two_nodes_and_observations();
    let pre_chain = g.chain_head;
    let _ = g.compact_log(g.audit.len() as u64);
    assert_ne!(pre_chain, g.chain_head);
    assert!(g.verify_chain().is_ok());
}

#[test]
fn compact_log_round_trips_through_replay() {
    let mut g = graph_with_two_nodes_and_observations();
    let _ = g.compact_log(g.audit.len() as u64);
    let pre_chain = g.chain_head;
    let pre_len = g.audit.len();
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("replay must accept v10 with StatsSnapshot events");
    assert_eq!(g2.chain_head, pre_chain);
    assert_eq!(g2.audit.len(), pre_len);
    let snap_count_a = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::StatsSnapshot { .. }))
        .count();
    let snap_count_b = g2
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::StatsSnapshot { .. }))
        .count();
    assert_eq!(snap_count_a, snap_count_b);
}

#[test]
fn compact_log_stats_hash_matches_node_state_at_emission() {
    let mut g = graph_with_two_nodes_and_observations();
    let pre = g.audit.len();
    // Capture each node's expected stats_hash *before* compact_log
    // (since compact_log doesn't mutate per-node state, the hash at
    // emission == the hash now).
    let h0 = g.nodes[0].stats.stats_hash();
    let h1 = g.nodes[1].stats.stats_hash();
    let _ = g.compact_log(g.audit.len() as u64);
    let snap0 = match &g.audit[pre].kind {
        AuditKind::StatsSnapshot { stats_hash, .. } => *stats_hash,
        _ => panic!(),
    };
    let snap1 = match &g.audit[pre + 1].kind {
        AuditKind::StatsSnapshot { stats_hash, .. } => *stats_hash,
        _ => panic!(),
    };
    assert_eq!(snap0, h0);
    assert_eq!(snap1, h1);
}

#[test]
fn compact_log_is_deterministic_across_two_runs() {
    let mk = || {
        let mut g = graph_with_two_nodes_and_observations();
        let _ = g.compact_log(g.audit.len() as u64);
        g.chain_head
    };
    assert_eq!(mk(), mk());
}

// ── Dispatch-layer tests ──────────────────────────────────────────────

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

#[test]
fn dispatch_abng_compact_log_returns_emitted_count() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(7)]) {
        Value::Int(i) => i,
        _ => panic!(),
    };
    let cb = cjc_runtime::tensor::Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[1, 3])
        .unwrap();
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(cb)],
    );
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    );
    let _ = call("abng_observe", &[Value::Int(g), Value::Int(0), Value::Float(0.5)]);
    let _ = call("abng_observe", &[Value::Int(g), Value::Int(1), Value::Float(0.7)]);
    let len = match call("abng_audit_len", &[Value::Int(g)]) {
        Value::Int(n) => n,
        _ => panic!(),
    };
    let result = call(
        "abng_compact_log",
        &[Value::Int(g), Value::Int(len)],
    );
    let n = match result {
        Value::Int(i) => i,
        _ => panic!("expected Int return"),
    };
    assert_eq!(n, 2, "two nodes observed → two StatsSnapshot events");
}

#[test]
fn dispatch_abng_compact_log_negative_until_seq_errors() {
    reset_arena();
    let g = match call("abng_new", &[Value::Int(0)]) {
        Value::Int(i) => i,
        _ => panic!(),
    };
    let result = dispatch_abng(
        "abng_compact_log",
        &[Value::Int(g), Value::Int(-1)],
    );
    let err = result.unwrap_err();
    assert!(err.contains("non-negative"), "got: {err}");
}
