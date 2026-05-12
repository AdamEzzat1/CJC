//! Pure-Rust unit tests of the ABNG arena and audit chain.

use cjc_abng::audit::{AuditEvent, AuditKind};
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::stats::NodeStats;
use cjc_abng::genesis_hash;

#[test]
fn welford_zero_var_with_constant_stream() {
    let mut s = NodeStats::new();
    for _ in 0..50 {
        s.observe(7.0);
    }
    assert_eq!(s.mean, 7.0);
    assert!(s.variance().abs() < 1e-12);
}

#[test]
fn welford_matches_naive_for_random_uniform() {
    use cjc_repro::Rng;
    let mut rng = Rng::seeded(42);
    let xs: Vec<f64> = (0..500).map(|_| rng.next_f64() * 100.0).collect();

    let mut welf = NodeStats::new();
    welf.observe_slice(&xs);

    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

    assert!((welf.mean - mean).abs() < 1e-9, "mean: {} vs {}", welf.mean, mean);
    // Welford and naive agree to within accumulator-noise tolerance.
    assert!((welf.variance() - var).abs() < 1e-9, "var: {} vs {}", welf.variance(), var);
}

#[test]
fn graph_starts_with_one_node_and_genesis_anchored_chain() {
    let g = AdaptiveBeliefGraph::new(0);
    assert_eq!(g.node_count(), 1);
    // Created event is always seq=0
    assert_eq!(g.audit.get(0).unwrap().seq, 0);
    assert_eq!(g.audit.get(0).unwrap().previous_hash, genesis_hash());
    assert!(g.verify_chain().is_ok());
}

#[test]
fn observe_increments_audit_log_and_chain_head() {
    let mut g = AdaptiveBeliefGraph::new(99);
    let head_genesis = g.chain_head;
    g.observe(0, 1.0).unwrap();
    let head_after_one = g.chain_head;
    g.observe(0, 2.0).unwrap();
    let head_after_two = g.chain_head;
    assert_ne!(head_genesis, head_after_one);
    assert_ne!(head_after_one, head_after_two);
    assert_eq!(g.audit_len(), 3); // Created + 2× BeliefUpdate
    assert_eq!(g.nodes[0].stats_version, 2);
}

#[test]
fn out_of_range_node_returns_err() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.observe(1, 1.0).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
}

#[test]
fn verify_chain_catches_value_tampering() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.observe(0, 1.0).unwrap();
    g.observe(0, 2.0).unwrap();
    // Phase 0.8 Item B4 — the audit log is columnar; tamper with the
    // event's `kind` via `kinds_mut`, not `get(i)` (which returns an
    // owned copy whose mutation would silently no-op).
    if let AuditKind::BeliefUpdate { value } = &mut g.audit.kinds_mut()[2] {
        *value = 999.0;
    }
    assert!(g.verify_chain().is_err());
}

#[test]
fn verify_chain_catches_previous_hash_tampering() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.observe(0, 1.0).unwrap();
    g.observe(0, 2.0).unwrap();
    // Phase 0.8 Item B4 — columnar mutator. The old
    // `g.audit[1].previous_hash` pattern would tamper an owned copy
    // under the new API and verify_chain would falsely pass.
    g.audit.previous_hashes_mut()[1][0] ^= 0xFF;
    assert!(g.verify_chain().is_err());
}

#[test]
fn double_run_chain_head_byte_identical() {
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(7);
        for i in 0..200 {
            g.observe(0, (i as f64) * 0.001).unwrap();
        }
        g
    };
    let a = mk();
    let b = mk();
    assert_eq!(a.chain_head, b.chain_head);
    assert_eq!(a.nodes[0].stats.canonical_bytes(), b.nodes[0].stats.canonical_bytes());
    assert_eq!(a.audit.len(), b.audit.len());
    for (x, y) in a.audit.iter().zip(b.audit.iter()) {
        assert_eq!(x.new_hash, y.new_hash);
    }
}

#[test]
fn audit_event_payload_layout() {
    // Spot-check that payload size matches the documented layout.
    let g = AdaptiveBeliefGraph::new(0);
    // Created event: 8+8+4+1+8+32 = 61 bytes
    assert_eq!(g.audit.get(0).unwrap().payload_bytes().len(), 61);
}

#[test]
fn belief_update_payload_includes_value_bits() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.observe(0, 1.5).unwrap();
    let payload = g.audit.get(1).unwrap().payload_bytes();
    // BeliefUpdate event: 8+8+4+1+8(value)+8+32 = 69 bytes
    assert_eq!(payload.len(), 69);
    // Tag at offset 20 should be 0x01
    assert_eq!(payload[20], 0x01);
    // Value bits at offset 21..29 (big-endian)
    let mut bits = [0u8; 8];
    bits.copy_from_slice(&payload[21..29]);
    assert_eq!(f64::from_bits(u64::from_be_bytes(bits)), 1.5);
}

#[test]
fn recompute_new_hash_round_trip() {
    let mut g = AdaptiveBeliefGraph::new(0);
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
        g.observe(0, v).unwrap();
    }
    for event in &g.audit {
        assert_eq!(
            event.new_hash,
            AuditEvent::compute_new_hash(&event.previous_hash, &event.payload_bytes())
        );
    }
}
