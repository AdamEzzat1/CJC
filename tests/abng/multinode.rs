//! Phase 0.2 — multi-node arena, children promotion, codebook,
//! descend routing, per-node stats chain decoupling.

use cjc_abng::audit::AuditKind;
use cjc_abng::children::ChildrenKind;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::serialize::{replay, serialize};

#[test]
fn add_node_appends_child_with_audit() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let n1 = g.add_node(0, 5).unwrap();
    assert_eq!(n1, 1);
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.nodes[1].parent, Some(0));
    // Created (root) + ChildrenPromoted (None→Node4) + NodeAdded.
    assert_eq!(g.audit_len(), 3);
    assert!(matches!(g.audit[1].kind, AuditKind::ChildrenPromoted { from: 0, to: 1 }));
    assert!(matches!(
        g.audit[2].kind,
        AuditKind::NodeAdded { parent: 0, key_byte: 5 }
    ));
}

#[test]
fn duplicate_key_byte_rejected() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.add_node(0, 5).unwrap();
    let err = g.add_node(0, 5).unwrap_err();
    assert_eq!(
        err,
        GraphError::KeyAlreadyBound {
            parent: 0,
            key_byte: 5
        }
    );
}

#[test]
fn promotion_chain_node4_to_node256() {
    let mut g = AdaptiveBeliefGraph::new(0);
    for k in 0u8..100 {
        g.add_node(0, k).unwrap();
    }
    assert_eq!(g.nodes[0].children.kind(), ChildrenKind::Node256);
    let promotions: Vec<_> = g
        .audit
        .iter()
        .filter_map(|e| match &e.kind {
            AuditKind::ChildrenPromoted { from, to } => Some((*from, *to)),
            _ => None,
        })
        .collect();
    assert_eq!(promotions, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
}

#[test]
fn descend_walks_match_until_unbound_byte() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let n1 = g.add_node(0, 7).unwrap();
    let n2 = g.add_node(n1, 9).unwrap();
    let _n3 = g.add_node(n2, 11).unwrap();

    let r = g.descend(&[7, 9, 11]);
    assert_eq!(r.matched_prefix, 3);
    assert_eq!(r.path.len(), 4);

    let r = g.descend(&[7, 9, 99]);
    assert_eq!(r.matched_prefix, 2);
    assert_eq!(r.path.len(), 3);

    let r = g.descend(&[42]);
    assert_eq!(r.matched_prefix, 0);
    assert_eq!(r.path, vec![0]);
}

#[test]
fn descend_handles_long_branchy_paths() {
    // Build a 5-deep, 4-wide trie; descend at every cell to confirm
    // matched_prefix aligns with the byte sequence.
    let mut g = AdaptiveBeliefGraph::new(0);
    let mut frontier = vec![0u32];
    for _ in 0..5 {
        let mut next_frontier = Vec::new();
        for parent in &frontier {
            for k in 0u8..4 {
                next_frontier.push(g.add_node(*parent, k).unwrap());
            }
        }
        frontier = next_frontier;
    }
    // Walk a deterministic path: bytes [0, 1, 2, 3, 0]
    let path_bytes = [0u8, 1, 2, 3, 0];
    let r = g.descend(&path_bytes);
    assert_eq!(r.matched_prefix, 5);
    assert_eq!(r.path.len(), 6);
    // Bailing on the first invalid byte:
    let r = g.descend(&[0, 99]);
    assert_eq!(r.matched_prefix, 1);
}

#[test]
fn codebook_set_then_encode() {
    let mut g = AdaptiveBeliefGraph::new(0);
    // 2 dims × 4 bins → 3 boundaries each
    g.set_codebook(2, 4, &[0.5, 1.5, 2.5, 0.5, 1.5, 2.5]).unwrap();
    let p = g.encode_prefix(&[0.0, 1.0]).unwrap();
    assert_eq!(p, vec![0, 1]);
}

#[test]
fn codebook_freeze_is_one_shot() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_codebook(1, 4, &[0.5, 1.5, 2.5]).unwrap();
    let err = g.set_codebook(1, 4, &[0.5, 1.5, 2.5]).unwrap_err();
    assert_eq!(err, GraphError::CodebookAlreadyFrozen);
}

#[test]
fn encode_without_codebook_errs() {
    let g = AdaptiveBeliefGraph::new(0);
    let err = g.encode_prefix(&[1.0]).unwrap_err();
    assert_eq!(err, GraphError::NoCodebook);
}

#[test]
fn per_node_chain_isolated_from_other_nodes() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let n1 = g.add_node(0, 1).unwrap();
    let head_n0_pre = g.nodes[0].stats_chain_head;
    let head_n1_pre = g.nodes[n1 as usize].stats_chain_head;

    // Observe only node 1; node 0's per-node chain must NOT advance.
    g.observe(n1, 5.0).unwrap();

    assert_eq!(g.nodes[0].stats_chain_head, head_n0_pre, "node 0 chain advanced unexpectedly");
    assert_ne!(g.nodes[n1 as usize].stats_chain_head, head_n1_pre, "node 1 chain didn't advance");

    // Global chain *did* advance.
    let global_after = g.chain_head;
    g.observe(0, 9.0).unwrap();
    assert_ne!(g.chain_head, global_after);
    assert_ne!(g.nodes[0].stats_chain_head, head_n0_pre);
}

#[test]
fn multinode_round_trip_byte_identical() {
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(7);
        let a = g.add_node(0, 1).unwrap();
        let b = g.add_node(0, 2).unwrap();
        let _c = g.add_node(a, 3).unwrap();
        for v in [1.0, 2.0, 3.0, 4.0] {
            g.observe(a, v).unwrap();
        }
        for v in [10.0, 20.0] {
            g.observe(b, v).unwrap();
        }
        g
    };
    let g1 = mk();
    let g2 = mk();
    let blob1 = serialize(&g1);
    let blob2 = serialize(&g2);
    assert_eq!(blob1, blob2, "double-run snapshot bytes differ");

    let g3 = replay(&blob1).unwrap();
    let blob3 = serialize(&g3);
    assert_eq!(blob1, blob3, "round-trip snapshot bytes differ");
}

#[test]
fn round_trip_with_codebook_and_promotion() {
    let mut g = AdaptiveBeliefGraph::new(42);
    // Trigger Node4 → Node16 → Node48 promotion on the root.
    for k in 0u8..20 {
        g.add_node(0, k).unwrap();
    }
    // Install a codebook.
    g.set_codebook(3, 4, &[0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]).unwrap();
    // Observe across multiple nodes.
    g.observe(0, 1.5).unwrap();
    g.observe(1, 2.5).unwrap();
    g.observe(2, 3.5).unwrap();

    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.nodes[0].children.kind(), g2.nodes[0].children.kind());
    assert_eq!(
        g.codebook.as_ref().unwrap().frozen_hash,
        g2.codebook.as_ref().unwrap().frozen_hash
    );
    // Per-node stats_chain_head also bit-equal.
    for (a, b) in g.nodes.iter().zip(g2.nodes.iter()) {
        assert_eq!(a.stats_chain_head, b.stats_chain_head);
    }
}

#[test]
fn replay_rejects_v1_magic() {
    // Synthesize a Phase 0.1-shaped magic header on a v2 blob.
    let g = AdaptiveBeliefGraph::new(0);
    let mut blob = serialize(&g);
    blob[4] = 0x01;
    let err = replay(&blob);
    assert!(matches!(
        err,
        Err(cjc_abng::serialize::DecodeError::BadMagic)
    ));
}

#[test]
fn chain_verifies_after_complex_multinode_mutation() {
    let mut g = AdaptiveBeliefGraph::new(0);
    // Mix structural + observational events at different rates.
    for k in 0u8..30 {
        g.add_node(0, k).unwrap();
    }
    for i in 0..50 {
        g.observe(0, i as f64).unwrap();
    }
    g.set_codebook(2, 8, &(0..14).map(|i| (i as f64) * 0.5).collect::<Vec<_>>())
        .unwrap();
    for child in 1..=10u32 {
        g.observe(child, child as f64).unwrap();
    }
    assert!(g.verify_chain().is_ok());
}
