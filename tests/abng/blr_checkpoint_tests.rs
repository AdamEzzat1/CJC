//! Phase 0.9.5 R0-3 (Tier 2 Option C) — periodic BLR audit checkpoints.
//!
//! Research Phase R0 found that `state_hash` (SHA-256 over the d×d BLR
//! precision matrix) is ~67 % of the per-row training cost. Option C
//! keeps the full witness only every [`BLR_CHECKPOINT_INTERVAL`]
//! updates: intermediate `TrainStep` / n=1 `BlrUpdated` events carry
//! the [`BLR_INTERMEDIATE_WITNESS`] zero sentinel, and a node trained
//! past a non-multiple of the interval must be flushed with
//! [`AdaptiveBeliefGraph::checkpoint_blr`] before serialization.
//!
//! These tests pin: the sentinel-vs-real witness cadence, the
//! flush-before-serialize contract (both directions), determinism, and
//! the `checkpoint_blr` skip rules.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::{AdaptiveBeliefGraph, BLR_CHECKPOINT_INTERVAL, BLR_INTERMEDIATE_WITNESS};
use cjc_abng::serialize::{replay, serialize, DecodeError};
use cjc_ad::pinn::Activation;

const SEED: u64 = 42;
const PHI: [f64; 2] = [1.0, 0.5];

/// Graph with a d=2 BLR head and four child leaves under the root.
fn graph() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(SEED);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for byte in 0..4u8 {
        g.add_node(0, byte).unwrap();
    }
    g
}

/// Every `BlrUpdated` witness on node 0 over the last `count` events.
fn root_blr_witnesses(g: &AdaptiveBeliefGraph, count: usize) -> Vec<[u8; 32]> {
    let n = g.audit.len();
    let mut out = Vec::new();
    for i in (n - count)..n {
        if let AuditKind::BlrUpdated { state_hash } = g.audit.get(i).unwrap().kind {
            out.push(state_hash);
        }
    }
    out
}

#[test]
fn intermediate_rows_carry_sentinel_witness() {
    // Fewer than BLR_CHECKPOINT_INTERVAL updates on one node: every
    // n=1 BlrUpdated witness is the zero sentinel.
    let mut g = graph();
    let rows = (BLR_CHECKPOINT_INTERVAL as usize) - 1;
    for i in 0..rows {
        g.blr_update(0, &PHI, &[0.3 + i as f64 * 1e-4]).unwrap();
    }
    let witnesses = root_blr_witnesses(&g, rows);
    assert_eq!(witnesses.len(), rows);
    for (i, w) in witnesses.iter().enumerate() {
        assert_eq!(
            *w, BLR_INTERMEDIATE_WITNESS,
            "update {i} (n_seen={}) must carry the sentinel",
            i + 1
        );
    }
}

#[test]
fn kth_row_carries_real_state_hash() {
    // Exactly BLR_CHECKPOINT_INTERVAL updates: the final event lands on
    // a checkpoint boundary and carries the real state_hash; the
    // earlier ones are sentinels.
    let mut g = graph();
    let k = BLR_CHECKPOINT_INTERVAL as usize;
    for i in 0..k {
        g.blr_update(0, &PHI, &[0.3 + i as f64 * 1e-4]).unwrap();
    }
    let witnesses = root_blr_witnesses(&g, k);
    for w in &witnesses[..k - 1] {
        assert_eq!(*w, BLR_INTERMEDIATE_WITNESS, "pre-checkpoint rows are sentinels");
    }
    let live = g.nodes[0].blr.as_ref().unwrap().state_hash();
    assert_ne!(witnesses[k - 1], BLR_INTERMEDIATE_WITNESS, "the kth row checkpoints");
    assert_eq!(
        witnesses[k - 1], live,
        "the checkpoint witness equals the node's live state_hash"
    );
}

#[test]
fn checkpoint_blr_then_replay_roundtrips() {
    // A node left mid-interval is flushed by checkpoint_blr; replay
    // then verifies cleanly and recovers the chain head + merkle root.
    let mut g = graph();
    for i in 0..10 {
        g.blr_update(0, &PHI, &[0.3 + i as f64 * 1e-4]).unwrap();
    }
    let emitted = g.checkpoint_blr();
    assert_eq!(emitted, 1, "only node 0 was trained mid-interval");
    let head = g.chain_head;
    let root = g.merkle_root();
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("checkpoint-flushed graph replays cleanly");
    assert_eq!(g2.chain_head, head);
    assert_eq!(g2.merkle_root(), root);
}

#[test]
fn replay_without_checkpoint_blr_fails() {
    // The contract, the other direction: a graph trained past a
    // non-multiple of the interval and serialized WITHOUT
    // checkpoint_blr fails replay loudly with BlrStateHashMismatch —
    // never a silent acceptance of an un-anchored final state.
    let mut g = graph();
    for i in 0..10 {
        g.blr_update(0, &PHI, &[0.3 + i as f64 * 1e-4]).unwrap();
    }
    let blob = serialize(&g);
    match replay(&blob) {
        Err(DecodeError::BlrStateHashMismatch { node_id, .. }) => {
            assert_eq!(node_id, 0, "the un-flushed trained node is rejected");
        }
        other => panic!("expected BlrStateHashMismatch, got {other:?}"),
    }
}

#[test]
fn checkpoint_blr_double_run_deterministic() {
    // Same seed + same updates + checkpoint_blr → byte-identical chain
    // head and merkle root across independent runs.
    let build = || {
        let mut g = graph();
        for i in 0..100 {
            g.blr_update(0, &PHI, &[0.3 + i as f64 * 1e-4]).unwrap();
        }
        g.checkpoint_blr();
        (g.chain_head, g.merkle_root())
    };
    assert_eq!(build(), build(), "Option C is deterministic across runs");
}

#[test]
fn checkpoint_blr_skips_boundary_and_untrained_nodes() {
    // checkpoint_blr emits only for nodes left mid-interval: a node
    // trained to an exact multiple of the interval (last event already
    // a real checkpoint) and never-trained nodes (witnessed by
    // BlrInitialized) are skipped.
    let mut g = graph();
    // Node 0: exactly one full interval -> on a checkpoint boundary.
    for i in 0..(BLR_CHECKPOINT_INTERVAL as usize) {
        g.blr_update(0, &PHI, &[0.3 + i as f64 * 1e-4]).unwrap();
    }
    // Node 1: mid-interval. Nodes 2..4: never trained.
    for i in 0..5 {
        g.blr_update(1, &PHI, &[0.5 + i as f64 * 1e-4]).unwrap();
    }
    let emitted = g.checkpoint_blr();
    assert_eq!(emitted, 1, "only the mid-interval node 1 needs a flush");
    // A second flush re-emits for the still-mid-interval node 1
    // (node 0 is now also mid-interval-free, node 1's n_seen unchanged).
    assert_eq!(g.checkpoint_blr(), 1, "flush is keyed on n_seen, deterministic");
}

#[test]
fn train_step_intermediate_witness_is_sentinel() {
    // The same periodic policy on the `train_step` entry point: a
    // single row routes to a leaf and emits a TrainStep whose witness
    // is the sentinel (n_seen=1 is mid-interval).
    let mut g = graph();
    let leaf = g.train_step(&[0.3], &PHI, 0.7).unwrap();
    let last = g.audit.last().unwrap();
    match last.kind {
        AuditKind::TrainStep { state_hash, .. } => {
            assert_eq!(state_hash, BLR_INTERMEDIATE_WITNESS);
        }
        ref k => panic!("expected TrainStep, got {k:?}"),
    }
    // The flush re-anchors the leaf.
    assert_eq!(g.checkpoint_blr(), 1);
    let flushed = g.audit.last().unwrap();
    assert_eq!(flushed.node_id, leaf);
    assert!(matches!(flushed.kind, AuditKind::BlrUpdated { .. }));
}
