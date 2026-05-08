//! Phase 0.4 Track C-2.3.3 — replay semantic invariants.
//!
//! `replay()` validates the audit-log hash chain, but the chain check
//! alone permits adversarially-constructed blobs whose hashes are
//! internally consistent (an attacker who recomputes the chain after
//! tampering) but whose semantic content is wrong:
//!
//! - non-monotonic `seq` sequence (events reordered, duplicated, or skipped)
//! - mismatched `epoch` between events and the snapshot header
//! - swapped `*Updated` events for the same node — the chain still
//!   validates, but each event's recorded `stats_version` no longer
//!   matches the live node's evolution
//! - missing or duplicate `Created` event (the chain anchor)
//!
//! Each test below builds a normal graph, mutates the audit log to
//! produce one specific malformation, recomputes the chain so it
//! validates, then reserializes and confirms `replay()` returns the
//! specific `DecodeError` variant for that invariant.
//!
//! These checks are a second line of defense behind the primary chain
//! check: random byte tampering still surfaces `ChainMismatch` first
//! (covered by existing tests). The new checks fire only on blobs that
//! pass chain validation but violate the higher-level structural
//! contract of the audit log.

use cjc_abng::audit::{AuditEvent, AuditKind};
use cjc_abng::genesis_hash;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize, DecodeError};

/// Build a graph with one Created + several BeliefUpdate events.
fn build_with_observes(seed: u64, n_observes: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    for i in 0..n_observes {
        g.observe(0, (i as f64) * 1.5).unwrap();
    }
    g
}

/// Walk the audit log, setting `previous_hash` from the running chain
/// and recomputing each event's `new_hash` from its current payload.
/// Returns the new final chain head.
fn rebuild_chain(audit: &mut [AuditEvent]) -> [u8; 32] {
    let mut prev = genesis_hash();
    for ev in audit.iter_mut() {
        ev.previous_hash = prev;
        ev.new_hash = ev.recompute_new_hash();
        prev = ev.new_hash;
    }
    prev
}

// ── CreatedMustBeFirst ─────────────────────────────────────────────────

#[test]
fn replay_rejects_zero_events_as_created_must_be_first() {
    // Pre-loop check: a blob whose audit log is empty has no chain
    // anchor. Targeted byte tamper at the known n_events u64 offset.
    let g = build_with_observes(0, 1);
    let mut blob = serialize(&g);

    // v11 layout, single root, no codebook/head/blr/density/calibration/policy:
    //   5 magic + 8 seed + 8 epoch + 32 final_hash
    //   + 1 codebook_present + 1 head_present + 1 blr_prior_present
    //   + 1 density_enabled + 1 calibration_present
    //   + 1 policy_present + 48 action_counts (u64 × 6) = 107
    //   + 8 unfreeze_count u64 (Phase 0.4-extended v11)    = 115
    //   + 4 n_nodes (u32)                                  = 119
    //   + 89 per-node base + 50 (Phase 0.4 Track B-2.2.2 stability buffers)
    //                      + 96 (Phase 0.4 Track B-2.2.1 Welford accumulators)
    //                      = 235 per-node bytes (single root, no extras)
    //   = 354 cumulative
    //   n_events u64 lives at 354..362
    let n_events_offset = 354;
    blob[n_events_offset..n_events_offset + 8].copy_from_slice(&0u64.to_be_bytes());

    let err = replay(&blob).unwrap_err();
    assert_eq!(err, DecodeError::CreatedMustBeFirst);
}

#[test]
fn replay_rejects_first_event_not_created() {
    // Swap the first two events so audit[0] is a BeliefUpdate. Recompute
    // the chain so it validates; the new check fires before the chain
    // check would.
    let mut g = build_with_observes(0, 2);
    g.audit.swap(0, 1);
    g.chain_head = rebuild_chain(&mut g.audit);

    let blob = serialize(&g);
    let err = replay(&blob).unwrap_err();
    assert_eq!(err, DecodeError::CreatedMustBeFirst);
}

#[test]
fn replay_rejects_duplicate_created_event() {
    // Replace audit[1] with a second Created event. Chain stays valid
    // after recompute; semantic check fires at event_index == 1.
    let mut g = build_with_observes(0, 2);
    g.audit[1] = AuditEvent {
        seq: 1,
        epoch: 0,
        node_id: 0,
        kind: AuditKind::Created,
        stats_version: g.audit[0].stats_version,
        stats_hash: g.audit[0].stats_hash,
        previous_hash: [0u8; 32], // overwritten by rebuild_chain
        new_hash: [0u8; 32],      // overwritten by rebuild_chain
    };
    g.chain_head = rebuild_chain(&mut g.audit);

    let blob = serialize(&g);
    let err = replay(&blob).unwrap_err();
    assert_eq!(err, DecodeError::CreatedMustBeFirst);
}

// ── NonMonotonicSeq ────────────────────────────────────────────────────

#[test]
fn replay_rejects_non_monotonic_seq() {
    // Event[1] has seq=1 in a clean graph. Set it to 99; chain stays
    // valid after recompute; new check fires at event_index == 1.
    let mut g = build_with_observes(0, 3);
    g.audit[1].seq = 99;
    g.chain_head = rebuild_chain(&mut g.audit);

    let blob = serialize(&g);
    let err = replay(&blob).unwrap_err();
    assert_eq!(
        err,
        DecodeError::NonMonotonicSeq {
            expected: 1,
            got: 99
        }
    );
}

#[test]
fn replay_rejects_seq_skip_at_first_event() {
    // First event must have seq=0; bumping it to 5 catches at event_index 0.
    let mut g = build_with_observes(0, 1);
    g.audit[0].seq = 5;
    g.chain_head = rebuild_chain(&mut g.audit);

    let blob = serialize(&g);
    let err = replay(&blob).unwrap_err();
    assert_eq!(
        err,
        DecodeError::NonMonotonicSeq {
            expected: 0,
            got: 5
        }
    );
}

// ── EpochMismatch ──────────────────────────────────────────────────────

#[test]
fn replay_rejects_epoch_mismatch() {
    // Event[1] is forged to claim epoch=42 while the header records 0.
    let mut g = build_with_observes(0, 2);
    g.audit[1].epoch = 42;
    g.chain_head = rebuild_chain(&mut g.audit);

    let blob = serialize(&g);
    let err = replay(&blob).unwrap_err();
    assert_eq!(
        err,
        DecodeError::EpochMismatch {
            expected: 0,
            got: 42
        }
    );
}

// ── StatsVersionMismatch ───────────────────────────────────────────────

#[test]
fn replay_rejects_stats_version_mismatch() {
    // After audit[1] (BeliefUpdate) is applied, root.stats_version == 1.
    // Forge the event to claim stats_version=99; chain stays valid;
    // post-apply the new check catches the mismatch before
    // StatsMismatch can fire.
    let mut g = build_with_observes(0, 2);
    g.audit[1].stats_version = 99;
    g.chain_head = rebuild_chain(&mut g.audit);

    let blob = serialize(&g);
    let err = replay(&blob).unwrap_err();
    assert!(
        matches!(
            err,
            DecodeError::StatsVersionMismatch {
                node_id: 0,
                at_seq: 1
            }
        ),
        "expected StatsVersionMismatch{{node_id:0, at_seq:1}}, got {err:?}"
    );
}

// ── Regression: valid blobs still replay ───────────────────────────────

#[test]
fn replay_passes_valid_blob_unchanged() {
    // Sanity: the new checks must not break any valid blob.
    let g = build_with_observes(123, 10);
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("valid blob still replays");
    assert_eq!(g2.chain_head, g.chain_head);
    assert_eq!(g2.audit_len(), g.audit_len());
}

#[test]
fn replay_passes_empty_observation_graph() {
    // The simplest valid blob: one root with only the Created event.
    // Pins that the n_events == 1 case (just the chain anchor) still
    // passes all four new checks.
    let g = AdaptiveBeliefGraph::new(7);
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("root-only graph replays");
    assert_eq!(g2.audit_len(), 1);
}

#[test]
fn replay_passes_multinode_graph() {
    // Pins that NodeAdded + ChildrenPromoted events, which advance
    // stats_version on a different (non-root) node, still pass the
    // stats_version check.
    let mut g = AdaptiveBeliefGraph::new(7);
    let a = g.add_node(0, 1).unwrap();
    let b = g.add_node(0, 2).unwrap();
    g.observe(a, 1.0).unwrap();
    g.observe(a, 2.0).unwrap();
    g.observe(b, 10.0).unwrap();
    let blob = serialize(&g);
    let g2 = replay(&blob).expect("multinode graph replays");
    assert_eq!(g2.chain_head, g.chain_head);
}
