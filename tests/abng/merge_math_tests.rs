//! Phase 0.4 Track B-2.2.6 — real NIG-aware merge math.
//!
//! Pre-0.4: `force_merge(absorbed, into)` only set `absorbed.is_active
//! = false`. All of absorbed's training history (Welford stats + BLR
//! posterior) was silently dropped — a node receiving merges *lost*
//! information instead of gaining it. Phase 0.4 makes Merge actually
//! fold absorbed's evidence into `into`:
//!
//! 1. Welford parallel-merge of `NodeStats` (Chan/Golub/LeVeque) so
//!    `into.n_seen` and `into.mean` reflect both streams.
//! 2. NIG-aware combine of the BLR posterior — sum precisions,
//!    precision-weighted-mean of the means, and `(a, b) ←
//!    (a_into + a_other - a_prior, b_into + b_other - b_prior)` to
//!    avoid double-counting the prior.
//! 3. Mark absorbed inactive (unchanged from pre-0.4).
//!
//! These tests pin the contract: combine increases `n_seen`, the
//! combined posterior is non-trivial, the audit log replays
//! byte-identically, and the chain integrity holds across the merge.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

fn graph_with_two_observed_nodes() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    let _a = g.add_node(0, 1).unwrap(); // child a (id=1)
    let _b = g.add_node(0, 2).unwrap(); // child b (id=2)

    // Drive observations into both children to build up Welford state.
    for v in &[1.0, 2.0, 3.0] {
        g.observe(1, *v).unwrap();
    }
    for v in &[10.0, 20.0] {
        g.observe(2, *v).unwrap();
    }

    // Drive BLR updates on both.
    g.blr_update(1, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    g.blr_update(2, &[2.0, 1.0, 1.0, 2.0], &[2.0, 2.0]).unwrap();
    g
}

#[test]
fn merge_combines_n_seen_into_target() {
    // Pre-0.4 this test would fail: `into.stats.n_seen` would stay at
    // its pre-merge value (3) instead of growing to 5 (3 + 2).
    let mut g = graph_with_two_observed_nodes();
    let into_n_pre = g.nodes[1].stats.n_seen;
    let absorbed_n = g.nodes[2].stats.n_seen;
    g.force_merge(2, 1).unwrap();
    assert_eq!(
        g.nodes[1].stats.n_seen,
        into_n_pre + absorbed_n,
        "merge must fold absorbed's n_seen into into"
    );
}

#[test]
fn merge_marks_absorbed_inactive_unchanged() {
    let mut g = graph_with_two_observed_nodes();
    g.force_merge(2, 1).unwrap();
    assert!(!g.nodes[2].is_active);
    assert!(g.nodes[1].is_active, "into stays active");
}

#[test]
fn merge_combines_blr_n_seen() {
    let mut g = graph_with_two_observed_nodes();
    let into_blr_pre = g.nodes[1].blr.as_ref().unwrap().n_seen;
    let absorbed_blr = g.nodes[2].blr.as_ref().unwrap().n_seen;
    assert!(into_blr_pre > 0 && absorbed_blr > 0);
    g.force_merge(2, 1).unwrap();
    assert_eq!(
        g.nodes[1].blr.as_ref().unwrap().n_seen,
        into_blr_pre + absorbed_blr,
        "BLR n_seen sums on combine"
    );
}

#[test]
fn merge_changes_into_state_hash() {
    // Combining absorbed's evidence into `into` changes into's
    // canonical bytes → different state hash.
    let mut g = graph_with_two_observed_nodes();
    let into_pre = g.nodes[1].blr.as_ref().unwrap().state_hash();
    g.force_merge(2, 1).unwrap();
    let into_post = g.nodes[1].blr.as_ref().unwrap().state_hash();
    assert_ne!(into_pre, into_post);
}

#[test]
fn merge_increments_action_count() {
    let mut g = graph_with_two_observed_nodes();
    let pre = g.action_counts[cjc_abng::graph::ActionKind::Merge as usize];
    g.force_merge(2, 1).unwrap();
    let post = g.action_counts[cjc_abng::graph::ActionKind::Merge as usize];
    assert_eq!(post, pre + 1);
}

#[test]
fn merge_emits_merge_event_followed_by_blr_updated_for_into() {
    // Phase 0.4 Track B-2.2.6: when both nodes carry BLR, force_merge
    // emits the structural `Merge` event for absorbed AND a follow-on
    // `BlrUpdated` witness on `into` carrying the post-combine
    // state_hash. The witness is required by the end-of-replay
    // per-node BLR verifier — without it, a graph that snapshots a
    // post-merge state would replay with a stale pre-merge witness
    // and reject as `BlrStateHashMismatch`.
    let mut g = graph_with_two_observed_nodes();
    let pre_audit = g.audit_len() as usize;
    g.force_merge(2, 1).unwrap();
    assert_eq!(g.audit.len(), pre_audit + 2);
    assert!(matches!(
        g.audit[pre_audit].kind,
        AuditKind::Merge {
            absorbed: 2,
            into: 1
        }
    ));
    let into_witness_state_hash = match g.audit[pre_audit + 1].kind {
        AuditKind::BlrUpdated { state_hash } => state_hash,
        ref k => panic!("expected BlrUpdated for into after Merge, got {k:?}"),
    };
    assert_eq!(g.audit[pre_audit + 1].node_id, 1);
    // The BlrUpdated witness must match the live combined state.
    assert_eq!(
        into_witness_state_hash,
        g.nodes[1].blr.as_ref().unwrap().state_hash()
    );
}

#[test]
fn merge_without_blr_emits_only_merge_event() {
    // No BLR prior installed → no BLR combine → no follow-on
    // BlrUpdated witness, just the Merge event.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
    let _a = g.add_node(0, 1).unwrap();
    let _b = g.add_node(0, 2).unwrap();
    g.observe(1, 1.0).unwrap();
    g.observe(2, 2.0).unwrap();
    let pre_audit = g.audit_len() as usize;
    g.force_merge(2, 1).unwrap();
    assert_eq!(g.audit.len(), pre_audit + 1);
    assert!(matches!(
        g.audit[pre_audit].kind,
        AuditKind::Merge { .. }
    ));
}

#[test]
fn merge_round_trips_byte_identical() {
    let mut g = graph_with_two_observed_nodes();
    g.force_merge(2, 1).unwrap();
    let chain_pre = g.chain_head;
    let into_n_pre = g.nodes[1].stats.n_seen;
    let blob1 = serialize(&g);
    let g2 = replay(&blob1).expect("replay accepts post-merge blob");
    assert_eq!(g2.chain_head, chain_pre);
    assert_eq!(g2.nodes[1].stats.n_seen, into_n_pre);
    assert!(!g2.nodes[2].is_active);
    let blob2 = serialize(&g2);
    assert_eq!(blob1, blob2);
}

#[test]
fn merge_followed_by_observe_chain_verifies() {
    // After merge, observing on `into` must keep the audit chain
    // intact: the BeliefUpdate event's `stats_hash` reflects the
    // post-combine + post-update state. Replay-side `apply_event` for
    // Merge must combine stats correctly so the witness check on the
    // subsequent BeliefUpdate matches.
    let mut g = graph_with_two_observed_nodes();
    g.force_merge(2, 1).unwrap();
    g.observe(1, 99.0).unwrap();
    g.verify_chain().expect("chain valid after merge + observe");

    let blob = serialize(&g);
    let g2 = replay(&blob).expect("replay accepts merge-then-observe");
    g2.verify_chain().expect("replay chain valid");
    assert_eq!(g2.chain_head, g.chain_head);
}

#[test]
fn merge_followed_by_blr_update_chain_verifies() {
    let mut g = graph_with_two_observed_nodes();
    g.force_merge(2, 1).unwrap();
    // Continue training the merged node — BLR update must verify
    // against the combined feature_version_hash (which is preserved
    // from `into` across combine).
    g.blr_update(1, &[3.0, 1.5, 1.5, 3.0], &[3.0, 3.0]).unwrap();
    g.verify_chain().expect("chain valid after merge + blr_update");

    let blob = serialize(&g);
    let g2 = replay(&blob).expect("replay accepts merge-then-blr_update");
    g2.verify_chain().expect("replay chain valid");
}

#[test]
fn merge_self_errs() {
    let mut g = graph_with_two_observed_nodes();
    let err = g.force_merge(1, 1).unwrap_err();
    assert!(matches!(
        err,
        cjc_abng::graph::GraphError::ForceMergeSelf
    ));
}

#[test]
fn merge_already_inactive_errs() {
    let mut g = graph_with_two_observed_nodes();
    g.force_merge(2, 1).unwrap();
    let err = g.force_merge(2, 1).unwrap_err();
    assert!(matches!(
        err,
        cjc_abng::graph::GraphError::ForceMergeAlreadyAbsorbed { node_id: 2 }
    ));
}

#[test]
fn double_merge_keeps_into_active_and_grows_evidence() {
    // A → into; B → into. After both, into.n_seen sums all three.
    let mut g = graph_with_two_observed_nodes();
    let _c = g.add_node(0, 3).unwrap();
    for v in &[100.0, 200.0, 300.0, 400.0] {
        g.observe(3, *v).unwrap();
    }
    let n_root = g.nodes[1].stats.n_seen;
    let n_b = g.nodes[2].stats.n_seen;
    let n_c = g.nodes[3].stats.n_seen;

    g.force_merge(2, 1).unwrap();
    g.force_merge(3, 1).unwrap();
    assert_eq!(g.nodes[1].stats.n_seen, n_root + n_b + n_c);
    assert!(g.nodes[1].is_active);
    assert!(!g.nodes[2].is_active);
    assert!(!g.nodes[3].is_active);
}
