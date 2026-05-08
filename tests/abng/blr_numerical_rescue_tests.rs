//! Phase 0.4 Track C-2.3.4 — `BlrNumericalRescue` audit kind (tag 0x18).
//!
//! When `BlrState::update` would set `b < f64::EPSILON`, the BLR layer
//! clamps to `f64::EPSILON` (preserves the InverseGamma posterior's
//! well-definedness) and surfaces `Some(b_pre_clamp)` to the graph
//! layer, which appends a `BlrNumericalRescue` event *after* the
//! corresponding `BlrUpdated`. The rescue event is diagnostic-only —
//! `apply_event` is a no-op for it during replay — so determinism is
//! preserved.
//!
//! Triggering the clamp deterministically: the formula
//!   `b_new = b_old + 0.5 * (μᵀΛμ + yᵀy − mᵀΛm)`
//! produces `b_new = b_old` whenever the SSR bracket is zero (i.e.
//! `y == 0` so `m_new == 0` with prior `μ == 0`). With a prior
//! constructed at `b = ε / 2` (already below the threshold), the
//! resulting `b_new < ε` triggers the clamp on every such update.

use cjc_abng::audit::{AuditKind, BLR_RESCUE_B_BELOW_EPSILON};
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

fn graph_with_degenerate_prior() -> AdaptiveBeliefGraph {
    // Tiny prior `b` deliberately below f64::EPSILON so any SSR-zero
    // update lands the clamp.
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, f64::EPSILON / 2.0).unwrap();
    g
}

fn graph_with_healthy_prior() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g
}

#[test]
fn rescue_event_emitted_after_blr_updated_when_clamp_fires() {
    let mut g = graph_with_degenerate_prior();
    let pre_audit = g.audit_len() as usize;

    // y = 0 with prior μ = 0 ⇒ SSR = 0 ⇒ b_pre_clamp = b_old < ε.
    g.blr_update(0, &[1.0, 0.5], &[0.0]).unwrap();

    // Two new events: BlrUpdated then BlrNumericalRescue, in that order.
    assert_eq!(g.audit.len(), pre_audit + 2);
    let n = g.audit.len();
    assert!(matches!(g.audit[n - 2].kind, AuditKind::BlrUpdated { .. }));
    let rescue = &g.audit[n - 1];
    let (reason, b_pre_clamp_bits) = match rescue.kind {
        AuditKind::BlrNumericalRescue {
            reason,
            b_pre_clamp_bits,
        } => (reason, b_pre_clamp_bits),
        ref k => panic!("expected BlrNumericalRescue, got {k:?}"),
    };
    assert_eq!(reason, BLR_RESCUE_B_BELOW_EPSILON);
    let pre = f64::from_bits(b_pre_clamp_bits);
    // pre-clamp value is ≈ b_old (= ε/2); SSR is exactly zero in this
    // construction, so pre is bit-identical to the prior's b.
    assert_eq!(pre, f64::EPSILON / 2.0);

    g.verify_chain().expect("chain still valid after rescue event");
}

#[test]
fn no_rescue_event_on_normal_update() {
    let mut g = graph_with_healthy_prior();
    let pre_audit = g.audit_len() as usize;
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();

    // Healthy prior + non-zero SSR — only BlrUpdated, no rescue.
    assert_eq!(g.audit.len(), pre_audit + 1);
    assert!(matches!(
        g.audit[pre_audit].kind,
        AuditKind::BlrUpdated { .. }
    ));
}

#[test]
fn rescue_events_paired_with_immediately_preceding_blr_updated() {
    // The clamp fires exactly once on the degenerate prior: the first
    // call drops `b` from `ε/2` to `ε`; subsequent SSR=0 calls leave
    // `b_pre_clamp = ε`, which is not strictly less than `ε`, so the
    // clamp does not re-fire. The structural invariant we pin here is:
    // *every* BlrNumericalRescue in the audit log is immediately
    // preceded by a BlrUpdated, regardless of how many fire.
    let mut g = graph_with_degenerate_prior();

    for _ in 0..5 {
        g.blr_update(0, &[1.0, 0.5], &[0.0]).unwrap();
    }

    let n_updated = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BlrUpdated { .. }))
        .count();
    let n_rescue = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BlrNumericalRescue { .. }))
        .count();
    assert_eq!(n_updated, 5, "one BlrUpdated per blr_update call");
    assert_eq!(
        n_rescue, 1,
        "clamp fires only on the first call (b drops ε/2 → ε; \
         subsequent calls have b == ε so b_pre_clamp == ε is not < ε)"
    );

    // Every BlrNumericalRescue immediately follows a BlrUpdated.
    for (i, e) in g.audit.iter().enumerate() {
        if matches!(e.kind, AuditKind::BlrNumericalRescue { .. }) {
            assert!(i > 0, "rescue event cannot be first");
            assert!(
                matches!(g.audit[i - 1].kind, AuditKind::BlrUpdated { .. }),
                "rescue event must follow BlrUpdated, got {:?}",
                g.audit[i - 1].kind
            );
        }
    }
    g.verify_chain().expect("chain still valid");
}

#[test]
fn rescue_event_round_trip_byte_identical() {
    // Snapshot a graph that emitted at least one rescue event;
    // replay must reproduce the chain head and audit log byte-for-byte
    // (the rescue event participates in chain hashing despite being
    // diagnostic-only at apply_event time).
    let mut g = graph_with_degenerate_prior();
    g.blr_update(0, &[1.0, 0.5], &[0.0]).unwrap(); // first call clamps
    g.blr_update(0, &[2.0, 1.0], &[0.0]).unwrap(); // subsequent: no clamp
    let head_before = g.chain_head;
    let audit_before = g.audit_len();
    let rescue_before = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BlrNumericalRescue { .. }))
        .count();
    assert!(rescue_before >= 1, "test premise: clamp fired");

    let blob1 = serialize(&g);
    let g2 = replay(&blob1).expect("replays cleanly");
    assert_eq!(g2.chain_head, head_before);
    assert_eq!(g2.audit_len(), audit_before);

    // Re-serialize and confirm byte equality (Phase 0.4 frozen-magic
    // contract: every write is bit-identical to its source blob).
    let blob2 = serialize(&g2);
    assert_eq!(blob1, blob2);

    // Same number of rescue events on both sides.
    let rescue_after = g2
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BlrNumericalRescue { .. }))
        .count();
    assert_eq!(rescue_after, rescue_before);
}

#[test]
fn rescue_event_apply_event_is_no_op() {
    // `apply_event` for BlrNumericalRescue must not mutate state. The
    // BLR posterior after replay must equal the BLR posterior on the
    // original graph (already covered transitively by the
    // round-trip-byte-identical test, but pinned here for clarity).
    let mut g = graph_with_degenerate_prior();
    g.blr_update(0, &[1.0, 0.5], &[0.0]).unwrap();
    let pre_b = g.nodes[0].blr.as_ref().unwrap().b;
    let pre_n_seen = g.nodes[0].blr.as_ref().unwrap().n_seen;

    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    let post_b = g2.nodes[0].blr.as_ref().unwrap().b;
    let post_n_seen = g2.nodes[0].blr.as_ref().unwrap().n_seen;
    assert_eq!(pre_b.to_bits(), post_b.to_bits());
    assert_eq!(pre_n_seen, post_n_seen);
}

#[test]
fn rescue_event_does_not_advance_blr_n_seen() {
    // The rescue event itself is metadata; only the immediately-prior
    // BlrUpdated is responsible for advancing `n_seen`.
    let mut g = graph_with_degenerate_prior();
    let n_before = g.blr_n_seen(0).unwrap();
    g.blr_update(0, &[1.0, 0.5], &[0.0]).unwrap();
    let n_after = g.blr_n_seen(0).unwrap();
    // One row in the y=[0] update advances n_seen by 1 (rescue itself
    // adds nothing).
    assert_eq!(n_after, n_before + 1);
}

#[test]
fn healthy_graph_replay_unchanged_by_phase_0_4_changes() {
    // Regression: a graph that never triggers the clamp produces a
    // blob with no 0x18 events, byte-identical to what pre-0.4 code
    // would produce (assuming the rest of the wire format is
    // unchanged). This pins that the audit log of a normal training
    // run gains zero new events from C-2.3.4.
    let mut g = graph_with_healthy_prior();
    for v in [1.0, 2.0, 3.0, 4.0] {
        g.blr_update(0, &[v, v * 0.5], &[v]).unwrap();
    }
    let n_rescue = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::BlrNumericalRescue { .. }))
        .count();
    assert_eq!(n_rescue, 0, "healthy training emits zero rescue events");
}
