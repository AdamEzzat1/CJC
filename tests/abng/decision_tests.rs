//! Phase 0.3d-3 — graph-level integration tests for `DecisionPolicy`,
//! the six force-* structural mutations, and the Dense children variant.

use cjc_abng::children::{AdaptiveChildren, ChildrenKind};
use cjc_abng::graph::{ActionKind, AdaptiveBeliefGraph, GraphError, N_ACTION_KINDS};
use cjc_abng::AuditKind;

fn ok_thresholds() -> [f64; 11] {
    [
        0.5, 64.0, 128.0, 0.05, 0.02,
        4.0, 0.1, 32.0, 10.0, 8.0,
        20.0,
    ]
}

// ─── DecisionPolicy install ──────────────────────────────────────

#[test]
fn install_decision_policy_records_hash() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    let h = g.decision_policy_hash().unwrap();
    assert_ne!(h, [0u8; 32]);
}

#[test]
fn install_decision_policy_one_shot_via_graph() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    let err = g.set_decision_policy(&ok_thresholds()).unwrap_err();
    assert_eq!(err, GraphError::DecisionPolicyAlreadyFrozen);
}

#[test]
fn install_decision_policy_validates_threshold_count() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.set_decision_policy(&[0.0]).unwrap_err();
    assert!(matches!(err, GraphError::DecisionPolicy(_)));
}

#[test]
fn install_decision_policy_after_evidence_is_allowed() {
    // Unlike other one-shot installs, decision_policy is permitted
    // after add_node — test pins this contract.
    let mut g = AdaptiveBeliefGraph::new(0);
    let _c = g.add_node(0, 7).unwrap();
    g.set_decision_policy(&ok_thresholds()).unwrap();
    assert!(g.decision_policy.is_some());
}

// ─── force_grow ──────────────────────────────────────────────────

#[test]
fn force_grow_creates_child_at_next_index() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let c = g.force_grow(0, 7).unwrap();
    assert_eq!(c, 1);
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.action_count(ActionKind::Grow), 1);
}

#[test]
fn force_grow_carries_subsystems_to_new_child() {
    use cjc_ad::pinn::Activation;
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    let c = g.force_grow(0, 7).unwrap();
    assert!(!g.nodes[c as usize].params.is_empty());
    assert!(g.nodes[c as usize].blr.is_some());
    assert!(g.nodes[c as usize].density.is_some());
    assert!(g.nodes[c as usize].calibration.is_some());
}

#[test]
fn force_grow_emits_promoted_grow_then_subsystem_inits() {
    use cjc_ad::pinn::Activation;
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    let pre_len = g.audit.len();
    let _ = g.force_grow(0, 7).unwrap();
    let new_events = &g.audit[pre_len..];
    // Root's children promoted None → Node4 → ChildrenPromoted fires
    // first. Then Grow, then LeafParamsInitialized.
    assert!(matches!(
        new_events[0].kind,
        AuditKind::ChildrenPromoted { .. }
    ));
    assert!(matches!(new_events[1].kind, AuditKind::Grow { .. }));
    assert!(matches!(
        new_events[2].kind,
        AuditKind::LeafParamsInitialized { .. }
    ));
}

#[test]
fn force_grow_key_already_bound_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_grow(0, 7).unwrap();
    let err = g.force_grow(0, 7).unwrap_err();
    assert!(matches!(
        err,
        GraphError::KeyAlreadyBound { parent: 0, key_byte: 7 }
    ));
}

// ─── force_split ─────────────────────────────────────────────────

#[test]
fn force_split_produces_two_distinct_children() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let (a, b) = g.force_split(0).unwrap();
    assert_ne!(a, b);
    assert_eq!(a, 1);
    assert_eq!(b, 2);
    assert_eq!(g.node_count(), 3);
    assert_eq!(g.action_count(ActionKind::Split), 1);
}

#[test]
fn force_split_deterministic_keys_from_seed() {
    let (g1_a, g1_b) = {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.force_split(0).unwrap()
    };
    let (g2_a, g2_b) = {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.force_split(0).unwrap()
    };
    assert_eq!(g1_a, g2_a);
    assert_eq!(g1_b, g2_b);
}

#[test]
fn force_split_blocked_on_non_leaf() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.add_node(0, 7).unwrap();
    let err = g.force_split(0).unwrap_err();
    assert!(matches!(err, GraphError::ForceSplitNotLeaf { .. }));
}

#[test]
fn force_split_blocked_on_frozen() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_freeze(0).unwrap();
    let err = g.force_split(0).unwrap_err();
    assert!(matches!(err, GraphError::NodeFrozen { node_id: 0 }));
}

// ─── force_merge ─────────────────────────────────────────────────

#[test]
fn force_merge_marks_absorbed_inactive_keeps_into_active() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let a = g.add_node(0, 1).unwrap();
    let b = g.add_node(0, 2).unwrap();
    g.force_merge(a, b).unwrap();
    assert!(!g.is_active(a).unwrap());
    assert!(g.is_active(b).unwrap());
    assert_eq!(g.action_count(ActionKind::Merge), 1);
}

#[test]
fn force_merge_self_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.force_merge(0, 0).unwrap_err();
    assert_eq!(err, GraphError::ForceMergeSelf);
}

#[test]
fn force_merge_already_absorbed_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let a = g.add_node(0, 1).unwrap();
    g.force_prune(a).unwrap();
    let err = g.force_merge(a, 0).unwrap_err();
    assert!(matches!(
        err,
        GraphError::ForceMergeAlreadyAbsorbed { .. }
    ));
}

#[test]
fn force_merge_into_frozen_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let a = g.add_node(0, 1).unwrap();
    g.force_freeze(0).unwrap();
    let err = g.force_merge(a, 0).unwrap_err();
    assert!(matches!(err, GraphError::NodeFrozen { node_id: 0 }));
}

// ─── force_prune ─────────────────────────────────────────────────

#[test]
fn force_prune_marks_inactive() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let c = g.add_node(0, 1).unwrap();
    g.force_prune(c).unwrap();
    assert!(!g.is_active(c).unwrap());
    assert_eq!(g.action_count(ActionKind::Prune), 1);
}

#[test]
fn force_prune_already_inactive_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_prune(0).unwrap();
    let err = g.force_prune(0).unwrap_err();
    assert!(matches!(err, GraphError::NodeNotActive { node_id: 0 }));
}

#[test]
fn force_prune_frozen_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_freeze(0).unwrap();
    let err = g.force_prune(0).unwrap_err();
    assert!(matches!(err, GraphError::NodeFrozen { node_id: 0 }));
}

// ─── force_compress ──────────────────────────────────────────────

#[test]
fn force_compress_replaces_children_with_dense() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.add_node(0, 1).unwrap();
    g.add_node(0, 2).unwrap();
    g.force_compress(0).unwrap();
    assert!(matches!(
        g.nodes[0].children,
        AdaptiveChildren::Dense { .. }
    ));
    assert_eq!(g.nodes[0].children.kind(), ChildrenKind::Dense);
    assert_eq!(g.action_count(ActionKind::Compress), 1);
}

#[test]
fn force_compress_orphans_descendants_in_arena() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let a = g.add_node(0, 1).unwrap();
    let b = g.add_node(0, 2).unwrap();
    let n_before = g.node_count();
    g.force_compress(0).unwrap();
    // Descendants persist in the arena per §7 #9 (no reordering).
    assert_eq!(g.node_count(), n_before);
    assert!(g.is_active(a).unwrap());
    assert!(g.is_active(b).unwrap());
    // But routing through the parent loses them.
    assert_eq!(g.nodes[0].children.iter().len(), 0);
}

#[test]
fn force_compress_on_dense_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_compress(0).unwrap();
    let err = g.force_compress(0).unwrap_err();
    assert!(matches!(err, GraphError::NodeIsDense { .. }));
}

// ─── force_freeze ────────────────────────────────────────────────

#[test]
fn force_freeze_idempotent_no_extra_event() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_freeze(0).unwrap();
    let pre_len = g.audit.len();
    g.force_freeze(0).unwrap();
    // Idempotent — no new audit event, no counter bump.
    assert_eq!(g.audit.len(), pre_len);
    assert_eq!(g.action_count(ActionKind::Freeze), 1);
}

#[test]
fn force_freeze_blocks_subsequent_grow() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_freeze(0).unwrap();
    let err = g.force_grow(0, 7).unwrap_err();
    assert!(matches!(err, GraphError::NodeFrozen { node_id: 0 }));
}

// ─── inspection: action_count / is_frozen ────────────────────────

#[test]
fn action_count_zero_for_fresh_graph() {
    let g = AdaptiveBeliefGraph::new(0);
    for kind in 0..N_ACTION_KINDS as u8 {
        let k = ActionKind::from_index(kind).unwrap();
        assert_eq!(g.action_count(k), 0);
    }
}

#[test]
fn is_frozen_default_false() {
    let g = AdaptiveBeliefGraph::new(0);
    assert!(!g.is_frozen(0).unwrap());
}

#[test]
fn is_frozen_after_freeze() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.force_freeze(0).unwrap();
    assert!(g.is_frozen(0).unwrap());
}

// ─── audit chain integrity ───────────────────────────────────────

#[test]
fn chain_verifies_after_full_p3d3_workflow() {
    use cjc_ad::pinn::Activation;
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_decision_policy(&ok_thresholds()).unwrap();
    let c1 = g.force_grow(0, 1).unwrap();
    let _c2 = g.force_grow(0, 2).unwrap();
    let (_a, _b) = g.force_split(c1).unwrap();
    let pruned = g.force_grow(0, 9).unwrap();
    g.force_prune(pruned).unwrap();
    g.force_freeze(0).unwrap();
    assert!(g.verify_chain().is_ok());
}

#[test]
fn round_trip_after_force_actions_byte_identical() {
    use cjc_ad::pinn::Activation;
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_decision_policy(&ok_thresholds()).unwrap();
    let c1 = g.force_grow(0, 1).unwrap();
    let _ = g.force_freeze(c1).unwrap();
    let blob1 = cjc_abng::serialize::serialize(&g);
    let g2 = cjc_abng::serialize::replay(&blob1).unwrap();
    let blob2 = cjc_abng::serialize::serialize(&g2);
    assert_eq!(blob1, blob2);
}

#[test]
fn determinism_double_run_full_p3d3_workflow() {
    // Split first (root must be a leaf), then grow under one leaf,
    // then freeze the other.
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let (a, b) = g.force_split(0).unwrap();
        let _ = g.force_grow(a, 33).unwrap();
        g.force_freeze(b).unwrap();
        (g.chain_head, g.action_counts)
    };
    assert_eq!(mk(), mk());
}

// ─── Phase 0.3d-4 — decide_step + unfreeze ───────────────────────

#[test]
fn decide_step_no_op_without_policy() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let counts = g.decide_step();
    assert_eq!(counts, [0u64; 6]);
}

#[test]
fn decide_step_idle_graph_increments_stability_then_freezes() {
    // freeze_after = 20 in ok_thresholds. After 21 calls (1 capture +
    // 20 stable advances), Freeze should fire on the root.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    let mut total_freeze = 0u64;
    for _ in 0..25 {
        let counts = g.decide_step();
        total_freeze += counts[ActionKind::Freeze as usize];
    }
    assert_eq!(total_freeze, 1, "Freeze should have fired exactly once");
    assert!(g.is_frozen(0).unwrap());
}

#[test]
fn decide_step_grow_after_observations() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    for _ in 0..70 {
        g.observe(0, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(counts[ActionKind::Grow as usize], 1);
    assert_eq!(g.node_count(), 2);
}

#[test]
fn decide_step_split_when_samples_high() {
    // split_min = 128. Observe 130 times then decide_step. But Grow
    // fires first (grow_min = 64) since both triggers are active and
    // Grow comes after Split in the fall-through. Wait — re-read
    // prompt §2.6: order is Compress, Merge, Split, Prune, Grow,
    // Freeze. Split before Grow, so Split should fire when both are
    // eligible. Confirm.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    for _ in 0..130 {
        g.observe(0, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(counts[ActionKind::Split as usize], 1);
    // After Split, Grow does NOT fire on the same node in the same call.
    assert_eq!(counts[ActionKind::Grow as usize], 0);
}

#[test]
fn decide_step_skips_frozen_node() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    g.force_freeze(0).unwrap();
    for _ in 0..70 {
        g.observe(0, 1.0).unwrap();
    }
    let counts = g.decide_step();
    // No Grow despite samples_seen >= grow_min — because root is frozen.
    assert_eq!(counts[ActionKind::Grow as usize], 0);
}

#[test]
fn decide_step_unfreeze_then_action_resumes() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    g.force_freeze(0).unwrap();
    for _ in 0..70 {
        g.observe(0, 1.0).unwrap();
    }
    // Frozen → no Grow.
    assert_eq!(g.decide_step()[ActionKind::Grow as usize], 0);
    // Unfreeze → Grow can fire on next call.
    g.unfreeze(0).unwrap();
    assert_eq!(g.decide_step()[ActionKind::Grow as usize], 1);
}

#[test]
fn decide_step_chain_verifies_after_many_passes() {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    for _ in 0..70 {
        g.observe(0, 1.0).unwrap();
    }
    for _ in 0..10 {
        g.decide_step();
    }
    assert!(g.verify_chain().is_ok());
}

#[test]
fn decide_step_round_trip_byte_identical() {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_decision_policy(&ok_thresholds()).unwrap();
    for _ in 0..70 {
        g.observe(0, 1.0).unwrap();
    }
    for _ in 0..3 {
        g.decide_step();
    }
    let blob1 = cjc_abng::serialize::serialize(&g);
    let g2 = cjc_abng::serialize::replay(&blob1).unwrap();
    let blob2 = cjc_abng::serialize::serialize(&g2);
    assert_eq!(blob1, blob2);
}

#[test]
fn decide_step_auto_captures_expected_epistemic_at_uncertainty_stable() {
    use cjc_ad::pinn::Activation;
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    // Use an isolating policy: grow_min and split_min are sky-high so
    // those triggers don't fire during the stability-accumulation
    // phase; we want auto-capture to be the ONLY thing happening.
    let isolating_thresholds: [f64; 11] = [
        0.5,         // H_grow
        1.0e9,       // grow_min — effectively disabled
        1.0e9,       // split_min — effectively disabled
        0.05,        // nll_split_gain
        0.02,        // impurity_min
        4.0,         // tau_merge
        0.1,         // kl_merge
        0.0,         // prune_floor — never prune
        1.0e9,       // prune_grace_epochs — effectively disabled
        8.0,         // tau_compress
        1.0e9,       // freeze_after — effectively disabled
    ];
    g.set_decision_policy(&isolating_thresholds).unwrap();
    // Train BLR so its posterior mean is non-zero — otherwise
    // epistemic_var at posterior_mean = epistemic_var at origin = 0
    // and auto-capture rejects it as degenerate.
    g.blr_update(0, &[1.0, 0.5, 2.0, 1.5, 3.0, 2.5], &[1.0, 2.0, 3.0])
        .unwrap();
    // Drive samples_seen above UNCERTAINTY_STABLE_MIN_SAMPLES (= 100).
    for _ in 0..150 {
        g.observe(0, 1.0).unwrap();
    }
    // First call: captures last_signature, stable_calls = 0 →
    //             uncertainty_stable still false → no auto-capture.
    g.decide_step();
    assert_eq!(g.expected_epistemic(0).unwrap(), None);
    // Second call: stable_calls = 1 → uncertainty_stable = true →
    //              auto-capture fires.
    g.decide_step();
    assert!(g.expected_epistemic(0).unwrap().is_some());
}

#[test]
fn unfreeze_on_active_is_no_op() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let pre = g.audit.len();
    g.unfreeze(0).unwrap();
    assert_eq!(g.audit.len(), pre);
}

#[test]
fn decide_step_determinism_with_observations() {
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(123);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        for i in 0..70 {
            g.observe(0, i as f64).unwrap();
        }
        for _ in 0..5 {
            g.decide_step();
        }
        (g.chain_head, g.action_counts, g.node_count())
    };
    assert_eq!(mk(), mk());
}
