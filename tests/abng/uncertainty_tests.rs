//! Phase 0.3c — pure-Rust integration tests of the OOD / calibration /
//! drift subsystems on top of the Phase 0.3a/b stack.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

fn build_full_stack() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    g
}

#[test]
fn full_stack_install_emits_expected_events() {
    let g = build_full_stack();
    // Created + LeafHeadConfigured + LeafParamsInitialized
    //   + BlrPriorConfigured + BlrInitialized
    //   + DensityTrackerInstalled + CalibrationInstalled = 7 events
    assert_eq!(g.audit_len(), 7);
}

#[test]
fn density_observe_updates_score_and_emits_event() {
    let mut g = build_full_stack();
    let pre = g.audit_len();
    g.density_observe(0, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
    assert_eq!(g.density_n_seen(0).unwrap(), 3);
    let s_at_mean = g.density_score(0, &[2.0, 2.0]).unwrap();
    let s_far = g.density_score(0, &[100.0, 100.0]).unwrap();
    assert!(s_far > s_at_mean);
    let last = g.audit.last().unwrap();
    assert!(matches!(last.kind, AuditKind::DensityUpdated { .. }));
    assert_eq!(g.audit_len(), pre + 1);
}

#[test]
fn calibration_observe_updates_ece_and_emits_event() {
    let mut g = build_full_stack();
    // Predict 0.9 every time, only 10% correct → high ECE.
    for i in 0..50 {
        g.calibration_observe(0, 0.9, i % 10 == 0).unwrap();
    }
    let ece = g.calibration_ece(0).unwrap();
    assert!(ece > 0.7, "ECE should be ~0.8, got {ece}");
    assert_eq!(g.calibration_n_seen(0).unwrap(), 50);
}

#[test]
fn drift_baseline_freeze_then_score() {
    let mut g = build_full_stack();
    g.density_observe(0, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        .unwrap();
    g.freeze_drift_baseline(0).unwrap();
    let s0 = g.drift_score(0).unwrap();
    assert!(s0.abs() < 1e-12, "drift at freeze should be 0, got {s0}");
    // Shift the live density tracker.
    g.density_observe(
        0,
        &[10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0],
    )
    .unwrap();
    let s1 = g.drift_score(0).unwrap();
    assert!(s1 > s0, "drift score should rise after shift");
}

#[test]
fn freeze_baseline_without_density_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    // No density tracker installed → freeze fails.
    let err = g.freeze_drift_baseline(0).unwrap_err();
    assert!(matches!(err, GraphError::Drift(_)));
}

#[test]
fn ood_score_is_max_of_components() {
    let mut g = build_full_stack();
    // Train density tracker on a tight cluster.
    for _ in 0..20 {
        g.density_observe(0, &[0.0, 0.0]).unwrap();
    }
    let s_inside = g.ood_score(0, &[0.0, 0.0], 5, 5).unwrap();
    // Component breakdown:
    //   density_score ≈ 0     (at the mean)
    //   prefix_distance = 0   (matched=5, max=5)
    //   epistemic_z ≈ small   (BLR posterior is broad initially)
    // So OOD ≈ small.
    let s_outside = g.ood_score(0, &[100.0, 100.0], 0, 5).unwrap();
    // density_score → ~1, prefix_distance = 1, OOD = max ≈ 1.
    assert!(s_outside > s_inside);
    assert!(s_outside > 0.99);
}

#[test]
fn density_score_zero_when_n_lt_two() {
    let g = build_full_stack();
    let s = g.density_score(0, &[1.0, 1.0]).unwrap();
    assert_eq!(s, 0.0);
}

#[test]
fn full_stack_round_trip_byte_identical() {
    let mk = || {
        let mut g = build_full_stack();
        g.density_observe(0, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        g.calibration_observe(0, 0.7, true).unwrap();
        g.calibration_observe(0, 0.3, false).unwrap();
        g.freeze_drift_baseline(0).unwrap();
        g
    };
    let g1 = mk();
    let g2 = mk();
    assert_eq!(serialize(&g1), serialize(&g2));

    let blob = serialize(&g1);
    let g3 = replay(&blob).unwrap();
    assert_eq!(g1.chain_head, g3.chain_head);
    assert_eq!(g1.density_n_seen(0).unwrap(), g3.density_n_seen(0).unwrap());
    assert_eq!(
        g1.calibration_n_seen(0).unwrap(),
        g3.calibration_n_seen(0).unwrap()
    );
    let s1 = g1.drift_score(0).unwrap();
    let s3 = g3.drift_score(0).unwrap();
    assert!((s1 - s3).abs() < 1e-12);
}

#[test]
fn install_ordering_density_requires_head() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.set_density_tracker().unwrap_err();
    assert!(matches!(err, GraphError::Density(_)));
}

#[test]
fn install_ordering_calibration_blocks_after_add_node() {
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.add_node(0, 1).unwrap();
    let err = g.set_calibration(15).unwrap_err();
    assert!(matches!(err, GraphError::Calibration(_)));
}

#[test]
fn chain_verifies_after_full_stack_workflow() {
    let mut g = build_full_stack();
    for i in 0..20 {
        let phi = [(i as f64) * 0.1, (i as f64) * 0.05];
        g.density_observe(0, &phi).unwrap();
        g.calibration_observe(0, 0.5 + (i as f64) * 0.01, i % 2 == 0).unwrap();
    }
    g.freeze_drift_baseline(0).unwrap();
    assert!(g.verify_chain().is_ok());
}

#[test]
fn child_nodes_inherit_density_and_calibration() {
    let mut g = build_full_stack();
    let n1 = g.add_node(0, 5).unwrap();
    assert!(g.nodes[n1 as usize].density.is_some());
    assert!(g.nodes[n1 as usize].calibration.is_some());
    // Per-child events: NodeAdded + LeafParamsInitialized + BlrInitialized
    //   + DensityTrackerInstalled + CalibrationInstalled (+ ChildrenPromoted)
    let n_inits_for_child: usize = g
        .audit
        .iter()
        .filter(|e| {
            e.node_id == n1
                && matches!(
                    e.kind,
                    AuditKind::DensityTrackerInstalled { .. }
                        | AuditKind::CalibrationInstalled { .. }
                )
        })
        .count();
    assert_eq!(n_inits_for_child, 2);
}
