//! Phase 0.3d-2 — graph-level integration tests for the
//! `expected_epistemic` capture and the calibrated OOD formula.

use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::AuditKind;
use cjc_ad::pinn::Activation;

fn graph_with_blr() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(7);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g
}

#[test]
fn capture_emits_expected_epistemic_event() {
    let mut g = graph_with_blr();
    let pre = g.audit.len();
    g.set_expected_epistemic(0, 0.5).unwrap();
    assert_eq!(g.audit.len(), pre + 1);
    assert!(matches!(
        g.audit.last().unwrap().kind,
        AuditKind::ExpectedEpistemicCaptured { .. }
    ));
}

#[test]
fn capture_advances_chain_head() {
    let mut g = graph_with_blr();
    let pre = g.chain_head;
    g.set_expected_epistemic(0, 0.5).unwrap();
    assert_ne!(pre, g.chain_head);
    assert!(g.verify_chain().is_ok());
}

#[test]
fn second_capture_errs() {
    let mut g = graph_with_blr();
    g.set_expected_epistemic(0, 0.5).unwrap();
    let err = g.set_expected_epistemic(0, 0.7).unwrap_err();
    assert!(matches!(
        err,
        GraphError::ExpectedEpistemicAlreadyCaptured { node_id: 0 }
    ));
}

#[test]
fn capture_without_blr_errs() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let err = g.set_expected_epistemic(0, 0.5).unwrap_err();
    assert!(matches!(err, GraphError::ExpectedEpistemicNoBlr));
}

#[test]
fn capture_invalid_value_errs() {
    let mut g = graph_with_blr();
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let err = g.set_expected_epistemic(0, bad).unwrap_err();
        assert!(
            matches!(err, GraphError::ExpectedEpistemicInvalidValue(_)),
            "expected InvalidValue, got {err:?} for {bad}"
        );
    }
}

#[test]
fn capture_per_node_independent() {
    let mut g = graph_with_blr();
    let c = g.add_node(0, 7).unwrap();
    g.set_expected_epistemic(0, 0.4).unwrap();
    assert_eq!(g.expected_epistemic(0).unwrap(), Some(0.4));
    assert_eq!(g.expected_epistemic(c).unwrap(), None);
    g.set_expected_epistemic(c, 0.9).unwrap();
    assert_eq!(g.expected_epistemic(c).unwrap(), Some(0.9));
    // The earlier node's value is untouched.
    assert_eq!(g.expected_epistemic(0).unwrap(), Some(0.4));
}

#[test]
fn ood_score_calibrated_branch_active() {
    // After capture, ood_score's epistemic_z component uses the ratio
    // formula. With a small captured reference (much less than the
    // current epistemic var) the ratio saturates at 1.0.
    let mut g = graph_with_blr();
    // Train BLR a bit so its predict() returns finite epistemic var.
    g.blr_update(0, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[1.0, 2.0, 3.0])
        .unwrap();
    let phi = vec![5.0, 5.0]; // far-from-training point → high epi var
    g.set_expected_epistemic(0, 1e-6).unwrap();
    let s = g.ood_score(0, &phi, 0, 0).unwrap();
    // With a tiny reference, calibrated z saturates at 1.0; OOD = max
    // over components, so s == 1.0.
    assert!(s >= 0.999, "expected saturated OOD, got {s}");
}

#[test]
fn snapshot_round_trip_preserves_capture() {
    let mut g = graph_with_blr();
    g.set_expected_epistemic(0, 0.31).unwrap();
    let blob = cjc_abng::serialize::serialize(&g);
    let g2 = cjc_abng::serialize::replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(
        g2.expected_epistemic(0).unwrap(),
        Some(0.31)
    );
}

#[test]
fn determinism_double_run_capture_chain_head() {
    let mk = || {
        let mut g = graph_with_blr();
        g.set_expected_epistemic(0, 0.42).unwrap();
        g.chain_head
    };
    assert_eq!(mk(), mk());
}
