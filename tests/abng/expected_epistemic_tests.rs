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
    // current epistemic leverage) the ratio saturates at 1.0.
    let mut g = graph_with_blr();
    // Train BLR a bit so its predict() returns finite epistemic
    // leverage (the dimensionless `‖L⁻¹φ‖²`; see C-2.3.1).
    g.blr_update(0, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[1.0, 2.0, 3.0])
        .unwrap();
    let phi = vec![5.0, 5.0]; // far-from-training point → high leverage
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

// ── Phase 0.4 Track C-2.3.12 — force_recapture_expected_epistemic ──

/// Helper that trains the root BLR with one batch so
/// `epistemic_leverage_at_posterior_mean` returns Some(positive).
fn graph_with_blr_trained() -> AdaptiveBeliefGraph {
    let mut g = graph_with_blr();
    g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
    g
}

#[test]
fn recapture_from_uncaptured_state_succeeds() {
    let mut g = graph_with_blr_trained();
    assert_eq!(g.expected_epistemic(0).unwrap(), None);
    let v = g.force_recapture_expected_epistemic(0).unwrap();
    assert!(v > 0.0 && v.is_finite());
    assert_eq!(g.expected_epistemic(0).unwrap(), Some(v));
}

#[test]
fn recapture_overwrites_existing_value() {
    // Capture initial value manually, then force-recapture: the new
    // value must be the deterministic-from-current-state leverage,
    // overwriting the manual value.
    let mut g = graph_with_blr_trained();
    g.set_expected_epistemic(0, 0.123).unwrap();
    assert_eq!(g.expected_epistemic(0).unwrap(), Some(0.123));
    let v = g.force_recapture_expected_epistemic(0).unwrap();
    assert_ne!(v, 0.123);
    assert_eq!(g.expected_epistemic(0).unwrap(), Some(v));
}

#[test]
fn recapture_emits_capture_event_each_time() {
    let mut g = graph_with_blr_trained();
    let pre = g.audit.len();
    let _ = g.force_recapture_expected_epistemic(0).unwrap();
    assert_eq!(g.audit.len(), pre + 1);
    assert!(matches!(
        g.audit.last().unwrap().kind,
        AuditKind::ExpectedEpistemicCaptured { .. }
    ));
    let _ = g.force_recapture_expected_epistemic(0).unwrap();
    assert_eq!(
        g.audit.len(),
        pre + 2,
        "second recapture must also emit a fresh event"
    );
    assert!(matches!(
        g.audit.last().unwrap().kind,
        AuditKind::ExpectedEpistemicCaptured { .. }
    ));
}

#[test]
fn recapture_reflects_post_blr_drift() {
    // Train, capture, drift posterior via more updates, recapture.
    // The two captured values should differ (posterior moved → leverage
    // at posterior mean shifted).
    let mut g = graph_with_blr_trained();
    let v1 = g.force_recapture_expected_epistemic(0).unwrap();
    // Add many more observations to shift the posterior.
    for _ in 0..5 {
        g.blr_update(0, &[3.0, 1.5, 2.0, 4.0], &[6.0, 8.0]).unwrap();
    }
    let v2 = g.force_recapture_expected_epistemic(0).unwrap();
    assert_ne!(
        v1.to_bits(),
        v2.to_bits(),
        "recapture after training drift should yield a different leverage"
    );
}

#[test]
fn recapture_node_out_of_range_errs() {
    let mut g = graph_with_blr_trained();
    let err = g.force_recapture_expected_epistemic(99).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { node_id: 99, .. }));
}

#[test]
fn recapture_no_blr_errs() {
    // Graph without set_blr_prior — recapture must error.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    let err = g.force_recapture_expected_epistemic(0).unwrap_err();
    assert!(matches!(err, GraphError::ExpectedEpistemicNoBlr));
}

#[test]
fn recapture_no_evidence_errs() {
    // Untrained BLR (n_seen == 0): predict at the prior mean (zero
    // vector) gives leverage = 0 → epistemic_leverage_at_posterior_mean
    // returns None → recapture errors with InvalidValue.
    let mut g = graph_with_blr();
    let err = g.force_recapture_expected_epistemic(0).unwrap_err();
    assert!(matches!(err, GraphError::ExpectedEpistemicInvalidValue(_)));
}

#[test]
fn recapture_determinism_double_run() {
    // Two graphs constructed and recaptured the same way must
    // produce bit-identical chain heads.
    let mk = || {
        let mut g = graph_with_blr_trained();
        let _ = g.force_recapture_expected_epistemic(0).unwrap();
        g.chain_head
    };
    assert_eq!(mk(), mk());
}

#[test]
fn recapture_snapshot_round_trip() {
    let mut g = graph_with_blr_trained();
    let v = g.force_recapture_expected_epistemic(0).unwrap();
    let blob = cjc_abng::serialize::serialize(&g);
    let g2 = cjc_abng::serialize::replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g2.expected_epistemic(0).unwrap(), Some(v));
}
