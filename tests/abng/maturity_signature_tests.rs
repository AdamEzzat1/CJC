//! Phase 0.3d-1 — pure-Rust graph-method tests for `Maturity` and
//! `NodeSignature` accessors on [`AdaptiveBeliefGraph`].

use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use cjc_abng::maturity::{Maturity, MAX_TRUST_LEVEL};
use cjc_abng::signature::{NodeSignature, SIGNATURE_LEN};
use cjc_ad::pinn::Activation;

#[test]
fn node_maturity_root_initial_state() {
    let g = AdaptiveBeliefGraph::new(7);
    let m = g.node_maturity(0).unwrap();
    assert_eq!(m.samples_seen, 0);
    assert!(!m.calibration_stable);
    assert!(!m.uncertainty_stable);
    assert_eq!(m.trust_level, 0);
}

#[test]
fn node_maturity_climbs_with_observations() {
    let mut g = AdaptiveBeliefGraph::new(0);
    for _ in 0..192 {
        g.observe(0, 1.0).unwrap();
    }
    let m = g.node_maturity(0).unwrap();
    assert_eq!(m.samples_seen, 192);
    assert_eq!(m.trust_level, 3);
}

#[test]
fn node_maturity_trust_caps_at_max() {
    let mut g = AdaptiveBeliefGraph::new(0);
    // 256 samples → trust_level == 4. Adding more does not exceed the cap.
    for _ in 0..(MAX_TRUST_LEVEL as u64 * 64 + 100) {
        g.observe(0, 1.0).unwrap();
    }
    let m = g.node_maturity(0).unwrap();
    assert_eq!(m.trust_level, MAX_TRUST_LEVEL);
}

#[test]
fn node_maturity_out_of_range_errs() {
    let g = AdaptiveBeliefGraph::new(0);
    let err = g.node_maturity(99).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
}

#[test]
fn node_maturity_stub_flags_remain_false() {
    // Pin the 0.3d-1 stub: even after lots of evidence the stability
    // flags stay false. 0.3d-2/4 will flip these and force this test
    // to be updated deliberately.
    let mut g = AdaptiveBeliefGraph::new(42);
    g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
    g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    g.set_calibration(15).unwrap();
    for i in 0..100 {
        g.observe(0, i as f64).unwrap();
        g.calibration_observe(0, 0.5, true).unwrap();
    }
    let m = g.node_maturity(0).unwrap();
    assert!(!m.calibration_stable);
    assert!(!m.uncertainty_stable);
}

#[test]
fn node_maturity_per_node_independent() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let c = g.add_node(0, 7).unwrap();
    g.observe(0, 1.0).unwrap();
    g.observe(0, 2.0).unwrap();
    g.observe(c, 5.0).unwrap();
    let m_root = g.node_maturity(0).unwrap();
    let m_child = g.node_maturity(c).unwrap();
    assert_eq!(m_root.samples_seen, 2);
    assert_eq!(m_child.samples_seen, 1);
}

#[test]
fn node_signature_root_initial_state() {
    let g = AdaptiveBeliefGraph::new(0);
    let s = g.node_signature(0).unwrap();
    assert_eq!(s.prediction, [0u8; 8]);
    assert_eq!(s.uncertainty, [0u8; 8]);
    assert_eq!(s.calibration, [0u8; 8]);
    assert_ne!(s.routing, [0u8; 8]);
}

#[test]
fn node_signature_out_of_range_errs() {
    let g = AdaptiveBeliefGraph::new(0);
    let err = g.node_signature(99).unwrap_err();
    assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
}

#[test]
fn node_signature_canonical_bytes_size() {
    let g = AdaptiveBeliefGraph::new(0);
    let s = g.node_signature(0).unwrap();
    assert_eq!(s.canonical_bytes().len(), SIGNATURE_LEN);
}

#[test]
fn node_signature_distinguishes_subsystems() {
    let bare = NodeSignature::from_node(&AdaptiveBeliefGraph::new(0).nodes[0]);

    let mut with_head = AdaptiveBeliefGraph::new(0);
    with_head
        .set_leaf_head(2, vec![], 1, Activation::None)
        .unwrap();
    let s_head = with_head.node_signature(0).unwrap();
    assert_ne!(s_head.prediction, bare.prediction);
    assert_eq!(s_head.uncertainty, [0u8; 8]);

    let mut with_blr = AdaptiveBeliefGraph::new(0);
    with_blr
        .set_leaf_head(2, vec![], 1, Activation::None)
        .unwrap();
    with_blr.set_blr_prior(1.0, 1.5, 1.0).unwrap();
    let s_blr = with_blr.node_signature(0).unwrap();
    assert_ne!(s_blr.uncertainty, [0u8; 8]);
}

#[test]
fn node_signature_changes_when_children_added() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let r0 = g.node_signature(0).unwrap().routing;
    let _c = g.add_node(0, 7).unwrap();
    let r1 = g.node_signature(0).unwrap().routing;
    assert_ne!(r0, r1);
    // Adding another child changes routing again.
    let _c2 = g.add_node(0, 9).unwrap();
    let r2 = g.node_signature(0).unwrap().routing;
    assert_ne!(r1, r2);
}

#[test]
fn node_signature_observe_does_not_change_routing() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let r0 = g.node_signature(0).unwrap().routing;
    g.observe(0, 3.14).unwrap();
    let r1 = g.node_signature(0).unwrap().routing;
    assert_eq!(r0, r1);
}

#[test]
fn determinism_double_run_full_stack() {
    let mk = || {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g.set_density_tracker().unwrap();
        g.set_calibration(15).unwrap();
        g.observe(0, 1.5).unwrap();
        let m = g.node_maturity(0).unwrap();
        let s = g.node_signature(0).unwrap();
        (m.canonical_bytes(), s.canonical_bytes())
    };
    let a = mk();
    let b = mk();
    assert_eq!(a, b);
}

#[test]
fn maturity_canonical_bytes_pinned_layout() {
    // Pin the 11-byte big-endian layout so a future format change is
    // forced through the snapshot v6 review.
    let m = Maturity {
        samples_seen: 0x_0011_2233_4455_6677,
        calibration_stable: true,
        uncertainty_stable: true,
        trust_level: 4,
    };
    let bytes = m.canonical_bytes();
    assert_eq!(
        &bytes,
        &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x01, 0x01, 0x04]
    );
}
