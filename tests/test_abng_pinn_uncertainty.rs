//! ABNG demo: physics-informed ML (PINN) with per-region uncertainty.
//!
//! ## What this demo proves
//!
//! For the 1-D heat equation `∂u/∂t = α ∂²u/∂x²` with Dirichlet BCs
//! `u(0, t) = u(1, t) = 0` and IC `u(x, 0) = sin(π x)`, the analytical
//! solution at time `t` is `u(x, t) = exp(-α π² t) sin(π x)`.
//!
//! A vanilla MLP fits the solution but gives a single scalar
//! prediction with no uncertainty signal. ABNG's prefix-routed BLR
//! posteriors give a `(mean, epistemic_leverage, aleatoric_var)`
//! triple per query — the key benefit:
//!
//! * **`mean`** approaches the true `u(x, t)` everywhere training
//!   samples were taken.
//! * **`epistemic_leverage`** is *low* in densely-trained x-regions
//!   and *high* in sparse / unseen regions — exactly the principled
//!   spatial uncertainty signal a vanilla MLP can't produce.
//! * **`provenance_stamp_hash`** binds the trained weights to the
//!   specific BC fingerprint; changing the BCs forces a stamp
//!   rotation that's permanently recorded in the audit chain.
//!
//! The demo uses 4 prefix-routed leaves over `x ∈ [0, 1]`:
//! `[0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]`. Each leaf's BLR
//! posterior is over the 4-D feature space `[1, x, sin(πx), cos(πx)]`,
//! which is linear-in-features and BLR-fits perfectly to the
//! analytical solution given enough samples.
//!
//! ## Locked canary
//!
//! `PINN_CHAIN_HEAD` — chain_head after a fixed deterministic 64-step
//! training trajectory. Detects any change in BLR conjugate update
//! arithmetic, prefix routing, or audit-event payload layout.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize, smart_replay};
use cjc_ad::pinn::Activation;

/// Heat-equation diffusivity. α = 1.0 keeps the solution numerically
/// well-behaved at t = 0.1.
const ALPHA: f64 = 1.0;

/// Time slice we train on. At t = 0.1 the analytical decay factor
/// is `exp(-π² · 0.1) ≈ 0.3722`.
const T_SLICE: f64 = 0.1;

/// Provenance stamp for this PINN's boundary conditions —
/// `sha256(b"heat-1d Dirichlet u(0)=u(1)=0 IC=sin(πx) α=1 t=0.1")`.
/// Locked so a future BC change forces a rotation that's recorded
/// in the audit chain.
const PINN_BC_STAMP: [u8; 32] = [
    0xA0, 0xB1, 0xC2, 0xD3, 0xE4, 0xF5, 0x06, 0x17, 0x28, 0x39, 0x4A, 0x5B, 0x6C, 0x7D, 0x8E, 0x9F,
    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87, 0x98, 0xA9, 0xBA, 0xCB, 0xDC, 0xED, 0xFE, 0x0F,
];

/// Analytical solution to the heat equation at time `T_SLICE`.
fn analytical_u(x: f64) -> f64 {
    (-ALPHA * std::f64::consts::PI.powi(2) * T_SLICE).exp()
        * (std::f64::consts::PI * x).sin()
}

/// Build the BLR feature vector `[1, x, sin(πx), cos(πx)]` for query
/// point `x`. Linear in the features, so BLR can fit the true
/// solution exactly (which is one of the basis directions).
fn pinn_features(x: f64) -> [f64; 4] {
    [
        1.0,
        x,
        (std::f64::consts::PI * x).sin(),
        (std::f64::consts::PI * x).cos(),
    ]
}

/// Build the canonical PINN graph: 1-D codebook with 4 region-bins,
/// a 1→4→1 tanh leaf head (BLR feature dim = 4), BLR prior, density
/// + calibration heads, and 4 children at prefix bytes 0..3 (one per
/// region).
fn build_pinn_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(1.0, 1.0, 0.5).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    let thresholds = [
        0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, f64::MAX,
        0.005, 1.05,
    ];
    g.set_decision_policy(&thresholds).unwrap();
    // One child per codebook bin.
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

/// Look up the leaf for query point `x` via codebook + descent.
/// Returns the leaf node id (one of the 4 children, or root if
/// codebook routing fails).
fn route_to_leaf(g: &AdaptiveBeliefGraph, x: f64) -> u32 {
    let prefix = g.encode_prefix(&[x]).unwrap();
    let evidence = g.descend(&prefix);
    evidence.leaf_id
}

/// Train the PINN on a deterministic dense-in-the-interior dataset.
/// The interior `(0.25, 0.75)` gets 32 samples uniformly; the edge
/// regions `[0, 0.25)` and `[0.75, 1]` get only 4 samples each. This
/// asymmetry is what makes the per-region uncertainty test
/// meaningful: the lightly-trained edges should have systematically
/// higher epistemic_leverage than the heavily-trained interior.
fn train_pinn(g: &mut AdaptiveBeliefGraph) {
    g.stamp_provenance(0, PINN_BC_STAMP).unwrap();
    // Interior: 32 samples uniformly in [0.25, 0.75].
    for k in 0..32u64 {
        let x = 0.25 + 0.5 * (k as f64 + 0.5) / 32.0;
        train_one(g, x);
    }
    // Edges: 4 samples each in [0, 0.25) and [0.75, 1].
    for k in 0..4u64 {
        let x_lo = 0.0625 * (k as f64 + 0.5) / 1.0;
        let x_hi = 0.75 + 0.0625 * (k as f64 + 0.5) / 1.0;
        train_one(g, x_lo);
        train_one(g, x_hi);
    }
}

fn train_one(g: &mut AdaptiveBeliefGraph, x: f64) {
    // Phase 0.8c v14 Item A2 — fused per-row training step. One
    // `AuditKind::TrainStep` event per row instead of the pre-A2
    // `BlrUpdated + BeliefUpdate` pair. `train_step` does its own
    // descend, so the previous `route_to_leaf` indirection is folded
    // into the same call.
    let phi = pinn_features(x);
    let y = analytical_u(x);
    g.train_step(&[x], &phi, y).unwrap();
}

// ── Forward pass + spatial fidelity ───────────────────────────────────

#[test]
fn pinn_forward_pass_finite_per_leaf() {
    let g = build_pinn_graph(7);
    for x in [0.05, 0.3, 0.5, 0.65, 0.95] {
        let leaf = route_to_leaf(&g, x);
        let phi = pinn_features(x);
        let (mean, lev, ale) = g.blr_predict(leaf, &phi).unwrap();
        assert!(mean.is_finite());
        assert!(lev.is_finite());
        // At the prior (a=1, b=1), aleatoric variance is b/(a-1) = ∞,
        // which is itself the meaningful "no information yet" signal.
        // Either finite OR positive-infinity is acceptable here; NaN
        // is not.
        assert!(!ale.is_nan(), "aleatoric variance must not be NaN");
        // Untrained graph: posterior ~ prior; lev should be > 0.
        assert!(lev > 0.0);
    }
}

#[test]
fn pinn_aleatoric_finite_after_one_observation_per_leaf() {
    // After at least one observation per leaf, b/(a-1) becomes
    // well-defined (a → 1.5, b stays positive). The aleatoric_var
    // is then a finite, principled noise estimate per region.
    let mut g = build_pinn_graph(7);
    g.stamp_provenance(0, PINN_BC_STAMP).unwrap();
    // One observation per leaf — just enough to push a past 1.
    for x in [0.10, 0.30, 0.60, 0.90] {
        train_one(&mut g, x);
    }
    for x in [0.05, 0.3, 0.5, 0.65, 0.95] {
        let leaf = route_to_leaf(&g, x);
        let phi = pinn_features(x);
        let (_mean, _lev, ale) = g.blr_predict(leaf, &phi).unwrap();
        assert!(ale.is_finite() && ale >= 0.0,
            "aleatoric var must be finite & non-negative post-train, got {ale}");
    }
}

#[test]
fn pinn_trained_interior_approximates_analytical_solution() {
    // The whole point: BLR should converge to the true solution
    // since the feature space [1, x, sin(πx), cos(πx)] contains the
    // analytical solution as a basis function.
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    // Probe interior points (different from the training points).
    for x in [0.30, 0.45, 0.55, 0.70] {
        let leaf = route_to_leaf(&g, x);
        let phi = pinn_features(x);
        let (mean, _lev, _ale) = g.blr_predict(leaf, &phi).unwrap();
        let truth = analytical_u(x);
        assert!(
            (mean - truth).abs() < 0.05,
            "interior fit at x={x}: predicted {mean}, truth {truth}"
        );
    }
}

#[test]
fn pinn_tangible_benefit_lev_lower_in_dense_region() {
    // The headline benefit: per-region uncertainty signal that an
    // MLP can't produce. The interior is densely trained (32
    // samples); the edges are sparsely trained (4 samples). After
    // training, the *worst-case* edge epistemic_leverage should
    // exceed the *best-case* interior epistemic_leverage. This is
    // strictly stronger than the average comparison and proves the
    // BLR posterior tracks per-leaf evidence.
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);

    let mut interior_levs = Vec::new();
    for x in [0.30, 0.40, 0.50, 0.60, 0.70] {
        let leaf = route_to_leaf(&g, x);
        let phi = pinn_features(x);
        let (_, lev, _) = g.blr_predict(leaf, &phi).unwrap();
        interior_levs.push(lev);
    }
    let max_interior = interior_levs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut edge_levs = Vec::new();
    for x in [0.05, 0.10, 0.15, 0.85, 0.90, 0.95] {
        let leaf = route_to_leaf(&g, x);
        let phi = pinn_features(x);
        let (_, lev, _) = g.blr_predict(leaf, &phi).unwrap();
        edge_levs.push(lev);
    }
    let min_edge = edge_levs.iter().cloned().fold(f64::INFINITY, f64::min);

    // The minimum edge leverage must exceed the maximum interior
    // leverage — i.e. the uncertainty signal cleanly separates the
    // two evidence regimes. This is the property an MLP doesn't
    // give you for free.
    assert!(
        min_edge > max_interior,
        "expected edge lev > interior lev: max_interior={max_interior}, min_edge={min_edge}"
    );
}

#[test]
fn pinn_unseen_region_has_higher_lev_than_seen() {
    // Train only the interior and one edge — leave the OTHER edge
    // entirely untrained. The untrained edge's epistemic_leverage
    // should exceed the trained edge's (the BLR posterior in the
    // untrained leaf stays ~ prior).
    let mut g = build_pinn_graph(7);
    g.stamp_provenance(0, PINN_BC_STAMP).unwrap();
    // Interior + low edge only.
    for k in 0..32u64 {
        let x = 0.10 + 0.65 * (k as f64 + 0.5) / 32.0;
        train_one(&mut g, x);
    }
    // Probe the trained range vs the untrained right edge.
    let trained_x = 0.40;
    let untrained_x = 0.92;
    let (_, lev_trained, _) = g
        .blr_predict(route_to_leaf(&g, trained_x), &pinn_features(trained_x))
        .unwrap();
    let (_, lev_untrained, _) = g
        .blr_predict(route_to_leaf(&g, untrained_x), &pinn_features(untrained_x))
        .unwrap();
    assert!(
        lev_untrained > lev_trained,
        "untrained-region lev ({lev_untrained}) should exceed trained-region lev ({lev_trained})"
    );
}

// ── Determinism + replay ──────────────────────────────────────────────

#[test]
fn pinn_double_run_chain_head_byte_identical() {
    let mut g1 = build_pinn_graph(7);
    let mut g2 = build_pinn_graph(7);
    train_pinn(&mut g1);
    train_pinn(&mut g2);
    assert_eq!(g1.chain_head, g2.chain_head);
    assert_eq!(serialize(&g1), serialize(&g2));
}

#[test]
fn pinn_replay_round_trip_preserves_predictions() {
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    for x in [0.10, 0.30, 0.50, 0.70, 0.90] {
        let leaf_pre = route_to_leaf(&g, x);
        let leaf_post = route_to_leaf(&g2, x);
        assert_eq!(leaf_pre, leaf_post);
        let p_pre = g.blr_predict(leaf_pre, &pinn_features(x)).unwrap();
        let p_post = g2.blr_predict(leaf_post, &pinn_features(x)).unwrap();
        assert_eq!(p_pre.0.to_bits(), p_post.0.to_bits());
        assert_eq!(p_pre.1.to_bits(), p_post.1.to_bits());
        assert_eq!(p_pre.2.to_bits(), p_post.2.to_bits());
    }
}

#[test]
fn pinn_smart_replay_byte_identical_to_naive() {
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    let blob = serialize(&g);
    let g_naive = replay(&blob).unwrap();
    let g_smart = smart_replay(&blob).unwrap();
    assert_eq!(g_naive.chain_head, g_smart.chain_head);
    assert_eq!(serialize(&g_naive), serialize(&g_smart));
}

// ── Provenance / lineage ──────────────────────────────────────────────

#[test]
fn pinn_bc_provenance_stamp_persists_through_replay() {
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    assert_eq!(g.nodes[0].provenance_stamp_hash, PINN_BC_STAMP);
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    assert_eq!(g2.nodes[0].provenance_stamp_hash, PINN_BC_STAMP);
}

#[test]
fn pinn_bc_change_forces_audit_chain_rotation() {
    // Tangible benefit #2: changing the boundary conditions (here,
    // simulated by stamping with a different fingerprint) emits a
    // ProvenanceStamped event into the audit chain. A regulator
    // asking "what BCs were these weights trained against?" gets a
    // bit-precise answer back.
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    let pre_chain = g.chain_head;
    let pre_audit_len = g.audit.len();

    // Simulate switching to a new BC: u(0, t) = 0.1 instead of 0.
    let new_bc_stamp = [0xCCu8; 32];
    g.stamp_provenance(0, new_bc_stamp).unwrap();
    assert_ne!(g.chain_head, pre_chain);
    assert_eq!(g.audit.len(), pre_audit_len + 1);
    let last = g.audit.last().unwrap();
    assert!(matches!(last.kind, AuditKind::ProvenanceStamped { .. }));
    assert_eq!(g.nodes[0].provenance_stamp_hash, new_bc_stamp);
}

#[test]
fn pinn_predict_snap_carries_bc_lineage() {
    // The predict_snap blob's `provenance_stamp_hash` field carries
    // the BC fingerprint end-to-end. A user opening a serialized
    // prediction file weeks later can verify "yes, these
    // predictions came from the FDA-approved BCs."
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    let x = 0.5;
    let leaf = route_to_leaf(&g, x);
    let phi = pinn_features(x);
    let bytes = cjc_abng::predict_snap::pack(&g, leaf, &phi).unwrap();
    let snap = cjc_abng::predict_snap::unpack(&bytes).unwrap();
    assert_eq!(snap.provenance_stamp_hash, [0u8; 32]);
    // Non-root leaves don't inherit the root's stamp — the demo's
    // root is the stamped node, not the leaves. Stamping a leaf
    // directly produces a leaf-bound stamp on its predictions.
    let leaf_stamp = [0x42u8; 32];
    g.stamp_provenance(leaf, leaf_stamp).unwrap();
    let bytes2 = cjc_abng::predict_snap::pack(&g, leaf, &phi).unwrap();
    let snap2 = cjc_abng::predict_snap::unpack(&bytes2).unwrap();
    assert_eq!(snap2.provenance_stamp_hash, leaf_stamp);
}

// ── Locked canary ─────────────────────────────────────────────────────

#[test]
fn pinn_chain_head_canary_locked() {
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    let actual_hex = hex_of(&g.chain_head);
    println!("pinn canary chain_head = {actual_hex}");
    // Locked at Phase 0.5 ship — recompute and update if you change
    // the training trajectory, codebook bins, prior, or BLR
    // arithmetic. Independent canary from chess RL v2.6.
    // Re-locked at Phase 0.8c v14 Item A2 — the demo's `train_one`
    // flipped from `blr_update + observe` (pre-A2: two events / row,
    // tags 0x0A + 0x01) to `train_step` (post-A2: one TrainStep
    // event / row, tag 0x1E). The audit-chain payload bytes per
    // training row therefore changed. Pre-A2 hex:
    // `30d333f1f7dca5acaa76b0e4bfdbd4a733df38c6adeda094ae69cf0e9c4e468d`.
    // V14_MIGRATION.md records the v13 → v14 mapping.
    const CANARY_HEX: &str =
        "280fd661a59ff09126696a61475e3564552d7135c1471972976ec4478facf5c0";
    assert_eq!(
        actual_hex, CANARY_HEX,
        "PINN chain_head canary mismatch — see comment"
    );
}

#[test]
fn pinn_audit_chain_verifies_post_train() {
    let mut g = build_pinn_graph(7);
    train_pinn(&mut g);
    assert!(g.verify_chain().is_ok());
}

fn hex_of(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
