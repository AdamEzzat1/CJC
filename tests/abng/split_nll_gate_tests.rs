//! Phase 0.4 Track B-2.2.4 — Split impurity + held-out ΔNLL gates.
//!
//! `try_split` was previously gated only on `is_leaf` and
//! `samples_seen ≥ split_min`. Phase 0.4 adds two more gates:
//!
//! 1. **Impurity gate**: `variance(NodeStats) ≥ policy.impurity_min()`.
//!    A node whose observations have collapsed to a constant value has
//!    no learning signal to split on.
//! 2. **Held-out ΔNLL gate**: deterministic bootstrap on samples drawn
//!    from the node's Gaussian model `N(μ, σ²)`. We synthesize a
//!    50/50 train/test split, fit pre-split (single Gaussian) and
//!    post-split (median-partitioned Gaussians) models on train, and
//!    compute NLL on test. The Δ must exceed `policy.nll_split_gain()`.
//!
//! The bootstrap is seeded from `(graph.seed, node_id, ∑action_counts)`
//! so the gate decisions are bit-deterministic across runs and replays.

use cjc_abng::graph::{ActionKind, AdaptiveBeliefGraph};

/// Thresholds that disable everything except the Split gates so we
/// study them in isolation. tau_compress=0 / tau_merge=0 / kl_merge=0
/// require identical signatures (rare). prune_floor / freeze_after
/// disabled.
fn isolating_thresholds(
    impurity_min: f64,
    nll_split_gain: f64,
) -> [f64; 14] {
    [
        100.0,           // h_grow — disable Grow
        1.0e9,           // grow_min — disable Grow
        4.0,             // split_min — low so easy to satisfy
        nll_split_gain,  // nll_split_gain
        impurity_min,    // impurity_min
        0.0,             // tau_merge
        0.0,             // kl_merge
        0.0,             // prune_floor
        1.0e9,           // prune_grace_epochs
        0.0,             // tau_compress
        1.0e9,           // freeze_after
        f64::MAX,        // drift_unfreeze — disabled
        0.005,           // ece_stability_max_delta (v11)
        1.05,            // sigma_stability_ratio (v11)
    ]
}

#[test]
fn split_blocked_when_variance_below_impurity_floor() {
    // All observations = 1.0 → variance = 0 → impurity gate fails.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&isolating_thresholds(0.02, 0.05)).unwrap();
    for _ in 0..10 {
        g.observe(0, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Split as usize],
        0,
        "zero-variance node should be blocked by impurity gate"
    );
}

#[test]
fn split_fires_when_variance_above_impurity_floor() {
    // Varied observations → non-zero variance → impurity gate passes.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&isolating_thresholds(0.02, 0.05)).unwrap();
    for i in 0..10 {
        g.observe(0, (i as f64) * 0.5).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Split as usize],
        1,
        "high-variance node should be allowed through impurity gate"
    );
}

#[test]
fn split_blocked_when_nll_gain_threshold_above_estimate() {
    // High nll_split_gain threshold → estimated gain insufficient → block.
    let mut g = AdaptiveBeliefGraph::new(0);
    g.set_decision_policy(&isolating_thresholds(0.0, 1.0e9)).unwrap();
    for i in 0..10 {
        g.observe(0, (i as f64) * 0.5).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Split as usize],
        0,
        "huge nll_split_gain should block Split via ΔNLL gate"
    );
}

#[test]
fn split_gates_are_deterministic_across_runs() {
    // Two independent runs with identical graph seed and observations
    // must produce the same Split decision (the bootstrap is
    // SplitMix64-deterministic from graph.seed + node_id +
    // action_counts_sum).
    let make_graph = || {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.set_decision_policy(&isolating_thresholds(0.02, 0.05))
            .unwrap();
        for i in 0..10 {
            g.observe(0, (i as f64) * 0.3).unwrap();
        }
        g
    };
    let mut g1 = make_graph();
    let mut g2 = make_graph();
    assert_eq!(g1.decide_step(), g2.decide_step());
}

#[test]
fn split_under_too_few_samples_blocked_at_split_min() {
    // samples_seen < split_min — pre-existing gate. Pin that the new
    // ΔNLL gates don't accidentally let through too-few-sample cases.
    let mut g = AdaptiveBeliefGraph::new(0);
    let thresholds: [f64; 14] = [
        100.0,
        1.0e9,
        100.0,  // split_min — high
        0.05,
        0.02,
        0.0,
        0.0,
        0.0,
        1.0e9,
        0.0,
        1.0e9,
        f64::MAX, // drift_unfreeze — disabled
        0.005,  // ece_stability_max_delta (v11)
        1.05,   // sigma_stability_ratio (v11)
    ];
    g.set_decision_policy(&thresholds).unwrap();
    for i in 0..10 {
        g.observe(0, (i as f64) * 0.5).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Split as usize],
        0,
        "samples_seen=10 < split_min=100 should block before reaching new gates"
    );
}
