//! Phase 0.4 Track B-2.2.5 — route-entropy gate for Grow.
//!
//! When a codebook is installed, `decide_step` gates Grow on the
//! Shannon entropy of the candidate's parent's children-key
//! distribution. The gate keeps Grow from firing in regions of the
//! tree that aren't being subdivided richly enough — a node whose
//! parent has only a single key bound has no diversity-of-routes
//! evidence to justify spawning a fresh subtree.
//!
//! Codebook-less graphs fall back to pre-0.4 Hamming-only Grow
//! behavior (the entropy concept is undefined without a codebook),
//! so existing decide_step tests that don't install a codebook still
//! work unchanged. The tests below exercise the gate exclusively in
//! the codebook-installed path.
//!
//! Implementation note: with unique key bytes per parent's children
//! map, the entropy of the candidate's parent's children-key
//! distribution simplifies to `log(n_children)` once `n_children ≥ 2`.
//! Single-child or root-without-parent cases bootstrap to
//! `f64::INFINITY` so the very first Grow at any depth is allowed.

use cjc_abng::graph::{ActionKind, AdaptiveBeliefGraph};

fn install_codebook(g: &mut AdaptiveBeliefGraph) {
    // 1-D codebook with 8 bins, boundaries 1..=7.0. Decision tests only
    // need *some* codebook to be installed so the entropy gate
    // engages; we never invoke `descend` here.
    g.set_codebook(1, 8, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .unwrap();
}

/// Single-feature codebook + fresh policy with grow_min low so we
/// can fire after a handful of observations.
fn graph_codebook_grow_policy() -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(0);
    install_codebook(&mut g);
    let thresholds: [f64; 14] = [
        0.5,    // h_grow — entropy threshold
        2.0,    // grow_min
        1.0e9,  // split_min — disabled
        0.05,   // nll_split_gain
        0.02,   // impurity_min
        4.0,    // tau_merge
        0.1,    // kl_merge
        0.0,    // prune_floor
        1.0e9,  // prune_grace_epochs
        8.0,    // tau_compress
        1.0e9,  // freeze_after — disabled
        f64::MAX, // drift_unfreeze — disabled
        0.005,  // ece_stability_max_delta (v11)
        1.05,   // sigma_stability_ratio (v11)
    ];
    g.set_decision_policy(&thresholds).unwrap();
    g
}

#[test]
fn root_grow_allowed_via_bootstrap() {
    // Root has no parent, so the entropy proxy returns INFINITY and
    // the gate is bypassed. Even with `h_grow = 0.5` (tight), root's
    // first Grow always fires.
    let mut g = graph_codebook_grow_policy();
    for _ in 0..10 {
        g.observe(0, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Grow as usize],
        1,
        "root Grow always allowed via bootstrap path"
    );
}

#[test]
fn leaf_with_single_sibling_grow_allowed_via_bootstrap() {
    // After the first Grow, root has 1 child. The new child is itself
    // a candidate — its parent (root) has 1 child, which is < 2 →
    // bootstrap → allowed.
    let mut g = graph_codebook_grow_policy();
    for _ in 0..10 {
        g.observe(0, 1.0).unwrap();
    }
    g.decide_step(); // first Grow on root → root now has 1 child
    // Drive observations into the child to make it grow-eligible.
    for _ in 0..10 {
        g.observe(1, 1.0).unwrap();
    }
    let counts = g.decide_step();
    // The child can grow (single-sibling bootstrap). Root could also
    // re-grow (if signature stable), but won't because root has the
    // child slot at the deterministic key already used. So at least
    // one Grow fires (on the child).
    assert!(
        counts[ActionKind::Grow as usize] >= 1,
        "single-sibling leaf grow allowed via bootstrap"
    );
}

/// Thresholds with all triggers EXCEPT Grow effectively disabled:
/// - tau_compress=0 → Compress only on identical signatures (rare)
/// - tau_merge=0, kl_merge=0 → Merge only on identical signatures+posteriors
/// - split_min, prune_grace_epochs, freeze_after huge
/// - prune_floor=0 (Prune requires samples_seen < 0, never)
///
/// Lets us study Grow in isolation without Compress accidentally
/// converting root to `Dense` (which empties `iter()` and would mask
/// the entropy gate).
fn isolating_thresholds(h_grow: f64) -> [f64; 14] {
    [
        h_grow, // h_grow
        2.0,    // grow_min — low so easy to satisfy
        1.0e9,  // split_min — disabled
        0.05,   // nll_split_gain
        0.02,   // impurity_min
        0.0,    // tau_merge — require identical sigs
        0.0,    // kl_merge — require identical posteriors
        0.0,    // prune_floor — never prune
        1.0e9,  // prune_grace_epochs — disabled
        0.0,    // tau_compress — require identical sigs (rare)
        1.0e9,  // freeze_after — disabled
        f64::MAX, // drift_unfreeze — disabled
        0.005,  // ece_stability_max_delta (v11)
        1.05,   // sigma_stability_ratio (v11)
    ]
}

#[test]
fn high_h_grow_blocks_grow_on_two_sibling_parent() {
    // Root with two children → entropy = log(2) ≈ 0.693. With
    // h_grow > 0.693 the gate blocks Grow on those children.
    let mut g = AdaptiveBeliefGraph::new(0);
    install_codebook(&mut g);
    g.force_grow(0, 5).unwrap();
    g.force_grow(0, 11).unwrap();
    assert_eq!(g.nodes[0].children.iter().len(), 2);

    g.set_decision_policy(&isolating_thresholds(1.0)).unwrap();
    for _ in 0..10 {
        g.observe(1, 1.0).unwrap();
        g.observe(2, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Grow as usize],
        0,
        "leaves with parent of 2 children should be blocked when h_grow > log(2)"
    );
}

#[test]
fn low_h_grow_unblocks_two_sibling_parent() {
    // Same setup but h_grow = 0.5 (below log(2) ≈ 0.693) → gate passes.
    let mut g = AdaptiveBeliefGraph::new(0);
    install_codebook(&mut g);
    g.force_grow(0, 5).unwrap();
    g.force_grow(0, 11).unwrap();
    g.set_decision_policy(&isolating_thresholds(0.5)).unwrap();
    for _ in 0..10 {
        g.observe(1, 1.0).unwrap();
        g.observe(2, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert!(
        counts[ActionKind::Grow as usize] >= 1,
        "h_grow < log(2) should let at least one leaf grow"
    );
}

#[test]
fn codebookless_graph_falls_back_to_pre_0_4_grow() {
    // No codebook → entropy gate skipped → pre-0.4 behavior preserved.
    // Pin this so existing decide_step tests (which don't install
    // codebooks) don't regress.
    let mut g = AdaptiveBeliefGraph::new(0);
    let thresholds: [f64; 14] = [
        100.0,  // h_grow — very strict (would block if gate engaged)
        2.0,    // grow_min
        1.0e9,
        0.05,
        0.02,
        4.0,
        0.1,
        0.0,
        1.0e9,
        8.0,
        1.0e9,
        f64::MAX, // drift_unfreeze — disabled
        0.005,  // ece_stability_max_delta (v11)
        1.05,   // sigma_stability_ratio (v11)
    ];
    g.set_decision_policy(&thresholds).unwrap();
    for _ in 0..10 {
        g.observe(0, 1.0).unwrap();
    }
    let counts = g.decide_step();
    assert_eq!(
        counts[ActionKind::Grow as usize],
        1,
        "no codebook → entropy gate skipped → pre-0.4 Grow fires"
    );
}
