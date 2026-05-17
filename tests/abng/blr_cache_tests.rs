//! Phase 0.8 Item D1 — tests for the cached Cholesky factor in
//! `BlrState`.
//!
//! Determinism contract: `predict` returns the same `(mean,
//! epistemic_leverage, aleatoric_var)` tuple regardless of whether
//! `cached_l` is `Some` (cache hit) or `None` (cache miss followed
//! by lazy populate). The cache is purely a performance hint; the
//! math goes through identical `cholesky` + `forward_subst` paths.
//!
//! Snapshot contract: the cache is NOT serialized. Round-tripping
//! a graph through `serialize` -> `replay` produces a graph whose
//! BLR states have `cached_l == None` (fresh-after-deserialize).
//!
//! Phase 0.9.5 — the `n = 1` `update` fast path maintains its own
//! incremental Cholesky factor (`chol_factor`) and *invalidates*
//! `cached_l`; the next `predict` lazily repopulates it. Batch
//! (`n >= 2`) `update` and `combine` still populate `cached_l`
//! eagerly. Either way the cache is never stale — it is `None` or
//! exactly `cholesky(precision)`.
//!
//! These tests gate D1 against silent regressions: if a future
//! refactor accidentally lets the cache go stale (e.g. mutates
//! `precision` without updating `cached_l`), one of the parity
//! tests below fires.

use cjc_abng::blr::{BlrPrior, BlrState};
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;

const D: u32 = 4;
const PHI: [f64; 4] = [1.0, 0.5, 0.25, 0.125];

fn primed_state(prior: &BlrPrior, n_updates: usize) -> BlrState {
    let mut s = BlrState::from_prior(prior, D);
    for i in 0..n_updates {
        let yi = [0.7 + (i as f64) * 0.0001];
        s.update(&PHI, &yi).unwrap();
    }
    s
}

// ---------------------------------------------------------------------------
// Cache-hit and cache-miss produce identical predict output
// ---------------------------------------------------------------------------

#[test]
fn cache_miss_and_hit_produce_identical_predict() {
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let s = primed_state(&prior, 100);
    // After 100 updates the cache is populated. Snapshot the first
    // predict result.
    let first = s.predict(&PHI).unwrap();
    // Force cache miss by clearing.
    *s.cached_l.borrow_mut() = None;
    let after_miss = s.predict(&PHI).unwrap();
    assert_eq!(
        first, after_miss,
        "predict on cache hit and cache miss must produce identical tuples"
    );
}

#[test]
fn predict_is_idempotent_across_repeated_calls() {
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let s = primed_state(&prior, 50);
    let r1 = s.predict(&PHI).unwrap();
    let r2 = s.predict(&PHI).unwrap();
    let r3 = s.predict(&PHI).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
}

#[test]
fn from_prior_starts_with_empty_cache() {
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let s = BlrState::from_prior(&prior, D);
    assert!(
        s.cached_l.borrow().is_none(),
        "fresh BlrState must have empty cache (lazy init)"
    );
}

#[test]
fn first_predict_populates_cache() {
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let s = BlrState::from_prior(&prior, D);
    assert!(s.cached_l.borrow().is_none());
    // Predict on the prior — Λ_0 = λ_0 · I, so this is well-defined.
    let _ = s.predict(&PHI).unwrap();
    assert!(
        s.cached_l.borrow().is_some(),
        "first predict must populate the cache"
    );
}

#[test]
fn update_n1_invalidates_cache_then_predict_repopulates() {
    // Phase 0.9.5 — the n = 1 fast path maintains its own incremental
    // factor and invalidates the predict cache rather than eagerly
    // populating it. The next `predict` lazily recomputes
    // `cholesky(precision)`.
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let mut s = BlrState::from_prior(&prior, D);
    s.update(&PHI, &[0.7]).unwrap(); // n = 1
    assert!(
        s.cached_l.borrow().is_none(),
        "an n=1 update invalidates the predict cache"
    );
    let _ = s.predict(&PHI).unwrap();
    assert!(
        s.cached_l.borrow().is_some(),
        "predict after an n=1 update repopulates the cache"
    );
}

#[test]
fn update_batch_populates_cache() {
    // An n >= 2 batch update still eagerly populates the cache — it
    // computes the full Cholesky for its own solve regardless.
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let mut s = BlrState::from_prior(&prior, D);
    let feats = [
        PHI[0], PHI[1], PHI[2], PHI[3], // row 0
        PHI[0], PHI[1], PHI[2], PHI[3], // row 1
    ];
    s.update(&feats, &[0.7, 0.8]).unwrap(); // n = 2
    assert!(
        s.cached_l.borrow().is_some(),
        "an n>=2 update eagerly populates the cache"
    );
}

#[test]
fn update_n1_does_not_leave_a_stale_cache() {
    // The cache tracks `precision`. After an n=1 update changes
    // `precision`, the cache must NOT hold a stale factor from before
    // the update — the n=1 path invalidates it, and `predict` then
    // recomputes the Cholesky of the *new* precision.
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let mut s = primed_state(&prior, 10);
    // Populate the cache via a predict, then do one more n=1 update.
    let _ = s.predict(&PHI).unwrap();
    assert!(s.cached_l.borrow().is_some());
    s.update(&PHI, &[0.9]).unwrap(); // n = 1 — must invalidate
    assert!(
        s.cached_l.borrow().is_none(),
        "n=1 update must invalidate, not leave a stale cache"
    );
    // `predict` recomputes from the post-update precision; the result
    // must match a from-scratch recompute (cache cleared).
    let predict_via_lazy = s.predict(&PHI).unwrap();
    *s.cached_l.borrow_mut() = None;
    let predict_via_fresh = s.predict(&PHI).unwrap();
    assert_eq!(
        predict_via_lazy, predict_via_fresh,
        "post-update cache must be the Cholesky of the post-update precision"
    );
}

#[test]
fn combine_populates_cache() {
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let mut s_a = primed_state(&prior, 5);
    let s_b = primed_state(&prior, 5);
    s_a.combine(&s_b, &prior).unwrap();
    assert!(
        s_a.cached_l.borrow().is_some(),
        "combine must populate the cache"
    );
    // And the populated cache must produce a correct predict.
    let predict_via_cache = s_a.predict(&PHI).unwrap();
    *s_a.cached_l.borrow_mut() = None;
    let predict_after_clear = s_a.predict(&PHI).unwrap();
    assert_eq!(predict_via_cache, predict_after_clear);
}

// ---------------------------------------------------------------------------
// Snapshot contract: cache is NOT serialized
// ---------------------------------------------------------------------------

#[test]
fn snapshot_round_trip_does_not_carry_cache() {
    // Train a graph with BLR enabled; serialize; replay; verify the
    // replayed graph's BLR states have empty caches; verify a predict
    // produces the same output as on the original.
    let mut g = AdaptiveBeliefGraph::new(42);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for _ in 0..20 {
        g.blr_update(0, &PHI, &[0.7]).unwrap();
    }

    // Pre-serialize predict (cache hit path).
    let pred_before = g.blr_predict(0, &PHI).unwrap();

    // Phase 0.9.5 R0-3 (Tier 2 Option C) — flush periodic-checkpoint
    // BLR witnesses so the trained node's final state is chain-anchored
    // for the end-of-replay verifier.
    g.checkpoint_blr();
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();

    // The replayed graph's BLR cache is fresh.
    let blr2 = g2.nodes[0].blr.as_ref().expect("blr state must round-trip");
    assert!(
        blr2.cached_l.borrow().is_none(),
        "replay must leave cached_l empty (lazy init contract)"
    );

    // First predict on replayed graph produces the same output as
    // the original.
    let pred_after = g2.blr_predict(0, &PHI).unwrap();
    assert_eq!(
        pred_before, pred_after,
        "replay -> first predict must produce bit-identical output to pre-serialize predict"
    );

    // And after that first predict, the cache is populated.
    let blr2_again =
        g2.nodes[0].blr.as_ref().expect("blr state must persist");
    assert!(
        blr2_again.cached_l.borrow().is_some(),
        "first predict on replayed graph must populate the cache"
    );
}

#[test]
fn cache_does_not_affect_canonical_bytes() {
    // canonical_bytes must NOT include cached_l. Two states with
    // identical mean/precision/a/b/n_seen/feature_version_hash but
    // different cache states (one populated, one empty) must produce
    // identical canonical_bytes.
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let s1 = primed_state(&prior, 10);
    let mut s2 = s1.clone();
    *s2.cached_l.borrow_mut() = None; // s1 has cache, s2 doesn't.
    assert_eq!(
        s1.canonical_bytes(),
        s2.canonical_bytes(),
        "cached_l must not affect canonical_bytes"
    );
    assert_eq!(s1.state_hash(), s2.state_hash());
}

// ---------------------------------------------------------------------------
// Multi-update sequences: predict matches recompute at every step
// ---------------------------------------------------------------------------

#[test]
fn predict_matches_uncached_after_n_updates() {
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let mut s = BlrState::from_prior(&prior, D);
    for i in 0..30 {
        let yi = [0.5 + (i as f64) * 0.01];
        s.update(&PHI, &yi).unwrap();
        // After each update, predict via cache and via fresh
        // recompute must agree.
        let cached = s.predict(&PHI).unwrap();
        *s.cached_l.borrow_mut() = None;
        let uncached = s.predict(&PHI).unwrap();
        assert_eq!(
            cached, uncached,
            "diverged at step {i} — cache and fresh decomposition disagree"
        );
        // Restore cache so the next iteration's update has a base to
        // overwrite (purely to exercise the "update with cached state"
        // path).
        let _ = s.predict(&PHI);
    }
}

#[test]
fn clone_carries_consistent_cache() {
    // BlrState: Clone. Cloning a state with a populated cache
    // produces a clone whose cache is also populated and matches
    // (because RefCell<Option<Vec<f64>>>: Clone via Vec::clone).
    let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
    let s_orig = primed_state(&prior, 5);
    let s_clone = s_orig.clone();
    assert_eq!(
        s_orig.cached_l.borrow().clone(),
        s_clone.cached_l.borrow().clone(),
        "Clone must produce identical cache state"
    );
    // Predict outputs match.
    assert_eq!(
        s_orig.predict(&PHI).unwrap(),
        s_clone.predict(&PHI).unwrap()
    );
}
