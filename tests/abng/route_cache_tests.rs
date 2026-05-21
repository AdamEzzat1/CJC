//! Phase 0.10 Track Q1 — D-HARHT route-memoization cache.
//!
//! The cache is opt-in and outcome-transparent: a hit must return
//! exactly the `RouteEvidence` the radix walk would compute, and a
//! topology change must clear it. These tests pin both properties,
//! plus the "disabled by default" contract that keeps every pre-0.10
//! caller byte-identical.

use cjc_abng::graph::AdaptiveBeliefGraph;

/// Two-level routing tree. Node ids, by `add_node` order:
/// root = 0; root's children (keys 10, 20, 30, 40) = 1, 2, 3, 4;
/// node 2's children (keys 11, 22, 33) = 5, 6, 7.
fn build_tree(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    for &k in &[10u8, 20, 30, 40] {
        g.add_node(0, k).unwrap();
    }
    for &k in &[11u8, 22, 33] {
        g.add_node(2, k).unwrap();
    }
    g
}

/// Prefixes exercising full matches, partial matches, and a
/// zero-match (unknown first byte). All are <= 7 bytes — cacheable.
const PREFIXES: &[&[u8]] = &[
    &[10u8],
    &[20u8],
    &[20u8, 22],
    &[20u8, 11],
    &[20u8, 99], // partial: key 99 unbound under node 2
    &[99u8],     // zero match: key 99 unbound at root
    &[30u8, 33], // partial: node 3 has no children
];

#[test]
fn route_cache_disabled_by_default() {
    let g = build_tree(1);
    assert!(!g.route_cache_enabled());
    assert_eq!(g.route_cache_stats(), None);
}

#[test]
fn descend_with_cache_matches_descend_without() {
    // The parity gate. A cached descend must be byte-identical to the
    // uncached walk for every prefix — including repeats, so cache
    // hits are exercised.
    let plain = build_tree(7);
    let mut cached = build_tree(7);
    cached.enable_route_cache();

    for _round in 0..3 {
        for &prefix in PREFIXES {
            let want = plain.descend(prefix);
            let got = cached.descend(prefix);
            assert_eq!(
                got, want,
                "cached descend diverged from the walk for prefix {prefix:?}"
            );
        }
    }
}

#[test]
fn repeated_descend_registers_cache_hits() {
    let mut g = build_tree(2);
    g.enable_route_cache();

    // First pass over the cacheable prefixes — all misses.
    let cacheable: &[&[u8]] = &[&[10u8], &[20u8], &[20u8, 22]];
    for &p in cacheable {
        let _ = g.descend(p);
    }
    let (hits1, misses1, _) = g.route_cache_stats().unwrap();
    assert_eq!(hits1, 0, "first pass should be all misses");
    assert_eq!(misses1, cacheable.len() as u64);

    // Second pass — every cacheable prefix is now a hit.
    for &p in cacheable {
        let _ = g.descend(p);
    }
    let (hits2, misses2, _) = g.route_cache_stats().unwrap();
    assert_eq!(hits2, cacheable.len() as u64, "second pass should all hit");
    assert_eq!(misses2, misses1, "no new misses on the second pass");
}

#[test]
fn topology_change_clears_cache() {
    let mut g = build_tree(3);
    g.enable_route_cache();

    // Populate: one miss for prefix [10].
    let before = g.descend(&[10u8]);
    let (_, misses_after_first, _) = g.route_cache_stats().unwrap();
    assert_eq!(misses_after_first, 1);

    // A topology change (NodeAdded) must clear the cache.
    g.add_node(0, 50).unwrap();

    // The same prefix is a miss again — proof the cache was cleared.
    let after = g.descend(&[10u8]);
    let (hits, misses, _) = g.route_cache_stats().unwrap();
    assert_eq!(hits, 0, "cache was cleared, so the re-descend cannot hit");
    assert_eq!(misses, 2, "second descend re-missed after the clear");

    // ...and the routing result is still correct across the clear.
    assert_eq!(before, after);
}

#[test]
fn uncacheable_long_prefix_still_routes() {
    // descend accepts an arbitrary &[u8]; a prefix longer than 7
    // bytes cannot be packed into a u64 key. It must still route
    // correctly (via the walk) and register as a skip.
    let plain = build_tree(4);
    let mut cached = build_tree(4);
    cached.enable_route_cache();

    let long: &[u8] = &[20u8, 22, 0, 0, 0, 0, 0, 0, 0]; // 9 bytes
    assert_eq!(cached.descend(long), plain.descend(long));

    let (hits, _misses, skips) = cached.route_cache_stats().unwrap();
    assert_eq!(hits, 0);
    assert!(skips >= 1, "a >7-byte prefix should count as a skip");
}

#[test]
fn cache_does_not_perturb_chain_head() {
    // descend is read-only (emits no audit event), so enabling the
    // cache and routing must leave the audit chain head untouched —
    // the determinism contract for the routing layer.
    let mut g = build_tree(5);
    let head_before = g.chain_head;

    g.enable_route_cache();
    for _ in 0..4 {
        for &p in PREFIXES {
            let _ = g.descend(p);
        }
    }
    assert_eq!(
        g.chain_head, head_before,
        "route caching must not touch the audit chain"
    );
}

#[test]
fn enable_route_cache_is_idempotent_reset() {
    let mut g = build_tree(6);
    g.enable_route_cache();
    let _ = g.descend(&[10u8]);
    let _ = g.descend(&[10u8]); // a hit
    let (hits, _, _) = g.route_cache_stats().unwrap();
    assert_eq!(hits, 1);

    // A second enable resets to a fresh, empty cache.
    g.enable_route_cache();
    let (hits, misses, _) = g.route_cache_stats().unwrap();
    assert_eq!((hits, misses), (0, 0), "re-enable starts fresh");

    g.disable_route_cache();
    assert!(!g.route_cache_enabled());
}
