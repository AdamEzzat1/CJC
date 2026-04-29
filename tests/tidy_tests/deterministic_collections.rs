//! v3 Phase 7 — Deterministic Collection family tests.
//!
//! Covers `IndexVec`, `TinyDetMap`, `SortedVecMap`, `DetOpenMap`. Every
//! structure must produce the same output for the same input across
//! runs, and (where order is part of the public contract) must produce
//! sorted iteration.

use cjc_data::detcoll::{DetOpenMap, IndexVec, SortedVecMap, TinyDetMap};

cjc_data::det_idx!(NodeId);

// ── IndexVec ────────────────────────────────────────────────────────────

#[test]
fn phase7_indexvec_dense_id_lookup() {
    let mut iv: IndexVec<NodeId, &'static str> = IndexVec::new();
    let a = iv.push("alpha");
    let b = iv.push("beta");
    let c = iv.push("gamma");
    assert_eq!(iv[a], "alpha");
    assert_eq!(iv[b], "beta");
    assert_eq!(iv[c], "gamma");
    assert_eq!(iv.len(), 3);
}

#[test]
fn phase7_indexvec_iter_is_insertion_order() {
    let mut iv: IndexVec<NodeId, i32> = IndexVec::new();
    for v in [3, 1, 4, 1, 5, 9, 2, 6] {
        iv.push(v);
    }
    let collected: Vec<i32> = iv.values().copied().collect();
    assert_eq!(collected, vec![3, 1, 4, 1, 5, 9, 2, 6]);
}

// ── TinyDetMap ──────────────────────────────────────────────────────────

#[test]
fn phase7_tiny_det_map_lookup_and_sorted_iter() {
    let mut m: TinyDetMap<&str, i32> = TinyDetMap::new();
    for &k in &["zebra", "apple", "mango", "banana"] {
        m.insert(k, 0);
    }
    let keys: Vec<&&str> = m.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, vec![&"apple", &"banana", &"mango", &"zebra"]);
}

#[test]
fn phase7_tiny_det_map_deterministic_under_permutation() {
    let mut a: TinyDetMap<i32, &str> = TinyDetMap::new();
    let mut b: TinyDetMap<i32, &str> = TinyDetMap::new();
    for &(k, v) in &[(1, "a"), (2, "b"), (3, "c"), (4, "d")] {
        a.insert(k, v);
    }
    for &(k, v) in &[(4, "d"), (1, "a"), (3, "c"), (2, "b")] {
        b.insert(k, v);
    }
    let av: Vec<_> = a.iter().collect();
    let bv: Vec<_> = b.iter().collect();
    assert_eq!(av, bv);
}

// ── SortedVecMap ────────────────────────────────────────────────────────

#[test]
fn phase7_sorted_vec_map_binary_lookup() {
    let mut m: SortedVecMap<i32, i32> = SortedVecMap::new();
    for k in 0..200 {
        m.insert(k, k * 10);
    }
    for k in 0..200 {
        assert_eq!(m.get(&k), Some(&(k * 10)));
    }
    assert_eq!(m.get(&200), None);
    assert_eq!(m.get(&-1), None);
}

#[test]
fn phase7_sorted_vec_map_iter_is_sorted() {
    let m: SortedVecMap<i32, &str> =
        SortedVecMap::from_iter_unsorted(vec![(3, "c"), (1, "a"), (4, "d"), (2, "b")]);
    let keys: Vec<i32> = m.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, vec![1, 2, 3, 4]);
}

#[test]
fn phase7_sorted_vec_map_range_query() {
    let m: SortedVecMap<i32, &str> = SortedVecMap::from_sorted_unique(vec![
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
        (5, "e"),
    ]);
    let r: Vec<_> = m.range(&2, &5).map(|(k, _)| *k).collect();
    assert_eq!(r, vec![2, 3, 4]);
}

// ── DetOpenMap ──────────────────────────────────────────────────────────

#[test]
fn phase7_det_open_map_basic_insert_get_remove() {
    let mut m: DetOpenMap<i32, &str> = DetOpenMap::new();
    m.insert(1, "a");
    m.insert(2, "b");
    m.insert(3, "c");
    assert_eq!(m.get(&1), Some(&"a"));
    assert_eq!(m.remove(&2), Some("b"));
    assert_eq!(m.get(&2), None);
    assert_eq!(m.len(), 2);
}

#[test]
fn phase7_det_open_map_resize_preserves_entries() {
    let mut m: DetOpenMap<i32, i32> = DetOpenMap::new();
    for i in 0..5_000 {
        m.insert(i, i * 7);
    }
    for i in 0..5_000 {
        assert_eq!(m.get(&i), Some(&(i * 7)));
    }
    assert_eq!(m.len(), 5_000);
}

#[test]
fn phase7_det_open_map_iter_sorted_is_canonical() {
    // Two maps inserted in different orders should produce identical
    // sorted iteration. Raw `iter()` is *not* a canonical surface;
    // `iter_sorted()` is.
    let mut a: DetOpenMap<i32, i32> = DetOpenMap::new();
    let mut b: DetOpenMap<i32, i32> = DetOpenMap::new();
    for i in [1, 2, 3, 4, 5] {
        a.insert(i, i * 10);
    }
    for i in [5, 4, 3, 2, 1] {
        b.insert(i, i * 10);
    }
    let av: Vec<i32> = a.iter_sorted().into_iter().map(|(_, v)| *v).collect();
    let bv: Vec<i32> = b.iter_sorted().into_iter().map(|(_, v)| *v).collect();
    assert_eq!(av, bv);
}

#[test]
fn phase7_det_open_map_overflow_count_zero_for_small_workload() {
    let mut m: DetOpenMap<i32, i32> = DetOpenMap::new();
    for i in 0..100 {
        m.insert(i, i);
    }
    // Small workload should not trigger probe-budget overflow.
    assert_eq!(m.overflow_count(), 0);
}

// ── Canonical-output rule ───────────────────────────────────────────────

#[test]
fn phase7_canonical_output_uses_sorted_structure() {
    // Rule: when surfacing iteration order to public output, a
    // collection MUST produce sorted order (BTreeMap, SortedVecMap,
    // TinyDetMap, or `*::iter_sorted()`). Random unsorted iter from
    // a hash-style map is internal-only.
    //
    // Pin: SortedVecMap and TinyDetMap iterate sorted by construction;
    // DHarht::iter_sorted and DetOpenMap::iter_sorted exist explicitly.
    let m: SortedVecMap<&str, i32> =
        SortedVecMap::from_iter_unsorted(vec![("c", 3), ("a", 1), ("b", 2)]);
    let serialized: Vec<&&str> = m.iter().map(|(k, _)| k).collect();
    assert_eq!(serialized, vec![&"a", &"b", &"c"]);
}
