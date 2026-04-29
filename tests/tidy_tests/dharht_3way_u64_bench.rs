//! v3 Phase 10 — 3-way comparison: BTreeMap vs DHarht v.01 (byte-key) vs
//! DHarhtMemory (port of user-supplied D-HARHT Memory profile, u64-key).
//!
//! Bench shape mirrors the user's source bench
//! (`D-HARHT-Blueprint-and-Code.md` `dharht_bench.rs`):
//! - 100k keys, 2M lookups
//! - keys are `deterministic_permutation_scatter(idx ^ golden_ratio)`
//! - lookup stream is a randomized walk through the key set (NOT cyclic
//!   `i % N`) — this matters for cache behavior; the user's bench
//!   reported `D-HARHT memory ≈ 37 ns/op, HashMap ≈ 37 ns/op`.
//!
//! For the byte-key DHarht v.01 we encode the u64 key as 8 little-endian
//! bytes so the same key set drives all three.
//!
//! Determinism + security tests live alongside the bench so the user's
//! "test for speed, determinism, and security" ask is covered in one
//! file.

use cjc_data::detcoll::dharht_memory::{
    deterministic_permutation_scatter, shape_hash, DHarhtMemory,
};
use cjc_data::detcoll::DHarht;
use std::collections::{BTreeMap, HashMap};

const N_KEYS: usize = 100_000;
const N_LOOKUPS: usize = 2_000_000;

fn make_keys(n: usize) -> Vec<u64> {
    (0..n)
        .map(|idx| {
            let base = idx as u64;
            deterministic_permutation_scatter(base ^ 0x9e37_79b9_7f4a_7c15)
        })
        .collect()
}

fn make_lookup_stream(keys: &[u64], n: usize) -> Vec<u64> {
    let mut stream = Vec::with_capacity(n);
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    for _ in 0..n {
        state = deterministic_permutation_scatter(state);
        let slot = state as usize % keys.len();
        stream.push(keys[slot]);
    }
    stream
}

// ── Determinism tests ─────────────────────────────────────────────────

#[test]
fn phase10_dharht_memory_double_build_byte_equal() {
    let keys = make_keys(2_000);
    fn build(keys: &[u64]) -> DHarhtMemory<u64> {
        let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
        for &k in keys {
            t.insert(k, k.wrapping_mul(31));
        }
        t.seal_for_lookup();
        t
    }
    let h1 = shape_hash(&build(&keys));
    let h2 = shape_hash(&build(&keys));
    let h3 = shape_hash(&build(&keys));
    assert_eq!(h1, h2);
    assert_eq!(h2, h3);
}

#[test]
fn phase10_dharht_memory_iter_sorted_canonical_regardless_of_insert_order() {
    let mut a: DHarhtMemory<u64> = DHarhtMemory::new();
    let mut b: DHarhtMemory<u64> = DHarhtMemory::new();
    for i in 0..500u64 {
        a.insert(i, i);
    }
    for i in (0..500u64).rev() {
        b.insert(i, i);
    }
    a.seal_for_lookup();
    b.seal_for_lookup();
    let av: Vec<(u64, u64)> = a.iter_sorted().into_iter().map(|(k, v)| (k, *v)).collect();
    let bv: Vec<(u64, u64)> = b.iter_sorted().into_iter().map(|(k, v)| (k, *v)).collect();
    assert_eq!(av, bv);
}

#[test]
fn phase10_dharht_memory_matches_btreemap_oracle_under_random_workload() {
    let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
    let mut oracle: BTreeMap<u64, u64> = BTreeMap::new();
    let mut x: u64 = 0xCAFEBABE;
    for _ in 0..10_000 {
        x = deterministic_permutation_scatter(x);
        let p1 = t.insert(x, x);
        let p2 = oracle.insert(x, x);
        assert_eq!(p1, p2);
    }
    t.seal_for_lookup();
    for (k, v) in &oracle {
        assert_eq!(t.get(*k), Some(v));
    }
}

// ── Security tests ────────────────────────────────────────────────────

#[test]
fn phase10_dharht_memory_no_silent_loss_at_scale() {
    let keys = make_keys(N_KEYS);
    let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
    for &k in &keys {
        t.insert(k, k);
    }
    t.seal_for_lookup();
    for &k in &keys {
        assert_eq!(t.get(k), Some(&k), "lost key {}", k);
    }
    assert_eq!(t.len() as usize, N_KEYS);
}

#[test]
fn phase10_dharht_memory_full_key_equality_no_false_positive() {
    let keys = make_keys(20_000);
    let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
    for &k in &keys {
        t.insert(k, k);
    }
    t.seal_for_lookup();
    // Probe with un-inserted keys — must return None even when their
    // scatter shares a shard+prefix with an inserted key.
    let probes = make_keys(40_000);
    let inserted: std::collections::BTreeSet<u64> = keys.iter().copied().collect();
    for &p in probes.iter().skip(20_000).take(5_000) {
        if !inserted.contains(&p) {
            assert_eq!(t.get(p), None);
        }
    }
}

#[test]
fn phase10_dharht_memory_collision_overflow_diagnostic_surface() {
    // Force overflow with synthetic adversarial input. We can't easily
    // construct prefix collisions without reverse-engineering the hash,
    // but we can document that the surface exists.
    let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
    for i in 0..100u64 {
        t.insert(i, i);
    }
    t.seal_for_lookup();
    // With 100 keys spread over 65k slots, overflow should be 0.
    assert_eq!(t.micro_overflow_count(), 0);
    // Max collision group is well-defined and queryable.
    let _ = t.max_collision_group();
}

// ── 3-way speed bench (ignored) ───────────────────────────────────────

#[test]
#[ignore]
fn bench_phase10_three_way_u64_keys() {
    use std::hint::black_box;
    use std::time::Instant;

    let keys = make_keys(N_KEYS);
    let lookup_keys = make_lookup_stream(&keys, N_LOOKUPS);

    // BTreeMap baseline.
    let mut bt: BTreeMap<u64, u64> = BTreeMap::new();
    let mut hm: HashMap<u64, u64> = HashMap::new();
    for (i, &k) in keys.iter().enumerate() {
        bt.insert(k, i as u64);
        hm.insert(k, i as u64);
    }

    // DHarht v.01 (byte-key, my implementation).
    let mut v01: DHarht<u64> = DHarht::new();
    for (i, &k) in keys.iter().enumerate() {
        v01.insert_bytes(&k.to_le_bytes(), i as u64);
    }
    v01.seal_for_lookup();

    // DHarht Memory (port of user-supplied blueprint).
    let mut mem: DHarhtMemory<u64> = DHarhtMemory::new();
    for (i, &k) in keys.iter().enumerate() {
        mem.insert(k, i as u64);
    }
    mem.seal_for_lookup();

    // Warm.
    for &k in lookup_keys.iter().take(8192) {
        black_box(bt.get(&k));
        black_box(hm.get(&k));
        black_box(v01.get_bytes(&k.to_le_bytes()));
        black_box(mem.get(k));
    }

    // BTreeMap.
    let t = Instant::now();
    let mut acc: u64 = 0;
    for &k in &lookup_keys {
        acc ^= bt.get(&k).copied().unwrap_or(0);
    }
    let bt_time = t.elapsed();
    black_box(acc);

    // HashMap (std SipHash baseline).
    let t = Instant::now();
    let mut acc: u64 = 0;
    for &k in &lookup_keys {
        acc ^= hm.get(&k).copied().unwrap_or(0);
    }
    let hm_time = t.elapsed();
    black_box(acc);

    // DHarht v.01.
    let t = Instant::now();
    let mut acc: u64 = 0;
    for &k in &lookup_keys {
        acc ^= v01.get_bytes(&k.to_le_bytes()).copied().unwrap_or(0);
    }
    let v01_time = t.elapsed();
    black_box(acc);

    // DHarht Memory.
    let t = Instant::now();
    let mut acc: u64 = 0;
    for &k in &lookup_keys {
        acc ^= mem.get(k).copied().unwrap_or(0);
    }
    let mem_time = t.elapsed();
    black_box(acc);

    let ns_per = |d: std::time::Duration| d.as_secs_f64() * 1e9 / N_LOOKUPS as f64;

    eprintln!(
        "\n[Phase 10 3-way bench] u64 keys, {} keys, {} lookups (random probe stream):\n\
         \x20\x20BTreeMap                {:>8.2} ns/op  ({:?})\n\
         \x20\x20HashMap (std SipHash)   {:>8.2} ns/op  ({:?})\n\
         \x20\x20DHarht v.01 (byte-key)  {:>8.2} ns/op  ({:?})\n\
         \x20\x20DHarht Memory (u64-key) {:>8.2} ns/op  ({:?})\n\
         \x20\x20DHarht-Mem / HashMap   = {:.2}× ({})\n\
         \x20\x20DHarht-Mem / BTreeMap  = {:.2}× ({})\n\
         \x20\x20v.01 / Memory          = {:.2}×\n",
        N_KEYS,
        N_LOOKUPS,
        ns_per(bt_time),
        bt_time,
        ns_per(hm_time),
        hm_time,
        ns_per(v01_time),
        v01_time,
        ns_per(mem_time),
        mem_time,
        hm_time.as_secs_f64() / mem_time.as_secs_f64(),
        if mem_time < hm_time { "Memory faster" } else { "HashMap faster" },
        bt_time.as_secs_f64() / mem_time.as_secs_f64(),
        if mem_time < bt_time { "Memory faster" } else { "BTreeMap faster" },
        v01_time.as_secs_f64() / mem_time.as_secs_f64(),
    );
}
