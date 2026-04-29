//! v3 Phase 7 — `DHarht` Memory profile backend tests.
//!
//! Pins the architectural contract:
//! - splitmix64 deterministic scattering (no randomized seeds)
//! - 256-shard power-of-two layout
//! - sealed sparse 16-bit front directory
//! - MicroBucket16 with deterministic `BTreeMap` overflow
//! - full key equality on every successful lookup
//! - no silent entry loss across `seal_for_lookup`
//! - deterministic shape across runs

use cjc_data::detcoll::dharht::{deterministic_shape_hash, MICROBUCKET_CAPACITY};
use cjc_data::detcoll::{DHarht, LookupProfile};
use std::collections::BTreeMap;

#[test]
fn phase7_default_profile_is_memory() {
    let t: DHarht<u32> = DHarht::new();
    assert_eq!(t.profile(), LookupProfile::Memory);
}

#[test]
fn phase7_insert_get_update_roundtrip() {
    let mut t: DHarht<u32> = DHarht::new();
    assert!(t.is_empty());
    assert_eq!(t.insert_bytes(b"alpha", 1), None);
    assert_eq!(t.insert_bytes(b"beta", 2), None);
    assert_eq!(t.insert_bytes(b"alpha", 11), Some(1));
    assert_eq!(t.get_bytes(b"alpha"), Some(&11));
    assert_eq!(t.get_bytes(b"beta"), Some(&2));
    assert_eq!(t.get_bytes(b"missing"), None);
    assert_eq!(t.len(), 2);
}

#[test]
fn phase7_seal_for_lookup_preserves_all_entries() {
    let mut t: DHarht<u32> = DHarht::new();
    for i in 0..5_000u32 {
        t.insert_bytes(&i.to_be_bytes(), i);
    }
    let len_before = t.len();
    t.seal_for_lookup();
    assert!(t.is_sealed());
    assert_eq!(t.len(), len_before);
    for i in 0..5_000u32 {
        assert_eq!(t.get_bytes(&i.to_be_bytes()), Some(&i));
    }
}

#[test]
fn phase7_sealed_lookup_after_compaction_preserves_values() {
    let mut t: DHarht<Vec<u8>> = DHarht::new();
    for i in 0..2_000u32 {
        let key = format!("user_{:08}", i).into_bytes();
        t.insert_bytes(&key, i.to_le_bytes().to_vec());
    }
    t.seal_for_lookup();
    for i in 0..2_000u32 {
        let key = format!("user_{:08}", i).into_bytes();
        assert_eq!(t.get_bytes(&key), Some(&i.to_le_bytes().to_vec()));
    }
}

#[test]
fn phase7_full_key_equality_no_false_positive() {
    // Construct a workload large enough that some shard+prefix pairs
    // collide — `get_bytes` must still return only the inserted key's
    // value, never another collision-bucket-mate's value.
    let mut t: DHarht<u32> = DHarht::new();
    for i in 0..10_000u32 {
        let mut k = b"row_".to_vec();
        k.extend_from_slice(&i.to_be_bytes());
        t.insert_bytes(&k, i);
    }
    // Probe with un-inserted keys that share length/prefix.
    for i in 10_000..15_000u32 {
        let mut k = b"row_".to_vec();
        k.extend_from_slice(&i.to_be_bytes());
        assert_eq!(t.get_bytes(&k), None, "false positive on key {}", i);
    }
}

#[test]
fn phase7_microbucket16_overflow_falls_back_safely() {
    // At 50_000 entries some shard+prefix microbucket should overflow.
    // The contract: every key is still findable; max bucket size
    // never exceeds MICROBUCKET_CAPACITY.
    let mut t: DHarht<u32> = DHarht::new();
    for i in 0..50_000u32 {
        t.insert_bytes(&i.to_le_bytes(), i);
    }
    assert!(t.max_bucket_size() as usize <= MICROBUCKET_CAPACITY);
    for i in 0..50_000u32 {
        assert_eq!(t.get_bytes(&i.to_le_bytes()), Some(&i));
    }
    assert_eq!(t.len(), 50_000);
}

#[test]
fn phase7_deterministic_shape_double_run() {
    fn build() -> DHarht<u32> {
        let mut t: DHarht<u32> = DHarht::new();
        // Mix of insertions and updates.
        for i in 0..1_000u32 {
            t.insert_bytes(&i.to_be_bytes(), i);
        }
        for i in 0..500u32 {
            t.insert_bytes(&i.to_be_bytes(), i + 10_000);
        }
        t.seal_for_lookup();
        t
    }
    let h1 = deterministic_shape_hash(&build());
    let h2 = deterministic_shape_hash(&build());
    let h3 = deterministic_shape_hash(&build());
    assert_eq!(h1, h2);
    assert_eq!(h2, h3);
}

#[test]
fn phase7_iter_sorted_is_canonical_regardless_of_insertion_order() {
    let mut a: DHarht<u32> = DHarht::new();
    let mut b: DHarht<u32> = DHarht::new();
    for i in 0..200u32 {
        a.insert_bytes(&i.to_be_bytes(), i);
    }
    for i in (0..200u32).rev() {
        b.insert_bytes(&i.to_be_bytes(), i);
    }
    let av: Vec<(Vec<u8>, u32)> = a
        .iter_sorted()
        .into_iter()
        .map(|(k, v)| (k.to_vec(), *v))
        .collect();
    let bv: Vec<(Vec<u8>, u32)> = b
        .iter_sorted()
        .into_iter()
        .map(|(k, v)| (k.to_vec(), *v))
        .collect();
    assert_eq!(av, bv);
}

#[test]
fn phase7_matches_btreemap_oracle() {
    use cjc_data::detcoll::DHarht;
    let mut t: DHarht<u32> = DHarht::new();
    let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    let mut x: u64 = 0xCAFEBABE;
    for _ in 0..3_000 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let k = (x as u32 % 1_000).to_be_bytes().to_vec();
        let v = (x >> 8) as u32;
        let p1 = t.insert_bytes(&k, v);
        let p2 = oracle.insert(k, v);
        assert_eq!(p1, p2, "insert returned different prev value");
    }
    for (k, v) in &oracle {
        assert_eq!(t.get_bytes(k), Some(v));
    }
    assert_eq!(t.len() as usize, oracle.len());
}

#[test]
fn phase7_empty_table_lookup_returns_none() {
    let t: DHarht<u32> = DHarht::new();
    assert_eq!(t.get_bytes(b"any"), None);
    assert!(!t.contains_bytes(b"any"));
    assert_eq!(t.len(), 0);
}

#[test]
fn phase7_zero_length_key_works() {
    let mut t: DHarht<u32> = DHarht::new();
    t.insert_bytes(b"", 42);
    assert_eq!(t.get_bytes(b""), Some(&42));
    assert!(t.contains_bytes(b""));
    assert_eq!(t.len(), 1);
}

#[test]
fn phase7_long_key_works() {
    let mut t: DHarht<u32> = DHarht::new();
    let key = vec![0xABu8; 4096];
    t.insert_bytes(&key, 7);
    assert_eq!(t.get_bytes(&key), Some(&7));
}

// ── Bench (ignored, same-process headline) ─────────────────────────────

#[test]
#[ignore]
fn bench_phase7_dharht_vs_btreemap_lookup() {
    use std::time::Instant;

    const N: usize = 100_000;
    const PROBE_ITERS: usize = 1_000_000;

    // Build inputs.
    let keys: Vec<Vec<u8>> = (0..N as u32)
        .map(|i| format!("user_id_{:08}", i).into_bytes())
        .collect();

    // BTreeMap baseline.
    let mut bt: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        bt.insert(k.clone(), i as u32);
    }

    // DHarht sealed.
    let mut dh: cjc_data::detcoll::DHarht<u32> = cjc_data::detcoll::DHarht::new();
    for (i, k) in keys.iter().enumerate() {
        dh.insert_bytes(k, i as u32);
    }
    dh.seal_for_lookup();

    // Hot probe sequence: cyclic over keys.
    let probe = |i: usize| &keys[i % N];

    // Warm.
    for i in 0..1000 {
        std::hint::black_box(bt.get(probe(i)));
        std::hint::black_box(dh.get_bytes(probe(i)));
    }

    let t = Instant::now();
    let mut acc: u32 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(*bt.get(probe(i)).unwrap());
    }
    let bt_time = t.elapsed();
    std::hint::black_box(acc);

    let t = Instant::now();
    let mut acc: u32 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(*dh.get_bytes(probe(i)).unwrap());
    }
    let dh_time = t.elapsed();
    std::hint::black_box(acc);

    let speedup = bt_time.as_secs_f64() / dh_time.as_secs_f64();
    eprintln!(
        "\n[Phase 7 bench] sealed lookup, {N} keys, {PROBE_ITERS} probes:\n\
         \x20\x20BTreeMap:    {bt_time:?}\n\
         \x20\x20DHarht (sealed): {dh_time:?}\n\
         \x20\x20Speedup:     {speedup:.2}×\n",
    );
}

#[test]
#[ignore]
fn bench_phase9_dharht_vs_btreemap_vs_hashmap() {
    use std::collections::HashMap;
    use std::time::Instant;

    const N: usize = 100_000;
    const PROBE_ITERS: usize = 1_000_000;

    let keys: Vec<Vec<u8>> = (0..N as u32)
        .map(|i| format!("user_id_{:08}", i).into_bytes())
        .collect();

    let mut bt: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    let mut hm: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut dh: cjc_data::detcoll::DHarht<u32> = cjc_data::detcoll::DHarht::new();
    for (i, k) in keys.iter().enumerate() {
        bt.insert(k.clone(), i as u32);
        hm.insert(k.clone(), i as u32);
        dh.insert_bytes(k, i as u32);
    }
    dh.seal_for_lookup();

    let probe = |i: usize| &keys[i % N];
    for i in 0..1000 {
        std::hint::black_box(bt.get(probe(i)));
        std::hint::black_box(hm.get(probe(i)));
        std::hint::black_box(dh.get_bytes(probe(i)));
    }

    let t = Instant::now();
    let mut acc: u32 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(*bt.get(probe(i)).unwrap());
    }
    let bt_time = t.elapsed();
    std::hint::black_box(acc);

    let t = Instant::now();
    let mut acc: u32 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(*hm.get(probe(i)).unwrap());
    }
    let hm_time = t.elapsed();
    std::hint::black_box(acc);

    let t = Instant::now();
    let mut acc: u32 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(*dh.get_bytes(probe(i)).unwrap());
    }
    let dh_time = t.elapsed();
    std::hint::black_box(acc);

    eprintln!(
        "\n[Phase 9 bench] sealed lookup, {N} keys, {PROBE_ITERS} probes:\n\
         \x20\x20BTreeMap:        {bt_time:?}\n\
         \x20\x20HashMap (std):   {hm_time:?}\n\
         \x20\x20DHarht (sealed): {dh_time:?}\n\
         \x20\x20DHarht / BTreeMap: {:.2}× (>1.0 = DHarht faster)\n\
         \x20\x20DHarht / HashMap:  {:.2}× (>1.0 = DHarht faster)\n",
        bt_time.as_secs_f64() / dh_time.as_secs_f64(),
        hm_time.as_secs_f64() / dh_time.as_secs_f64(),
    );
}

#[test]
#[ignore]
fn bench_phase7_categorical_dictionary_lookup() {
    use cjc_data::byte_dict::ByteDictionary;
    use std::time::Instant;

    const N: usize = 50_000;
    const PROBE_ITERS: usize = 500_000;

    let keys: Vec<Vec<u8>> = (0..N as u32)
        .map(|i| format!("cat_{:08}", i).into_bytes())
        .collect();

    // Pre-seal dictionary (BTreeMap-backed).
    let mut pre = ByteDictionary::new();
    for k in &keys {
        pre.intern(k).unwrap();
    }

    // Sealed dictionary (DHarht-backed).
    let mut sealed = ByteDictionary::new();
    for k in &keys {
        sealed.intern(k).unwrap();
    }
    sealed.seal_for_lookup();

    let probe = |i: usize| &keys[i % N];

    // Warm.
    for i in 0..1000 {
        std::hint::black_box(pre.lookup(probe(i)));
        std::hint::black_box(sealed.lookup(probe(i)));
    }

    let t = Instant::now();
    let mut acc: u64 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(pre.lookup(probe(i)).unwrap());
    }
    let pre_time = t.elapsed();
    std::hint::black_box(acc);

    let t = Instant::now();
    let mut acc: u64 = 0;
    for i in 0..PROBE_ITERS {
        acc = acc.wrapping_add(sealed.lookup(probe(i)).unwrap());
    }
    let sealed_time = t.elapsed();
    std::hint::black_box(acc);

    let speedup = pre_time.as_secs_f64() / sealed_time.as_secs_f64();
    eprintln!(
        "\n[Phase 7 bench] ByteDictionary lookup, {N} categories, {PROBE_ITERS} probes:\n\
         \x20\x20Pre-seal (BTreeMap):    {pre_time:?}\n\
         \x20\x20Sealed (DHarht Memory): {sealed_time:?}\n\
         \x20\x20Speedup:                {speedup:.2}×\n",
    );
}
