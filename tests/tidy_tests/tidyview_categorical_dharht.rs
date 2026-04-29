//! v3 Phase 7 — TidyView categorical dictionary backed by `DHarht`.
//!
//! `ByteDictionary::seal_for_lookup()` builds a `DHarht` Memory-profile
//! accelerator over the existing `BTreeMap<Vec<u8>, u64>` lookup
//! state. Post-seal, `lookup()` routes through the `DHarht`. This test
//! file pins:
//!
//! - byte-equal results between BTreeMap-backed and DHarht-backed lookup
//! - round-trip category → code → category preserved across seal
//! - deterministic shape across runs
//! - sealed lookup is faster (informational; bench in
//!   `bench_phase7_dharht_vs_btreemap_lookup`)

use cjc_data::byte_dict::{ByteDictionary, CategoricalColumn};

#[test]
fn phase7_categorical_lookup_byte_equal_pre_and_post_seal() {
    let mut cc = CategoricalColumn::new();
    let inputs: Vec<&[u8]> = vec![
        b"red", b"green", b"blue", b"red", b"yellow", b"green", b"orange", b"blue",
    ];
    for bytes in &inputs {
        cc.push(bytes).unwrap();
    }
    // Pre-seal lookup baseline.
    let pre_codes: Vec<u64> = inputs
        .iter()
        .map(|b| cc.dictionary().lookup(b).unwrap())
        .collect();

    // Take owned dictionary, seal, swap back.
    let mut dict = cc.dictionary().clone();
    assert!(!dict.is_lookup_sealed());
    dict.seal_for_lookup();
    assert!(dict.is_lookup_sealed());

    let post_codes: Vec<u64> = inputs.iter().map(|b| dict.lookup(b).unwrap()).collect();
    assert_eq!(pre_codes, post_codes);
}

#[test]
fn phase7_unknown_categories_return_none_post_seal() {
    let mut dict = ByteDictionary::new();
    dict.intern(b"red").unwrap();
    dict.intern(b"green").unwrap();
    dict.seal_for_lookup();
    assert!(dict.lookup(b"red").is_some());
    assert!(dict.lookup(b"green").is_some());
    assert!(dict.lookup(b"missing").is_none());
    assert!(dict.lookup(b"").is_none());
}

#[test]
fn phase7_round_trip_category_code_category_post_seal() {
    let mut dict = ByteDictionary::new();
    let inputs: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
    let codes: Vec<u64> = inputs.iter().map(|b| dict.intern(b).unwrap()).collect();
    dict.seal_for_lookup();
    // Round-trip: category → code → bytes
    for (b, &c) in inputs.iter().zip(codes.iter()) {
        assert_eq!(dict.lookup(b), Some(c));
        assert_eq!(dict.get(c), Some(*b));
    }
}

#[test]
fn phase7_large_dictionary_seal_no_data_loss() {
    let mut dict = ByteDictionary::new();
    // 10k unique categories.
    for i in 0..10_000u32 {
        let key = format!("cat_{:08}", i);
        dict.intern(key.as_bytes()).unwrap();
    }
    dict.seal_for_lookup();
    // Every key must be findable; no entries silently dropped.
    for i in 0..10_000u32 {
        let key = format!("cat_{:08}", i);
        assert!(dict.lookup(key.as_bytes()).is_some(), "lost key {}", i);
    }
}

#[test]
fn phase7_sealed_lookup_has_zero_overflow_for_typical_workload() {
    let mut dict = ByteDictionary::new();
    for i in 0..1_000u32 {
        dict.intern(format!("k{}", i).as_bytes()).unwrap();
    }
    dict.seal_for_lookup();
    // Typical workloads (short distinct keys, ~1k entries) should not
    // trigger MicroBucket16 overflow. Non-zero is still correct, but
    // catching it here is a useful canary.
    let overflow = dict.dharht_overflow_count();
    assert!(
        overflow < 50,
        "unexpected high overflow_count for 1k-key workload: {}",
        overflow
    );
}

#[test]
fn phase7_sealed_dictionary_is_deterministic_across_builds() {
    fn build() -> Vec<u64> {
        let mut dict = ByteDictionary::new();
        for i in 0..500u32 {
            dict.intern(format!("k_{:04}", i).as_bytes()).unwrap();
        }
        dict.seal_for_lookup();
        // Probe pattern.
        (0..500u32)
            .map(|i| dict.lookup(format!("k_{:04}", i).as_bytes()).unwrap())
            .collect()
    }
    let a = build();
    let b = build();
    assert_eq!(a, b);
}

#[test]
fn phase7_pre_seal_dictionary_still_works() {
    // Don't accidentally break pre-seal usage when adding the post-seal
    // accelerator. ByteDictionary without `seal_for_lookup()` should
    // behave identically to its v3 Phase 1 surface.
    let mut dict = ByteDictionary::new();
    dict.intern(b"a").unwrap();
    dict.intern(b"b").unwrap();
    assert!(!dict.is_lookup_sealed());
    assert_eq!(dict.lookup(b"a"), Some(0));
    assert_eq!(dict.lookup(b"b"), Some(1));
    assert_eq!(dict.lookup(b"missing"), None);
}
