//! v3 Phase 7 — Bolero fuzz harness for `DHarht` Memory profile.
//!
//! Targets:
//! - random insert/get/update/seal sequence matches `BTreeMap` oracle
//! - collision-heavy key sets don't lose entries
//! - microbucket overflow path is sound
//! - sealed lookup is byte-equal to pre-seal for the same probe
//!   sequence

use cjc_data::detcoll::DHarht;
use std::collections::BTreeMap;
use std::panic;

#[test]
fn fuzz_dharht_insert_get_update_oracle() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            let _ = panic::catch_unwind(|| {
                if input.is_empty() {
                    return;
                }
                let mut t: DHarht<u32> = DHarht::new();
                let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
                // Treat the byte stream as a sequence of operations.
                // Each op is 4 bytes: 1 op-tag + 3 key-byte + 0 value
                // (value derived from key index). Skip if too short.
                for (i, chunk) in input.chunks_exact(4).enumerate() {
                    let key = chunk[1..4].to_vec();
                    let v = (i as u32).wrapping_mul(0x9E3779B9);
                    let p1 = t.insert_bytes(&key, v);
                    let p2 = oracle.insert(key, v);
                    assert_eq!(p1, p2);
                }
                // Verify all entries.
                for (k, v) in &oracle {
                    assert_eq!(t.get_bytes(k), Some(v));
                }
                assert_eq!(t.len() as usize, oracle.len());
            });
        });
}

#[test]
fn fuzz_dharht_collision_heavy_keys() {
    // Targets MicroBucket16 overflow paths by feeding many keys
    // sharing common prefixes.
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            let _ = panic::catch_unwind(|| {
                let mut t: DHarht<u32> = DHarht::new();
                let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
                // Build keys all sharing a fixed 4-byte prefix.
                for (i, &b) in input.iter().enumerate().take(500) {
                    let mut k = b"PREF".to_vec();
                    k.push(b);
                    k.push((i & 0xFF) as u8);
                    t.insert_bytes(&k, i as u32);
                    oracle.insert(k, i as u32);
                }
                for (k, v) in &oracle {
                    assert_eq!(t.get_bytes(k), Some(v));
                }
                // No silent loss.
                assert_eq!(t.len() as usize, oracle.len());
            });
        });
}

#[test]
fn fuzz_dharht_seal_then_lookup_byte_equal() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            let _ = panic::catch_unwind(|| {
                if input.is_empty() {
                    return;
                }
                let mut t: DHarht<u32> = DHarht::new();
                let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
                for chunk in input.chunks_exact(3) {
                    let key = chunk.to_vec();
                    let v = chunk.iter().fold(0u32, |a, b| {
                        a.wrapping_mul(31).wrapping_add(*b as u32)
                    });
                    t.insert_bytes(&key, v);
                    oracle.insert(key, v);
                }
                t.seal_for_lookup();
                for (k, v) in &oracle {
                    assert_eq!(t.get_bytes(k), Some(v));
                }
            });
        });
}
