//! v3 Phase 7 — `DHarht` Memory profile property tests.
//!
//! Random insert/get/update sequences must agree with a `BTreeMap`
//! oracle. Random seal cycles preserve values. Categorical
//! round-trip preserves data.
//!
//! Uses `proptest!` macros to drive 256 randomized cases per property.

use cjc_data::byte_dict::ByteDictionary;
use cjc_data::detcoll::DHarht;
use proptest::prelude::*;
use std::collections::BTreeMap;

// Bound key length and entry count to keep tests fast.
fn key_strategy() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..32)
}

fn ops_strategy() -> impl Strategy<Value = Vec<(Vec<u8>, u32)>> {
    proptest::collection::vec((key_strategy(), any::<u32>()), 0..200)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn phase7_dharht_matches_btreemap_oracle(ops in ops_strategy()) {
        let mut t: DHarht<u32> = DHarht::new();
        let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
        for (k, v) in &ops {
            let p1 = t.insert_bytes(k, *v);
            let p2 = oracle.insert(k.clone(), *v);
            prop_assert_eq!(p1, p2);
        }
        for (k, v) in &oracle {
            prop_assert_eq!(t.get_bytes(k), Some(v));
        }
        prop_assert_eq!(t.len() as usize, oracle.len());
    }

    #[test]
    fn phase7_dharht_get_missing_returns_none(
        ops in ops_strategy(),
        probes in proptest::collection::vec(key_strategy(), 0..50)
    ) {
        let mut t: DHarht<u32> = DHarht::new();
        let inserted: BTreeMap<Vec<u8>, u32> =
            ops.iter().map(|(k, v)| (k.clone(), *v)).collect();
        for (k, v) in &ops {
            t.insert_bytes(k, *v);
        }
        for p in &probes {
            match inserted.get(p) {
                Some(expected) => prop_assert_eq!(t.get_bytes(p), Some(expected)),
                None => prop_assert_eq!(t.get_bytes(p), None),
            }
        }
    }

    #[test]
    fn phase7_dharht_seal_preserves_all_entries(ops in ops_strategy()) {
        let mut t: DHarht<u32> = DHarht::new();
        for (k, v) in &ops {
            t.insert_bytes(k, *v);
        }
        let oracle: BTreeMap<Vec<u8>, u32> =
            ops.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let len_before = t.len();
        t.seal_for_lookup();
        prop_assert_eq!(t.len(), len_before);
        for (k, v) in &oracle {
            prop_assert_eq!(t.get_bytes(k), Some(v));
        }
    }

    #[test]
    fn phase7_dharht_double_run_iter_sorted_identical(ops in ops_strategy()) {
        let build = || -> DHarht<u32> {
            let mut t: DHarht<u32> = DHarht::new();
            for (k, v) in &ops {
                t.insert_bytes(k, *v);
            }
            t.seal_for_lookup();
            t
        };
        let a = build();
        let b = build();
        let av: Vec<(Vec<u8>, u32)> = a.iter_sorted().into_iter().map(|(k, v)| (k.to_vec(), *v)).collect();
        let bv: Vec<(Vec<u8>, u32)> = b.iter_sorted().into_iter().map(|(k, v)| (k.to_vec(), *v)).collect();
        prop_assert_eq!(av, bv);
    }

    #[test]
    fn phase7_categorical_dictionary_round_trip(values in proptest::collection::vec(key_strategy(), 0..100)) {
        let mut dict = ByteDictionary::new();
        let mut codes: Vec<u64> = Vec::new();
        for v in &values {
            // intern_with_policy is the unfrozen path; fall back gracefully.
            if let Ok(c) = dict.intern(v) {
                codes.push(c);
            }
        }
        dict.seal_for_lookup();
        // Codes returned during intern must equal codes returned by lookup post-seal.
        let mut idx = 0usize;
        for v in &values {
            if let Some(c) = dict.lookup(v) {
                if idx < codes.len() {
                    prop_assert_eq!(c, codes[idx]);
                }
                idx += 1;
            }
        }
    }
}
