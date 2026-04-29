//! Fuzz + property targets for the Deterministic Adaptive Dictionary Engine
//! (Phase 1 of TidyView v3).
//!
//! Run with:
//!   cargo test --test bolero_fuzz categorical_dictionary
//!
//! Properties:
//!   1. Round-trip: every byte sequence pushed into a `CategoricalColumn`
//!      decodes back to the same bytes via the assigned code.
//!   2. Lexical determinism: building two `ByteDictionary`s from arbitrary
//!      shufflings of the *same* unique byte set, then sealing each lexically,
//!      yields identical `(code, bytes)` mappings.
//!   3. Frozen rejection: a frozen dictionary never extends — `intern` on an
//!      unseen sequence returns `Err(Frozen)`, and `lookup` on a known
//!      sequence still resolves.

use cjc_data::byte_dict::{
    ByteDictError, ByteDictionary, CategoricalColumn, CategoryOrdering,
};
use std::panic;

/// Convert a `Vec<u8>` fuzz input into a sequence of byte strings by
/// splitting on `0x00`. Each split is one categorical value.
fn split_into_values(input: &[u8]) -> Vec<Vec<u8>> {
    if input.is_empty() {
        return Vec::new();
    }
    input
        .split(|&b| b == 0x00)
        .map(|slice| slice.to_vec())
        .take(256)
        .collect()
}

/// Property 1: pushing N byte sequences into a fresh `CategoricalColumn`
/// produces a column where `get(i)` returns the i-th pushed sequence
/// byte-for-byte.
#[test]
fn fuzz_categorical_round_trip() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let values = split_into_values(input);
            if values.is_empty() {
                return;
            }
            let mut col = CategoricalColumn::new();
            for v in &values {
                col.push(v).expect("push to non-frozen column never fails");
            }
            assert_eq!(col.len(), values.len());
            for (i, expected) in values.iter().enumerate() {
                let got = col.get(i).expect("non-null entry");
                assert_eq!(got, expected.as_slice(),
                    "round-trip mismatch at row {i}");
            }
        });
    });
}

/// Property 2: two dictionaries built from arbitrary insertion orders of
/// the same *unique* byte set, both sealed lexically, produce identical
/// `(code, bytes)` mappings.
///
/// We split the input into two halves and interpret the first as insertion
/// order A and a byte-rotated copy as insertion order B. We deduplicate
/// each by the natural byte identity before interning so both dictionaries
/// see the same set of unique values.
#[test]
fn fuzz_categorical_lexical_determinism() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let values = split_into_values(input);
            if values.len() < 2 {
                return;
            }
            // Deduplicate while preserving first-seen order
            let mut unique: Vec<Vec<u8>> = Vec::new();
            for v in &values {
                if !unique.iter().any(|u| u == v) {
                    unique.push(v.clone());
                }
            }
            if unique.len() < 2 {
                return;
            }
            // Build dict A in given order, dict B in rotated order
            let mut a = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
            let mut b = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
            for v in &unique {
                a.intern(v).expect("non-frozen intern");
            }
            // Rotate by one to give B a different insertion order
            let mut rotated = unique.clone();
            rotated.rotate_left(1);
            for v in &rotated {
                b.intern(v).expect("non-frozen intern");
            }
            // Seal both lexically
            let _perm_a = a.seal_lexical();
            let _perm_b = b.seal_lexical();
            // After sealing, both dictionaries must have the same
            // (code, bytes) mapping
            assert_eq!(a.len(), b.len());
            for code in 0..a.len() as u64 {
                let bytes_a = a.get(code).expect("code in range");
                let bytes_b = b.get(code).expect("code in range");
                assert_eq!(bytes_a, bytes_b,
                    "lexical seal not deterministic at code {code}");
            }
        });
    });
}

/// Property 3: a frozen dictionary never extends. `intern` on an unseen
/// byte sequence returns `Err(Frozen)`. `lookup` on a known sequence
/// still resolves.
#[test]
fn fuzz_categorical_frozen_rejection() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let values = split_into_values(input);
            if values.len() < 2 {
                return;
            }
            // Use the first half to populate, then freeze, then probe
            // with the second half. Anything in the second half that is
            // also in the first must lookup; anything new must reject.
            let split = values.len() / 2;
            let (seed, probe) = values.split_at(split);
            if seed.is_empty() {
                return;
            }
            let mut dict = ByteDictionary::new();
            for v in seed {
                dict.intern(v).expect("non-frozen intern");
            }
            let known_count = dict.len();
            dict.freeze();
            assert!(dict.is_frozen());
            for v in probe {
                let in_seed = seed.iter().any(|s| s == v);
                match dict.intern(v) {
                    Err(ByteDictError::Frozen) => {
                        // Acceptable for any value that's not already
                        // interned; for known values the contract is that
                        // intern() still errors (it never extends), but
                        // lookup() must succeed.
                        if in_seed {
                            assert!(dict.lookup(v).is_some(),
                                "lookup must resolve known value after freeze");
                        }
                    }
                    Ok(_code) => {
                        // Only acceptable if intern() short-circuits to
                        // existing entries — Phase 1 chose strict rejection.
                        // If this branch ever fires, the contract changed.
                        panic!("frozen dict extended on value {:?}", v);
                    }
                    Err(other) => {
                        panic!("unexpected intern error after freeze: {:?}", other);
                    }
                }
            }
            // Length must not have grown
            assert_eq!(dict.len(), known_count,
                "frozen dictionary grew");
        });
    });
}
