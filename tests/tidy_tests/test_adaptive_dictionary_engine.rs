//! Phase 1 — Deterministic Adaptive Dictionary Engine integration tests.
//!
//! These tests pin Phase-1 contract behaviour at the public-API level:
//!   - End-to-end build / read / null cycle
//!   - Code-width promotion across the integer thresholds
//!   - Lexical seal determinism across permuted insertion orders
//!   - Frozen-dict rejection contract
//!   - Unknown-category policies (Error / MapToOther / MapToNull / ExtendDictionary)
//!   - Profile statistics on a moderately sized column
//!
//! Phase 1 deliberately does NOT wire `CategoricalColumn` into `TidyView`
//! — that lives in Phase 2. These tests therefore exercise only the
//! standalone engine API; the verbs/parity tests stay untouched.

use cjc_data::byte_dict::{
    AdaptiveCodes, ByteDictError, ByteDictionary, CategoricalColumn,
    CategoryOrdering, UnknownCategoryPolicy,
};

#[test]
fn end_to_end_build_read_null_cycle() {
    let mut col = CategoricalColumn::new();
    col.push(b"alpha").unwrap();
    col.push(b"beta").unwrap();
    col.push_null();
    col.push(b"alpha").unwrap();
    col.push(b"gamma").unwrap();

    assert_eq!(col.len(), 5);
    assert_eq!(col.get(0).unwrap(), b"alpha");
    assert_eq!(col.get(1).unwrap(), b"beta");
    assert!(col.is_null(2));
    assert_eq!(col.get(2), None);
    assert_eq!(col.get(3).unwrap(), b"alpha");
    assert_eq!(col.get(4).unwrap(), b"gamma");

    // Dictionary holds 3 unique values, codes assigned in first-seen order.
    let dict = col.dictionary();
    assert_eq!(dict.len(), 3);
    assert_eq!(dict.lookup(b"alpha"), Some(0));
    assert_eq!(dict.lookup(b"beta"), Some(1));
    assert_eq!(dict.lookup(b"gamma"), Some(2));
}

#[test]
fn code_width_promotes_at_256() {
    let mut codes = AdaptiveCodes::new();
    assert_eq!(codes.width_bytes(), 1);
    for i in 0..256u64 {
        codes.push(i);
    }
    // Last code (255) still fits in U8.
    assert_eq!(codes.width_bytes(), 1);
    codes.push(256);
    assert_eq!(codes.width_bytes(), 2);
    // Round-trip the promoted value
    assert_eq!(codes.get(256), 256);
}

#[test]
fn code_width_promotes_at_65536() {
    let mut codes = AdaptiveCodes::with_capacity(65_537);
    for i in 0..65_536u64 {
        codes.push(i);
    }
    assert_eq!(codes.width_bytes(), 2);
    codes.push(65_536);
    assert_eq!(codes.width_bytes(), 4);
    assert_eq!(codes.get(65_536), 65_536);
}

#[test]
fn lexical_seal_is_permutation_invariant() {
    // Two dictionaries fed the same set of unique byte strings in two
    // different orders must, after lexical seal, agree on (code → bytes).
    let words: Vec<&[u8]> = vec![b"banana", b"apple", b"cherry", b"date", b"fig"];

    let mut a = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
    for w in &words {
        a.intern(w).unwrap();
    }

    let mut b = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
    let mut shuffled = words.clone();
    shuffled.reverse();
    for w in &shuffled {
        b.intern(w).unwrap();
    }

    a.seal_lexical();
    b.seal_lexical();

    // After seal, codes 0..N must map to bytes in lexical (raw byte) order:
    //   apple, banana, cherry, date, fig
    let expected: Vec<&[u8]> = vec![b"apple", b"banana", b"cherry", b"date", b"fig"];
    for (code, exp) in expected.iter().enumerate() {
        assert_eq!(a.get(code as u64).unwrap(), *exp);
        assert_eq!(b.get(code as u64).unwrap(), *exp);
    }
}

#[test]
fn frozen_dict_rejects_new_entries() {
    let mut dict = ByteDictionary::new();
    dict.intern(b"red").unwrap();
    dict.intern(b"green").unwrap();
    dict.freeze();
    assert!(dict.is_frozen());

    // Lookup of known values still works
    assert_eq!(dict.lookup(b"red"), Some(0));
    assert_eq!(dict.lookup(b"green"), Some(1));

    // Intern of any unseen value returns Err(Frozen)
    match dict.intern(b"blue") {
        Err(ByteDictError::Frozen) => {}
        other => panic!("expected Frozen, got {other:?}"),
    }
    assert_eq!(dict.len(), 2, "frozen dict must not grow");
}

#[test]
fn unknown_policy_error_returns_unknown_error() {
    // With Error policy, push_with_policy on an unseen value returns
    // Err(UnknownCategory). The column does NOT grow.
    let mut dict = ByteDictionary::new();
    dict.intern(b"x").unwrap();
    dict.freeze();
    let mut col = CategoricalColumn::with_dictionary(dict);
    let res = col.push_with_policy(b"y", &UnknownCategoryPolicy::Error);
    assert!(matches!(res, Err(ByteDictError::UnknownCategory)));
    assert_eq!(col.len(), 0, "Error policy must not grow the column");
}

#[test]
fn unknown_policy_map_to_null_records_null() {
    let mut dict = ByteDictionary::new();
    dict.intern(b"x").unwrap();
    dict.freeze();
    let mut col = CategoricalColumn::with_dictionary(dict);
    col.push_with_policy(b"y", &UnknownCategoryPolicy::MapToNull).unwrap();
    assert_eq!(col.len(), 1);
    assert!(col.is_null(0));
}

#[test]
fn unknown_policy_map_to_other_uses_other_code() {
    let mut dict = ByteDictionary::new();
    let known = dict.intern(b"known").unwrap();
    let other = dict.intern(b"<other>").unwrap();
    dict.freeze();
    let mut col = CategoricalColumn::with_dictionary(dict);

    // Known passes through to its real code (not null)
    col.push_with_policy(b"known", &UnknownCategoryPolicy::MapToOther { other_code: other }).unwrap();
    // Unknown gets mapped to the <other> sentinel (also not null)
    col.push_with_policy(b"unknown", &UnknownCategoryPolicy::MapToOther { other_code: other }).unwrap();

    assert_eq!(col.len(), 2);
    assert!(!col.is_null(0));
    assert!(!col.is_null(1));
    assert_eq!(col.get(0).unwrap(), b"known");
    assert_eq!(col.get(1).unwrap(), b"<other>");
    // Verify the codes themselves
    assert_eq!(col.codes().get(0), known);
    assert_eq!(col.codes().get(1), other);
}

#[test]
fn unknown_policy_extend_bypasses_frozen() {
    let mut dict = ByteDictionary::new();
    dict.intern(b"a").unwrap();
    dict.freeze();
    let mut col = CategoricalColumn::with_dictionary(dict);
    col.push_with_policy(b"b", &UnknownCategoryPolicy::ExtendDictionary).unwrap();
    assert_eq!(col.len(), 1);
    assert_eq!(col.dictionary().len(), 2);
    assert_eq!(col.get(0).unwrap(), b"b");
}

#[test]
fn profile_reports_cardinality_and_width() {
    let mut col = CategoricalColumn::new();
    for i in 0..1024u32 {
        let s = format!("v{i:04}");
        col.push(s.as_bytes()).unwrap();
    }
    // Push a few duplicates and a null
    col.push(b"v0000").unwrap();
    col.push(b"v0001").unwrap();
    col.push_null();

    let prof = col.profile();
    assert_eq!(prof.nrows, 1027);
    assert_eq!(prof.cardinality, 1024);
    assert_eq!(prof.missing, 1);
    // 1024 distinct codes still fit in u16 (256..65536)
    assert_eq!(prof.code_width_bytes, 2);
}
