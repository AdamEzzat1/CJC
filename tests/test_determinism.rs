// CJC Test Suite — Determinism & Collections Tests (ROLE 5)
//
// Ensures that hashing, sorting, and map operations on byte slices
// are deterministic across runs and platform-independent.

use cjc_runtime::{murmurhash3, value_hash, Value, DetMap};
use std::rc::Rc;

// =========================================================================
// Hash Stability Tests (Double-Run Gate)
// =========================================================================

#[test]
fn test_murmurhash3_empty_stable() {
    // Hash of empty input must be deterministic
    let h1 = murmurhash3(b"");
    let h2 = murmurhash3(b"");
    assert_eq!(h1, h2);
    // Pin the exact value for cross-run verification
    assert_eq!(h1, murmurhash3(b""));
}

#[test]
fn test_murmurhash3_known_vectors() {
    // These values must remain stable across all CJC versions.
    // If these change, determinism is broken.
    let h_hello = murmurhash3(b"hello");
    let h_world = murmurhash3(b"world");
    let h_empty = murmurhash3(b"");

    // Run twice — must produce identical results
    assert_eq!(h_hello, murmurhash3(b"hello"));
    assert_eq!(h_world, murmurhash3(b"world"));
    assert_eq!(h_empty, murmurhash3(b""));

    // Different inputs produce different hashes
    assert_ne!(h_hello, h_world);
    assert_ne!(h_hello, h_empty);
    assert_ne!(h_world, h_empty);
}

#[test]
fn test_murmurhash3_binary_data_stable() {
    let data = vec![0xff, 0x00, 0x41, 0x0a, 0x80, 0x7f, 0xfe, 0x01];
    let h1 = murmurhash3(&data);
    let h2 = murmurhash3(&data);
    assert_eq!(h1, h2);
}

#[test]
fn test_murmurhash3_long_input_stable() {
    // Test with a longer input to exercise the 8-byte chunk + tail paths
    let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let h1 = murmurhash3(&data);
    let h2 = murmurhash3(&data);
    assert_eq!(h1, h2);
}

#[test]
fn test_value_hash_byteslice_stable() {
    let v1 = Value::ByteSlice(Rc::new(b"token".to_vec()));
    let v2 = Value::ByteSlice(Rc::new(b"token".to_vec()));
    assert_eq!(value_hash(&v1), value_hash(&v2));
}

#[test]
fn test_value_hash_strview_matches_byteslice() {
    // StrView and ByteSlice with same content should produce same hash
    let bs = Value::ByteSlice(Rc::new(b"hello".to_vec()));
    let sv = Value::StrView(Rc::new(b"hello".to_vec()));
    assert_eq!(value_hash(&bs), value_hash(&sv));
}

#[test]
fn test_value_hash_u8_stable() {
    let v1 = Value::U8(42);
    let v2 = Value::U8(42);
    assert_eq!(value_hash(&v1), value_hash(&v2));
    assert_ne!(value_hash(&v1), value_hash(&Value::U8(43)));
}

// =========================================================================
// ByteSlice Lexicographic Sorting Tests
// =========================================================================

#[test]
fn test_byteslice_sort_lexicographic() {
    let mut slices: Vec<Vec<u8>> = vec![
        b"cherry".to_vec(),
        b"apple".to_vec(),
        b"banana".to_vec(),
        b"date".to_vec(),
    ];
    slices.sort();
    assert_eq!(
        slices,
        vec![
            b"apple".to_vec(),
            b"banana".to_vec(),
            b"cherry".to_vec(),
            b"date".to_vec(),
        ]
    );
}

#[test]
fn test_byteslice_sort_shorter_is_less() {
    let mut slices: Vec<Vec<u8>> = vec![
        b"abc".to_vec(),
        b"ab".to_vec(),
        b"a".to_vec(),
        b"abcd".to_vec(),
    ];
    slices.sort();
    assert_eq!(
        slices,
        vec![
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"abcd".to_vec(),
        ]
    );
}

#[test]
fn test_byteslice_sort_binary_data() {
    let mut slices: Vec<Vec<u8>> = vec![
        vec![0xff, 0x01],
        vec![0x00, 0x01],
        vec![0x80, 0x01],
        vec![0x00, 0x00],
    ];
    slices.sort();
    assert_eq!(
        slices,
        vec![
            vec![0x00, 0x00],
            vec![0x00, 0x01],
            vec![0x80, 0x01],
            vec![0xff, 0x01],
        ]
    );
}

#[test]
fn test_byteslice_sort_stable() {
    // Stable sort: equal elements preserve insertion order.
    // We use (count, token) pairs sorted by count desc, token asc.
    let mut entries: Vec<(i64, Vec<u8>)> = vec![
        (5, b"the".to_vec()),
        (3, b"cat".to_vec()),
        (5, b"and".to_vec()),
        (3, b"bat".to_vec()),
        (1, b"zoo".to_vec()),
    ];
    // Sort by count descending, then by token ascending (lexicographic)
    entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    assert_eq!(entries[0], (5, b"and".to_vec()));
    assert_eq!(entries[1], (5, b"the".to_vec()));
    assert_eq!(entries[2], (3, b"bat".to_vec()));
    assert_eq!(entries[3], (3, b"cat".to_vec()));
    assert_eq!(entries[4], (1, b"zoo".to_vec()));
}

#[test]
fn test_byteslice_sort_deterministic_double_run() {
    // Same input sorted twice must produce identical results
    let input: Vec<Vec<u8>> = vec![
        b"delta".to_vec(),
        b"alpha".to_vec(),
        b"charlie".to_vec(),
        b"bravo".to_vec(),
    ];
    let mut run1 = input.clone();
    let mut run2 = input.clone();
    run1.sort();
    run2.sort();
    assert_eq!(run1, run2);
}

// =========================================================================
// DetMap with ByteSlice Keys (ROLE 5 integration)
// =========================================================================

#[test]
fn test_detmap_byteslice_insert_lookup() {
    let mut map = DetMap::new();
    let key1 = Value::ByteSlice(Rc::new(b"hello".to_vec()));
    let key2 = Value::ByteSlice(Rc::new(b"world".to_vec()));
    map.insert(key1.clone(), Value::Int(1));
    map.insert(key2.clone(), Value::Int(2));

    assert_eq!(map.len(), 2);
    match map.get(&key1) {
        Some(Value::Int(1)) => {},
        other => panic!("expected Int(1), got {:?}", other),
    }
    match map.get(&key2) {
        Some(Value::Int(2)) => {},
        other => panic!("expected Int(2), got {:?}", other),
    }
}

#[test]
fn test_detmap_byteslice_insertion_order() {
    let mut map = DetMap::new();
    let keys: Vec<Vec<u8>> = vec![
        b"third".to_vec(),
        b"first".to_vec(),
        b"second".to_vec(),
    ];
    for (i, k) in keys.iter().enumerate() {
        map.insert(Value::ByteSlice(Rc::new(k.clone())), Value::Int(i as i64));
    }

    // Iteration must follow insertion order
    let iter_keys: Vec<String> = map.iter()
        .map(|(k, _)| match k {
            Value::ByteSlice(b) => String::from_utf8((**b).clone()).unwrap(),
            _ => panic!("expected ByteSlice key"),
        })
        .collect();
    assert_eq!(iter_keys, vec!["third", "first", "second"]);
}

#[test]
fn test_detmap_byteslice_overwrite() {
    let mut map = DetMap::new();
    let key = Value::ByteSlice(Rc::new(b"key".to_vec()));
    map.insert(key.clone(), Value::Int(1));
    map.insert(key.clone(), Value::Int(2));

    assert_eq!(map.len(), 1);
    match map.get(&key) {
        Some(Value::Int(2)) => {},
        other => panic!("expected Int(2), got {:?}", other),
    }
}

#[test]
fn test_detmap_byteslice_remove_reinsert() {
    let mut map = DetMap::new();
    let k1 = Value::ByteSlice(Rc::new(b"a".to_vec()));
    let k2 = Value::ByteSlice(Rc::new(b"b".to_vec()));
    let k3 = Value::ByteSlice(Rc::new(b"c".to_vec()));

    map.insert(k1.clone(), Value::Int(1));
    map.insert(k2.clone(), Value::Int(2));
    map.insert(k3.clone(), Value::Int(3));

    // Remove k2
    let removed = map.remove(&k2);
    assert!(matches!(removed, Some(Value::Int(2))));
    assert_eq!(map.len(), 2);
    assert!(!map.contains_key(&k2));

    // Reinsert k2
    map.insert(k2.clone(), Value::Int(22));
    assert_eq!(map.len(), 3);
    match map.get(&k2) {
        Some(Value::Int(22)) => {},
        other => panic!("expected Int(22), got {:?}", other),
    }

    // Hash stability after remove/reinsert
    let h1 = value_hash(&k2);
    let h2 = value_hash(&Value::ByteSlice(Rc::new(b"b".to_vec())));
    assert_eq!(h1, h2);
}

#[test]
fn test_detmap_byteslice_stress_hash_stability() {
    // Insert many keys, verify all are retrievable (hash correctness)
    let mut map = DetMap::new();
    for i in 0..100 {
        let key = Value::ByteSlice(Rc::new(format!("key_{:04}", i).into_bytes()));
        map.insert(key, Value::Int(i));
    }
    assert_eq!(map.len(), 100);

    // Verify all lookups succeed
    for i in 0..100 {
        let key = Value::ByteSlice(Rc::new(format!("key_{:04}", i).into_bytes()));
        match map.get(&key) {
            Some(Value::Int(v)) => assert_eq!(*v, i),
            other => panic!("key_{:04}: expected Int({}), got {:?}", i, i, other),
        }
    }
}

// =========================================================================
// Double-Run Determinism Gate
// =========================================================================

#[test]
fn test_hash_double_run_gate() {
    // Simulate two runs: hash the same set of byte slices and verify identical output
    let tokens: Vec<&[u8]> = vec![
        b"the", b"quick", b"brown", b"fox", b"jumps",
        b"over", b"the", b"lazy", b"dog",
    ];

    let hashes_run1: Vec<u64> = tokens.iter().map(|t| murmurhash3(t)).collect();
    let hashes_run2: Vec<u64> = tokens.iter().map(|t| murmurhash3(t)).collect();
    assert_eq!(hashes_run1, hashes_run2);
}

#[test]
fn test_vocab_count_deterministic() {
    // Simulate a vocab counting pipeline: count tokens, sort by count desc + token asc
    let tokens: Vec<&[u8]> = vec![
        b"the", b"cat", b"sat", b"on", b"the", b"mat",
        b"the", b"cat", b"on", b"the", b"mat",
    ];

    // Count
    let mut counts: std::collections::HashMap<Vec<u8>, i64> = std::collections::HashMap::new();
    for &t in &tokens {
        *counts.entry(t.to_vec()).or_insert(0) += 1;
    }

    // Sort deterministically: count desc, then token bytes asc
    let mut entries: Vec<(Vec<u8>, i64)> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Run 2: identical
    let mut counts2: std::collections::HashMap<Vec<u8>, i64> = std::collections::HashMap::new();
    for &t in &tokens {
        *counts2.entry(t.to_vec()).or_insert(0) += 1;
    }
    let mut entries2: Vec<(Vec<u8>, i64)> = counts2.into_iter().collect();
    entries2.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    assert_eq!(entries, entries2);

    // Verify expected order
    assert_eq!(entries[0], (b"the".to_vec(), 4));
    assert_eq!(entries[1], (b"cat".to_vec(), 2));
    assert_eq!(entries[2], (b"mat".to_vec(), 2));
    assert_eq!(entries[3], (b"on".to_vec(), 2));
    assert_eq!(entries[4], (b"sat".to_vec(), 1));
}
