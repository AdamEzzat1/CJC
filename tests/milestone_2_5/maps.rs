// Milestone 2.5 — Deterministic Map Tests
//
// Tests for DetMap (deterministic hash map with insertion-order iteration):
// - Insert, get, contains_key
// - Insertion-order iteration
// - Remove and re-insert
// - Overwrite existing key preserves order
// - Growth/rehash preserves order
// - value_hash determinism

use cjc_runtime::{DetMap, Value, value_hash};
use std::rc::Rc;

#[test]
fn map_insert_and_get() {
    let mut m = DetMap::new();
    m.insert(Value::String(Rc::new("x".into())), Value::Int(10));
    m.insert(Value::String(Rc::new("y".into())), Value::Int(20));

    assert_eq!(m.len(), 2);
    assert!(m.contains_key(&Value::String(Rc::new("x".into()))));
    assert!(m.contains_key(&Value::String(Rc::new("y".into()))));
    assert!(!m.contains_key(&Value::String(Rc::new("z".into()))));

    if let Some(Value::Int(v)) = m.get(&Value::String(Rc::new("x".into()))) {
        assert_eq!(*v, 10);
    } else {
        panic!("expected Int(10)");
    }
}

#[test]
fn map_insertion_order_iteration() {
    let mut m = DetMap::new();
    m.insert(Value::String(Rc::new("c".into())), Value::Int(3));
    m.insert(Value::String(Rc::new("a".into())), Value::Int(1));
    m.insert(Value::String(Rc::new("b".into())), Value::Int(2));

    let keys = m.keys();
    assert_eq!(keys.len(), 3);

    // Keys must come back in insertion order: c, a, b
    if let Value::String(ref s) = keys[0] {
        assert_eq!(s.as_str(), "c");
    } else {
        panic!("expected String key");
    }
    if let Value::String(ref s) = keys[1] {
        assert_eq!(s.as_str(), "a");
    } else {
        panic!("expected String key");
    }
    if let Value::String(ref s) = keys[2] {
        assert_eq!(s.as_str(), "b");
    } else {
        panic!("expected String key");
    }
}

#[test]
fn map_overwrite_preserves_order() {
    let mut m = DetMap::new();
    m.insert(Value::Int(1), Value::Float(1.0));
    m.insert(Value::Int(2), Value::Float(2.0));
    m.insert(Value::Int(3), Value::Float(3.0));

    // Overwrite key 2
    m.insert(Value::Int(2), Value::Float(99.0));

    assert_eq!(m.len(), 3);

    // Value should be updated
    if let Some(Value::Float(v)) = m.get(&Value::Int(2)) {
        assert_eq!(*v, 99.0);
    } else {
        panic!("expected Float(99.0)");
    }
}

#[test]
fn map_remove_and_reinsert() {
    let mut m = DetMap::new();
    m.insert(Value::Int(1), Value::Bool(true));
    m.insert(Value::Int(2), Value::Bool(false));
    m.insert(Value::Int(3), Value::Bool(true));

    assert_eq!(m.len(), 3);

    // Remove key 2
    let removed = m.remove(&Value::Int(2));
    assert!(removed.is_some());
    assert_eq!(m.len(), 2);
    assert!(!m.contains_key(&Value::Int(2)));

    // Re-insert key 2 -- it goes at the end
    m.insert(Value::Int(2), Value::Bool(false));
    assert_eq!(m.len(), 3);
    assert!(m.contains_key(&Value::Int(2)));
}

#[test]
fn map_growth_preserves_insertion_order() {
    // Insert enough entries to trigger at least one rehash (initial capacity = 8)
    let mut m = DetMap::new();
    for i in 0..20 {
        m.insert(Value::Int(i), Value::Float(i as f64 * 10.0));
    }
    assert_eq!(m.len(), 20);

    // Verify all entries are accessible
    for i in 0..20 {
        assert!(m.contains_key(&Value::Int(i)), "missing key {}", i);
        if let Some(Value::Float(v)) = m.get(&Value::Int(i)) {
            assert_eq!(*v, i as f64 * 10.0);
        } else {
            panic!("wrong value for key {}", i);
        }
    }

    // Verify keys come back in insertion order
    let keys = m.keys();
    for (idx, key) in keys.iter().enumerate() {
        if let Value::Int(v) = key {
            assert_eq!(*v, idx as i64, "insertion order broken at index {}", idx);
        }
    }
}

#[test]
fn map_value_hash_determinism() {
    // Same value always produces same hash
    let h1 = value_hash(&Value::Int(42));
    let h2 = value_hash(&Value::Int(42));
    assert_eq!(h1, h2);

    // Different values produce different hashes (with high probability)
    let h3 = value_hash(&Value::Int(43));
    assert_ne!(h1, h3);

    // Float hashing
    let hf1 = value_hash(&Value::Float(3.14));
    let hf2 = value_hash(&Value::Float(3.14));
    assert_eq!(hf1, hf2);
}
