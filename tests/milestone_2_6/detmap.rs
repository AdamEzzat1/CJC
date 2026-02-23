// Milestone 2.6 — DetMap Removal Order Preservation Tests
//
// Tests for the Phase 1 bug fix: DetMap remove must preserve insertion-order
// iteration for remaining entries. After removing an element from the middle,
// the order of all other elements must remain stable.

use cjc_runtime::{DetMap, Value};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Helper: create a string Value
// ---------------------------------------------------------------------------

fn str_val(s: &str) -> Value {
    Value::String(Rc::new(s.to_string()))
}

fn int_val(i: i64) -> Value {
    Value::Int(i)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn detmap_remove_middle_preserves_order() {
    let mut m = DetMap::new();
    m.insert(str_val("a"), int_val(1));
    m.insert(str_val("b"), int_val(2));
    m.insert(str_val("c"), int_val(3));

    // Remove the middle element
    let removed = m.remove(&str_val("b"));
    assert!(removed.is_some());
    assert_eq!(m.len(), 2);

    // Remaining keys must be in original insertion order: a, c
    let keys = m.keys();
    assert_eq!(keys.len(), 2);

    match &keys[0] {
        Value::String(s) => assert_eq!(s.as_str(), "a", "first key should be 'a'"),
        other => panic!("expected String key, got {:?}", other),
    }
    match &keys[1] {
        Value::String(s) => assert_eq!(s.as_str(), "c", "second key should be 'c'"),
        other => panic!("expected String key, got {:?}", other),
    }

    // Values should match
    let vals = m.values_vec();
    assert!(matches!(vals[0], Value::Int(1)));
    assert!(matches!(vals[1], Value::Int(3)));
}

#[test]
fn detmap_remove_then_reinsert_preserves_order() {
    let mut m = DetMap::new();
    m.insert(int_val(10), str_val("ten"));
    m.insert(int_val(20), str_val("twenty"));
    m.insert(int_val(30), str_val("thirty"));

    // Remove key 20
    m.remove(&int_val(20));
    assert_eq!(m.len(), 2);

    // Re-insert key 20 -- it should go at the end
    m.insert(int_val(20), str_val("twenty-again"));
    assert_eq!(m.len(), 3);

    // Order should now be: 10, 30, 20
    let keys = m.keys();
    assert!(matches!(keys[0], Value::Int(10)));
    assert!(matches!(keys[1], Value::Int(30)));
    assert!(matches!(keys[2], Value::Int(20)));

    // Verify value was updated
    match m.get(&int_val(20)) {
        Some(Value::String(s)) => assert_eq!(s.as_str(), "twenty-again"),
        other => panic!("expected String 'twenty-again', got {:?}", other),
    }
}

#[test]
fn detmap_remove_first_preserves_remaining_order() {
    let mut m = DetMap::new();
    m.insert(str_val("x"), int_val(1));
    m.insert(str_val("y"), int_val(2));
    m.insert(str_val("z"), int_val(3));

    // Remove the first element
    m.remove(&str_val("x"));
    assert_eq!(m.len(), 2);

    let keys = m.keys();
    match &keys[0] {
        Value::String(s) => assert_eq!(s.as_str(), "y"),
        other => panic!("expected 'y', got {:?}", other),
    }
    match &keys[1] {
        Value::String(s) => assert_eq!(s.as_str(), "z"),
        other => panic!("expected 'z', got {:?}", other),
    }
}

#[test]
fn detmap_remove_last_preserves_remaining_order() {
    let mut m = DetMap::new();
    m.insert(str_val("x"), int_val(1));
    m.insert(str_val("y"), int_val(2));
    m.insert(str_val("z"), int_val(3));

    // Remove the last element
    m.remove(&str_val("z"));
    assert_eq!(m.len(), 2);

    let keys = m.keys();
    match &keys[0] {
        Value::String(s) => assert_eq!(s.as_str(), "x"),
        other => panic!("expected 'x', got {:?}", other),
    }
    match &keys[1] {
        Value::String(s) => assert_eq!(s.as_str(), "y"),
        other => panic!("expected 'y', got {:?}", other),
    }
}

#[test]
fn detmap_bulk_insert_remove_stress_order() {
    // Insert 100 items, remove every other one, verify remaining order.
    let mut m = DetMap::new();
    for i in 0..100i64 {
        m.insert(int_val(i), int_val(i * 10));
    }
    assert_eq!(m.len(), 100);

    // Remove all even-indexed keys (0, 2, 4, ..., 98)
    for i in (0..100i64).step_by(2) {
        let removed = m.remove(&int_val(i));
        assert!(removed.is_some(), "failed to remove key {}", i);
    }
    assert_eq!(m.len(), 50);

    // Remaining keys should be the odd numbers in insertion order: 1, 3, 5, ..., 99
    let keys = m.keys();
    assert_eq!(keys.len(), 50);

    for (idx, key) in keys.iter().enumerate() {
        let expected = (idx as i64) * 2 + 1;
        match key {
            Value::Int(v) => assert_eq!(
                *v, expected,
                "at position {}, expected key {}, got {}",
                idx, expected, v
            ),
            other => panic!("expected Int key, got {:?}", other),
        }
    }

    // Verify values match
    for i in (1..100i64).step_by(2) {
        match m.get(&int_val(i)) {
            Some(Value::Int(v)) => assert_eq!(
                *v,
                i * 10,
                "value for key {} should be {}, got {}",
                i,
                i * 10,
                v
            ),
            other => panic!("expected Int value for key {}, got {:?}", i, other),
        }
    }
}
