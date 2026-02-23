// Milestone 2.5 — COW String Tests
//
// CJC strings are stored as Value::String(Rc<String>), giving O(1)
// clone (reference sharing) with copy-on-write semantics when the
// Rc is made unique. These tests verify:
// - Rc sharing on clone
// - Independence after Rc::make_mut
// - Display correctness
// - value_hash determinism on string values

use cjc_runtime::{Value, murmurhash3, value_hash};
use std::rc::Rc;

#[test]
fn cow_string_clone_shares_rc() {
    let s1 = Value::String(Rc::new("hello world".into()));
    let s2 = s1.clone();

    // Cloning a Value::String should share the same Rc
    if let (Value::String(a), Value::String(b)) = (&s1, &s2) {
        assert!(Rc::ptr_eq(a, b), "clone must share Rc pointer");
        assert_eq!(Rc::strong_count(a), 2);
    } else {
        panic!("expected Value::String variants");
    }
}

#[test]
fn cow_string_make_mut_decouples() {
    let original = Rc::new(String::from("shared"));
    let mut fork = original.clone();
    assert_eq!(Rc::strong_count(&original), 2);

    // Rc::make_mut clones the inner data when shared
    Rc::make_mut(&mut fork).push_str(" modified");

    // After make_mut, original is unmodified
    assert_eq!(*original, "shared");
    assert_eq!(*fork, "shared modified");
    assert_eq!(Rc::strong_count(&original), 1);
    assert_eq!(Rc::strong_count(&fork), 1);
}

#[test]
fn cow_string_display_format() {
    let v = Value::String(Rc::new("CJC runtime".into()));
    assert_eq!(format!("{}", v), "CJC runtime");
}

#[test]
fn cow_string_hash_deterministic() {
    let v1 = Value::String(Rc::new("test_key".into()));
    let v2 = Value::String(Rc::new("test_key".into()));
    let v3 = Value::String(Rc::new("different".into()));

    let h1 = value_hash(&v1);
    let h2 = value_hash(&v2);
    let h3 = value_hash(&v3);

    assert_eq!(h1, h2, "same string content must produce same hash");
    assert_ne!(h1, h3, "different strings should (almost certainly) differ in hash");
}

#[test]
fn cow_string_empty_and_unicode() {
    // Empty string
    let empty = Value::String(Rc::new(String::new()));
    assert_eq!(format!("{}", empty), "");

    // Unicode content
    let unicode = Value::String(Rc::new("tensor".into()));
    assert_eq!(format!("{}", unicode), "tensor");

    // Hash of empty string is deterministic
    let h1 = value_hash(&Value::String(Rc::new(String::new())));
    let h2 = value_hash(&Value::String(Rc::new(String::new())));
    assert_eq!(h1, h2);
}

#[test]
fn cow_murmurhash3_basic() {
    // murmurhash3 with same input always gives same output
    let data = b"hello";
    let h1 = murmurhash3(data);
    let h2 = murmurhash3(data);
    assert_eq!(h1, h2, "murmurhash3 must be deterministic");

    // Different inputs produce different hashes
    let h3 = murmurhash3(b"world");
    assert_ne!(h1, h3);
}
