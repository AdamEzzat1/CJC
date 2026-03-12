//! Snap (serialization) hardening tests.

use std::rc::Rc;
use cjc_runtime::Value;

/// Encode and decode an integer value.
#[test]
fn snap_roundtrip_int() {
    let val = Value::Int(42);
    let encoded = cjc_snap::snap_encode(&val);
    let decoded = cjc_snap::snap_decode(&encoded).expect("decode should succeed");
    match decoded {
        Value::Int(v) => assert_eq!(v, 42),
        other => panic!("Expected Int(42), got {other:?}"),
    }
}

/// Encode and decode a float value.
#[test]
fn snap_roundtrip_float() {
    let val = Value::Float(3.14);
    let encoded = cjc_snap::snap_encode(&val);
    let decoded = cjc_snap::snap_decode(&encoded).expect("decode should succeed");
    match decoded {
        Value::Float(v) => assert!((v - 3.14).abs() < 1e-10),
        other => panic!("Expected Float(3.14), got {other:?}"),
    }
}

/// Encode and decode a boolean value.
#[test]
fn snap_roundtrip_bool() {
    let val = Value::Bool(true);
    let encoded = cjc_snap::snap_encode(&val);
    let decoded = cjc_snap::snap_decode(&encoded).expect("decode should succeed");
    match decoded {
        Value::Bool(v) => assert!(v),
        other => panic!("Expected Bool(true), got {other:?}"),
    }
}

/// Encode and decode a string value.
#[test]
fn snap_roundtrip_string() {
    let val = Value::String(Rc::new("hello world".to_string()));
    let encoded = cjc_snap::snap_encode(&val);
    let decoded = cjc_snap::snap_decode(&encoded).expect("decode should succeed");
    match decoded {
        Value::String(s) => assert_eq!(s.as_ref(), "hello world"),
        other => panic!("Expected String, got {other:?}"),
    }
}

/// Encode and decode Void.
#[test]
fn snap_roundtrip_void() {
    let val = Value::Void;
    let encoded = cjc_snap::snap_encode(&val);
    let decoded = cjc_snap::snap_decode(&encoded).expect("decode should succeed");
    match decoded {
        Value::Void => {} // OK
        other => panic!("Expected Void, got {other:?}"),
    }
}

/// SHA-256 hash determinism.
#[test]
fn snap_hash_deterministic() {
    let val = Value::Int(42);
    let h1 = cjc_snap::sha256(&cjc_snap::snap_encode(&val));
    let h2 = cjc_snap::sha256(&cjc_snap::snap_encode(&val));
    assert_eq!(h1, h2, "SHA-256 hash should be deterministic");
}

/// Empty byte array doesn't crash.
#[test]
fn snap_decode_empty_bytes() {
    let result = cjc_snap::snap_decode(&[]);
    // Should either return an error or a default value — not panic
    let _ = result;
}

/// Arbitrary bytes don't crash decoder.
#[test]
fn snap_decode_garbage() {
    let garbage = vec![0xFF, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];
    let result = cjc_snap::snap_decode(&garbage);
    let _ = result; // Must not panic
}
