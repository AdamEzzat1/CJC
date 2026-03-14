//! Content-addressable serialization for CJC values.
//!
//! `cjc-snap` provides deterministic binary encoding of `Value` types with
//! SHA-256 content hashing. Zero external dependencies -- the SHA-256
//! implementation is hand-rolled following FIPS 180-4.
//!
//! # Overview
//!
//! ```text
//!   Value ──snap_encode──> bytes ──sha256──> hash
//!          ◄──snap_decode──
//! ```
//!
//! The high-level API is `snap()` / `restore()`:
//!
//! ```ignore
//! let blob = cjc_snap::snap(&value);
//! let restored = cjc_snap::restore(&blob).unwrap();
//! ```

pub mod hash;
pub mod encode;
pub mod decode;
pub mod json;
pub mod persist;

pub use hash::sha256;
pub use encode::snap_encode;
pub use decode::snap_decode;
pub use json::snap_to_json;

use cjc_runtime::Value;
use std::fmt;

// ---------------------------------------------------------------------------
// SnapError
// ---------------------------------------------------------------------------

/// Errors that can occur during decoding or integrity verification.
#[derive(Debug)]
pub enum SnapError {
    /// The tag byte does not correspond to any known `Value` variant.
    InvalidTag(u8),
    /// The byte stream ended before the value was fully decoded.
    UnexpectedEof,
    /// A string field contained invalid UTF-8.
    Utf8Error,
    /// The SHA-256 hash of the data does not match the stored content hash.
    HashMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
}

impl fmt::Display for SnapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnapError::InvalidTag(tag) => write!(f, "invalid tag byte: 0x{:02x}", tag),
            SnapError::UnexpectedEof => write!(f, "unexpected end of data"),
            SnapError::Utf8Error => write!(f, "invalid UTF-8 in string field"),
            SnapError::HashMismatch { expected, actual } => {
                write!(
                    f,
                    "hash mismatch: expected {}, got {}",
                    hash::hex_string(expected),
                    hash::hex_string(actual)
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SnapBlob
// ---------------------------------------------------------------------------

/// A content-addressable blob: the canonical binary encoding of a `Value`
/// together with its SHA-256 digest.
#[derive(Debug, Clone)]
pub struct SnapBlob {
    /// SHA-256 hash of `data`.
    pub content_hash: [u8; 32],
    /// Canonical binary encoding of the value.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

/// Encode a `Value` into a content-addressable `SnapBlob`.
///
/// The blob contains the canonical binary encoding and its SHA-256 hash.
/// Two values that are logically equal will always produce blobs with the
/// same `content_hash`, regardless of HashMap iteration order or Rc identity.
pub fn snap(value: &Value) -> SnapBlob {
    let data = snap_encode(value);
    let content_hash = sha256(&data);
    SnapBlob { content_hash, data }
}

/// Restore a `Value` from a `SnapBlob`, verifying data integrity.
///
/// Returns `SnapError::HashMismatch` if the SHA-256 of `blob.data` does not
/// match `blob.content_hash`.
pub fn restore(blob: &SnapBlob) -> Result<Value, SnapError> {
    // Verify integrity
    let actual_hash = sha256(&blob.data);
    if actual_hash != blob.content_hash {
        return Err(SnapError::HashMismatch {
            expected: blob.content_hash,
            actual: actual_hash,
        });
    }
    snap_decode(&blob.data)
}

/// Check whether a value can be snap-encoded without panicking.
///
/// Returns `false` for runtime-only variants (Fn, Closure, ClassRef,
/// GradGraph, OptimizerState, etc.) that cannot be meaningfully serialized.
pub fn is_snappable(value: &Value) -> bool {
    match value {
        Value::Void | Value::Int(_) | Value::Float(_) | Value::Bool(_)
        | Value::String(_) | Value::U8(_) | Value::Bytes(_)
        | Value::ByteSlice(_) | Value::StrView(_) | Value::Bf16(_)
        | Value::F16(_) | Value::Complex(_) | Value::Tensor(_) => true,
        Value::Array(arr) => arr.iter().all(is_snappable),
        Value::Tuple(elems) => elems.iter().all(is_snappable),
        Value::Struct { fields, .. } => fields.values().all(is_snappable),
        Value::Enum { fields, .. } => fields.iter().all(is_snappable),
        Value::Map(m) => {
            let map = m.borrow();
            let result = map.iter().all(|(k, v)| is_snappable(k) && is_snappable(v));
            result
        }
        // Runtime-only: not serializable
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_runtime::Tensor;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    #[test]
    fn test_snap_restore_int() {
        let original = Value::Int(42);
        let blob = snap(&original);
        let restored = restore(&blob).unwrap();
        match restored {
            Value::Int(v) => assert_eq!(v, 42),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn test_snap_restore_string() {
        let original = Value::String(Rc::new("hello".to_string()));
        let blob = snap(&original);
        let restored = restore(&blob).unwrap();
        match restored {
            Value::String(s) => assert_eq!(s.as_str(), "hello"),
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn test_snap_restore_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let original = Value::Tensor(t);
        let blob = snap(&original);
        let restored = restore(&blob).unwrap();
        match restored {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_snap_restore_nested() {
        let mut fields = BTreeMap::new();
        fields.insert("x".to_string(), Value::Int(1));
        fields.insert("data".to_string(), Value::Array(Rc::new(vec![
            Value::Float(1.0),
            Value::Float(2.0),
        ])));
        let original = Value::Struct {
            name: "Complex".to_string(),
            fields,
        };
        let blob = snap(&original);
        let restored = restore(&blob).unwrap();
        match restored {
            Value::Struct { name, fields } => {
                assert_eq!(name, "Complex");
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_content_addressable_same_value() {
        let v1 = Value::Int(42);
        let v2 = Value::Int(42);
        let blob1 = snap(&v1);
        let blob2 = snap(&v2);
        assert_eq!(blob1.content_hash, blob2.content_hash);
        assert_eq!(blob1.data, blob2.data);
    }

    #[test]
    fn test_content_addressable_different_values() {
        let v1 = Value::Int(42);
        let v2 = Value::Int(43);
        let blob1 = snap(&v1);
        let blob2 = snap(&v2);
        assert_ne!(blob1.content_hash, blob2.content_hash);
    }

    #[test]
    fn test_struct_determinism() {
        // Two structs with same fields inserted in different order
        let mut f1 = BTreeMap::new();
        f1.insert("b".to_string(), Value::Int(2));
        f1.insert("a".to_string(), Value::Int(1));
        f1.insert("c".to_string(), Value::Int(3));

        let mut f2 = BTreeMap::new();
        f2.insert("c".to_string(), Value::Int(3));
        f2.insert("a".to_string(), Value::Int(1));
        f2.insert("b".to_string(), Value::Int(2));

        let blob1 = snap(&Value::Struct {
            name: "S".to_string(),
            fields: f1,
        });
        let blob2 = snap(&Value::Struct {
            name: "S".to_string(),
            fields: f2,
        });

        assert_eq!(blob1.content_hash, blob2.content_hash);
    }

    #[test]
    fn test_hash_mismatch_detection() {
        let blob = snap(&Value::Int(42));
        let tampered = SnapBlob {
            content_hash: blob.content_hash,
            data: snap_encode(&Value::Int(999)), // different data
        };
        let result = restore(&tampered);
        assert!(matches!(result, Err(SnapError::HashMismatch { .. })));
    }

    #[test]
    fn test_snap_void() {
        let blob = snap(&Value::Void);
        assert_eq!(blob.data, vec![0x00]);
        let restored = restore(&blob).unwrap();
        assert!(matches!(restored, Value::Void));
    }

    #[test]
    fn test_hex_display() {
        let blob = snap(&Value::Int(0));
        let hex = hash::hex_string(&blob.content_hash);
        assert_eq!(hex.len(), 64, "SHA-256 hex should be 64 chars");
    }
}
