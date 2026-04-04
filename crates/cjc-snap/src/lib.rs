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
pub use encode::snap_encode_v2;
pub use encode::{encode_typed_tensor, encode_chunked_tensor, encode_sparse_csr, encode_categorical, encode_schema, encode_dataframe};
pub use encode::{DataFrameColumnData, DEFAULT_CHUNK_SIZE};
pub use encode::{
    TAG_TYPED_TENSOR, TAG_CHUNKED_TENSOR, TAG_SPARSE_CSR, TAG_CATEGORICAL,
    TAG_SCHEMA, TAG_DATAFRAME, SNAP_MAGIC, SNAP_VERSION,
    COL_TYPE_INT, COL_TYPE_FLOAT, COL_TYPE_STR, COL_TYPE_BOOL,
    COL_TYPE_CATEGORICAL, COL_TYPE_DATETIME,
};
pub use decode::snap_decode;
pub use decode::snap_decode_v2;
pub use json::snap_to_json;
pub use persist::{snap_save, snap_save_v2, snap_load};

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

/// Encode a `Value` into a content-addressable `SnapBlob` (v1 format).
///
/// The blob contains the canonical binary encoding and its SHA-256 hash.
/// Two values that are logically equal will always produce blobs with the
/// same `content_hash`, regardless of HashMap iteration order or Rc identity.
pub fn snap(value: &Value) -> SnapBlob {
    let data = snap_encode(value);
    let content_hash = sha256(&data);
    SnapBlob { content_hash, data }
}

/// Encode a `Value` into a content-addressable `SnapBlob` (v2 format with header).
///
/// Uses the v2 format: [MAGIC][version][flags][payload...].
/// Supports all new tags (typed tensors, sparse CSR, categorical, etc.).
pub fn snap_v2(value: &Value) -> SnapBlob {
    let data = snap_encode_v2(value);
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

/// Restore a `Value` from a `SnapBlob`, auto-detecting v1 or v2 format.
///
/// Verifies SHA-256 integrity, then detects format version from magic bytes.
pub fn restore_v2(blob: &SnapBlob) -> Result<Value, SnapError> {
    let actual_hash = sha256(&blob.data);
    if actual_hash != blob.content_hash {
        return Err(SnapError::HashMismatch {
            expected: blob.content_hash,
            actual: actual_hash,
        });
    }
    snap_decode_v2(&blob.data)
}

/// Check whether a value can be snap-encoded without panicking.
///
/// Returns `false` for runtime-only variants (Fn, Closure, ClassRef,
/// GradGraph, OptimizerState, etc.) that cannot be meaningfully serialized.
pub fn is_snappable(value: &Value) -> bool {
    match value {
        Value::Void | Value::Na | Value::Int(_) | Value::Float(_) | Value::Bool(_)
        | Value::String(_) | Value::U8(_) | Value::Bytes(_)
        | Value::ByteSlice(_) | Value::StrView(_) | Value::Bf16(_)
        | Value::F16(_) | Value::Complex(_) | Value::Tensor(_)
        | Value::SparseTensor(_) => true,
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
    use cjc_runtime::{Tensor, SparseCsr};
    use std::collections::BTreeMap;
    use std::rc::Rc;

    // -- Existing v1 tests --

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
        let mut f1 = BTreeMap::new();
        f1.insert("b".to_string(), Value::Int(2));
        f1.insert("a".to_string(), Value::Int(1));
        f1.insert("c".to_string(), Value::Int(3));

        let mut f2 = BTreeMap::new();
        f2.insert("c".to_string(), Value::Int(3));
        f2.insert("a".to_string(), Value::Int(1));
        f2.insert("b".to_string(), Value::Int(2));

        let blob1 = snap(&Value::Struct { name: "S".to_string(), fields: f1 });
        let blob2 = snap(&Value::Struct { name: "S".to_string(), fields: f2 });
        assert_eq!(blob1.content_hash, blob2.content_hash);
    }

    #[test]
    fn test_hash_mismatch_detection() {
        let blob = snap(&Value::Int(42));
        let tampered = SnapBlob {
            content_hash: blob.content_hash,
            data: snap_encode(&Value::Int(999)),
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

    // -- v2 high-level API tests --

    #[test]
    fn test_snap_v2_roundtrip_int() {
        let blob = snap_v2(&Value::Int(123));
        // v2 data starts with magic header
        assert_eq!(&blob.data[0..4], SNAP_MAGIC);
        assert_eq!(blob.data[4], SNAP_VERSION);
        let restored = restore_v2(&blob).unwrap();
        assert!(matches!(restored, Value::Int(123)));
    }

    #[test]
    fn test_snap_v2_roundtrip_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let blob = snap_v2(&Value::Tensor(t));
        let restored = restore_v2(&blob).unwrap();
        match restored {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_snap_v2_restore_falls_back_to_v1() {
        // v1 blob should be decodable by restore_v2
        let blob = snap(&Value::Int(42));
        let restored = restore_v2(&blob).unwrap();
        assert!(matches!(restored, Value::Int(42)));
    }

    // -- SparseTensor roundtrip tests --

    #[test]
    fn test_snap_sparse_tensor_roundtrip() {
        let sparse = SparseCsr {
            nrows: 3,
            ncols: 3,
            row_offsets: vec![0, 1, 2, 3],
            col_indices: vec![0, 1, 2],
            values: vec![1.0, 2.0, 3.0],
        };
        let blob = snap(&Value::SparseTensor(sparse));
        let restored = restore(&blob).unwrap();
        match restored {
            Value::SparseTensor(s) => {
                assert_eq!(s.nrows, 3);
                assert_eq!(s.ncols, 3);
                assert_eq!(s.row_offsets, vec![0, 1, 2, 3]);
                assert_eq!(s.col_indices, vec![0, 1, 2]);
                assert_eq!(s.values, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected SparseTensor"),
        }
    }

    #[test]
    fn test_snap_sparse_empty() {
        let sparse = SparseCsr {
            nrows: 2,
            ncols: 2,
            row_offsets: vec![0, 0, 0],
            col_indices: vec![],
            values: vec![],
        };
        let blob = snap(&Value::SparseTensor(sparse));
        let restored = restore(&blob).unwrap();
        match restored {
            Value::SparseTensor(s) => {
                assert_eq!(s.nrows, 2);
                assert_eq!(s.ncols, 2);
                assert_eq!(s.values.len(), 0);
            }
            _ => panic!("expected SparseTensor"),
        }
    }

    #[test]
    fn test_is_snappable_sparse() {
        let sparse = SparseCsr {
            nrows: 1,
            ncols: 1,
            row_offsets: vec![0, 1],
            col_indices: vec![0],
            values: vec![1.0],
        };
        assert!(is_snappable(&Value::SparseTensor(sparse)));
    }

    // -- Chunked tensor tests --

    #[test]
    fn test_chunked_tensor_roundtrip_small() {
        // Small tensor that fits in one chunk
        let raw_bytes: Vec<u8> = (0..24u8).collect(); // 3 f64s worth
        let mut buf = Vec::new();
        encode_chunked_tensor(0, &[3], &raw_bytes, 1024, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3]);
                assert_eq!(t.to_vec().len(), 3);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_chunked_tensor_roundtrip_multi_chunk() {
        // Force multiple chunks with small chunk size
        let n = 100;
        let mut raw_bytes = Vec::with_capacity(n * 8);
        for i in 0..n {
            raw_bytes.extend_from_slice(&(i as f64).to_bits().to_le_bytes());
        }
        let mut buf = Vec::new();
        // 64 bytes per chunk = 8 f64s per chunk, so 13 chunks for 100 elements
        encode_chunked_tensor(0, &[n], &raw_bytes, 64, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[n]);
                for i in 0..n {
                    assert_eq!(t.to_vec()[i], i as f64);
                }
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_chunked_tensor_empty() {
        let mut buf = Vec::new();
        encode_chunked_tensor(0, &[0], &[], 1024, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0]);
                assert_eq!(t.to_vec().len(), 0);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_chunked_tensor_hash_integrity() {
        // Tamper with a chunk and verify hash mismatch is detected
        let raw_bytes: Vec<u8> = vec![0u8; 16]; // 2 f64s
        let mut buf = Vec::new();
        encode_chunked_tensor(0, &[2], &raw_bytes, 8, &mut buf);

        // Find and tamper with a data byte (after the chunk hash)
        // TAG(1) + dtype(1) + ndim(8) + shape(8) + chunk_size(8) + n_chunks(8) = 34
        // chunk 0: len(8) + hash(32) + data(8) = 48
        // Tamper with the data of chunk 0
        let data_start = 34 + 8 + 32; // skip tag+meta + first chunk header
        if buf.len() > data_start {
            buf[data_start] = 0xFF; // tamper
        }
        let result = snap_decode(&buf);
        assert!(matches!(result, Err(SnapError::HashMismatch { .. })));
    }

    #[test]
    fn test_chunked_tensor_deterministic() {
        let raw_bytes: Vec<u8> = (0..80u8).collect();
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        encode_chunked_tensor(0, &[10], &raw_bytes, 32, &mut buf1);
        encode_chunked_tensor(0, &[10], &raw_bytes, 32, &mut buf2);
        assert_eq!(buf1, buf2, "chunked encoding must be deterministic");
    }

    // -- DataFrame encoding tests --

    #[test]
    fn test_dataframe_roundtrip_basic() {
        let int_data = vec![1i64, 2, 3];
        let float_data = vec![1.5f64, 2.5, 3.5];
        let str_data = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let mut buf = Vec::new();
        encode_dataframe(
            &["id", "value", "name"],
            &[COL_TYPE_INT, COL_TYPE_FLOAT, COL_TYPE_STR],
            &[
                DataFrameColumnData::Int(&int_data),
                DataFrameColumnData::Float(&float_data),
                DataFrameColumnData::Str(&str_data),
            ],
            3,
            &mut buf,
        );

        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Struct { name, fields } => {
                assert_eq!(name, "DataFrame");
                assert!(fields.contains_key("__nrows"));
                assert!(fields.contains_key("__columns"));
                match fields.get("__nrows") {
                    Some(Value::Int(n)) => assert_eq!(*n, 3),
                    _ => panic!("expected __nrows = 3"),
                }
                // Check id column
                match fields.get("id") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 3);
                        assert!(matches!(arr[0], Value::Int(1)));
                        assert!(matches!(arr[1], Value::Int(2)));
                    }
                    _ => panic!("expected id array"),
                }
                // Check value column
                match fields.get("value") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 3);
                        match &arr[0] {
                            Value::Float(f) => assert_eq!(*f, 1.5),
                            _ => panic!("expected Float"),
                        }
                    }
                    _ => panic!("expected value array"),
                }
                // Check name column
                match fields.get("name") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 3);
                        match &arr[0] {
                            Value::String(s) => assert_eq!(s.as_str(), "a"),
                            _ => panic!("expected String"),
                        }
                    }
                    _ => panic!("expected name array"),
                }
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_dataframe_bool_column() {
        let bool_data = vec![true, false, true];
        let mut buf = Vec::new();
        encode_dataframe(
            &["flag"],
            &[COL_TYPE_BOOL],
            &[DataFrameColumnData::Bool(&bool_data)],
            3,
            &mut buf,
        );
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Struct { fields, .. } => {
                match fields.get("flag") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 3);
                        assert!(matches!(arr[0], Value::Bool(true)));
                        assert!(matches!(arr[1], Value::Bool(false)));
                        assert!(matches!(arr[2], Value::Bool(true)));
                    }
                    _ => panic!("expected flag array"),
                }
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_dataframe_categorical_column() {
        let levels = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let codes = vec![0u32, 1, 2, 0, 1];
        let mut buf = Vec::new();
        encode_dataframe(
            &["animal"],
            &[COL_TYPE_CATEGORICAL],
            &[DataFrameColumnData::Categorical { levels: &levels, codes: &codes }],
            5,
            &mut buf,
        );
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Struct { fields, .. } => {
                match fields.get("animal") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 5);
                        match &arr[0] {
                            Value::String(s) => assert_eq!(s.as_str(), "cat"),
                            _ => panic!("expected String"),
                        }
                        match &arr[2] {
                            Value::String(s) => assert_eq!(s.as_str(), "fish"),
                            _ => panic!("expected String"),
                        }
                    }
                    _ => panic!("expected animal array"),
                }
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_dataframe_datetime_column() {
        let dt_data = vec![1000i64, 2000, 3000];
        let mut buf = Vec::new();
        encode_dataframe(
            &["timestamp"],
            &[COL_TYPE_DATETIME],
            &[DataFrameColumnData::DateTime(&dt_data)],
            3,
            &mut buf,
        );
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Struct { fields, .. } => {
                match fields.get("timestamp") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 3);
                        assert!(matches!(arr[0], Value::Int(1000)));
                    }
                    _ => panic!("expected timestamp array"),
                }
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_dataframe_empty() {
        let mut buf = Vec::new();
        encode_dataframe(&[], &[], &[], 0, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Struct { name, fields } => {
                assert_eq!(name, "DataFrame");
                match fields.get("__nrows") {
                    Some(Value::Int(0)) => {}
                    _ => panic!("expected __nrows = 0"),
                }
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_dataframe_deterministic() {
        let int_data = vec![1i64, 2, 3];
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        encode_dataframe(
            &["x"],
            &[COL_TYPE_INT],
            &[DataFrameColumnData::Int(&int_data)],
            3,
            &mut buf1,
        );
        encode_dataframe(
            &["x"],
            &[COL_TYPE_INT],
            &[DataFrameColumnData::Int(&int_data)],
            3,
            &mut buf2,
        );
        assert_eq!(buf1, buf2, "dataframe encoding must be deterministic");
    }

    // -- Typed tensor standalone tests --

    #[test]
    fn test_typed_tensor_f64_roundtrip() {
        let raw: Vec<u8> = vec![1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let mut buf = Vec::new();
        encode_typed_tensor(0, &[3], &raw, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Tensor(t) => {
                assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_typed_tensor_i32_roundtrip() {
        let raw: Vec<u8> = vec![10i32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut buf = Vec::new();
        encode_typed_tensor(3, &[3], &raw, &mut buf); // dtype 3 = I32
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Tensor(t) => {
                assert_eq!(t.to_vec(), vec![10.0, 20.0, 30.0]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    // -- Schema roundtrip test --

    #[test]
    fn test_schema_roundtrip() {
        let fields = vec![
            ("id".to_string(), 0x01u8),
            ("name".to_string(), 0x04u8),
            ("value".to_string(), 0x02u8),
        ];
        let mut buf = Vec::new();
        encode_schema(&fields, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Struct { name, fields } => {
                assert_eq!(name, "Schema");
                assert_eq!(fields.len(), 3);
                assert!(matches!(fields.get("id"), Some(Value::Int(1))));
                assert!(matches!(fields.get("name"), Some(Value::Int(4))));
                assert!(matches!(fields.get("value"), Some(Value::Int(2))));
            }
            _ => panic!("expected Schema struct"),
        }
    }

    // -- Categorical standalone roundtrip --

    #[test]
    fn test_categorical_roundtrip() {
        let levels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let codes = vec![0u32, 1, 2, 0];
        let mut buf = Vec::new();
        encode_categorical(&levels, &codes, &mut buf);
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 4);
                match &arr[0] { Value::String(s) => assert_eq!(s.as_str(), "a"), _ => panic!() }
                match &arr[1] { Value::String(s) => assert_eq!(s.as_str(), "b"), _ => panic!() }
                match &arr[2] { Value::String(s) => assert_eq!(s.as_str(), "c"), _ => panic!() }
                match &arr[3] { Value::String(s) => assert_eq!(s.as_str(), "a"), _ => panic!() }
            }
            _ => panic!("expected Array"),
        }
    }

    // -- SparseCsr standalone roundtrip --

    #[test]
    fn test_sparse_csr_standalone_roundtrip() {
        let mut buf = Vec::new();
        encode_sparse_csr(
            2, 3,
            &[0, 2, 3],
            &[0, 2, 1],
            &[1.0, 2.0, 3.0],
            &mut buf,
        );
        let decoded = snap_decode(&buf).unwrap();
        match decoded {
            Value::SparseTensor(s) => {
                assert_eq!(s.nrows, 2);
                assert_eq!(s.ncols, 3);
                assert_eq!(s.values, vec![1.0, 2.0, 3.0]);
                assert_eq!(s.col_indices, vec![0, 2, 1]);
                assert_eq!(s.row_offsets, vec![0, 2, 3]);
            }
            _ => panic!("expected SparseTensor"),
        }
    }
}
