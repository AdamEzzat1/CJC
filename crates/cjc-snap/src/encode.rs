//! Canonical binary encoding for CJC `Value` types.
//!
//! Encoding is deterministic: the same logical value always produces the same
//! byte sequence regardless of HashMap iteration order, Rc pointer identity,
//! etc. This is critical for content-addressable hashing.

use cjc_runtime::Value;

// ---------------------------------------------------------------------------
// Tag bytes -- one byte identifies each variant
// ---------------------------------------------------------------------------

pub const TAG_VOID: u8 = 0x00;
pub const TAG_INT: u8 = 0x01;
pub const TAG_FLOAT: u8 = 0x02;
pub const TAG_BOOL: u8 = 0x03;
pub const TAG_STRING: u8 = 0x04;
pub const TAG_ARRAY: u8 = 0x05;
pub const TAG_TUPLE: u8 = 0x06;
pub const TAG_STRUCT: u8 = 0x07;
pub const TAG_TENSOR: u8 = 0x08;
pub const TAG_ENUM: u8 = 0x09;
pub const TAG_BYTES: u8 = 0x0A;
pub const TAG_BYTESLICE: u8 = 0x0B;
pub const TAG_STRVIEW: u8 = 0x0C;
pub const TAG_U8: u8 = 0x0D;
pub const TAG_BF16: u8 = 0x0E;
pub const TAG_F16: u8 = 0x0F;
pub const TAG_COMPLEX: u8 = 0x10;
pub const TAG_MAP: u8 = 0x11;
pub const TAG_TYPED_TENSOR: u8 = 0x12;
pub const TAG_CHUNKED_TENSOR: u8 = 0x13;
pub const TAG_SPARSE_CSR: u8 = 0x14;
pub const TAG_CATEGORICAL: u8 = 0x15;
pub const TAG_SCHEMA: u8 = 0x16;
pub const TAG_DATAFRAME: u8 = 0x17;

/// Snap format magic bytes and version.
pub const SNAP_MAGIC: &[u8; 4] = b"CJS\x01";
pub const SNAP_VERSION: u8 = 2;

/// Canonical NaN representation for f64 (quiet NaN with no payload).
const CANONICAL_NAN_BITS: u64 = 0x7FF8_0000_0000_0000;

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Encode a CJC `Value` into a canonical byte representation.
///
/// The encoding is fully deterministic:
/// - Struct fields are sorted by name.
/// - Floats: NaN is canonicalized to a single bit pattern.
/// - Integers: little-endian 8 bytes.
/// - Strings: 8-byte length prefix + UTF-8 bytes.
///
/// Unsupported runtime-only variants (Fn, Closure, ClassRef, Scratchpad,
/// GradGraph, etc.) will panic. These are not data values and cannot be
/// meaningfully serialized.
pub fn snap_encode(value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    encode_value(value, &mut buf);
    buf
}

fn encode_value(value: &Value, buf: &mut Vec<u8>) {
    match value {
        Value::Void => {
            buf.push(TAG_VOID);
        }
        Value::Int(v) => {
            buf.push(TAG_INT);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Value::Float(v) => {
            buf.push(TAG_FLOAT);
            let bits = if v.is_nan() {
                CANONICAL_NAN_BITS
            } else {
                v.to_bits()
            };
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        Value::Bool(v) => {
            buf.push(TAG_BOOL);
            buf.push(if *v { 0x01 } else { 0x00 });
        }
        Value::String(s) => {
            buf.push(TAG_STRING);
            encode_string(s.as_str(), buf);
        }
        Value::Bytes(b) => {
            buf.push(TAG_BYTES);
            let data = b.borrow();
            let len = data.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&data);
        }
        Value::ByteSlice(b) => {
            buf.push(TAG_BYTESLICE);
            let len = b.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(b);
        }
        Value::StrView(b) => {
            buf.push(TAG_STRVIEW);
            let len = b.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(b);
        }
        Value::U8(v) => {
            buf.push(TAG_U8);
            buf.push(*v);
        }
        Value::Array(arr) => {
            buf.push(TAG_ARRAY);
            let len = arr.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            for elem in arr.iter() {
                encode_value(elem, buf);
            }
        }
        Value::Tuple(elems) => {
            buf.push(TAG_TUPLE);
            let len = elems.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            for elem in elems.iter() {
                encode_value(elem, buf);
            }
        }
        Value::Struct { name, fields } => {
            buf.push(TAG_STRUCT);
            // Encode struct name
            encode_string(name, buf);
            // Sort fields by name for determinism
            let mut sorted_fields: Vec<(&String, &Value)> = fields.iter().collect();
            sorted_fields.sort_by_key(|(k, _)| *k);
            // Encode field count
            let count = sorted_fields.len() as u64;
            buf.extend_from_slice(&count.to_le_bytes());
            // Encode each field: name + value
            for (key, val) in sorted_fields {
                encode_string(key, buf);
                encode_value(val, buf);
            }
        }
        Value::Tensor(t) => {
            buf.push(TAG_TENSOR);
            let shape = t.shape();
            let ndim = shape.len() as u64;
            buf.extend_from_slice(&ndim.to_le_bytes());
            for &dim in shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            // Write contiguous f64 data
            let data = t.to_vec();
            for &val in &data {
                let bits = if val.is_nan() {
                    CANONICAL_NAN_BITS
                } else {
                    val.to_bits()
                };
                buf.extend_from_slice(&bits.to_le_bytes());
            }
        }
        Value::Enum {
            enum_name,
            variant,
            fields,
        } => {
            buf.push(TAG_ENUM);
            encode_string(enum_name, buf);
            encode_string(variant, buf);
            let count = fields.len() as u64;
            buf.extend_from_slice(&count.to_le_bytes());
            for field in fields {
                encode_value(field, buf);
            }
        }
        Value::Bf16(v) => {
            buf.push(TAG_BF16);
            buf.extend_from_slice(&v.0.to_le_bytes());
        }
        Value::F16(v) => {
            buf.push(TAG_F16);
            buf.extend_from_slice(&v.0.to_le_bytes());
        }
        Value::Complex(z) => {
            buf.push(TAG_COMPLEX);
            let re_bits = if z.re.is_nan() {
                CANONICAL_NAN_BITS
            } else {
                z.re.to_bits()
            };
            let im_bits = if z.im.is_nan() {
                CANONICAL_NAN_BITS
            } else {
                z.im.to_bits()
            };
            buf.extend_from_slice(&re_bits.to_le_bytes());
            buf.extend_from_slice(&im_bits.to_le_bytes());
        }
        Value::Map(m) => {
            buf.push(TAG_MAP);
            let map = m.borrow();
            // DetMap preserves insertion order, but for canonical encoding
            // we sort entries by their encoded key representation.
            let entries: Vec<_> = map.iter().collect();
            // Sort by key's canonical encoding for determinism
            let mut sorted: Vec<(Vec<u8>, &Value, &Value)> = entries
                .iter()
                .map(|(k, v)| {
                    let mut key_buf = Vec::new();
                    encode_value(k, &mut key_buf);
                    (key_buf, *k, *v)
                })
                .collect();
            sorted.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

            let count = sorted.len() as u64;
            buf.extend_from_slice(&count.to_le_bytes());
            for (key_bytes, _, val) in &sorted {
                buf.extend_from_slice(key_bytes);
                encode_value(val, buf);
            }
        }

        Value::SparseTensor(s) => {
            encode_sparse_csr(s.nrows, s.ncols, &s.row_offsets, &s.col_indices, &s.values, buf);
        }

        // Runtime-only variants that cannot be meaningfully serialized:
        Value::ClassRef(_)
        | Value::Fn(_)
        | Value::Closure { .. }
        | Value::Regex { .. }
        | Value::Scratchpad(_)
        | Value::PagedKvCache(_)
        | Value::AlignedBytes(_)
        | Value::GradGraph(_)
        | Value::OptimizerState(_)
        | Value::TidyView(_)
        | Value::GroupedTidyView(_)
        | Value::VizorPlot(_) => {
            panic!(
                "snap_encode: cannot serialize runtime-only variant: {}",
                value.type_name()
            );
        }
    }
}

/// Encode a CJC `Value` into the v2 snap format with magic header and version byte.
///
/// Format: [MAGIC: "CJS\x01"][version: u8][flags: u8][payload...]
/// Flags: 0x00 = uncompressed, no special features
pub fn snap_encode_v2(value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(SNAP_MAGIC);
    buf.push(SNAP_VERSION);
    buf.push(0x00); // flags: uncompressed
    encode_value(value, &mut buf);
    buf
}

/// Encode a typed tensor (DType + raw bytes).
/// Format: [TAG_TYPED_TENSOR][dtype: u8][ndim: u64][shape...][raw_bytes]
pub fn encode_typed_tensor(
    dtype_tag: u8,
    shape: &[usize],
    raw_bytes: &[u8],
    buf: &mut Vec<u8>,
) {
    buf.push(TAG_TYPED_TENSOR);
    buf.push(dtype_tag);
    let ndim = shape.len() as u64;
    buf.extend_from_slice(&ndim.to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    let byte_len = raw_bytes.len() as u64;
    buf.extend_from_slice(&byte_len.to_le_bytes());
    buf.extend_from_slice(raw_bytes);
}

/// Encode a sparse CSR matrix.
/// Format: [TAG_SPARSE_CSR][dtype: u8][nrows: u64][ncols: u64][nnz: u64]
///         [row_ptr bytes][col_idx bytes][values bytes]
pub fn encode_sparse_csr(
    nrows: usize,
    ncols: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    buf: &mut Vec<u8>,
) {
    buf.push(TAG_SPARSE_CSR);
    buf.push(0x00); // dtype = f64
    buf.extend_from_slice(&(nrows as u64).to_le_bytes());
    buf.extend_from_slice(&(ncols as u64).to_le_bytes());
    let nnz = values.len() as u64;
    buf.extend_from_slice(&nnz.to_le_bytes());
    // row_ptr: (nrows+1) entries
    for &rp in row_ptr {
        buf.extend_from_slice(&(rp as u64).to_le_bytes());
    }
    // col_idx: nnz entries
    for &ci in col_idx {
        buf.extend_from_slice(&(ci as u64).to_le_bytes());
    }
    // values: nnz f64s
    for &v in values {
        let bits = if v.is_nan() { CANONICAL_NAN_BITS } else { v.to_bits() };
        buf.extend_from_slice(&bits.to_le_bytes());
    }
}

/// Encode a categorical column.
/// Format: [TAG_CATEGORICAL][n_levels: u32][level_strings...][n_rows: u64][codes: n_rows × u32]
pub fn encode_categorical(
    levels: &[String],
    codes: &[u32],
    buf: &mut Vec<u8>,
) {
    buf.push(TAG_CATEGORICAL);
    let n_levels = levels.len() as u32;
    buf.extend_from_slice(&n_levels.to_le_bytes());
    for level in levels {
        encode_string(level, buf);
    }
    let n_rows = codes.len() as u64;
    buf.extend_from_slice(&n_rows.to_le_bytes());
    for &c in codes {
        buf.extend_from_slice(&c.to_le_bytes());
    }
}

/// Encode a schema (field names + types).
/// Format: [TAG_SCHEMA][n_fields: u32][name: str, type_tag: u8]...
pub fn encode_schema(
    fields: &[(String, u8)],
    buf: &mut Vec<u8>,
) {
    buf.push(TAG_SCHEMA);
    let n_fields = fields.len() as u32;
    buf.extend_from_slice(&n_fields.to_le_bytes());
    for (name, type_tag) in fields {
        encode_string(name, buf);
        buf.push(*type_tag);
    }
}

/// Default chunk size for chunked tensor encoding: 4 MB.
pub const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Column type tags for DataFrame encoding.
pub const COL_TYPE_INT: u8 = 0;
pub const COL_TYPE_FLOAT: u8 = 1;
pub const COL_TYPE_STR: u8 = 2;
pub const COL_TYPE_BOOL: u8 = 3;
pub const COL_TYPE_CATEGORICAL: u8 = 4;
pub const COL_TYPE_DATETIME: u8 = 5;

/// Encode a tensor as chunked format with per-chunk SHA-256 hashes.
///
/// Format: [TAG_CHUNKED_TENSOR][dtype: u8][ndim: u64][shape...]
///         [chunk_size: u64][n_chunks: u64]
///         [chunk_0_len: u64][chunk_0_hash: 32 bytes][chunk_0_bytes...]
///         [chunk_1_len: u64][chunk_1_hash: 32 bytes][chunk_1_bytes...]
///         ...
///
/// This enables streaming, content-addressable chunks, and resumable I/O.
pub fn encode_chunked_tensor(
    dtype_tag: u8,
    shape: &[usize],
    raw_bytes: &[u8],
    chunk_size: usize,
    buf: &mut Vec<u8>,
) {
    buf.push(TAG_CHUNKED_TENSOR);
    buf.push(dtype_tag);
    let ndim = shape.len() as u64;
    buf.extend_from_slice(&ndim.to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&(dim as u64).to_le_bytes());
    }

    let cs = if chunk_size == 0 { DEFAULT_CHUNK_SIZE } else { chunk_size };
    buf.extend_from_slice(&(cs as u64).to_le_bytes());

    // Calculate number of chunks
    let n_chunks = if raw_bytes.is_empty() {
        0usize
    } else {
        (raw_bytes.len() + cs - 1) / cs
    };
    buf.extend_from_slice(&(n_chunks as u64).to_le_bytes());

    // Encode each chunk: [len][sha256][bytes]
    for i in 0..n_chunks {
        let start = i * cs;
        let end = (start + cs).min(raw_bytes.len());
        let chunk = &raw_bytes[start..end];
        let chunk_len = chunk.len() as u64;
        let chunk_hash = crate::sha256(chunk);

        buf.extend_from_slice(&chunk_len.to_le_bytes());
        buf.extend_from_slice(&chunk_hash);
        buf.extend_from_slice(chunk);
    }
}

/// Encode a DataFrame as columnar binary format.
///
/// Format: [TAG_DATAFRAME][n_cols: u32][n_rows: u64]
///         [col_name: str][col_type: u8][col_data...]...
///
/// Column data formats:
/// - Int:         [i64 × n_rows]
/// - Float:       [f64 bits × n_rows] (NaN canonicalized)
/// - Str:         [string × n_rows]
/// - Bool:        [u8 × n_rows] (0x00/0x01)
/// - Categorical: [n_levels: u32][level_strings...][codes: u32 × n_rows]
/// - DateTime:    [i64 × n_rows] (epoch millis)
pub fn encode_dataframe(
    column_names: &[&str],
    column_types: &[u8],
    column_data: &[DataFrameColumnData<'_>],
    n_rows: usize,
    buf: &mut Vec<u8>,
) {
    buf.push(TAG_DATAFRAME);
    let n_cols = column_names.len() as u32;
    buf.extend_from_slice(&n_cols.to_le_bytes());
    buf.extend_from_slice(&(n_rows as u64).to_le_bytes());

    for i in 0..column_names.len() {
        encode_string(column_names[i], buf);
        buf.push(column_types[i]);

        match &column_data[i] {
            DataFrameColumnData::Int(vals) => {
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            DataFrameColumnData::Float(vals) => {
                for &v in vals.iter() {
                    let bits = if v.is_nan() { CANONICAL_NAN_BITS } else { v.to_bits() };
                    buf.extend_from_slice(&bits.to_le_bytes());
                }
            }
            DataFrameColumnData::Str(vals) => {
                for s in vals.iter() {
                    encode_string(s, buf);
                }
            }
            DataFrameColumnData::Bool(vals) => {
                for &b in vals.iter() {
                    buf.push(if b { 0x01 } else { 0x00 });
                }
            }
            DataFrameColumnData::Categorical { levels, codes } => {
                let n_levels = levels.len() as u32;
                buf.extend_from_slice(&n_levels.to_le_bytes());
                for level in levels.iter() {
                    encode_string(level, buf);
                }
                for &c in codes.iter() {
                    buf.extend_from_slice(&c.to_le_bytes());
                }
            }
            DataFrameColumnData::DateTime(vals) => {
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

/// Typed column data for DataFrame encoding.
pub enum DataFrameColumnData<'a> {
    Int(&'a [i64]),
    Float(&'a [f64]),
    Str(&'a [String]),
    Bool(&'a [bool]),
    Categorical { levels: &'a [String], codes: &'a [u32] },
    DateTime(&'a [i64]),
}

/// Encode a string as 8-byte little-endian length + UTF-8 bytes.
fn encode_string(s: &str, buf: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u64;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    #[test]
    fn test_encode_void() {
        let bytes = snap_encode(&Value::Void);
        assert_eq!(bytes, vec![TAG_VOID]);
    }

    #[test]
    fn test_encode_int() {
        let bytes = snap_encode(&Value::Int(42));
        assert_eq!(bytes[0], TAG_INT);
        assert_eq!(bytes.len(), 9);
        let val = i64::from_le_bytes(bytes[1..9].try_into().unwrap());
        assert_eq!(val, 42);
    }

    #[test]
    fn test_encode_negative_int() {
        let bytes = snap_encode(&Value::Int(-1));
        let val = i64::from_le_bytes(bytes[1..9].try_into().unwrap());
        assert_eq!(val, -1);
    }

    #[test]
    fn test_encode_float() {
        let bytes = snap_encode(&Value::Float(3.14));
        assert_eq!(bytes[0], TAG_FLOAT);
        assert_eq!(bytes.len(), 9);
        let bits = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
        assert_eq!(f64::from_bits(bits), 3.14);
    }

    #[test]
    fn test_encode_nan_canonicalized() {
        let nan1 = snap_encode(&Value::Float(f64::NAN));
        let nan2 = snap_encode(&Value::Float(-f64::NAN));
        // Both NaN variants produce the same canonical encoding
        assert_eq!(nan1, nan2);
        let bits = u64::from_le_bytes(nan1[1..9].try_into().unwrap());
        assert_eq!(bits, CANONICAL_NAN_BITS);
    }

    #[test]
    fn test_encode_bool() {
        let t = snap_encode(&Value::Bool(true));
        let f = snap_encode(&Value::Bool(false));
        assert_eq!(t, vec![TAG_BOOL, 0x01]);
        assert_eq!(f, vec![TAG_BOOL, 0x00]);
    }

    #[test]
    fn test_encode_string() {
        let val = Value::String(Rc::new("hello".to_string()));
        let bytes = snap_encode(&val);
        assert_eq!(bytes[0], TAG_STRING);
        let len = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
        assert_eq!(len, 5);
        assert_eq!(&bytes[9..14], b"hello");
    }

    #[test]
    fn test_encode_array() {
        let val = Value::Array(Rc::new(vec![Value::Int(1), Value::Int(2)]));
        let bytes = snap_encode(&val);
        assert_eq!(bytes[0], TAG_ARRAY);
        let len = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
        assert_eq!(len, 2);
    }

    #[test]
    fn test_encode_struct_sorted_fields() {
        // Fields in HashMap may come in any order, but encoding must be sorted
        let mut fields = BTreeMap::new();
        fields.insert("z".to_string(), Value::Int(3));
        fields.insert("a".to_string(), Value::Int(1));
        fields.insert("m".to_string(), Value::Int(2));
        let val = Value::Struct {
            name: "Test".to_string(),
            fields,
        };
        let bytes1 = snap_encode(&val);

        // Encode again -- must produce identical bytes
        let mut fields2 = BTreeMap::new();
        fields2.insert("m".to_string(), Value::Int(2));
        fields2.insert("a".to_string(), Value::Int(1));
        fields2.insert("z".to_string(), Value::Int(3));
        let val2 = Value::Struct {
            name: "Test".to_string(),
            fields: fields2,
        };
        let bytes2 = snap_encode(&val2);

        assert_eq!(bytes1, bytes2, "struct encoding must be deterministic regardless of insertion order");
    }

    #[test]
    fn test_encode_deterministic() {
        let v1 = Value::Float(1.0);
        let v2 = Value::Float(1.0);
        assert_eq!(snap_encode(&v1), snap_encode(&v2));
    }
}
