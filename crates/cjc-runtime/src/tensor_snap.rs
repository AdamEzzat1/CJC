//! Deterministic binary serialization for tensors and tensor lists.
//!
//! This module provides a small, self-contained, byte-stable format for
//! saving and loading f64 tensors. It lives in `cjc-runtime` (not
//! `cjc-snap`) because `cjc-snap` already depends on `cjc-runtime`, so
//! adding the reverse dependency would create a cycle.
//!
//! # Wire format
//!
//! All integers are little-endian, all floats are IEEE-754 f64 little-endian.
//!
//! ```text
//! offset  bytes  field
//! ------  -----  ------------------------------------------------------
//!      0      4  magic = b"CJCT"
//!      4      1  format version = 1
//!      5      3  reserved = 0
//!      8      8  n_tensors : u64
//!     16      -  tensor[0], tensor[1], ...
//!
//! tensor layout:
//!      0      8  ndim : u64
//!      8  ndim*8  shape[0..ndim] : u64[]
//!      *  numel*8  data : f64[]      (row-major, contiguous)
//!
//! after the last tensor:
//!      0      8  footer_hash : u64  (SplitMix64 fold of all preceding bytes)
//! ```
//!
//! The footer hash lets a reader cheaply detect corruption and lets tests
//! assert byte-identity across executors without parsing the payload.
//!
//! Determinism properties:
//! - Tensors are materialized with `to_vec()` so stride/offset views are
//!   flattened to a canonical row-major order before serialization.
//! - NaN bit patterns are preserved as-written (we do not canonicalize);
//!   CJC-Lang produces NaN only through explicit `0.0 / 0.0`-style paths,
//!   and the determinism contract never introduces spurious NaNs.
//! - The footer hash is a SplitMix64 fold, which is order-sensitive and
//!   deterministic across platforms (pure integer arithmetic, no FP).

use crate::tensor::Tensor;

const MAGIC: &[u8; 4] = b"CJCT";
const FORMAT_VERSION: u8 = 1;
const HEADER_LEN: usize = 16; // magic(4) + ver(1) + reserved(3) + n_tensors(8)

/// Errors returned by the tensor snap codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorSnapError {
    TooShort,
    BadMagic,
    BadVersion(u8),
    TrailingGarbage,
    BadShape,
    BadHash { expected: u64, actual: u64 },
}

impl std::fmt::Display for TensorSnapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "tensor snap: input too short"),
            Self::BadMagic => write!(f, "tensor snap: bad magic (expected CJCT)"),
            Self::BadVersion(v) => write!(f, "tensor snap: unsupported version {v}"),
            Self::TrailingGarbage => write!(f, "tensor snap: trailing garbage after footer"),
            Self::BadShape => write!(f, "tensor snap: corrupt shape header"),
            Self::BadHash { expected, actual } => {
                write!(f, "tensor snap: hash mismatch (expected {expected:#x}, got {actual:#x})")
            }
        }
    }
}

/// SplitMix64 folding hash. Pure integer arithmetic, deterministic across
/// platforms. Not cryptographic — used only for integrity/parity checks.
#[inline]
fn splitmix64_fold(bytes: &[u8]) -> u64 {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    // Mix length first so `[]` hashes differently from `[0x00 * 0]`.
    state ^= bytes.len() as u64;
    state = mix64(state);

    // Fold 8 bytes at a time.
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let mut chunk = [0u8; 8];
        chunk.copy_from_slice(&bytes[i..i + 8]);
        state ^= u64::from_le_bytes(chunk);
        state = mix64(state);
        i += 8;
    }
    // Fold the trailing tail (0..7 bytes).
    if i < bytes.len() {
        let mut chunk = [0u8; 8];
        for (j, b) in bytes[i..].iter().enumerate() {
            chunk[j] = *b;
        }
        state ^= u64::from_le_bytes(chunk);
        state = mix64(state);
    }
    state
}

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Encode a list of tensors into the wire format. Infallible for any
/// well-formed `Tensor` (materializes views via `to_vec`).
pub fn encode_list(tensors: &[Tensor]) -> Vec<u8> {
    // Estimate capacity to avoid reallocation.
    let mut cap = HEADER_LEN + 8; // + footer
    for t in tensors {
        cap += 8 + 8 * t.ndim() + 8 * t.shape().iter().product::<usize>();
    }
    let mut buf = Vec::with_capacity(cap);

    // Header
    buf.extend_from_slice(MAGIC);
    buf.push(FORMAT_VERSION);
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    // Tensors
    for t in tensors {
        let shape = t.shape();
        buf.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        for &d in shape {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        let data = t.to_vec();
        for v in &data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Footer hash over everything so far.
    let hash = splitmix64_fold(&buf);
    buf.extend_from_slice(&hash.to_le_bytes());
    buf
}

/// Encode a single tensor (convenience wrapper).
pub fn encode_one(tensor: &Tensor) -> Vec<u8> {
    encode_list(std::slice::from_ref(tensor))
}

/// Decode a list of tensors from the wire format. Verifies magic, version,
/// and footer hash.
pub fn decode_list(bytes: &[u8]) -> Result<Vec<Tensor>, TensorSnapError> {
    if bytes.len() < HEADER_LEN + 8 {
        return Err(TensorSnapError::TooShort);
    }
    if &bytes[0..4] != MAGIC {
        return Err(TensorSnapError::BadMagic);
    }
    let version = bytes[4];
    if version != FORMAT_VERSION {
        return Err(TensorSnapError::BadVersion(version));
    }

    // Verify footer hash over everything except the last 8 bytes.
    let footer_start = bytes.len() - 8;
    let expected_hash = u64::from_le_bytes([
        bytes[footer_start],
        bytes[footer_start + 1],
        bytes[footer_start + 2],
        bytes[footer_start + 3],
        bytes[footer_start + 4],
        bytes[footer_start + 5],
        bytes[footer_start + 6],
        bytes[footer_start + 7],
    ]);
    let actual_hash = splitmix64_fold(&bytes[..footer_start]);
    if expected_hash != actual_hash {
        return Err(TensorSnapError::BadHash {
            expected: expected_hash,
            actual: actual_hash,
        });
    }

    // n_tensors
    let n_tensors = read_u64(bytes, 8)? as usize;
    let mut cursor = HEADER_LEN;
    let mut out = Vec::with_capacity(n_tensors);

    for _ in 0..n_tensors {
        if cursor + 8 > footer_start {
            return Err(TensorSnapError::TooShort);
        }
        let ndim = read_u64(bytes, cursor)? as usize;
        cursor += 8;

        // Cap ndim to a sane value to avoid pathological allocation.
        if ndim > 16 {
            return Err(TensorSnapError::BadShape);
        }
        if cursor + 8 * ndim > footer_start {
            return Err(TensorSnapError::TooShort);
        }

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let d = read_u64(bytes, cursor)? as usize;
            shape.push(d);
            cursor += 8;
        }

        // Guard against shape-overflow DoS.
        let numel = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(TensorSnapError::BadShape)?;

        if cursor + 8 * numel > footer_start {
            return Err(TensorSnapError::TooShort);
        }

        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&bytes[cursor..cursor + 8]);
            data.push(f64::from_le_bytes(chunk));
            cursor += 8;
        }

        let t = Tensor::from_vec(data, &shape).map_err(|_| TensorSnapError::BadShape)?;
        out.push(t);
    }

    if cursor != footer_start {
        return Err(TensorSnapError::TrailingGarbage);
    }
    Ok(out)
}

/// Decode a single tensor from the wire format. Errors if the payload
/// contains zero or more than one tensor.
pub fn decode_one(bytes: &[u8]) -> Result<Tensor, TensorSnapError> {
    let list = decode_list(bytes)?;
    if list.len() != 1 {
        return Err(TensorSnapError::BadShape);
    }
    Ok(list.into_iter().next().unwrap())
}

/// Deterministic content hash of a tensor list. Separate from the wire
/// format's footer hash — this one hashes *logical* content (shape + data)
/// only, so it is invariant under re-encoding.
pub fn hash_list(tensors: &[Tensor]) -> u64 {
    let mut state: u64 = 0x243F_6A88_85A3_08D3; // arbitrary constant
    state ^= tensors.len() as u64;
    state = mix64(state);
    for t in tensors {
        let shape = t.shape();
        state ^= shape.len() as u64;
        state = mix64(state);
        for &d in shape {
            state ^= d as u64;
            state = mix64(state);
        }
        let data = t.to_vec();
        // Fold the f64 bits (NaN-bit-preserving — determinism relies on the
        // caller not producing garbage NaN bit patterns).
        for v in &data {
            state ^= v.to_bits();
            state = mix64(state);
        }
    }
    state
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, TensorSnapError> {
    if offset + 8 > bytes.len() {
        return Err(TensorSnapError::TooShort);
    }
    let mut chunk = [0u8; 8];
    chunk.copy_from_slice(&bytes[offset..offset + 8]);
    Ok(u64::from_le_bytes(chunk))
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f64>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape).unwrap()
    }

    #[test]
    fn empty_list_roundtrips() {
        let bytes = encode_list(&[]);
        let out = decode_list(&bytes).unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn scalar_tensor_roundtrips() {
        let a = t(vec![42.0], &[1]);
        let bytes = encode_one(&a);
        let b = decode_one(&bytes).unwrap();
        assert_eq!(b.shape(), &[1]);
        assert_eq!(b.to_vec(), vec![42.0]);
    }

    #[test]
    fn matrix_roundtrips() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let bytes = encode_one(&a);
        let b = decode_one(&bytes).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn multiple_tensors_roundtrip() {
        let a = t(vec![1.0, 2.0], &[2]);
        let b = t(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]);
        let c = t(vec![7.0], &[1, 1]);
        let bytes = encode_list(&[a.clone(), b.clone(), c.clone()]);
        let out = decode_list(&bytes).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].to_vec(), a.to_vec());
        assert_eq!(out[1].to_vec(), b.to_vec());
        assert_eq!(out[2].to_vec(), c.to_vec());
    }

    #[test]
    fn encoding_is_deterministic() {
        let a = t(vec![1.5, -2.5, 3.25], &[3]);
        let e1 = encode_one(&a);
        let e2 = encode_one(&a);
        assert_eq!(e1, e2, "encoding must be byte-identical for the same input");
    }

    #[test]
    fn different_tensors_produce_different_encodings() {
        let a = t(vec![1.0, 2.0], &[2]);
        let b = t(vec![1.0, 2.0, 3.0], &[3]);
        assert_ne!(encode_one(&a), encode_one(&b));
    }

    #[test]
    fn different_shapes_same_data_produce_different_encodings() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let b = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_ne!(encode_one(&a), encode_one(&b));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let a = t(vec![1.0], &[1]);
        let mut bytes = encode_one(&a);
        bytes[0] = b'X';
        assert!(matches!(decode_list(&bytes), Err(TensorSnapError::BadMagic)));
    }

    #[test]
    fn bad_version_is_rejected() {
        let a = t(vec![1.0], &[1]);
        let mut bytes = encode_one(&a);
        bytes[4] = 99;
        assert!(matches!(decode_list(&bytes), Err(TensorSnapError::BadVersion(99))));
    }

    #[test]
    fn hash_mismatch_is_rejected() {
        let a = t(vec![1.0, 2.0, 3.0], &[3]);
        let mut bytes = encode_one(&a);
        // Flip one data byte
        let idx = HEADER_LEN + 8 + 8; // past ndim + shape[0]
        bytes[idx] ^= 0xFF;
        assert!(matches!(decode_list(&bytes), Err(TensorSnapError::BadHash { .. })));
    }

    #[test]
    fn too_short_is_rejected() {
        assert!(matches!(decode_list(&[]), Err(TensorSnapError::TooShort)));
        assert!(matches!(decode_list(&[0u8; 10]), Err(TensorSnapError::TooShort)));
    }

    #[test]
    fn hash_list_is_deterministic() {
        let a = t(vec![1.0, 2.0, 3.0], &[3]);
        let b = t(vec![4.0, 5.0], &[2]);
        let h1 = hash_list(&[a.clone(), b.clone()]);
        let h2 = hash_list(&[a.clone(), b.clone()]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_list_is_order_sensitive() {
        let a = t(vec![1.0, 2.0, 3.0], &[3]);
        let b = t(vec![4.0, 5.0], &[2]);
        let h1 = hash_list(&[a.clone(), b.clone()]);
        let h2 = hash_list(&[b, a]);
        assert_ne!(h1, h2, "hash must change when order changes");
    }

    #[test]
    fn hash_list_distinguishes_shapes_and_data() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let b = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = t(vec![1.0, 2.0, 3.0, 5.0], &[4]);
        assert_ne!(hash_list(&[a.clone()]), hash_list(&[b]));
        assert_ne!(hash_list(&[a]), hash_list(&[c]));
    }

    #[test]
    fn pathological_ndim_is_rejected() {
        // Construct a blob with ndim=1000 in the header — should error out
        // before attempting to allocate.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.extend_from_slice(&[0u8; 3]);
        bytes.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        bytes.extend_from_slice(&1000u64.to_le_bytes()); // ndim = 1000
        let hash = splitmix64_fold(&bytes);
        bytes.extend_from_slice(&hash.to_le_bytes());
        assert!(matches!(decode_list(&bytes), Err(TensorSnapError::BadShape)));
    }

    #[test]
    fn shape_overflow_is_rejected() {
        // ndim=2, shape=[usize::MAX, 2] — multiplying should overflow.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.extend_from_slice(&[0u8; 3]);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes()); // ndim
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        let hash = splitmix64_fold(&bytes);
        bytes.extend_from_slice(&hash.to_le_bytes());
        assert!(matches!(decode_list(&bytes), Err(TensorSnapError::BadShape)));
    }
}
