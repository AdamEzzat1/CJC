//! CPB2 — CANA Memory Bundle: on-disk weight persistence for the
//! trained Phase-F1 memory head (`pinn_memory_v1`).
//!
//! Byte-for-byte the CPB0 design (`pinn_bundle.rs`) with a different
//! magic, head type and feature count — the format, determinism
//! contract and validation rules are identical:
//!
//! ```text
//! [0..4)   b"CPB2"                       file magic
//! [4..8)   u32 LE                        bundle schema version
//! [8..12)  u32 LE                        compressed payload length
//! [12..)   lossless payload              (embeds its own input hash)
//! ```
//!
//! Payload: `model_id (len-prefixed) · model_version u32 ·
//! feature_count u32 · means [f64; 8] · stds [f64; 8] ·
//! coefficients [f64; 8] · intercept f64`, all little-endian,
//! f64 by bit pattern. Same weights → byte-identical bundle.

use std::fs;
use std::io;
use std::path::Path;

use cjc_cana::pinn_memory_v1::{PinnMemoryV1, PINN_MEMORY_V1_FEATURE_COUNT};

use crate::candidate::CompressionError;
use crate::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};

/// On-disk magic for a memory-head weight bundle.
const BUNDLE_MAGIC: &[u8; 4] = b"CPB2";

/// Bump on incompatible bundle layout changes.
pub const MEMORY_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// A trained memory head plus the model identity it was trained under.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryBundle {
    /// Model identifier (e.g. `"pinn_memory_v1"`).
    pub model_id: String,
    /// Monotonic model version.
    pub model_version: u32,
    /// The trained weights.
    pub head: PinnMemoryV1,
}

/// Errors from bundle IO / decoding.
#[derive(Debug)]
pub enum MemoryBundleError {
    /// Filesystem error.
    Io(io::Error),
    /// Structural error (magic / version / framing / field validity).
    Malformed(&'static str),
    /// Payload failed decompression or integrity verification.
    Compression(CompressionError),
}

impl std::fmt::Display for MemoryBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryBundleError::Io(e) => write!(f, "memory bundle io error: {e}"),
            MemoryBundleError::Malformed(m) => write!(f, "memory bundle malformed: {m}"),
            MemoryBundleError::Compression(e) => write!(f, "memory bundle payload error: {e}"),
        }
    }
}

impl std::error::Error for MemoryBundleError {}

impl MemoryBundle {
    /// Fixed-order canonical encoding of identity + weights.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + PINN_MEMORY_V1_FEATURE_COUNT * 24);
        out.extend_from_slice(&(self.model_id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.model_id.as_bytes());
        out.extend_from_slice(&self.model_version.to_le_bytes());
        out.extend_from_slice(&(PINN_MEMORY_V1_FEATURE_COUNT as u32).to_le_bytes());
        for v in &self.head.feature_means {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in &self.head.feature_stds {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in &self.head.coefficients {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&self.head.intercept.to_bits().to_le_bytes());
        out
    }

    /// Decode from the canonical encoding. Rejects wrong feature
    /// counts, truncation, trailing bytes, and invalid weights.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, MemoryBundleError> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], MemoryBundleError> {
            if *pos + n > bytes.len() {
                return Err(MemoryBundleError::Malformed("bundle payload truncated"));
            }
            let s = &bytes[*pos..*pos + n];
            *pos += n;
            Ok(s)
        };
        let id_len = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        let model_id = String::from_utf8(take(&mut pos, id_len)?.to_vec())
            .map_err(|_| MemoryBundleError::Malformed("non-utf8 model id"))?;
        let model_version = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
        let feature_count = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        if feature_count != PINN_MEMORY_V1_FEATURE_COUNT {
            return Err(MemoryBundleError::Malformed(
                "feature count does not match this build's basis",
            ));
        }
        let read_arr =
            |pos: &mut usize| -> Result<[f64; PINN_MEMORY_V1_FEATURE_COUNT], MemoryBundleError> {
                let mut arr = [0.0f64; PINN_MEMORY_V1_FEATURE_COUNT];
                for slot in arr.iter_mut() {
                    let raw = u64::from_le_bytes(take(pos, 8)?.try_into().unwrap());
                    *slot = f64::from_bits(raw);
                }
                Ok(arr)
            };
        let feature_means = read_arr(&mut pos)?;
        let feature_stds = read_arr(&mut pos)?;
        let coefficients = read_arr(&mut pos)?;
        let intercept =
            f64::from_bits(u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()));
        if pos != bytes.len() {
            return Err(MemoryBundleError::Malformed("trailing bytes after bundle"));
        }
        let head = PinnMemoryV1 {
            feature_means,
            feature_stds,
            coefficients,
            intercept,
        };
        if !head.is_valid() {
            return Err(MemoryBundleError::Malformed(
                "decoded weights fail validity (non-finite or zero std)",
            ));
        }
        Ok(Self {
            model_id,
            model_version,
            head,
        })
    }
}

/// Write a bundle to `path` (overwrites). Rejects invalid weights at
/// the boundary.
pub fn write_memory_bundle(path: &Path, bundle: &MemoryBundle) -> Result<(), MemoryBundleError> {
    if !bundle.head.is_valid() {
        return Err(MemoryBundleError::Malformed(
            "refusing to persist invalid weights",
        ));
    }
    let payload = lossless_compress_bytes(&bundle.canonical_bytes());
    let mut out = Vec::with_capacity(12 + payload.bytes.len());
    out.extend_from_slice(BUNDLE_MAGIC);
    out.extend_from_slice(&MEMORY_BUNDLE_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload.bytes);
    fs::write(path, &out).map_err(MemoryBundleError::Io)
}

/// Read a bundle from `path`, validating magic, schema version,
/// payload integrity, and weight validity.
pub fn read_memory_bundle(path: &Path) -> Result<MemoryBundle, MemoryBundleError> {
    let bytes = fs::read(path).map_err(MemoryBundleError::Io)?;
    if bytes.len() < 12 {
        return Err(MemoryBundleError::Malformed("file shorter than header"));
    }
    if &bytes[0..4] != BUNDLE_MAGIC {
        return Err(MemoryBundleError::Malformed("bad bundle magic"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != MEMORY_BUNDLE_SCHEMA_VERSION {
        return Err(MemoryBundleError::Malformed("unsupported bundle version"));
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if 12 + len != bytes.len() {
        return Err(MemoryBundleError::Malformed("payload length mismatch"));
    }
    let payload =
        lossless_decompress_bytes(&bytes[12..]).map_err(MemoryBundleError::Compression)?;
    MemoryBundle::from_canonical_bytes(&payload)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_cana::pinn_memory_v1::{PINN_MEMORY_V1_MODEL_ID, PINN_MEMORY_V1_MODEL_VERSION};

    fn sample_bundle() -> MemoryBundle {
        MemoryBundle {
            model_id: PINN_MEMORY_V1_MODEL_ID.to_string(),
            model_version: PINN_MEMORY_V1_MODEL_VERSION,
            head: PinnMemoryV1 {
                feature_means: [3.1, 8.2, 0.5, 0.1, 0.2, 2.7, 4.4, 0.05],
                feature_stds: [1.5, 2.0, 0.7, 0.3, 0.4, 1.9, 2.2, 0.06],
                coefficients: [0.01, -0.02, 0.003, 0.0, 0.004, 0.12, 0.4, 0.31],
                intercept: 0.15,
            },
        }
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let b = sample_bundle();
        let back = MemoryBundle::from_canonical_bytes(&b.canonical_bytes()).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn file_round_trip_preserves_bits_and_double_write_is_identical() {
        let dir = std::env::temp_dir().join("cana_memory_bundle_test_rt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.cpb");
        let b = sample_bundle();
        write_memory_bundle(&path, &b).unwrap();
        let back = read_memory_bundle(&path).unwrap();
        assert_eq!(b, back);
        for i in 0..PINN_MEMORY_V1_FEATURE_COUNT {
            assert_eq!(
                b.head.coefficients[i].to_bits(),
                back.head.coefficients[i].to_bits()
            );
        }
        let first_bytes = fs::read(&path).unwrap();
        write_memory_bundle(&path, &b).unwrap();
        assert_eq!(first_bytes, fs::read(&path).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_and_corrupted_bundles_rejected() {
        let dir = std::env::temp_dir().join("cana_memory_bundle_test_bad");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Invalid weights refused on write.
        let mut bad = sample_bundle();
        bad.head.feature_stds[2] = 0.0;
        assert!(write_memory_bundle(&dir.join("bad.cpb"), &bad).is_err());
        // NaN payload rejected by the decode validity gate.
        let mut nan = sample_bundle();
        nan.head.intercept = f64::NAN;
        assert!(MemoryBundle::from_canonical_bytes(&nan.canonical_bytes()).is_err());
        // Corruption / truncation / bad magic / bad version.
        let path = dir.join("weights.cpb");
        write_memory_bundle(&path, &sample_bundle()).unwrap();
        let good = fs::read(&path).unwrap();
        let mut corrupt = good.clone();
        let mid = corrupt.len() - 4;
        corrupt[mid] ^= 0xFF;
        fs::write(dir.join("corrupt.cpb"), &corrupt).unwrap();
        assert!(read_memory_bundle(&dir.join("corrupt.cpb")).is_err());
        for cut in [0, 3, 11, good.len() / 2] {
            let t = dir.join(format!("trunc_{cut}.cpb"));
            fs::write(&t, &good[..cut]).unwrap();
            assert!(read_memory_bundle(&t).is_err());
        }
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        fs::write(dir.join("badmagic.cpb"), &bad_magic).unwrap();
        assert!(read_memory_bundle(&dir.join("badmagic.cpb")).is_err());
        let mut bad_ver = good.clone();
        bad_ver[4] = 0xEE;
        fs::write(dir.join("badver.cpb"), &bad_ver).unwrap();
        assert!(read_memory_bundle(&dir.join("badver.cpb")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
