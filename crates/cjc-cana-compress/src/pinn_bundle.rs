//! CPB0 — CANA PINN Bundle v0: on-disk weight persistence for the
//! trained PINN v2 thermal head (handoff §2.4 / Phase-A §8 Q2).
//!
//! ## Format
//!
//! ```text
//! [0..4)   b"CPB0"                       file magic
//! [4..8)   u32 LE                        bundle schema version
//! [8..12)  u32 LE                        compressed payload length
//! [12..)   lossless payload              (embeds its own input hash →
//!                                         per-bundle integrity on read)
//! ```
//!
//! The payload decompresses to a fixed-order little-endian encoding of
//! the model identity + weights:
//!
//! ```text
//! model_id (len-prefixed str) · model_version u32 · feature_count u32
//! · means [f64; n] · stds [f64; n] · coefficients [f64; n]
//! · intercept f64
//! ```
//!
//! ## Determinism contract
//!
//! Same weights → byte-identical bundle (fixed field order, `to_bits`
//! f64 encoding, deterministic compressor). `model_id`/`model_version`
//! travel INSIDE the bundle so a loaded head always reports the
//! identity it was trained under — that identity flows into report
//! hashes via `CostModel::name()/version()` (the v1 precedent:
//! `PINN_V1_MODEL_ID`).
//!
//! Loaders validate magic, schema version, feature count, payload
//! integrity (via the lossless codec's embedded hash), and weight
//! validity (`PinnThermalV2::is_valid`) — a corrupted or NaN bundle is
//! an `Err`, never a silently-degraded model.

use std::fs;
use std::io;
use std::path::Path;

use cjc_cana::pinn_thermal_v2::{PinnThermalV2, PINN_V2_FEATURE_COUNT};

use crate::candidate::CompressionError;
use crate::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};

/// On-disk magic for a PINN weight bundle.
const BUNDLE_MAGIC: &[u8; 4] = b"CPB0";

/// Bump on incompatible bundle layout changes.
pub const PINN_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// A trained head plus the model identity it was trained under.
#[derive(Debug, Clone, PartialEq)]
pub struct PinnBundle {
    /// Model identifier (e.g. `"pinn_thermal_v2"`); flows into report
    /// hashes when the head is attached to a cost model.
    pub model_id: String,
    /// Monotonic model version.
    pub model_version: u32,
    /// The trained weights.
    pub head: PinnThermalV2,
}

/// Errors from bundle IO / decoding.
#[derive(Debug)]
pub enum PinnBundleError {
    /// Filesystem error.
    Io(io::Error),
    /// Structural error (magic / version / framing / field validity).
    Malformed(&'static str),
    /// Payload failed decompression or integrity verification.
    Compression(CompressionError),
}

impl std::fmt::Display for PinnBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinnBundleError::Io(e) => write!(f, "pinn bundle io error: {e}"),
            PinnBundleError::Malformed(m) => write!(f, "pinn bundle malformed: {m}"),
            PinnBundleError::Compression(e) => write!(f, "pinn bundle payload error: {e}"),
        }
    }
}

impl std::error::Error for PinnBundleError {}

impl PinnBundle {
    /// Fixed-order canonical encoding of identity + weights (the
    /// pre-compression payload).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + PINN_V2_FEATURE_COUNT * 24);
        out.extend_from_slice(&(self.model_id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.model_id.as_bytes());
        out.extend_from_slice(&self.model_version.to_le_bytes());
        out.extend_from_slice(&(PINN_V2_FEATURE_COUNT as u32).to_le_bytes());
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
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, PinnBundleError> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], PinnBundleError> {
            if *pos + n > bytes.len() {
                return Err(PinnBundleError::Malformed("bundle payload truncated"));
            }
            let s = &bytes[*pos..*pos + n];
            *pos += n;
            Ok(s)
        };
        let id_len = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        let model_id = String::from_utf8(take(&mut pos, id_len)?.to_vec())
            .map_err(|_| PinnBundleError::Malformed("non-utf8 model id"))?;
        let model_version = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
        let feature_count = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        if feature_count != PINN_V2_FEATURE_COUNT {
            return Err(PinnBundleError::Malformed(
                "feature count does not match this build's basis",
            ));
        }
        let read_arr =
            |pos: &mut usize| -> Result<[f64; PINN_V2_FEATURE_COUNT], PinnBundleError> {
                let mut arr = [0.0f64; PINN_V2_FEATURE_COUNT];
                for slot in arr.iter_mut() {
                    let raw = u64::from_le_bytes(take(pos, 8)?.try_into().unwrap());
                    *slot = f64::from_bits(raw);
                }
                Ok(arr)
            };
        let feature_means = read_arr(&mut pos)?;
        let feature_stds = read_arr(&mut pos)?;
        let coefficients = read_arr(&mut pos)?;
        let intercept = f64::from_bits(u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()));
        if pos != bytes.len() {
            return Err(PinnBundleError::Malformed("trailing bytes after bundle"));
        }
        let head = PinnThermalV2 {
            feature_means,
            feature_stds,
            coefficients,
            intercept,
        };
        if !head.is_valid() {
            return Err(PinnBundleError::Malformed(
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
/// the boundary — an unloadable bundle must be impossible to produce
/// through this API.
pub fn write_bundle(path: &Path, bundle: &PinnBundle) -> Result<(), PinnBundleError> {
    if !bundle.head.is_valid() {
        return Err(PinnBundleError::Malformed(
            "refusing to persist invalid weights",
        ));
    }
    let payload = lossless_compress_bytes(&bundle.canonical_bytes());
    let mut out = Vec::with_capacity(12 + payload.bytes.len());
    out.extend_from_slice(BUNDLE_MAGIC);
    out.extend_from_slice(&PINN_BUNDLE_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload.bytes);
    fs::write(path, &out).map_err(PinnBundleError::Io)
}

/// Read a bundle from `path`, validating magic, schema version,
/// payload integrity, and weight validity.
pub fn read_bundle(path: &Path) -> Result<PinnBundle, PinnBundleError> {
    let bytes = fs::read(path).map_err(PinnBundleError::Io)?;
    if bytes.len() < 12 {
        return Err(PinnBundleError::Malformed("file shorter than header"));
    }
    if &bytes[0..4] != BUNDLE_MAGIC {
        return Err(PinnBundleError::Malformed("bad bundle magic"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != PINN_BUNDLE_SCHEMA_VERSION {
        return Err(PinnBundleError::Malformed("unsupported bundle version"));
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if 12 + len != bytes.len() {
        return Err(PinnBundleError::Malformed("payload length mismatch"));
    }
    let payload = lossless_decompress_bytes(&bytes[12..]).map_err(PinnBundleError::Compression)?;
    PinnBundle::from_canonical_bytes(&payload)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_cana::pinn_thermal_v2::{PINN_V2_MODEL_ID, PINN_V2_MODEL_VERSION};

    fn sample_bundle() -> PinnBundle {
        PinnBundle {
            model_id: PINN_V2_MODEL_ID.to_string(),
            model_version: PINN_V2_MODEL_VERSION,
            head: PinnThermalV2 {
                feature_means: [3.1, 8.2, 0.5, 0.1, 0.2, 2.7, 0.05],
                feature_stds: [1.5, 2.0, 0.7, 0.3, 0.4, 1.9, 0.06],
                coefficients: [0.01, -0.02, 0.003, 0.0, 0.004, 0.12, 0.31],
                intercept: 0.15,
            },
        }
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let b = sample_bundle();
        let back = PinnBundle::from_canonical_bytes(&b.canonical_bytes()).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn encoding_is_deterministic() {
        let b = sample_bundle();
        let first = b.canonical_bytes();
        for _ in 0..20 {
            assert_eq!(first, b.canonical_bytes());
        }
    }

    #[test]
    fn file_round_trip_preserves_bits() {
        let dir = std::env::temp_dir().join("cana_pinn_bundle_test_rt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.cpb");
        let b = sample_bundle();
        write_bundle(&path, &b).unwrap();
        let back = read_bundle(&path).unwrap();
        assert_eq!(b, back);
        // Bit-exactness, not approximate equality.
        for i in 0..PINN_V2_FEATURE_COUNT {
            assert_eq!(
                b.head.coefficients[i].to_bits(),
                back.head.coefficients[i].to_bits()
            );
        }
        // Double-write is byte-identical (determinism contract).
        let first_bytes = fs::read(&path).unwrap();
        write_bundle(&path, &b).unwrap();
        assert_eq!(first_bytes, fs::read(&path).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_weights_refused_on_write() {
        let dir = std::env::temp_dir().join("cana_pinn_bundle_test_invalid");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut b = sample_bundle();
        b.head.feature_stds[2] = 0.0;
        assert!(write_bundle(&dir.join("bad.cpb"), &b).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_file_rejected_not_panicked() {
        let dir = std::env::temp_dir().join("cana_pinn_bundle_test_corrupt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.cpb");
        write_bundle(&path, &sample_bundle()).unwrap();
        let mut bytes = fs::read(&path).unwrap();

        // Flip a byte inside the compressed payload.
        let mid = bytes.len() - 4;
        bytes[mid] ^= 0xFF;
        let corrupted = dir.join("corrupted.cpb");
        fs::write(&corrupted, &bytes).unwrap();
        assert!(read_bundle(&corrupted).is_err());

        // Truncation at several depths.
        for cut in [0, 3, 11, bytes.len() / 2] {
            let t = dir.join(format!("trunc_{cut}.cpb"));
            fs::write(&t, &bytes[..cut]).unwrap();
            assert!(read_bundle(&t).is_err(), "cut at {cut} must be rejected");
        }

        // Bad magic / bad version.
        let good = fs::read(&path).unwrap();
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        fs::write(dir.join("badmagic.cpb"), &bad_magic).unwrap();
        assert!(read_bundle(&dir.join("badmagic.cpb")).is_err());
        let mut bad_ver = good.clone();
        bad_ver[4] = 0xEE;
        fs::write(dir.join("badver.cpb"), &bad_ver).unwrap();
        assert!(read_bundle(&dir.join("badver.cpb")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn nan_payload_rejected_by_validity_gate() {
        let mut b = sample_bundle();
        b.head.intercept = f64::NAN;
        // Bypass write_bundle's gate by encoding directly.
        let bytes = b.canonical_bytes();
        assert!(PinnBundle::from_canonical_bytes(&bytes).is_err());
    }
}
