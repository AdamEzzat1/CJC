//! CPB1 — CANA PINN Bundle v1: on-disk weight persistence for the
//! trained energy head (Phase B). Copies the CPB0 pattern
//! ([`crate::pinn_bundle`]) with one structural difference: the basis
//! is VARIABLE-LENGTH — the pass-name vocabulary is part of the
//! trained artifact, so the bundle carries it and the loader
//! reconstructs the exact feature alignment the trainer used.
//!
//! ## Format
//!
//! ```text
//! [0..4)   b"CPB1"                       file magic
//! [4..8)   u32 LE                        bundle schema version
//! [8..12)  u32 LE                        compressed payload length
//! [12..)   lossless payload              (embeds its own input hash)
//! ```
//!
//! Payload (fixed-order little-endian):
//!
//! ```text
//! model_id (len-prefixed str) · model_version u32
//! · n_passes u32 · pass_names [len-prefixed str; n_passes]
//! · feature_count u32 (validated = 10 + n_passes + 4)
//! · means [f64; n] · stds [f64; n] · coefficients [f64; n]
//! · intercept f64
//! ```
//!
//! ## Determinism contract
//!
//! Same weights → byte-identical bundle. Loaders validate magic,
//! schema version, payload integrity, feature-count consistency with
//! THIS build's basis constants, and weight validity
//! ([`PinnEnergyV1::is_valid`]) — corrupted/NaN bundles are `Err`,
//! never a silently-degraded model.

use std::fs;
use std::io;
use std::path::Path;

use cjc_cana::pinn_energy_v1::{PinnEnergyV1, ENERGY_TAIL_FEATURES, ENERGY_WORKLOAD_FEATURES};

use crate::candidate::CompressionError;
use crate::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};

/// On-disk magic for an energy-head weight bundle.
const BUNDLE_MAGIC: &[u8; 4] = b"CPB1";

/// Bump on incompatible bundle layout changes.
pub const ENERGY_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Sanity cap on the pass vocabulary — far above any real pass list;
/// guards the decoder against adversarial length prefixes.
const MAX_PASS_NAMES: usize = 256;

/// A trained energy head plus the model identity it was trained under.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyBundle {
    /// Model identifier (e.g. `"pinn_energy_v1"`); flows into report
    /// hashes when a consumer activates the head.
    pub model_id: String,
    /// Monotonic model version.
    pub model_version: u32,
    /// The trained weights (pass vocabulary included).
    pub head: PinnEnergyV1,
}

/// Errors from bundle IO / decoding.
#[derive(Debug)]
pub enum EnergyBundleError {
    /// Filesystem error.
    Io(io::Error),
    /// Structural error (magic / version / framing / field validity).
    Malformed(&'static str),
    /// Payload failed decompression or integrity verification.
    Compression(CompressionError),
}

impl std::fmt::Display for EnergyBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnergyBundleError::Io(e) => write!(f, "energy bundle io error: {e}"),
            EnergyBundleError::Malformed(m) => write!(f, "energy bundle malformed: {m}"),
            EnergyBundleError::Compression(e) => write!(f, "energy bundle payload error: {e}"),
        }
    }
}

impl std::error::Error for EnergyBundleError {}

impl EnergyBundle {
    /// Fixed-order canonical encoding of identity + vocabulary +
    /// weights (the pre-compression payload).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let n = self.head.feature_count();
        let mut out = Vec::with_capacity(64 + n * 24);
        out.extend_from_slice(&(self.model_id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.model_id.as_bytes());
        out.extend_from_slice(&self.model_version.to_le_bytes());
        out.extend_from_slice(&(self.head.pass_names.len() as u32).to_le_bytes());
        for name in &self.head.pass_names {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
        }
        out.extend_from_slice(&(n as u32).to_le_bytes());
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

    /// Decode from the canonical encoding. Rejects inconsistent
    /// dimensions, truncation, trailing bytes, and invalid weights.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, EnergyBundleError> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], EnergyBundleError> {
            if *pos + n > bytes.len() {
                return Err(EnergyBundleError::Malformed("bundle payload truncated"));
            }
            let s = &bytes[*pos..*pos + n];
            *pos += n;
            Ok(s)
        };
        let read_u32 = |pos: &mut usize| -> Result<u32, EnergyBundleError> {
            Ok(u32::from_le_bytes(take(pos, 4)?.try_into().unwrap()))
        };
        let read_str = |pos: &mut usize| -> Result<String, EnergyBundleError> {
            let len = read_u32(pos)? as usize;
            String::from_utf8(take(pos, len)?.to_vec())
                .map_err(|_| EnergyBundleError::Malformed("non-utf8 string in bundle"))
        };

        let model_id = read_str(&mut pos)?;
        let model_version = read_u32(&mut pos)?;
        let n_passes = read_u32(&mut pos)? as usize;
        if n_passes > MAX_PASS_NAMES {
            return Err(EnergyBundleError::Malformed("pass vocabulary too large"));
        }
        let mut pass_names = Vec::with_capacity(n_passes);
        for _ in 0..n_passes {
            pass_names.push(read_str(&mut pos)?);
        }
        let feature_count = read_u32(&mut pos)? as usize;
        let expected = ENERGY_WORKLOAD_FEATURES + n_passes + ENERGY_TAIL_FEATURES;
        if feature_count != expected {
            return Err(EnergyBundleError::Malformed(
                "feature count inconsistent with pass vocabulary and this build's basis",
            ));
        }
        let read_vec = |pos: &mut usize| -> Result<Vec<f64>, EnergyBundleError> {
            let mut v = Vec::with_capacity(feature_count);
            for _ in 0..feature_count {
                let raw = u64::from_le_bytes(take(pos, 8)?.try_into().unwrap());
                v.push(f64::from_bits(raw));
            }
            Ok(v)
        };
        let feature_means = read_vec(&mut pos)?;
        let feature_stds = read_vec(&mut pos)?;
        let coefficients = read_vec(&mut pos)?;
        let intercept =
            f64::from_bits(u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()));
        if pos != bytes.len() {
            return Err(EnergyBundleError::Malformed("trailing bytes after bundle"));
        }
        let head = PinnEnergyV1 {
            pass_names,
            feature_means,
            feature_stds,
            coefficients,
            intercept,
        };
        if !head.is_valid() {
            return Err(EnergyBundleError::Malformed(
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

/// Write a bundle to `path` (overwrites). Invalid weights are refused
/// at the boundary.
pub fn write_energy_bundle(path: &Path, bundle: &EnergyBundle) -> Result<(), EnergyBundleError> {
    if !bundle.head.is_valid() {
        return Err(EnergyBundleError::Malformed(
            "refusing to persist invalid weights",
        ));
    }
    let payload = lossless_compress_bytes(&bundle.canonical_bytes());
    let mut out = Vec::with_capacity(12 + payload.bytes.len());
    out.extend_from_slice(BUNDLE_MAGIC);
    out.extend_from_slice(&ENERGY_BUNDLE_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload.bytes);
    fs::write(path, &out).map_err(EnergyBundleError::Io)
}

/// Read a bundle from `path`, validating magic, schema version,
/// payload integrity, dimensional consistency, and weight validity.
pub fn read_energy_bundle(path: &Path) -> Result<EnergyBundle, EnergyBundleError> {
    let bytes = fs::read(path).map_err(EnergyBundleError::Io)?;
    if bytes.len() < 12 {
        return Err(EnergyBundleError::Malformed("file shorter than header"));
    }
    if &bytes[0..4] != BUNDLE_MAGIC {
        return Err(EnergyBundleError::Malformed("bad bundle magic"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != ENERGY_BUNDLE_SCHEMA_VERSION {
        return Err(EnergyBundleError::Malformed("unsupported bundle version"));
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if 12 + len != bytes.len() {
        return Err(EnergyBundleError::Malformed("payload length mismatch"));
    }
    let payload =
        lossless_decompress_bytes(&bytes[12..]).map_err(EnergyBundleError::Compression)?;
    EnergyBundle::from_canonical_bytes(&payload)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_cana::pinn_energy_v1::{PINN_ENERGY_V1_MODEL_ID, PINN_ENERGY_V1_MODEL_VERSION};

    fn sample_bundle() -> EnergyBundle {
        let pass_names: Vec<String> = ["constant_fold", "dce", "licm", "loop_unroll"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let n = ENERGY_WORKLOAD_FEATURES + pass_names.len() + ENERGY_TAIL_FEATURES;
        EnergyBundle {
            model_id: PINN_ENERGY_V1_MODEL_ID.to_string(),
            model_version: PINN_ENERGY_V1_MODEL_VERSION,
            head: PinnEnergyV1 {
                pass_names,
                feature_means: (0..n).map(|i| 0.1 * i as f64).collect(),
                feature_stds: (0..n).map(|i| 1.0 + 0.05 * i as f64).collect(),
                coefficients: (0..n).map(|i| 0.01 * (i as f64 - 4.0)).collect(),
                intercept: -0.025,
            },
        }
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let b = sample_bundle();
        let back = EnergyBundle::from_canonical_bytes(&b.canonical_bytes()).unwrap();
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
    fn empty_pass_vocabulary_round_trips() {
        let mut b = sample_bundle();
        let n = ENERGY_WORKLOAD_FEATURES + ENERGY_TAIL_FEATURES;
        b.head.pass_names.clear();
        b.head.feature_means = vec![0.0; n];
        b.head.feature_stds = vec![1.0; n];
        b.head.coefficients = vec![0.0; n];
        let back = EnergyBundle::from_canonical_bytes(&b.canonical_bytes()).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn file_round_trip_preserves_bits_and_double_write_is_stable() {
        let dir = std::env::temp_dir().join("cana_energy_bundle_test_rt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("energy.cpb");
        let b = sample_bundle();
        write_energy_bundle(&path, &b).unwrap();
        let back = read_energy_bundle(&path).unwrap();
        assert_eq!(b, back);
        for (a, c) in b.head.coefficients.iter().zip(back.head.coefficients.iter()) {
            assert_eq!(a.to_bits(), c.to_bits());
        }
        let first_bytes = fs::read(&path).unwrap();
        write_energy_bundle(&path, &b).unwrap();
        assert_eq!(first_bytes, fs::read(&path).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dimension_inconsistency_rejected() {
        // feature_count field disagreeing with the vocabulary must be
        // rejected even when the arrays themselves are self-consistent.
        let b = sample_bundle();
        let mut bytes = b.canonical_bytes();
        // The feature_count u32 sits right after the pass names; locate
        // it by re-encoding a modified bundle instead of byte surgery:
        let mut wrong = b.clone();
        wrong.head.pass_names.pop(); // arrays no longer match vocab
        bytes = wrong.canonical_bytes();
        assert!(EnergyBundle::from_canonical_bytes(&bytes).is_err());
    }

    #[test]
    fn invalid_weights_refused_on_write() {
        let dir = std::env::temp_dir().join("cana_energy_bundle_test_invalid");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut b = sample_bundle();
        b.head.feature_stds[2] = 0.0;
        assert!(write_energy_bundle(&dir.join("bad.cpb"), &b).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_file_rejected_not_panicked() {
        let dir = std::env::temp_dir().join("cana_energy_bundle_test_corrupt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("energy.cpb");
        write_energy_bundle(&path, &sample_bundle()).unwrap();
        let mut bytes = fs::read(&path).unwrap();

        let mid = bytes.len() - 4;
        bytes[mid] ^= 0xFF;
        let corrupted = dir.join("corrupted.cpb");
        fs::write(&corrupted, &bytes).unwrap();
        assert!(read_energy_bundle(&corrupted).is_err());

        for cut in [0, 3, 11, bytes.len() / 2] {
            let t = dir.join(format!("trunc_{cut}.cpb"));
            fs::write(&t, &bytes[..cut]).unwrap();
            assert!(read_energy_bundle(&t).is_err(), "cut {cut} must reject");
        }

        let good = fs::read(&path).unwrap();
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        fs::write(dir.join("badmagic.cpb"), &bad_magic).unwrap();
        assert!(read_energy_bundle(&dir.join("badmagic.cpb")).is_err());
        let mut bad_ver = good.clone();
        bad_ver[4] = 0xEE;
        fs::write(dir.join("badver.cpb"), &bad_ver).unwrap();
        assert!(read_energy_bundle(&dir.join("badver.cpb")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn nan_payload_rejected_by_validity_gate() {
        let mut b = sample_bundle();
        b.head.intercept = f64::NAN;
        let bytes = b.canonical_bytes();
        assert!(EnergyBundle::from_canonical_bytes(&bytes).is_err());
    }

    #[test]
    fn adversarial_pass_count_rejected() {
        // A length prefix claiming 2^31 pass names must reject cleanly
        // (and fast), not allocate.
        let b = sample_bundle();
        let mut bytes = b.canonical_bytes();
        // n_passes sits after model_id (len 4+14) + version 4.
        let off = 4 + b.model_id.len() + 4;
        bytes[off..off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(EnergyBundle::from_canonical_bytes(&bytes).is_err());
    }
}
