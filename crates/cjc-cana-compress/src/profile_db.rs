//! Phase A4 — "expected vs actual" compilation-profile database.
//!
//! ## What this is
//!
//! An append-only, deterministic row store for
//! [`CompilationProfile`] records: one row per
//! `(program × ranker-configuration)` experiment, joining to the A3
//! sidecar via `sidecar_bundle_hash`. This is the training-data corpus
//! the PINN v2 (neural) phase trains from, and the substrate for the
//! 5-way ablation comparison (Phase A6).
//!
//! ## Schema corrections vs the Phase-A handoff §5.1
//!
//! The handoff's sketch referenced types that do not exist in this
//! workspace (`VerifierOutcome`, `ParityOutcome`) and placed the module
//! in `cjc-cana` (impossible: the row codec and the sidecar join target
//! live HERE, and `cjc-cana-compress` already depends on `cjc-cana` —
//! the reverse edge would be a cycle). Corrections:
//!
//! - verifier outcome → `legality_approved: bool` +
//!   `legality_violation_count: u32` (from the real `LegalityVerdict`).
//! - parity outcome → `parity_match: Option<bool>` — `None` when the
//!   harness didn't run both executors; `Some(eq)` when it did.
//! - model identity → `cost_model_id: String` + `cost_model_version:
//!   u32` captured from `CostModel::name()/version()` at rank time.
//! - hashes → plain `u64` (the workspace's `ProgramHash`/`FeatureHash`
//!   newtypes wrap `u64`; rows store the raw value to keep this module
//!   free of cross-crate type churn).
//!
//! ## Determinism contract
//!
//! - [`CompilationProfile::canonical_bytes`] is a fixed-order
//!   little-endian encoding; same row → byte-identical encoding.
//! - [`CompilationProfile::row_hash`] is FNV-1a over the canonical
//!   bytes **excluding diagnostic fields** (`compile_wall_micros`) —
//!   wall-clock is stored for human analysis but two runs of the same
//!   deterministic experiment produce the same `row_hash` even though
//!   their wall-clocks differ (determinism invariant #7).
//! - The on-disk format is a header + length-prefixed
//!   losslessly-compressed rows; readers verify magic, version, and
//!   per-row integrity (the RLE payload embeds its own input hash).
//! - Appending is strictly ordered: rows read back in append order.

use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::Path;

use cjc_cana::hash::hash_bytes;

use crate::candidate::CompressionError;
use crate::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};

/// On-disk magic for a profile-DB file.
const FILE_MAGIC: &[u8; 4] = b"CPDB";
/// On-disk magic for a single encoded row (inside the compressed
/// payload).
const ROW_MAGIC: &[u8; 4] = b"CPR0";
/// Bump on incompatible schema changes.
///
/// v2 (PINN v2 §2.2): added `estimated_float_ops` — the §2.1
/// data-sanity pass proved the v1 feature set was type-blind, making
/// recorded thermal (FP-op density) unpredictable by ANY model class.
/// The corpus regenerates deterministically in ~35 s, so no v1→v2
/// migration path is provided; v1 files are rejected on read.
pub const PROFILE_SCHEMA_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// CompilationProfile — one experiment row
// ---------------------------------------------------------------------------

/// One `(program × configuration)` experiment record.
///
/// Every field is either a deterministic compile-time fact, a
/// deterministic prediction, or an explicitly-diagnostic measurement
/// (see [`Self::row_hash`] for which is which).
#[derive(Debug, Clone, PartialEq)]
pub struct CompilationProfile {
    /// Schema version stamped at construction.
    pub schema_version: u32,
    /// Human-readable program name (bench corpus key).
    pub program_name: String,
    /// `ProgramHash.0` from `cjc_cana::features::extract`.
    pub program_hash: u64,
    /// `FeatureHash.0` from the same extraction.
    pub feature_hash: u64,
    /// `CompressedCanaSidecar::bundle_hash` join key; `0` when no
    /// sidecar was produced for this experiment.
    pub sidecar_bundle_hash: u64,
    /// Ablation configuration id (e.g. `"baseline"`, `"nss"`,
    /// `"quantum"`, `"nss_quantum"`, `"full_pinn"`).
    pub config_id: String,
    /// `CostModel::name()` of the model that drove ranking.
    pub cost_model_id: String,
    /// `CostModel::version()` of the same.
    pub cost_model_version: u32,
    /// Per-function pass sequences, sorted by function name. Flattened
    /// as `(function, passes)` pairs to keep the codec simple.
    pub pass_sequence: Vec<(String, Vec<String>)>,

    // -- Workload estimates (deterministic; from the physical layer) --
    /// Sum of per-function `flops_estimate` (saturating).
    pub estimated_flops: u64,
    /// Sum of per-function `bytes_read_estimate` (saturating).
    pub estimated_bytes_read: u64,
    /// Sum of per-function `bytes_written_estimate` (saturating).
    pub estimated_bytes_written: u64,
    /// Sum of per-function `allocation_bytes_estimate` (saturating).
    pub estimated_alloc_bytes: u64,
    /// Sum of per-function `working_set_bytes_estimate` (saturating).
    pub estimated_working_set: u64,
    /// Sum of per-function `float_ops_estimate` (saturating). Schema
    /// v2: the FP-density signal (`estimated_float_ops /
    /// estimated_flops`) the thermal head trains on.
    pub estimated_float_ops: u64,

    // -- Predictions (deterministic; recorded at rank time) --
    /// Max per-function NSS CPU saturation.
    pub nss_predicted_cpu_max: f64,
    /// Max per-function NSS memory peak.
    pub nss_predicted_memory_max: f64,
    /// Max per-function NSS thermal pressure.
    pub nss_predicted_thermal_max: f64,
    /// Max per-function PINN energy estimate.
    pub pinn_predicted_energy_max: f64,
    /// Max per-function PINN thermal pressure.
    pub pinn_predicted_thermal_max: f64,
    /// Max per-function PINN bandwidth pressure.
    pub pinn_predicted_bandwidth_max: f64,

    // -- Observed (deterministic counters) --
    /// MIR expression-node count before optimization.
    pub mir_nodes_before: u64,
    /// MIR expression-node count after applying the chosen plan.
    pub mir_nodes_after: u64,
    /// Total recommendations across functions.
    pub recommended_count: u32,
    /// Total dropped across functions.
    pub dropped_count: u32,

    // -- Outcomes --
    /// `LegalityVerdict` collapsed to its approval bit.
    pub legality_approved: bool,
    /// Number of legality violations recorded (0 when approved).
    pub legality_violation_count: u32,
    /// AST-eval vs MIR-exec output equality, when the harness ran
    /// both. `None` = not measured.
    pub parity_match: Option<bool>,

    // -- Diagnostics (EXCLUDED from row_hash) --
    /// Wall-clock of the compile+rank phase, microseconds. Diagnostic
    /// only; never feeds a decision path.
    pub compile_wall_micros: u64,

    // -- Final --
    /// Deterministic experiment score (lower = better), as defined by
    /// the emitting harness. For the A6 v3 ablation this is the
    /// baseline-relative modeled energy of the optimized run
    /// (`raw_energy / baseline_config_energy`); earlier harness
    /// versions stored the post-optimization MIR-size ratio.
    pub score: f64,
}

impl CompilationProfile {
    /// Canonical fixed-order encoding of the full row (including
    /// diagnostics).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(ROW_MAGIC);
        push_u32(&mut out, self.schema_version);
        push_str(&mut out, &self.program_name);
        push_u64(&mut out, self.program_hash);
        push_u64(&mut out, self.feature_hash);
        push_u64(&mut out, self.sidecar_bundle_hash);
        push_str(&mut out, &self.config_id);
        push_str(&mut out, &self.cost_model_id);
        push_u32(&mut out, self.cost_model_version);
        push_u32(&mut out, self.pass_sequence.len() as u32);
        for (func, passes) in &self.pass_sequence {
            push_str(&mut out, func);
            push_u32(&mut out, passes.len() as u32);
            for p in passes {
                push_str(&mut out, p);
            }
        }
        push_u64(&mut out, self.estimated_flops);
        push_u64(&mut out, self.estimated_bytes_read);
        push_u64(&mut out, self.estimated_bytes_written);
        push_u64(&mut out, self.estimated_alloc_bytes);
        push_u64(&mut out, self.estimated_working_set);
        push_u64(&mut out, self.estimated_float_ops);
        push_f64(&mut out, self.nss_predicted_cpu_max);
        push_f64(&mut out, self.nss_predicted_memory_max);
        push_f64(&mut out, self.nss_predicted_thermal_max);
        push_f64(&mut out, self.pinn_predicted_energy_max);
        push_f64(&mut out, self.pinn_predicted_thermal_max);
        push_f64(&mut out, self.pinn_predicted_bandwidth_max);
        push_u64(&mut out, self.mir_nodes_before);
        push_u64(&mut out, self.mir_nodes_after);
        push_u32(&mut out, self.recommended_count);
        push_u32(&mut out, self.dropped_count);
        out.push(self.legality_approved as u8);
        push_u32(&mut out, self.legality_violation_count);
        match self.parity_match {
            None => out.push(0),
            Some(false) => out.push(1),
            Some(true) => out.push(2),
        }
        push_u64(&mut out, self.compile_wall_micros);
        push_f64(&mut out, self.score);
        out
    }

    /// Stable content hash of the row, **excluding diagnostics**.
    ///
    /// Two runs of the same deterministic experiment produce the same
    /// `row_hash` even though their `compile_wall_micros` differ. This
    /// is the §5.3 metric-11 canary ("report hash stability") at row
    /// granularity.
    pub fn row_hash(&self) -> u64 {
        let mut stable = self.clone();
        stable.compile_wall_micros = 0;
        hash_bytes(&stable.canonical_bytes())
    }

    /// Decode a row from its canonical encoding.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CompressionError> {
        let mut r = Reader::new(bytes);
        r.expect_magic(ROW_MAGIC)?;
        let schema_version = r.read_u32()?;
        if schema_version != PROFILE_SCHEMA_VERSION {
            return Err(CompressionError::MalformedPayload {
                at_byte: 4,
                reason: "unsupported profile schema version",
            });
        }
        let program_name = r.read_str()?;
        let program_hash = r.read_u64()?;
        let feature_hash = r.read_u64()?;
        let sidecar_bundle_hash = r.read_u64()?;
        let config_id = r.read_str()?;
        let cost_model_id = r.read_str()?;
        let cost_model_version = r.read_u32()?;
        let n_fns = r.read_u32()? as usize;
        let mut pass_sequence = Vec::with_capacity(n_fns.min(1024));
        for _ in 0..n_fns {
            let func = r.read_str()?;
            let n_passes = r.read_u32()? as usize;
            let mut passes = Vec::with_capacity(n_passes.min(64));
            for _ in 0..n_passes {
                passes.push(r.read_str()?);
            }
            pass_sequence.push((func, passes));
        }
        let estimated_flops = r.read_u64()?;
        let estimated_bytes_read = r.read_u64()?;
        let estimated_bytes_written = r.read_u64()?;
        let estimated_alloc_bytes = r.read_u64()?;
        let estimated_working_set = r.read_u64()?;
        let estimated_float_ops = r.read_u64()?;
        let nss_predicted_cpu_max = r.read_f64()?;
        let nss_predicted_memory_max = r.read_f64()?;
        let nss_predicted_thermal_max = r.read_f64()?;
        let pinn_predicted_energy_max = r.read_f64()?;
        let pinn_predicted_thermal_max = r.read_f64()?;
        let pinn_predicted_bandwidth_max = r.read_f64()?;
        let mir_nodes_before = r.read_u64()?;
        let mir_nodes_after = r.read_u64()?;
        let recommended_count = r.read_u32()?;
        let dropped_count = r.read_u32()?;
        let legality_approved = r.read_u8()? != 0;
        let legality_violation_count = r.read_u32()?;
        let parity_match = match r.read_u8()? {
            0 => None,
            1 => Some(false),
            2 => Some(true),
            _ => {
                return Err(CompressionError::MalformedPayload {
                    at_byte: r.pos,
                    reason: "invalid parity_match tag",
                })
            }
        };
        let compile_wall_micros = r.read_u64()?;
        let score = r.read_f64()?;
        if !r.is_exhausted() {
            return Err(CompressionError::MalformedPayload {
                at_byte: r.pos,
                reason: "trailing bytes after profile row",
            });
        }
        Ok(Self {
            schema_version,
            program_name,
            program_hash,
            feature_hash,
            sidecar_bundle_hash,
            config_id,
            cost_model_id,
            cost_model_version,
            pass_sequence,
            estimated_flops,
            estimated_bytes_read,
            estimated_bytes_written,
            estimated_alloc_bytes,
            estimated_working_set,
            estimated_float_ops,
            nss_predicted_cpu_max,
            nss_predicted_memory_max,
            nss_predicted_thermal_max,
            pinn_predicted_energy_max,
            pinn_predicted_thermal_max,
            pinn_predicted_bandwidth_max,
            mir_nodes_before,
            mir_nodes_after,
            recommended_count,
            dropped_count,
            legality_approved,
            legality_violation_count,
            parity_match,
            compile_wall_micros,
            score,
        })
    }
}

// ---------------------------------------------------------------------------
// File-level append/read
// ---------------------------------------------------------------------------

/// Append one row to the profile DB at `path`, creating the file (with
/// header) if absent.
///
/// Row bytes are losslessly compressed (the payload embeds its own
/// input hash, giving per-row integrity on read-back) and
/// length-prefixed, so the file supports streaming append without
/// rewriting.
pub fn append_row(path: &Path, row: &CompilationProfile) -> io::Result<()> {
    let exists = path.exists();
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    if !exists {
        f.write_all(FILE_MAGIC)?;
        f.write_all(&PROFILE_SCHEMA_VERSION.to_le_bytes())?;
    }
    let payload = lossless_compress_bytes(&row.canonical_bytes());
    f.write_all(&(payload.bytes.len() as u32).to_le_bytes())?;
    f.write_all(&payload.bytes)?;
    Ok(())
}

/// Read every row from the profile DB at `path`, in append order.
pub fn read_all(path: &Path) -> Result<Vec<CompilationProfile>, ProfileDbError> {
    let bytes = fs::read(path).map_err(ProfileDbError::Io)?;
    if bytes.len() < 8 {
        return Err(ProfileDbError::Malformed("file shorter than header"));
    }
    if &bytes[0..4] != FILE_MAGIC {
        return Err(ProfileDbError::Malformed("bad file magic"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != PROFILE_SCHEMA_VERSION {
        return Err(ProfileDbError::Malformed("unsupported file version"));
    }
    let mut rows = Vec::new();
    let mut pos = 8usize;
    while pos < bytes.len() {
        if pos + 4 > bytes.len() {
            return Err(ProfileDbError::Malformed("truncated row length prefix"));
        }
        let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > bytes.len() {
            return Err(ProfileDbError::Malformed("truncated row payload"));
        }
        let row_bytes = lossless_decompress_bytes(&bytes[pos..pos + len])
            .map_err(ProfileDbError::Compression)?;
        rows.push(
            CompilationProfile::from_canonical_bytes(&row_bytes)
                .map_err(ProfileDbError::Compression)?,
        );
        pos += len;
    }
    Ok(rows)
}

/// Errors from the file-level profile DB API.
#[derive(Debug)]
pub enum ProfileDbError {
    /// Filesystem error.
    Io(io::Error),
    /// Structural error in the file framing.
    Malformed(&'static str),
    /// Row payload failed decompression or decoding.
    Compression(CompressionError),
}

impl std::fmt::Display for ProfileDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileDbError::Io(e) => write!(f, "profile db io error: {e}"),
            ProfileDbError::Malformed(m) => write!(f, "profile db malformed: {m}"),
            ProfileDbError::Compression(e) => write!(f, "profile db row error: {e}"),
        }
    }
}

impl std::error::Error for ProfileDbError {}

// ---------------------------------------------------------------------------
// Little-endian push/read helpers
// ---------------------------------------------------------------------------

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], CompressionError> {
        if self.pos + n > self.bytes.len() {
            return Err(CompressionError::MalformedPayload {
                at_byte: self.pos,
                reason: "row truncated",
            });
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn expect_magic(&mut self, magic: &[u8; 4]) -> Result<(), CompressionError> {
        if self.take(4)? != magic {
            return Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "bad row magic",
            });
        }
        Ok(())
    }
    fn read_u8(&mut self) -> Result<u8, CompressionError> {
        Ok(self.take(1)?[0])
    }
    fn read_u32(&mut self) -> Result<u32, CompressionError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> Result<u64, CompressionError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64, CompressionError> {
        Ok(f64::from_bits(self.read_u64()?))
    }
    fn read_str(&mut self) -> Result<String, CompressionError> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| CompressionError::MalformedPayload {
            at_byte: self.pos,
            reason: "non-utf8 string in profile row",
        })
    }
    fn is_exhausted(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> CompilationProfile {
        CompilationProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            program_name: "arith".to_string(),
            program_hash: 0xDEAD_BEEF,
            feature_hash: 0xCAFE_F00D,
            sidecar_bundle_hash: 42,
            config_id: "full_pinn".to_string(),
            cost_model_id: "pinn_coeffs_v1".to_string(),
            cost_model_version: 1,
            pass_sequence: vec![
                (
                    "compute".to_string(),
                    vec!["constant_fold".to_string(), "dce".to_string()],
                ),
                ("main".to_string(), vec![]),
            ],
            estimated_flops: 1_000_000,
            estimated_bytes_read: 8_000_000,
            estimated_bytes_written: 2_000_000,
            estimated_alloc_bytes: 64_000,
            estimated_working_set: 128_000,
            estimated_float_ops: 250_000,
            nss_predicted_cpu_max: 0.5,
            nss_predicted_memory_max: 0.25,
            nss_predicted_thermal_max: 0.75,
            pinn_predicted_energy_max: 0.4,
            pinn_predicted_thermal_max: 0.3,
            pinn_predicted_bandwidth_max: 0.2,
            mir_nodes_before: 120,
            mir_nodes_after: 95,
            recommended_count: 4,
            dropped_count: 2,
            legality_approved: true,
            legality_violation_count: 0,
            parity_match: Some(true),
            compile_wall_micros: 1234,
            score: 0.7916,
        }
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let row = sample_row();
        let bytes = row.canonical_bytes();
        let back = CompilationProfile::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(row, back);
    }

    #[test]
    fn parity_tristate_round_trips() {
        for parity in [None, Some(false), Some(true)] {
            let mut row = sample_row();
            row.parity_match = parity;
            let back = CompilationProfile::from_canonical_bytes(&row.canonical_bytes()).unwrap();
            assert_eq!(back.parity_match, parity);
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let row = sample_row();
        let first = row.canonical_bytes();
        for _ in 0..20 {
            assert_eq!(first, row.canonical_bytes());
        }
    }

    #[test]
    fn row_hash_ignores_wall_clock_but_not_score() {
        let row = sample_row();
        let mut other_wall = row.clone();
        other_wall.compile_wall_micros = 999_999;
        assert_eq!(
            row.row_hash(),
            other_wall.row_hash(),
            "wall-clock is diagnostic; must not perturb the stable hash"
        );
        let mut other_score = row.clone();
        other_score.score = 0.5;
        assert_ne!(
            row.row_hash(),
            other_score.row_hash(),
            "score is a deterministic output; must perturb the stable hash"
        );
    }

    #[test]
    fn row_hash_sensitive_to_config_and_model() {
        let row = sample_row();
        let mut other = row.clone();
        other.config_id = "baseline".to_string();
        assert_ne!(row.row_hash(), other.row_hash());
        let mut other2 = row.clone();
        other2.cost_model_version = 2;
        assert_ne!(row.row_hash(), other2.row_hash());
    }

    #[test]
    fn truncated_row_rejected_not_panicked() {
        let bytes = sample_row().canonical_bytes();
        for cut in [0, 3, 10, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                CompilationProfile::from_canonical_bytes(&bytes[..cut]).is_err(),
                "cut at {cut} must be rejected"
            );
        }
    }

    #[test]
    fn trailing_bytes_rejected() {
        let mut bytes = sample_row().canonical_bytes();
        bytes.push(0);
        assert!(CompilationProfile::from_canonical_bytes(&bytes).is_err());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = sample_row().canonical_bytes();
        bytes[0] = b'X';
        assert!(CompilationProfile::from_canonical_bytes(&bytes).is_err());
    }

    #[test]
    fn file_append_and_read_back_in_order() {
        let dir = std::env::temp_dir().join("cana_profile_db_test_append");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("profiles.cpdb");

        let mut rows = Vec::new();
        for i in 0..5u32 {
            let mut row = sample_row();
            row.program_name = format!("prog_{i}");
            row.recommended_count = i;
            append_row(&path, &row).unwrap();
            rows.push(row);
        }
        let back = read_all(&path).unwrap();
        assert_eq!(back.len(), 5);
        for (a, b) in rows.iter().zip(back.iter()) {
            assert_eq!(a, b);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_file_rejected_not_panicked() {
        let dir = std::env::temp_dir().join("cana_profile_db_test_corrupt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("profiles.cpdb");
        append_row(&path, &sample_row()).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        // Flip a byte inside the compressed row body.
        let mid = bytes.len() - 4;
        bytes[mid] ^= 0xFF;
        let corrupted = dir.join("corrupted.cpdb");
        fs::write(&corrupted, &bytes).unwrap();
        assert!(read_all(&corrupted).is_err());

        // Truncated file.
        let truncated = dir.join("truncated.cpdb");
        fs::write(&truncated, &bytes[..bytes.len() - 10]).unwrap();
        assert!(read_all(&truncated).is_err());

        // Bad header.
        let bad = dir.join("bad.cpdb");
        fs::write(&bad, b"NOPE").unwrap();
        assert!(read_all(&bad).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_db_file_reads_as_zero_rows() {
        let dir = std::env::temp_dir().join("cana_profile_db_test_empty");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("profiles.cpdb");
        // Create header-only file by appending then... simpler: write
        // header manually via the public constants.
        fs::write(
            &path,
            [&FILE_MAGIC[..], &PROFILE_SCHEMA_VERSION.to_le_bytes()].concat(),
        )
        .unwrap();
        let rows = read_all(&path).unwrap();
        assert!(rows.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
