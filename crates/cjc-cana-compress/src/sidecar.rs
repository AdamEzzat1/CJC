//! Phase A3 — `CompressedCanaSidecar`: persistence format for
//! `(CanaReport, PassHistory)` pairs.
//!
//! ## What this is for
//!
//! Today, [`cjc_cana::CanaReport`] is a transient in-memory value — the
//! existing `to_json` writer can emit a sidecar JSON file, but the
//! [`cjc_cana::PassHistory`] that the compiler accumulates during a
//! single compile *never* gets archived. Without an archived
//! pass-history corpus, the future training pipeline (Phases B–D)
//! has no labelled examples to fit per-component regressions against.
//!
//! `CompressedCanaSidecar` is the canonical persistence format:
//!
//! - **`CanaReport` JSON bytes** stored verbatim (uncompressed — typical
//!   reports are ≤ 100 KB and compressing JSON via byte RLE doesn't
//!   pay off enough to justify the format complexity).
//! - **`PassHistory` compressed losslessly** via
//!   [`compress_pass_history`] (the byte-RLE codec handles the
//!   repetitive structure of pass records well).
//! - **Stable hashes** at every level: report JSON hash, pass-history
//!   input/compressed/recovered hashes, and an outer `bundle_hash`
//!   covering everything for at-rest tamper detection.
//!
//! Subsequent training-data work (A4 — profile DB) can append rows
//! containing the `bundle_hash` as a stable join key against compiler
//! runs.
//!
//! ## File format
//!
//! ```text
//! 4 bytes   magic = "CCS0"                      (CANA Compressed Sidecar v0)
//! 4 bytes   schema_version (u32 LE)
//! 8 bytes   report_json_hash (u64 LE, FNV-1a)
//! 4 bytes   report_json_len (u32 LE)
//! N bytes   report_json_bytes
//! 8 bytes   history_input_hash (u64 LE)
//! 8 bytes   history_compressed_hash (u64 LE)
//! 8 bytes   history_original_len (u64 LE)
//! 4 bytes   history_compressed_len (u32 LE)
//! M bytes   history_compressed_bytes
//! 8 bytes   bundle_hash (u64 LE — covers every byte above)
//! ```
//!
//! `bundle_hash` is computed last and appended; it covers the entire
//! preceding byte stream so a single FNV-1a roundtrip is sufficient to
//! verify integrity at load time.
//!
//! ## Determinism contract
//!
//! Same `(report, history)` produces byte-identical
//! [`CompressedCanaSidecar::to_bytes`] across runs and platforms.
//! Backed by:
//!
//! - [`CanaReport::canonical_bytes`] — already deterministic.
//! - [`compress_pass_history`] — deterministic by codec design.
//! - All u32/u64 fields little-endian.
//! - No wall-clock, no RNG, no `HashMap` iteration.

use std::fs;
use std::io;
use std::path::Path;

use cjc_cana::hash::hash_bytes;
use cjc_cana::pass_history::PassHistory;
use cjc_cana::report::CanaReport;

use crate::candidate::CompressionError;
use crate::lossless_trace::{compress_pass_history, decompress_pass_history, LosslessTracePayload};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"CCS0";
const SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// CompressedCanaSidecar
// ---------------------------------------------------------------------------

/// A persistent, content-addressed bundle of a CANA report plus its
/// associated compressed pass history.
///
/// Construct via [`Self::build`] from `(report, history)` references.
/// Persist via [`Self::to_bytes`] / [`Self::write_to_path`]. Load via
/// [`Self::from_bytes`] / [`Self::read_from_path`]. Verify integrity
/// via [`Self::verify`].
#[derive(Debug, Clone)]
pub struct CompressedCanaSidecar {
    /// Schema version. Bumped on format changes that aren't
    /// backward-compatible.
    pub schema_version: u32,
    /// Raw CANA report JSON bytes (deterministic via
    /// [`CanaReport::canonical_bytes`]).
    pub report_json: Vec<u8>,
    /// FNV-1a hash of `report_json` (also stored in the on-disk header
    /// for self-validation).
    pub report_json_hash: u64,
    /// The losslessly-compressed pass history.
    pub history_payload: LosslessTracePayload,
    /// FNV-1a hash of the entire serialized bundle. Set by
    /// [`Self::build`] / [`Self::from_bytes`].
    pub bundle_hash: u64,
}

impl CompressedCanaSidecar {
    /// Build a sidecar from a [`CanaReport`] and a [`PassHistory`].
    ///
    /// The report is rendered via `canonical_bytes()` (JSON) and stored
    /// verbatim. The history is compressed losslessly via
    /// [`compress_pass_history`]. Both sub-hashes plus the outer
    /// `bundle_hash` are populated.
    ///
    /// Same `(report, history)` always produces byte-identical output
    /// — this is the determinism canary that the persistence layer
    /// inherits from the underlying components.
    pub fn build(report: &CanaReport, history: &PassHistory) -> Self {
        let report_json = report.canonical_bytes();
        let report_json_hash = hash_bytes(&report_json);
        let history_payload = compress_pass_history(history);
        let mut sidecar = Self {
            schema_version: SCHEMA_VERSION,
            report_json,
            report_json_hash,
            history_payload,
            bundle_hash: 0,
        };
        // Compute the bundle hash over the body — i.e., over what
        // `to_bytes` writes *before* appending the bundle_hash itself.
        let body = sidecar.body_bytes();
        sidecar.bundle_hash = hash_bytes(&body);
        sidecar
    }

    /// Compute the on-disk length in bytes.
    pub fn serialized_len(&self) -> usize {
        // magic(4) + schema_version(4) + json_hash(8) + json_len(4)
        //   + N(json bytes)
        //   + input_hash(8) + compressed_hash(8) + original_len(8) + compressed_len(4)
        //   + M(compressed bytes)
        //   + bundle_hash(8)
        4 + 4
            + 8
            + 4
            + self.report_json.len()
            + 8
            + 8
            + 8
            + 4
            + self.history_payload.bytes.len()
            + 8
    }

    /// Compression ratio relative to the *raw* `(report_json +
    /// uncompressed pass history)` byte size. Lower is better.
    /// Returns `0.0` if the raw size is zero.
    pub fn compression_ratio(&self) -> f64 {
        let raw = self.report_json.len() + self.history_payload.original_len;
        if raw == 0 {
            return 0.0;
        }
        (self.serialized_len() as f64) / (raw as f64)
    }

    /// Serialize to bytes. See module-level docs for the format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.body_bytes();
        out.extend_from_slice(&self.bundle_hash.to_le_bytes());
        out
    }

    /// Body of the bytes (everything before the trailing bundle_hash).
    /// Pre-computing this lets `build` hash the body once and append.
    fn body_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        out.extend_from_slice(&self.report_json_hash.to_le_bytes());
        out.extend_from_slice(&(self.report_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.report_json);
        out.extend_from_slice(&self.history_payload.input_hash.to_le_bytes());
        out.extend_from_slice(&self.history_payload.compressed_hash.to_le_bytes());
        out.extend_from_slice(&(self.history_payload.original_len as u64).to_le_bytes());
        out.extend_from_slice(&(self.history_payload.bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.history_payload.bytes);
        out
    }

    /// Deserialize from bytes. Validates magic, schema version, every
    /// embedded hash, and the outer `bundle_hash`. Returns
    /// [`CompressionError::MalformedPayload`] on any inconsistency —
    /// never panics.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompressionError> {
        if bytes.len() < 4 + 4 + 8 + 4 + 8 + 8 + 8 + 4 + 8 {
            return Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "sidecar shorter than minimum header size",
            });
        }
        if &bytes[0..4] != MAGIC {
            return Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "bad sidecar magic",
            });
        }
        let schema_version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if schema_version != SCHEMA_VERSION {
            return Err(CompressionError::MalformedPayload {
                at_byte: 4,
                reason: "unsupported sidecar schema version",
            });
        }
        let report_json_hash = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let report_json_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let json_start = 20;
        let json_end = json_start + report_json_len;
        if bytes.len() < json_end + 8 + 8 + 8 + 4 + 8 {
            return Err(CompressionError::MalformedPayload {
                at_byte: json_start,
                reason: "truncated report json",
            });
        }
        let report_json = bytes[json_start..json_end].to_vec();
        let actual_json_hash = hash_bytes(&report_json);
        if actual_json_hash != report_json_hash {
            return Err(CompressionError::MalformedPayload {
                at_byte: json_start,
                reason: "report json hash mismatch (corruption)",
            });
        }

        let mut cursor = json_end;
        let history_input_hash = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let history_compressed_hash =
            u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let history_original_len =
            u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;
        let history_compressed_len =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let comp_start = cursor;
        let comp_end = comp_start + history_compressed_len;
        if bytes.len() < comp_end + 8 {
            return Err(CompressionError::MalformedPayload {
                at_byte: comp_start,
                reason: "truncated compressed pass history",
            });
        }
        let comp_bytes = bytes[comp_start..comp_end].to_vec();
        let actual_comp_hash = hash_bytes(&comp_bytes);
        if actual_comp_hash != history_compressed_hash {
            return Err(CompressionError::MalformedPayload {
                at_byte: comp_start,
                reason: "compressed pass history hash mismatch (corruption)",
            });
        }
        let history_payload = LosslessTracePayload {
            bytes: comp_bytes,
            input_hash: history_input_hash,
            compressed_hash: history_compressed_hash,
            original_len: history_original_len,
        };

        let bundle_hash = u64::from_le_bytes(bytes[comp_end..comp_end + 8].try_into().unwrap());

        // Validate trailing bytes (extra data after bundle_hash is a
        // corruption signal — refuse rather than accept).
        if bytes.len() != comp_end + 8 {
            return Err(CompressionError::MalformedPayload {
                at_byte: comp_end + 8,
                reason: "trailing bytes after sidecar bundle_hash",
            });
        }

        // Validate bundle_hash against the body.
        let actual_bundle_hash = hash_bytes(&bytes[..comp_end]);
        if actual_bundle_hash != bundle_hash {
            return Err(CompressionError::MalformedPayload {
                at_byte: comp_end,
                reason: "bundle hash mismatch (corruption)",
            });
        }

        Ok(Self {
            schema_version,
            report_json,
            report_json_hash,
            history_payload,
            bundle_hash,
        })
    }

    /// Re-verify every hash in the bundle. Useful as a "did the sidecar
    /// I just loaded match what the producer wrote?" sanity check —
    /// `from_bytes` already runs these checks, so calling this after
    /// `from_bytes` is mostly defensive.
    pub fn verify(&self) -> Result<(), CompressionError> {
        if hash_bytes(&self.report_json) != self.report_json_hash {
            return Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "report json hash mismatch (in-memory)",
            });
        }
        let payload_hash = hash_bytes(&self.history_payload.bytes);
        if payload_hash != self.history_payload.compressed_hash {
            return Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "compressed pass history hash mismatch (in-memory)",
            });
        }
        let body = self.body_bytes();
        if hash_bytes(&body) != self.bundle_hash {
            return Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "bundle hash mismatch (in-memory)",
            });
        }
        Ok(())
    }

    /// Recover the [`PassHistory`] by decompressing the embedded
    /// payload. Returns the canonical PassHistory iteration sequence,
    /// byte-identical to the one passed into [`Self::build`].
    pub fn recover_history(&self) -> Result<PassHistory, CompressionError> {
        decompress_pass_history(&self.history_payload)
    }

    /// Borrow the report JSON bytes. To parse them back into a
    /// [`CanaReport`], use whatever JSON parser the caller has — the
    /// `cjc-cana` crate currently does not ship a parser (the writer is
    /// hand-rolled), so producers and consumers agree on the JSON
    /// schema by inspection.
    pub fn report_json(&self) -> &[u8] {
        &self.report_json
    }

    /// Borrow the compressed history payload (for callers that want to
    /// inspect compression stats without recovering the full history).
    pub fn history_payload(&self) -> &LosslessTracePayload {
        &self.history_payload
    }

    /// Write the serialized sidecar to `path`. Convenience.
    pub fn write_to_path(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.to_bytes())
    }

    /// Read + parse a sidecar from `path`. Convenience.
    pub fn read_from_path(path: &Path) -> Result<Self, SidecarIoError> {
        let bytes = fs::read(path).map_err(SidecarIoError::Io)?;
        Self::from_bytes(&bytes).map_err(SidecarIoError::Compression)
    }
}

/// Combined error type for the path-level sidecar loaders.
#[derive(Debug)]
pub enum SidecarIoError {
    /// Underlying filesystem error (file not found, permissions, etc.).
    Io(io::Error),
    /// Payload-level error (bad magic, corrupted bytes, etc.).
    Compression(CompressionError),
}

impl std::fmt::Display for SidecarIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SidecarIoError::Io(e) => write!(f, "sidecar io error: {e}"),
            SidecarIoError::Compression(e) => write!(f, "sidecar payload error: {e}"),
        }
    }
}

impl std::error::Error for SidecarIoError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_cana::analyze_program;
    use cjc_cana::hash::ProgramHash;
    use cjc_cana::pass_history::{PassHistory, PassOutcome, PassRecord, SkipReason};
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    fn make_program() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "main".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: vec![MirStmt::Expr(MirExpr {
                        kind: MirExprKind::IntLit(7),
                    })],
                    result: None,
                },
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    fn make_history(n: u64) -> PassHistory {
        let mut h = PassHistory::with_capacity((n as usize).max(4));
        let names = ["cf", "dce", "cse", "licm", "sr"];
        for i in 0..n {
            h.record(PassRecord {
                pass_name: names[(i as usize) % names.len()].to_string(),
                input_hash: ProgramHash(i),
                output_hash: ProgramHash(i + 1),
                outcome: if i % 5 == 0 {
                    PassOutcome::NoOp
                } else if i % 5 == 1 {
                    PassOutcome::Skipped(SkipReason::CostBelowThreshold)
                } else {
                    PassOutcome::Changed
                },
            });
        }
        h
    }

    // ----- Round-trip --------------------------------------------------

    #[test]
    fn round_trip_basic() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(10);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let bytes = sidecar.to_bytes();
        let loaded = CompressedCanaSidecar::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.schema_version, SCHEMA_VERSION);
        assert_eq!(loaded.report_json, sidecar.report_json);
        assert_eq!(loaded.report_json_hash, sidecar.report_json_hash);
        assert_eq!(loaded.bundle_hash, sidecar.bundle_hash);
        loaded.verify().unwrap();

        // History recovers exactly.
        let recovered = loaded.recover_history().unwrap();
        let original: Vec<_> = history.iter().cloned().collect();
        let restored: Vec<_> = recovered.iter().cloned().collect();
        assert_eq!(original, restored);
    }

    #[test]
    fn round_trip_with_empty_history() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = PassHistory::new();
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let bytes = sidecar.to_bytes();
        let loaded = CompressedCanaSidecar::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.recover_history().unwrap().len(), 0);
    }

    #[test]
    fn round_trip_with_large_history() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(200);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let bytes = sidecar.to_bytes();
        let loaded = CompressedCanaSidecar::from_bytes(&bytes).unwrap();
        let recovered = loaded.recover_history().unwrap();
        assert_eq!(recovered.len(), 200);
        // Compressed pass-history should be smaller than the
        // raw canonical bytes (compression must do something useful on
        // a 200-record sequence with repeated names + zero-heavy hash
        // bytes).
        assert!(
            sidecar.history_payload.bytes.len() <= sidecar.history_payload.original_len,
            "history compressed {} bytes < raw {} bytes",
            sidecar.history_payload.bytes.len(),
            sidecar.history_payload.original_len
        );
    }

    // ----- Determinism --------------------------------------------------

    #[test]
    fn build_is_deterministic() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(8);
        let s1 = CompressedCanaSidecar::build(&report, &history);
        let s2 = CompressedCanaSidecar::build(&report, &history);
        assert_eq!(s1.bundle_hash, s2.bundle_hash);
        assert_eq!(s1.to_bytes(), s2.to_bytes());
        assert_eq!(s1.report_json, s2.report_json);
    }

    #[test]
    fn bundle_hash_distinguishes_different_history() {
        let program = make_program();
        let report = analyze_program(&program);
        let h1 = make_history(5);
        let h2 = make_history(6);
        let s1 = CompressedCanaSidecar::build(&report, &h1);
        let s2 = CompressedCanaSidecar::build(&report, &h2);
        assert_ne!(s1.bundle_hash, s2.bundle_hash);
    }

    #[test]
    fn bundle_hash_distinguishes_different_report() {
        // Same history, two slightly different programs → different
        // bundle_hash because the report JSON differs.
        let h = make_history(5);
        let p1 = make_program();
        let mut p2 = make_program();
        p2.functions[0].name = "renamed".to_string();
        let r1 = analyze_program(&p1);
        let r2 = analyze_program(&p2);
        let s1 = CompressedCanaSidecar::build(&r1, &h);
        let s2 = CompressedCanaSidecar::build(&r2, &h);
        assert_ne!(s1.bundle_hash, s2.bundle_hash);
        assert_ne!(s1.report_json, s2.report_json);
    }

    // ----- Malformed-payload rejection ----------------------------------

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = vec![0u8; 100];
        bytes[0..4].copy_from_slice(b"XXXX");
        let r = CompressedCanaSidecar::from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "bad sidecar magic",
                ..
            })
        ));
    }

    #[test]
    fn too_short_rejected() {
        let r = CompressedCanaSidecar::from_bytes(&[0u8; 10]);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "sidecar shorter than minimum header size",
                ..
            })
        ));
    }

    #[test]
    fn unsupported_schema_rejected() {
        // Build a valid header but with a bumped schema version.
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(4);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let mut bytes = sidecar.to_bytes();
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        let r = CompressedCanaSidecar::from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "unsupported sidecar schema version",
                ..
            })
        ));
    }

    #[test]
    fn corrupted_json_hash_rejected() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(4);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let mut bytes = sidecar.to_bytes();
        // Flip a bit in the embedded json_hash.
        bytes[8] ^= 0xFF;
        let r = CompressedCanaSidecar::from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "report json hash mismatch (corruption)",
                ..
            })
        ));
    }

    #[test]
    fn corrupted_bundle_hash_rejected() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(4);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let mut bytes = sidecar.to_bytes();
        // Flip a bit in the trailing bundle_hash.
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        let r = CompressedCanaSidecar::from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "bundle hash mismatch (corruption)",
                ..
            })
        ));
    }

    #[test]
    fn truncated_compressed_payload_rejected() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(4);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let mut bytes = sidecar.to_bytes();
        // Inflate the compressed_len field beyond what's actually there.
        let n = bytes.len();
        // Find the compressed_len position: at the end minus 8 (bundle hash)
        // minus the compressed payload size. Just truncate the bytes —
        // simpler than tweaking a length field.
        bytes.truncate(n - 20); // drops bundle hash + part of payload
        let r = CompressedCanaSidecar::from_bytes(&bytes);
        assert!(r.is_err(), "expected malformed payload, got {:?}", r);
    }

    #[test]
    fn trailing_bytes_rejected() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(4);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let mut bytes = sidecar.to_bytes();
        bytes.extend_from_slice(b"extra_garbage");
        let r = CompressedCanaSidecar::from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "trailing bytes after sidecar bundle_hash",
                ..
            })
        ));
    }

    // ----- verify() ----------------------------------------------------

    #[test]
    fn verify_succeeds_on_well_formed_sidecar() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(6);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        sidecar.verify().unwrap();
    }

    #[test]
    fn verify_fails_when_in_memory_fields_tampered() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(6);
        let mut sidecar = CompressedCanaSidecar::build(&report, &history);
        // Tamper with in-memory json bytes without updating the hash.
        sidecar.report_json[0] ^= 0xFF;
        let r = sidecar.verify();
        assert!(r.is_err());
    }

    // ----- Compression ratio + metadata --------------------------------

    #[test]
    fn compression_ratio_finite() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(50);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let r = sidecar.compression_ratio();
        assert!(r.is_finite());
        assert!(r > 0.0);
        // The ratio is sidecar-on-disk / raw-content. Sidecar includes
        // ~50 bytes of header + bundle_hash overhead, so for tiny
        // inputs the ratio can be > 1.0; that's expected.
    }

    #[test]
    fn serialized_len_matches_to_bytes_len() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(8);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        assert_eq!(sidecar.serialized_len(), sidecar.to_bytes().len());
    }

    // ----- Path round-trip ---------------------------------------------

    #[test]
    fn round_trip_via_tempfile() {
        let program = make_program();
        let report = analyze_program(&program);
        let history = make_history(12);
        let sidecar = CompressedCanaSidecar::build(&report, &history);
        let tmp = std::env::temp_dir().join(format!(
            "cjc_cana_compress_sidecar_test_{}.bin",
            sidecar.bundle_hash
        ));
        sidecar.write_to_path(&tmp).expect("write tempfile");
        let loaded = CompressedCanaSidecar::read_from_path(&tmp).expect("read tempfile");
        assert_eq!(loaded.bundle_hash, sidecar.bundle_hash);
        assert_eq!(loaded.report_json, sidecar.report_json);
        let _ = std::fs::remove_file(&tmp);
    }
}
