//! Lossless compression of pass histories and arbitrary byte traces.
//!
//! ## Why a hand-written RLE
//!
//! The workspace deliberately avoids pulling general-purpose compression
//! crates (`flate2`, `zstd`, `lz4`) into the IR-adjacent dep graph. Those
//! crates are tuned for throughput, link in C dependencies, and carry their
//! own determinism considerations (compression-level heuristics, parallel
//! workers). For CANA's audit-trail use case we want:
//!
//! - **A 200-LOC byte-RLE codec** with one deterministic encoder rule
//!   ("emit a run when next-byte-equality holds for ≥ 2 bytes; else emit
//!   a literal block").
//! - **Exact round-trip** with explicit integrity checks at the byte level
//!   (magic header, original-length stamp, FNV-1a hash of decoded bytes).
//! - **No allocations after the worst-case capacity is reserved.**
//!
//! ## Byte format
//!
//! ```text
//! header:
//!   4 bytes  magic = b"CLT0"     (CANA Lossless Trace v0)
//!   8 bytes  original_len      (u64 LE)
//!   8 bytes  input FNV-1a hash (u64 LE) — bound at encode time
//!
//! body:
//!   one or more chunks, each:
//!     1 byte  control = run_bit << 7 | (count - 1)
//!     if run_bit == 1:  1 byte  value
//!     else:             count bytes literal
//! ```
//!
//! Counts are stored as `count - 1` so a single control byte addresses
//! 1..128 (not 0..127). Runs require `count >= 2`; literal blocks accept
//! `count >= 1`.
//!
//! ## PassHistory adapter
//!
//! [`compress_pass_history`] / [`decompress_pass_history`] wrap the byte
//! codec around a canonical serializer for
//! [`cjc_cana::PassHistory`] entries. The canonical encoding lives in this
//! module so it's reviewed alongside the codec invariants it has to
//! satisfy.

use cjc_cana::hash::{hash_bytes, ProgramHash};
use cjc_cana::pass_history::{PassHistory, PassOutcome, PassRecord, SkipReason};

use crate::candidate::CompressionError;

// ---------------------------------------------------------------------------
// Public payload type
// ---------------------------------------------------------------------------

/// Output of [`lossless_compress_bytes`] (or the
/// [`compress_pass_history`] convenience). Carries everything needed to
/// verify the round-trip without re-running the encoder.
#[derive(Debug, Clone)]
pub struct LosslessTracePayload {
    /// The compressed bytes (header + body).
    pub bytes: Vec<u8>,
    /// FNV-1a hash of the original input (also embedded in the header for
    /// at-encode integrity).
    pub input_hash: u64,
    /// FNV-1a hash of the compressed bytes (covers header + body).
    pub compressed_hash: u64,
    /// Length of the original input.
    pub original_len: usize,
}

impl LosslessTracePayload {
    /// Compression ratio = `compressed_bytes.len() / original_len`. `1.0`
    /// means no savings; `0.0` would mean infinite savings (impossible
    /// because the header alone is 20 bytes).
    pub fn ratio(&self) -> f64 {
        if self.original_len == 0 {
            return 0.0;
        }
        (self.bytes.len() as f64) / (self.original_len as f64)
    }
}

// ---------------------------------------------------------------------------
// Byte-level codec
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"CLT0";
const HEADER_LEN: usize = 4 + 8 + 8; // magic + original_len + input_hash
const MAX_CHUNK: usize = 128; // (count - 1) fits in 7 bits => count in 1..128

/// Compress an arbitrary byte slice losslessly.
///
/// Encoder rule (deterministic): walk the input left-to-right; at each
/// position, emit the longest run of `(MAX_CHUNK).min(remaining)` equal
/// bytes if `current == next`, otherwise emit a literal block of length up
/// to `MAX_CHUNK` ending at the next position where a run of `>= 2` would
/// begin. The encoder NEVER backtracks. Two distinct outputs cannot exist
/// for the same input.
///
/// Panic-free for any byte slice up to `u64::MAX` length (we use `u64` for
/// the length stamp).
pub fn lossless_compress_bytes(input: &[u8]) -> LosslessTracePayload {
    let mut out = Vec::with_capacity(HEADER_LEN + input.len() + input.len() / 64 + 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(input.len() as u64).to_le_bytes());
    let input_hash = hash_bytes(input);
    out.extend_from_slice(&input_hash.to_le_bytes());

    let mut i = 0;
    while i < input.len() {
        // Look ahead for the longest run starting at `i`.
        let mut run_len = 1usize;
        while i + run_len < input.len() && input[i + run_len] == input[i] && run_len < MAX_CHUNK {
            run_len += 1;
        }
        if run_len >= 2 {
            // Emit a run chunk.
            let ctrl = 0x80u8 | ((run_len - 1) as u8);
            out.push(ctrl);
            out.push(input[i]);
            i += run_len;
        } else {
            // Emit a literal block. Scan forward until either we run out of
            // input, hit MAX_CHUNK literals, or find a position where a run
            // of >= 2 begins (so the next pass can capture it).
            let mut lit_len = 1usize;
            while i + lit_len < input.len() && lit_len < MAX_CHUNK {
                // Would the next chunk start a run? If yes, stop literal here.
                if i + lit_len + 1 < input.len() && input[i + lit_len] == input[i + lit_len + 1] {
                    break;
                }
                lit_len += 1;
            }
            let ctrl = (lit_len - 1) as u8; // run bit = 0
            out.push(ctrl);
            out.extend_from_slice(&input[i..i + lit_len]);
            i += lit_len;
        }
    }

    let compressed_hash = hash_bytes(&out);
    LosslessTracePayload {
        bytes: out,
        input_hash,
        compressed_hash,
        original_len: input.len(),
    }
}

/// Inverse of [`lossless_compress_bytes`]. Verifies the magic header, the
/// original-length stamp, and (on success) the FNV-1a hash of the decoded
/// output.
pub fn lossless_decompress_bytes(compressed: &[u8]) -> Result<Vec<u8>, CompressionError> {
    if compressed.len() < HEADER_LEN {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "header truncated",
        });
    }
    if &compressed[0..4] != MAGIC {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "bad magic",
        });
    }
    let original_len = u64::from_le_bytes(compressed[4..12].try_into().unwrap()) as usize;
    let expected_input_hash = u64::from_le_bytes(compressed[12..20].try_into().unwrap());

    let mut out = Vec::with_capacity(original_len);
    let mut cursor = HEADER_LEN;
    while cursor < compressed.len() {
        let ctrl = compressed[cursor];
        cursor += 1;
        let count = ((ctrl & 0x7F) as usize) + 1;
        let is_run = (ctrl & 0x80) != 0;
        if is_run {
            if cursor >= compressed.len() {
                return Err(CompressionError::MalformedPayload {
                    at_byte: cursor,
                    reason: "truncated run value",
                });
            }
            let v = compressed[cursor];
            cursor += 1;
            for _ in 0..count {
                out.push(v);
            }
        } else if cursor + count > compressed.len() {
            return Err(CompressionError::MalformedPayload {
                at_byte: cursor,
                reason: "truncated literal block",
            });
        } else {
            out.extend_from_slice(&compressed[cursor..cursor + count]);
            cursor += count;
        }
    }

    if out.len() != original_len {
        return Err(CompressionError::MalformedPayload {
            at_byte: cursor,
            reason: "decoded length mismatches header stamp",
        });
    }
    let observed_hash = hash_bytes(&out);
    if observed_hash != expected_input_hash {
        return Err(CompressionError::MalformedPayload {
            at_byte: HEADER_LEN,
            reason: "input hash mismatch (corruption)",
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// PassHistory adapter
// ---------------------------------------------------------------------------

const PASS_HISTORY_MAGIC: &[u8; 4] = b"CLP0";

/// Outcome tag bytes — declaration order matches
/// [`PassOutcome`]/[`SkipReason`] for stable serialization.
const OUTCOME_CHANGED: u8 = 0;
const OUTCOME_NOOP: u8 = 1;
const OUTCOME_SKIP_LEGALITY: u8 = 2;
const OUTCOME_SKIP_USER: u8 = 3;
const OUTCOME_SKIP_COST: u8 = 4;
const OUTCOME_SKIP_OTHER: u8 = 5;

/// Canonical serialization of a [`PassHistory`]. Used as the input to
/// [`lossless_compress_bytes`] when compressing a pass history. Exposed
/// `pub(crate)` so the tests can assert against it.
pub(crate) fn pass_history_to_bytes(history: &PassHistory) -> Vec<u8> {
    let mut out = Vec::with_capacity(PASS_HISTORY_MAGIC.len() + 8 + history.len() * 32);
    out.extend_from_slice(PASS_HISTORY_MAGIC);
    out.extend_from_slice(&(history.len() as u64).to_le_bytes());
    for record in history.iter() {
        // Pass name (length-prefixed).
        out.extend_from_slice(&(record.pass_name.len() as u32).to_le_bytes());
        out.extend_from_slice(record.pass_name.as_bytes());
        // Hashes.
        out.extend_from_slice(&record.input_hash.0.to_le_bytes());
        out.extend_from_slice(&record.output_hash.0.to_le_bytes());
        // Outcome tag.
        out.push(match record.outcome {
            PassOutcome::Changed => OUTCOME_CHANGED,
            PassOutcome::NoOp => OUTCOME_NOOP,
            PassOutcome::Skipped(SkipReason::LegalityGate) => OUTCOME_SKIP_LEGALITY,
            PassOutcome::Skipped(SkipReason::UserDisabled) => OUTCOME_SKIP_USER,
            PassOutcome::Skipped(SkipReason::CostBelowThreshold) => OUTCOME_SKIP_COST,
            PassOutcome::Skipped(SkipReason::Other) => OUTCOME_SKIP_OTHER,
        });
    }
    out
}

fn pass_history_from_bytes(bytes: &[u8]) -> Result<PassHistory, CompressionError> {
    if bytes.len() < PASS_HISTORY_MAGIC.len() + 8 {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "pass-history header truncated",
        });
    }
    if &bytes[0..4] != PASS_HISTORY_MAGIC {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "bad pass-history magic",
        });
    }
    let record_count = u64::from_le_bytes(bytes[4..12].try_into().unwrap()) as usize;
    let mut history = PassHistory::with_capacity(record_count.max(1));
    let mut cursor = 12;
    for _ in 0..record_count {
        if cursor + 4 > bytes.len() {
            return Err(CompressionError::MalformedPayload {
                at_byte: cursor,
                reason: "truncated record name length",
            });
        }
        let name_len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + name_len + 8 + 8 + 1 > bytes.len() {
            return Err(CompressionError::MalformedPayload {
                at_byte: cursor,
                reason: "truncated record body",
            });
        }
        let pass_name = std::str::from_utf8(&bytes[cursor..cursor + name_len]).map_err(|_| {
            CompressionError::MalformedPayload {
                at_byte: cursor,
                reason: "non-UTF8 pass name",
            }
        })?;
        cursor += name_len;
        let input_hash = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let output_hash = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let outcome_tag = bytes[cursor];
        cursor += 1;
        let outcome = match outcome_tag {
            OUTCOME_CHANGED => PassOutcome::Changed,
            OUTCOME_NOOP => PassOutcome::NoOp,
            OUTCOME_SKIP_LEGALITY => PassOutcome::Skipped(SkipReason::LegalityGate),
            OUTCOME_SKIP_USER => PassOutcome::Skipped(SkipReason::UserDisabled),
            OUTCOME_SKIP_COST => PassOutcome::Skipped(SkipReason::CostBelowThreshold),
            OUTCOME_SKIP_OTHER => PassOutcome::Skipped(SkipReason::Other),
            _ => {
                return Err(CompressionError::MalformedPayload {
                    at_byte: cursor - 1,
                    reason: "unknown outcome tag",
                });
            }
        };
        history.record(PassRecord {
            pass_name: pass_name.to_string(),
            input_hash: ProgramHash(input_hash),
            output_hash: ProgramHash(output_hash),
            outcome,
        });
    }
    if cursor != bytes.len() {
        return Err(CompressionError::MalformedPayload {
            at_byte: cursor,
            reason: "extra trailing bytes",
        });
    }
    Ok(history)
}

/// Compress a [`PassHistory`] losslessly. Returns a payload whose
/// `input_hash` is the FNV-1a hash of the canonical pass-history byte
/// representation (NOT of the in-memory `PassHistory` struct).
pub fn compress_pass_history(history: &PassHistory) -> LosslessTracePayload {
    let bytes = pass_history_to_bytes(history);
    lossless_compress_bytes(&bytes)
}

/// Inverse of [`compress_pass_history`]. Round-trips exactly: the returned
/// `PassHistory` iterates the same records in the same order as the input.
pub fn decompress_pass_history(
    payload: &LosslessTracePayload,
) -> Result<PassHistory, CompressionError> {
    let raw = lossless_decompress_bytes(&payload.bytes)?;
    pass_history_from_bytes(&raw)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Byte codec --------------------------------------------------

    #[test]
    fn single_byte_round_trips() {
        let payload = lossless_compress_bytes(&[7]);
        let back = lossless_decompress_bytes(&payload.bytes).unwrap();
        assert_eq!(back, vec![7]);
    }

    #[test]
    fn long_run_compresses_well() {
        let input = vec![0xFFu8; 1000];
        let payload = lossless_compress_bytes(&input);
        // 1000 / 128 = 7.81 → 8 chunks × 2 bytes = 16, plus 20-byte header = 36.
        assert!(
            payload.bytes.len() < 50,
            "1000-byte run should compress to < 50 bytes, got {}",
            payload.bytes.len()
        );
        let back = lossless_decompress_bytes(&payload.bytes).unwrap();
        assert_eq!(back, input);
    }

    #[test]
    fn mixed_runs_and_literals_round_trip() {
        // Pattern with: short run, literals, long run, literals.
        let mut input = vec![1, 1, 2, 3, 4, 5];
        input.extend(std::iter::repeat(7).take(200));
        input.extend([8, 9, 10]);
        let payload = lossless_compress_bytes(&input);
        let back = lossless_decompress_bytes(&payload.bytes).unwrap();
        assert_eq!(back, input);
    }

    #[test]
    fn random_bytes_round_trip_exactly() {
        // Worst case for RLE: every byte different. The compressor still
        // round-trips, even if compression ratio is > 1.
        let input: Vec<u8> = (0..255).collect();
        let payload = lossless_compress_bytes(&input);
        let back = lossless_decompress_bytes(&payload.bytes).unwrap();
        assert_eq!(back, input);
    }

    #[test]
    fn compress_is_deterministic() {
        let input = b"abracadabra-abracadabra-the-quick-brown-fox";
        let p1 = lossless_compress_bytes(input);
        let p2 = lossless_compress_bytes(input);
        assert_eq!(p1.bytes, p2.bytes);
        assert_eq!(p1.input_hash, p2.input_hash);
        assert_eq!(p1.compressed_hash, p2.compressed_hash);
    }

    #[test]
    fn truncated_header_rejected() {
        let r = lossless_decompress_bytes(&[1, 2, 3]);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "header truncated",
            })
        ));
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(b"XXXX");
        let r = lossless_decompress_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "bad magic",
            })
        ));
    }

    #[test]
    fn truncated_run_rejected() {
        // Header valid, body has 0x81 (run of 2) but no following value byte.
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&2u64.to_le_bytes()); // original_len = 2
        bytes.extend_from_slice(&hash_bytes(&[5, 5]).to_le_bytes());
        bytes.push(0x81); // run of 2, but no value
        let r = lossless_decompress_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "truncated run value",
                ..
            })
        ));
    }

    #[test]
    fn truncated_literal_rejected() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&hash_bytes(&[1, 2, 3]).to_le_bytes());
        bytes.push(0x02); // literal block of 3, only 1 byte present
        bytes.push(0x01);
        let r = lossless_decompress_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "truncated literal block",
                ..
            })
        ));
    }

    #[test]
    fn hash_mismatch_detected() {
        // Build a valid payload, then corrupt the embedded input hash.
        let input = b"hello world";
        let mut payload = lossless_compress_bytes(input).bytes;
        // Flip a bit in the input_hash region (bytes 12..20).
        payload[12] ^= 0xFF;
        let r = lossless_decompress_bytes(&payload);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "input hash mismatch (corruption)",
                ..
            })
        ));
    }

    #[test]
    fn payload_ratio_for_homogeneous_input() {
        // 10000 identical bytes — ratio should be tiny.
        let input = vec![0xAA; 10000];
        let p = lossless_compress_bytes(&input);
        assert!(
            p.ratio() < 0.05,
            "homogeneous input compressed to ratio {}",
            p.ratio()
        );
    }

    #[test]
    fn payload_ratio_returns_zero_on_empty() {
        let p = LosslessTracePayload {
            bytes: vec![],
            input_hash: 0,
            compressed_hash: 0,
            original_len: 0,
        };
        assert_eq!(p.ratio(), 0.0);
    }

    // ----- PassHistory adapter ----------------------------------------

    fn make_record(name: &str, before: u64, after: u64, outcome: PassOutcome) -> PassRecord {
        PassRecord {
            pass_name: name.to_string(),
            input_hash: ProgramHash(before),
            output_hash: ProgramHash(after),
            outcome,
        }
    }

    #[test]
    fn pass_history_round_trips_exactly() {
        let mut h = PassHistory::with_capacity(16);
        h.record(make_record("cf", 1, 2, PassOutcome::Changed));
        h.record(make_record("dce", 2, 3, PassOutcome::Changed));
        h.record(make_record("cse", 3, 3, PassOutcome::NoOp));
        h.record(make_record(
            "licm",
            3,
            3,
            PassOutcome::Skipped(SkipReason::LegalityGate),
        ));
        h.record(make_record(
            "sr",
            3,
            3,
            PassOutcome::Skipped(SkipReason::CostBelowThreshold),
        ));

        let payload = compress_pass_history(&h);
        let back = decompress_pass_history(&payload).unwrap();

        let original: Vec<_> = h.iter().cloned().collect();
        let restored: Vec<_> = back.iter().cloned().collect();
        assert_eq!(original, restored);
    }

    #[test]
    fn pass_history_empty_is_degenerate_but_handled() {
        // Empty PassHistory has 0 records but still has a valid header.
        let h = PassHistory::new();
        let payload = compress_pass_history(&h);
        let back = decompress_pass_history(&payload).unwrap();
        assert_eq!(back.len(), 0);
    }

    #[test]
    fn pass_history_compressed_is_deterministic_across_calls() {
        let mut h = PassHistory::with_capacity(8);
        for i in 0..8 {
            h.record(make_record("cf", i, i + 1, PassOutcome::Changed));
        }
        let p1 = compress_pass_history(&h);
        let p2 = compress_pass_history(&h);
        assert_eq!(p1.bytes, p2.bytes);
        assert_eq!(p1.input_hash, p2.input_hash);
        assert_eq!(p1.compressed_hash, p2.compressed_hash);
    }

    #[test]
    fn pass_history_with_zero_hashes_compresses_to_under_full_size() {
        // A pass history with all-zero hash bytes — RLE catches the
        // long runs of zeros inside each record. We don't compare to a
        // diverse history because the canonical record encoding always
        // interleaves zero-rich hash bytes with non-zero name bytes,
        // which makes the diverse-vs-homogeneous ratio comparison
        // brittle: byte-level RLE benefits roughly equally from both
        // because both contain the same per-record zero runs. What we
        // can robustly assert is that the lossless codec compresses a
        // homogeneous, mostly-zero stream below 1.0.
        let mut h = PassHistory::with_capacity(256);
        for _ in 0..200 {
            h.record(make_record("cf", 0, 0, PassOutcome::NoOp));
        }
        let p = compress_pass_history(&h);
        assert!(
            p.ratio() < 1.0,
            "expected ratio < 1.0 on zero-rich history, got {}",
            p.ratio()
        );
    }

    #[test]
    fn pass_history_corrupted_inner_magic_rejected() {
        // Round-trip canonical bytes, then corrupt the pass-history magic
        // before re-compressing. The byte decoder will succeed (hash
        // matches), but pass_history_from_bytes will reject.
        let mut h = PassHistory::with_capacity(4);
        h.record(make_record("cf", 1, 2, PassOutcome::Changed));
        let mut bytes = pass_history_to_bytes(&h);
        bytes[0] = b'X'; // break the magic
        let r = pass_history_from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "bad pass-history magic",
                ..
            })
        ));
    }

    #[test]
    fn pass_history_unknown_outcome_tag_rejected() {
        let mut h = PassHistory::with_capacity(4);
        h.record(make_record("cf", 1, 2, PassOutcome::Changed));
        let mut bytes = pass_history_to_bytes(&h);
        // The outcome tag is the last byte of the only record.
        *bytes.last_mut().unwrap() = 99;
        let r = pass_history_from_bytes(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "unknown outcome tag",
                ..
            })
        ));
    }
}
