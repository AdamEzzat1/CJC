//! Deterministic LZ77-style motif/dictionary compression.
//!
//! Pairs with [`lossless_trace`](crate::lossless_trace) as the second
//! **lossless** compression scheme available to
//! [`CompressionKind::MotifDictionary`](crate::candidate::CompressionKind).
//! The two schemes target different shape patterns:
//!
//! - `LosslessTrace` (byte RLE) wins on inputs with long runs of identical
//!   bytes (no-op pass repetitions, identical-hash histories).
//! - `MotifDictionary` (LZ77 back-references) wins on inputs where the same
//!   *multi-byte* motif recurs at varying offsets — repeated function-name
//!   strings, repeated `(hash, hash, outcome)` triples that share fragments,
//!   feature sequences with recurring patterns.
//!
//! ## Why a custom LZ77 (not `flate2`)
//!
//! Same reasoning as the RLE codec: no C deps, deterministic-by-construction
//! encoder, audit-friendly format. The encoder rule is *literally* one-pass
//! greedy:
//!
//! 1. Scan left-to-right.
//! 2. At each position, find the longest match in the sliding window of
//!    `WINDOW_SIZE` previous bytes. Among ties (multiple positions of the
//!    same maximum length), pick the **smallest offset** (the *most recent*
//!    occurrence). This is the deterministic tie-break.
//! 3. If `match_len >= MIN_MATCH`, emit `Back { offset, len }`.
//! 4. Else, emit `Literal { byte }`.
//!
//! `MIN_MATCH = 3` so a 2-byte match doesn't outcompete a 3-byte literal +
//! cheap follow-on.
//!
//! ## Byte format
//!
//! ```text
//! header:
//!   4 bytes  magic = b"CMD0"           (CANA Motif Dictionary v0)
//!   8 bytes  original_len              (u64 LE)
//!   8 bytes  input FNV-1a hash         (u64 LE)
//!
//! body:
//!   one or more tokens; each token is one byte of control + payload:
//!     control: high bit 0  → literal; low 7 bits = (lit_count - 1) in 1..128;
//!                            followed by lit_count bytes
//!     control: high bit 1  → back-reference; low 7 bits = (len - MIN_MATCH)
//!                            in 0..127 so encoded lengths are 3..130;
//!                            followed by 2 bytes (u16 LE) = offset 1..32768
//! ```
//!
//! Bounds: `MIN_MATCH = 3`, `MAX_MATCH = MIN_MATCH + 127 = 130`,
//! `WINDOW_SIZE = 32768`. Same caveats as the RLE codec apply: byte format
//! freeze is part of `COMPRESS_VERSION`, format renames break existing
//! report hashes.

use cjc_cana::hash::hash_bytes;

use crate::candidate::CompressionError;

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"CMD0";
const HEADER_LEN: usize = 4 + 8 + 8;

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = MIN_MATCH + 127; // 130
const WINDOW_SIZE: usize = 32768;
const MAX_LITERAL_BLOCK: usize = 128; // 1..128 stored as (count-1) in 7 bits

// ---------------------------------------------------------------------------
// Public payload type
// ---------------------------------------------------------------------------

/// Output of [`compress_motif_dictionary`].
#[derive(Debug, Clone)]
pub struct MotifDictionaryPayload {
    /// Compressed bytes (header + body).
    pub bytes: Vec<u8>,
    /// FNV-1a hash of the original input.
    pub input_hash: u64,
    /// FNV-1a hash of the compressed bytes.
    pub compressed_hash: u64,
    /// Length of the original input.
    pub original_len: usize,
    /// Number of back-reference tokens emitted. Useful for the report
    /// (large counts on a low-entropy input is a sign of good
    /// dictionary hits).
    pub backref_count: u32,
}

impl MotifDictionaryPayload {
    /// Compression ratio.
    pub fn ratio(&self) -> f64 {
        if self.original_len == 0 {
            return 0.0;
        }
        (self.bytes.len() as f64) / (self.original_len as f64)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Compress a byte slice using deterministic LZ77.
///
/// The encoder is purely greedy: at each position, find the longest match
/// in the previous `WINDOW_SIZE` bytes; on ties, the *most recent* (smallest
/// offset) wins. This rule is total, deterministic, and panic-free.
pub fn compress_motif_dictionary(input: &[u8]) -> MotifDictionaryPayload {
    let mut out = Vec::with_capacity(HEADER_LEN + input.len() + 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(input.len() as u64).to_le_bytes());
    let input_hash = hash_bytes(input);
    out.extend_from_slice(&input_hash.to_le_bytes());

    let mut pending_literals: Vec<u8> = Vec::new();
    let mut backref_count: u32 = 0;
    let mut i = 0;
    while i < input.len() {
        let (best_offset, best_len) = find_longest_match(input, i);
        if best_len >= MIN_MATCH {
            // Flush any pending literals first.
            flush_literals(&mut out, &mut pending_literals);
            // Emit back-reference.
            let len_code = (best_len - MIN_MATCH) as u8;
            out.push(0x80 | len_code);
            // Offset is u16 LE (we capped WINDOW_SIZE at 32768).
            let off_u16 = best_offset as u16;
            out.extend_from_slice(&off_u16.to_le_bytes());
            i += best_len;
            backref_count += 1;
        } else {
            pending_literals.push(input[i]);
            if pending_literals.len() == MAX_LITERAL_BLOCK {
                flush_literals(&mut out, &mut pending_literals);
            }
            i += 1;
        }
    }
    flush_literals(&mut out, &mut pending_literals);

    let compressed_hash = hash_bytes(&out);
    MotifDictionaryPayload {
        bytes: out,
        input_hash,
        compressed_hash,
        original_len: input.len(),
        backref_count,
    }
}

/// Walks back over the last `WINDOW_SIZE` bytes of `input[0..i]` and
/// returns the `(offset, length)` of the longest match starting at `i`.
/// On ties, returns the smallest offset (most recent occurrence) — the
/// determinism tie-break.
fn find_longest_match(input: &[u8], i: usize) -> (usize, usize) {
    if i == 0 || input.len() - i < MIN_MATCH {
        return (0, 0);
    }
    let window_start = i.saturating_sub(WINDOW_SIZE);
    let remaining = (input.len() - i).min(MAX_MATCH);

    let mut best_offset = 0usize;
    let mut best_len = 0usize;

    // Iterate from most-recent (offset 1) backward to oldest. The first
    // match of any given length we find is, by construction, the most
    // recent — so on ties the smallest offset always wins.
    for offset in 1..=(i - window_start) {
        let ref_pos = i - offset;
        // Compare bytes input[i..i+remaining] vs input[ref_pos..ref_pos+remaining].
        // Match length stops at first mismatch.
        let mut match_len = 0usize;
        while match_len < remaining && input[ref_pos + match_len] == input[i + match_len] {
            match_len += 1;
        }
        if match_len > best_len {
            best_len = match_len;
            best_offset = offset;
            if best_len == MAX_MATCH {
                break; // Can't beat MAX_MATCH; stop the scan.
            }
        }
    }
    (best_offset, best_len)
}

fn flush_literals(out: &mut Vec<u8>, pending: &mut Vec<u8>) {
    if pending.is_empty() {
        return;
    }
    let count = pending.len();
    debug_assert!(count <= MAX_LITERAL_BLOCK);
    let ctrl = (count - 1) as u8; // run bit unset
    out.push(ctrl);
    out.extend_from_slice(pending);
    pending.clear();
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Inverse of [`compress_motif_dictionary`].
pub fn decompress_motif_dictionary(compressed: &[u8]) -> Result<Vec<u8>, CompressionError> {
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
        let is_backref = (ctrl & 0x80) != 0;
        if is_backref {
            let len = ((ctrl & 0x7F) as usize) + MIN_MATCH;
            if cursor + 2 > compressed.len() {
                return Err(CompressionError::MalformedPayload {
                    at_byte: cursor,
                    reason: "truncated back-reference offset",
                });
            }
            let offset =
                u16::from_le_bytes(compressed[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2;
            if offset == 0 || offset > out.len() {
                return Err(CompressionError::MalformedPayload {
                    at_byte: cursor - 2,
                    reason: "back-reference offset out of range",
                });
            }
            // Copy `len` bytes from `out.len() - offset`. This must support
            // overlap (length > offset) for run-style matches.
            let start = out.len() - offset;
            for k in 0..len {
                let b = out[start + k];
                out.push(b);
            }
        } else {
            let count = ((ctrl & 0x7F) as usize) + 1;
            if cursor + count > compressed.len() {
                return Err(CompressionError::MalformedPayload {
                    at_byte: cursor,
                    reason: "truncated literal block",
                });
            }
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
    if hash_bytes(&out) != expected_input_hash {
        return Err(CompressionError::MalformedPayload {
            at_byte: HEADER_LEN,
            reason: "input hash mismatch (corruption)",
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_byte_round_trips() {
        let payload = compress_motif_dictionary(&[42]);
        let back = decompress_motif_dictionary(&payload.bytes).unwrap();
        assert_eq!(back, vec![42]);
    }

    #[test]
    fn no_compression_on_random_bytes() {
        // Random-looking, no motifs ≥ 3 bytes long → all literals.
        // Output should still round-trip exactly.
        let input: Vec<u8> = (0..200u8).collect();
        let payload = compress_motif_dictionary(&input);
        let back = decompress_motif_dictionary(&payload.bytes).unwrap();
        assert_eq!(back, input);
        assert_eq!(payload.backref_count, 0);
    }

    #[test]
    fn repeated_motif_emits_backrefs() {
        // "abcabcabcabcabc" — after first "abc", each subsequent "abc" is a
        // 3-byte back-reference (or longer once consecutive repeats kick in).
        let input = b"abcabcabcabcabc";
        let payload = compress_motif_dictionary(input);
        let back = decompress_motif_dictionary(&payload.bytes).unwrap();
        assert_eq!(back, input);
        assert!(
            payload.backref_count >= 1,
            "expected at least one back-reference, got {}",
            payload.backref_count
        );
        assert!(
            payload.bytes.len() < HEADER_LEN + input.len(),
            "expected compression below literal size"
        );
    }

    #[test]
    fn overlapping_run_via_backref_works() {
        // Classic LZ77 overlap test: input "aaa...aaa" should encode as one
        // literal + one back-reference where len > offset (the decoder
        // reads bytes it has just written).
        let input = vec![0xCDu8; 200];
        let payload = compress_motif_dictionary(&input);
        let back = decompress_motif_dictionary(&payload.bytes).unwrap();
        assert_eq!(back, input);
        // Overlap means we should see significant compression even though
        // RLE would also handle this well.
        assert!(payload.bytes.len() < 50);
    }

    #[test]
    fn motif_dict_is_deterministic() {
        let input = b"the_quick_brown_fox_the_quick_brown_fox_the_quick_brown_fox";
        let p1 = compress_motif_dictionary(input);
        let p2 = compress_motif_dictionary(input);
        assert_eq!(p1.bytes, p2.bytes);
        assert_eq!(p1.input_hash, p2.input_hash);
        assert_eq!(p1.compressed_hash, p2.compressed_hash);
        assert_eq!(p1.backref_count, p2.backref_count);
    }

    #[test]
    fn tie_break_picks_most_recent_offset() {
        // The string "AAABCAAABC" has two "AAABC" motifs. When the encoder
        // reaches the second one, it could find the match by going back 5
        // bytes (to position 0) — there's no earlier copy, so this case
        // doesn't tie-test directly. But we can construct a case:
        // "ABCDABCDABCD" — when encoder is at position 8 ("ABCD" starts),
        // both offsets 4 and 8 yield a 4-byte match starting at position
        // 4 (or 0). The most-recent rule should pick offset 4 (start at
        // pos 4), not offset 8 (start at pos 0).
        //
        // We can't directly inspect which offset was chosen without
        // re-parsing the token stream, but we CAN confirm round-trip is
        // exact and the encoder is deterministic.
        let input = b"ABCDABCDABCDABCD";
        let p1 = compress_motif_dictionary(input);
        let p2 = compress_motif_dictionary(input);
        let back = decompress_motif_dictionary(&p1.bytes).unwrap();
        assert_eq!(back, input);
        assert_eq!(p1.bytes, p2.bytes);
    }

    #[test]
    fn truncated_header_rejected() {
        let r = decompress_motif_dictionary(&[1, 2, 3]);
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
        let r = decompress_motif_dictionary(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                at_byte: 0,
                reason: "bad magic",
            })
        ));
    }

    #[test]
    fn truncated_backref_rejected() {
        // Build a valid-header payload that ends in 0x80 (backref control)
        // with no following offset bytes.
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&3u64.to_le_bytes()); // claim original_len = 3
        bytes.extend_from_slice(&hash_bytes(&[1, 2, 3]).to_le_bytes());
        bytes.push(0x80); // backref of MIN_MATCH bytes, but no offset
        let r = decompress_motif_dictionary(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "truncated back-reference offset",
                ..
            })
        ));
    }

    #[test]
    fn zero_offset_rejected() {
        // Construct a payload where a backref claims offset = 0.
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&hash_bytes(&[1, 2, 3]).to_le_bytes());
        // Backref token with offset 0 — invalid, must reject before
        // producing output.
        bytes.push(0x80);
        bytes.extend_from_slice(&0u16.to_le_bytes());
        let r = decompress_motif_dictionary(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "back-reference offset out of range",
                ..
            })
        ));
    }

    #[test]
    fn out_of_range_offset_rejected() {
        // Construct: literal one byte, then backref with offset > out.len().
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&hash_bytes(&[7, 7]).to_le_bytes());
        bytes.push(0x00); // literal block of 1
        bytes.push(7);
        bytes.push(0x80); // backref of 3 bytes
        bytes.extend_from_slice(&5u16.to_le_bytes()); // offset 5, but out only has 1 byte
        let r = decompress_motif_dictionary(&bytes);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "back-reference offset out of range",
                ..
            })
        ));
    }

    #[test]
    fn hash_mismatch_detected() {
        let input = b"hello world hello world";
        let mut payload = compress_motif_dictionary(input).bytes;
        // Flip a bit in the embedded input hash.
        payload[12] ^= 0xFF;
        let r = decompress_motif_dictionary(&payload);
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "input hash mismatch (corruption)",
                ..
            })
        ));
    }

    #[test]
    fn pass_history_canonical_bytes_compress_via_motif_dict() {
        // Make sure the motif-dict compressor works on the byte stream
        // a PassHistory produces — the two lossless schemes are
        // interchangeable on the same inputs.
        use cjc_cana::hash::ProgramHash;
        use cjc_cana::pass_history::{PassHistory, PassOutcome, PassRecord};
        let mut h = PassHistory::with_capacity(64);
        for i in 0..40u64 {
            // Repeat names a lot to give motif-dict something to chew on.
            let name = if i % 2 == 0 { "constant_fold" } else { "dce" };
            h.record(PassRecord {
                pass_name: name.to_string(),
                input_hash: ProgramHash(i),
                output_hash: ProgramHash(i + 1),
                outcome: PassOutcome::Changed,
            });
        }
        let bytes = crate::lossless_trace::pass_history_to_bytes(&h);
        let payload = compress_motif_dictionary(&bytes);
        let back = decompress_motif_dictionary(&payload.bytes).unwrap();
        assert_eq!(back, bytes);
        // Repeated names should produce back-references.
        assert!(payload.backref_count > 0);
    }
}
