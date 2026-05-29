//! Deterministic byte-pair-encoding (BPE) tokenizer (v0.7+ heavy, A5.1).
//!
//! The text-drift detectors in v0.7+ A5 need a way to convert arbitrary
//! string columns into comparable distributions over discrete units
//! (tokens). Existing tokenizers (HuggingFace, SentencePiece, tiktoken)
//! pull in language runtimes / large dependency trees — incompatible
//! with Locke's zero-external-dep contract and with the byte-identical
//! determinism guarantee.
//!
//! This module hand-rolls a minimal byte-level BPE:
//!
//! 1. **Initial vocab** is the 256 raw byte tokens. Every token ID in
//!    `0..256` is the byte with the same value.
//! 2. **Training** counts adjacent token-pair frequencies across the
//!    corpus, picks the most-frequent pair (with explicit lexicographic
//!    tie-break for determinism), and adds the merged token to the vocab.
//!    Repeats until the configured target vocab size is reached or no
//!    pair occurs more than once.
//! 3. **Encoding** applies each merge rule in training order to the byte
//!    sequence of the input string. Produces a `Vec<u32>` of token IDs.
//! 4. **Decoding** concatenates the byte sequences for each ID and
//!    interprets the result as UTF-8 (lossy on invalid sequences — Locke
//!    is downstream of validation, so invalid UTF-8 in a `Column::Str`
//!    is the producer's bug, not the tokenizer's).
//!
//! ## Determinism contract
//!
//! - Pair counting uses `BTreeMap<(u32, u32), u64>` — sorted iteration.
//! - Tie-break on equal pair frequency uses lexicographic comparison of
//!   `(left_token_bytes, right_token_bytes)`. Two training runs over the
//!   same corpus produce a byte-identical [`Tokenizer`].
//! - [`Tokenizer::fingerprint`] is a content-addressed hash over the
//!   vocab and merge rules; equal tokenizers have equal fingerprints.
//! - Encoding is deterministic by construction (merges are applied in
//!   training order).
//!
//! ## Scope of A5.1
//!
//! - No word-boundary marker. The corpus is treated as a flat byte
//!   stream per input string; the tokenizer can produce merges that
//!   span what a human would call word boundaries. For text drift this
//!   is acceptable — what matters is that the same input always
//!   produces the same tokens.
//! - No special tokens (`<bos>`, `<eos>`, etc.).
//! - No vocab serialization yet. A serialized form is A5.4.
//! - No support for byte-level pair merges that overflow `u32::MAX`
//!   token IDs (the worst case is corpus_bytes + target_vocab_size,
//!   well under 2^32 for any realistic Locke input).

use std::collections::BTreeMap;

use crate::id::{fingerprint, fingerprint_compose, FingerprintId, IdDomain};

// ─── Public types ────────────────────────────────────────────────────────

/// A trained, frozen BPE tokenizer.
///
/// Cheap to clone (vocab + merges are owned but small for the corpus
/// sizes Locke handles — typically `target_vocab_size <= 4096`).
#[derive(Clone, Debug, PartialEq)]
pub struct Tokenizer {
    /// `vocab[id as usize]` is the byte sequence the token represents.
    /// IDs `0..256` are raw bytes; IDs `>=256` are learned merges.
    vocab: Vec<Vec<u8>>,
    /// Merge rules in training application order:
    /// `(left_id, right_id, merged_id)`. Encoding applies them in this
    /// order; the merged_id always equals `256 + index` for the merge's
    /// position in `merges`.
    merges: Vec<(u32, u32, u32)>,
    /// Reverse lookup for encoding: token bytes → token ID. Only used
    /// by `decode_id_by_bytes` / direct lookups; encoding proper goes
    /// through the merge rules.
    byte_to_id: BTreeMap<Vec<u8>, u32>,
}

/// Optional knobs for [`Tokenizer::train`].
#[derive(Clone, Debug)]
pub struct TokenizerTrainConfig {
    /// Maximum number of tokens (including the 256 base bytes). At
    /// least 256. Default `1024`.
    pub target_vocab_size: u32,
    /// Stop training if no candidate pair occurs at least this many
    /// times. Avoids spending merge rules on coincidences. Default `2`.
    pub min_pair_frequency: u64,
}

impl Default for TokenizerTrainConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 1024,
            min_pair_frequency: 2,
        }
    }
}

// ─── Construction / training ──────────────────────────────────────────────

impl Tokenizer {
    /// Train a deterministic BPE tokenizer on a corpus of strings.
    ///
    /// The 256 raw byte tokens are always present (IDs `0..256`); learned
    /// merges occupy IDs `256..target_vocab_size`. Returns when either
    /// the target vocab is full or no pair satisfies
    /// `cfg.min_pair_frequency`.
    pub fn train(corpus: &[&str], cfg: &TokenizerTrainConfig) -> Self {
        let mut vocab: Vec<Vec<u8>> = (0u32..256).map(|b| vec![b as u8]).collect();
        let mut byte_to_id: BTreeMap<Vec<u8>, u32> =
            (0u32..256).map(|b| (vec![b as u8], b)).collect();
        let mut merges: Vec<(u32, u32, u32)> = Vec::new();

        // Initial corpus: each input string → its sequence of byte token IDs.
        let mut seqs: Vec<Vec<u32>> =
            corpus.iter().map(|s| s.bytes().map(|b| b as u32).collect()).collect();

        let target = cfg.target_vocab_size.max(256) as usize;

        while vocab.len() < target {
            // Count adjacent pairs across all sequences (BTreeMap → sorted).
            let mut counts: BTreeMap<(u32, u32), u64> = BTreeMap::new();
            for seq in &seqs {
                if seq.len() < 2 {
                    continue;
                }
                for w in seq.windows(2) {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                break;
            }

            // Pick the pair with the highest frequency; break ties by
            // lexicographic comparison of `(left_bytes, right_bytes)`.
            // The (count, lex) key produces a fully deterministic choice
            // independent of BTreeMap iteration order (which is already
            // deterministic — belt and suspenders).
            let chosen = counts
                .iter()
                .map(|(&(l, r), &c)| {
                    let lex = (vocab[l as usize].clone(), vocab[r as usize].clone());
                    (c, lex, l, r)
                })
                // Reverse-sort on count, then forward-sort on lex.
                .min_by(|a, b| {
                    b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1))
                });
            let (best_count, _lex, left_id, right_id) = match chosen {
                Some(x) => x,
                None => break,
            };
            if best_count < cfg.min_pair_frequency {
                break;
            }

            // Build merged token bytes and add to vocab.
            let mut merged_bytes = vocab[left_id as usize].clone();
            merged_bytes.extend_from_slice(&vocab[right_id as usize]);
            // If, somehow, this token already exists in the vocab (e.g.
            // two distinct merge sequences produce the same bytes), stop
            // — that's a structural impossibility in standard BPE but
            // worth a defensive check.
            if byte_to_id.contains_key(&merged_bytes) {
                break;
            }
            let merged_id = vocab.len() as u32;
            vocab.push(merged_bytes.clone());
            byte_to_id.insert(merged_bytes, merged_id);
            merges.push((left_id, right_id, merged_id));

            // Rewrite all sequences applying the new merge.
            for seq in seqs.iter_mut() {
                *seq = apply_merge(seq, left_id, right_id, merged_id);
            }
        }

        Self { vocab, merges, byte_to_id }
    }
}

/// Replace every adjacent `(left_id, right_id)` in `seq` with
/// `merged_id`. Single left-to-right pass.
fn apply_merge(seq: &[u32], left_id: u32, right_id: u32, merged_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left_id && seq[i + 1] == right_id {
            out.push(merged_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

// ─── Encoding / decoding ─────────────────────────────────────────────────

impl Tokenizer {
    /// Encode a string into a sequence of token IDs by replaying each
    /// trained merge rule in training order against the input's byte
    /// stream. Produces a result consistent with the training-time
    /// tokenization of the same byte sequence.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut seq: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        for &(left_id, right_id, merged_id) in &self.merges {
            seq = apply_merge(&seq, left_id, right_id, merged_id);
        }
        seq
    }

    /// Decode a sequence of token IDs back into a string by concatenating
    /// each token's byte sequence. Invalid UTF-8 in the output is
    /// replaced with `U+FFFD` via `String::from_utf8_lossy`.
    ///
    /// Unknown IDs (outside `[0, vocab_size())`) are skipped silently —
    /// the alternative is to panic, which would be wrong for downstream
    /// drift code that handles arbitrary user data.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(token) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(token);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Number of unique tokens (always `>= 256`).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Number of learned merge rules. Equal to
    /// `vocab_size() - 256`.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// Stable content-addressed fingerprint over the vocab and merge
    /// rules. Two tokenizers with the same trained state produce the
    /// same fingerprint; any change to vocab or merges produces a new
    /// fingerprint.
    pub fn fingerprint(&self) -> FingerprintId {
        let mut parts: Vec<FingerprintId> = Vec::with_capacity(self.vocab.len() + self.merges.len());
        for tok in &self.vocab {
            parts.push(fingerprint(IdDomain::Idea, tok));
        }
        for &(l, r, m) in &self.merges {
            let mut buf = Vec::with_capacity(12);
            buf.extend_from_slice(&l.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
            buf.extend_from_slice(&m.to_le_bytes());
            parts.push(fingerprint(IdDomain::Idea, &buf));
        }
        fingerprint_compose(IdDomain::Idea, "tokenizer", &parts)
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn train_default(corpus: &[&str]) -> Tokenizer {
        Tokenizer::train(corpus, &TokenizerTrainConfig::default())
    }

    // ── Vocab / construction invariants ───────────────────────────────

    #[test]
    fn fresh_tokenizer_has_256_base_tokens() {
        let t = train_default(&[]);
        assert_eq!(t.vocab_size(), 256);
        assert_eq!(t.merge_count(), 0);
    }

    #[test]
    fn base_token_ids_match_byte_values() {
        let t = train_default(&[]);
        // ID i must represent the byte with value i.
        for i in 0u32..256 {
            assert_eq!(t.vocab[i as usize], vec![i as u8]);
        }
    }

    // ── Round-trip encode/decode ──────────────────────────────────────

    #[test]
    fn encode_decode_round_trip_on_ascii() {
        let t = train_default(&["hello", "world", "hello world"]);
        let original = "hello world";
        let ids = t.encode(original);
        assert_eq!(t.decode(&ids), original);
    }

    #[test]
    fn encode_decode_round_trip_on_utf8_with_combining_marks() {
        // "café" with a precomposed é
        let t = train_default(&["café au lait", "café"]);
        let original = "café";
        let ids = t.encode(original);
        assert_eq!(t.decode(&ids), original);
    }

    #[test]
    fn empty_string_encodes_to_empty_sequence() {
        let t = train_default(&["abc"]);
        let ids = t.encode("");
        assert!(ids.is_empty());
        assert_eq!(t.decode(&ids), "");
    }

    #[test]
    fn decoding_unknown_id_is_skipped() {
        let t = train_default(&[]);
        // ID 99999 is out of range; should be skipped without panicking.
        let s = t.decode(&[b'a' as u32, 99_999, b'b' as u32]);
        assert_eq!(s, "ab");
    }

    // ── Training behaviour ────────────────────────────────────────────

    #[test]
    fn training_learns_at_least_one_merge_on_repeated_bigrams() {
        // "th" appears 4 times — well above min_pair_frequency=2.
        let cfg = TokenizerTrainConfig {
            target_vocab_size: 260,
            ..Default::default()
        };
        let t = Tokenizer::train(
            &["this thread the throne thaw"],
            &cfg,
        );
        assert!(t.merge_count() >= 1);
        // The first merge is "t" + "h" → "th".
        let (l, r, _) = t.merges[0];
        assert_eq!(t.vocab[l as usize], b"t");
        assert_eq!(t.vocab[r as usize], b"h");
    }

    #[test]
    fn training_stops_at_target_vocab_size() {
        let cfg = TokenizerTrainConfig {
            target_vocab_size: 258, // base 256 + 2 merges max
            min_pair_frequency: 1,
            ..Default::default()
        };
        let t = Tokenizer::train(&["aaaabbbb"], &cfg);
        assert!(t.vocab_size() <= 258);
    }

    #[test]
    fn training_stops_when_min_pair_frequency_not_met() {
        let cfg = TokenizerTrainConfig {
            target_vocab_size: 4096,
            min_pair_frequency: 10,
        };
        // Only single-occurrence bigrams.
        let t = Tokenizer::train(&["abc"], &cfg);
        assert_eq!(t.merge_count(), 0);
    }

    #[test]
    fn training_on_empty_corpus_produces_only_base_tokens() {
        let cfg = TokenizerTrainConfig::default();
        let t = Tokenizer::train(&[], &cfg);
        assert_eq!(t.vocab_size(), 256);
    }

    // ── Determinism ───────────────────────────────────────────────────

    #[test]
    fn training_is_byte_identical_across_runs() {
        let corpus = &["the quick brown fox", "the lazy dog", "fox over fox"];
        let cfg = TokenizerTrainConfig::default();
        let a = Tokenizer::train(corpus, &cfg);
        let b = Tokenizer::train(corpus, &cfg);
        assert_eq!(a, b);
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn encoding_is_byte_identical_across_runs() {
        let t = train_default(&["the lazy fox", "the quick fox"]);
        let s = "the fox is quick";
        let a = t.encode(s);
        let b = t.encode(s);
        assert_eq!(a, b);
    }

    #[test]
    fn tie_breaks_use_lex_order_on_equal_frequency_pairs() {
        // Pairs (a,b), (b,c), (c,d) each occur twice within "abcdabcd";
        // (d,' '), (' ',a) each occur once and are excluded by
        // min_pair_frequency=2. Among the tied candidates, lex order on
        // (left_bytes, right_bytes) picks ('a','b') first because
        // [b'a'] (= [97]) < [b'b'] (= [98]) < [b'c'].
        let cfg = TokenizerTrainConfig {
            target_vocab_size: 257, // exactly one merge
            min_pair_frequency: 2,
        };
        let t = Tokenizer::train(&["abcd abcd"], &cfg);
        assert_eq!(t.merge_count(), 1);
        let (l, r, _) = t.merges[0];
        assert_eq!(t.vocab[l as usize], b"a");
        assert_eq!(t.vocab[r as usize], b"b");
    }

    // ── Fingerprint behaviour ─────────────────────────────────────────

    #[test]
    fn fingerprint_changes_when_corpus_changes() {
        // Use min_pair_frequency=1 so even tiny corpora learn distinct
        // merges. Otherwise both would produce vocab=256 base bytes +
        // 0 merges → identical fingerprint, which is a real (and
        // boring) behavior worth not testing here.
        let cfg = TokenizerTrainConfig {
            target_vocab_size: 512,
            min_pair_frequency: 1,
        };
        let t1 = Tokenizer::train(&["the quick brown fox"], &cfg);
        let t2 = Tokenizer::train(&["entirely different content"], &cfg);
        assert_ne!(t1.fingerprint(), t2.fingerprint());
    }

    #[test]
    fn fingerprint_is_equal_when_corpora_train_to_identical_vocab() {
        // Edge case: corpora that produce no merges have equal
        // fingerprints. Document this as expected behavior.
        let cfg = TokenizerTrainConfig {
            target_vocab_size: 4096,
            min_pair_frequency: 99, // unreachable on small corpora
        };
        let t1 = Tokenizer::train(&["the quick brown fox"], &cfg);
        let t2 = Tokenizer::train(&["the lazy dog"], &cfg);
        // No merges either side → same 256-byte vocab → same fingerprint.
        assert_eq!(t1.merge_count(), 0);
        assert_eq!(t2.merge_count(), 0);
        assert_eq!(t1.fingerprint(), t2.fingerprint());
    }

    #[test]
    fn fingerprint_is_stable_across_repeated_calls() {
        let t = train_default(&["hello", "world"]);
        let f1 = t.fingerprint();
        let f2 = t.fingerprint();
        assert_eq!(f1, f2);
    }

    // ── Pairs-of-merges sanity ────────────────────────────────────────

    #[test]
    fn merged_token_id_is_always_at_or_after_256() {
        let t = train_default(&["aaaa bbbb cccc aaaa bbbb cccc"]);
        for &(_, _, merged) in &t.merges {
            assert!(
                merged >= 256,
                "merged token ID {} must be in the learned range",
                merged
            );
        }
    }

    #[test]
    fn merge_ids_are_sequential_starting_from_256() {
        let t = train_default(&["aaaabbbbccccddddeeeffffaaaa"]);
        for (idx, &(_, _, merged)) in t.merges.iter().enumerate() {
            assert_eq!(merged as usize, 256 + idx);
        }
    }

    #[test]
    fn encoded_sequence_decodes_back_to_input_bytes_under_all_merges() {
        let t = train_default(&["the quick brown fox jumps over the lazy dog"]);
        let original = "the fox";
        let ids = t.encode(original);
        let decoded = t.decode(&ids);
        assert_eq!(decoded, original);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn invalid_utf8_in_training_corpus_does_not_panic() {
        // This is a "should not happen" path — Column::Str enforces
        // UTF-8 at construction. But the tokenizer operates on bytes
        // and must not panic if a caller hands it pre-mangled data.
        // We can't construct an &str with invalid UTF-8 directly, but
        // we can verify trains and encodes work on edge-case strings
        // containing every possible byte category.
        let s = "\u{0001}\u{007F}\u{0080}\u{00FF}\u{D7FF}\u{E000}\u{FFFD}\u{10FFFF}";
        let t = train_default(&[s]);
        let ids = t.encode(s);
        assert_eq!(t.decode(&ids), s);
    }

    #[test]
    fn single_byte_inputs_round_trip() {
        let t = train_default(&["a", "b", "c"]);
        for s in ["", "a", "ab", "abc"] {
            let ids = t.encode(s);
            assert_eq!(t.decode(&ids), s);
        }
    }
}
