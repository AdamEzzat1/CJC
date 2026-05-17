//! Phase 0.9.5 — deterministic categorical preprocessing.
//!
//! ABNG's routing codebook ([`crate::codebook::QuantileCodebook`]) bins
//! `f64` inputs; it has no path for categorical features. This module
//! adds the first piece of categorical support: a deterministic
//! per-feature category vocabulary.
//!
//! # Determinism
//!
//! Everything here is built from the **training split only** (never the
//! test split — that would leak) and is a pure function of the observed
//! category counts:
//!
//! * The vocabulary is a [`BTreeMap`] — never a `HashMap` — so iteration
//!   and the canonical-byte encoding are order-stable.
//! * Real categories are assigned codes by `(-count, label)` — most
//!   frequent first, ties broken lexicographically. Shuffling the input
//!   rows cannot change the result ([`CategoryDictionary::canonical_bytes`]
//!   is row-order invariant).
//! * Three reserved codes ([`CODE_MISSING`], [`CODE_UNKNOWN`],
//!   [`CODE_RARE`]) occupy the low slots, so they are fixed regardless of
//!   vocabulary size.
//! * Rare-category folding ([`RarePolicy`]) is a pure function of the
//!   training counts; the policy is retained on the dictionary so a
//!   schema snapshot reproduces the exact fold set.
//!
//! See `docs/abng/PHASE_0_9_5_HANDOFF.md` §3 for the full design.

use std::collections::{BTreeMap, BTreeSet};

/// Reserved code — the source cell was empty or a missing-marker.
pub const CODE_MISSING: u32 = 0;
/// Reserved code — a category absent from the frozen training
/// vocabulary (only reachable at transform / inference time).
pub const CODE_UNKNOWN: u32 = 1;
/// Reserved code — a training category folded by the [`RarePolicy`].
pub const CODE_RARE: u32 = 2;
/// First code available to real (kept, non-reserved) categories.
pub const FIRST_REAL_CODE: u32 = 3;

/// Rare-category folding policy.
///
/// A training category folds into [`CODE_RARE`] if it fails **either**
/// floor: its absolute training count is `< min_count`, **or** its
/// training frequency `count / total` (total counts every observed row,
/// including missing) is `< min_frac`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RarePolicy {
    /// Absolute training-count floor. A category with `count < min_count`
    /// folds.
    pub min_count: u64,
    /// Fractional training-frequency floor. A category with
    /// `count / total < min_frac` folds.
    pub min_frac: f64,
}

impl RarePolicy {
    /// The Phase 0.9.5 defaults: `min_count = 30`, `min_frac = 0.001`.
    pub const DEFAULT: RarePolicy = RarePolicy {
        min_count: 30,
        min_frac: 0.001,
    };

    /// A policy that folds nothing — every observed category is kept.
    pub const KEEP_ALL: RarePolicy = RarePolicy {
        min_count: 0,
        min_frac: 0.0,
    };
}

impl Default for RarePolicy {
    fn default() -> Self {
        RarePolicy::DEFAULT
    }
}

/// Accumulates training-split category observations for one feature,
/// then freezes into an immutable [`CategoryDictionary`].
#[derive(Debug, Clone)]
pub struct CategoryDictionaryBuilder {
    /// Per-real-category raw counts. Allocates a `String` only on the
    /// first sight of a category, not per row.
    counts: BTreeMap<String, u64>,
    /// Raw strings treated as "missing".
    missing_markers: BTreeSet<String>,
    missing_count: u64,
    total: u64,
}

impl CategoryDictionaryBuilder {
    /// Start a builder. `missing_markers` are the raw strings that mean
    /// "missing" — e.g. `["?"]` for the Diabetes-130 dataset, `[]` for a
    /// dataset with no missing sentinel.
    pub fn new(missing_markers: &[&str]) -> Self {
        Self {
            counts: BTreeMap::new(),
            missing_markers: missing_markers.iter().map(|s| s.to_string()).collect(),
            missing_count: 0,
            total: 0,
        }
    }

    /// Observe one raw cell value from the training split.
    pub fn observe(&mut self, raw: &str) {
        self.total += 1;
        if self.missing_markers.contains(raw) {
            self.missing_count += 1;
        } else if let Some(c) = self.counts.get_mut(raw) {
            *c += 1;
        } else {
            self.counts.insert(raw.to_string(), 1);
        }
    }

    /// Observe every value in a training column.
    pub fn observe_all<I, S>(&mut self, values: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for v in values {
            self.observe(v.as_ref());
        }
    }

    /// Freeze into an immutable [`CategoryDictionary`], applying `policy`.
    pub fn build(self, policy: RarePolicy) -> CategoryDictionary {
        CategoryDictionary::from_builder(self, policy)
    }
}

/// A frozen, deterministic per-feature category vocabulary.
///
/// Built once from the training split via [`CategoryDictionaryBuilder`]
/// and then immutable — the builder-consumes-into-dictionary transition
/// *is* the freeze. [`encode`](Self::encode) maps any raw cell value to
/// its `u32` code; never panics, never allocates.
#[derive(Debug, Clone)]
pub struct CategoryDictionary {
    /// Kept real category -> code (`>= FIRST_REAL_CODE`).
    codes: BTreeMap<String, u32>,
    /// Real categories folded into [`CODE_RARE`] (seen in training, but
    /// below the policy floors).
    rare: BTreeSet<String>,
    /// Raw strings treated as missing.
    missing_markers: BTreeSet<String>,
    /// Kept-category labels, indexed by `code - FIRST_REAL_CODE`.
    labels: Vec<String>,
    /// Training count per kept category, parallel to `labels`.
    counts: Vec<u64>,
    missing_count: u64,
    /// Total training occurrences folded into [`CODE_RARE`].
    rare_count: u64,
    /// Total training rows observed (including missing).
    total: u64,
    /// The folding policy used (retained for the schema snapshot).
    policy: RarePolicy,
}

impl CategoryDictionary {
    fn from_builder(b: CategoryDictionaryBuilder, policy: RarePolicy) -> Self {
        let total = b.total;
        // `count < min_frac * total` <=> `count / total < min_frac`.
        // Exact `as f64` casts (counts < 2^53) + one IEEE multiply +
        // one compare — deterministic across platforms.
        let frac_floor = policy.min_frac * (total as f64);

        let mut kept: Vec<(String, u64)> = Vec::new();
        let mut rare: BTreeSet<String> = BTreeSet::new();
        let mut rare_count: u64 = 0;
        for (label, count) in b.counts {
            let folds = count < policy.min_count || (count as f64) < frac_floor;
            if folds {
                rare_count += count;
                rare.insert(label);
            } else {
                kept.push((label, count));
            }
        }

        // Deterministic code order: descending count, ties lexicographic.
        kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut codes = BTreeMap::new();
        let mut labels = Vec::with_capacity(kept.len());
        let mut counts = Vec::with_capacity(kept.len());
        for (i, (label, count)) in kept.into_iter().enumerate() {
            codes.insert(label.clone(), FIRST_REAL_CODE + i as u32);
            labels.push(label);
            counts.push(count);
        }

        CategoryDictionary {
            codes,
            rare,
            missing_markers: b.missing_markers,
            labels,
            counts,
            missing_count: b.missing_count,
            rare_count,
            total,
            policy,
        }
    }

    /// Map a raw cell value to its code. Pure — never panics, never
    /// allocates. A missing-marker yields [`CODE_MISSING`]; a kept
    /// category yields its real code; a folded category yields
    /// [`CODE_RARE`]; anything else yields [`CODE_UNKNOWN`].
    pub fn encode(&self, raw: &str) -> u32 {
        if self.missing_markers.contains(raw) {
            CODE_MISSING
        } else if let Some(&code) = self.codes.get(raw) {
            code
        } else if self.rare.contains(raw) {
            CODE_RARE
        } else {
            CODE_UNKNOWN
        }
    }

    /// Number of kept (real, non-folded) categories.
    pub fn n_real(&self) -> u32 {
        self.labels.len() as u32
    }

    /// Distinct codes including the three reserved slots:
    /// `FIRST_REAL_CODE + n_real`.
    pub fn vocab_size(&self) -> u32 {
        FIRST_REAL_CODE + self.n_real()
    }

    /// Number of distinct training categories folded into [`CODE_RARE`].
    pub fn n_rare_folded(&self) -> usize {
        self.rare.len()
    }

    /// Total training rows observed (including missing).
    pub fn total_observed(&self) -> u64 {
        self.total
    }

    /// Training rows that were a missing-marker.
    pub fn missing_count(&self) -> u64 {
        self.missing_count
    }

    /// Total training occurrences across all folded rare categories.
    pub fn rare_count(&self) -> u64 {
        self.rare_count
    }

    /// The folding policy this dictionary was built with.
    pub fn policy(&self) -> RarePolicy {
        self.policy
    }

    /// Human-readable label for a code. Reserved codes get sentinel
    /// names; an out-of-range code gets `"<INVALID>"`.
    pub fn label(&self, code: u32) -> &str {
        match code {
            CODE_MISSING => "<MISSING>",
            CODE_UNKNOWN => "<UNKNOWN>",
            CODE_RARE => "<RARE>",
            c => {
                let idx = (c - FIRST_REAL_CODE) as usize;
                self.labels.get(idx).map(String::as_str).unwrap_or("<INVALID>")
            }
        }
    }

    /// Deterministic byte encoding of the whole dictionary — the input
    /// to the categorical-vocab hash in the schema snapshot. Two
    /// dictionaries built from the same training values **in any row
    /// order** produce byte-identical output.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Policy.
        buf.extend_from_slice(&self.policy.min_count.to_be_bytes());
        buf.extend_from_slice(&self.policy.min_frac.to_bits().to_be_bytes());
        // Totals.
        buf.extend_from_slice(&self.total.to_be_bytes());
        buf.extend_from_slice(&self.missing_count.to_be_bytes());
        buf.extend_from_slice(&self.rare_count.to_be_bytes());
        // Missing markers (BTreeSet — sorted iteration).
        buf.extend_from_slice(&(self.missing_markers.len() as u32).to_be_bytes());
        for m in &self.missing_markers {
            write_str(&mut buf, m);
        }
        // Real categories, in code order.
        buf.extend_from_slice(&self.n_real().to_be_bytes());
        for (i, label) in self.labels.iter().enumerate() {
            buf.extend_from_slice(&(FIRST_REAL_CODE + i as u32).to_be_bytes());
            buf.extend_from_slice(&self.counts[i].to_be_bytes());
            write_str(&mut buf, label);
        }
        // Folded rare categories (BTreeSet — sorted iteration).
        buf.extend_from_slice(&(self.rare.len() as u32).to_be_bytes());
        for r in &self.rare {
            write_str(&mut buf, r);
        }
        buf
    }

    /// SHA-256 of [`canonical_bytes`](Self::canonical_bytes) — the
    /// categorical-vocab hash for the Phase 0.9.5 schema snapshot.
    pub fn vocab_hash(&self) -> [u8; 32] {
        cjc_snap::hash::sha256(&self.canonical_bytes())
    }
}

/// Append a length-prefixed (`u32` BE) UTF-8 string to `buf`.
fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
    buf.extend_from_slice(s.as_bytes());
}

// ──────────────────────────────────────────────────────────────────
// Phase 0.9.5 COMMIT 2 — encoding modes + schema snapshot.
// ──────────────────────────────────────────────────────────────────

/// One-hot encoder for the predictive feature vector `phi`.
///
/// Maps a [`CategoryDictionary`] code to a slot in a one-hot vector.
/// Slots `0/1/2` are always MISSING/UNKNOWN/RARE; slots `3..` are real
/// categories in dictionary-code order (most-frequent first).
///
/// `max_real` caps how many real categories get their own slot — real
/// codes beyond the cap collapse into the RARE slot. This bounds `phi`
/// width for high-cardinality features (e.g. ICD-9 diagnosis codes) —
/// the explosion guard applied on the predictive side. Effect coding
/// is a documented Phase 1.0 alternative (`PHASE_0_9_5_HANDOFF.md` §9).
#[derive(Debug, Clone, Copy)]
pub struct OneHotEncoder {
    max_real: Option<u32>,
}

impl OneHotEncoder {
    /// An encoder with no cap — every real category gets its own slot.
    pub fn new() -> Self {
        Self { max_real: None }
    }

    /// An encoder that distinctly encodes at most `max_real` real
    /// categories; lower-frequency reals collapse into the RARE slot.
    pub fn with_max_real(max_real: u32) -> Self {
        Self {
            max_real: Some(max_real),
        }
    }

    /// One-hot width for `dict`: 3 reserved slots + (capped) reals.
    pub fn width(&self, dict: &CategoryDictionary) -> usize {
        let reals = match self.max_real {
            Some(m) => dict.n_real().min(m),
            None => dict.n_real(),
        };
        (FIRST_REAL_CODE + reals) as usize
    }

    /// The one-hot slot for `raw`. A real category beyond the
    /// `max_real` cap collapses into the RARE slot. Always `< width`.
    pub fn slot(&self, dict: &CategoryDictionary, raw: &str) -> usize {
        let code = dict.encode(raw);
        match self.max_real {
            Some(m) if code >= FIRST_REAL_CODE && code - FIRST_REAL_CODE >= m => {
                CODE_RARE as usize
            }
            _ => code as usize,
        }
    }

    /// Write the one-hot vector for `raw` into `out`: `out` is resized
    /// to [`width`](Self::width), zeroed, then a single `1.0` is set at
    /// [`slot`](Self::slot). Reuses the caller's buffer (the ABNG
    /// buffer-reuse convention).
    pub fn encode_into(&self, dict: &CategoryDictionary, raw: &str, out: &mut Vec<f64>) {
        let w = self.width(dict);
        out.clear();
        out.resize(w, 0.0);
        out[self.slot(dict, raw)] = 1.0;
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Ordinal routing bucket for a category code — the routing-side (`x`)
/// encoding.
///
/// Clamps a [`CategoryDictionary`] code into `0..route_bins`: the
/// reserved codes and the most-frequent real categories get distinct
/// buckets; lower-frequency reals beyond `route_bins - 1` all collapse
/// into the last bucket. Because dictionary codes are frequency-ranked,
/// this keeps the *common* categories distinct for routing and lumps
/// the tail — the route-explosion guard: a categorical feature
/// contributes at most `route_bins` distinct routing values.
pub fn route_bucket(code: u32, route_bins: u8) -> u8 {
    let last = (route_bins as u32).saturating_sub(1);
    code.min(last) as u8
}

/// Phase 0.9.5 feature-transform version. Bump on any change to how
/// raw rows become `(x, phi)`; the schema snapshot embeds it so a
/// replay detects a transform-pipeline mismatch.
pub const FEATURE_TRANSFORM_VERSION: u32 = 1;

/// The deterministic provenance bundle for one Phase 0.9.5 run.
///
/// Every field is a hash or a scalar fixed by the
/// `(dataset, split seed, transform)` triple.
/// [`snapshot_hash`](Self::snapshot_hash) over all of them is the
/// single value a replay checks: same dataset + same seed + same build
/// ⇒ identical snapshot hash. See `PHASE_0_9_5_HANDOFF.md` §3.6.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaSnapshot {
    /// SHA-256 of the raw source CSV bytes.
    pub raw_dataset_hash: [u8; 32],
    /// SHA-256 of the canonical column-name + role list.
    pub schema_hash: [u8; 32],
    /// Combined hash of every per-feature [`CategoryDictionary`]
    /// (see [`hash_vocabularies`]).
    pub categorical_vocab_hash: [u8; 32],
    /// SHA-256 of the per-numeric-column `(mean, std)` standardization
    /// statistics.
    pub numeric_standardization_hash: [u8; 32],
    /// [`FEATURE_TRANSFORM_VERSION`] at the time of the run.
    pub feature_transform_version: u32,
    /// The deterministic train/test split seed.
    pub split_seed: u64,
    /// Total dataset row count.
    pub row_count: u64,
    /// Human-readable target definition, e.g. `"readmitted == '<30'"`.
    pub target_definition: String,
}

impl SchemaSnapshot {
    /// Deterministic byte encoding — fixed field order, big-endian
    /// integers.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(160 + self.target_definition.len());
        buf.extend_from_slice(&self.raw_dataset_hash);
        buf.extend_from_slice(&self.schema_hash);
        buf.extend_from_slice(&self.categorical_vocab_hash);
        buf.extend_from_slice(&self.numeric_standardization_hash);
        buf.extend_from_slice(&self.feature_transform_version.to_be_bytes());
        buf.extend_from_slice(&self.split_seed.to_be_bytes());
        buf.extend_from_slice(&self.row_count.to_be_bytes());
        write_str(&mut buf, &self.target_definition);
        buf
    }

    /// SHA-256 of [`canonical_bytes`](Self::canonical_bytes) — the
    /// single provenance value a replay checks.
    pub fn snapshot_hash(&self) -> [u8; 32] {
        cjc_snap::hash::sha256(&self.canonical_bytes())
    }
}

/// Combine the per-feature category dictionaries into one deterministic
/// `categorical_vocab_hash` for a [`SchemaSnapshot`].
///
/// Features are sorted by name before hashing, so the result does not
/// depend on the order the dictionaries were built or supplied in.
pub fn hash_vocabularies(features: &[(&str, &CategoryDictionary)]) -> [u8; 32] {
    let mut sorted: Vec<&(&str, &CategoryDictionary)> = features.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut buf = Vec::new();
    buf.extend_from_slice(&(sorted.len() as u32).to_be_bytes());
    for (name, dict) in sorted {
        write_str(&mut buf, name);
        let cb = dict.canonical_bytes();
        buf.extend_from_slice(&(cb.len() as u32).to_be_bytes());
        buf.extend_from_slice(&cb);
    }
    cjc_snap::hash::sha256(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(values: &[&str], markers: &[&str], policy: RarePolicy) -> CategoryDictionary {
        let mut b = CategoryDictionaryBuilder::new(markers);
        b.observe_all(values.iter().copied());
        b.build(policy)
    }

    #[test]
    fn reserved_codes_are_fixed() {
        assert_eq!(CODE_MISSING, 0);
        assert_eq!(CODE_UNKNOWN, 1);
        assert_eq!(CODE_RARE, 2);
        assert_eq!(FIRST_REAL_CODE, 3);
    }

    #[test]
    fn codes_assigned_by_descending_frequency_then_label() {
        // "b" x3 (most frequent); "a" x2 and "c" x2 tie, broken
        // lexicographically -> a before c.
        let d = build(&["b", "b", "b", "a", "a", "c", "c"], &[], RarePolicy::KEEP_ALL);
        assert_eq!(d.encode("b"), FIRST_REAL_CODE);
        assert_eq!(d.encode("a"), FIRST_REAL_CODE + 1);
        assert_eq!(d.encode("c"), FIRST_REAL_CODE + 2);
    }

    #[test]
    fn row_order_invariance() {
        let p = RarePolicy::KEEP_ALL;
        let a = build(&["x", "y", "y", "z", "z", "z"], &["?"], p);
        // Same multiset, shuffled order.
        let b = build(&["z", "y", "z", "x", "z", "y"], &["?"], p);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.vocab_hash(), b.vocab_hash());
    }

    #[test]
    fn missing_marker_encodes_to_code_missing() {
        let d = build(&["a", "a", "?", "a"], &["?"], RarePolicy::KEEP_ALL);
        assert_eq!(d.encode("?"), CODE_MISSING);
        assert_eq!(d.missing_count(), 1);
        assert_eq!(d.total_observed(), 4);
    }

    #[test]
    fn unknown_category_encodes_to_code_unknown() {
        let d = build(&["a", "a", "b"], &[], RarePolicy::KEEP_ALL);
        assert_eq!(d.encode("never-seen-in-training"), CODE_UNKNOWN);
    }

    #[test]
    fn encode_never_panics_on_arbitrary_input() {
        let d = build(&["a"], &[], RarePolicy::KEEP_ALL);
        let _ = d.encode("");
        let _ = d.encode("\u{1F600}\u{0000}\t\n");
        let _ = d.encode(&"z".repeat(10_000));
    }

    #[test]
    fn rare_category_folds_by_min_count() {
        let mut vals: Vec<&str> = vec!["common"; 50];
        vals.extend(vec!["rare"; 5]);
        let d = build(&vals, &[], RarePolicy { min_count: 30, min_frac: 0.0 });
        assert_eq!(d.encode("common"), FIRST_REAL_CODE);
        assert_eq!(d.encode("rare"), CODE_RARE);
        assert_eq!(d.n_rare_folded(), 1);
        assert_eq!(d.rare_count(), 5);
    }

    #[test]
    fn rare_category_folds_by_min_frac() {
        // "rare" is 10/1000 = 1% — below a 5% min_frac floor.
        let mut vals: Vec<&str> = vec!["common"; 990];
        vals.extend(vec!["rare"; 10]);
        let d = build(&vals, &[], RarePolicy { min_count: 0, min_frac: 0.05 });
        assert_eq!(d.encode("rare"), CODE_RARE);
        assert_eq!(d.encode("common"), FIRST_REAL_CODE);
    }

    #[test]
    fn rare_folds_when_either_floor_fails() {
        // "mid" passes the count floor (40 >= 30) but fails the frac
        // floor (40/1000 = 0.04 < 0.10) -> folds.
        let mut vals: Vec<&str> = vec!["common"; 960];
        vals.extend(vec!["mid"; 40]);
        let d = build(&vals, &[], RarePolicy { min_count: 30, min_frac: 0.10 });
        assert_eq!(d.encode("mid"), CODE_RARE);
    }

    #[test]
    fn keep_all_policy_folds_nothing() {
        let d = build(&["a", "b", "c"], &[], RarePolicy::KEEP_ALL);
        assert_eq!(d.n_rare_folded(), 0);
        assert!(d.encode("a") >= FIRST_REAL_CODE);
    }

    #[test]
    fn vocab_size_is_reserved_plus_real() {
        let d = build(&["a", "b", "c"], &[], RarePolicy::KEEP_ALL);
        assert_eq!(d.n_real(), 3);
        assert_eq!(d.vocab_size(), FIRST_REAL_CODE + 3);
    }

    #[test]
    fn empty_builder_has_no_real_categories() {
        let d = build(&[], &["?"], RarePolicy::DEFAULT);
        assert_eq!(d.n_real(), 0);
        assert_eq!(d.vocab_size(), FIRST_REAL_CODE);
        assert_eq!(d.encode("anything"), CODE_UNKNOWN);
    }

    #[test]
    fn canonical_bytes_change_with_vocabulary() {
        let a = build(&["a", "b"], &[], RarePolicy::KEEP_ALL);
        let b = build(&["a", "b", "c"], &[], RarePolicy::KEEP_ALL);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn canonical_bytes_change_with_policy() {
        let mut vals: Vec<&str> = vec!["common"; 50];
        vals.extend(vec!["rare"; 5]);
        let keep = build(&vals, &[], RarePolicy::KEEP_ALL);
        let fold = build(&vals, &[], RarePolicy { min_count: 30, min_frac: 0.0 });
        assert_ne!(keep.canonical_bytes(), fold.canonical_bytes());
    }

    #[test]
    fn encode_is_pure() {
        let d = build(&["a", "a", "b"], &["?"], RarePolicy::KEEP_ALL);
        for v in ["a", "b", "?", "unknown", ""] {
            assert_eq!(d.encode(v), d.encode(v));
        }
    }

    #[test]
    fn label_round_trips_for_real_and_reserved_codes() {
        let d = build(&["b", "b", "b", "a", "a"], &[], RarePolicy::KEEP_ALL);
        assert_eq!(d.label(d.encode("a")), "a");
        assert_eq!(d.label(d.encode("b")), "b");
        assert_eq!(d.label(CODE_MISSING), "<MISSING>");
        assert_eq!(d.label(CODE_UNKNOWN), "<UNKNOWN>");
        assert_eq!(d.label(CODE_RARE), "<RARE>");
        assert_eq!(d.label(9_999), "<INVALID>");
    }

    #[test]
    fn missing_markers_excluded_from_real_vocabulary() {
        // "?" is a marker — it must not become a real category even
        // though it is observed many times.
        let d = build(&["?", "?", "?", "a", "a"], &["?"], RarePolicy::KEEP_ALL);
        assert_eq!(d.n_real(), 1);
        assert_eq!(d.encode("?"), CODE_MISSING);
        assert_eq!(d.encode("a"), FIRST_REAL_CODE);
    }

    // ── COMMIT 2 — encoding modes + schema snapshot ──────────────────

    fn sample_snapshot() -> SchemaSnapshot {
        SchemaSnapshot {
            raw_dataset_hash: [1u8; 32],
            schema_hash: [2u8; 32],
            categorical_vocab_hash: [3u8; 32],
            numeric_standardization_hash: [4u8; 32],
            feature_transform_version: FEATURE_TRANSFORM_VERSION,
            split_seed: 42,
            row_count: 101_766,
            target_definition: "readmitted == '<30'".to_string(),
        }
    }

    #[test]
    fn one_hot_width_is_reserved_plus_reals() {
        let d = build(&["a", "b", "c"], &[], RarePolicy::KEEP_ALL);
        assert_eq!(
            OneHotEncoder::new().width(&d),
            (FIRST_REAL_CODE + 3) as usize
        );
    }

    #[test]
    fn one_hot_slot_equals_code_when_uncapped() {
        let d = build(&["b", "b", "a"], &[], RarePolicy::KEEP_ALL);
        let e = OneHotEncoder::new();
        assert_eq!(e.slot(&d, "b"), d.encode("b") as usize);
        assert_eq!(e.slot(&d, "a"), d.encode("a") as usize);
    }

    #[test]
    fn one_hot_reserved_slots() {
        let d = build(&["a", "a", "?"], &["?"], RarePolicy::KEEP_ALL);
        let e = OneHotEncoder::new();
        assert_eq!(e.slot(&d, "?"), CODE_MISSING as usize);
        assert_eq!(e.slot(&d, "never-seen"), CODE_UNKNOWN as usize);
    }

    #[test]
    fn one_hot_cap_collapses_tail_to_rare() {
        // 4 real categories; cap at 2 -> codes 3,4 keep distinct slots,
        // codes 5,6 collapse into the RARE slot.
        let d = build(
            &["a", "a", "a", "a", "b", "b", "b", "c", "c", "d"],
            &[],
            RarePolicy::KEEP_ALL,
        );
        let e = OneHotEncoder::with_max_real(2);
        assert_eq!(e.width(&d), (FIRST_REAL_CODE + 2) as usize);
        assert_eq!(e.slot(&d, "a"), FIRST_REAL_CODE as usize);
        assert_eq!(e.slot(&d, "b"), (FIRST_REAL_CODE + 1) as usize);
        assert_eq!(e.slot(&d, "c"), CODE_RARE as usize);
        assert_eq!(e.slot(&d, "d"), CODE_RARE as usize);
    }

    #[test]
    fn one_hot_slot_always_within_width() {
        let d = build(&["a", "a", "b", "c"], &["?"], RarePolicy::KEEP_ALL);
        for e in [
            OneHotEncoder::new(),
            OneHotEncoder::with_max_real(1),
            OneHotEncoder::with_max_real(0),
        ] {
            let w = e.width(&d);
            for raw in ["a", "b", "c", "?", "unknown", ""] {
                assert!(e.slot(&d, raw) < w, "slot out of bounds for {raw:?}");
            }
        }
    }

    #[test]
    fn one_hot_encode_into_sets_single_one_and_reuses_buffer() {
        let d = build(&["a", "a", "b"], &[], RarePolicy::KEEP_ALL);
        let e = OneHotEncoder::new();
        let mut buf = vec![9.0; 99]; // pre-filled garbage
        e.encode_into(&d, "a", &mut buf);
        assert_eq!(buf.len(), e.width(&d));
        assert_eq!(buf.iter().sum::<f64>(), 1.0);
        assert_eq!(buf[e.slot(&d, "a")], 1.0);
    }

    #[test]
    fn route_bucket_clamps_and_keeps_frequent_distinct() {
        assert_eq!(route_bucket(CODE_MISSING, 4), 0);
        assert_eq!(route_bucket(CODE_UNKNOWN, 4), 1);
        assert_eq!(route_bucket(CODE_RARE, 4), 2);
        assert_eq!(route_bucket(FIRST_REAL_CODE, 4), 3);
        assert_eq!(route_bucket(FIRST_REAL_CODE + 9, 4), 3); // tail lumps
    }

    #[test]
    fn route_bucket_wider_bins_keep_more_distinct() {
        assert_eq!(route_bucket(FIRST_REAL_CODE + 3, 8), FIRST_REAL_CODE as u8 + 3);
        assert_eq!(route_bucket(FIRST_REAL_CODE + 99, 8), 7); // last bucket
    }

    #[test]
    fn schema_snapshot_canonical_bytes_deterministic() {
        let s = sample_snapshot();
        assert_eq!(s.canonical_bytes(), s.clone().canonical_bytes());
        assert_eq!(s.snapshot_hash(), s.snapshot_hash());
    }

    #[test]
    fn schema_snapshot_hash_changes_with_every_field() {
        let base = sample_snapshot();
        let h = base.snapshot_hash();
        let mut v;
        v = base.clone();
        v.raw_dataset_hash[0] ^= 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.schema_hash[0] ^= 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.categorical_vocab_hash[0] ^= 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.numeric_standardization_hash[0] ^= 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.feature_transform_version += 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.split_seed += 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.row_count += 1;
        assert_ne!(v.snapshot_hash(), h);
        v = base.clone();
        v.target_definition.push('!');
        assert_ne!(v.snapshot_hash(), h);
    }

    #[test]
    fn hash_vocabularies_is_feature_order_independent() {
        let da = build(&["x", "x", "y"], &[], RarePolicy::KEEP_ALL);
        let db = build(&["p", "q", "q"], &[], RarePolicy::KEEP_ALL);
        let ab = hash_vocabularies(&[("alpha", &da), ("beta", &db)]);
        let ba = hash_vocabularies(&[("beta", &db), ("alpha", &da)]);
        assert_eq!(ab, ba);
    }

    #[test]
    fn hash_vocabularies_changes_when_a_dictionary_changes() {
        let da = build(&["x", "x", "y"], &[], RarePolicy::KEEP_ALL);
        let db = build(&["p", "q", "q"], &[], RarePolicy::KEEP_ALL);
        let dc = build(&["p", "q", "q", "r"], &[], RarePolicy::KEEP_ALL);
        assert_ne!(
            hash_vocabularies(&[("a", &da), ("b", &db)]),
            hash_vocabularies(&[("a", &da), ("b", &dc)]),
        );
    }
}
