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

use cjc_repro::{kahan_sum_f64, KahanAccumulatorF64};

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

// ──────────────────────────────────────────────────────────────────
// Phase 0.9.5 COMMIT 4 — CategoricalTransform: raw rows -> (x, phi, y).
// ──────────────────────────────────────────────────────────────────

/// How one source column is consumed by a [`CategoricalTransform`].
///
/// The two `*PhiOnly` roles are the Phase 0.9.5 §4 route-explosion hard
/// guard: a high-cardinality nominal column (ICD-9 `diag_1/2/3`, the 23
/// medication columns) feeds the predictive vector `phi` but is **never**
/// eligible as a routing feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRole {
    /// The binary target / label column.
    Target,
    /// Dropped entirely — identifiers and leakage-risk columns.
    Ignore,
    /// Categorical: one-hot encoded into `phi`, eligible for routing.
    Categorical,
    /// Categorical, one-hot into `phi` only — never a routing feature.
    CategoricalPhiOnly,
    /// Numeric: standardized into `phi`, eligible for routing.
    Numeric,
    /// Numeric, standardized into `phi` only — never a routing feature.
    NumericPhiOnly,
}

impl ColumnRole {
    /// True if a column with this role may be picked as a routing feature.
    fn is_routing_candidate(self) -> bool {
        matches!(self, ColumnRole::Categorical | ColumnRole::Numeric)
    }
}

/// One source column's tag byte for the schema hash.
fn role_tag(role: ColumnRole) -> u8 {
    match role {
        ColumnRole::Target => 0,
        ColumnRole::Ignore => 1,
        ColumnRole::Categorical => 2,
        ColumnRole::CategoricalPhiOnly => 3,
        ColumnRole::Numeric => 4,
        ColumnRole::NumericPhiOnly => 5,
    }
}

/// The ordered column layout of a dataset — one `(name, role)` per
/// column, in raw row order (CSV column order).
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    columns: Vec<(String, ColumnRole)>,
}

impl Schema {
    /// Build a schema from `(name, role)` pairs in column order.
    pub fn new(columns: Vec<(String, ColumnRole)>) -> Self {
        Self { columns }
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// True if the schema has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The role of column `idx`.
    pub fn role(&self, idx: usize) -> ColumnRole {
        self.columns[idx].1
    }

    /// The name of column `idx`.
    pub fn name(&self, idx: usize) -> &str {
        &self.columns[idx].0
    }

    /// Index of the sole [`ColumnRole::Target`] column.
    fn target_index(&self) -> Result<usize, TransformError> {
        let mut found: Option<usize> = None;
        for (i, (_, role)) in self.columns.iter().enumerate() {
            if *role == ColumnRole::Target {
                if found.is_some() {
                    return Err(TransformError::MultipleTargets);
                }
                found = Some(i);
            }
        }
        found.ok_or(TransformError::NoTarget)
    }

    /// Deterministic canonical bytes — column count, then each name
    /// (length-prefixed) followed by its role tag, in column order.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.columns.len() as u32).to_be_bytes());
        for (name, role) in &self.columns {
            write_str(&mut buf, name);
            buf.push(role_tag(*role));
        }
        buf
    }
}

/// Errors from [`CategoricalTransform::fit`] / [`CategoricalTransform::transform`].
#[derive(Debug, Clone, PartialEq)]
pub enum TransformError {
    /// The schema has no columns.
    EmptySchema,
    /// The schema has no `Target` column.
    NoTarget,
    /// The schema has more than one `Target` column.
    MultipleTargets,
    /// `route_bins` is not a power of two in `[2, 128]`.
    BadRouteBins(u8),
    /// A row's cell count does not match the schema's column count.
    RowArityMismatch {
        /// The schema's column count.
        expected: usize,
        /// The offending row's cell count.
        got: usize,
    },
    /// A row's target cell is empty or a missing-marker.
    MissingTarget,
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::EmptySchema => {
                write!(f, "categorical transform: schema has no columns")
            }
            TransformError::NoTarget => {
                write!(f, "categorical transform: schema has no Target column")
            }
            TransformError::MultipleTargets => write!(
                f,
                "categorical transform: schema has more than one Target column"
            ),
            TransformError::BadRouteBins(n) => write!(
                f,
                "categorical transform: route_bins must be a power of two in [2, 128], got {n}"
            ),
            TransformError::RowArityMismatch { expected, got } => write!(
                f,
                "categorical transform: row has {got} cells, schema has {expected} columns"
            ),
            TransformError::MissingTarget => {
                write!(f, "categorical transform: row has a missing target cell")
            }
        }
    }
}

/// Per-column z-score standardization statistics, fit on the **training
/// split only** (Phase 0.9.5 directive 6 — no test-split leakage).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Standardizer {
    /// Training-split column mean.
    pub mean: f64,
    /// Training-split column standard deviation, floored at `1e-12` so a
    /// constant column cannot divide by zero.
    pub std: f64,
}

impl Standardizer {
    /// Fit `(mean, std)` over `values` — population variance (`1/n`
    /// divisor). The values are sorted before summation so the result is
    /// invariant to training-row order, and both reductions use Kahan
    /// compensated summation (per the determinism rules). An empty input
    /// yields the identity standardizer `(0.0, 1.0)`.
    pub fn fit(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                std: 1.0,
            };
        }
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mean = kahan_sum_f64(&sorted) / (n as f64);
        let mut var_acc = KahanAccumulatorF64::new();
        for &v in &sorted {
            let d = v - mean;
            var_acc.add(d * d);
        }
        let std = (var_acc.finalize() / (n as f64)).sqrt().max(1e-12);
        Self { mean, std }
    }

    /// Standardize one value: `(v - mean) / std`.
    pub fn apply(&self, v: f64) -> f64 {
        (v - self.mean) / self.std
    }
}

/// Mutual information `I(F; Y)` in nats between a discrete feature `F`
/// (bucket indices `0..n_buckets`) and a binary target `Y` (`0` / `1`),
/// estimated from exact co-occurrence counts.
///
/// Deterministic: counts are integers, the joint table is walked in a
/// fixed `(bucket, y)` order, and the running sum is Kahan-compensated.
/// Returns `0.0` for an empty / length-mismatched input.
fn mutual_information(buckets: &[u8], target: &[u8], n_buckets: usize) -> f64 {
    let n = buckets.len();
    if n == 0 || n != target.len() {
        return 0.0;
    }
    let mut joint: Vec<[u64; 2]> = vec![[0, 0]; n_buckets];
    let mut margin_y = [0u64; 2];
    for (&b, &y) in buckets.iter().zip(target.iter()) {
        let bi = b as usize;
        if bi >= n_buckets {
            continue;
        }
        let yi = (y != 0) as usize;
        joint[bi][yi] += 1;
        margin_y[yi] += 1;
    }
    let total = (margin_y[0] + margin_y[1]) as f64;
    if total == 0.0 {
        return 0.0;
    }
    let mut acc = KahanAccumulatorF64::new();
    for counts in &joint {
        let nb = counts[0] + counts[1];
        if nb == 0 {
            continue;
        }
        let p_b = (nb as f64) / total;
        for yi in 0..2 {
            let nby = counts[yi];
            if nby == 0 {
                continue;
            }
            let p_by = (nby as f64) / total;
            let p_y = (margin_y[yi] as f64) / total;
            acc.add(p_by * (p_by / (p_b * p_y)).ln());
        }
    }
    // MI is non-negative; clamp tiny round-off below zero.
    acc.finalize().max(0.0)
}

/// `route_bins - 1` quantile cut points for a numeric column, from its
/// training values. Cut `k` is the `k / route_bins` quantile. A
/// low-variety column can yield repeated cuts — [`numeric_bucket`]'s
/// `partition_point` lookup tolerates that. Sorting makes the result
/// invariant to training-row order.
fn quantile_cuts(values: &[f64], route_bins: u8) -> Vec<f64> {
    let want = route_bins as usize - 1;
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n == 0 {
        return vec![0.0; want];
    }
    let mut cuts = Vec::with_capacity(want);
    for k in 1..route_bins as usize {
        let q = (k as f64) / (route_bins as f64);
        let idx = ((q * n as f64) as usize).min(n - 1);
        cuts.push(sorted[idx]);
    }
    cuts
}

/// Routing bucket for a numeric value: the count of cut points strictly
/// below it, clamped into `0..route_bins`. Mirrors the codebook's
/// `partition_point(|b| b < v)` binning convention.
fn numeric_bucket(value: f64, cuts: &[f64], route_bins: u8) -> u8 {
    let b = cuts.partition_point(|&c| c < value);
    b.min(route_bins as usize - 1) as u8
}

/// Parse a numeric cell. `None` for a missing-marker or an unparseable
/// string — the caller imputes those to the training-split column mean.
fn parse_numeric(raw: &str, markers: &[&str]) -> Option<f64> {
    if markers.contains(&raw) {
        return None;
    }
    match raw.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

/// Map a raw target cell to a `0` / `1` label byte. A missing-marker
/// target counts as `0` here — used only for MI feature selection;
/// `transform` rejects a missing target rather than mislabelling it.
fn binarise_target(raw: &str, positives: &BTreeSet<String>, markers: &[&str]) -> u8 {
    if !markers.contains(&raw) && positives.contains(raw) {
        1
    } else {
        0
    }
}

/// Deterministic hash of every numeric column's `(mean, std)` — the
/// `numeric_standardization_hash` for the schema snapshot. Columns are
/// taken in schema order; `f64`s via `to_bits`.
fn hash_standardizers(schema: &Schema, columns: &[ColumnTransform]) -> [u8; 32] {
    let mut buf = Vec::new();
    for (c, ct) in columns.iter().enumerate() {
        if let ColumnTransform::Numeric { stdz } = ct {
            write_str(&mut buf, schema.name(c));
            buf.extend_from_slice(&stdz.mean.to_bits().to_be_bytes());
            buf.extend_from_slice(&stdz.std.to_bits().to_be_bytes());
        }
    }
    cjc_snap::hash::sha256(&buf)
}

/// One selected routing feature: a source column plus the rule that
/// derives its `0..route_bins` routing bucket from a raw cell.
#[derive(Debug, Clone, PartialEq)]
struct RoutingFeature {
    /// Original column index.
    col: usize,
    /// Bucketing rule.
    kind: RoutingKind,
}

/// How a [`RoutingFeature`] derives its routing bucket.
#[derive(Debug, Clone, PartialEq)]
enum RoutingKind {
    /// Categorical: `route_bucket(dict.encode(cell), route_bins)`, where
    /// `dict` is the column's entry in `CategoricalTransform::columns`.
    Categorical,
    /// Numeric: [`numeric_bucket`] of the (mean-imputed) value against
    /// these `route_bins - 1` quantile cut points.
    Numeric { cuts: Vec<f64> },
}

/// Per-column fitted state inside a [`CategoricalTransform`].
#[derive(Debug, Clone)]
enum ColumnTransform {
    /// The target column — carries the positive-label set.
    Target { positives: BTreeSet<String> },
    /// A dropped column.
    Ignore,
    /// A categorical column: frozen dictionary + one-hot encoder.
    Categorical {
        dict: CategoryDictionary,
        encoder: OneHotEncoder,
    },
    /// A numeric column: its standardizer.
    Numeric { stdz: Standardizer },
}

/// Configuration for [`CategoricalTransform::fit`].
///
/// The Phase 0.9.5 defaults (`route_bins = 4`, `k_routing = 4`,
/// `max_real = 32`, [`RarePolicy::DEFAULT`]) come from the handoff §9.
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// Routing buckets per routing feature — the codebook `n_bins`.
    /// Must be a power of two in `[2, 128]`.
    pub route_bins: u8,
    /// How many routing features to select by mutual information.
    /// Clamped to the number of routing candidates in the schema.
    pub k_routing: usize,
    /// One-hot width cap for wide categorical columns — the `phi`-side
    /// explosion guard (see [`OneHotEncoder::with_max_real`]).
    pub max_real: u32,
    /// Rare-category folding policy applied to every dictionary.
    pub rare_policy: RarePolicy,
    /// Raw cell strings treated as missing.
    pub missing_markers: Vec<String>,
    /// Target cell values that map to the positive class (`y = 1.0`);
    /// every other non-missing value is negative (`y = 0.0`).
    pub target_positives: Vec<String>,
    /// Human-readable target definition for the schema snapshot.
    pub target_definition: String,
    /// SHA-256 of the raw source CSV bytes (supplied by the harness).
    pub raw_dataset_hash: [u8; 32],
    /// The train/test split seed.
    pub split_seed: u64,
    /// Total dataset row count (train + test).
    pub row_count: u64,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            route_bins: 4,
            k_routing: 4,
            max_real: 32,
            rare_policy: RarePolicy::DEFAULT,
            missing_markers: vec![String::from("?"), String::new()],
            target_positives: Vec::new(),
            target_definition: String::new(),
            raw_dataset_hash: [0u8; 32],
            split_seed: 0,
            row_count: 0,
        }
    }
}

/// A fitted categorical preprocessing pipeline — turns raw heterogeneous
/// string rows into the `(x, phi, y)` triples ABNG's
/// `train_step(x, phi, y)` consumes.
///
/// Built once from the **training split** via [`fit`](Self::fit): the
/// dictionaries, standardization statistics, and routing-feature
/// selection are a pure function of the train rows (Phase 0.9.5
/// directive 6 — leakage-free). The result is immutable.
#[derive(Debug, Clone)]
pub struct CategoricalTransform {
    /// Per-column fitted state, indexed by original column position.
    columns: Vec<ColumnTransform>,
    /// Index of the `Target` column.
    target_col: usize,
    /// Selected routing features (the `x` layout), in MI-rank order.
    routing: Vec<RoutingFeature>,
    /// Total `phi` width.
    phi_width: usize,
    /// Routing bucket count per `x` dimension (the codebook `n_bins`).
    route_bins: u8,
    /// Raw cell strings treated as missing (for numeric parsing in
    /// `transform`).
    missing_markers: Vec<String>,
    /// Deterministic provenance bundle.
    snapshot: SchemaSnapshot,
}

impl CategoricalTransform {
    /// Fit the transform on the **training split**.
    ///
    /// `train_rows` are the raw cell rows of the train split only; each
    /// must have exactly one cell per `schema` column. Builds a
    /// dictionary per categorical column and a [`Standardizer`] per
    /// numeric column, selects the top-`k_routing` routing features by
    /// mutual information with the target, and assembles the
    /// [`SchemaSnapshot`].
    ///
    /// # Determinism
    ///
    /// `fit` is a pure function of the schema, the *multiset* of train
    /// rows, and the config: dictionaries are frequency+lexically sorted
    /// (COMMIT 1), standardization sums over sorted values, and MI
    /// selection scores order-free counts — so a permutation of the
    /// train rows yields a byte-identical transform and snapshot.
    pub fn fit(
        schema: &Schema,
        train_rows: &[Vec<String>],
        config: &TransformConfig,
    ) -> Result<Self, TransformError> {
        if schema.is_empty() {
            return Err(TransformError::EmptySchema);
        }
        let target_col = schema.target_index()?;
        if !matches!(config.route_bins, 2 | 4 | 8 | 16 | 32 | 64 | 128) {
            return Err(TransformError::BadRouteBins(config.route_bins));
        }
        let n_cols = schema.len();
        for row in train_rows {
            if row.len() != n_cols {
                return Err(TransformError::RowArityMismatch {
                    expected: n_cols,
                    got: row.len(),
                });
            }
        }

        let markers: Vec<&str> = config.missing_markers.iter().map(String::as_str).collect();

        // ── Per-column fitted state (train rows only). ───────────────
        let mut columns: Vec<ColumnTransform> = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let ct = match schema.role(c) {
                ColumnRole::Target => ColumnTransform::Target {
                    positives: config.target_positives.iter().cloned().collect(),
                },
                ColumnRole::Ignore => ColumnTransform::Ignore,
                ColumnRole::Categorical | ColumnRole::CategoricalPhiOnly => {
                    let mut b = CategoryDictionaryBuilder::new(&markers);
                    for row in train_rows {
                        b.observe(row[c].as_str());
                    }
                    ColumnTransform::Categorical {
                        dict: b.build(config.rare_policy),
                        encoder: OneHotEncoder::with_max_real(config.max_real),
                    }
                }
                ColumnRole::Numeric | ColumnRole::NumericPhiOnly => {
                    let mut vals: Vec<f64> = Vec::new();
                    for row in train_rows {
                        if let Some(v) = parse_numeric(row[c].as_str(), &markers) {
                            vals.push(v);
                        }
                    }
                    ColumnTransform::Numeric {
                        stdz: Standardizer::fit(&vals),
                    }
                }
            };
            columns.push(ct);
        }

        // ── phi width — sum of every column's phi contribution. ──────
        let mut phi_width = 0usize;
        for ct in &columns {
            phi_width += match ct {
                ColumnTransform::Categorical { dict, encoder } => encoder.width(dict),
                ColumnTransform::Numeric { .. } => 1,
                ColumnTransform::Target { .. } | ColumnTransform::Ignore => 0,
            };
        }

        // ── Binarised train target (for MI selection). ───────────────
        let target: Vec<u8> = match &columns[target_col] {
            ColumnTransform::Target { positives } => train_rows
                .iter()
                .map(|row| binarise_target(row[target_col].as_str(), positives, &markers))
                .collect(),
            _ => unreachable!("target_index points at the Target column"),
        };

        // ── MI-based routing-feature selection. ──────────────────────
        // Each routing candidate is scored by I(routing-bucket; target)
        // — exactly the signal that reaches `x`. Sort (MI desc, col asc).
        let mut scored: Vec<(usize, f64, Option<Vec<f64>>)> = Vec::new();
        for c in 0..n_cols {
            if !schema.role(c).is_routing_candidate() {
                continue;
            }
            let (mi, cuts) = match &columns[c] {
                ColumnTransform::Categorical { dict, .. } => {
                    let buckets: Vec<u8> = train_rows
                        .iter()
                        .map(|row| route_bucket(dict.encode(row[c].as_str()), config.route_bins))
                        .collect();
                    let mi = mutual_information(&buckets, &target, config.route_bins as usize);
                    (mi, None)
                }
                ColumnTransform::Numeric { stdz } => {
                    let raw: Vec<f64> = train_rows
                        .iter()
                        .map(|row| parse_numeric(row[c].as_str(), &markers).unwrap_or(stdz.mean))
                        .collect();
                    let cuts = quantile_cuts(&raw, config.route_bins);
                    let buckets: Vec<u8> = raw
                        .iter()
                        .map(|&v| numeric_bucket(v, &cuts, config.route_bins))
                        .collect();
                    let mi = mutual_information(&buckets, &target, config.route_bins as usize);
                    (mi, Some(cuts))
                }
                _ => unreachable!("a routing candidate is Categorical or Numeric"),
            };
            scored.push((c, mi, cuts));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let k = config.k_routing.min(scored.len());
        let routing: Vec<RoutingFeature> = scored
            .into_iter()
            .take(k)
            .map(|(col, _mi, cuts)| RoutingFeature {
                col,
                kind: match cuts {
                    Some(cuts) => RoutingKind::Numeric { cuts },
                    None => RoutingKind::Categorical,
                },
            })
            .collect();

        // ── Schema snapshot. ─────────────────────────────────────────
        let categorical_vocab_hash = {
            let mut vocab_pairs: Vec<(&str, &CategoryDictionary)> = Vec::new();
            for (c, ct) in columns.iter().enumerate() {
                if let ColumnTransform::Categorical { dict, .. } = ct {
                    vocab_pairs.push((schema.name(c), dict));
                }
            }
            hash_vocabularies(&vocab_pairs)
        };
        let numeric_standardization_hash = hash_standardizers(schema, &columns);
        let snapshot = SchemaSnapshot {
            raw_dataset_hash: config.raw_dataset_hash,
            schema_hash: cjc_snap::hash::sha256(&schema.canonical_bytes()),
            categorical_vocab_hash,
            numeric_standardization_hash,
            feature_transform_version: FEATURE_TRANSFORM_VERSION,
            split_seed: config.split_seed,
            row_count: config.row_count,
            target_definition: config.target_definition.clone(),
        };

        Ok(Self {
            columns,
            target_col,
            routing,
            phi_width,
            route_bins: config.route_bins,
            missing_markers: config.missing_markers.clone(),
            snapshot,
        })
    }

    /// Transform one raw row into `(x, phi, y)`:
    ///
    /// * `x` — one routing-bucket value (in `0.0..route_bins`) per
    ///   selected routing feature, in MI-rank order; feeds the codebook.
    /// * `phi` — concatenated one-hot (categorical, `max_real`-capped)
    ///   and standardized-numeric features, in column order; feeds the
    ///   leaf BLR. Length is always [`phi_width`](Self::phi_width).
    /// * `y` — `1.0` if the target cell is a positive value, else `0.0`.
    ///
    /// A missing or unparseable numeric cell is imputed to its column's
    /// training-split mean. Errors on a row-arity mismatch or a missing
    /// target cell.
    pub fn transform(&self, row: &[String]) -> Result<(Vec<f64>, Vec<f64>, f64), TransformError> {
        if row.len() != self.columns.len() {
            return Err(TransformError::RowArityMismatch {
                expected: self.columns.len(),
                got: row.len(),
            });
        }
        let markers: Vec<&str> = self.missing_markers.iter().map(String::as_str).collect();

        // ── y — the binary label. ────────────────────────────────────
        let target_cell = row[self.target_col].as_str();
        if markers.contains(&target_cell) {
            return Err(TransformError::MissingTarget);
        }
        let positives = match &self.columns[self.target_col] {
            ColumnTransform::Target { positives } => positives,
            _ => unreachable!("target_col points at the Target column"),
        };
        let y = if positives.contains(target_cell) {
            1.0
        } else {
            0.0
        };

        // ── x — one routing bucket per selected feature. ─────────────
        let mut x: Vec<f64> = Vec::with_capacity(self.routing.len());
        for rf in &self.routing {
            let cell = row[rf.col].as_str();
            let bucket: u8 = match &rf.kind {
                RoutingKind::Categorical => {
                    let dict = match &self.columns[rf.col] {
                        ColumnTransform::Categorical { dict, .. } => dict,
                        _ => unreachable!("RoutingKind::Categorical column is Categorical"),
                    };
                    route_bucket(dict.encode(cell), self.route_bins)
                }
                RoutingKind::Numeric { cuts } => {
                    let mean = match &self.columns[rf.col] {
                        ColumnTransform::Numeric { stdz } => stdz.mean,
                        _ => unreachable!("RoutingKind::Numeric column is Numeric"),
                    };
                    let v = parse_numeric(cell, &markers).unwrap_or(mean);
                    numeric_bucket(v, cuts, self.route_bins)
                }
            };
            x.push(bucket as f64);
        }

        // ── phi — one-hot + standardized numeric, in column order. ───
        let mut phi: Vec<f64> = Vec::with_capacity(self.phi_width);
        for (c, ct) in self.columns.iter().enumerate() {
            match ct {
                ColumnTransform::Categorical { dict, encoder } => {
                    let base = phi.len();
                    let width = encoder.width(dict);
                    phi.resize(base + width, 0.0);
                    phi[base + encoder.slot(dict, row[c].as_str())] = 1.0;
                }
                ColumnTransform::Numeric { stdz } => {
                    let v = parse_numeric(row[c].as_str(), &markers).unwrap_or(stdz.mean);
                    phi.push(stdz.apply(v));
                }
                ColumnTransform::Target { .. } | ColumnTransform::Ignore => {}
            }
        }

        Ok((x, phi, y))
    }

    /// The deterministic provenance bundle for this fit.
    pub fn snapshot(&self) -> &SchemaSnapshot {
        &self.snapshot
    }

    /// Width of the `phi` vector every [`transform`](Self::transform)
    /// call produces.
    pub fn phi_width(&self) -> usize {
        self.phi_width
    }

    /// Number of selected routing features — the length of `x`.
    pub fn n_routing_features(&self) -> usize {
        self.routing.len()
    }

    /// Routing bucket count per `x` dimension — use as the codebook
    /// `n_bins` when wiring the routing tree.
    pub fn route_bins(&self) -> u8 {
        self.route_bins
    }

    /// Original column indices of the selected routing features, in
    /// MI-rank (descending mutual information) order.
    pub fn routing_feature_columns(&self) -> Vec<usize> {
        self.routing.iter().map(|rf| rf.col).collect()
    }
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

    // ── COMMIT 4 — CategoricalTransform: raw rows -> (x, phi, y) ──────

    fn rows(data: &[&[&str]]) -> Vec<Vec<String>> {
        data.iter()
            .map(|r| r.iter().map(|s| s.to_string()).collect())
            .collect()
    }

    fn row(cells: &[&str]) -> Vec<String> {
        cells.iter().map(|s| s.to_string()).collect()
    }

    fn schema(cols: &[(&str, ColumnRole)]) -> Schema {
        Schema::new(cols.iter().map(|(n, r)| (n.to_string(), *r)).collect())
    }

    /// A test config — `KEEP_ALL` policy so the tiny fixtures don't all
    /// rare-fold; positives `{"yes"}`.
    fn cfg() -> TransformConfig {
        TransformConfig {
            route_bins: 4,
            k_routing: 2,
            max_real: 32,
            rare_policy: RarePolicy::KEEP_ALL,
            missing_markers: vec!["?".to_string()],
            target_positives: vec!["yes".to_string()],
            target_definition: "label == 'yes'".to_string(),
            raw_dataset_hash: [7u8; 32],
            split_seed: 42,
            row_count: 6,
        }
    }

    fn mixed_schema() -> Schema {
        schema(&[
            ("color", ColumnRole::Categorical),
            ("size", ColumnRole::Numeric),
            ("tag", ColumnRole::CategoricalPhiOnly),
            ("label", ColumnRole::Target),
            ("rowid", ColumnRole::Ignore),
        ])
    }

    fn mixed_rows() -> Vec<Vec<String>> {
        rows(&[
            &["red", "1.0", "a", "yes", "0"],
            &["red", "2.0", "b", "yes", "1"],
            &["blue", "8.0", "c", "no", "2"],
            &["blue", "9.0", "d", "no", "3"],
            &["green", "5.0", "e", "yes", "4"],
            &["green", "4.0", "f", "no", "5"],
        ])
    }

    #[test]
    fn standardizer_fit_basic() {
        // [0, 2, 4]: mean 2, population variance 8/3.
        let s = Standardizer::fit(&[0.0, 2.0, 4.0]);
        assert_eq!(s.mean, 2.0);
        let expect_std = (8.0_f64 / 3.0).sqrt();
        assert!((s.std - expect_std).abs() < 1e-12);
        assert!((s.apply(4.0) - 2.0 / expect_std).abs() < 1e-12);
    }

    #[test]
    fn standardizer_empty_is_identity() {
        let s = Standardizer::fit(&[]);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std, 1.0);
        assert_eq!(s.apply(3.5), 3.5);
    }

    #[test]
    fn standardizer_constant_column_floors_std() {
        let s = Standardizer::fit(&[7.0, 7.0, 7.0, 7.0]);
        assert_eq!(s.mean, 7.0);
        assert_eq!(s.std, 1e-12);
        // No div-by-zero; a value at the mean standardizes to exactly 0.
        assert_eq!(s.apply(7.0), 0.0);
    }

    #[test]
    fn standardizer_fit_is_row_order_invariant() {
        // Sorted-before-summed -> bit-identical regardless of input order.
        let a = Standardizer::fit(&[1.0, 5.0, 2.0, 9.0, 3.0]);
        let b = Standardizer::fit(&[9.0, 3.0, 1.0, 2.0, 5.0]);
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        assert_eq!(a.std.to_bits(), b.std.to_bits());
    }

    #[test]
    fn mutual_information_zero_when_independent() {
        // Each bucket sees both targets equally — the feature is useless.
        let buckets = [0u8, 0, 1, 1];
        let target = [0u8, 1, 0, 1];
        assert_eq!(mutual_information(&buckets, &target, 2), 0.0);
    }

    #[test]
    fn mutual_information_positive_when_predictive() {
        // The bucket perfectly predicts the target.
        let buckets = [0u8, 0, 1, 1];
        let target = [0u8, 0, 1, 1];
        assert!(mutual_information(&buckets, &target, 2) > 0.0);
    }

    #[test]
    fn mutual_information_empty_or_mismatched_is_zero() {
        assert_eq!(mutual_information(&[], &[], 4), 0.0);
        assert_eq!(mutual_information(&[0], &[0, 1], 4), 0.0);
    }

    #[test]
    fn numeric_bucket_clamps_into_route_bins() {
        let cuts = [1.0, 2.0, 3.0];
        assert_eq!(numeric_bucket(0.0, &cuts, 4), 0);
        assert_eq!(numeric_bucket(1.5, &cuts, 4), 1);
        assert_eq!(numeric_bucket(2.5, &cuts, 4), 2);
        assert_eq!(numeric_bucket(100.0, &cuts, 4), 3);
        assert!(numeric_bucket(-1e9, &cuts, 4) < 4);
        assert!(numeric_bucket(1e9, &cuts, 4) < 4);
    }

    #[test]
    fn quantile_cuts_empty_input_is_degenerate() {
        // No data -> `route_bins - 1` placeholder cuts, no panic.
        assert_eq!(quantile_cuts(&[], 4).len(), 3);
    }

    #[test]
    fn quantile_cuts_are_nondecreasing() {
        let cuts = quantile_cuts(&[5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0], 4);
        assert_eq!(cuts.len(), 3);
        for w in cuts.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn schema_canonical_bytes_deterministic() {
        let s = schema(&[("a", ColumnRole::Categorical), ("y", ColumnRole::Target)]);
        assert_eq!(s.canonical_bytes(), s.clone().canonical_bytes());
    }

    #[test]
    fn fit_rejects_empty_schema() {
        let s = Schema::new(vec![]);
        assert_eq!(
            CategoricalTransform::fit(&s, &[], &cfg()).unwrap_err(),
            TransformError::EmptySchema
        );
    }

    #[test]
    fn fit_rejects_no_target() {
        let s = schema(&[("a", ColumnRole::Categorical)]);
        assert_eq!(
            CategoricalTransform::fit(&s, &rows(&[&["x"]]), &cfg()).unwrap_err(),
            TransformError::NoTarget
        );
    }

    #[test]
    fn fit_rejects_multiple_targets() {
        let s = schema(&[("y1", ColumnRole::Target), ("y2", ColumnRole::Target)]);
        assert_eq!(
            CategoricalTransform::fit(&s, &[], &cfg()).unwrap_err(),
            TransformError::MultipleTargets
        );
    }

    #[test]
    fn fit_rejects_bad_route_bins() {
        let s = schema(&[("y", ColumnRole::Target)]);
        let mut c = cfg();
        c.route_bins = 3; // not a power of two
        assert_eq!(
            CategoricalTransform::fit(&s, &[], &c).unwrap_err(),
            TransformError::BadRouteBins(3)
        );
    }

    #[test]
    fn fit_rejects_row_arity_mismatch() {
        let s = schema(&[("a", ColumnRole::Categorical), ("y", ColumnRole::Target)]);
        let bad = rows(&[&["x", "yes", "EXTRA"]]);
        assert_eq!(
            CategoricalTransform::fit(&s, &bad, &cfg()).unwrap_err(),
            TransformError::RowArityMismatch { expected: 2, got: 3 }
        );
    }

    #[test]
    fn fit_then_transform_produces_x_phi_y() {
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        let (x, phi, y) = t
            .transform(&row(&["red", "1.0", "a", "yes", "99"]))
            .unwrap();
        assert_eq!(x.len(), t.n_routing_features());
        assert_eq!(x.len(), 2);
        assert_eq!(phi.len(), t.phi_width());
        assert_eq!(y, 1.0);
        for &xi in &x {
            assert!(xi >= 0.0 && xi < t.route_bins() as f64);
        }
    }

    #[test]
    fn transform_phi_width_constant_across_rows() {
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        for r in mixed_rows() {
            let (_, phi, _) = t.transform(&r).unwrap();
            assert_eq!(phi.len(), t.phi_width());
        }
        // Unknown categories / missing numerics keep `phi` the same width.
        let (_, phi, _) = t
            .transform(&row(&["MAGENTA", "?", "ZZZ", "no", "x"]))
            .unwrap();
        assert_eq!(phi.len(), t.phi_width());
    }

    #[test]
    fn transform_phi_is_one_hot_plus_standardized() {
        // Fixture widths: color 3 cats -> 6, size -> 1, tag 6 cats -> 9.
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        let (_, phi, _) = t
            .transform(&row(&["red", "1.0", "a", "yes", "0"]))
            .unwrap();
        assert_eq!(phi.len(), 16);
        let color_seg: f64 = phi[0..6].iter().sum();
        assert_eq!(color_seg, 1.0);
        let tag_seg: f64 = phi[7..16].iter().sum();
        assert_eq!(tag_seg, 1.0);
    }

    #[test]
    fn transform_rejects_missing_target() {
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        assert_eq!(
            t.transform(&row(&["red", "1.0", "a", "?", "0"])).unwrap_err(),
            TransformError::MissingTarget
        );
    }

    #[test]
    fn transform_rejects_row_arity_mismatch() {
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        assert_eq!(
            t.transform(&row(&["red", "1.0"])).unwrap_err(),
            TransformError::RowArityMismatch { expected: 5, got: 2 }
        );
    }

    #[test]
    fn phi_only_column_is_never_a_routing_feature() {
        let mut c = cfg();
        c.k_routing = 4; // ask for more than the candidate count
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &c).unwrap();
        let routing = t.routing_feature_columns();
        // tag (2, CategoricalPhiOnly), label (3, Target), rowid (4,
        // Ignore) are never routing features.
        assert!(!routing.contains(&2));
        assert!(!routing.contains(&3));
        assert!(!routing.contains(&4));
        assert_eq!(t.n_routing_features(), 2); // only color + size
    }

    #[test]
    fn routing_features_clamped_to_candidate_count() {
        let s = schema(&[
            ("a", ColumnRole::Categorical),
            ("b", ColumnRole::CategoricalPhiOnly),
            ("y", ColumnRole::Target),
        ]);
        let mut c = cfg();
        c.k_routing = 3;
        let t = CategoricalTransform::fit(
            &s,
            &rows(&[&["p", "q", "yes"], &["p", "r", "no"]]),
            &c,
        )
        .unwrap();
        assert_eq!(t.n_routing_features(), 1); // only "a" is a candidate
    }

    #[test]
    fn fit_is_deterministic_double_run() {
        let t1 = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        let t2 = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        assert_eq!(t1.snapshot().snapshot_hash(), t2.snapshot().snapshot_hash());
        assert_eq!(t1.routing_feature_columns(), t2.routing_feature_columns());
        assert_eq!(t1.phi_width(), t2.phi_width());
        let r = row(&["green", "5.0", "e", "yes", "0"]);
        assert_eq!(t1.transform(&r).unwrap(), t2.transform(&r).unwrap());
    }

    #[test]
    fn fit_is_row_order_invariant() {
        let forward = mixed_rows();
        let mut reversed = forward.clone();
        reversed.reverse();
        let t1 = CategoricalTransform::fit(&mixed_schema(), &forward, &cfg()).unwrap();
        let t2 = CategoricalTransform::fit(&mixed_schema(), &reversed, &cfg()).unwrap();
        // The whole snapshot — including numeric standardization — is
        // row-order invariant (the standardizer sorts before summing).
        assert_eq!(t1.snapshot().snapshot_hash(), t2.snapshot().snapshot_hash());
        assert_eq!(t1.routing_feature_columns(), t2.routing_feature_columns());
        let r = row(&["blue", "8.0", "c", "no", "0"]);
        assert_eq!(t1.transform(&r).unwrap(), t2.transform(&r).unwrap());
    }

    #[test]
    fn transform_x_buckets_within_route_bins() {
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        for r in mixed_rows() {
            let (x, _, _) = t.transform(&r).unwrap();
            for &xi in &x {
                assert!(xi >= 0.0 && xi < t.route_bins() as f64);
            }
        }
    }

    #[test]
    fn transform_numeric_missing_imputed_to_mean() {
        let s = schema(&[("n", ColumnRole::Numeric), ("y", ColumnRole::Target)]);
        let t = CategoricalTransform::fit(
            &s,
            &rows(&[&["2.0", "yes"], &["4.0", "no"], &["6.0", "yes"]]),
            &cfg(),
        )
        .unwrap();
        // Train mean of [2, 4, 6] is 4; a "?" cell imputes to 4 -> z = 0.
        let (_, phi, _) = t.transform(&row(&["?", "no"])).unwrap();
        assert_eq!(phi.len(), 1);
        assert_eq!(phi[0], 0.0);
    }

    #[test]
    fn snapshot_carries_config_provenance() {
        let t = CategoricalTransform::fit(&mixed_schema(), &mixed_rows(), &cfg()).unwrap();
        let snap = t.snapshot();
        assert_eq!(snap.raw_dataset_hash, [7u8; 32]);
        assert_eq!(snap.split_seed, 42);
        assert_eq!(snap.row_count, 6);
        assert_eq!(snap.target_definition, "label == 'yes'");
        assert_eq!(snap.feature_transform_version, FEATURE_TRANSFORM_VERSION);
    }

    #[test]
    fn mi_selection_prefers_the_predictive_feature() {
        // "signal" perfectly predicts the target; "noise" is constant.
        let s = schema(&[
            ("signal", ColumnRole::Categorical),
            ("noise", ColumnRole::Categorical),
            ("y", ColumnRole::Target),
        ]);
        let data = rows(&[
            &["A", "k", "yes"],
            &["A", "k", "yes"],
            &["B", "k", "no"],
            &["B", "k", "no"],
        ]);
        let mut c = cfg();
        c.route_bins = 8; // wide enough to keep the two reals distinct
        c.k_routing = 1;
        let t = CategoricalTransform::fit(&s, &data, &c).unwrap();
        assert_eq!(t.routing_feature_columns(), vec![0]);
    }
}
