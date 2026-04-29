//! Deterministic Adaptive Dictionary Engine — Phase 1 core (v3 brief).
//!
//! A byte-first categorical engine designed to coexist with the existing
//! `Column::Categorical` and `FctColumn` types (which stay for backwards
//! compat). New code that needs a deterministic, memory-efficient
//! categorical column should prefer `CategoricalColumn` from this module.
//!
//! # Design summary
//!
//! - `ByteStringPool` — append-only `Vec<u8>` arena with stable
//!   `(offset, len)` handles. No `String` in the hot path.
//! - `ByteStrView` — opaque `(offset, len)` handle into a pool.
//! - `AdaptiveCodes` — 4-arm enum (U8/U16/U32/U64). Promotes deterministically
//!   when cardinality crosses 256 / 65 536 / 2³² boundaries.
//! - `ByteDictionary` — `BTreeMap<Vec<u8>, u64>` lookup, `frozen` flag,
//!   `CategoryOrdering` policy. Deterministic by construction.
//! - `CategoricalColumn` — codes + dictionary + optional null bitmap.
//!
//! # Determinism contract
//!
//! 1. All thresholds are integer-only. No float math anywhere.
//! 2. Lookup is `BTreeMap`, not `HashMap` — no randomized hashing.
//! 3. Lexical ordering uses raw byte comparison (`Vec<u8>::cmp`), not
//!    Unicode-aware sort. Cross-machine reproducibility is guaranteed.
//! 4. Code-width promotion is *lazy* — triggered only when the current arm
//!    physically cannot hold the next code (i.e., inserting code 256 into
//!    a U8 arm). It is not predictive.
//! 5. `intern()` on a frozen dictionary returns `Err`, never silently
//!    extends.
//! 6. The same byte sequence interned in two fresh dictionaries with the
//!    same `CategoryOrdering` produces bit-identical code sequences.
//!
//! # What this module does NOT do (deferred)
//!
//! - Wiring into TidyView verbs (Phase 2)
//! - Categorical-aware group_by/join (Phase 3)
//! - Replacement of existing `Column::Categorical` / `FctColumn`
//! - Language-level `.cjcl` builtins
//!
//! # Layout
//!
//! Public API at the top, helpers below, `#[cfg(test)] mod tests` at the
//! bottom. Inline unit tests pin every invariant in the determinism
//! contract; bolero fuzz lives in `tests/bolero_fuzz/categorical_dictionary_fuzz.rs`.

use crate::BitMask;
use std::collections::BTreeMap;

// ════════════════════════════════════════════════════════════════════════
//  ByteStringPool — append-only Vec<u8> arena with stable handles
// ════════════════════════════════════════════════════════════════════════

/// Append-only byte arena. Each interned byte sequence gets a stable
/// `ByteStrView` handle whose `(offset, len)` survives all subsequent
/// insertions (the underlying `Vec<u8>` may reallocate, but the indices
/// into it are stable).
///
/// Pool capacity:
/// - `bytes`: up to 4 GiB total raw payload (`u32` offsets).
/// - `offsets`: up to 2³² distinct entries.
/// - `lens`: up to 2³² individual byte length (per-entry cap is also 2³²).
///
/// These limits are documented but not actively enforced beyond a
/// `try_push` API: in v3 use the pool well below those numbers; if a
/// future workload approaches the cap, promote to a `u64`-offset variant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ByteStringPool {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    lens: Vec<u32>,
}

impl ByteStringPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries currently in the pool.
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// True if the pool has no entries.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Total bytes stored (sum of all entry lengths). O(1).
    pub fn byte_size(&self) -> usize {
        self.bytes.len()
    }

    /// Append a byte sequence and return a stable view.
    ///
    /// Returns `Err` if the per-entry length exceeds `u32::MAX` or the
    /// total byte payload would exceed `u32::MAX`. These limits are wide
    /// enough that real-world pipelines never hit them, but we surface
    /// them as errors rather than panic for safety at boundaries.
    pub fn push(&mut self, bytes: &[u8]) -> Result<ByteStrView, ByteDictError> {
        let len = u32::try_from(bytes.len()).map_err(|_| ByteDictError::EntryTooLong {
            got: bytes.len(),
        })?;
        let new_total = self.bytes.len().checked_add(bytes.len()).ok_or(
            ByteDictError::PoolOverflow {
                attempted: bytes.len(),
                current: self.bytes.len(),
            },
        )?;
        let offset = u32::try_from(self.bytes.len()).map_err(|_| ByteDictError::PoolOverflow {
            attempted: bytes.len(),
            current: self.bytes.len(),
        })?;
        if new_total > u32::MAX as usize {
            return Err(ByteDictError::PoolOverflow {
                attempted: bytes.len(),
                current: self.bytes.len(),
            });
        }
        self.bytes.extend_from_slice(bytes);
        self.offsets.push(offset);
        self.lens.push(len);
        Ok(ByteStrView { offset, len })
    }

    /// Resolve a view back to its byte slice. Panics in debug builds if the
    /// view points outside the pool; in release builds returns an empty
    /// slice for an out-of-bounds view (defensive — should never happen if
    /// views are only constructed by `push`).
    pub fn get(&self, view: ByteStrView) -> &[u8] {
        let start = view.offset as usize;
        let end = start.saturating_add(view.len as usize);
        if end > self.bytes.len() {
            debug_assert!(false, "ByteStrView out of bounds for pool");
            return &[];
        }
        &self.bytes[start..end]
    }

    /// Resolve a view by entry index (position in insertion order).
    ///
    /// Returns `None` if `idx >= self.len()`.
    pub fn get_by_index(&self, idx: usize) -> Option<&[u8]> {
        let offset = *self.offsets.get(idx)? as usize;
        let len = *self.lens.get(idx)? as usize;
        Some(&self.bytes[offset..offset + len])
    }

    /// Iterate `(index, bytes)` in insertion order. Deterministic.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[u8])> + '_ {
        self.offsets.iter().zip(self.lens.iter()).enumerate().map(
            move |(i, (&off, &len))| {
                let start = off as usize;
                let end = start + len as usize;
                (i, &self.bytes[start..end])
            },
        )
    }

    /// Iterate views in insertion order. Use when you want to keep the
    /// `(offset, len)` representation (e.g. to thread it through a
    /// dictionary lookup).
    pub fn iter_views(&self) -> impl Iterator<Item = ByteStrView> + '_ {
        self.offsets
            .iter()
            .zip(self.lens.iter())
            .map(|(&offset, &len)| ByteStrView { offset, len })
    }
}

/// Opaque handle into a `ByteStringPool`. Cheap to copy.
///
/// Two views are equal if and only if their `(offset, len)` are equal —
/// this does NOT compare bytes. Use `ByteStringPool::get` to fetch the
/// payload before doing a content comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteStrView {
    pub offset: u32,
    pub len: u32,
}

impl ByteStrView {
    /// Length in bytes.
    pub fn byte_len(self) -> usize {
        self.len as usize
    }

    /// True if the view points to an empty byte sequence.
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

// ════════════════════════════════════════════════════════════════════════
//  AdaptiveCodes — code storage with deterministic width promotion
// ════════════════════════════════════════════════════════════════════════

/// Adaptive-width code storage. Promotes to a wider arm only when the
/// current arm physically cannot hold the next code:
///
/// - U8  → U16  when inserting a code ≥ 256
/// - U16 → U32  when inserting a code ≥ 65 536
/// - U32 → U64  when inserting a code ≥ 2³²
///
/// Promotion preserves all existing codes bit-for-bit. The `len()` and
/// the sequence of values returned by `iter()` are invariant under
/// promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveCodes {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl AdaptiveCodes {
    /// Empty U8-arm storage.
    pub fn new() -> Self {
        AdaptiveCodes::U8(Vec::new())
    }

    /// Pre-allocated empty U8-arm storage.
    pub fn with_capacity(cap: usize) -> Self {
        AdaptiveCodes::U8(Vec::with_capacity(cap))
    }

    /// Number of codes stored.
    pub fn len(&self) -> usize {
        match self {
            AdaptiveCodes::U8(v) => v.len(),
            AdaptiveCodes::U16(v) => v.len(),
            AdaptiveCodes::U32(v) => v.len(),
            AdaptiveCodes::U64(v) => v.len(),
        }
    }

    /// True if no codes stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a code. Promotes the arm in-place if `code` does not fit.
    ///
    /// Promotion is *lazy*: a U8 arm with codes [0, 1, 2] and an insert
    /// of `255` stays U8; insert of `256` promotes to U16 with codes
    /// [0, 1, 2, 256].
    pub fn push(&mut self, code: u64) {
        // Promote if needed.
        if self.needs_promotion_for(code) {
            self.promote_to_fit(code);
        }
        match self {
            AdaptiveCodes::U8(v) => v.push(code as u8),
            AdaptiveCodes::U16(v) => v.push(code as u16),
            AdaptiveCodes::U32(v) => v.push(code as u32),
            AdaptiveCodes::U64(v) => v.push(code),
        }
    }

    /// Get the code at index `i`. Panics if `i >= len()`.
    pub fn get(&self, i: usize) -> u64 {
        match self {
            AdaptiveCodes::U8(v) => v[i] as u64,
            AdaptiveCodes::U16(v) => v[i] as u64,
            AdaptiveCodes::U32(v) => v[i] as u64,
            AdaptiveCodes::U64(v) => v[i],
        }
    }

    /// Iterate codes as `u64` (the widest representation, lossless for all
    /// arms).
    pub fn iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            AdaptiveCodes::U8(v) => Box::new(v.iter().map(|&x| x as u64)),
            AdaptiveCodes::U16(v) => Box::new(v.iter().map(|&x| x as u64)),
            AdaptiveCodes::U32(v) => Box::new(v.iter().map(|&x| x as u64)),
            AdaptiveCodes::U64(v) => Box::new(v.iter().copied()),
        }
    }

    /// Width in bytes per code (1, 2, 4, or 8). Useful for memory
    /// accounting and benchmarks.
    pub fn width_bytes(&self) -> usize {
        match self {
            AdaptiveCodes::U8(_) => 1,
            AdaptiveCodes::U16(_) => 2,
            AdaptiveCodes::U32(_) => 4,
            AdaptiveCodes::U64(_) => 8,
        }
    }

    /// True if this arm cannot represent `code`.
    fn needs_promotion_for(&self, code: u64) -> bool {
        match self {
            AdaptiveCodes::U8(_) => code > u8::MAX as u64,
            AdaptiveCodes::U16(_) => code > u16::MAX as u64,
            AdaptiveCodes::U32(_) => code > u32::MAX as u64,
            AdaptiveCodes::U64(_) => false,
        }
    }

    /// Promote to the smallest arm that can hold `code`.
    fn promote_to_fit(&mut self, code: u64) {
        // Move out of self temporarily.
        let old = std::mem::replace(self, AdaptiveCodes::U64(Vec::new()));
        let target = if code <= u16::MAX as u64 {
            // U8 → U16
            let v: Vec<u16> = match old {
                AdaptiveCodes::U8(v) => v.into_iter().map(|x| x as u16).collect(),
                AdaptiveCodes::U16(v) => v,
                AdaptiveCodes::U32(_) | AdaptiveCodes::U64(_) => {
                    debug_assert!(false, "promote_to_fit called with shrinking target");
                    Vec::new()
                }
            };
            AdaptiveCodes::U16(v)
        } else if code <= u32::MAX as u64 {
            let v: Vec<u32> = match old {
                AdaptiveCodes::U8(v) => v.into_iter().map(|x| x as u32).collect(),
                AdaptiveCodes::U16(v) => v.into_iter().map(|x| x as u32).collect(),
                AdaptiveCodes::U32(v) => v,
                AdaptiveCodes::U64(_) => {
                    debug_assert!(false, "promote_to_fit called with shrinking target");
                    Vec::new()
                }
            };
            AdaptiveCodes::U32(v)
        } else {
            let v: Vec<u64> = match old {
                AdaptiveCodes::U8(v) => v.into_iter().map(|x| x as u64).collect(),
                AdaptiveCodes::U16(v) => v.into_iter().map(|x| x as u64).collect(),
                AdaptiveCodes::U32(v) => v.into_iter().map(|x| x as u64).collect(),
                AdaptiveCodes::U64(v) => v,
            };
            AdaptiveCodes::U64(v)
        };
        *self = target;
    }
}

impl Default for AdaptiveCodes {
    fn default() -> Self {
        Self::new()
    }
}

// ════════════════════════════════════════════════════════════════════════
//  CategoryOrdering — how codes are assigned to byte sequences
// ════════════════════════════════════════════════════════════════════════

/// Policy for assigning codes to byte sequences during dictionary
/// construction.
///
/// - `FirstSeen`: insertion order. Code = position of first occurrence.
///   Best for ingestion stability (the first row sets the order, and
///   subsequent rows of the same value reuse the same code).
/// - `Lexical`: byte-lexicographic order. The dictionary collects all
///   distinct bytes first, then assigns codes by sorting via
///   `Vec<u8>::cmp`. Best for cross-dataset reproducibility (any
///   permutation of the same input set produces the same code map).
/// - `Explicit(values)`: user provides the canonical order. Codes are
///   assigned `0..values.len()` in the order given. Best for
///   production ML schemas where the inference-time code map must
///   match training exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoryOrdering {
    FirstSeen,
    Lexical,
    Explicit(Vec<Vec<u8>>),
}

// ════════════════════════════════════════════════════════════════════════
//  UnknownCategoryPolicy — handling values not in the dictionary
// ════════════════════════════════════════════════════════════════════════

/// Policy for `intern_with_policy` when a byte sequence is not in the
/// dictionary AND the dictionary is `frozen`.
///
/// - `Error`: return `Err(UnknownCategory)`. Strictest. Recommended for
///   safety-critical inference pipelines where a previously-unseen
///   category indicates a real data issue.
/// - `MapToOther`: return the code of a designated "Other" bucket. The
///   bucket must be added before freezing — the dictionary does not
///   silently create one.
/// - `MapToNull`: return `Ok(None)`, signalling the caller should mark
///   the row as null in the categorical column's null bitmap.
/// - `ExtendDictionary`: ignore the frozen flag and add the value. Only
///   safe in training; never in inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownCategoryPolicy {
    Error,
    MapToOther { other_code: u64 },
    MapToNull,
    ExtendDictionary,
}

/// Outcome of `intern_with_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternedCode {
    /// A concrete code was assigned or looked up.
    Code(u64),
    /// `MapToNull` was active and the value was unknown — the caller
    /// should set the null bit at this row.
    Null,
}

// ════════════════════════════════════════════════════════════════════════
//  Errors
// ════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ByteDictError {
    /// Single entry exceeded `u32::MAX` bytes.
    EntryTooLong { got: usize },
    /// Total pool would exceed `u32::MAX` bytes.
    PoolOverflow { attempted: usize, current: usize },
    /// Attempted to intern into a frozen dictionary with `Error` policy.
    UnknownCategory,
    /// Attempted to intern into a frozen dictionary.
    Frozen,
    /// Explicit ordering was provided but contains duplicates.
    ExplicitOrderingHasDuplicates,
    /// `MapToOther` policy was used but the `other_code` is not present
    /// in the dictionary.
    OtherCodeNotInDictionary { code: u64 },
}

impl std::fmt::Display for ByteDictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ByteDictError::EntryTooLong { got } => {
                write!(f, "ByteStringPool entry too long: {got} bytes (max {})", u32::MAX)
            }
            ByteDictError::PoolOverflow { attempted, current } => write!(
                f,
                "ByteStringPool overflow: cannot append {attempted} bytes (current {current})"
            ),
            ByteDictError::UnknownCategory => {
                write!(f, "unknown category in frozen dictionary (Error policy)")
            }
            ByteDictError::Frozen => write!(f, "dictionary is frozen"),
            ByteDictError::ExplicitOrderingHasDuplicates => {
                write!(f, "explicit ordering contains duplicate values")
            }
            ByteDictError::OtherCodeNotInDictionary { code } => {
                write!(f, "MapToOther policy: other_code {code} not in dictionary")
            }
        }
    }
}

impl std::error::Error for ByteDictError {}

// ════════════════════════════════════════════════════════════════════════
//  ByteDictionary — pool + BTreeMap lookup + ordering policy + frozen flag
// ════════════════════════════════════════════════════════════════════════

/// Deterministic byte-keyed dictionary.
///
/// Owns:
/// - a `ByteStringPool` storing every distinct byte sequence
/// - a `BTreeMap<Vec<u8>, u64>` mapping bytes → code (deterministic
///   iteration; no randomized hashing)
/// - an `ordering` policy controlling code assignment
/// - a `frozen` flag preventing inadvertent extension
///
/// # Code stability across runs
///
/// For a given `(input bytes sequence, ordering)` pair, the dictionary
/// produces bit-identical codes on every machine, every architecture,
/// every locale. The only inputs to the code map are the raw bytes and
/// the ordering policy.
#[derive(Debug, Clone)]
pub struct ByteDictionary {
    pool: ByteStringPool,
    /// Maps owned `Vec<u8>` keys → assigned code.
    /// `Vec<u8>` (owned) is required because the pool may reallocate
    /// during interning — we cannot key by `ByteStrView` until we know
    /// every key has stable storage.
    lookup: BTreeMap<Vec<u8>, u64>,
    /// View-by-code lookup. `code_to_view[c as usize]` returns the
    /// `ByteStrView` for code `c`. Codes are dense `0..n_categories`.
    code_to_view: Vec<ByteStrView>,
    frozen: bool,
    ordering: CategoryOrdering,
    /// v3 Phase 7: optional sealed `DHarht` lookup accelerator. Built
    /// by `seal_for_lookup()`. Pre-seal the field is `None` and lookups
    /// fall through to `lookup` (BTreeMap). Post-seal `dharht` becomes
    /// the primary lookup; `lookup` is retained for canonical iteration
    /// (insertion-order display, snapshot output, range scans).
    dharht: Option<crate::detcoll::DHarht<u64>>,
    /// v3 Phase 11: optional u64-hash content-cache built on
    /// `SealedU64Map` (which wraps `DHarhtMemory`). Lets callers who
    /// already have the deterministic 64-bit hash of bytes (from
    /// snapshot diffing, content-addressed storage, etc.) look up the
    /// dictionary code via `lookup_by_hash` without rehashing.
    /// Built by `seal_with_u64_hash_index()`; `None` otherwise.
    hash_index: Option<crate::detcoll::SealedU64Map<u64>>,
}

impl ByteDictionary {
    /// Empty dictionary with `FirstSeen` ordering. Not frozen.
    pub fn new() -> Self {
        Self::with_ordering(CategoryOrdering::FirstSeen)
    }

    /// Empty dictionary with the given ordering. For `Explicit`, codes are
    /// pre-populated `0..values.len()` immediately.
    pub fn with_ordering(ordering: CategoryOrdering) -> Self {
        let mut dict = ByteDictionary {
            pool: ByteStringPool::new(),
            lookup: BTreeMap::new(),
            code_to_view: Vec::new(),
            frozen: false,
            ordering: ordering.clone(),
            dharht: None,
            hash_index: None,
        };
        if let CategoryOrdering::Explicit(values) = ordering {
            for v in values {
                let _ = dict.intern_internal(&v, false);
            }
        }
        dict
    }

    /// Build an `Explicit`-ordered dictionary from the given values. The
    /// dictionary is returned NOT frozen — the caller can choose to
    /// freeze it. Returns `Err` if `values` contains duplicates.
    pub fn from_explicit(values: Vec<Vec<u8>>) -> Result<Self, ByteDictError> {
        // Detect duplicates via a sorted scan (deterministic).
        let mut sorted = values.clone();
        sorted.sort();
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                return Err(ByteDictError::ExplicitOrderingHasDuplicates);
            }
        }
        Ok(Self::with_ordering(CategoryOrdering::Explicit(values)))
    }

    /// Number of categories.
    pub fn len(&self) -> usize {
        self.code_to_view.len()
    }

    /// True if no categories.
    pub fn is_empty(&self) -> bool {
        self.code_to_view.is_empty()
    }

    /// True if the dictionary is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// The ordering policy.
    pub fn ordering(&self) -> &CategoryOrdering {
        &self.ordering
    }

    /// Freeze the dictionary. After freezing, `intern` returns
    /// `Err(Frozen)` for unknown values; `intern_with_policy` honours
    /// `UnknownCategoryPolicy`. Lookups continue to work.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Direct read access to the underlying pool.
    pub fn pool(&self) -> &ByteStringPool {
        &self.pool
    }

    /// Resolve a code back to its byte payload. Returns `None` for
    /// out-of-range codes.
    pub fn get(&self, code: u64) -> Option<&[u8]> {
        let idx = usize::try_from(code).ok()?;
        let view = *self.code_to_view.get(idx)?;
        Some(self.pool.get(view))
    }

    /// Look up a byte sequence. Does not extend the dictionary. Returns
    /// `None` if not present.
    ///
    /// v3 Phase 7: when the dictionary has been `seal_for_lookup()`-ed,
    /// the primary lookup goes through the `DHarht` Memory profile.
    /// Falls back to the `BTreeMap` for unsealed dictionaries (which is
    /// also the canonical iteration source).
    pub fn lookup(&self, bytes: &[u8]) -> Option<u64> {
        if let Some(d) = &self.dharht {
            return d.get_bytes(bytes).copied();
        }
        self.lookup.get(bytes).copied()
    }

    /// v3 Phase 7: build the `DHarht` lookup accelerator and seal it.
    /// After this call, `lookup()` routes through the `DHarht` and the
    /// dictionary should be treated as read-only for performance
    /// reasons (mutations are not blocked but invalidate the
    /// accelerator — they trigger a debug-build assertion). The
    /// `BTreeMap` lookup table is preserved for canonical iteration
    /// and for `range`-style queries that the `DHarht` does not
    /// support.
    ///
    /// Spec compliance:
    /// - splitmix64 deterministic scattering ✓
    /// - 256 shards (power of two) ✓
    /// - sealed sparse 16-bit front directory ✓
    /// - MicroBucket16 with deterministic BTreeMap overflow on
    ///   bucket > 16 (no silent entry loss) ✓
    /// - full key equality on every successful lookup ✓
    pub fn seal_for_lookup(&mut self) {
        let mut d = crate::detcoll::DHarht::new();
        for (k, &code) in &self.lookup {
            d.insert_bytes(k.as_slice(), code);
        }
        d.seal_for_lookup();
        self.dharht = Some(d);
    }

    /// True if the dictionary has been sealed with a `DHarht`
    /// accelerator. Distinct from the `frozen` flag (which controls
    /// extension, not lookup backend).
    pub fn is_lookup_sealed(&self) -> bool {
        self.dharht.is_some()
    }

    /// Diagnostic: number of entries that overflowed to the per-shard
    /// `BTreeMap` fallback in the `DHarht`. Always 0 if not sealed.
    pub fn dharht_overflow_count(&self) -> u64 {
        self.dharht.as_ref().map(|d| d.overflow_count()).unwrap_or(0)
    }

    /// v3 Phase 11: build a u64-hash content-addressed lookup index
    /// using `SealedU64Map` (DHarhtMemory profile). After this call,
    /// `lookup_by_hash(h)` returns the dictionary code for whichever
    /// byte sequence hashes to `h`. The hash function is the
    /// workspace's deterministic `crate::detcoll::hash_bytes` so the
    /// caller's pre-computed hash and the index agree byte-for-byte.
    ///
    /// Use case: snapshot diffing, content-addressed storage,
    /// reproducibility-critical pipelines where the hash is the
    /// canonical identifier.
    ///
    /// This is **independent** of `seal_for_lookup()` — you can call
    /// either, both, or neither. Both indices, when built, are
    /// mutually consistent: they reference the same code space.
    pub fn seal_with_u64_hash_index(&mut self) {
        let mut idx = crate::detcoll::SealedU64Map::new();
        for (k, &code) in &self.lookup {
            let h = crate::detcoll::hash_bytes(k.as_slice());
            idx.insert(h, code);
        }
        idx.seal();
        self.hash_index = Some(idx);
    }

    /// True if the u64-hash index has been built.
    pub fn is_hash_indexed(&self) -> bool {
        self.hash_index.is_some()
    }

    /// Look up a code by the deterministic u64 hash of its bytes.
    /// Returns `None` if the hash is unknown OR if the hash index has
    /// not been built (call `seal_with_u64_hash_index()` first).
    ///
    /// **Hash collision safety**: this is a hash-only lookup with no
    /// full byte equality check. Two distinct byte sequences hashing
    /// to the same `u64` would return one of the two codes — that's
    /// `O(2^-64)` for `splitmix64`-mixed hashes (well-distributed
    /// inputs). For safety-critical paths use `lookup_by_hash_verify`
    /// which carries the original bytes and verifies.
    pub fn lookup_by_hash(&self, hash: u64) -> Option<u64> {
        self.hash_index.as_ref()?.get(hash).copied()
    }

    /// Hash-keyed lookup with explicit byte verification. Returns the
    /// code only if both (a) the hash maps to a known code AND (b) the
    /// stored bytes for that code match `bytes` exactly. This closes
    /// the `O(2^-64)` collision window of `lookup_by_hash`.
    pub fn lookup_by_hash_verify(&self, hash: u64, bytes: &[u8]) -> Option<u64> {
        let code = self.lookup_by_hash(hash)?;
        let stored = self.get(code)?;
        if stored == bytes { Some(code) } else { None }
    }

    /// Intern a byte sequence. If not present and the dictionary is not
    /// frozen, assigns a new code (only valid under `FirstSeen` /
    /// `ExtendDictionary`-ish flows). Under `Lexical` ordering this method
    /// works in *streaming mode* — codes are assigned in encounter order,
    /// then `seal_lexical()` re-sorts them at the end. For `Explicit`,
    /// `intern` errors on unknown values (the explicit list is the
    /// authority).
    pub fn intern(&mut self, bytes: &[u8]) -> Result<u64, ByteDictError> {
        self.intern_internal(bytes, true)
    }

    /// Intern with explicit unknown policy. This is the inference-time
    /// API: it does not require the dictionary be unfrozen.
    pub fn intern_with_policy(
        &mut self,
        bytes: &[u8],
        policy: &UnknownCategoryPolicy,
    ) -> Result<InternedCode, ByteDictError> {
        if let Some(c) = self.lookup.get(bytes) {
            return Ok(InternedCode::Code(*c));
        }
        if !self.frozen {
            // If not frozen, behave like intern — extend the dictionary.
            return self.intern_internal(bytes, true).map(InternedCode::Code);
        }
        match policy {
            UnknownCategoryPolicy::Error => Err(ByteDictError::UnknownCategory),
            UnknownCategoryPolicy::MapToNull => Ok(InternedCode::Null),
            UnknownCategoryPolicy::MapToOther { other_code } => {
                if (*other_code as usize) < self.code_to_view.len() {
                    Ok(InternedCode::Code(*other_code))
                } else {
                    Err(ByteDictError::OtherCodeNotInDictionary { code: *other_code })
                }
            }
            UnknownCategoryPolicy::ExtendDictionary => {
                // Bypass the freeze for this call only.
                self.intern_internal(bytes, false).map(InternedCode::Code)
            }
        }
    }

    /// Internal interning. `respect_frozen` controls whether the frozen
    /// flag is honoured (used to implement `ExtendDictionary` policy).
    fn intern_internal(
        &mut self,
        bytes: &[u8],
        respect_frozen: bool,
    ) -> Result<u64, ByteDictError> {
        if let Some(&c) = self.lookup.get(bytes) {
            return Ok(c);
        }
        if respect_frozen && self.frozen {
            return Err(ByteDictError::Frozen);
        }
        // For Explicit ordering, a non-explicit value is unknown.
        if let CategoryOrdering::Explicit(_) = &self.ordering {
            if respect_frozen {
                return Err(ByteDictError::UnknownCategory);
            }
        }
        let view = self.pool.push(bytes)?;
        let code = self.code_to_view.len() as u64;
        self.code_to_view.push(view);
        self.lookup.insert(bytes.to_vec(), code);
        Ok(code)
    }

    /// Re-assign codes lexicographically (only useful when ordering is
    /// `Lexical`). Returns the permutation `old_code → new_code` so
    /// callers can rewrite their code arrays.
    ///
    /// For `FirstSeen` and `Explicit`, this is a no-op and returns the
    /// identity permutation.
    pub fn seal_lexical(&mut self) -> Vec<u64> {
        if !matches!(self.ordering, CategoryOrdering::Lexical) {
            return (0..self.code_to_view.len() as u64).collect();
        }
        // Collect (bytes_owned, old_code), sort by bytes, derive perm.
        let mut entries: Vec<(Vec<u8>, u64)> = self
            .lookup
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut perm = vec![0u64; self.code_to_view.len()];
        // Rebuild code_to_view + lookup.
        let mut new_pool = ByteStringPool::new();
        let mut new_code_to_view: Vec<ByteStrView> = Vec::with_capacity(entries.len());
        let mut new_lookup: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        for (new_code, (bytes, old_code)) in entries.into_iter().enumerate() {
            let view = new_pool
                .push(&bytes)
                .expect("seal_lexical: re-push cannot exceed original pool size");
            new_code_to_view.push(view);
            new_lookup.insert(bytes, new_code as u64);
            perm[old_code as usize] = new_code as u64;
        }
        self.pool = new_pool;
        self.code_to_view = new_code_to_view;
        self.lookup = new_lookup;
        perm
    }

    /// Iterate `(code, bytes)` pairs in code order. Deterministic.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &[u8])> + '_ {
        self.code_to_view.iter().enumerate().map(move |(i, &v)| {
            (i as u64, self.pool.get(v))
        })
    }
}

impl Default for ByteDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// ════════════════════════════════════════════════════════════════════════
//  CategoricalColumn — codes + dictionary + optional null bitmap
// ════════════════════════════════════════════════════════════════════════

/// A categorical column: a vector of codes pointing into a shared
/// `ByteDictionary`, plus an optional null bitmap.
///
/// # Invariant
///
/// `codes.len() == nrows`. If `nulls` is present, `nulls.nrows() ==
/// nrows`. A row is valid iff `nulls.is_none() || nulls.unwrap().get(i)`.
/// (A `1` bit means valid — same convention as `BitMask` elsewhere in
/// cjc-data.)
///
/// # Equality
///
/// Two `CategoricalColumn`s are deep-equal iff they have the same codes,
/// same dictionary contents (including ordering), and same null pattern.
#[derive(Debug, Clone)]
pub struct CategoricalColumn {
    codes: AdaptiveCodes,
    dictionary: ByteDictionary,
    nulls: Option<BitMask>,
}

impl CategoricalColumn {
    /// Empty column with a fresh `FirstSeen` dictionary.
    pub fn new() -> Self {
        Self {
            codes: AdaptiveCodes::new(),
            dictionary: ByteDictionary::new(),
            nulls: None,
        }
    }

    /// Empty column with the given dictionary (used to share a dictionary
    /// across columns or to seed with `Explicit` ordering).
    pub fn with_dictionary(dictionary: ByteDictionary) -> Self {
        Self {
            codes: AdaptiveCodes::new(),
            dictionary,
            nulls: None,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// True if no rows.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Read-only access to the dictionary.
    pub fn dictionary(&self) -> &ByteDictionary {
        &self.dictionary
    }

    /// Read-only access to the codes.
    pub fn codes(&self) -> &AdaptiveCodes {
        &self.codes
    }

    /// Read-only access to the null bitmap if present.
    pub fn nulls(&self) -> Option<&BitMask> {
        self.nulls.as_ref()
    }

    /// True if the row at index `i` is null.
    pub fn is_null(&self, i: usize) -> bool {
        match &self.nulls {
            None => false,
            Some(b) => !b.get(i),
        }
    }

    /// Append a non-null value, interning it in the dictionary.
    ///
    /// Returns the assigned code.
    pub fn push(&mut self, bytes: &[u8]) -> Result<u64, ByteDictError> {
        let code = self.dictionary.intern(bytes)?;
        self.codes.push(code);
        if let Some(b) = &mut self.nulls {
            // Extend null bitmap with a "valid" bit. BitMask is fixed-size,
            // so we rebuild — this is fine because nulls grow with the
            // column.
            *b = bitmask_with_extra_valid(b);
        }
        Ok(code)
    }

    /// Append a null. Allocates the null bitmap on first call if absent.
    pub fn push_null(&mut self) {
        // Use sentinel code 0 for the null cell — readers must check
        // `is_null` first. (We deliberately do not use a "null code" in
        // the dictionary itself: the dictionary stays clean of synthetic
        // entries.)
        self.codes.push(0);
        match &mut self.nulls {
            None => {
                // Existing rows are all valid. Build a fresh bitmap of
                // length codes.len() with all-ones for prior rows and a
                // zero for this null.
                let n = self.codes.len();
                let mut words = vec![0u64; (n + 63) / 64];
                // Set bits 0..n-1 to 1 (valid for prior rows; the new
                // null is at bit n-1).
                for i in 0..(n - 1) {
                    words[i / 64] |= 1u64 << (i % 64);
                }
                self.nulls =
                    Some(BitMask::from_words_for_test(words, n));
            }
            Some(b) => {
                *b = bitmask_with_extra_invalid(b);
            }
        }
    }

    /// Append a value with explicit unknown-policy handling. If the
    /// policy returns `InternedCode::Null`, the row is recorded as null.
    pub fn push_with_policy(
        &mut self,
        bytes: &[u8],
        policy: &UnknownCategoryPolicy,
    ) -> Result<(), ByteDictError> {
        match self.dictionary.intern_with_policy(bytes, policy)? {
            InternedCode::Code(c) => {
                self.codes.push(c);
                if let Some(b) = &mut self.nulls {
                    *b = bitmask_with_extra_valid(b);
                }
            }
            InternedCode::Null => {
                self.push_null();
            }
        }
        Ok(())
    }

    /// Resolve the bytes for row `i`. Returns `None` if the row is null.
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        if self.is_null(i) {
            return None;
        }
        let code = self.codes.get(i);
        self.dictionary.get(code)
    }

    /// Iterate `(row_index, Option<bytes>)`. Deterministic.
    pub fn iter(&self) -> impl Iterator<Item = (usize, Option<&[u8]>)> + '_ {
        (0..self.len()).map(move |i| (i, self.get(i)))
    }

    /// Run `seal_lexical` on the dictionary and rewrite the codes. After
    /// this call the dictionary's iteration order is byte-lex; the codes
    /// are remapped accordingly. Bit-identical results regardless of
    /// insertion order.
    pub fn seal_lexical(&mut self) {
        if !matches!(self.dictionary.ordering(), CategoryOrdering::Lexical) {
            return;
        }
        let perm = self.dictionary.seal_lexical();
        let mut new_codes = AdaptiveCodes::with_capacity(self.codes.len());
        for c in self.codes.iter() {
            new_codes.push(perm[c as usize]);
        }
        self.codes = new_codes;
    }

    /// Build a `CategoricalProfile` summarising this column. Internal
    /// stats only; not exposed as a language-level builtin in Phase 1.
    pub fn profile(&self) -> CategoricalProfile {
        let cardinality = self.dictionary.len();
        let nrows = self.len();
        let missing = match &self.nulls {
            None => 0,
            Some(b) => nrows - b.count_ones(),
        };
        let bytes_used = self.dictionary.pool().byte_size();
        let avg_byte_len = if cardinality == 0 {
            0
        } else {
            bytes_used / cardinality
        };
        let mut max_byte_len = 0usize;
        for (_, b) in self.dictionary.iter() {
            if b.len() > max_byte_len {
                max_byte_len = b.len();
            }
        }
        let unique_ratio_thousandths = if nrows == 0 {
            0
        } else {
            (cardinality as u64).saturating_mul(1000) / nrows as u64
        };
        CategoricalProfile {
            nrows,
            cardinality,
            missing,
            bytes_used,
            avg_byte_len,
            max_byte_len,
            code_width_bytes: self.codes.width_bytes(),
            unique_ratio_thousandths,
        }
    }
}

impl Default for CategoricalColumn {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal stats computed by `CategoricalColumn::profile()`. Integer
/// fields only — no float math, deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoricalProfile {
    pub nrows: usize,
    pub cardinality: usize,
    pub missing: usize,
    pub bytes_used: usize,
    pub avg_byte_len: usize,
    pub max_byte_len: usize,
    pub code_width_bytes: usize,
    /// `(cardinality / nrows) * 1000` rounded toward zero. Reported in
    /// thousandths to keep the profile integer-only.
    pub unique_ratio_thousandths: u64,
}

// ════════════════════════════════════════════════════════════════════════
//  BitMask helpers (private)
// ════════════════════════════════════════════════════════════════════════

/// Build a new BitMask one bit longer than `b`, with the new bit set
/// (valid).
fn bitmask_with_extra_valid(b: &BitMask) -> BitMask {
    let n = b.nrows() + 1;
    let mut words: Vec<u64> = b.words_slice().to_vec();
    let needed = (n + 63) / 64;
    while words.len() < needed {
        words.push(0);
    }
    let i = n - 1;
    words[i / 64] |= 1u64 << (i % 64);
    BitMask::from_words_for_test(words, n)
}

/// Build a new BitMask one bit longer than `b`, with the new bit clear
/// (invalid).
fn bitmask_with_extra_invalid(b: &BitMask) -> BitMask {
    let n = b.nrows() + 1;
    let mut words: Vec<u64> = b.words_slice().to_vec();
    let needed = (n + 63) / 64;
    while words.len() < needed {
        words.push(0);
    }
    // The new bit is bit (n-1); leave it clear by default.
    BitMask::from_words_for_test(words, n)
}

// ════════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── ByteStringPool ──────────────────────────────────────────────────

    #[test]
    fn pool_round_trip_simple() {
        let mut p = ByteStringPool::new();
        let v_hello = p.push(b"hello").unwrap();
        let v_world = p.push(b"world").unwrap();
        assert_eq!(p.get(v_hello), b"hello");
        assert_eq!(p.get(v_world), b"world");
        assert_eq!(p.len(), 2);
        assert_eq!(p.byte_size(), 10);
    }

    #[test]
    fn pool_round_trip_empty_strings() {
        // Empty byte strings are valid and must round-trip.
        let mut p = ByteStringPool::new();
        let v_empty = p.push(b"").unwrap();
        let v_x = p.push(b"x").unwrap();
        let v_empty2 = p.push(b"").unwrap();
        assert!(p.get(v_empty).is_empty());
        assert_eq!(p.get(v_x), b"x");
        assert!(p.get(v_empty2).is_empty());
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn pool_round_trip_embedded_nul_and_high_bytes() {
        let mut p = ByteStringPool::new();
        let payload: &[u8] = &[0u8, 1, 2, 0, 0xff, 0xfe, b'a'];
        let v = p.push(payload).unwrap();
        assert_eq!(p.get(v), payload);
    }

    #[test]
    fn pool_get_by_index_matches_views() {
        let mut p = ByteStringPool::new();
        p.push(b"a").unwrap();
        p.push(b"bb").unwrap();
        p.push(b"ccc").unwrap();
        assert_eq!(p.get_by_index(0).unwrap(), b"a");
        assert_eq!(p.get_by_index(1).unwrap(), b"bb");
        assert_eq!(p.get_by_index(2).unwrap(), b"ccc");
        assert!(p.get_by_index(3).is_none());
    }

    #[test]
    fn pool_iter_is_insertion_order() {
        let mut p = ByteStringPool::new();
        p.push(b"z").unwrap();
        p.push(b"a").unwrap();
        p.push(b"m").unwrap();
        let collected: Vec<&[u8]> = p.iter().map(|(_, b)| b).collect();
        assert_eq!(collected, vec![b"z" as &[u8], b"a", b"m"]);
    }

    // ── AdaptiveCodes promotion ─────────────────────────────────────────

    #[test]
    fn adaptive_codes_starts_u8() {
        let c = AdaptiveCodes::new();
        assert!(matches!(c, AdaptiveCodes::U8(_)));
        assert_eq!(c.width_bytes(), 1);
    }

    #[test]
    fn adaptive_codes_stays_u8_at_255() {
        let mut c = AdaptiveCodes::new();
        for i in 0u64..=255 {
            c.push(i);
        }
        assert!(matches!(c, AdaptiveCodes::U8(_)));
        assert_eq!(c.len(), 256);
        // Verify all values round-trip.
        for i in 0u64..=255 {
            assert_eq!(c.get(i as usize), i);
        }
    }

    #[test]
    fn adaptive_codes_promotes_u8_to_u16_at_256() {
        let mut c = AdaptiveCodes::new();
        for i in 0u64..=255 {
            c.push(i);
        }
        assert!(matches!(c, AdaptiveCodes::U8(_)));
        c.push(256);
        assert!(matches!(c, AdaptiveCodes::U16(_)));
        // Existing codes preserved bit-for-bit.
        for i in 0u64..=255 {
            assert_eq!(c.get(i as usize), i);
        }
        assert_eq!(c.get(256), 256);
    }

    #[test]
    fn adaptive_codes_promotes_u16_to_u32_at_65536() {
        let mut c = AdaptiveCodes::U16(vec![0u16, 1, 2, 65_535]);
        c.push(65_536);
        assert!(matches!(c, AdaptiveCodes::U32(_)));
        assert_eq!(c.get(0), 0);
        assert_eq!(c.get(3), 65_535);
        assert_eq!(c.get(4), 65_536);
    }

    #[test]
    fn adaptive_codes_promotes_u32_to_u64_above_4b() {
        let mut c = AdaptiveCodes::U32(vec![0u32, 1, u32::MAX]);
        let huge = (u32::MAX as u64) + 1;
        c.push(huge);
        assert!(matches!(c, AdaptiveCodes::U64(_)));
        assert_eq!(c.get(0), 0);
        assert_eq!(c.get(2), u32::MAX as u64);
        assert_eq!(c.get(3), huge);
    }

    #[test]
    fn adaptive_codes_iter_matches_get() {
        let mut c = AdaptiveCodes::new();
        for i in 0u64..10 {
            c.push(i * 7);
        }
        let via_iter: Vec<u64> = c.iter().collect();
        let via_get: Vec<u64> = (0..c.len()).map(|i| c.get(i)).collect();
        assert_eq!(via_iter, via_get);
    }

    // ── ByteDictionary basic ────────────────────────────────────────────

    #[test]
    fn dict_intern_assigns_sequential_codes() {
        let mut d = ByteDictionary::new();
        assert_eq!(d.intern(b"red").unwrap(), 0);
        assert_eq!(d.intern(b"green").unwrap(), 1);
        assert_eq!(d.intern(b"blue").unwrap(), 2);
        // Re-intern returns existing code.
        assert_eq!(d.intern(b"red").unwrap(), 0);
        assert_eq!(d.intern(b"green").unwrap(), 1);
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn dict_get_round_trips_bytes() {
        let mut d = ByteDictionary::new();
        d.intern(b"red").unwrap();
        d.intern(b"green").unwrap();
        d.intern(b"blue").unwrap();
        assert_eq!(d.get(0).unwrap(), b"red");
        assert_eq!(d.get(1).unwrap(), b"green");
        assert_eq!(d.get(2).unwrap(), b"blue");
        assert!(d.get(99).is_none());
    }

    #[test]
    fn dict_lookup_does_not_extend() {
        let mut d = ByteDictionary::new();
        d.intern(b"red").unwrap();
        assert_eq!(d.lookup(b"red"), Some(0));
        assert_eq!(d.lookup(b"green"), None);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn dict_first_seen_is_insertion_order() {
        // Inserting B, A, C must give codes 0, 1, 2 in that order.
        let mut d = ByteDictionary::new();
        assert_eq!(d.intern(b"B").unwrap(), 0);
        assert_eq!(d.intern(b"A").unwrap(), 1);
        assert_eq!(d.intern(b"C").unwrap(), 2);
        assert_eq!(d.get(0).unwrap(), b"B");
        assert_eq!(d.get(1).unwrap(), b"A");
        assert_eq!(d.get(2).unwrap(), b"C");
    }

    #[test]
    fn dict_lexical_seal_reorders_by_bytes() {
        let mut d = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
        d.intern(b"banana").unwrap();
        d.intern(b"apple").unwrap();
        d.intern(b"cherry").unwrap();
        let perm = d.seal_lexical();
        // After seal, codes are in byte-lex order.
        assert_eq!(d.get(0).unwrap(), b"apple");
        assert_eq!(d.get(1).unwrap(), b"banana");
        assert_eq!(d.get(2).unwrap(), b"cherry");
        // Permutation maps old → new.
        // banana was 0 → now 1
        // apple  was 1 → now 0
        // cherry was 2 → now 2
        assert_eq!(perm, vec![1, 0, 2]);
    }

    #[test]
    fn dict_lexical_two_insertion_orders_seal_to_same_codes() {
        let mut d1 = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
        let mut d2 = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
        d1.intern(b"banana").unwrap();
        d1.intern(b"apple").unwrap();
        d1.intern(b"cherry").unwrap();
        d2.intern(b"cherry").unwrap();
        d2.intern(b"banana").unwrap();
        d2.intern(b"apple").unwrap();
        d1.seal_lexical();
        d2.seal_lexical();
        for code in 0u64..3 {
            assert_eq!(d1.get(code), d2.get(code), "lex ordering diverges at {code}");
        }
    }

    #[test]
    fn dict_explicit_pre_populates_codes() {
        let d = ByteDictionary::from_explicit(vec![
            b"low".to_vec(),
            b"med".to_vec(),
            b"high".to_vec(),
        ])
        .unwrap();
        assert_eq!(d.lookup(b"low"), Some(0));
        assert_eq!(d.lookup(b"med"), Some(1));
        assert_eq!(d.lookup(b"high"), Some(2));
    }

    #[test]
    fn dict_explicit_rejects_duplicates() {
        let err = ByteDictionary::from_explicit(vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"a".to_vec(),
        ]);
        assert!(matches!(err, Err(ByteDictError::ExplicitOrderingHasDuplicates)));
    }

    #[test]
    fn dict_explicit_rejects_unknown_intern_when_frozen() {
        let mut d = ByteDictionary::from_explicit(vec![b"a".to_vec(), b"b".to_vec()]).unwrap();
        d.freeze();
        let res = d.intern(b"c");
        assert!(matches!(res, Err(ByteDictError::Frozen)));
    }

    // ── Frozen + UnknownCategoryPolicy ──────────────────────────────────

    #[test]
    fn dict_frozen_intern_errors() {
        let mut d = ByteDictionary::new();
        d.intern(b"a").unwrap();
        d.freeze();
        let err = d.intern(b"b");
        assert!(matches!(err, Err(ByteDictError::Frozen)));
    }

    #[test]
    fn policy_error_returns_unknown_category() {
        let mut d = ByteDictionary::new();
        d.intern(b"a").unwrap();
        d.freeze();
        let res = d.intern_with_policy(b"b", &UnknownCategoryPolicy::Error);
        assert!(matches!(res, Err(ByteDictError::UnknownCategory)));
    }

    #[test]
    fn policy_map_to_null_returns_null() {
        let mut d = ByteDictionary::new();
        d.intern(b"a").unwrap();
        d.freeze();
        let res = d.intern_with_policy(b"b", &UnknownCategoryPolicy::MapToNull);
        assert_eq!(res, Ok(InternedCode::Null));
    }

    #[test]
    fn policy_map_to_other_returns_other_code() {
        let mut d = ByteDictionary::new();
        let _a = d.intern(b"a").unwrap();
        let other = d.intern(b"Other").unwrap();
        d.freeze();
        let res = d.intern_with_policy(
            b"unseen",
            &UnknownCategoryPolicy::MapToOther { other_code: other },
        );
        assert_eq!(res, Ok(InternedCode::Code(other)));
    }

    #[test]
    fn policy_map_to_other_rejects_invalid_other_code() {
        let mut d = ByteDictionary::new();
        d.intern(b"a").unwrap();
        d.freeze();
        let res = d.intern_with_policy(
            b"x",
            &UnknownCategoryPolicy::MapToOther { other_code: 99 },
        );
        assert!(matches!(
            res,
            Err(ByteDictError::OtherCodeNotInDictionary { code: 99 })
        ));
    }

    #[test]
    fn policy_extend_dictionary_bypasses_frozen() {
        let mut d = ByteDictionary::new();
        d.intern(b"a").unwrap();
        d.freeze();
        let res = d
            .intern_with_policy(b"b", &UnknownCategoryPolicy::ExtendDictionary)
            .unwrap();
        assert_eq!(res, InternedCode::Code(1));
        assert_eq!(d.len(), 2);
        // Dictionary stays frozen for subsequent strict calls.
        assert!(d.is_frozen());
        let strict = d.intern_with_policy(b"c", &UnknownCategoryPolicy::Error);
        assert!(matches!(strict, Err(ByteDictError::UnknownCategory)));
    }

    // ── CategoricalColumn ───────────────────────────────────────────────

    #[test]
    fn col_push_round_trip() {
        let mut c = CategoricalColumn::new();
        c.push(b"red").unwrap();
        c.push(b"blue").unwrap();
        c.push(b"red").unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.get(0).unwrap(), b"red");
        assert_eq!(c.get(1).unwrap(), b"blue");
        assert_eq!(c.get(2).unwrap(), b"red");
        // Two distinct categories.
        assert_eq!(c.dictionary().len(), 2);
    }

    #[test]
    fn col_push_null_is_observed() {
        let mut c = CategoricalColumn::new();
        c.push(b"red").unwrap();
        c.push_null();
        c.push(b"blue").unwrap();
        assert_eq!(c.len(), 3);
        assert!(!c.is_null(0));
        assert!(c.is_null(1));
        assert!(!c.is_null(2));
        assert_eq!(c.get(0).unwrap(), b"red");
        assert!(c.get(1).is_none());
        assert_eq!(c.get(2).unwrap(), b"blue");
    }

    #[test]
    fn col_seal_lexical_remaps_codes() {
        let mut c =
            CategoricalColumn::with_dictionary(ByteDictionary::with_ordering(CategoryOrdering::Lexical));
        c.push(b"banana").unwrap();
        c.push(b"apple").unwrap();
        c.push(b"banana").unwrap();
        c.push(b"cherry").unwrap();
        c.seal_lexical();
        // After seal, banana → 1, apple → 0, cherry → 2.
        assert_eq!(c.codes().get(0), 1, "row0 was banana → seal to 1");
        assert_eq!(c.codes().get(1), 0, "row1 was apple → seal to 0");
        assert_eq!(c.codes().get(2), 1, "row2 was banana → seal to 1");
        assert_eq!(c.codes().get(3), 2, "row3 was cherry → seal to 2");
        // Round-trip preserved.
        assert_eq!(c.get(0).unwrap(), b"banana");
        assert_eq!(c.get(1).unwrap(), b"apple");
        assert_eq!(c.get(2).unwrap(), b"banana");
        assert_eq!(c.get(3).unwrap(), b"cherry");
    }

    #[test]
    fn col_promotion_at_256_distinct() {
        // Force the U8 → U16 transition by pushing 257 distinct values.
        let mut c = CategoricalColumn::new();
        for i in 0u32..257 {
            // Generate distinct byte strings.
            let s = format!("v{}", i);
            c.push(s.as_bytes()).unwrap();
        }
        assert_eq!(c.dictionary().len(), 257);
        // Code arm has been promoted to U16 (or wider).
        assert!(matches!(
            c.codes(),
            AdaptiveCodes::U16(_) | AdaptiveCodes::U32(_) | AdaptiveCodes::U64(_)
        ));
        // Every row round-trips.
        for i in 0u32..257 {
            let s = format!("v{}", i);
            assert_eq!(c.get(i as usize).unwrap(), s.as_bytes());
        }
    }

    #[test]
    fn col_push_with_policy_null_records_null() {
        let mut c = CategoricalColumn::new();
        c.push(b"a").unwrap();
        c.push(b"b").unwrap();
        // Freeze the dictionary, then push an unknown with MapToNull.
        // We need to freeze after seeding the column, so reach in via
        // a fresh setup: build with explicit dict.
        let mut dict = ByteDictionary::new();
        dict.intern(b"a").unwrap();
        dict.intern(b"b").unwrap();
        dict.freeze();
        let mut c2 = CategoricalColumn::with_dictionary(dict);
        c2.push_with_policy(b"a", &UnknownCategoryPolicy::Error).unwrap();
        c2.push_with_policy(b"unseen", &UnknownCategoryPolicy::MapToNull)
            .unwrap();
        c2.push_with_policy(b"b", &UnknownCategoryPolicy::Error).unwrap();
        assert_eq!(c2.len(), 3);
        assert!(!c2.is_null(0));
        assert!(c2.is_null(1));
        assert!(!c2.is_null(2));
    }

    // ── Profile ─────────────────────────────────────────────────────────

    #[test]
    fn profile_reports_basic_stats() {
        let mut c = CategoricalColumn::new();
        c.push(b"red").unwrap();
        c.push(b"blue").unwrap();
        c.push_null();
        c.push(b"red").unwrap();
        let p = c.profile();
        assert_eq!(p.nrows, 4);
        assert_eq!(p.cardinality, 2);
        assert_eq!(p.missing, 1);
        assert_eq!(p.bytes_used, 7); // "red" + "blue"
        // unique_ratio: 2/4 = 0.5 = 500 thousandths
        assert_eq!(p.unique_ratio_thousandths, 500);
        assert_eq!(p.code_width_bytes, 1);
    }

    // ── Determinism property ────────────────────────────────────────────

    #[test]
    fn determinism_same_input_first_seen_two_dicts() {
        let inputs: &[&[u8]] = &[b"alpha", b"beta", b"alpha", b"gamma", b"beta", b"delta"];
        let mut d1 = ByteDictionary::new();
        let mut d2 = ByteDictionary::new();
        let codes1: Vec<u64> = inputs.iter().map(|b| d1.intern(b).unwrap()).collect();
        let codes2: Vec<u64> = inputs.iter().map(|b| d2.intern(b).unwrap()).collect();
        assert_eq!(codes1, codes2, "FirstSeen must be deterministic");
        // Sanity: alpha=0, beta=1, gamma=2, delta=3.
        assert_eq!(codes1, vec![0, 1, 0, 2, 1, 3]);
    }

    #[test]
    fn determinism_lexical_two_permutations_seal_identical() {
        let perm1: &[&[u8]] = &[b"alpha", b"beta", b"gamma", b"delta"];
        let perm2: &[&[u8]] = &[b"delta", b"gamma", b"beta", b"alpha"];

        let mut d1 = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
        let mut d2 = ByteDictionary::with_ordering(CategoryOrdering::Lexical);
        for s in perm1 {
            d1.intern(s).unwrap();
        }
        for s in perm2 {
            d2.intern(s).unwrap();
        }
        d1.seal_lexical();
        d2.seal_lexical();
        // After sealing, both dictionaries must have identical code → bytes
        // mapping for every category.
        for code in 0u64..4 {
            assert_eq!(d1.get(code), d2.get(code), "code {code}");
        }
    }
}
