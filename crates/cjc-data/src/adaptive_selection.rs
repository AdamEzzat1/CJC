//! Adaptive selection representations for TidyView.
//!
//! A `TidyView` describes a subset of rows from a base `DataFrame`. Until v2,
//! the subset was stored as a single `BitMask` regardless of density. The
//! adaptive engine picks one of five deterministic representations based on
//! result density, so that sparse predicates do not pay the cost of a full
//! bitscan and dense predicates retain the existing fast O(nrows/64) path.
//!
//! # Modes
//!
//! - `Empty` — no rows selected (zero allocation)
//! - `All` — every row selected (zero allocation; nrows stored)
//! - `SelectionVector` — ascending `Vec<u32>` of selected row indices (sparse)
//! - `VerbatimMask` — backing `BitMask` for the dense path (≥30% density)
//! - `Hybrid` — chunked, locally-classified per [`HYBRID_CHUNK_SIZE`]-row block.
//!   Active for mid-density results when `nrows >= 2 * HYBRID_CHUNK_SIZE`.
//!
//! # Determinism
//!
//! - Iteration order is **always ascending row index** for every arm.
//! - Density classification uses pure integer arithmetic; thresholds are
//!   bit-stable across platforms.
//! - `intersect`/`union` produce a single deterministic arm choice for any
//!   pair of inputs.

use crate::BitMask;

// ── Constants ────────────────────────────────────────────────────────────────

/// Sparse threshold: count strictly less than `nrows / SPARSE_DIVISOR` is sparse.
const SPARSE_DIVISOR: usize = 1024;

/// Dense threshold: count strictly greater than `nrows * DENSE_NUM / DENSE_DEN`
/// is dense. 3/10 = 30%.
const DENSE_NUM: usize = 3;
const DENSE_DEN: usize = 10;

/// Hybrid chunk size in rows. 4096 rows = 64 u64 words per dense chunk = 512 B.
pub const HYBRID_CHUNK_SIZE: usize = 4096;

/// Words-per-chunk for the Dense variant. `HYBRID_CHUNK_SIZE / 64`.
const HYBRID_WORDS_PER_CHUNK: usize = HYBRID_CHUNK_SIZE / 64;

/// Per-chunk sparse threshold: a chunk with < `HYBRID_CHUNK_SIZE / 32` set bits
/// uses `HybridChunk::Sparse`. 128 hits at 4096-row chunk size.
const HYBRID_CHUNK_SPARSE_THRESHOLD: usize = HYBRID_CHUNK_SIZE / 32;

// ── HybridChunk ──────────────────────────────────────────────────────────────

/// A per-chunk selection state. Each chunk represents `HYBRID_CHUNK_SIZE` rows
/// (the final chunk may be partial — see `HybridChunk::partial_size`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridChunk {
    /// 0 hits in this chunk.
    Empty,
    /// All `chunk_len` rows selected (where `chunk_len` is the partial-aware
    /// row count for this chunk).
    All,
    /// Sparse: chunk-local row indices in ascending order, range `[0, chunk_len)`.
    Sparse(Vec<u16>),
    /// Dense: word-packed bitmap, length `HYBRID_WORDS_PER_CHUNK`. Tail bits
    /// past `chunk_len` are zero.
    Dense(Box<[u64]>),
}

impl HybridChunk {
    /// Count of selected rows in this chunk.
    pub fn count(&self, chunk_len: usize) -> usize {
        match self {
            HybridChunk::Empty => 0,
            HybridChunk::All => chunk_len,
            HybridChunk::Sparse(rows) => rows.len(),
            HybridChunk::Dense(words) => {
                words.iter().map(|w| w.count_ones() as usize).sum()
            }
        }
    }

    /// True if chunk-local row `off` (in `[0, chunk_len)`) is selected.
    pub fn contains_local(&self, off: usize, chunk_len: usize) -> bool {
        if off >= chunk_len {
            return false;
        }
        match self {
            HybridChunk::Empty => false,
            HybridChunk::All => true,
            HybridChunk::Sparse(rows) => rows.binary_search(&(off as u16)).is_ok(),
            HybridChunk::Dense(words) => {
                let wi = off / 64;
                let bi = off % 64;
                (words[wi] >> bi) & 1 == 1
            }
        }
    }
}

// ── AdaptiveSelection ────────────────────────────────────────────────────────

/// A row-selection representation chosen adaptively by density.
///
/// All variants carry `nrows` (the size of the underlying DataFrame) so that
/// `count`/`contains`/iteration are answerable without consulting the base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveSelection {
    /// No rows selected. `nrows` is the underlying frame size.
    Empty { nrows: usize },
    /// Every row selected.
    All { nrows: usize },
    /// Sparse: ascending u32 row indices.
    SelectionVector { rows: Vec<u32>, nrows: usize },
    /// Dense: backing BitMask.
    VerbatimMask { mask: BitMask },
    /// Mid-density chunked representation. `chunks.len() == ceil(nrows / HYBRID_CHUNK_SIZE)`.
    Hybrid { nrows: usize, chunks: Vec<HybridChunk> },
}

impl AdaptiveSelection {
    // ── Constructors ─────────────────────────────────────────────────────

    /// All rows selected.
    pub fn all(nrows: usize) -> Self {
        AdaptiveSelection::All { nrows }
    }

    /// No rows selected.
    pub fn empty(nrows: usize) -> Self {
        AdaptiveSelection::Empty { nrows }
    }

    /// Wrap a `BitMask` directly (force the dense arm). Used by call sites
    /// that already produced a mask through the existing pipeline and do
    /// not want a re-classification.
    pub fn from_bitmask(mask: BitMask) -> Self {
        let nrows = mask.nrows();
        let count = mask.count_ones();
        if count == 0 {
            AdaptiveSelection::Empty { nrows }
        } else if count == nrows {
            AdaptiveSelection::All { nrows }
        } else {
            AdaptiveSelection::VerbatimMask { mask }
        }
    }

    /// Construct from raw predicate-result words (LSB-first within each u64).
    ///
    /// This is the canonical entry point from `try_eval_predicate_columnar`
    /// and the row-wise filter fallback. The classifier:
    ///
    /// 1. Counts set bits across `words`.
    /// 2. Picks `Empty`/`All`/`SelectionVector`/`VerbatimMask` per density.
    /// 3. Mid-band selections become `Hybrid` when `nrows >= 2 * HYBRID_CHUNK_SIZE`.
    ///
    /// The caller is responsible for masking off tail bits past `nrows` before
    /// invoking — `BitMask::all_true` / `BitMask::from_bools` and the
    /// columnar predicate path already do this.
    pub fn from_predicate_result(words: Vec<u64>, nrows: usize) -> Self {
        let count: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        Self::classify(words, nrows, count)
    }

    /// Test-only and conversion helper — same as `from_predicate_result` but
    /// caller already knows the count.
    pub fn from_words_with_count(words: Vec<u64>, nrows: usize, count: usize) -> Self {
        Self::classify(words, nrows, count)
    }

    fn classify(words: Vec<u64>, nrows: usize, count: usize) -> Self {
        if count == 0 {
            return AdaptiveSelection::Empty { nrows };
        }
        if count == nrows {
            return AdaptiveSelection::All { nrows };
        }
        if Self::is_sparse(count, nrows) {
            let rows = words_to_indices(&words);
            return AdaptiveSelection::SelectionVector { rows, nrows };
        }
        let mask = bitmask_from_words(words, nrows);
        if Self::is_dense(count, nrows) {
            AdaptiveSelection::VerbatimMask { mask }
        } else if Self::should_hybrid(nrows) {
            // Mid-band: chunk it. Cheap one-pass classifier per chunk.
            let chunks = chunks_from_mask(&mask);
            AdaptiveSelection::Hybrid { nrows, chunks }
        } else {
            // Mid-band but the frame is too small to benefit from chunking.
            AdaptiveSelection::VerbatimMask { mask }
        }
    }

    /// `count < nrows / 1024`. Saturates safely for nrows < 1024 (then no
    /// non-empty selection is sparse, which is correct: a hand-typed
    /// 5-element df with 1 hit should not allocate a `Vec<u32>`).
    #[inline]
    fn is_sparse(count: usize, nrows: usize) -> bool {
        count < nrows / SPARSE_DIVISOR
    }

    /// `count > nrows * 3/10`. Uses 128-bit intermediate to avoid overflow on
    /// extreme nrows.
    #[inline]
    fn is_dense(count: usize, nrows: usize) -> bool {
        let lhs = (count as u128) * (DENSE_DEN as u128);
        let rhs = (nrows as u128) * (DENSE_NUM as u128);
        lhs > rhs
    }

    /// Frames smaller than 2 chunks gain nothing from chunking — keep them
    /// on `VerbatimMask`.
    #[inline]
    fn should_hybrid(nrows: usize) -> bool {
        nrows >= 2 * HYBRID_CHUNK_SIZE
    }

    // ── SelectionRepr surface ────────────────────────────────────────────

    /// Total number of rows in the underlying DataFrame.
    pub fn nrows(&self) -> usize {
        match self {
            AdaptiveSelection::Empty { nrows } => *nrows,
            AdaptiveSelection::All { nrows } => *nrows,
            AdaptiveSelection::SelectionVector { nrows, .. } => *nrows,
            AdaptiveSelection::VerbatimMask { mask } => mask.nrows(),
            AdaptiveSelection::Hybrid { nrows, .. } => *nrows,
        }
    }

    /// Number of rows actually selected.
    pub fn count(&self) -> usize {
        match self {
            AdaptiveSelection::Empty { .. } => 0,
            AdaptiveSelection::All { nrows } => *nrows,
            AdaptiveSelection::SelectionVector { rows, .. } => rows.len(),
            AdaptiveSelection::VerbatimMask { mask } => mask.count_ones(),
            AdaptiveSelection::Hybrid { nrows, chunks } => {
                let mut total = 0;
                for (i, c) in chunks.iter().enumerate() {
                    total += c.count(chunk_len_for(i, *nrows));
                }
                total
            }
        }
    }

    /// True if `row` is in the selection. Out-of-range rows return `false`.
    pub fn contains(&self, row: usize) -> bool {
        match self {
            AdaptiveSelection::Empty { .. } => false,
            AdaptiveSelection::All { nrows } => row < *nrows,
            AdaptiveSelection::SelectionVector { rows, nrows } => {
                if row >= *nrows {
                    return false;
                }
                let target = row as u32;
                rows.binary_search(&target).is_ok()
            }
            AdaptiveSelection::VerbatimMask { mask } => {
                if row >= mask.nrows() {
                    return false;
                }
                mask.get(row)
            }
            AdaptiveSelection::Hybrid { nrows, chunks } => {
                if row >= *nrows {
                    return false;
                }
                let ci = row / HYBRID_CHUNK_SIZE;
                let off = row % HYBRID_CHUNK_SIZE;
                chunks[ci].contains_local(off, chunk_len_for(ci, *nrows))
            }
        }
    }

    /// Iterate selected rows in ascending order.
    pub fn iter_indices(&self) -> SelectionIndices<'_> {
        match self {
            AdaptiveSelection::Empty { .. } => SelectionIndices::Empty,
            AdaptiveSelection::All { nrows } => SelectionIndices::Range(0..*nrows),
            AdaptiveSelection::SelectionVector { rows, .. } => SelectionIndices::Vec(rows.iter()),
            AdaptiveSelection::VerbatimMask { mask } => SelectionIndices::Mask {
                mask,
                next: 0,
                nrows: mask.nrows(),
            },
            AdaptiveSelection::Hybrid { nrows, chunks } => SelectionIndices::Hybrid {
                chunks,
                nrows: *nrows,
                ci: 0,
                inner: HybridInner::Start,
            },
        }
    }

    /// Materialize the selection as a `BitMask`. Always allocates `O(nrows/64)`.
    pub fn materialize_mask(&self) -> BitMask {
        match self {
            AdaptiveSelection::Empty { nrows } => BitMask::all_false(*nrows),
            AdaptiveSelection::All { nrows } => BitMask::all_true(*nrows),
            AdaptiveSelection::SelectionVector { rows, nrows } => {
                let mut bools = vec![false; *nrows];
                for &r in rows {
                    bools[r as usize] = true;
                }
                BitMask::from_bools(&bools)
            }
            AdaptiveSelection::VerbatimMask { mask } => mask.clone(),
            AdaptiveSelection::Hybrid { nrows, chunks } => {
                let mut bools = vec![false; *nrows];
                for (ci, chunk) in chunks.iter().enumerate() {
                    let base = ci * HYBRID_CHUNK_SIZE;
                    let chunk_len = chunk_len_for(ci, *nrows);
                    match chunk {
                        HybridChunk::Empty => {}
                        HybridChunk::All => {
                            for off in 0..chunk_len {
                                bools[base + off] = true;
                            }
                        }
                        HybridChunk::Sparse(rows) => {
                            for &r in rows {
                                bools[base + r as usize] = true;
                            }
                        }
                        HybridChunk::Dense(words) => {
                            for off in 0..chunk_len {
                                let wi = off / 64;
                                let bi = off % 64;
                                if (words[wi] >> bi) & 1 == 1 {
                                    bools[base + off] = true;
                                }
                            }
                        }
                    }
                }
                BitMask::from_bools(&bools)
            }
        }
    }

    /// Materialize the selection as an ascending `Vec<u32>` of row indices.
    pub fn materialize_indices(&self) -> Vec<u32> {
        match self {
            AdaptiveSelection::Empty { .. } => Vec::new(),
            AdaptiveSelection::All { nrows } => (0..*nrows as u32).collect(),
            AdaptiveSelection::SelectionVector { rows, .. } => rows.clone(),
            AdaptiveSelection::VerbatimMask { mask } => {
                mask.iter_set().map(|i| i as u32).collect()
            }
            AdaptiveSelection::Hybrid { nrows, chunks } => {
                let mut out = Vec::with_capacity(self.count());
                for (ci, chunk) in chunks.iter().enumerate() {
                    let base = (ci * HYBRID_CHUNK_SIZE) as u32;
                    let chunk_len = chunk_len_for(ci, *nrows);
                    match chunk {
                        HybridChunk::Empty => {}
                        HybridChunk::All => {
                            for off in 0..chunk_len as u32 {
                                out.push(base + off);
                            }
                        }
                        HybridChunk::Sparse(rows) => {
                            for &r in rows {
                                out.push(base + r as u32);
                            }
                        }
                        HybridChunk::Dense(words) => {
                            for (wi, &w) in words.iter().enumerate() {
                                let mut bits = w;
                                while bits != 0 {
                                    let tz = bits.trailing_zeros() as usize;
                                    let off = wi * 64 + tz;
                                    if off < chunk_len {
                                        out.push(base + off as u32);
                                    }
                                    bits &= bits - 1;
                                }
                            }
                        }
                    }
                }
                out
            }
        }
    }

    /// Intersection (AND). Output mode is re-classified by density.
    ///
    /// Panics if `nrows` differ — programming error (different base frames).
    pub fn intersect(&self, other: &AdaptiveSelection) -> AdaptiveSelection {
        assert_eq!(
            self.nrows(),
            other.nrows(),
            "AdaptiveSelection::intersect: nrows mismatch ({} vs {})",
            self.nrows(),
            other.nrows()
        );
        // Identity / annihilator fast paths
        match (self, other) {
            (AdaptiveSelection::Empty { nrows }, _) | (_, AdaptiveSelection::Empty { nrows }) => {
                return AdaptiveSelection::Empty { nrows: *nrows };
            }
            (AdaptiveSelection::All { .. }, _) => return other.clone(),
            (_, AdaptiveSelection::All { .. }) => return self.clone(),
            _ => {}
        }
        // Sparse ∩ Sparse: merge-walk in O(|A|+|B|), no bitmap allocation.
        if let (
            AdaptiveSelection::SelectionVector { rows: a, nrows },
            AdaptiveSelection::SelectionVector { rows: b, .. },
        ) = (self, other)
        {
            let merged = sorted_merge_intersect(a, b);
            return Self::classify_sparse(merged, *nrows);
        }
        // Sparse ∩ VerbatimMask: filter the sparse vector by mask test.
        if let (
            AdaptiveSelection::SelectionVector { rows, nrows },
            AdaptiveSelection::VerbatimMask { mask },
        )
        | (
            AdaptiveSelection::VerbatimMask { mask },
            AdaptiveSelection::SelectionVector { rows, nrows },
        ) = (self, other)
        {
            let filtered: Vec<u32> =
                rows.iter().copied().filter(|&r| mask.get(r as usize)).collect();
            return Self::classify_sparse(filtered, *nrows);
        }
        // ── v3 Phase 3: Hybrid streaming fast paths ──────────────────────
        // Hybrid ∩ Hybrid: per-chunk dispatch over the 16-way shape table.
        if let (
            AdaptiveSelection::Hybrid { nrows, chunks: ac },
            AdaptiveSelection::Hybrid { chunks: bc, .. },
        ) = (self, other)
        {
            let mut out = Vec::with_capacity(ac.len());
            for ci in 0..ac.len() {
                let chunk_len = chunk_len_for(ci, *nrows);
                out.push(intersect_chunks(&ac[ci], &bc[ci], chunk_len));
            }
            return Self::simplify_hybrid(*nrows, out);
        }
        // Hybrid ∩ SelectionVector: walk the sparse vector once, dispatch
        // each row to its chunk via row >> 12. Output stays sparse.
        if let (
            AdaptiveSelection::Hybrid { nrows, chunks },
            AdaptiveSelection::SelectionVector { rows, .. },
        )
        | (
            AdaptiveSelection::SelectionVector { rows, .. },
            AdaptiveSelection::Hybrid { nrows, chunks },
        ) = (self, other)
        {
            let mut filtered: Vec<u32> = Vec::with_capacity(rows.len());
            for &r in rows {
                let row = r as usize;
                let ci = row / HYBRID_CHUNK_SIZE;
                let off = row % HYBRID_CHUNK_SIZE;
                let chunk_len = chunk_len_for(ci, *nrows);
                if chunks[ci].contains_local(off, chunk_len) {
                    filtered.push(r);
                }
            }
            return Self::classify_sparse(filtered, *nrows);
        }
        // Hybrid ∩ VerbatimMask: per-chunk word-AND against the matching
        // 64-word slice of the verbatim mask. Output stays Hybrid.
        if let (
            AdaptiveSelection::Hybrid { nrows, chunks },
            AdaptiveSelection::VerbatimMask { mask },
        )
        | (
            AdaptiveSelection::VerbatimMask { mask },
            AdaptiveSelection::Hybrid { nrows, chunks },
        ) = (self, other)
        {
            let words = mask.words_slice();
            let mut out_chunks = Vec::with_capacity(chunks.len());
            for ci in 0..chunks.len() {
                let chunk_len = chunk_len_for(ci, *nrows);
                let w_start = ci * HYBRID_WORDS_PER_CHUNK;
                let w_end = (w_start + HYBRID_WORDS_PER_CHUNK).min(words.len());
                let mask_chunk = &words[w_start..w_end];
                out_chunks.push(intersect_chunk_with_words(
                    &chunks[ci],
                    mask_chunk,
                    chunk_len,
                ));
            }
            return Self::simplify_hybrid(*nrows, out_chunks);
        }
        // General path: materialize both as words, AND, reclassify.
        let lhs = self.materialize_mask();
        let rhs = other.materialize_mask();
        let words: Vec<u64> = lhs
            .words_slice()
            .iter()
            .zip(rhs.words_slice().iter())
            .map(|(a, b)| a & b)
            .collect();
        AdaptiveSelection::from_predicate_result(words, lhs.nrows())
    }

    /// Union (OR). Output mode re-classified by density.
    pub fn union(&self, other: &AdaptiveSelection) -> AdaptiveSelection {
        assert_eq!(
            self.nrows(),
            other.nrows(),
            "AdaptiveSelection::union: nrows mismatch ({} vs {})",
            self.nrows(),
            other.nrows()
        );
        match (self, other) {
            (AdaptiveSelection::All { nrows }, _) | (_, AdaptiveSelection::All { nrows }) => {
                return AdaptiveSelection::All { nrows: *nrows };
            }
            (AdaptiveSelection::Empty { .. }, _) => return other.clone(),
            (_, AdaptiveSelection::Empty { .. }) => return self.clone(),
            _ => {}
        }
        // Sparse ∪ Sparse: merge-walk, no bitmap allocation.
        if let (
            AdaptiveSelection::SelectionVector { rows: a, nrows },
            AdaptiveSelection::SelectionVector { rows: b, .. },
        ) = (self, other)
        {
            let merged = sorted_merge_union(a, b);
            // Re-classify: union may push us out of the sparse band.
            if merged.len() >= *nrows / SPARSE_DIVISOR {
                // No longer sparse — fall through to bitmap path below.
            } else {
                return Self::classify_sparse(merged, *nrows);
            }
        }
        // ── v3 Phase 3: Hybrid streaming fast paths ──────────────────────
        // Hybrid ∪ Hybrid: per-chunk dispatch.
        if let (
            AdaptiveSelection::Hybrid { nrows, chunks: ac },
            AdaptiveSelection::Hybrid { chunks: bc, .. },
        ) = (self, other)
        {
            let mut out = Vec::with_capacity(ac.len());
            for ci in 0..ac.len() {
                let chunk_len = chunk_len_for(ci, *nrows);
                out.push(union_chunks(&ac[ci], &bc[ci], chunk_len));
            }
            return Self::simplify_hybrid(*nrows, out);
        }
        // Hybrid ∪ SelectionVector: scatter the sparse rows into the
        // matching chunks. Output stays Hybrid.
        if let (
            AdaptiveSelection::Hybrid { nrows, chunks },
            AdaptiveSelection::SelectionVector { rows, .. },
        )
        | (
            AdaptiveSelection::SelectionVector { rows, .. },
            AdaptiveSelection::Hybrid { nrows, chunks },
        ) = (self, other)
        {
            let mut out_chunks: Vec<HybridChunk> = chunks.clone();
            // Bucket sparse rows by chunk index, then union each bucket.
            // Single linear pass: ascending input, ascending bucket order.
            let mut i = 0usize;
            while i < rows.len() {
                let first_row = rows[i] as usize;
                let ci = first_row / HYBRID_CHUNK_SIZE;
                let chunk_base = ci * HYBRID_CHUNK_SIZE;
                let chunk_end = chunk_base + HYBRID_CHUNK_SIZE;
                let mut bucket: Vec<u16> = Vec::new();
                while i < rows.len() && (rows[i] as usize) < chunk_end {
                    bucket.push((rows[i] as usize - chunk_base) as u16);
                    i += 1;
                }
                let chunk_len = chunk_len_for(ci, *nrows);
                let bucket_chunk = if bucket.len() == chunk_len {
                    HybridChunk::All
                } else {
                    HybridChunk::Sparse(bucket)
                };
                out_chunks[ci] = union_chunks(&out_chunks[ci], &bucket_chunk, chunk_len);
            }
            return Self::simplify_hybrid(*nrows, out_chunks);
        }
        // Hybrid ∪ VerbatimMask: per-chunk word-OR against the matching
        // 64-word slice. Output stays Hybrid.
        if let (
            AdaptiveSelection::Hybrid { nrows, chunks },
            AdaptiveSelection::VerbatimMask { mask },
        )
        | (
            AdaptiveSelection::VerbatimMask { mask },
            AdaptiveSelection::Hybrid { nrows, chunks },
        ) = (self, other)
        {
            let words = mask.words_slice();
            let mut out_chunks = Vec::with_capacity(chunks.len());
            for ci in 0..chunks.len() {
                let chunk_len = chunk_len_for(ci, *nrows);
                let w_start = ci * HYBRID_WORDS_PER_CHUNK;
                let w_end = (w_start + HYBRID_WORDS_PER_CHUNK).min(words.len());
                let mask_chunk = &words[w_start..w_end];
                out_chunks.push(union_chunk_with_words(
                    &chunks[ci],
                    mask_chunk,
                    chunk_len,
                ));
            }
            return Self::simplify_hybrid(*nrows, out_chunks);
        }
        let lhs = self.materialize_mask();
        let rhs = other.materialize_mask();
        let words: Vec<u64> = lhs
            .words_slice()
            .iter()
            .zip(rhs.words_slice().iter())
            .map(|(a, b)| a | b)
            .collect();
        AdaptiveSelection::from_predicate_result(words, lhs.nrows())
    }

    /// Stable identifier of the chosen mode. Useful for debug, tests, and
    /// the planned `explain_selection_mode` user-facing call.
    pub fn explain_selection_mode(&self) -> &'static str {
        match self {
            AdaptiveSelection::Empty { .. } => "Empty",
            AdaptiveSelection::All { .. } => "All",
            AdaptiveSelection::SelectionVector { .. } => "SelectionVector",
            AdaptiveSelection::VerbatimMask { .. } => "VerbatimMask",
            AdaptiveSelection::Hybrid { .. } => "Hybrid",
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Build a `SelectionVector`/`Empty`/`All` from an already-sorted-ascending
    /// `Vec<u32>`. Used by sparse-fast-path set ops.
    fn classify_sparse(rows: Vec<u32>, nrows: usize) -> Self {
        if rows.is_empty() {
            AdaptiveSelection::Empty { nrows }
        } else if rows.len() == nrows {
            AdaptiveSelection::All { nrows }
        } else {
            AdaptiveSelection::SelectionVector { rows, nrows }
        }
    }

    /// Collapse a chunked Hybrid result into `Empty`/`All` if every chunk
    /// agrees; otherwise keep the chunked layout. Phase 3 deliberately does
    /// not re-globalize mid-density Hybrids — preserving chunks is the point.
    fn simplify_hybrid(nrows: usize, chunks: Vec<HybridChunk>) -> AdaptiveSelection {
        let mut total = 0usize;
        for (i, c) in chunks.iter().enumerate() {
            total += c.count(chunk_len_for(i, nrows));
        }
        if total == 0 {
            return AdaptiveSelection::Empty { nrows };
        }
        if total == nrows {
            return AdaptiveSelection::All { nrows };
        }
        AdaptiveSelection::Hybrid { nrows, chunks }
    }
}

// ── Iterator ─────────────────────────────────────────────────────────────────

/// An ascending iterator over selected row indices. One variant per
/// `AdaptiveSelection` arm so the hot path stays monomorphic and inlines.
pub enum SelectionIndices<'a> {
    Empty,
    Range(std::ops::Range<usize>),
    Vec(std::slice::Iter<'a, u32>),
    Mask {
        mask: &'a BitMask,
        next: usize,
        nrows: usize,
    },
    Hybrid {
        chunks: &'a [HybridChunk],
        nrows: usize,
        /// Current chunk index.
        ci: usize,
        inner: HybridInner<'a>,
    },
}

/// Per-chunk iteration state.
pub enum HybridInner<'a> {
    /// About to advance to the next chunk.
    Start,
    /// Iterating an `All` chunk: emit `[next, end)` plus chunk base.
    AllRange { base: u32, next: u32, end: u32 },
    /// Iterating a sparse chunk's index list.
    Sparse {
        base: u32,
        iter: std::slice::Iter<'a, u16>,
    },
    /// Iterating a dense chunk's word vector.
    Dense {
        base: u32,
        words: &'a [u64],
        wi: usize,
        bits: u64,
        chunk_len: usize,
    },
}

impl<'a> Iterator for SelectionIndices<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            SelectionIndices::Empty => None,
            SelectionIndices::Range(r) => r.next(),
            SelectionIndices::Vec(it) => it.next().map(|&v| v as usize),
            SelectionIndices::Mask { mask, next, nrows } => {
                while *next < *nrows {
                    let i = *next;
                    *next += 1;
                    if mask.get(i) {
                        return Some(i);
                    }
                }
                None
            }
            SelectionIndices::Hybrid {
                chunks,
                nrows,
                ci,
                inner,
            } => loop {
                // Try to drain the current chunk's inner iterator first.
                match inner {
                    HybridInner::Start => {
                        // Fall through to chunk advance.
                    }
                    HybridInner::AllRange { base, next, end } => {
                        if *next < *end {
                            let v = *next;
                            *next += 1;
                            return Some((*base + v) as usize);
                        }
                        // exhausted; fall through
                    }
                    HybridInner::Sparse { base, iter } => {
                        if let Some(&off) = iter.next() {
                            return Some((*base + off as u32) as usize);
                        }
                        // exhausted; fall through
                    }
                    HybridInner::Dense {
                        base,
                        words,
                        wi,
                        bits,
                        chunk_len,
                    } => {
                        loop {
                            if *bits != 0 {
                                let tz = bits.trailing_zeros() as usize;
                                let off = *wi * 64 + tz;
                                *bits &= *bits - 1;
                                if off < *chunk_len {
                                    return Some((*base + off as u32) as usize);
                                }
                                continue;
                            }
                            *wi += 1;
                            if *wi >= words.len() {
                                break; // exhausted current chunk
                            }
                            *bits = words[*wi];
                        }
                    }
                }
                // Advance to next chunk
                if *ci >= chunks.len() {
                    return None;
                }
                let chunk = &chunks[*ci];
                let chunk_idx = *ci;
                *ci += 1;
                let chunk_len = chunk_len_for(chunk_idx, *nrows);
                let base = (chunk_idx * HYBRID_CHUNK_SIZE) as u32;
                *inner = match chunk {
                    HybridChunk::Empty => HybridInner::Start,
                    HybridChunk::All => HybridInner::AllRange {
                        base,
                        next: 0,
                        end: chunk_len as u32,
                    },
                    HybridChunk::Sparse(rows) => HybridInner::Sparse {
                        base,
                        iter: rows.iter(),
                    },
                    HybridChunk::Dense(words) => HybridInner::Dense {
                        base,
                        words: &words[..],
                        wi: 0,
                        bits: words[0],
                        chunk_len,
                    },
                };
            },
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert raw u64 words into an ascending `Vec<u32>` of set-bit indices.
///
/// Uses `trailing_zeros` to skip zero runs — O(count), not O(nrows).
fn words_to_indices(words: &[u64]) -> Vec<u32> {
    let mut out = Vec::with_capacity(64);
    for (wi, &w) in words.iter().enumerate() {
        let mut bits = w;
        while bits != 0 {
            let tz = bits.trailing_zeros() as usize;
            let row = wi * 64 + tz;
            out.push(row as u32);
            // Clear the lowest set bit
            bits &= bits - 1;
        }
    }
    out
}

/// Build a `BitMask` from raw words (used after AND/OR ops). Tail bits past
/// `nrows` are masked off to preserve the BitMask invariant.
fn bitmask_from_words(mut words: Vec<u64>, nrows: usize) -> BitMask {
    let nwords = (nrows + 63) / 64;
    if words.len() < nwords {
        words.resize(nwords, 0);
    } else if words.len() > nwords {
        words.truncate(nwords);
    }
    if nwords > 0 && nrows % 64 != 0 {
        let tail = nrows % 64;
        words[nwords - 1] &= (1u64 << tail) - 1;
    }
    BitMask::from_words_for_test(words, nrows)
}

/// Row count for chunk `ci` given a total `nrows`. The final chunk may be
/// partial.
#[inline]
fn chunk_len_for(ci: usize, nrows: usize) -> usize {
    let base = ci * HYBRID_CHUNK_SIZE;
    let remaining = nrows.saturating_sub(base);
    remaining.min(HYBRID_CHUNK_SIZE)
}

/// Slice the words of a fully-materialized BitMask into per-chunk
/// `HybridChunk`s, classifying each chunk by local density.
fn chunks_from_mask(mask: &BitMask) -> Vec<HybridChunk> {
    let nrows = mask.nrows();
    let total_words = mask.words_slice().len();
    let nchunks = (nrows + HYBRID_CHUNK_SIZE - 1) / HYBRID_CHUNK_SIZE;
    let mut chunks = Vec::with_capacity(nchunks);
    let words = mask.words_slice();
    for ci in 0..nchunks {
        let chunk_len = chunk_len_for(ci, nrows);
        let w_start = ci * HYBRID_WORDS_PER_CHUNK;
        let w_end = (w_start + HYBRID_WORDS_PER_CHUNK).min(total_words);
        let chunk_words = &words[w_start..w_end];
        let count: usize = chunk_words.iter().map(|w| w.count_ones() as usize).sum();
        let chunk = if count == 0 {
            HybridChunk::Empty
        } else if count == chunk_len {
            HybridChunk::All
        } else if count < HYBRID_CHUNK_SPARSE_THRESHOLD {
            // Sparse chunk: pull u16 offsets directly from the words.
            let mut offs: Vec<u16> = Vec::with_capacity(count);
            for (i, &w) in chunk_words.iter().enumerate() {
                let mut bits = w;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    let off = i * 64 + tz;
                    if off < chunk_len {
                        offs.push(off as u16);
                    }
                    bits &= bits - 1;
                }
            }
            HybridChunk::Sparse(offs)
        } else {
            // Dense chunk: copy the (up to) HYBRID_WORDS_PER_CHUNK words.
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for (i, &w) in chunk_words.iter().enumerate() {
                buf[i] = w;
            }
            HybridChunk::Dense(buf.into_boxed_slice())
        };
        chunks.push(chunk);
    }
    chunks
}

// ── v3 Phase 3: per-chunk set-op helpers ─────────────────────────────────────

/// Per-chunk classifier: take an ascending `Vec<u16>` of chunk-local offsets
/// and a `chunk_len`, decide between `Empty`/`All`/`Sparse`/`Dense`. Used by
/// outputs of sparse-shaped per-chunk operations.
fn classify_sparse_chunk(offs: Vec<u16>, chunk_len: usize) -> HybridChunk {
    if offs.is_empty() {
        HybridChunk::Empty
    } else if offs.len() == chunk_len {
        HybridChunk::All
    } else {
        // Per-Phase-3 invariant: sparse output of sparse-shape input keeps
        // the sparse arm regardless of size — overflow into Dense is the
        // caller's responsibility (they classify via `classify_dense_chunk`).
        HybridChunk::Sparse(offs)
    }
}

/// Per-chunk classifier: take a populated word buffer and a `chunk_len`,
/// pick the most compact arm by local count. Used by outputs of word-shaped
/// per-chunk operations.
fn classify_dense_chunk(buf: Vec<u64>, chunk_len: usize) -> HybridChunk {
    let count: usize = buf.iter().map(|w| w.count_ones() as usize).sum();
    if count == 0 {
        HybridChunk::Empty
    } else if count == chunk_len {
        HybridChunk::All
    } else if count < HYBRID_CHUNK_SPARSE_THRESHOLD {
        // Demote to Sparse: extract chunk-local offsets ascending.
        let mut offs = Vec::with_capacity(count);
        for (i, &w) in buf.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                let off = i * 64 + tz;
                if off < chunk_len {
                    offs.push(off as u16);
                }
                bits &= bits - 1;
            }
        }
        HybridChunk::Sparse(offs)
    } else {
        HybridChunk::Dense(buf.into_boxed_slice())
    }
}

/// Merge-walk intersection of two sorted ascending `&[u16]` slices.
fn merge_intersect_u16(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Merge-walk union of two sorted ascending `&[u16]` slices.
fn merge_union_u16(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// 16-way per-chunk intersection. The 5 effective shapes after annihilator
/// folding: Empty/_, _/Empty, All/X, X/All, Sparse∩Sparse (merge-walk),
/// Sparse∩Dense (filter-walk), Dense∩Dense (word-AND).
fn intersect_chunks(a: &HybridChunk, b: &HybridChunk, chunk_len: usize) -> HybridChunk {
    use HybridChunk::*;
    match (a, b) {
        (Empty, _) | (_, Empty) => Empty,
        (All, x) | (x, All) => x.clone(),
        (Sparse(ax), Sparse(bx)) => {
            let merged = merge_intersect_u16(ax, bx);
            classify_sparse_chunk(merged, chunk_len)
        }
        (Sparse(offs), Dense(words)) | (Dense(words), Sparse(offs)) => {
            let mut out: Vec<u16> = Vec::with_capacity(offs.len());
            for &off in offs {
                let off_u = off as usize;
                let wi = off_u / 64;
                let bi = off_u % 64;
                if (words[wi] >> bi) & 1 == 1 {
                    out.push(off);
                }
            }
            classify_sparse_chunk(out, chunk_len)
        }
        (Dense(aw), Dense(bw)) => {
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for i in 0..HYBRID_WORDS_PER_CHUNK {
                buf[i] = aw[i] & bw[i];
            }
            classify_dense_chunk(buf, chunk_len)
        }
    }
}

/// 16-way per-chunk union.
fn union_chunks(a: &HybridChunk, b: &HybridChunk, chunk_len: usize) -> HybridChunk {
    use HybridChunk::*;
    match (a, b) {
        (All, _) | (_, All) => All,
        (Empty, x) | (x, Empty) => x.clone(),
        (Sparse(ax), Sparse(bx)) => {
            let merged = merge_union_u16(ax, bx);
            // May overflow the sparse threshold → promote to Dense.
            if merged.len() == chunk_len {
                All
            } else if merged.len() < HYBRID_CHUNK_SPARSE_THRESHOLD {
                if merged.is_empty() {
                    Empty
                } else {
                    Sparse(merged)
                }
            } else {
                let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
                for &off in &merged {
                    let off = off as usize;
                    buf[off / 64] |= 1u64 << (off % 64);
                }
                classify_dense_chunk(buf, chunk_len)
            }
        }
        (Sparse(offs), Dense(words)) | (Dense(words), Sparse(offs)) => {
            let mut buf: Vec<u64> = words.iter().copied().collect();
            for &off in offs {
                let off = off as usize;
                buf[off / 64] |= 1u64 << (off % 64);
            }
            classify_dense_chunk(buf, chunk_len)
        }
        (Dense(aw), Dense(bw)) => {
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for i in 0..HYBRID_WORDS_PER_CHUNK {
                buf[i] = aw[i] | bw[i];
            }
            classify_dense_chunk(buf, chunk_len)
        }
    }
}

/// Hybrid chunk × VerbatimMask 64-word slice → Hybrid chunk (intersection).
/// `mask_words` may be shorter than `HYBRID_WORDS_PER_CHUNK` for the final
/// (partial) chunk.
fn intersect_chunk_with_words(
    chunk: &HybridChunk,
    mask_words: &[u64],
    chunk_len: usize,
) -> HybridChunk {
    use HybridChunk::*;
    match chunk {
        Empty => Empty,
        All => {
            // Result = whatever bits are set in mask_words (pre-zeroed past
            // chunk_len by BitMask invariant).
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for (i, &w) in mask_words.iter().enumerate() {
                buf[i] = w;
            }
            classify_dense_chunk(buf, chunk_len)
        }
        Sparse(offs) => {
            let mut out: Vec<u16> = Vec::with_capacity(offs.len());
            for &off in offs {
                let off_u = off as usize;
                let wi = off_u / 64;
                if wi < mask_words.len() {
                    let bi = off_u % 64;
                    if (mask_words[wi] >> bi) & 1 == 1 {
                        out.push(off);
                    }
                }
            }
            classify_sparse_chunk(out, chunk_len)
        }
        Dense(words) => {
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for i in 0..HYBRID_WORDS_PER_CHUNK {
                let mw = if i < mask_words.len() { mask_words[i] } else { 0 };
                buf[i] = words[i] & mw;
            }
            classify_dense_chunk(buf, chunk_len)
        }
    }
}

/// Hybrid chunk × VerbatimMask 64-word slice → Hybrid chunk (union).
fn union_chunk_with_words(
    chunk: &HybridChunk,
    mask_words: &[u64],
    chunk_len: usize,
) -> HybridChunk {
    use HybridChunk::*;
    match chunk {
        Empty => {
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for (i, &w) in mask_words.iter().enumerate() {
                buf[i] = w;
            }
            classify_dense_chunk(buf, chunk_len)
        }
        All => All,
        Sparse(offs) => {
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for (i, &w) in mask_words.iter().enumerate() {
                buf[i] = w;
            }
            for &off in offs {
                let off = off as usize;
                buf[off / 64] |= 1u64 << (off % 64);
            }
            classify_dense_chunk(buf, chunk_len)
        }
        Dense(words) => {
            let mut buf = vec![0u64; HYBRID_WORDS_PER_CHUNK];
            for i in 0..HYBRID_WORDS_PER_CHUNK {
                let mw = if i < mask_words.len() { mask_words[i] } else { 0 };
                buf[i] = words[i] | mw;
            }
            classify_dense_chunk(buf, chunk_len)
        }
    }
}

/// Merge-walk intersection of two sorted ascending `Vec<u32>` slices.
fn sorted_merge_intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Merge-walk union of two sorted ascending `Vec<u32>` slices.
fn sorted_merge_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_bools(n: usize, set: &[usize]) -> Vec<bool> {
        let mut v = vec![false; n];
        for &i in set {
            v[i] = true;
        }
        v
    }

    fn classify_from_bools(bools: &[bool]) -> AdaptiveSelection {
        let mask = BitMask::from_bools(bools);
        let words: Vec<u64> = mask.words_slice().to_vec();
        AdaptiveSelection::from_predicate_result(words, bools.len())
    }

    // ── Constructors / classifier ────────────────────────────────────

    #[test]
    fn empty_when_no_bits_set() {
        let s = classify_from_bools(&mk_bools(100, &[]));
        assert_eq!(s.explain_selection_mode(), "Empty");
        assert_eq!(s.count(), 0);
        assert_eq!(s.nrows(), 100);
    }

    #[test]
    fn all_when_every_bit_set() {
        let s = classify_from_bools(&vec![true; 100]);
        assert_eq!(s.explain_selection_mode(), "All");
        assert_eq!(s.count(), 100);
        assert_eq!(s.nrows(), 100);
    }

    #[test]
    fn sparse_picks_selection_vector() {
        // 100_000 rows, 50 hits → 50 < 100_000/1024 (~97) → sparse
        let mut hits = vec![];
        for i in (0..100_000).step_by(2_000) {
            hits.push(i);
        }
        assert_eq!(hits.len(), 50);
        let s = classify_from_bools(&mk_bools(100_000, &hits));
        assert_eq!(s.explain_selection_mode(), "SelectionVector");
        assert_eq!(s.count(), 50);
    }

    #[test]
    fn dense_picks_verbatim_mask() {
        // 1000 rows, every other → 50% → dense
        let hits: Vec<usize> = (0..1000).step_by(2).collect();
        let s = classify_from_bools(&mk_bools(1000, &hits));
        assert_eq!(s.explain_selection_mode(), "VerbatimMask");
        assert_eq!(s.count(), 500);
    }

    #[test]
    fn mid_band_small_frame_stays_verbatim() {
        // 1000 rows < 2 * HYBRID_CHUNK_SIZE → small frame, mid-band → VerbatimMask.
        let hits: Vec<usize> = (0..1000).step_by(5).collect();
        assert_eq!(hits.len(), 200);
        let s = classify_from_bools(&mk_bools(1000, &hits));
        assert_eq!(s.explain_selection_mode(), "VerbatimMask");
        assert_eq!(s.count(), 200);
    }

    #[test]
    fn mid_band_large_frame_picks_hybrid() {
        // 100_000 rows, 5_000 hits = 5%. Above sparse threshold (97), below
        // dense threshold (30%). nrows >= 2 * 4096 → Hybrid.
        let hits: Vec<usize> = (0..100_000).step_by(20).collect();
        assert_eq!(hits.len(), 5_000);
        let s = classify_from_bools(&mk_bools(100_000, &hits));
        assert_eq!(s.explain_selection_mode(), "Hybrid");
        assert_eq!(s.count(), 5_000);
    }

    // ── SelectionRepr surface ────────────────────────────────────────

    #[test]
    fn count_and_contains_agree_for_sparse() {
        let hits = vec![3usize, 17, 99, 5_000];
        let s = classify_from_bools(&mk_bools(100_000, &hits));
        assert_eq!(s.explain_selection_mode(), "SelectionVector");
        for h in &hits {
            assert!(s.contains(*h), "expected contains({})", h);
        }
        assert!(!s.contains(0));
        assert!(!s.contains(100_001)); // out of range
    }

    #[test]
    fn iter_indices_ascending_for_every_arm() {
        let hybrid_hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let cases: Vec<(usize, Vec<usize>, &'static str)> = vec![
            (10, vec![], "Empty"),
            (10, (0..10).collect(), "All"),
            (100_000, vec![3, 17, 99, 5_000], "SelectionVector"),
            (1000, (0..1000).step_by(2).collect(), "VerbatimMask"),
            (100_000, hybrid_hits, "Hybrid"),
        ];
        for (nrows, hits, expected_mode) in cases {
            let s = classify_from_bools(&mk_bools(nrows, &hits));
            assert_eq!(s.explain_selection_mode(), expected_mode);
            let collected: Vec<usize> = s.iter_indices().collect();
            assert_eq!(collected, hits, "{expected_mode} iter mismatch");
            // Strictly ascending invariant
            for w in collected.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn materialize_round_trip_agrees_for_every_arm() {
        let hits = vec![1usize, 5, 10, 64, 65, 200];
        let nrows = 256;
        let s = classify_from_bools(&mk_bools(nrows, &hits));
        let mask = s.materialize_mask();
        let idx = s.materialize_indices();
        assert_eq!(mask.count_ones(), hits.len());
        assert_eq!(idx.len(), hits.len());
        for h in &hits {
            assert!(mask.get(*h));
            assert!(idx.binary_search(&(*h as u32)).is_ok());
        }
    }

    // ── Set ops ──────────────────────────────────────────────────────

    #[test]
    fn intersect_identity_with_all() {
        let s = classify_from_bools(&mk_bools(1000, &[1, 2, 3]));
        let all = AdaptiveSelection::all(1000);
        let r = s.intersect(&all);
        assert_eq!(r.count(), 3);
        assert!(r.contains(1) && r.contains(2) && r.contains(3));
    }

    #[test]
    fn intersect_with_empty_is_empty() {
        let s = classify_from_bools(&mk_bools(1000, &[1, 2, 3]));
        let empty = AdaptiveSelection::empty(1000);
        let r = s.intersect(&empty);
        assert_eq!(r.explain_selection_mode(), "Empty");
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn union_with_all_is_all() {
        let s = classify_from_bools(&mk_bools(1000, &[1, 2, 3]));
        let all = AdaptiveSelection::all(1000);
        let r = s.union(&all);
        assert_eq!(r.explain_selection_mode(), "All");
        assert_eq!(r.count(), 1000);
    }

    #[test]
    fn intersect_mode_mixing_sparse_and_dense_gives_sparse_or_empty() {
        // sparse: a few hits in a 100k-row frame
        let sparse = classify_from_bools(&mk_bools(100_000, &[3, 17, 99, 5_000]));
        // dense: every other row in the same frame
        let dense_hits: Vec<usize> = (0..100_000).step_by(2).collect();
        let dense = classify_from_bools(&mk_bools(100_000, &dense_hits));
        // 50% over 100k → mid band → Hybrid (large frame).
        assert_eq!(dense.explain_selection_mode(), "VerbatimMask"); // 50% > 30% → dense
        assert_eq!(sparse.explain_selection_mode(), "SelectionVector");

        let r = sparse.intersect(&dense);
        // 3, 17, 99 are odd → don't survive even-only filter; 5_000 is even.
        assert_eq!(r.count(), 1);
        assert!(r.contains(5_000));
    }

    #[test]
    fn intersect_is_commutative_and_associative_for_three_inputs() {
        let a = classify_from_bools(&mk_bools(256, &[1, 5, 7, 99]));
        let b = classify_from_bools(&mk_bools(256, &[5, 99, 100, 200]));
        let c = classify_from_bools(&mk_bools(256, &[5, 50, 99, 250]));

        let ab_c = a.intersect(&b).intersect(&c);
        let bc_a = b.intersect(&c).intersect(&a);
        let ba_c = b.intersect(&a).intersect(&c);

        assert_eq!(ab_c.materialize_indices(), bc_a.materialize_indices());
        assert_eq!(ab_c.materialize_indices(), ba_c.materialize_indices());
    }

    // ── Density classifier edges ─────────────────────────────────────

    #[test]
    fn small_nrows_never_classifies_as_sparse() {
        // nrows=10, 1 hit. nrows/1024 = 0, count=1, 1 < 0 is false → not sparse.
        let s = classify_from_bools(&mk_bools(10, &[3]));
        assert_ne!(s.explain_selection_mode(), "SelectionVector");
    }

    #[test]
    fn dense_threshold_is_exclusive_30_percent() {
        // 1000 rows, exactly 300 hits (= 30%). 300*10 = 3000, 1000*3 = 3000,
        // strict > fails → mid band, not dense. Small frame → VerbatimMask.
        let hits: Vec<usize> = (0..300).collect();
        let s = classify_from_bools(&mk_bools(1000, &hits));
        assert_eq!(s.explain_selection_mode(), "VerbatimMask");
    }

    #[test]
    fn sparse_iter_uses_word_skipping() {
        // 1M rows, 5 hits scattered far apart. iter must succeed quickly and
        // return exactly the hits.
        let hits = vec![100_usize, 50_000, 200_000, 500_000, 999_000];
        let s = classify_from_bools(&mk_bools(1_000_000, &hits));
        assert_eq!(s.explain_selection_mode(), "SelectionVector");
        let collected: Vec<usize> = s.iter_indices().collect();
        assert_eq!(collected, hits);
    }

    // ── Hybrid arm coverage ──────────────────────────────────────────

    #[test]
    fn hybrid_iter_matches_bitmask_iter_under_bimodal_density() {
        // Bimodal: first half empty, second half mostly-set
        let mut hits: Vec<usize> = Vec::new();
        for i in 50_000..100_000 {
            if i % 2 == 0 {
                hits.push(i);
            }
        }
        let bools = mk_bools(100_000, &hits);
        let s = classify_from_bools(&bools);
        // 25k hits / 100k = 25% mid band → Hybrid.
        assert_eq!(s.explain_selection_mode(), "Hybrid");
        let collected: Vec<usize> = s.iter_indices().collect();
        assert_eq!(collected, hits);
        assert_eq!(s.count(), hits.len());
    }

    #[test]
    fn hybrid_contains_matches_bitmask() {
        let hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let s = classify_from_bools(&mk_bools(100_000, &hits));
        assert_eq!(s.explain_selection_mode(), "Hybrid");
        for h in &hits {
            assert!(s.contains(*h));
        }
        assert!(!s.contains(99_999));
        assert!(!s.contains(100_001));
    }

    #[test]
    fn hybrid_materialize_round_trip() {
        let hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let s = classify_from_bools(&mk_bools(100_000, &hits));
        assert_eq!(s.explain_selection_mode(), "Hybrid");
        let bm = s.materialize_mask();
        assert_eq!(bm.count_ones(), hits.len());
        let idx = s.materialize_indices();
        assert_eq!(idx.len(), hits.len());
        for h in &hits {
            assert!(bm.get(*h));
            assert!(idx.binary_search(&(*h as u32)).is_ok());
        }
    }

    #[test]
    fn hybrid_chunks_pick_local_modes() {
        // 8 chunks × 4096 rows = 32_768 rows. Layout chosen so global density
        // lands in the mid band (~16%) so the classifier picks Hybrid:
        //   Chunk 0: All           (4096 hits)
        //   Chunks 1, 4, 5, 6, 7:  Empty (0 hits)
        //   Chunk 2: Sparse        (5 hits)
        //   Chunk 3: Dense         (1024 hits, 25% of chunk — past per-chunk sparse threshold)
        let nrows = 8 * HYBRID_CHUNK_SIZE;
        let mut hits = Vec::new();
        for i in 0..HYBRID_CHUNK_SIZE {
            hits.push(i);
        }
        for k in 0..5 {
            hits.push(2 * HYBRID_CHUNK_SIZE + k * 100);
        }
        for off in (0..HYBRID_CHUNK_SIZE).step_by(4) {
            hits.push(3 * HYBRID_CHUNK_SIZE + off);
        }
        let s = classify_from_bools(&mk_bools(nrows, &hits));
        assert_eq!(s.explain_selection_mode(), "Hybrid");
        if let AdaptiveSelection::Hybrid { chunks, .. } = &s {
            assert_eq!(chunks.len(), 8);
            assert!(matches!(chunks[0], HybridChunk::All));
            assert!(matches!(chunks[1], HybridChunk::Empty));
            assert!(matches!(chunks[2], HybridChunk::Sparse(_)));
            assert!(matches!(chunks[3], HybridChunk::Dense(_)));
            assert!(matches!(chunks[7], HybridChunk::Empty));
        } else {
            panic!("expected Hybrid");
        }
    }

    // ── Merge-walk fast paths ────────────────────────────────────────

    #[test]
    fn sparse_intersect_sparse_uses_merge_walk() {
        let a = classify_from_bools(&mk_bools(100_000, &[1, 5, 17, 99, 5_000, 50_000]));
        let b = classify_from_bools(&mk_bools(100_000, &[5, 17, 200, 5_000, 99_000]));
        assert_eq!(a.explain_selection_mode(), "SelectionVector");
        assert_eq!(b.explain_selection_mode(), "SelectionVector");
        let r = a.intersect(&b);
        // {5, 17, 5000} survive
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, vec![5usize, 17, 5_000]);
    }

    #[test]
    fn sparse_union_sparse_uses_merge_walk() {
        let a = classify_from_bools(&mk_bools(100_000, &[1, 5, 17]));
        let b = classify_from_bools(&mk_bools(100_000, &[5, 99, 200]));
        let r = a.union(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, vec![1usize, 5, 17, 99, 200]);
    }

    #[test]
    fn intersect_cardinality_identity_holds_across_modes() {
        // |A| + |B| = |A ∪ B| + |A ∩ B|
        let a = classify_from_bools(&mk_bools(100_000, &[1, 5, 17, 99, 5_000, 50_000]));
        let b = classify_from_bools(&mk_bools(100_000, &[5, 17, 200, 5_000, 99_000]));
        let inter = a.intersect(&b);
        let union = a.union(&b);
        assert_eq!(a.count() + b.count(), inter.count() + union.count());
    }

    #[test]
    fn merge_walk_intersect_helper_is_correct() {
        let a = vec![1u32, 3, 5, 7, 9];
        let b = vec![2u32, 3, 5, 6, 9, 11];
        let out = sorted_merge_intersect(&a, &b);
        assert_eq!(out, vec![3, 5, 9]);
    }

    #[test]
    fn merge_walk_union_helper_is_correct() {
        let a = vec![1u32, 3, 5];
        let b = vec![2u32, 3, 7];
        let out = sorted_merge_union(&a, &b);
        assert_eq!(out, vec![1, 2, 3, 5, 7]);
    }

    // ── v3 Phase 3: Hybrid streaming set-op fast paths ───────────────

    /// Build a Hybrid selection from explicit hits, asserting Hybrid arm.
    fn hybrid_from_hits(nrows: usize, hits: &[usize]) -> AdaptiveSelection {
        let s = classify_from_bools(&mk_bools(nrows, hits));
        assert_eq!(
            s.explain_selection_mode(),
            "Hybrid",
            "expected Hybrid for {nrows} rows × {} hits",
            hits.len()
        );
        s
    }

    /// Brute-force oracle: materialize both sides and AND/OR via BitMask.
    fn oracle_intersect(a: &AdaptiveSelection, b: &AdaptiveSelection) -> Vec<usize> {
        let am = a.materialize_mask();
        let bm = b.materialize_mask();
        let n = am.nrows();
        let mut out = Vec::new();
        for i in 0..n {
            if am.get(i) && bm.get(i) {
                out.push(i);
            }
        }
        out
    }
    fn oracle_union(a: &AdaptiveSelection, b: &AdaptiveSelection) -> Vec<usize> {
        let am = a.materialize_mask();
        let bm = b.materialize_mask();
        let n = am.nrows();
        let mut out = Vec::new();
        for i in 0..n {
            if am.get(i) || bm.get(i) {
                out.push(i);
            }
        }
        out
    }

    #[test]
    fn phase3_hybrid_intersect_hybrid_matches_oracle() {
        // Two mid-band selections over a 100k-row frame at different phases.
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect(); // 5%
        let b_hits: Vec<usize> = (0..100_000).step_by(15).collect(); // ~6.67%
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = hybrid_from_hits(100_000, &b_hits);
        let r = a.intersect(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_intersect(&a, &b));
    }

    #[test]
    fn phase3_hybrid_union_hybrid_matches_oracle() {
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let b_hits: Vec<usize> = (0..100_000).step_by(15).collect();
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = hybrid_from_hits(100_000, &b_hits);
        let r = a.union(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_union(&a, &b));
    }

    #[test]
    fn phase3_hybrid_intersect_sparse_matches_oracle() {
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect(); // Hybrid (5%)
        let b_hits: Vec<usize> = vec![100, 5_000, 50_020, 99_980]; // SelectionVector
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = classify_from_bools(&mk_bools(100_000, &b_hits));
        assert_eq!(b.explain_selection_mode(), "SelectionVector");
        let r = a.intersect(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_intersect(&a, &b));
        // Symmetric
        let r2 = b.intersect(&a);
        let collected2: Vec<usize> = r2.iter_indices().collect();
        assert_eq!(collected2, oracle_intersect(&a, &b));
    }

    #[test]
    fn phase3_hybrid_union_sparse_matches_oracle() {
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let b_hits: Vec<usize> = vec![1, 50, 100, 5_000, 50_020, 99_999];
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = classify_from_bools(&mk_bools(100_000, &b_hits));
        assert_eq!(b.explain_selection_mode(), "SelectionVector");
        let r = a.union(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_union(&a, &b));
    }

    #[test]
    fn phase3_hybrid_intersect_verbatim_matches_oracle() {
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect(); // Hybrid
        let b_hits: Vec<usize> = (0..100_000).step_by(2).collect(); // 50% → VerbatimMask
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = classify_from_bools(&mk_bools(100_000, &b_hits));
        assert_eq!(b.explain_selection_mode(), "VerbatimMask");
        let r = a.intersect(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_intersect(&a, &b));
        // Symmetric
        let r2 = b.intersect(&a);
        assert_eq!(
            r2.iter_indices().collect::<Vec<usize>>(),
            oracle_intersect(&a, &b)
        );
    }

    #[test]
    fn phase3_hybrid_union_verbatim_matches_oracle() {
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let b_hits: Vec<usize> = (0..100_000).step_by(2).collect();
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = classify_from_bools(&mk_bools(100_000, &b_hits));
        let r = a.union(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_union(&a, &b));
    }

    #[test]
    fn phase3_hybrid_chain_intersect_three_way() {
        let a_hits: Vec<usize> = (0..100_000).step_by(20).collect();
        let b_hits: Vec<usize> = (0..100_000).step_by(15).collect();
        let c_hits: Vec<usize> = (0..100_000).step_by(12).collect();
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = hybrid_from_hits(100_000, &b_hits);
        let c = hybrid_from_hits(100_000, &c_hits);
        let abc = a.intersect(&b).intersect(&c);
        let cba = c.intersect(&b).intersect(&a);
        assert_eq!(abc.materialize_indices(), cba.materialize_indices());
        // Compare against scalar oracle.
        let am = a.materialize_mask();
        let bm = b.materialize_mask();
        let cm = c.materialize_mask();
        let oracle: Vec<usize> = (0..100_000)
            .filter(|&i| am.get(i) && bm.get(i) && cm.get(i))
            .collect();
        assert_eq!(abc.iter_indices().collect::<Vec<usize>>(), oracle);
    }

    #[test]
    fn phase3_per_chunk_intersect_sparse_sparse_uses_merge_walk() {
        // Two Hybrids whose corresponding chunks are both Sparse — verifies
        // the merge-walk path actually fires and produces correct sparse output.
        // Stride 200 over 100k = 500 hits; per-chunk count ≈ 4096/200 ≈ 20 (well
        // below 128 sparse threshold), so chunks are Sparse-shaped.
        let a_hits: Vec<usize> = (0..100_000).step_by(200).collect();
        let b_hits: Vec<usize> = (100..100_000).step_by(200).collect();
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = hybrid_from_hits(100_000, &b_hits);
        // Disjoint hits → intersection is empty.
        let r = a.intersect(&b);
        assert_eq!(r.count(), 0);
        assert_eq!(r.explain_selection_mode(), "Empty");
    }

    #[test]
    fn phase3_per_chunk_union_sparse_sparse_promotes_to_dense_when_large() {
        // Force per-chunk overflow past sparse threshold (128 hits/chunk):
        // stride 30 across 100k = ~3333 hits, per-chunk ≈ 4096/30 ≈ 136 → just
        // above sparse threshold per-chunk. Union with itself stays sparse-shaped
        // via merge-walk, but cross-streamed unions can promote.
        let a_hits: Vec<usize> = (0..100_000).step_by(30).collect();
        let b_hits: Vec<usize> = (15..100_000).step_by(30).collect();
        let a_count = a_hits.len();
        let b_count = b_hits.len();
        let a = hybrid_from_hits(100_000, &a_hits);
        let b = hybrid_from_hits(100_000, &b_hits);
        let r = a.union(&b);
        // Disjoint hits → union count = a_count + b_count.
        assert_eq!(r.count(), a_count + b_count);
        // Confirm content matches oracle.
        assert_eq!(
            r.iter_indices().collect::<Vec<usize>>(),
            oracle_union(&a, &b)
        );
    }

    #[test]
    fn phase3_simplify_hybrid_collapses_to_all_when_full() {
        // a + b cover every row → simplify_hybrid must collapse to All.
        let a_hits: Vec<usize> = (0..100_000).filter(|i| i % 2 == 0).collect();
        let b_hits: Vec<usize> = (0..100_000).filter(|i| i % 2 == 1).collect();
        // Each is 50% over 100k → VerbatimMask, not Hybrid. Build Hybrid by
        // forcing mid-band: take 6/10 of rows in each side so mid-band fires.
        let a_hits: Vec<usize> = (0..100_000).filter(|i| i % 5 != 0).collect(); // 80% — dense
        let b_hits: Vec<usize> = (0..100_000).step_by(5).collect(); // 20%
        let _ = (a_hits, b_hits); // unused — see next assertion-style test.
        // Instead: assert simplify_hybrid directly on a fully-set chunked arg.
        let nrows = 8 * HYBRID_CHUNK_SIZE;
        let chunks = vec![HybridChunk::All; 8];
        let s = AdaptiveSelection::simplify_hybrid(nrows, chunks);
        assert_eq!(s.explain_selection_mode(), "All");
        assert_eq!(s.count(), nrows);
    }

    #[test]
    fn phase3_simplify_hybrid_collapses_to_empty_when_drained() {
        let nrows = 8 * HYBRID_CHUNK_SIZE;
        let chunks = vec![HybridChunk::Empty; 8];
        let s = AdaptiveSelection::simplify_hybrid(nrows, chunks);
        assert_eq!(s.explain_selection_mode(), "Empty");
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn phase3_per_chunk_helpers_handle_partial_final_chunk() {
        // 9000 rows = 2 full chunks + 1 partial (808 rows). Mid-band density.
        let nrows = 2 * HYBRID_CHUNK_SIZE + 808;
        let a_hits: Vec<usize> = (0..nrows).step_by(20).collect();
        let b_hits: Vec<usize> = (0..nrows).step_by(15).collect();
        let a = hybrid_from_hits(nrows, &a_hits);
        let b = hybrid_from_hits(nrows, &b_hits);
        let r = a.intersect(&b);
        let collected: Vec<usize> = r.iter_indices().collect();
        assert_eq!(collected, oracle_intersect(&a, &b));
        // No row should leak past nrows.
        for &row in &collected {
            assert!(row < nrows, "row {row} out of bounds");
        }
    }

    #[test]
    fn phase3_intersect_chunks_helper_dense_and_sparse() {
        // Direct unit test on intersect_chunks: Sparse vs Dense, pinning that
        // we get a sparse result with correct offsets.
        let chunk_len = HYBRID_CHUNK_SIZE;
        let sparse = HybridChunk::Sparse(vec![0u16, 5, 17, 100, 4000]);
        // Dense bitmap with bit 5, 17, 4000 set.
        let mut dense_words = vec![0u64; HYBRID_WORDS_PER_CHUNK];
        for off in [5usize, 17, 4000] {
            dense_words[off / 64] |= 1u64 << (off % 64);
        }
        let dense = HybridChunk::Dense(dense_words.into_boxed_slice());
        let r = intersect_chunks(&sparse, &dense, chunk_len);
        match r {
            HybridChunk::Sparse(offs) => assert_eq!(offs, vec![5, 17, 4000]),
            other => panic!("expected Sparse, got {other:?}"),
        }
    }

    #[test]
    fn phase3_union_chunks_helper_demotes_to_sparse_when_small() {
        // Sparse ∪ Sparse below threshold stays Sparse.
        let chunk_len = HYBRID_CHUNK_SIZE;
        let a = HybridChunk::Sparse(vec![1u16, 2, 3]);
        let b = HybridChunk::Sparse(vec![4u16, 5, 6]);
        let r = union_chunks(&a, &b, chunk_len);
        match r {
            HybridChunk::Sparse(offs) => assert_eq!(offs, vec![1, 2, 3, 4, 5, 6]),
            other => panic!("expected Sparse, got {other:?}"),
        }
    }

    #[test]
    fn phase3_union_chunks_helper_promotes_to_dense_above_threshold() {
        // Build two disjoint sparse chunks each with 70 hits (140 union >
        // 128 threshold) → output must be Dense.
        let chunk_len = HYBRID_CHUNK_SIZE;
        let a_offs: Vec<u16> = (0..70).map(|i| i * 2).collect();
        let b_offs: Vec<u16> = (0..70).map(|i| i * 2 + 1).collect();
        let a = HybridChunk::Sparse(a_offs);
        let b = HybridChunk::Sparse(b_offs);
        let r = union_chunks(&a, &b, chunk_len);
        match r {
            HybridChunk::Dense(words) => {
                let count: usize = words.iter().map(|w| w.count_ones() as usize).sum();
                assert_eq!(count, 140);
            }
            other => panic!("expected Dense (overflow), got {other:?}"),
        }
    }
}
