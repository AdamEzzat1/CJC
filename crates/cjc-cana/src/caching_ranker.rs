//! Phase 5 — `CachingPassRanker`: eliminate the recommendation overhead
//! on repeat compilations.
//!
//! ## What this does
//!
//! Wraps a `PassRanker` with an in-memory cache keyed by
//! `(ProgramHash, CostModel.version())`. On a cache hit, the wrapped
//! ranker is skipped entirely — the cached `RankingReport` is returned.
//! On a miss, the inner ranker runs and the result is cached.
//!
//! ## Why this matters
//!
//! Phase 2's benchmark showed CANA's recommendation pipeline adds ~1.66×
//! median compile-time overhead (analyze → rank → convert). On the
//! second compilation of the same program with the same model, the
//! recommendation hasn't changed — we shouldn't pay the cost twice.
//! With caching:
//!
//! - First compile: cache miss → full pipeline (no change vs Phase 2)
//! - Second+ compile: cache hit → near-zero overhead
//!
//! For interactive workflows (rapid iteration, REPL re-evaluation,
//! incremental compilation), this turns CANA from a 1.66× slowdown into
//! a no-op after the first run.
//!
//! ## Determinism contract
//!
//! Caching preserves determinism trivially:
//! - The cache key includes everything that determines the recommendation
//!   (ProgramHash captures all inputs to `extract`; CostModel.version()
//!   invalidates when coefficients change).
//! - The cached value is a `RankingReport`, which is itself deterministic.
//! - Cache hits return *byte-identical* reports to a miss-then-compute.
//!
//! Same MIR + same model → same recommendation, whether cached or not.
//!
//! ## Eviction policy
//!
//! Phase 5's policy is intentionally simple: when at capacity, drop one
//! arbitrary entry (the BTreeMap's first key). A proper LRU would
//! require extra bookkeeping; for current workloads where the cache is
//! checked at compile time (not per-execution), simple eviction is fine.
//!
//! Default capacity: 256 entries (covers ~most multi-file projects).

use std::cell::RefCell;
use std::collections::BTreeMap;

use cjc_mir::MirProgram;

use crate::cost_model::CostModel;
use crate::features::CanaFeatures;
use crate::hash::ProgramHash;
use crate::legality::LegalityGate;
use crate::pass_ranker::{PassRanker, RankingReport};

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Aggregated cache statistics. Exposed via [`CachingPassRanker::stats`]
/// for benchmark and observability surfaces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of `rank()` calls that hit the cache.
    pub hits: u64,
    /// Number of `rank()` calls that missed (and ran the inner ranker).
    pub misses: u64,
    /// Number of evictions performed by the simple capacity policy.
    pub evictions: u64,
    /// Current number of entries held.
    pub size: usize,
    /// Maximum capacity configured for this cache.
    pub capacity: usize,
}

impl CacheStats {
    /// Total `rank()` calls (hits + misses).
    pub fn total_calls(&self) -> u64 {
        self.hits + self.misses
    }

    /// Cache hit rate in `[0, 1]`. Returns 0.0 if no calls yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_calls();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// CachingPassRanker
// ---------------------------------------------------------------------------

/// A `PassRanker` wrapper that caches recommendations by
/// `(ProgramHash, CostModel.version())`.
///
/// Public API mirrors `PassRanker::rank` — the cache is transparent.
/// Callers that need to observe cache behaviour can call
/// [`CachingPassRanker::stats`].
pub struct CachingPassRanker<M: CostModel, G: LegalityGate> {
    inner: PassRanker<M, G>,
    cache: RefCell<BTreeMap<(ProgramHash, u32), RankingReport>>,
    /// Tracked separately from the cache map's len() so we don't borrow
    /// the RefCell to read stats.
    state: RefCell<CacheStats>,
}

/// Default cache capacity (entries).
///
/// Standalone rather than associated-const so it can be referenced
/// without specifying the `<M, G>` type parameters of
/// [`CachingPassRanker`].
pub const DEFAULT_CACHE_CAPACITY: usize = 256;

impl<M: CostModel, G: LegalityGate> CachingPassRanker<M, G> {
    /// Wrap an existing `PassRanker` with caching at the default capacity.
    pub fn new(inner: PassRanker<M, G>) -> Self {
        Self::with_capacity(inner, DEFAULT_CACHE_CAPACITY)
    }

    /// Wrap an existing `PassRanker` with caching at a specified capacity.
    /// Capacity is clamped to at least 1.
    pub fn with_capacity(inner: PassRanker<M, G>, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner,
            cache: RefCell::new(BTreeMap::new()),
            state: RefCell::new(CacheStats {
                capacity,
                ..CacheStats::default()
            }),
        }
    }

    /// Borrow the wrapped ranker. Useful for tests and audit surfaces.
    pub fn inner(&self) -> &PassRanker<M, G> {
        &self.inner
    }

    /// Snapshot the cache stats. Cheap (atomic-like read; the underlying
    /// RefCell borrow is held for the duration of this call only).
    pub fn stats(&self) -> CacheStats {
        *self.state.borrow()
    }

    /// Clear the cache and reset counters (preserves capacity).
    pub fn clear(&self) {
        self.cache.borrow_mut().clear();
        let mut state = self.state.borrow_mut();
        let cap = state.capacity;
        *state = CacheStats {
            capacity: cap,
            ..CacheStats::default()
        };
    }

    /// Rank a program with cache consultation.
    ///
    /// On cache hit: returns the cached report (clone). The wrapped
    /// ranker is NOT consulted. Cost model and legality gate are NOT
    /// invoked.
    ///
    /// On cache miss: defers to the wrapped ranker, caches the result,
    /// and returns it.
    ///
    /// Determinism: cache hits return byte-identical reports to a fresh
    /// computation (by construction — the cache stores the exact
    /// `RankingReport` produced by the inner ranker).
    pub fn rank(&self, program: &MirProgram, features: &CanaFeatures) -> RankingReport {
        let key = (
            features.program_hash,
            self.inner.cost_model().version(),
        );

        // Hit path: fast.
        if let Some(cached) = self.cache.borrow().get(&key) {
            let mut state = self.state.borrow_mut();
            state.hits = state.hits.saturating_add(1);
            return cached.clone();
        }

        // Miss path: compute, then store.
        let report = self.inner.rank(program, features);

        let mut cache = self.cache.borrow_mut();
        let mut state = self.state.borrow_mut();
        state.misses = state.misses.saturating_add(1);

        // Evict if at capacity. Simple policy: drop the first entry by
        // BTreeMap iteration order. For Phase 5 this is intentional —
        // it's deterministic (BTreeMap key order is stable) and cheap.
        if cache.len() >= state.capacity {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
                state.evictions = state.evictions.saturating_add(1);
            }
        }
        cache.insert(key, report.clone());
        state.size = cache.len();

        report
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Build a caching ranker with Phase-2 defaults (LinearCostModel +
/// DefaultLegalityGate) at the default cache capacity.
///
/// This is the standard entry point for Phase-5-enabled callers.
pub fn default_caching_ranker() -> CachingPassRanker<
    crate::linear_cost_model::LinearCostModel,
    crate::legality::DefaultLegalityGate,
> {
    CachingPassRanker::new(crate::pass_ranker::default_ranker())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::extract;
    use crate::pass_ranker::default_ranker;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    fn simple_program(name: &str) -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: name.to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: vec![MirStmt::Expr(MirExpr {
                        kind: MirExprKind::IntLit(42),
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

    #[test]
    fn fresh_cache_has_zero_stats() {
        let r = default_caching_ranker();
        let s = r.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.size, 0);
        assert_eq!(s.capacity, DEFAULT_CACHE_CAPACITY);
    }

    #[test]
    fn first_call_is_a_miss_second_is_a_hit() {
        let prog = simple_program("f");
        let feats = extract(&prog);
        let r = default_caching_ranker();

        let _ = r.rank(&prog, &feats);
        let s = r.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 0);

        let _ = r.rank(&prog, &feats);
        let s = r.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 1);
    }

    #[test]
    fn cached_report_equals_fresh_report() {
        // Byte-identicality between cache hit and a fresh computation.
        let prog = simple_program("f");
        let feats = extract(&prog);

        let cached_ranker = default_caching_ranker();
        let fresh_ranker = default_ranker();

        let cached_first = cached_ranker.rank(&prog, &feats); // miss, fills cache
        let cached_second = cached_ranker.rank(&prog, &feats); // hit
        let fresh = fresh_ranker.rank(&prog, &feats);

        assert_eq!(cached_first.per_fn, cached_second.per_fn);
        assert_eq!(cached_first.per_fn, fresh.per_fn);
        assert_eq!(cached_first.sequence, fresh.sequence);
    }

    #[test]
    fn distinct_programs_produce_distinct_misses() {
        let r = default_caching_ranker();

        let pa = simple_program("alpha");
        let fa = extract(&pa);
        let _ = r.rank(&pa, &fa);

        let pb = simple_program("beta");
        let fb = extract(&pb);
        let _ = r.rank(&pb, &fb);

        let s = r.stats();
        assert_eq!(s.misses, 2);
        assert_eq!(s.hits, 0);
        assert_eq!(s.size, 2);
    }

    #[test]
    fn hit_rate_is_meaningful_after_many_calls() {
        let prog = simple_program("f");
        let feats = extract(&prog);
        let r = default_caching_ranker();

        for _ in 0..10 {
            let _ = r.rank(&prog, &feats);
        }

        let s = r.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 9);
        assert!((s.hit_rate() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn capacity_eviction_drops_old_entries() {
        let r = CachingPassRanker::with_capacity(default_ranker(), 2);

        let pa = simple_program("a");
        let pb = simple_program("b");
        let pc = simple_program("c");
        let fa = extract(&pa);
        let fb = extract(&pb);
        let fc = extract(&pc);

        let _ = r.rank(&pa, &fa); // miss, size=1
        let _ = r.rank(&pb, &fb); // miss, size=2
        let _ = r.rank(&pc, &fc); // miss, evict, size=2

        let s = r.stats();
        assert_eq!(s.size, 2);
        assert_eq!(s.evictions, 1);
        assert_eq!(s.misses, 3);
    }

    #[test]
    fn clear_resets_state_but_keeps_capacity() {
        let r = CachingPassRanker::with_capacity(default_ranker(), 10);
        let prog = simple_program("f");
        let feats = extract(&prog);

        let _ = r.rank(&prog, &feats);
        let _ = r.rank(&prog, &feats);
        assert_eq!(r.stats().hits, 1);

        r.clear();
        let s = r.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.size, 0);
        assert_eq!(s.capacity, 10);
    }

    #[test]
    fn zero_capacity_is_clamped_to_one() {
        let r = CachingPassRanker::with_capacity(default_ranker(), 0);
        assert_eq!(r.stats().capacity, 1);
    }

    #[test]
    fn cache_is_deterministic_across_runs() {
        // Same program, same model: cached entries must be byte-identical
        // across CachingPassRanker instances.
        let prog = simple_program("f");
        let feats = extract(&prog);

        let r1 = default_caching_ranker();
        let report1 = r1.rank(&prog, &feats);

        let r2 = default_caching_ranker();
        let report2 = r2.rank(&prog, &feats);

        assert_eq!(report1.per_fn, report2.per_fn);
        assert_eq!(report1.sequence, report2.sequence);
    }

    #[test]
    fn fifty_iteration_repeat_after_first_miss_produces_one_miss() {
        // Stress test: 50 calls on the same program should produce
        // exactly 1 miss and 49 hits.
        let prog = simple_program("f");
        let feats = extract(&prog);
        let r = default_caching_ranker();

        for _ in 0..50 {
            let _ = r.rank(&prog, &feats);
        }

        let s = r.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 49);
    }
}
