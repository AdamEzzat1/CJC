//! Phase 0.10 Track Q1 — D-HARHT-backed route memoization cache.
//!
//! [`RouteCache`] memoizes `prefix -> RouteEvidence` so a repeated
//! [`descend`](crate::graph::AdaptiveBeliefGraph::descend) over the same
//! routing prefix skips the radix walk. It is **opt-in**
//! ([`enable_route_cache`](crate::graph::AdaptiveBeliefGraph::enable_route_cache))
//! and **in-memory only** — never serialized, never part of the audit
//! chain. A cache entry can therefore never change `descend`'s
//! observable output: a hit returns exactly the [`RouteEvidence`] the
//! walk would have produced (pinned by the cache-vs-walk parity tests).
//!
//! # Invalidation
//!
//! The cache is cleared whenever a topology-changing audit event is
//! appended (`NodeAdded`, `Grow`, `Split`, `Merge`, `Prune`,
//! `Compress` — see `AdaptiveBeliefGraph::append_event`). Routing
//! depends only on the tree's `children` edges, so a non-topology
//! event (a `BeliefUpdate`, a `TrainStep`, a `ChildrenPromoted` —
//! which preserves every key→child mapping) leaves cached routes valid.
//!
//! # Key encoding
//!
//! D-HARHT keys are `u64`; a routing prefix is `&[u8]` of length
//! `codebook.n_dims`. Prefixes of length `1..=7` pack **losslessly**
//! into a `u64` — the length is stored in the leading bits, so `[1]`
//! and `[0, 1]` never collide — giving a collision-free, exact-match
//! cache. Prefixes of length `0` or `> 7` are uncacheable: `descend`
//! falls through to the walk for them (still correct, just not
//! memoized). ABNG codebooks are low-dimensional (Phase 0.9 routed on
//! 4 features), so the cacheable range covers the realistic workload.

use cjc_dharht::{DHarht, LookupProfile};

use crate::route::RouteEvidence;

/// D-HARHT shard count. Must be a power of two (`DHarht::new`'s
/// contract); 256 matches the reference memory-profile configuration.
const ROUTE_CACHE_SHARDS: usize = 256;

/// Longest prefix that packs losslessly into a `u64` key (1 length
/// nibble + up to 7 data bytes).
const MAX_CACHEABLE_PREFIX: usize = 7;

/// Opt-in `prefix -> RouteEvidence` memoizer backing
/// [`AdaptiveBeliefGraph::descend`](crate::graph::AdaptiveBeliefGraph::descend).
#[derive(Debug, Clone)]
pub struct RouteCache {
    table: DHarht<RouteEvidence>,
    hits: u64,
    misses: u64,
    skips: u64,
}

impl RouteCache {
    /// A fresh, empty cache (D-HARHT memory profile, 256 shards).
    pub fn new() -> Self {
        Self {
            table: Self::fresh_table(),
            hits: 0,
            misses: 0,
            skips: 0,
        }
    }

    fn fresh_table() -> DHarht<RouteEvidence> {
        let mut table = DHarht::new(ROUTE_CACHE_SHARDS);
        table.set_lookup_profile(LookupProfile::Memory);
        table
    }

    /// Pack a routing prefix into a collision-free `u64` key. The
    /// length is folded into the leading bits, so prefixes that share
    /// a byte suffix but differ in length get distinct keys. Returns
    /// `None` for the uncacheable lengths (`0`, or `> 7`).
    fn pack_key(prefix: &[u8]) -> Option<u64> {
        let n = prefix.len();
        if n == 0 || n > MAX_CACHEABLE_PREFIX {
            return None;
        }
        let mut key = n as u64;
        for &byte in prefix {
            key = (key << 8) | byte as u64;
        }
        Some(key)
    }

    /// Look up a memoized route. On a hit returns the stored evidence;
    /// on a miss (or an uncacheable prefix) returns `None`, and the
    /// caller falls through to the walk. Updates the hit/miss/skip
    /// counters.
    pub fn lookup(&mut self, prefix: &[u8]) -> Option<RouteEvidence> {
        match Self::pack_key(prefix) {
            None => {
                self.skips += 1;
                None
            }
            Some(key) => match self.table.get(key) {
                Some(evidence) => {
                    self.hits += 1;
                    Some(evidence.clone())
                }
                None => {
                    self.misses += 1;
                    None
                }
            },
        }
    }

    /// Memoize the result of a walk. A no-op for uncacheable prefixes
    /// (those already counted as a skip by [`lookup`](Self::lookup)).
    pub fn record(&mut self, prefix: &[u8], evidence: &RouteEvidence) {
        if let Some(key) = Self::pack_key(prefix) {
            self.table.insert(key, evidence.clone());
        }
    }

    /// Drop every memoized route. Called on any topology change. The
    /// cumulative hit/miss/skip counters are observability over the
    /// cache's whole lifetime, not cache contents, so they survive.
    pub fn clear(&mut self) {
        self.table = Self::fresh_table();
    }

    /// Cumulative cache hits over this cache's lifetime.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cumulative cache misses (a cacheable prefix not yet memoized).
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Cumulative skips (prefix length `0` or `> 7` — uncacheable).
    pub fn skips(&self) -> u64 {
        self.skips
    }
}

impl Default for RouteCache {
    fn default() -> Self {
        Self::new()
    }
}
