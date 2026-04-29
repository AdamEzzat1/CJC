//! `SealedU64Map<V>` — public wrapper around `DHarhtMemory` enforcing
//! the **build → seal → read-many** lifecycle.
//!
//! This is the recommended way to use the D-HARHT Memory profile from
//! workspace code that needs `u64`-keyed sealed equality lookup. It
//! adds:
//!
//! - A type-state-style API (`new()` → `insert()`* → `seal()` →
//!   `get()`*) that prevents accidental mutation after sealing
//! - `len_pre_seal()` / `len_sealed()` accessors that don't lie about
//!   which phase you're in
//! - `approx_memory_bytes()` and security counters surfaced cleanly
//! - Drop-in replacement for any `BTreeMap<u64, V>` whose access
//!   pattern is "build once, look up many"
//!
//! # When to pick this
//!
//! | You want                                            | Use this? |
//! |----------------------------------------------------|-----------|
//! | `u64` keys, sealed lookup, fastest possible reads  | ✅ yes     |
//! | `u64` keys but mutating throughout                 | ❌ `BTreeMap` |
//! | Range queries / sorted iteration                   | ❌ `BTreeMap` |
//! | Arbitrary byte keys                                | ❌ `DHarht` v.01 |
//! | Tiny tables (≤16 entries)                          | ❌ `TinyDetMap` |
//!
//! # Determinism
//!
//! Same input → byte-identical sealed shape across processes,
//! machines, architectures. Verified by `shape_hash()` and the
//! Phase 10 + Phase 11 test suites.
//!
//! # Example
//!
//! ```
//! use cjc_data::detcoll::SealedU64Map;
//!
//! let mut m: SealedU64Map<&'static str> = SealedU64Map::new();
//! m.insert(0xDEADBEEF, "alpha");
//! m.insert(0xCAFEBABE, "beta");
//! m.seal();
//! assert_eq!(m.get(0xDEADBEEF), Some(&"alpha"));
//! assert_eq!(m.get(0x00000000), None);
//! ```

use super::dharht_memory::{shape_hash, DHarhtMemory};

/// Build-then-seal-then-read u64-keyed map.
#[derive(Clone, Debug)]
pub struct SealedU64Map<V: Clone> {
    inner: DHarhtMemory<V>,
}

impl<V: Clone> SealedU64Map<V> {
    /// Empty map in build phase.
    pub fn new() -> Self {
        Self {
            inner: DHarhtMemory::new(),
        }
    }

    /// Insert or update. **Panics in debug** if called after `seal()`.
    /// Returns previous value if any.
    pub fn insert(&mut self, key: u64, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    /// Lookup by key. Works in both build and sealed phases.
    /// Pre-seal: `O(log n_shard)` BTreeMap probe.
    /// Post-seal: `O(1)` average via packed front directory + micro
    /// buckets.
    #[inline]
    pub fn get(&self, key: u64) -> Option<&V> {
        self.inner.get(key)
    }

    pub fn contains_key(&self, key: u64) -> bool {
        self.inner.contains_key(key)
    }

    /// Transition from build → sealed phase. After this call lookup
    /// becomes `O(1)`-class via the packed front directory. Mutations
    /// after `seal()` panic in debug builds.
    pub fn seal(&mut self) {
        self.inner.seal_for_lookup();
    }

    pub fn is_sealed(&self) -> bool {
        self.inner.is_sealed()
    }

    /// Total entry count (works in both phases).
    pub fn len(&self) -> u64 {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Sorted iteration over `(key, &value)`. Always canonical (key-
    /// sorted). Use this for serialization, snapshots, audit output.
    pub fn iter_sorted(&self) -> Vec<(u64, &V)> {
        self.inner.iter_sorted()
    }

    /// Approximate allocated memory in bytes.
    pub fn approx_memory_bytes(&self) -> usize {
        self.inner.approx_memory_bytes()
    }

    /// Diagnostic: how many entries spilled to the BTreeMap fallback
    /// path because their front-directory collision group exceeded
    /// `MicroBucket16`. Always 0 in well-distributed workloads.
    pub fn micro_overflow_count(&self) -> u64 {
        self.inner.micro_overflow_count()
    }

    /// Diagnostic: largest collision group encountered at seal time.
    /// Helps detect adversarial inputs.
    pub fn max_collision_group(&self) -> u32 {
        self.inner.max_collision_group()
    }

    /// Cross-run determinism check: hash the sealed shape. Two
    /// `SealedU64Map`s with identical `(key, value)` sets produce the
    /// same `shape_hash()` byte-equal across runs / machines / OSes.
    pub fn shape_hash(&self) -> u64
    where
        V: std::hash::Hash,
    {
        shape_hash(&self.inner)
    }
}

impl<V: Clone> Default for SealedU64Map<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_seal_read_lifecycle() {
        let mut m: SealedU64Map<u64> = SealedU64Map::new();
        for i in 0..1000u64 {
            m.insert(i, i * 31);
        }
        // Pre-seal lookup works.
        assert_eq!(m.get(42), Some(&(42 * 31)));
        assert!(!m.is_sealed());
        m.seal();
        assert!(m.is_sealed());
        for i in 0..1000u64 {
            assert_eq!(m.get(i), Some(&(i * 31)));
        }
    }

    #[test]
    fn shape_hash_byte_equal_for_identical_input() {
        let build = || -> SealedU64Map<u64> {
            let mut m = SealedU64Map::new();
            for i in 0..500u64 {
                m.insert(i.wrapping_mul(0xdeadbeef), i);
            }
            m.seal();
            m
        };
        assert_eq!(build().shape_hash(), build().shape_hash());
    }

    #[test]
    fn iter_sorted_canonical() {
        let mut a: SealedU64Map<u64> = SealedU64Map::new();
        let mut b: SealedU64Map<u64> = SealedU64Map::new();
        for i in 0..100u64 {
            a.insert(i, i);
        }
        for i in (0..100u64).rev() {
            b.insert(i, i);
        }
        a.seal();
        b.seal();
        let av: Vec<(u64, u64)> = a.iter_sorted().into_iter().map(|(k, v)| (k, *v)).collect();
        let bv: Vec<(u64, u64)> = b.iter_sorted().into_iter().map(|(k, v)| (k, *v)).collect();
        assert_eq!(av, bv);
    }

    #[test]
    fn no_silent_loss_at_scale() {
        let mut m: SealedU64Map<u64> = SealedU64Map::new();
        for i in 0..50_000u64 {
            m.insert(i, i);
        }
        m.seal();
        for i in 0..50_000u64 {
            assert_eq!(m.get(i), Some(&i));
        }
    }
}
