//! `SortedVecMap<K, V>` — small-to-medium sealed sorted map.
//!
//! Sorted `Vec<(K, V)>` with binary-search lookup. The "graduation" of
//! `TinyDetMap` for sizes ~16–~10_000 where `BTreeMap`'s pointer-chasing
//! cost outweighs `Vec`'s contiguous-memory wins.
//!
//! Iteration is sorted by `K`, so output is canonical. Iteration cost
//! is `O(n)`; lookup is `O(log n)`. Insert/remove cost is `O(n)`
//! (shift), so this is **not** the structure for hot mutation paths.
//!
//! Two construction modes:
//! - `insert` one-by-one — `O(n²)` total, fine for sealed/build-once.
//! - `from_sorted_unique` — `O(n)`, caller has already sorted.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct SortedVecMap<K: Ord, V> {
    entries: Vec<(K, V)>,
}

impl<K: Ord, V> Default for SortedVecMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> SortedVecMap<K, V> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
        }
    }

    /// Construct from a vector that the caller asserts is sorted by
    /// `K` and contains no duplicates. Debug-build asserts both
    /// invariants; release builds trust the caller.
    pub fn from_sorted_unique(entries: Vec<(K, V)>) -> Self {
        debug_assert!(
            entries.windows(2).all(|w| w[0].0 < w[1].0),
            "from_sorted_unique requires strictly sorted unique keys"
        );
        Self { entries }
    }

    /// Construct from any iterator. Sorts and dedups (last value wins).
    pub fn from_iter_unsorted<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self
    where
        K: Clone,
    {
        let mut entries: Vec<(K, V)> = iter.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        // Dedup keeping the *last* occurrence per key, which matches
        // `BTreeMap::insert` semantics.
        let mut deduped: Vec<(K, V)> = Vec::with_capacity(entries.len());
        for (k, v) in entries.into_iter() {
            if let Some(last) = deduped.last_mut() {
                if last.0 == k {
                    last.1 = v;
                    continue;
                }
            }
            deduped.push((k, v));
        }
        Self { entries: deduped }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `O(log n)` lookup. Full key equality on hit (binary search
    /// returns the exact match index).
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|i| &self.entries[i].1)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .is_ok()
    }

    /// Insert or update. `O(n)` worst case (shift). Returns previous
    /// value on update.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.entries.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            Err(i) => {
                self.entries.insert(i, (key, value));
                None
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.entries.binary_search_by(|(k, _)| k.cmp(key)) {
            Ok(i) => Some(self.entries.remove(i).1),
            Err(_) => None,
        }
    }

    /// Iterate `(&K, &V)` pairs sorted by `K`. Canonical order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// Range iteration. `[start, end)`.
    pub fn range<Q>(&self, start: &Q, end: &Q) -> impl Iterator<Item = (&K, &V)> + '_
    where
        K: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // Find lower and upper bounds via binary search.
        let lo = self
            .entries
            .binary_search_by(|(k, _)| match k.borrow().cmp(start) {
                Ordering::Less => Ordering::Less,
                _ => Ordering::Greater,
            })
            .unwrap_or_else(|i| i);
        let hi = self
            .entries
            .binary_search_by(|(k, _)| match k.borrow().cmp(end) {
                Ordering::Less => Ordering::Less,
                _ => Ordering::Greater,
            })
            .unwrap_or_else(|i| i);
        self.entries[lo..hi].iter().map(|(k, v)| (k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_search_lookup() {
        let mut m = SortedVecMap::new();
        for k in 0..100i32 {
            m.insert(k, k * 10);
        }
        for k in 0..100 {
            assert_eq!(m.get(&k), Some(&(k * 10)));
        }
        assert_eq!(m.get(&100), None);
        assert_eq!(m.get(&-1), None);
    }

    #[test]
    fn iter_is_sorted() {
        let m = SortedVecMap::from_iter_unsorted(vec![
            (3, "c"),
            (1, "a"),
            (4, "d"),
            (2, "b"),
        ]);
        let keys: Vec<i32> = m.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3, 4]);
    }

    #[test]
    fn from_iter_unsorted_dedup_last_wins() {
        let m = SortedVecMap::from_iter_unsorted(vec![(1, "a"), (1, "b"), (1, "c")]);
        assert_eq!(m.get(&1), Some(&"c"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn range_query() {
        let m: SortedVecMap<i32, &str> = SortedVecMap::from_sorted_unique(vec![
            (1, "a"),
            (2, "b"),
            (3, "c"),
            (4, "d"),
            (5, "e"),
        ]);
        let r: Vec<_> = m.range(&2, &5).map(|(k, _)| *k).collect();
        assert_eq!(r, vec![2, 3, 4]);
    }

    #[test]
    fn from_sorted_unique_debug_panics_on_unsorted() {
        // Only check in debug. Skip this test in release.
        if cfg!(debug_assertions) {
            let result = std::panic::catch_unwind(|| {
                SortedVecMap::from_sorted_unique(vec![(2, "b"), (1, "a")])
            });
            assert!(result.is_err());
        }
    }
}
