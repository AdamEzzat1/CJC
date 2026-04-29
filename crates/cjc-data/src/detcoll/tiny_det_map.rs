//! `TinyDetMap<K, V>` — small map backed by a sorted `Vec`.
//!
//! Optimized for ≤ ~16 entries. Linear scan beats `BTreeMap` at this
//! size (no node allocation, contiguous memory, branch predictor wins).
//! Iteration order is sorted by `K`, so output is canonical.
//!
//! Use for: small parser keyword tables, tiny schema metadata, short
//! enum/member lists, anywhere `BTreeMap` would feel like overkill but
//! you still want sorted iteration.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct TinyDetMap<K: Ord, V> {
    /// Sorted by `K`. Linear scan beats binary search up to ~16 entries
    /// because of cache locality and branch predictor friendliness.
    entries: Vec<(K, V)>,
}

impl<K: Ord, V> Default for TinyDetMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> TinyDetMap<K, V> {
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

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert or update. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.entries.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            Err(i) => {
                self.entries.insert(i, (key, value));
                None
            }
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        // Linear scan with branch on key cmp.
        for (k, v) in &self.entries {
            match k.cmp(key) {
                Ordering::Less => continue,
                Ordering::Equal => return Some(v),
                Ordering::Greater => return None,
            }
        }
        None
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.entries.binary_search_by(|(k, _)| k.cmp(key)) {
            Ok(i) => Some(self.entries.remove(i).1),
            Err(_) => None,
        }
    }

    /// Iterate sorted `(K, V)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

impl<K: Ord + Clone, V: Clone> FromIterator<(K, V)> for TinyDetMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut m = Self::new();
        for (k, v) in iter {
            m.insert(k, v);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_basic() {
        let mut m = TinyDetMap::new();
        assert!(m.insert("apple", 1).is_none());
        assert!(m.insert("banana", 2).is_none());
        assert!(m.insert("cherry", 3).is_none());
        assert_eq!(m.get(&"apple"), Some(&1));
        assert_eq!(m.get(&"banana"), Some(&2));
        assert_eq!(m.get(&"cherry"), Some(&3));
        assert_eq!(m.get(&"date"), None);
    }

    #[test]
    fn insert_overwrites_returns_old() {
        let mut m = TinyDetMap::new();
        m.insert("k", 1);
        let prev = m.insert("k", 2);
        assert_eq!(prev, Some(1));
        assert_eq!(m.get(&"k"), Some(&2));
    }

    #[test]
    fn iter_is_sorted_regardless_of_insertion_order() {
        let mut m = TinyDetMap::new();
        for &k in &["zebra", "apple", "mango", "banana"] {
            m.insert(k, 0);
        }
        let keys: Vec<&&str> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![&"apple", &"banana", &"mango", &"zebra"]);
    }

    #[test]
    fn remove_works() {
        let mut m = TinyDetMap::new();
        m.insert(1, "a");
        m.insert(2, "b");
        m.insert(3, "c");
        assert_eq!(m.remove(&2), Some("b"));
        assert_eq!(m.get(&2), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn deterministic_under_permutation() {
        let mut a = TinyDetMap::new();
        let mut b = TinyDetMap::new();
        for &(k, v) in &[(1, "a"), (2, "b"), (3, "c"), (4, "d")] {
            a.insert(k, v);
        }
        for &(k, v) in &[(4, "d"), (1, "a"), (3, "c"), (2, "b")] {
            b.insert(k, v);
        }
        let av: Vec<_> = a.iter().collect();
        let bv: Vec<_> = b.iter().collect();
        assert_eq!(av, bv);
    }
}
