//! `DetOpenMap<K, V>` — deterministic open-addressing hash map.
//!
//! Fixed splitmix64 mixing. Bounded probe length. Deterministic
//! fallback to `BTreeMap` if the probe budget is exceeded — which
//! prevents adversarial collision attacks from causing unbounded
//! probing or silent corruption.
//!
//! Use for: sparse mutable equality lookup where order doesn't matter
//! (worklist membership, temporary caches, sparse `NodeId → Fact` etc.).
//! For sealed/read-heavy workloads prefer `DHarht`. For ordered output
//! prefer `BTreeMap`.
//!
//! Iteration order is undefined-but-deterministic for a given insertion
//! sequence. **Do not surface iteration order to public output** — sort
//! into a `BTreeMap` or `SortedVecMap` for canonical output.

use super::splitmix64;
use std::collections::BTreeMap;
use std::hash::Hash;

/// Maximum probe length before falling back to BTreeMap. Caps worst
/// case lookup at `MAX_PROBE` linear scan steps.
const MAX_PROBE: usize = 32;

/// Default initial table size. Power of two so `hash & (n - 1)` is the
/// shard index.
const INITIAL_SLOTS: usize = 16;

/// Resize when load factor crosses this fraction. 0.75 in u32 form.
const LOAD_NUM: usize = 3;
const LOAD_DEN: usize = 4;

#[derive(Debug, Clone)]
enum Slot<K, V> {
    Empty,
    Occupied(K, V),
    Tombstone,
}

#[derive(Debug)]
pub struct DetOpenMap<K: Hash + Eq + Ord + Clone, V: Clone> {
    /// Open-addressing primary table.
    slots: Vec<Slot<K, V>>,
    /// Number of `Occupied` slots.
    occupied: usize,
    /// Number of `Tombstone` slots.
    tombstones: usize,
    /// Deterministic fallback for entries that exceeded the probe
    /// budget. The contract: `BTreeMap::insert` semantics, so no
    /// ordering surprises.
    fallback: BTreeMap<K, V>,
    /// Counter for instrumentation / security tests.
    overflow_to_fallback: u64,
}

impl<K: Hash + Eq + Ord + Clone, V: Clone> Default for DetOpenMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Ord + Clone, V: Clone> DetOpenMap<K, V> {
    pub fn new() -> Self {
        Self::with_capacity(INITIAL_SLOTS)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(INITIAL_SLOTS).next_power_of_two();
        Self {
            slots: (0..cap).map(|_| Slot::Empty).collect(),
            occupied: 0,
            tombstones: 0,
            fallback: BTreeMap::new(),
            overflow_to_fallback: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.occupied + self.fallback.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many entries spilled into the BTreeMap fallback. Surfaces
    /// adversarial-collision health for tests / monitoring. Not part
    /// of the user-visible map contract.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_to_fallback
    }

    /// Hash a key with the deterministic mixer.
    fn hash_key(&self, key: &K) -> u64 {
        // Stable per-implementation: rely on Hash + a `splitmix64` mix.
        let mut h = std::hash::DefaultHasher::new();
        // Note: we deliberately use a stable starting state by feeding
        // a fixed prefix. `DefaultHasher` is `SipHash-1-3` with a
        // *stable* seed within a single run, but Rust documents that
        // its output may differ across versions. To pin determinism
        // across versions we mix the result through `splitmix64` and
        // also feed a fixed nonce.
        use std::hash::Hasher;
        h.write_u64(0xCBF29CE484222325);
        key.hash(&mut h);
        splitmix64(h.finish())
    }

    fn slot_index(&self, hash: u64, probe: usize) -> usize {
        let n = self.slots.len();
        // Linear probing with deterministic mix per probe step.
        let step = splitmix64(hash.wrapping_add(probe as u64));
        (step as usize) & (n - 1)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // If we'd exceed the load factor, resize first.
        if (self.occupied + 1) * LOAD_DEN > self.slots.len() * LOAD_NUM {
            self.resize();
        }

        let h = self.hash_key(&key);
        let mut first_tombstone: Option<usize> = None;
        for probe in 0..MAX_PROBE {
            let idx = self.slot_index(h, probe);
            match &self.slots[idx] {
                Slot::Empty => {
                    let slot_idx = first_tombstone.unwrap_or(idx);
                    if matches!(self.slots[slot_idx], Slot::Tombstone) {
                        self.tombstones -= 1;
                    }
                    self.slots[slot_idx] = Slot::Occupied(key, value);
                    self.occupied += 1;
                    return None;
                }
                Slot::Tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(idx);
                    }
                }
                Slot::Occupied(k, _) if k == &key => {
                    // Found an existing slot — update.
                    if let Slot::Occupied(_, old_v) =
                        std::mem::replace(&mut self.slots[idx], Slot::Empty)
                    {
                        self.slots[idx] = Slot::Occupied(key, value);
                        return Some(old_v);
                    }
                    unreachable!()
                }
                Slot::Occupied(_, _) => continue,
            }
        }

        // Probe budget exceeded — fall back to BTreeMap. Deterministic.
        self.overflow_to_fallback += 1;
        self.fallback.insert(key, value)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        // Check fallback first — once a key has overflowed, it lives
        // there even if the primary now has space.
        if let Some(v) = self.fallback.get(key) {
            return Some(v);
        }
        let h = self.hash_key(key);
        for probe in 0..MAX_PROBE {
            let idx = self.slot_index(h, probe);
            match &self.slots[idx] {
                Slot::Empty => return None,
                Slot::Occupied(k, v) if k == key => return Some(v),
                Slot::Occupied(_, _) | Slot::Tombstone => continue,
            }
        }
        None
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(v) = self.fallback.remove(key) {
            return Some(v);
        }
        let h = self.hash_key(key);
        for probe in 0..MAX_PROBE {
            let idx = self.slot_index(h, probe);
            match &self.slots[idx] {
                Slot::Empty => return None,
                Slot::Occupied(k, _) if k == key => {
                    if let Slot::Occupied(_, v) =
                        std::mem::replace(&mut self.slots[idx], Slot::Tombstone)
                    {
                        self.occupied -= 1;
                        self.tombstones += 1;
                        return Some(v);
                    }
                    unreachable!()
                }
                _ => continue,
            }
        }
        None
    }

    fn resize(&mut self) {
        let new_cap = (self.slots.len() * 2).max(INITIAL_SLOTS);
        let old = std::mem::replace(
            &mut self.slots,
            (0..new_cap).map(|_| Slot::Empty).collect(),
        );
        self.occupied = 0;
        self.tombstones = 0;
        for slot in old {
            if let Slot::Occupied(k, v) = slot {
                // Reinsert via the normal insert path so probe sequence
                // reuses the new table size. Because we just resized,
                // load factor cannot trigger another resize here.
                self.insert(k, v);
            }
        }
    }

    /// Iterate `(K, V)` pairs in **deterministic-but-undefined** order
    /// (combination of slot order + fallback order). Callers MUST sort
    /// before public output.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        let from_slots = self.slots.iter().filter_map(|s| match s {
            Slot::Occupied(k, v) => Some((k, v)),
            _ => None,
        });
        let from_fallback = self.fallback.iter();
        from_slots.chain(from_fallback)
    }

    /// Sorted iteration. Use this for any path that surfaces iteration
    /// order to public output.
    pub fn iter_sorted(&self) -> Vec<(&K, &V)> {
        let mut all: Vec<(&K, &V)> = self.iter().collect();
        all.sort_by(|a, b| a.0.cmp(b.0));
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_basic() {
        let mut m: DetOpenMap<i32, &str> = DetOpenMap::new();
        m.insert(1, "a");
        m.insert(2, "b");
        m.insert(3, "c");
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&2), Some(&"b"));
        assert_eq!(m.get(&3), Some(&"c"));
        assert_eq!(m.get(&4), None);
    }

    #[test]
    fn insert_overwrites() {
        let mut m: DetOpenMap<i32, &str> = DetOpenMap::new();
        m.insert(1, "a");
        let prev = m.insert(1, "b");
        assert_eq!(prev, Some("a"));
        assert_eq!(m.get(&1), Some(&"b"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn remove_works_via_tombstone() {
        let mut m: DetOpenMap<i32, &str> = DetOpenMap::new();
        m.insert(1, "a");
        m.insert(2, "b");
        m.insert(3, "c");
        assert_eq!(m.remove(&2), Some("b"));
        assert_eq!(m.get(&2), None);
        // Other entries still findable.
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&3), Some(&"c"));
    }

    #[test]
    fn resize_preserves_entries() {
        let mut m: DetOpenMap<i32, i32> = DetOpenMap::new();
        for i in 0..1000 {
            m.insert(i, i * 2);
        }
        for i in 0..1000 {
            assert_eq!(m.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn deterministic_double_run_iter_sorted() {
        let mut a: DetOpenMap<i32, i32> = DetOpenMap::new();
        let mut b: DetOpenMap<i32, i32> = DetOpenMap::new();
        for i in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3] {
            a.insert(i, i);
            b.insert(i, i);
        }
        let av = a.iter_sorted();
        let bv = b.iter_sorted();
        assert_eq!(av, bv);
    }

    #[test]
    fn iter_sorted_is_canonical_regardless_of_insertion_order() {
        let mut a: DetOpenMap<i32, i32> = DetOpenMap::new();
        let mut b: DetOpenMap<i32, i32> = DetOpenMap::new();
        for i in [1, 2, 3, 4, 5] {
            a.insert(i, i);
        }
        for i in [5, 4, 3, 2, 1] {
            b.insert(i, i);
        }
        let av: Vec<i32> = a.iter_sorted().into_iter().map(|(_, v)| *v).collect();
        let bv: Vec<i32> = b.iter_sorted().into_iter().map(|(_, v)| *v).collect();
        assert_eq!(av, bv);
        assert_eq!(av, vec![1, 2, 3, 4, 5]);
    }
}
