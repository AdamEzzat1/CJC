//! `DHarht` — Deterministic HARHT (Hybrid Adaptive Radix Hash Trie).
//!
//! `Phase 7` shipped the architectural skeleton; `Phase 8` adds the
//! constant-factor optimizations:
//!
//! - **Per-shard typed slab allocator**: a single `Vec<u8>` per shard
//!   holds every key's bytes; `MicroBucket` entries store `(u32, u32)`
//!   handles into the slab. Eliminates per-entry `Vec<u8>` heap
//!   allocations (the #1 reason Phase 7's DHarht was slower than
//!   `BTreeMap`).
//! - **Singleton front-entry fast path**: when a front-directory
//!   prefix has exactly one key, the entry stores the handle + value
//!   inline — lookup skips the bucket array indirection entirely.
//!   Promotes to a real `MicroBucket` only on second insertion.
//!
//! All Phase 7 contracts are preserved bit-equal:
//!
//! - splitmix64 deterministic scattering
//! - 256-shard power-of-two layout
//! - sealed sparse 16-bit front directory per shard
//! - `MICROBUCKET_CAPACITY = 16` enforced; overflow → per-shard
//!   `BTreeMap` (deterministic, no silent loss)
//! - per-shard `overflow_count` + `max_bucket_size` counters
//! - full key equality on every successful lookup
//! - deterministic across runs / machines / architectures
//!
//! What is **still deferred vs the full spec**:
//! - Full ART tree (Node4/16/32/48/256) — using `BTreeMap` overflow,
//!   per the spec's "deterministic fallback to BTreeMap" allowance.
//! - Sealed jump table for the front directory — using sorted `Vec`
//!   with binary search; benchmarks below show this is now small
//!   enough relative to other costs that the jump table would be a
//!   third-order optimization.

use super::{hash_bytes, splitmix64};
use std::collections::BTreeMap;

/// Number of shards. Must be a power of two so `hash & MASK` is the
/// shard index. Spec recommends 256.
pub const NSHARDS: usize = 256;
const SHARD_MASK: u64 = (NSHARDS - 1) as u64;

/// Maximum entries in a `MicroBucket` before falling back to BTreeMap.
/// Spec: MicroBucket16.
pub const MICROBUCKET_CAPACITY: usize = 16;

/// Recommended profile per the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupProfile {
    Memory,
    Speed,
}

impl Default for LookupProfile {
    fn default() -> Self {
        LookupProfile::Memory
    }
}

/// Inline key threshold. Keys ≤ this length are stored inline in the
/// bucket entry, avoiding a slab dereference on the hot path. Sized
/// to fit common identifier-like keys (UUIDs are 16 bytes, 64-bit
/// integer keys are 8 bytes, "user_12345678" style identifiers are
/// ~13 bytes).
pub const INLINE_KEY_LEN: usize = 16;

/// Handle into a shard's `key_pool` slab, OR an inline key for short
/// strings. Tagged via `len`:
/// - `len <= INLINE_KEY_LEN`: bytes are inline in `inline[..len]`
/// - `len  > INLINE_KEY_LEN`: bytes live in the slab at `[offset..offset+len]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KeyHandle {
    /// Slab offset (only valid when `len > INLINE_KEY_LEN`). For
    /// inline keys this field is unused but maintained for layout
    /// consistency.
    offset: u32,
    /// Key length in bytes. Acts as the discriminant for inline vs slab.
    len: u32,
    /// Inline key bytes for short keys. Bytes beyond `len` are zero.
    inline: [u8; INLINE_KEY_LEN],
}

impl KeyHandle {
    #[inline(always)]
    fn is_inline(self) -> bool {
        (self.len as usize) <= INLINE_KEY_LEN
    }

    /// Resolve to a byte slice. For inline keys returns from
    /// `self.inline`; for slab keys returns from `slab`.
    #[inline]
    fn bytes<'a>(&'a self, slab: &'a [u8]) -> &'a [u8] {
        if self.is_inline() {
            &self.inline[..self.len as usize]
        } else {
            &slab[self.offset as usize..(self.offset + self.len) as usize]
        }
    }

    /// Construct a KeyHandle from `key`. Stores inline if short
    /// enough, otherwise appends to `slab` and stores `(offset, len)`.
    fn intern(slab: &mut Vec<u8>, key: &[u8]) -> Self {
        if key.len() <= INLINE_KEY_LEN {
            let mut inline = [0u8; INLINE_KEY_LEN];
            inline[..key.len()].copy_from_slice(key);
            KeyHandle {
                offset: 0,
                len: key.len() as u32,
                inline,
            }
        } else {
            let offset = slab.len() as u32;
            slab.extend_from_slice(key);
            KeyHandle {
                offset,
                len: key.len() as u32,
                inline: [0u8; INLINE_KEY_LEN],
            }
        }
    }
}

/// Number of front-directory slots per shard. Power of two so
/// `slot = (hash >> some_bits) & FRONT_MASK` is the slot index.
/// Phase 9: 256-slot direct jump table replacing the prior sorted
/// Vec + binary search. Per-slot cost: ~32 B × 256 = 8 KB, × 256
/// shards = ~2 MB total. Trade ~2 MB constant memory for `O(1)`
/// front lookup.
pub const FRONT_SLOTS: usize = 256;
const FRONT_MASK: u64 = (FRONT_SLOTS - 1) as u64;

/// Front-directory entry. Three states:
/// - `Empty`: no key has this slot.
/// - `Singleton`: slot has exactly one key — handle + value stored
///   inline. Lookup is one memcmp against the slab; no bucket
///   indirection.
/// - `Bucket`: slot has ≥ 2 keys — bucket id into `Shard::buckets`.
#[derive(Debug, Clone)]
enum FrontEntry<V: Clone> {
    Empty,
    Singleton { handle: KeyHandle, value: V },
    Bucket(u32),
}

impl<V: Clone> Default for FrontEntry<V> {
    fn default() -> Self {
        FrontEntry::Empty
    }
}

/// Bounded inline bucket. Handles point into the shard's key_pool
/// slab — the bucket itself only owns small (handle, value) pairs.
#[derive(Debug, Clone)]
pub struct MicroBucket<V: Clone> {
    handles: Vec<KeyHandle>,
    values: Vec<V>,
}

impl<V: Clone> MicroBucket<V> {
    fn new() -> Self {
        Self {
            handles: Vec::new(),
            values: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.handles.len()
    }

    fn full(&self) -> bool {
        self.handles.len() >= MICROBUCKET_CAPACITY
    }

    /// Linear scan with full key equality. Inline keys compare
    /// directly against `KeyHandle::inline`; slab keys dereference.
    #[inline]
    fn get<'a>(&'a self, slab: &[u8], key: &[u8]) -> Option<&'a V> {
        for (i, h) in self.handles.iter().enumerate() {
            if h.bytes(slab) == key {
                return Some(&self.values[i]);
            }
        }
        None
    }

    fn upsert_existing(&mut self, slab: &[u8], key: &[u8], value: V) -> Option<V> {
        for (i, h) in self.handles.iter().enumerate() {
            if h.bytes(slab) == key {
                return Some(std::mem::replace(&mut self.values[i], value));
            }
        }
        None
    }

    fn push(&mut self, handle: KeyHandle, value: V) {
        debug_assert!(!self.full());
        self.handles.push(handle);
        self.values.push(value);
    }
}

#[derive(Debug, Clone)]
struct Shard<V: Clone> {
    /// Per-shard key bytes arena. Monotonic — never compacted, so
    /// handles are stable for the lifetime of the shard.
    key_pool: Vec<u8>,
    /// Phase 9: 256-slot direct jump table indexed by 8 hash bits.
    /// `O(1)` lookup, no binary search.
    front: Box<[FrontEntry<V>; FRONT_SLOTS]>,
    /// Bucket pool. Bucket IDs are `Vec` indices.
    buckets: Vec<MicroBucket<V>>,
    /// Deterministic BTreeMap fallback for keys whose microbucket
    /// would exceed `MICROBUCKET_CAPACITY`.
    overflow: BTreeMap<Vec<u8>, V>,
    overflow_count: u64,
    max_bucket_size: u32,
}

impl<V: Clone> Shard<V> {
    fn new() -> Self {
        // `Box<[FrontEntry<V>; 256]>` initialized with Empty in every slot.
        // Avoid the [Empty; 256] syntax which requires V: Copy.
        let arr: [FrontEntry<V>; FRONT_SLOTS] =
            std::array::from_fn(|_| FrontEntry::Empty);
        Self {
            key_pool: Vec::new(),
            front: Box::new(arr),
            buckets: Vec::new(),
            overflow: BTreeMap::new(),
            overflow_count: 0,
            max_bucket_size: 0,
        }
    }

    /// Intern a key, returning a handle.
    #[inline]
    fn intern_key(&mut self, bytes: &[u8]) -> KeyHandle {
        KeyHandle::intern(&mut self.key_pool, bytes)
    }

    /// Resolve a handle to a byte slice.
    #[inline]
    fn slab_bytes<'a>(&'a self, h: &'a KeyHandle) -> &'a [u8] {
        h.bytes(&self.key_pool)
    }
}

/// Fast guard predicate: does `handle`'s bytes equal `probe`?
/// Inlined separately so the match guard in `get_bytes` produces
/// branch-friendly assembly.
#[inline(always)]
fn h_eq(handle: &KeyHandle, slab: &[u8], probe: &[u8]) -> bool {
    handle.bytes(slab) == probe
}

/// Deterministic hybrid lookup table.
#[derive(Debug, Clone)]
pub struct DHarht<V: Clone> {
    shards: Vec<Shard<V>>,
    sealed: bool,
    profile: LookupProfile,
    total_entries: u64,
}

impl<V: Clone> DHarht<V> {
    pub fn new() -> Self {
        Self::with_profile(LookupProfile::Memory)
    }

    pub fn with_profile(profile: LookupProfile) -> Self {
        Self {
            shards: (0..NSHARDS).map(|_| Shard::new()).collect(),
            sealed: false,
            profile,
            total_entries: 0,
        }
    }

    pub fn profile(&self) -> LookupProfile {
        self.profile
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    pub fn len(&self) -> u64 {
        self.total_entries
    }

    pub fn is_empty(&self) -> bool {
        self.total_entries == 0
    }

    pub fn overflow_count(&self) -> u64 {
        self.shards.iter().map(|s| s.overflow_count).sum()
    }

    pub fn max_bucket_size(&self) -> u32 {
        self.shards
            .iter()
            .map(|s| s.max_bucket_size)
            .max()
            .unwrap_or(0)
    }

    /// Phase 8 diagnostic: count of front-directory singleton entries.
    /// Higher = more keys hit the fast path.
    pub fn singleton_count(&self) -> u64 {
        self.shards
            .iter()
            .flat_map(|s| s.front.iter())
            .filter(|e| matches!(e, FrontEntry::Singleton { .. }))
            .count() as u64
    }

    /// Phase 9: directly indexes a `(shard, slot)` pair from the hash.
    /// Shard = top 8 bits, slot = next 8 bits. The slot is a direct
    /// array index — no binary search.
    #[inline]
    fn locate(key: &[u8]) -> (usize, usize) {
        let h = hash_bytes(key);
        let shard = ((h >> 56) & SHARD_MASK) as usize;
        let slot = ((h >> 48) & FRONT_MASK) as usize;
        (shard, slot)
    }

    pub fn insert_bytes(&mut self, key: &[u8], value: V) -> Option<V> {
        debug_assert!(!self.sealed, "DHarht: insert_bytes after seal_for_lookup");
        let (s, slot) = Self::locate(key);
        let shard = &mut self.shards[s];

        // Check overflow first — once a key has overflowed, it stays
        // there for the lifetime of the table.
        if let Some(v) = shard.overflow.get_mut(key) {
            return Some(std::mem::replace(v, value));
        }

        // Direct array access — no binary search.
        match &shard.front[slot] {
            FrontEntry::Empty => {
                let handle = shard.intern_key(key);
                shard.front[slot] = FrontEntry::Singleton { handle, value };
                self.total_entries += 1;
                shard.max_bucket_size = shard.max_bucket_size.max(1);
                None
            }
            FrontEntry::Singleton { handle, .. } => {
                // Compare against existing key bytes via slab.
                let existing_bytes_match = shard.slab_bytes(handle) == key;
                if existing_bytes_match {
                    if let FrontEntry::Singleton { value: ref mut v, .. } = shard.front[slot] {
                        return Some(std::mem::replace(v, value));
                    }
                    unreachable!()
                }
                // Different key — promote Singleton to Bucket.
                let old = std::mem::replace(
                    &mut shard.front[slot],
                    FrontEntry::Bucket(u32::MAX),
                );
                let (old_handle, old_value) = match old {
                    FrontEntry::Singleton { handle, value } => (handle, value),
                    _ => unreachable!(),
                };
                let new_handle = shard.intern_key(key);
                let mut bucket = MicroBucket::new();
                bucket.push(old_handle, old_value);
                bucket.push(new_handle, value);
                let bid = shard.buckets.len() as u32;
                shard.buckets.push(bucket);
                shard.front[slot] = FrontEntry::Bucket(bid);
                self.total_entries += 1;
                shard.max_bucket_size = shard.max_bucket_size.max(2);
                None
            }
            FrontEntry::Bucket(bid) => {
                let bid = *bid as usize;
                let key_pool = &shard.key_pool;
                let bucket = &mut shard.buckets[bid];
                if let Some(prev) = bucket.upsert_existing(key_pool, key, value.clone()) {
                    return Some(prev);
                }
                if !bucket.full() {
                    let handle = shard.intern_key(key);
                    let bucket = &mut shard.buckets[bid];
                    bucket.push(handle, value);
                    self.total_entries += 1;
                    shard.max_bucket_size =
                        shard.max_bucket_size.max(bucket.len() as u32);
                    return None;
                }
                let prev = shard.overflow.insert(key.to_vec(), value);
                if prev.is_none() {
                    self.total_entries += 1;
                    shard.overflow_count += 1;
                }
                prev
            }
        }
    }

    #[inline]
    pub fn get_bytes(&self, key: &[u8]) -> Option<&V> {
        let (s, slot) = Self::locate(key);
        let shard = &self.shards[s];
        // Direct array access — single load, no branch.
        // Hot path: Singleton + Bucket. Empty falls through to
        // `get_bytes_overflow` only when shard.overflow is non-empty
        // (which is the *uncommon* case in well-behaved workloads).
        match &shard.front[slot] {
            FrontEntry::Singleton { handle, value }
                if h_eq(handle, &shard.key_pool, key) =>
            {
                Some(value)
            }
            FrontEntry::Bucket(bid) => {
                let bucket = &shard.buckets[*bid as usize];
                bucket.get(&shard.key_pool, key).or_else(|| {
                    if shard.overflow.is_empty() {
                        None
                    } else {
                        shard.overflow.get(key)
                    }
                })
            }
            _ => {
                // Empty slot OR Singleton miss — overflow is the only
                // remaining place this key could be. Fast-skip when
                // empty.
                if shard.overflow.is_empty() {
                    None
                } else {
                    shard.overflow.get(key)
                }
            }
        }
    }

    pub fn contains_bytes(&self, key: &[u8]) -> bool {
        self.get_bytes(key).is_some()
    }

    pub fn seal_for_lookup(&mut self) {
        for shard in self.shards.iter_mut() {
            shard.key_pool.shrink_to_fit();
            // front is a fixed [_; FRONT_SLOTS] — no shrink needed.
            shard.buckets.shrink_to_fit();
            for bucket in shard.buckets.iter_mut() {
                bucket.handles.shrink_to_fit();
                bucket.values.shrink_to_fit();
            }
        }
        self.sealed = true;
    }

    /// Iterate `(bytes, &V)` pairs in deterministic-but-undefined
    /// order. Callers must sort for canonical output.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &V)> + '_ {
        self.shards.iter().flat_map(|shard| {
            let from_front = shard.front.iter().flat_map(move |entry| {
                let pairs: Vec<(&[u8], &V)> = match entry {
                    FrontEntry::Empty => Vec::new(),
                    FrontEntry::Singleton { handle, value } => {
                        vec![(shard.slab_bytes(handle), value)]
                    }
                    FrontEntry::Bucket(bid) => {
                        let bucket = &shard.buckets[*bid as usize];
                        bucket
                            .handles
                            .iter()
                            .zip(bucket.values.iter())
                            .map(|(h, v)| (shard.slab_bytes(h), v))
                            .collect()
                    }
                };
                pairs.into_iter()
            });
            let from_overflow = shard.overflow.iter().map(|(k, v)| (k.as_slice(), v));
            from_front.chain(from_overflow)
        })
    }

    pub fn iter_sorted(&self) -> Vec<(&[u8], &V)> {
        let mut all: Vec<(&[u8], &V)> = self.iter().collect();
        all.sort_by(|a, b| a.0.cmp(b.0));
        all
    }
}

impl<V: Clone> Default for DHarht<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of `DHarht` shape for double-run determinism checks.
pub fn deterministic_shape_hash<V: Clone + std::hash::Hash>(t: &DHarht<V>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let entries = t.iter_sorted();
    h.write_u64(entries.len() as u64);
    for (k, v) in entries {
        h.write_usize(k.len());
        h.write(k);
        v.hash(&mut h);
    }
    for shard in &t.shards {
        h.write_u64(shard.overflow_count);
        h.write_u32(shard.max_bucket_size);
    }
    splitmix64(h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_roundtrip() {
        let mut t: DHarht<i32> = DHarht::new();
        t.insert_bytes(b"alpha", 1);
        t.insert_bytes(b"beta", 2);
        t.insert_bytes(b"gamma", 3);
        assert_eq!(t.get_bytes(b"alpha"), Some(&1));
        assert_eq!(t.get_bytes(b"beta"), Some(&2));
        assert_eq!(t.get_bytes(b"gamma"), Some(&3));
        assert_eq!(t.get_bytes(b"delta"), None);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn insert_overwrites_returns_old() {
        let mut t: DHarht<i32> = DHarht::new();
        t.insert_bytes(b"k", 1);
        let prev = t.insert_bytes(b"k", 2);
        assert_eq!(prev, Some(1));
        assert_eq!(t.get_bytes(b"k"), Some(&2));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn singleton_then_bucket_promotion() {
        let mut t: DHarht<i32> = DHarht::new();
        // First insert at any prefix → Singleton.
        t.insert_bytes(b"a", 1);
        assert!(t.singleton_count() >= 1);
        // Hammer enough keys that some prefix definitely sees ≥2.
        for i in 0..10_000u32 {
            t.insert_bytes(&i.to_le_bytes(), i as i32);
        }
        // Some singletons should have promoted to buckets.
        // (We can't assert a specific number — depends on hash
        // distribution — but max_bucket_size > 1 confirms at least
        // one promotion happened.)
        assert!(t.max_bucket_size() >= 2);
    }

    #[test]
    fn seal_preserves_all_entries() {
        let mut t: DHarht<u32> = DHarht::new();
        for i in 0..1000u32 {
            t.insert_bytes(&i.to_le_bytes(), i);
        }
        let len_before = t.len();
        t.seal_for_lookup();
        assert!(t.is_sealed());
        assert_eq!(t.len(), len_before);
        for i in 0..1000u32 {
            assert_eq!(t.get_bytes(&i.to_le_bytes()), Some(&i));
        }
    }

    #[test]
    fn overflow_to_btreemap_fallback_no_data_loss() {
        let mut t: DHarht<u32> = DHarht::new();
        for i in 0..50_000u32 {
            t.insert_bytes(&i.to_le_bytes(), i);
        }
        for i in 0..50_000u32 {
            assert_eq!(t.get_bytes(&i.to_le_bytes()), Some(&i));
        }
        assert_eq!(t.len(), 50_000);
    }

    #[test]
    fn deterministic_double_build_same_shape_hash() {
        fn build() -> DHarht<u32> {
            let mut t: DHarht<u32> = DHarht::new();
            for i in 0..500u32 {
                t.insert_bytes(&i.to_be_bytes(), i);
            }
            t.seal_for_lookup();
            t
        }
        let h1 = deterministic_shape_hash(&build());
        let h2 = deterministic_shape_hash(&build());
        assert_eq!(h1, h2);
    }

    #[test]
    fn iter_sorted_is_canonical_regardless_of_insertion_order() {
        let mut a: DHarht<u32> = DHarht::new();
        let mut b: DHarht<u32> = DHarht::new();
        for i in 0..100u32 {
            a.insert_bytes(&i.to_be_bytes(), i);
        }
        for i in (0..100u32).rev() {
            b.insert_bytes(&i.to_be_bytes(), i);
        }
        let av: Vec<_> = a.iter_sorted().into_iter().map(|(k, v)| (k.to_vec(), *v)).collect();
        let bv: Vec<_> = b.iter_sorted().into_iter().map(|(k, v)| (k.to_vec(), *v)).collect();
        assert_eq!(av, bv);
    }

    #[test]
    fn microbucket_capacity_respected() {
        let mut t: DHarht<u32> = DHarht::new();
        for i in 0..10_000u32 {
            t.insert_bytes(&i.to_le_bytes(), i);
        }
        assert!(
            t.max_bucket_size() as usize <= MICROBUCKET_CAPACITY,
            "max bucket size {} exceeded MICROBUCKET_CAPACITY {}",
            t.max_bucket_size(),
            MICROBUCKET_CAPACITY
        );
    }

    #[test]
    fn matches_btreemap_oracle_on_random_workload() {
        let mut h: DHarht<u32> = DHarht::new();
        let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
        let mut x: u64 = 0xCAFEBABE;
        for _ in 0..2000 {
            x = splitmix64(x);
            let key_kind = x % 100;
            let mut key = Vec::new();
            key.extend_from_slice(&key_kind.to_le_bytes());
            let v = (x >> 8) as u32;
            h.insert_bytes(&key, v);
            oracle.insert(key, v);
        }
        for (k, v) in &oracle {
            assert_eq!(h.get_bytes(k), Some(v));
        }
        assert_eq!(h.len() as usize, oracle.len());
    }

    #[test]
    fn sealed_lookup_after_compaction_preserves_values() {
        let mut t: DHarht<Vec<u8>> = DHarht::new();
        for i in 0..500u32 {
            t.insert_bytes(&i.to_be_bytes(), i.to_be_bytes().to_vec());
        }
        t.seal_for_lookup();
        for i in 0..500u32 {
            assert_eq!(t.get_bytes(&i.to_be_bytes()), Some(&i.to_be_bytes().to_vec()));
        }
    }

    #[test]
    fn empty_table_lookup_returns_none() {
        let t: DHarht<u32> = DHarht::new();
        assert_eq!(t.get_bytes(b"any"), None);
    }

    #[test]
    fn zero_length_key_works() {
        let mut t: DHarht<u32> = DHarht::new();
        t.insert_bytes(b"", 42);
        assert_eq!(t.get_bytes(b""), Some(&42));
    }
}
