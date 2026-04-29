//! Deterministic collection family — Phase 7.
//!
//! A small set of collections each tuned to one workload shape, sharing a
//! determinism contract:
//!
//! 1. **No randomized hashing.** All hash functions are fixed
//!    (`splitmix64` for `DHarht` and `DetOpenMap`).
//! 2. **No pointer-address ordering.** Iteration order is driven by
//!    keys, sort order, or insertion order — never raw addresses.
//! 3. **Bounded behavior.** Every structure has a documented worst
//!    case; failure mode is deterministic fallback (typically
//!    `BTreeMap`), never silent corruption.
//! 4. **Full key equality on success.** Hash collision never returns
//!    a wrong value.
//!
//! # When to use which
//!
//! | Workload                                  | Pick           |
//! |-------------------------------------------|----------------|
//! | Tiny maps (≤ ~16 entries)                 | `TinyDetMap`   |
//! | Small sealed sorted maps                  | `SortedVecMap` |
//! | Dense `IdType -> Value` ID tables         | `IndexVec`     |
//! | Sparse mutable equality lookup            | `DetOpenMap`   |
//! | Large sealed equality lookup, prefix-heavy| `DHarht`       |
//! | Range / prefix queries / canonical output | `BTreeMap`     |
//!
//! `DHarht` is **not** a global `BTreeMap` replacement. It is best for
//! byte-addressable, sealed/read-heavy, deterministic equality lookup
//! workloads. `BTreeMap` remains the choice for canonical ordering,
//! diagnostics, serialization, and range behavior.

pub mod dharht;
pub mod dharht_memory;
pub mod det_open;
pub mod index_vec;
pub mod sealed_u64_map;
pub mod sorted_vec_map;
pub mod tiny_det_map;

pub use dharht::{DHarht, LookupProfile, MicroBucket};
pub use dharht_memory::DHarhtMemory;
pub use det_open::DetOpenMap;
pub use index_vec::{Idx, IndexVec};
pub use sealed_u64_map::SealedU64Map;
pub use sorted_vec_map::SortedVecMap;
pub use tiny_det_map::TinyDetMap;

/// Splitmix64 mixer used as the deterministic hash function across
/// `DHarht` and `DetOpenMap`. Pure function: same input → same output
/// on every machine, every architecture, every locale.
///
/// This is the same mixing function `cjc-repro` uses for its RNG —
/// reusing it here keeps the workspace's "one deterministic mixer" rule.
#[inline(always)]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Hash arbitrary bytes deterministically. Used as the keying function
/// for `DHarht::insert_bytes` and (post-Phase-11) as the canonical
/// 64-bit content hash for `ByteDictionary::seal_with_u64_hash_index`.
///
/// **Phase 9: switched from splitmix64+FNV folding to multiplicative
/// hash on 8-byte chunks.** Same determinism contract (no platform-
/// dependent seeds, byte-equal across runs / machines), but ~5× faster
/// on short keys because we skip the inner splitmix64 mixer per chunk.
/// Final mix at the end folds in length to maintain distribution
/// quality.
#[inline]
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    // FxHash-style multiplicative constant (golden ratio in 64-bit).
    // Fixed across runs — no random seed.
    const K: u64 = 0x517cc1b727220a95;
    let mut h: u64 = 0xCBF29CE484222325; // FNV offset basis as initial seed
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let v = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = h.rotate_left(5).wrapping_mul(K) ^ v;
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rem.len()].copy_from_slice(rem);
        let v = u64::from_le_bytes(tail);
        h = h.rotate_left(5).wrapping_mul(K) ^ v;
    }
    // Final avalanche — splitmix64 finalizer ensures shard / front
    // indices are well-distributed. Cost: paid once per hash, not
    // per chunk.
    splitmix64(h ^ (bytes.len() as u64))
}
