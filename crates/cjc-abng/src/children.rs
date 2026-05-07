//! Adaptive children variants for [`AdaptiveBeliefNode`](crate::AdaptiveBeliefNode).
//!
//! The five variants `None / Node4 / Node16 / Node48 / Node256` follow the same
//! density-classification pattern as `cjc-data::AdaptiveSelection` (which itself
//! follows ART — Adaptive Radix Trees). Each variant stores children at a size
//! class chosen to fit the current population:
//!
//! | Variant   | Capacity | Layout                                    | Lookup |
//! |-----------|---------:|-------------------------------------------|--------|
//! | `None`    |        0 | terminal — leaf                           | —      |
//! | `Node4`   |        4 | parallel `keys[4]` / `slots[4]` arrays    | linear |
//! | `Node16`  |       16 | parallel `keys[16]` / `slots[16]` arrays  | linear |
//! | `Node48`  |       48 | `index[256]` indirects into `slots: Vec`  | direct |
//! | `Node256` |      256 | direct-indexed `slots[byte]`              | direct |
//!
//! `Node48`'s two-array trick is the load-bearing trick from ART: a 256-byte
//! index array maps `byte → slot`, and the `slots` Vec is densely packed.
//! That keeps `Node48` ~1/5 the memory of `Node256` while preserving O(1)
//! lookup, which is the right shape for the typical "20-30 children" middle
//! ground.
//!
//! Promotion happens at insert time, not lazily — when a `Node4` would receive
//! a 5th child, it becomes a `Node16` *before* the insert lands. This keeps
//! the audit log linear (no deferred-promotion bookkeeping in event payloads).

use crate::node::NodeId;

/// Sentinel byte for "empty slot" inside the `Node48` `index` array.
const NODE48_EMPTY: u8 = 0xFF;

/// Numeric code for a children variant. Embedded in audit events
/// (`ChildrenPromoted { from, to }`) and the snapshot format.
///
/// The numeric values are part of the on-disk contract — adding a new variant
/// must use a fresh code, never re-number an existing one. Phase 0.3d-3 adds
/// `Dense = 5` for the post-Compress sub-tree representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChildrenKind {
    None = 0,
    Node4 = 1,
    Node16 = 2,
    Node48 = 3,
    Node256 = 4,
    /// Phase 0.3d-3 — terminal sub-tree representation after Compress.
    /// The 32-byte signature replaces the node's children container; the
    /// original descendants persist in the arena (orphaned) per the
    /// "never reorder pushes" invariant in `ABNG_CURRENT_ARCHITECTURE.md`
    /// §7 #9.
    Dense = 5,
}

impl ChildrenKind {
    /// Decode a u8 tag back to a kind, or `None` for an unknown value.
    pub fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            0 => ChildrenKind::None,
            1 => ChildrenKind::Node4,
            2 => ChildrenKind::Node16,
            3 => ChildrenKind::Node48,
            4 => ChildrenKind::Node256,
            5 => ChildrenKind::Dense,
            _ => return None,
        })
    }
}

/// Adaptive child container. The variant is chosen based on the current
/// child count; insert-time promotion handles transitions.
#[derive(Debug, Clone)]
pub enum AdaptiveChildren {
    /// Terminal — no children. The default at node creation.
    None,
    /// Up to 4 children. Linear scan over a parallel `keys`/`slots` array
    /// (cache-friendly at this size).
    Node4 {
        keys: [u8; 4],
        slots: [Option<NodeId>; 4],
    },
    /// 5–16 children. Same shape as `Node4`, larger arrays.
    Node16 {
        keys: [u8; 16],
        slots: [Option<NodeId>; 16],
    },
    /// 17–48 children. Two-array trick: `index[byte]` indirects into the
    /// densely-packed `slots` Vec. `0xFF` in `index` marks an empty slot.
    Node48 {
        index: Box<[u8; 256]>,
        slots: Vec<Option<NodeId>>,
    },
    /// 49–256 children. Direct-indexed `byte → child` table.
    Node256 {
        slots: Box<[Option<NodeId>; 256]>,
    },
    /// Phase 0.3d-3 — terminal sub-tree marker after Compress. Holds a
    /// 32-byte signature representing the compressed sub-tree's
    /// fingerprint. Routing (`get`, `iter`) returns empty; the original
    /// descendants persist in the arena but are unreachable through
    /// this node.
    Dense { signature: [u8; 32] },
}

impl AdaptiveChildren {
    /// Construct an empty (leaf) container.
    pub fn new() -> Self {
        AdaptiveChildren::None
    }

    /// Numeric kind code for audit logs and snapshots.
    pub fn kind(&self) -> ChildrenKind {
        match self {
            AdaptiveChildren::None => ChildrenKind::None,
            AdaptiveChildren::Node4 { .. } => ChildrenKind::Node4,
            AdaptiveChildren::Node16 { .. } => ChildrenKind::Node16,
            AdaptiveChildren::Node48 { .. } => ChildrenKind::Node48,
            AdaptiveChildren::Node256 { .. } => ChildrenKind::Node256,
            AdaptiveChildren::Dense { .. } => ChildrenKind::Dense,
        }
    }

    /// Count of live children. `Dense` returns 0 — the sub-tree is
    /// represented by a single signature, no routable children.
    pub fn len(&self) -> usize {
        match self {
            AdaptiveChildren::None => 0,
            AdaptiveChildren::Node4 { slots, .. } => slots.iter().filter(|s| s.is_some()).count(),
            AdaptiveChildren::Node16 { slots, .. } => slots.iter().filter(|s| s.is_some()).count(),
            AdaptiveChildren::Node48 { slots, .. } => slots.iter().filter(|s| s.is_some()).count(),
            AdaptiveChildren::Node256 { slots } => slots.iter().filter(|s| s.is_some()).count(),
            AdaptiveChildren::Dense { .. } => 0,
        }
    }

    /// Whether this container is at maximum capacity for its current variant.
    /// Used by `add_child` to decide whether to promote before inserting.
    /// `Dense` reports `true` so the graph layer's structural-mutation
    /// guards reject any attempt to mutate a compressed sub-tree.
    pub fn is_full(&self) -> bool {
        match self {
            AdaptiveChildren::None => true, // any insert promotes to Node4
            AdaptiveChildren::Node4 { slots, .. } => slots.iter().all(|s| s.is_some()),
            AdaptiveChildren::Node16 { slots, .. } => slots.iter().all(|s| s.is_some()),
            // Node48's `slots: Vec` only ever contains `Some(_)` (we push on
            // insert), so `iter().all(|s| s.is_some())` is `true` for any
            // non-empty vec. The right capacity test is the vec's length
            // against the fixed cap.
            AdaptiveChildren::Node48 { slots, .. } => slots.len() == 48,
            AdaptiveChildren::Node256 { slots } => slots.iter().all(|s| s.is_some()),
            AdaptiveChildren::Dense { .. } => true,
        }
    }

    /// Whether this container is a Dense (compressed) sub-tree. The
    /// graph layer uses this to gate force_grow / force_split / add_node.
    pub fn is_dense(&self) -> bool {
        matches!(self, AdaptiveChildren::Dense { .. })
    }

    /// Lookup a child by key byte. Returns `None` if no child is bound to
    /// that byte (or if `self == None`).
    pub fn get(&self, key: u8) -> Option<NodeId> {
        match self {
            AdaptiveChildren::None => None,
            AdaptiveChildren::Node4 { keys, slots } => {
                for i in 0..4 {
                    if slots[i].is_some() && keys[i] == key {
                        return slots[i];
                    }
                }
                None
            }
            AdaptiveChildren::Node16 { keys, slots } => {
                for i in 0..16 {
                    if slots[i].is_some() && keys[i] == key {
                        return slots[i];
                    }
                }
                None
            }
            AdaptiveChildren::Node48 { index, slots } => {
                let slot_idx = index[key as usize];
                if slot_idx == NODE48_EMPTY {
                    None
                } else {
                    slots[slot_idx as usize]
                }
            }
            AdaptiveChildren::Node256 { slots } => slots[key as usize],
            AdaptiveChildren::Dense { .. } => None,
        }
    }

    /// Iterate `(key_byte, child_id)` pairs in ascending key order.
    /// Used by snapshot serialization and `abng_node_child_count`.
    /// `Dense` returns an empty iterator — the sub-tree's descendants
    /// remain in the arena but are unreachable from this node.
    pub fn iter(&self) -> Vec<(u8, NodeId)> {
        let mut out = Vec::new();
        match self {
            AdaptiveChildren::None => {}
            AdaptiveChildren::Dense { .. } => {}
            AdaptiveChildren::Node4 { keys, slots } => {
                for i in 0..4 {
                    if let Some(id) = slots[i] {
                        out.push((keys[i], id));
                    }
                }
                out.sort_by_key(|&(k, _)| k);
            }
            AdaptiveChildren::Node16 { keys, slots } => {
                for i in 0..16 {
                    if let Some(id) = slots[i] {
                        out.push((keys[i], id));
                    }
                }
                out.sort_by_key(|&(k, _)| k);
            }
            AdaptiveChildren::Node48 { index, slots } => {
                for byte in 0u16..=255 {
                    let slot_idx = index[byte as usize];
                    if slot_idx != NODE48_EMPTY {
                        if let Some(id) = slots[slot_idx as usize] {
                            out.push((byte as u8, id));
                        }
                    }
                }
            }
            AdaptiveChildren::Node256 { slots } => {
                for byte in 0u16..=255 {
                    if let Some(id) = slots[byte as usize] {
                        out.push((byte as u8, id));
                    }
                }
            }
        }
        out
    }

    /// Insert a child, promoting the variant first if needed. Returns the
    /// kind *before* and *after* insertion so the caller can emit a
    /// `ChildrenPromoted` audit event when they differ.
    ///
    /// If a child already exists at `key`, the old value is replaced and
    /// no promotion happens. This is intentional: callers (the graph)
    /// only call `add_child` after confirming the key is unbound.
    pub fn add_child(&mut self, key: u8, id: NodeId) -> (ChildrenKind, ChildrenKind) {
        let from_kind = self.kind();
        // Promote first if at capacity.
        if self.is_full() {
            self.promote();
        }
        let to_kind = self.kind();

        match self {
            AdaptiveChildren::None => unreachable!("promote() never leaves None"),
            AdaptiveChildren::Node4 { keys, slots } => {
                // Replace if already bound.
                for i in 0..4 {
                    if slots[i].is_some() && keys[i] == key {
                        slots[i] = Some(id);
                        return (from_kind, to_kind);
                    }
                }
                // Otherwise find an empty slot.
                for i in 0..4 {
                    if slots[i].is_none() {
                        keys[i] = key;
                        slots[i] = Some(id);
                        return (from_kind, to_kind);
                    }
                }
                unreachable!("Node4 promote() guaranteed an empty slot");
            }
            AdaptiveChildren::Node16 { keys, slots } => {
                for i in 0..16 {
                    if slots[i].is_some() && keys[i] == key {
                        slots[i] = Some(id);
                        return (from_kind, to_kind);
                    }
                }
                for i in 0..16 {
                    if slots[i].is_none() {
                        keys[i] = key;
                        slots[i] = Some(id);
                        return (from_kind, to_kind);
                    }
                }
                unreachable!("Node16 promote() guaranteed an empty slot");
            }
            AdaptiveChildren::Node48 { index, slots } => {
                if index[key as usize] != NODE48_EMPTY {
                    let slot_idx = index[key as usize] as usize;
                    slots[slot_idx] = Some(id);
                    return (from_kind, to_kind);
                }
                // Append into the next free Vec slot.
                let new_slot = slots.len();
                debug_assert!(new_slot < 48, "Node48 promote() guaranteed room");
                slots.push(Some(id));
                index[key as usize] = new_slot as u8;
                (from_kind, to_kind)
            }
            AdaptiveChildren::Node256 { slots } => {
                slots[key as usize] = Some(id);
                (from_kind, to_kind)
            }
            AdaptiveChildren::Dense { .. } => unreachable!(
                "add_child called on Dense container; graph-layer guards must reject \
                 force_grow / add_node on a compressed sub-tree before reaching this point"
            ),
        }
    }

    /// In-place promote to the next size class. Called by `add_child` when
    /// at capacity. Does not insert anything; the caller follows up with
    /// the actual insert.
    fn promote(&mut self) {
        let promoted = match std::mem::replace(self, AdaptiveChildren::None) {
            AdaptiveChildren::None => AdaptiveChildren::Node4 {
                keys: [0u8; 4],
                slots: [None; 4],
            },
            AdaptiveChildren::Node4 { keys, slots } => {
                let mut new_keys = [0u8; 16];
                let mut new_slots: [Option<NodeId>; 16] = [None; 16];
                for i in 0..4 {
                    new_keys[i] = keys[i];
                    new_slots[i] = slots[i];
                }
                AdaptiveChildren::Node16 {
                    keys: new_keys,
                    slots: new_slots,
                }
            }
            AdaptiveChildren::Node16 { keys, slots } => {
                let mut index: Box<[u8; 256]> = Box::new([NODE48_EMPTY; 256]);
                let mut new_slots: Vec<Option<NodeId>> = Vec::with_capacity(48);
                for i in 0..16 {
                    if let Some(id) = slots[i] {
                        let slot_idx = new_slots.len();
                        new_slots.push(Some(id));
                        index[keys[i] as usize] = slot_idx as u8;
                    }
                }
                AdaptiveChildren::Node48 {
                    index,
                    slots: new_slots,
                }
            }
            AdaptiveChildren::Node48 { index, slots } => {
                let mut new_slots: Box<[Option<NodeId>; 256]> = Box::new([None; 256]);
                for byte in 0u16..=255 {
                    let slot_idx = index[byte as usize];
                    if slot_idx != NODE48_EMPTY {
                        new_slots[byte as usize] = slots[slot_idx as usize];
                    }
                }
                AdaptiveChildren::Node256 { slots: new_slots }
            }
            AdaptiveChildren::Node256 { slots } => {
                // Already maxed; no further promotion. Restore in place.
                AdaptiveChildren::Node256 { slots }
            }
            AdaptiveChildren::Dense { signature } => {
                // Dense never promotes — graph guards must prevent
                // promote() from being called on a compressed sub-tree.
                AdaptiveChildren::Dense { signature }
            }
        };
        *self = promoted;
    }
}

impl Default for AdaptiveChildren {
    fn default() -> Self {
        AdaptiveChildren::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_is_none_kind() {
        let c = AdaptiveChildren::new();
        assert_eq!(c.kind(), ChildrenKind::None);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn first_insert_promotes_to_node4() {
        let mut c = AdaptiveChildren::new();
        let (from, to) = c.add_child(7, 1);
        assert_eq!(from, ChildrenKind::None);
        assert_eq!(to, ChildrenKind::Node4);
        assert_eq!(c.get(7), Some(1));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn fifth_insert_promotes_node4_to_node16() {
        let mut c = AdaptiveChildren::new();
        for k in 0..4u8 {
            c.add_child(k, k as NodeId);
        }
        assert_eq!(c.kind(), ChildrenKind::Node4);
        let (from, to) = c.add_child(4, 4);
        assert_eq!(from, ChildrenKind::Node4);
        assert_eq!(to, ChildrenKind::Node16);
        for k in 0..5u8 {
            assert_eq!(c.get(k), Some(k as NodeId));
        }
    }

    #[test]
    fn seventeenth_insert_promotes_to_node48() {
        let mut c = AdaptiveChildren::new();
        for k in 0..16u8 {
            c.add_child(k, k as NodeId);
        }
        assert_eq!(c.kind(), ChildrenKind::Node16);
        let (from, to) = c.add_child(16, 16);
        assert_eq!(from, ChildrenKind::Node16);
        assert_eq!(to, ChildrenKind::Node48);
        for k in 0..17u8 {
            assert_eq!(c.get(k), Some(k as NodeId));
        }
    }

    #[test]
    fn forty_ninth_insert_promotes_to_node256() {
        let mut c = AdaptiveChildren::new();
        for k in 0..48u8 {
            c.add_child(k, k as NodeId);
        }
        assert_eq!(c.kind(), ChildrenKind::Node48);
        let (from, to) = c.add_child(48, 48);
        assert_eq!(from, ChildrenKind::Node48);
        assert_eq!(to, ChildrenKind::Node256);
        for k in 0..49u8 {
            assert_eq!(c.get(k), Some(k as NodeId));
        }
    }

    #[test]
    fn full_node256_no_further_promotion() {
        let mut c = AdaptiveChildren::new();
        for k in 0u16..256 {
            c.add_child(k as u8, k as NodeId);
        }
        assert_eq!(c.kind(), ChildrenKind::Node256);
        // Replacing an existing key is fine.
        let (from, to) = c.add_child(7, 999);
        assert_eq!(from, ChildrenKind::Node256);
        assert_eq!(to, ChildrenKind::Node256);
        assert_eq!(c.get(7), Some(999));
    }

    #[test]
    fn replace_existing_key_no_promotion() {
        let mut c = AdaptiveChildren::new();
        c.add_child(5, 100);
        let (from, to) = c.add_child(5, 200);
        assert_eq!(from, ChildrenKind::Node4);
        assert_eq!(to, ChildrenKind::Node4);
        assert_eq!(c.get(5), Some(200));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn iter_returns_keys_in_ascending_order() {
        let mut c = AdaptiveChildren::new();
        for &k in &[7u8, 3, 11, 1, 9] {
            c.add_child(k, k as NodeId);
        }
        let pairs = c.iter();
        let keys: Vec<u8> = pairs.iter().map(|&(k, _)| k).collect();
        assert_eq!(keys, vec![1, 3, 7, 9, 11]);
    }

    #[test]
    fn get_after_each_promotion_consistent() {
        let mut c = AdaptiveChildren::new();
        // Insert 100 distinct keys; spot-check at every promotion boundary.
        for k in 0u8..100 {
            c.add_child(k, 1000 as NodeId + k as NodeId);
            assert_eq!(c.get(k), Some(1000 as NodeId + k as NodeId));
        }
        // Promotions happened: Node4 → Node16 → Node48 → Node256.
        assert_eq!(c.kind(), ChildrenKind::Node256);
        for k in 0u8..100 {
            assert_eq!(c.get(k), Some(1000 as NodeId + k as NodeId));
        }
    }

    #[test]
    fn from_tag_round_trip() {
        for k in [
            ChildrenKind::None,
            ChildrenKind::Node4,
            ChildrenKind::Node16,
            ChildrenKind::Node48,
            ChildrenKind::Node256,
            ChildrenKind::Dense,
        ] {
            assert_eq!(ChildrenKind::from_tag(k as u8), Some(k));
        }
        assert_eq!(ChildrenKind::from_tag(99), None);
    }

    // ── Phase 0.3d-3: Dense variant ──────────────────────────────

    #[test]
    fn dense_variant_basic_shape() {
        let c = AdaptiveChildren::Dense {
            signature: [42u8; 32],
        };
        assert_eq!(c.kind(), ChildrenKind::Dense);
        assert_eq!(ChildrenKind::Dense as u8, 5);
        assert_eq!(c.len(), 0);
        assert!(c.is_full());
        assert!(c.is_dense());
        // Lookup always misses.
        for k in 0u16..=255 {
            assert_eq!(c.get(k as u8), None);
        }
        // Iter is empty.
        assert!(c.iter().is_empty());
    }

    #[test]
    fn non_dense_variants_report_not_dense() {
        let mut c = AdaptiveChildren::new();
        assert!(!c.is_dense());
        c.add_child(7, 1);
        assert!(!c.is_dense()); // Node4
        for k in 0..16u8 {
            c.add_child(k, k as NodeId);
        }
        assert!(!c.is_dense()); // Node16/48 depending
    }

    #[test]
    fn dense_signature_distinct_kinds() {
        let a = AdaptiveChildren::Dense { signature: [1u8; 32] };
        let b = AdaptiveChildren::Dense { signature: [2u8; 32] };
        // Both report Dense kind, but the underlying signatures differ
        // — used by the snapshot encoder/decoder for round-trip equality.
        assert_eq!(a.kind(), b.kind());
        match (&a, &b) {
            (
                AdaptiveChildren::Dense { signature: sa },
                AdaptiveChildren::Dense { signature: sb },
            ) => assert_ne!(sa, sb),
            _ => unreachable!(),
        }
    }
}
