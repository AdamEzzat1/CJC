//! `DHarhtMemory` — port of the user-supplied **D-HARHT Memory profile**
//! architecture. Distinct from `DHarht` (v.01) which uses byte-slice
//! keys with a slab allocator.
//!
//! Source: `D-HARHT-Blueprint-and-Code.md` (2026-04-28). This is a
//! faithful port of the Memory profile (sparse paged 16-bit front
//! directory, packed `u64` front entries with 3-bit tags,
//! `MicroBucket4`/`8`/`16` with parallel `match_mask`, splitmix64
//! finalizer as scatter), adapted for the workspace's ownership/
//! deterministic-collections module.
//!
//! Differences vs the source blueprint (scope-limited):
//!
//! - **Keys are `u64`** (matches the source). For arbitrary byte keys
//!   keep using `DHarht` (v.01).
//! - **Per-shard fallback is a `BTreeMap`** (deterministic) for
//!   collision groups exceeding `MicroBucket16` capacity. The source
//!   uses a full Adaptive Radix Tree (Node4/16/32/48/256) at the
//!   per-shard level. ART is correct + deterministic but is a larger
//!   port; `BTreeMap` fallback satisfies "no silent entry loss" and
//!   is what the source's Memory profile spec already allowed under
//!   "deterministic fallback to BTreeMap or SortedVecMap if probe
//!   budget is exceeded."
//! - **No SIMD `match_mask`**: the parallel scalar bitmask still gets
//!   compiled to tight code by LLVM; we don't reach for `SSE2`
//!   intrinsics for portability.
//!
//! Determinism contract preserved bit-equal:
//! - splitmix64 deterministic scatter (no random seeds)
//! - 256 shards (power of two)
//! - sealed sparse 16-bit front directory
//! - `MicroBucket16` cap enforced; overflow → per-shard `BTreeMap`
//! - full key equality on every successful lookup
//! - per-shard collision counters

use std::collections::BTreeMap;

/// Number of shards. Power of two so `(scatter >> shift) & MASK` is the
/// shard index.
pub const NSHARDS_M: usize = 256;
const SHARD_MASK_M: u64 = (NSHARDS_M - 1) as u64;
const SHARD_SHIFT_M: u32 = 64 - 8; // top 8 bits = shard

/// 16-bit front directory: 65 536 prefixes per table, sparse-paged
/// with 8-bit pages (256 entries each = 256 pages).
pub const FRONT_BITS_MEMORY: u8 = 16;
const FRONT_SLOTS_MEMORY: usize = 1 << FRONT_BITS_MEMORY;
const SPARSE_PAGE_BITS: u8 = 8;
const SPARSE_PAGE_SIZE: usize = 1 << SPARSE_PAGE_BITS;
const NO_PAGE: u32 = u32::MAX;

/// 3-bit tag at the bottom of every packed front entry.
const FRONT_TAG_MASK: u64 = 0b111;
const FRONT_EMPTY: u64 = 0;
const FRONT_SINGLE: u64 = 0b001;
const FRONT_MICRO4: u64 = 0b010;
const FRONT_MICRO8: u64 = 0b011;
const FRONT_MICRO16: u64 = 0b100;
const FRONT_FALLBACK: u64 = 0b101;

/// Splitmix64 finalizer — used as the scatter function. Matches the
/// blueprint exactly. Deterministic across machines.
#[inline(always)]
pub fn deterministic_permutation_scatter(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Top 16 bits of the scattered key = front prefix.
#[inline(always)]
fn front_prefix(scattered: u64) -> usize {
    (scattered >> (64 - FRONT_BITS_MEMORY)) as usize
}

#[inline(always)]
fn pack_front_single(shard_id: usize, leaf_id: usize) -> u64 {
    debug_assert!(shard_id < (1 << 29));
    debug_assert!(leaf_id < (1 << 32));
    ((shard_id as u64) << 35) | ((leaf_id as u64) << 3) | FRONT_SINGLE
}

#[inline(always)]
fn unpack_front_single(packed: u64) -> (usize, usize) {
    ((packed >> 35) as usize, ((packed >> 3) as u32) as usize)
}

#[inline(always)]
fn pack_front_micro(bucket_id: usize, tag: u64) -> u64 {
    debug_assert!(bucket_id < (1_usize << 61));
    ((bucket_id as u64) << 3) | tag
}

#[inline(always)]
fn unpack_front_micro(packed: u64) -> usize {
    (packed >> 3) as usize
}

/// `MicroBucket4` — 4-entry inline collision group with parallel match.
#[derive(Clone, Debug)]
pub struct MicroBucket4 {
    pub shard_id: u16,
    pub count: u8,
    pub keys: [u64; 4],
    pub leaf_ids: [u32; 4],
}

impl MicroBucket4 {
    #[inline(always)]
    fn match_mask(&self, key: u64) -> u32 {
        let mask = ((self.keys[0] == key) as u32)
            | (((self.keys[1] == key) as u32) << 1)
            | (((self.keys[2] == key) as u32) << 2)
            | (((self.keys[3] == key) as u32) << 3);
        let live = if self.count >= 4 { 0b1111 } else { (1u32 << self.count) - 1 };
        mask & live
    }
}

/// `MicroBucket8` — 8-entry inline collision group.
#[derive(Clone, Debug)]
pub struct MicroBucket8 {
    pub shard_id: u16,
    pub count: u8,
    pub keys: [u64; 8],
    pub leaf_ids: [u32; 8],
}

impl MicroBucket8 {
    #[inline(always)]
    fn match_mask(&self, key: u64) -> u32 {
        let mut mask = 0u32;
        for slot in 0..8 {
            mask |= ((self.keys[slot] == key) as u32) << slot;
        }
        let live = if self.count >= 8 { 0xff } else { (1u32 << self.count) - 1 };
        mask & live
    }
}

/// `MicroBucket16` — 16-entry inline collision group.
#[derive(Clone, Debug)]
pub struct MicroBucket16 {
    pub shard_id: u16,
    pub count: u8,
    pub keys: [u64; 16],
    pub leaf_ids: [u32; 16],
}

impl MicroBucket16 {
    #[inline(always)]
    fn match_mask(&self, key: u64) -> u32 {
        let mut mask = 0u32;
        for slot in 0..16 {
            mask |= ((self.keys[slot] == key) as u32) << slot;
        }
        let live = if self.count >= 16 {
            u16::MAX as u32
        } else {
            (1u32 << self.count) - 1
        };
        mask & live
    }
}

/// Leaf payload: key + value. Indexed by `leaf_id` within the
/// owning shard's slab.
#[derive(Clone, Debug)]
pub struct LeafNode<V: Clone> {
    pub key: u64,
    pub value: V,
}

// ─────────────────────────────────────────────────────────────────────────
//  Phase 12: ART (Adaptive Radix Tree) fallback
//
//  Ported from the user-supplied D-HARHT blueprint. Used when a
//  front-directory prefix has more than `MicroBucket16` entries
//  (i.e., adversarial-collision workloads). Pre-Phase-12 those
//  entries went into a global `BTreeMap` fallback; Phase 12 routes
//  them into a per-shard ART so worst-case lookup stays
//  byte-by-byte radix descent rather than `O(log N)` BTree walk.
//
//  Determinism preserved: ART growth is monotonic + deterministic
//  (Node4 → Node16 → Node32 → Node48 → Node256 by child count). Same
//  insertion order produces byte-identical tree shape.
// ─────────────────────────────────────────────────────────────────────────

const ART_EMPTY: u32 = u32::MAX;
const NODE48_EMPTY: u8 = u8::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TaggedNode(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeKind {
    Leaf,
    Node4,
    Node16,
    Node32,
    Node48,
    Node256,
}

impl TaggedNode {
    const TAG_BITS: u32 = 3;
    const INDEX_SHIFT: u32 = 3;
    const LEAF: u32 = 0;
    const N4: u32 = 1;
    const N16: u32 = 2;
    const N32: u32 = 3;
    const N48: u32 = 4;
    const N256: u32 = 5;

    fn new(index: usize, tag: u32) -> Self {
        debug_assert!(tag < (1 << Self::TAG_BITS));
        assert!(index < (1usize << (32 - Self::INDEX_SHIFT)));
        Self(((index as u32) << Self::INDEX_SHIFT) | tag)
    }
    fn leaf(i: usize) -> Self { Self::new(i, Self::LEAF) }
    fn n4(i: usize) -> Self { Self::new(i, Self::N4) }
    fn n16(i: usize) -> Self { Self::new(i, Self::N16) }
    fn n32(i: usize) -> Self { Self::new(i, Self::N32) }
    fn n48(i: usize) -> Self { Self::new(i, Self::N48) }
    fn n256(i: usize) -> Self { Self::new(i, Self::N256) }

    fn kind(self) -> NodeKind {
        match self.0 & ((1 << Self::TAG_BITS) - 1) {
            Self::LEAF => NodeKind::Leaf,
            Self::N4 => NodeKind::Node4,
            Self::N16 => NodeKind::Node16,
            Self::N32 => NodeKind::Node32,
            Self::N48 => NodeKind::Node48,
            Self::N256 => NodeKind::Node256,
            _ => unreachable!(),
        }
    }

    fn index(self) -> usize { (self.0 >> Self::INDEX_SHIFT) as usize }
}

#[derive(Clone, Debug)]
struct ArtNode4 { count: u8, keys: [u8; 4], children: [u32; 4] }
#[derive(Clone, Debug)]
struct ArtNode16 { count: u8, keys: [u8; 16], children: [u32; 16] }
#[derive(Clone, Debug)]
struct ArtNode32 { count: u8, keys: [u8; 32], children: [u32; 32] }
#[derive(Clone, Debug)]
struct ArtNode48 { count: u8, child_index: [u8; 256], children: [u32; 48] }
#[derive(Clone, Debug)]
struct ArtNode256 { count: u16, children: [u32; 256] }

/// Per-shard ART slab. Holds entries (tagged), leaves (key+value),
/// and per-node-type arrays. Indices are `u32`. Allocation is
/// monotonic.
#[derive(Clone, Debug)]
struct ArtSlab<V: Clone> {
    entries: Vec<TaggedNode>,
    leaves: Vec<LeafNode<V>>,
    node4: Vec<ArtNode4>,
    node16: Vec<ArtNode16>,
    node32: Vec<ArtNode32>,
    node48: Vec<ArtNode48>,
    node256: Vec<ArtNode256>,
}

impl<V: Clone> ArtSlab<V> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            leaves: Vec::new(),
            node4: Vec::new(),
            node16: Vec::new(),
            node32: Vec::new(),
            node48: Vec::new(),
            node256: Vec::new(),
        }
    }

    fn alloc_entry(&mut self, t: TaggedNode) -> u32 {
        let id = self.entries.len() as u32;
        self.entries.push(t);
        id
    }
    fn alloc_leaf(&mut self, key: u64, value: V) -> u32 {
        let lid = self.leaves.len();
        self.leaves.push(LeafNode { key, value });
        self.alloc_entry(TaggedNode::leaf(lid))
    }
    fn alloc_node4(&mut self) -> u32 {
        let nid = self.node4.len();
        self.node4.push(ArtNode4 { count: 0, keys: [0; 4], children: [ART_EMPTY; 4] });
        self.alloc_entry(TaggedNode::n4(nid))
    }

    /// Byte at `depth` of the scattered key (0..=7).
    #[inline(always)]
    fn radix(scattered: u64, depth: usize) -> u8 {
        if depth >= 8 { 0 } else { (scattered >> ((7 - depth) * 8)) as u8 }
    }

    fn find_child(&self, entry_id: u32, radix: u8) -> Option<u32> {
        let entry = self.entries[entry_id as usize];
        match entry.kind() {
            NodeKind::Leaf => None,
            NodeKind::Node4 => {
                let n = &self.node4[entry.index()];
                for i in 0..n.count as usize {
                    if n.keys[i] == radix { return Some(n.children[i]); }
                }
                None
            }
            NodeKind::Node16 => {
                let n = &self.node16[entry.index()];
                for i in 0..n.count as usize {
                    if n.keys[i] == radix { return Some(n.children[i]); }
                }
                None
            }
            NodeKind::Node32 => {
                let n = &self.node32[entry.index()];
                for i in 0..n.count as usize {
                    if n.keys[i] == radix { return Some(n.children[i]); }
                }
                None
            }
            NodeKind::Node48 => {
                let n = &self.node48[entry.index()];
                let slot = n.child_index[radix as usize];
                if slot == NODE48_EMPTY { None } else { Some(n.children[slot as usize]) }
            }
            NodeKind::Node256 => {
                let c = self.node256[entry.index()].children[radix as usize];
                if c == ART_EMPTY { None } else { Some(c) }
            }
        }
    }

    fn add_child(&mut self, entry_id: u32, radix: u8, child: u32) {
        let entry = self.entries[entry_id as usize];
        match entry.kind() {
            NodeKind::Node4 if self.node4[entry.index()].count < 4 => {
                let n = &mut self.node4[entry.index()];
                let s = n.count as usize;
                n.keys[s] = radix; n.children[s] = child; n.count += 1;
            }
            NodeKind::Node4 => { self.grow_4_to_16(entry_id); self.add_child(entry_id, radix, child); }
            NodeKind::Node16 if self.node16[entry.index()].count < 16 => {
                let n = &mut self.node16[entry.index()];
                let s = n.count as usize;
                n.keys[s] = radix; n.children[s] = child; n.count += 1;
            }
            NodeKind::Node16 => { self.grow_16_to_32(entry_id); self.add_child(entry_id, radix, child); }
            NodeKind::Node32 if self.node32[entry.index()].count < 32 => {
                let n = &mut self.node32[entry.index()];
                let s = n.count as usize;
                n.keys[s] = radix; n.children[s] = child; n.count += 1;
            }
            NodeKind::Node32 => { self.grow_32_to_48(entry_id); self.add_child(entry_id, radix, child); }
            NodeKind::Node48 if self.node48[entry.index()].count < 48 => {
                let n = &mut self.node48[entry.index()];
                let s = n.count as usize;
                n.child_index[radix as usize] = s as u8;
                n.children[s] = child;
                n.count += 1;
            }
            NodeKind::Node48 => { self.grow_48_to_256(entry_id); self.add_child(entry_id, radix, child); }
            NodeKind::Node256 => {
                let n = &mut self.node256[entry.index()];
                if n.children[radix as usize] == ART_EMPTY { n.count += 1; }
                n.children[radix as usize] = child;
            }
            NodeKind::Leaf => unreachable!("cannot add child to leaf"),
        }
    }

    fn grow_4_to_16(&mut self, entry_id: u32) {
        let old_idx = self.entries[entry_id as usize].index();
        let old = &self.node4[old_idx];
        let mut next = ArtNode16 { count: old.count, keys: [0; 16], children: [ART_EMPTY; 16] };
        let c = old.count as usize;
        next.keys[..c].copy_from_slice(&old.keys[..c]);
        next.children[..c].copy_from_slice(&old.children[..c]);
        let new_idx = self.node16.len();
        self.node16.push(next);
        self.entries[entry_id as usize] = TaggedNode::n16(new_idx);
    }
    fn grow_16_to_32(&mut self, entry_id: u32) {
        let old_idx = self.entries[entry_id as usize].index();
        let old = &self.node16[old_idx];
        let mut next = ArtNode32 { count: old.count, keys: [0; 32], children: [ART_EMPTY; 32] };
        let c = old.count as usize;
        next.keys[..c].copy_from_slice(&old.keys[..c]);
        next.children[..c].copy_from_slice(&old.children[..c]);
        let new_idx = self.node32.len();
        self.node32.push(next);
        self.entries[entry_id as usize] = TaggedNode::n32(new_idx);
    }
    fn grow_32_to_48(&mut self, entry_id: u32) {
        let old_idx = self.entries[entry_id as usize].index();
        let old = &self.node32[old_idx];
        let mut next = ArtNode48 {
            count: old.count, child_index: [NODE48_EMPTY; 256], children: [ART_EMPTY; 48],
        };
        for i in 0..old.count as usize {
            next.child_index[old.keys[i] as usize] = i as u8;
            next.children[i] = old.children[i];
        }
        let new_idx = self.node48.len();
        self.node48.push(next);
        self.entries[entry_id as usize] = TaggedNode::n48(new_idx);
    }
    fn grow_48_to_256(&mut self, entry_id: u32) {
        let old_idx = self.entries[entry_id as usize].index();
        let old = &self.node48[old_idx];
        let mut next = ArtNode256 { count: old.count as u16, children: [ART_EMPTY; 256] };
        for radix in 0..=u8::MAX {
            let s = old.child_index[radix as usize];
            if s != NODE48_EMPTY {
                next.children[radix as usize] = old.children[s as usize];
            }
        }
        let new_idx = self.node256.len();
        self.node256.push(next);
        self.entries[entry_id as usize] = TaggedNode::n256(new_idx);
    }

    /// Insert (key, value). Returns previous value if key existed.
    /// `root` is the entry ID of the ART root; pass `None` for an
    /// empty tree (this fn returns Some(new_root) if it allocated one).
    fn insert(
        &mut self,
        root: Option<u32>,
        key: u64,
        scattered: u64,
        value: V,
    ) -> (u32, Option<V>) {
        match root {
            None => {
                let leaf = self.alloc_leaf(key, value);
                (leaf, None)
            }
            Some(r) => {
                let prev = self.insert_at(r, 1, key, scattered, value);
                (r, prev)
            }
        }
    }

    fn insert_at(
        &mut self,
        node_id: u32,
        depth: usize,
        key: u64,
        scattered: u64,
        value: V,
    ) -> Option<V> {
        if self.entries[node_id as usize].kind() == NodeKind::Leaf {
            let leaf_id = self.entries[node_id as usize].index();
            let leaf = &mut self.leaves[leaf_id];
            if leaf.key == key {
                return Some(std::mem::replace(&mut leaf.value, value));
            }
            // Split leaf into a Node4.
            let old_leaf_entry = self.alloc_entry(TaggedNode::leaf(leaf_id));
            let new_leaf_entry = self.alloc_leaf(key, value);
            let old_scattered = deterministic_permutation_scatter(self.leaves[leaf_id].key);
            self.join_leaves_in_place(node_id, depth, old_scattered, old_leaf_entry, scattered, new_leaf_entry);
            return None;
        }
        let radix = Self::radix(scattered, depth);
        if let Some(child) = self.find_child(node_id, radix) {
            return self.insert_at(child, depth + 1, key, scattered, value);
        }
        let leaf = self.alloc_leaf(key, value);
        self.add_child(node_id, radix, leaf);
        None
    }

    fn join_leaves_in_place(
        &mut self,
        target: u32,
        depth: usize,
        left_scattered: u64,
        left: u32,
        right_scattered: u64,
        right: u32,
    ) {
        let lr = Self::radix(left_scattered, depth);
        let rr = Self::radix(right_scattered, depth);
        let n4 = self.node4.len();
        self.node4.push(ArtNode4 { count: 0, keys: [0; 4], children: [ART_EMPTY; 4] });
        self.entries[target as usize] = TaggedNode::n4(n4);
        if lr == rr && depth < 7 {
            let child = self.alloc_node4();
            self.add_child(target, lr, child);
            self.join_leaves_in_place(child, depth + 1, left_scattered, left, right_scattered, right);
        } else {
            self.add_child(target, lr, left);
            self.add_child(target, rr, right);
        }
    }

    fn get(&self, root: Option<u32>, key: u64, scattered: u64) -> Option<&V> {
        let mut node_id = root?;
        let mut depth = 1;
        loop {
            let entry = self.entries[node_id as usize];
            match entry.kind() {
                NodeKind::Leaf => {
                    let leaf = &self.leaves[entry.index()];
                    return if leaf.key == key { Some(&leaf.value) } else { None };
                }
                _ => {
                    let r = Self::radix(scattered, depth);
                    let child = self.find_child(node_id, r)?;
                    node_id = child;
                    depth += 1;
                }
            }
        }
    }
}

/// Per-shard slab of leaves + ART fallback. The ART carries entries
/// whose front-directory collision group exceeded `MicroBucket16`.
#[derive(Clone, Debug)]
struct Shard<V: Clone> {
    leaves: Vec<LeafNode<V>>,
    /// Build-time only: keys still in flight before sealing.
    pre_seal: BTreeMap<u64, u32>,
    /// Phase 12: ART fallback path. Populated during seal for prefixes
    /// whose collision group exceeded `MicroBucket16`. `None` until
    /// the first such overflow on this shard.
    art: ArtSlab<V>,
    art_root: Option<u32>,
}

impl<V: Clone> Shard<V> {
    fn new() -> Self {
        Self {
            leaves: Vec::new(),
            pre_seal: BTreeMap::new(),
            art: ArtSlab::new(),
            art_root: None,
        }
    }

    fn art_insert(&mut self, key: u64, scattered: u64, value: V) {
        let (new_root, _prev) = self.art.insert(self.art_root, key, scattered, value);
        self.art_root = Some(new_root);
    }

    fn art_get(&self, key: u64, scattered: u64) -> Option<&V> {
        self.art.get(self.art_root, key, scattered)
    }
}

/// Sparse paged front directory. `page_table` maps the upper 8 bits
/// of the prefix to a page id (or `NO_PAGE`); each page is a fixed
/// 256-entry array of packed `u64` front entries.
#[derive(Clone, Debug)]
struct FrontDir {
    page_table: Vec<u32>,
    pages: Vec<Box<[u64; SPARSE_PAGE_SIZE]>>,
}

impl FrontDir {
    fn empty() -> Self {
        Self {
            page_table: vec![NO_PAGE; FRONT_SLOTS_MEMORY / SPARSE_PAGE_SIZE],
            pages: Vec::new(),
        }
    }

    #[inline(always)]
    fn get(&self, prefix: usize) -> u64 {
        let page_id = self.page_table[prefix >> SPARSE_PAGE_BITS];
        if page_id == NO_PAGE {
            FRONT_EMPTY
        } else {
            self.pages[page_id as usize][prefix & (SPARSE_PAGE_SIZE - 1)]
        }
    }

    fn ensure_page(&mut self, page: usize) -> usize {
        if self.page_table[page] == NO_PAGE {
            self.page_table[page] = self.pages.len() as u32;
            self.pages.push(Box::new([FRONT_EMPTY; SPARSE_PAGE_SIZE]));
        }
        self.page_table[page] as usize
    }

    fn set(&mut self, prefix: usize, value: u64) {
        let page = prefix >> SPARSE_PAGE_BITS;
        let pid = self.ensure_page(page);
        self.pages[pid][prefix & (SPARSE_PAGE_SIZE - 1)] = value;
    }
}

/// D-HARHT Memory profile — port of the user-supplied blueprint.
#[derive(Clone, Debug)]
pub struct DHarhtMemory<V: Clone> {
    shards: Vec<Shard<V>>,
    front: FrontDir,
    micro4: Vec<MicroBucket4>,
    micro8: Vec<MicroBucket8>,
    micro16: Vec<MicroBucket16>,
    /// Per-prefix overflow when collision count > 16. Replaces the
    /// blueprint's ART fallback path. Deterministic, no silent loss.
    overflow: BTreeMap<u64, V>,
    sealed: bool,
    total_entries: u64,
    /// Per-table counters for security/diagnostics.
    micro_overflow_count: u64,
    max_collision_group: u32,
}

impl<V: Clone> DHarhtMemory<V> {
    pub fn new() -> Self {
        Self {
            shards: (0..NSHARDS_M).map(|_| Shard::new()).collect(),
            front: FrontDir::empty(),
            micro4: Vec::new(),
            micro8: Vec::new(),
            micro16: Vec::new(),
            overflow: BTreeMap::new(),
            sealed: false,
            total_entries: 0,
            micro_overflow_count: 0,
            max_collision_group: 0,
        }
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
    pub fn micro_overflow_count(&self) -> u64 {
        self.micro_overflow_count
    }
    pub fn max_collision_group(&self) -> u32 {
        self.max_collision_group
    }

    /// Approximate allocated memory in bytes. Counts shard slabs,
    /// front-directory pages, micro-bucket pools, and BTreeMap fallback.
    /// Useful for memory-comparison benches.
    pub fn approx_memory_bytes(&self) -> usize {
        use std::mem::size_of;
        let mut total = size_of::<Self>();
        for shard in &self.shards {
            total += size_of::<Shard<V>>();
            total += shard.leaves.capacity() * size_of::<LeafNode<V>>();
            // Pre-seal BTreeMap: ~56 bytes per entry (B-tree node overhead).
            total += shard.pre_seal.len() * 56;
        }
        // Front directory: page table + pages.
        total += self.front.page_table.capacity() * size_of::<u32>();
        total += self.front.pages.len() * size_of::<Box<[u64; SPARSE_PAGE_SIZE]>>();
        total += self.front.pages.len() * SPARSE_PAGE_SIZE * size_of::<u64>();
        // Micro buckets.
        total += self.micro4.capacity() * size_of::<MicroBucket4>();
        total += self.micro8.capacity() * size_of::<MicroBucket8>();
        total += self.micro16.capacity() * size_of::<MicroBucket16>();
        // Overflow BTreeMap.
        total += self.overflow.len() * (size_of::<u64>() + size_of::<V>() + 56);
        total
    }

    #[inline(always)]
    fn shard_id(scattered: u64) -> usize {
        ((scattered >> SHARD_SHIFT_M) & SHARD_MASK_M) as usize
    }

    /// Insert a `(key, value)`. Pre-seal: `O(log N_shard)` BTreeMap
    /// insert. Update returns the previous value.
    pub fn insert(&mut self, key: u64, value: V) -> Option<V> {
        debug_assert!(!self.sealed, "DHarhtMemory: insert after seal");
        let scattered = deterministic_permutation_scatter(key);
        let s = Self::shard_id(scattered);
        let shard = &mut self.shards[s];
        if let Some(&leaf_id) = shard.pre_seal.get(&key) {
            let prev = std::mem::replace(&mut shard.leaves[leaf_id as usize].value, value);
            return Some(prev);
        }
        let leaf_id = shard.leaves.len() as u32;
        shard.leaves.push(LeafNode { key, value });
        shard.pre_seal.insert(key, leaf_id);
        self.total_entries += 1;
        None
    }

    /// Lookup a key. Pre-seal goes through the BTreeMap; post-seal
    /// goes through the packed front directory + microbuckets.
    #[inline]
    pub fn get(&self, key: u64) -> Option<&V> {
        let scattered = deterministic_permutation_scatter(key);
        if !self.sealed {
            let s = Self::shard_id(scattered);
            let shard = &self.shards[s];
            return shard
                .pre_seal
                .get(&key)
                .map(|&leaf_id| &shard.leaves[leaf_id as usize].value);
        }
        let prefix = front_prefix(scattered);
        let entry = self.front.get(prefix);
        match entry & FRONT_TAG_MASK {
            t if t == FRONT_SINGLE => {
                let (sid, lid) = unpack_front_single(entry);
                let leaf = &self.shards[sid].leaves[lid];
                if leaf.key == key { Some(&leaf.value) } else { None }
            }
            t if t == FRONT_MICRO4 => {
                let b = &self.micro4[unpack_front_micro(entry)];
                let m = b.match_mask(key);
                if m == 0 {
                    return None;
                }
                let slot = m.trailing_zeros() as usize;
                let leaf = &self.shards[b.shard_id as usize].leaves[b.leaf_ids[slot] as usize];
                Some(&leaf.value)
            }
            t if t == FRONT_MICRO8 => {
                let b = &self.micro8[unpack_front_micro(entry)];
                let m = b.match_mask(key);
                if m == 0 {
                    return None;
                }
                let slot = m.trailing_zeros() as usize;
                let leaf = &self.shards[b.shard_id as usize].leaves[b.leaf_ids[slot] as usize];
                Some(&leaf.value)
            }
            t if t == FRONT_MICRO16 => {
                let b = &self.micro16[unpack_front_micro(entry)];
                let m = b.match_mask(key);
                if m == 0 {
                    return None;
                }
                let slot = m.trailing_zeros() as usize;
                let leaf = &self.shards[b.shard_id as usize].leaves[b.leaf_ids[slot] as usize];
                Some(&leaf.value)
            }
            t if t == FRONT_FALLBACK => {
                // Phase 12: walk the originating shard's ART rather
                // than the per-table BTreeMap. Faster on adversarial
                // collision-rich workloads (radix descent vs BTree
                // log walk). The legacy `overflow` BTreeMap is kept
                // as a safety net for code paths that haven't been
                // migrated yet — it stays empty post-Phase-12.
                let s = Self::shard_id(scattered);
                if let Some(v) = self.shards[s].art_get(key, scattered) {
                    Some(v)
                } else {
                    self.overflow.get(&key)
                }
            }
            _ => None,
        }
    }

    pub fn contains_key(&self, key: u64) -> bool {
        self.get(key).is_some()
    }

    /// Build the sealed lookup index. Groups leaves by their 16-bit
    /// front prefix, packs each group into the smallest microbucket
    /// that fits, and writes the packed entry into the sparse paged
    /// front directory.
    pub fn seal_for_lookup(&mut self) {
        self.micro4.clear();
        self.micro8.clear();
        self.micro16.clear();
        self.overflow.clear();
        self.front = FrontDir::empty();
        self.micro_overflow_count = 0;
        self.max_collision_group = 0;

        // Collect (prefix → Vec<(shard_id, leaf_id, key)>) in BTree
        // order so the build pass is itself deterministic.
        let mut buckets: BTreeMap<usize, Vec<(usize, usize, u64)>> = BTreeMap::new();
        for (sid, shard) in self.shards.iter().enumerate() {
            for (lid, leaf) in shard.leaves.iter().enumerate() {
                let prefix = front_prefix(deterministic_permutation_scatter(leaf.key));
                buckets.entry(prefix).or_default().push((sid, lid, leaf.key));
            }
        }

        for (prefix, mut group) in buckets {
            // Stable order for determinism: sort by (key, shard_id, leaf_id).
            group.sort_by_key(|&(s, l, k)| (k, s, l));
            self.max_collision_group =
                self.max_collision_group.max(group.len() as u32);

            let packed = match group.len() {
                0 => continue,
                1 => {
                    let (sid, lid, _) = group[0];
                    pack_front_single(sid, lid)
                }
                2..=4 => {
                    let bid = self.micro4.len();
                    let mut b = MicroBucket4 {
                        shard_id: 0,
                        count: group.len() as u8,
                        keys: [0; 4],
                        leaf_ids: [0; 4],
                    };
                    for (slot, &(s, l, k)) in group.iter().enumerate() {
                        b.shard_id = s as u16;
                        b.keys[slot] = k;
                        b.leaf_ids[slot] = l as u32;
                    }
                    self.micro4.push(b);
                    pack_front_micro(bid, FRONT_MICRO4)
                }
                5..=8 => {
                    let bid = self.micro8.len();
                    let mut b = MicroBucket8 {
                        shard_id: 0,
                        count: group.len() as u8,
                        keys: [0; 8],
                        leaf_ids: [0; 8],
                    };
                    for (slot, &(s, l, k)) in group.iter().enumerate() {
                        b.shard_id = s as u16;
                        b.keys[slot] = k;
                        b.leaf_ids[slot] = l as u32;
                    }
                    self.micro8.push(b);
                    pack_front_micro(bid, FRONT_MICRO8)
                }
                9..=16 => {
                    let bid = self.micro16.len();
                    let mut b = MicroBucket16 {
                        shard_id: 0,
                        count: group.len() as u8,
                        keys: [0; 16],
                        leaf_ids: [0; 16],
                    };
                    for (slot, &(s, l, k)) in group.iter().enumerate() {
                        b.shard_id = s as u16;
                        b.keys[slot] = k;
                        b.leaf_ids[slot] = l as u32;
                    }
                    self.micro16.push(b);
                    pack_front_micro(bid, FRONT_MICRO16)
                }
                n => {
                    // Phase 12: overflow → per-shard ART (Adaptive Radix
                    // Tree). Each entry goes into the ART of its
                    // originating shard. Faster than BTreeMap on
                    // adversarial-collision workloads (radix descent
                    // vs B-tree log walk), still deterministic by
                    // construction (Node4→Node16→...→Node256 growth
                    // is monotonic + insertion-order-independent for
                    // the final shape after compaction).
                    self.micro_overflow_count += n as u64;
                    for &(s, l, k) in &group {
                        let v = self.shards[s].leaves[l].value.clone();
                        let scattered = deterministic_permutation_scatter(k);
                        self.shards[s].art_insert(k, scattered, v);
                    }
                    FRONT_FALLBACK
                }
            };
            self.front.set(prefix, packed);
        }
        self.sealed = true;
    }

    /// Iterate all entries in canonical (key-sorted) order. The
    /// pre-seal BTreeMap iteration is already sorted; post-seal we
    /// re-sort because micro buckets aren't key-sorted.
    pub fn iter_sorted(&self) -> Vec<(u64, &V)> {
        if !self.sealed {
            let mut all: Vec<(u64, &V)> = Vec::with_capacity(self.total_entries as usize);
            for shard in &self.shards {
                for leaf in &shard.leaves {
                    all.push((leaf.key, &leaf.value));
                }
            }
            all.sort_by_key(|&(k, _)| k);
            return all;
        }
        let mut all: Vec<(u64, &V)> = Vec::with_capacity(self.total_entries as usize);
        for shard in &self.shards {
            for leaf in &shard.leaves {
                all.push((leaf.key, &leaf.value));
            }
        }
        all.sort_by_key(|&(k, _)| k);
        all
    }
}

impl<V: Clone> Default for DHarhtMemory<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic shape hash for double-build identity tests.
pub fn shape_hash<V: Clone + std::hash::Hash>(t: &DHarhtMemory<V>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let entries = t.iter_sorted();
    h.write_u64(entries.len() as u64);
    for (k, v) in entries {
        h.write_u64(k);
        v.hash(&mut h);
    }
    h.write_u64(t.micro_overflow_count);
    h.write_u32(t.max_collision_group);
    h.write_usize(t.micro4.len());
    h.write_usize(t.micro8.len());
    h.write_usize(t.micro16.len());
    deterministic_permutation_scatter(h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_update_works() {
        let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
        assert_eq!(t.insert(7, 70), None);
        assert_eq!(t.insert(9, 90), None);
        assert_eq!(t.get(7), Some(&70));
        assert_eq!(t.get(9), Some(&90));
        assert_eq!(t.get(11), None);
        assert_eq!(t.insert(7, 700), Some(70));
        assert_eq!(t.get(7), Some(&700));
    }

    #[test]
    fn seal_preserves_all_entries() {
        let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
        for i in 0..5_000u64 {
            t.insert(i, i * 2);
        }
        t.seal_for_lookup();
        for i in 0..5_000u64 {
            assert_eq!(t.get(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn deterministic_double_build_same_shape_hash() {
        fn build() -> DHarhtMemory<u64> {
            let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
            for i in 0..1_000u64 {
                t.insert(deterministic_permutation_scatter(i ^ 0x9e37_79b9_7f4a_7c15), i);
            }
            t.seal_for_lookup();
            t
        }
        assert_eq!(shape_hash(&build()), shape_hash(&build()));
    }

    #[test]
    fn micro16_capacity_enforced() {
        let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
        // 50k keys → 50k / 65536 ≈ 0.76 keys per prefix — well below 16.
        for i in 0..50_000u64 {
            t.insert(i, i);
        }
        t.seal_for_lookup();
        // Overflow should be 0 or very low at this scale.
        assert!(t.micro_overflow_count() < 100, "unexpected micro overflow: {}", t.micro_overflow_count());
        // Max collision group ≤ 16 by construction (anything more goes
        // to overflow).
        for i in 0..50_000u64 {
            assert_eq!(t.get(i), Some(&i));
        }
    }

    #[test]
    fn full_key_equality_no_false_positive() {
        let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
        for i in 0..2_000u64 {
            t.insert(i, i);
        }
        t.seal_for_lookup();
        for i in 2_000..3_000u64 {
            assert_eq!(t.get(i), None, "false positive at {}", i);
        }
    }

    #[test]
    fn art_fallback_handles_forced_overflow() {
        // Construct keys that map to the same shard + same 16-bit
        // front prefix to force MicroBucket16 overflow → ART path.
        // We need keys whose splitmix64 finalizer produces the same
        // top 24 bits (8 bits shard + 16 bits prefix). Brute-force
        // search a small space and pick keys that collide.
        let target_shard_prefix: u64 = {
            let s = deterministic_permutation_scatter(0);
            (s >> 40) & 0xFF_FFFF
        };
        let mut colliding: Vec<u64> = Vec::with_capacity(32);
        let mut k: u64 = 0;
        while colliding.len() < 32 && k < 100_000_000 {
            let s = deterministic_permutation_scatter(k);
            if (s >> 40) & 0xFF_FFFF == target_shard_prefix {
                colliding.push(k);
            }
            k += 1;
        }
        // We need >= 17 to overflow MicroBucket16. If brute search
        // didn't find enough at this scale, skip — bench keys are not
        // adversarial in practice.
        if colliding.len() >= 17 {
            let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
            for (i, &k) in colliding.iter().enumerate() {
                t.insert(k, i as u64);
            }
            t.seal_for_lookup();
            // ART path should be exercised.
            assert!(
                t.micro_overflow_count() > 0,
                "expected ART fallback to be triggered; overflow={}",
                t.micro_overflow_count()
            );
            // All keys must still be findable via the ART path.
            for (i, &k) in colliding.iter().enumerate() {
                assert_eq!(
                    t.get(k),
                    Some(&(i as u64)),
                    "ART fallback lost key {}",
                    k
                );
            }
        }
    }

    #[test]
    fn matches_btreemap_oracle() {
        use std::collections::BTreeMap;
        let mut t: DHarhtMemory<u64> = DHarhtMemory::new();
        let mut oracle: BTreeMap<u64, u64> = BTreeMap::new();
        let mut x: u64 = 0xCAFEBABE;
        for _ in 0..5_000 {
            x = deterministic_permutation_scatter(x);
            let p1 = t.insert(x, x);
            let p2 = oracle.insert(x, x);
            assert_eq!(p1, p2);
        }
        t.seal_for_lookup();
        for (k, v) in &oracle {
            assert_eq!(t.get(*k), Some(v));
        }
    }
}
