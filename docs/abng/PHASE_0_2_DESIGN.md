# ABNG Phase 0.2 — Multi-node Arena, Children, Routing (Design Note)

**Date:** 2026-05-05
**Builds on:** [Phase 0.1](PHASE_0_1_DESIGN.md), [ADR-0023](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023%20ABNG%20Adaptive%20Belief%20Radix%20Graph%20%28Phase%200.1%29.md)
**Scope:** Multi-node topology, `AdaptiveChildren` enum, prefix encoder, descend routing, per-node stats chain. **No structural decision triggers, no neural head.**

## Why this slice

Phase 0.1 shipped the substrate (arena + audit + replay) on a single root node. Phase 0.2 ships the *radix-tree topology* — multiple nodes, parent-child edges, the `AdaptiveChildren` enum that mirrors `cjc-data::AdaptiveSelection`, and a frozen-codebook prefix encoder that turns an input vector into a route.

Phase 0.2 deliberately stops short of:

* **Structural decision triggers** (Grow/Split/Merge/Prune/Compress/Freeze) — these depend on calibration error, NLL gain, signature divergence. None of those exist without a per-leaf neural head. Shipping empty policy buckets in 0.2 would freeze API shapes before semantics are designed.
* **Per-leaf MLPs / BLR head / OOD score / calibration bins** — Phase 0.3.
* **`Dense` compressed children variant** — Phase 0.4 with log compaction.

What 0.2 *does* deliver:

1. `AdaptiveBeliefNode.parent: Option<u32>` + `children: AdaptiveChildren`
2. `AdaptiveChildren::{None, Node4, Node16, Node48, Node256}` — five variants, the same density-classification idea as `AdaptiveSelection` (Empty/All/SelectionVector/VerbatimMask/Hybrid)
3. Auto-promotion (Node4→16→48→256) on capacity overflow
4. Quantile-codebook prefix encoder; deterministic; codebook frozen at first install
5. `descend(prefix)` returns a `RouteEvidence` (matched prefix length + leaf id + path)
6. Per-node stats chain decoupled from the global event chain (a node's stats chain only advances on its own observations)
7. Snapshot format **v2** — magic bumps `\x01` → `\x02`. Phase 0.1 snapshots no longer load. Documented break.

## AdaptiveChildren

Mirrors the `cjc-data::AdaptiveSelection` density-classification pattern: pick the smallest representation that fits the current population.

```rust
pub enum AdaptiveChildren {
    /// Leaf — terminal node, no children.
    None,
    /// 0–4 children. Linear scan over a parallel `keys`/`slots` array.
    Node4 {
        keys:  [u8; 4],
        slots: [Option<NodeId>; 4],
    },
    /// 5–16 children. Linear scan; extends Node4 once filled.
    Node16 {
        keys:  [u8; 16],
        slots: [Option<NodeId>; 16],
    },
    /// 17–48 children. ART's two-array trick: a 256-entry index byte,
    /// indirecting into a packed `slots` Vec.
    Node48 {
        index: Box<[u8; 256]>,        // 0xFF = empty slot
        slots: Vec<Option<NodeId>>,    // length up to 48
    },
    /// 49–256 children. Direct-indexed `[byte → child]` array.
    Node256 {
        slots: Box<[Option<NodeId>; 256]>,
    },
}
```

### Promotion

A child insertion that would push count past capacity *promotes* to the next size class. Promotion is deterministic, audited, and atomic — the old children variant is consumed; the new variant is constructed; the audit log records the transition.

```
Node4   (4)   → on 5th insert  → Node16 (16)
Node16  (16)  → on 17th insert → Node48 (48)
Node48  (48)  → on 49th insert → Node256 (256)
Node256       → no further promotion (max byte branching)
```

`Dense` (a fully-collapsed signature representation) is not in 0.2 — it requires a `NodeSignature` and behavior comparison, both of which need belief tensors.

### Why these breakpoints

Same design call as ART (Adaptive Radix Tree) and `AdaptiveSelection`: density-classify the children at construction. `Node48`'s two-array trick keeps memory ~1/5 of `Node256` for medium-arity nodes; `Node4`/`Node16` keep memory tiny for sparse nodes. Linear scan is fastest under 16 entries (cache-friendly); direct-index wins above ~48.

## Prefix encoder

```rust
pub struct QuantileCodebook {
    /// Per-dimension quantile boundaries. `bins[d]` is a sorted slice of
    /// length `n_bins-1` separating the `n_bins` bins for dimension `d`.
    pub bins:        Vec<Vec<f64>>,
    pub n_dims:      u8,
    pub n_bins:      u16,            // typically 256
    pub frozen_hash: [u8; 32],       // sha256 of canonical bytes
}
```

Once installed via `abng_set_codebook`, the codebook is **frozen**: subsequent calls error. A `CodebookFrozen` audit event is emitted. The codebook's `frozen_hash` becomes part of every subsequent snapshot's header so a snapshot is only replayable against a graph with the matching codebook.

Encoding `x: f64[D]` to `prefix: [u8; D]`:

```
for d in 0..D:
    let v = x[d];
    let bins = &codebook.bins[d];
    prefix[d] = binary_search(bins, v) as u8;     // 0..n_bins-1
```

Phase 0.2 supports `n_bins ∈ {2, 4, 8, 16, 32, 64, 128, 256}` (powers of two, clean byte mapping). 256 is the typical choice for tabular data; smaller values for low-cardinality categorical features.

Categorical and missing-value handling are deferred — Phase 0.3 adds explicit tag bytes (`0xFE` = missing, `0xFD..0xC0` = reserved) once we have schema tracking.

## RouteEvidence

```rust
pub struct RouteEvidence {
    /// Number of prefix bytes successfully matched, 0..D.
    pub matched_prefix: u8,
    /// Final node reached (might be intermediate if descent bailed).
    pub leaf_id:        u32,
    /// Path taken, root-first; length == matched_prefix + 1.
    pub path:           Vec<u32>,
}
```

`descend(prefix)` walks from the root, matching one byte per hop. When a child for the next byte doesn't exist, descent stops and returns the current node as the (effective) leaf. This is what gives a Phase 0.3+ ABNG model the ability to *abstain* with `RouteToFallback` when `matched_prefix < min_prefix_len`.

## Per-node stats chain

Phase 0.1: each event mutated stats *and* the global chain; per-node `stats_chain_head` was kept in lockstep with the global head.

Phase 0.2: per-node chain advances only when *that* node's stats change.

```
node.stats_chain_head_new = sha256(
    node.stats_chain_head_prev || node.stats.canonical_bytes()
)
```

The global event chain still records every event; per-node chains are an *additional* hash chain that lets you audit one node's evolution in isolation. This is what makes `cjcl abng inspect --node ID` a real thing in 0.4.

## New audit kinds

```rust
pub enum AuditKind {
    Created,                                            // Phase 0.1
    BeliefUpdate { value: f64 },                        // Phase 0.1
    NodeAdded { parent: u32, key_byte: u8 },            // Phase 0.2
    ChildrenPromoted { from: u8, to: u8 },              // Phase 0.2
    CodebookFrozen { codebook_hash: [u8; 32] },         // Phase 0.2
}
```

Tag bytes (frozen part of canonical encoding):

| Tag  | Kind |
|------|------|
| 0x00 | Created |
| 0x01 | BeliefUpdate { value } |
| 0x02 | NodeAdded { parent, key_byte } |
| 0x03 | ChildrenPromoted { from, to } |
| 0x04 | CodebookFrozen { codebook_hash } |

`from`/`to` use the same code as `abng_node_kind`: 0=None, 1=Node4, 2=Node16, 3=Node48, 4=Node256.

## Snapshot format v2

Magic `b"ABNG\x02"`. Phase 0.1 snapshots (`b"ABNG\x01"`) are rejected with `BadMagic`/`UnsupportedVersion`. This is a deliberate clean break — Phase 0.1 shipped 5 minutes before 0.2 and never had real users.

```
magic         "ABNG\x02"     (5)
seed          u64 BE         (8)
epoch         u64 BE         (8)
final_hash    [u8; 32]       (32)
codebook_present u8          (1)        0x00 = no codebook, 0x01 = present
if codebook_present:
  n_dims      u8             (1)
  n_bins      u16 BE         (2)
  for each dim d in 0..n_dims:
    n_boundaries u16 BE      (2)        always n_bins-1 in current spec
    boundaries  [f64 BE × n_boundaries]
  frozen_hash [u8; 32]       (32)       sha256 of canonical encoding above
n_nodes       u32 BE         (4)
per node:
  parent      i32 BE         (4)        -1 = root
  children_kind u8           (1)        0=None,1=Node4,2=Node16,3=Node48,4=Node256
  children_payload (variable, per kind, see below)
  stats canonical_bytes (24)
  stats_version u64 BE       (8)
  stats_chain_head [u8; 32]  (32)
n_events      u64 BE         (8)
per event:
  payload_len u32 BE         (4)
  payload     ...
  previous_hash [u8; 32]     (32)
  new_hash      [u8; 32]     (32)
```

Children payload by kind:

```
None:     <empty>
Node4:    keys[4] (4 bytes) + slots[4] (4 × i32 BE; -1 = empty)
Node16:   keys[16] (16) + slots[16] (16 × i32 BE)
Node48:   index[256] (256) + n_slots u8 (1) + slots[n_slots] × i32 BE
Node256:  256 × i32 BE
```

Replay rebuilds the graph from events alone (deterministic), then verifies:
1. Each replayed event's `new_hash` matches the stored value.
2. Each replayed node's `canonical_bytes` matches the stored value.
3. Each replayed node's children equal the stored children.
4. The final `chain_head` matches the stored `final_hash`.

## New builtins (~10)

| Name | Args | Returns | Purpose |
|---|---|---|---|
| `abng_set_codebook` | `graph_id, quantiles_2d_tensor, n_bins: i64` | `Void` | install + freeze codebook |
| `abng_codebook_dims` | `graph_id` | `Int` | n_dims (0 if none installed) |
| `abng_codebook_hash` | `graph_id` | `String` (hex) | frozen codebook hash, "" if none |
| `abng_encode_prefix` | `graph_id, x_tensor` | `Tensor[D]` (i64 bytes) | encode input → prefix bytes |
| `abng_add_node` | `graph_id, parent: i64, key_byte: i64` | `Int` (new node_id) | manual add child |
| `abng_node_parent` | `graph_id, node_id` | `Int` | parent (-1 for root) |
| `abng_node_kind` | `graph_id, node_id` | `Int` | children variant code 0..4 |
| `abng_node_child_count` | `graph_id, node_id` | `Int` | live children |
| `abng_node_child` | `graph_id, node_id, key_byte: i64` | `Int` | child id by key byte (-1 if absent) |
| `abng_descend` | `graph_id, prefix_tensor` | `Tensor[2]` | `[matched_prefix, leaf_id]` |
| `abng_route_path` | `graph_id, prefix_tensor` | `Tensor[matched+1]` | full root-to-leaf path |

Total surface after 0.2: **499 dispatch arms** (488 + 11 new).

## Test growth

- Unit tests in `cjc-abng`: 28 → ~50 (add: AdaptiveChildren transitions, codebook freeze, prefix encoding, descend, per-node chain isolation, multi-node replay)
- Integration tests in `tests/abng/`: 44 → ~80 (add: each new builtin, parity for each, multi-node determinism, codebook hashing)

## Determinism contract (unchanged from 0.1)

* All maps `BTreeMap`. All sets `BTreeSet`.
* All summations through Kahan or pairwise.
* All hashes through `cjc_snap::hash::sha256`.
* Codebook bins are `f64::to_bits().to_be_bytes()` for canonical encoding.
* Prefix encoding does **not** use FMA — the binary search compares `value` against `boundary[mid]` with plain `<`.

## Risks

1. **Snapshot break.** v1 → v2 is a clean break, not gradual. Mitigation: documented; 0.1 had no users.
2. **Codebook freeze semantics.** Once frozen, you can't refit on new training data without making a new graph. Mitigation: this is intentional — refit ⇒ different graph_id ⇒ different snapshot, same as model versioning elsewhere.
3. **Promotion ordering.** Promotion happens at insert time, not lazily. Mitigation: makes the audit log linear (no "deferred promotions" hidden state); per-event payload encoding is simple.
4. **Per-node chain divergence.** Phase 0.1 callers might expect node[0].stats_chain_head == graph.chain_head. Mitigation: the equivalence only held in 0.1 because there was one node; documenting the new contract.
