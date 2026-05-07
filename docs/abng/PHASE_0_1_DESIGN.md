# ABNG Phase 0.1 — Arena, Audit, Replay (Design Note)

**Date:** 2026-05-05
**ADR:** [ADR-0022 ABNG Adaptive Belief Radix Graph](../../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0022%20ABNG%20Adaptive%20Belief%20Radix%20Graph.md)
**Scope:** standalone `cjc-abng` crate. No chess wiring, no neural head, no
adaptive children variants. Goal: a *replayable, hash-chained* arena with
`abng_*` builtins reachable from `.cjcl` source.

## Why this slice

The full ABNG v0.3 design (eight roles, ~50 builtins, structural decisions,
per-leaf MLPs) is too big for one PR. Phase 0.1 ships the **substrate**
that everything else assumes: an arena, a tamper-evident event log, a
deterministic replay path, and a serialization format. Once that is
verified bit-identical end-to-end, Phase 0.2 layers in the radix-children
variants (`Node4`/`Node16`/`Node48`/`Node256`/`Dense`) and the structural
decision engine, and Phase 0.3 attaches per-leaf MLPs via existing
`grad_graph_*` builtins.

Concretely Phase 0.1 contains:

* a single root node with running belief stats (Welford μ + Kahan-compensated M2)
* an append-only `AuditEvent` log with SHA-256 chain
* `abng_serialize` / `abng_replay` that round-trip bit-identically
* ~13 `abng_*` builtins exposed to user `.cjcl` source via satellite dispatch

What Phase 0.1 explicitly does **not** include:

* multiple nodes, children, prefix encoding (Phase 0.2)
* per-node MLPs / belief tensors / BLR head (Phase 0.3)
* structural decisions (Grow/Split/Merge/Prune/Compress/Freeze) (Phase 0.2)
* drift detector, OOD score, calibration bins (Phase 0.3+)
* `cjcl abng …` CLI subcommands (Phase 0.4)

## Crate placement

`crates/cjc-abng/` — new workspace member, satellite-dispatch pattern that
mirrors `cjc-ad/src/dispatch.rs` (Phase 3c) and `cjc-quantum/src/dispatch.rs`
(Phase 3a). Routed from both `cjc-eval` and `cjc-mir-exec` *after*
`dispatch_grad_graph`, so `dispatch_builtin` → `dispatch_quantum` →
`dispatch_grad_graph` → `dispatch_abng` is the canonical fall-through chain.

## Determinism contract

* **Hash:** SHA-256 from `cjc_snap::hash::sha256` (zero external deps,
  hand-rolled FIPS 180-4 in the workspace already).
* **Sums:** Kahan-compensated `M2` via `cjc_repro::KahanAccumulatorF64`.
  Welford mean update is the bit-stable recurrence
  `mean += (x − mean) / n`; deterministic for a fixed sample order.
* **RNG:** SplitMix64 from `cjc_repro::Rng`, seeded by graph creation.
  Phase 0.1 doesn't use the RNG yet but threads it for future phases.
* **Maps:** `BTreeMap` only. Arena indexing by `Vec` position.
* **Floats in canonical bytes:** `f64::to_bits().to_be_bytes()` so
  signaling-NaN bit-patterns are preserved for hash equality.

## Type sketch

```rust
pub struct AdaptiveBeliefGraph {
    pub seed: u64,
    pub epoch: u64,
    pub nodes: Vec<AdaptiveBeliefNode>,   // arena, NodeId = index
    pub audit: Vec<AuditEvent>,           // append-only
    pub chain_head: [u8; 32],             // hash of last event
}

pub struct AdaptiveBeliefNode {
    pub node_id: u32,
    pub stats: NodeStats,
    pub stats_version: u64,               // bumped on every update
    pub stats_chain_head: [u8; 32],       // per-node chain (Phase 0.1: equal to global on root)
}

pub struct NodeStats {
    pub n_seen: u64,
    pub mean: f64,
    pub m2: KahanAccumulatorF64,          // Kahan-compensated sum-of-squared-deviations
}

pub struct AuditEvent {
    pub seq: u64,
    pub epoch: u64,
    pub node_id: u32,
    pub kind: AuditKind,
    pub stats_version: u64,
    pub stats_hash: [u8; 32],             // hash of post-update NodeStats
    pub previous_hash: [u8; 32],
    pub new_hash: [u8; 32],               // sha256(previous_hash || canonical_payload)
}

pub enum AuditKind {
    Created,
    BeliefUpdate { value: f64 },
}
```

## Canonical encoding

Bit-stability requires a *fixed* byte order across runs and platforms.

* `NodeStats` (24 bytes): `n_seen` (8 BE) ‖ `mean.to_bits()` (8 BE) ‖
  `m2.finalize().to_bits()` (8 BE).
* `AuditEvent.payload` (variable): `seq` (8 BE) ‖ `epoch` (8 BE) ‖
  `node_id` (4 BE) ‖ kind tag (`0x00`=Created, `0x01`=BeliefUpdate) ‖
  if kind=BeliefUpdate, `value.to_bits()` (8 BE) ‖
  `stats_version` (8 BE) ‖ `stats_hash` (32).
* Chain step: `new_hash = sha256(previous_hash ‖ payload)`.

## Snapshot format (`abng_serialize` output)

```
"ABNG" 0x01                      // 5-byte magic + version
seed:        u64 BE              // 8 bytes
epoch:       u64 BE              // 8 bytes
final_hash:  [u8; 32]            // current chain_head, for end-to-end check
n_nodes:     u32 BE              // 4 bytes (Phase 0.1: always 1)
n_events:    u64 BE              // 8 bytes
events:      [AuditEvent ...]    // n_events × event_record_bytes
```

Each event record stores its full payload + `previous_hash` + `new_hash`.
The replay path *recomputes* `new_hash` from the running chain and asserts
equality with the stored value (defense-in-depth — corruption is detected
even if `previous_hash` was correctly tampered with).

## Builtin surface (Phase 0.1)

All names dispatched via `cjc_abng::dispatch_abng()`:

| Name | Args | Returns | Purpose |
|---|---|---|---|
| `abng_new` | `seed: i64` | `Int` (graph_id) | create new graph |
| `abng_drop` | `graph_id: i64` | `Void` | free a graph |
| `abng_root` | `graph_id: i64` | `Int` | root node id (always 0 in 0.1) |
| `abng_observe` | `graph_id, node_id: i64, value: f64` | `Void` | apply BeliefUpdate, append audit event |
| `abng_observe_batch` | `graph_id, node_id: i64, values: Tensor` | `Void` | sequenced batch of observations |
| `abng_node_count` | `graph_id: i64` | `Int` | number of nodes |
| `abng_node_stats` | `graph_id, node_id: i64` | `Tensor[3]` | `[n_seen, mean, variance]` |
| `abng_node_stats_version` | `graph_id, node_id: i64` | `Int` | per-node stats version |
| `abng_audit_len` | `graph_id: i64` | `Int` | number of audit events |
| `abng_chain_head` | `graph_id: i64` | `String` (hex) | current chain head |
| `abng_verify_chain` | `graph_id: i64` | `Bool` | recompute chain, return true on match |
| `abng_serialize` | `graph_id: i64` | `Bytes` | snapshot blob |
| `abng_replay` | `bytes: Bytes` | `Int` (new graph_id) | replay → new graph; **errors on mismatch** |

## Tests

* **Unit (`tests/abng/unit.rs`)** — Welford correctness against direct
  formulas; bit-identity of `NodeStats::canonical_bytes`; Kahan vs naive
  on long sequences.
* **Audit chain (`tests/abng/audit_chain.rs`)** — chain step formula;
  tampering detection (flip one byte → `verify_chain → false`); empty
  graph chain head is the genesis hash.
* **Replay (`tests/abng/replay.rs`)** — train → serialize → replay →
  byte-identical `chain_head`; replay is independent of the original
  graph instance.
* **Determinism (`tests/abng/determinism.rs`)** — train twice with the
  same seed and same sample sequence → byte-identical serialized
  snapshot.
* **Dispatch (`tests/abng/dispatch.rs`)** — every builtin exercised
  via `cjc_abng::dispatch_abng` with both happy-path and Err-path
  assertions.
* **Parity (`tests/abng/parity.rs`)** — same `.cjcl` source via
  `cjc_eval::Interpreter` and `cjc_mir_exec::run_program_with_executor`
  produces byte-identical `abng_chain_head`.

## Risks (Phase 0.1)

1. **Numerical edge case in Welford with `n=1`:** variance is undefined;
   we return `0.0` and document. ECE/Brier/etc. land in 0.3.
2. **`m2.finalize()` ordering:** Kahan accumulator state depends on
   sample arrival order. Determinism is preserved as long as the
   batch builtin observes elements in tensor order — enforced by the
   dispatch layer (we iterate `to_vec()` left-to-right).
3. **Snapshot growth:** Phase 0.1 stores every event uncompressed.
   Acceptable because event records are 117 bytes; a million events is
   ~117 MB. Compaction lands in Phase 0.4 alongside `cjcl abng`.

## Out-of-scope clarifications

* No `Tensor` of `u8` for hash output; we return `Value::String` with
  hex-encoded chain head. Bytes blobs are returned as `Value::Bytes`.
* No JSON snapshot path in Phase 0.1; `abng_serialize` is the binary
  format only. JSON via `cjcl abng` is Phase 0.4.
* No multi-thread support yet (the arena is `thread_local!`).
* No `verify_nogc` test on the inference path until Phase 0.3 (the
  inference path doesn't exist yet).
