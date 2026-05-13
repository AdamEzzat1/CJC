# v13 → v14 ABNG Snapshot Migration

**Phase:** 0.8c
**Status:** all three v14 wire-format items shipped (A4 + A1 + A2).
**Magic byte:** `ABNG\x0D` (v13) → `ABNG\x0E` (v14).

This doc records what changed in the on-disk and audit-chain layouts
between v13 and v14, what's forward-compatible, and how an existing
v13 snapshot is migrated. It accompanies the three v14 commits on
the `claude/abng-v14-wire-format` branch:

| Item | Commit | What changed | Canary impact |
|---|---|---|---|
| Scaffolding | [`0db8522`](#) | Magic byte bump + dual-acceptance replay (v13 and v14 both readable) | none |
| A4 | [`d2ce894`](#) | Sparse `Node48`/`Node256` snapshot encoding (`count u32 + (byte, child_id)×count` replaces dense `index[256] + slots[]`) | none — Path B (encoding-only) |
| A1 | [`d61a366`](#) | Packed lower-triangular BLR precision (`d(d+1)/2` entries on disk vs `d×d`) | none — Path B (encoding-only; `canonical_bytes` unchanged) |
| A2 | [`457c3c1`](#) + [`eb18f1a`](#) | Fused `AuditKind::TrainStep` (tag `0x1E`) replaces the pre-A2 `BlrUpdated + BeliefUpdate` pair when `Graph::train_step` is called | 6 canaries re-locked (Path A — audit-event content changed) |

## Wire-format changes summary

### Magic bytes
- v13: `ABNG\x0D` (5 bytes)
- v14: `ABNG\x0E` (5 bytes)

Writers always emit v14 from this branch onwards. Readers accept
both: `decode_payload` in `crates/cjc-abng/src/serialize.rs` threads
a `WireVersion` enum from the magic-byte check at archive open
through every per-variant decoder.

### A4 — sparse `AdaptiveChildren::Node48`/`Node256` encoding

**Pre-v14 (dense):**
```
index: [u8; 256]              // child id at each prefix byte (255 = unused)
slots: variable               // child node ids referenced by index
```

**Post-v14 (sparse):**
```
count: u32 BE                 // number of populated entries
pairs: (key u8, child_id u32 BE) × count
```

For a `Node48` with 25 children the v14 layout is 5 + 25×5 = 130
bytes; v13 was 256 + 48×4 = 448 bytes. Saving: ~228 B per node.
`Node256` (256 children max) saves less but still benefits.

No `canonical_bytes`/`state_hash` interaction — the children layout
was never in any hash, so no canary feels this change.

### A1 — packed lower-triangular `BlrState.precision` snapshot

**Pre-v14:** the precision matrix `Λ ∈ ℝ^{d×d}` was written as
`d*d` `f64` entries in row-major order. For `d=16` that's 256 ×
8 = 2 KB per node's BLR state.

**Post-v14:** because Λ is symmetric (NIG posterior), only the
lower-triangular half is written: `d(d+1)/2` `f64` entries in
row-major order. For `d=16` that's 136 × 8 = 1088 B — a saving of
960 B per node.

Reader symmetrizes back on load: the upper triangle is filled from
the lower at `decode_blr_state` time so all downstream code sees
the same `d×d` `Tensor` shape as before.

Critically, `BlrState::canonical_bytes` is unchanged — it still
emits the full `d×d` matrix, so `state_hash` is unchanged. Path B
(encoding-only): every chain-witness that includes a BLR state
hash is bit-identical to its v13 value.

### A2 — fused `AuditKind::TrainStep` (tag `0x1E`)

**Pre-A2 audit sequence for `Graph::train_step(x, phi, y)`:**

```
event N+0  AuditKind::BlrUpdated { state_hash }
event N+1  AuditKind::BeliefUpdate { value: y }
```

(Plus an optional `BlrNumericalRescue` after `BlrUpdated` if the
`b<ε` rescue branch fired.)

**Post-A2 audit sequence:**

```
event N+0  AuditKind::TrainStep { value: y, state_hash }
```

(Plus an optional `BlrNumericalRescue` after `TrainStep` if the
rescue branch fired — same shape as pre-A2.)

**Payload layout** (after the 21-byte common header):

```
[21..29]   value.to_bits()       u64 BE   (8 bytes)
[29..61]   state_hash             [u8; 32] (32 bytes)
```

Full event size: 21 (header) + 40 (body) + 8 (`stats_version`) + 32
(`stats_hash`) = **101 bytes**. The pre-A2 two-event sequence took
93 + 69 = 162 bytes. **38 % reduction per row.**

**Decoder gating.** Tag `0x1E` is gated by `wire == WireVersion::V14`.
A v13 archive containing `0x1E` is rejected as
`DecodeError::UnknownKindTag(0x1E)`.

**Replay semantics:**

* **Welford side** — encoded `value` is reapplied as
  `nodes[node_id].observe(value)` at replay time (same as
  `BeliefUpdate { value }`).
* **BLR side** — witness-only. The post-update BLR state lives in
  the per-node section of the snapshot. The end-of-replay verifier
  loop in `serialize.rs:1881..1907` was extended to recognize
  `TrainStep { state_hash }` as an equivalent witness to
  `BlrUpdated { state_hash }`.

## Forward-compatibility

* **v13 archives stay readable.** Open through `replay(&bytes)` →
  dual-magic dispatch detects `ABNG\x0D`, sets
  `WireVersion::V13`, and threads it through the per-component
  decoders. The 28 SHA-256 canaries in archives produced under v13
  remain verifiable byte-for-byte.
* **Writers always emit v14.** Once a v13 archive is loaded and
  re-serialized through `serialize(&graph)`, it becomes a v14
  archive. There is no opt-in flag — v14 is the default emit path.
* **Mixed-version archives are rejected.** A v13 archive with a
  v14-only audit kind (e.g. `0x1E`) is malformed and decode errors
  out with `UnknownKindTag(0x1E)`. This is hygiene against future
  codepaths that might construct mixed blobs.

## Migration steps for an existing v13 snapshot

```rust
use cjc_abng::serialize::{replay, serialize};
use std::fs;

let bytes = fs::read("my_graph.abng")?;     // v13 magic detected
let graph = replay(&bytes)?;                 // decoded through V13 path
let v14_bytes = serialize(&graph);           // emits v14 magic + layout
fs::write("my_graph_v14.abng", &v14_bytes)?;
```

**No state mutation needed.** The migration is purely encoding —
the in-memory `AdaptiveBeliefGraph` is identical between the load
and re-serialize calls. If the workload was idle during migration
(no `train_step` calls between load and save), the chain head of
the saved v14 archive equals the chain head of the original v13
archive.

If the workload **was** active between load and save (e.g., the
graph trained more rows via `train_step` before being saved), the
new chain steps go through the v14 `TrainStep` path and the
resulting chain head reflects v14 audit events for those new rows.

## Verifying a migration

```rust
let pre = replay(&fs::read("pre.abng")?)?;
let post = replay(&fs::read("post.abng")?)?;

// If no training happened between pre and post, chain heads match.
assert_eq!(pre.chain_head, post.chain_head);

// In either case, both archives independently verify their chains.
pre.verify_chain()?;
post.verify_chain()?;
```

## Canary re-lock — what shifted and why

Six demo workloads call the new fused path post-A2 (commit
[`eb18f1a`](#)). The other 19 canary demos don't go through
`train_step`, so their hex stayed locked at their pre-A2 values.

| File | Canary | Pre-A2 hex (truncated) | Post-A2 hex (truncated) |
|---|---|---|---|
| [test_abng_tabular_gp.rs](../../tests/test_abng_tabular_gp.rs) | `tabular_chain_head_canary_locked` | `cd3f5c…87397e6` | `26ab2b…fef40d6745` |
| [test_abng_tabular_gp_cjcl.rs](../../tests/test_abng_tabular_gp_cjcl.rs) | `tabular_cjcl_chain_head_canary_locked` | `4ffaca…2af09be54` | `6b3374…ae6ea4bd762510fe` |
| [test_abng_pinn_uncertainty.rs](../../tests/test_abng_pinn_uncertainty.rs) | `pinn_chain_head_canary_locked` | `30d333…9c4e468d` | `280fd6…78facf5c0` |
| [test_abng_pinn_uncertainty_cjcl.rs](../../tests/test_abng_pinn_uncertainty_cjcl.rs) | `pinn_cjcl_chain_head_canary_locked` | `e5d6c4…0d684a64a` | `be14b7…f81242f3c766388d` |
| [test_abng_lineage_attestation.rs](../../tests/test_abng_lineage_attestation.rs) | `lineage_chain_head_canary_locked` | `789acc…98f0606c2` | `7892bd…6fd22418fddf81` |
| [test_abng_lineage_attestation_cjcl.rs](../../tests/test_abng_lineage_attestation_cjcl.rs) | `lineage_cjcl_chain_head_canary_locked` | `20f5f9…94fa39b40` | `223906…7e8c706432f5a2250617d7` |

Each canary's comment block records the pre-A2 hex inline for
historical traceability. To recover a v13-era chain head from a
v14-era archive, replay the migration in reverse is not possible
— the v14 audit events for the affected rows fundamentally encode
different bytes than v13 did, so the v13 chain head is a frozen
historical fact, not a derivable one.

### NOT re-locked (would-be candidates that aren't):

* **Independent fingerprint canary.**
  `lineage_dataset_a_fingerprint_canary_locked` hashes dataset
  bytes, not the audit chain. Stayed locked.
* **Batch-pattern demos** (`tabular_scaled`, `pinn_scaled`,
  `lineage_scaled`) keep emitting `BlrUpdated + BeliefUpdateBatch`
  — switching them to N `TrainStep` events would regress log
  size. Their canaries stayed locked.
* **`test_chess_rl_v2_6_abng`** — out of scope; its
  `CHESS_V2_6_CHAIN_HEAD` and `CHESS_V2_6_BLR_STATE_HASH` stayed
  locked. A separate migration decision.

## Audit-log consumers — heads-up

If your code scans the audit log for specific `AuditKind` variants,
note that the per-row training events for `train_step`-using
workloads now arrive as `TrainStep` instead of `BlrUpdated +
BeliefUpdate`. For example, the
`lineage_audit_log_contains_observation_history_in_order` test
previously counted `BeliefUpdate` kinds and asserted the count
equalled the dataset size; under v14 it now filters on `TrainStep`
instead. External consumers (regulators, lineage tools, audit-log
analytics) should add `TrainStep` to their kind-recognition tables.

The `BeliefUpdate` and `BlrUpdated` variants are NOT removed —
they still fire when callers use the unfused `observe(...)` or
`blr_update(...)` APIs directly. v14 only changes what
`Graph::train_step` emits.

## Going forward

* **A3 (Merkle-indexed audit chain)** and **C2 (parallel
  `verify_chain`)** are still planned but not in v14 yet. When
  they land they'll append a trailer block to the v14 archive
  layout; the audit-event encoding above remains stable.
* **No new `Value` enum variants** were introduced. Both
  executors (`cjc-eval` + `cjc-mir-exec`) route through the same
  `g.train_step(...)` Rust API via the `abng_train_step` builtin.
* **Determinism contract is preserved.** Same inputs +
  same seed = bit-identical Welford stats + BLR state. The chain
  head reflects the v14 audit shape but the underlying numerical
  state is independent of wire format.
