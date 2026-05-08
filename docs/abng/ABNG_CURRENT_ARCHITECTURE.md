# ABNG — Current Architecture (post-Phase 0.4, snapshot v11)

**As of:** 2026-05-07
**Crate:** `crates/cjc-abng/`
**Tests:** `tests/abng/` (391 integration) + `crates/cjc-abng/src/*` (252 in-crate) + `tests/prop_tests/abng_decision_props.rs` (4 properties × 256 cases) + `tests/bolero_fuzz/abng_decision_fuzz.rs` (4 fuzz targets) — all passing
**Snapshot magic:** `ABNG\x0B` (v11 — Phase 0.4-extended bumped from `\x0A` to absorb two consolidated v11 changes: DecisionPolicy thresholds 12 + 13 (`ece_stability_max_delta`, `sigma_stability_ratio` — Item A) and graph header `unfreeze_count: u64` (Item B). Phase 0.4 mid-track bumps were `\x08 → \x09` (C-2.3.5 `BlrState.feature_version_hash`) → `\x0A` (B-2.2.7 `DecisionPolicy.drift_unfreeze` + B-2.2.{1,2} per-node Welford state) → `\x0B` (Items A+B).)
**Builtin count:** 73 user-facing `abng_*` arms in `dispatch.rs` (65 from Phase 0.3d; +1 C-2.3.5 `abng_reset_blr`; +1 C-2.3.6 `abng_leaf_set_params_batch`; +1 C-2.3.8 `abng_blr_predict_with_fallback`; +1 C-2.3.12 `abng_force_recapture_expected_epistemic`; +3 Track A `abng_descend_traced`, `abng_predict_snap`, `abng_compact_log`; +1 v11 `abng_unfreeze_count`)
**Audit kinds:** 28 (tags `0x00`..`0x1B` — Phase 0.4 added `0x18 BlrNumericalRescue`, `0x19 LeafParamsUpdatedBatch`, `0x1A StatsSnapshot`, and `0x1B Routed`. Tag `0x1C` is reserved for `ProvenanceStamped` (Phase 0.5).)

This document is a *source-of-truth* handoff after Phase 0.3d. Where the
phase design notes (`PHASE_0_*_DESIGN.md`) and the actual code disagree,
**the code wins** and this document records what the code does. Gaps
between design intent and implementation are listed in §8.

---

## 1. Current Phase and Scope

| Phase | Status | Adds |
|---|---|---|
| 0.1   | DONE | Arena + audit log + SHA-256 chain + replay |
| 0.2   | DONE | Multi-node arena, `AdaptiveChildren` (Node4/16/48/256), prefix codebook + descend routing, per-node stats chain |
| 0.3a  | DONE | Per-node MLP head (Xavier init, `leaf_forward` into ambient `cjc-ad` GradGraph) |
| 0.3b  | DONE | Per-node BLR (Cholesky + NIG conjugate update; `(mean, epi, ale)` predict) |
| 0.3c  | DONE | Density tracker (diag Mahalanobis), 15-bin ECE calibration, drift baseline + score, composite `ood_score` |
| 0.3d-1 | DONE | `Maturity` + `NodeSignature` types (lazy), 2 read-only inspection builtins |
| 0.3d-2 | DONE | `expected_epistemic` per-node capture, calibrated `ood_score` ratio formula (`epi / expected`), audit kind `0x17`, snapshot `v6` |
| 0.3d-3 | DONE | `DecisionPolicy` install, 6 force-* structural mutations, `Dense` children variant (kind `5`), audit kinds `0x10..0x15`, snapshot `v7` |
| 0.3d-4 | DONE | `decide_step` policy engine (6 triggers, fall-through order), `unfreeze` (audit `0x16`), persistent signature-stability state, real `Maturity` flags, auto-capture of `expected_epistemic`, snapshot `v8` |
| 0.3d-5 | DONE | proptest properties + bolero fuzz targets + decoder allocation hardening + this doc |
| 0.4 Track C | DONE | post-0.3d audit fixes (7 items): BLR predict rename, NaN/Inf input validation, replay semantic invariants, BLR clamp audit (0x18), feature_version_hash + reset_blr (snapshot v9), batched leaf-params (0x19), per-leaf → per-node doc rename |
| 0.4 Track B | DONE | trigger refinement (7 items): NIG-aware merge math (`combine`), KL-divergence merge gate, route-entropy grow gate, bootstrap held-out ΔNLL split gate, drift-trip auto-Unfreeze (snapshot v10, 12th threshold), 3-window ECE/σ stability buffers per node, Welford-smoothed `NodeSignature` profiles |
| 0.4 Track A | DONE | `cjcl abng …` CLI surface complete: 5/5 subcommands (`inspect`, `replay`, `diff`, `explain`, `train`). JSON output via `--json` on each. Audit kinds `0x1A StatsSnapshot` (G3.7) and `0x1B Routed` (G3.5) shipped; `0x1C ProvenanceStamped` deferred to Phase 0.5. `train` ships with explicit-flag config; TOML `--config` files defer to Phase 0.5. Track A shipped under snapshot magic `\x0A` (no further bump for the CLI surface itself). |
| 0.4-extended (v11) | DONE | Snapshot magic `\x0A → \x0B`. **Item A:** configurable `Maturity` thresholds — `DecisionPolicy.thresholds[12]` = `ece_stability_max_delta` (replaces compile-time `ECE_STABILITY_MAX_DELTA` const), `[13]` = `sigma_stability_ratio` (replaces `SIGMA_STABILITY_RATIO` const). N_THRESHOLDS 12 → 14, POLICY_BYTES_LEN 96 → 112. `Maturity::from_node_with_policy(node, Some(policy))` reads the configurable thresholds; the legacy `Maturity::from_node(node)` falls back to the compile-time consts. **Item B:** `AdaptiveBeliefGraph.unfreeze_count: u64` field + `abng_unfreeze_count(g) -> Int` builtin — observability for the `Unfreeze` audit kind (manual + drift-trip auto-unfreeze paths). Replay verifies the field matches what `apply_event` accumulates. |
| 0.5   | LATER | Chess-RL retrofit (value head first, then policy head); per-node provenance_stamp_hash + 0x1C ProvenanceStamped (forces v11 → v12); smart-replay using StatsSnapshot to fast-forward; TOML `--config` files for `cjcl abng train`; NodeStats canonical_bytes 24B → 32B (Kahan compensation); also v12. |

ABNG is now a **Bayesian-inspired structurally-adaptive belief graph** with:
- Topology + routing (0.2)
- Neural per-node head (0.3a)
- Per-node calibrated uncertainty (0.3b)
- Per-node OOD/calibration/drift signals (0.3c)
- Lazy + persistent maturity / signature evidence (0.3d-1, 0.3d-4)
- 3-window ECE/σ stability ring buffers per node (0.4 B-2.2.2)
- Welford-smoothed 4-channel `NodeSignature` profiles per node (0.4 B-2.2.1)
- Calibrated OOD ratio with auto-captured training-time σ (0.3d-2, 0.3d-4)
- Frozen-threshold `DecisionPolicy` (12 thresholds) + 6 structural-action audit kinds + drift-trip auto-Unfreeze (0.3d-3, 0.4 B-2.2.7)
- KL-divergence merge gate, route-entropy grow gate, ΔNLL split gate, NIG-aware merge math (0.4 B-2.2.{3,4,5,6})
- One-pass deterministic `decide_step` engine (0.3d-4)
- Property-tested + fuzz-hardened replay/decoder boundary (0.3d-5)
- Per-input NaN/Inf rejection at observe / density / calibration / blr_update boundaries (0.4 C-2.3.2)
- Replay validates seq monotonicity, Created-first, epoch match, and stats_version match (0.4 C-2.3.3)
- BLR feature-version stamping + `abng_reset_blr` recovery (0.4 C-2.3.5)
- A complete tamper-evident audit chain over every state mutation (28 kinds, tags `0x00..0x1B`)
- Bit-deterministic snapshot round-trip across both AST and MIR backends
- `cjcl abng {inspect, replay, diff, explain, train}` CLI surface (0.4 Track A)
- Prediction snapshots (`abng_predict_snap` + `predict_snap` module) for explainable lineage (0.4 Track A G3.5)
- Read-side log-compaction marker (`abng_compact_log` + `0x1A StatsSnapshot`) (0.4 Track A G3.7)

ABNG is now structurally adaptive at runtime: install a policy, observe
evidence, call `decide_step` to fire structural mutations whose actions
are deterministic functions of (Maturity, NodeSignature, current
graph topology). Phase 0.4 is complete — Tracks B + C shipped the
trigger-refinement + audit-fix work; Track A shipped the
`cjcl abng {inspect,replay,diff,explain,train}` CLI surface, the
`abng_predict_snap`/`abng_compact_log`/`abng_descend_traced` builtins,
and the `predict_snap` + `Routed` (`0x1B`) + `StatsSnapshot` (`0x1A`)
audit kinds — all extending snapshot v10 in place. Phase 0.5 picks
up the v11-bump items deferred to consolidate magic bumps.

---

## 2. Core Invariants

These hold at every public API boundary today and **must** continue to
hold across all future phases. Violating any of them is a regression.

### 2.1 Determinism

* **Hashing:** SHA-256 from `cjc_snap::hash::sha256` (zero-deps, FIPS 180-4).
* **Summation:** all f64 sums go through `cjc_repro::KahanAccumulatorF64`
  or `pairwise_sum_f64`. No FMA in any kernel that touches belief state.
* **RNG:** `cjc_repro::Rng` (SplitMix64) seeded explicitly from
  `(graph.seed, …)` derivations.
* **Maps/Sets:** `BTreeMap` / `BTreeSet` only. `HashMap`/`HashSet` are
  banned in canonical paths.
* **Float canonical bytes:** `f64::to_bits().to_be_bytes()` everywhere
  — preserves NaN bit patterns; platform-stable.
* **Welford ordering:** sample arrival order is the canonical update
  order; reordering changes (M2, mean) bit-for-bit.

### 2.2 Audit chain

* Every state mutation appends one `AuditEvent` with
  `new_hash = sha256(previous_hash ‖ canonical_payload)`.
* **Genesis hash** is the constant `sha256(b"ABNG-GENESIS-v1")`,
  exposed as `cjc_abng::genesis_hash()`. It is the `previous_hash`
  fed into the first event in any graph (always a `Created` event at
  `seq = 0`). It is **not** the same as `chain_head` for a freshly-
  created graph — see next bullet.
* **Empty-graph chain head:** `abng_new(seed)` immediately appends a
  `Created` event, so `abng_chain_head` on a graph with no other
  observations returns the **post-Created** hash
  (`sha256(genesis_hash() ‖ Created.canonical_payload)`), **not**
  genesis itself. The "internal genesis state" exists only as the
  `previous_hash` of the `Created` event; it never equals
  `chain_head` at any externally-observable moment.
* `AdaptiveBeliefGraph::verify_chain()` walks the audit log from the
  genesis hash through every event, recomputing each `new_hash` and
  asserting equality with the stored value. Any tamper anywhere
  surfaces as `GraphError::ChainBroken { at_seq }`.

### 2.3 Replay equality

`abng_serialize → abng_replay` is the golden round-trip. Replay
- rebuilds the graph from `Created`/`NodeAdded` events alone (deterministic re-init),
- verifies every event's `new_hash` against the replayed previous-hash chain,
- installs each per-node state blob (params, BLR, density, calibration, drift)
  from the snapshot's per-node section,
- verifies each blob's `state_hash()` against the most-recent
  `*Initialized` / `*Updated` event for that node,
- asserts the final `chain_head` matches the stored `final_hash`.

Any mismatch → `DecodeError::{ChainMismatch, StatsMismatch,
LeafParamsHashMismatch, BlrStateHashMismatch, …}`.

### 2.4 AST↔MIR parity

Every `abng_*` builtin must produce **byte-identical printed output**
when the same `.cjcl` snippet is run through `cjc_eval::Interpreter`
and `cjc_mir_exec::run_program_with_executor`. The ABNG arena is a
per-thread `BTreeMap<i64, AdaptiveBeliefGraph>`; both executors share
the same arena via `cjc_abng::dispatch::dispatch_abng`.

### 2.5 No silent allocations on inference paths

The hot path (`leaf_forward`, `blr_predict`, `density_score`,
`ood_score`, `descend`) does not allocate after warm-up except for
the `GradGraph` parameter clones in `leaf_forward` (documented
trade-off; Phase 0.3d may reuse via `set_tensor`+`reforward`).

### 2.6 Frozen on-disk encoding

* `MAGIC = b"ABNG\x0A"` — bumped on every breaking format change.
  Old magic bytes (`\x01`..`\x09`) are explicitly rejected. Phase 0.3d
  alone bumped through 3 versions (`\x05 → \x06 → \x07 → \x08`); Phase
  0.4 bumped `\x08 → \x09` (Track C-2.3.5: `BlrState.feature_version_hash`)
  then `\x09 → \x0A` (Track B-2.2.7: `DecisionPolicy` 12th threshold
  `drift_unfreeze`). Each transition is a clean break with no
  backward-compatibility path.
* Audit-kind tag bytes `0x00`..`0x19` are **frozen forever**. New
  kinds must allocate fresh tags. Phase 0.3d allocated `0x10..0x17`
  (8 new tags). Phase 0.4 added `0x18 BlrNumericalRescue` (C-2.3.4)
  and `0x19 LeafParamsUpdatedBatch` (C-2.3.6); both are opt-in (`0x18`
  fires only on numerical rescue; `0x19` fires only when callers
  explicitly use the batch builtin). Track C-2.3.5 then bumped magic
  `\x08 → \x09` to absorb `BlrState.feature_version_hash`, and Track
  B-2.2.7 bumped `\x09 → \x0A` to absorb `DecisionPolicy.drift_unfreeze`
  (12th threshold) plus the per-node B-2.2.{1,2} state additions
  (`SignatureWelford × 4` + `ece_history` / `sigma_history` ring
  buffers, ~146 extra bytes per node). Future audit-kind additions in
  Track A (`0x1A..0x1C` — `StatsSnapshot`, `Routed`,
  `ProvenanceStamped`) may extend v10 in place if they only add tag
  bytes (no per-node state) — the planned per-node
  `provenance_stamp_hash` is deferred to Phase 0.5 to keep v10 frozen
  for the full Phase 0.4 lifetime.
  See §3.6 for the full table.
* `ChildrenKind` numeric codes `0..5` are frozen (`5 = Dense` shipped
  with Phase 0.3d-3).
* `Activation` tag bytes `0x00`..`0x08` are frozen.

### 2.7 Defensive decoder allocation (Phase 0.3d-5)

The replay path treats any length field decoded from the blob as
**untrusted** when it could drive a `Vec::with_capacity` allocation.
Specifically:

* `n_nodes`, `n_events`, `n_params`, and `decode_tensor`'s `numel`
  use `Vec::new()` (or are bounded by the cursor's remaining bytes
  before `with_capacity`) so a fuzz-corrupted blob cannot trigger an
  unbounded allocation panic.
* Established by `tests/bolero_fuzz/abng_decision_fuzz.rs::fuzz_abng_tamper_no_panic`.

---

## 3. Frozen API Decisions

These are the surface contracts that user code (and future phases)
depend on.

### 3.1 Multi-graph thread-local arena

```rust
thread_local! { static ARENA: RefCell<Arena> = ... }
struct Arena { graphs: BTreeMap<i64, AdaptiveBeliefGraph>, next_id: i64 }
```

- `abng_new(seed) -> Int` returns a graph id. Multiple graphs coexist.
- `abng_drop(id)` frees a graph.
- `dispatch::reset_arena()` is the per-test entry; tests must call it
  to avoid arena leakage on shared cargo test threads.

### 3.2 One-shot install order

Each of these is **frozen-on-first-call**. The first five **must be
installed before any `add_node`** (i.e. when `n_nodes ≤ 1`); the
decision policy is install-anytime. Re-installation errors with the
corresponding `*AlreadyFrozen` variant.

| Subsystem | Builtin | Depends on | Installable after `add_node`? |
|---|---|---|---|
| Codebook | `abng_set_codebook(g, q2d, n_bins)` | (none) | No |
| Leaf head | `abng_set_leaf_head(g, in, hidden_1d, out, act)` | (none) | No |
| BLR prior | `abng_set_blr_prior(g, λ, a, b)` | leaf head | No |
| Density tracker | `abng_set_density_tracker(g)` | leaf head | No |
| Calibration bins | `abng_set_calibration(g, n_bins ∈ [2,100])` | (none) | No |
| Decision policy | `abng_set_decision_policy(g, thresholds: Tensor[14])` | (none) | **Yes** (install-anytime) |

The leaf head is required by both BLR (which needs penultimate-feature
dim `d`) and density (which uses the same `d`). Calibration is
independent. **Decision policy** is one-shot but install-anytime —
the policy thresholds drive `decide_step` and don't require an empty
graph (Phase 0.3d-3).

### 3.3 Per-node state mounting

When a subsystem is installed, the **root** is back-filled. When
`abng_add_node(parent, key_byte)` runs *after* a subsystem is
installed, the new child is initialized for that subsystem too. The
order of init events on `add_node` is fixed:

```
NodeAdded
[ChildrenPromoted]   if the parent's children variant was promoted
LeafParamsInitialized      if leaf head was installed
BlrInitialized              if BLR prior was installed
DensityTrackerInstalled    if density was enabled
CalibrationInstalled       if calibration n_bins was set
```

### 3.4 Builtin surface (73 arms)

Construction / lifecycle: `abng_new`, `abng_drop`, `abng_root`,
`abng_node_count`, `abng_audit_len`, `abng_chain_head`,
`abng_verify_chain`, `abng_serialize`, `abng_replay`.

Observation / stats: `abng_observe`, `abng_observe_batch`,
`abng_node_stats`, `abng_node_stats_version`,
`abng_node_stats_chain_head`.

Topology: `abng_add_node`, `abng_node_parent`, `abng_node_kind`,
`abng_node_child`, `abng_node_child_count`.

Codebook + routing: `abng_set_codebook`, `abng_codebook_dims`,
`abng_codebook_hash`, `abng_encode_prefix`, `abng_descend`,
`abng_route_path`.

Leaf head: `abng_set_leaf_head`, `abng_leaf_head_dims`,
`abng_leaf_param_count`, `abng_leaf_param`, `abng_leaf_set_param`,
`abng_leaf_set_params_batch` (Phase 0.4 Track C-2.3.6 — emits one
`LeafParamsUpdatedBatch` event for the whole vector instead of `2(L+1)`
`LeafParamsUpdated` events), `abng_leaf_params_hash`, `abng_leaf_forward`.

BLR: `abng_set_blr_prior`, `abng_blr_features`, `abng_blr_update`,
`abng_blr_predict`, `abng_blr_state_hash`, `abng_blr_n_seen`,
`abng_reset_blr` (Phase 0.4 Track C-2.3.5 — clears posterior to prior
and refreshes `feature_version_hash` to current MLP params; recovery
from `BlrError::FeatureVersionStale`),
`abng_blr_predict_with_fallback` (Phase 0.4 Track C-2.3.8 —
read-only parent-walk variant of `abng_blr_predict` that returns the
prediction at the **nearest ancestor (incl. self) with `n_seen ≥ 1`**.
Returns `Tensor[4] = [mean, epistemic_leverage, aleatoric_var,
source_node_id_as_f64]`. Errors with `BlrError::NoEvidence { walked }`
when no ancestor has any observations. No audit event, no RNG —
suitable for use inside `decide_step` if a future trigger needs a
"likely useful prediction here" signal).

Density / calibration / drift / OOD:
`abng_set_density_tracker`, `abng_density_observe`, `abng_density_score`,
`abng_density_n_seen`, `abng_set_calibration`,
`abng_calibration_observe`, `abng_calibration_ece`,
`abng_calibration_n_seen`, `abng_freeze_drift_baseline`,
`abng_drift_score`, `abng_ood_score`.

**Phase 0.3d additions (16 builtins):**

Maturity / signature inspection (0.3d-1):
`abng_node_maturity`, `abng_node_signature`.

Expected epistemic σ capture (0.3d-2 + 0.4 C-2.3.12):
`abng_set_expected_epistemic`, `abng_expected_epistemic`,
`abng_force_recapture_expected_epistemic` (Phase 0.4 Track C-2.3.12 —
overwrites the captured value with a fresh deterministic capture from
the current BLR posterior; required after `abng_reset_blr` to keep
`ood_score`'s calibrated ratio aligned with the post-reset posterior
shape; emits a fresh `0x17 ExpectedEpistemicCaptured` audit event
each call so replay rebuilds the same sequence).

Decision policy + structural mutations (0.3d-3):
`abng_set_decision_policy`, `abng_decision_policy_hash`,
`abng_force_grow`, `abng_force_split`, `abng_force_merge`,
`abng_force_prune`, `abng_force_compress`, `abng_force_freeze`,
`abng_is_frozen`, `abng_action_count`.

Decision engine + unfreeze (0.3d-4):
`abng_decide_step`, `abng_unfreeze`.

### 3.5 Cross-language handle representation

- Graph id, node id, prefix bytes, params count, audit-event seq, etc.
  cross the language boundary as `Value::Int(i64)`.
- Tensors as `Value::Tensor`.
- Hashes as `Value::String` (lowercase 64-char hex).
- Blobs as `Value::Bytes(Rc<RefCell<Vec<u8>>>)`.
- Booleans as `Value::Bool` (e.g. `abng_is_frozen`).
- `leaf_forward` returns `Value::Array([y_idx, p_0, …, p_{n-1}])` so
  the user can pass param indices to `grad_graph_backward_collect`.
- `abng_decide_step` returns `Value::Tensor` of shape `[6]` indexed by
  [`ActionKind`]: `[Grow, Split, Merge, Prune, Compress, Freeze]`.
- `abng_force_split` returns `Value::Tensor` of shape `[2]` carrying
  the two new child node ids.
- `abng_node_maturity` returns `Value::Tensor` of shape `[4]`:
  `[samples_seen, calibration_stable, uncertainty_stable, trust_level]`.
- `abng_node_signature` returns `Value::Bytes` of length 32.
- `abng_expected_epistemic` returns `Value::Float`; the sentinel
  `-1.0` indicates "not captured" (since `set_expected_epistemic`
  rejects non-positive values, `-1.0` cannot be a real reference).

### 3.6 Frozen audit-kind tag table (post-0.3d)

| Tag | Kind | Phase | Payload shape | Witness type |
|---|---|---|---|---|
| `0x00` | `Created` | 0.1 | (header only) | structural |
| `0x01` | `BeliefUpdate` | 0.1 | `value: f64` (8B) | full payload |
| `0x02` | `NodeAdded` | 0.2 | `parent: u32, key_byte: u8` (5B) | full payload |
| `0x03` | `ChildrenPromoted` | 0.2 | `from: u8, to: u8` (2B) | full payload |
| `0x04` | `CodebookFrozen` | 0.2 | `codebook_hash: [u8; 32]` | full payload |
| `0x05` | `LeafHeadConfigured` | 0.3a | `config_hash: [u8; 32]` | full payload |
| `0x06` | `LeafParamsInitialized` | 0.3a | `params_hash: [u8; 32]` | hash witness |
| `0x07` | `LeafParamsUpdated` | 0.3a | `params_hash: [u8; 32]` | hash witness |
| `0x08` | `BlrPriorConfigured` | 0.3b | `config_hash: [u8; 32]` | full payload |
| `0x09` | `BlrInitialized` | 0.3b | `state_hash: [u8; 32]` | hash witness |
| `0x0A` | `BlrUpdated` | 0.3b | `state_hash: [u8; 32]` | hash witness |
| `0x0B` | `DensityTrackerInstalled` | 0.3c | `state_hash: [u8; 32]` | hash witness |
| `0x0C` | `DensityUpdated` | 0.3c | `state_hash: [u8; 32]` | hash witness |
| `0x0D` | `CalibrationInstalled` | 0.3c | `state_hash: [u8; 32]` | hash witness |
| `0x0E` | `CalibrationUpdated` | 0.3c | `state_hash: [u8; 32]` | hash witness |
| `0x0F` | `DriftBaselineFrozen` | 0.3c | `state_hash: [u8; 32]` | hash witness |
| `0x10` | `Grow` | 0.3d-3 | `parent: u32, key: u8, child: u32` (9B) | full payload |
| `0x11` | `Split` | 0.3d-3 | `parent: u32, child_a: u32, child_b: u32` (12B) | full payload |
| `0x12` | `Merge` | 0.3d-3 | `absorbed: u32, into: u32` (8B) | full payload |
| `0x13` | `Prune` | 0.3d-3 | (header `node_id`) | structural |
| `0x14` | `Compress` | 0.3d-3 | `signature: [u8; 32]` | full payload |
| `0x15` | `Freeze` | 0.3d-3 | (header `node_id`) | structural |
| `0x16` | `Unfreeze` | 0.3d-4 | (header `node_id`) | structural |
| `0x17` | `ExpectedEpistemicCaptured` | 0.3d-2 | `state_hash: [u8; 32]` | hash witness |
| `0x18` | `BlrNumericalRescue` | 0.4 (C-2.3.4) | `reason: u8, b_pre_clamp_bits: u64` (9B) | full payload (diagnostic) |
| `0x19` | `LeafParamsUpdatedBatch` | 0.4 (C-2.3.6) | `params_hash: [u8; 32]` | hash witness |
| `0x1A` | `StatsSnapshot` | 0.4 (Track A G3.7) | `node_id: u32, stats_hash: [u8; 32]` (36B) | hash witness (log-compaction marker; replay no-op in 0.4, smart-replay deferred to 0.5) |
| `0x1B` | `Routed` | 0.4 (Track A G3.5) | `leaf: u32, matched_prefix: u8` (5B) | full payload (opt-in trace event from `descend_traced`) |

**Witness vs full-payload rationale:** *hash witness* events keep the
audit log compact under heavy training (`*Updated` events fire per
batch; the actual state lives in the per-node snapshot section).
*Full-payload* events carry the data needed to reconstruct graph
topology during replay — e.g. `Grow` must know the new child id so
the arena re-built event-by-event matches the original index order.

### 3.7 Decision engine — `decide_step` semantics (Phase 0.3d-4)

`AdaptiveBeliefGraph::decide_step()` runs **one structural-decision
pass** with the following deterministic contract:

1. **No-op without policy.** If `decision_policy` is `None`, returns
   `[0; 6]` and does not advance signature stability.
2. **NodeId-ascending iteration.** Iterates over an arena snapshot
   captured at call entry; new nodes from this pass are NOT visited
   in the same call.
3. **Always advances signature stability.** Even for frozen / inactive
   nodes — so the stability counter accumulates while frozen
   (Phase 0.4 drift-trip un-freeze will use the same counter).
4. **Skip frozen / inactive.** No structural action fires on
   `is_frozen == true` or `is_active == false` nodes.
5. **Auto-capture `expected_epistemic`.** Before the trigger ladder:
   if `Maturity.uncertainty_stable && expected_epistemic.is_none() &&
   blr.is_some()`, capture `epistemic_leverage(blr.predict(blr.mean))`
   through the existing `set_expected_epistemic` path. (The captured
   value is leverage, not variance in y-units — see §6.5; the field
   name `expected_epistemic` is preserved for snapshot stability.)
6. **Drift-trip auto-Unfreeze (Phase 0.4 Track B-2.2.7).** Before the
   skip-if-frozen check: if the node is frozen, has both a density
   tracker and a drift baseline, and
   `drift_score(current_density) > drift_unfreeze`, call `unfreeze`
   (emits `0x16 Unfreeze` audit) and let the regular ladder run on
   the now-thawed node. Default `drift_unfreeze = f64::MAX` keeps the
   gate disabled in policies that don't opt in.
7. **Trigger fall-through (at most one fires per node per call):**
   1. Compress — children present + all child signatures within
      `tau_compress` Hamming of node's signature
   2. Merge — sibling with smaller `NodeId` whose signature is within
      `tau_merge` Hamming **AND** posterior `KL ≤ kl_merge`
      (Phase 0.4 Track B-2.2.3) — the `combine` math runs only if both
      gates pass (B-2.2.6)
   3. Split — leaf + `samples_seen ≥ split_min` **AND** bootstrap
      held-out ΔNLL gain ≥ `nll_split_gain` (Phase 0.4 Track B-2.2.4)
   4. Prune — `samples_seen < prune_floor` AND
      `signature_stable_calls ≥ prune_grace_epochs` (root never pruned)
   5. Grow — leaf + `samples_seen ≥ grow_min` + deterministic-from-
      `(seed, node_id)` key byte not bound **AND** route-key entropy at
      candidate depth > `H_grow` (Phase 0.4 Track B-2.2.5)
   6. Freeze — `signature_stable_calls ≥ freeze_after`, where the
      "stability" signal is now Welford-smoothed across observations
      (Phase 0.4 Track B-2.2.1) and ECE / σ stability are gated on
      a 3-window ring buffer (Phase 0.4 Track B-2.2.2)
8. **Returns `[u64; 6]`** indexed by [`ActionKind`]:
   `[Grow, Split, Merge, Prune, Compress, Freeze]`. (`Unfreeze` does
   not bump action_counts — see §7 #13.)

**Phase 0.4 trigger refinements (DONE).** The Track B work refined
every gate into its prompt-spec form: Welford-smoothed signatures
(B-2.2.1), 3-window ECE/σ stability (B-2.2.2), KL-merge gate
(B-2.2.3), ΔNLL split gate (B-2.2.4), route-entropy grow gate
(B-2.2.5), NIG-aware merge math (B-2.2.6), drift-trip auto-Unfreeze
(B-2.2.7). The pre-0.4 single-threshold simplifications are gone;
all gates are now defensible against the original prompt spec.
Compare the inline `Phase 0.4 will…` markers in
`crates/cjc-abng/src/graph.rs` — they have all flipped to
`// Phase 0.4 Track B-2.2.x: …`.

The `Value` enum layout has not been extended for ABNG. HARD RULE #1.

---

## 4. Open Risks

### R1. ECE on small bins
Bins with `< 5` samples produce noisy ECE. Phase 0.3c exposes raw ECE
without a maturity gate. **Mitigation:** Phase 0.3d's
`Maturity.calibration_stable` will require minimum bin populations
before declaring stable.

### R2. OOD `epistemic_z` is a placeholder ✅ RESOLVED in Phase 0.3d
Phase 0.3d-2 added `expected_epistemic: Option<f64>` per node, the
`set_expected_epistemic` install path, and the calibrated formula
`(epi / expected).clamp(0, 1)` (falls back to raw clamp when not
captured). Phase 0.3d-4 added auto-capture inside `decide_step` at
`Maturity.uncertainty_stable` first holding.

### R3. Diagonal-only Mahalanobis
Off-axis correlation in features isn't captured. Acceptable for
post-tanh penultimate features (approximately diagonal); will need
revisiting if a specific failure mode shows up in chess-RL.

### R4. `LeafParamsUpdated` / `BlrUpdated` / `DensityUpdated` /
`CalibrationUpdated` event volume
Every batch update fires a 32-byte witness event. A million SGD steps
on a 10-leaf graph emits 10M events ≈ 320 MB audit log. Witness-only
(no payload) keeps the cost bounded but log compaction is Phase 0.4.

### R5. Snapshot break each phase ⚠️ DEFERRED to Phase 0.4 Track A
v1 → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9 (C-2.3.5) → v10 (B-2.2.7).
Phase 0.3d alone bumped through 3 versions; Phase 0.4 added 2 more
(C-2.3.5 added `BlrState.feature_version_hash`, B-2.2.7 added
`DecisionPolicy.drift_unfreeze` 12th threshold and absorbed B-2.2.{1,2}
per-node Welford + ring-buffer state in-place). The original "single
bump per phase" goal slipped because B-2.2.{1,2} were resequenced
*after* B-2.2.7, but the second bump did absorb both Stage B state
additions in-place — so v10 is final for Phase 0.4 Track A. The CLI
must NOT bump magic again — audit tags `0x1A..0x1C` extend v10 in
place; per-node `provenance_stamp_hash` is deferred to Phase 0.5.

### R6. `leaf_forward` clones every param into the GradGraph each call
Fine for small heads. For large heads or PINN-style high-frequency
forward, switch to `set_tensor`+`reforward`. Phase 0.3d optimization.

### R7. Drift baseline freeze policy is user-driven ✅ RESOLVED in Phase 0.4 Track B-2.2.7
Phase 0.3c only provides the primitive. Phase 0.3d-4 shipped
`abng_unfreeze` (manual). Phase 0.4 Track B-2.2.7 wired the
**drift-trip auto-unfreeze** path inside `decide_step`: a 12th
`DecisionPolicy.drift_unfreeze` threshold (snapshot v10) gates an
`unfreeze` call when a frozen node's `drift_score(current_density) >
drift_unfreeze`. Default `f64::MAX` keeps the gate disabled in
policies that don't opt in. See §3.7 step 6 and §8.11.

### R8. `Maturity` not yet plumbed ✅ RESOLVED in Phase 0.3d
Phase 0.3d-1 shipped lazy `Maturity { samples_seen,
calibration_stable, uncertainty_stable, trust_level }` and
`NodeSignature` (4 × 8B profile hashes). Phase 0.3d-4 promoted both
to participate in `decide_step`'s triggers — `calibration_stable`
flips on `ECE < 0.05`, `uncertainty_stable` requires BLR + samples ≥
100 + signature-stable for ≥ 1 decide_step call.

### R9. Decision-engine simplifications ✅ RESOLVED in Phase 0.4 Track B
The `decide_step` engine pre-0.4 shipped with deliberately simplified
triggers. Phase 0.4 Track B refined every gate into its prompt-spec
form across 7 items: Welford-smoothed signatures (B-2.2.1), 3-window
ECE/σ stability (B-2.2.2), KL-divergence merge (B-2.2.3), bootstrap
held-out ΔNLL split (B-2.2.4), route-entropy grow (B-2.2.5), real
NIG-aware merge math via `combine` (B-2.2.6), and drift-trip
auto-Unfreeze (B-2.2.7). The *event channel* was full strength
since 0.3d-4; the *quality* of the signal now matches the original
prompt spec. See §3.7 and §8.10.

### R10. BLR `predict()` returns dimensionless leverage, not variance contribution ✅ RESOLVED in Phase 0.4 Track C-2.3.1
**Surfaced 2026-05-07 retrospective; resolved 2026-05-07 in C-2.3.1.**
`BlrState::predict()` returns `(mean, epistemic_leverage,
aleatoric_var)` where `epistemic_leverage = ‖L⁻¹φ‖² = φᵀΛ⁻¹φ` is
**dimensionless leverage**, not variance in output units. Pre-0.4 docs
called the middle slot `epistemic_var`, which was misleading because
units of variance would require multiplying by `aleatoric_var`. The
0.3b design note's predictive-variance formula `total = aleatoric_var
× (1 + epistemic_leverage)` treats it as leverage internally — and the
API now matches. 0.3d-2's `expected_epistemic` capture and 0.3d-4's
auto-capture store leverage (the field name is preserved for snapshot
stability), so the calibrated OOD ratio `(lev / expected).clamp(0, 1)`
works on its own terms (units cancel). C-2.3.1 was a rename + doc
update only — no math change, no snapshot bump.

### R11. `observe()` accepts NaN/Inf — no input validation (post-0.3d audit)
**Surfaced 2026-05-07 retrospective.** `AdaptiveBeliefGraph::observe()`,
`density_observe()`, `calibration_observe()`, and `blr_update()`
accept non-finite f64 inputs without validation. A single
`observe(node, f64::NAN)` poisons the Welford state forever, but
replay still passes (bytes are bit-identical). **Phase 0.4 Track C-2.3.2**
— reject non-finite values at every input boundary.

### R12. Replay missing semantic invariant checks (post-0.3d audit)
**Surfaced 2026-05-07 retrospective.** `replay()` validates the hash
chain but does NOT validate seq monotonicity, "Created event must
be first," `event.epoch == header.epoch`, or
`event.stats_version == post-apply node.stats_version`. An
adversarial blob with consistent hashes but reordered seqs / missing
Created / forged epochs currently passes replay silently.
**Phase 0.4 Track C-2.3.3** — add four new `DecodeError` variants
and the corresponding checks.

### R13. Silent `b < 0` clamp in BLR ✅ RESOLVED in Phase 0.4 Track C-2.3.4
**Surfaced 2026-05-07 retrospective; resolved 2026-05-07 in C-2.3.4.**
`BlrState::update` still clamps `b_new < f64::EPSILON` to
`f64::EPSILON` (the IG posterior must stay well-defined), but the
event is no longer silent: the function now returns
`Ok(Some(b_pre_clamp))` and the graph layer's `blr_update` appends a
`BlrNumericalRescue { reason: 0x00, b_pre_clamp_bits }` audit event
immediately after the corresponding `BlrUpdated`. Determinism is
preserved (the clamped state is bit-identical to pre-C-2.3.4); the
new event is purely diagnostic and `apply_event` is a no-op for it
during replay. Existing snapshots from healthy training contain no
0x18 events and replay byte-identically.

### R14. MLP feature space drift after BLR install ✅ RESOLVED in Phase 0.4 Track C-2.3.5
**Surfaced 2026-05-07 retrospective; resolved 2026-05-07 in C-2.3.5
(snapshot v8 → v9).** `BlrState` now carries
`feature_version_hash: [u8; 32]`, stamped from the per-node MLP
params hash at install (`set_blr_prior`, `add_node`, `force_grow`,
`force_split`) and on every reset. `blr_update` rejects with
`BlrError::FeatureVersionStale { stored, current }` when current
params hash differs. Recovery: `abng_reset_blr(node_id)` clears the
posterior to prior and refreshes `feature_version_hash` to current
MLP. The replay path's `BlrInitialized` apply_event was extended to
reset live state + refresh fvh on every fire (handles install AND
reset cases uniformly). `apply_event` is no-op for `BlrUpdated` as
before; the post-replay per-node hash matcher already recognizes the
extended canonical bytes.

### R15. `NodeStats::canonical_bytes` future-hostile for compaction (post-0.3d audit)
**Surfaced 2026-05-07 retrospective.** Canonical bytes serialize
only `m2.finalize()`, dropping the Kahan compensation register.
Replay-from-events is fine because the compensation rebuilds. But
Phase 0.4's planned log compaction needs to resume from canonical
bytes — that requires the compensation state too. **Phase 0.4
Track C-2.3.9** — extend canonical bytes 24B → 32B (snapshot v9).

### R16. Per-leaf vs per-node naming drift ✅ RESOLVED in Phase 0.4 Track C-2.3.7
**Surfaced 2026-05-07 retrospective; resolved 2026-05-07 in C-2.3.7.**
Pre-0.4 the architecture doc and design notes called MLP / BLR heads
"per-leaf" — but `init_params` and BLR init are called for *every*
node (root + every `add_node` / `force_grow` / `force_split`). Code
was always per-node; docs were wrong. C-2.3.7 shipped a doc-only
rename: "per-leaf" → "per-node" across this doc, the Phase 0.3a/0.3b
design notes, and the crate-level docs in `leaf_head.rs`, `blr.rs`,
`graph.rs`, `node.rs`, `audit.rs`, `serialize.rs`, `dispatch.rs`,
and `lib.rs`. No semantics changed; no code changes required.

---

## 5. Phase Roadmap

```
0.3d (DONE 2026-05-07)
  Structural decisions over 0.3a/b/c evidence — shipped across 5 sub-steps:
    0.3d-1  Maturity + NodeSignature (lazy types, 2 inspection builtins)
    0.3d-2  expected_epistemic capture + calibrated OOD (snapshot v6,
            audit 0x17, 2 builtins)
    0.3d-3  DecisionPolicy + 6 force-* + Dense children variant
            (snapshot v7, audit 0x10..0x15, 10 builtins)
    0.3d-4  decide_step engine + Unfreeze + persistent stability
            (snapshot v8, audit 0x16, 2 builtins)
    0.3d-5  proptest + bolero + decoder hardening + this doc

0.4 Track C (DONE 2026-05-07) — post-0.3d audit fixes (7 items)
  - C-2.3.1 BLR predict() rename → epistemic_leverage (no math change)
  - C-2.3.2 NaN/Inf input validation at observe / density / calibration / blr_update
  - C-2.3.3 Replay semantic invariants — seq monotonic, Created-first,
            epoch match, stats_version match (4 new DecodeError variants)
  - C-2.3.4 BLR b<ε clamp audit event (0x18 BlrNumericalRescue)
  - C-2.3.5 BlrState.feature_version_hash + abng_reset_blr (snapshot v9)
  - C-2.3.6 abng_leaf_set_params_batch (single 0x19 event for 2(L+1) params)
  - C-2.3.7 per-leaf → per-node naming rename (doc only)

0.4 Track B (DONE 2026-05-07) — trigger refinement (7 items)
  - B-2.2.6 NIG-aware merge math: BlrState::combine + NodeStats::combine
  - B-2.2.3 KL-divergence gate for Merge (BlrState::kl_divergence)
  - B-2.2.5 Route-entropy gate for Grow (route_key_entropy_at_candidate_depth)
  - B-2.2.4 Bootstrap held-out ΔNLL gain for Split (synthetic Gaussian sampling)
  - B-2.2.7 Drift-trip auto-Unfreeze + DecisionPolicy.drift_unfreeze 12th
            threshold (snapshot v10, DecisionPolicy 88B → 96B)
  - B-2.2.2 3-window ECE/σ stability ring buffers per node (per-node +50B)
  - B-2.2.1 Welford-smoothed NodeSignature profiles per node (per-node +96B)

0.4 Track A (DONE 2026-05-07) — CLI + JSON view + log compaction
  - cjcl abng {inspect,replay,diff,explain,train} CLI (5/5 shipped)
  - JSON output via --json on every subcommand (hand-rolled, no
    external dep)
  - Log compaction marker (0x1A StatsSnapshot + abng_compact_log
    builtin); smart-replay deferred to Phase 0.5
  - Routed audit kind (0x1B) + descend_traced + abng_predict_snap
    builtin + predict_snap module
  - Audit tags 0x1A and 0x1B allocated; 0x1C reserved for
    Phase 0.5 ProvenanceStamped
  - cjcl abng train ships with explicit-flag config; TOML --config
    deferred to Phase 0.5

0.5
  - Per-node provenance_stamp_hash + 0x1C ProvenanceStamped audit kind
    (would force snapshot magic v10 → v11)
  - Configurable Maturity constants (ECE_STABILITY_MAX_DELTA,
    SIGMA_STABILITY_RATIO promoted to DecisionPolicy thresholds 13 + 14
    — also forces v10 → v11; deferred per architecture-doc §8.23 to
    consolidate magic bumps)
  - unfreeze_count observability (extend action_counts to [u64; 7]
    OR add separate field — also forces v10 → v11; same consolidation
    rationale)
  - TOML config files for cjcl abng train
  - Smart-replay using StatsSnapshot to fast-forward past *Updated
    runs (the read half of log compaction)
  - NodeStats canonical_bytes 24B → 32B (Kahan compensation register
    for full compaction support — also forces v10 → v11)
  - Chess-RL retrofit: replace value head with ABNG, then policy head
  - Uncertainty-gated bootstrap in A2C/PPO
  - End-to-end determinism gate against existing chess-rl-v2 weight hashes
```

---

## 6. Key Contracts

### 6.1 Replay contract

Public API: `abng_replay(bytes) -> Int` (graph_id) or `Err(DecodeError)`.

Replay accepts bytes that begin with `MAGIC == b"ABNG\x08"`. Any
deviation in:
- magic bytes
- header layout (codebook → leaf head → BLR prior → density flag →
  calibration n_bins → decision policy → action_counts)
- per-node section (parent, children, stats, params, BLR, density,
  calibration, drift, expected_epistemic, is_frozen, is_active,
  last_signature, signature_stable_calls)
- per-event payload bytes
- per-event `previous_hash` chain link
- per-event `new_hash` recomputed value
- per-node `canonical_bytes()` after replay
- per-node `state_hash()` of installed subsystem blobs vs most-recent witness event
- replayed `action_counts` vs stored `action_counts`
- final `chain_head` vs stored `final_hash`
- `decision_policy.policy_hash` vs recomputed hash

…produces a **specific** `DecodeError` variant. There is no fallback,
no "best effort" replay. Replay is bit-equality or it errors. Phase
0.3d-5 also added defensive bounds-checking on length fields to
prevent fuzz-driven allocation panics (see §2.7).

### 6.2 Hashes / canonical bytes

Each frozen subsystem has a **canonical byte encoding** used for
`state_hash`. The encoding is part of the on-disk contract and **must
not change** without a version bump.

| Subsystem | Function | Layout |
|---|---|---|
| `NodeStats` | `canonical_bytes() -> [u8; 24]` | n_seen u64 BE ‖ mean f64-bits BE ‖ M2 f64-bits BE |
| `AuditEvent` | `payload_bytes() -> Vec<u8>` | seq u64 ‖ epoch u64 ‖ node_id u32 ‖ kind u8 ‖ kind-payload ‖ stats_version u64 ‖ stats_hash 32B |
| `QuantileCodebook` | canonical bytes (codebook.rs) | n_dims u8 ‖ n_bins u16 ‖ ∀d: n_boundaries u16 ‖ bins f64×k |
| `LeafHead` | `canonical_bytes() -> Vec<u8>` | input_dim u32 ‖ output_dim u32 ‖ activation u8 ‖ n_hidden u16 ‖ hidden_dims u32×N |
| `BlrPrior` | `canonical_bytes() -> [u8; 24]` | precision f64 ‖ a f64 ‖ b f64 |
| `BlrState` | `canonical_bytes()` | d u32 ‖ mean f64×d ‖ precision f64×d² ‖ a f64 ‖ b f64 ‖ n_seen u64 ‖ feature_version_hash 32B (v9; Track C-2.3.5) |
| `DensityTracker` | `canonical_bytes()` | d u32 ‖ n u64 ‖ mean f64×d ‖ M2 f64×d (76 bytes for d=4) |
| `CalibrationBins` | `canonical_bytes()` | n_bins u8 ‖ counts u32×n ‖ correct u32×n ‖ conf_sum_bits u64×n |
| `DriftBaseline` | `canonical_bytes()` | d u32 ‖ n_at_freeze u64 ‖ mean f64×d ‖ std f64×d ‖ frozen_hash 32B |
| `Maturity` | `canonical_bytes() -> [u8; 11]` | samples_seen u64 BE ‖ cal_stable u8 ‖ unc_stable u8 ‖ trust_level u8 |
| `NodeSignature` | `canonical_bytes() -> [u8; 32]` | prediction[8] ‖ uncertainty[8] ‖ calibration[8] ‖ routing[8] |
| `DecisionPolicy` | `canonical_bytes() -> [u8; 88]` | thresholds: f64 BE × 11 (canonical order — see `policy.rs` module docs) |

All multi-byte integers and f64 bit patterns are **big-endian**.

### 6.3 Codebook

- 1-D quantile codebook per feature dim. Frozen at `set_codebook`.
- `frozen_hash` embedded in `CodebookFrozen` audit + snapshot header.
- `encode_prefix(x: f64×D) -> [u8; D]` is bit-deterministic.
- `descend(prefix) -> RouteEvidence` walks the trie matching one byte
  per hop; on miss, descent stops and the current node is the
  effective leaf.

### 6.4 Leaf head / MLP

- Architecture: `[in] → [h_1, …, h_L] → [out]`, with `Activation`
  applied between layers; final layer linear (`Activation::None`).
- Per-node `params: Vec<Tensor>` is `[W_1, b_1, …, W_out, b_out]` with
  shapes determined by `LeafHead::layer_shape(layer_idx)`.
- **Xavier-uniform** init from
  `seed_local = splitmix64(graph.seed) ⊕ splitmix64(node_id) ⊕ splitmix64(layer_idx<<1 | kind_bit)`.
  Three independent SplitMix64 streams XORed avoids low-order
  correlation across adjacent node ids.
- `leaf_forward(node_id, x_idx)` registers each `params[k]` as a
  parameter on the **ambient `cjc_ad::GradGraph`** via
  `cjc_ad::dispatch::with_ambient`, builds the MLP layer-by-layer with
  `g.mlp_layer(...)`, and returns `(y_idx, param_indices)` for the
  caller to pass to `grad_graph_backward_collect`.
- After the user runs Adam (or any optimizer), they write back via
  `leaf_set_param(node_id, k, t)`. Each writeback emits one
  `LeafParamsUpdated` event with the post-update params hash witness.

### 6.5 BLR (Bayesian linear regression)

- Per-node `BlrState { d, mean, precision (d×d), a, b, n_seen,
  feature_version_hash: [u8; 32] }` (snapshot v9; the 32-byte hash
  was added by Phase 0.4 Track C-2.3.5 — see §6.5 "Feature-version
  contract" below).
- **NIG conjugate update** on `(features [n,d], y [n])`:
  - `Λ_new = Λ + Xᵀ X`
  - `m_new = Λ_new⁻¹ (Λ μ + Xᵀ y)`
  - `a_new = a + n/2`
  - `b_new = b + 0.5 (yᵀy + μᵀΛμ − m_newᵀ Λ_new m_new)`
- **Numerical rescue (Phase 0.4 Track C-2.3.4):** if `b_new <
  f64::EPSILON` the value is clamped to `f64::EPSILON` to keep the
  IG posterior well-defined. `BlrState::update` returns
  `Result<Option<f64>, BlrError>` — `Ok(Some(b_pre_clamp))` when the
  clamp fired. The graph layer's `blr_update` then appends a
  `BlrNumericalRescue { reason: 0x00, b_pre_clamp_bits }` audit event
  (tag `0x18`) immediately after the corresponding `BlrUpdated`.
  The clamped state is bit-identical with or without observability,
  so determinism is preserved across runs whether or not callers
  inspect the returned `Option<f64>`.
- Λ_new⁻¹φ is computed via **hand-rolled Kahan-compensated Cholesky**
  + triangular solves. No external linear-algebra dep. No FMA.
- **No silent regularization on Cholesky.** Any non-positive pivot
  errors with `BlrError::NonPositiveDefinite`. The earlier 0.3b
  design note claimed an `f64::EPSILON` diagonal regularization
  would be applied before decomposition; this was never shipped, and
  Phase 0.4 Track C-2.3.10 corrected the design note to match the
  code. The hard-error contract is intentional: silent regularization
  would mask corrupt state and break bit-determinism. With the NIG
  update `Λ_new = Λ + XᵀX`, positive-definiteness is preserved by
  construction whenever `Λ` is PD — a non-PD pivot signals a corrupt
  state, not a numerically-tight one.
- `predict(phi) -> (mean, epistemic_leverage, aleatoric_var)`
  (post-Phase 0.4 Track C-2.3.1 — pre-0.4 docs called the middle slot
  `epistemic_var`, which was misleading; see R10):
  - `mean = μᵀφ` — posterior mean prediction.
  - `epistemic_leverage = ‖L⁻¹φ‖² = φᵀΛ⁻¹φ` — **dimensionless
    leverage**, NOT variance in y-units. Decreases monotonically with
    evidence.
  - `aleatoric_var = b/(a−1)` if `a > 1` else `+∞` (i.e. unbounded
    aleatoric pre-update). In y² units.
  - To recover predictive variance in y-units a caller multiplies:
    `total_var = aleatoric_var × (1 + epistemic_leverage)`. 0.3d-2's
    `expected_epistemic` field stores leverage too — its name is
    preserved for snapshot stability; the `(lev / expected)` ratio
    used in `ood_score` is a unit-cancelling division so the
    semantics are unaffected.
- Numerically validated: `y = 2x₁ + 3x₂` recovered to <0.01 from 200
  deterministic samples; epistemic leverage monotonically decreases
  with evidence (see in-crate test `epistemic_leverage_decreases_with_evidence`).

### 6.6 OOD / calibration / drift

**`ood_score(node, phi, matched_prefix, prefix_max)` (post-Phase 0.3d-2):**
```
density_score    = node.density.density_score(phi)         else 0.0
prefix_distance  = (prefix_max − matched_prefix) / prefix_max  else 0.0
epistemic_z      = if node.expected_epistemic.is_some():
                      (epi / expected).clamp(0, 1)
                   else:
                      epi.clamp(0, 1)                       else 0.0
OOD              = max(density_score, prefix_distance, epistemic_z)
```
`max` is deliberate — any single strong signal triggers high OOD.
The calibrated ratio formula activates per-node once
`expected_epistemic` is captured (manual via
`abng_set_expected_epistemic` or auto via `decide_step` at
`Maturity.uncertainty_stable` first holding).

**Calibration ECE:**
```
ECE = Σ_b (counts[b] / N) · |correct[b]/counts[b] − conf_sum[b]/counts[b]|
```
Empty bins contribute 0. Kahan-accumulated.

**Drift score:** L2-normalised z-shift of current density mean vs
frozen baseline mean, divided by baseline std (with `1e-12` floor
on σ to bound the divide).

### 6.7 Lineage

Every `prediction.snap` design intent — provenance pointers from a
prediction back to (model_hash, dataset_hash, codebook_hash,
feature_transform_hash) — is **partially** wired:

- `chain_head` itself is the model fingerprint (covers all installs
  + all training events).
- `codebook_hash` (and `LeafHead.config_hash`, `BlrPrior.config_hash`)
  are exposed as builtins.
- Dataset / feature-transform hash slots exist in design but **are not
  yet plumbed** through the audit log. Phase 0.4 Track A
  (`cjcl abng explain`) is when this gets formalized.

### 6.8 Maturity / NodeSignature stability state (Phase 0.4 B-2.2.{1,2})

Each node carries persistent stability state used by `decide_step`
to gate `Maturity.calibration_stable` / `Maturity.uncertainty_stable`
flags and `signature_stable_calls` increments.

**3-window ECE/σ ring buffers (B-2.2.2, per-node +50B):**

```rust
pub ece_history:    [f64; 3],   // most-recent 3 ECE samples
pub ece_fill_count: u8,         // 0..3 (saturating)
pub sigma_history:  [f64; 3],   // most-recent 3 σ (epistemic-leverage) samples
pub sigma_fill_count: u8,       // 0..3 (saturating)
```

`Maturity.calibration_stable` flips on iff `ece_fill_count == 3`
AND `max(ece_history) - min(ece_history) ≤ ECE_STABILITY_MAX_DELTA`.
`Maturity.uncertainty_stable` flips on iff `sigma_fill_count == 3`
AND `max(sigma_history) / min(sigma_history) ≤ SIGMA_STABILITY_RATIO`.
The constants live in `maturity.rs` and are intentionally compile-time
(see §8.23 — Phase 0.5 may promote to `DecisionPolicy` thresholds).

**4-channel Welford signatures (B-2.2.1, per-node +96B = 4 × 24B):**

```rust
pub welford_prediction:   SignatureWelford,  // posterior mean projection
pub welford_uncertainty:  SignatureWelford,  // epistemic leverage
pub welford_calibration:  SignatureWelford,  // running ECE
pub welford_routing:      SignatureWelford,  // route_key signal
```

`SignatureWelford { n: u64, mean: f64, m2: f64 }` — 24 bytes per
channel, canonical f64 bit pattern. `NodeSignature::from_node` reads
the four Welford means (canonical bits) and packs them into a
`[u8; 32]` (8B per channel) instead of the pre-0.4 sha256-truncate of
subsystem state. The Welford-folded summaries change *gradually* with
observations, so `signature_stable_calls` increments are now *lenient*
rather than *strict*; one new sample no longer resets stability to
zero unless the running mean shifts beyond Hamming sensitivity.

The `routing_observation_value` helper in `signature.rs` projects the
descend trace into a scalar so routing changes participate in the
Welford fold. All four channels are advanced inside `decide_step` per
the contract in §3.7 step 3 ("Always advances signature stability").

---

## 7. Do Not Change Assumptions

These are the assumptions other parts of the codebase rely on. Breaking
any of them silently is a regression even if all tests pass — because
some are about *what's not tested yet*.

1. **`Value` enum layout is unchanged.** Every ABNG handle is
   `Value::Int(i64)`, `Value::Tensor`, `Value::String`, `Value::Bytes`,
   `Value::Array`, `Value::Bool`, or `Value::Float`. No new `Value`
   variant.
2. **`MAGIC` is `b"ABNG\x0A"` and is the *only* accepted magic.**
   Phase 0.4 already bumped through `\x09` (C-2.3.5) and `\x0A`
   (B-2.2.7); Track A must NOT bump again — audit tags `0x1A..0x1C`
   extend v10 in place. A future field that needs a bump goes to
   `\x0B` in Phase 0.5. Never adds a fallback for older versions.
3. **Audit-kind tag bytes `0x00..0x19` keep their current semantics.**
   Track A's new kinds use `0x1A..` only. **Never** re-number.
4. **`ChildrenKind` codes `0..5` and `Activation` codes `0x00..0x08`
   are frozen.** Code `5 = Dense` was added in 0.3d-3 and is now
   permanent.
5. **All `set_*` install builtins are one-shot.** The first five
   require `n_nodes ≤ 1` (`*NotEmptyGraph` if late);
   `set_decision_policy` is install-anytime. Subsequent calls error
   with `*AlreadyFrozen`.
6. **`AdaptiveBeliefGraph` field order is the canonical-bytes order
   for snapshots.** Adding a field means bumping the snapshot magic
   and updating the encoder/decoder symmetrically.
7. **Welford updates apply in slice / observation order.** No
   reordering, no parallel folds (Phase 0.4 may add parallelism,
   must use binned accumulators if so).
8. **`with_ambient` from `cjc_ad::dispatch` is the *only* GradGraph
   entry point.** Constructing a fresh `GradGraph` inside ABNG breaks
   AST↔MIR parity — both executors share one ambient graph per thread.
9. **Per-node-event `node_id` is the arena index after creation.**
   Replay relies on `event.node_id == graph.nodes.len()` at
   `NodeAdded` / `Grow` / `Split` time. Don't reorder pushes.
   `force_prune` and `force_merge` mark nodes inactive but **never
   shrink** the arena.
10. **Drift baseline freeze requires `density.n ≥ 2`.** Earlier freezes
    error with `DriftError::InsufficientEvidence`. This is what makes
    `drift_score` always finite.
11. **`decide_step` iteration order is contract.** Iterates nodes in
    `NodeId` ascending order over an arena snapshot taken at call
    entry. Trigger fall-through is fixed (Compress → Merge → Split →
    Prune → Grow → Freeze, at most one per node per call). Replay
    determinism depends on this exact order.
12. **`force_compress` orphans descendants in the arena.** The Dense
    children variant replaces the routable container; descendants
    persist (per #9) but become unreachable through `descend`. They
    are NOT pruned automatically.
13. **`Unfreeze` does not bump `action_counts`.** The counter tracks
    structural mutations, not state-flag flips. `Freeze` increments;
    `Unfreeze` does not.

---

## 8. Still Undecided / Gaps Between Design and Implementation

Items where the phase design notes leave decisions open *or* the
implementation simplified the design. Phase 0.4 should explicitly
choose for each remaining gap.

### 8.1 OOD `epistemic_z` ✅ RESOLVED in Phase 0.3d-2/4
Phase 0.3d-2 added per-node `expected_epistemic` capture and the
calibrated `(epi / expected).clamp(0, 1)` formula (falls back to raw
clamp when not captured). Phase 0.3d-4 added auto-capture inside
`decide_step` when `Maturity.uncertainty_stable` first holds.

### 8.2 `Maturity` struct ✅ FULLY RESOLVED in Phase 0.3d-1/4 + 0.4 B-2.2.2
Phase 0.3d-1 shipped lazy `Maturity { samples_seen,
calibration_stable, uncertainty_stable, trust_level }` with stub
flags. Phase 0.3d-4 promoted the flags to single-threshold variants
(`ECE < 0.05`; BLR + samples ≥ 100 + signature-stable ≥ 1). Phase
0.4 Track B-2.2.2 then upgraded both flags to **3-window ring
buffers** per node (`ece_history: [f64; 3]`, `sigma_history: [f64; 3]`
with per-buffer `fill_count` saturating at 3). `calibration_stable`
flips on when `max(ece_history) - min(ece_history) ≤
ECE_STABILITY_MAX_DELTA`; `uncertainty_stable` flips on when
`max(sigma_history) / min(sigma_history) ≤ SIGMA_STABILITY_RATIO`.
Both constants are currently compile-time in `maturity.rs` —
promoting them to `DecisionPolicy` thresholds is deferred to Phase
0.5 (see §8.23). Snapshot v10 absorbs the per-node +50B in place.
`min_required_samples` field promotion was not in Phase 0.4 scope —
deferred to Phase 0.5.

### 8.3 NodeSignature for merge ✅ FULLY RESOLVED in Phase 0.4 Track B-2.2.1
Phase 0.3d-1 shipped lazy 4 × 8-byte profile hashes (prediction,
uncertainty, calibration, routing). Phase 0.3d-3 used Hamming-byte
distance for Compress/Merge. Phase 0.3d-4 added persistent stability
tracking via `last_signature` + `signature_stable_calls`. Phase 0.4
Track B-2.2.1 then replaced the sha256-truncate construction with
**4-channel Welford-folded summaries** per node:
`welford_{prediction,uncertainty,calibration,routing}:
SignatureWelford { n: u64, mean: f64, m2: f64 }` (24B per channel,
+96B per node, absorbed into snapshot v10 in place). `from_node`
reads the four Welford means and packs canonical f64 bit patterns
into a `[u8; 32]`. The Welford fold makes `signature_stable_calls`
**lenient** — small post-stability observations no longer reset the
counter to zero unless the running mean shifts beyond Hamming
sensitivity. `decide_step` advances all four channels per the
contract in §3.7 step 3, including for frozen / inactive nodes so
B-2.2.7's drift-trip auto-Unfreeze sees up-to-date Welford state.
See §6.8 for the data layout.

### 8.4 Categorical features in codebook
- **Design:** categorical = stable-hashed canonical id, missing = 0xFE.
- **Code:** `QuantileCodebook` is real-valued only. No categorical
  support.
- **Decision needed:** Phase 0.4 — extend codebook variant or
  build a sibling `CategoricalCodebook`.

### 8.5 Drift signals beyond feature mean
- **Design:** label_shift_score, missingness_shift, category_shift,
  route_shift.
- **Code:** only feature-mean L2 z-shift.
- **Decision needed:** Phase 0.4 if needed by drift-trip auto-Unfreeze
  inside `decide_step`; otherwise defer.

### 8.6 `Routed` audit events
- **Design:** every `descend` call could emit a `Routed { leaf,
  matched_prefix }` event for explainability.
- **Code:** none. `descend` is a pure read; emits no event.
- **Decision needed:** Phase 0.4 (`cjcl abng explain`) — likely keep
  it as an opt-in trace mode, not a chain event, to avoid log
  explosion.

### 8.7 ChildrenPromoted hash witness ✅ SUPERSEDED by 0.3d-3
Phase 0.3d-3's structural-action events (`Grow`/`Split`/`Merge`/
`Compress`) carry full payloads (parent + child IDs, signature)
instead of hash witnesses, sidestepping the original concern.
`ChildrenPromoted` itself is unchanged but is no longer the only
topology-mutating event.

### 8.8 Snapshot-version negotiation
- Currently no version negotiation: replay either accepts `\x08` or
  errors. Phase 0.4's CLI may want a "this snapshot is from v5, please
  upgrade" diagnostic, which means decoding the magic byte and
  branching to a friendly error rather than `BadMagic`.

### 8.9 Decision-engine simplifications ✅ RESOLVED in Phase 0.4 Track B
Phase 0.3d-4 shipped defensible single-threshold triggers; Phase 0.4
Track B refined every gate into its prompt-spec form:

| Trigger | 0.3d-4 baseline | 0.4 outcome |
|---|---|---|
| Compress | sibling Hamming ≤ τ_compress | unchanged in B; full sub-tree signature equivalence deferred to 0.5 |
| Merge | sibling Hamming ≤ τ_merge | ✅ B-2.2.3 added posterior `KL ≤ kl_merge` gate; ✅ B-2.2.6 shipped real NIG-aware `combine` math |
| Split | leaf + samples_seen ≥ split_min | ✅ B-2.2.4 added bootstrap held-out ΔNLL gain ≥ `nll_split_gain` (synthetic Gaussian sampling from BLR posterior) |
| Prune | samples_seen < prune_floor + signature_stable ≥ prune_grace_epochs | unchanged |
| Grow | leaf + samples_seen ≥ grow_min + key unbound | ✅ B-2.2.5 added route-key entropy at candidate depth > `H_grow` gate |
| Freeze | signature_stable_calls ≥ freeze_after | ✅ stability signal upgraded by B-2.2.1 (Welford-smoothed) and B-2.2.2 (3-window ECE/σ ring buffers) |
| (drift-trip Unfreeze) | (no auto path) | ✅ B-2.2.7 wired auto-Unfreeze rung above the regular ladder + 12th `drift_unfreeze` threshold |

The pre-0.4 single-threshold simplifications in `crates/cjc-abng/src/graph.rs`
have all been replaced; `// Phase 0.4 will…` markers are gone.

### 8.10 Real merge math ✅ RESOLVED in Phase 0.4 Track B-2.2.6
Pre-0.4 `force_merge` and policy-driven Merge only set
`absorbed.is_active = false`, dropping all of absorbed's training
history. Phase 0.4 Track B-2.2.6 shipped: `BlrState::combine(&mut
self, other: &BlrState, prior: &BlrPrior)` and `NodeStats::combine
(&mut self, other: &NodeStats)`, both pure functions of the input
states. `force_merge` and the replay-side `apply_event` for the
`Merge` audit kind both fold absorbed's BLR posterior (NIG-aware:
sum precisions, precision-weighted-mean of means, `(a, b)` with
prior subtract) and absorbed's Welford stats (Chan/Golub/LeVeque
parallel merge) into `into` before deactivating absorbed. No
snapshot bump (combines existing fields). The `feature_version_hash`
on `into` is preserved across combine — into's feature space wins.

### 8.11 Drift-trip auto-Unfreeze ✅ RESOLVED in Phase 0.4 Track B-2.2.7
Phase 0.4 Track B-2.2.7 wired the auto-unfreeze ladder step into
`decide_step`. A new threshold `DecisionPolicy.drift_unfreeze()` (12th
slot, post-v10 magic bump) gates it: when a frozen node has both a
density tracker and a drift baseline, and `drift_score(current_density)
> drift_unfreeze`, `decide_step` calls `unfreeze` before any other
trigger considers the node. Default `f64::MAX` keeps the gate
disabled in policies that don't opt in. The unfreeze emits the same
`Unfreeze` audit kind (`0x16`) as the manual builtin so replay
treats both the same way.

### 8.12 BLR `predict()` API name vs unit ✅ RESOLVED in Phase 0.4 Track C-2.3.1
See R10. Phase 0.4 Track C-2.3.1 shipped the rename: `predict()`'s
second tuple element is now `epistemic_leverage`, the helper
`epistemic_leverage_at_posterior_mean` matches, and the doc clarifies
that `expected_epistemic` stores leverage (field name preserved for
snapshot stability). No math change, no snapshot bump.

### 8.13 NaN/Inf input validation ✅ RESOLVED in Phase 0.4 Track C-2.3.2
See R11. Phase 0.4 Track C-2.3.2 shipped non-finite rejection at four
input boundaries: `AdaptiveBeliefGraph::observe`,
`density_observe`, `calibration_observe`, and `blr_update`. New error
variants — `GraphError::NonFiniteInput`, `BlrError::NonFiniteInput` —
reject NaN, +Inf, and -Inf with a clear error before any state
mutation. Replay byte-identical for healthy training (no events fire
for rejected inputs). Test coverage in
`tests/abng/observe_validation_tests.rs`. No snapshot bump.

### 8.14 Replay missing semantic invariants ✅ RESOLVED in Phase 0.4 Track C-2.3.3
See R12. Phase 0.4 Track C-2.3.3 shipped four new `DecodeError`
variants: `SeqNonMonotonic { at_seq, expected }`,
`MissingCreatedEvent`, `EpochMismatch { event_seq, event_epoch,
header_epoch }`, `StatsVersionMismatch { event_seq, event_version,
post_apply_version }`. Replay now validates each event's seq
strictly increases by 1, the first non-genesis event is `Created`,
every event's `epoch` matches the header epoch, and every event's
`stats_version` matches the post-apply node `stats_version`.
Adversarial blobs with consistent hash chains but reordered seqs /
missing Created / forged epochs now error specifically rather than
silently passing. Test coverage in `tests/abng/replay_invariant_tests.rs`.
No snapshot bump (new error variants only).

### 8.15 BLR silent `b<0` clamp ✅ RESOLVED in Phase 0.4 Track C-2.3.4
See R13. Phase 0.4 Track C-2.3.4 shipped the audit event: tag `0x18
BlrNumericalRescue { reason: u8, b_pre_clamp_bits: u64 }` (9-byte
body). `BlrState::update` returns `Result<Option<f64>, BlrError>` —
`Some(b_pre_clamp)` when the clamp fires; the graph layer's
`blr_update` emits the event after `BlrUpdated`. `apply_event` is a
no-op for the new kind. No magic bump (consolidated into the planned
`\x08 → \x09` end-of-Phase-0.4 freeze).

### 8.16 MLP-vs-BLR feature space contract ✅ RESOLVED in Phase 0.4 Track C-2.3.5
See R14. Phase 0.4 Track C-2.3.5 shipped: `feature_version_hash` on
BlrState (snapshot v9), `BlrError::FeatureVersionStale` on stale
update, `abng_reset_blr` recovery builtin (dispatch surface 66 → 67),
fvh stamping at every BLR-init site (`set_blr_prior`, `add_node`,
`force_grow`, `force_split`), and replay-side reset semantics in
`apply_event` for `BlrInitialized`.

### 8.17 LeafParamsUpdated event volume ✅ RESOLVED in Phase 0.4 Track C-2.3.6
Phase 0.4 Track C-2.3.6 shipped the batch builtin
`abng_leaf_set_params_batch(g, node_id, params: Tensor[]) -> Void`
(dispatch.rs +1 arm: 65 → 66) and the matching audit kind
`LeafParamsUpdatedBatch { params_hash }` at tag `0x19`. The graph
method validates atomically — if any tensor's count or shape is
wrong, the node's params are unchanged and no audit event is
appended. The post-replay per-node verify path treats `0x19` as a
valid latest-hash source alongside `LeafParamsInitialized` and
`LeafParamsUpdated`. A 2-layer head's optimizer step now fires one
event instead of six; a 100-epoch / 10-leaf loop drops from ~6,000
events to ~1,000 (6× reduction).

### 8.18 Per-leaf vs per-node naming ✅ RESOLVED in Phase 0.4 Track C-2.3.7
See R16. Phase 0.4 Track C-2.3.7 shipped the doc-only rename. No
code changes; behavior was always per-node.

### 8.19 Lineage belief / inherited prior ✅ PARTIALLY RESOLVED in Phase 0.4 Track C-2.3.8
Each node's BLR prior is independent — no ancestor-conditioned
prediction or parent-as-prior on `add_node`. Phase 0.4 Track C-2.3.8
shipped the read-only fallback variant
`abng_blr_predict_with_fallback`: walks up the parent chain from the
target node to the **nearest ancestor (incl. self) with `n_seen ≥ 1`**
and returns its prediction tuple plus the source node id. Errors with
`BlrError::NoEvidence { walked }` if no ancestor has any observations.
This solves the "fresh leaf returns uninformative prior moments"
problem for read paths. **Remaining 0.5 work:** full lineage belief
(parent-as-prior on `add_node`, ancestor-conditioned posterior
combine, lineage-aware `blr_update`) is still deferred — those touch
the audit log and require careful scope decisions.

### 8.20 NodeStats canonical bytes for compaction (post-0.3d audit)
See R15. Phase 0.4 Track C-2.3.9 — extend canonical_bytes 24B → 32B
to include Kahan compensation register. Required for log compaction.

### 8.21 Cholesky regularization design-vs-code drift ✅ RESOLVED in Phase 0.4 Track C-2.3.10
Pre-0.4 PHASE_0_3b_DESIGN.md claimed Cholesky uses `f64::EPSILON`
diagonal regularization. Code does NOT — it errors on non-positive
pivot (`BlrError::NonPositiveDefinite`). Code is correct (no silent
regularization → reproducibility intact). Phase 0.4 Track C-2.3.10
deleted the regularization claim from PHASE_0_3b_DESIGN.md
"Numerical safeguards" and added an explicit no-regularization
contract to §6.5 of this doc. Doc-only fix; no code or wire-format
change.

### 8.22 Empty-graph chain-head wording ✅ RESOLVED in Phase 0.4 Track C-2.3.11
See §2.2. Phase 0.4 Track C-2.3.11 added explicit definitions of
genesis hash (`sha256(b"ABNG-GENESIS-v1")`, exposed as
`cjc_abng::genesis_hash()`) and clarified that the "internal genesis
state" exists only as the `previous_hash` of the `Created` event —
`abng_chain_head` for a freshly-created graph is always the
**post-Created** hash, never the genesis hash itself. Doc-only fix;
no code or wire-format change.

### 8.23 Audit findings (independent verification, post-0.3d)
- ✅ RESOLVED in Phase 0.4 Track C-2.3.12 — `expected_epistemic`
  re-capture: shipped `abng_force_recapture_expected_epistemic`. Each
  call clears the captured field and re-runs the same deterministic
  capture logic `decide_step` uses (`epistemic_leverage(blr.predict
  (blr.mean))`), emitting a fresh `ExpectedEpistemicCaptured` audit
  event. Required after `abng_reset_blr` to align `ood_score`'s
  calibrated ratio with the post-reset posterior shape.
- ⚠️ DEFERRED — `Maturity` thresholds (`ECE_STABILITY_MAX`,
  `UNCERTAINTY_STABLE_MIN_SAMPLES`) hardcoded as compile-time
  constants. Promoting them to `DecisionPolicy` thresholds would push
  the threshold count from 12 → 14 and force snapshot v10 → v11 —
  conflicts with the "v10 frozen for Phase 0.4" contract. Deferred
  to Phase 0.5.
- ⚠️ DEFERRED — `Unfreeze` doesn't bump `action_counts`; if 0.4's
  auto-Unfreeze fires often, observability suffers. Either approach
  (extending `action_counts` to `[u64; 7]` or adding a separate
  `unfreeze_count: u64`) extends the snapshot header and forces
  v10 → v11 — same magic-bump conflict as above. Deferred to Phase 0.5.
- No determinism canary specifically for `decide_step`; property
  tests cover monotonicity but a dedicated chain-head canary would
  catch regressions earlier. Phase 0.4 Track C-2.3.12 added one;
  see §8.24 below for the explicit entry.
- `force_compress` orphans descendants stay `is_active = true`;
  policy-driven Compress in 0.4 should mark them inactive. Deferred
  to Phase 0.5.

---

## Appendix A — File Map

| File | Role | Phase |
|---|---|---|
| `crates/cjc-abng/src/lib.rs` | re-exports + genesis hash | 0.1+ |
| `crates/cjc-abng/src/graph.rs` | `AdaptiveBeliefGraph`, install / observe / score / structural mutation / decide_step engine + KL-merge / ΔNLL-split / route-entropy-grow gates / drift-trip auto-Unfreeze | 0.1+ |
| `crates/cjc-abng/src/node.rs` | `AdaptiveBeliefNode` (per-node state) + ECE/σ ring buffers + 4 × `SignatureWelford` channels (B-2.2.{1,2}) | 0.1+ |
| `crates/cjc-abng/src/audit.rs` | `AuditEvent`, `AuditKind` (26 variants — added `0x18 BlrNumericalRescue`, `0x19 LeafParamsUpdatedBatch`), payload-bytes | 0.1+ |
| `crates/cjc-abng/src/serialize.rs` | snapshot v10 encode + replay + defensive bounds + 4 new C-2.3.3 `DecodeError` variants | 0.1+ |
| `crates/cjc-abng/src/dispatch.rs` | 69 `abng_*` builtins (`+1` from C-2.3.5 `abng_reset_blr`, `+1` from C-2.3.6 `abng_leaf_set_params_batch`, `+1` from C-2.3.8 `abng_blr_predict_with_fallback`, `+1` from C-2.3.12 `abng_force_recapture_expected_epistemic`) | 0.1+ |
| `crates/cjc-abng/src/stats.rs` | `NodeStats` (Welford + Kahan M2) + `combine` (B-2.2.6 — Chan/Golub/LeVeque parallel merge) | 0.1 + 0.4 |
| `crates/cjc-abng/src/children.rs` | `AdaptiveChildren` (Node4/16/48/256/Dense) + promotion | 0.2 + 0.3d-3 |
| `crates/cjc-abng/src/codebook.rs` | quantile codebook + prefix encoder | 0.2 |
| `crates/cjc-abng/src/route.rs` | `RouteEvidence` | 0.2 |
| `crates/cjc-abng/src/leaf_head.rs` | `LeafHead` + Xavier init | 0.3a |
| `crates/cjc-abng/src/blr.rs` | BLR prior + state + Cholesky + NIG + predict + `combine` (B-2.2.6) + `kl_divergence` (B-2.2.3) + `feature_version_hash` (C-2.3.5) + `BlrNumericalRescue` path (C-2.3.4) + `NonFiniteInput` / `FeatureVersionStale` errors | 0.3b + 0.4 |
| `crates/cjc-abng/src/density.rs` | density tracker (diag Mahalanobis) | 0.3c |
| `crates/cjc-abng/src/calibration.rs` | 15-bin ECE | 0.3c |
| `crates/cjc-abng/src/drift.rs` | drift baseline + z-score | 0.3c |
| `crates/cjc-abng/src/maturity.rs` | `Maturity` lazy + 3-window ring buffers (B-2.2.2) + `ECE_STABILITY_MAX_DELTA` / `SIGMA_STABILITY_RATIO` constants | 0.3d-1 + 0.3d-4 + 0.4 |
| `crates/cjc-abng/src/signature.rs` | `NodeSignature` 4 × 8B profiles (B-2.2.1: read from per-node `SignatureWelford` state) + `routing_observation_value` helper | 0.3d-1 + 0.4 |
| `crates/cjc-abng/src/policy.rs` | `DecisionPolicy` (12 thresholds incl. `drift_unfreeze`, validators) | 0.3d-3 + 0.4 |

| Test file | Phase |
|---|---|
| `tests/abng/unit.rs` | 0.1 |
| `tests/abng/replay.rs` | 0.1 |
| `tests/abng/dispatch.rs` | 0.1 |
| `tests/abng/parity.rs` | 0.1 |
| `tests/abng/multinode.rs` | 0.2 |
| `tests/abng/dispatch_p2.rs` | 0.2 |
| `tests/abng/parity_p2.rs` | 0.2 |
| `tests/abng/leaf_head_tests.rs` | 0.3a |
| `tests/abng/dispatch_p3a.rs` | 0.3a |
| `tests/abng/parity_p3a.rs` | 0.3a |
| `tests/abng/blr_tests.rs` | 0.3b |
| `tests/abng/dispatch_p3b.rs` | 0.3b |
| `tests/abng/parity_p3b.rs` | 0.3b |
| `tests/abng/uncertainty_tests.rs` | 0.3c |
| `tests/abng/dispatch_p3c.rs` | 0.3c |
| `tests/abng/parity_p3c.rs` | 0.3c |
| `tests/abng/maturity_signature_tests.rs` | 0.3d-1 + 0.4 (B-2.2.{1,2} updates) |
| `tests/abng/expected_epistemic_tests.rs` | 0.3d-2 |
| `tests/abng/decision_tests.rs` | 0.3d-3 + 0.3d-4 + 0.4 |
| `tests/abng/dispatch_p3d.rs` | 0.3d-1..4 + 0.4 |
| `tests/abng/parity_p3d.rs` | 0.3d-1..4 + 0.4 |
| `tests/abng/observe_validation_tests.rs` | 0.4 C-2.3.2 |
| `tests/abng/replay_invariant_tests.rs` | 0.4 C-2.3.3 |
| `tests/abng/blr_numerical_rescue_tests.rs` | 0.4 C-2.3.4 |
| `tests/abng/blr_feature_version_tests.rs` | 0.4 C-2.3.5 |
| `tests/abng/blr_predict_fallback_tests.rs` | 0.4 C-2.3.8 |
| `tests/abng/compact_log_tests.rs` | 0.4 Track A G3.7 |
| `tests/abng/decide_step_canary_tests.rs` | 0.4 C-2.3.12 |
| `tests/abng/leaf_params_batch_tests.rs` | 0.4 C-2.3.6 |
| `tests/abng/route_trace_tests.rs` | 0.4 Track A G3.5 |
| `tests/abng/merge_math_tests.rs` | 0.4 B-2.2.6 |
| `tests/abng/route_entropy_grow_tests.rs` | 0.4 B-2.2.5 |
| `tests/abng/split_nll_gate_tests.rs` | 0.4 B-2.2.4 |
| `tests/prop_tests/abng_decision_props.rs` | 0.3d-5 |
| `tests/bolero_fuzz/abng_decision_fuzz.rs` | 0.3d-5 |

## Appendix B — Most Recent Verified Test Counts (post-Phase 0.4 B+C+polish, 2026-05-07)

| Gate | Result | Δ from end-of-0.3d |
|---|---|---|
| `cargo test -p cjc-abng --lib` | **267 passed, 0 failed** | +40 (B+C: +25; +9 from G3.5 predict_snap unit; +6 from v11 policy threshold-12/13 tests) |
| `cargo test --test abng` | **442 passed, 0 failed** | +139 (B+C: +88; +13 from C-2.3.8 fallback; +9 from C-2.3.12 recapture; +6 from C-2.3.12 decide_step canary; +13 from G3.5 route_trace; +10 from G3.7 compact_log) |
| `cargo test --test prop_tests abng_decision` | **4 passed** (× 256 cases each) | +0 |
| `cargo test --test bolero_fuzz abng_decision` | **4 passed** | +0 |
| `cargo test -p cjc-cli --test abng_cli_integration` | **32 passed** (NEW Track A gate) | +32 |
| `cargo test --workspace --release --lib` | (re-run before Phase 0.5 merge) | — |
| `cargo test --test physics_ml --release` | (re-run before Phase 0.5 merge) | — |

Total ABNG-direct `#[test]` markers: **703** (261 in-crate + 442
integration), plus 4 properties (× 256 cases each), 4 fuzz targets,
and 32 cjc-cli `cjcl abng …` integration tests.

---

*This document is the source of truth as of end-of-Phase-0.4
(2026-05-07). All three tracks (A, B, C) are shipped under snapshot
magic `\x0A`; Phase 0.5 picks up the four magic-bump items
consolidated into one v10 → v11 bump (per-node
`provenance_stamp_hash` + `0x1C ProvenanceStamped`, configurable
Maturity constants, `unfreeze_count` observability,
`NodeStats::canonical_bytes` 24B → 32B). When the code and a phase
design note disagree, this document — and the code it references —
wins.*
