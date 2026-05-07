# ABNG — Current Architecture (post-Phase 0.3d)

**As of:** 2026-05-07
**Crate:** `crates/cjc-abng/`
**Tests:** `tests/abng/` (303 integration) + `crates/cjc-abng/src/*` (227 in-crate) + `tests/prop_tests/abng_decision_props.rs` (4 properties × 256 cases) + `tests/bolero_fuzz/abng_decision_fuzz.rs` (4 fuzz targets) — all passing
**Snapshot magic:** `ABNG\x08` (v8)
**Builtin count:** 65 user-facing `abng_*` arms in `dispatch.rs`
**Audit kinds:** 24 (tags `0x00`..`0x17`)

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
| 0.3a  | DONE | Per-leaf MLP head (Xavier init, `leaf_forward` into ambient `cjc-ad` GradGraph) |
| 0.3b  | DONE | Per-leaf BLR (Cholesky + NIG conjugate update; `(mean, epi, ale)` predict) |
| 0.3c  | DONE | Density tracker (diag Mahalanobis), 15-bin ECE calibration, drift baseline + score, composite `ood_score` |
| 0.3d-1 | DONE | `Maturity` + `NodeSignature` types (lazy), 2 read-only inspection builtins |
| 0.3d-2 | DONE | `expected_epistemic` per-leaf capture, calibrated `ood_score` ratio formula (`epi / expected`), audit kind `0x17`, snapshot `v6` |
| 0.3d-3 | DONE | `DecisionPolicy` install, 6 force-* structural mutations, `Dense` children variant (kind `5`), audit kinds `0x10..0x15`, snapshot `v7` |
| 0.3d-4 | DONE | `decide_step` policy engine (6 triggers, fall-through order), `unfreeze` (audit `0x16`), persistent signature-stability state, real `Maturity` flags, auto-capture of `expected_epistemic`, snapshot `v8` |
| 0.3d-5 | DONE | proptest properties + bolero fuzz targets + decoder allocation hardening + this doc |
| 0.4   | NEXT | `cjcl abng …` CLI, JSON snapshot, log compaction, Welford-smoothed signatures, KL-merge, ΔNLL split |
| 0.5   | LATER | Chess-RL retrofit (value head first, then policy head) |

ABNG is now a **Bayesian-inspired structurally-adaptive belief graph** with:
- Topology + routing (0.2)
- Neural per-leaf head (0.3a)
- Per-leaf calibrated uncertainty (0.3b)
- Per-node OOD/calibration/drift signals (0.3c)
- Lazy + persistent maturity / signature evidence (0.3d-1, 0.3d-4)
- Calibrated OOD ratio with auto-captured training-time σ (0.3d-2, 0.3d-4)
- Frozen-threshold `DecisionPolicy` + 6 structural-action audit kinds (0.3d-3)
- One-pass deterministic `decide_step` engine (0.3d-4)
- Property-tested + fuzz-hardened replay/decoder boundary (0.3d-5)
- A complete tamper-evident audit chain over every state mutation
- Bit-deterministic snapshot round-trip across both AST and MIR backends

ABNG is now structurally adaptive at runtime: install a policy, observe
evidence, call `decide_step` to fire structural mutations whose actions
are deterministic functions of (Maturity, NodeSignature, current
graph topology). Phase 0.4 layers a CLI and quality refinements on top.

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
* `AdaptiveBeliefGraph::verify_chain()` recomputes every event's
  `new_hash` from genesis and asserts equality. Any tamper anywhere
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

* `MAGIC = b"ABNG\x08"` — bumped on every breaking format change.
  Old magic bytes (`\x01`..`\x07`) are explicitly rejected. Phase 0.3d
  alone bumped through 3 versions (`\x05 → \x06 → \x07 → \x08`); each
  was a clean break with no backward-compatibility path.
* Audit-kind tag bytes `0x00`..`0x17` are **frozen forever**. New
  kinds must allocate fresh tags. Phase 0.3d allocated `0x10..0x17`
  (8 new tags); see §3.6 for the full table.
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
| Decision policy | `abng_set_decision_policy(g, thresholds: Tensor[11])` | (none) | **Yes** (install-anytime) |

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

### 3.4 Builtin surface (65 arms)

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
`abng_leaf_params_hash`, `abng_leaf_forward`.

BLR: `abng_set_blr_prior`, `abng_blr_features`, `abng_blr_update`,
`abng_blr_predict`, `abng_blr_state_hash`, `abng_blr_n_seen`.

Density / calibration / drift / OOD:
`abng_set_density_tracker`, `abng_density_observe`, `abng_density_score`,
`abng_density_n_seen`, `abng_set_calibration`,
`abng_calibration_observe`, `abng_calibration_ece`,
`abng_calibration_n_seen`, `abng_freeze_drift_baseline`,
`abng_drift_score`, `abng_ood_score`.

**Phase 0.3d additions (16 builtins):**

Maturity / signature inspection (0.3d-1):
`abng_node_maturity`, `abng_node_signature`.

Expected epistemic σ capture (0.3d-2):
`abng_set_expected_epistemic`, `abng_expected_epistemic`.

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
   blr.is_some()`, capture `epistemic_var(blr.predict(blr.mean))`
   through the existing `set_expected_epistemic` path.
6. **Trigger fall-through (at most one fires per node per call):**
   1. Compress — children present + all child signatures within
      `tau_compress` Hamming of node's signature
   2. Merge — sibling with smaller `NodeId` whose signature is within
      `tau_merge` Hamming
   3. Split — leaf + `samples_seen ≥ split_min`
   4. Prune — `samples_seen < prune_floor` AND
      `signature_stable_calls ≥ prune_grace_epochs` (root never pruned)
   5. Grow — leaf + `samples_seen ≥ grow_min` + deterministic-from-
      `(seed, node_id)` key byte not bound
   6. Freeze — `signature_stable_calls ≥ freeze_after`
7. **Returns `[u64; 6]`** indexed by [`ActionKind`]:
   `[Grow, Split, Merge, Prune, Compress, Freeze]`.

**Triggers are simplified.** Phase 0.3d-4 ships defensible single-
threshold variants; Phase 0.4 will refine to: Welford-smoothed
signatures, KL-divergence merge, bootstrap held-out ΔNLL split, route-
entropy grow, real NIG-aware merge math. The simplifications are
documented inline in `crates/cjc-abng/src/graph.rs` with `Phase 0.4
will…` markers.

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

### R5. Snapshot break each phase
v1 → v2 → v3 → v4 → v5 → v6 → v7 → v8 — Phase 0.3d alone bumped
through 3 versions (one per sub-step that added persistent state).
Acceptable so far because ABNG had no real users, but Phase 0.4's
CLI ships first to external users — **version 0.4 should freeze
the format for the CLI's lifetime** (likely as v9 with all known
0.4 additions consolidated into a single bump).

### R6. `leaf_forward` clones every param into the GradGraph each call
Fine for small heads. For large heads or PINN-style high-frequency
forward, switch to `set_tensor`+`reforward`. Phase 0.3d optimization.

### R7. Drift baseline freeze policy is user-driven
Phase 0.3c only provides the primitive. Phase 0.3d-4 ships
`abng_unfreeze` (manual) but the **drift-trip auto-unfreeze** path
inside `decide_step` is deferred to Phase 0.4 alongside the CLI.

### R8. `Maturity` not yet plumbed ✅ RESOLVED in Phase 0.3d
Phase 0.3d-1 shipped lazy `Maturity { samples_seen,
calibration_stable, uncertainty_stable, trust_level }` and
`NodeSignature` (4 × 8B profile hashes). Phase 0.3d-4 promoted both
to participate in `decide_step`'s triggers — `calibration_stable`
flips on `ECE < 0.05`, `uncertainty_stable` requires BLR + samples ≥
100 + signature-stable for ≥ 1 decide_step call.

### R9. Decision-engine simplifications (Phase 0.3d-4)
The `decide_step` engine ships with deliberately simplified triggers
(see §3.7). Phase 0.4 will refine to: Welford-smoothed signatures,
KL-divergence merge, bootstrap held-out ΔNLL split, route-entropy
grow, real NIG-aware merge math, and 3-window ECE/σ stability. The
*event channel* is full strength now; the *quality* of the signal
is the 0.4 concern.

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

0.4 (NEXT — CLI + quality refinements)
  - cjcl abng {train,inspect,explain,replay,diff} CLI
  - Welford-smoothed NodeSignature profiles
  - 3-window ECE / σ stability buffers for Maturity
  - KL-divergence gate for Merge
  - Bootstrap held-out ΔNLL gain for Split
  - Route-entropy gate for Grow
  - Real NIG-aware merge math (combine BLR posteriors)
  - Drift-trip auto-Unfreeze inside decide_step
  - JSON-safe snapshot view via cjc_snap::snap_to_json
  - Log compaction (squash N consecutive *Updated into one Snapshot)
  - Snapshot-version negotiation diagnostics
  - Categorical features in codebook
  - `Routed` audit events (opt-in trace mode)
  - Snapshot version frozen for CLI lifetime (likely v9, single bump)

0.5
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
| `BlrState` | `canonical_bytes()` | d u32 ‖ mean f64×d ‖ precision f64×d² ‖ a f64 ‖ b f64 ‖ n_seen u64 |
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

- Per-node `BlrState { d, mean, precision (d×d), a, b, n_seen }`.
- **NIG conjugate update** on `(features [n,d], y [n])`:
  - `Λ_new = Λ + Xᵀ X`
  - `m_new = Λ_new⁻¹ (Λ μ + Xᵀ y)`
  - `a_new = a + n/2`
  - `b_new = b + 0.5 (yᵀy + μᵀΛμ − m_newᵀ Λ_new m_new)`
- Λ_new⁻¹φ is computed via **hand-rolled Kahan-compensated Cholesky**
  + triangular solves. No external linear-algebra dep. No FMA.
- `predict(phi) -> (mean, epistemic_var, aleatoric_var)`:
  - `mean = μᵀφ`
  - `epistemic_var = ‖L⁻¹φ‖²`
  - `aleatoric_var = b/(a−1)` if `a > 1` else `+∞` (i.e. unbounded
    aleatoric pre-update)
- Numerically validated: `y = 2x₁ + 3x₂` recovered to <0.01 from 200
  deterministic samples; epistemic variance monotonically decreases
  with evidence.

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
The calibrated ratio formula activates per-leaf once
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
  yet plumbed** through the audit log. Phase 0.4 (`cjcl abng explain`)
  is when this gets formalized.

---

## 7. Do Not Change Assumptions

These are the assumptions other parts of the codebase rely on. Breaking
any of them silently is a regression even if all tests pass — because
some are about *what's not tested yet*.

1. **`Value` enum layout is unchanged.** Every ABNG handle is
   `Value::Int(i64)`, `Value::Tensor`, `Value::String`, `Value::Bytes`,
   `Value::Array`, `Value::Bool`, or `Value::Float`. No new `Value`
   variant.
2. **`MAGIC` is `b"ABNG\x08"` and is the *only* accepted magic.**
   Phase 0.4 that needs a new field bumps to `b"ABNG\x09"` — never
   adds a fallback for older versions.
3. **Audit-kind tag bytes `0x00..0x17` keep their current semantics.**
   Phase 0.4's new kinds use `0x18..` only. **Never** re-number.
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
Phase 0.3d-2 added per-leaf `expected_epistemic` capture and the
calibrated `(epi / expected).clamp(0, 1)` formula (falls back to raw
clamp when not captured). Phase 0.3d-4 added auto-capture inside
`decide_step` when `Maturity.uncertainty_stable` first holds.

### 8.2 `Maturity` struct ✅ RESOLVED in Phase 0.3d-1/4
Phase 0.3d-1 shipped lazy `Maturity { samples_seen,
calibration_stable, uncertainty_stable, trust_level }` with stub
flags. Phase 0.3d-4 promoted the flags to single-threshold variants
(`ECE < 0.05`; BLR + samples ≥ 100 + signature-stable ≥ 1).
**Remaining 0.4 work:** 3-window stability buffers (currently
single-threshold), `min_required_samples` field promotion (currently
just `samples_seen`).

### 8.3 NodeSignature for merge ✅ PARTIALLY RESOLVED
Phase 0.3d-1 shipped lazy 4 × 8-byte profile hashes (prediction,
uncertainty, calibration, routing). Phase 0.3d-3 used Hamming-byte
distance for Compress/Merge. Phase 0.3d-4 added persistent stability
tracking via `last_signature` + `signature_stable_calls`.
**Remaining 0.4 work:** Welford-folded summaries (currently lazy
sha256-truncate of subsystem state — changes on every observation,
which makes `signature_stable_calls` strict; Welford-smoothing makes
it lenient).

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

### 8.9 Decision-engine simplifications (Phase 0.3d-4 deferrals)
Phase 0.3d-4 ships defensible single-threshold triggers; full
prompt-spec triggers land in 0.4:

| Trigger | 0.3d-4 (current) | 0.4 (planned) |
|---|---|---|
| Compress | sibling Hamming ≤ τ_compress | full sub-tree signature equivalence |
| Merge | sibling Hamming ≤ τ_merge **only** | + posterior KL < kl_merge gate |
| Split | leaf + samples_seen ≥ split_min | + held-out ΔNLL gain ≥ nll_split_gain + impurity ≥ impurity_min |
| Prune | unchanged from 0.4 design | unchanged |
| Grow | leaf + samples_seen ≥ grow_min + key unbound | + route entropy > H_grow gate |
| Freeze | unchanged from 0.4 design | unchanged |

### 8.10 Real merge math (Phase 0.3d-4 deferral)
`force_merge` and policy-driven Merge in 0.3d-4 only set
`absorbed.is_active = false`. Phase 0.4 will combine BLR posteriors
(NIG-aware mean / precision / a / b combination) and fold stats
into the surviving node so `into` actually inherits `absorbed`'s
evidence.

### 8.11 Drift-trip auto-Unfreeze (Phase 0.3d-4 deferral)
`abng_unfreeze` exists as a manual builtin. The architecture-doc
intent is that `decide_step` would auto-unfreeze on drift signals
exceeding a threshold. Phase 0.4 wires this through.

---

## Appendix A — File Map

| File | Role | Phase |
|---|---|---|
| `crates/cjc-abng/src/lib.rs` | re-exports + genesis hash | 0.1+ |
| `crates/cjc-abng/src/graph.rs` | `AdaptiveBeliefGraph`, install / observe / score / structural mutation / decide_step engine | 0.1+ |
| `crates/cjc-abng/src/node.rs` | `AdaptiveBeliefNode` (per-node state) | 0.1+ |
| `crates/cjc-abng/src/audit.rs` | `AuditEvent`, `AuditKind` (24 variants), payload-bytes | 0.1+ |
| `crates/cjc-abng/src/serialize.rs` | snapshot v8 encode + replay + defensive bounds | 0.1+ |
| `crates/cjc-abng/src/dispatch.rs` | 65 `abng_*` builtins | 0.1+ |
| `crates/cjc-abng/src/stats.rs` | `NodeStats` (Welford + Kahan M2) | 0.1 |
| `crates/cjc-abng/src/children.rs` | `AdaptiveChildren` (Node4/16/48/256/Dense) + promotion | 0.2 + 0.3d-3 |
| `crates/cjc-abng/src/codebook.rs` | quantile codebook + prefix encoder | 0.2 |
| `crates/cjc-abng/src/route.rs` | `RouteEvidence` | 0.2 |
| `crates/cjc-abng/src/leaf_head.rs` | `LeafHead` + Xavier init | 0.3a |
| `crates/cjc-abng/src/blr.rs` | BLR prior + state + Cholesky + NIG + predict | 0.3b |
| `crates/cjc-abng/src/density.rs` | density tracker (diag Mahalanobis) | 0.3c |
| `crates/cjc-abng/src/calibration.rs` | 15-bin ECE | 0.3c |
| `crates/cjc-abng/src/drift.rs` | drift baseline + z-score | 0.3c |
| `crates/cjc-abng/src/maturity.rs` | `Maturity` lazy + real flags | 0.3d-1 + 0.3d-4 |
| `crates/cjc-abng/src/signature.rs` | `NodeSignature` 4 × 8B profiles | 0.3d-1 |
| `crates/cjc-abng/src/policy.rs` | `DecisionPolicy` (11 thresholds, validators) | 0.3d-3 |

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
| `tests/abng/maturity_signature_tests.rs` | 0.3d-1 |
| `tests/abng/expected_epistemic_tests.rs` | 0.3d-2 |
| `tests/abng/decision_tests.rs` | 0.3d-3 + 0.3d-4 |
| `tests/abng/dispatch_p3d.rs` | 0.3d-1..4 |
| `tests/abng/parity_p3d.rs` | 0.3d-1..4 |
| `tests/prop_tests/abng_decision_props.rs` | 0.3d-5 |
| `tests/bolero_fuzz/abng_decision_fuzz.rs` | 0.3d-5 |

## Appendix B — Most Recent Verified Test Counts (post-Phase 0.3d)

| Gate | Result |
|---|---|
| `cargo test -p cjc-abng --lib` | **227 passed, 0 failed** |
| `cargo test --test abng` | **303 passed, 0 failed** |
| `cargo test --test prop_tests abng_decision` | **4 passed** (× 256 cases each) |
| `cargo test --test bolero_fuzz abng_decision` | **4 passed** |
| `cargo test --workspace --release --lib` | **2,363 passed, 0 failed** |
| `cargo test --test physics_ml --release` | **107 passed, 0 failed, 2 ignored** |

Total ABNG-direct `#[test]` markers: **~530** (227 in-crate + 303
integration), plus 4 properties (× 256 cases each) and 4 fuzz targets.

---

*This document is the source of truth as of end-of-Phase-0.3d
(2026-05-07). When the code and a phase design note disagree, this
document — and the code it references — wins. Phase 0.4 work should
update this doc in lockstep with the code, not after.*
