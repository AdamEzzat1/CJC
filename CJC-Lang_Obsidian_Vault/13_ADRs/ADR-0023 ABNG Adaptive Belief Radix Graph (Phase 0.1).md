---
title: "ADR-0023: ABNG Adaptive Belief Radix Graph — Phase 0.1 (Arena, Audit, Replay)"
tags: [adr, ml, abng, audit, determinism, satellite-dispatch]
status: Accepted
date: 2026-05-05
---

# ADR-0023: ABNG Adaptive Belief Radix Graph — Phase 0.1 (Arena, Audit, Replay)

## Status

Accepted

## Context

ABNG (Adaptive Belief Network Graph) is the project's
auditable-Bayesian-inspired neural architecture. The v0.3 vision adds an
**adaptive radix-tree-shaped belief graph** whose nodes grow / split /
merge / compress under explicit evidence, calibration, and OOD
pressure — see the eight-role design document under
[[Showcase|`docs/abng/PHASE_0_1_DESIGN.md`]].

The full v0.3 surface is too large for one PR (~50 builtins, structural
decision engine, per-leaf BLR head, drift detector, calibration bins,
OOD scoring, four hash chains). This ADR records the **Phase 0.1**
slice — the substrate that everything else assumes.

Phase 0.1 ships:

* a single root node with running belief statistics (Welford μ +
  Kahan-compensated M2)
* an append-only `AuditEvent` log with SHA-256 hash chain
  (`sha256(previous_hash ‖ canonical_payload)`)
* a binary snapshot format (`abng_serialize` / `abng_replay`) that is
  bit-identical across runs and platforms
* 13 user-facing builtins (`abng_*`) reachable from `.cjcl` source via
  satellite dispatch

Phase 0.1 explicitly defers:

* multiple nodes, prefix encoder, `Node4`/`Node16`/`Node48`/`Node256`/
  `Dense` children variants → Phase 0.2
* per-node MLPs / belief tensors / BLR head → Phase 0.3
* structural decisions (Grow/Split/Merge/Prune/Compress/Freeze) → Phase 0.2
* drift detector, OOD score, calibration bins → Phase 0.3+
* `cjcl abng …` CLI subcommands → Phase 0.4

## Decision

### 1. New crate `cjc-abng/` using satellite-dispatch pattern

Following the [[ADR-0016 Language-Level GradGraph Primitives|`cjc-ad/dispatch.rs`]]
and `cjc-quantum/dispatch.rs` precedents, ABNG lives in its own crate
with a `dispatch_abng(name, args)` entry point routed from both
`cjc-eval` and `cjc-mir-exec` *after* `dispatch_grad_graph`.

This avoids a `cjc-runtime → cjc-abng` dependency cycle. The fall-through
chain in both executors is now:

```
dispatch_builtin → dispatch_quantum → dispatch_grad_graph → dispatch_abng
```

### 2. Multi-graph thread-local arena

Instead of `cjc-ad`'s single ambient graph, ABNG uses a per-thread
`BTreeMap<i64, AdaptiveBeliefGraph>` keyed by an integer `graph_id`
returned from `abng_new(seed)`. Reasons:

* **Tests:** independent graphs without resetting global state.
* **Future `cjcl abng diff a.snap b.snap`:** two graphs in one process.
* **Natural language API:** `let g = abng_new(seed)` instead of hidden
  ambient state.

`graph_id` crosses the language boundary as `Value::Int(i64)`; this
preserves the `Value` enum layout (HARD RULE #1).

### 3. SHA-256 hash chain over canonical bytes

Every audit event is hash-chained:

```
new_hash = sha256(previous_hash ‖ canonical_payload)
```

Canonical-payload byte layout is frozen as part of the chain contract:

```
seq:           u64 BE (8)
epoch:         u64 BE (8)
node_id:       u32 BE (4)
kind tag:      u8     (1)        0x00=Created  0x01=BeliefUpdate
if BeliefUpdate:
  value:       u64 BE (8)        f64 bit pattern
stats_version: u64 BE (8)
stats_hash:    [u8; 32]
```

Hashing is via `cjc_snap::hash::sha256` — the existing zero-dependency
hand-rolled FIPS 180-4 implementation. No new external dependency.

### 4. Welford + Kahan for sufficient statistics

`NodeStats` uses Welford's streaming recurrence for the mean
(`mean += (x − mean) / n`) and a `cjc_repro::KahanAccumulatorF64` for
the sum of squared deviations. Variance is `M2 / (n − 1)` (sample
variance) when `n ≥ 2`.

For a fixed sample order, `NodeStats::canonical_bytes()` is bit-identical
across runs and platforms, which makes the chain hash bit-identical.

### 5. Snapshot format: hand-rolled binary, not `Value`-tagged

Phase 0.1 defines its own binary format rather than routing through
`cjc-snap`'s `Value`-tagged framing. Reasons:

* Smaller blobs (~117 bytes per event vs `Value::Struct` framing).
* Replay is direct — recover the typed struct, not `Value` decode then
  field extraction.
* The format is internal; `cjc-snap` is the right path for serializing
  *user* values, not internal arenas.

Layout:

```
magic         "ABNG\x01"     (5)
seed          u64 BE         (8)
epoch         u64 BE         (8)
final_hash    [u8; 32]       (32)
n_nodes       u32 BE         (4)         Phase 0.1: always 1
per node:     canonical_bytes(24) + stats_version u64(8) + stats_chain_head(32)
n_events      u64 BE         (8)
per event:    payload_len u32(4) + payload + previous_hash + new_hash
```

`abng_replay` rebuilds the graph **from the events alone** and asserts:

1. each event's recomputed `new_hash` matches the stored value;
2. each replayed node's `canonical_bytes` matches the stored value;
3. the final `chain_head` matches the stored `final_hash`.

All three must hold or the snapshot is rejected (`DecodeError`).

### 6. Builtin surface (13 names)

| Name | Purpose |
|---|---|
| `abng_new(seed)` → `Int` | create new graph, return graph_id |
| `abng_drop(graph_id)` | free a graph |
| `abng_root(graph_id)` → `Int` | root node id (always 0 in Phase 0.1) |
| `abng_observe(graph_id, node_id, value)` | apply BeliefUpdate, append audit |
| `abng_observe_batch(graph_id, node_id, tensor)` | sequenced batch observation |
| `abng_node_count(graph_id)` → `Int` | number of nodes |
| `abng_node_stats(graph_id, node_id)` → `Tensor[3]` | `[n_seen, mean, variance]` |
| `abng_node_stats_version(graph_id, node_id)` → `Int` | per-node stats version |
| `abng_audit_len(graph_id)` → `Int` | number of audit events |
| `abng_chain_head(graph_id)` → `String` | hex-encoded chain head |
| `abng_verify_chain(graph_id)` → `Bool` | recompute chain and check |
| `abng_serialize(graph_id)` → `Bytes` | snapshot blob |
| `abng_replay(bytes)` → `Int` | replay → new graph; errs on mismatch |

## Consequences

### Positive

* **Determinism by construction.** Every observation is hash-chained;
  any non-determinism upstream (in Welford order, in tensor iteration,
  in event sequencing) immediately surfaces as a chain-head mismatch.
* **Tamper detection.** Flipping a single byte in any event payload
  breaks every subsequent `new_hash`. `abng_verify_chain` catches it
  in O(audit_len) time.
* **Replay = verification.** `abng_serialize → abng_replay` is the
  canonical regression test for the whole substrate; one assertion
  exercises Welford correctness, Kahan ordering, hash-chain integrity,
  and serializer/parser symmetry.
* **Foundation is honest.** Phase 0.1 ships *only* what it claims;
  no placeholder fields for structural decisions or per-leaf MLPs that
  would silently freeze byte layouts before their semantics are
  designed.

### Negative

* **No structural changes yet.** The single-root-only constraint is
  Phase 0.1's defining limitation — the radix-tree topology that
  motivates the whole project is Phase 0.2's deliverable.
* **No neural head yet.** The chess RL value-head application is
  Phase 0.3.
* **Snapshot format is byte-frozen.** Phase 0.2 must be
  format-compatible (we expanded `n_nodes` to `u32` for that; we
  reject `n_nodes != 1` at decode in 0.1 explicitly so older binaries
  don't silently misread newer snapshots).

### Neutral

* **Builtin count grew by 13** (475 → 488). No collisions with
  existing names.
* **One incidental fix:** the 16 typed-NodeIdx call-sites in
  `cjc-ad/src/dispatch.rs` Phase 3a/3b arms (`grad_graph_softmax`,
  `cross_entropy`, `layer_norm`, `gelu`, `silu`, `reshape`,
  `batch_norm`, `gather`, `cat`, `reforward`, `backward_collect`)
  needed `.index()` / `NodeIdx::from_usize(...)` boundary adapters
  to compile. This was a pre-existing partial-migration state from
  the [[ADR-0022 Typed Index Newtypes for ML Metadata|typed-index-newtypes
  PR]]; ABNG's regression gate surfaced and fixed it. All 107
  `physics_ml` PINN parity tests continue to pass.

## Tests

* **Unit tests in `cjc-abng/src/{audit,graph,node,serialize,stats}`.rs**:
  28 tests covering Welford correctness, audit chain, snapshot
  round-trip, tampering detection, double-run determinism.
* **Integration tests in `tests/abng/`**: 44 tests across `unit.rs`,
  `replay.rs`, `dispatch.rs`, `parity.rs`. Includes:
  * Welford accuracy vs naive on random uniforms.
  * Chain-break detection on tampered events.
  * Full serialize → replay byte-identity gate.
  * Every builtin exercised through `dispatch_abng` with happy-path
    and Err-path coverage.
  * AST↔MIR parity on `.cjcl` source for every public builtin.
  * `parity_double_run_chain_head_byte_identical` runs the same
    program four times (eval×2, mir×2) and asserts all four chain
    heads are byte-equal.
* **Workspace regression gate**: 2,164 lib tests across 23 crates
  pass with zero failures; `physics_ml` 107/107 (PINN suite) pass
  after the cjc-ad boundary fix.

## Roadmap

* **Phase 0.2:** prefix encoder + `AdaptiveChildren` enum (`Node4` /
  `Node16` / `Node48` / `Node256` / `Dense`) + structural decisions
  (Grow/Split/Merge/Prune/Compress/Freeze) + per-node stats chain.
* **Phase 0.3:** per-leaf MLPs via existing `grad_graph_*` builtins;
  per-leaf Bayesian-linear-regression head; OOD scoring; calibration
  bins.
* **Phase 0.4:** `cjcl abng {train,inspect,explain,replay,diff}`
  CLI; JSON snapshot path; log compaction.
* **Phase 0.5:** chess RL retrofit — replace value head only first,
  then policy head; uncertainty-gated bootstrap.

## See also

* [[ADR-0016 Language-Level GradGraph Primitives]] — satellite-dispatch
  pattern that ABNG mirrors.
* [[ADR-0017 Adaptive TidyView Selection]] — five-armed enum (Empty /
  All / SelectionVector / VerbatimMask / Hybrid) that the
  `AdaptiveChildren` enum mirrors.
* [[ADR-0022 Typed Index Newtypes for ML Metadata]] — the typed-NodeIdx
  migration whose tail-end inconsistency ABNG's regression gate
  surfaced.

## Phase 0.2 amendment (2026-05-05) — Multi-node, Children, Routing

Phase 0.2 layered the radix-tree topology on top of Phase 0.1's substrate
without changing any of the original determinism / hash-chain / replay
contracts. Snapshot magic bumped `\x01` → `\x02` to signal the format
break; v1 snapshots are explicitly rejected with `BadMagic`.

### Decisions

1. **`AdaptiveChildren` as a five-armed enum** mirroring
   [[ADR-0017 Adaptive TidyView Selection]]. Insert-time auto-promotion
   (Node4 → Node16 → Node48 → Node256). `Node48` uses ART's two-array
   trick (256-byte index + packed `slots: Vec`) for ~5× memory savings
   over `Node256` at medium arity. `Dense` variant (compressed-signature
   terminal) deferred to Phase 0.4 alongside log compaction.
2. **`ChildrenPromoted` event before `NodeAdded`** so the audit log
   replays in structural-then-mutational order; replay's `add_child`
   call already triggers the promotion deterministically, so
   `ChildrenPromoted` is a witness, not an action.
3. **Per-node stats chain decouples from global event chain.**
   `node.stats_chain_head` advances *only* on observations to that node;
   the global chain still records every event. Phase 0.1 had nothing to
   decouple from (one node ⇒ both chains coincide). This is what makes
   `cjcl abng inspect --node ID` (Phase 0.4) auditable in isolation.
4. **Quantile codebook is one-shot frozen.** Once installed, subsequent
   `set_codebook` calls error with `CodebookAlreadyFrozen`. The codebook
   bytes live in the snapshot header; the `CodebookFrozen` audit event
   carries only a 32-byte hash witness. Replay re-derives the codebook
   from the header and verifies the witness matches.
5. **Structural decisions (Grow/Split/Merge/Prune/Compress/Freeze)
   deferred to Phase 0.3.** They depend on per-leaf calibration error
   and NLL gain, neither of which exists without a neural head. Shipping
   empty policy buckets in 0.2 would freeze API shapes before semantics
   were designed.

### Surface

* **11 new builtins** (488 → 499 total dispatch arms): `abng_add_node`,
  `abng_node_parent`, `abng_node_kind`, `abng_node_child_count`,
  `abng_node_child`, `abng_set_codebook`, `abng_codebook_dims`,
  `abng_codebook_hash`, `abng_encode_prefix`, `abng_descend`,
  `abng_route_path`, `abng_node_stats_chain_head` (one extra inspection
  helper alongside the planned 10).
* New audit kinds with frozen tag bytes: `NodeAdded` (0x02),
  `ChildrenPromoted` (0x03), `CodebookFrozen` (0x04).
* Snapshot format v2 (see [[Showcase|`docs/abng/PHASE_0_2_DESIGN.md`]]
  for the byte layout).

### Tests

* In-crate: 28 → 62 (+34) covering `AdaptiveChildren` promotion,
  `QuantileCodebook` encoding + freeze, multi-node graph mutation,
  per-node chain isolation, snapshot v2 round-trip with codebook +
  promotion.
* Integration (`tests/abng/`): 44 → 89 (+45) covering all new builtins
  + AST↔MIR parity for each + multi-node determinism double-runs.
* Workspace lib regression: 2,164 → 2,198 lib tests pass, 0 failures.
* `physics_ml` 107/107 still pass (cjc-ad incidental fix from 0.1
  remains correct).

### Bug surfaced and fixed during Phase 0.2

`AdaptiveChildren::is_full()` for `Node48` originally checked
`slots.iter().all(|s| s.is_some())`. Since `Node48` only ever pushes
`Some(_)` into its `Vec<Option<NodeId>>`, that predicate is `true` for
any non-empty vec — promotion fired 47 inserts early. The right test is
`slots.len() == 48`. Caught by the
`forty_ninth_insert_promotes_to_node256` unit test before any code
shipped past the in-crate gate.

## Phase 0.3a amendment (2026-05-06) — Per-leaf MLP head

Phase 0.3a is the slice that makes ABNG a *neural* architecture. Every
node now carries a fused MLP whose architecture is described by a
graph-wide [`LeafHead`](../../crates/cjc-abng/src/leaf_head.rs); per-leaf
Xavier-initialised weights/biases live in `AdaptiveBeliefNode.params`;
the new `leaf_forward(node_id, x_idx)` method wires the leaf's MLP into
the ambient `cjc_ad::GradGraph` and returns `(y_idx, param_indices)`
ready for `grad_graph_backward_collect`. Snapshot magic bumped
`\x02` → `\x03`.

### Decisions

1. **`set_leaf_head` is one-shot and must precede any `add_node`.** Same
   freeze pattern as the codebook. The "before any add_node" guard makes
   the params-shape contract a structural invariant: if a head is
   configured, every node has correctly-shaped params; if not, no node
   has any.
2. **Three-way XOR'd SplitMix64 streams for Xavier seed derivation.**
   Naively seeding `Rng::seeded(graph_seed ^ node_id ^ ...)` produces
   correlated low-order bits across adjacent node ids; XORing three
   independent SplitMix64 outputs (one for graph_seed, one for node_id,
   one for layer_idx+kind_bit) breaks the correlation while preserving
   bit-exact determinism.
3. **`LeafParamsUpdated` events carry only a 32-byte witness, not the
   tensor.** The actual post-update params live in the snapshot's
   per-node section. This keeps events small under heavy training (a
   million SGD steps adds ~32 MB of audit log instead of 32 MB ×
   params/leaf). Replay verifies the stored params hash matches the
   most-recent witness for each node.
4. **`leaf_forward` clones params into the ambient graph each call.**
   Phase 0.3a takes the simple "fresh GradGraph each step" pattern. The
   `set_tensor`+`reforward` graph-reuse optimisation from Phase 3c is
   left to Phase 0.3d (when training-loop perf actually matters).
5. **`cjc-abng` gains a direct `cjc-ad` dependency.** No cycle (cjc-ad
   doesn't depend on cjc-abng). The alternative — orchestrating
   `dispatch_grad_graph` re-entrancy from the cjc-eval/cjc-mir-exec
   layer — would push architecture knowledge into user-facing dispatch
   code. Direct dep is cleaner.
6. **Snapshot v3 is a clean break from v2** for the same reason v2 was
   a clean break from v1: ABNG is in its first dev cycle, no real users
   depend on the older format. Each phase bumps the magic byte rather
   than maintaining backward compatibility shims that would lock in
   half-baked layouts.

### Surface

* **7 new builtins** (499 → 506): `abng_set_leaf_head`,
  `abng_leaf_head_dims`, `abng_leaf_param_count`, `abng_leaf_param`,
  `abng_leaf_set_param`, `abng_leaf_params_hash`, `abng_leaf_forward`.
* **3 new audit kinds** with frozen tag bytes:
  `LeafHeadConfigured` (0x05), `LeafParamsInitialized` (0x06),
  `LeafParamsUpdated` (0x07).
* **Snapshot v3** layout extends v2 with an optional head section in
  the header and a per-node `n_params + tensor blob` section
  (see [[Showcase|`docs/abng/PHASE_0_3a_DESIGN.md`]]).

### Tests

* In-crate: 62 → 77 (+15) covering Xavier init determinism + bound
  invariants, head freeze + before-add_node guard, params-hash on init
  vs after update, snapshot v3 round-trip with head + per-node params.
* Integration (`tests/abng/`): 89 → 122 (+33) covering each new
  builtin with happy-path and Err-path assertions, `leaf_forward`
  shape correctness, AST↔MIR parity for all 7 builtins, **a flagship
  end-to-end train-step parity** that runs forward → backward →
  manual SGD → writeback through both backends and asserts byte-equal
  `abng_chain_head`.
* Workspace lib regression: 2,198 → 2,213 lib tests pass, 0 failures.
* `physics_ml` 107/107 still pass — Phase 0.3a's cjc-ad dependency
  doesn't perturb the existing PINN suite.

### Roadmap (revised after 0.3a)

| Phase | Scope |
|-------|-------|
| 0.3b  | Per-leaf BLR head (Bayesian linear regression on penultimate features for epistemic σ²) |
| 0.3c  | OOD scoring + calibration bins + drift detector |
| 0.3d  | Structural decisions (Grow/Split/Merge/Prune/Compress/Freeze) — needs evidence from 0.3b/c |
| 0.4   | `cjcl abng …` CLI + JSON snapshot + log compaction |
| 0.5   | Chess RL retrofit — value head first, then policy head |

---

## Phase 0.3d Amendment (2026-05-07)

Phase 0.3d shipped the **structural-decision engine** across five
sub-steps. Snapshot magic walked `\x05 → \x06 → \x07 → \x08`; the
audit-tag block grew from 16 (`0x00..0x0F`) to 24 (`0x00..0x17`);
dispatch surface grew from 49 to 65 builtins.

### 0.3d-1 — `Maturity` + `NodeSignature` (lazy, snapshot v5 unchanged)
Lazy types computed on-demand from existing per-node state.
`Maturity { samples_seen, calibration_stable, uncertainty_stable,
trust_level }` with stub stability flags; `NodeSignature` with 4 ×
8B profile hashes (prediction / uncertainty / calibration / routing).
Two read-only inspection builtins (`abng_node_maturity`,
`abng_node_signature`). Zero invariant changes.

### 0.3d-2 — `expected_epistemic` capture, calibrated OOD (v6, audit `0x17`)
Added per-node `expected_epistemic: Option<f64>` field. Manual
install via `abng_set_expected_epistemic(g, node_id, value)`;
revised `ood_score` to use `(epi / expected).clamp(0, 1)` when
captured. Closes architecture-doc gap §8.1's manual half. Snapshot
bumped `\x05 → \x06`. New audit kind `0x17 ExpectedEpistemicCaptured`
deliberately skips `0x10..0x16` to keep the structural-action block
contiguous.

### 0.3d-3 — `DecisionPolicy` + 6 force-* + `Dense` (v7, audits `0x10..0x15`)
* `DecisionPolicy` with 11 thresholds, install-anytime (one-shot,
  unlike other `set_*`).
* 6 force-* graph methods + builtins for direct testing of each
  structural mutation: `abng_force_grow`, `abng_force_split`,
  `abng_force_merge`, `abng_force_prune`, `abng_force_compress`,
  `abng_force_freeze`. Plus `abng_is_frozen` and `abng_action_count`.
* Per-node `is_frozen` + `is_active` flags persisted.
* Per-graph `action_counts: [u64; 6]` cross-checked on replay.
* New `AdaptiveChildren::Dense { signature: [u8; 32] }` variant
  (kind code `5`); descendants of a compressed sub-tree persist in
  the arena (per the never-reorder-pushes invariant) but become
  unreachable through `descend`.
* Six new audit kinds at `0x10..0x15`. Full payloads (not hash
  witnesses) — these events drive replay topology reconstruction.
* Snapshot bumped `\x06 → \x07`.

### 0.3d-4 — `decide_step` engine + Unfreeze (v8, audit `0x16`)
* `abng_decide_step(g) -> Tensor[6]` — one-pass policy engine.
  Iterates nodes in `NodeId` ascending order over an arena snapshot
  taken at call entry; fires at most one structural action per node
  in fixed fall-through order: Compress → Merge → Split → Prune →
  Grow → Freeze.
* Per-node persistent `last_signature: Option<[u8; 32]>` +
  `signature_stable_calls: u64` for stability tracking.
* `Maturity.calibration_stable` flips on `ECE < 0.05`;
  `Maturity.uncertainty_stable` requires BLR + samples ≥ 100 +
  signature stable for ≥ 1 decide_step call.
* Auto-capture of `expected_epistemic` inside `decide_step` when
  `Maturity.uncertainty_stable` first holds — closes architecture-
  doc gap §8.1's auto half.
* `abng_unfreeze` builtin + new audit kind `0x16 Unfreeze` (the
  drift-trip auto-unfreeze path remains a 0.4 deferral).
* Snapshot bumped `\x07 → \x08`.

### 0.3d-5 — proptest, bolero, decoder hardening, doc updates
* 4 properties × 256 cases each
  (`tests/prop_tests/abng_decision_props.rs`): replay byte-equality,
  decide_step monotonicity, density-score monotonicity in
  Mahalanobis distance, force_grow replay.
* 4 bolero fuzz targets (`tests/bolero_fuzz/abng_decision_fuzz.rs`):
  structural fuzz, numerical-bounds fuzz, tamper fuzz, observe-
  determinism fuzz.
* **Decoder hardening (security fix):** the tamper fuzz revealed a
  decoder vulnerability where untrusted length fields (`n_nodes`,
  `n_events`, `n_params`, tensor `numel`) drove `Vec::with_capacity`
  to attempt 100+ GB allocations on corrupt inputs. Replaced with
  `Vec::new()` (or bounded `with_capacity` against remaining cursor
  bytes). Replay is now panic-free under arbitrary byte flips.
* Architecture doc, this ADR, project memory, MEMORY.md, and
  Phase 0.4 prompt updated.

### Cumulative gate movement (Phase 0.3d)

| Gate | Pre-0.3d | Post-0.3d-5 | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 122 | **227** | +105 |
| `cargo test --test abng` | 175 | **303** | +128 |
| `cargo test --test prop_tests abng_decision` | (n/a) | **4 × 256 cases** | new |
| `cargo test --test bolero_fuzz abng_decision` | (n/a) | **4 targets** | new |
| `cargo test --workspace --release --lib` | 2,258 | **2,363** | +105 |
| `cargo test --test physics_ml --release` | 107 | **107** | 0 (canary unchanged) |

### Key architectural decisions deferred to Phase 0.4

The decision-engine triggers are deliberately simplified in 0.3d-4.
Phase 0.4 will refine to the full prompt-spec criteria — see
architecture doc §8.9 / §8.10 / §8.11 for the deferral table.

### Roadmap (revised after 0.3d)

| Phase | Scope |
|-------|-------|
| 0.4   | `cjcl abng {train,inspect,explain,replay,diff}` CLI + Welford-smoothed signatures + KL-merge + ΔNLL split + drift-trip auto-Unfreeze + JSON snapshot + log compaction + categorical codebook + snapshot-version frozen for CLI lifetime |
| 0.5   | Chess RL retrofit — value head first, then policy head |

---

## Phase 0.4 amendment (2026-05-07) — Trigger refinement, audit fixes, CLI surface

Phase 0.4 split into three independent tracks once the post-0.3d
audit findings clarified — Track C (audit fixes, ships first), Track B
(trigger refinement to prompt-spec form), Track A (CLI, the remaining
piece). Tracks B and C are SHIPPED; Track A is the next concrete
deliverable.

Snapshot magic walked `\x08 → \x09 → \x0A`; the audit-tag block grew
from 24 (`0x00..0x17`) to 26 (`0x00..0x19`); dispatch surface grew
from 65 to 67 builtins.

See [[Showcase|`docs/abng/PHASE_0_4_DESIGN.md`]] for the post-hoc
design note covering all decisions.

### Track C — post-0.3d audit fixes (7 items)

The independent post-0.3d audit produced 12 findings; the 7 highest-
priority items shipped as Track C before any trigger refinement.

* **C-2.3.1** — BLR `predict()` rename → `epistemic_leverage`. The
  middle tuple slot is dimensionless leverage (φᵀΛ⁻¹φ), not variance
  in y-units; pre-0.4 docs misnamed it `epistemic_var`. Rename + doc
  only; no math change, no snapshot bump.
* **C-2.3.2** — NaN/Inf input rejection at four boundaries: `observe`,
  `density_observe`, `calibration_observe`, `blr_update`. New error
  variants `GraphError::NonFiniteInput`, `BlrError::NonFiniteInput`.
  No snapshot bump.
* **C-2.3.3** — Replay semantic invariants. 4 new `DecodeError`
  variants: `SeqNonMonotonic`, `MissingCreatedEvent`, `EpochMismatch`,
  `StatsVersionMismatch`. Adversarial blobs with consistent hash
  chains but reordered seqs / missing Created / forged epochs now
  error specifically. No snapshot bump.
* **C-2.3.4** — BLR `b<ε` clamp audit event. `BlrState::update`
  returns `Result<Option<f64>, BlrError>`; on clamp,
  graph-layer `blr_update` emits `0x18 BlrNumericalRescue { reason:
  u8, b_pre_clamp_bits: u64 }` (9-byte body). Replay treats it as
  no-op (diagnostic-only). New audit tag `0x18`; no snapshot bump.
* **C-2.3.5** — MLP/BLR `feature_version_hash`. `BlrState` carries
  `[u8; 32]` stamped from the per-node MLP params hash at every
  BLR-init site. `blr_update` rejects with
  `BlrError::FeatureVersionStale { stored, current }` when current
  params hash differs. New builtin `abng_reset_blr` clears posterior
  to prior + refreshes fvh. **Snapshot bump `\x08 → \x09`.**
* **C-2.3.6** — `abng_leaf_set_params_batch` + new audit kind
  `LeafParamsUpdatedBatch` at tag `0x19`. Atomic — if any tensor's
  count or shape is wrong, params are unchanged and no event
  appended. 6× event reduction on a 2-layer head's optimizer step.
  New audit tag `0x19`; no further snapshot bump.
* **C-2.3.7** — "per-leaf" → "per-node" doc rename. Code was always
  per-node; docs were wrong. Doc-only.

### Track B — trigger refinement to prompt-spec form (7 items)

The prompt's §2.3 trigger spec listed rich criteria; Phase 0.3d-4
shipped defensible single-threshold simplifications. Track B replaced
each with the spec form.

The compute-only items B-2.2.{3,4,5,6} shipped before the wire-format
items B-2.2.{1,2,7} so each PR could land independently. B-2.2.7
bumped magic; B-2.2.{1,2} resequenced *after* B-2.2.7 and absorbed
their per-node state additions into the same v10 bump in-place.

* **B-2.2.6** — NIG-aware merge math: `BlrState::combine(&mut self,
  other, prior)` (sum precisions, precision-weighted-mean of means,
  `(a, b)` with prior subtract) + `NodeStats::combine` (Chan/Golub/
  LeVeque parallel Welford merge). `force_merge` and replay-side
  `apply_event(Merge)` both call `combine` before deactivating
  absorbed. No snapshot bump (combines existing fields). Emits an
  extra `BlrUpdated` (tag `0x0A`) witness on the `into` node so
  replay re-validates `into`'s post-combine state.
* **B-2.2.3** — KL-divergence gate for Merge. `BlrState::kl_divergence`
  (closed-form for d-D Gaussian + scalar IG). `try_merge` requires
  both Hamming ≤ τ_merge AND posterior `KL ≤ kl_merge`. No snapshot
  bump.
* **B-2.2.5** — Route-entropy gate for Grow.
  `route_key_entropy_at_candidate_depth`. `try_grow` requires both
  `samples_seen ≥ grow_min` AND route-key entropy at candidate depth
  > `H_grow`. No snapshot bump.
* **B-2.2.4** — Bootstrap held-out ΔNLL gain for Split.
  `estimate_split_nll_gain` uses synthetic Gaussian sampling from the
  BLR posterior (deterministic via SplitMix64-derived seed).
  `try_split` requires both `samples_seen ≥ split_min` AND ΔNLL gain
  ≥ `nll_split_gain`. No snapshot bump.
* **B-2.2.7** — Drift-trip auto-Unfreeze. When a frozen node has both
  a density tracker and a drift baseline, and `drift_score(current
  density) > drift_unfreeze`, `decide_step` calls `unfreeze` before
  the regular ladder. New 12th `DecisionPolicy.drift_unfreeze`
  threshold (default `f64::MAX` keeps the gate disabled).
  **Snapshot bump `\x09 → \x0A`; `DecisionPolicy` 88B → 96B.**
* **B-2.2.2** — 3-window ECE/σ stability ring buffers per node.
  `ece_history: [f64; 3]`, `ece_fill_count: u8`, `sigma_history:
  [f64; 3]`, `sigma_fill_count: u8`. `Maturity.calibration_stable`
  flips on `max(ece_history) - min ≤ ECE_STABILITY_MAX_DELTA`;
  `Maturity.uncertainty_stable` flips on `max(sigma_history) / min ≤
  SIGMA_STABILITY_RATIO`. Per-node +50B (absorbed into v10 in place).
* **B-2.2.1** — Welford-smoothed `NodeSignature` profiles per node. 4
  × `SignatureWelford { n: u64, mean: f64, m2: f64 }` channels. The
  Welford fold makes `signature_stable_calls` lenient — small
  post-stability observations no longer reset the counter. Per-node
  +96B (absorbed into v10 in place).

### Decisions worth recording (Phase 0.4)

1. **Two magic bumps in Phase 0.4 (`\x08 → \x09 → \x0A`) — but v10
   absorbs both Stage B state additions in-place.** The "single bump
   per phase" goal slipped because C-2.3.5 (audit fix) shipped before
   B-2.2.7 (feature). Mid-phase resequencing of B-2.2.{1,2} after
   B-2.2.7 made the second bump absorb their per-node state additions
   too — so v10 is the final magic for Phase 0.4 and Track A extends
   it in place via audit tags `0x1A..0x1C`. Per-node
   `provenance_stamp_hash` is deferred to Phase 0.5.
2. **Track C ships before Track B.** Audit fixes have priority over
   feature work — building trigger refinement on top of unsound
   primitives compounds the problem. Order: C-2.3.{1..7} → B-2.2.{6,
   3, 5, 4, 7, 2, 1}.
3. **NIG-aware merge math is a primitive, not an algorithm.**
   `combine` lives in `blr.rs` / `stats.rs` alongside `update` /
   `predict` / `kl_divergence`; the *decision* of whether to call it
   is in `try_merge`. This separation matches the 0.3d-3 D4 principle
   ("force-* mutations have minimal semantics").
4. **Welford signatures shift `signature_stable_calls` from strict to
   lenient.** Pre-0.4, any observation reset stability to zero. The
   Welford fold makes the signature change gradually — only when the
   running mean shifts beyond Hamming sensitivity does the counter
   reset. The wire shape (`[u8; 32]`) is unchanged; the read
   implementation is what changed.
5. **Audit tags `0x18` / `0x19` are opt-in.** `BlrNumericalRescue`
   only fires on a clamp event (pathological data); `LeafParamsUpdated
   Batch` only fires when callers explicitly use the batch builtin.
   Healthy training snapshots from pre-0.4 contain neither tag and
   replay byte-identical through the new code paths. This discipline
   is what allowed shipping under v9/v10 instead of bumping again
   for these tags.

### Surface

* **Builtin count:** 65 → 67 (`abng_reset_blr` from C-2.3.5,
  `abng_leaf_set_params_batch` from C-2.3.6).
* **Audit tags:** 24 → 26 (`0x18 BlrNumericalRescue`,
  `0x19 LeafParamsUpdatedBatch`).
* **Snapshot magic:** `\x08 → \x09 → \x0A`.
* **`DecisionPolicy`:** 11 thresholds (88B) → 12 thresholds (96B).
* **Per-node state:** +50B (ECE/σ ring buffers) +96B (Welford × 4).
* **8 new test files** (see PHASE_0_4_DESIGN.md test-surface table).

### Cumulative gate movement (Phase 0.4 B+C)

| Gate | Pre-0.4 (end-of-0.3d-5) | Post-0.4 B+C | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 227 | **252** | +25 |
| `cargo test --test abng` | 303 | **391** | +88 |
| `cargo test --test prop_tests abng_decision` | 4 × 256 cases | 4 × 256 cases | +0 (re-tuned for 12 thresholds) |
| `cargo test --test bolero_fuzz abng_decision` | 4 targets | 4 targets | +0 |

### Roadmap (revised after 0.4 B+C)

| Phase | Scope |
|-------|-------|
| 0.4 Track A | `cjcl abng {train,inspect,explain,replay,diff}` CLI + JSON snapshot view + log compaction + audit tags `0x1A..0x1C` (StatsSnapshot, Routed, ProvenanceStamped) extending v10 in place |
| 0.5   | Per-node `provenance_stamp_hash` (deferred from 0.4 to keep v10 frozen); audit findings polish (C-2.3.{8,9,10,11,12} if not in Track A); Chess RL retrofit (value head first, then policy head); Maturity constants promoted to `DecisionPolicy` thresholds |

---

## Phase 0.4 Track A amendment (2026-05-07)

Phase 0.4 Track A shipped the user-facing CLI surface and the
supporting builtins / audit kinds in 5 sequential PRs (G3.0+1+3+4 →
G3.5 → G3.6 → G3.7 → G3.8). All extensions land **in-place under
snapshot magic `\x0A`** — no further bump in Phase 0.4.

### Surface

* **CLI subcommands (5/5 shipped):**
  * `cjcl abng inspect <model.snap> [--node ID] [--audit] [--stats]
    [--tree] [--json]` — read-only viewer; validates audit chain on
    load; drill-in flags for per-node detail / audit histogram /
    Welford stats / arena topology; hand-rolled JSON output.
  * `cjcl abng replay <model.snap> [--verify] [--json]` — wrapper
    around `cjc_abng::serialize::replay`; reports `DecodeError`
    cleanly with exit 1; `--verify` additionally calls
    `verify_chain()`.
  * `cjcl abng diff <a.snap> <b.snap> [--json]` — chain_head +
    n_nodes + n_events + action_counts + per-node
    `stats_chain_head` diff. Exits 1 when chain heads differ.
  * `cjcl abng explain <prediction.snap> [--model <model.snap>]
    [--json]` — reads prediction-snapshots produced by
    `abng_predict_snap`, reports lineage hashes, predicting node id
    + BLR `n_seen`, prediction tuple, and a categorical
    abstain-or-trust verdict (UNCALIBRATED / LOW EVIDENCE /
    OOD SATURATED / SUPPORTED). Optional `--model` verifies
    `chain_head` + lineage hashes match the model snapshot.
  * `cjcl abng train [OPTIONS] --out <PATH>` — deterministic
    SplitMix64-seeded driver loop (observe N times → `decide_step`
    every K obs → serialize). Phase 0.4 ships the explicit-flag
    config (`--seed`, `--n-obs`, `--obs-seed`, `--decide-every`,
    `--max-decide`); TOML `--config` files defer to Phase 0.5.

* **Audit kinds added (2 new, opt-in, both extend v10 in place):**
  * `0x1A StatsSnapshot { node_id: u32, stats_hash: [u8; 32] }` —
    log-compaction marker. Phase 0.4 emits the marker only;
    smart-replay (fast-forward past `*Updated` runs) defers to
    Phase 0.5. `apply_event` is a pure no-op.
  * `0x1B Routed { leaf: u32, matched_prefix: u8 }` — opt-in
    descend trace event. Untraced `descend()` remains silent;
    `descend_traced()` emits one event per call. Replay no-op.

* **Builtins added (3 new, dispatch.rs surface 69 → 72):**
  * `abng_descend_traced(g, prefix_tensor) -> Tensor[2]` — the
    Routed-event-emitting variant of `abng_descend`.
  * `abng_predict_snap(g, node_id, phi: Tensor[d]) -> Bytes` — pack
    a predict + lineage tuple into a `b"ABNG-PRED\x01"`-prefixed
    byte blob. Drives `cjcl abng explain`.
  * `abng_compact_log(g, until_seq) -> Int` — emit one
    `StatsSnapshot` per distinct node touched in `[0, until_seq)`,
    in `NodeId`-ascending order. Returns count emitted.

* **New module:** `cjc_abng::predict_snap`. Defines `PRED_MAGIC`
  (`b"ABNG-PRED\x01"`), `PredictionSnap` struct, `pack`/`unpack`
  free functions, and `PredictionSnapError` (Truncated / BadMagic /
  ShortBody / SuspiciousPhiDim — defensive bounds-check discipline
  matches the 0.3d-5 model-snapshot decoder hardening).

### Decisions worth recording (Track A)

1. **`cjcl abng` is a CLI suite command** in the existing
   `cjc-cli/src/lib.rs` infrastructure — uses the same hand-rolled
   argument parser as `cjcl inspect` / `cjcl trace` / etc. Zero
   external deps; the suite handler in lib.rs forwards
   subcommand-and-after to the abng module's own dispatcher.
2. **Two distinct snapshot magics now coexist**: `b"ABNG\x0A"` for
   model snapshots, `b"ABNG-PRED\x01"` for prediction snapshots.
   Distinct length + prefix means the `serialize::replay` decoder
   rejects prediction snapshots with `DecodeError::BadMagic` and
   the `predict_snap::unpack` decoder rejects model snapshots with
   `PredictionSnapError::BadMagic { got: ... }` — neither can be
   silently confused for the other.
3. **JSON output is hand-rolled** in every subcommand. cjc-cli has
   zero external dependencies; adding `serde_json` for one feature
   would break the contract. The format is intentionally simple
   and stable — downstream tooling can rely on the field set.
4. **`cjcl abng train` defaults match the canary fixture** — same
   seed (42), same codebook (1×4), same head (1→[2]→1 tanh), same
   BLR prior, same DecisionPolicy. This means a default training
   run produces a snapshot whose `chain_head` is reproducible by
   reading the canary's locked-in hex. Surfaces drift between the
   canary fixture and the train default if either changes
   independently.
5. **TOML `--config` deferred** because cjc-cli has no TOML parser.
   Adding one is a feature in itself; explicit flags cover the v1
   training surface adequately. Phase 0.5 adds a hand-rolled
   minimal TOML parser alongside the configurable Maturity
   constants and `provenance_stamp_hash` work.
6. **`StatsSnapshot` ships marker-only.** Smart-replay that uses
   it to fast-forward past `*Updated` runs is significantly more
   complex (needs a new replay state machine) and was deferred to
   Phase 0.5. The marker emission is deterministic and replay-safe
   today; smart-replay is purely an optimization on top.
7. **`descend_traced` is opt-in.** A regular `descend` call must
   not emit Routed (would explode the audit log under heavy
   inference). Callers explicitly choose the traced variant when
   they want a route witness in the chain — typically only
   `cjcl abng explain` workflows need this.

### Cumulative gate movement (Phase 0.4 Track A)

| Gate | Pre-Track-A | Post-Track-A | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 252 | **261** | +9 (predict_snap unit) |
| `cargo test --test abng` | 419 | **442** | +23 (route_trace +13, compact_log +10) |
| `cargo test --test prop_tests abng_decision` | 4 × 256 cases | 4 × 256 cases | +0 |
| `cargo test --test bolero_fuzz abng_decision` | 4 targets | 4 targets | +0 |
| `cargo test -p cjc-cli --test abng_cli_integration` | (n/a) | **32 passed** | new |

### Roadmap (revised after Phase 0.4 complete)

| Phase | Scope |
|-------|-------|
| 0.5   | Per-node `provenance_stamp_hash` + `0x1C ProvenanceStamped` audit kind (forces v10 → v11); configurable Maturity constants (`ECE_STABILITY_MAX_DELTA`, `SIGMA_STABILITY_RATIO` → DecisionPolicy thresholds 13 + 14, also forces v11); `unfreeze_count` observability (extends `action_counts` to `[u64; 7]` OR adds a separate field, also v11). Smart-replay using `StatsSnapshot` to fast-forward; TOML config files for `cjcl abng train`; `NodeStats::canonical_bytes` 24B → 32B (Kahan compensation for full compaction support — also v11). All four v11-bump items consolidate into one magic bump. |
| 0.5+  | Chess-RL retrofit — value head first (uncertainty-gated bootstrap), then policy head; end-to-end determinism gate against existing chess-rl-v2 weight hashes. |

---

## Phase 0.5 amendment (2026-05-08)

Phase 0.5 shipped on the `claude/abng-phase-0-5` branch (10 commits)
and merged into master at commit `fe0b602`. Net surface delta from
Phase 0.4: snapshot magic bumped `\x0B → \x0C` (v12), 73 → 75
builtins, 28 → 29 audit kinds, `NodeStats::canonical_bytes` 24B →
32B, new prediction snapshot magic `b"ABNG-PRED\x02"`. Plus an
extensive demo layer that proves 11 distinct ABNG capabilities at
small scale.

### What landed

**Joint v12 wire-format bump (Items 1 + 4 — single commit `1c92ab0`):**

* **Item 1 — Per-node `provenance_stamp_hash` + `0x1C
  ProvenanceStamped`.** Every node carries a 32-byte caller-chosen
  fingerprint (typically `sha256(dataset_bytes ‖ feature_version)`).
  The `stamp_provenance` graph method writes the field and emits a
  `ProvenanceStamped` event into the audit chain. Idempotent for
  same-hash re-stamps. Two new builtins:
  `abng_stamp_provenance(g, node_id, hex_string)` and
  `abng_provenance_stamp(g, node_id) -> hex_string`. Predict snap
  (PRED_MAGIC v2) absorbs a trailing `provenance_stamp_hash`. New
  `DecodeError::ProvenanceMismatch` fired when the per-node section's
  stored stamp disagrees with what event replay produces.
* **Item 4 — `NodeStats::canonical_bytes` 24B → 32B.** Append the
  Kahan compensation register's bit pattern so log compaction can
  resume the full Welford state from canonical bytes alone.
  Compensation-register bit-stable for fixed observation order. New
  `KahanAccumulatorF64::compensation_bits()` accessor +
  `from_components(sum, comp, count)` constructor in `cjc-repro`.

**Item 2 — `smart_replay` API + StatsSnapshot consistency check
(commit `ca2aa18`):** `smart_replay(bytes)` and
`replay_with_options(bytes, opts)` plus a tamper-detection layer:
the snapshot's payload `stats_hash` field must equal the event-level
`stats_hash` (compact_log writes both from the same source). New
`DecodeError::StatsSnapshotMismatch`. **The cycle-saving fast-forward
layer is explicitly deferred to Phase 0.6 Item 3.**

**Item 3 — TOML `--config` files for `cjcl abng train` (commit
`b2c48bf`):** Hand-rolled minimal TOML parser in
`crates/cjc-cli/src/toml_min.rs` (~600 LOC, 29 tests) — zero
external deps to preserve cjc-cli's published-package contract.

**Item 5 — Chess RL v2.6 ABNG retrofit scaffold (commit `9568e29`):**
Sibling test file `tests/test_chess_rl_v2_6_abng.rs` (19 tests)
demonstrating the API pattern. The full v2.5 PRELUDE rewrite was
deferred — v2.5's chess RL fleet (97/97 passing) stays unchanged.
Two locked canary hashes:
* `V2_6_CHAIN_HEAD = 27d547b8…6847b`
* `V2_6_BLR_STATE_HASH = 869b32bd…fabc`

**Application demos (Rust API + CJC-Lang sibling for each):**

| Demo | Rust tests | CJC-Lang tests | Locked Rust canary |
|---|---:|---:|---|
| PINN per-region uncertainty (`tests/test_abng_pinn_uncertainty*`) | 13 | 9 | `30d333f1…e468d` |
| Tabular GP-like (`tests/test_abng_tabular_gp*`) | 10 | 9 | `cd3f5c7b…e87e6` |
| Lineage attestation (`tests/test_abng_lineage_attestation*`) | 16 | 9 | `789acce7…06c2` |

The lineage demo also locks `DATASET_A_FINGERPRINT = 3e85d52f…3634`.

**Six capability-only CJC-Lang demos (close the demo gap from Phase
0.4):**

| Capability | Demo file | Tests | Locked canary |
|---|---|---:|---|
| OOD detection composite | `test_abng_ood_detection_cjcl.rs` | 9 | `85970ca5…533e` |
| Adaptive structural triggers | `test_abng_adaptive_triggers_cjcl.rs` | 9 | **`d064fb08…7807` (matches Rust `decide_step_canary` byte-for-byte)** |
| Calibration / ECE | `test_abng_calibration_cjcl.rs` | 10 | `4c625f08…5a25` |
| Distribution-drift detection | `test_abng_drift_detection_cjcl.rs` | 9 | `a3a41c5b…8121` |
| Log compaction | `test_abng_compact_log_cjcl.rs` | 7 | (parity-checked, no separate canary) |
| Maturity inspection | `test_abng_maturity_inspection_cjcl.rs` | 10 | `c7b92726…928b` |

### Surface

* **Snapshot magic:** `b"ABNG\x0C"` (v12)
* **Builtin count:** 75 (`abng_stamp_provenance` and
  `abng_provenance_stamp` added in 0.5)
* **Audit kinds:** 29 (tags `0x00..0x1C` — `0x1C ProvenanceStamped`
  added)
* **NodeStats::canonical_bytes:** 32 bytes (was 24)
* **PRED_MAGIC:** `b"ABNG-PRED\x02"` (was `\x01`)

### Cumulative gate movement (Phase 0.5)

| Gate | Pre-0.5 | Post-0.5 | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 261 | **275** | +14 |
| `cargo test --test abng` | 442 | **460** | +18 |
| `cargo test --test prop_tests abng_decision` | 4 × 256 | 6 × 256 | +2 properties |
| `cargo test --test bolero_fuzz abng_decision` | 4 | 8 | +4 fuzz targets |
| `cargo test -p cjc-cli --test abng_cli_integration` | 32 | **43** | +11 |
| `cargo test -p cjc-cli --lib toml_min` (NEW) | — | **29** | +29 |
| 4 Rust application demos (NEW) | — | 19 + 13 + 10 + 16 = **58** | +58 |
| 9 CJC-Lang demos (NEW) | — | 9+9+9+9+9+10+9+7+10 = **81** | +81 |

**Net new tests added by Phase 0.5: ~213 across 13 new demo files
plus the integration / unit / property / fuzz extensions. Total
project tests on merged master: 3,683 across 13 separate suites,
zero failures.**

### Decisions worth recording (Phase 0.5)

1. **The `provenance_stamp_hash` is per-node, not per-graph.**
   Each node can carry its own fingerprint. Supports federation /
   fine-tuning workflows where different leaves are trained against
   different dataset slices.
2. **`smart_replay`'s cycle-saving optimization is deferred to
   Phase 0.6.** Phase 0.5 ships only the API + StatsSnapshot
   consistency check. The full optimization requires relaxing the
   per-event `stats_hash` check for fast-forwarded events; that's
   a structural refactor better landed in its own phase.
3. **Demos in CJC-Lang source serve dual purpose.** They validate
   the language-level surface AND give AST↔MIR parity coverage for
   ~30 builtins by construction.
4. **The chess RL v2.6 retrofit is intentionally a scaffold, not
   a benefit demonstration.** v2.5's bottleneck is interpreter
   throughput, not ML algorithm. The retrofit's job is to lock the
   API contract.
5. **The adaptive demo's chain_head matches the Rust-side
   `decide_step_canary` byte-for-byte.** Same workload, three
   pipelines (direct Rust API, CJC-Lang AST, CJC-Lang MIR),
   bit-identical SHA-256 over 23+ events. Strongest cross-pipeline
   determinism gate the project supports.

### Honest gaps (preserved as Phase 0.6 work items)

* Cross-platform determinism CI not yet wired (canaries
  Windows-only).
* Performance benchmarks at scale not yet run (asymptotic claims
  unmeasured at n > 200).
* Smart-replay fast-forward optimization deferred.
* Adaptive triggers — only Merge demonstrated in a workload (5/6
  trigger types still unfired in the demo layer).
* Real-world case study (someone using ABNG to attest a real
  model) — not yet.

### Roadmap (revised after Phase 0.5 complete)

| Phase | Scope |
|-------|-------|
| 0.6 | Cross-platform determinism CI; performance benchmarks at scale (`bench/abng_*` family); smart-replay fast-forward optimization (Phase 0.5 Item 2 second half); native `batch_observe` + bulk BLR update (forces v13 bump); adaptive triggers — fire all 6 types in CJC-Lang demos; Phase 0.6 demos at scale + with noise (every Phase 0.5 demo gets a `_scaled` sibling); compiler / interpreter perf prep work (LICM, CSE, function inlining; native specialization for ABNG hot paths; AOT explicitly OUT OF SCOPE); TidyView-parity training pipeline foundation (profile + analysis + one demonstration kernel; multi-phase work begins here). See `docs/abng/PHASE_0_6_HANDOFF.md`. |
| 0.7+ | AOT compilation (`cjcl compile foo.cjcl → foo.exe`); destructive log truncation (Phase 0.6's compact_log markers + smart_replay enable this); training pipeline at TidyView-parity perf; real-world case study deployment. |
