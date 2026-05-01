# Phase 2 Audit — Deterministic Collection Policy for ML Metadata

**Status:** Design draft for Phase 2 of the deterministic ML training stack.
**Branched off:** master (does not depend on Phase 1 / 3a / 3b being merged).
**Author:** generated 2026-05-01 to position Phase 2 to ship fast once Phase 1 lands.

---

## TL;DR

| Metric | Count |
|---|---|
| Sites surveyed | 11 |
| **Already deterministic** (BTreeMap / sorted Vec / insertion-ordered) | 7 |
| **Migration candidates** for Phase 2 (typed-ID + IndexVec) | 3 |
| **Watch-list** (deterministic but fragile to future change) | 2 |
| **Hash-based or unkeyed maps in ML metadata paths** | **0** |

**Headline finding:** the codebase already practices deterministic-collection discipline — every ML-relevant map is a `BTreeMap`, an explicitly-sorted `Vec`, or an insertion-ordered append-only `Vec`. **No `HashMap` / `HashSet` / unsorted-Vec patterns were found in any ML metadata path.**

That changes the framing of Phase 2. The goal is **not** "fix nondeterminism" — it's **"introduce typed IDs and the right detcoll variant for each workload, replacing untyped `Vec<T>` and untyped `BTreeMap<String, T>` where the ergonomic / performance win justifies the churn."**

The biggest migration candidate is **Site 4 (GradGraph parallel arrays)** — three parallel `Vec<T>` indexed by `usize`. Migrating to `IndexVec<NodeIdx, ...>` adds compile-time type safety against passing a parameter ID where a node ID is expected. That's an ergonomics win, not a correctness win — but it's the kind of typing discipline that pays for itself once Phase 6's training manifest needs to record parameter IDs distinctly from node IDs.

---

## Policy reference (from the brief)

| Workload | Pick |
|---|---|
| Dense `Id → value` tables | `IndexVec<I, V>` |
| Tiny configs / metric maps (≤ ~16 entries) | `TinyDetMap<K, V>` |
| Canonical small sealed sorted maps (≤ ~10k) | `SortedVecMap<K, V>` |
| Mutable deterministic equality lookup | `DetOpenMap<K, V>` |
| Build-once / read-many `u64` lookup | `SealedU64Map<V>` / `DHarhtMemory` |
| Range queries / canonical sorted output | `BTreeMap<K, V>` |

Hot-path `u64` IDs are preferred for: vocab token IDs, feature IDs, dataset row IDs, shard IDs, parameter IDs, optimizer-state IDs, checkpoint hash indexes.

---

## Inventory

### A. Already deterministic — confirm and document

These sites already obey the policy. No code change needed; Phase 2's job is to **cite them as canonical examples** in the eventual ADR.

| Site | File:line | Current | Workload | Notes |
|---|---|---|---|---|
| **byte_dict / lookup** | `cjc-data/src/byte_dict.rs:480` | `BTreeMap<Vec<u8>, u64>` | Build-once + optional sealed `DHarht` accelerator | Optional `seal_for_lookup()` graduates to `DHarht`; optional `seal_with_u64_hash_index()` builds a `SealedU64Map<u64>`. **Already exemplary** — Phase 2 should propagate this pattern. |
| **byte_dict / code_to_view** | `cjc-data/src/byte_dict.rs:483` | `Vec<ByteStrView>` | Dense `code → bytes` | Insertion-ordered, deterministic by construction. Conceptually an `IndexVec<Code, ByteStrView>`. |
| **dict_encoding** | `cjc-data/src/dict_encoding.rs:13` | `BTreeMap<String, u32>` | Build-once string → code | Determinism contract verified in tests. |
| **DatasetPlan encodings** (Phase 1) | `cjc-data/src/dataset_plan.rs` | `SortedVecMap<String, EncodingSpec>` | Canonical small sealed map | Already migrated. Confirms the policy table is internalized. |
| **profile counters** | `cjc-runtime/src/profile.rs:106-108` | `BTreeMap<String, ZoneStats>` + `BTreeMap<i64, (String, Instant)>` | Range/canonical CSV output | Documented determinism contract. Ensures profiled and non-profiled runs produce bit-identical weight hashes (Chess RL v2.3). |
| **state_space ARENA** | `cjc-runtime/src/state_space.rs:79+` | thread-local `BTreeMap<usize, StateSpaceCell>` | Mutable equality lookup | ✓ Deterministic. **Phase 2 candidate (low priority): `DetOpenMap<CellIdx, StateSpaceCell>`** — would gain typed handles + hash-table speed without losing determinism. |

### B. Migration candidates — typed-ID + detcoll wins

These are the actionable Phase 2 work items, **ordered by leverage**.

#### Site 4 — GradGraph parallel arrays ★ HIGHEST LEVERAGE

- **File**: `crates/cjc-ad/src/lib.rs:354-356`
- **Current**:
  ```rust
  ops: Vec<GradOp>,
  tensors: Vec<Tensor>,
  param_grads: Vec<Option<Tensor>>,
  ```
- **Cardinality**: large (thousands of nodes per graph in PINN/transformer demos)
- **Mutation**: append-only during construction, mutated during backward
- **Why migrate**: every method on `GradGraph` takes a `usize` index. There's no compile-time prevention against passing a *parameter* index where a *node* index is expected, or vice versa. Phase 6's training manifest will need to record parameter IDs distinctly from arbitrary node IDs — typed IDs make that natural.
- **Recommended**: introduce `NodeIdx(u32)` newtype + `IndexVec<NodeIdx, GradOp>` etc. Three parallel `IndexVec`s preserve the existing layout and cache behavior; only the type signatures change.
- **Risk**: every `usize` parameter on `GradGraph` and every dispatch arm in `cjc-ad/src/dispatch.rs` flips. Phase 3a/3b's helpers (`arg_idx`, `arg_idx_checked`) need to return `NodeIdx` instead of `usize`.
- **Backward compat**: `Value::Int(i64)` representation at the `.cjcl` boundary stays unchanged — the typed ID is a *Rust-side* affordance.

#### Site 7 — `AdamState` moment buffers

- **File**: `crates/cjc-runtime/src/ml.rs:128-136`
- **Current**:
  ```rust
  pub struct AdamState {
      pub lr: f64, pub beta1: f64, pub beta2: f64, pub eps: f64, pub t: u64,
      pub m: Vec<f64>,
      pub v: Vec<f64>,
  }
  ```
- **Cardinality**: large (one entry per parameter; PINN demos use ~hundreds, real workloads use 10k-1M+)
- **Mutation**: read-many, mutated on every `adam_step` call
- **Note**: this is the *Rust-side* trainer state (used by baked-in trainers). The language-level `adam_step` builtin in `builtins.rs:1853` is separately stateless — it takes `m`/`v` Tensors as args and returns updated ones. Both code paths exist; only the Rust-side one is in scope here.
- **Why migrate**: the `m`/`v` Vecs are parameter-position-keyed by implicit convention. Any change in parameter registration order silently invalidates the state. A typed `IndexVec<ParamIdx, f64>` makes the contract explicit.
- **Recommended**: introduce `ParamIdx(u32)` newtype; migrate to:
  ```rust
  pub m: IndexVec<ParamIdx, f64>,
  pub v: IndexVec<ParamIdx, f64>,
  ```
- **Risk**: medium. `AdamState::new` callers and `adam_step(&mut [f64], ...)` signature change. Bisect-able by Site (ParamIdx isolated to the optimizer module first, then propagated).
- **Priority**: medium. Bundle with Site 4 (`NodeIdx`) and Site 8 (`LayerIdx`) so the typed-ID pattern lands as one consistent change across the AD + optimizer layers.

#### Site 8 — MLP layer specification

- **File**: `crates/cjc-ad/src/pinn.rs:66-69`
- **Current**: `Vec<DenseLayer>`
- **Cardinality**: small (2–10 layers)
- **Mutation**: build-once
- **Why migrate**: like Site 4 but lower-stakes. A typed `IndexVec<LayerIdx, DenseLayer>` clarifies that "layer index" is a distinct concept from "node index" (a `DenseLayer` *contains* node indices for its weights and biases).
- **Risk**: low — small surface, build-once.
- **Priority**: medium. Bundle with Site 4 so the `IndexVec` typing pattern lands as one consistent change.

### C. Watch-list — already-deterministic, future-fragile

Sites that are correct today but might silently lose determinism if extended carelessly. (Site 7 / `AdamState` was on this list in an earlier draft; it's been promoted to a Site B migration candidate above. Site 10 / SGD `velocity` is symmetric to Site 7 and should land in the same PR using the same `ParamIdx` newtype.)

#### Snap struct-field encoding (Site 9)

- **File**: `cjc-snap/src/encode.rs:656-760`
- **Current**: encodes struct fields in **alphabetical** order — already deterministic.
- **Risk**: Phase 6 (training manifest) will hash entire struct outputs. If anyone "optimizes" snap to use insertion order instead of alphabetical, every checkpoint hash changes and Phase 6 manifests stop validating across tool versions.
- **Mitigation**: Phase 6's ADR should pin alphabetical struct-field encoding as a determinism invariant, not an implementation detail.

---

## Recommendations for Phase 2 implementation

### Concrete code changes (one PR)

1. **Introduce `NodeIdx(u32)` newtype** in `cjc-ad`. Migrate `GradGraph::{ops, tensors, param_grads}` to `IndexVec<NodeIdx, _>`. Update all internal methods, dispatch helpers, and the satellite `cjc-ad/src/dispatch.rs`. **No `.cjcl`-facing change** — `Value::Int(i64)` stays.
2. **Introduce `LayerIdx(u32)` newtype** in `cjc-ad/src/pinn.rs`. Migrate `Vec<DenseLayer>` to `IndexVec<LayerIdx, DenseLayer>`.
3. **Introduce `ParamIdx(u32)` newtype** in `cjc-runtime/src/ml.rs`. Migrate `AdamState::{m, v}` and the `SgdState`/equivalent velocity buffer to `IndexVec<ParamIdx, f64>`.
4. **Add a determinism doc-comment** on every Vec-shaped ML metadata container that *isn't* migrated, noting why insertion order is sufficient and what would break if reordered.

### Design notes for later phases

- **Phase 5 (tensor pooling + optimizer state)**: persistent optimizer-state container must be `IndexVec<ParamIdx, OptState>`. Do not introduce `BTreeMap<String, OptState>` for "named parameter" support.
- **Phase 6 (training manifest)**: pin alphabetical struct-field encoding in `cjc-snap` as a manifest invariant.
- **Phase 6 (manifest)**: vocab/feature dictionaries should hash via `ByteDictionary::seal_with_u64_hash_index`, then the resulting `SealedU64Map<u64>` becomes the canonical hash input for the manifest field.

### Things Phase 2 should NOT do

- **Don't migrate `BTreeMap<String, _>` to `SortedVecMap<String, _>` blindly.** SortedVecMap is faster for ≤16 entries; `BTreeMap` is fine above that. Most ML-metadata sites are exactly the cardinality where measurement matters — don't churn without evidence.
- **Don't introduce `DHarhtMemory` for small dictionaries.** It's the right tool for build-once-read-many `u64` keys at thousand-plus scale. Adding it to a 4-entry MLP layer index just adds dependencies without gain.
- **Don't change `Value::Int(i64)` to a typed wrapper at the language boundary.** That violates HARD RULE #1 from the Phase 3c brief and breaks every existing builtin.
- **Don't add a `repro-ml` policy gate yet.** That's Phase 6 work and depends on the manifest format being stabilized.

---

## What this audit changes about Phase 2

Before this audit, Phase 2 was framed as *"apply the deterministic-collection policy to ML metadata."* That framing implies **finding and fixing** non-deterministic sites.

After the audit, the actual Phase 2 work is:

1. **Typed-ID newtypes** for `NodeIdx`, `LayerIdx`, `ParamIdx` (a few hundred lines, mostly mechanical).
2. **Migrate the four `Vec<T>` ML-metadata containers** at Sites 4, 7, 8, 10 to `IndexVec<TypedId, T>`.
3. **Doc-comments** on Vec-shaped containers explaining their determinism contract.
4. **A short ADR** (call it ADR-0022) recording the policy decisions for future contributors so the next person who adds `BTreeMap<String, Tensor>` somewhere has a doc to read.

That's **a 1-2-day PR**, not a multi-week refactor. The reason is the codebase already did the hard work — it just hasn't *named* the discipline yet.

---

## Site index (for traceability)

| ID | Tag | File:line | Status |
|---|---|---|---|
| 1 | byte-dict-lookup | `cjc-data/src/byte_dict.rs:480` | ✓ deterministic |
| 2 | byte-dict-code-to-view | `cjc-data/src/byte_dict.rs:483` | ✓ deterministic (IndexVec-shaped) |
| 3 | dict-encoding | `cjc-data/src/dict_encoding.rs:13` | ✓ deterministic |
| 4 | grad-graph-tape | `cjc-ad/src/lib.rs:354-356` | ★ migration candidate (typed-ID) |
| 5 | profile-counters | `cjc-runtime/src/profile.rs:106-108` | ✓ deterministic |
| 6 | state-space-arena | `cjc-runtime/src/state_space.rs:79+` | ✓ deterministic |
| 7 | adam-state | `cjc-runtime/src/ml.rs:128-136` | migration candidate (`ParamIdx`-typed `IndexVec`) |
| 8 | mlp-layer-vec | `cjc-ad/src/pinn.rs:66-69` | migration candidate (`LayerIdx`-typed `IndexVec`) |
| 9 | snap-struct-fields | `cjc-snap/src/encode.rs:656-760` | ✓ deterministic (lock invariant in Phase 6) |
| 10 | sgd-velocity | `cjc-runtime/src/ml.rs:107-117` | migration candidate (`ParamIdx`-typed `IndexVec`, bundle with Site 7) |
| 11 | dataset-plan-encodings | `cjc-data/src/dataset_plan.rs` (Phase 1) | ✓ deterministic |

---

## Phase 2 PR shape (proposal)

When Phase 1 (PR #3) merges and Phase 2 implementation begins, the PR should be:

- ~3 commits, ~600 LOC total
- One commit per typed-ID newtype (`NodeIdx`, `LayerIdx`)
- One commit adding doc-comments + the ADR-0022 draft
- Tests: a regression suite gating that `cjc-ad` and the demos produce bit-identical output before/after — the migration is type-only, no value should change

This audit document itself should be merged independently as a citation target for the implementation PR.
