# ADR-0022 — Typed Index Newtypes for ML Metadata

**Date:** 2026-05-01
**Status:** Accepted, partially deferred (Phase 2b pending)
**Related:** [[ADR-0019 Deterministic Collection Family]]
**Audit:** `docs/ml_training/PHASE_2_AUDIT.md`

## Context

The Phase 2 audit (`docs/ml_training/PHASE_2_AUDIT.md`) surveyed every
site in the codebase where ML-relevant metadata is stored in a map-
shaped or table-shaped structure. The headline finding was:

> 11 sites surveyed; 7 already deterministic; 0 use HashMap/HashSet
> in any audit-visible path. The codebase already practices
> deterministic-collection discipline. Phase 2 is therefore *not*
> "fix nondeterminism" — it's "introduce typed IDs and the right
> detcoll variant where the ergonomic / safety win justifies the
> churn."

The audit identified four migration candidates (Sites 4, 7, 8, 10 in
`PHASE_2_AUDIT.md`) and recommended introducing three typed-ID
newtypes (`NodeIdx`, `ParamIdx`, `LayerIdx`) to prevent silent
confusion between "which integer index is this?" — particularly
around `cjc-ad`'s `GradGraph` (parallel `Vec<T>` indexed by `usize`),
`cjc-runtime::ml::AdamState::{m, v}`, and `cjc-ad::pinn::DenseLayer`.

The audit estimated "1-2 day PR" for the full migration. On execution,
this estimate was **incorrect**: `cjc-ad/src/lib.rs` alone has 48
methods using `usize` and 255 internal index accesses. A holistic
migration is closer to a 3-4 day refactor with non-trivial weight-
hash regression risk.

## Decision

Split the Phase 2 work into two PRs:

### Phase 2a (this ADR's scope) — typed-ID infrastructure

Ships:

1. **Three `repr(transparent)` newtypes** in `crates/cjc-ad/src/idx.rs`:
   - `NodeIdx(u32)` — index into a `GradGraph` parallel array.
   - `ParamIdx(u32)` — index into a per-parameter optimizer-state buffer.
   - `LayerIdx(u32)` — index into a layered architecture description.
2. **Typed dispatch boundary** in `crates/cjc-ad/src/dispatch.rs`:
   - `arg_idx`, `arg_idx_checked` return `NodeIdx`.
   - `idx_value` takes `NodeIdx`.
   - Each dispatch arm converts `NodeIdx → usize` via `.index()` only
     at the `GradGraph` call site (since `GradGraph` itself is not yet
     typed).
3. **Documentation** (this ADR, plus the audit doc).

### Phase 2b (deferred) — bulk migration

Defers to a follow-up PR:

1. Migrating `GradGraph`'s 48 method signatures (`usize` → `NodeIdx`).
2. Migrating `GradGraph`'s ~255 internal `self.tensors[a]` accesses.
3. Migrating `cjc-ad::pinn::DenseLayer`'s `Vec<DenseLayer>` to use
   `LayerIdx` for layer indexing and `NodeIdx` for the contained node
   refs.
4. Migrating `cjc-runtime::ml::AdamState::{m, v}` and the SGD velocity
   buffer to `Vec<f64>` keyed by `ParamIdx` (or `IndexVec<ParamIdx,
   f64>` if the `IndexVec` primitive is lifted from `cjc-data` to
   `cjc-runtime` first).

### Why this split is justified

The audit's stated goal — "typed IDs at the boundary, no value should
change" — is achieved by Phase 2a alone. The dispatch layer is the
audit-visible surface (it's what user `.cjcl` code touches via
`Value::Int`); making it typed is the highest-leverage step. Phase 2b
extends the discipline inward but does not enable any new capability.

Splitting also dovetails with the open Phase 3a (#4) and 3b (#5) PRs,
which add new dispatch arms using the same `arg_idx_checked` helper.
Phase 2a's typed boundary lets those PRs gain typed nodes "for free"
on rebase; Phase 2b can land later without coordinating across all
five Phase 3 branches.

## Consequences

### Positive

- **Compile-time safety** at the language boundary: a `Value::Int`
  decoded into a `NodeIdx` cannot be passed to a method expecting
  `ParamIdx` (when Phase 2b lands those signatures).
- **Zero runtime overhead**: `repr(transparent)` over `u32` means the
  ABI is identical to plain `u32`. No allocation, no indirection.
- **Self-documenting API surface**: `fn arg_idx_checked() -> NodeIdx`
  conveys intent; `fn arg_idx_checked() -> usize` did not.
- **Backward-compatible at the language boundary**: `Value::Int(i64)`
  representation is unchanged. User `.cjcl` code is not affected.

### Negative

- **Phase 2b deferral** means `GradGraph` itself is still typed
  `usize`. Internal `cjc-ad` code can still confuse the three index
  kinds. The dispatch layer is the *only* place where the typing is
  enforced today.
- **Conversion noise** in dispatch arms: each arm now has
  `NodeIdx::from_usize(...)` and `.index()` calls bracketing the
  `with_ambient(|g| g.op(...))` call. This is intentional — it
  highlights the boundary — but reads as visual clutter until Phase
  2b removes the need for the conversions.
- **API doubling temptation**: a future contributor might be tempted
  to add `pub fn add_typed(a: NodeIdx, b: NodeIdx) -> NodeIdx`
  alongside the existing `pub fn add(a: usize, b: usize) -> usize`.
  This ADR rejects that — Phase 2b will *replace* the `usize` API,
  not duplicate it.

### Neutral

- The `IndexVec<NodeIdx, T>` storage migration (audit's literal
  recommendation) is *not* in either Phase 2a or Phase 2b scope. The
  audit's choice of `IndexVec` was an implementation detail; typed
  IDs at the API achieve the safety goal without requiring a cross-
  crate dependency on `cjc-data::detcoll`. If `IndexVec` is later
  lifted to `cjc-runtime` (a separate refactor), Phase 2c can revisit
  storage layout.

## Sites updated by Phase 2a (this PR)

- `crates/cjc-ad/src/idx.rs` (new file, ~150 LOC).
- `crates/cjc-ad/src/lib.rs`: +1 module declaration, +1 re-export.
- `crates/cjc-ad/src/dispatch.rs`: helpers re-typed; ~30 dispatch arms
  updated with `.index()` and `NodeIdx::from_usize` conversions.

## Sites deferred to Phase 2b

| Audit Site | File:line | Migration |
|---|---|---|
| 4 | `cjc-ad/src/lib.rs:354-356` | `GradGraph::{ops, tensors, param_grads}` storage + 48 method sigs |
| 7 | `cjc-runtime/src/ml.rs:128-136` | `AdamState::{m, v}` keyed by `ParamIdx` |
| 8 | `cjc-ad/src/pinn.rs:66-69` | `Vec<DenseLayer>` keyed by `LayerIdx`; `DenseLayer` fields keyed by `NodeIdx` |
| 10 | `cjc-runtime/src/ml.rs:107-117` | SGD velocity keyed by `ParamIdx` (bundle with Site 7) |

## Regression contract

Any future Phase 2b PR (or any PR touching the dispatch arms or the
new typed boundary) must demonstrate **byte-equal weight hashes**
across the existing flagship demos:

- `examples/physics_ml/pinn_heat_1d_pure.cjcl`
- `examples/ml_training/mlp_classifier_pure.cjcl` (Phase 3 demo, when
  it merges)

Any divergence is a regression; any divergence in Phase 2b would
indicate a real indexing bug introduced by the migration, since the
typed-ID change is supposed to be value-neutral.

## Acceptance criteria for this ADR

Phase 2a:

- [x] `crates/cjc-ad/src/idx.rs` defines `NodeIdx`, `ParamIdx`,
  `LayerIdx` with `repr(transparent)`, `from_usize`, `index`, and the
  full `Display` / `From<NodeIdx> for usize` set of conversions.
- [x] `crates/cjc-ad/src/dispatch.rs` `arg_idx_checked` returns
  `NodeIdx`; every dispatch arm uses `.index()` at the `GradGraph`
  call site and `NodeIdx::from_usize(...)` at the construction-method
  return site.
- [x] `cargo test -p cjc-ad` passes (86/86 — was 83/83 + 3 new
  `idx::tests`).
- [x] `cargo test --test physics_ml --release` passes (71/71 — no
  printed-output drift).
- [x] This ADR documents the Phase 2a/2b split rationale.
