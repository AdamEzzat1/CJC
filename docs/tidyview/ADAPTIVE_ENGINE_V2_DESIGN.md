# Adaptive TidyView Engine v2 — Design Note

**Status:** Proposed (2026-04-26)
**Author:** Stacked-roles team (Architect + Data-engine engineer + Determinism auditor)
**Scope:** Single feature surface — adaptive selection representations behind the existing public TidyView API.

---

## 1. Goal

Replace the single `BitMask`-backed selection inside `TidyView` with an **adaptive selection** that picks one of three deterministic representations based on result density:

| Mode             | When                                  | Backing                |
|------------------|---------------------------------------|------------------------|
| `Empty`          | `count == 0`                          | unit (no allocation)   |
| `All`            | `count == nrows`                      | unit                   |
| `SelectionVector`| `count < nrows / 1024` (sparse)       | `Vec<u32>` ascending   |
| `VerbatimMask`   | `count > 30% of nrows` (dense)        | existing `BitMask`     |
| `Hybrid`         | `nrows/1024 ≤ count ≤ 30%` (mid)      | **deferred** (see §6)  |

The mid band currently routes to `VerbatimMask` via an adapter trait so callers
already see the eventual `Hybrid` arm in the enum and never need to be patched
again when it lands.

## 2. Why this is safe to add now

Reconnaissance confirms the entire downstream pipeline funnels through one
predictable choke point: **20+ call sites** in `crates/cjc-data/src/lib.rs`
consume the per-view selection via `self.mask.iter_set()`. Every join,
group_by, distinct, set op, formatter, head/tail and filter pushdown lands in
this exact pattern. Therefore:

- A trait method `iter_indices(&self) -> impl Iterator<Item=usize>` covers
  every existing consumer with no semantic change.
- The existing `BitMask` becomes the backing store of the `VerbatimMask` arm
  unmodified — no behaviour change in the dense path.
- The new `SelectionVector` arm bypasses the bitscan when the result is sparse,
  which is exactly the regime where the bitscan is wasteful (≥99.9% zero
  words).

No public API changes. `TidyView::filter`, `TidyView::select`, `LazyView`,
`tidy_dispatch.rs`, and all 30 user-visible builtins keep their current
signatures.

## 3. Module layout

```
crates/cjc-data/src/
  adaptive_selection.rs   ← NEW: enum, trait, density classifier, ops
  lib.rs                  ← TidyView field type changes BitMask → AdaptiveSelection
  lazy.rs                 ← unchanged surface (still consumes via .iter_indices())
```

Public `BitMask` stays exported (it's already a documented type referenced by
test files); we add `AdaptiveSelection` alongside it.

## 4. Shared interface

```rust
pub trait SelectionRepr {
    fn nrows(&self) -> usize;
    fn count(&self) -> usize;
    fn contains(&self, row: usize) -> bool;
    fn iter_indices(&self) -> SelectionIndices<'_>;          // ascending
    fn intersect(&self, other: &Self) -> Self;
    fn union(&self, other: &Self) -> Self;
    fn materialize_mask(&self) -> BitMask;
    fn materialize_indices(&self) -> Vec<u32>;
    fn explain_selection_mode(&self) -> &'static str;
}
```

`AdaptiveSelection` implements `SelectionRepr` directly — no dyn dispatch on
the hot path. `intersect` and `union` are mode-mixing: the result picks the
mode best suited to the **output** density, not either input mode.

`SelectionIndices<'_>` is a custom iterator enum that wraps the per-arm
iterators, avoiding `Box<dyn Iterator>` and preserving inlining.

## 5. Density-aware construction

A single `AdaptiveSelection::from_predicate_result(words: &[u64], nrows: usize)`
constructor classifies once based on popcnt over the raw words and returns the
chosen arm:

```text
count = popcnt(words & tail_mask(nrows))

count == 0            → Empty
count == nrows        → All
count <  nrows/1024   → SelectionVector(extract via word.trailing_zeros() loop)
count >  3*nrows/10   → VerbatimMask(BitMask)
otherwise             → Hybrid (today: VerbatimMask via adapter)
```

**Determinism note:** The classifier uses pure integer arithmetic. No
floating-point thresholds. `nrows / 1024` is exact integer division, so the
boundary is bit-stable across platforms and runs.

## 6. Hybrid: deferred, but reserved

A roaring-style chunked container is the right answer for the mid band, but
shipping it correctly requires:
- chunk-level format negotiation (run-length / dense / sparse per chunk)
- chunk-level intersect/union with stable iteration order
- a dedicated fuzz target for cross-mode `intersect/union` agreement

We defer it to a follow-up (Adaptive v2.1) and route mid-band selections to
`VerbatimMask` today. The enum variant exists from day one so callers do not
need to be patched twice.

## 7. Optimizer impact

`lazy.rs::optimize` does not change. The existing 3-pass pipeline
(`merge_filters` → `push_predicates_down` → `eliminate_redundant_selects`) is
purely tree-form; selection representation is an execution-time concept. We
add **late materialization** as a property the executor inherits from the
adaptive enum: `AdaptiveSelection::All` skips the per-row test inside
`try_eval_predicate_columnar`, and `Empty` short-circuits to an empty
result before any column work.

A planned-but-not-shipped pass (`predicate_grouping_by_column`) is documented
in §10 as the natural place to land further optimization.

## 8. Determinism contract (preserved)

- Iteration order is **always ascending row index**, regardless of arm.
- Intersect/union of two selections produces the same `AdaptiveSelection`
  variant for the same input shapes (deterministic mode pick).
- No `HashMap`/`HashSet`. No FMA. No rayon. No `unsafe`.
- All `Vec<u32>` lengths and contents are bit-stable per (df, predicate, seed).

A new parity test asserts that for any predicate result, the four operations
(`iter_indices`, `count`, `materialize_mask`, `materialize_indices`) agree
across every arm.

## 9. Scope of changes

- **NEW** `crates/cjc-data/src/adaptive_selection.rs` (enum, trait impl,
  classifier, ops, iterator types)
- **EDIT** `crates/cjc-data/src/lib.rs`:
  - `TidyView.mask` field type: `BitMask` → `AdaptiveSelection`
  - The 20+ `self.mask.iter_set()` call sites become `self.mask.iter_indices()`
  - `try_eval_predicate_columnar` continues to produce raw `Vec<u64>` words,
    then funnels through `AdaptiveSelection::from_predicate_result`
- **NO change** to `lazy.rs`, `tidy_dispatch.rs`, public builtins, snapshot
  format, or any test file outside `tests/tidy_tests/`.

## 10. Test plan

`tests/tidy_tests/` gains:

- `test_adaptive_selection_unit.rs` — per-arm unit (Empty, All, Sparse,
  Dense, mid-band → adapter), per-method (count, contains, iter_indices,
  intersect, union, materialize_*).
- `test_adaptive_selection_property.rs` — proptest: for any
  `Vec<bool>`, the constructed AdaptiveSelection round-trips through every
  arm method and matches the equivalent `BitMask::from_bools` answer.
- `test_adaptive_selection_fuzz.rs` — bolero structural target on
  `(Vec<u8> bytes → mode-mixing intersect/union)`. Asserts no panic and
  ascending-order invariant.
- `test_adaptive_selection_determinism.rs` — same predicate against same
  df, ten repetitions, every selection bit-equal byte-for-byte.
- `test_adaptive_selection_join_adversarial.rs` — joins where one side is
  Sparse, the other Dense. Same `BTreeMap` join answer regardless.
- `bench/adaptive_selection_bench.rs` (Criterion harness, not in the
  workspace test gate) — sweeps density 0%–100% in 5% steps and reports
  mode-mix crossover. Used for design validation, not regression.

## 11. Acceptance gates

1. `cargo test -p cjc-data` ≥ baseline + new tests, zero failures.
2. `cargo test --test tidy_tests --release` ≥ baseline + new tests.
3. `cargo build -p cjc-data --release` clean.
4. `cargo test parity --workspace` unchanged (selection representation is
   internal — parity tests only see public API output).
5. Determinism: ten consecutive runs of the new determinism test produce a
   single hash.

## 12. Out of scope (explicit non-goals)

- JIT / LLVM lowering of expression kernels
- SIMD bitscan intrinsics (the `count_ones`/`trailing_zeros` calls already
  lower to popcnt/tzcnt on x86 via stdlib)
- Snapshot format change (selection is in-memory; snapshots already serialize
  by row enumeration)
- `Hybrid` chunked container (deferred to v2.1 per §6)
- Predicate fragment bytecode (out of scope for this pass — current columnar
  fast-path already covers the common shapes; revisit after density crossover
  data tells us which shapes still go row-wise)

---

## Cross-references

- `crates/cjc-data/src/lib.rs:1858` — `BitMask` (becomes `VerbatimMask` backing)
- `crates/cjc-data/src/lib.rs:2005` — `TidyView` struct (single field changes)
- `crates/cjc-data/src/lib.rs:2024` — `try_eval_predicate_columnar` (sole producer site)
- `crates/cjc-data/src/lazy.rs:21` — `ViewNode` (unchanged)
- `crates/cjc-data/src/lazy.rs:195` — `optimize` pipeline (unchanged)
