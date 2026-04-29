---
title: "ADR-0018: Deterministic Adaptive Dictionary Engine"
tags: [adr, data, tidyview, categorical, determinism, dictionary]
status: Accepted
date: 2026-04-28
---

# ADR-0018: Deterministic Adaptive Dictionary Engine

## Status

Accepted (Phase 1 of TidyView v3, shipping in v0.1.7-dev).

## Context

CJC-Lang's existing categorical surfaces — `Column::Categorical` (in
`cjc-data`) and `FctColumn` (in `cjc-runtime`'s forcats) — store category
values as `Vec<String>` plus a `HashMap<String, u32>` lookup. That works
for small dictionaries today, but it has three fault lines that show up
the moment v3 needs cardinality-aware joins, group_by, and chunked
streaming:

1. **`HashMap` is non-deterministic.** Iteration order depends on
   `RandomState`, which means the same insertion sequence can produce
   different dictionary orderings on different runs. v3 verbs (cat-aware
   group_by, chunked spill-resume) need stable code assignment to
   guarantee bit-identical output across runs and machines.
2. **`String` is owned-UTF-8.** Every interned value pays one allocation
   plus a length header per row of distinct text, even when dictionaries
   share long substrings or hold short tokens. Spill-to-disk in Phase 6
   wants a flat byte arena with stable `(offset, len)` handles, not a
   forest of `String`s.
3. **`u32` codes are pre-committed.** Below 256 distinct categories, two
   thirds of the code byte is wasted; above 2³², the dictionary cannot
   represent the column at all. v3 chunks must be able to widen
   in-flight without re-encoding everything from row zero.

Phase 1 introduces a *parallel* engine — a fresh module
`cjc-data::byte_dict` — that fixes all three. It does **not** replace
`Column::Categorical` or `FctColumn` (those stay for backwards compat).
Phase 2 will migrate TidyView verbs that benefit (group_by, joins,
distinct) onto the new types; Phase 1 ships only the standalone core
plus tests.

## Decision

Adopt a **byte-first, BTreeMap-backed, lazily-promoted dictionary** as
the primary representation for new categorical work. Concretely, six
new public types in `crates/cjc-data/src/byte_dict.rs`:

- `ByteStringPool` — append-only `Vec<u8>` arena. Each interned byte
  sequence gets a stable `ByteStrView { offset: u32, len: u32 }` handle.
  No `String` in the hot path; embedded NULs and arbitrary high bytes
  are first-class.
- `ByteStrView` — the `(offset, len)` handle. Stable across all
  subsequent inserts (the underlying `Vec<u8>` may reallocate, but the
  indices into it do not move).
- `AdaptiveCodes` — 4-arm enum (`U8` / `U16` / `U32` / `U64`) holding
  the per-row code array. Promotes deterministically when cardinality
  crosses 256, 65 536, or 2³². Promotion is **lazy**: it triggers only
  when the *next* code physically cannot fit the current arm. It is
  never predictive and never speculative.
- `ByteDictionary` — `BTreeMap<Vec<u8>, u64>` lookup, plus `frozen` flag
  and `CategoryOrdering` policy (`FirstSeen` / `Lexical` / `Explicit`).
  The BTree, not a HashMap, is the determinism load-bearer. `Vec<u8>::cmp`
  (raw byte order) is the seal-key — not Unicode collation — so the same
  dictionary serialises identically on Linux, Windows, macOS, and any
  future target.
- `CategoricalColumn` — codes (`AdaptiveCodes`) + dictionary
  (`ByteDictionary`) + optional null bitmap (`Option<BitMask>`,
  reusing the existing `BitMask` from `cjc-data`).
- `UnknownCategoryPolicy` — four-arm enum decided per call:
  `Error` (return `Err(UnknownCategory)`),
  `MapToNull` (record null bit),
  `MapToOther { other_code: u64 }` (route to a sentinel "other" code),
  `ExtendDictionary` (bypass the freeze for that single push).

### Determinism contract

The contract that v3 builds on is six clauses, all integer-only and all
test-pinned:

1. **No float math anywhere in the engine.** All thresholds, ratios,
   and profile statistics are `u64`/`usize`.
2. **`BTreeMap`, not `HashMap`.** The lookup index is ordered.
   `RandomState` does not appear.
3. **Byte-lex ordering, not Unicode.** `seal_lexical()` sorts by
   `Vec<u8>::cmp`, which is the same on every platform.
4. **Lazy code-width promotion.** `AdaptiveCodes` widens only when the
   incoming code physically does not fit the current arm. Never
   predictive.
5. **Frozen rejection is strict.** `intern` on a frozen dictionary
   returns `Err(Frozen)` — never silently extends. `lookup` continues
   to resolve known values.
6. **Round-trip is byte-equal.** For any push sequence,
   `col.get(i)` returns the exact bytes that were pushed at row `i`
   (or `None` if `push_null` was called). Bolero fuzz pins this for
   arbitrary input shapes.

### What Phase 1 deliberately defers

- **Wiring into TidyView verbs.** `Column::Categorical` and `FctColumn`
  stay unchanged. No verb (filter, mutate, group_by, join, pivot) reads
  or writes a `CategoricalColumn` yet. That's Phase 2.
- **Categorical-aware `group_by` / `join`.** Phase 3 adds a
  cat-aware `GroupIndex` that reads codes directly instead of
  re-hashing strings.
- **Language-level `.cjcl` builtins.** No `categorical(...)` builtin in
  Phase 1. The engine is Rust-only until verbs need it.
- **Higher-order / native code-width arm `U16x4` packing.** The four-arm
  enum is sufficient for v3's target workloads. A packed
  variable-width arm can land later under the same `AdaptiveCodes`
  trait surface if benchmarks justify it.
- **Zero-copy serde.** `ByteStringPool` is layout-stable but Phase 1
  does not yet expose a memory-mapped or `bytemuck`-cast path. Phase 6
  (spill-to-disk) will define the on-disk format on top of the
  arena.

## Consequences

### Positive

- **Determinism is structural.** Two fresh dictionaries built from any
  permutation of the same byte set, both sealed lexically, produce
  bit-identical `(code, bytes)` mappings. Verified by integration test
  `lexical_seal_is_permutation_invariant` and bolero fuzz
  `fuzz_categorical_lexical_determinism`.
- **Code-width is paid as needed.** A 4-category column costs `nrows`
  bytes for codes, not `4 * nrows`. A 100k-category column promotes
  to `u32` once and stays there — no second re-encoding pass.
- **Spill-to-disk has a target.** The pool's flat `Vec<u8>` + `Vec<u32>`
  offsets is exactly the on-disk format Phase 6 wants. No
  serialise/deserialise round-trip required.
- **The existing categorical surfaces are untouched.** The Stage 17
  forcats suite (`FctColumn`) and existing TidyView categorical
  cells continue to work identically; nothing migrates until Phase 2.

### Negative

- **Two parallel categorical types now exist.** `CategoricalColumn`
  (Phase 1) and `Column::Categorical` (legacy). Phase 2's job is to
  eliminate the duplication by routing TidyView verbs through the new
  type and either deleting the legacy column or making it a thin
  wrapper.
- **`bitmask_with_extra_valid` rebuilds the null `BitMask` per push when
  nulls are present.** This is `O(nrows)` per `push`, so a column with
  10 000 rows of mixed null/non-null pays ~5 × 10⁷ bit-ops to construct.
  Phase 1 accepts this — null-rich columns are not the v3 hot path —
  but the optimization (in-place flag-array → `BitMask` at `freeze`
  time) is queued for Phase 2.
- **`MapToOther` requires the caller to pre-intern the sentinel.** The
  policy is `MapToOther { other_code }`, not `MapToOther { other_label:
  &[u8] }`, on the assumption that pipelines decide their sentinel up
  front. This trades ergonomics for one fewer string-comparison branch
  on the hot path.

## Test surface

- **34 inline unit tests** in `byte_dict.rs` covering every public
  surface: pool round-trip (including embedded NUL, high bytes, empty
  strings), code promotion at every threshold, ordering policy
  determinism, frozen rejection, all four unknown-policy variants,
  null bitmap interactions, profile statistics.
- **3 bolero structural fuzz tests** in
  `tests/bolero_fuzz/categorical_dictionary_fuzz.rs`: round-trip,
  lexical determinism across permuted insertion orders, frozen
  rejection.
- **10 integration tests** in
  `tests/tidy_tests/test_adaptive_dictionary_engine.rs`: end-to-end
  build/read/null cycle, both promotion thresholds, lexical-seal
  permutation invariance, frozen rejection, all four unknown policies,
  profile reporting.

## Related ADRs

- [[ADR-0017 Adaptive TidyView Selection]] — the v2/v2.1/v2.2 selection
  engine. ADR-0018 is the categorical companion: ADR-0017 made the row
  axis adaptive, ADR-0018 makes the column axis (for categoricals)
  adaptive.

## References

- Phase 1 source: [byte_dict.rs](../../../crates/cjc-data/src/byte_dict.rs)
- Inline unit tests: same file, `#[cfg(test)] mod tests`
- Bolero fuzz: [categorical_dictionary_fuzz.rs](../../../tests/bolero_fuzz/categorical_dictionary_fuzz.rs)
- Integration tests: [test_adaptive_dictionary_engine.rs](../../../tests/tidy_tests/test_adaptive_dictionary_engine.rs)
- Architecture note: [[TidyView Architecture]] (Phase-2 wiring stub)
