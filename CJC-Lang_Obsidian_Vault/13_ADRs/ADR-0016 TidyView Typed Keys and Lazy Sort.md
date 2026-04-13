---
title: ADR-0016 TidyView Typed Keys and Lazy Sort
tags: [adr, data, performance, determinism]
status: Accepted
date: 2026-04-12
---

# ADR-0016: TidyView Typed Keys and Lazy Sort

**Status:** Accepted
**Date:** 2026-04-12
**Deciders:** Lead Language Architect, Numerical Computing Engineer, Determinism Auditor

## Context

TidyView performance profiling (38-benchmark suite, 100K-row datasets) revealed three bottlenecks:

1. **Sort comparator** resolved column references by name (`HashMap` lookup) on every comparison — O(N log N) lookups for an N-row sort.
2. **GroupMeta keys** stored as `Vec<String>`, requiring `.to_string()` formatting on every row and `.parse::<T>()` on readback. For `Int` and `Float` columns this is pure waste.
3. **Arrange materialized eagerly** — `arrange()` copied all columns into a new `DataFrame` even when the next operation only needed ordering (e.g., `head()`, `slice()`).

These are classic database-engine optimizations (pre-resolved schemas, typed keys, lazy evaluation) applied to CJC's TidyView layer.

## Decision

Six optimizations, implemented as a single coherent batch:

### P1: Cache column indices in sort comparator

`arrange()` pre-resolves column references before entering `sort_by`:

```rust
let key_cols: Vec<(&Column, bool)> = keys.iter()
    .map(|k| (base.get_column(&k.col_name).unwrap(), k.descending))
    .collect();
row_indices.sort_by(|&a, &b| {
    for &(col, desc) in &key_cols { ... }
});
```

Eliminates O(N log N × K) name lookups. Sort benchmarks improved ~30x.

### P2: Store typed keys in GroupMeta

`GroupKey` enum replaces `Vec<String>`:

```rust
pub enum GroupKey {
    Int(i64),
    Float(FloatKey),
    Code(u32),
    Str(String),
    Bool(bool),
    DateTime(i64),
}
```

`FloatKey` wraps `f64` with NaN-last ordering matching `compare_column_rows`. Group construction skips `.to_string()` formatting entirely.

### P3: Vectorized aggregation kernels

Aggregate functions pre-resolve column references once per group set, then dispatch to type-specialized inner loops (`fast_agg_sum`, `fast_agg_count`, etc.) that iterate `&[usize]` row indices directly against the resolved `&Column`.

### P4: Lazy sort via permutation vector

`TidyView` gains an `ordering: Option<Rc<Vec<usize>>>` field. `arrange()` computes the permutation but does **not** materialize a new `DataFrame`. Methods that need physical data call `resolve_ordering()`, which materializes once and caches.

Operations that pass through ordering without resolving: `select`, `rename`, `relocate`.
Operations that resolve immediately: `filter`, `slice`, `join`, `group_by`, `materialize`.

### P5: Code-based categorical sort

`Column::Categorical` levels are always sorted (BTreeMap-backed `DictEncoding`). The sort comparator compares `u32` codes instead of dereferencing string levels:

```rust
Column::Categorical { codes, .. } => codes[a].cmp(&codes[b])
```

### P6: Shared-dictionary join optimization

`detect_shared_dict_flags()` compares Categorical level vectors between left and right join columns. When dictionaries are identical, join key comparison uses `u32` codes instead of string cloning.

## Constraints

### Determinism preservation

- **Kahan summation order**: All vectorized aggregation kernels iterate `row_indices` in forward order, identical to the scalar path. No SIMD lane splitting, no tree reduction. Per [[ADR-0002 Kahan Accumulator]], the compensation term is order-dependent.
- **Stable sort**: Rust's `sort_by` is stable; P1 and P4 preserve this. P5's code-based comparison produces identical ordering because BTreeMap guarantees sorted levels.
- **GroupKey determinism**: `FloatKey::Ord` matches `compare_column_rows` NaN-last semantics exactly. `BTreeMap<Vec<GroupKey>, ...>` iteration order is deterministic.

### Breaking change

`GroupMeta::key_values` changed from `Vec<String>` to `Vec<GroupKey>`. All code comparing group keys against string literals must use `GroupKey::Str(...)`, `GroupKey::Int(...)`, etc.

## Consequences

### Positive (post regression-fix)

- Sort benchmarks (D1, D2) improved **34-69x** (column-index caching + lazy materialization)
- Multi-key sort (D4) improved **2.4x**
- Group+summarise (C1-C4) improved **2.0-2.6x**
- String filter (A3) improved **248x** (columnar fast path)
- Join (E2, E3) improved **1.5-1.7x**
- Scaling (G2 1M rows) improved **1.7x**
- Full pipeline (F3 join+filter+group) improved **1.7x**
- All 38 benchmarks passing, all determinism gates (H1-H3) passing
- 5 proptest property suites (200 cases each) + 2 bolero fuzz targets added

### Negative

- Minor regressions on string-only workloads (E1: 1.5x slower, C5: 1.2x slower) due to `GroupKey` enum being larger than plain `String`.
- F2 (filter+mutate+group) 1.5x slower — mutate path exercises materialization.

### Regression fix (2026-04-13)

Initial P4 implementation caused widespread regressions (joins 2-3.5x slower, scaling 2-5x slower). Root causes:
1. `resolve_ordering()` returned `self.clone()` when no ordering existed — wasteful BitMask copy on every method call.
2. `filter()` forced full DataFrame materialization when ordering was present.

Fix: `resolve_ordering()` now returns `Option<TidyView>` (None = no work needed). `filter()` composes with ordering by filtering the permutation vector directly instead of materializing. All regressions resolved.

## Test coverage

| Category | Count | Purpose |
|---|---|---|
| P1 unit tests | 5 | Sort cache correctness |
| P2 unit tests | 4 | Typed GroupKey round-trip |
| P3 unit tests | 4 | Vectorized agg parity |
| P4 unit tests | 14 | Lazy sort semantics |
| P5 unit tests | 3 | Categorical code sort |
| P6 unit tests | 7 | Shared-dict join |
| proptest properties | 5 | Random-input invariants (200 cases each) |
| bolero fuzz targets | 2 | Panic-freedom under arbitrary data |
| Benchmark suite | 38 | Performance + determinism gates |

## Related

- [[ADR-0002 Kahan Accumulator]] — order-dependence constraint on P3
- [[TidyView Architecture]] — updated with performance optimizations section
- [[Determinism Contract]] — all six optimizations preserve bit-identical output
