---
title: TidyView Architecture
tags: [data, runtime, architecture, determinism]
status: Implemented
---

# TidyView Architecture

**Crate**: `cjc-data` — `crates/cjc-data/src/lib.rs` (~7.3K LOC) + `tidy_dispatch.rs` (~1.3K) + `lazy.rs` (~2.2K) + supporting modules.

## Summary

TidyView is a three-layer data manipulation system combining a **tidyverse-inspired API** with **database-engine internals**. It is designed so that determinism, auditability, and memory safety emerge from the architecture rather than being bolted on.

## The Three Layers

### Layer 1: TidyView (Immutable, Lazy, Zero-Copy)

```rust
pub struct TidyView {
    base: Rc<DataFrame>,    // shared immutable base
    mask: BitMask,           // which rows are visible (1 bit/row)
    proj: ProjectionMap,     // which columns are visible (index list)
}
```

- **Database analog**: SQL view / cursor
- `filter()` creates a new bitmask — never copies row data
- `select()` updates projection indices — never copies column data
- Data only materializes on `.materialize()` or operations that structurally require it (e.g., `arrange`)
- All intermediate views share the same `Rc<DataFrame>` base — the original data is never modified

### Layer 2: TidyFrame (Materialized, Copy-on-Write)

```rust
pub struct TidyFrame {
    inner: Rc<RefCell<DataFrame>>,
}
```

- **Database analog**: Materialized view
- Returned by `mutate()`, joins, pivot operations
- **Copy-on-write**: if `Rc::strong_count > 1` when writing, deep-clones first
- Prevents aliasing mutations (the Pandas `SettingWithCopyWarning` problem)

### Layer 3: GroupedTidyView (Grouped, Aggregatable)

```rust
pub struct GroupedTidyView {
    view: TidyView,
    index: GroupIndex,
}
```

- **Database analog**: `GROUP BY` state
- Group order = **first-occurrence** order (deterministic)
- Built via `BTreeMap` lookup — O(N log G), no hash nondeterminism
- `summarise()` collapses groups; `ungroup()` restores flat view

## Auditability by Construction

### Immutable base data

Every `filter()`, `select()`, `distinct()`, `semi_join()`, `anti_join()` returns a new `TidyView` pointing at the **same** underlying `Rc<DataFrame>`. The original data is never touched. If step 5 of a 10-step pipeline produces wrong results, you can inspect the base data and it's still pristine.

### Inspection methods

```rust
view.mask()          // → &BitMask — which rows are visible
view.proj()          // → &ProjectionMap — which columns are visible
grouped.group_index() // → &GroupIndex — group keys + row indices
lazy.plan()          // → &ViewNode — the logical query tree
lazy.optimized_plan() // → ViewNode — the tree after optimizer passes
```

At any pipeline stage:
- `mask().iter_set()` gives exact visible row indices
- `mask().count_ones()` gives the count of surviving rows
- `proj().indices()` gives the visible column indices
- The original data is still there behind the `Rc<DataFrame>`

### Snapshot semantics in mutate

```rust
let snapshot_names: Vec<String> = df.columns.iter().map(|(n, _)| n.clone()).collect();
for &(col_name, ref expr) in assignments {
    validate_expr_columns_snapshot(expr, &snapshot_names)?;
}
```

All column references validate against the column list frozen **before** any mutations. A reference to a column being created in the same `mutate` call is `TidyError::ColumnNotFound`. Results never depend on assignment order.

### Null quarantine

- Base `Column` type has **no nulls** (no `Option` wrapper)
- `NullCol` is a separate type used only by operations that can introduce nulls (left joins, pivot_wider, full joins)
- If you have a `TidyView`, its data is proven null-free by type

## Memory Safety

| Problem | Traditional risk | TidyView prevention |
|---|---|---|
| Aliasing mutation | Pandas `SettingWithCopyWarning` | `Rc::strong_count` check + deep clone before write |
| Dangling references | View into freed buffer | `Rc<DataFrame>` keeps base alive as long as any view exists |
| Order-dependent mutate | `assign(a=x+1, b=a*2)` ambiguity | Snapshot semantics — column list frozen at call entry |
| Hidden nulls | NaN/None in any column at any time | `Column` (null-free) vs `NullCol` (explicit) |

## Database-Inspired Internals

### Columnar storage

`DataFrame` stores data column-major: `Vec<(String, Column)>`. Same layout as Apache Arrow, DuckDB, Parquet. Aggregations scan contiguous memory; projections are index lookups.

### Lazy query optimizer (lazy.rs)

Three classic passes over the `ViewNode` tree:

1. **Filter merging** — `Filter(Filter(input, p1), p2)` → `Filter(input, p1 AND p2)`
2. **Predicate pushdown** — push filters past Select, Arrange, and (when safe) Mutate; push into Join branches
3. **Redundant select elimination** — remove `Select` nodes that project all columns

Rules mirror standard relational algebra: never push past `GroupSummarise` or `Distinct` (they change row identity).

### Zone maps (column_meta.rs)

`ColumnStats` tracks min/max, sorted flags, and distinct counts per column. Enables:
- `can_skip_gt(threshold)` — skip entire column if max ≤ threshold
- `binary_search_range_f64()` — O(log N) filter on sorted columns

### Columnar filter fast-path

For predicates matching `Col op Literal`, evaluation is bitwise over the column buffer rather than row-by-row interpreter calls.

### BitMask (O(N/8) memory)

```rust
pub struct BitMask {
    words: Vec<u64>,
    nrows: usize,
}
```

1 bit per row, packed into u64. Chained filters AND their bitmasks. Same approach as Oracle bitmap indexes and DuckDB selection vectors.

## Determinism Guarantees

- `BTreeMap`/`BTreeSet` everywhere — never `HashMap`/`HashSet`
- Kahan summation for all floating-point reductions (via `cjc-repro`)
- Stable sorting (Rust's `sort_by` is stable)
- First-occurrence group ordering
- Deterministic sampling via LCG-based Fisher-Yates with explicit seed
- No hidden parallelism

## Error Reporting

```rust
pub enum TidyError {
    ColumnNotFound(String),
    DuplicateColumn(String),
    PredicateNotBool { got: String },
    TypeMismatch { expected: String, got: String },
    LengthMismatch { expected: usize, got: usize },
    EmptyGroup,
    CapacityExceeded { limit: usize, got: usize },
    Internal(String),
}
```

Every variant names the specific column or value that caused the problem.

## Integration

Both executors route through `tidy_dispatch.rs`:

```
cjc-eval ──────┐
               ├──→ tidy_dispatch::dispatch_tidy_method()
cjc-mir-exec ──┘    tidy_dispatch::dispatch_grouped_method()
```

Type erasure via `Rc<dyn Any>` avoids circular dependency between `cjc-runtime` and `cjc-data`.

## Related

- [[DataFrame DSL]]
- [[Numerical Truth]]
- [[Runtime Architecture]]
- [[Dispatch Layer]]
- [[Vizor]]
