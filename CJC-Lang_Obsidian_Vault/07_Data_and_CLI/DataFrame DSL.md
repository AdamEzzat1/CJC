---
title: DataFrame DSL
tags: [data, runtime]
status: Implemented
---

# DataFrame DSL

**Crate**: `cjc-data` — `crates/cjc-data/src/` (~307K lib.rs + 56K tidy_dispatch.rs).

## Summary

A tidyverse / dplyr-inspired DataFrame library for CJC-Lang. Operations are lazy and pipe-friendly. Both executors route through `tidy_dispatch.rs` for parity.

## Column types

- Int
- Float
- Str
- Bool
- Categorical (u16 code + level dictionary — see [[Bastion]] Phase 17)
- DateTime

## Operations

### Shape
- `nrows`, `ncols`, `column_names`

### Row operations
- `filter(pred)` — row-wise predicate
- `arrange(col, ...)` — sort
- `slice_head(n)`, `slice_tail(n)`

### Column operations
- `select(cols)`
- `mutate(name = expr)` — add/modify computed columns
- `rename(old = new)`

### Grouping and aggregation
- `group_by(cols)` → `GroupedTidyView`
- `summarize(name = agg)` — reduce groups
- `ungroup()`

### Joins and reshaping
- `inner_join`, `left_join`, `right_join`, `full_join` (**Needs verification** of exact list)
- `pivot_longer`, `pivot_wider`

### Window functions
- `window_sum`, `window_mean`, rolling aggregations

## The `DExpr` lazy layer

`DExpr` is an expression tree that lets you write predicates and computed columns declaratively:

```cjcl
df |> filter(col("age") > 18) |> mutate(adult_score = col("x") * 2.0)
```

`col("age") > 18` builds a `DExpr::Gt(Col("age"), Lit(18))` which is evaluated lazily per row or per group.

## Determinism

- All iteration uses sorted order or preserves insertion order.
- Hash joins are avoided in favor of sort-based joins (or use a deterministic hashing scheme — **Needs verification**).
- Group keys iterate in `BTreeMap` order.
- Reductions use Kahan/binned accumulation — see [[Numerical Truth]].

## Streaming CSV

Per the performance manifesto, Phase 8 added Kahan-stable streaming CSV parsing. See `docs/spec/phase_8_data_logistics.md`.

## TidyView name

Early design docs use "TidyView" as the name for the immutable lazy dataframe reference. The public surface talks about DataFrames; internally the lazy view type is sometimes called `TidyView` or `GroupedTidyView`.

## Related

- [[Runtime Architecture]]
- [[Vizor]] — output destination for many DF pipelines
- [[Bastion]] — statistics over DFs
- [[Statistics and Distributions]]
- [[Dispatch Layer]]
