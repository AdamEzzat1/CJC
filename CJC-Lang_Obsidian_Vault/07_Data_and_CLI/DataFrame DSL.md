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
- `inner_join`, `left_join`, `right_join`
- `df_full_join(df1, df2, on)` — full outer join (v0.1.7)
- `df_anti_join(df1, df2, on)` — rows of `df1` with no match in `df2` (v0.1.7)
- `df_semi_join(df1, df2, on)` — rows of `df1` that have a match in `df2` (v0.1.7)
- `pivot_wider(df, id_cols, names_from, values_from)` — long → wide, 4 args (v0.1.7 dispatch-arm)
- `pivot_longer(df, cols, names_to, values_to)` — wide → long (v0.1.7 dispatch-arm)

### Deduplication and renaming (v0.1.7)
- `df_distinct(df)` — drop duplicate rows
- `df_rename(df, old_name, new_name)` — rename one column

### Missing-value operations (v0.1.7)
- `df_fill_na(df, col_name, fill_val)` — **per-column** NA replacement (3 args, not a global fill)
- `df_drop_na(df)` — drop rows containing any NA

See [[NA Handling]] for the underlying `Value::Na` representation and the generic `is_na` / `fill_na` / `coalesce` builtins.

### CSV I/O (v0.1.7)
- `df_read_csv(path)` — parse a CSV file into a DataFrame.

**Gotcha**: the CSV reader infers `"0"` and `"1"` as `Bool`, not `Int`. If you need them as integers, cast explicitly after reading, or ensure the CSV has wider numeric ranges in the column so the inference picks `Int`.

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

- [[TidyView Architecture]] — three-layer design, auditability, database internals
- [[Runtime Architecture]]
- [[Vizor]] — output destination for many DF pipelines
- [[Bastion]] — statistics over DFs
- [[Statistics and Distributions]]
- [[Dispatch Layer]]
