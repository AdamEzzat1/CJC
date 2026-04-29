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
    base: Rc<DataFrame>,         // shared immutable base
    mask: AdaptiveSelection,     // which rows are visible (density-aware; see below)
    proj: ProjectionMap,         // which columns are visible (index list)
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

### Columnar filter fast-path (v2.1: predicate bytecode)

For predicates matching `Col op Literal` (and And/Or compounds thereof),
the filter is lowered once per `filter()` call to a flat stack-bytecode
program in `crates/cjc-data/src/predicate_bytecode.rs`, then interpreted
in a tight loop. Three opcodes:

```rust
pub enum PredicateOp {
    Cmp { kind: LeafKind, op: CmpKind },
    And,
    Or,
}
```

`Cmp` carries `(col_idx, literal, op)` already resolved at lowering
time, so interpretation involves no symbol lookup. The result is bit-
identical to the legacy AST-walk path on every shape it accepts; the AST
walk is retained as a no-cost fallback oracle for unsupported shapes.

### Sparse-gather predicate (v2.2)

When the existing selection has already narrowed below 25 % of base
rows (`count * 4 < nrows`), the same bytecode program is interpreted
*scalar-wise* over the existing index iterator instead of by full
column scan. AND/OR are monotone, so the sparse path returns a
selection bounded to the input set without a final AND with the
existing mask. Bit-identical to the dense path.

`materialize_mask()` is deferred until *after* the dense/sparse
decision, so chained filters where the parent is very sparse pay no
bitmap allocation cost. On a 1M-row two-step chain the second filter
drops from ~1 ms to ~25 µs at p1 = 1–10 hits. See
[[ADR-0017 Adaptive TidyView Selection|ADR-0017 v2.2 amendment]] for
the full crossover table.

### Tidyverse integration verdict (Phase 0 audit, 2026-04-27)

CJC-Lang has no second "tidyverse wrapper" sitting above TidyView — the
language-level surface (the `view.filter(...)` / `grouped.summarise(...)`
syntax visible in `.cjcl` source) is the dispatch table
`tidy_dispatch::dispatch_tidy_method` + `dispatch_grouped_method`, which
forwards 1:1 to the same Rust API used by `cjc-data` internally. Both
`cjc-eval` and `cjc-mir-exec` route their method calls into this single
dispatch table, matching the canonical wiring pattern.

| Verb | Routes through TidyView | Adaptive-selection-aware |
|---|---|---|
| filter | yes | yes (predicate bytecode + sparse-gather) |
| select | yes | yes (mask preserved, projection rewritten) |
| mutate | yes (returns TidyFrame) | mask applied **before** expression eval, then base rebuilt — no hidden bypass |
| arrange | yes (returns TidyFrame) | iter_indices() collected, full DataFrame rebuilt — re-materialisation by design |
| group_by | yes | yes (uses `iter_indices()` directly via v2.2 IntoIterator signature) |
| summarise (grouped) | yes | uses group → row_indices already from iter_indices() |
| slice / head / tail / sample | yes | yes (builds new AdaptiveSelection mask) |
| distinct | yes | yes (iter_indices + view_from_row_indices) |
| inner / left / full join | yes (returns TidyFrame) | iter_indices() on both sides, then column gather — re-materialisation by design |
| semi / anti join | yes (returns TidyView) | yes (view_from_row_indices) |
| pivot_longer / pivot_wider | yes (returns TidyFrame) | re-materialisation by design |
| bind_rows / bind_cols | yes (returns TidyFrame) | re-materialisation by design |
| rename / drop_cols | yes | mask preserved, columns metadata rewritten |

The "re-materialisation by design" entries (mutate, arrange, joins,
pivots, binds) all consult the mask **before** doing work — there is no
case of a verb walking the full base while ignoring the selection.
Where they materialise, it is because they need to produce new row
identity (joins, pivots) or a new sort order (arrange) or new column
buffers (mutate, binds), which cannot be expressed as a view delta.

Pinned by 6 dispatch-vs-direct parity tests in
[`tests/tidy_tests/test_tidyverse_integration_parity.rs`]:

- filter chain (1M rows, sparse-gather active) — dispatch and direct
  produce identical row indices, identical selection mode, and
  identical materialised columns.
- filter → select — selection cardinality preserved through dispatch.
- distinct, arrange, group_by + summarise — full pipeline parity.
- sparse filter → group_by + summarise — explicitly asserts the
  post-filter mode is `SelectionVector`, proving the dispatch did not
  silently materialise.

Phase 0 regression gate: 369/369 `test_phase10_tidy`, 23/23
`bolero_fuzz`, 38/38 `tidyview_benchmarks`, 159/159 `cjc-data` unit. No
v2.2 number changed.

### Phase 1 — Deterministic Adaptive Dictionary Engine (2026-04-28)

Phase 1 of TidyView v3 lands a parallel, byte-first categorical engine
in [`crates/cjc-data/src/byte_dict.rs`](../../crates/cjc-data/src/byte_dict.rs).
See [[ADR-0018 Deterministic Adaptive Dictionary Engine]] for the full
decision and trade-offs.

Six new public types:

- `ByteStringPool` — append-only `Vec<u8>` arena with stable
  `(offset, len)` handles. No `String` in the hot path.
- `ByteStrView` — opaque `(offset, len)` handle into a pool.
- `AdaptiveCodes` — 4-arm enum (U8/U16/U32/U64) that promotes lazily
  at the integer thresholds 256, 65 536, 2³².
- `ByteDictionary` — `BTreeMap<Vec<u8>, u64>` lookup, `frozen` flag,
  `CategoryOrdering::FirstSeen` / `Lexical` / `Explicit`.
- `CategoricalColumn` — codes + dictionary + optional `BitMask` nulls.
- `UnknownCategoryPolicy` — `Error` / `MapToNull` /
  `MapToOther { other_code }` / `ExtendDictionary`.

**Phase 1 ships only the standalone engine.** TidyView verbs are
unchanged — `Column::Categorical` and `FctColumn` remain the live
categorical surfaces. Phase 2 will route group_by, joins, and distinct
through `CategoricalColumn` so cardinality-aware paths can read codes
directly instead of re-hashing strings.

Test surface: 34 inline unit + 3 bolero fuzz + 10 integration
(`tests/tidy_tests/test_adaptive_dictionary_engine.rs`).

Phase 1 regression gate: 379/379 `test_phase10_tidy` (+10 from Phase 0),
26/26 `bolero_fuzz` (+3), 38/38 `tidyview_benchmarks`, 193/193
`cjc-data` unit (+34). No earlier number changed.

### Phase 2 — Cat-aware key path for `group_by` + `distinct` (2026-04-28)

When every key column is `Column::Categorical`, the lookup BTreeMap
inside `GroupIndex::build_fast` and the dedup BTreeSet inside
`TidyView::distinct` switch from `Vec<String>` keys to `Vec<u32>` keys
of category codes. The `levels[code].clone()` per row per key column
disappears; display strings are materialised once per group (or once
per unique row, in distinct), not once per row scan.

**Bit-identity is structural, not asymptotic.** Inside a single
`DataFrame`, codes are in 1:1 correspondence with display strings
(`levels[code]` is a deterministic lookup), so the fast path produces
identical group slot assignment, identical `key_values`, identical
`row_indices`. The string path remains as a fallback for any key set
that isn't all-categorical (e.g., one categorical + one int key) — the
cat-detector returns `None` and the existing string path runs unchanged.

**Joins are deliberately out of scope.** Joins compare keys *across*
two DataFrames whose categorical levels are independent string sets —
left-side code 3 means "Boston" while right-side code 3 means
"Athens". Cross-DataFrame code comparison without a translation layer
would be wrong. Cat-aware joins are deferred until the broader
`Column` refactor wires Phase 1's `CategoricalColumn` (with shared
frozen dictionaries) into the column enum.

**Headline benchmark** (1M rows, 100 distinct categorical keys, single
key column, 5-run average):

| Path | Wall time | Speedup |
|---|---|---|
| String (`Column::Str`) | 859.9 ms | 1.0× |
| Categorical | 114.8 ms | **7.49×** |

The win is dominated by eliminating O(rows × key_cols) String allocations
and BTreeMap key comparisons — `Vec<u32>` comparison is a `memcmp` over
4 × key_cols bytes, while `Vec<String>` comparison walks the Vec, then
each String, then each character.

Test surface: 11 integration parity tests at
[`tests/tidy_tests/test_v3_phase2_categorical_keys.rs`](../../tests/tidy_tests/test_v3_phase2_categorical_keys.rs)
covering single-key, two-key, post-filter, mixed-type fallback,
single-row, empty-after-filter, all-same-value cases for both
`group_by_fast` and `distinct`. Plus 1 ignored design-validation bench.

Phase 2 regression gate: 390/390 `test_phase10_tidy` (+11 from
Phase 1, +1 ignored bench), 26/26 `bolero_fuzz` unchanged, 38/38
`tidyview_benchmarks` unchanged, 193/193 `cjc-data` unit unchanged.

### BitMask (O(N/8) memory)

```rust
pub struct BitMask {
    words: Vec<u64>,
    nrows: usize,
}
```

1 bit per row, packed into u64. Chained filters AND their bitmasks. Same approach as Oracle bitmap indexes and DuckDB selection vectors.

### Adaptive selection (v0.1.6, [[ADR-0017 Adaptive TidyView Selection]])

`TidyView::mask` is no longer a raw `BitMask` — it's an `AdaptiveSelection` enum that classifies the result of every predicate evaluation by density:

```rust
pub enum AdaptiveSelection {
    Empty { nrows: usize },
    All { nrows: usize },
    SelectionVector { rows: Vec<u32>, nrows: usize },
    VerbatimMask { mask: BitMask },
    Hybrid { nrows: usize, chunks: Vec<HybridChunk> }, // v2.1: chunked, per-block classified
}

pub enum HybridChunk {
    Empty,
    All,
    Sparse(Vec<u16>),
    Dense(Box<[u64]>),
}
```

| Density (`count` vs `nrows`) | Arm chosen |
|---|---|
| `count == 0` | `Empty` |
| `count == nrows` | `All` |
| `count < nrows / 1024` | `SelectionVector` (sparse) |
| `count * 10 > nrows * 3` | `VerbatimMask` (dense, > 30 %) |
| otherwise, `nrows >= 8192` | `Hybrid` (mid-band, 4096-row chunks) |
| otherwise | `VerbatimMask` (small frame mid-band) |

`Hybrid` splits the row space into 4096-row chunks and re-classifies each chunk independently — chunks with zero hits become `HybridChunk::Empty` (skipped by the iterator), chunks that pass entirely become `HybridChunk::All` (a single range emit), sparse chunks store ≤ 128 indices in a `Vec<u16>`, and dense chunks store the chunk's 64-word verbatim bitmap. This gives mid-band selections (the typical post-WHERE-clause shape on real data) the O(count) iteration cost of `SelectionVector` without paying the dense-path word-walk on Empty/All chunks.

All consumers reach selections through a mode-invariant trait: `count`, `contains`, `iter_indices`, `intersect`, `union`, `materialize_mask`, `materialize_indices`, `explain_selection_mode`. The 20+ pre-existing `iter_set` call sites in joins, group_by, distinct, materialize, pivot, summarise, and mutate-with-mask migrate to a single `iter_indices()` call that enum-dispatches to either the sparse `Vec<u32>` or the dense bitmap iterator.

Why it matters: a 9-row sparse filter on a 1M-row table now enumerates 9 indices in O(9), instead of word-walking 15,625 u64s. Empty / All-pass filters short-circuit to O(1) in `count()` and skip allocation entirely.

Determinism is preserved by construction: thresholds are integer-only (no float density boundary), every arm yields strictly ascending row indices, and `intersect` / `union` re-classify after each set op, so the cardinality identity `|A| + |B| = |A ∪ B| + |A ∩ B|` holds across mode-mixed inputs (fuzzed via bolero).

`mask()` still returns an owned `BitMask` (now materialized via the trait) to keep existing chain calls working; new code prefers `selection() -> &AdaptiveSelection`.

## v3 Phase 3 — Hybrid streaming set ops (2026-04-28)

`AdaptiveSelection::intersect` / `union` now have a **per-chunk dispatch
path** when one or both operands are `Hybrid`. Pre-Phase-3 these routed
through `to_verbatim_mask() → AND/OR → re-classify`, allocating an
`nrows/64` u64 buffer for each Hybrid operand. Phase 3 keeps the chunk
layout end-to-end.

### Per-chunk dispatch table

`HybridChunk` is one of `Empty | All | Sparse(Vec<u16>) | Dense(Box<[u64]>)`.
Cross-product is 4×4 = 16 cells; Empty/All folding collapses to **5
effective shapes**:

| left × right (intersect) | path                                                     |
|---|---|
| `Sparse ∩ Sparse`        | merge-walk on two ascending `Vec<u16>` → `Sparse` or `Empty` |
| `Sparse ∩ Dense`         | filter-walk: bit-test each sparse row against partner words |
| `Dense ∩ Dense`          | per-word AND over `chunk_len.div_ceil(64)` u64s          |
| `Empty ∩ *`              | `Empty`                                                  |
| `All ∩ x`                | `x` unchanged                                            |

Symmetric union table; result chunks are re-classified after each op
via `classify_sparse_chunk` / `classify_dense_chunk` so they land in
the cheapest shape (Sparse stays Sparse if hits ≤ 128, else promotes
to Dense).

### Helper inventory (`crates/cjc-data/src/adaptive_selection.rs`)

| Helper | Role |
|---|---|
| `intersect_chunks(a, b, chunk_len) -> HybridChunk` | dispatch one cell of the table |
| `union_chunks(a, b, chunk_len) -> HybridChunk`     | symmetric union |
| `intersect_chunk_with_words(chunk, words)`         | Hybrid × VerbatimMask cell |
| `union_chunk_with_words(chunk, words, chunk_len)`  | Hybrid ∪ VerbatimMask cell |
| `merge_intersect_u16(a, b)` / `merge_union_u16`    | two-cursor merge over sorted `Vec<u16>` |
| `classify_sparse_chunk(rows, chunk_len)`           | promote `Sparse → All` / `Sparse → Dense` if needed |
| `classify_dense_chunk(words, chunk_len)`           | demote `Dense → All` / `Dense → Sparse` / `Dense → Empty` |
| `Self::simplify_hybrid(nrows, chunks)`             | collapse to `All`/`Empty` only when **every** chunk agrees — never re-globalize a mid-band Hybrid into a single `VerbatimMask` |

### Surface contract — production reach today

`TidyView::filter` AND-collapses inside `predicate_bytecode::interpret`
/ `interpret_sparse` (those routines apply the final mask AND
themselves and return the merged result). `AdaptiveSelection::intersect`
/ `union` are **not** on `filter()`'s hot path today — they are public
algebraic operations on the selection lattice, called from tests and
reserved for **Phase 4 = cat-aware joins + Column wiring**, where a
left-mask × right-mask probe-and-build over join keys lands directly
on `Hybrid ∩ Hybrid`.

### Headline bench (favorable case)

`bench_phase3_hybrid_set_op`, 100k rows, sparse-chunked × dense-chunked
Hybrid, same-process comparison vs pre-Phase-3 materialize-and-AND
oracle:

```
Phase 3 chunked path:           4.69 μs avg (n=100)
Pre-Phase-3 materialize path:  391.41 μs avg (n=100)
Speedup:                        83.46×
```

The 83× headline is the favorable case where allocation dominates
the old path. Contiguous-Hybrid (Dense × Dense everywhere) shrinks
the win to ~5–10× — still net positive but no longer
allocation-bound.

## v3 Phase 4 — Cat-aware joins + Column wiring (2026-04-28)

### Cat-aware joins

`TidyView::inner_join` / `left_join` / `semi_join` / `anti_join` now
auto-detect when every join-key column is `Column::Categorical` on
**both** frames and route through a `BTreeMap<Vec<u32>, Vec<usize>>`
probe with a per-column **right-code → left-code remap**. Pre-Phase-4
these built `Vec<String>` keys per row via `get_display(row)` — for
100k × 100k joins on a 100-level categorical that was 200 000
`String::clone()` calls before the BTreeMap even saw a key.

Each DataFrame owns its own dictionary, so the remap is what makes
cross-frame comparison safe:

```rust
right_to_left[ki][right_code] = Some(left_code) | None
```

`None` = the level doesn't exist on the left dictionary; that right
row is skipped before BTreeMap insertion (it can never join). Build
order of the remap is `BTreeMap<&str, u32>` over left levels — BTree
not Hash, so the remap construction is itself deterministic.

### Cat-aware join surface

```
collect_categorical_join_keys(&left.base, &left_cols, &right.base, &right_cols)
    -> Option<CategoricalJoinKeys<'_>>
build_right_lookup_btree_categorical(&cat, right.mask.iter_indices())
    -> BTreeMap<Vec<u32>, Vec<usize>>
left_join_key_codes(&cat, l_row, &mut Vec<u32>)
    -> ()  // fills key_buf in left-code space
```

Mixed-type keys (e.g., one categorical + one int) cause
`collect_categorical_join_keys` to return `None`; callers fall back
to the existing string-key path automatically.

### Column ↔ CategoricalColumn wiring (limited scope)

Phase 1's `byte_dict::CategoricalColumn` (adaptive-width codes,
frozen dictionaries, `UnknownCategoryPolicy`) is reachable from the
DataFrame surface via two lossless conversions:

| Method | Direction | Notes |
|---|---|---|
| `Column::to_categorical_column(&self)` | `Column::Categorical → CategoricalColumn` | Uses `CategoryOrdering::Explicit` to pin level→code byte-equal. |
| `Column::from_categorical_column(&cc)` | `CategoricalColumn → Column::Categorical` | Returns `None` if levels are non-UTF-8 or if `cc` has nulls (`Column::Categorical` has no null bitmap). |

Round-trip identity:
`from_categorical_column(to_categorical_column(c))` byte-equal to `c`.

The full replacement of `Column::Categorical { levels: Vec<String>,
codes: Vec<u32> }` with `byte_dict::CategoricalColumn` is **deferred** —
hundreds of column-reader call sites would migrate. Phase 4's
conversion methods are the gateway for callers that explicitly need
adaptive widths or shared/frozen dictionaries.

### Headline bench (`bench_phase4_categorical_inner_join`, ignored)

100k × 100k rows, 100 unique categorical keys, single-key inner
join, same-process comparison (5-run average):

```
String-key path:   16.82 s
Cat-aware path:     2.77 s
Speedup:            6.08×
```

The win is purely from eliminating per-row `String::clone()` and
allocator pressure in `row_key`. BTreeMap structure / lookup cost is
identical between paths — `Vec<u32>` keys just don't allocate per row.

## v3 Phase 5 — Full Column wiring + filter Hybrid path + cat-aware arrange (2026-04-28)

Three deliverables in one phase, closing the deferred items from
Phases 3 and 4.

### (a) `Column::CategoricalAdaptive` — first-class adaptive variant

`Column::Categorical` (legacy `Vec<String>` + `Vec<u32>`) and
`Column::CategoricalAdaptive(Box<CategoricalColumn>)` (Phase 1's
adaptive engine: `AdaptiveCodes` U8/U16/U32/U64 + `ByteDictionary`
with optional shared/frozen state) now coexist as sibling variants.
The 21 column-reader sites all dispatch through
`Column::to_legacy_categorical()` — a lossless `CategoricalAdaptive
→ Categorical` shim (falls back to `Column::Str` for null-bearing or
non-UTF-8 columns). Hot paths (Phase 2 group_by, Phase 4 joins,
Phase 5(c) arrange) bypass the shim entirely.

### (b) Filter-chain Hybrid intersect path

`predicate_bytecode::PredicateBytecode::evaluate_to_selection(base,
nrows)` is new — pure predicate evaluation (no existing-mask AND),
result classified as `AdaptiveSelection`. `TidyView::filter` routes
through Phase 3's per-chunk dispatch when the existing mask is
already `Hybrid`:

```rust
if matches!(self.mask, AdaptiveSelection::Hybrid { .. })
    && !predicate_bytecode::should_use_sparse_path(count, nrows_base)
{
    let fresh = bc.evaluate_to_selection(&self.base, nrows_base);
    let intersected = self.mask.intersect(&fresh);
    return Ok(TidyView { mask: intersected, .. });
}
```

This is **Phase 3's first production wiring beyond joins** — Hybrid
left mask + chunk-sparse predicate result intersect through the
per-chunk dispatch (the 5 effective shape paths) instead of
allocating an `nrows/64`-word BitMask twice and ANDing.

### (c) Cat-aware arrange

`TidyView::arrange` resolves each key once into either:

- `CatCodes { codes: &[u32], descending }` — when the column is
  `Column::Categorical` with **lex-sorted levels** (the Phase 17
  `forcats` invariant)
- `Legacy { col: &Column, descending }` — everything else

The comparator dispatch is per-key, so composite keys mix sorted-cat
fast path with int / float / unsorted-cat slow path freely.
`levels_are_sorted(levels)` is checked once before sort, not per
comparison.

### Headline bench (`bench_phase5_arrange_cat_vs_string`, ignored)

100k rows, 100 unique categorical levels, single-key ascending sort,
5-run average:

```
Str arrange:   92.73 ms
Cat arrange:    9.58 ms
Speedup:        9.68×
```

Marginally faster than Phase 4's 6.08× on joins because arrange's
hot path is more comparison-bound — the stable sort calls the
comparator ~1.6M times for 100k rows. Every eliminated `String::cmp`
counts.

### Production reach summary (post-Phase-5)

| Verb | Cat-aware? | Mechanism |
|---|---|---|
| `group_by` / `distinct` | ✓ Phase 2 | `BTreeMap<Vec<u32>, _>` keyed group lookup |
| `inner_join` / `left_join` / `semi_join` / `anti_join` | ✓ Phase 4 | `BTreeMap<Vec<u32>, _>` + cross-frame remap |
| `filter` (Hybrid existing mask) | ✓ Phase 5(b) | `evaluate_to_selection` + Phase 3 `intersect` |
| `arrange` | ✓ Phase 5(c) | `u32` code compare when levels lex-sorted |
| `mutate` / `select` / `slice` / `pivot_*` | Through legacy shim | `to_legacy_categorical()` |

## v3 Phase 6 — Streaming summarise + cat-aware mutate + lazy optimizer (2026-04-28)

Three deliverables in one phase, opening new surface beyond the v3
Phase 0–5 closure.

### (a) `TidyView::summarise_streaming`

Single-pass aggregation that **skips the GroupIndex materialisation
step entirely**. Walks visible rows once, maintaining a
`BTreeMap<key, Vec<AccState>>` where each accumulator holds running
state of constant size:

| `StreamingAgg` | Algorithm | State size |
|---|---|---|
| `Count` | counter | 8 B |
| `Sum(col)` | Kahan running sum | 16 B |
| `Mean(col)` | Kahan + count | 24 B |
| `Min(col)` / `Max(col)` | NaN-aware extremum | 16 B |
| `Var(col)` / `Sd(col)` | Welford online variance | 24 B |

Median / Quantile / NDistinct / IQR / First / Last require the full
row index list — not streaming. Callers fall back to the legacy
`summarise`.

**Memory:** O(K · acc_size) instead of O(N · usize). For 100M rows
/ 1000 groups: ~32 KB vs ~800 MB — **25 000× less memory.**

Cat-aware: key tuple is `Vec<u32>` codes when every key is
`Column::Categorical`, `Vec<String>` displays otherwise.

### (b) Cat-aware mutate (Categorical pass-through)

`eval_expr_column` now early-returns when the expression is
`DExpr::Col(name)` and the source is `Column::Categorical` /
`Column::CategoricalAdaptive` — the column is cloned verbatim
(structural Vec clone, not row-by-row re-encoding). Pre-Phase-6 the
row-wise fallback degraded the result to `Column::Str`, breaking
downstream cat-aware fast paths on the new column.

### (c) Lazy-plan optimizer pass

New variant `ViewNode::StreamingGroupSummarise { input, group_keys,
aggregations: Vec<(String, StreamingAgg)> }`. Optimizer pass
`annotate_streamable_summarise` rewrites `GroupSummarise` →
`StreamingGroupSummarise` when every aggregation is in {Count, Sum,
Mean, Min, Max, Var, Sd}. All-or-nothing: any non-streamable agg
keeps the legacy path. Mixed-mode dispatch would require executing
the node twice.

Wired into `execute`, `execute_batched`, `is_pipeline_breaker`, and
all the plan-inspection helpers (`node_output_columns`,
`count_filters`, `innermost`, `kind`, `node_kinds`).

### Headline bench (`bench_phase6_streaming_vs_legacy_summarise`, ignored)

1M rows × 1000 groups, Count + Sum, 3-run avg:

```
Legacy summarise:    588.7 ms
Streaming summarise: 536.9 ms
Speedup:             1.10×
```

The CPU speedup is modest at this shape — both paths are dominated
by per-row Kahan sums + BTreeMap operations; GroupIndex alloc
overhead is small relative to the workload. **The headline Phase 6
streaming win is memory** (25 000× at 100M × 1000), not throughput,
which enables datasets that previously OOM'd. CPU savings scale with
cardinality / row count beyond the L2/L3 fit boundary.

## v3 Phase 7 — Deterministic collection family + DHarht (2026-04-28)

New module `cjc_data::detcoll` ships five deterministic collections,
each tuned to one workload niche. ADR-0019 is the canonical
architecture record.

### Family overview

| Type | Niche | Lookup | Iteration |
|---|---|---|---|
| `IndexVec<I, V>` | dense `IdType → Value` | `O(1)` | insertion order |
| `TinyDetMap<K, V>` | ≤ 16 entries | linear scan | sorted |
| `SortedVecMap<K, V>` | small-medium sealed sorted | binary search | sorted |
| `DetOpenMap<K, V>` | sparse mutable equality | open addressing + BTreeMap fallback | undefined-but-deterministic; `iter_sorted()` for canonical |
| `DHarht<V>` | large sealed equality, byte-addressable | shard → front → microbucket linear scan + BTreeMap overflow | undefined-but-deterministic; `iter_sorted()` for canonical |

### `DHarht` Memory profile

256 shards, splitmix64 scattering, sealed sparse 16-bit front
directory per shard, MicroBucket16 with deterministic BTreeMap
overflow on bucket > 16. Full key equality on every successful
lookup. Per-shard `overflow_count` and `max_bucket_size` counters.
Builds twice from the same input → same `deterministic_shape_hash`.

### TidyView wiring

`ByteDictionary::seal_for_lookup()` is new — it builds a `DHarht<u64>`
mirroring the existing `BTreeMap` lookup state and routes
`lookup()` through it. **Opt-in**: pre-Phase-7 dictionaries that
don't call `seal_for_lookup` see no behavior change. `is_lookup_sealed()`
and `dharht_overflow_count()` surface the post-seal state.

### Backend selection rule

D-HARHT is **not** a global BTreeMap replacement. Use it for
byte-addressable, sealed/read-heavy, deterministic equality lookup.
BTreeMap remains the right choice for canonical ordering, range
queries, diagnostics, serialization.

### Honest performance status

`bench_phase7_dharht_vs_btreemap_lookup` (100k keys, 1M probes):

```
BTreeMap:        296.4 ms
DHarht (sealed): 768.2 ms
Speedup:         0.39× (i.e. ~2.5× SLOWER)
```

The current `DHarht` build is **slower than BTreeMap** on this
workload. The architectural shape, determinism contract, and security
guarantees are delivered today. The constant-factor speed claim is
**deferred** pending the per-shard typed slab allocator + singleton
front-entry fast path + ART fallback. ADR-0019 documents what is
deferred and why.

This is honest reporting — `DHarht` is shipped because the
*architecture* is real (deterministic, secure, correct) and the
infrastructure for the future speed claim is now in place. Users who
need speed today should keep using `BTreeMap`; users who need the
sealed determinism contract for audit/reproducibility benefit
immediately.

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
