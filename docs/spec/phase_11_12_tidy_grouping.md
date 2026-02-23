# Phase 11–12: Tidy Grouping, Sorting, Slicing, and Joins — Specification and Implementation Log

## Objective

Extend the Phase 10 zero-allocation tidy view layer (`cjc-data`) with **grouping**
(`group_by` / `ungroup`), **aggregation** (`summarise`), **stable sorting** (`arrange`),
**row slicing** (`slice` / `slice_head` / `slice_tail` / `slice_sample`), **deduplication**
(`distinct`), and **relational joins** (`semi_join`, `anti_join`, `inner_join`, `left_join`).

All operations share the same non-negotiables as Phase 10:
- **Bit-deterministic**: identical inputs → identical outputs, every run, every platform.
- **Allocation-transparent**: view-only ops produce no column buffers; only materialising
  ops (summarise, arrange, joins) allocate new column data.
- **NoGC-safe builtins**: view ops are registered in the `cjc-mir` safe-builtin list;
  materialising ops are intentionally excluded so the NoGC verifier rejects them inside
  `@nogc` functions.
- **Zero regressions** across the full CJC test suite.

---

## Changes

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| `cjc-data/src/lib.rs` | Phase 10: filter/select/mutate only | Added `RowIndexMap`, `GroupMeta`, `GroupIndex`, `GroupedTidyView`, `TidyAgg`, `ArrangeKey`; extended `TidyView` with `group_by`, `arrange`, `slice*`, `distinct`, and four join methods |
| `cjc-data/src/lib.rs` — `TidyError` | 6 variants | Added `EmptyGroup` variant |
| `cjc-mir/src/nogc_verify.rs` | Phase 10 tidy safe builtins | Added Phase 11-12 view-safe builtins: `tidy_group_by`, `tidy_ungroup`, `tidy_ngroups`, `tidy_slice`, `tidy_slice_head`, `tidy_slice_tail`, `tidy_slice_sample`, `tidy_distinct`, `tidy_semi_join`, `tidy_anti_join` |
| `tests/tidy_tests/` | 11 files (Phase 10) | Added 7 new test files: 89 new tests (no perf-gated additions) |
| `tests/tidy_tests/mod.rs` | Phase 10 modules | Extended with Phase 11 and 12 module declarations |

### Why

Phase 10's view layer covers per-row predicates and column projection. The next tier of
data-frame operations requires: grouping rows by key columns (no allocation), computing
per-group aggregates (allocates one result row per group), sorting rows by one or more
sort keys (materialises a permuted copy), subsetting rows by position (RowIndexMap view),
deduplicating by key column set (RowIndexMap view), and matching rows across two frames
(semi/anti are views; inner/left materialise new frames with possibly multiplied rows).

---

## Spec-Lock Table

The following invariants are tested and must not regress.

| Property | Rule |
|----------|------|
| **Group order** | First-occurrence: groups are emitted in the order the first matching row appears in the visible row sequence. No hash-map nondeterminism. |
| **Null / missing** | CJC has no `null` type for Int/Float/Bool. Str columns use `""` as sentinel for absent join keys in left outer joins. Int uses `i64::MIN` as the null sentinel. |
| **NaN sort position** | In ascending sort, `NaN` sorts **last**. In descending sort, `NaN` sorts **first** (first ≡ largest). This mirrors R's `na.last = TRUE` default. |
| **Stable sort** | `arrange()` uses Rust `slice::sort_by` (guaranteed stable). Equal rows preserve their original relative order. |
| **Slice out-of-bounds** | `slice(start, end)` clamps to `[0, nrows]`. Never panics. |
| **slice_sample output order** | Selected row indices are sorted ascending before materialisation. Output is deterministic for a given seed regardless of LCG internal traversal order. |
| **Distinct order** | First-occurrence: first row with each distinct key tuple is retained, in the order rows appear. |
| **Join key matching** | Keys are compared as strings (via `Display` on `Column`). Cross-type key matching (Int key on left, Str key on right with same printed value) is intentionally undefined — callers must use matching column types. |
| **Join left-row order** | `inner_join` and `left_join` preserve the order of left rows. For each left row, matched right rows appear in their left-to-right order in the right frame. |
| **Join right-lookup determinism** | The right-side lookup is built as a `Vec<(key_tuple, row_index)>` sorted by `(key, row_index)` before binary-search lookup. No HashMap. |
| **Left join null fill** | Unmatched left rows yield the left columns unchanged, right columns filled with type-appropriate sentinels (`i64::MIN` for Int, `f64::NAN` for Float, `""` for Str, `false` for Bool). |
| **Empty group aggregate** | `count` → `0.0`, `sum` → `0.0`, `mean` → `f64::NAN`, `min/max` → `f64::NAN`, `first/last` → `TidyError::EmptyGroup`. |
| **summarise output** | One row per group. First `K` columns are key columns (in `group_by` key order), followed by aggregate output columns. |
| **arrange materialises** | `arrange()` always materialises a new `DataFrame` with all visible columns in sorted order. Returns a fresh `TidyView` (identity mask, identity projection) over the new frame. |
| **NoGC boundary** | View ops (group_by, ungroup, slice*, distinct, semi_join, anti_join) are registered safe builtins. Materialising ops (arrange, summarise, inner_join, left_join) are NOT registered → NoGC verifier rejects them inside `@nogc`. |

---

## Data Structures

### `RowIndexMap`

```
pub struct RowIndexMap {
    indices: Vec<usize>,   // row indices into the TidyView's visible rows
}
```

- Used by `slice`, `slice_head`, `slice_tail`, `slice_sample`, `distinct`, `semi_join`,
  `anti_join` to produce a view over a selected/filtered subset of rows without column
  allocation.
- `indices` are always in **ascending order** (stable left-to-right row traversal).

### `GroupMeta`

```
pub struct GroupMeta {
    key_values: Vec<String>,   // stringified key column values for this group
    row_indices: Vec<usize>,   // indices into the base DataFrame (absolute)
}
```

### `GroupIndex`

```
pub struct GroupIndex {
    groups: Vec<GroupMeta>,   // in first-occurrence order
    key_names: Vec<String>,   // key column names in the order passed to group_by()
}
```

- Built by `GroupIndex::build()` via a linear scan over visible rows. Groups are
  appended to a `Vec<(key_tuple, slot)>` on first encounter, looked up via linear
  search for subsequent rows in the same group.
- **No `HashMap`** — eliminates hash-order nondeterminism.
- Complexity: O(N × G) where G = number of unique groups. Acceptable for the group
  counts typical in tidy analytics; a hash-accelerated path can be added in a later
  phase without changing the observable ordering contract.

### `GroupedTidyView`

```
pub struct GroupedTidyView {
    view: TidyView,
    index: GroupIndex,
}
```

- Produced by `TidyView::group_by(key_cols)`.
- `ungroup()` → returns the inner `TidyView` unchanged.
- `summarise(aggs)` → materialises one-row-per-group `DataFrame` → `TidyView`.
- `group_index()` → `&GroupIndex` (white-box access for tests).

### `TidyAgg`

```
pub enum TidyAgg {
    Count,
    Sum(String),    // column name
    Mean(String),
    Min(String),
    Max(String),
    First(String),
    Last(String),
}
```

All numeric aggregations (`Sum`, `Mean`, `Min`, `Max`) operate on `f64`. Integer columns
are cast to `f64` before aggregation; the output column is always `Column::Float`.
`Count` output is `Column::Float` (value = group size as `f64`).
`First` / `Last` preserve the source column type.

Float summation uses **Kahan compensated summation** (via `cjc_repro::kahan_sum_f64`)
for deterministic float accumulation independent of compiler vectorisation.

### `ArrangeKey`

```
pub struct ArrangeKey {
    col_name: String,
    descending: bool,
}
// Constructors: ArrangeKey::asc(name), ArrangeKey::desc(name)
```

Multi-key sort: primary key first, secondary key breaks ties, etc.
Stability: equal rows under all keys preserve original relative order.

---

## API Surface

### Phase 11

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `TidyView::group_by(keys)` | `&[&str] → Result<GroupedTidyView, TidyError>` | Grouped view | `GroupIndex` (O(N) scan, O(G×K) storage) |
| `GroupedTidyView::ungroup()` | `→ TidyView` | Inner view | None |
| `GroupedTidyView::summarise(aggs)` | `&[(&str, TidyAgg)] → Result<TidyView, TidyError>` | 1-row-per-group frame | O(G × A) column buffers |
| `GroupedTidyView::group_index()` | `→ &GroupIndex` | Index ref | None |
| `TidyView::arrange(keys)` | `&[ArrangeKey] → Result<TidyView, TidyError>` | Sorted view | O(N × K) — new DataFrame |
| `TidyView::slice(start, end)` | `usize, usize → TidyView` | Row-range view | O(end-start) indices |
| `TidyView::slice_head(n)` | `usize → TidyView` | First-N view | O(n) indices |
| `TidyView::slice_tail(n)` | `usize → TidyView` | Last-N view | O(n) indices |
| `TidyView::slice_sample(n, seed)` | `usize, u64 → TidyView` | Random-N view | O(n) indices |

### Phase 12

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `TidyView::distinct(key_cols)` | `&[&str] → Result<TidyView, TidyError>` | De-duped view | O(M) indices (M = distinct count) |
| `TidyView::semi_join(right, left_on, right_on)` | `&TidyView, &[&str], &[&str] → Result<TidyView, TidyError>` | Left rows with match | O(M) indices |
| `TidyView::anti_join(right, left_on, right_on)` | `&TidyView, &[&str], &[&str] → Result<TidyView, TidyError>` | Left rows without match | O(M) indices |
| `TidyView::inner_join(right, left_on, right_on)` | `&TidyView, &[&str], &[&str] → Result<TidyFrame, TidyError>` | Matched rows, both cols | O(N×K) — new DataFrame |
| `TidyView::left_join(right, left_on, right_on)` | `&TidyView, &[&str], &[&str] → Result<TidyFrame, TidyError>` | All left rows + matched right | O(N×K) — new DataFrame |

---

## Semantics

### group_by

- **Key columns** must exist in the view's projected columns. Unknown column →
  `TidyError::ColumnNotFound`.
- **Group identity**: key tuple is the concatenated `Display` representations of the key
  columns for each row, separated by `\x00` (NUL byte), ensuring no false collisions
  between `("a", "b")` and `("ab", "")`.
- **Group order**: first-occurrence in the visible row sequence (post-filter, post-mask).
- **Ungrouped view**: `ungroup()` returns the original `TidyView` with mask and
  projection unchanged. Group metadata is discarded.

### summarise

- **Output**: one row per group, plus one row for the degenerate case of a 0-group
  input (returns empty frame, 0 rows).
- **Key columns first**: first K output columns are the group key values, as `Column::Str`
  (stringified from group key). Aggregate output columns follow.
- **Duplicate output name**: `TidyError::DuplicateColumn`.
- **Unknown aggregate column**: `TidyError::ColumnNotFound`.
- **Kahan sum**: deterministic float accumulation — same result regardless of instruction
  reordering, FMA fusion, or vectorisation choices by the compiler.

### arrange

- **Stable**: equal rows (under all sort keys) preserve relative order from the input.
- **NaN last**: in ascending order, `NaN` is treated as greater than any non-NaN float.
  In descending order, `NaN` appears first (largest position).
- **Materialises**: `arrange()` copies all visible rows and all visible columns into a new
  `DataFrame`, then returns a `TidyView` with identity mask and identity projection over
  the new frame. This is the only Phase 11 operation that allocates column buffers.
- **Unknown key column**: `TidyError::ColumnNotFound`.
- **After filter**: visible (masked-in) rows only are sorted and materialised into the
  new frame. The resulting frame contains exactly `mask.count_ones()` rows.

### slice / slice_head / slice_tail

- **Indices**: positional within the visible row sequence (i.e., the i-th row according
  to `iter_set()` of the current mask).
- **Clamping**: `start` and `end` are silently clamped to `[0, nrows]`. No panic or error.
- **Zero-n**: `slice_head(0)` → empty view; `slice_tail(0)` → empty view.
- **No column allocation**: results are `TidyView` backed by a `RowIndexMap`.

### slice_sample

- **LCG**: Knuth multiplicative LCG — `rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)`.
- **Partial Fisher-Yates**: builds a random permutation of `[0, nrows)` using the LCG,
  taking the first `n` elements, then sorts them ascending.
- **Clamping**: if `n > nrows`, returns all rows (in original order). If `n == 0`,
  returns empty view.
- **Determinism**: same `(n, seed, nrows)` → identical output indices, every run,
  every platform.
- **Output order**: row indices are sorted ascending (stable left-to-right order in
  result), not in random order.

### distinct

- **Key columns**: the subset of columns to deduplicate on. Unknown column →
  `TidyError::ColumnNotFound`.
- **Empty key list**: treated as all-columns-are-key; every row is its own key, so all
  rows are retained (no deduplication).
- **First-occurrence**: of each distinct key tuple is retained; duplicates are dropped.
- **Output**: a `TidyView` backed by a `RowIndexMap` of the retained row indices.
  No column buffers allocated.

### semi_join / anti_join

- **Left-only output**: result contains only the columns (and rows) of the left view.
  Right columns never appear in the output.
- **semi_join**: left rows for which at least one match exists in the right view on the
  specified key columns.
- **anti_join**: left rows for which no match exists in the right view.
- **Right-side lookup**: built as a `Vec<key_tuple>`, sorted and deduplicated, for
  O(log R) binary-search lookup per left row. No HashMap.
- **Preserves left row order**.

### inner_join

- **Output columns**: all left columns, then all right columns excluding the join key
  columns (to avoid duplicate key columns).
- **Row multiplication**: one output row for each (left_row, right_row) match pair.
- **No match**: left row is excluded from output (inner semantics).
- **Left row order preserved**: all match rows for left row i precede all match rows for
  left row j when i < j.
- **Right match order**: for a given left row, right match rows appear in ascending
  right-row-index order (deterministic due to sorted right-side lookup).
- **Materialises**: returns `TidyFrame`.

### left_join

- **Output columns**: same as `inner_join` (left cols + non-key right cols).
- **Unmatched left rows**: included with right columns filled by type sentinels:
  - `Column::Int` → `i64::MIN`
  - `Column::Float` → `f64::NAN`
  - `Column::Str` → `""`
  - `Column::Bool` → `false`
- **Matched left rows**: expanded to one output row per right match, same as inner_join.
- **Left row order preserved**.
- **Materialises**: returns `TidyFrame`.

---

## Lowering Strategy (MIR / Runtime)

Phase 11-12 operations lower to named builtins in the CJC MIR layer, identical to
Phase 10's strategy.

**NoGC-safe builtins** (registered in `cjc-mir/src/nogc_verify.rs`):

| Builtin | Why safe |
|---------|----------|
| `tidy_group_by` | Produces `GroupIndex` (Vec of Vecs) — no GC-traced column buffers |
| `tidy_ungroup` | Returns inner TidyView — no allocation |
| `tidy_ngroups` | Read-only metadata access |
| `tidy_slice` | RowIndexMap (Vec<usize>) — no column buffers |
| `tidy_slice_head` | RowIndexMap — no column buffers |
| `tidy_slice_tail` | RowIndexMap — no column buffers |
| `tidy_slice_sample` | RowIndexMap — no column buffers |
| `tidy_distinct` | RowIndexMap — no column buffers |
| `tidy_semi_join` | RowIndexMap — no column buffers |
| `tidy_anti_join` | RowIndexMap — no column buffers |

**Intentionally NOT registered** (materialise column buffers → rejected inside `@nogc`):

| Builtin | Why rejected |
|---------|-------------|
| `tidy_arrange` | Copies O(N×K) column data into new DataFrame |
| `tidy_summarise` | Allocates O(G×A) aggregate result columns |
| `tidy_inner_join` | Allocates joined output frame, possibly O(N²) rows |
| `tidy_left_join` | Same as inner_join |

---

## Edge Cases Covered

### group_by
- [x] Empty DataFrame → 0 groups, empty GroupIndex
- [x] Single key column → correct group membership
- [x] Multi-key grouping (composite key)
- [x] First-occurrence ordering
- [x] Row assignments to groups (white-box GroupIndex check)
- [x] Unknown key column → `TidyError::ColumnNotFound`
- [x] group_by after filter → only masked-in rows considered
- [x] ungroup → view is unchanged
- [x] Int column as key
- [x] Bool column as key

### summarise
- [x] Count per group
- [x] Sum per group
- [x] Mean per group
- [x] Min per group
- [x] Max per group
- [x] First / Last per group
- [x] Empty group: count → 0.0, sum → 0.0, mean → NaN
- [x] Empty group: first/last → `TidyError::EmptyGroup`
- [x] Float determinism (Kahan sum, 1000-element groups)
- [x] Duplicate output name → `TidyError::DuplicateColumn`
- [x] Unknown aggregate column → `TidyError::ColumnNotFound`
- [x] Key columns present in output (first K columns)
- [x] Int column sum (cast to Float output)
- [x] Multiple groups, one row each

### arrange
- [x] Basic ascending sort
- [x] Descending sort
- [x] Stable tie-breaking (equal keys preserve original order)
- [x] NaN sorts last in ascending
- [x] NaN sorts first in descending
- [x] Multi-key sort (primary + secondary key)
- [x] Unknown key column → `TidyError::ColumnNotFound`
- [x] Empty DataFrame → 0-row result, no panic
- [x] arrange after filter → only visible rows sorted
- [x] group_by after arrange → groups on sorted order

### slice
- [x] Basic range `[start, end)`
- [x] Empty range `[n, n)` → 0-row view
- [x] Out-of-bounds clamped silently
- [x] Empty DataFrame → 0-row view
- [x] `slice_head(n)` → first n rows
- [x] `slice_head(n > nrows)` → all rows
- [x] `slice_head(0)` → empty view
- [x] `slice_tail(n)` → last n rows
- [x] `slice_tail(n > nrows)` → all rows
- [x] `slice_tail(0)` → empty view
- [x] `slice_sample` deterministic (same seed)
- [x] `slice_sample` different seeds produce different results
- [x] `slice_sample(n > nrows)` → all rows
- [x] `slice_sample(0)` → empty view
- [x] `slice_sample` output indices in ascending order
- [x] slice after filter → operates on visible rows only

### distinct
- [x] Single key column deduplication
- [x] Multi-column key deduplication
- [x] Empty key list → all rows retained
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] First-occurrence ordering preserved
- [x] Empty DataFrame → 0-row view
- [x] All unique → all rows retained
- [x] All same → one row retained
- [x] distinct after filter → operates on visible rows
- [x] distinct after select/projection → deduplicates on projected columns
- [x] `to_tensor()` on distinct result

### semi_join / anti_join
- [x] semi_join basic (some matches)
- [x] semi_join no matches → empty result
- [x] semi_join all match → all left rows
- [x] semi_join unknown left key → `TidyError::ColumnNotFound`
- [x] semi_join unknown right key → `TidyError::ColumnNotFound`
- [x] semi_join preserves left row order
- [x] anti_join basic (some matches excluded)
- [x] anti_join no matches → all left rows returned
- [x] anti_join all match → empty result

### inner_join
- [x] One-to-one join
- [x] Many-to-one join (fan-out on left)
- [x] Output ordering deterministic (left-row order, then right-row order for ties)
- [x] Unknown left key column → `TidyError::ColumnNotFound`
- [x] Empty result (no key matches)
- [x] Left row order preserved
- [x] `to_tensor()` on join result

### left_join
- [x] Basic left join (some matches)
- [x] All left rows match → no null fills
- [x] No matches → all right columns null-filled
- [x] One-to-many explosion (left row duplicated for each right match)
- [x] left join after filter + arrange (pipeline composition)
- [x] Many-to-many explosion ordering deterministic

### NoGC
- [x] `gc_alloc` inside `@nogc` rejected (baseline)
- [x] Clean `@nogc` function passes
- [x] `tidy_slice` inside `@nogc` does not cite `gc_alloc`
- [x] `tidy_semi_join` inside `@nogc` does not cite `gc_alloc`
- [x] `tidy_group_by` inside `@nogc` does not cite `gc_alloc`
- [x] `tidy_distinct` inside `@nogc` does not cite `gc_alloc`

---

## Tests Added

All new tests are under `tests/tidy_tests/`, entry point `tests/test_phase10_tidy.rs`
(shared with Phase 10).

| File | Tests | Description |
|------|-------|-------------|
| `test_phase11_group_by.rs` | 10 | Empty df, single/multi key, first-occurrence order, row assignments, unknown col, group_by after filter, ungroup, Int key, Bool key |
| `test_phase11_summarise.rs` | 14 | count/sum/mean/min/max/first/last, empty-group sentinels, Kahan float determinism, duplicate output name, unknown col, key cols in output, Int col sum, multiple groups |
| `test_phase11_arrange.rs` | 10 | Asc/desc, stable ties, NaN last/first, multi-key, unknown col, empty df, arrange after filter, group_by after arrange |
| `test_phase11_slice.rs` | 16 | slice range/empty/clamp, empty df, slice_head/tail (normal/clamp/zero), slice_sample (determinism/seeds/clamp/zero/order), slice after filter |
| `test_phase12_distinct.rs` | 11 | Single/multi col, zero cols, unknown col, first-occurrence, empty df, all-unique, all-same, after filter, after projection, to_tensor |
| `test_phase12_joins.rs` | 22 | semi/anti (basic/no-match/all-match/errors/order), inner (1:1, M:1, deterministic order, unknown col, empty, left-order, to_tensor), left (basic, all-match, null-fill, 1:M, pipeline, M:M determinism) |
| `test_phase11_12_nogc.rs` | 6 | gc_alloc rejected, clean passes, slice/semi_join/group_by/distinct safe |

**Phase 11-12 total: 89 new tests (no perf-gated additions).**

Combined with Phase 10 (37 non-ignored + 1 ignored):
**Tidy test suite total: 126 non-ignored tests, 1 ignored (perf gate).**

---

## Regression Results

**Tidy tests (Phase 10 + 11 + 12):**

```
running 127 tests
test result: ok. 126 passed; 0 failed; 1 ignored; finished in 0.05s
```

**Full CJC suite:**

```
test result: ok. 1316 passed; 0 failed; 1 ignored
```

All 46 test binaries passed. Zero regressions introduced.

Previous totals by phase:
- Stages 1–2.4: 535 tests
- Phase 10: +37 → 572 active
- Phase 11-12: +89 → 661 active + 1 ignored

(The 1,316 total includes all test binaries including stress tests, benchmarks, and
the MIR/optimizer/parity/milestone suites.)

---

## Known Limitations

1. **GroupIndex is O(N × G)** in the degenerate case (many unique groups). For analytic
   workloads with G << N this is fast in practice. A hash-accelerated variant can be
   added in a later phase without changing the observable first-occurrence ordering
   contract.

2. **Join key type matching** uses `Display` stringification. Cross-type keys (e.g., Int
   on left matching Str on right) are not detected at validation time — they will simply
   fail to match. Future phases should add key-type validation at `resolve_join_keys()`.

3. **Left-join null sentinels** are type-specific magic values (`i64::MIN`, `f64::NAN`,
   `""`, `false`). CJC has no first-class `null` type in the current runtime. A proper
   nullable column type (e.g., `Column::NullableInt(Vec<Option<i64>>)`) would be cleaner
   and is planned for a future phase.

4. **Kahan sum** is used for `TidyAgg::Sum` and `TidyAgg::Mean`. `Min` and `Max` operate
   on raw `f64` comparisons and are not sensitive to accumulation order. `First` and
   `Last` do no arithmetic. No special handling is needed beyond Kahan for the current
   aggregate set.

5. **`slice_sample` with replacement** is not supported. All sampling is without
   replacement. `n > nrows` silently returns all rows.

6. **Many-to-many inner joins** can produce O(N²) output rows. There is no guard or
   warning. Callers are responsible for ensuring join cardinality is appropriate for
   available memory.

7. **`arrange` on a 0-column view** returns a 0-column, 0-row frame (all columns are
   invisible through the projection). This is correct but may be surprising; callers
   should ensure the view has visible columns before sorting.

---

## Appendix: Algorithm Notes

### GroupIndex::build() — first-occurrence linear scan

```
key_to_slot: Vec<(String, usize)>  // (key_tuple, group_slot_index)

for each visible_row in visible_rows:
    key = join(key_col_values_for_row, "\x00")
    if let Some(slot) = key_to_slot.linear_find(key):
        groups[slot].row_indices.push(visible_row)
    else:
        slot = groups.len()
        key_to_slot.push((key, slot))
        groups.push(GroupMeta { key_values, row_indices: [visible_row] })
```

Rationale: O(N × G) is acceptable for typical tidy analytics (G is small relative to N).
The deliberate avoidance of `HashMap` eliminates all hash-seed nondeterminism.

### build_right_lookup() — sorted Vec for deterministic binary search

```
let mut lookup: Vec<(String, usize)> = right_visible_rows
    .map(|row| (row_key(right_base, key_cols, row), row))
    .collect();
lookup.sort_unstable();  // lexicographic on (key_string, row_index)
```

Binary search via `partition_point` returns a deterministic contiguous range of matches.
Row index is part of the sort key to break ties deterministically.

### compare_column_rows() — NaN-last float comparison

```
match (a_is_nan, b_is_nan):
    (true,  true)  => Equal
    (true,  false) => Greater  // NaN last in ascending
    (false, true)  => Less
    (false, false) => a_val.partial_cmp(b_val).unwrap_or(Equal)
```

For descending sort, the outer `arrange` loop calls `.reverse()` on the `Ordering`,
so `NaN` becomes `Less` → sorts to the front (NaN first in descending).

### slice_sample LCG + partial Fisher-Yates

```
seed = knuth_lcg(seed)
indices = (0..nrows).collect::<Vec<_>>()
for i in 0..min(n, nrows):
    j = (seed as usize % (nrows - i)) + i
    indices.swap(i, j)
    seed = knuth_lcg(seed)
let mut selected = indices[..n].to_vec()
selected.sort_unstable()  // ascending output order
```

`knuth_lcg(x) = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)`
