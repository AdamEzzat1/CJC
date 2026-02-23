# Phase 10: Tidy Primitives — Specification and Implementation Log

## Objective

Integrate CrowleyFrame-style tidy operations directly into CJC's `cjc-data` runtime
as **deterministic, allocation-controlled view generators** (`filter`, `select`) and a
**copy-on-write mutate kernel** (`mutate`), with full regression safety across the
entire CJC test suite.

---

## Changes

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| `cjc-data/src/lib.rs` | No view layer; filter materializes immediately | Added `BitMask`, `ProjectionMap`, `TidyView`, `TidyFrame` + `filter`/`select`/`mutate`/`materialize` |
| `cjc-data/src/lib.rs` — `DExpr` | No boolean literal | Added `LitBool(bool)` variant + eval + Display |
| `cjc-mir/src/nogc_verify.rs` | No tidy builtins | Added `tidy_filter`, `tidy_select`, `tidy_mask_and`, `tidy_nrows`, `tidy_ncols`, `tidy_column_names` to safe-builtin list |
| `tests/tidy_tests/` | Did not exist | 11 test files, 37 non-ignored tests, 1 perf gate (ignored) |
| `tests/test_phase10_tidy.rs` | Did not exist | Entry point for `cargo test --test test_phase10_tidy` |

### Why

The existing `Pipeline` / `LogicalPlan` API materializes at every step. For hot paths
that chain many filters or projections, this allocates O(N × K) new column buffers
per step. Phase 10 introduces a lazy view layer: filter and select produce
zero-allocation views backed by a bitmask and an index list; data is only copied at
explicit `materialize()` / `to_tensor()` / `mutate()` calls.

---

## Data Structures

### `BitMask`

```
struct BitMask {
    words: Vec<u64>,   // u64 words, LSB-first within word
    nrows: usize,      // total row count (tail bits above nrows are zero)
}
```

- **Bit `i`**: word `i/64`, bit position `i % 64`.
- **Tail bits**: always zero — construction zeroes bits `nrows..64*nwords`.
- **Chaining**: `a.and(&b)` ≡ bitwise AND of all words — O(N/64).
- **Memory**: O(N/8) bytes for N rows (1 bit/row).
- **Iteration**: `iter_set()` yields row indices in ascending order.

### `ProjectionMap`

```
struct ProjectionMap {
    indices: Vec<usize>,   // Column indices into base DataFrame.columns
}
```

- **Empty selection**: `ProjectionMap::from_indices(vec![])` is valid.
- **Ordering**: exactly the order provided by the caller.
- **No allocation** of column buffers — metadata only.

### `TidyView`

```
struct TidyView {
    base: Rc<DataFrame>,     // shared, immutable reference to original data
    mask: BitMask,           // which rows are visible
    proj: ProjectionMap,     // which columns are visible and in what order
}
```

- Cloning a `TidyView` clones the `Rc` pointer (cheaply) and deep-copies the mask
  and projection (O(N/64 + K)).
- The base `DataFrame` is never mutated through a `TidyView`.

### `TidyFrame`

```
struct TidyFrame {
    inner: Rc<RefCell<DataFrame>>,
}
```

- Produced by `TidyView::mutate()` (always materialized).
- `.mutate()` on a `TidyFrame` with refcount > 1 triggers a deep clone before
  mutation (copy-on-write alias safety).

---

## API Surface

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `DataFrame::tidy()` | `self → TidyView` | Full view | 1 `Rc` + mask + proj alloc |
| `DataFrame::tidy_mut()` | `self → TidyFrame` | Mutable frame | 1 `Rc<RefCell<_>>` alloc |
| `TidyView::filter(predicate)` | `&DExpr → Result<TidyView,TidyError>` | New view with tighter mask | O(N/64) words |
| `TidyView::select(cols)` | `&[&str] → Result<TidyView,TidyError>` | New view with narrower projection | O(K) indices |
| `TidyView::mutate(assignments)` | `&[(&str, DExpr)] → Result<TidyFrame,TidyError>` | Materialized frame | O(N×K) |
| `TidyView::materialize()` | `→ Result<DataFrame,TidyError>` | New owned DataFrame | O(N×K) |
| `TidyView::to_tensor(cols)` | `&[&str] → Result<Tensor,TidyError>` | New Tensor | O(N×K) |
| `TidyFrame::mutate(assignments)` | `&mut _, &[(&str,DExpr)] → Result<(),TidyError>` | In-place or CoW | CoW if shared |

---

## Semantics

### filter

- **Input validation**: all column references in predicate must exist in `base`. Evaluated against full base (not just projected columns), consistent with SQL WHERE semantics.
- **Predicate type**: must evaluate to `Bool`. Non-bool → `TidyError::PredicateNotBool`.
- **NaN floats**: IEEE 754 — `NaN != NaN` is `false`, `NaN > x` is `false`. Deterministic.
- **Chain composition**: `v.filter(A).filter(B)` merges bitmasks with AND. Equivalent to single `v.filter(A && B)`.
- **Tail bits**: guaranteed zero — iteration never yields phantom rows outside `nrows`.
- **Empty frame**: 0-row input produces 0-row view; no panic.

### select

- **Zero columns**: allowed; yields a valid `TidyView` with `ncols() == 0`.
- **Unknown column**: `TidyError::ColumnNotFound(name)`.
- **Duplicate names in request**: `TidyError::DuplicateColumn(name)`.
- **Ordering**: exactly as supplied by caller; reordering is correct in `materialize()` and `to_tensor()`.
- **No buffer allocation**.

### mutate

- **Snapshot semantics**: all assignments see the column state *before* the `mutate` call. A new column created in assignment `i` is NOT visible to assignment `j > i` in the same call. (Use chained `.view().mutate(...)` for sequential semantics.)
- **Existing column**: overwritten.
- **New column**: appended after existing columns.
- **Duplicate target in one call**: `TidyError::DuplicateColumn(name)`.
- **Type promotion**: `Int + Float → Float`. `Int + Int → Int` (wrapping). `Float + LitInt → Float`.
- **Empty DataFrame**: succeeds; creates empty-length columns.
- **Mask-awareness**: `TidyView::mutate()` materializes only masked-in rows. The result has exactly `mask.count_ones()` rows.
- **Alias safety**: `TidyFrame::mutate()` copy-on-write — if `Rc` refcount > 1, deep-clones before mutation.

### materialize

- Rows emitted in ascending index order (stable/deterministic).
- Edge: empty rows → 0-row DataFrame. Empty projection → 0-column DataFrame.
- Allocation: exactly one new buffer per visible column.

---

## Lowering Strategy (MIR / Runtime)

Phase 10 is a pure Rust library layer in `cjc-data`. CJC language programs access
it via builtin function calls that lower to:

- `tidy_filter(view, predicate_fn)` → safe builtin (no GC), calls bitmask builder
- `tidy_select(view, col_names)` → safe builtin (no GC), updates projection map
- `tidy_materialize(view)` → allocating builtin (NOT safe inside `@nogc`)
- `tidy_mutate(view, assignments)` → allocating builtin (NOT safe inside `@nogc`)

The `cjc-mir/src/nogc_verify.rs` safe-builtin list includes `tidy_filter`,
`tidy_select`, `tidy_mask_and`, `tidy_nrows`, `tidy_ncols`, `tidy_column_names`.
Materializing operations are excluded from the safe list, causing the NoGC verifier
to reject them inside `@nogc` functions.

---

## Edge Cases Covered

### Filter
- [x] Empty DataFrame (0 rows) — returns 0-row view, no panic
- [x] Predicate returns non-bool type → `TidyError::PredicateNotBool`
- [x] Float NaN comparisons — deterministic IEEE 754 semantics
- [x] Chained filters — AND mask composition
- [x] Mask tail bits — never leak phantom rows
- [x] Missing column in predicate → `TidyError::ColumnNotFound`

### Select
- [x] Selecting 0 columns — valid empty-column view
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] Duplicate column names → `TidyError::DuplicateColumn`
- [x] Reordering columns — correct in `materialize()` and `to_tensor()`
- [x] Select after filter, select after mutate (via chain)

### Mutate
- [x] Mutate on 0-row DataFrame — succeeds, empty columns
- [x] Type promotion: Int + Float → Float
- [x] Int + Int → Int (stays Int)
- [x] Float + LitInt → Float (promotes)
- [x] Overwriting existing column
- [x] Snapshot semantics — later assignment cannot see earlier in same call
- [x] Duplicate target in one call → `TidyError::DuplicateColumn`
- [x] Mutate over masked view — only visible rows in result
- [x] Mutate on empty mask → 0-row materialized result
- [x] Alias safety via CoW in `TidyFrame`

### Materialization
- [x] Empty rows → 0-row result
- [x] Empty projection → 0-column result
- [x] Masked view → rows in ascending order
- [x] Projection-reordered → columns in projection order
- [x] `to_tensor()` on projected + masked view

### Performance / NoGC
- [x] Filter allocates only O(N/64) mask words
- [x] Select allocates only O(K) index entries
- [x] Mutate allocates exactly one buffer per new column
- [x] NoGC verifier rejects `gc_alloc` inside `@nogc` (existing + new test)
- [x] `tidy_filter` / `tidy_select` in safe-builtin list

---

## Tests Added

All tests under `tests/tidy_tests/`, entry point `tests/test_phase10_tidy.rs`.

| File | Tests | Description |
|------|-------|-------------|
| `test_tidy_filter_empty_result.rs` | 2 | Zero-match filter, column names preserved |
| `test_tidy_filter_empty_df.rs` | 3 | 0-row DataFrame filter, mask validity, unknown col still errors |
| `test_tidy_filter_chain_mask_merge.rs` | 4 | Chained vs single-pass equivalence, 3-level chain, all-masked-out |
| `test_tidy_select_reorder.rs` | 5 | Column reorder, to_tensor, single col, all cols, reorder+filter |
| `test_tidy_select_zero_cols.rs` | 3 | 0-col view, materialize, empty df |
| `test_tidy_mutate_type_promotion.rs` | 4 | Int+Float→Float, Int+Int→Int, Float+LitInt→Float, empty df |
| `test_tidy_mutate_ordering.rs` | 5 | Snapshot semantics, independent assignments, dup target, overwrite, sequential chain |
| `test_tidy_mutate_masked_view.rs` | 3 | Masked mutate, empty mask, projected+masked mutate |
| `test_tidy_alias_safety.rs` | 3 | View safety, two independent frames, clone+mutate CoW |
| `test_tidy_nogc_rejection.rs` | 3 | gc_alloc rejected, clean passes, tidy_filter is safe |
| `test_tidy_speed_gate.rs` | 3 | Smoke test, allocation budget, perf gate (ignored) |

**Total: 37 non-ignored tests, 1 perf-gated (#[ignore]) test.**

---

## Regression Results

**Tidy tests:** 37 passed, 0 failed, 1 ignored (perf gate)

**Full CJC suite:**

```
test result: ok. 1227 passed; 0 failed; 1 ignored
```

All 46 test binaries passed. Zero regressions introduced.

Previously: 535 tests (Stages 1–2.4).
Phase 10 adds 37 active tests → suite total: 572 active + 1 ignored.

(The 1,227 figure includes all test binaries including stress tests and benchmarks.)

---

## Known Limitations

1. **No `DExpr::LitBool` before Phase 10** — added as part of this phase to support boolean literal predicates in filter expressions.
2. **`validate_expr_columns_snapshot`** validates only column names, not types. Type mismatches surface at `eval_expr_column` time (runtime error).
3. **`mutate` on a `TidyView`** always materializes first (mask applied). There is no lazy mutate path that defers materialization.
4. **Int overflow** is wrapping (Rust `i64` arithmetic). Overflow is silent and deterministic, consistent with CJC's existing Int semantics in `eval_binop`.
5. **`Str` columns in mutate**: string arithmetic is limited to equality/inequality comparisons; arithmetic ops on strings return `TidyError::Internal` (propagated from `DataError`).
6. **`Bool + Int`** promotion is not defined — returns `TidyError::TypeMismatch`. This is intentional to avoid ambiguous semantics.
