# Phase 13–16: Tidy Completion — Specification and Implementation Log

## Objective

Complete the CJC tidy layer by adding **reshaping** (pivot_longer, pivot_wider), **column
management** (rename, relocate, drop_cols, bind_rows, bind_cols), **scoped transforms**
(across()), a **real nullability model** (NullableColumn / NullCol / NullableFrame),
**join maturity** (typed keys, suffix collision handling, right_join, full_join), and a
**group performance upgrade** (BTree-accelerated GroupIndex) — while preserving all Phase
10-12 semantics and achieving zero regressions across the full CJC test suite.

---

## Changes

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| `cjc-data/src/lib.rs` — TidyError | 7 variants | Added `schema_mismatch`, `join_type_mismatch`, `duplicate_key`, `empty_selection` helper constructors |
| `cjc-data/src/lib.rs` — NullableColumn | Did not exist | Added `NullableColumn<T>` (values + validity bitmap); `NullCol` enum; `NullableFrame` |
| `cjc-data/src/lib.rs` — TidyView | Phase 10-12 methods | Added `pivot_longer`, `pivot_wider`, `rename`, `relocate`, `drop_cols`, `bind_rows`, `bind_cols`, `mutate_across`, `right_join`, `full_join`, `inner_join_typed`, `left_join_typed`, `group_by_fast` |
| `cjc-data/src/lib.rs` — GroupedTidyView | `summarise` only | Added `mutate_across`, `summarise_across` |
| `cjc-data/src/lib.rs` — GroupIndex | `build()` linear scan | Added `build_fast()` BTree-accelerated variant |
| `cjc-data/src/lib.rs` — new types | None | `NullableColumn<T>`, `NullCol`, `NullableFrame`, `AcrossTransform`, `AcrossSpec`, `JoinSuffix`, `RelocatePos` |
| `cjc-mir/src/nogc_verify.rs` — safe builtins | Phase 10-12 list | Added `tidy_relocate`, `tidy_drop_cols`, `tidy_group_by_fast` |
| `cjc-mir/src/nogc_verify.rs` — may_gc loop | Only skipped `is_gc_builtin` | Also skips `is_safe_builtin` (bug fix: safe builtins were incorrectly reported as may-GC) |
| `tests/tidy_tests/` | 18 files (Phase 10-12) | Added 9 new test files, 127 new tests |
| `tests/tidy_tests/mod.rs` | Phase 10-12 modules | Extended with Phase 13-16 module declarations |

### Why

Phase 10-12 established the lazy view layer and core grouping/join/sort operations.
Phase 13-16 completes the "full dplyr/tidyr surface":
- **Pivot** is the canonical "tidy" reshape operation; strict mode ensures determinism.
- **Column verbs** (rename, relocate, drop, bind) handle the schema manipulation that
  every real analysis workflow needs.
- **Across** enables batch column transforms without hand-enumerating column names.
- **Nullables** replace sentinel fills (i64::MIN, NaN, "", false) with a proper validity
  bitmap model, enabling callers to distinguish "absent" from "zero/NaN/empty".
- **Join maturity** adds type validation to prevent silent non-matches from type mismatches,
  and suffix handling for collision-safe full-outer join output.
- **Group perf** replaces the O(N×G) linear key scan with a O(N log G) BTree lookup
  while preserving the external first-occurrence ordering contract.

---

## Spec-Lock Table

| Property | Rule |
|----------|------|
| **pivot_longer row order** | Original row order preserved; within each source row, value columns appear in the order supplied in `value_cols`. |
| **pivot_longer col order** | `[id_cols..., names_to, values_to]`. |
| **pivot_longer zero value_cols** | `TidyError::Internal("empty selection: ...")` |
| **pivot_longer mixed dtypes** | Strict: all value_cols must have the same type. `TidyError::TypeMismatch`. |
| **pivot_longer duplicate in value_cols** | `TidyError::DuplicateColumn`. |
| **pivot_wider col order** | `[id_cols..., unique_key_values_in_first_occurrence_order]`. |
| **pivot_wider row order** | One row per unique id-col combination; first-occurrence order. |
| **pivot_wider duplicate (id, key) combo** | Strict: `TidyError::DuplicateColumn("duplicate key: ...")`. |
| **pivot_wider missing combinations** | Null fill via `NullCol` validity bitmap (NOT sentinels). |
| **rename collision** | `TidyError::DuplicateColumn`. |
| **rename unknown col** | `TidyError::ColumnNotFound`. |
| **rename noop (old==new)** | Silently accepted. |
| **relocate ordering** | Stable: non-moved columns retain their relative order. |
| **relocate unknown col or anchor** | `TidyError::ColumnNotFound`. |
| **drop_cols unknown col** | `TidyError::ColumnNotFound`. |
| **drop_cols all cols** | Valid: returns 0-col view. |
| **bind_rows schema** | Strict: column names must match exactly in order. `TidyError::Internal("schema mismatch: ...")`. |
| **bind_rows row order** | Left rows, then right rows. |
| **bind_cols row mismatch** | `TidyError::LengthMismatch`. |
| **bind_cols name collision** | `TidyError::DuplicateColumn`. |
| **across expansion order** | Stable column iteration (order of `cols` in AcrossSpec). |
| **across generated names** | `{col}_{fn_name}` (default) or user-specified template with `{col}` and `{fn}` placeholders. |
| **across name collision** | Overwrite for existing cols; `TidyError::DuplicateColumn` only for new-name conflicts. |
| **NullableColumn validity bitmap** | Same layout as BitMask: LSB-first per word, tail bits zeroed. |
| **NullCol in pivot_wider null fill** | Validity = false (absent), value field = type zero (not observable). |
| **NullCol to_column_filled** | Float nulls → NaN; Int nulls → 0; Str nulls → ""; Bool nulls → false. |
| **join type validation** | Only compatible types allowed: Int↔Int, Float↔Float, Str↔Str, Bool↔Bool, Int↔Float (numeric widening). Otherwise `TidyError::TypeMismatch`. |
| **join suffix** | Default `.x`/`.y`. User may specify any strings. Applied only on non-key column name collisions. |
| **right_join row order** | Right outer loop order preserved. Left unmatched cols → null-filled. |
| **full_join row order** | Left rows first (matched and unmatched), then unmatched right rows. |
| **group_by_fast semantics** | Identical to `group_by`: first-occurrence group order, same row assignments, same key_names. |
| **NoGC boundary additions** | `tidy_relocate`, `tidy_drop_cols`, `tidy_group_by_fast` → SAFE. All materialising ops → NOT SAFE. |
| **NoGC verifier bug fix** | Safe builtins are now also skipped in the "may_gc" check loop (was: only gc_builtins were skipped, causing safe builtins to be falsely reported as may-GC). |

---

## Data Structures

### `NullableColumn<T>`

```
pub struct NullableColumn<T: Clone> {
    pub values: Vec<T>,
    pub validity: BitMask,  // validity.get(i) = true → values[i] is valid
}
```

- **Validity bit**: `true` = valid (non-null); `false` = null.
- **Tail bits**: always zero (same contract as `BitMask`).
- **Memory**: O(N) for values + O(N/8) for bitmap.
- **Zero-copy gather**: `NullableColumn::gather(&indices)` produces a new column with
  gathered values and validity bits.

### `NullCol`

```
pub enum NullCol {
    Int(NullableColumn<i64>),
    Float(NullableColumn<f64>),
    Str(NullableColumn<String>),
    Bool(NullableColumn<bool>),
}
```

Key conversions:
- `NullCol::from_column(col)` → fully valid (all bits set).
- `NullCol::to_column_strict()` → `Ok(Column)` if all valid, `Err` if any null.
- `NullCol::to_column_filled()` → fills nulls with type-appropriate zero values.
- `NullCol::null_of_type(type_name, len)` → all-null column of given type.
- `NullCol::get_display(i)` → "null" for null cells.

### `NullableFrame`

```
pub struct NullableFrame {
    pub columns: Vec<(String, NullCol)>,
}
```

- Output type for `pivot_wider`, `right_join`, `full_join`.
- `to_dataframe_filled()` → converts to regular `DataFrame` with null fills.
- `to_tidy_view_filled()` → convenience: materialize + wrap as `TidyView`.
- `to_tidy_frame_filled()` → same for `TidyFrame`.

### `AcrossSpec` / `AcrossTransform`

```
pub struct AcrossTransform {
    pub fn_name: String,
    pub func: Box<dyn Fn(&str, &Column) -> Result<Column, TidyError>>,
}

pub struct AcrossSpec {
    pub cols: Vec<String>,
    pub transform: AcrossTransform,
    pub name_template: Option<String>,  // None → "{col}_{fn}"
}
```

- `AcrossTransform::new(fn_name, fn_body)` — construct a named transform.
- `AcrossSpec::new(cols, transform)` — select columns and apply transform.
- `AcrossSpec::with_template(tmpl)` — override the generated name format.
- `AcrossSpec::output_name(col_name)` — compute output column name for a given input.

### `JoinSuffix`

```
pub struct JoinSuffix {
    pub left: String,   // default ".x"
    pub right: String,  // default ".y"
}
```

Applied to non-key columns when left and right frames share a column name.

### `RelocatePos<'a>`

```
pub enum RelocatePos<'a> {
    Front,
    Back,
    Before(&'a str),
    After(&'a str),
}
```

### `GroupIndex::build_fast()` (BTree-accelerated)

```
pub fn build_fast(base: &DataFrame, key_col_indices: &[usize],
                  visible_rows: &[usize], key_names: Vec<String>) -> Self
// Uses BTreeMap<Vec<String>, usize> for O(log G) key lookup per row.
// Output ordering: identical to build() (first-occurrence).
```

---

## API Surface

### Phase 13 — Reshape + Column Management

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `TidyView::pivot_longer` | `(&[&str], &str, &str) → Result<TidyFrame, TidyError>` | Long-format frame | O(N×K) — new columns |
| `TidyView::pivot_wider` | `(&[&str], &str, &str) → Result<NullableFrame, TidyError>` | Wide nullable frame | O(M×K) — new columns |
| `TidyView::rename` | `(&[(&str, &str)]) → Result<TidyView, TidyError>` | View over new base | O(N×K) base rebuild |
| `TidyView::relocate` | `(&[&str], RelocatePos) → Result<TidyView, TidyError>` | Reordered projection | O(K) indices |
| `TidyView::drop_cols` | `(&[&str]) → Result<TidyView, TidyError>` | Narrowed projection | O(K) indices |
| `TidyView::bind_rows` | `(&TidyView) → Result<TidyFrame, TidyError>` | Concatenated rows | O(N×K) |
| `TidyView::bind_cols` | `(&TidyView) → Result<TidyFrame, TidyError>` | Concatenated columns | O(N×K) |

### Phase 14 — Across + Nullable

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `TidyView::mutate_across` | `(&[AcrossSpec]) → Result<TidyFrame, TidyError>` | Mutated frame | O(N×K) |
| `GroupedTidyView::mutate_across` | `(&[AcrossSpec]) → Result<TidyFrame, TidyError>` | Mutated frame | O(N×K) |
| `GroupedTidyView::summarise_across` | `(&[AcrossSpec]) → Result<TidyFrame, TidyError>` | Aggregate frame | O(G×A) |
| `NullCol::from_column` | `(&Column) → NullCol` | Fully valid nullable | O(N/8) bitmap |
| `NullCol::to_column_strict` | `→ Result<Column, TidyError>` | Unwrapped column | None |
| `NullCol::to_column_filled` | `→ Column` | Filled column | O(N) |
| `NullableFrame::to_dataframe_filled` | `→ DataFrame` | Filled DataFrame | O(N×K) |

### Phase 15 — Join Maturity

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `TidyView::inner_join_typed` | `(&TidyView, &[(&str,&str)], &JoinSuffix) → Result<TidyFrame, TidyError>` | Inner-join frame | O(N×K) |
| `TidyView::left_join_typed` | `(&TidyView, &[(&str,&str)], &JoinSuffix) → Result<TidyFrame, TidyError>` | Left-join frame | O(N×K) |
| `TidyView::right_join` | `(&TidyView, &[(&str,&str)], &JoinSuffix) → Result<NullableFrame, TidyError>` | Right-join nullable frame | O(N×K) |
| `TidyView::full_join` | `(&TidyView, &[(&str,&str)], &JoinSuffix) → Result<NullableFrame, TidyError>` | Full-outer nullable frame | O(N×K) |

### Phase 16 — Group Perf

| Method | Signature | Returns | Allocation |
|--------|-----------|---------|-----------|
| `TidyView::group_by_fast` | `(&[&str]) → Result<GroupedTidyView, TidyError>` | Grouped view (BTree) | O(G log G) BTree + O(G×K) GroupIndex |

---

## Semantics

### pivot_longer

- Explicit column list required. All value columns must have the same type.
- Output row `j` (0-indexed): source row `j / len(value_cols)`, value column
  `j % len(value_cols)` (in the order supplied in `value_cols`).
- Schema: `[projected_non_value_cols, names_to, values_to]`.
- After `filter()`: only visible rows are pivoted.

### pivot_wider

- `names_from` column values become new output column headers.
- `values_from` column values fill the cells.
- `id_cols` combination determines output row identity.
- Duplicate (id, name) pair → strict error.
- Missing combination → null fill via `NullCol` validity bitmap.
- Key values in first-occurrence order (scan of visible rows).

### rename

- Rebuilds the base `DataFrame` with renamed columns. The mask and projection are
  preserved (indices still valid since only names change, not positions).
- Collision check: if `new_name` already exists in the base (among columns not being
  renamed), → `TidyError::DuplicateColumn`.

### relocate

- Pure projection-map update: no column buffers copied.
- `Front`: `[moved_cols, remaining_in_original_order]`
- `Back`: `[remaining_in_original_order, moved_cols]`
- `Before(anchor)`: insert moved_cols immediately before anchor in the remaining list.
- `After(anchor)`: insert moved_cols immediately after anchor in the remaining list.

### bind_rows

- Strict schema: same column names in same order. Any deviation → error.
- Row concatenation is via `gather_column` on the two visible row sets.
- Type widening: Int + Float → Float (same as existing `concat_columns`).

### bind_cols

- Row count must match exactly.
- No column name collision allowed.
- Pure column concatenation: both sets of visible rows gathered and merged.

### mutate_across / summarise_across

- **mutate_across**: materializes self, then applies each transform column-by-column.
  Result is appended or replaces existing column with the generated name.
- **summarise_across**: for each group, applies the transform to the group's row slice.
  The transform function MUST return a single-element column (scalar reduction).
- Both respect the `AcrossSpec::name_template` if provided.

### inner_join_typed / left_join_typed

- Same semantics as `inner_join` / `left_join` but with:
  1. Type validation before join (Int↔Float allowed; others must match).
  2. Suffix collision handling: non-key columns with the same name get `.x`/`.y` (or user suffix) appended.

### right_join

- Implemented as swapped left-join then column reordering:
  `right_join(right, on)` = conceptually `right.left_join(self, swapped_on)` with
  column order [left_cols (nullable), right_cols].
- Left columns are `NullCol` (nullable); unmatched right rows produce null left cols.

### full_join

- Phase 1: left outer loop — all left rows, right match or null for unmatched.
- Phase 2: right rows not matched in Phase 1 — prepend left cols as null.
- Output row order: all Phase 1 rows (left order), then Phase 2 rows (right order).
- Returned as `NullableFrame` since both sides may be null.

### group_by_fast

- Identical semantics to `group_by`. Drop-in replacement.
- Internal: `BTreeMap<Vec<String>, usize>` for O(log G) key lookup during scan.
- First-occurrence ordering guaranteed: groups are appended to `groups: Vec<GroupMeta>`
  on their first encounter; the BTree is only used for lookup, not for output ordering.

---

## Lowering Strategy (MIR / Runtime)

### NoGC-safe builtins (newly registered)

| Builtin | Why safe |
|---------|----------|
| `tidy_relocate` | Updates `ProjectionMap` only — O(K) usize allocation |
| `tidy_drop_cols` | Updates `ProjectionMap` via `select()` — O(K) |
| `tidy_group_by_fast` | BTree-based `GroupIndex` — no column buffer allocation |

### Intentionally NOT registered (materialising)

| Builtin | Why excluded |
|---------|-------------|
| `tidy_pivot_longer` | Allocates O(N×K) column buffers |
| `tidy_pivot_wider` | Allocates `NullableFrame` with O(M×K) column buffers |
| `tidy_bind_rows` | Allocates concatenated column buffers |
| `tidy_bind_cols` | Allocates merged column buffers |
| `tidy_mutate_across` | Materialises base + applies transforms |
| `tidy_summarise_across` | Allocates aggregate result frame |
| `tidy_right_join` | Allocates `NullableFrame` |
| `tidy_full_join` | Allocates `NullableFrame` |
| `tidy_inner_join_typed` | Allocates `TidyFrame` |
| `tidy_left_join_typed` | Allocates `TidyFrame` |
| `tidy_rename` | Rebuilds base `DataFrame` (O(N×K) clone) |

### NoGC verifier bug fix

Prior to Phase 13-16, the "may_gc function call" check loop only skipped `is_gc_builtin`
before consulting `may_gc_map`. Because safe builtins (like `tidy_filter`) are not in
`may_gc_map`, they were hitting `unwrap_or(true)` → incorrectly reported as "may trigger GC"
when transitively called from @nogc functions. The fix adds `|| is_safe_builtin(callee)`
to the skip guard, making safe builtins invisible to the may-GC propagation path.

---

## Edge Cases Covered

### pivot_longer
- [x] Empty DataFrame → 0-row output
- [x] Explicit column list order is preserved in output
- [x] Empty `value_cols` → `TidyError::Internal("empty selection")`
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] Duplicate in `value_cols` → `TidyError::DuplicateColumn`
- [x] Mixed types → `TidyError::TypeMismatch`
- [x] String value columns → `Column::Str` output
- [x] Determinism across two runs
- [x] After filter (only visible rows pivoted)

### pivot_wider
- [x] Basic schema: id_cols first, then key values
- [x] Row count = unique id combinations
- [x] Values correct
- [x] First-occurrence key column ordering
- [x] Determinism across two runs
- [x] Missing combination → null fill via NullCol validity
- [x] Duplicate (id, key) → `TidyError::DuplicateColumn`
- [x] Unknown names_from, values_from, id_col → `TidyError::ColumnNotFound`
- [x] Empty DataFrame → 0-row output
- [x] `to_tidy_view_filled()` works on result

### bind_rows
- [x] Row count = left + right
- [x] Row order: left then right
- [x] Column order matches left frame
- [x] Empty right → left unchanged
- [x] Empty left → right rows only
- [x] After filter (only visible rows concatenated)
- [x] Schema mismatch → `TidyError::Internal`
- [x] Type mismatch in column → error

### bind_cols
- [x] Column count = left + right
- [x] Column order: left then right
- [x] Values correct
- [x] Row count mismatch → `TidyError::LengthMismatch`
- [x] Name collision → `TidyError::DuplicateColumn`

### rename
- [x] Basic rename
- [x] Multiple renames
- [x] Noop (`old == new`) accepted silently
- [x] Values preserved after rename
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] Collision → `TidyError::DuplicateColumn`
- [x] Mask preserved after rename

### relocate
- [x] Move to front, back, before, after
- [x] Multiple columns moved
- [x] Non-moved columns retain relative order
- [x] Values preserved after relocation
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] Unknown anchor → `TidyError::ColumnNotFound`

### drop_cols
- [x] Single column dropped
- [x] Multiple columns dropped
- [x] All columns dropped → 0-col view
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] Row count preserved after drop

### across (mutate_across)
- [x] Basic: generates `{col}_{fn}` columns
- [x] Values correct
- [x] Original columns preserved
- [x] Custom name template
- [x] Empty cols list → no-op
- [x] Unknown column → `TidyError::ColumnNotFound`
- [x] Expansion order stable (cols list order)
- [x] Determinism across two runs
- [x] Ungrouped and grouped produce same row-wise result

### summarise_across
- [x] Basic: one aggregate row per group
- [x] Values correct
- [x] Duplicate output name → `TidyError::DuplicateColumn`
- [x] Unknown column → `TidyError::ColumnNotFound`

### Nullable semantics
- [x] `NullableColumn::from_values` → all valid
- [x] Explicit validity with nulls
- [x] `get()` returns None for null
- [x] `gather()` preserves nulls
- [x] `gather()` on null index preserves null
- [x] `NullCol::from_column` → fully valid
- [x] `to_column_strict` succeeds when all valid
- [x] `to_column_strict` fails when any null
- [x] `to_column_filled` → Float nulls → NaN
- [x] type_name correct for all variants
- [x] `get_display` returns "null" for null
- [x] `null_of_type` → all-null column
- [x] Validity bitmap tail bits clean (65-row test)
- [x] `NullableFrame::to_dataframe_filled` works
- [x] `NullableFrame::column_names` correct

### join maturity
- [x] Type mismatch → `TidyError::TypeMismatch` for inner_join_typed
- [x] Int↔Float compatible (no error)
- [x] Type mismatch for left_join_typed
- [x] Suffix applied on collision (default `.x`/`.y`)
- [x] Custom suffix works
- [x] No suffix when no collision
- [x] left_join_typed: all left rows retained
- [x] left_join_typed: null fill for unmatched
- [x] right_join: all right rows retained
- [x] right_join: right columns present
- [x] right_join: type mismatch error
- [x] full_join: row count = matched + unmatched left + unmatched right
- [x] full_join: left unmatched → right cols null
- [x] full_join: type mismatch error
- [x] full_join: deterministic across runs
- [x] inner_join_typed: correct row count
- [x] inner_join_typed: left row order preserved

### group perf upgrade
- [x] `group_by_fast` produces same `ngroups` as `group_by`
- [x] First-occurrence ordering preserved
- [x] Identical row assignments to slow variant
- [x] `summarise` output identical
- [x] Empty DataFrame → 0 groups
- [x] Single group
- [x] All-unique groups
- [x] Multi-key identical to slow
- [x] Deterministic across two runs
- [x] After filter → same groups as slow

### NoGC negative (Phase 13-16)
- [x] `tidy_pivot_longer` rejected in @nogc
- [x] `tidy_pivot_wider` rejected in @nogc
- [x] `tidy_bind_rows` rejected in @nogc
- [x] `tidy_bind_cols` rejected in @nogc
- [x] `tidy_mutate_across` rejected in @nogc
- [x] `tidy_right_join` rejected in @nogc
- [x] `tidy_full_join` rejected in @nogc
- [x] `tidy_inner_join_typed` rejected in @nogc
- [x] `tidy_summarise_across` rejected in @nogc
- [x] `tidy_rename` rejected in @nogc
- [x] `tidy_relocate` ACCEPTED in @nogc
- [x] `tidy_drop_cols` ACCEPTED in @nogc
- [x] `tidy_group_by_fast` ACCEPTED in @nogc
- [x] Clean function with mixed safe builtins passes

---

## Tests Added

All new tests in `tests/tidy_tests/`, entry point `tests/test_phase10_tidy.rs`.

| File | Tests | Description |
|------|-------|-------------|
| `test_phase13_pivot_longer.rs` | 13 | Basic row count/schema/order, empty df, zero cols error, unknown col, duplicate col, mixed type error, Str values, determinism, after filter |
| `test_phase13_pivot_wider.rs` | 12 | Schema, row count, values, first-occurrence col order, determinism, null fill via NullCol, duplicate key error, unknown col errors, empty df, to_tidy_view_filled |
| `test_phase13_bind.rs` | 13 | bind_rows: row count/order/schema/empty/after-filter/errors; bind_cols: col count/order/values/row-mismatch/collision |
| `test_phase13_rename_relocate_drop.rs` | 20 | Rename: basic/multiple/noop/values/errors; Relocate: front/back/before/after/multi/errors/values; Drop: single/multi/all/error/row-count |
| `test_phase14_across.rs` | 13 | mutate_across: basic/values/original-cols/template/noop/unknown/order/determinism/grouped-ungrouped; summarise_across: basic/values/dup-name-error/unknown |
| `test_phase14_nullable_semantics.rs` | 15 | NullableColumn: validity/get/gather/null-gather; NullCol: from_column/to_column_strict/strict-with-null/filled-float-NaN/type_name/display/null_of_type; tail-bit safety; NullableFrame conversions |
| `test_phase15_join_maturity.rs` | 17 | Type mismatch/Int-Float-widening/suffix-collision/custom-suffix/no-suffix; left_join_typed: rows/null-fill; right_join: rows/cols/type-error; full_join: row-count/null-check/type-error/determinism; inner_join_typed: row-count/order |
| `test_phase16_group_perf_semantics.rs` | 10 | ngroups/first-occurrence/row-assignments/summarise-output/empty/single-group/all-unique/multi-key/determinism/after-filter — all vs slow baseline |
| `test_phase16_nogc_negative.rs` | 14 | 10 materialising ops rejected + 3 view ops accepted + clean-function baseline |

**Phase 13-16 total: 127 new tests (0 perf-gated).**

Combined tidy test suite (Phase 10 + 11 + 12 + 13-16):
**253 non-ignored tests, 1 ignored (Phase 10 perf gate).**

---

## Regression Results

**Tidy tests (Phase 10 + 11 + 12 + 13-16):**

```
running 254 tests
test result: ok. 253 passed; 0 failed; 1 ignored; finished in 0.12s
```

**Full CJC suite:**

```
test result: ok. 1443 passed; 0 failed; 1 ignored
```

All 46+ test binaries passed. Zero regressions introduced.

Cumulative test growth:
- Stages 1–2.4: 535 tests
- Phase 10: +37 → 572 active
- Phase 11-12: +89 → 661 active
- Phase 13-16: +127 → **788 active + 1 ignored**

(The 1,443 total includes all test binaries: unit tests, stress tests, benchmarks,
MIR/optimizer/parity/milestone suites, and all tidy tests.)

---

## Known Limitations

1. **`NullCol` is not integrated into `Column`** — the two types coexist. Operations that
   return `NullableFrame` require explicit conversion (`to_tidy_view_filled()` etc.). A
   future phase should unify `Column` and `NullCol` under a single nullable column type.

2. **`bind_rows` requires exact schema match** (strict mode). A "coercion mode" that
   fills missing columns with nulls (like R's `dplyr::bind_rows`) is not yet implemented.

3. **`summarise_across` transform functions must return a scalar (1-element column)**.
   Multi-row reduce functions (like `quantile`) are not supported in the current interface.

4. **`rename` rebuilds the entire base `DataFrame`** (O(N×K) clone), making it ineligible
   for `@nogc`. A future optimization could store a rename map in the view metadata
   without copying column data.

5. **`pivot_wider` uses `Display` for key matching** — same limitation as Phase 12 joins.
   Cross-type keys (Int column value "42" vs Str "42") would falsely match or fail. A
   typed key comparison should be added in a future phase.

6. **Full-join null semantics**: callers must check `NullCol::is_null()` explicitly or
   use `to_column_filled()`. There is no automatic propagation of null through arithmetic
   expressions in `DExpr` (a future "three-valued logic DExpr" extension).

7. **`across` function closures are boxed** (`Box<dyn Fn(...)>`). For hot loops, callers
   should use `mutate()` with explicit `DExpr` instead.
