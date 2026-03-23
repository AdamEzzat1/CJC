# CJC TidyView Performance Optimization Plan

**Date:** 2025-03-22
**Status:** PLAN ONLY — No implementation yet
**Constraint:** Determinism is the primary objective. All optimizations must preserve bit-identical output.

---

## Current Performance Baseline (Release Mode)

| Operation | 100K rows | 1M rows |
|-----------|-----------|---------|
| Select (zero-copy) | 9.5µs | ~10µs |
| Filter (50% selectivity) | 8.4ms + 8.2ms materialize | 365ms |
| Filter→Select chain | 13.5ms | — |
| Group-by (3 groups) | 76ms | — |
| Group-by (1000 groups) | 915ms | — |
| Group-by (100 groups) | — | 1.62s |
| Full pipeline | 75ms | 773ms |
| Arrange (sort) | 66ms | — |

---

## Bottleneck Analysis (Ranked by Impact)

### CRITICAL: group_by Uses O(N×G) Linear Scan

**Current:** `GroupIndex::build()` uses `Vec::position()` to find existing group keys — a linear scan through all known groups for every row. With 1000 groups and 100K rows, this is **100M string comparisons**.

**Already exists but unused:** `GroupIndex::build_fast()` uses a BTreeMap for O(log G) lookup per row, reducing to O(N × log G). With 1000 groups, that's a **100× reduction** in comparisons.

**Impact:** This single change would likely cut the 100K/1000-group benchmark from 915ms to ~50-100ms.

### CRITICAL: Pervasive String Cloning

Every operation that touches string columns clones the string data:
- `group_by` clones group keys per row
- `filter` evaluation clones string column values for comparison
- `join` converts column values to `String` via `.to_string()` per row
- `distinct` clones key tuples per row
- `materialize` clones every string in every column

For a 100K-row DataFrame with 3 string columns, this means **300K+ String allocations** just to materialize.

### HIGH: mutate() Forces Full Materialization

`mutate()` currently materializes the entire view, then appends a column. Even if only adding one column to a 20-column DataFrame, it copies all 20 columns.

### MEDIUM: summarise() Allocates Vec Per Group Per Aggregator

For each (group, aggregator) pair, a new `Vec<f64>` is allocated, values are gathered, then reduced. With 1000 groups and 5 aggregators, that's 5000 temporary Vec allocations.

### MEDIUM: Row-wise Predicate Evaluation in filter()

Filter evaluates predicates one row at a time via `eval_expr_row()`. No vectorization — each comparison is a scalar operation with branches for type checking and NaN handling.

### LOW: Kahan Summation Overhead

3 FP operations per value (necessary for determinism). Cannot be optimized away without sacrificing reproducibility. This is an accepted cost.

---

## Optimization Plan

### Phase O1: Switch group_by to BTree-Accelerated Build (CRITICAL)

**Estimated speedup: 5-20× on many-group workloads**

**Change:** Make `group_by()` use `GroupIndex::build_fast()` (which already exists) instead of `GroupIndex::build()`.

**Details:**
- `build_fast()` uses `BTreeMap<Vec<String>, usize>` for group key lookup
- O(N × log G) instead of O(N × G)
- BTreeMap guarantees deterministic group ordering (lexicographic)
- First-occurrence ordering can be preserved by using an insertion-order tracker alongside the BTreeMap

**Verification:**
- Result must be bit-identical to current `build()` output
- Same group ordering (first-occurrence or lexicographic — document which)
- 3-run determinism check on 100K rows / 1000 groups

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — change `group_by()` to call `build_fast()`

**Estimated LOC:** ~10 (swapping one function call)

---

### Phase O2: String Interning for Categorical Columns (CRITICAL)

**Estimated speedup: 2-5× on string-heavy operations**

**Change:** Introduce a deterministic string interner that maps strings to compact u32 IDs. Use `Rc<str>` or intern pool to avoid cloning.

**Design:**
```rust
/// Thread-local deterministic string pool.
/// BTreeMap ensures stable ordering across runs.
pub struct StringPool {
    forward: BTreeMap<Rc<str>, u32>,
    reverse: Vec<Rc<str>>,
}

impl StringPool {
    pub fn intern(&mut self, s: &str) -> u32;
    pub fn resolve(&self, id: u32) -> &str;
}
```

**Where to apply:**
1. `Column::Str` storage: Replace `Vec<String>` with `Vec<Rc<str>>` or dictionary-encoded `Vec<u32>` + pool
2. Group key construction: Intern keys instead of cloning strings
3. Join key comparison: Compare interned IDs (u32) instead of string bytes
4. Distinct: Use interned IDs in BTreeSet
5. Filter: Compare interned IDs for equality predicates

**Determinism guarantee:**
- BTreeMap ensures same interning order across runs
- Same string → same ID, always
- Dictionary encoding (already built in Phase 4) can be reused here

**Verification:**
- All existing tests must pass unchanged
- String operations produce identical output
- Memory usage drops measurably

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — Column storage change
- Or: use existing `DictEncoding` from `dict_encoding.rs` as the internal representation

**Estimated LOC:** ~200

---

### Phase O3: Columnar Predicate Evaluation for filter() (HIGH)

**Estimated speedup: 2-4× on filter-heavy pipelines**

**Change:** Instead of evaluating predicates row-by-row, evaluate them column-at-a-time to produce a boolean mask directly.

**Current flow:**
```
for each row:
    if eval_expr_row(predicate, row) == true:
        mask.set(row)
```

**Optimized flow:**
```
// For simple predicates like col("x") > 5.0:
let col_data: &[f64] = get_float_column("x");
let mut mask_words: Vec<u64> = vec![0; (nrows + 63) / 64];
for (i, &val) in col_data.iter().enumerate() {
    if val > 5.0 {
        mask_words[i / 64] |= 1 << (i % 64);
    }
}
```

**Benefits:**
- Sequential memory access on column buffer (cache-friendly)
- No per-row type dispatch overhead
- No DExpr tree walk per row
- Enables future SIMD vectorization (compare 4 f64s at once with AVX2)

**Scope:**
- Fast path for simple predicates: `Col op Literal` (covers ~80% of real filters)
- Fall back to row-wise for complex nested expressions
- Must produce identical BitMask as current implementation

**Verification:**
- Bit-identical BitMask for all predicate types
- No behavior change for NaN handling

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — add `eval_predicate_columnar()` path in `filter()`

**Estimated LOC:** ~150

---

### Phase O4: Lazy mutate() Without Full Materialization (HIGH)

**Estimated speedup: 2-3× on mutate-heavy pipelines**

**Change:** Instead of materializing the entire view to add a column, compute the new column lazily and carry it alongside the view.

**Design:**
```rust
struct TidyView {
    base: Rc<DataFrame>,
    mask: BitMask,
    proj: ProjectionMap,
    // NEW: pending derived columns (computed but not yet materialized into base)
    pending_columns: Vec<(String, Column)>,
}
```

When `mutate("new_col", expr)` is called:
1. Evaluate the expression against visible rows → produce a `Column`
2. Store it in `pending_columns` instead of materializing
3. When `materialize()` or `collect()` is called, merge pending columns with base columns

**Benefits:**
- Avoids copying existing columns when only adding new ones
- Chains of mutate calls only allocate the new columns
- Base DataFrame remains untouched (zero-copy for existing columns)

**Constraint:**
- If mutate replaces an existing column, must update `pending_columns` or materialize
- Subsequent filter/select must be aware of pending columns

**Verification:**
- Identical results to current implementation
- Memory reduction measurable on multi-column DataFrames

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — TidyView struct, mutate(), materialize()

**Estimated LOC:** ~200

---

### Phase O5: Segment-Based Aggregation in summarise() (MEDIUM)

**Estimated speedup: 1.5-3× on grouped aggregation**

**Change:** Use the specialized aggregate kernels from `agg_kernels.rs` (already built!) in the main `summarise()` path.

**Current flow:**
```
for each group:
    for each aggregator:
        gather values into Vec<f64>
        reduce to scalar
```

**Optimized flow:**
```
// Sort data by group keys (or reuse existing group index)
// Build contiguous segments
let segments: Vec<(usize, usize)> = build_segments_from_groups(&group_index);

// For each aggregator, run the specialized kernel once over all segments
let sums = agg_sum_f64(&column_data, &segments);       // single pass
let means = agg_mean_f64(&column_data, &segments);      // single pass
let medians = agg_median_f64(&column_data, &segments);  // single pass
```

**Benefits:**
- Eliminates per-group Vec allocation (5000 allocations → 5)
- Sequential memory access per aggregation pass (cache-friendly)
- Specialized kernels are already written and tested

**Approach:**
1. After grouping, sort data by group keys to make groups contiguous
2. Convert `GroupIndex.groups[i].row_indices` → contiguous segments
3. Run `agg_kernels::agg_*` functions on the contiguous segments
4. Or use `gather_agg_*` functions for non-contiguous indices (slower but no sort needed)

**Trade-off:**
- Sorting data by group keys adds O(N log N) cost up front
- But eliminates O(N × n_aggs) gather allocations
- Net win when n_aggs ≥ 2 (common case)

**Verification:**
- Bit-identical output (Kahan summation preserved)
- Same group ordering

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — `summarise()` implementation

**Estimated LOC:** ~100

---

### Phase O6: Join Key Caching and Index Reuse (MEDIUM)

**Estimated speedup: 1.5-2× on join-heavy workflows**

**Change:** Cache join key string conversions and reuse join indexes when the same table is joined multiple times.

**Current waste:**
- For every row in the left table, `column_value_str()` allocates a new String
- For every row in the right table, same allocation for building the BTreeMap index
- If the same right table is joined against multiple left tables, the index is rebuilt each time

**Optimization:**
1. **Pre-compute join keys as interned IDs** (reuse Phase O2's string pool)
2. **Cache the BTreeMap index** on the right TidyView for reuse
3. **Use u32 key comparison** instead of String comparison in the lookup

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — join implementations

**Estimated LOC:** ~150

---

### Phase O7: Vectorized Column Operations in DExpr (MEDIUM)

**Estimated speedup: 1.5-3× on expression-heavy mutate/filter**

**Change:** Evaluate DExpr operations column-at-a-time instead of row-at-a-time.

**Current:**
```
for each row:
    result[row] = eval_expr_row(expr, row)  // tree walk per row
```

**Optimized:**
```
fn eval_expr_column(expr: &DExpr, df: &DataFrame, mask: &BitMask) -> Column {
    match expr {
        DExpr::BinOp { op: Add, left, right } => {
            let left_col = eval_expr_column(left, df, mask);
            let right_col = eval_expr_column(right, df, mask);
            column_add(&left_col, &right_col)  // tight loop, no dispatch
        }
        DExpr::Col(name) => df.get_column(name).gather(mask),
        DExpr::LitFloat(v) => Column::Float(vec![*v; mask.count_ones()]),
        ...
    }
}
```

**Benefits:**
- Eliminates per-row expression tree walk
- Column operations are tight loops over contiguous memory
- Better CPU pipeline utilization (no branches in inner loop)
- Enables future SIMD on column arithmetic

**Scope:**
- Arithmetic: `+`, `-`, `*`, `/` on Float columns → vectorized
- Comparison: `>`, `<`, `==` → produce boolean mask directly
- Literal broadcast: `col("x") + 5.0` → add scalar to column
- FnCall: `log(col("x"))` → apply to entire column at once

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — add `eval_expr_column()` alongside existing `eval_expr_row()`

**Estimated LOC:** ~250

---

### Phase O8: Distinct via BTreeSet Acceleration (LOW)

**Estimated speedup: 2-10× on high-cardinality distinct**

**Change:** Replace linear duplicate scan in `distinct()` with BTreeSet-based deduplication.

**Current:** O(N × D) — linear scan through seen keys per row
**Optimized:** O(N × log D) — BTreeSet insertion per row

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — `distinct()` implementation

**Estimated LOC:** ~30

---

### Phase O9: Arena Allocator for Temporary Group State (LOW)

**Estimated speedup: 1.2-1.5× on allocation-heavy group-by**

**Change:** Use a bump allocator for the temporary Vecs created during summarise, then free them all at once.

**Current:** Each group's gather Vec is individually heap-allocated and dropped.
**Optimized:** All gather Vecs share a single arena, freed in one deallocation.

**Implementation:**
```rust
struct GroupArena {
    buffer: Vec<f64>,
    offsets: Vec<(usize, usize)>,  // (start, len) per group
}
```

Pre-allocate `buffer` to total visible row count, then slice it for each group's values. Zero individual allocations.

**Files to modify:**
- `crates/cjc-data/src/lib.rs` — `summarise()` gather logic

**Estimated LOC:** ~80

---

## Implementation Order (Recommended)

```
Phase O1 (Fast group_by)               ← CRITICAL, ~10 LOC, biggest single win
  ↓
Phase O2 (String interning)            ← CRITICAL, ~200 LOC, eliminates clone overhead
  ↓
Phase O3 (Columnar filter eval)        ← HIGH, ~150 LOC, cache-friendly filter
  ↓
Phase O4 (Lazy mutate)                 ← HIGH, ~200 LOC, avoids full materialization
  ↓
Phase O5 (Segment-based aggregation)   ← MEDIUM, ~100 LOC, uses existing kernels
  ↓
Phase O6 (Join key caching)            ← MEDIUM, ~150 LOC, reduces string allocation
  ↓
Phase O7 (Vectorized DExpr)            ← MEDIUM, ~250 LOC, eliminates row-wise tree walk
  ↓
Phase O8 (Fast distinct)               ← LOW, ~30 LOC, quick win
  ↓
Phase O9 (Arena allocator)             ← LOW, ~80 LOC, reduces allocation pressure
```

---

## Expected Performance After All Optimizations

| Operation | Current (100K) | After Optimizations | Speedup |
|-----------|---------------|---------------------|---------|
| Group-by (1000 groups) | 915ms | ~50-100ms | **9-18×** |
| Group-by (3 groups) | 76ms | ~15-25ms | **3-5×** |
| Filter (50%) | 16.6ms total | ~5-8ms | **2-3×** |
| Summarise (5 aggs, 50 groups) | 394ms | ~80-150ms | **2-5×** |
| Full pipeline 1M | 773ms | ~150-300ms | **2-5×** |
| Mutate (add 1 col to 20-col df) | ~25ms | ~5-10ms | **2-5×** |
| Inner join (string keys) | — | ~2× faster | **~2×** |
| Distinct (high cardinality) | O(N×D) | O(N log D) | **5-20×** |

**Overall realistic improvement for typical data science pipelines: 3-8× faster**

---

## Determinism Verification Protocol

For EACH optimization phase:

1. **Bit-identical output:** Run the exact same pipeline before and after, compare all output values at the bit level (`.to_bits()` for f64)
2. **3-run determinism:** Same input → identical output across 3 runs
3. **Cross-optimization determinism:** Output with all optimizations enabled must match output with all disabled
4. **Kahan audit:** Verify all floating-point reductions still use Kahan summation
5. **BTreeMap audit:** No HashMap/HashSet in any new code
6. **No FMA:** Verify no fused multiply-add instructions

---

## What This Does NOT Touch

- The compiler pipeline (Lexer → Parser → AST → HIR → MIR → Exec)
- The type system or type inference
- The memory model or GC/NoGC boundary
- Existing API contracts (all public function signatures unchanged)
- The determinism contract (strengthened, not weakened)
- View Fusion IR (deferred to separate plan)

---

## Estimated Total Scope

| Phase | LOC | Priority | Main Win |
|-------|-----|----------|----------|
| O1: Fast group_by | ~10 | CRITICAL | 9-18× on many-group |
| O2: String interning | ~200 | CRITICAL | 2-5× on string-heavy |
| O3: Columnar filter | ~150 | HIGH | 2-3× on filter |
| O4: Lazy mutate | ~200 | HIGH | 2-3× on mutate chains |
| O5: Segment aggregation | ~100 | MEDIUM | 1.5-3× on summarise |
| O6: Join key caching | ~150 | MEDIUM | ~2× on joins |
| O7: Vectorized DExpr | ~250 | MEDIUM | 1.5-3× on expressions |
| O8: Fast distinct | ~30 | LOW | 5-20× on distinct |
| O9: Arena allocator | ~80 | LOW | 1.2-1.5× on alloc-heavy |
| **Total** | **~1,170** | — | **3-8× overall** |
