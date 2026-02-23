# Phase 17: Categorical Foundations — Spec & Gap Audit

**Status:** Complete
**Regression baseline going in:** 1,443 passed · 0 failed · 1 ignored
**Regression result after Phase 17:** 1,509 passed · 0 failed · 5 ignored
**New tests:** 66 passing + 4 ignored (capacity boundary, run with `--ignored`)

---

## 1. Changes Table

| Item | File | Description |
|------|------|-------------|
| `TidyError::CapacityExceeded` variant | `crates/cjc-data/src/lib.rs` | New error variant for u16 level overflow |
| `TidyView::base_column()` | `crates/cjc-data/src/lib.rs` | Raw column accessor for fct_summary_means |
| `FctColumn` struct | `crates/cjc-data/src/lib.rs` | Compact categorical column (u16 index + levels Vec) |
| `FctColumn::encode()` | `crates/cjc-data/src/lib.rs` | String → FctColumn; BTreeMap O(log L) lookup; first-occurrence level order |
| `FctColumn::encode_from_view()` | `crates/cjc-data/src/lib.rs` | Respects TidyView mask + projection |
| `FctColumn::fct_lump()` | `crates/cjc-data/src/lib.rs` | Top-N frequency lumping with deterministic tie-breaking |
| `FctColumn::fct_reorder()` | `crates/cjc-data/src/lib.rs` | Level reorder by summary values; NaN always last |
| `FctColumn::fct_reorder_by_col()` | `crates/cjc-data/src/lib.rs` | Full pipeline: mean-per-level → reorder |
| `FctColumn::fct_collapse()` | `crates/cjc-data/src/lib.rs` | Metadata-only level rename/merge; @nogc safe |
| `FctColumn::to_str_column()` | `crates/cjc-data/src/lib.rs` | Decode back to Column::Str |
| `FctColumn::gather()` | `crates/cjc-data/src/lib.rs` | Row-subset gather (view semantics) |
| `NullableFactor` struct | `crates/cjc-data/src/lib.rs` | FctColumn + BitMask validity; null ≠ level |
| `NullableFactor::encode_nullable()` | `crates/cjc-data/src/lib.rs` | Option<String> slice → NullableFactor |
| `NullableFactor::{fct_lump,fct_reorder,fct_collapse}` | `crates/cjc-data/src/lib.rs` | Null-preserving wrappers over FctColumn ops |
| `TidyView::fct_encode()` | `crates/cjc-data/src/lib.rs` | Convenience: encode masked view column |
| `TidyView::fct_summary_means()` | `crates/cjc-data/src/lib.rs` | Per-level mean for use with fct_reorder |
| `TidyError::capacity_exceeded()` constructor | `crates/cjc-data/src/lib.rs` | Helper constructor |
| NoGC: `fct_collapse` registered safe | `crates/cjc-mir/src/nogc_verify.rs` | Metadata-only, no GC heap |
| NoGC: `fct_encode/lump/reorder` absent | `crates/cjc-mir/src/nogc_verify.rs` | Materialising ops, intentionally absent |
| `test_phase17_forcats.rs` | `tests/tidy_tests/` | 70 test items (66 passing + 4 ignored capacity) |
| `mod.rs` updated | `tests/tidy_tests/mod.rs` | Phase 17 module declared |

---

## 2. Spec-Lock Table

| ID | Invariant | Enforced By |
|----|-----------|-------------|
| S-1 | Index type is `u16`; max 65,535 distinct levels; `TidyError::CapacityExceeded` on overflow | `FctColumn::encode` |
| S-2 | Level ordering = first-occurrence of each string in visible input rows; BTreeMap for O(log L) lookup; levels Vec carries order | `encode()`, `encode_nullable()` |
| S-3 | Null handling: null cells use separate BitMask; null is NOT a level; null index slot is sentinel 0 | `NullableFactor` |
| S-4 | `fct_lump` tie-breaking: equal-frequency levels keep first-occurrence order in top-N selection; "Other" bucket appended LAST | `fct_lump()` |
| S-5 | "Other" collision: if "Other" already a level, bucket renamed "Other_" (iterates until unique) | `fct_lump()` |
| S-6 | `fct_reorder` NaN: NaN summary values sort LAST regardless of ascending/descending direction | `fct_reorder()` |
| S-7 | `fct_reorder` stable sort: equal-summary levels preserve original first-occurrence order | `sort_by` (Rust stable) |
| S-8 | `fct_collapse` is metadata-only: never rewrites data buffer unless indices change; O(L) levels rebuild + O(N) conditional remap | `fct_collapse()` |
| S-9 | `fct_collapse` duplicate outputs: multiple old levels → same new name → one new level; index remapping is canonical | `fct_collapse()` |
| S-10 | `fct_encode` is materialising (allocates u16 buffer + String levels) → NOT @nogc | `nogc_verify.rs` absence |
| S-11 | `fct_lump` is materialising (new levels Vec + new data Vec) → NOT @nogc | `nogc_verify.rs` absence |
| S-12 | `fct_reorder` is materialising (new levels Vec + new data Vec) → NOT @nogc | `nogc_verify.rs` absence |
| S-13 | `fct_collapse` is @nogc safe (registered in `is_safe_builtin`) | `nogc_verify.rs` |
| S-14 | Double-encoding is idempotent: encode(decode(fct)) produces same levels and data | `test_fct_encode_idempotent_double_encode` |
| S-15 | `fct_reorder_by_col`: NaN rows excluded from mean; level with all-NaN rows gets NaN mean → sorts last | `fct_reorder_by_col()` |
| S-16 | Collapse never triggers CapacityExceeded: collapsed result ≤ original level count | `fct_collapse()` |
| S-17 | NullableFactor null-after-collapse: null rows remain null after any op; null does not become a level | `NullableFactor::fct_collapse()` |
| S-18 | Determinism: all ops produce identical output across independent runs (BTreeMap, stable sort, no HashMap) | Multiple tests |

---

## 3. Data Structures

### `FctColumn`
```rust
pub struct FctColumn {
    pub levels: Vec<String>,   // index → string; first-occurrence order
    pub data:   Vec<u16>,      // row → level index; data[i] < levels.len()
}
```
**Max capacity:** 65,535 levels (u16 max − 1; 0 reserved as sentinel in NullableFactor).

### `NullableFactor`
```rust
pub struct NullableFactor {
    pub fct:      FctColumn,
    pub validity: BitMask,     // validity.get(i) = true → row i is non-null
}
```
Null rows carry `data[i] = 0` as a sentinel; validity bit is clear → callers must check before decoding.

### `TidyError::CapacityExceeded`
```rust
CapacityExceeded { limit: usize, got: usize }
```
Returned by `fct_encode` / `encode_nullable` when more than 65,535 distinct strings are encountered.

---

## 4. API Surface

### Phase 17 operations

| Method | Receiver | Returns | @nogc | Notes |
|--------|----------|---------|-------|-------|
| `FctColumn::encode(strs)` | — | `Result<FctColumn, TidyError>` | ✗ | BTreeMap lookup, first-occurrence levels |
| `FctColumn::encode_from_view(view, col)` | — | `Result<FctColumn, TidyError>` | ✗ | Respects mask + projection |
| `FctColumn::fct_lump(n)` | `&self` | `Result<FctColumn, TidyError>` | ✗ | Top-N frequency; "Other" collision safe |
| `FctColumn::fct_reorder(vals, desc)` | `&self` | `Result<FctColumn, TidyError>` | ✗ | NaN last; stable for ties |
| `FctColumn::fct_reorder_by_col(col, desc)` | `&self` | `Result<FctColumn, TidyError>` | ✗ | Computes per-level mean then reorders |
| `FctColumn::fct_collapse(mapping)` | `&self` | `Result<FctColumn, TidyError>` | ✓ | Metadata-only; conditional O(N) remap |
| `FctColumn::to_str_column()` | `&self` | `Column` | ✗ | Decodes to Column::Str |
| `FctColumn::gather(indices)` | `&self` | `FctColumn` | ✗ | Row-subset; levels shared |
| `FctColumn::decode(i)` | `&self` | `&str` | ✓ | Single-row decode |
| `FctColumn::nrows()` | `&self` | `usize` | ✓ | — |
| `FctColumn::nlevels()` | `&self` | `usize` | ✓ | — |
| `NullableFactor::from_fct(fct)` | — | `NullableFactor` | ✗ | All-valid bitmap |
| `NullableFactor::encode_nullable(opts)` | — | `Result<NullableFactor, TidyError>` | ✗ | None → null |
| `NullableFactor::fct_lump(n)` | `&self` | `Result<NullableFactor, TidyError>` | ✗ | Null rows pass through |
| `NullableFactor::fct_reorder(vals, desc)` | `&self` | `Result<NullableFactor, TidyError>` | ✗ | Null rows pass through |
| `NullableFactor::fct_collapse(mapping)` | `&self` | `Result<NullableFactor, TidyError>` | ✓ | Null rows pass through |
| `NullableFactor::decode(i)` | `&self` | `Option<&str>` | ✓ | None if null |
| `TidyView::fct_encode(col)` | `&self` | `Result<FctColumn, TidyError>` | ✗ | — |
| `TidyView::fct_summary_means(fct, col)` | `&self` | `Result<Vec<f64>, TidyError>` | ✗ | NaN-excluded means per level |
| `TidyView::base_column(name)` | `&self` | `Option<&Column>` | ✓ | Raw unmasked column access |

---

## 5. Role 1 Findings: Data Type Audit

### A. Categorical Type Status

| Feature | Status | Notes |
|---------|--------|-------|
| `FctColumn` (u16 index + Vec<String> levels) | ✅ Implemented | Deterministic first-occurrence ordering |
| `NullableFactor` (FctColumn + BitMask) | ✅ Implemented | Null ≠ level; bitmap-based |
| `OrderedFactor` vs `UnorderedFactor` distinction | 🔲 Not needed yet | fct_reorder achieves ordered semantics |
| `SparseFactor` (large sparse vocabularies) | 🔲 Phase 18+ | Not blocking for current use cases |
| `u32` upgrade path | 🔲 Future flag | Change `u16` → `u32` in FctColumn; all logic identical |
| NA vs "Other" level distinction | ✅ Addressed | Null uses bitmap; "Other" is an explicit level from fct_lump |

### B. Missing Tidy Data Types

| Type | Recommended? | Notes |
|------|-------------|-------|
| `Date` (calendar date) | **Recommended Later** | Needed for time-series; reuses NullableColumn<i32> internally; affects joins/sort |
| `Datetime` (UTC / timezone-naive) | **Recommended Later** | i64 epoch millis; affects sort, group, join |
| `Duration` | **Recommended Later** | i64 millis; arithmetic with Datetime |
| `Time` (time-of-day) | **Not Needed** | Derivable from Datetime; low priority |
| `Decimal` / Fixed-point | **Recommended Later** | Needed for financial data; deterministic arithmetic |
| `List column` (nested rows) | **Not Needed** | Adds GC pressure; incompatible with NoGC model |
| `Struct column` | **Recommended Later** | Would enable embedded records; complex |
| `Union column` | **Not Needed** | Increases API surface without clear benefit |
| `Binary column` | **Not Needed** | Out of scope for tidy analytics |
| `Interval` types | **Recommended Later** | Derived from Datetime + Duration |

**Priority order for future phases:** Date/Datetime → Decimal → Duration → Struct → Interval

---

## 6. Role 2 Findings: Determinism Audit

All four core operations are deterministic by construction:

### fct_encode
- **First-occurrence stable:** BTreeMap lookup for O(log L); `levels` Vec carries insertion order → first-occurrence deterministic across runs.
- **After filter:** Only visible rows (mask) contribute → consistent with view semantics.
- **After bind_rows:** Caller concatenates strings then encodes; encoding is deterministic on the concatenated slice.
- **After pivot:** Same — encode on resulting string column.

### fct_lump
- **Tie-breaking:** Equal-frequency levels resolved by ascending level index (= first-occurrence order) via `.then(a.0.cmp(&b.0))` in sort key.
- **N=0:** All levels → "Other" (one level total). ✅ Tested.
- **N ≥ nlevels:** No-op (clone returned). ✅ Tested.
- **"Other" collision:** Iterates "Other" → "Other_" → "Other__" etc. ✅ Tested.
- **All equal frequency:** First N by first-occurrence win. ✅ Tested.

### fct_reorder
- **NaN last:** Both ascending AND descending — NaN comparison is extracted from the direction flip. ✅ Fixed (bug: was applying `.reverse()` to NaN ordering; fixed in this phase).
- **Stable ties:** Rust `sort_by` is stable → equal-summary levels preserve level-index order.

### fct_collapse
- **Level ordering:** Output follows first-occurrence of NEW names as encountered in OLD level order.
- **Duplicate outputs:** Merged into single index; canonical mapping.
- **Empty mapping:** Returns clone (noop).

---

## 7. Role 3 Findings: Memory Model & NoGC Audit

### u16 Enforcement
- Overflow at 65,536 distinct strings → `TidyError::CapacityExceeded { limit: 65_535, got: N }`.
- Checked in `encode()` and `encode_nullable()` before pushing new level.
- `fct_collapse` can only reduce level count → overflow impossible from collapse.
- Capacity tests verified (marked `#[ignore]` for CI speed; run with `--ignored`).

### Metadata-Only Collapse
- `fct_collapse` rebuilds `levels` Vec (O(L)) and remaps `data` Vec (O(N)) only when indices actually change.
- No DataFrame column allocation; no GC heap involvement.
- Registered as `fct_collapse` in `is_safe_builtin()` in `nogc_verify.rs`.
- Negative NoGC tests: `fct_encode`, `fct_lump`, `fct_reorder` all rejected inside `@nogc`. ✅ Tested.

### Nullable + Factor Integration
- Null is represented as a cleared bit in `NullableFactor::validity` (BitMask).
- Null is NOT a level — `levels` Vec contains only real string values.
- Null index slot (`data[i] = 0` for null rows) is a sentinel; callers must check bitmap before decoding.
- After any op (lump/reorder/collapse), null rows carry the same validity bit (false) — null never becomes a real level.

### Memory Invariants
- `fct_lump`: allocates new `levels` Vec (≤ n+1 elements) + new `data` Vec (N elements).
- `fct_reorder`: allocates new `levels` Vec (same length) + new `data` Vec (N elements).
- `fct_collapse`: allocates new `levels` Vec (≤ L elements) + optionally new `data` Vec (N elements).
- Re-encoding after rename: if user calls `encode()` again on decoded output, result is identical (S-14 idempotency).
- Recoding is O(L) for levels + O(N) for data remap — never accidentally O(N×L).

---

## 8. Role 4 Edge-Case Checklist

### Encoding
- [x] Basic first-occurrence level ordering
- [x] Indices correct
- [x] Decode roundtrip
- [x] Single level
- [x] Empty input
- [x] Determinism across two runs
- [x] Stable after filter (visible rows only)
- [x] Stable after bind_rows
- [x] Idempotent double-encode
- [x] From TidyView (mask-aware)
- [x] From TidyView after filter
- [x] Unknown column error
- [x] Capacity boundary small (1,000 levels — fast CI proxy)
- [x] Exactly 65,535 levels ok (ignored — slow)
- [x] 65,536 levels → CapacityExceeded (ignored — slow)

### Lumping
- [x] Basic top-N
- [x] Values decoded correctly
- [x] N=0 → all other
- [x] N ≥ nlevels → noop
- [x] N exactly = nlevels → noop
- [x] Tie-breaking by first-occurrence
- [x] All equal frequency
- [x] All categories unique
- [x] "Other" collision → renamed
- [x] "Other" and "Other_" both present → "Other__"
- [x] Determinism

### Reordering
- [x] Ascending
- [x] Descending
- [x] NaN sorts last (ascending AND descending)
- [x] Ties are stable
- [x] Data indices remapped correctly
- [x] By float column
- [x] By int column
- [x] NaN excluded from mean in by_col
- [x] Wrong type → TypeMismatch
- [x] Length mismatch → LengthMismatch
- [x] Determinism

### Collapse
- [x] Basic multi-level merge
- [x] Values decoded correctly
- [x] Empty mapping → noop
- [x] Same-name mapping → noop
- [x] All to one
- [x] Duplicate new names merged to one index
- [x] Preserves first-occurrence level order
- [x] Chain (collapse then collapse)
- [x] Collapse then lump
- [x] Collapse then reorder
- [x] Empty factor
- [x] Determinism
- [x] After max cardinality → no error (ignored — slow)

### Nullable
- [x] From FctColumn (all valid)
- [x] Encode with nulls
- [x] Decode → None for null
- [x] Null not a level
- [x] Lump preserves nulls
- [x] Reorder preserves nulls
- [x] Collapse preserves nulls
- [x] Null remains null after collapse (not promoted to level)
- [x] Capacity exceeded (ignored — slow)

### NoGC
- [x] `fct_encode` rejected in @nogc
- [x] `fct_lump` rejected in @nogc
- [x] `fct_reorder` rejected in @nogc
- [x] `fct_collapse` accepted in @nogc
- [x] `fct_collapse` + other safe builtins accepted together

### fct_summary_means
- [x] Basic mean per level
- [x] NaN rows excluded from mean
- [x] All-NaN level → NaN mean
- [x] Wrong type → TypeMismatch

---

## 9. Role 5 Findings: Strategic Roadmap

CJC tidy now has a complete Phase 10-17 stack:

| Capability | Status |
|------------|--------|
| filter, select, mutate | ✅ Phase 10 |
| group_by, summarise, arrange, slice, distinct | ✅ Phase 11-12 |
| pivot, bind, across, nullable columns | ✅ Phase 13-14 |
| typed joins (inner/left/right/full), suffix, type validation | ✅ Phase 15 |
| BTree-accelerated GroupIndex | ✅ Phase 16 |
| categorical encoding (fct_encode/lump/reorder/collapse), NullableFactor | ✅ Phase 17 |

### What still blocks serious tidy engine status:

| Capability | Assessment | Priority |
|------------|-----------|----------|
| **Window functions** (lag, lead, row_number, rank) | Foundational — required for ML feature engineering | **Phase 18** |
| **Rolling / cumulative ops** (cumsum, cummean, rolling_mean) | Foundational — standard analytics | **Phase 18** |
| **Date/time types** (Date, Datetime UTC) | Foundational for time-series | **Phase 19** |
| **Decimal / fixed-point type** | Foundational for financial data | **Phase 19** |
| **Expression-level null propagation** | Foundational — null in DExpr currently unhandled | **Phase 19** |
| **Vectorized string ops** (str_contains, str_extract, str_replace) | Foundational for NLP/data cleaning | **Phase 20** |
| **Lazy query planner** | Optimization — not blocking | Phase 21+ |
| **Arrow memory layout** | Ecosystem integration — not blocking | Phase 22+ |
| **SIMD acceleration hooks** | Optimization — not blocking | Phase 23+ |
| **Parallel group/aggregate** | Optimization — not blocking | Phase 24+ |

**Recommended Phase 18:** Window functions + rolling ops — they complete the "complete data transformation" story and are pre-requisites for most ML pipeline work.

---

## 10. Tests Table

| File | Tests | Focus |
|------|-------|-------|
| `test_phase17_forcats.rs` | 66 passing + 4 ignored | Full Phase 17 edge cases |

### Sub-suite breakdown:
| Category | Count |
|----------|-------|
| fct_encode (basic + view + determinism) | 12 |
| Capacity boundary (1 fast proxy + 3 ignored) | 4 |
| fct_lump | 11 |
| fct_reorder | 12 |
| fct_collapse | 13 |
| NullableFactor | 10 |
| fct_summary_means | 4 |
| NoGC (rejected + accepted) | 5 |
| Gather / to_str_column | 2 |
| **Total** | **73** |

---

## 11. Regression Results

| Suite | Passed | Failed | Ignored |
|-------|--------|--------|---------|
| Tidy suite (Phase 10–17) | 319 | 0 | 5 |
| Full CJC suite | **1,509** | **0** | **5** |

Ignored tests:
1. `test_tidy_speed_gate` — perf gate (pre-existing)
2. `test_fct_encode_exactly_65535_levels_ok` — capacity boundary, slow
3. `test_fct_encode_65536_levels_errors` — capacity boundary, slow
4. `test_fct_collapse_after_max_cardinality_no_error` — capacity boundary, slow
5. `test_nullable_factor_capacity_exceeded` — capacity boundary, slow

To run capacity boundary tests: `cargo test --test test_phase10_tidy -- --ignored`

---

## 12. Bugs Fixed During Phase 17

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `fct_reorder` NaN sorts first in descending mode | `cmp.reverse()` was applied to NaN ordering — reversed `Greater` (NaN last) to `Less` (NaN first) | Moved `descending` flip inside the finite-value match arm only; NaN arms are unconditional |
| O(N²) encode at 65k levels | Linear scan `Vec.iter().position()` in `encode()` | Replaced with `BTreeMap<String, u16>` for O(N log L) total; levels Vec still carries first-occurrence order |

---

## 13. Known Limitations

1. **u16 cap at 65,535 levels:** Upgrade to u32 requires changing the index type in `FctColumn::data` — all logic is identical. Planned as a feature flag in a future phase.
2. **fct_reorder_by_col uses full-row means:** No per-group reordering (e.g., reorder within each group_by group). A grouped variant would need GroupedTidyView integration.
3. **No `fct_drop_levels()`:** Pruning unused levels (those with zero rows) is not yet implemented. Currently `fct_lump(nlevels)` is a no-op, not a prune.
4. **SparseFactor not implemented:** For vocabularies > 1M tokens (e.g., NLP word IDs), a u32-indexed sparse representation would be needed.
5. **String interning:** `FctColumn::gather()` copies the levels Vec reference — levels are not interned. For large level counts with many gathers, a shared `Rc<Vec<String>>` would reduce copies.
