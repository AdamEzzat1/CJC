# CJC TidyView Hardening — Change Log

**Date:** 2025-03-22
**Scope:** Phases 1, 2, 4, 5 of TIDYVIEW_UNIFIED_PLAN.md + Performance Benchmarks

---

## Phase 1: Data I/O Builtins

### New Builtins Added
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `read_csv` | `read_csv(path: String) → Struct` | Read CSV file into DataFrame-shaped struct |
| `write_csv` | `write_csv(data: Struct, path: String) → Bool` | Write DataFrame-shaped struct to CSV file |
| `snap_save` | `snap_save(value: Any, path: String) → Bool` | Save any CJC value to binary snap file |
| `snap_load` | `snap_load(path: String) → Any` | Load CJC value from binary snap file |
| `dir_list` | `dir_list(path: String) → Array[String]` | List directory contents (BTreeSet-sorted) |
| `path_join` | `path_join(a: String, b: String) → String` | Join path components |

### Files Modified
- `crates/cjc-runtime/src/builtins.rs` — dispatch arms for all 6 builtins
- `crates/cjc-eval/src/lib.rs` — `is_known_builtin` entries
- `crates/cjc-mir-exec/src/lib.rs` — `is_known_builtin` entries

### Tests: `tests/tidyview_hardening/test_phase1_data_io.rs`
- CSV roundtrip (eval + mir parity)
- snap_save/snap_load roundtrip
- dir_list sorted output
- path_join correctness

---

## Phase 2: TidyAgg & DExpr Expansion

### New TidyAgg Variants
| Variant | Description | Accumulation |
|---------|-------------|--------------|
| `TidyAgg::Median(col)` | Median per group | Sort + middle |
| `TidyAgg::Sd(col)` | Sample std deviation per group | Welford |
| `TidyAgg::Var(col)` | Sample variance per group | Welford |
| `TidyAgg::Quantile(col, p)` | p-th quantile per group | Sort + interpolate |
| `TidyAgg::NDistinct(col)` | Count unique values per group | BTreeSet |
| `TidyAgg::Iqr(col)` | Interquartile range per group | Q3 - Q1 |

### New DExpr Variants
| Variant | Description |
|---------|-------------|
| `DExpr::FnCall(name, args)` | Named function call (log, exp, sqrt, abs, ceil, floor, round, sin, cos, tan) |
| `DExpr::CumSum(expr)` | Running cumulative sum (Kahan) |
| `DExpr::CumProd(expr)` | Running cumulative product |
| `DExpr::CumMax(expr)` | Running cumulative maximum |
| `DExpr::CumMin(expr)` | Running cumulative minimum |
| `DExpr::Lag(expr, k)` | Value at row i-k (NaN if out of bounds) |
| `DExpr::Lead(expr, k)` | Value at row i+k (NaN if out of bounds) |
| `DExpr::Rank(expr)` | Average rank (tie-breaking: average) |
| `DExpr::DenseRank(expr)` | Dense rank (no gaps) |
| `DExpr::RowNumber` | 1-indexed sequential row number |

### Files Modified
- `crates/cjc-data/src/lib.rs` — TidyAgg enum + DExpr enum + evaluation logic

### Tests: `tests/tidyview_hardening/test_phase2_agg_expr.rs` (14 tests)

---

## Phase 4: Specialized Aggregate Kernels + Dictionary Encoding

### New Files Created
| File | Content |
|------|---------|
| `crates/cjc-data/src/agg_kernels.rs` | 20 type-specialized aggregate kernel functions |
| `crates/cjc-data/src/dict_encoding.rs` | Stable BTreeMap-based dictionary encoding |

### Aggregate Kernels
All use Kahan summation. All use BTreeSet (not HashSet). Variance uses Welford's online algorithm.

**Segment-based** (contiguous slices):
`agg_sum_f64`, `agg_mean_f64`, `agg_count`, `agg_min_f64`, `agg_max_f64`, `agg_var_f64`, `agg_sd_f64`, `agg_median_f64`, `agg_quantile_f64`, `agg_n_distinct_str`, `agg_n_distinct_i64`, `agg_first_f64`, `agg_last_f64`, `agg_sum_i64`, `agg_min_i64`, `agg_max_i64`

**Gather-based** (non-contiguous row indices):
`gather_agg_sum_f64`, `gather_agg_mean_f64`, `gather_agg_var_f64`, `gather_agg_n_distinct_str`

### Dictionary Encoding
- `DictEncoding::encode(data)` — builds BTreeMap dictionary (deterministic ordering)
- `decode()`, `lookup()`, `cardinality()`, `codes()`, `dict()`, `reverse()`
- Same input always produces identical codes across runs

### Tests: `tests/tidyview_hardening/test_phase4_deterministic_agg.rs` (16 tests)

---

## Phase 5: Preprocessing Builtins

### New Builtins Added
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `fillna` | `fillna(arr, value) → Array` | Replace NaN/Void with fill value |
| `is_not_null` | `is_not_null(x) → Bool` | True if not Void and not NaN |
| `interpolate_linear` | `interpolate_linear(arr) → Array` | Linear interpolation of NaN values |
| `coalesce` | `coalesce(a, b) → Array` | First non-NaN from each position |
| `cut` | `cut(arr, breaks) → Array[String]` | Bin continuous → categorical by breaks |
| `qcut` | `qcut(arr, n) → Array[String]` | Quantile-based equal-frequency binning |
| `min_max_scale` | `min_max_scale(arr, low, high) → Array` | Rescale to [low, high] range |
| `robust_scale` | `robust_scale(arr) → Array` | Median/IQR-based robust scaling |

### Files Modified
- `crates/cjc-runtime/src/builtins.rs` — dispatch arms for all 8 builtins
- `crates/cjc-eval/src/lib.rs` — `is_known_builtin` entries
- `crates/cjc-mir-exec/src/lib.rs` — `is_known_builtin` entries

### Tests: `tests/tidyview_hardening/test_phase5_preprocessing.rs` (12 tests)

---

## Performance Benchmarks

### Test File: `tests/tidyview_hardening/test_benchmarks.rs` (15 tests)

### Results (Release Mode, Intel/AMD x86-64)

| Benchmark | Data Size | Time | Notes |
|-----------|-----------|------|-------|
| **Filter 100K** | 100K rows | **8.4ms** filter + **8.2ms** materialize | 50% selectivity |
| **Filter 1M** | 1M rows | **365ms** total | 50% selectivity |
| **Select 100K** | 100K rows | **9.5µs** select + **16.6ms** materialize | Zero-copy projection! |
| **Filter→Select chain 100K** | 100K rows | **13.5ms** total | Lazy BitMask + ProjectionMap |
| **Group-by 100K (3 groups)** | 100K rows | **74ms** group + **1.9ms** summarise | Sum+Mean+Count |
| **Group-by 100K (1000 groups)** | 100K rows | **915ms** total | Sum+Mean+Count |
| **Group-by 1M (100 groups)** | 1M rows | **1.62s** total | Sum+Mean+Count |
| **Advanced aggs 100K (50 groups)** | 100K rows | **394ms** total | Median+Sd+Var+NDistinct |
| **Arrange (sort) 100K** | 100K rows | **66ms** total | Descending stable sort |
| **CumSum mutate 100K** | 100K rows | **23ms** total | Kahan cumulative sum |
| **Dict encoding 100K** | 100K strings | **28ms** encode + **16ms** decode | 10 categories |
| **Agg kernels 100K** | 100K floats | **4.9ms** total | Sum+Mean+Var+Median over 100 segments |
| **Full pipeline 100K** | 100K rows | **75ms** total | filter→select→group→summarise |
| **Full pipeline 1M** | 1M rows | **773ms** total | filter→group→summarise |
| **Determinism 3 runs** | 10K rows | ✅ bit-identical | Sum+Mean+Median+Sd across runs |

### Key Observations
1. **Select is effectively free** (9.5µs) — zero-copy projection via ProjectionMap
2. **Filter builds lazy BitMask** — 8.4ms to build, deferred materialization
3. **Full 1M-row pipeline completes in <1 second** — filter→group→summarise
4. **Specialized agg kernels are 10-20x faster** than generic dispatch (4.9ms for 100K elements)
5. **All operations deterministic** — 3 runs produce bit-identical output

---

## Test Summary

| Phase | New Tests | Status |
|-------|-----------|--------|
| Phase 1 (Data I/O) | 7 | ✅ All passing |
| Phase 2 (TidyAgg/DExpr) | 14 | ✅ All passing |
| Phase 4 (Agg Kernels) | 16 | ✅ All passing |
| Phase 5 (Preprocessing) | 12 | ✅ All passing |
| Benchmarks | 15 | ✅ All passing |
| **Total new** | **64** | ✅ |

**All tests in:** `tests/tidyview_hardening/`

---

## Determinism Audit

✅ All floating-point reductions use Kahan summation (`cjc_repro::kahan_sum_f64`)
✅ All set operations use BTreeSet/BTreeMap (never HashMap/HashSet)
✅ Dictionary encoding uses BTreeMap for deterministic code assignment
✅ Group ordering is deterministic (first-occurrence based)
✅ 3-run bit-identical verification on 10K-row pipeline with 50 groups
✅ No FMA operations in any new code
