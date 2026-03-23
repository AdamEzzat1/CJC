# CJC TidyView & Data Science Unified Improvement Plan

**Date:** 2025-03-22
**Sources:** TIDYVIEW_IMPROVEMENT_PLAN.md + Performance Analysis + Assessment & Fusion Architecture
**Status:** PLAN ONLY — No implementation yet

---

## Verification Baseline: What Already Exists

Before planning, we verified the codebase. **Many items from the original plan already exist.**

### Already Implemented (DO NOT RE-IMPLEMENT)

| Category | What Exists | Location |
|----------|-------------|----------|
| **Descriptive Stats** | variance, sd, median, quantile, iqr, mad, mode, skewness, kurtosis, z_score, standardize | `cjc-runtime/src/stats.rs` |
| **Correlation** | cor, cov, sample_cov, cor_matrix, cov_matrix, spearman_cor, kendall_cor, partial_cor, cor_ci | `cjc-runtime/src/stats.rs` |
| **Cumulative/Window** | cumsum, cumprod, cummax, cummin, lag, lead, window_sum/mean/min/max | `cjc-runtime/src/stats.rs` |
| **Ranking** | rank, dense_rank, row_number, percent_rank, percentile_rank, cume_dist, ntile | `cjc-runtime/src/stats.rs` |
| **Distributions** | Normal, t, chi-sq, F, Beta, Gamma, Exponential, Weibull, Binomial, Poisson (PDFs, CDFs, PPFs, sampling) | `cjc-runtime/src/distributions.rs` |
| **Hypothesis Tests** | t_test (1-sample, 2-sample, paired), chi_squared, f_test, anova_oneway, mann_whitney, kruskal_wallis, wilcoxon, tukey_hsd, bonferroni, fdr_bh, lm, logistic_regression | `cjc-runtime/src/hypothesis.rs` |
| **Linear Algebra** | LU, QR, Cholesky, SVD, svd_truncated, eigh, Schur, solve, lstsq, inverse, pinv, det, trace, matrix_rank, cond, norms, kron, matrix_exp | `cjc-runtime/src/linalg.rs` |
| **ML Core** | mse_loss, cross_entropy_loss, binary_cross_entropy, huber_loss, hinge_loss, sgd_step, adam_step, l1/l2_penalty, confusion_matrix, precision, recall, f1, accuracy, auc_roc, pca, kfold_indices, train_test_split, batch_norm, dropout, lstm_cell, gru_cell, multi_head_attention, relu, leaky_relu, sigmoid, tanh, mish, silu, softmax | `cjc-runtime/src/ml.rs` |
| **Optimizers** | bisect, brentq, newton_scalar, secant, minimize_gd, minimize_bfgs, minimize_lbfgs, minimize_nelder_mead, penalty_objective, project_box, projected_gd_step, wolfe_line_search, lbfgs_step | `cjc-runtime/src/optimize.rs` |
| **Clustering** | kmeans, dbscan, agglomerative | `cjc-runtime/src/clustering.rs` |
| **Time Series** | acf, pacf, ewma, ema, arima_diff, diff, seasonal_decompose, ar_fit, ar_forecast | `cjc-runtime/src/timeseries.rs` |
| **Stationarity** | adf_test, kpss_test, pp_test | `cjc-runtime/src/stationarity.rs` |
| **Signal/FFT** | fft, ifft, rfft, psd, fft_2d, hann/hamming/blackman windows | `cjc-runtime/src/fft.rs` |
| **Sparse LA** | lanczos_eigsh, arnoldi_eigs, cg_solve, gmres_solve, bicgstab_solve | `cjc-runtime/src/sparse*.rs` |
| **ODE/PDE** | ode_step_euler, ode_step_rk4, ode_solve_rk45, pde_laplacian_1d, pde_step_diffusion, adjoint_solve | `cjc-runtime/src/ode.rs` |
| **File I/O** | file_read, file_write, file_exists, file_lines, getenv | `cjc-runtime/src/builtins.rs` |
| **Encoding** | label_encode, one_hot | `cjc-runtime/src/builtins.rs` |
| **Vectorized** | case_when, argsort, argmin, argmax, topk, einsum | `cjc-runtime/src/builtins.rs` |
| **Tensor Ops** | flatten, squeeze, unsqueeze, chunk, gather, scatter, index_select, where, masked_fill, nonzero, sort, argsort_axis, stack, broadcast | `cjc-runtime/src/builtins.rs` |
| **Vizor** | 19 geom types, faceting, color scales, legends, annotations, BMP/SVG export | `cjc-vizor/` |
| **Snap** | Binary serialization protocol (all Value types, Tensors, DataFrames) | `cjc-snap/` |
| **Interpolation** | interp1d_linear, interp1d_nearest, polyfit, polyval, spline_cubic_natural, spline_eval | builtins |

### Actually Missing (What This Plan Addresses)

| Gap | Impact | Why It Matters |
|-----|--------|---------------|
| `read_csv(path)` / `write_csv(df, path)` as CJC builtins | CRITICAL | CSV parsing exists as Rust API but not exposed as language functions |
| `snap_save(value, path)` / `snap_load(path)` builtins | HIGH | Snap protocol exists but no language-level save/load |
| `read_json(path)` / `write_json(value, path)` builtins | HIGH | JSON serialization exists, no file I/O wrapper |
| TidyAgg expansion (Median, Sd, Var, Quantile, NDistinct) | HIGH | Only Count/Sum/Mean/Min/Max/First/Last in summarise |
| DExpr function calls (`col("x") \|> log()`) | HIGH | Only binary ops in DExpr, no function calls |
| Partitioned window functions (per-group) | MEDIUM | window_sum etc. exist for flat arrays, not within grouped TidyView |
| View Fusion / Lazy Execution | STRATEGIC | TidyView is eager; views copy data on every step |
| Zero-copy projections | STRATEGIC | select() currently copies columns |
| Filtered index views | STRATEGIC | filter() currently copies matching rows |
| DataFrame → Tensor direct export | MEDIUM | to_tensor exists but could be more ergonomic |
| Vizor statistical plots (QQ, pairs, corrplot) | LOW | Polish items |
| Interactive HTML export | LOW | SVG exists, JS interactivity doesn't |
| dir_list, path_join | LOW | file_exists exists but no directory listing |

---

## Stack Role Group

You are a stacked data engineering team working inside the CJC compiler repository.

You consist of:

1. **Data Pipeline Architect** — owns DataFrame/TidyView architecture, column types, lazy evaluation, query optimization, and the view fusion IR
2. **Statistical Computing Engineer** — owns TidyAgg expansion, DExpr function calls, partitioned window functions, and summarise completeness
3. **I/O & Serialization Engineer** — owns read_csv/write_csv builtins, snap_save/snap_load, JSON I/O, and directory utilities
4. **Query Execution Engineer** — owns zero-copy projections, filtered index views, fused execution passes, materialization boundaries, and VM-friendly op lowering
5. **Visualization Engineer** — owns Vizor statistical plots, interactive HTML export, and annotation layer
6. **Determinism & Performance Auditor** — enforces bit-identical output, Kahan/Binned accumulation, BTreeMap ordering, deterministic segmented aggregation, and stable dictionary encoding

Your goal is to close the remaining gaps in CJC's data infrastructure while laying the architectural foundation for view fusion — making TidyView evolve from an eager copy-based system into a deterministic logical view system over columnar buffers.

---

## PRIME DIRECTIVES

1. **Never break determinism** — same seed = bit-identical output
2. **All new builtins wired in THREE places** — `cjc-runtime/builtins.rs`, `cjc-eval/lib.rs`, `cjc-mir-exec/lib.rs`
3. **All floating-point reductions use Kahan or BinnedAccumulator**
4. **BTreeMap/BTreeSet everywhere** — never HashMap/HashSet
5. **Parity tests for every new feature** — eval == mir-exec
6. **COW semantics** — data operations return new values, never mutate in place
7. **Do not re-implement what already exists** — verify before coding
8. **Preserve future VM lowering path** — all new operations must map to compact instruction forms

---

## PHASE 1: Data I/O Builtins (CRITICAL — Blocks Real Data Work)

**Priority: CRITICAL**
**Rationale:** CSV parsing, Snap serialization, and JSON encoding all exist as Rust APIs. They just need thin wrappers exposed as CJC builtins.

### 1.1 `read_csv(path)` / `write_csv(df, path)`
- Wrap existing `CsvReader` from `cjc-data/src/csv.rs` as a language builtin
- `read_csv(path)` → constructs DataFrame from file path
- `write_csv(df, path)` → serializes DataFrame columns to CSV file
- Must use BTreeMap column ordering for deterministic output
- Optional params: `delimiter`, `has_header`

### 1.2 `snap_save(value, path)` / `snap_load(path)`
- Wrap existing `cjc-snap` encode/decode as language builtins
- `snap_save(value, path)` → encode any CJC value to binary file
- `snap_load(path)` → decode binary file back to CJC value
- Uses existing CJS\x01 v2 format — no protocol changes needed

### 1.3 `read_json(path)` / `write_json(value, path)`
- Wrap existing JSON serialization (if present) or add minimal parser
- Flat JSON arrays → auto-convert to DataFrame
- Nested objects → struct conversion

### 1.4 `dir_list(path)` / `path_join(parts)`
- `dir_list(path)` → sorted array of filenames (BTreeSet for deterministic order)
- `path_join(a, b)` → platform-aware path concatenation

**Files to modify:**
- `crates/cjc-runtime/src/builtins.rs` — new dispatch arms
- `crates/cjc-eval/src/lib.rs` — wire builtins
- `crates/cjc-mir-exec/src/lib.rs` — wire builtins

**Tests:** `tests/tidyview_hardening/test_phase1_data_io.rs`
- read_csv roundtrip (write then read, verify identical)
- snap_save/snap_load roundtrip for all Value types
- dir_list returns sorted results
- Parity: eval == mir-exec for all new builtins

**Estimated LOC:** ~200 (thin wrappers over existing APIs)

---

## PHASE 2: TidyAgg & DExpr Completeness (HIGH — Enables Real Summarise Workflows)

**Priority: HIGH**
**Rationale:** TidyView's summarise() currently only supports Count/Sum/Mean/Min/Max/First/Last. Real data science requires median, sd, var, quantile, n_distinct in grouped aggregations.

### 2.1 Expand TidyAgg Enum
Add new variants to `TidyAgg` in `cjc-data/`:
```
TidyAgg::Median(col)
TidyAgg::Sd(col)
TidyAgg::Var(col)
TidyAgg::Quantile(col, p)
TidyAgg::NDistinct(col)
TidyAgg::Iqr(col)
```

Each must use Kahan summation where applicable.

### 2.2 TidyAgg Builders
Add builder functions matching existing pattern:
- `tidy_median(col)`, `tidy_sd(col)`, `tidy_var(col)`, `tidy_quantile(col, p)`, `tidy_n_distinct(col)`, `tidy_iqr(col)`

### 2.3 DExpr Function Call Support
Add `DExpr::FnCall(name, args)` variant to allow expressions like:
```
mutate("log_x", fn_call("log", [col("x")]))
mutate("abs_diff", fn_call("abs", [col("a") - col("b")]))
```

Functions permitted inside DExpr: `log`, `exp`, `sqrt`, `abs`, `ceil`, `floor`, `round`, `sin`, `cos`, `tan`

### 2.4 Partitioned Window Functions
Allow cumulative/window functions within grouped TidyView:
```
grouped_view.mutate("running_total", DExpr::CumSum(col("sales")))
grouped_view.mutate("prev_value", DExpr::Lag(col("price"), 1))
```

New DExpr variants:
- `DExpr::CumSum(col)`, `DExpr::CumProd(col)`, `DExpr::CumMax(col)`, `DExpr::CumMin(col)`
- `DExpr::Lag(col, k)`, `DExpr::Lead(col, k)`
- `DExpr::RollingMean(col, window)`, `DExpr::RollingSd(col, window)`
- `DExpr::Rank(col)`, `DExpr::DenseRank(col)`, `DExpr::RowNumber()`

**Files to modify:**
- `crates/cjc-data/src/lib.rs` or `tidy.rs` — TidyAgg enum expansion
- `crates/cjc-data/src/tidy_dispatch.rs` — dispatch new agg/expr variants
- `crates/cjc-runtime/src/builtins.rs` — wire builders

**Tests:** `tests/tidyview_hardening/test_phase2_agg_expr.rs`
- summarise with median, sd, var per group
- DExpr::FnCall with log, abs
- Partitioned cumsum within groups
- Determinism: 3 runs identical
- Parity: eval == mir-exec

**Estimated LOC:** ~400

---

## PHASE 3: View Fusion Foundation (STRATEGIC — Architecture for Performance)

**Priority: STRATEGIC**
**Rationale:** This is the single highest-leverage change for TidyView performance. Current TidyView copies data at every step. View fusion makes it a logical plan system that only materializes when necessary.

### 3.1 Logical View IR

Define `ViewNode` enum (internal to `cjc-data`):

```rust
enum ViewNode {
    Scan { frame: DataFrame },
    Project { input: Box<ViewNode>, columns: Vec<String> },
    Filter { input: Box<ViewNode>, predicate: DExpr },
    Derive { input: Box<ViewNode>, name: String, expr: DExpr },
    StableSort { input: Box<ViewNode>, keys: Vec<ArrangeKey> },
    Group { input: Box<ViewNode>, keys: Vec<String> },
    Aggregate { input: Box<ViewNode>, aggs: Vec<(String, TidyAgg)> },
    Join { left: Box<ViewNode>, right: Box<ViewNode>, kind: JoinKind, left_on: String, right_on: String },
    Window { input: Box<ViewNode>, expr: DExpr, partition_by: Vec<String> },
    Slice { input: Box<ViewNode>, start: usize, end: usize },
    Materialize { input: Box<ViewNode> },
}
```

Each node carries:
- Schema in / schema out (column names + types)
- Determinism requirements (all nodes must be deterministic)
- Fusibility flag (can this fuse with its input?)
- Barrier flag (forces materialization before this step?)

### 3.2 View Fusion Rules

Safe fusions:
- `Scan + Filter` → single filtered scan
- `Scan + Project` → zero-copy column subset
- `Filter + Project` → fused filter-then-project
- `Project + Derive` → derive only computes projected columns
- `Derive + Derive` → single pass computing all derived columns
- `Filter + Filter` → conjunction of predicates
- `Redundant Project elimination` → skip project that doesn't change columns

Fusion blockers (force materialization):
- `StableSort` — must materialize to sort
- `Join` — requires materialized index/state
- `Window` with cross-row dependencies
- Explicit `.collect()` / `.materialize()`
- Export to Vizor / tensor

### 3.3 Zero-Copy Projections

`select()` on a TidyView should NOT copy column data. Instead:
- Store a `Vec<usize>` column index mapping
- When materializing, only touch selected columns
- This alone eliminates most copy overhead in select-heavy pipelines

### 3.4 Filtered Index Views

`filter()` on a TidyView should NOT copy matching rows. Instead:
- Build a `Vec<u32>` row index (sorted, for cache locality)
- Carry the index through subsequent operations
- Materialize only when a barrier is reached

### 3.5 Materialization Boundaries

TidyView materializes ONLY at:
- `.collect()` — explicit user request
- `.arrange()` — stable sort requires full data
- `.inner_join()` etc. — joins require realized state
- `.to_tensor()` — export to tensor
- Vizor plot data extraction
- `snap_save()` — serialization
- `print()` — display

All other operations remain lazy views.

### 3.6 VM Bytecode Mapping (Future)

Document how ViewNodes lower to future CJC VM instructions:
```
SCAN_COL_F64   reg, frame_id, col_idx
FILTER_MASK    reg_out, reg_pred
TAKE_ROWS      reg_out, reg_frame, reg_mask
DERIVE_COL     reg_out, expr_id
STABLE_GROUP   reg_out, reg_frame, key_cols
AGG_SUM_F64    reg_out, reg_group, col_idx
AGG_MEAN_F64   reg_out, reg_group, col_idx
AGG_MEDIAN     reg_out, reg_group, col_idx
STABLE_JOIN    reg_out, reg_left, reg_right, kind
MATERIALIZE    reg_out, reg_view
```

**Files to create:**
- `crates/cjc-data/src/view_ir.rs` — ViewNode enum, schema tracking
- `crates/cjc-data/src/view_fusion.rs` — fusion optimizer
- `crates/cjc-data/src/view_exec.rs` — materialization executor

**Files to modify:**
- `crates/cjc-data/src/tidy_dispatch.rs` — build ViewNode tree instead of eager execution
- `crates/cjc-data/src/lib.rs` — TidyView struct updated to hold ViewNode

**Tests:** `tests/tidyview_hardening/test_phase3_view_fusion.rs`
- Verify lazy: filter+select doesn't allocate until collect
- Verify fusion: filter+filter → single predicate
- Verify materialization: sort forces materialization
- Determinism: fused plan == eager plan (bit-identical output)
- Parity: eval == mir-exec
- Memory test: fused pipeline uses less memory than eager

**Estimated LOC:** ~800

---

## PHASE 4: Deterministic Aggregation Kernels (STRATEGIC — Performance + Correctness)

**Priority: STRATEGIC**
**Rationale:** Group-by aggregation is the core operation for data science. Making it fast AND deterministic is where CJC can differentiate from hash-based systems.

### 4.1 Sort-Based Grouping (Primary Strategy)

Replace or augment current grouping with sort-then-segment:
1. Stable sort on group keys (already have stable sort)
2. Segment scan: walk sorted data, emit segment boundaries
3. Run aggregation kernels on each segment

Benefits:
- Deterministic group order (lexicographic)
- Excellent cache locality (sequential memory access)
- Thread-count invariant (each segment is independent)
- No hash collisions or load-factor issues

### 4.2 Specialized Aggregate Kernels

Tight, type-specialized kernels for each aggregation:
```rust
fn agg_sum_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64>  // Kahan
fn agg_mean_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> // Kahan
fn agg_count(segments: &[(usize, usize)]) -> Vec<i64>
fn agg_min_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64>
fn agg_max_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64>
fn agg_var_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64>  // Welford
fn agg_median_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64>
fn agg_quantile_f64(data: &[f64], p: f64, segments: &[(usize, usize)]) -> Vec<f64>
fn agg_n_distinct(data: &[String], segments: &[(usize, usize)]) -> Vec<i64> // BTreeSet
```

All reductions use Kahan summation. No FMA. No platform-dependent behavior.

### 4.3 Stable Dictionary Encoding

For string/categorical columns:
1. Build deterministic dictionary (BTreeMap<String, u32>)
2. Replace string data with dictionary indices
3. Grouping/joining operates on u32 indices (much faster)
4. Dictionary is stable across runs (BTree ordering)

This accelerates:
- `group_by()` on string columns
- `inner_join()` on string keys
- `filter()` with string equality
- `distinct()` on string columns

### 4.4 Deterministic Parallel Reductions (Future)

When parallel execution is added:
1. Partition rows into fixed-size chunks (deterministic partitioning)
2. Each chunk produces partial aggregates (using same accumulator semantics)
3. Merge partials in fixed left-to-right order (NOT reduce-tree)
4. Emit groups in stable key order

This ensures:
- Same result regardless of thread count
- Same result regardless of chunk size
- Bit-identical with single-threaded execution

**Files to create:**
- `crates/cjc-data/src/agg_kernels.rs` — specialized aggregate functions
- `crates/cjc-data/src/dict_encoding.rs` — stable dictionary encoding

**Files to modify:**
- `crates/cjc-data/src/tidy_dispatch.rs` — use new kernels in summarise

**Tests:** `tests/tidyview_hardening/test_phase4_deterministic_agg.rs`
- Sort-based grouping produces same result as current grouping
- Specialized kernels match generic implementations (bit-identical)
- Dictionary encoding roundtrip preserves data
- Parallel reduction (simulated) matches single-threaded (bit-identical)
- Kahan audit: all reductions use Kahan
- BTreeMap audit: no HashMap in new code
- Stress test: 100K rows, 1000 groups, deterministic

**Estimated LOC:** ~600

---

## PHASE 5: Preprocessing Pipeline Integration (MEDIUM — ML Data Prep)

**Priority: MEDIUM**
**Rationale:** standardize() and one_hot() exist as standalone functions. This phase makes them work as TidyView pipeline steps.

### 5.1 Scaling as TidyView Mutate Operations
```
view.mutate("x_scaled", DExpr::FnCall("standardize", [col("x")]))
view.mutate("y_norm", DExpr::FnCall("min_max_scale", [col("y"), lit(0.0), lit(1.0)]))
view.mutate("z_robust", DExpr::FnCall("robust_scale", [col("z")]))
```

These are enabled by Phase 2's `DExpr::FnCall` — no new infrastructure needed, just whitelist the functions.

### 5.2 Missing Data Handling in TidyView
```
view.mutate("x_filled", DExpr::FnCall("fillna", [col("x"), lit(0.0)]))
view.filter(DExpr::FnCall("is_not_null", [col("y")]))  // dropna equivalent
```

New functions to add:
- `fillna(arr, value)` → replace null/NaN with value
- `is_not_null(x)` → boolean, for use in filter predicates
- `interpolate_linear(arr)` → linearly interpolate missing values
- `coalesce(a, b)` → first non-null

### 5.3 Binning & Discretization
- `cut(arr, breaks)` → categorical from continuous
- `qcut(arr, n)` → quantile-based equal-frequency bins

### 5.4 DataFrame → Tensor Export Enhancement
Improve `to_tensor(cols)` to handle:
- Automatic numeric-only column selection
- Categorical columns → automatic one-hot encoding
- Missing values → configurable fill strategy
- Output shape annotation

**Files to modify:**
- `crates/cjc-runtime/src/builtins.rs` — add fillna, is_not_null, cut, qcut
- `crates/cjc-data/src/tidy_dispatch.rs` — whitelist new functions in DExpr

**Tests:** `tests/tidyview_hardening/test_phase5_preprocessing.rs`
- fillna replaces NaN values
- cut creates correct bin labels
- qcut produces equal-frequency bins
- to_tensor with mixed columns
- Parity: eval == mir-exec

**Estimated LOC:** ~300

---

## PHASE 6: Vizor Statistical Plots (LOW — Polish)

**Priority: LOW**
**Rationale:** Vizor already has 19 geom types. These are polish items for publication quality.

### 6.1 Statistical Geom Types
- `geom_qq()` — QQ plot against normal distribution
- `geom_pairs()` — scatterplot matrix
- `geom_corrplot()` — correlation matrix heatmap with coefficient values

### 6.2 Interactive HTML Export
- `plot.save_html(path)` — self-contained HTML with embedded SVG
- Minimal JS for hover tooltips (data point values)
- No framework dependencies

### 6.3 Annotation Layer
- `annotate_text(x, y, label)` — text annotation
- `annotate_hline(y)` / `annotate_vline(x)` — reference lines
- `annotate_rect(x1, y1, x2, y2)` — highlight region

**Files to modify:**
- `crates/cjc-vizor/src/geom.rs` — new geom types
- `crates/cjc-vizor/src/render.rs` — HTML export

**Tests:** `tests/tidyview_hardening/test_phase6_vizor.rs`

**Estimated LOC:** ~400

---

## Implementation Order

```
Phase 1 (Data I/O Builtins)           ← CRITICAL, do first, ~200 LOC
  ↓                                      Unblocks loading real data
Phase 2 (TidyAgg & DExpr)             ← HIGH, ~400 LOC
  ↓                                      Unblocks real summarise workflows
Phase 3 (View Fusion Foundation)       ← STRATEGIC, ~800 LOC
  ↓                                      Architecture for performance
Phase 4 (Deterministic Agg Kernels)    ← STRATEGIC, ~600 LOC
  ↓                                      Performance + correctness
Phase 5 (Preprocessing Pipeline)       ← MEDIUM, ~300 LOC
  ↓                                      ML data prep integration
Phase 6 (Vizor Polish)                 ← LOW, ~400 LOC
                                         Publication quality
```

**Total estimated: ~2,700 LOC** (down from original 3,500 because many items already exist)

---

## Verification Protocol

For each phase:

1. **Compile gate:** `cargo check --workspace` passes
2. **Unit tests:** Each new function has ≥2 tests
3. **Parity tests:** eval == mir-exec for every new builtin
4. **Determinism tests:** 3 runs with same seed → identical output
5. **Regression gate:** `cargo test --workspace` — zero new failures (currently 5,300+ tests)
6. **Kahan audit:** All floating-point reductions use Kahan/BinnedAccumulator
7. **BTreeMap audit:** No HashMap/HashSet in new code
8. **Memory audit:** View fusion actually reduces allocations vs eager

All new tests go in: `tests/tidyview_hardening/`

After all phases, run full regression:
```bash
cargo test --workspace 2>&1 | grep "test result:"
# Expect: 0 failures across all crates and integration tests
```

Document all changes in: `docs/TIDYVIEW_HARDENING_CHANGELOG.md`

---

## Architecture Decision: Why View Fusion Matters

From the performance analysis document:

> The biggest speedups would not come from one thing. They would come from **stacking multiple non-glamorous wins together**.

The key insight is that CJC's data layer should evolve from:
- **Current:** "nice high-level data API" (eager, copies on every step)
- **Target:** "serious execution layer for tabular and preprocessing workloads" (lazy, fused, zero-copy)

The performance ceiling estimates:
- **vs naive eager row-oriented:** 2x–10x faster on common transforms
- **vs well-written eager columnar:** 1.5x–4x on realistic pipelines
- **vs Polars/DuckDB:** competitive in deterministic workloads, wins on reproducibility + predictable memory

The strategic framing:
> TidyView should not primarily be "a dataframe object." It should be **a deterministic logical view system over columnar buffers**.

This aligns with CJC's core philosophy: speed with control, determinism, and architectural coherence.

---

## What This Does NOT Change

This plan does NOT touch:
- The compiler pipeline (Lexer → Parser → AST → HIR → MIR → Exec)
- The type system or type inference
- The memory model or GC/NoGC boundary
- Existing builtins or their behavior
- The determinism contract
- The Bastion kernel architecture (future)
- The VM design (future)

All changes are additive. No existing code is modified except to add new dispatch arms and new enum variants.

---

## Estimated Scope Summary

| Phase | New Items | New Files | Est. LOC | Priority |
|-------|-----------|-----------|----------|----------|
| 1. Data I/O | 8 builtins | 0 (wrappers) | ~200 | CRITICAL |
| 2. TidyAgg/DExpr | 6 aggs + DExpr variants | 0 (extend) | ~400 | HIGH |
| 3. View Fusion | ViewNode IR + fusion | 3 new | ~800 | STRATEGIC |
| 4. Agg Kernels | Specialized kernels + dict encoding | 2 new | ~600 | STRATEGIC |
| 5. Preprocessing | 4 builtins + pipeline integration | 0 (extend) | ~300 | MEDIUM |
| 6. Vizor Polish | 3 geoms + HTML export | 0 (extend) | ~400 | LOW |
| **Total** | **~21 builtins + architecture** | **~5 new** | **~2,700** | — |
