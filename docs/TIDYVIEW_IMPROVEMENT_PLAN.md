# CJC TidyView & Data Science Improvement Plan

## Stack Role Group

You are a stacked data engineering team working inside the CJC compiler repository.

You consist of:

1. **Data Pipeline Architect** — owns DataFrame/TidyView architecture, column types, lazy evaluation, and query optimization
2. **Statistical Computing Engineer** — owns descriptive stats, inferential tests, distributions, correlation/covariance, and sampling
3. **I/O & Serialization Engineer** — owns CSV/file reading, data import/export, format conversion, and streaming
4. **Visualization Engineer** — owns Vizor grammar-of-graphics, plot types, themes, interactive output, and export formats
5. **ML Feature Engineer** — owns preprocessing pipelines, encoding, feature extraction, and model-data integration
6. **Determinism Auditor** — enforces bit-identical output, Kahan/Binned accumulation, BTreeMap ordering, and reproducibility

Your goal is to make CJC's data infrastructure competitive with pandas/tidyverse/polars for real data science workflows while preserving the determinism contract.

---

## PRIME DIRECTIVES

1. **Never break determinism** — same seed = bit-identical output
2. **All new builtins wired in THREE places** — `cjc-runtime/builtins.rs`, `cjc-eval/lib.rs`, `cjc-mir-exec/lib.rs`
3. **All floating-point reductions use Kahan or BinnedAccumulator**
4. **BTreeMap/BTreeSet everywhere** — never HashMap/HashSet
5. **Parity tests for every new feature** — eval == mir-exec
6. **COW semantics** — data operations return new values, never mutate in place

---

## CURRENT STATE AUDIT

### What CJC TidyView Already Has (Strong)
- 20+ tidy verbs: filter, select, mutate, arrange, distinct, slice, slice_head/tail/sample
- 7 join types: inner, left, right, full, semi, anti (+ key type validation)
- Grouping: group_by, group_by_fast (BTree-accelerated), summarise, summarise_across
- Reshaping: pivot_longer, pivot_wider
- Binding: bind_rows, bind_cols
- Column ops: rename, relocate, drop_cols, mutate_across
- String functions: 12 stringr-style builtins (detect, extract, replace, split, trim, case, etc.)
- Regex: Thompson NFA engine (zero-dependency, NoGC-safe)
- Basic stats: median, sd, variance, n_distinct
- Aggregation: sum, mean, min, max, first, last, count
- Nullable columns: bitmap-based validity model
- Categorical: factor levels, fct_lump, fct_reorder, fct_collapse
- CSV I/O: parse from string, streaming aggregation
- Visualization: 25+ geom types, faceting, themes, SVG/BMP export

### What's Missing (Gaps to Fill)

---

## PHASE 1: File I/O (Blocks Real Data Work)

**Priority: CRITICAL — without this, users can't load data files**

### 1.1 `read_csv(path)` / `write_csv(df, path)` builtins
- Read CSV from filesystem path → DataFrame
- Write DataFrame to filesystem path as CSV
- Deterministic column ordering (BTreeMap-based)
- UTF-8 validation, configurable delimiter/header/quoting

### 1.2 `read_lines(path)` / `write_lines(lines, path)` builtins
- Line-by-line text file reading
- Returns array of strings

### 1.3 `read_json(path)` / `write_json(value, path)` builtins
- Minimal JSON parser (no external deps)
- Supports nested objects → struct conversion
- Flat JSON arrays → DataFrame auto-conversion

### 1.4 File existence / path utilities
- `file_exists(path)` → bool
- `path_join(parts)` → string
- `dir_list(path)` → array of filenames

**Files to modify:**
- `crates/cjc-runtime/src/builtins.rs` — new file I/O dispatch
- `crates/cjc-eval/src/lib.rs` — wire new builtins
- `crates/cjc-mir-exec/src/lib.rs` — wire new builtins

**Tests:** `tests/beta_hardening/test_file_io.rs`

---

## PHASE 2: Descriptive Statistics Expansion

**Priority: HIGH — data exploration requires these**

### 2.1 Correlation & Covariance
- `cor(x, y)` → Pearson correlation coefficient (Kahan summation)
- `cor_matrix(df, cols)` → correlation matrix as Tensor
- `cov(x, y)` → sample covariance
- `cov_matrix(df, cols)` → covariance matrix as Tensor

### 2.2 Quantiles & Distribution Summaries
- `quantile(arr, p)` → p-th quantile (linear interpolation)
- `iqr(arr)` → interquartile range (Q3 - Q1)
- `summary(arr)` → struct { min, q1, median, mean, q3, max, sd, n }
- `skewness(arr)` → sample skewness
- `kurtosis(arr)` → sample excess kurtosis

### 2.3 Cumulative & Window Functions
- `cumsum(arr)` → running cumulative sum (Kahan)
- `cumprod(arr)` → running cumulative product
- `cummax(arr)` / `cummin(arr)` → running extrema
- `lag(arr, k)` → shift array by k positions (fill with Void/NaN)
- `lead(arr, k)` → shift array backward by k positions
- `rolling_mean(arr, window)` → rolling average (Kahan)
- `rolling_sd(arr, window)` → rolling standard deviation

### 2.4 Ranking Functions
- `rank(arr)` → average rank (tie-breaking: average)
- `dense_rank(arr)` → dense rank (no gaps)
- `row_number(arr)` → 1-indexed ordinal rank
- `percent_rank(arr)` → percentile rank [0, 1]

### 2.5 TidyView Integration
- All cumulative/window/rank functions usable inside `mutate()` as DExpr operations
- `DExpr::CumSum(col)`, `DExpr::Lag(col, k)`, etc.

**Files to modify:**
- `crates/cjc-runtime/src/stats.rs` (new file or extend existing)
- `crates/cjc-runtime/src/builtins.rs`
- `crates/cjc-data/src/tidy_dispatch.rs` — new DExpr kinds
- Both executors

**Tests:** `tests/beta_hardening/test_descriptive_stats.rs`

---

## PHASE 3: Inferential Statistics

**Priority: HIGH — hypothesis testing is core data science**

### 3.1 Hypothesis Tests
- `t_test_one(arr, mu0)` → struct { statistic, p_value, df, ci_low, ci_high }
- `t_test_two(arr1, arr2, equal_var)` → struct { statistic, p_value, df }
- `chi_sq_test(observed, expected)` → struct { statistic, p_value, df }
- `f_test(arr1, arr2)` → struct { statistic, p_value, df1, df2 }

### 3.2 Probability Distributions (CDFs/PDFs/Quantile functions)
- Normal: `dnorm(x, mu, sigma)`, `pnorm(x, mu, sigma)`, `qnorm(p, mu, sigma)`
- Student-t: `dt(x, df)`, `pt(x, df)`, `qt(p, df)`
- Chi-squared: `dchisq(x, df)`, `pchisq(x, df)`
- F distribution: `df_dist(x, df1, df2)`, `pf(x, df1, df2)`
- Binomial: `dbinom(k, n, p)`, `pbinom(k, n, p)`
- Poisson: `dpois(k, lambda)`, `ppois(k, lambda)`

### 3.3 Confidence Intervals
- `ci_mean(arr, alpha)` → struct { lower, upper, level }
- `ci_proportion(successes, n, alpha)` → struct { lower, upper }

### 3.4 Non-parametric Tests
- `wilcoxon_test(arr1, arr2)` → struct { statistic, p_value }
- `ks_test(arr1, arr2)` → Kolmogorov-Smirnov test

**Implementation notes:**
- All distribution functions implemented from scratch (no external deps)
- Use Lanczos approximation for gamma function
- Use regularized incomplete beta for F/t/chi-sq CDFs
- All reductions via Kahan summation

**Files to create:**
- `crates/cjc-runtime/src/distributions.rs`
- `crates/cjc-runtime/src/hypothesis.rs`

**Tests:** `tests/beta_hardening/test_inferential_stats.rs`

---

## PHASE 4: Linear Algebra Completeness

**Priority: MEDIUM — needed for PCA, regression, dimensionality reduction**

### 4.1 Missing Decompositions
- `svd(tensor)` → struct { u, s, vt } (Golub-Kahan bidiagonalization)
- `eig(tensor)` → struct { values, vectors } (QR algorithm)
- `det(tensor)` → float (via LU decomposition, already have LU)

### 4.2 Solvers
- `solve(A, b)` → x such that Ax = b (LU-based, expose existing)
- `lstsq(A, b)` → x minimizing ||Ax - b||^2 (via QR, already have QR)

### 4.3 Convenience
- `pinv(tensor)` → Moore-Penrose pseudoinverse (via SVD)
- `trace(tensor)` → sum of diagonal
- `diag(tensor)` → diagonal vector, or diag(vec) → diagonal matrix
- `eye(n)` → n×n identity matrix

### 4.4 Statistical Applications
- `pca(data, n_components)` → struct { components, explained_variance, loadings }
- `linear_regression(X, y)` → struct { coefficients, r_squared, residuals, p_values }

**Files to modify:**
- `crates/cjc-runtime/src/linalg.rs` (extend existing)
- `crates/cjc-runtime/src/builtins.rs`

**Tests:** `tests/beta_hardening/test_linalg_completeness.rs`

---

## PHASE 5: Data Preprocessing & Feature Engineering

**Priority: MEDIUM — ML workflows require these**

### 5.1 Encoding
- `one_hot(arr)` → Tensor (one-hot matrix)
- `label_encode(arr)` → array of ints + mapping struct
- `ordinal_encode(arr, levels)` → array of ints in specified order

### 5.2 Scaling & Normalization
- `standardize(arr)` → z-score normalized array
- `min_max_scale(arr, low, high)` → rescaled to [low, high]
- `robust_scale(arr)` → median/IQR-based scaling

### 5.3 Missing Data
- `fillna(arr, value)` → replace null/NaN with value
- `dropna(df, cols)` → remove rows with any null in specified columns
- `interpolate_linear(arr)` → linearly interpolate missing values
- `coalesce(arr1, arr2, ...)` → first non-null from each position

### 5.4 Binning & Discretization
- `cut(arr, breaks)` → categorical from continuous (like R's cut)
- `qcut(arr, n)` → quantile-based binning into n equal-frequency bins

### 5.5 TidyView case_when
- `case_when(conditions, values, default)` → vectorized conditional
- Usable inside `mutate()` as a DExpr

**Files to modify:**
- `crates/cjc-runtime/src/preprocessing.rs` (new)
- `crates/cjc-data/src/tidy_dispatch.rs`
- `crates/cjc-runtime/src/builtins.rs`

**Tests:** `tests/beta_hardening/test_preprocessing.rs`

---

## PHASE 6: ML Convenience Layer

**Priority: MEDIUM — completes the ML story**

### 6.1 Optimizers (deterministic)
- `sgd_step(params, grads, lr)` → updated params
- `adam_step(params, grads, m, v, lr, beta1, beta2, t)` → struct { params, m, v }
- `rmsprop_step(params, grads, cache, lr, decay)` → struct { params, cache }

### 6.2 Loss Functions
- `mse_loss(predicted, actual)` → float
- `cross_entropy_loss(predicted, actual)` → float
- `binary_cross_entropy(predicted, actual)` → float
- `hinge_loss(predicted, actual)` → float

### 6.3 Activation Functions (as builtins)
- `sigmoid(x)` → element-wise logistic (already exists for scalars, extend to tensors)
- `relu(x)` → element-wise max(0, x)
- `softmax(arr)` → normalized exponentials
- `tanh(x)` → already exists, verify tensor support

### 6.4 Tensor Utilities
- `argmax(arr)` / `argmin(arr)` → index of extreme
- `topk(arr, k)` → top-k values + indices
- `concat(tensors, axis)` → concatenate along axis
- `stack(tensors, axis)` → stack along new axis
- `split(tensor, n, axis)` → split into n parts

### 6.5 Model Persistence
- `snap_save(value, path)` → save any CJC value to file (snap format exists, add file I/O)
- `snap_load(path)` → load CJC value from file

**Files to modify:**
- `crates/cjc-runtime/src/ml.rs` (extend existing)
- `crates/cjc-runtime/src/builtins.rs`
- `crates/cjc-snap/` (add file I/O layer)

**Tests:** `tests/beta_hardening/test_ml_convenience.rs`

---

## PHASE 7: Vizor Visualization Enhancements

**Priority: LOW — Vizor is already strong, these are polish items**

### 7.1 Statistical Plot Integration
- `geom_qq()` — QQ plot (quantile-quantile against normal)
- `geom_pairs()` — scatterplot matrix (pairs plot)
- `geom_corrplot()` — correlation matrix heatmap with values
- `geom_residuals()` — residual diagnostic panel (4 plots)

### 7.2 Interactive Export
- `plot.save_html(path)` — self-contained HTML with SVG + JS tooltips
- Hover tooltips showing data values
- Pan/zoom (minimal JS, no framework dependency)

### 7.3 Annotation Layer
- `annotate_text(x, y, label)` — add text annotation
- `annotate_hline(y)` / `annotate_vline(x)` — reference lines
- `annotate_rect(x1, y1, x2, y2)` — highlight region

**Files to modify:**
- `crates/cjc-vizor/`

---

## IMPLEMENTATION ORDER (Recommended)

```
Phase 1 (File I/O)          ← CRITICAL, do first
  ↓
Phase 2 (Descriptive Stats) ← enables data exploration
  ↓
Phase 3 (Inferential Stats) ← enables hypothesis testing
  ↓
Phase 4 (Linear Algebra)    ← enables PCA, regression
  ↓
Phase 5 (Preprocessing)     ← enables ML pipelines
  ↓
Phase 6 (ML Convenience)    ← completes the ML story
  ↓
Phase 7 (Vizor Polish)      ← polish for publication
```

---

## VERIFICATION PROTOCOL

For each phase:

1. **Compile gate:** `cargo check --workspace` passes
2. **Unit tests:** Each new function has ≥2 tests
3. **Parity tests:** eval == mir-exec for every new builtin
4. **Determinism tests:** 3 runs with same seed → identical output
5. **Regression gate:** `cargo test --workspace` — zero new failures
6. **Kahan audit:** All floating-point reductions use Kahan/BinnedAccumulator
7. **BTreeMap audit:** No HashMap/HashSet in new code

---

## ESTIMATED SCOPE

| Phase | New Builtins | New Files | Estimated LOC |
|-------|-------------|-----------|---------------|
| 1. File I/O | 8 | 1 | ~400 |
| 2. Descriptive Stats | 18 | 1-2 | ~600 |
| 3. Inferential Stats | 16 | 2 | ~800 |
| 4. Linear Algebra | 10 | extend 1 | ~500 |
| 5. Preprocessing | 12 | 1 | ~400 |
| 6. ML Convenience | 15 | extend 2 | ~500 |
| 7. Vizor Polish | 6 | extend 1 | ~300 |
| **Total** | **~85** | **~8** | **~3,500** |
