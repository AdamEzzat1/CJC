# Bastion Primitive ABI

**Version:** 1.0
**Date:** 2026-03-08

---

## Design Philosophy

Bastion is not 200 builtins. It is a thin statistical vocabulary layered on top of
CJC's deterministic numerical substrate. Each primitive exists because it enables
5-20 higher-level functions to become thin wrappers or clean compositions.

**Primitive Admission Rule:** A primitive may enter CJC core/runtime only if it
enables approximately 5-20 higher-level statistical functions to become thin
wrappers or clean compositions. If it only helps one function, it belongs in
Bastion library code, not in the runtime.

---

## The 15-Primitive Model

### A. Reduction Primitives

| # | Primitive | Status | Location | Unlocks |
|---|-----------|--------|----------|---------|
| 1 | `sum_kahan(x)` | EXISTS | `cjc-repro/src/lib.rs:55` + `KahanAccumulatorF64` | mean, variance, std, se, cov, cor, all reductions |
| 2 | `mean_welford(x)` | EXISTS (as Kahan mean) | `cjc-runtime/src/stats.rs:18` | All mean-based stats |
| 3 | `var_welford(x, ddof)` | EXISTS (as two-pass Kahan) | `cjc-runtime/src/stats.rs:41` | variance, sd, se, z-score, standardize |
| 4 | `min_max(x)` | EXISTS (separate) | Tensor min/max folds | range, normalization, outlier detection |

**Why two-pass Kahan instead of Welford:** CJC's two-pass Kahan variance
(mean first, then sum of squared deviations) is numerically equivalent to
Welford for non-streaming use and integrates with the existing Kahan/Binned
accumulator infrastructure. Welford would only matter for single-pass streaming,
which CJC doesn't need yet.

### B. Ordering / Selection Primitives

| # | Primitive | Status | Location | Unlocks |
|---|-----------|--------|----------|---------|
| 5 | `stable_sort(x)` | EXISTS | `builtins.rs:356` | rank, quantile, ECDF, order statistics |
| 6 | `argsort_stable(x)` | EXISTS | `tensor.rs:1719` | rank, permutation-based ops |
| 7 | `nth_element(x, k)` | **NEW** | `stats.rs` | O(n) median, quantile, percentile, MAD, IQR, all robust stats |
| 8 | `rank(x, method)` | EXISTS | `stats.rs:430` | Spearman, Kendall, Mann-Whitney, Kruskal-Wallis |

**Why nth_element is the highest-leverage addition:** Before nth_element, median
required O(n log n) full sort. With nth_element (introselect), median is O(n)
expected. This cascades to: quantile, percentile, IQR, MAD, trimmed_mean,
winsorize, and all robust estimators that depend on order statistics.

### C. Elementwise / Transform Primitives

| # | Primitive | Status | Location | Unlocks |
|---|-----------|--------|----------|---------|
| 9 | `map(x, fn)` | EXISTS | `tensor.rs:495` + SIMD | z-score, scaling, any elementwise transform |
| 10| `zip_map(x, y, fn)` | EXISTS (as broadcast2) | `builtins.rs:2360` | covariance, correlation, residuals, pairwise ops |
| 11| `filter_mask(x, mask)` | **NEW** | `stats.rs` | NaN filtering, outlier removal, conditional selection |
| 12| `gather(x, idx)` | EXISTS | `tensor.rs:1731` | bootstrap sample construction, permutation ops |

### D. Window / Recurrence Primitives

| # | Primitive | Status | Location | Unlocks |
|---|-----------|--------|----------|---------|
| 13| `rolling_reduce(x, w, r)` | EXISTS (concrete) | `window.rs:33-122` | rolling mean/std/min/max, EWMA |
| 14| `prefix_scan(x, op)` | EXISTS (concrete) | `stats.rs:359-395` | cumsum, cumprod, cummax, cummin |

**Concrete vs generalized:** CJC has 4 concrete rolling ops (sum/mean/min/max)
and 4 concrete prefix ops (sum/prod/max/min). A generalized `rolling_reduce`
and `prefix_scan` would be cleaner but the concrete versions are sufficient
for Bastion Phase 1.

### E. Random / Sampling Primitive

| # | Primitive | Status | Location | Unlocks |
|---|-----------|--------|----------|---------|
| 15| `sample_indices(n, k, replace, seed)` | **NEW** | `stats.rs` | bootstrap, jackknife, permutation tests, cross-validation |

---

## Special Functions

| Function | Status | Location | Precision | Unlocks |
|----------|--------|----------|-----------|---------|
| `erf(x)` | **NEW** | `distributions.rs` | ~1.5e-7 | Higher-level dist functions, Bastion completeness |
| `erfc(x)` | **NEW** | `distributions.rs` | ~1.5e-7 | Complementary error function |
| `norm_cdf(x)` | EXISTS | `distributions.rs:177` | ~1.5e-7 | All hypothesis tests, confidence intervals |
| `norm_pdf(x)` | EXISTS | `distributions.rs:195` | Exact | Likelihood, density estimation |
| `norm_ppf(p)` | EXISTS | `distributions.rs:202` | Good | Confidence intervals, quantile functions |

---

## What Was Deferred

| Candidate | Reason for Deferral | When to Revisit |
|-----------|---------------------|-----------------|
| Fused min_max | Separate min/max work; minor perf gain | When profiling shows it matters |
| Welford online mean/var | Two-pass Kahan is sufficient; streaming not needed | If CJC adds streaming data support |
| Generalized rolling_reduce | Concrete window ops suffice for Bastion | When CJC adds closures-as-values |
| Generalized prefix_scan | Concrete cum ops suffice | Same as above |

---

## What Each Primitive Unlocks (Leverage Analysis)

### nth_element (NEW) -- unlocks ~15 functions
median_fast, quantile_fast, percentile, IQR, MAD, trimmed_mean, winsorize,
robust_scale, biweight_midvariance, Qn scale, whisker bounds (boxplot),
outlier detection, all robust estimators

### sample_indices (NEW) -- unlocks ~12 functions
bootstrap_mean, bootstrap_ci, bootstrap_se, bootstrap_bca,
jackknife_mean, jackknife_ci, permutation_test_mean, permutation_test_corr,
cross_validation_split, train_test_split, stratified_sample, shuffle

### erf/erfc (NEW) -- unlocks ~6 functions
Higher-precision normal CDF, probit function, Mills ratio,
inverse erf, Gaussian kernel density, error function integral

### filter_mask (NEW) -- unlocks ~8 functions
NaN-aware mean/var/std/median, outlier removal, conditional statistics,
masked reduction, boolean indexing on tensors
