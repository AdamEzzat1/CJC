# Bastion Feature Classification Matrix

**Decision criteria:**
- **P = CJC Primitive** -- Must be a builtin or runtime intrinsic (hot path, used by everything, needs SIMD/AD hooks)
- **R = Runtime/Stdlib** -- Implemented in Rust inside `cjc-runtime`, callable from CJC (needs unsafe/FFI/complex numerics)
- **B = Bastion Library** -- Pure CJC code, no runtime changes needed (composable from P + R)
- **X = Postpone** -- Requires features CJC doesn't have yet (parallel, complex types, or low priority)
- **! = Reject/Redesign** -- Doesn't fit CJC's model; needs fundamental rethink

---

## Legend: Determinism / Memory / Size / Priority

Each entry also gets rated on the four design axes:
- **Det** = Determinism fit (Y/N/Seed) -- does it produce identical output for same input?
- **Mem** = Memory model fit (NoGC / GC / Alloc) -- can it run in NoGC mode?
- **Size** = Binary size impact (S/M/L) -- how much code does it add to the runtime?
- **Pri** = Priority (1=critical, 2=high, 3=medium, 4=low)

---

## Category A: Descriptive Statistics

| Feature              | Class | Det | Mem   | Size | Pri | Notes                                          |
|----------------------|-------|-----|-------|------|-----|-------------------------------------------------|
| mean(xs)             | **P** | Y   | NoGC  | S    | 1   | Already exists as Tensor.mean()                |
| var(xs, ddof)        | **P** | Y   | NoGC  | S    | 1   | Already exists as Tensor.std() variant         |
| std(xs, ddof)        | **P** | Y   | NoGC  | S    | 1   | Already exists                                 |
| sum(xs)              | **P** | Y   | NoGC  | S    | 1   | Already exists as Tensor.sum()                 |
| min(xs) / max(xs)    | **P** | Y   | NoGC  | S    | 1   | Already exists                                 |
| median(xs)           | **R** | Y   | Alloc | S    | 1   | Needs quickselect (mutates scratch buffer)     |
| sort(xs)             | **P** | Y   | Alloc | S    | 1   | Already exists                                 |
| skewness(xs)         | **B** | Y   | NoGC  | S    | 2   | Pure formula: m3 / m2^1.5                      |
| kurtosis(xs)         | **B** | Y   | NoGC  | S    | 2   | Pure formula: m4 / m2^2 - 3                    |
| zscore(xs)           | **B** | Y   | Alloc | S    | 2   | (xs - mean) / std, pure CJC                    |
| percentile(xs, q)    | **R** | Y   | Alloc | S    | 2   | Needs partial sort (select_nth)                |
| iqr(xs)              | **B** | Y   | Alloc | S    | 2   | percentile(75) - percentile(25)                |
| cumsum(xs)           | **B** | Y   | Alloc | S    | 2   | Simple scan, pure CJC                          |
| cummean(xs)          | **B** | Y   | Alloc | S    | 3   | Running mean, pure CJC                         |
| ecdf(xs)             | **B** | Y   | Alloc | S    | 3   | Sort + rank / n                                |
| diff(xs, n)          | **B** | Y   | Alloc | S    | 2   | xs[i] - xs[i-n]                                |
| pct_change(xs, n)    | **B** | Y   | Alloc | S    | 3   | (xs[i] - xs[i-n]) / xs[i-n]                    |
| quantile_bins(xs, k) | **B** | Y   | Alloc | S    | 3   | Sort + assign bins                             |
| mean_skipna           | **R** | Y   | NoGC  | S    | 2   | NaN-aware variant (needs NaN concept in CJC)   |
| var_skipna            | **R** | Y   | NoGC  | S    | 2   | NaN-aware variant                              |
| minmax_scale(xs)     | **B** | Y   | Alloc | S    | 3   | (x - min) / (max - min)                        |
| robust_scale(xs)     | **B** | Y   | Alloc | S    | 3   | (x - median) / iqr                             |
| winsorize(xs, lo, hi)| **B** | Y   | Alloc | S    | 3   | Clamp to percentile bounds                     |

---

## Category B: Rolling Window Statistics

| Feature                      | Class | Det | Mem   | Size | Pri | Notes                                     |
|------------------------------|-------|-----|-------|------|-----|-------------------------------------------|
| rolling_mean(xs, w)          | **R** | Y   | Alloc | S    | 1   | Kahan-compensated O(n), runtime for perf  |
| rolling_std(xs, w)           | **R** | Y   | Alloc | S    | 1   | Fused with mean for numerical stability   |
| rolling_var(xs, w)           | **B** | Y   | Alloc | S    | 2   | std^2, composable from rolling_std        |
| rolling_zscore(xs, w)        | **B** | Y   | Alloc | S    | 2   | (x - rolling_mean) / rolling_std          |
| rolling_multi(xs, w, mask)   | **R** | Y   | Alloc | M    | 2   | Fused multi-stat kernel, runtime for perf |
| rolling_cov(xs, ys, w)       | **R** | Y   | Alloc | S    | 2   | Needs Kahan dual-accumulator              |
| rolling_corr(xs, ys, w)      | **B** | Y   | Alloc | S    | 3   | cov / (std_x * std_y)                     |
| rolling_beta(xs, ys, w)      | **B** | Y   | Alloc | S    | 3   | cov / var_x                               |
| RollingConfig                | **B** | Y   | NoGC  | S    | 2   | Struct with window/alignment/nan_policy   |
| Alignment (trailing/centered)| **B** | Y   | NoGC  | S    | 2   | Enum, pure CJC                            |
| NanPolicy                    | **R** | Y   | NoGC  | S    | 2   | Needs runtime NaN detection               |
| rolling_mean_axis0(mat, w)   | **X** | Y   | Alloc | M    | 3   | Needs 2D tensor support maturation        |
| ewma(xs, alpha)              | **B** | Y   | Alloc | S    | 3   | Simple recurrence, pure CJC              |

---

## Category C: Matrix Kernels

| Feature                   | Class | Det | Mem   | Size | Pri | Notes                                      |
|---------------------------|-------|-----|-------|------|-----|--------------------------------------------|
| cov_matrix(X)             | **R** | Y   | Alloc | M    | 1   | O(n*p^2), needs efficient 2D access        |
| corr_matrix(X)            | **R** | Y   | Alloc | M    | 1   | cov / (std_i * std_j), runtime for speed   |
| dot(a, b)                 | **P** | Y   | Alloc | S    | 1   | Already exists in CJC                      |
| diag(mat)                 | **B** | Y   | Alloc | S    | 3   | Extract diagonal, pure CJC                 |
| trace(mat)                | **B** | Y   | NoGC  | S    | 3   | Sum of diagonal                            |
| is_symmetric(mat, tol)    | **B** | Y   | NoGC  | S    | 4   | Double-loop comparison                     |
| XTX(X) / XXT(X)           | **B** | Y   | Alloc | S    | 3   | Gram matrices, compose from matmul         |
| pairwise_euclidean(X)     | **B** | Y   | Alloc | M    | 3   | O(n^2*p), pure CJC                         |
| pairwise_cosine(X)        | **B** | Y   | Alloc | M    | 3   | O(n^2*p), pure CJC                         |
| corr_distance(X)          | **B** | Y   | Alloc | S    | 4   | 1 - corr_matrix                            |
| cov_matrix_skipna(X)      | **R** | Y   | Alloc | M    | 3   | NaN-pairwise deletion, complex             |
| cov/corr (parallel)       | **X** | Y   | Alloc | M    | 4   | Needs Rayon / CJC parallel primitives      |

---

## Category D: Robust Estimators

| Feature                   | Class | Det | Mem   | Size | Pri | Notes                                      |
|---------------------------|-------|-----|-------|------|-----|--------------------------------------------|
| median(xs)                | **R** | Y   | Alloc | S    | 1   | quickselect, already classified above      |
| mad(xs)                   | **R** | Y   | Alloc | S    | 1   | median(|x - median(x)|), two quickselects  |
| trimmed_mean(xs, trim)    | **B** | Y   | Alloc | S    | 2   | Sort + slice + mean                        |
| trimmed_std(xs, trim)     | **B** | Y   | Alloc | S    | 3   | Sort + slice + std                         |
| winsorized_mean(xs, lo,hi)| **B** | Y   | Alloc | S    | 3   | Clamp + mean                               |
| mad_std(xs)               | **B** | Y   | Alloc | S    | 3   | mad * 1.4826                               |
| biweight_midvariance(xs)  | **B** | Y   | Alloc | S    | 4   | Iterative formula, pure CJC               |
| qn_scale(xs)              | **R** | Y   | Alloc | M    | 4   | O(n^2) pairwise diffs + quickselect        |
| huber_location(xs)        | **B** | Y   | Alloc | S    | 3   | Iterative M-estimator, pure CJC           |
| rolling_median(xs, w)     | **R** | Y   | Alloc | M    | 3   | Needs sorted-window data structure         |
| RobustConfig              | **B** | Y   | NoGC  | S    | 3   | Policy enums, pure CJC struct              |
| RobustWorkspace           | **!** | Y   | Alloc | S    | -   | Redesign: CJC doesn't expose scratch APIs  |

---

## Category E: Resampling

| Feature                        | Class | Det  | Mem   | Size | Pri | Notes                                   |
|--------------------------------|-------|------|-------|------|-----|-----------------------------------------|
| bootstrap_mean(xs, B, seed)    | **B** | Seed | Alloc | S    | 2   | Pure CJC with rand_uniform_int          |
| bootstrap_ci(xs, B, conf, seed)| **B** | Seed | Alloc | S    | 2   | Bootstrap + percentile                  |
| bootstrap_se(xs, B, seed)      | **B** | Seed | Alloc | S    | 3   | std of bootstrap samples                |
| bootstrap_bca_ci               | **B** | Seed | Alloc | M    | 3   | Bias-corrected accelerated              |
| bootstrap_t_ci                 | **B** | Seed | Alloc | M    | 3   | Studentized bootstrap                   |
| bayesian_bootstrap_ci          | **B** | Seed | Alloc | M    | 4   | Dirichlet weights                       |
| jackknife_mean(xs)             | **B** | Y    | Alloc | S    | 2   | Leave-one-out, pure CJC                |
| jackknife_ci(xs)               | **B** | Y    | Alloc | S    | 3   | LOO + percentile                        |
| influence_mean(xs)             | **B** | Y    | Alloc | S    | 3   | LOO influence values                    |
| delete_d_jackknife             | **B** | Y    | Alloc | M    | 4   | Delete-d generalization                 |
| permutation_corr_test          | **B** | Seed | Alloc | M    | 3   | Shuffle + recompute                     |
| permutation_mean_diff_test     | **B** | Seed | Alloc | M    | 3   | Shuffle + recompute                     |
| moving_block_bootstrap         | **X** | Seed | Alloc | M    | 4   | Needs TSA maturity first                |
| circular_block_bootstrap       | **X** | Seed | Alloc | M    | 4   | Needs TSA maturity first                |
| stationary_bootstrap           | **X** | Seed | Alloc | M    | 4   | Needs geometric-distributed blocks      |
| jackknife_after_bootstrap      | **X** | Seed | Alloc | L    | 4   | Complex nested resampling               |
| Parallel bootstrap (Rayon)     | **X** | Seed | Alloc | M    | 4   | CJC has no parallel primitives yet      |

---

## Category F: Time-Series Analysis

| Feature                   | Class | Det | Mem   | Size | Pri | Notes                                      |
|---------------------------|-------|-----|-------|------|-----|--------------------------------------------|
| acf(xs, nlags)            | **B** | Y   | Alloc | S    | 1   | Direct O(n*k), pure CJC                    |
| pacf(xs, nlags)           | **B** | Y   | Alloc | S    | 2   | Levinson-Durbin recursion, pure CJC        |
| acovf(xs, nlags)          | **B** | Y   | Alloc | S    | 2   | Autocovariance, pure CJC                   |
| ccf(xs, ys, nlags)        | **B** | Y   | Alloc | S    | 3   | Cross-correlation, pure CJC                |
| acf_with_ci(xs, nlags)    | **B** | Y   | Alloc | S    | 3   | ACF + 1.96/sqrt(n) bands                   |
| ljung_box(xs, lags)       | **R** | Y   | Alloc | S    | 2   | Needs chi-square CDF (dist primitive)      |
| box_pierce(xs, lags)      | **R** | Y   | Alloc | S    | 3   | Needs chi-square CDF                       |
| durbin_watson(xs)         | **B** | Y   | NoGC  | S    | 2   | Pure sum-of-squares formula                |
| runs_test(xs)             | **R** | Y   | Alloc | S    | 3   | Needs Normal CDF for p-value               |
| adf_test(xs)              | **R** | Y   | Alloc | M    | 2   | OLS + critical value table interpolation   |
| kpss_test(xs)             | **R** | Y   | Alloc | M    | 2   | OLS + long-run variance + critical tables  |
| pp_test(xs)               | **R** | Y   | Alloc | M    | 3   | Variant of ADF                             |
| variance_ratio_test(xs)   | **R** | Y   | Alloc | S    | 3   | Needs Normal CDF for p-value               |
| zivot_andrews_test(xs)    | **R** | Y   | Alloc | L    | 4   | Structural break test, heavy OLS           |
| periodogram(xs)           | **R** | Y   | Alloc | M    | 2   | Needs FFT primitive                        |
| welch_psd(xs)             | **R** | Y   | Alloc | M    | 3   | Needs FFT + windowing                      |
| spectral_entropy(xs)      | **B** | Y   | Alloc | S    | 3   | -sum(p * log(p)), pure CJC                 |
| dominant_frequency(xs)    | **B** | Y   | Alloc | S    | 3   | argmax of periodogram                      |
| spectral_flatness(xs)     | **B** | Y   | Alloc | S    | 4   | geo_mean / arith_mean of PSD               |
| band_power(xs, f1, f2)    | **B** | Y   | Alloc | S    | 4   | Integrate PSD in band                      |
| rolling_autocorr(xs, w, k)| **B** | Y   | Alloc | S    | 3   | Rolling ACF(k), pure CJC                   |
| hp_filter(xs, lambda)     | **R** | Y   | Alloc | M    | 3   | Tridiagonal solve (needs linear algebra)   |
| fractional_diff(xs, d)    | **B** | Y   | Alloc | S    | 3   | Hosking weights, pure CJC                  |
| stl_decompose(xs)         | **X** | Y   | Alloc | L    | 4   | LOESS + iteration, complex                 |
| engle_granger(y, x)       | **X** | Y   | Alloc | L    | 4   | OLS + ADF on residuals, needs OLS infra    |
| seasonal_unit_root        | **X** | Y   | Alloc | L    | 4   | HEGY test, very specialized                |
| integration_order(xs)     | **X** | Y   | Alloc | M    | 4   | Repeated ADF, depends on adf_test          |

---

## Category G: Statistical Inference

| Feature                   | Class | Det | Mem   | Size | Pri | Notes                                      |
|---------------------------|-------|-----|-------|------|-----|--------------------------------------------|
| t_test_1samp(xs, mu)      | **R** | Y   | NoGC  | S    | 1   | Needs Student-t CDF                        |
| t_test_2samp(xs, ys)      | **R** | Y   | NoGC  | S    | 1   | Needs Student-t CDF (Welch correction)     |
| t_test_paired(xs, ys)     | **R** | Y   | Alloc | S    | 2   | diff + 1samp t-test                        |
| chi2_gof(obs, exp)        | **R** | Y   | NoGC  | S    | 2   | Needs chi-square CDF                       |
| chi2_independence(table)  | **R** | Y   | Alloc | S    | 2   | Needs chi-square CDF                       |
| f_test_oneway(groups...)  | **R** | Y   | Alloc | S    | 2   | Needs F-distribution CDF                   |
| levene_test(groups...)    | **R** | Y   | Alloc | S    | 3   | Needs F-distribution CDF                   |
| jarque_bera(xs)           | **B** | Y   | NoGC  | S    | 3   | Pure formula using skew/kurtosis + chi2 CDF|
| anderson_darling(xs)      | **R** | Y   | Alloc | M    | 3   | Sort + CDF evaluation + critical tables    |
| pearson_corr_test(xs, ys) | **R** | Y   | NoGC  | S    | 2   | Needs Student-t CDF                        |
| spearman_corr_test(xs,ys) | **R** | Y   | Alloc | S    | 2   | Rank + pearson + t CDF                     |
| mann_whitney_u(xs, ys)    | **R** | Y   | Alloc | S    | 3   | Rank sum + Normal approx                   |
| ks_1samp(xs)              | **R** | Y   | Alloc | M    | 3   | Empirical vs theoretical CDF               |
| cohens_d(xs, ys)          | **B** | Y   | NoGC  | S    | 2   | Pure formula                               |
| hedges_g(xs, ys)          | **B** | Y   | NoGC  | S    | 3   | Cohen's d with correction factor           |
| mean_diff_ci(xs, ys)      | **B** | Y   | NoGC  | S    | 2   | t-based CI                                 |
| p_adjust(pvals, method)   | **B** | Y   | Alloc | S    | 3   | Sort + adjust (BH, Bonferroni)             |
| proportion_ztest(x, n, p0)| **R** | Y   | NoGC  | S    | 3   | Needs Normal CDF                           |
| corr_ci(r, n)             | **B** | Y   | NoGC  | S    | 3   | Fisher z-transform + Normal                |
| var_ci(xs)                | **R** | Y   | NoGC  | S    | 3   | Needs chi-square PPF                       |
| odds_ratio(table)         | **B** | Y   | NoGC  | S    | 4   | Pure formula + log CI                      |
| rank_biserial(xs, ys)     | **B** | Y   | Alloc | S    | 4   | Mann-Whitney based                         |
| cliffs_delta(xs, ys)      | **B** | Y   | Alloc | S    | 4   | Pairwise comparison                        |

---

## Category H: Probability Distributions

| Feature               | Class | Det | Mem   | Size | Pri | Notes                                        |
|-----------------------|-------|-----|-------|------|-----|----------------------------------------------|
| norm_cdf(x, mu, sig)  | **P** | Y   | NoGC  | S    | 1   | **New primitive**: erfc-based, needed everywhere |
| norm_pdf(x, mu, sig)  | **P** | Y   | NoGC  | S    | 1   | **New primitive**: exp + const               |
| norm_ppf(x, mu, sig)  | **R** | Y   | NoGC  | M    | 1   | Inverse CDF, needs rational approx          |
| norm_sf(x)             | **B** | Y   | NoGC  | S    | 2   | 1 - cdf or erfc-based                       |
| norm_logpdf(x)         | **B** | Y   | NoGC  | S    | 3   | -0.5*((x-mu)/sig)^2 - log(sig*sqrt(2pi))    |
| norm_logsf(x)          | **B** | Y   | NoGC  | S    | 3   | log(sf(x))                                   |
| norm_cumhazard(x)      | **B** | Y   | NoGC  | S    | 4   | -log(sf(x))                                  |
| t_cdf(x, df)           | **R** | Y   | NoGC  | M    | 1   | **New runtime**: regularized incomplete beta |
| t_ppf(x, df)           | **R** | Y   | NoGC  | M    | 2   | Inverse of t_cdf                             |
| chi2_cdf(x, df)        | **R** | Y   | NoGC  | M    | 1   | **New runtime**: regularized gamma           |
| chi2_ppf(x, df)        | **R** | Y   | NoGC  | M    | 2   | Inverse of chi2_cdf                          |
| f_cdf(x, df1, df2)     | **R** | Y   | NoGC  | M    | 2   | **New runtime**: regularized incomplete beta |
| exp_pdf/cdf/ppf        | **B** | Y   | NoGC  | S    | 2   | Simple formulas, pure CJC                    |
| unif_pdf/cdf/ppf       | **B** | Y   | NoGC  | S    | 3   | Trivial formulas                             |
| erf(x) / erfc(x)       | **P** | Y   | NoGC  | S    | 1   | **New primitive**: needed for Normal CDF     |

---

## Summary Counts

| Classification                  | Count | % of features |
|---------------------------------|-------|---------------|
| **P** -- CJC Primitive          | 8     | 7%            |
| **R** -- Runtime/Stdlib         | 33    | 28%           |
| **B** -- Bastion Library (CJC)  | 55    | 47%           |
| **X** -- Postpone               | 15    | 13%           |
| **!** -- Reject/Redesign        | 1     | 1%            |
| Existing in CJC already         | ~6    | 5%            |
| **Total unique features**       | ~118  |               |

### Key Insight
**47% of bunker-stats can be written as pure CJC code** once the runtime provides
~33 numerical primitives (distribution CDFs, FFT, quickselect, rolling accumulators).
Only 8 features need to become first-class CJC primitives (erf/erfc, norm_cdf,
norm_pdf, and the existing mean/sum/min/max/sort/dot).
