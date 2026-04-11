> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# Bastion Migration Map

**From:** bunker-stats v0.2.9 (Rust/PyO3, ~18K LOC)
**To:** Bastion (CJC-native statistics library)

---

## Architecture Overview

```
+-------------------------------------------------------+
|                    Bastion Library                      |
|                    (pure CJC code)                      |
|                                                         |
|  bastion/descriptive.cjc   -- skew, kurtosis, zscore   |
|  bastion/rolling.cjc       -- rolling_var, ewma, etc.  |
|  bastion/robust.cjc        -- trimmed_mean, huber, etc.|
|  bastion/resampling.cjc    -- bootstrap, jackknife      |
|  bastion/tsa.cjc           -- acf, pacf, durbin_watson  |
|  bastion/infer.cjc         -- cohens_d, hedges_g, etc. |
|  bastion/dist.cjc          -- exp_pdf, unif_cdf, etc.  |
|  bastion/transform.cjc     -- minmax, robust_scale     |
+-------------------------------------------------------+
                          |
                   calls into
                          v
+-------------------------------------------------------+
|              CJC Runtime (Rust)                         |
|                                                         |
|  Distributions:  t_cdf, chi2_cdf, f_cdf, norm_ppf     |
|  Rolling:        rolling_mean, rolling_std, rolling_cov |
|  Matrix:         cov_matrix, corr_matrix               |
|  Selection:      median, percentile (quickselect)      |
|  Stationarity:   adf_test, kpss_test, pp_test          |
|  Spectral:       periodogram, welch_psd (FFT)          |
|  Inference:      t_test, chi2_test, anova, mann_whitney |
+-------------------------------------------------------+
                          |
                   calls into
                          v
+-------------------------------------------------------+
|              CJC Primitives (builtins)                  |
|                                                         |
|  Existing:  mean, sum, std, min, max, sort, dot,       |
|             sqrt, log, exp, sin, cos, abs, floor,      |
|             ceil, round, len, rand (seeded)             |
|  NEW:       erf, erfc, norm_cdf, norm_pdf              |
+-------------------------------------------------------+
```

---

## Phase 0: Primitive Foundation (prerequisite)

**Goal:** Add the 4 new CJC primitives that everything else depends on.

| Primitive    | Where to add                    | Implementation                     | Tests needed |
|-------------|--------------------------------|-------------------------------------|-------------|
| `erf(x)`    | `cjc-runtime/src/builtins.rs`  | Abramowitz & Stegun or libm port   | 10          |
| `erfc(x)`   | `cjc-runtime/src/builtins.rs`  | 1 - erf(x) or direct               | 10          |
| `norm_cdf(x)` | `cjc-runtime/src/builtins.rs`| 0.5 * erfc(-x / sqrt(2))           | 10          |
| `norm_pdf(x)` | `cjc-runtime/src/builtins.rs`| exp(-0.5*x^2) / sqrt(2*pi)         | 5           |

**Estimated effort:** ~200 LOC Rust, 35 tests.
**Binary size impact:** ~2 KB.

---

## Phase 1: Runtime Numerics (unblocks 70% of Bastion)

**Goal:** Add the runtime functions that require Rust (special functions, FFT, etc.)

### Phase 1a: Distribution CDFs (unblocks all of inference)

| Function       | Algorithm                          | LOC est | Priority |
|----------------|------------------------------------|---------|----------|
| `t_cdf(x, df)` | Regularized incomplete beta       | 150     | 1        |
| `t_ppf(x, df)` | Newton-Raphson on t_cdf           | 80      | 2        |
| `chi2_cdf(x,df)`| Regularized lower gamma           | 120     | 1        |
| `chi2_ppf(x,df)`| Newton-Raphson on chi2_cdf        | 80      | 2        |
| `f_cdf(x,d1,d2)`| Regularized incomplete beta       | 100     | 2        |
| `norm_ppf(x)`   | Rational approximation (Beasley-Springer-Moro) | 80 | 1 |
| `beta_inc(a,b,x)`| Continued fraction (Lentz)       | 200     | 1        |
| `gamma_inc(a,x)` | Series + continued fraction      | 200     | 1        |

**Total:** ~1,010 LOC Rust, ~80 tests.
**Dependencies unlocked:** All inference tests, stationarity p-values, ACF confidence bands.

### Phase 1b: Quickselect & Rolling Accumulators

| Function                       | Algorithm                    | LOC est | Priority |
|-------------------------------|------------------------------|---------|----------|
| `median(xs)`                   | Introselect (quickselect)   | 80      | 1        |
| `percentile(xs, q)`            | Introselect + interpolation | 60      | 2        |
| `mad(xs)`                      | Two quickselects            | 40      | 1        |
| `rolling_mean(xs, w)`          | Kahan sliding window        | 80      | 1        |
| `rolling_std(xs, w)`           | Kahan sum + sumsq           | 100     | 1        |
| `rolling_cov(xs, ys, w)`       | Dual Kahan accumulators     | 100     | 2        |
| `rolling_multi(xs, w, mask)`   | Fused kernel                | 200     | 2        |
| `rolling_median(xs, w)`        | Sorted-window structure     | 200     | 3        |

**Total:** ~860 LOC Rust, ~60 tests.

### Phase 1c: FFT & Spectral

| Function                | Algorithm                        | LOC est | Priority |
|------------------------|----------------------------------|---------|----------|
| `fft(xs)`               | Radix-2 + Bluestein (or port)   | 300     | 2        |
| `periodogram(xs)`       | FFT + power spectrum             | 80      | 2        |
| `welch_psd(xs, nperseg)`| Windowed FFT + averaging         | 120     | 3        |

**Total:** ~500 LOC Rust, ~30 tests.

### Phase 1d: Matrix Kernels

| Function                   | Algorithm                          | LOC est | Priority |
|---------------------------|------------------------------------|---------|----------|
| `cov_matrix(X)`            | Column-mean + cross products      | 120     | 1        |
| `corr_matrix(X)`           | cov / (std_i * std_j)             | 80      | 1        |
| `cov_matrix_skipna(X)`     | Pairwise deletion                 | 150     | 3        |

**Total:** ~350 LOC Rust, ~30 tests.

### Phase 1e: Stationarity Tests

| Function                 | Algorithm                          | LOC est | Priority |
|-------------------------|------------------------------------|---------|----------|
| `adf_test(xs)`           | OLS regression + DF critical table| 200     | 2        |
| `kpss_test(xs)`          | OLS + LRV estimation              | 200     | 2        |
| `pp_test(xs)`            | Modified ADF                      | 150     | 3        |

**Total:** ~550 LOC Rust, ~30 tests.

### Phase 1f: Hypothesis Tests

| Function                     | Algorithm                     | LOC est | Priority |
|-----------------------------|-------------------------------|---------|----------|
| `t_test_1samp(xs, mu)`      | Formula + t_cdf               | 40      | 1        |
| `t_test_2samp(xs, ys)`      | Welch formula + t_cdf         | 60      | 1        |
| `t_test_paired(xs, ys)`     | diff + t_test_1samp           | 30      | 2        |
| `chi2_gof(obs, exp)`        | Formula + chi2_cdf            | 50      | 2        |
| `chi2_independence(table)`   | Expected counts + chi2        | 70      | 2        |
| `f_test_oneway(groups...)`   | SS between/within + F_cdf    | 80      | 2        |
| `levene_test(groups...)`     | Median-centered ANOVA        | 60      | 3        |
| `mann_whitney_u(xs, ys)`     | Rank sum + normal approx     | 80      | 3        |
| `ks_1samp(xs)`               | Empirical CDF comparison     | 100     | 3        |
| `anderson_darling(xs)`       | Sorted CDF + formula         | 80      | 3        |
| `pearson_corr_test(xs, ys)`  | Corr + t-transform           | 40      | 2        |
| `spearman_corr_test(xs, ys)` | Rank + pearson               | 50      | 2        |
| `normality_tests (JB, AD)`   | Skew/kurt + chi2 / AD table  | 60      | 3        |

**Total:** ~800 LOC Rust, ~60 tests.

---

## Phase 2: Bastion Library (pure CJC)

Once the runtime provides Phase 1 functions, write these as `.cjc` library files.

### bastion/descriptive.cjc (~150 LOC CJC)
```
fn skewness(xs: [f64]) -> f64
fn kurtosis(xs: [f64]) -> f64
fn zscore(xs: [f64]) -> [f64]
fn cumsum(xs: [f64]) -> [f64]
fn cummean(xs: [f64]) -> [f64]
fn diff(xs: [f64], n: i64) -> [f64]
fn pct_change(xs: [f64], n: i64) -> [f64]
fn ecdf(xs: [f64]) -> ([f64], [f64])
fn quantile_bins(xs: [f64], k: i64) -> [i64]
```

### bastion/rolling.cjc (~100 LOC CJC)
```
fn rolling_var(xs: [f64], w: i64) -> [f64]
fn rolling_zscore(xs: [f64], w: i64) -> [f64]
fn rolling_corr(xs: [f64], ys: [f64], w: i64) -> [f64]
fn rolling_beta(xs: [f64], ys: [f64], w: i64) -> [f64]
fn ewma(xs: [f64], alpha: f64) -> [f64]
```

### bastion/robust.cjc (~120 LOC CJC)
```
fn trimmed_mean(xs: [f64], trim: f64) -> f64
fn trimmed_std(xs: [f64], trim: f64) -> f64
fn winsorized_mean(xs: [f64], lo: f64, hi: f64) -> f64
fn mad_std(xs: [f64]) -> f64
fn huber_location(xs: [f64], c: f64, tol: f64) -> f64
fn biweight_midvariance(xs: [f64]) -> f64
```

### bastion/resampling.cjc (~200 LOC CJC)
```
fn bootstrap_mean(xs: [f64], B: i64, seed: i64) -> f64
fn bootstrap_ci(xs: [f64], B: i64, conf: f64, seed: i64) -> (f64, f64, f64)
fn bootstrap_se(xs: [f64], B: i64, seed: i64) -> f64
fn bootstrap_bca_ci(xs: [f64], B: i64, conf: f64, seed: i64) -> (f64, f64, f64)
fn jackknife_mean(xs: [f64]) -> (f64, f64, f64)
fn jackknife_ci(xs: [f64], conf: f64) -> (f64, f64, f64)
fn permutation_test_mean(xs: [f64], ys: [f64], B: i64, seed: i64) -> (f64, f64)
fn permutation_test_corr(xs: [f64], ys: [f64], B: i64, seed: i64) -> (f64, f64)
```

### bastion/tsa.cjc (~250 LOC CJC)
```
fn acf(xs: [f64], nlags: i64) -> [f64]
fn pacf(xs: [f64], nlags: i64) -> [f64]
fn acovf(xs: [f64], nlags: i64) -> [f64]
fn ccf(xs: [f64], ys: [f64], nlags: i64) -> [f64]
fn acf_with_ci(xs: [f64], nlags: i64) -> ([f64], f64)
fn durbin_watson(residuals: [f64]) -> f64
fn spectral_entropy(psd: [f64]) -> f64
fn dominant_frequency(freqs: [f64], psd: [f64]) -> f64
fn spectral_flatness(psd: [f64]) -> f64
fn band_power(freqs: [f64], psd: [f64], f1: f64, f2: f64) -> f64
fn rolling_autocorr(xs: [f64], w: i64, k: i64) -> [f64]
fn fractional_diff(xs: [f64], d: f64, thresh: f64) -> [f64]
```

### bastion/infer.cjc (~180 LOC CJC)
```
fn cohens_d(xs: [f64], ys: [f64]) -> f64
fn hedges_g(xs: [f64], ys: [f64]) -> f64
fn mean_diff_ci(xs: [f64], ys: [f64], conf: f64) -> (f64, f64, f64)
fn p_adjust_bonferroni(pvals: [f64]) -> [f64]
fn p_adjust_bh(pvals: [f64]) -> [f64]
fn corr_ci(r: f64, n: i64, conf: f64) -> (f64, f64)
fn odds_ratio(a: f64, b: f64, c: f64, d: f64) -> (f64, f64, f64)
fn rank_biserial(xs: [f64], ys: [f64]) -> f64
fn cliffs_delta(xs: [f64], ys: [f64]) -> f64
```

### bastion/dist.cjc (~100 LOC CJC)
```
fn exp_pdf(x: f64, rate: f64) -> f64
fn exp_cdf(x: f64, rate: f64) -> f64
fn exp_ppf(q: f64, rate: f64) -> f64
fn unif_pdf(x: f64, a: f64, b: f64) -> f64
fn unif_cdf(x: f64, a: f64, b: f64) -> f64
fn unif_ppf(q: f64, a: f64, b: f64) -> f64
fn norm_sf(x: f64) -> f64
fn norm_logpdf(x: f64, mu: f64, sig: f64) -> f64
fn norm_logsf(x: f64, mu: f64, sig: f64) -> f64
```

### bastion/transform.cjc (~80 LOC CJC)
```
fn minmax_scale(xs: [f64]) -> [f64]
fn robust_scale(xs: [f64]) -> [f64]
fn winsorize(xs: [f64], lo: f64, hi: f64) -> [f64]
fn demean(xs: [f64]) -> [f64]
```

**Total pure CJC:** ~1,180 LOC across 8 files.

---

## Phase 3: Postponed Features (requires CJC evolution)

These features are blocked on CJC capabilities that don't exist yet:

| Feature                     | Blocker                                    | When to revisit     |
|----------------------------|--------------------------------------------|---------------------|
| Parallel bootstrap          | No threading / task parallelism in CJC    | After Stage 3       |
| Parallel cov_matrix         | No threading                              | After Stage 3       |
| Rolling on 2D axis0         | 2D Tensor maturity                        | After Tensor v2     |
| Block bootstrap (moving, circular, stationary) | TSA + geometric dist | After Phase 2 TSA  |
| Jackknife-after-bootstrap   | Complex nested resampling patterns        | After Phase 2       |
| STL decompose               | LOESS implementation + iteration          | After Phase 2       |
| Engle-Granger cointegration | Full OLS infrastructure                   | After Phase 1e      |
| Seasonal unit root (HEGY)   | Very specialized, low demand              | v2.0                |
| Integration order test      | Depends on iterated ADF                   | After adf_test      |
| Zivot-Andrews               | Structural break, heavy computation       | v2.0                |

---

## Phase 4: Rejected / Redesigned

| Feature           | Reason                                              | Alternative                         |
|-------------------|-----------------------------------------------------|-------------------------------------|
| RobustWorkspace   | CJC has no mutable scratch buffer API               | Use CJC array allocation patterns   |
| PyO3 wrappers     | CJC is not Python                                   | Native CJC function signatures      |
| StatsMask bitflags| Over-engineered for CJC; use integer constants      | `let MEAN = 1; let STD = 2;` etc.  |
| NaN propagation variants | CJC may adopt a different NaN philosophy    | Design CJC NaN policy holistically  |

---

## Dependency Graph (build order)

```
Phase 0: erf, erfc, norm_cdf, norm_pdf
    |
    v
Phase 1a: beta_inc, gamma_inc -> t_cdf, chi2_cdf, f_cdf, norm_ppf
    |
    +---> Phase 1f: t_test, chi2_test, anova, etc.
    |
Phase 1b: median, percentile, mad, rolling_mean, rolling_std
    |
    +---> Phase 2: bastion/descriptive.cjc, bastion/robust.cjc
    |
Phase 1c: fft, periodogram, welch_psd
    |
    +---> Phase 2: bastion/tsa.cjc (spectral parts)
    |
Phase 1d: cov_matrix, corr_matrix
    |
    +---> Phase 2: bastion/infer.cjc (correlation tests)
    |
Phase 1e: adf_test, kpss_test
    |
    +---> Phase 2: bastion/tsa.cjc (stationarity parts)
```

---

## Sizing Summary

| Phase    | LOC (Rust) | LOC (CJC) | Tests | Effort |
|----------|-----------|-----------|-------|--------|
| Phase 0  | 200       | 0         | 35    | 1 day  |
| Phase 1a | 1,010     | 0         | 80    | 3 days |
| Phase 1b | 860       | 0         | 60    | 2 days |
| Phase 1c | 500       | 0         | 30    | 2 days |
| Phase 1d | 350       | 0         | 30    | 1 day  |
| Phase 1e | 550       | 0         | 30    | 2 days |
| Phase 1f | 800       | 0         | 60    | 2 days |
| Phase 2  | 0         | 1,180     | 200+  | 3 days |
| **Total**| **4,270** | **1,180** | **525+**| ~16 days |

### Comparison to bunker-stats
- bunker-stats: 17,975 LOC Rust (includes PyO3 boilerplate, duplicate skipna variants)
- Bastion total: ~5,450 LOC (4,270 Rust + 1,180 CJC)
- **Reduction: 3.3x smaller** (by eliminating PyO3/numpy, deduplicating, composing)

---

## Design Principles for Bastion

1. **Determinism first:** Every function produces identical output for identical input.
   Seeded RNG via CJC's existing deterministic random.

2. **NoGC where possible:** Descriptive stats on slices should pass NoGC verification.
   Allocation-requiring functions (sort, bootstrap) use standard CJC array allocation.

3. **Primitive-first:** Push complexity into ~33 runtime primitives; keep Bastion
   library code simple and auditable.

4. **Small binary:** The 4 new builtins add ~2 KB. Runtime functions add ~4 KB.
   No external crate dependencies (no statrs, no nalgebra, no rustfft).
   Implement special functions from scratch using well-known algorithms.

5. **Composability:** Bastion functions compose from primitives. Users can see the
   source and modify it. No opaque black boxes.

6. **No NaN surprise:** Design a single CJC-wide NaN policy (likely: reject at
   boundaries, propagate internally). Don't replicate bunker-stats' 3-variant pattern.
