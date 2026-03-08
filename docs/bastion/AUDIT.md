# Bunker-Stats Audit Report

**Date:** 2026-03-08
**Source:** `bunker-stats` v0.2.9 (~18K LOC Rust, single crate)
**Target:** CJC Bastion library (pure CJC + runtime primitives)

---

## 1. Crate-Level Summary

| Property          | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Rust edition      | 2021                                                         |
| Crate type        | `cdylib` + `rlib` (Python extension via PyO3)                |
| Total source LOC  | ~17,975 (excluding worktrees/target)                         |
| Source files       | 62 `.rs` files                                               |
| External deps     | `pyo3`, `numpy`, `statrs`, `nalgebra`, `rand`/`rand_pcg`, `rustfft`, `libm`, `bitflags`, opt `rayon` |
| PyO3 exports      | ~160 `#[pyfunction]`s + 1 `#[pyclass]` (`RobustStats`)      |
| Test files         | 4 integration test files + inline `#[cfg(test)]` modules     |
| Benchmarks         | 11 Criterion bench files                                     |

### Dependency Concern Matrix (for CJC migration)

| Dep       | Purpose                       | CJC Impact                                    |
|-----------|-------------------------------|-----------------------------------------------|
| pyo3      | Python FFI                    | **DROP** -- CJC is not Python                 |
| numpy     | Array interop                 | **DROP** -- CJC has native Tensor             |
| statrs    | t/F/chi2/Normal distributions | **REPLACE** with CJC dist primitives          |
| nalgebra  | Dense matrix ops (OLS, SVD)   | **REPLACE** or expose via runtime primitives  |
| rand/pcg  | Seeded RNG                    | **EXISTS** -- CJC has deterministic RNG       |
| rustfft   | FFT for spectral analysis     | **NEW PRIMITIVE** needed                      |
| libm      | erfc, erf for Normal CDF      | **NEW PRIMITIVE** needed (erfc/erf)           |
| bitflags  | StatsMask bitmask             | **DROP** -- CJC can use integer bitmask       |
| rayon     | Parallel iteration            | **POSTPONE** -- CJC is single-threaded        |

---

## 2. Module Architecture

```
bunker-stats/src/
  lib.rs              (2,712 LOC) -- PyO3 wrappers + core slice helpers
  infer/              (2,685 LOC) -- Statistical inference tests
    common.rs         -- Alternative enum, reject_nonfinite, mean/var helpers
    ttest.rs          -- One-sample, two-sample t-tests
    chi2.rs           -- Chi-square GOF + independence
    anova.rs          -- One-way ANOVA F-test, Levene
    normality.rs      -- Jarque-Bera, Anderson-Darling
    correlation.rs    -- Pearson/Spearman with p-values
    variance_tests.rs -- F-test for variance, Bartlett
    mann_whitney.rs   -- Mann-Whitney U
    ks.rs             -- Kolmogorov-Smirnov (1-sample)
    effect.rs         -- Cohen's d, Hedges' g, mean diff CI
    effect_nonparam.rs-- Rank biserial, Cliff's delta, eta/omega sq
    paired.rs         -- Paired t-test
    p_adjust.rs       -- Multiple testing correction (Bonferroni, BH)
    proportion.rs     -- Proportion z-tests
    corr_ci.rs        -- Correlation CI (Fisher z transform)
    var_ci.rs         -- Variance CI (chi-square)
    odds_ratio.rs     -- 2x2 odds ratio
  kernels/
    rolling/          (1,518 LOC) -- Sliding window statistics
      engine.rs       -- Kahan-compensated rolling mean/std
      config.rs       -- RollingConfig, Alignment, NanPolicy enums
      masks.rs        -- StatsMask bitflags (MEAN|STD|VAR|COUNT|MIN|MAX)
      multi.rs        -- Fused multi-stat 1D rolling kernel
      multi_axis0.rs  -- Fused multi-stat 2D rolling (along axis 0)
      axis0.rs        -- Rolling mean/std along axis 0
      covcorr.rs      -- Rolling covariance
      bounds.rs       -- Output length calculation
      var/std/mean/zscore/state -- Single-stat rolling variants
    quantile/         (123 LOC) -- Selection-based quantile ops
      select.rs       -- nth_element (partial sort)
      percentile.rs   -- Arbitrary percentile
      iqr.rs          -- Interquartile range
      winsor.rs       -- Winsorization
    matrix/           (887 LOC) -- Covariance/correlation matrices
      cov.rs          -- cov_matrix, cov_bias, cov_centered, cov_skipna,
                         XTX, XXT, pairwise Euclidean/cosine distance
      corr.rs         -- corr_matrix, corr_skipna, corr_distance
    robust/           (1,091 LOC) -- Robust estimators
      extended.rs     -- Median, MAD, trimmed mean, IQR, winsorized mean,
                         Qn, Huber M-estimator, biweight midvariance
      policy.rs       -- LocationPolicy/ScalePolicy enums, RobustConfig
      fit.rs          -- Robust fit/score (enum-dispatched)
      rolling.rs      -- Adaptive rolling median
      pyrobust.rs     -- RobustStats PyO3 class wrapper
      mad.rs          -- (legacy, thin)
      trimmed_mean.rs -- (legacy, thin)
    resampling/       (2,538 LOC) -- Bootstrap & jackknife
      core.rs         -- Pure Rust core (no PyO3), Criterion-benchable
      bootstrap.rs    -- Bootstrap mean/CI, BCa, studentized-t, Bayesian,
                         moving-block, circular-block, stationary bootstrap,
                         permutation tests (corr, mean diff)
      jackknife.rs    -- Jackknife mean/CI, influence, delete-d, JaB
    tsa/              (3,831 LOC) -- Time-series analysis
      stationarity.rs -- ADF, KPSS, Phillips-Perron, variance ratio,
                         Zivot-Andrews, trend stationarity, integration order,
                         seasonal diff, seasonal unit root
      diagnostics.rs  -- Ljung-Box, Durbin-Watson, BG test, Box-Pierce,
                         runs test, ACF zero-crossing
      acf_pacf.rs     -- ACF (raw), PACF (Levinson-Durbin, Yule-Walker),
                         ACOVF, ACF with CI, CCF, PACF-innovations, PACF-Burg
      spectral.rs     -- Periodogram (FFT + DFT), Welch PSD, cumulative
                         periodogram, dominant frequency, spectral entropy,
                         Bartlett PSD, spectral peaks/flatness/centroid/rolloff,
                         band power
      rolling.rs      -- Rolling TSA stats (rolling AR(1), etc.)
      rolling_autocorr.rs -- Rolling autocorrelation, rolling cross-correlation
      cointegration.rs -- Engle-Granger test (WIP, not wired)
      decomposition.rs -- HP filter, fractional diff, STL decompose (WIP)
      config.rs       -- KPSS/TSA config structs
    dist/             (766 LOC) -- Probability distributions
      normal.rs       -- PDF, logPDF, CDF, SF, logSF, cumhazard, PPF
      exponential.rs  -- PDF, logPDF, CDF, SF, logSF, cumhazard, PPF
      uniform.rs      -- PDF, logPDF, CDF, SF, logSF, PPF
    stats/            (1 LOC) -- Placeholder module
```

---

## 3. Kernel Categories

### Category A: Descriptive Statistics (core slice ops)
Implemented inline in `lib.rs`: mean, var, std, skew, kurtosis, zscore,
percentile, IQR, MAD, trimmed mean, median, and their `_skipna` variants.
Also: diff, pct_change, cumsum, cummean, ECDF, quantile_bins, sign_mask,
demean_with_signs, minmax_scale, robust_scale, winsorize.

### Category B: Rolling Window Statistics
Kahan-compensated sliding window with configurable alignment (trailing/centered),
NaN policy (propagate/ignore), and min_periods. Single-stat and fused multi-stat
kernels. 1D and 2D (axis-0). Rolling cov/corr/beta/linreg with skipna.
EWMA (exponentially weighted moving average).

### Category C: Matrix Kernels
Sample covariance (ddof=1), biased cov, centered cov, skipna cov, Gram matrices
(XTX, XXT), pairwise Euclidean/cosine distance, correlation matrix, correlation
distance. Optional Rayon parallelism via `par_chunks_mut`.
Also: diag, trace, is_symmetric.

### Category D: Robust Estimators
O(n) median via `select_nth_unstable`, fused median+MAD, workspace API for
zero-allocation pipelines. Configurable via LocationPolicy/ScalePolicy enums.
Trimmed std, MAD-based std, biweight midvariance, Qn scale, Huber M-estimator.
Adaptive rolling median.

### Category E: Resampling (Bootstrap & Jackknife)
Deterministic via Pcg64 with golden-ratio seed mixing. Parallel via Rayon.
Bootstrap: mean, CI (percentile/BCa/studentized-t/Bayesian), SE, variance.
Block bootstrap: moving-block, circular-block, stationary.
Permutation tests: correlation, mean difference.
Jackknife: mean, CI, influence, delete-d, jackknife-after-bootstrap.

### Category F: Time-Series Analysis
**Stationarity:** ADF (with critical value tables), KPSS, Phillips-Perron,
variance ratio, Zivot-Andrews, trend stationarity, integration order,
seasonal differencing, seasonal unit root.
**Diagnostics:** Ljung-Box, Box-Pierce, Durbin-Watson, BG test, runs test.
**ACF/PACF:** Raw ACF, Levinson-Durbin PACF, Yule-Walker PACF, ACOVF,
ACF with CI, CCF, PACF-innovations, PACF-Burg.
**Spectral:** FFT periodogram, Welch PSD, Bartlett PSD, cumulative periodogram,
spectral entropy/peaks/flatness/centroid/rolloff, band power, dominant frequency.
**Rolling TSA:** Rolling autocorrelation, rolling cross-correlation.
**WIP:** Engle-Granger cointegration, HP filter, fractional diff, STL decompose.

### Category G: Statistical Inference
One-sample / two-sample / paired t-tests, chi-square (GOF + independence),
ANOVA (one-way F-test), Levene's test, Jarque-Bera, Anderson-Darling,
Pearson/Spearman correlation with p-values, F-test for variance, Bartlett,
Mann-Whitney U, Kolmogorov-Smirnov (1-sample), p-value adjustment
(Bonferroni/BH), proportion z-tests, correlation CI (Fisher z), variance CI
(chi-square), odds ratio, rank biserial, Cliff's delta, eta/omega squared.

### Category H: Probability Distributions
Normal, Exponential, Uniform -- each with PDF, logPDF, CDF, SF, logSF,
cumhazard, PPF (inverse CDF). Normal uses libm::erfc for precision.

---

## 4. Algorithmic Highlights

| Algorithm                      | Complexity | Notes                                    |
|--------------------------------|------------|------------------------------------------|
| Rolling mean/std               | O(n)       | Kahan-compensated, numerically stable    |
| Fused multi-stat rolling       | O(n)       | Single pass for mean+std+var+count+min+max|
| Median                         | O(n) avg   | `select_nth_unstable` (quickselect)      |
| ACF/PACF                       | O(n*k)     | Direct computation (no FFT for ACF)      |
| Periodogram                    | O(n log n) | FFT via rustfft, DFT fallback for n<64   |
| Cov matrix                     | O(n*p^2)   | Upper-triangle + mirror, optional Rayon  |
| Bootstrap                      | O(B*n)     | Pcg64 per-stream, Rayon parallel         |
| Levinson-Durbin PACF           | O(k^2)     | Recursive (not O(k^3) Yule-Walker)       |
| Huber M-estimator              | O(n*iter)  | Iterative, ~10 iterations typical        |
| ADF test                       | O(n)       | OLS regression + critical value table    |
