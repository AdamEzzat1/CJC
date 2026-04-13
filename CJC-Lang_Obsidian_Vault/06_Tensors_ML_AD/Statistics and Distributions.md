---
title: Statistics and Distributions
tags: [runtime, stats]
status: Implemented
---

# Statistics and Distributions

**Source**: `crates/cjc-runtime/src/stats.rs` (~55K), `distributions.rs` (~57K).

## Summary

A self-contained statistical library: descriptive statistics, distributions (PDF/CDF/PPF), random sampling, and quantile estimation. All numerically stable (see [[Numerical Truth]]) and seeded ([[SplitMix64]]).

## Descriptive statistics (~35+ functions)

- Central tendency: `mean`, `median`, `mode`
- Dispersion: `sd`, `variance`, `mad`, `iqr`
- Shape: `skewness`, `kurtosis`
- Extremes: `min`, `max`, `quantile`
- Rank: `rank`, `argsort`
- Correlation: `cor`, `cov`, `spearman`, `kendall_tau`
- Standardization: `z_score`, `scale`

All sums are Kahan-stable or binned; all sort-based operations use `total_cmp`.

## Distributions (24)

Each distribution exposes at least `pdf`, `cdf`, and `ppf` (inverse CDF / quantile). Sampling uses [[SplitMix64]].

| Family | Examples |
|---|---|
| Normal | `normal_pdf`, `normal_cdf`, `normal_ppf` |
| Student-t | `t_pdf`, `t_cdf`, `t_ppf` |
| Chi-squared | `chi2_pdf`, `chi2_cdf`, `chi2_ppf` |
| F | `f_pdf`, `f_cdf`, `f_ppf` |
| Beta, Gamma | `beta_*`, `gamma_*` |
| Discrete | `binomial_pmf`, `poisson_pmf` |
| Extremes | `weibull_*`, `exponential_*`, `pareto_*` |
| Multivariate | Normal, Dirichlet (**Needs verification**) |

## Bastion library

Beyond the runtime, `docs/bastion/` describes [[Bastion]] — a pure-CJC statistics library that implements ~55 higher-level functions on top of ~15 runtime primitives. The philosophy is documented in `docs/bastion/CLASSIFICATION.md` and `BASTION_PRIMITIVE_ABI.md`.

## Related

- [[Numerical Truth]]
- [[Kahan Summation]]
- [[SplitMix64]]
- [[Hypothesis Tests]]
- [[Bastion]]
- [[DataFrame DSL]]
