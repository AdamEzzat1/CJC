---
title: Hypothesis Tests
tags: [runtime, stats]
status: Implemented
---

# Hypothesis Tests

**Source**: `crates/cjc-runtime/src/hypothesis.rs` (~50K).

## Summary

24 classical hypothesis tests implemented as builtin functions. Output includes test statistic, p-value, and (where applicable) effect size or confidence interval.

## Tests available

- **t-tests**: `t_test` (one-sample), `t_test_two_sample` (independent), `t_test_paired`
- **Non-parametric**: `wilcoxon_signed_rank`, `mann_whitney_u` (**Needs verification**)
- **ANOVA**: `anova_oneway`, `tukey_hsd` (post-hoc)
- **Goodness of fit**: `chi_squared_test`, `kolmogorov_smirnov` (**Needs verification**)
- **Time series**: `adf_test` (Augmented Dickey-Fuller)
- **Correlation**: Pearson, Spearman — feed into `cor` from stats

## Determinism

All p-values depend only on input data and distribution CDFs from [[Statistics and Distributions]] — no randomness, no hidden state, no platform-dependent libm. Bit-identical across runs.

## Related

- [[Statistics and Distributions]]
- [[DataFrame DSL]]
- [[Numerical Truth]]
- [[Bastion]]
