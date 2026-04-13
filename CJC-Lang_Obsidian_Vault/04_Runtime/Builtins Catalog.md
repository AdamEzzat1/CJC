---
title: Builtins Catalog
tags: [runtime, builtins, reference]
status: Implemented
---

# Builtins Catalog

**Source of truth**: `crates/cjc-runtime/src/builtins.rs` (~4,407 lines). This single file registers every builtin that both [[cjc-eval]] and [[cjc-mir-exec]] can call.

## Headline counts (verified 2026-04-09)

Counted by dispatch-arm pattern `^\s*"name"\s*=>` in each source file:

| Surface | File | Arms | Unique names |
|---|---|---|---|
| Core runtime builtins | `crates/cjc-runtime/src/builtins.rs` | **363** | 363 |
| Quantum builtins (separate dispatch) | `crates/cjc-quantum/src/dispatch.rs` via `dispatch_quantum()` | **83** | 78 (5 aliases) |
| **Total exposed to user programs** |  | **446** | **441** |

Routing arms in `cjc-eval/src/lib.rs` and `cjc-mir-exec/src/lib.rs` (66 each) are **not** separate builtins — they are the executor side of the [[Wiring Pattern]] that forwards into `cjc-runtime/src/builtins.rs`. Do not add them to the total.

The README's "221+" number is pre-v0.1.2 and should be updated. Earlier internal surveys giving 334/336/419 were undercounts because they used over-narrow grep patterns that missed several dispatch arms (string interpolation helpers, datetime builtins, and some COW-array helpers).

## Categories

From README and survey:

| Category | Count | Examples |
|---|---|---|
| Mathematics | 19+ | `sqrt`, `log`, `exp`, `sin`, `cos`, `tan`, `asin`, `atan2`, `abs`, `clamp`, `hypot`, `pow` |
| Statistics | 35+ | `mean`, `sd`, `variance`, `median`, `quantile`, `rank`, `cor`, `cov`, `z_score`, `min`, `max` |
| Distributions | 24 | `normal_pdf`, `normal_cdf`, `normal_ppf`, `t_cdf`, `chi2_cdf`, `beta_pdf`, `gamma_cdf`, `f_cdf`, `binomial_pmf`, `weibull_pdf`, ... |
| Hypothesis tests | 24 | `t_test`, `t_test_two_sample`, `t_test_paired`, `wilcoxon_signed_rank`, `anova_oneway`, `chi_squared_test`, `adf_test`, `tukey_hsd` |
| Linear algebra | 9+ | `matmul`, `dot`, `cross`, `norm`, `det`, `solve`, `lstsq`, `eigh`, `svd`, `qr`, `cholesky`, `schur` |
| ML primitives | 40+ | `relu`, `gelu`, `sigmoid`, `softmax`, `tanh`, `attention`, `conv2d`, `conv1d`, `batch_norm`, `dropout_mask`, `embedding`, `Adam.new`, `binary_cross_entropy` |
| Signal processing | 14+ | `fft`, `rfft`, `ifft`, `psd`, window functions, `cumsum`, `diff`, `trapz`, `simps` |
| Data wrangling | 73+ | `filter`, `group_by`, `join`, `pivot_longer`, `pivot_wider`, `window_sum`, `select`, `mutate`, `arrange` (see [[DataFrame DSL]]) |
| Tensor ops | many | `reshape`, `slice`, `broadcast`, `einsum`, `sparse_matmul`, `transpose` |
| Strings | some | `str_upper`, `str_lower`, `str_split`, `str_join`, `str_contains` |
| Arrays | some | `sort`, `push`, `flatten`, `argmax`, `argsort` |
| I/O | some | `file_read`, `file_write`, `json_parse`, `datetime_from_parts` |
| Regex | several | via `~=`, `!~`, `find`, `find_all`, `split` (see [[Regex Engine]]) |

## The Wiring Rule

Every builtin here is registered in three places per [[Wiring Pattern]]:
1. `cjc-runtime/src/builtins.rs` — the implementation
2. `cjc-eval` — the AST interpreter's call site
3. `cjc-mir-exec` — the MIR executor's call site

Forgetting one breaks [[Parity Gates]].

## Sub-topic notes

- [[Linear Algebra]]
- [[Statistics and Distributions]]
- [[Hypothesis Tests]]
- [[ML Primitives]]
- [[Signal Processing]]
- [[Regex Engine]]

## Related

- [[Dispatch Layer]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Wiring Pattern]]
- [[Parity Gates]]
