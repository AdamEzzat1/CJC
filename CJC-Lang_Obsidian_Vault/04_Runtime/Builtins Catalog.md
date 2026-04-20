---
title: Builtins Catalog
tags: [runtime, builtins, reference]
status: Implemented
---

# Builtins Catalog

**Source of truth**: `crates/cjc-runtime/src/builtins.rs` (~4,407 lines). This single file registers every builtin that both [[cjc-eval]] and [[cjc-mir-exec]] can call.

## Headline counts (updated 2026-04-19)

Counted by dispatch-arm pattern `^\s*"name"\s*=>` in each source file:

| Surface | File | Arms | Unique names |
|---|---|---|---|
| Core runtime builtins | `crates/cjc-runtime/src/builtins.rs` | **395** | 395 |
| Quantum builtins (separate dispatch) | `crates/cjc-quantum/src/dispatch.rs` via `dispatch_quantum()` | **83** | 78 (5 aliases) |
| **Total exposed to user programs** |  | **478** | **473** |

**Added 2026-04-18** (regex engine upgrade v0.1.5): `regex_or`, `regex_seq`, `regex_explain` (+3 arms).
**Added 2026-04-18** (capture groups v0.1.6): `regex_captures`, `regex_named_capture`, `regex_capture_count` (+3 arms).
**Added 2026-04-19** (v0.1.7 data-science surface expansion, +23 arms):

- DataFrame (+10, in `crates/cjc-data/src/tidy_dispatch.rs`): `df_read_csv`, `pivot_wider`, `pivot_longer`, `df_distinct`, `df_rename`, `df_anti_join`, `df_semi_join`, `df_full_join`, `df_fill_na`, `df_drop_na`. See [[DataFrame DSL]].
- Datetime (+10, in `crates/cjc-runtime/src/datetime.rs`): `parse_date`, `date_format`, `year`, `month`, `day`, `hour`, `minute`, `second`, `date_diff`, `date_add`, `now`. Timestamps are epoch milliseconds (i64). See [[Date Time Surface]].
- Regularized regression (+3, in `crates/cjc-runtime/src/builtins.rs`): `ridge_regression`, `lasso_regression`, `elastic_net`. See [[Regularized Regression]].
- NA handling (+0 net — the `is_na`, `fill_na`/`fillna`, `drop_na`, `is_not_null`, `coalesce` arms already existed; v0.1.7 promotes `NA` to a lexer keyword producing `Value::Na`). See [[NA Handling]].
- f-strings (+0 — syntax, not a builtin): `f"{name}"` now lexes and lowers through HIR. See [[Format Strings]].

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
| Data wrangling | 83+ | `filter`, `group_by`, `join`, `pivot_longer`, `pivot_wider`, `window_sum`, `select`, `mutate`, `arrange`, `df_read_csv`, `df_distinct`, `df_rename`, `df_anti_join`, `df_semi_join`, `df_full_join`, `df_fill_na`, `df_drop_na` (see [[DataFrame DSL]]) |
| Regularized regression | 3 | `ridge_regression`, `lasso_regression`, `elastic_net` (see [[Regularized Regression]]) |
| Date / time | 10+ | `parse_date`, `date_format`, `year`, `month`, `day`, `hour`, `minute`, `second`, `date_diff`, `date_add`, `now` (see [[Date Time Surface]]) |
| NA handling | 5 | `is_na`, `fill_na`/`fillna`, `drop_na`, `is_not_null`, `coalesce` (see [[NA Handling]]) |
| Tensor ops | many | `reshape`, `slice`, `broadcast`, `einsum`, `sparse_matmul`, `transpose` |
| Strings | some | `str_upper`, `str_lower`, `str_split`, `str_join`, `str_contains` |
| Arrays | some | `sort`, `push`, `flatten`, `argmax`, `argsort` |
| I/O | some | `file_read`, `file_write`, `json_parse`, `datetime_from_parts` |
| Regex | several | via `~=`, `!~`, `find`, `find_all`, `split`; composition: `regex_or`, `regex_seq`, `regex_explain`; captures: `regex_captures`, `regex_named_capture`, `regex_capture_count` (see [[Regex Engine]]) |

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
- [[Regularized Regression]]
- [[Date Time Surface]]
- [[NA Handling]]

## Related

- [[Dispatch Layer]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Wiring Pattern]]
- [[Parity Gates]]
