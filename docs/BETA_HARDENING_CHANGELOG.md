# CJC Beta Hardening — Change Log

**Date:** 2025-03-22
**Scope:** All 5 phases of BETA_HARDENING_PLAN.md executed

---

## Regression Fix: MirExprKind::Col DExpr parity

**Root cause:** Adding TidyView dispatch to mir-exec (Phase 1.1) exposed a latent bug where
`MirExprKind::Col(name)` produced `Value::String("col:{name}")` instead of the proper
`Value::Struct { name: "DExpr", kind: "col", value: name }` that eval produces.

**Fix:** Changed `crates/cjc-mir-exec/src/lib.rs` line 466 to produce a proper DExpr struct,
matching the eval behavior exactly.

**Tests restored:** 5 tidy_tests (prop_filter_idempotent, prop_mutate_preserves_nrows_adds_col,
tidy_filter_select, tidy_empty_df, tidy_pipeline)

---

## Phase 1: Wiring Gap Fixes (Pre-Beta Blocker)

### 1.1 DataFrame.view() wired into MIR-exec
**Files modified:**
- `crates/cjc-mir-exec/src/lib.rs`
  - Added `use cjc_data::{..., TidyView}` and `use cjc_data::tidy_dispatch` imports
  - Added `"view"` arm to DataFrame method dispatch (creates TidyView from struct)
  - Added TidyView method dispatch block (delegates to `tidy_dispatch::dispatch_tidy_method`)
  - Added GroupedTidyView method dispatch block (delegates to `tidy_dispatch::dispatch_grouped_method`)
  - Added `rebuild_dataframe_from_struct()` helper function (mirrors eval's version)
  - Added `value_array_to_column()` helper function (supports Float, Str, Bool columns)

**Impact:** `df.view()` now works in both executors. TidyView verbs (filter, select, arrange, group_by, etc.) now dispatched in MIR-exec.

### 1.2 sample_indices() wired into eval
**Files modified:**
- `crates/cjc-eval/src/lib.rs`
  - Added inline `"sample_indices"` handler after `categorical_sample`
  - Signature: `sample_indices(n, k, [replace], [seed])` → `array[i64]`
  - Uses interpreter RNG (`self.rng.next_u64()`) when no explicit seed provided

**Impact:** `sample_indices()` now works in both executors with identical RNG behavior.

### 1.3 Fuzz harness allocation cap
**Files modified:**
- `tests/cjc_v0_1_hardening/fuzz/test_fuzz_hardening.rs`
  - Added `if input.len() > 4096 { return; }` guard to `fuzz_mir_pipeline_no_crash`
  - Added same guard to `fuzz_eval_pipeline_no_crash`

**Impact:** Prevents 100GB+ allocation bomb from random fuzz inputs.

### 1.4 Parity tests
**Files created:**
- `tests/beta_hardening/test_phase1_parity.rs` — 7 tests
  - DataFrame.view() creates TidyView (Rust API)
  - sample_indices parity (eval == mir-exec)
  - sample_indices with replacement parity
  - sample_indices determinism (3 runs identical)
  - sample_indices implicit RNG parity
  - Snap roundtrip parity

---

## Phase 2: Essential Language Features

### 2.1 args()/getenv() builtins
**Files modified:**
- `crates/cjc-runtime/src/builtins.rs` — Added `getenv` to dispatch_builtin
- `crates/cjc-eval/src/lib.rs` — Added inline `args()` handler
- `crates/cjc-mir-exec/src/lib.rs` — Added inline `args()` handler + known builtin entries

### 2.2 array_slice verified
- `array_slice(arr, start, end)` already wired via dispatch_builtin fallthrough
- Confirmed working in both executors via parity tests

### 2.3 Module system
- Deferred to post-beta (too invasive for this phase)
- Current state documented: `cjc-module` crate has infrastructure, `--multi-file` flag works

### 2.4 Map builtins
**Files modified:**
- `crates/cjc-runtime/src/builtins.rs` — Added to dispatch_builtin:
  - `map_new()` → empty Map
  - `map_set(m, k, v)` → new Map with key set (COW semantics)
  - `map_get(m, k)` → value or Void
  - `map_keys(m)` → array of keys
  - `map_values(m)` → array of values
  - `map_contains(m, k)` → bool

**Files created:**
- `tests/beta_hardening/test_phase2_features.rs` — 15 tests

---

## Phase 3: Numerical Computing

### 3.1 Numerical Integration
**Files created:**
- `crates/cjc-runtime/src/integrate.rs`
  - `trapezoid(xs, ys)` — composite trapezoidal rule with Kahan summation
  - `simpson(xs, ys)` — composite Simpson's 1/3 rule with Kahan summation
  - `cumtrapz(xs, ys)` — cumulative trapezoidal integration

**Files modified:**
- `crates/cjc-runtime/src/lib.rs` — Added `pub mod integrate;`
- `crates/cjc-runtime/src/builtins.rs` — Wired `trapz`, `simps`, `cumtrapz`

### 3.2 Numerical Differentiation
**Added to** `crates/cjc-runtime/src/integrate.rs`:
- `diff_central(xs, ys)` — central difference derivative
- `diff_forward(xs, ys)` — forward difference derivative
- `gradient_1d(ys, dx)` — numerical gradient with uniform spacing

**Wired as builtins:** `diff_central`, `diff_forward`, `gradient_1d`

### 3.3 Adaptive ODE exposure
- `ode_step_euler` and `ode_step_rk4` already wired as builtins (verified)
- `ode_solve_rk45` remains internal (requires closure argument)

### 3.4 Constrained Optimization
**Added to** `crates/cjc-runtime/src/optimize.rs`:
- `penalty_objective(f_val, violations, penalty)` — quadratic penalty
- `project_box(x, lower, upper)` — box constraint projection
- `projected_gd_step(x, grad, lr, lower, upper)` — projected gradient descent

**Wired as builtins:** `penalty_objective`, `project_box`, `projected_gd_step`

**Files created:**
- `tests/beta_hardening/test_phase3_numerics.rs` — 10 tests

---

## Phase 4: ML/DL Expansion

### 4.1 LSTM/GRU Cell Primitives
**Added to** `crates/cjc-runtime/src/ml.rs`:
- `lstm_cell(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh)` → `(h_new, c_new)`
  - Full LSTM with i/f/g/o gates, sigmoid/tanh activations
- `gru_cell(x, h_prev, w_ih, w_hh, b_ih, b_hh)` → `h_new`
  - Full GRU with r/z/n gates

**Wired as builtins:** `lstm_cell`, `gru_cell`

### 4.2 Multi-Head Attention Wrapper
**Added to** `crates/cjc-runtime/src/ml.rs`:
- `multi_head_attention(q, k, v, w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o, num_heads)` → Tensor
  - Linear projections → split_heads → SDPA → merge_heads → output projection

**Wired as builtin:** `multi_head_attention`

### 4.3 ARIMA for Time Series
**Added to** `crates/cjc-runtime/src/timeseries.rs`:
- `arima_diff(data, d)` — d-th order differencing
- `ar_fit(data, p)` — AR(p) fitting via Yule-Walker + Levinson-Durbin
- `ar_forecast(coeffs, history, steps)` — AR forecasting

**Wired as builtins:** `arima_diff`, `ar_fit`, `ar_forecast`

**Files created:**
- `tests/beta_hardening/test_phase4_ml.rs` — 15 tests

---

## Phase 5: Polish

### 5.1 E3xxx Error Codes
**Files modified:**
- `crates/cjc-diag/src/error_codes.rs`
  - Added E3001–E3008 (borrow/ownership error codes)
  - Reserved for future borrow checker implementation

### 5.2 Tutorial Progression
**Files created:**
- `docs/GETTING_STARTED.md` — Installation, first program, REPL guide
- `docs/TUTORIAL.md` — 10-lesson progressive tutorial

### 5.3 Package Manager Design ADR
**Files created:**
- `docs/adr/ADR-0013-package-manager.md` — Architecture Decision Record

---

## Test Summary

| Phase | Tests Added | All Passing |
|-------|------------|-------------|
| Phase 1 | 7 | Yes |
| Phase 2 | 15 | Yes |
| Phase 3 | 10 | Yes |
| Phase 4 | 15 | Yes |
| Phase 5 | 0 (docs only) | N/A |
| **Total** | **49** (new) | **Yes** |

**Test file location:** `tests/beta_hardening/`
- `test_phase1_parity.rs` — Wiring gap fixes
- `test_phase2_features.rs` — Language features
- `test_phase3_numerics.rs` — Numerical computing
- `test_phase4_ml.rs` — ML/DL expansion

**Entry point:** `tests/test_beta_hardening.rs`

---

## New Builtins Added

| Builtin | Category | Stateless | Both Executors |
|---------|----------|-----------|----------------|
| `args()` | I/O | No (env) | Yes |
| `getenv(name)` | I/O | Yes | Yes |
| `map_new()` | Data | Yes | Yes |
| `map_set(m, k, v)` | Data | Yes | Yes |
| `map_get(m, k)` | Data | Yes | Yes |
| `map_keys(m)` | Data | Yes | Yes |
| `map_values(m)` | Data | Yes | Yes |
| `map_contains(m, k)` | Data | Yes | Yes |
| `trapz(xs, ys)` | Numerics | Yes | Yes |
| `simps(xs, ys)` | Numerics | Yes | Yes |
| `cumtrapz(xs, ys)` | Numerics | Yes | Yes |
| `diff_central(xs, ys)` | Numerics | Yes | Yes |
| `diff_forward(xs, ys)` | Numerics | Yes | Yes |
| `gradient_1d(ys, dx)` | Numerics | Yes | Yes |
| `penalty_objective(f, g, p)` | Optimization | Yes | Yes |
| `project_box(x, lo, hi)` | Optimization | Yes | Yes |
| `projected_gd_step(...)` | Optimization | Yes | Yes |
| `lstm_cell(...)` | ML | Yes | Yes |
| `gru_cell(...)` | ML | Yes | Yes |
| `multi_head_attention(...)` | ML | Yes | Yes |
| `arima_diff(data, d)` | Time Series | Yes | Yes |
| `ar_fit(data, p)` | Time Series | Yes | Yes |
| `ar_forecast(c, h, n)` | Time Series | Yes | Yes |
