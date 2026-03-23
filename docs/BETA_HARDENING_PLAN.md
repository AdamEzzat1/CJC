# CJC Beta Hardening — Stack Role Group Implementation Plan

## ROLE

You are a stacked systems team working inside the CJC compiler repository.

You consist of:

1. **Lead Language Architect** — owns language semantics, type system soundness, and feature design
2. **Compiler Pipeline Engineer** — owns Lexer → Parser → AST → HIR → MIR → Exec data flow
3. **Runtime Systems Engineer** — owns memory model, GC/NoGC boundary, dispatch, and builtins
4. **Numerical Computing Engineer** — owns deterministic BLAS, SIMD, accumulator correctness, and AD
5. **Determinism & Reproducibility Auditor** — enforces bit-identical output across runs and platforms
6. **QA Automation Engineer** — owns test infrastructure, parity gates, and regression prevention

Your goal is to harden CJC for its first public beta release.
You must never break determinism, the memory model, or the compiler pipeline.

---

## PRIME DIRECTIVES (unchanged from CLAUDE.md)

1. Do not break the compiler pipeline: Lexer → Parser → AST → [TypeChecker] → HIR → MIR → [Optimize] → Exec
2. Do not introduce hidden allocations or GC usage in NoGC-verified paths
3. Maintain deterministic execution — same seed = bit-identical output
4. Preserve backward compatibility unless explicitly impossible
5. Never silently refactor unrelated systems — scope changes to the feature being implemented
6. Language primitives must stay minimal — higher-level functionality belongs in libraries (Bastion, Vizor)
7. Both executors must agree — every feature must work in `cjc-eval` AND `cjc-mir-exec`

---

## CURRENT STATE (Verified 2025-03-22)

### What's Strong (No Action Needed)
- 4,500+ tests passing, 0 failures (except fuzz harness OOM — see Phase 1)
- 258+ builtins with three-place wiring
- `if` as expression — DONE (parser + both executors)
- Default parameters — DONE (`fn f(x: f64 = 1.0)`)
- Variadic parameters — DONE (`fn f(...args: T)`)
- Decorators — DONE (`@memoize`, `@trace`)
- Snap save/load — DONE (6 builtins: `snap`, `restore`, `snap_save`, `snap_load`, `snap_hash`, `snap_to_json`)
- Multi-head attention building blocks — DONE (`split_heads`, `scaled_dot_product_attention`, `merge_heads`)
- Elman RNN — buildable manually (bench/bench_rnn_latency.cjc proves this)
- Adaptive ODE solver — DONE internally (`ode_solve_rk45` in ode.rs, not exposed)
- Standard library docs — 9 comprehensive guides in `CJC V 0.1/`
- 100+ example programs across examples/, demos/, bench/
- Paged KV cache — DONE (PagedKvCache with full wiring)

---

## PHASE 1: Fix Wiring Gaps (Pre-Beta Blocker)

**Owner: Runtime Systems Engineer + QA Automation Engineer**

These are parity violations — features that work in one executor but crash in the other. They MUST be fixed before any beta release.

### 1.1 Wire `DataFrame.view()` into MIR-exec

**Problem:** `df.view()` works in cjc-eval (line 2729) but is missing from cjc-mir-exec.
8+ tidy fixture files call this method.

**Work:**
- [ ] Read eval implementation at `cjc-eval/src/lib.rs:2729` (the `"view"` match arm)
- [ ] Add identical `"view"` case to DataFrame method dispatch in `cjc-mir-exec/src/lib.rs`
- [ ] Requires: `rebuild_dataframe_from_struct`, `TidyView::from_df`, `tidy_dispatch::wrap_view`
- [ ] Ensure imports match eval (cjc-data types)

**Verification:**
- [ ] Run all tidy fixture files through BOTH executors
- [ ] Add parity test: `test_dataframe_view_parity` (eval output == mir-exec output)

### 1.2 Wire `sample_indices()` into eval

**Problem:** `sample_indices()` works in cjc-mir-exec (line 1436, inline with RNG) but is missing from cjc-eval.
Not in `dispatch_builtin` either — needs RNG state, so must be inline.

**Work:**
- [ ] Read mir-exec implementation at `cjc-mir-exec/src/lib.rs:1436-1466`
- [ ] Add identical inline handling in `cjc-eval/src/lib.rs` call dispatch
- [ ] Must use `self.rng.next_u64()` for default seed (same as mir-exec)
- [ ] Signature: `sample_indices(n: i64, k: i64, [replace: bool], [seed: i64]) → array[i64]`

**Verification:**
- [ ] Add parity test: `test_sample_indices_parity` (eval == mir-exec for same seed)
- [ ] Add determinism test: same seed → identical indices across runs

### 1.3 Fix Fuzz Harness Allocation Bomb

**Problem:** `fuzz_mir_pipeline_no_crash` in `tests/cjc_v0_1_hardening/fuzz/test_fuzz_hardening.rs` feeds random bytes → parser may produce programs that trigger 100GB+ allocation, crashing the test suite.

**Work:**
- [ ] Add input-size cap to fuzz_mir_pipeline_no_crash (e.g., skip inputs > 4KB)
- [ ] OR add a global allocation limit guard in the executor for fuzz mode
- [ ] Same fix for `fuzz_eval_pipeline_no_crash` if affected

**Verification:**
- [ ] Full `cargo test --workspace` passes with zero crashes
- [ ] Fuzz tests run for 10 seconds without OOM

### 1.4 Parity Test Coverage Expansion

**Work:**
- [ ] Add parity tests for all DataFrame/Tidy methods (view, filter, group_by, join)
- [ ] Add parity tests for sample_indices
- [ ] Add parity tests for snap builtins (snap_save → snap_load roundtrip in both executors)

**Exit Criteria:** `cargo test --workspace` passes cleanly (0 failures, no OOM).

---

## PHASE 2: Essential Language Features for Beta

**Owner: Lead Language Architect + Compiler Pipeline Engineer**

These are the syntax/usability gaps that would frustrate a data scientist trying CJC for the first time.

### 2.1 `input()` Builtin + Command-Line Args

**Priority: HIGH** — CJC programs cannot read user input or accept CLI arguments.

**Work:**
- [ ] Add `input()` builtin: reads one line from stdin, returns `String`
- [ ] Add `input(prompt: str)` variant: prints prompt, reads line
- [ ] Add `args()` builtin: returns `array[str]` of command-line arguments
- [ ] Add `getenv(name: str)` builtin: returns environment variable or empty string
- [ ] Wire all three into `dispatch_builtin` (stateless) or inline in both executors
- [ ] Mark `input()` and `args()` as nondeterministic in NoGC verifier (they cause I/O)

**Determinism Note:** `input()` is inherently nondeterministic. Document this. The `--reproducible` flag should warn if `input()` is used.

**Verification:**
- [ ] Unit test: `input()` with mocked stdin
- [ ] Parity test: `args()` returns identical arrays in both executors
- [ ] NoGC test: `input()` inside `nogc {}` block produces compile error

### 2.2 Range Slicing Syntax `arr[1..5]`

**Priority: HIGH** — `array_slice(arr, 1, 5)` exists but `arr[1..5]` does not parse.

**Work:**
- [ ] Add `Range` expression to parser: `start..end` and `start..=end`
- [ ] Add `RangeExpr` to AST (or reuse existing if present)
- [ ] Handle `arr[range]` in index expression lowering → desugar to `array_slice(arr, start, end)`
- [ ] Support `tensor[1..3, 0..2]` for multi-dim slicing
- [ ] HIR/MIR lowering for range index expressions

**Verification:**
- [ ] Parse test: `arr[1..5]` produces correct AST
- [ ] Eval test: `let s = arr[1..5]; assert len(s) == 4`
- [ ] Parity test: both executors agree
- [ ] Tensor test: `tensor[0..2, 1..3]` returns correct sub-tensor

### 2.3 Module System Completion

**Priority: HIGH** — `cjc-module` crate has infrastructure but multi-file requires `--multi-file` flag.

**Work:**
- [ ] Audit `cjc-module/src/lib.rs` — identify what's connected vs stub
- [ ] Enable module resolution by default (remove `--multi-file` gate)
- [ ] Support `import math;` and `import stats.linear;` syntax
- [ ] Deterministic module resolution order (alphabetical, depth-first)
- [ ] Clear compile errors for circular dependencies
- [ ] Wire resolved symbols into both executors

**Verification:**
- [ ] Two-file program: `main.cjc` imports `helper.cjc` → runs correctly
- [ ] Circular dependency → clear error message with file paths
- [ ] Parity test: multi-file program produces identical output in both executors

### 2.4 Dict/Map Literal Syntax

**Priority: MEDIUM** — `Value::Map` exists with `DetMap` backend but no literal syntax.

**Work:**
- [ ] Add map literal syntax to parser: `{ "key": value, "key2": value2 }`
- [ ] Or use typed syntax: `Map { "key": value }` to avoid ambiguity with blocks
- [ ] Lower to `Value::Map(DetMap)` construction
- [ ] Ensure deterministic iteration order (BTreeMap-backed)

**Verification:**
- [ ] Parse test: map literal produces correct AST
- [ ] Eval test: `let m = { "a": 1, "b": 2 }; print(m["a"]);` → `1`
- [ ] Parity test: both executors agree
- [ ] Determinism test: iteration order is always alphabetical

---

## PHASE 3: Numerical Computing Completeness

**Owner: Numerical Computing Engineer + Determinism Auditor**

These fill gaps in CJC's scientific computing story. All implementations MUST use deterministic accumulation.

### 3.1 Numerical Integration (Quadrature)

**Work:**
- [ ] `trapezoid(f, a, b, n)` — composite trapezoidal rule
- [ ] `simpson(f, a, b, n)` — composite Simpson's 1/3 rule
- [ ] `gauss_quad(f, a, b, n)` — Gauss-Legendre quadrature (precomputed nodes/weights for n ≤ 20)
- [ ] All use KahanAccumulator for summation
- [ ] Add to `crates/cjc-runtime/src/` (new file `integrate.rs` or extend `optimize.rs`)
- [ ] Wire as builtins in both executors

**Verification:**
- [ ] `trapezoid(sin, 0, pi, 1000)` ≈ 2.0 (within 1e-6)
- [ ] `simpson(|x| x*x, 0, 1, 100)` ≈ 0.333333 (within 1e-10)
- [ ] Determinism: identical results across runs
- [ ] Parity: eval == mir-exec

### 3.2 Numerical Differentiation

**Work:**
- [ ] `diff_forward(f, x, h)` — forward difference: (f(x+h) - f(x)) / h
- [ ] `diff_central(f, x, h)` — central difference: (f(x+h) - f(x-h)) / 2h
- [ ] `diff_second(f, x, h)` — second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
- [ ] `gradient(f, x_vec, h)` — numerical gradient vector
- [ ] Default h = 1e-8 (or sqrt(eps) scaled)

**Verification:**
- [ ] `diff_central(sin, 0, 1e-8)` ≈ 1.0 (cos(0))
- [ ] `gradient(|v| v[0]^2 + v[1]^2, [3, 4], 1e-8)` ≈ [6, 8]
- [ ] Determinism + parity tests

### 3.3 Expose Adaptive ODE Solver as Builtin

**Note:** `ode_solve_rk45` already exists in `crates/cjc-runtime/src/ode.rs`. Just needs wiring.

**Work:**
- [ ] Add `ode_solve(f, y0, t_span, tol)` builtin wrapping `ode_solve_rk45`
- [ ] Wire into both executors
- [ ] Document: f is a CJC closure `fn(t: f64, y: Tensor) -> Tensor`

**Verification:**
- [ ] Exponential decay: dy/dt = -y, y(0) = 1 → y(1) ≈ e^(-1)
- [ ] Parity + determinism tests

### 3.4 Constrained Optimization Stubs

**Work:**
- [ ] `minimize_penalty(f, g_constraints, x0, penalty_weight)` — quadratic penalty method
- [ ] `minimize_augmented_lagrangian(f, g_eq, h_ineq, x0)` — augmented Lagrangian
- [ ] Both use existing `minimize_lbfgs` or `minimize_bfgs` as inner solver
- [ ] Deterministic convergence (fixed iteration order, Kahan accumulation)

**Verification:**
- [ ] Minimize x² subject to x ≥ 1 → x* = 1
- [ ] Parity + determinism tests

---

## PHASE 4: ML/DL Expansion

**Owner: Numerical Computing Engineer + Runtime Systems Engineer**

### 4.1 LSTM/GRU Cell Primitives

**Context:** Elman RNN is already buildable manually. LSTM/GRU need gate-level primitives.

**Work:**
- [ ] `lstm_cell(x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh)` → `(h_new, c_new)`
  - Implements: i/f/g/o gates with sigmoid/tanh activations
  - Returns tuple of (hidden_state, cell_state)
- [ ] `gru_cell(x, h_prev, W_ih, W_hh, b_ih, b_hh)` → `h_new`
  - Implements: r/z gates with sigmoid/tanh
- [ ] Wire as builtins in both executors
- [ ] All matrix ops use existing `matmul` + `sigmoid` + `tanh_activation`

**Verification:**
- [ ] Forward pass produces finite values
- [ ] Sequence of 10 steps doesn't explode or vanish (gradient health check)
- [ ] Determinism: same seed → identical hidden states
- [ ] Parity: eval == mir-exec

### 4.2 Multi-Head Attention Convenience Wrapper

**Context:** `split_heads`, `scaled_dot_product_attention`, `merge_heads` exist. Need a single-call wrapper.

**Work:**
- [ ] `multi_head_attention(Q, K, V, num_heads, W_q, W_k, W_v, W_o)` → `Tensor`
  - Linear projections → split_heads → SDPA → merge_heads → output projection
- [ ] Optional: causal mask parameter for autoregressive decoding
- [ ] Wire as builtin in both executors

**Verification:**
- [ ] Output shape matches expected `[batch, seq, model_dim]`
- [ ] Parity + determinism tests
- [ ] Matches manual split_heads → SDPA → merge_heads pipeline

### 4.3 ARIMA for Time Series

**Context:** ACF, PACF, EWMA, seasonal decomposition exist. No ARIMA model.

**Work:**
- [ ] `arima_fit(data, p, d, q)` → model struct (AR coefficients, MA coefficients, sigma²)
  - Differencing (d) → fit AR(p) via Yule-Walker or OLS → fit MA(q) residuals
- [ ] `arima_forecast(model, steps)` → array of forecasted values
- [ ] `arima_residuals(model, data)` → array of residuals
- [ ] All use deterministic accumulation (Kahan/Binned)

**Verification:**
- [ ] AR(1) on synthetic data recovers known coefficient
- [ ] Forecast of constant series = constant
- [ ] Determinism + parity tests

---

## PHASE 5: Polish for Publication

**Owner: QA Automation Engineer + Lead Language Architect**

### 5.1 E3xxx Borrow/Ownership Error Codes

**Context:** Error code range E3xxx is reserved in `cjc-diag/src/error_codes.rs` but has 0 codes defined.

**Work:**
- [ ] Define E3001–E3010 for common ownership/borrow scenarios
- [ ] Wire into diagnostics where currently generic E2xxx codes are used
- [ ] Ensure error messages include fix suggestions

### 5.2 Tutorial Progression

**Context:** 100+ examples exist, 9 reference guides exist. Missing: structured learning path.

**Work:**
- [ ] Create `docs/GETTING_STARTED.md` — install, first program, REPL basics
- [ ] Create `docs/TUTORIAL.md` — progressive 10-lesson tutorial:
  1. Variables, types, printing
  2. Functions, closures
  3. Control flow (if/match/for/while)
  4. Arrays, tensors, basic math
  5. Statistics and distributions
  6. DataFrames and tidy operations
  7. Linear algebra and optimization
  8. Automatic differentiation
  9. ML training loop (Chess RL walkthrough)
  10. Determinism guarantees and snap serialization

### 5.3 Package Manager / Dependency Resolution Design

**Work:**
- [ ] Write ADR (Architecture Decision Record) for package system design
- [ ] Key questions: registry vs git-based? Lockfile format? Version resolution strategy?
- [ ] Design `cjc.toml` manifest format
- [ ] Do NOT implement — design document only for community feedback

---

## IMPLEMENTATION ORDER

```
Phase 1 (BLOCKER)     →  Phase 2 (Beta UX)    →  Phase 3 (Numerics)  →  Phase 4 (ML)  →  Phase 5 (Polish)
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────┐    ┌────────────────┐
│ 1.1 view() wire │    │ 2.1 input/args   │    │ 3.1 Quadrature  │    │ 4.1 LSTM   │    │ 5.1 E3xxx      │
│ 1.2 sample_idx  │    │ 2.2 arr[1..5]    │    │ 3.2 Num. diff   │    │ 4.2 MHA    │    │ 5.2 Tutorial   │
│ 1.3 Fuzz fix    │    │ 2.3 Modules      │    │ 3.3 ODE expose  │    │ 4.3 ARIMA  │    │ 5.3 Pkg design │
│ 1.4 Parity tests│    │ 2.4 Map literals  │    │ 3.4 Constrained │    │            │    │                │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └────────────┘    └────────────────┘
     ~1 day                 ~3-5 days               ~3-4 days           ~2-3 days          ~2-3 days
```

---

## VERIFICATION PROTOCOL (Every Phase)

After completing each phase, run this verification loop:

```bash
# 1. Full test suite
cargo test --workspace 2>&1 | grep -E "test result:|FAILED|failures"

# 2. Parity gate (eval == mir-exec for all new features)
cargo test parity 2>&1 | grep -E "test result:|FAILED"

# 3. Determinism gate (same seed → identical output, 3 runs)
cargo test determinism 2>&1 | grep -E "test result:|FAILED"

# 4. NoGC verification (new builtins don't break NoGC paths)
cargo test nogc 2>&1 | grep -E "test result:|FAILED"

# 5. Check for unwired builtins (grep for function name in all 3 locations)
# For each new builtin X:
grep -r "\"X\"" crates/cjc-runtime/src/builtins.rs crates/cjc-eval/src/lib.rs crates/cjc-mir-exec/src/lib.rs
```

**Exit criteria per phase:** ALL tests pass, ALL parity gates green, ALL determinism checks pass.

---

## ITEMS REMOVED FROM PLAN (Already Implemented)

| Item | Status | Evidence |
|------|--------|----------|
| `if` as expression | DONE | Parser line 1410, eval line 848, mir-exec line 514 |
| Default parameters | DONE | `fn f(x: f64 = 1.0)` works in both executors |
| Variadic parameters | DONE | `fn f(...args: T)` fully supported |
| Decorators | DONE | `@memoize`, `@trace` in both executors |
| Snap save/load | DONE | 6 builtins: snap, restore, snap_save, snap_load, snap_hash, snap_to_json |
| Multi-head attention blocks | DONE | split_heads, SDPA, merge_heads wired in both executors |
| Adaptive ODE (internal) | DONE | ode_solve_rk45 in ode.rs (Phase 3.3 exposes it) |
| Standard library docs | DONE | 9 guides in CJC V 0.1/ |
| Example programs | DONE | 100+ across examples/, demos/, bench/ |
| Paged KV cache | DONE | PagedKvCache with full wiring |
