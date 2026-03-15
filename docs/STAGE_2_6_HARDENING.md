# Stage 2.6: Determinism Hardening & Infrastructure

This document covers all changes made in the Stage 2.6 hardening pass, which
focused on fixing determinism issues, improving code hygiene, and adding
infrastructure for CI, benchmarking, and test tracking.

---

## 1. Linalg Determinism Fixes

### Cholesky Decomposition (CRITICAL FIX)

**File:** `crates/cjc-runtime/src/linalg.rs`

The Cholesky decomposition (`Tensor::cholesky()`) was using naive summation
(`sum += l[j*n+k] * l[j*n+k]`) for the inner loop accumulations. This is a
determinism violation -- with large matrices and adversarial floating-point
data, results could differ across compilation flags, optimization levels, or
platforms due to different evaluation order of the floating-point additions.

**Fix:** Replaced all naive `sum +=` loops with `BinnedAccumulatorF64`, which
provides order-invariant, deterministic summation:

```rust
// Before (non-deterministic):
let mut sum = 0.0;
for k in 0..j {
    sum += l[j * n + k] * l[j * n + k];
}

// After (deterministic):
let mut acc = BinnedAccumulatorF64::new();
for k in 0..j {
    acc.add(l[j * n + k] * l[j * n + k]);
}
let sum = acc.finalize();
```

### Least Squares (lstsq) Q^T*b Dot Product

The `lstsq()` function was using naive accumulation for the Q^T * b
matrix-vector product. Fixed to use `BinnedAccumulatorF64`.

### Schur Decomposition Householder Reflections

The Hessenberg reduction in `schur()` was using naive dot products and norms
for Householder reflector computation. All six accumulation loops have been
upgraded to `BinnedAccumulatorF64`.

### LU Decomposition Determinism Contract

Added documentation to `lu_decompose()` specifying the pivot tie-breaking
contract: when two candidates have identical absolute values, the first
(lowest row index) is chosen. This is deterministic given identical input bits.

---

## 2. HashMap to BTreeMap Migration

**Files changed:**
- `crates/cjc-mir/src/escape.rs`
- `crates/cjc-mir/src/nogc_verify.rs`
- `crates/cjc-mir/src/optimize.rs`
- `crates/cjc-mir/src/ssa_optimize.rs`
- `crates/cjc-types/src/effect_registry.rs`
- `crates/cjc-analyzer/src/server.rs`

All `HashMap`/`HashSet` usage in compiler crates has been replaced with
`BTreeMap`/`BTreeSet`. While most of these were in compiler internals (not
runtime paths), the optimizer and SSA passes iterate over their maps, and
non-deterministic iteration order could theoretically cause different
optimizations to fire in different orders.

After this change, **zero** `HashMap`/`HashSet` imports remain in any crate
under `crates/`. The only remaining mentions are in documentation comments
in `cjc-snap` and `cjc-runtime/stats.rs` (which document that HashMap is
explicitly avoided).

---

## 3. GitHub Actions CI Pipeline

**File:** `.github/workflows/ci.yml`

New CI pipeline with five jobs:

| Job | Purpose |
|-----|---------|
| `check` | `cargo check` + clippy on Ubuntu and Windows |
| `test` | Full `cargo test --workspace` on Ubuntu and Windows |
| `determinism-gate` | Runs parity and determinism tests specifically |
| `fuzz-smoke` | Runs bolero fuzz targets in proptest mode |
| `proptest` | Runs property-based tests |

The pipeline runs on push to master/main and on all pull requests.
Cross-platform testing (Ubuntu + Windows) ensures the determinism
contract holds across operating systems.

---

## 4. Ignored Test Audit

All 16+ `#[ignore]` tests were audited and categorized:

| Category | Count | Reason | Action |
|----------|-------|--------|--------|
| Vizor snapshot generators | 11 | Manual visual review | Added tracking comment |
| Performance benchmarks | 6 | Run-on-demand timing tests | Added tracking comment |
| Memory model perf | 1 | Performance harness | Added tracking comment |
| Tidy perf | 1 | Fixture timing | Added tracking comment |
| Chess RL debug | 3 | Known parser limitation (`;` after `while {}`) | Already documented |

All ignored tests now have inline comments explaining:
1. Why they are ignored
2. How to run them manually

---

## 5. Benchmark Regression Suite

**File:** `tests/bench_regression.rs`

New performance regression smoke tests (all `#[ignore]`, run on demand):

| Test | What it measures | Budget |
|------|-----------------|--------|
| `bench_matmul_64x64_regression` | Tensor matmul throughput | 5s for 10 runs |
| `bench_parse_eval_regression` | Lexer + parser + eval pipeline | 10s for 50 runs |
| `bench_cholesky_regression` | Cholesky decomposition | 5s for 100 runs |
| `bench_kahan_sum_regression` | Kahan accumulator throughput | 5s for 10M values |
| `bench_mir_exec_regression` | MIR pipeline (fib(20)) | 30s for 5 runs |

These use generous budgets (10x expected) to catch order-of-magnitude
regressions without flaking on slow CI runners.

---

## 6. Existing Infrastructure Verified

The following were verified as already present and correctly wired:

- **Bolero fuzzing:** 5 targets in `tests/bolero_fuzz/mod.rs` (lexer, parser,
  MIR pipeline, complex determinism, optimizer parity)
- **clip_grad() builtin:** Already implemented in `builtins.rs` with 3-arg
  (value, min, max) signature
- **Benchmark crates:** `bench/ad_bench` and `bench/nlp_preprocess_bench`
  already exist as standalone benchmark programs

---

## Test Results

After all changes:

```
Total tests:   ~2,500+
Failures:      0
Ignored:       16+ (all audited and documented)
```

All existing determinism, parity, and functional tests pass unchanged.
