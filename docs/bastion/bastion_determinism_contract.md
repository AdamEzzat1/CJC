# Bastion Determinism Contract

**Version:** 1.0
**Date:** 2026-03-08

---

## Invariant

**Identical input + identical seed = identical output bit pattern.**

Every Bastion function, given the same input data and the same RNG seed, must
produce the exact same IEEE 754 bit pattern on every invocation, on every platform,
regardless of thread scheduling or evaluation order.

---

## Scope

This contract applies to:

1. **All 15 Bastion primitives** (P-layer)
2. **All runtime kernels** built on those primitives (R-layer)
3. **All Bastion library functions** composed from P and R (B-layer)

---

## Mechanisms

### 1. Kahan-Compensated Summation

All reductions (sum, mean, variance, covariance) use `KahanAccumulatorF64`
from `cjc-repro`. This eliminates summation-order sensitivity that plagues
naive floating-point accumulation.

**Location:** `cjc-repro/src/kahan.rs`

For parallel or order-invariant contexts, `BinnedAccumulatorF64` (2048-bin
superaccumulator in `cjc-runtime/src/accumulator.rs`) provides reproducible
results regardless of reduction order.

### 2. Deterministic Sorting

All sort operations use `f64::total_cmp` which defines a total order on
IEEE 754 doubles, including NaN placement. This ensures:

- `stable_sort` produces identical permutations across runs
- `argsort_stable` produces identical index vectors
- `rank` / `dense_rank` / `row_number` produce identical rankings
- `nth_element` (introselect) uses `select_nth_unstable_by(f64::total_cmp)`

**Key:** Rust's `sort_by` is guaranteed stable (merge sort). Introselect
(`select_nth_unstable_by`) is not stable but is deterministic for a given input.

### 3. Deterministic RNG

CJC uses `SplitMix64` (from `cjc-repro/src/lib.rs`) with:

- **Seeded construction:** `Rng::seeded(seed: u64)`
- **Deterministic forking:** `rng.fork()` produces a child RNG with a
  deterministic derived seed
- **No external entropy:** Never reads from OS random sources

Functions using RNG:
- `sample_indices(n, k, replace, seed)` — explicit seed parameter
- `Tensor.randn(shape)` — uses interpreter RNG (seeded at program start)
- `categorical_sample(probs)` — uses interpreter RNG

### 4. No HashMap-Dependent Ordering

CJC avoids `HashMap` iteration in any path that affects output ordering.
Where associative containers are needed, `BTreeMap` is used to guarantee
deterministic iteration order.

**Exception:** `n_distinct` uses a `HashSet` internally for counting, but
its output is a single integer (count), not an ordered collection.

### 5. No Floating-Point Contraction

CJC does not use `-ffast-math` or equivalent. All operations follow IEEE 754
semantics. Fused multiply-add (FMA) is used only where explicitly called via
`f64::mul_add`, never implicitly by the compiler.

---

## Per-Primitive Determinism Guarantees

| # | Primitive | Deterministic? | Mechanism |
|---|-----------|---------------|-----------|
| 1 | sum_kahan | Yes | KahanAccumulatorF64, sequential |
| 2 | mean (Kahan) | Yes | sum_kahan / n |
| 3 | variance (Kahan) | Yes | Two-pass Kahan: mean then sum of sq deviations |
| 4 | min / max | Yes | Sequential fold, total_cmp for NaN |
| 5 | stable_sort | Yes | Stable merge sort + total_cmp |
| 6 | argsort_stable | Yes | Stable sort on indices + total_cmp |
| 7 | nth_element | Yes | Introselect + total_cmp, deterministic for given input |
| 8 | rank | Yes | Stable argsort + deterministic tie-breaking |
| 9 | map | Yes | Sequential elementwise |
| 10 | zip_map (broadcast2) | Yes | Sequential pairwise with shape broadcast |
| 11 | filter_mask | Yes | Sequential scan, order-preserving |
| 12 | gather | Yes | Index-based lookup, no ambiguity |
| 13 | rolling_reduce | Yes | Sequential window with Kahan sum |
| 14 | prefix_scan | Yes | Sequential left-to-right accumulation |
| 15 | sample_indices | Yes | SplitMix64 with explicit seed |

| # | Special Function | Deterministic? | Mechanism |
|---|-----------------|---------------|-----------|
| S1 | erf | Yes | Polynomial approximation (A&S 7.1.26), no branching on platform |
| S2 | erfc | Yes | Same polynomial, special cases for 0/inf |
| S3 | normal_cdf | Yes | A&S approximation |
| S4 | normal_pdf | Yes | Exact formula: exp(-x^2/2) / sqrt(2*pi) |

---

## Testing Strategy

### Determinism Gate Tests

Every primitive has a determinism test that:
1. Runs the function twice with identical input
2. Asserts bitwise equality of output (`f64::to_bits()`)
3. For RNG-dependent functions, uses identical seeds

**Test locations:**
- `cjc-runtime/src/stats.rs` — `test_sample_indices_determinism`
- `tests/test_bastion_primitives.rs` — `bastion_sample_indices_determinism`

### Cross-Seed Independence

For RNG-dependent functions, tests verify that different seeds produce
different outputs (no degenerate RNG).

### Parity Tests

`median_fast` is tested against `median` (sort-based) to verify that the
faster introselect path produces identical results.

---

## What Breaks Determinism (Anti-Patterns)

| Anti-Pattern | Why It Breaks | CJC's Mitigation |
|-------------|--------------|------------------|
| Naive summation | Float addition is not associative | Kahan/Binned accumulators |
| HashMap iteration | Random hash seed changes order | BTreeMap or sorted output |
| Thread scheduling | Parallel reduction order varies | Sequential execution (no Rayon) |
| Platform RNG | OS entropy differs | SplitMix64, no external entropy |
| Unstable sort without total_cmp | NaN comparison is non-transitive | total_cmp everywhere |
| FMA contraction | Compiler may fuse mul+add differently | No -ffast-math, explicit mul_add only |

---

## Guarantees NOT Made

1. **Cross-version determinism:** A future CJC version may change algorithms
   (e.g., switch from A&S to minimax for erf). Output may differ across CJC versions.
2. **Cross-precision determinism:** f32 and f64 results will differ.
3. **Streaming determinism:** If CJC adds streaming/online algorithms, they
   may produce different results from batch algorithms on the same data.
