# CJC Phase B: Data Science Gap Audit -- Stack Role Implementation Prompt

## Instructions for Use

This is a **master prompt** to be given to an AI coding assistant to implement
all data science gaps identified in the Phase A readiness audit. The work is
organized into **8 sub-sprints** (B1 through B8). Each sub-sprint should be run
as a separate conversation. Copy the relevant sub-sprint section plus the
"Context" and "Wiring Pattern" sections into each conversation.

**CRITICAL**: All 6 original sprints (S1-S6) from `docs/STACK_ROLE_PROMPT.md`
are **already fully implemented**. Do NOT re-implement anything listed in the
"What Already Exists" section below. Phase B fills in the **gaps** identified
by the audit.

---

## Context (Include in EVERY sub-sprint conversation)

### What CJC Is

CJC is a deterministic scientific computing language with:
- Two parallel executors: `cjc-eval` (AST tree-walk) and `cjc-mir-exec` (MIR interpreter)
- Every builtin must be registered in **4-6 places** (the "wiring pattern")
- All floating-point reductions must use Kahan or Binned summation
- `BTreeMap`/`BTreeSet` everywhere -- no `HashMap` with random iteration order
- Same input must produce bit-identical output on every run
- 18 crates in the workspace, zero external dependencies

### Workspace Layout

```
crates/
  cjc-runtime/src/
    builtins.rs        -- shared stateless builtin dispatch (BOTH executors call this)
    tensor.rs          -- Tensor type (~1430 lines, has sum/mean/matmul/attention/conv/sigmoid/tanh/etc.)
    linalg.rs          -- LU, QR, Cholesky, inverse, det, solve, lstsq, eigh, kron, trace, norm_frobenius
    stats.rs           -- variance, sd, median, quantile, cor, cov, cumsum, lag/lead, rank, etc.
    distributions.rs   -- Normal/t/chi2/F CDF/PDF/PPF, binomial, poisson
    hypothesis.rs      -- t-test, chi2, ANOVA, f-test, lm()
    ml.rs              -- loss functions, SGD, Adam, confusion_matrix, AUC, kfold
    fft.rs             -- Cooley-Tukey radix-2 FFT, IFFT, RFFT, PSD
    window.rs          -- window_sum/mean/min/max
    accumulator.rs     -- BinnedAccumulatorF64/F32 (order-invariant summation)
    value.rs           -- Value enum (Int, Float, Bool, String, Tensor, Array, Struct, etc.)
    complex.rs         -- ComplexF64 with fixed-sequence arithmetic
    sparse.rs          -- SparseCsr, SparseCoo
    buffer.rs          -- COW Buffer<T>
    tensor_tiled.rs    -- TiledMatmul (64x64 L2-friendly tiling)
    lib.rs             -- pub mod declarations + re-exports
  cjc-types/src/
    effect_registry.rs -- EffectSet classification for ALL builtins
    lib.rs             -- Type enum, TypeEnv, TypeChecker
  cjc-eval/src/
    lib.rs             -- AST interpreter with is_known_builtin() list
  cjc-mir-exec/src/
    lib.rs             -- MIR executor with is_known_builtin() list
  cjc-ad/src/
    lib.rs             -- Forward (Dual) + Reverse (GradGraph) AD (~670 lines)
  cjc-data/src/
    lib.rs             -- DataFrame, TidyView, Column, joins, pivots (~5500 lines)
  cjc-repro/src/
    lib.rs             -- Rng (SplitMix64), KahanAccumulatorF64, pairwise_sum
tests/
  hardening_tests/
    mod.rs             -- pub mod declarations for H1-H18
    test_h1_span_unify.rs ... test_h18_regression_anova.rs
  test_hardening.rs    -- mod hardening_tests;
  audit_tests/         -- existing audit tests (mod.rs + 19 files)
```

### What Already Exists (DO NOT re-implement)

**Sprint 1 (stats.rs)**: `mean`, `variance`, `sample_variance`, `pop_variance`,
`sd`, `sample_sd`, `pop_sd`, `se`, `median`, `quantile`, `iqr`, `skewness`,
`kurtosis`, `z_score`, `standardize`, `n_distinct`, `cumsum`, `cumprod`,
`cummax`, `cummin`, `lag`, `lead`, `rank`, `dense_rank`, `row_number`, `histogram`

**Sprint 2 (distributions.rs, hypothesis.rs, stats.rs)**: `cor`, `cov`,
`sample_cov`, `cor_matrix`, `cov_matrix`, `normal_cdf`, `normal_pdf`,
`normal_ppf`, `t_cdf`, `chi2_cdf`, `f_cdf`, `t_test` (1-sample),
`t_test_two_sample` (Welch), `t_test_paired`, `chi_squared_test`

**Sprint 3 (linalg.rs)**: `lu_decompose`, `qr_decompose`, `cholesky`, `det`,
`solve`, `lstsq`, `trace`, `norm_frobenius`, `eigh` (Jacobi), `svd`,
`matrix_rank`, `inverse`, `kron`

**Sprint 4 (ml.rs, tensor.rs)**: `mse_loss`, `cross_entropy_loss`,
`binary_cross_entropy`, `huber_loss`, `hinge_loss`, `sgd_step` + `SgdState`,
`adam_step` + `AdamState`, `sigmoid`, `tanh_activation`, `leaky_relu`, `silu`,
`mish`, `argmax`, `argmin`, `clamp`, `one_hot`, `confusion_matrix`, `precision`,
`recall`, `f1_score`, `accuracy`, `auc_roc`

**Sprint 5 (hypothesis.rs)**: `lm` (OLS with auto-intercept, R^2/adj-R^2,
F-test, t-values, p-values, residuals)

**Sprint 6 (fft.rs, distributions.rs, hypothesis.rs, ml.rs)**: `fft`, `ifft`,
`rfft`, `psd`, `t_ppf`, `chi2_ppf`, `f_ppf`, `binomial_pmf`, `binomial_cdf`,
`poisson_pmf`, `poisson_cdf`, `anova_oneway`, `f_test`, `kfold_indices`,
`train_test_split`

**Autodiff (cjc-ad/src/lib.rs)**: Forward mode (`Dual`) with +,-,*,/,neg,sin,
cos,exp,ln,sqrt,pow. Reverse mode (`GradGraph`) with Add, Sub, Mul, Div, Neg,
MatMul, Sum, Mean, ScalarMul, Exp, Ln, StructField, MapLookup.

### The Wiring Pattern (CRITICAL -- follow for every new builtin)

Every new builtin function requires changes in **exactly 4-6 files**:

**1. Implementation module** (e.g., `crates/cjc-runtime/src/stats.rs`):
   - Write the pure Rust function
   - Add `#[cfg(test)] mod tests` with unit tests (or add to existing)
   - Use `KahanAccumulatorF64` from `cjc_repro` for any summation

**2. `crates/cjc-runtime/src/builtins.rs`** -- add dispatch arm:
```rust
// Inside dispatch_builtin() match, BEFORE the `_ => Ok(None)` catch-all:
"weighted_mean" => {
    if args.len() != 2 { return Err("weighted_mean requires 2 arguments".into()); }
    let data = value_to_f64_vec(&args[0])?;
    let weights = value_to_f64_vec(&args[1])?;
    Ok(Some(Value::Float(crate::stats::weighted_mean(&data, &weights)?)))
}
```

**3. `crates/cjc-mir-exec/src/lib.rs`** -- add to `is_known_builtin()`:
```rust
| "weighted_mean"
```

**4. `crates/cjc-eval/src/lib.rs`** -- add to `is_known_builtin()`:
```rust
| "weighted_mean"
```

**5. `crates/cjc-types/src/effect_registry.rs`** -- add effect classification:
```rust
m.insert("weighted_mean", alloc);
```

**6. `crates/cjc-runtime/src/lib.rs`** -- add `pub mod` (only if creating a new module)

### Effect Classification Guide

| Effect | Flag | When to use |
|--------|------|-------------|
| PURE | `pure` (var already exists) | No side effects, no allocation |
| ALLOC | `alloc` | Allocates new Value (arrays, strings, tensors) |
| IO | `io` | File/network/clock access |
| NONDET | `EffectSet::new(EffectSet::NONDET)` | Result depends on external state |
| MUTATES | `mutates` | Modifies an argument in place |
| GC | `gc` | Triggers garbage collection |

Most stats/linalg builtins are `alloc`. Distribution PDF/CDF/PMF returning a
single float are `pure`. LR schedule functions returning a single float are `pure`.

### Return Value Conventions

**Scalar**: `Ok(Some(Value::Float(...)))` or `Ok(Some(Value::Int(...)))`

**Array**: `Ok(Some(Value::Array(Rc::new(vec_of_values))))`

**Tensor**: `Ok(Some(Value::Tensor(tensor)))`

**Struct** (for result objects):
```rust
let mut fields = std::collections::HashMap::new();
fields.insert("statistic".into(), Value::Float(r.statistic));
fields.insert("p_value".into(), Value::Float(r.p_value));
Ok(Some(Value::Struct { name: "ResultName".into(), fields }))
```

**Tuple** (for paired results):
```rust
Ok(Some(Value::Tuple(Rc::new(vec![Value::Float(a), Value::Float(b)]))))
```

### Testing Pattern

Phase B tests go in `tests/audit_phase_b/` (new directory). Each sub-sprint
creates a test file `tests/audit_phase_b/test_b{N}_{name}.rs`:

```rust
//! Phase B audit test B{N}: {Description}

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn b{N}_feature_name() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let wm = weighted_mean(data, [1.0, 1.0, 1.0, 1.0, 1.0]);
print(wm);
"#);
    assert_eq!(out, vec!["3"]);
}
```

Create `tests/audit_phase_b/mod.rs`:
```rust
pub mod test_b1_weighted_stats;
pub mod test_b2_rank_correlations;
pub mod test_b3_linalg_extensions;
pub mod test_b4_ml_extensions;
pub mod test_b5_analyst_qol;
pub mod test_b6_fft_distributions;
pub mod test_b7_nonparametric;
pub mod test_b8_autodiff;
```

Create `tests/test_audit_phase_b.rs`:
```rust
mod audit_phase_b;
```

### Property Test Pattern

For numerical functions, add property-based tests using `proptest`:

```rust
#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_weighted_mean_uniform_weights(
            data in prop::collection::vec(-1000.0..1000.0f64, 2..50)
        ) {
            let weights = vec![1.0; data.len()];
            let wm = crate::stats::weighted_mean(&data, &weights).unwrap();
            let m = crate::stats::mean(&data);
            prop_assert!((wm - m).abs() < 1e-10);
        }
    }
}
```

### Determinism Rules

1. **No `HashMap` with iteration** -- use `BTreeMap` or `Vec` with deterministic ordering
2. **No `f64` as hash key** -- use integer indices or `to_bits()` for exact comparison
3. **No `par_iter()` in new code** -- sequential only
4. **Kahan summation for all reductions** -- `KahanAccumulatorF64::new()` / `.add()` / `.finalize()`
5. **Deterministic sorting** -- `sort_by(|a, b| a.total_cmp(b))` for NaN handling
6. **No `SystemTime`** -- only `datetime_now()` is allowed to be NONDET
7. **Fixed iteration order** -- for-loops over ranges, not iterators over hash structures

---

## Sub-Sprint B1: Weighted & Robust Statistics (stats.rs additions)

### Goal

Add weighted statistics, trimmed/winsorized means, MAD, mode, and percentile rank
to `crates/cjc-runtime/src/stats.rs`. These fill the "no robust statistics, no
weighted operations" gap from the Sprint 1 audit.

### Functions to Implement

```rust
/// Weighted mean: sum(data[i] * weights[i]) / sum(weights).
/// Uses Kahan summation for both numerator and denominator.
/// Errors if lengths differ or weights sum to zero.
pub fn weighted_mean(data: &[f64], weights: &[f64]) -> Result<f64, String>

/// Weighted variance: sum(w[i] * (x[i] - weighted_mean)^2) / sum(w).
/// Two-pass: first weighted mean (Kahan), then weighted sum of squared deviations.
/// Errors if lengths differ or weights sum to zero.
pub fn weighted_var(data: &[f64], weights: &[f64]) -> Result<f64, String>

/// Trimmed mean: mean of data with `proportion` fraction removed from each tail.
/// proportion=0.1 removes bottom 10% and top 10%, computing mean of middle 80%.
/// Clone and sort; then compute mean of the middle (1 - 2*proportion) fraction.
/// Errors if proportion < 0 or proportion >= 0.5, or data empty.
/// DETERMINISM: sort with total_cmp, Kahan summation for mean.
pub fn trimmed_mean(data: &[f64], proportion: f64) -> Result<f64, String>

/// Winsorize: replace values below the `proportion` quantile with the lower
/// boundary, and values above the `(1-proportion)` quantile with the upper
/// boundary. Returns a new Vec<f64>.
/// Errors if proportion < 0 or proportion >= 0.5, or data empty.
/// DETERMINISM: uses existing quantile() function.
pub fn winsorize(data: &[f64], proportion: f64) -> Result<Vec<f64>, String>

/// Median absolute deviation: median(|x[i] - median(x)|).
/// Robust measure of spread. Convention: does NOT multiply by 1.4826 scaling
/// factor (user can do this manually for normal-consistency).
/// DETERMINISM: two sorts, both with total_cmp.
pub fn mad(data: &[f64]) -> Result<f64, String>

/// Mode: most frequent value. Uses bit-exact comparison via to_bits().
/// Ties broken by smallest value (deterministic via sorted copy + sequential scan).
/// Returns the most frequent value.
/// Errors if data is empty.
pub fn mode(data: &[f64]) -> Result<f64, String>

/// Percentile rank: fraction of data values strictly less than the given value,
/// plus half the fraction equal to the value.
/// Returns a value in [0, 1].
/// DETERMINISM: sequential scan with total_cmp comparison.
pub fn percentile_rank(data: &[f64], value: f64) -> Result<f64, String>
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `weighted_mean(data, weights)` | `"weighted_mean"` | 2 (Array, Array) | Float | ALLOC |
| `weighted_var(data, weights)` | `"weighted_var"` | 2 (Array, Array) | Float | ALLOC |
| `trimmed_mean(data, prop)` | `"trimmed_mean"` | 2 (Array, Float) | Float | ALLOC |
| `winsorize(data, prop)` | `"winsorize"` | 2 (Array, Float) | Array | ALLOC |
| `mad(data)` | `"mad"` | 1 (Array) | Float | ALLOC |
| `mode(data)` | `"mode"` | 1 (Array) | Float | ALLOC |
| `percentile_rank(data, value)` | `"percentile_rank"` | 2 (Array, Float) | Float | ALLOC |

### Unit Tests (add to stats.rs `mod tests`)

- `test_weighted_mean_uniform`: equal weights should equal simple mean
- `test_weighted_mean_skewed`: `[1,2,3]` with weights `[3,0,0]` → 1.0
- `test_weighted_mean_empty`: empty data → error
- `test_weighted_var_uniform`: equal weights should match pop_variance
- `test_trimmed_mean_10pct`: verify trimming removes outliers
- `test_trimmed_mean_zero`: proportion=0.0 should equal regular mean
- `test_trimmed_mean_invalid_proportion`: proportion >= 0.5 → error
- `test_winsorize_basic`: verify extreme values are clipped to quantile boundaries
- `test_winsorize_no_change`: proportion=0.0 should return original data
- `test_mad_symmetric`: MAD of `[-2,-1,0,1,2]` → 1.0
- `test_mad_constant`: MAD of `[5,5,5]` → 0.0
- `test_mode_simple`: `[1,2,2,3]` → 2.0
- `test_mode_tie`: tie broken by smallest value
- `test_mode_single`: `[42]` → 42.0
- `test_percentile_rank_median`: percentile_rank of median value ≈ 0.5
- `test_percentile_rank_min`: percentile_rank of min value
- `test_determinism`: same input → bit-identical output

### Integration Tests

Create `tests/audit_phase_b/test_b1_weighted_stats.rs` -- 10+ tests through MIR executor.

### Validation

```
cargo test -p cjc-runtime stats                  # unit tests
cargo test --test test_audit_phase_b -- b1        # integration tests
cargo test --workspace                            # 0 regressions
```

---

## Sub-Sprint B2: Rank Correlations & Partial Correlation (stats.rs additions)

### Goal

Add Spearman rank correlation, Kendall tau-b, partial correlation, and Fisher
z-transform confidence intervals to `crates/cjc-runtime/src/stats.rs`. These
fill the "no non-Pearson correlation" gap from the Sprint 2 audit.

### Functions to Implement

```rust
/// Spearman rank correlation: Pearson correlation of the ranks of x and y.
/// Uses existing rank() for rank assignment, then existing cor() on ranks.
/// DETERMINISM: stable sort for rank assignment via existing rank().
pub fn spearman_cor(x: &[f64], y: &[f64]) -> Result<f64, String>

/// Kendall tau-b correlation coefficient.
/// Counts concordant and discordant pairs with tie adjustment.
/// Uses O(n^2) pairwise comparison (deterministic, no approximation).
/// tau_b = (concordant - discordant) / sqrt((n0 - n1) * (n0 - n2))
/// where n0 = n*(n-1)/2, n1 = sum of ties in x, n2 = sum of ties in y.
/// DETERMINISM: sequential pairwise enumeration i < j.
pub fn kendall_cor(x: &[f64], y: &[f64]) -> Result<f64, String>

/// Partial correlation: correlation of x and y controlling for z.
/// Uses the recursive formula:
/// partial_cor(x,y|z) = (cor(x,y) - cor(x,z)*cor(y,z))
///                    / (sqrt(1 - cor(x,z)^2) * sqrt(1 - cor(y,z)^2))
/// Errors if any denominator term is zero (perfect correlation with z).
pub fn partial_cor(x: &[f64], y: &[f64], z: &[f64]) -> Result<f64, String>

/// Confidence interval for Pearson correlation using Fisher z-transform.
/// z_r = atanh(r), se = 1/sqrt(n-3), CI = tanh(z_r +/- z_{alpha/2} * se).
/// alpha is the significance level (e.g., 0.05 for 95% CI).
/// Returns (lower_bound, upper_bound).
/// Uses existing normal_ppf() for the critical value.
/// Errors if n < 4 or alpha outside (0, 1).
pub fn cor_ci(x: &[f64], y: &[f64], alpha: f64) -> Result<(f64, f64), String>
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `spearman_cor(x, y)` | `"spearman_cor"` | 2 (Array, Array) | Float | ALLOC |
| `kendall_cor(x, y)` | `"kendall_cor"` | 2 (Array, Array) | Float | ALLOC |
| `partial_cor(x, y, z)` | `"partial_cor"` | 3 (Array, Array, Array) | Float | ALLOC |
| `cor_ci(x, y, alpha)` | `"cor_ci"` | 3 (Array, Array, Float) | Tuple(Float, Float) | ALLOC |

### Return Value Convention for cor_ci

```rust
"cor_ci" => {
    if args.len() != 3 { return Err("cor_ci requires 3 arguments".into()); }
    let x = value_to_f64_vec(&args[0])?;
    let y = value_to_f64_vec(&args[1])?;
    let alpha = value_to_f64(&args[2])?;
    let (lo, hi) = crate::stats::cor_ci(&x, &y, alpha)?;
    Ok(Some(Value::Tuple(Rc::new(vec![Value::Float(lo), Value::Float(hi)]))))
}
```

### Unit Tests

- `test_spearman_perfect_monotone`: perfectly monotonic → 1.0
- `test_spearman_perfect_reverse`: perfectly reverse monotonic → -1.0
- `test_spearman_nonlinear`: x vs x^2 (positive, less than 1)
- `test_spearman_equals_pearson_for_linear`: linear data → same as cor()
- `test_kendall_concordant`: perfectly concordant → 1.0
- `test_kendall_discordant`: perfectly discordant → -1.0
- `test_kendall_with_ties`: verify tie adjustment formula
- `test_kendall_known_values`: small dataset with known tau-b value
- `test_partial_cor_removes_confounder`: z explains both x and y → partial_cor ≈ 0
- `test_partial_cor_no_confounding`: z independent → partial_cor ≈ cor(x,y)
- `test_cor_ci_95pct`: 95% CI should be narrower than 99% CI
- `test_cor_ci_contains_r`: point estimate r should be inside CI
- `test_cor_ci_large_n`: large n → narrow CI
- `test_determinism`: bit-identical on double run

### Integration Tests

Create `tests/audit_phase_b/test_b2_rank_correlations.rs` -- 10+ tests.

---

## Sub-Sprint B3: Linear Algebra Extensions (linalg.rs additions)

### Goal

Add condition number, L1/infinity norms, Schur decomposition, and matrix
exponential to `crates/cjc-runtime/src/linalg.rs`. These fill the gaps
from the Sprint 3 audit.

### Functions to Implement

```rust
impl Tensor {
    /// Condition number via eigenvalue ratio.
    /// For symmetric matrices: uses eigh(), returns |lambda_max| / |lambda_min|.
    /// For general matrices: computes eigh(A^T * A), returns
    /// sqrt(sigma_max / sigma_min) where sigma are eigenvalues of A^T A.
    /// Returns f64::INFINITY if matrix is singular (zero smallest eigenvalue).
    /// DETERMINISM: uses existing eigh() which is deterministic.
    pub fn cond(&self) -> Result<f64, RuntimeError>

    /// 1-norm: maximum absolute column sum.
    /// For m x n matrix: max over j of sum_i |a[i][j]|.
    /// Uses Kahan summation for each column sum.
    pub fn norm_1(&self) -> Result<f64, RuntimeError>

    /// Infinity norm: maximum absolute row sum.
    /// For m x n matrix: max over i of sum_j |a[i][j]|.
    /// Uses Kahan summation for each row sum.
    pub fn norm_inf(&self) -> Result<f64, RuntimeError>

    /// Real Schur decomposition: A = Q * T * Q^T.
    /// T is quasi-upper-triangular (1x1 and 2x2 blocks on diagonal).
    ///
    /// Algorithm:
    /// 1. Reduce A to upper Hessenberg form via Householder reflections
    /// 2. Apply implicit double-shift QR iterations (Francis QR step)
    /// 3. Deflate when sub-diagonal elements converge to zero
    /// 4. Accumulate transformations in Q
    ///
    /// DETERMINISM:
    /// - Householder reflections: fixed sign convention (positive diagonal)
    /// - QR shifts: Wilkinson shift from trailing 2x2 submatrix
    /// - Convergence: |h[i+1,i]| < eps * (|h[i,i]| + |h[i+1,i+1]|), eps = 1e-14
    /// - Maximum iterations: 200 * n -- returns error if not converged
    ///
    /// Returns (Q: n x n orthogonal, T: n x n quasi-upper-triangular).
    pub fn schur(&self) -> Result<(Tensor, Tensor), RuntimeError>

    /// Matrix exponential: exp(A) via scaling and squaring with Pade approximation.
    ///
    /// Algorithm (Al-Mohy and Higham, 2009):
    /// 1. s = max(0, ceil(log2(||A||_1 / theta_13))), theta_13 = 5.371920351148152
    /// 2. B = A / 2^s
    /// 3. Compute Pade(13,13) approximant: r_13(B) = p_13(B) / q_13(B)
    ///    where p_13 and q_13 are degree-13 polynomials with known coefficients
    /// 4. Square the result s times: exp(A) = r_13(B)^(2^s)
    ///
    /// The Pade coefficients are fixed constants (b_0 through b_13).
    /// Step 3 requires computing B^2, B^4, B^6, then building U and V matrices
    /// from these powers, then solving (V - U) * r = (V + U).
    ///
    /// DETERMINISM: all operations are matrix multiply + solve, fully deterministic.
    /// Returns n x n matrix.
    pub fn matrix_exp(&self) -> Result<Tensor, RuntimeError>
}
```

### Schur Decomposition: Detailed Algorithm

```
1. Hessenberg reduction:
   For k = 0 to n-3:
     - Compute Householder reflector v for column k below diagonal
     - Apply from left: H = (I - 2vv^T) * H
     - Apply from right: H = H * (I - 2vv^T)
     - Accumulate in Q: Q = Q * (I - 2vv^T)

2. Implicit QR iteration:
   while unconverged sub-diagonal elements exist:
     - Find active unreduced block (largest i where h[i+1,i] != 0)
     - Compute Wilkinson shift from trailing 2x2 of active block
     - Apply Francis double-shift QR step
     - Check convergence of sub-diagonal elements
     - Deflate when h[i+1,i] < eps * (|h[i,i]| + |h[i+1,i+1]|)
     - If iteration count > 200*n, return error
```

### Matrix Exponential: Pade Coefficients

```rust
const PADE_COEFFS: [f64; 14] = [
    64764752532480000.0,    // b_0
    32382376266240000.0,    // b_1
    7771770303897600.0,     // b_2
    1187353796428800.0,     // b_3
    129060195264000.0,      // b_4
    10559470521600.0,       // b_5
    670442572800.0,         // b_6
    33522128640.0,          // b_7
    1323241920.0,           // b_8
    40840800.0,             // b_9
    960960.0,               // b_10
    16380.0,                // b_11
    182.0,                  // b_12
    1.0,                    // b_13
];
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `cond(A)` | `"cond"` | 1 (Tensor) | Float | ALLOC |
| `norm_1(A)` | `"norm_1"` | 1 (Tensor) | Float | PURE |
| `norm_inf(A)` | `"norm_inf"` | 1 (Tensor) | Float | PURE |
| `schur(A)` | `"schur"` | 1 (Tensor) | Tuple(Tensor, Tensor) | ALLOC |
| `matrix_exp(A)` | `"matrix_exp"` | 1 (Tensor) | Tensor | ALLOC |

### Unit Tests

- `test_norm_1_identity`: identity → 1.0
- `test_norm_1_known`: `[[1,2],[3,4]]` → max(|1|+|3|, |2|+|4|) = 6.0
- `test_norm_inf_known`: `[[1,2],[3,4]]` → max(|1|+|2|, |3|+|4|) = 7.0
- `test_norm_inf_is_norm_1_transpose`: ||A||_inf = ||A^T||_1
- `test_cond_identity`: cond(I) = 1.0
- `test_cond_diagonal`: cond(diag(2,4)) = 2.0
- `test_cond_ill_conditioned`: near-singular matrix has cond >> 1
- `test_schur_identity`: identity is its own Schur form
- `test_schur_reconstruction`: Q * T * Q^T ≈ A (frobenius norm < 1e-10)
- `test_schur_orthogonal_Q`: Q^T * Q ≈ I
- `test_schur_triangular_T`: T is quasi-upper-triangular (sub-diagonal small)
- `test_schur_diagonal_matrix`: diagonal matrix → T = A, Q = I
- `test_matrix_exp_zero`: exp(0) = I
- `test_matrix_exp_diagonal`: exp(diag(a,b)) = diag(exp(a), exp(b))
- `test_matrix_exp_known_2x2`: verify against known analytical result
- `test_matrix_exp_nilpotent`: exp([[0,1],[0,0]]) = [[1,1],[0,1]]
- `test_determinism`: all operations bit-identical on double run

### Integration Tests

Create `tests/audit_phase_b/test_b3_linalg_extensions.rs` -- 15+ tests.

---

## Sub-Sprint B4: ML Training Extensions (ml.rs + tensor.rs additions)

### Goal

Add tensor concatenation/stacking, top-k, batch normalization, dropout, learning
rate scheduling, regularization, and early stopping to `ml.rs` and `tensor.rs`.
These fill the "can't build a real neural network" gap from Sprint 4 audit.

### Functions: tensor.rs additions

```rust
impl Tensor {
    /// Concatenate tensors along existing axis.
    /// All tensors must have same shape except along the concat axis.
    /// DETERMINISM: sequential concatenation in input order.
    pub fn cat(tensors: &[&Tensor], axis: usize) -> Result<Tensor, RuntimeError>

    /// Stack tensors along a new axis.
    /// All tensors must have identical shape.
    /// Inserts a new dimension at `axis`.
    /// E.g., stacking three [2,3] tensors at axis=0 gives [3,2,3].
    pub fn stack(tensors: &[&Tensor], axis: usize) -> Result<Tensor, RuntimeError>

    /// Top-k values and indices along last axis.
    /// Returns (values_tensor, indices_vec).
    /// DETERMINISM: stable sort with index tie-breaking (lower index wins).
    pub fn topk(&self, k: usize) -> Result<(Tensor, Vec<usize>), RuntimeError>
}
```

### Functions: ml.rs additions

```rust
/// Batch normalization (inference mode).
/// Normalizes: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta.
/// All parameter arrays must have same length as feature dimension.
pub fn batch_norm(
    x: &[f64],
    running_mean: &[f64],
    running_var: &[f64],
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Result<Vec<f64>, String>

/// Dropout mask generation.
/// Returns a mask of 0.0 and scale values (1/(1-p)) using inverted dropout.
/// Uses seeded SplitMix64 RNG for determinism.
pub fn dropout_mask(n: usize, drop_prob: f64, seed: u64) -> Vec<f64>

/// Apply dropout: element-wise multiply data by mask.
pub fn apply_dropout(data: &[f64], mask: &[f64]) -> Result<Vec<f64>, String>

/// Learning rate schedule: step decay.
/// lr = initial_lr * decay_rate^(floor(epoch / step_size))
pub fn lr_step_decay(initial_lr: f64, decay_rate: f64, epoch: usize, step_size: usize) -> f64

/// Learning rate schedule: cosine annealing.
/// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
pub fn lr_cosine(max_lr: f64, min_lr: f64, epoch: usize, total_epochs: usize) -> f64

/// Learning rate schedule: linear warmup.
/// lr = initial_lr * min(1.0, epoch / warmup_epochs).
pub fn lr_linear_warmup(initial_lr: f64, epoch: usize, warmup_epochs: usize) -> f64

/// L1 regularization penalty: lambda * sum(|params|).
/// Uses Kahan summation.
pub fn l1_penalty(params: &[f64], lambda: f64) -> f64

/// L2 regularization penalty: 0.5 * lambda * sum(params^2).
/// Uses Kahan summation.
pub fn l2_penalty(params: &[f64], lambda: f64) -> f64

/// L1 regularization gradient: lambda * sign(params).
pub fn l1_grad(params: &[f64], lambda: f64) -> Vec<f64>

/// L2 regularization gradient: lambda * params.
pub fn l2_grad(params: &[f64], lambda: f64) -> Vec<f64>

/// Early stopping state tracker.
pub struct EarlyStoppingState {
    pub patience: usize,
    pub min_delta: f64,
    pub best_loss: f64,
    pub wait: usize,
    pub stopped: bool,
}

impl EarlyStoppingState {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }

    /// Check if training should stop. Returns true if no improvement
    /// for `patience` epochs (improvement = decrease > min_delta).
    pub fn check(&mut self, current_loss: f64) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
        }
        if self.wait >= self.patience {
            self.stopped = true;
        }
        self.stopped
    }
}
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `cat(tensors, axis)` | `"cat"` | 2 (Array[Tensor], Int) | Tensor | ALLOC |
| `stack(tensors, axis)` | `"stack"` | 2 (Array[Tensor], Int) | Tensor | ALLOC |
| `topk(tensor, k)` | `"topk"` | 2 (Tensor, Int) | Tuple(Tensor, Array) | ALLOC |
| `batch_norm(x, mean, var, gamma, beta, eps)` | `"batch_norm"` | 6 | Array | ALLOC |
| `dropout_mask(n, p, seed)` | `"dropout_mask"` | 3 (Int, Float, Int) | Array | ALLOC |
| `lr_step_decay(lr, rate, epoch, step)` | `"lr_step_decay"` | 4 (Float, Float, Int, Int) | Float | PURE |
| `lr_cosine(max, min, epoch, total)` | `"lr_cosine"` | 4 (Float, Float, Int, Int) | Float | PURE |
| `lr_linear_warmup(lr, epoch, warmup)` | `"lr_linear_warmup"` | 3 (Float, Int, Int) | Float | PURE |
| `l1_penalty(params, lambda)` | `"l1_penalty"` | 2 (Array, Float) | Float | ALLOC |
| `l2_penalty(params, lambda)` | `"l2_penalty"` | 2 (Array, Float) | Float | ALLOC |

### Unit Tests

- `test_cat_axis0`: cat two [2,3] tensors along axis 0 → [4,3]
- `test_cat_axis1`: cat two [2,3] tensors along axis 1 → [2,6]
- `test_cat_shape_mismatch`: mismatched shapes (non-concat dims) → error
- `test_stack_axis0`: stack three [2,3] tensors at axis 0 → [3,2,3]
- `test_stack_axis1`: stack at axis 1 → [2,3,3]
- `test_topk_basic`: topk([3,1,4,1,5,9], k=3) → values=[9,5,4], indices=[5,4,2]
- `test_topk_ties`: ties broken by lower index
- `test_topk_k_equals_n`: returns all values sorted
- `test_batch_norm_identity`: mean=0, var=1, gamma=1, beta=0 → input unchanged
- `test_batch_norm_shift_scale`: verify shift and scale
- `test_dropout_mask_seed`: same seed → identical mask
- `test_dropout_mask_different_seeds`: different seeds → different masks
- `test_dropout_fraction`: approximately drop_prob fraction of zeros
- `test_lr_step_decay_schedule`: verify at epoch 0, step_size, 2*step_size
- `test_lr_cosine_endpoints`: epoch=0 → max_lr, epoch=total → min_lr
- `test_lr_linear_warmup`: epoch=0 → 0, epoch >= warmup → initial_lr
- `test_l1_penalty_known`: |[1,-2,3]| * 0.1 = 0.6
- `test_l2_penalty_known`: 0.5 * 0.1 * (1+4+9) = 0.7
- `test_early_stopping_triggers`: verify stops after patience exhausted
- `test_early_stopping_resets`: improvement resets wait counter

### Integration Tests

Create `tests/audit_phase_b/test_b4_ml_extensions.rs` -- 15+ tests.

---

## Sub-Sprint B5: Analyst Quality-of-Life Extensions

### Goal

Add `case_when`, `ntile`, `percent_rank`, `cume_dist`, and weighted least
squares. These fill the "no case_when, no weighted regression" gaps from Sprint 5.

### Functions: stats.rs additions

```rust
/// Divide data into n roughly equal groups (ntile/quantile binning).
/// Returns 1-based group assignments matching original data order.
/// Groups are assigned by sorted rank: first ceil(N/n) get group 1, etc.
/// DETERMINISM: stable sort preserves original order for ties.
pub fn ntile(data: &[f64], n: usize) -> Result<Vec<f64>, String>

/// Percent rank: (rank - 1) / (n - 1), range [0, 1].
/// Uses average-tie ranking from existing rank() function.
/// For n=1, returns 0.0.
pub fn percent_rank(data: &[f64]) -> Result<Vec<f64>, String>

/// Cumulative distribution: count(x_i <= x_j) / n for each x_j.
/// Returns values in (0, 1] matching original data order.
/// DETERMINISM: sequential scan over sorted indices.
pub fn cume_dist(data: &[f64]) -> Result<Vec<f64>, String>
```

### Functions: builtins.rs case_when dispatch

`case_when` is a conditional recoding function. It takes parallel arrays of
conditions and values, returning the value at the first true condition:

```rust
// In builtins.rs dispatch:
"case_when" => {
    if args.len() != 3 { return Err("case_when requires 3 arguments (conditions, values, default)".into()); }
    // conditions: Array[Bool], values: Array[Any], default: Any
    let conditions = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Bool(b) => Ok(*b),
            _ => Err("case_when conditions must be booleans".into()),
        }).collect::<Result<Vec<bool>, String>>()?,
        _ => return Err("case_when conditions must be an array".into()),
    };
    let values = match &args[1] {
        Value::Array(arr) => arr.as_ref().clone(),
        _ => return Err("case_when values must be an array".into()),
    };
    if conditions.len() != values.len() {
        return Err("case_when conditions and values must have same length".into());
    }
    for (i, &cond) in conditions.iter().enumerate() {
        if cond { return Ok(Some(values[i].clone())); }
    }
    Ok(Some(args[2].clone())) // default
}
```

### Functions: hypothesis.rs additions

```rust
/// Weighted least squares regression.
/// Transforms to OLS: X_w = W^{1/2} * X, y_w = W^{1/2} * y.
/// Then applies standard QR-based least squares.
/// Returns LmResult (same struct as lm()).
/// Errors if weights contain non-positive values.
pub fn wls(
    x_flat: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
) -> Result<LmResult, String>
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `case_when(conds, vals, default)` | `"case_when"` | 3 (Array, Array, Any) | Any | ALLOC |
| `ntile(data, n)` | `"ntile"` | 2 (Array, Int) | Array | ALLOC |
| `percent_rank(data)` | `"percent_rank"` | 1 (Array) | Array | ALLOC |
| `cume_dist(data)` | `"cume_dist"` | 1 (Array) | Array | ALLOC |
| `wls(X, y, w, n, p)` | `"wls"` | 5 (Array, Array, Array, Int, Int) | Struct(LmResult) | ALLOC |

### Unit Tests

- `test_case_when_first_true`: first condition true → first value
- `test_case_when_second_true`: second condition true → second value
- `test_case_when_none_true`: no true → default
- `test_case_when_all_true`: returns first (deterministic)
- `test_ntile_even`: 12 items into 4 groups → [1,1,1,2,2,2,3,3,3,4,4,4]
- `test_ntile_uneven`: 10 into 3 groups → reasonable distribution
- `test_ntile_n_equals_len`: each element gets unique group
- `test_percent_rank_sorted`: sorted data → [0.0, 0.25, 0.5, 0.75, 1.0]
- `test_percent_rank_ties`: tied values get same percent rank
- `test_cume_dist_sorted`: sorted data → [0.2, 0.4, 0.6, 0.8, 1.0]
- `test_cume_dist_ties`: tied values get same cume_dist
- `test_wls_uniform_weights`: uniform weights → same as lm()
- `test_wls_downweight_outlier`: outlier with low weight has less influence
- `test_wls_known_result`: known dataset with known WLS solution
- `test_determinism`: bit-identical double run

### Integration Tests

Create `tests/audit_phase_b/test_b5_analyst_qol.rs` -- 12+ tests.

---

## Sub-Sprint B6: Advanced FFT & Distributions

### Goal

Add windowing functions, arbitrary-length FFT (Bluestein/chirp-z), 2D FFT,
and Beta/Gamma/Exponential/Weibull distributions. These fill the "only
power-of-2 FFT" and "narrow distribution set" gaps from Sprint 6.

### Functions: fft.rs additions

```rust
/// Hann window: w[k] = 0.5 * (1 - cos(2*pi*k / (N-1))).
/// Returns N-element Vec. For N=1, returns [1.0].
pub fn hann_window(n: usize) -> Vec<f64>

/// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k / (N-1)).
/// Returns N-element Vec.
pub fn hamming_window(n: usize) -> Vec<f64>

/// Blackman window: w[k] = 0.42 - 0.5*cos(2*pi*k/(N-1)) + 0.08*cos(4*pi*k/(N-1)).
/// Returns N-element Vec.
pub fn blackman_window(n: usize) -> Vec<f64>

/// Arbitrary-length FFT using Bluestein's algorithm (chirp-z transform).
/// Works for ANY input length N (not just powers of 2).
///
/// Algorithm:
/// 1. Compute chirp sequence: w[k] = exp(-i*pi*k^2/N)
/// 2. Multiply input by chirp: a[k] = x[k] * w[k]
/// 3. Compute convolution sequence: b[k] = conj(w[k])
/// 4. Zero-pad a and b to next power of 2 >= 2N-1
/// 5. Convolve via radix-2 FFT: A = fft(a_padded), B = fft(b_padded), C = ifft(A*B)
/// 6. Extract and scale: X[k] = w[k] * C[k]
///
/// DETERMINISM: all twiddle factors computed in fixed sequence from closed-form.
pub fn fft_arbitrary(data: &[(f64, f64)]) -> Vec<(f64, f64)>

/// 2D FFT: apply 1D FFT along rows, then along columns.
/// Input is flattened row-major complex data (rows * cols elements).
/// Both rows and cols must be powers of 2 (uses existing radix-2 fft).
/// Returns flattened row-major result.
pub fn fft_2d(data: &[(f64, f64)], rows: usize, cols: usize) -> Result<Vec<(f64, f64)>, String>

/// 2D inverse FFT.
pub fn ifft_2d(data: &[(f64, f64)], rows: usize, cols: usize) -> Result<Vec<(f64, f64)>, String>
```

### Functions: distributions.rs additions

```rust
/// Beta distribution PDF: x^(a-1) * (1-x)^(b-1) / B(a,b).
/// x in [0,1], a > 0, b > 0.
/// B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b), computed via ln_gamma.
pub fn beta_pdf(x: f64, a: f64, b: f64) -> f64

/// Beta distribution CDF via regularized incomplete beta function.
/// Uses existing regularized_incomplete_beta() helper.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64

/// Gamma distribution PDF: x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k)).
/// x >= 0, k > 0 (shape), theta > 0 (scale).
pub fn gamma_pdf(x: f64, k: f64, theta: f64) -> f64

/// Gamma distribution CDF via regularized lower incomplete gamma function.
/// Uses existing regularized_gamma_p() helper.
pub fn gamma_cdf(x: f64, k: f64, theta: f64) -> f64

/// Exponential distribution PDF: lambda * exp(-lambda * x).
/// x >= 0, lambda > 0 (rate).
pub fn exp_pdf(x: f64, lambda: f64) -> f64

/// Exponential distribution CDF: 1 - exp(-lambda * x).
pub fn exp_cdf(x: f64, lambda: f64) -> f64

/// Weibull distribution PDF: (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k).
/// x >= 0, k > 0 (shape), lambda > 0 (scale).
pub fn weibull_pdf(x: f64, k: f64, lambda: f64) -> f64

/// Weibull distribution CDF: 1 - exp(-(x/lambda)^k).
pub fn weibull_cdf(x: f64, k: f64, lambda: f64) -> f64
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `hann(n)` | `"hann"` | 1 (Int) | Array | ALLOC |
| `hamming(n)` | `"hamming"` | 1 (Int) | Array | ALLOC |
| `blackman(n)` | `"blackman"` | 1 (Int) | Array | ALLOC |
| `fft_arbitrary(data)` | `"fft_arbitrary"` | 1 (Array) | Array | ALLOC |
| `fft_2d(data, rows, cols)` | `"fft_2d"` | 3 (Array, Int, Int) | Array | ALLOC |
| `ifft_2d(data, rows, cols)` | `"ifft_2d"` | 3 (Array, Int, Int) | Array | ALLOC |
| `beta_pdf(x, a, b)` | `"beta_pdf"` | 3 (Float, Float, Float) | Float | PURE |
| `beta_cdf(x, a, b)` | `"beta_cdf"` | 3 (Float, Float, Float) | Float | PURE |
| `gamma_pdf(x, k, theta)` | `"gamma_pdf"` | 3 (Float, Float, Float) | Float | PURE |
| `gamma_cdf(x, k, theta)` | `"gamma_cdf"` | 3 (Float, Float, Float) | Float | PURE |
| `exp_pdf(x, lambda)` | `"exp_pdf"` | 2 (Float, Float) | Float | PURE |
| `exp_cdf(x, lambda)` | `"exp_cdf"` | 2 (Float, Float) | Float | PURE |
| `weibull_pdf(x, k, lambda)` | `"weibull_pdf"` | 3 (Float, Float, Float) | Float | PURE |
| `weibull_cdf(x, k, lambda)` | `"weibull_cdf"` | 3 (Float, Float, Float) | Float | PURE |

### Unit Tests

- `test_hann_endpoints`: w[0] = 0.0 and w[N-1] = 0.0
- `test_hann_midpoint`: w[N/2] = 1.0
- `test_hann_symmetry`: w[k] = w[N-1-k]
- `test_hamming_endpoints`: w[0] ≈ 0.08 and w[N-1] ≈ 0.08
- `test_blackman_endpoints`: w[0] ≈ 0.0 and w[N-1] ≈ 0.0
- `test_fft_arbitrary_prime`: 7-element signal matches brute-force DFT
- `test_fft_arbitrary_matches_radix2`: for power-of-2 input, matches existing fft()
- `test_fft_arbitrary_parseval`: energy in time domain = energy in freq domain
- `test_fft_2d_constant`: constant 2D → single nonzero DC component
- `test_fft_2d_roundtrip`: fft_2d then ifft_2d recovers original
- `test_beta_pdf_symmetric`: beta_pdf(0.5, 2, 2) = 1.5
- `test_beta_cdf_uniform`: beta_cdf(x, 1, 1) = x (uniform)
- `test_beta_cdf_endpoints`: beta_cdf(0, a, b) = 0, beta_cdf(1, a, b) = 1
- `test_gamma_cdf_exponential`: gamma_cdf(x, 1, 1/lambda) ≈ exp_cdf(x, lambda)
- `test_exp_cdf_memoryless`: exp_cdf(1/lambda, lambda) ≈ 1 - 1/e
- `test_exp_pdf_integral`: sum of pdf * dx ≈ 1.0 (numerical)
- `test_weibull_cdf_exponential`: weibull(k=1, lambda) = exponential(1/lambda)
- `test_weibull_pdf_mode`: verify mode location for k > 1
- `test_determinism`: all operations bit-identical

### Integration Tests

Create `tests/audit_phase_b/test_b6_fft_distributions.rs` -- 15+ tests.

---

## Sub-Sprint B7: Non-parametric Tests & Multiple Comparisons

### Goal

Add Tukey HSD post-hoc, Mann-Whitney U, Kruskal-Wallis H, Wilcoxon signed-rank,
Bonferroni correction, Benjamini-Hochberg FDR, and logistic regression to
`crates/cjc-runtime/src/hypothesis.rs`. These fill the "no non-parametric tests"
and "no logistic regression" gaps from Sprint 6.

### Functions to Implement

```rust
/// Tukey HSD pairwise comparison result.
pub struct TukeyHsdPair {
    pub group_i: usize,
    pub group_j: usize,
    pub mean_diff: f64,
    pub se: f64,         // standard error of difference
    pub q_statistic: f64,
    pub p_value: f64,    // approximated from studentized range
}

/// Tukey HSD post-hoc test: all pairwise comparisons after one-way ANOVA.
/// q = |mean_i - mean_j| / se, where se = sqrt(MSW / n_group).
/// p-value approximated via the studentized range distribution
/// (use normal CDF approximation for tractability).
/// DETERMINISM: pairs enumerated in fixed (i,j) order with i < j.
pub fn tukey_hsd(groups: &[&[f64]]) -> Result<Vec<TukeyHsdPair>, String>

/// Mann-Whitney U test result.
pub struct MannWhitneyResult {
    pub u_statistic: f64,
    pub z_score: f64,    // normal approximation
    pub p_value: f64,    // two-tailed
}

/// Mann-Whitney U test (Wilcoxon rank-sum test).
/// Non-parametric test for difference between two independent groups.
/// Computes U = R1 - n1*(n1+1)/2 where R1 = sum of ranks of group 1.
/// p-value via normal approximation: z = (U - mu) / sigma.
/// DETERMINISM: stable rank assignment with index tie-breaking.
pub fn mann_whitney(x: &[f64], y: &[f64]) -> Result<MannWhitneyResult, String>

/// Kruskal-Wallis H test result.
pub struct KruskalWallisResult {
    pub h_statistic: f64,
    pub p_value: f64,
    pub df: f64,
}

/// Kruskal-Wallis H test: non-parametric one-way ANOVA on ranks.
/// H = (12 / (N*(N+1))) * sum(R_i^2 / n_i) - 3*(N+1).
/// p-value via chi-squared approximation with k-1 df.
/// DETERMINISM: deterministic rank assignment across all groups.
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<KruskalWallisResult, String>

/// Wilcoxon signed-rank test result.
pub struct WilcoxonResult {
    pub w_statistic: f64,   // smaller of W+ and W-
    pub z_score: f64,       // normal approximation
    pub p_value: f64,       // two-tailed
}

/// Wilcoxon signed-rank test for paired data.
/// 1. Compute differences d[i] = x[i] - y[i]
/// 2. Remove zero differences
/// 3. Rank absolute differences (stable sort)
/// 4. W+ = sum of ranks where d > 0, W- = sum where d < 0
/// 5. W = min(W+, W-), z-approximation for p-value
/// DETERMINISM: stable sort of absolute differences.
pub fn wilcoxon_signed_rank(x: &[f64], y: &[f64]) -> Result<WilcoxonResult, String>

/// Bonferroni correction: adjusted_p[i] = min(p[i] * m, 1.0) where m = number of tests.
pub fn bonferroni(p_values: &[f64]) -> Vec<f64>

/// Benjamini-Hochberg FDR correction.
/// 1. Sort p-values (keeping track of original indices)
/// 2. For rank i (1-based): adjusted_p[i] = p[i] * m / i
/// 3. Enforce monotonicity: adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
///    working backwards from largest rank
/// 4. Cap at 1.0
/// DETERMINISM: stable sort with index tracking.
pub fn fdr_bh(p_values: &[f64]) -> Vec<f64>

/// Logistic regression result.
pub struct LogisticResult {
    pub coefficients: Vec<f64>,  // [intercept, beta_1, ..., beta_p]
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,      // Wald z-statistics
    pub p_values: Vec<f64>,      // two-tailed p-values from normal CDF
    pub log_likelihood: f64,
    pub aic: f64,                // -2*LL + 2*k
    pub iterations: usize,
}

/// Logistic regression via Iteratively Reweighted Least Squares (IRLS).
/// Model: P(y=1|x) = sigmoid(X * beta).
///
/// Algorithm:
/// 1. Initialize beta = zeros
/// 2. For each iteration:
///    a. mu = sigmoid(X * beta)
///    b. W = diag(mu * (1 - mu))
///    c. z = X*beta + W^{-1} * (y - mu)     (working response)
///    d. beta_new = (X^T W X)^{-1} X^T W z  (weighted least squares)
///    e. If ||beta_new - beta|| < tol, converge
/// 3. Compute Hessian = X^T W X, std_errors = sqrt(diag((X^T W X)^{-1}))
///
/// Auto-adds intercept column.
/// DETERMINISM: fixed iteration order, max 100 iterations, tol = 1e-8.
pub fn logistic_regression(
    x_flat: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
) -> Result<LogisticResult, String>
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `tukey_hsd(g1, g2, ...)` | `"tukey_hsd"` | variadic (Array...) | Array[Struct] | ALLOC |
| `mann_whitney(x, y)` | `"mann_whitney"` | 2 (Array, Array) | Struct | ALLOC |
| `kruskal_wallis(g1, g2, ...)` | `"kruskal_wallis"` | variadic (Array...) | Struct | ALLOC |
| `wilcoxon_signed_rank(x, y)` | `"wilcoxon_signed_rank"` | 2 (Array, Array) | Struct | ALLOC |
| `bonferroni(p_values)` | `"bonferroni"` | 1 (Array) | Array | ALLOC |
| `fdr_bh(p_values)` | `"fdr_bh"` | 1 (Array) | Array | ALLOC |
| `logistic_regression(X, y, n, p)` | `"logistic_regression"` | 4 (Array, Array, Int, Int) | Struct | ALLOC |

### Variadic Group Arguments Pattern

For variadic functions (tukey_hsd, kruskal_wallis), use the same dispatch
pattern as the existing `anova_oneway`:

```rust
"tukey_hsd" => {
    let groups: Vec<Vec<f64>> = args.iter()
        .map(|a| value_to_f64_vec(a))
        .collect::<Result<Vec<_>, _>>()?;
    let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
    let results = crate::hypothesis::tukey_hsd(&group_refs)?;
    let result_values: Vec<Value> = results.iter().map(|pair| {
        let mut fields = std::collections::HashMap::new();
        fields.insert("group_i".into(), Value::Int(pair.group_i as i64));
        fields.insert("group_j".into(), Value::Int(pair.group_j as i64));
        fields.insert("mean_diff".into(), Value::Float(pair.mean_diff));
        fields.insert("q_statistic".into(), Value::Float(pair.q_statistic));
        fields.insert("p_value".into(), Value::Float(pair.p_value));
        Value::Struct { name: "TukeyHsdPair".into(), fields }
    }).collect();
    Ok(Some(Value::Array(Rc::new(result_values))))
}
```

### Unit Tests

- `test_mann_whitney_identical`: identical groups → p ≈ 1.0
- `test_mann_whitney_separated`: non-overlapping groups → p < 0.05
- `test_mann_whitney_known_U`: small dataset with known U statistic
- `test_kruskal_wallis_identical`: identical groups → p ≈ 1.0
- `test_kruskal_wallis_different`: clearly different groups → p < 0.05
- `test_kruskal_wallis_two_groups_matches_mann_whitney`: consistent results
- `test_wilcoxon_no_difference`: paired data, d ≈ 0 → p ≈ 1.0
- `test_wilcoxon_clear_shift`: paired data with constant shift → p < 0.05
- `test_wilcoxon_known_W`: small dataset with known W statistic
- `test_bonferroni_basic`: [0.01, 0.04, 0.5] → [0.03, 0.12, 1.0]
- `test_bonferroni_cap`: p * m > 1.0 → capped at 1.0
- `test_fdr_bh_known`: verify against known BH-adjusted values
- `test_fdr_bh_monotonicity`: adjusted p-values monotone non-decreasing (in sorted order)
- `test_fdr_bh_preserves_order`: original order preserved in output
- `test_tukey_hsd_all_same`: all groups identical → all p ≈ 1.0
- `test_tukey_hsd_one_different`: one group different → its comparisons significant
- `test_logistic_intercept_only`: no predictors → intercept = log(p/(1-p)) of mean(y)
- `test_logistic_perfect_separation`: well-separated data → large coefficients
- `test_logistic_known_result`: known dataset with known coefficients (compare to R/statsmodels)
- `test_logistic_convergence`: verify converges within 100 iterations for well-conditioned data
- `test_determinism`: all operations bit-identical on double run

### Integration Tests

Create `tests/audit_phase_b/test_b7_nonparametric.rs` -- 15+ tests.

---

## Sub-Sprint B8: Autodiff Engine Improvements (cjc-ad/src/lib.rs)

### Goal

Add missing reverse-mode gradient operations (sin, cos, sqrt, pow, sigmoid, relu,
tanh), improve gradient accumulation with Kahan summation, and optionally add
graph optimization passes. All improvements preserve the determinism contract.

### Priority 1: Missing Ops in Reverse Mode

Add new `GradOp` variants:

```rust
pub enum GradOp {
    // ... existing variants (Input, Parameter, Add, Sub, Mul, Div, Neg,
    //     MatMul, Sum, Mean, ScalarMul, Exp, Ln, StructField, MapLookup) ...

    // NEW: Priority 1 -- transcendental ops
    Sin(usize),        // gradient: cos(x) * upstream_grad
    Cos(usize),        // gradient: -sin(x) * upstream_grad
    Sqrt(usize),       // gradient: upstream_grad / (2 * sqrt(x))
    Pow(usize, f64),   // gradient: n * x^(n-1) * upstream_grad

    // NEW: Priority 2 -- activation function ops
    Sigmoid(usize),    // gradient: sigmoid(x) * (1 - sigmoid(x)) * upstream_grad
    Relu(usize),       // gradient: (x > 0 ? 1 : 0) * upstream_grad
    TanhAct(usize),    // gradient: (1 - tanh(x)^2) * upstream_grad
}
```

### Forward Methods on GradGraph

```rust
impl GradGraph {
    /// Element-wise sine.
    pub fn sin(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.sin()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sin(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise cosine.
    pub fn cos(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.cos()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Cos(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise square root.
    pub fn sqrt(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.sqrt()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sqrt(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Element-wise power with constant exponent.
    pub fn pow(&mut self, a: usize, n: f64) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.powf(n)).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Pow(a, n),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Sigmoid(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::Relu(a),
            tensor: result,
            grad: None,
        })));
        idx
    }

    /// Tanh activation.
    pub fn tanh_act(&mut self, a: usize) -> usize {
        let a_t = self.nodes[a].borrow().tensor.clone();
        let data = a_t.to_vec();
        let result = Tensor::from_vec_unchecked(
            data.iter().map(|&x| x.tanh()).collect(),
            a_t.shape(),
        );
        let idx = self.nodes.len();
        self.nodes.push(Rc::new(RefCell::new(GradNode {
            op: GradOp::TanhAct(a),
            tensor: result,
            grad: None,
        })));
        idx
    }
}
```

### Backward Cases (add to backward() match)

```rust
GradOp::Sin(a) => {
    // d/da sin(a) = cos(a)
    let a_val = self.nodes[a].borrow().tensor.clone();
    let cos_a = Tensor::from_vec_unchecked(
        a_val.to_vec().iter().map(|&x| x.cos()).collect(),
        a_val.shape(),
    );
    let grad_a = grad.mul_elem_unchecked(&cos_a);
    accumulate_grad(&mut grads, a, &grad_a);
}
GradOp::Cos(a) => {
    // d/da cos(a) = -sin(a)
    let a_val = self.nodes[a].borrow().tensor.clone();
    let neg_sin_a = Tensor::from_vec_unchecked(
        a_val.to_vec().iter().map(|&x| -x.sin()).collect(),
        a_val.shape(),
    );
    let grad_a = grad.mul_elem_unchecked(&neg_sin_a);
    accumulate_grad(&mut grads, a, &grad_a);
}
GradOp::Sqrt(a) => {
    // d/da sqrt(a) = 1 / (2 * sqrt(a)) = 0.5 / node_tensor
    let inv_2sqrt = Tensor::from_vec_unchecked(
        node_tensor.to_vec().iter().map(|&x| 0.5 / x).collect(),
        node_tensor.shape(),
    );
    let grad_a = grad.mul_elem_unchecked(&inv_2sqrt);
    accumulate_grad(&mut grads, a, &grad_a);
}
GradOp::Pow(a, n) => {
    // d/da a^n = n * a^(n-1)
    let a_val = self.nodes[a].borrow().tensor.clone();
    let coeff = Tensor::from_vec_unchecked(
        a_val.to_vec().iter().map(|&x| n * x.powf(n - 1.0)).collect(),
        a_val.shape(),
    );
    let grad_a = grad.mul_elem_unchecked(&coeff);
    accumulate_grad(&mut grads, a, &grad_a);
}
GradOp::Sigmoid(a) => {
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    // node_tensor IS the sigmoid output
    let sig = &node_tensor;
    let one_minus = Tensor::from_vec_unchecked(
        sig.to_vec().iter().map(|&s| 1.0 - s).collect(),
        sig.shape(),
    );
    let local = sig.mul_elem_unchecked(&one_minus);
    let grad_a = grad.mul_elem_unchecked(&local);
    accumulate_grad(&mut grads, a, &grad_a);
}
GradOp::Relu(a) => {
    // relu'(x) = 1 if x > 0, else 0
    let a_val = self.nodes[a].borrow().tensor.clone();
    let mask = Tensor::from_vec_unchecked(
        a_val.to_vec().iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect(),
        a_val.shape(),
    );
    let grad_a = grad.mul_elem_unchecked(&mask);
    accumulate_grad(&mut grads, a, &grad_a);
}
GradOp::TanhAct(a) => {
    // tanh'(x) = 1 - tanh(x)^2
    // node_tensor IS the tanh output
    let t = &node_tensor;
    let one_minus_sq = Tensor::from_vec_unchecked(
        t.to_vec().iter().map(|&x| 1.0 - x * x).collect(),
        t.shape(),
    );
    let grad_a = grad.mul_elem_unchecked(&one_minus_sq);
    accumulate_grad(&mut grads, a, &grad_a);
}
```

### Priority 3: Kahan Summation in Gradient Accumulation

Replace the current `accumulate_grad` with a Kahan-stable version:

```rust
fn accumulate_grad(grads: &mut [Option<Tensor>], idx: usize, grad: &Tensor) {
    if let Some(existing) = &grads[idx] {
        // Element-wise Kahan summation for numerical stability
        let a = existing.to_vec();
        let b = grad.to_vec();
        debug_assert_eq!(a.len(), b.len());
        let result: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        grads[idx] = Some(Tensor::from_vec_unchecked(result, existing.shape()));
    } else {
        grads[idx] = Some(grad.clone());
    }
}
```

For graphs with many gradient contributions to the same node, add a
`KahanTensorAccumulator` utility:

```rust
/// Kahan-compensated tensor accumulator for deep graphs.
struct KahanTensorAccumulator {
    sum: Vec<f64>,
    comp: Vec<f64>,
    shape: Vec<usize>,
}

impl KahanTensorAccumulator {
    fn new(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            sum: vec![0.0; n],
            comp: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    fn add(&mut self, tensor: &Tensor) {
        let data = tensor.to_vec();
        for i in 0..self.sum.len() {
            let y = data[i] - self.comp[i];
            let t = self.sum[i] + y;
            self.comp[i] = (t - self.sum[i]) - y;
            self.sum[i] = t;
        }
    }

    fn finalize(self) -> Tensor {
        Tensor::from_vec_unchecked(self.sum, &self.shape)
    }
}
```

### Priority 4: Graph Optimization (Optional -- implement if time permits)

**Dead node pruning**: Before backward, walk from loss node backwards and mark
reachable nodes. Skip unreachable nodes during backward pass. This is already
partially achieved by the `grads[i].take()` → `None => continue` pattern, but
explicit pruning avoids even the iteration.

**Constant folding**: During forward construction, if both operands are `Input`
nodes (no gradient needed), compute the result eagerly and store as `Input`.
Must check that neither operand is transitively connected to any Parameter.

### Unit Tests (add to cjc-ad/src/lib.rs `mod tests`)

```rust
// Priority 1: Transcendental ops
#[test]
fn test_reverse_sin() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
    let b = g.sin(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    // d/dx sin(x) at x=0 = cos(0) = 1.0
    assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_reverse_cos() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
    let b = g.cos(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    // d/dx cos(x) at x=0 = -sin(0) = 0.0
    assert!(ga.to_vec()[0].abs() < 1e-10);
}

#[test]
fn test_reverse_sqrt() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![4.0], &[1]));
    let b = g.sqrt(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    // d/dx sqrt(x) at x=4 = 1/(2*2) = 0.25
    assert!((ga.to_vec()[0] - 0.25).abs() < 1e-10);
}

#[test]
fn test_reverse_pow() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
    let b = g.pow(a, 3.0);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    // d/dx x^3 at x=2 = 3*4 = 12.0
    assert!((ga.to_vec()[0] - 12.0).abs() < 1e-10);
}

// Priority 2: Activation ops
#[test]
fn test_reverse_sigmoid() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
    let b = g.sigmoid(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    // sigmoid(0) = 0.5, sigmoid'(0) = 0.5 * 0.5 = 0.25
    assert!((ga.to_vec()[0] - 0.25).abs() < 1e-10);
}

#[test]
fn test_reverse_relu_positive() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
    let b = g.relu(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_reverse_relu_negative() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![-1.0], &[1]));
    let b = g.relu(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    assert!(ga.to_vec()[0].abs() < 1e-10);
}

#[test]
fn test_reverse_tanh_act() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![0.0], &[1]));
    let b = g.tanh_act(a);
    g.backward(b);
    let ga = g.grad(a).unwrap();
    // tanh(0) = 0, tanh'(0) = 1 - 0^2 = 1.0
    assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
}

// Composition tests
#[test]
fn test_reverse_sin_cos_chain() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec_unchecked(vec![1.0], &[1]));
    let s = g.sin(x);
    let c = g.cos(s);
    let loss = g.sum(c);
    g.backward(loss);
    let gx = g.grad(x).unwrap();
    // d/dx cos(sin(x)) = -sin(sin(x)) * cos(x)
    let expected = -(1.0f64.sin().sin()) * 1.0f64.cos();
    assert!((gx.to_vec()[0] - expected).abs() < 1e-10);
}

// Finite-difference validation for all new ops
#[test]
fn test_all_new_ops_finite_diff() {
    let eps = 1e-7;
    let tol = 1e-5;

    // Sin
    assert!(check_grad_finite_diff(|x| x.sin(), 1.5, 1.5f64.cos(), eps, tol));
    // Cos
    assert!(check_grad_finite_diff(|x| x.cos(), 1.5, -1.5f64.sin(), eps, tol));
    // Sqrt
    assert!(check_grad_finite_diff(|x| x.sqrt(), 4.0, 0.25, eps, tol));
    // Pow
    assert!(check_grad_finite_diff(|x| x.powi(3) as f64, 2.0, 12.0, eps, tol));
    // Sigmoid
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let sig_grad = |x: f64| { let s = sig(x); s * (1.0 - s) };
    assert!(check_grad_finite_diff(sig, 0.5, sig_grad(0.5), eps, tol));
    // Tanh
    let tanh_grad = |x: f64| 1.0 - x.tanh().powi(2);
    assert!(check_grad_finite_diff(|x| x.tanh(), 0.5, tanh_grad(0.5), eps, tol));
}
```

### Integration Tests

Create `tests/audit_phase_b/test_b8_autodiff.rs` -- 15+ tests. These test the
`cjc_ad` crate directly at the Rust level (not through MIR executor), verifying:

1. Each new GradOp produces correct gradients
2. Compositions of new ops produce correct gradients
3. All gradients match finite difference approximation
4. Results are bit-identical on double run (determinism)
5. Kahan accumulation is more stable than naive for deep graphs

### Validation

```
cargo test -p cjc-ad                              # AD unit tests
cargo test --test test_audit_phase_b -- b8         # integration tests
cargo test --workspace                             # 0 regressions
```

---

## Test Plan Summary

### Directory Structure

```
tests/
  audit_phase_b/
    mod.rs                        -- pub mod declarations for B1-B8
    test_b1_weighted_stats.rs     -- 10+ tests
    test_b2_rank_correlations.rs  -- 10+ tests
    test_b3_linalg_extensions.rs  -- 15+ tests
    test_b4_ml_extensions.rs      -- 15+ tests
    test_b5_analyst_qol.rs        -- 12+ tests
    test_b6_fft_distributions.rs  -- 15+ tests
    test_b7_nonparametric.rs      -- 15+ tests
    test_b8_autodiff.rs           -- 15+ tests
  test_audit_phase_b.rs           -- mod audit_phase_b;
```

### Test Categories per File

Each test file includes:
1. **Correctness tests**: output matches known values (R, NumPy, scipy reference)
2. **Edge case tests**: empty input, single element, NaN/Inf handling
3. **Determinism tests**: run twice → assert bit-identical output
4. **Property tests**: mathematical invariants (CDF in [0,1], symmetry, etc.)

### Expected Test Count

| Sub-Sprint | Min Tests |
|------------|-----------|
| B1 | 10 |
| B2 | 10 |
| B3 | 15 |
| B4 | 15 |
| B5 | 12 |
| B6 | 15 |
| B7 | 15 |
| B8 | 15 |
| **Total** | **107+** |

---

## Regression Testing Instructions

After EACH sub-sprint, run:

```bash
# Unit tests for modified crate:
cargo test -p cjc-runtime     # stats, distributions, hypothesis, ml, fft, linalg
cargo test -p cjc-ad           # autodiff (B8 only)

# Phase B integration tests:
cargo test --test test_audit_phase_b

# All existing hardening tests (H1-H18 must remain green):
cargo test --test test_hardening

# Full workspace regression:
cargo test --workspace

# Expected: 0 failures, all H1-H18 still passing
```

If any existing test breaks, **fix the regression before proceeding**. Phase B
additions are purely additive -- no behavioral change to existing functions.

---

## Documentation Requirements

After all 8 sub-sprints, create or update:

1. **Create `docs/phase_b_changelog.md`**: List every new function by sub-sprint
   with brief descriptions and test counts.

2. **Update `docs/CJC_DataScience_Readiness_Audit.md`**: Update the sprint grades.
   Expected improvements:
   - Sprint 1: A (93) → A+ (97) -- weighted/robust stats fill remaining gaps
   - Sprint 2: A- (90) → A (95) -- rank correlations + CIs complete the picture
   - Sprint 3: B+ (85) → A (93) -- norms + Schur + matrix_exp are strong additions
   - Sprint 4: B (83) → A- (90) -- cat/stack/topk + training utilities
   - Sprint 5: A- (88) → A (94) -- case_when + percent_rank + WLS
   - Sprint 6: B (80) → A- (90) -- arbitrary FFT + distributions + nonparametric

3. **Update `docs/CJC_Feature_Capabilities.md`**: Add entries for all new builtins.

---

## Final Validation (after all sub-sprints)

```bash
# Full workspace must pass with 0 failures:
cargo test --workspace

# Run ALL test suites:
cargo test --test test_hardening         # H1-H18 all green
cargo test --test test_audit_phase_b     # B1-B8 all green

# Expected: 0 failures across the board
```

### Post-Phase B Checklist

- [ ] Every new builtin appears in `is_known_builtin()` in BOTH executors
- [ ] Every new builtin has an effect classification in `effect_registry.rs`
- [ ] Every new function has Rust unit tests AND MIR-executor integration tests
- [ ] All floating-point reductions use Kahan or Binned summation
- [ ] No `HashMap` with iteration -- only `BTreeMap` or `Vec`
- [ ] No `par_iter()` in any new code
- [ ] All new sorting uses `f64::total_cmp`
- [ ] Schur decomposition and matrix_exp produce bit-identical results on double-run
- [ ] Logistic regression converges deterministically with fixed iteration cap
- [ ] All new AD gradient ops validated against finite differences
- [ ] `cargo test --workspace` passes with 0 failures
- [ ] Documentation updated in `docs/phase_b_changelog.md`
- [ ] Audit scorecard grades updated in `docs/CJC_DataScience_Readiness_Audit.md`
