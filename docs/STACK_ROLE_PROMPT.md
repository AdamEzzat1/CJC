# CJC Data Science Completeness — Stack Role Implementation Prompt

## Instructions for Use

This is a **master prompt** to be given to an AI coding assistant to implement
6 sprints of data science features for CJC. Each sprint should be run as a
separate conversation. Copy the relevant sprint section plus the "Context"
and "Wiring Pattern" sections into each conversation.

---

## Context (Include in EVERY sprint conversation)

### What CJC Is

CJC is a deterministic scientific computing language with:
- Two parallel executors: `cjc-eval` (AST tree-walk) and `cjc-mir-exec` (MIR interpreter)
- Every builtin must be registered in THREE places (the "wiring pattern")
- All floating-point reductions must use Kahan or Binned summation
- `BTreeMap`/`BTreeSet` everywhere — no `HashMap` with random iteration order
- Same input must produce bit-identical output on every run

### Workspace Layout

```
crates/
  cjc-runtime/src/
    builtins.rs        -- shared stateless builtin dispatch (BOTH executors call this)
    tensor.rs          -- Tensor type (~1340 lines, has sum/mean/matmul/attention/conv/etc.)
    linalg.rs          -- LU, QR, Cholesky, inverse (~212 lines)
    window.rs          -- window_sum/mean/min/max (~214 lines)
    accumulator.rs     -- BinnedAccumulatorF64/F32 (order-invariant summation)
    value.rs           -- Value enum (Int, Float, Bool, String, Tensor, Array, etc.)
    json.rs            -- hand-rolled JSON parse/emit
    datetime.rs        -- epoch millis UTC arithmetic
    complex.rs         -- ComplexF64 with fixed-sequence arithmetic
    sparse.rs          -- SparseCsr, SparseCoo
    f16.rs             -- IEEE 754 binary16
    quantized.rs       -- i8/i4 dequantization
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
    lib.rs             -- Forward (Dual) + Reverse (GradGraph) AD
  cjc-data/src/
    lib.rs             -- DataFrame, TidyView, Column, joins, pivots (~5500 lines)
    csv.rs             -- CSV I/O
    tidy_dispatch.rs   -- tidy_filter/select/mutate/arrange/etc.
  cjc-repro/src/
    lib.rs             -- Rng (SplitMix64), KahanAccumulatorF64, pairwise_sum
tests/
  hardening_tests/
    mod.rs             -- pub mod declarations for H1-H12
    test_h1_span_unify.rs ... test_h12_perf.rs
  test_hardening.rs    -- mod hardening_tests;
```

### The Wiring Pattern (CRITICAL — follow for every new builtin)

Every new builtin function requires changes in **exactly 4 files**:

**1. Implementation module** (e.g., `crates/cjc-runtime/src/stats.rs`):
   - Write the pure Rust function
   - Add `#[cfg(test)] mod tests` with unit tests
   - Use `KahanAccumulatorF64` from `cjc_repro` for any summation

**2. `crates/cjc-runtime/src/builtins.rs`** — add dispatch arm:
   ```rust
   // Inside dispatch_builtin() match, BEFORE the `_ => Ok(None)` catch-all:
   "variance" => {
       if args.len() != 1 { return Err("variance requires 1 argument".into()); }
       let data = value_to_f64_vec(&args[0])?;
       Ok(Some(Value::Float(crate::stats::variance(&data))))
   }
   ```

**3. `crates/cjc-mir-exec/src/lib.rs`** — add to `is_known_builtin()`:
   ```rust
   // In the big `matches!()` inside is_known_builtin():
   | "variance"
   | "sd"
   ```

**4. `crates/cjc-eval/src/lib.rs`** — add to `is_known_builtin()`:
   ```rust
   // Same list, kept in sync with mir-exec:
   | "variance"
   | "sd"
   ```

**5. `crates/cjc-types/src/effect_registry.rs`** — add effect classification:
   ```rust
   // NOTE: median, sd, variance, n_distinct are ALREADY registered (lines 290-293).
   // For NEW builtins, add before the `m` return:
   m.insert("quantile", alloc);    // allocates result
   m.insert("cor", alloc);         // allocates matrix
   ```

**6. `crates/cjc-runtime/src/lib.rs`** — add `pub mod stats;` (once per new module)

### Effect Classification Guide

| Effect | Flag | When to use |
|--------|------|-------------|
| PURE | `pure` (var already exists) | No side effects, no allocation |
| ALLOC | `alloc` | Allocates new Value (arrays, strings, tensors) |
| IO | `io` | File/network/clock access |
| NONDET | `EffectSet::new(EffectSet::NONDET)` | Result depends on external state |
| MUTATES | `mutates` | Modifies an argument in place |
| GC | `gc` | Triggers garbage collection |

Most stats/linalg builtins are `alloc` (they return new arrays/tensors).
Pure scalar returns (like `variance` returning a single Float) can use `alloc`
since that's the conservative classification.

### Testing Pattern

Every sprint creates a test file `tests/hardening_tests/test_h{N}_{name}.rs`:

```rust
//! Hardening test H{N}: {Description}

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h{N}_feature_name() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let v = variance(data);
print(v);
"#);
    assert_eq!(out, vec!["2.5"]);  // population variance
}
```

Then register in `tests/hardening_tests/mod.rs`:
```rust
pub mod test_h{N}_{name};
```

### Determinism Rules

1. **No `HashMap` with iteration** — use `BTreeMap` or `Vec` with deterministic ordering
2. **No `f64` as hash key** — use integer indices or string keys
3. **No `par_iter()` in new code** — sequential only (parallel paths already exist in tensor.rs)
4. **Kahan summation for all reductions** — use `KahanAccumulatorF64::new()` then `.add()` then `.finalize()`
5. **Deterministic sorting** — use `sort_by()` with total ordering (handle NaN explicitly)
6. **No `SystemTime`** — only `datetime_now()` is allowed to be NONDET
7. **Fixed iteration order** — for-loops over ranges, not iterators over hash structures

### Pre-existing Stubs in effect_registry.rs

These are already registered but have NO implementation:
- `"median"` (line 290) — classified as `alloc`
- `"sd"` (line 291) — classified as `alloc`
- `"variance"` (line 292) — classified as `alloc`
- `"n_distinct"` (line 293) — classified as `ALLOC | NONDET`

Do NOT re-register these. Only add implementation.

---

## Sprint 1: Descriptive Statistics (1 week)

### Goal
Create `crates/cjc-runtime/src/stats.rs` with core descriptive statistics.
All functions operate on `&[f64]` and return `f64` or `Vec<f64>`.

### Functions to Implement

```rust
// All use KahanAccumulatorF64 for summation where applicable.

/// Population variance: Σ(xi - mean)² / n
/// Uses two-pass algorithm: first pass for mean (Kahan), second for deviations.
pub fn variance(data: &[f64]) -> Result<f64, String>

/// Sample variance: Σ(xi - mean)² / (n-1)
/// Returns error if data.len() < 2.
pub fn sample_variance(data: &[f64]) -> Result<f64, String>

/// Population standard deviation: sqrt(variance)
pub fn sd(data: &[f64]) -> Result<f64, String>

/// Sample standard deviation: sqrt(sample_variance)
pub fn sample_sd(data: &[f64]) -> Result<f64, String>

/// Standard error of the mean: sd / sqrt(n)
pub fn se(data: &[f64]) -> Result<f64, String>

/// Median: middle value of sorted data.
/// For even n, average of two middle values.
/// IMPORTANT: clone and sort the data — do NOT mutate input.
/// Use f64 total_cmp for deterministic NaN handling.
pub fn median(data: &[f64]) -> Result<f64, String>

/// Quantile at probability p (0.0 to 1.0).
/// Linear interpolation between adjacent ranks (R type 7 / NumPy default).
pub fn quantile(data: &[f64], p: f64) -> Result<f64, String>

/// Interquartile range: Q3 - Q1
pub fn iqr(data: &[f64]) -> Result<f64, String>

/// Skewness (Fisher's definition): E[(X-μ)³] / σ³
pub fn skewness(data: &[f64]) -> Result<f64, String>

/// Kurtosis (excess kurtosis, Fisher's): E[(X-μ)⁴] / σ⁴ - 3
pub fn kurtosis(data: &[f64]) -> Result<f64, String>

/// Z-scores: (xi - mean) / sd for each element
pub fn z_score(data: &[f64]) -> Result<Vec<f64>, String>

/// Min-max normalization: (xi - min) / (max - min)
pub fn standardize(data: &[f64]) -> Result<Vec<f64>, String>
```

### Builtin Names for Wiring

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `variance(data)` | `"variance"` | 1 (Array) | Float | ALLOC (already registered) |
| `sd(data)` | `"sd"` | 1 (Array) | Float | ALLOC (already registered) |
| `se(data)` | `"se"` | 1 (Array) | Float | ALLOC (new) |
| `median(data)` | `"median"` | 1 (Array) | Float | ALLOC (already registered) |
| `quantile(data, p)` | `"quantile"` | 2 (Array, Float) | Float | ALLOC (new) |
| `iqr(data)` | `"iqr"` | 1 (Array) | Float | ALLOC (new) |
| `skewness(data)` | `"skewness"` | 1 (Array) | Float | ALLOC (new) |
| `kurtosis(data)` | `"kurtosis"` | 1 (Array) | Float | ALLOC (new) |
| `z_score(data)` | `"z_score"` | 1 (Array) | Array[Float] | ALLOC (new) |
| `standardize(data)` | `"standardize"` | 1 (Array) | Array[Float] | ALLOC (new) |

### Unit Tests (in stats.rs)

- `test_variance_basic`: `[2, 4, 4, 4, 5, 5, 7, 9]` → population var = 4.0
- `test_sd_basic`: same data → sd = 2.0
- `test_median_odd`: `[1, 3, 5]` → 3.0
- `test_median_even`: `[1, 2, 3, 4]` → 2.5
- `test_quantile_basic`: `[1..100]` at p=0.25, 0.5, 0.75
- `test_skewness_symmetric`: symmetric data → ~0.0
- `test_kurtosis_normal`: normal-like data → ~0.0 (excess)
- `test_z_score_basic`: verify mean ≈ 0, sd ≈ 1 of result
- `test_determinism`: same input → identical output
- `test_empty_data_error`: empty array → error
- `test_kahan_stability`: 10000 × 0.1 doesn't drift

### Hardening Tests

Create `tests/hardening_tests/test_h13_descriptive_stats.rs` with 10+ tests
running through the MIR executor pipeline using `run_mir()`.

### Validation

After implementation:
```
cargo test -p cjc-runtime stats         # unit tests
cargo test --test test_hardening -- h13  # hardening tests
cargo test --workspace                   # 0 regressions
```

---

## Sprint 2: Correlation + Inference (1 week)

### Goal
Add correlation/covariance to `stats.rs`. Create `crates/cjc-runtime/src/distributions.rs`
for probability distributions. Create `crates/cjc-runtime/src/hypothesis.rs` for
statistical tests.

### Functions: stats.rs additions

```rust
/// Pearson correlation coefficient between two arrays.
/// cor(x, y) = cov(x,y) / (sd(x) * sd(y))
/// Uses Kahan summation for cross-products.
pub fn cor(x: &[f64], y: &[f64]) -> Result<f64, String>

/// Correlation matrix for a set of variables (columns).
/// Input: &[&[f64]] (each inner slice is one variable).
/// Returns: flat Vec<f64> of n×n correlation matrix.
pub fn cor_matrix(vars: &[&[f64]]) -> Result<Vec<f64>, String>

/// Population covariance: Σ(xi-μx)(yi-μy) / n
pub fn cov(x: &[f64], y: &[f64]) -> Result<f64, String>

/// Sample covariance: Σ(xi-μx)(yi-μy) / (n-1)
pub fn sample_cov(x: &[f64], y: &[f64]) -> Result<f64, String>

/// Covariance matrix.
pub fn cov_matrix(vars: &[&[f64]]) -> Result<Vec<f64>, String>
```

### Functions: distributions.rs (NEW file)

```rust
/// Normal distribution CDF using Abramowitz & Stegun approximation (7.1.26).
/// Deterministic — no table lookups or interpolation.
/// Maximum error: |ε| < 1.5 × 10⁻⁷
pub fn normal_cdf(x: f64) -> f64

/// Normal distribution PDF: (1/√(2π)) * exp(-x²/2)
pub fn normal_pdf(x: f64) -> f64

/// Normal distribution PPF (inverse CDF / quantile function).
/// Uses rational approximation (Beasley-Springer-Moro or similar).
/// p must be in (0, 1).
pub fn normal_ppf(p: f64) -> Result<f64, String>

/// Student's t-distribution CDF.
/// Uses regularized incomplete beta function.
/// df = degrees of freedom.
pub fn t_cdf(x: f64, df: f64) -> f64

/// Chi-squared distribution CDF.
/// Uses regularized lower incomplete gamma function.
pub fn chi2_cdf(x: f64, df: f64) -> f64

/// F-distribution CDF.
/// Uses regularized incomplete beta function.
pub fn f_cdf(x: f64, df1: f64, df2: f64) -> f64

// Helper functions (private):
// - regularized_incomplete_beta(a, b, x)  -- Lentz continued fraction
// - regularized_gamma_p(a, x)             -- series expansion
// - ln_gamma(x)                           -- Lanczos approximation
```

### Functions: hypothesis.rs (NEW file)

```rust
/// One-sample t-test result.
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,      // two-tailed
    pub df: f64,           // degrees of freedom
    pub mean: f64,
    pub se: f64,
}

/// One-sample t-test: is the mean significantly different from mu?
pub fn t_test(data: &[f64], mu: f64) -> Result<TTestResult, String>

/// Two-sample independent t-test (Welch's — unequal variance).
pub fn t_test_two_sample(x: &[f64], y: &[f64]) -> Result<TTestResult, String>

/// Paired t-test.
pub fn t_test_paired(x: &[f64], y: &[f64]) -> Result<TTestResult, String>

/// Chi-squared goodness-of-fit result.
pub struct ChiSquaredResult {
    pub chi2: f64,
    pub p_value: f64,
    pub df: f64,
}

/// Chi-squared goodness-of-fit test.
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> Result<ChiSquaredResult, String>
```

### Builtin Names

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `cor(x, y)` | `"cor"` | 2 (Array, Array) | Float | ALLOC |
| `cov(x, y)` | `"cov"` | 2 (Array, Array) | Float | ALLOC |
| `normal_cdf(x)` | `"normal_cdf"` | 1 (Float) | Float | PURE |
| `normal_pdf(x)` | `"normal_pdf"` | 1 (Float) | Float | PURE |
| `normal_ppf(p)` | `"normal_ppf"` | 1 (Float) | Float | PURE |
| `t_test(data, mu)` | `"t_test"` | 2 (Array, Float) | Struct | ALLOC |
| `t_test_two(x, y)` | `"t_test_two_sample"` | 2 (Array, Array) | Struct | ALLOC |
| `chi_squared_test(obs, exp)` | `"chi_squared_test"` | 2 (Array, Array) | Struct | ALLOC |

### Return Value Convention for Struct Results

t-test and chi-squared return `Value::Struct` with named fields:
```rust
Value::Struct {
    name: "TTestResult".into(),
    fields: {
        let mut m = std::collections::HashMap::new();
        m.insert("t_statistic".into(), Value::Float(result.t_statistic));
        m.insert("p_value".into(), Value::Float(result.p_value));
        m.insert("df".into(), Value::Float(result.df));
        m
    },
}
```

### Hardening Tests

`tests/hardening_tests/test_h14_correlation_inference.rs` — 15+ tests:
- Correlation of perfectly correlated data → 1.0
- Correlation of uncorrelated data → ~0.0
- Normal CDF(0) → 0.5, CDF(1.96) ≈ 0.975
- t-test on data with known mean → non-significant
- t-test on shifted data → significant (p < 0.05)
- Chi-squared on uniform observed vs expected → non-significant
- All tests must be deterministic (double-run check)

---

## Sprint 3: Core Linear Algebra (2 weeks)

### Goal
Extend `crates/cjc-runtime/src/linalg.rs` with SVD, eigenvalues, determinant,
solve, least squares, and utility functions.

### Functions to Add to linalg.rs

```rust
impl Tensor {
    /// Determinant via LU decomposition: product of U diagonal * parity.
    pub fn det(&self) -> Result<f64, RuntimeError>

    /// Solve Ax = b via LU decomposition.
    /// self = A (n×n), b = vector (n) or matrix (n×m).
    pub fn solve(&self, b: &Tensor) -> Result<Tensor, RuntimeError>

    /// Least squares solution: min ||Ax - b||₂ via QR decomposition.
    /// self = A (m×n, m >= n), b = vector (m).
    /// Returns x (n).
    pub fn lstsq(&self, b: &Tensor) -> Result<Tensor, RuntimeError>

    /// Matrix trace: sum of diagonal elements.
    pub fn trace(&self) -> Result<f64, RuntimeError>

    /// Frobenius norm: sqrt(Σ aij²).
    pub fn norm_frobenius(&self) -> Result<f64, RuntimeError>

    /// 1-norm: max column sum of absolute values.
    pub fn norm_1(&self) -> Result<f64, RuntimeError>

    /// Infinity norm: max row sum of absolute values.
    pub fn norm_inf(&self) -> Result<f64, RuntimeError>

    /// Singular Value Decomposition: A = U Σ V^T
    /// Uses Golub-Kahan bidiagonalization + implicit QR shifts.
    ///
    /// DETERMINISM: iteration order is fixed. Convergence uses
    /// Wilkinson shift with deterministic tie-breaking. Maximum
    /// 100*n iterations with deterministic fallback.
    ///
    /// Returns (U: m×m, sigma: min(m,n) vector, Vt: n×n).
    pub fn svd(&self) -> Result<(Tensor, Vec<f64>, Tensor), RuntimeError>

    /// Eigenvalue decomposition for symmetric matrices.
    /// Uses Jacobi eigenvalue algorithm (deterministic rotation order).
    ///
    /// DETERMINISM: sweeps through (i,j) pairs in fixed row-major order.
    /// Convergence criterion: off-diagonal norm < ε * diagonal norm.
    /// Maximum 100*n² iterations.
    ///
    /// Returns (eigenvalues: Vec<f64> sorted ascending, eigenvectors: n×n).
    pub fn eigh(&self) -> Result<(Vec<f64>, Tensor), RuntimeError>

    /// Matrix rank via SVD: count of singular values > tolerance.
    /// Default tolerance: max(m,n) * eps * sigma_max.
    pub fn rank(&self) -> Result<usize, RuntimeError>

    /// Condition number via SVD: sigma_max / sigma_min.
    pub fn cond(&self) -> Result<f64, RuntimeError>

    /// Kronecker product: A ⊗ B.
    pub fn kron(&self, other: &Tensor) -> Result<Tensor, RuntimeError>
}
```

### SVD Implementation Notes

Use the **Golub-Kahan bidiagonalization** approach:
1. Householder reduce A to bidiagonal form B = U₁ᵀ A V₁
2. Apply implicit QR shifts to chase the bulge on B
3. Iterate until off-diagonal elements < ε
4. Accumulate transformations in U and V

**Determinism requirements:**
- Householder reflections: use fixed sign convention (always positive diagonal)
- QR shifts: Wilkinson shift (eigenvalue of trailing 2×2 closer to b_nn)
- Convergence test: `|b[i]| < eps * (|d[i]| + |d[i+1]|)` with fixed eps
- Deflation: when converged, partition and continue (deterministic ordering)
- Maximum iterations: `100 * min(m,n)` — return error if not converged

### Eigenvalue Implementation Notes

Use **Jacobi eigenvalue algorithm** for symmetric matrices:
1. Find largest off-diagonal element (scan in row-major order for determinism)
2. Compute Givens rotation to zero it
3. Apply rotation to A and accumulate in V
4. Repeat until off-diagonal Frobenius norm < ε

**Determinism requirements:**
- Classic Jacobi (not cyclic) — always rotate the largest off-diagonal
- Tie-breaking: if multiple elements have equal magnitude, pick smallest (i,j) in row-major order
- Eigenvalues sorted ascending in final output
- Eigenvectors normalized and sign-canonical (first nonzero component positive)

### Builtin Names

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `det(A)` | `"det"` | 1 (Tensor) | Float | ALLOC |
| `solve(A, b)` | `"solve"` | 2 (Tensor, Tensor) | Tensor | ALLOC |
| `lstsq(A, b)` | `"lstsq"` | 2 (Tensor, Tensor) | Tensor | ALLOC |
| `trace(A)` | `"trace"` | 1 (Tensor) | Float | PURE |
| `norm(A)` | `"norm_frobenius"` | 1 (Tensor) | Float | PURE |
| `svd(A)` | `"svd"` | 1 (Tensor) | Tuple(Tensor,Array,Tensor) | ALLOC |
| `eigh(A)` | `"eigh"` | 1 (Tensor) | Tuple(Array,Tensor) | ALLOC |
| `rank(A)` | `"matrix_rank"` | 1 (Tensor) | Int | ALLOC |
| `cond(A)` | `"cond"` | 1 (Tensor) | Float | ALLOC |
| `kron(A, B)` | `"kron"` | 2 (Tensor, Tensor) | Tensor | ALLOC |

### Hardening Tests

`tests/hardening_tests/test_h15_linalg.rs` — 20+ tests:
- `det` of identity → 1.0
- `det` of known 3×3 → exact value
- `solve` Ax=b then verify A*x ≈ b
- `lstsq` overdetermined system → verify residual minimized
- `svd` of identity → singular values all 1.0
- `svd` of rank-1 matrix → one nonzero singular value
- `svd` reconstruction: U * diag(sigma) * Vt ≈ A
- `eigh` of diagonal matrix → eigenvalues = diagonal
- `eigh` of symmetric 3×3 → verify A*v = λ*v
- `rank` of rank-deficient matrix
- `cond` of well-conditioned vs ill-conditioned
- `trace` = sum of eigenvalues
- `det` = product of eigenvalues
- Determinism: SVD run twice → bit-identical

---

## Sprint 4: ML Toolkit Foundation (1 week)

### Goal
Create `crates/cjc-runtime/src/ml.rs` with loss functions, optimizers, activations,
and tensor utilities.

### Activations (add as methods on Tensor in tensor.rs or ml.rs)

```rust
impl Tensor {
    /// Sigmoid: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor

    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    pub fn tanh_activation(&self) -> Tensor

    /// Leaky ReLU: max(alpha*x, x), default alpha=0.01
    pub fn leaky_relu(&self, alpha: f64) -> Tensor

    /// SiLU / Swish: x * sigmoid(x)
    pub fn silu(&self) -> Tensor

    /// Mish: x * tanh(softplus(x)) where softplus = ln(1+exp(x))
    pub fn mish(&self) -> Tensor
}
```

### Tensor Utilities (add to tensor.rs)

```rust
impl Tensor {
    /// Concatenate tensors along axis.
    pub fn cat(tensors: &[&Tensor], axis: usize) -> Result<Tensor, RuntimeError>

    /// Stack tensors along new axis.
    pub fn stack(tensors: &[&Tensor], axis: usize) -> Result<Tensor, RuntimeError>

    /// Argmax along axis (returns indices).
    pub fn argmax(&self, axis: usize) -> Result<Tensor, RuntimeError>

    /// Argmin along axis.
    pub fn argmin(&self, axis: usize) -> Result<Tensor, RuntimeError>

    /// Top-k values and indices along last axis.
    pub fn topk(&self, k: usize) -> Result<(Tensor, Vec<usize>), RuntimeError>

    /// One-hot encoding: indices (n,) + num_classes → (n, num_classes)
    pub fn one_hot(indices: &[usize], num_classes: usize) -> Result<Tensor, RuntimeError>

    /// Clip/clamp values to [min, max].
    pub fn clamp(&self, min: f64, max: f64) -> Tensor
}
```

### Loss Functions (in ml.rs)

```rust
/// Mean Squared Error: Σ(pred - target)² / n
pub fn mse_loss(pred: &[f64], target: &[f64]) -> Result<f64, String>

/// Cross-entropy loss: -Σ target * ln(pred + eps)
/// For classification with softmax outputs.
pub fn cross_entropy_loss(pred: &[f64], target: &[f64]) -> Result<f64, String>

/// Binary cross-entropy: -Σ [t*ln(p) + (1-t)*ln(1-p)]
pub fn binary_cross_entropy(pred: &[f64], target: &[f64]) -> Result<f64, String>

/// Huber loss: quadratic for small errors, linear for large.
pub fn huber_loss(pred: &[f64], target: &[f64], delta: f64) -> Result<f64, String>

/// Hinge loss: Σ max(0, 1 - target * pred), for SVM.
pub fn hinge_loss(pred: &[f64], target: &[f64]) -> Result<f64, String>
```

### Optimizers (in ml.rs)

```rust
/// SGD optimizer state.
pub struct SgdState {
    pub lr: f64,
    pub momentum: f64,
    pub velocity: Vec<f64>,  // per-parameter momentum
}

/// SGD step: param -= lr * grad (with optional momentum).
/// Modifies `params` in place. Deterministic — sequential updates.
pub fn sgd_step(params: &mut [f64], grads: &[f64], state: &mut SgdState)

/// Adam optimizer state.
pub struct AdamState {
    pub lr: f64,
    pub beta1: f64,       // default 0.9
    pub beta2: f64,       // default 0.999
    pub eps: f64,          // default 1e-8
    pub t: u64,           // timestep counter
    pub m: Vec<f64>,      // first moment
    pub v: Vec<f64>,      // second moment
}

/// Adam step. Deterministic — sequential parameter updates.
pub fn adam_step(params: &mut [f64], grads: &[f64], state: &mut AdamState)
```

### Builtin Names

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `sigmoid(t)` | `"sigmoid"` | 1 (Tensor) | Tensor | ALLOC |
| `tanh(t)` | `"tanh_activation"` | 1 (Tensor) | Tensor | ALLOC |
| `leaky_relu(t, alpha)` | `"leaky_relu"` | 2 (Tensor, Float) | Tensor | ALLOC |
| `silu(t)` | `"silu"` | 1 (Tensor) | Tensor | ALLOC |
| `argmax(t, axis)` | `"argmax"` | 2 (Tensor, Int) | Tensor | ALLOC |
| `argmin(t, axis)` | `"argmin"` | 2 (Tensor, Int) | Tensor | ALLOC |
| `one_hot(indices, n)` | `"one_hot"` | 2 (Array, Int) | Tensor | ALLOC |
| `mse_loss(pred, target)` | `"mse_loss"` | 2 (Array, Array) | Float | ALLOC |
| `cross_entropy(pred, target)` | `"cross_entropy_loss"` | 2 (Array, Array) | Float | ALLOC |
| `huber_loss(pred, target, d)` | `"huber_loss"` | 3 (Array, Array, Float) | Float | ALLOC |

### Hardening Tests

`tests/hardening_tests/test_h16_ml_toolkit.rs` — 15+ tests

---

## Sprint 5: Analyst Quality-of-Life (1 week)

### Goal
Add cumulative operations, ranking, lag/lead, case_when, histogram,
and linear regression.

### Functions: stats.rs or new cumulative.rs

```rust
/// Cumulative sum with Kahan summation.
pub fn cumsum(data: &[f64]) -> Vec<f64>

/// Cumulative product.
pub fn cumprod(data: &[f64]) -> Vec<f64>

/// Cumulative max.
pub fn cummax(data: &[f64]) -> Vec<f64>

/// Cumulative min.
pub fn cummin(data: &[f64]) -> Vec<f64>

/// Lag: shift values forward by n positions, fill with NaN.
pub fn lag(data: &[f64], n: usize) -> Vec<f64>

/// Lead: shift values backward by n positions, fill with NaN.
pub fn lead(data: &[f64], n: usize) -> Vec<f64>

/// Rank (average ties). Returns 1-based ranks.
/// DETERMINISM: uses stable sort with index tracking.
pub fn rank(data: &[f64]) -> Vec<f64>

/// Dense rank (no gaps for ties).
pub fn dense_rank(data: &[f64]) -> Vec<f64>

/// Row number (sequential, tie-broken by original position — stable).
pub fn row_number(data: &[f64]) -> Vec<f64>

/// Histogram: bin data into n equal-width bins.
/// Returns (bin_edges: Vec<f64>, counts: Vec<usize>).
pub fn histogram(data: &[f64], n_bins: usize) -> Result<(Vec<f64>, Vec<usize>), String>
```

### Linear Regression (in hypothesis.rs or new regression.rs)

```rust
pub struct LmResult {
    pub coefficients: Vec<f64>,    // [intercept, slope1, slope2, ...]
    pub std_errors: Vec<f64>,      // standard errors of coefficients
    pub t_values: Vec<f64>,        // t-statistics
    pub p_values: Vec<f64>,        // p-values (two-tailed)
    pub r_squared: f64,            // R²
    pub adj_r_squared: f64,        // adjusted R²
    pub residuals: Vec<f64>,       // y - y_hat
    pub f_statistic: f64,          // overall F-test
    pub f_p_value: f64,            // p-value for F-test
}

/// Ordinary least squares regression: y = Xβ + ε
/// x_matrix: &[f64] flattened row-major (n × p), y: &[f64] (n).
/// Adds intercept column automatically.
/// Uses QR decomposition for numerical stability.
pub fn lm(x: &[f64], y: &[f64], n: usize, p: usize) -> Result<LmResult, String>
```

### Builtin Names

| Function | Builtin name | Args | Returns | Effect |
|----------|-------------|------|---------|--------|
| `cumsum(data)` | `"cumsum"` | 1 (Array) | Array | ALLOC |
| `cumprod(data)` | `"cumprod"` | 1 (Array) | Array | ALLOC |
| `cummax(data)` | `"cummax"` | 1 (Array) | Array | ALLOC |
| `cummin(data)` | `"cummin"` | 1 (Array) | Array | ALLOC |
| `lag(data, n)` | `"lag"` | 2 (Array, Int) | Array | ALLOC |
| `lead(data, n)` | `"lead"` | 2 (Array, Int) | Array | ALLOC |
| `rank(data)` | `"rank"` | 1 (Array) | Array | ALLOC |
| `dense_rank(data)` | `"dense_rank"` | 1 (Array) | Array | ALLOC |
| `histogram(data, n)` | `"histogram"` | 2 (Array, Int) | Tuple(Array,Array) | ALLOC |
| `lm(x, y, n, p)` | `"lm"` | 4 (Array, Array, Int, Int) | Struct | ALLOC |

### Hardening Tests

`tests/hardening_tests/test_h17_analyst_qol.rs` — 15+ tests

---

## Sprint 6: Advanced (2 weeks)

### Goal
FFT, additional distributions, ANOVA, cross-validation, and ML metrics.

### Functions: fft.rs (NEW)

```rust
/// Cooley-Tukey radix-2 FFT. Input length must be power of 2.
/// Returns Vec of (re, im) pairs.
/// DETERMINISM: bit-reversal permutation is deterministic,
/// butterfly operations in fixed order.
pub fn fft(data: &[(f64, f64)]) -> Vec<(f64, f64)>

/// Inverse FFT.
pub fn ifft(data: &[(f64, f64)]) -> Vec<(f64, f64)>

/// Real-valued FFT: input is real, output is complex.
/// Zero-pads to next power of 2 if needed (deterministic).
pub fn rfft(data: &[f64]) -> Vec<(f64, f64)>

/// Power spectral density: |FFT(x)|².
pub fn psd(data: &[f64]) -> Vec<f64>
```

### Functions: distributions.rs additions

```rust
/// Student's t-distribution PPF.
pub fn t_ppf(p: f64, df: f64) -> Result<f64, String>

/// Chi-squared PPF.
pub fn chi2_ppf(p: f64, df: f64) -> Result<f64, String>

/// F-distribution PPF.
pub fn f_ppf(p: f64, df1: f64, df2: f64) -> Result<f64, String>

/// Binomial PMF: C(n,k) * p^k * (1-p)^(n-k)
pub fn binomial_pmf(k: u64, n: u64, p: f64) -> f64

/// Binomial CDF: Σ_{i=0}^{k} binomial_pmf(i, n, p)
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64

/// Poisson PMF: (λ^k * e^-λ) / k!
pub fn poisson_pmf(k: u64, lambda: f64) -> f64

/// Poisson CDF: Σ_{i=0}^{k} poisson_pmf(i, λ)
pub fn poisson_cdf(k: u64, lambda: f64) -> f64
```

### Functions: hypothesis.rs additions

```rust
pub struct AnovaResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_between: f64,
    pub df_within: f64,
    pub ss_between: f64,
    pub ss_within: f64,
}

/// One-way ANOVA: compare means across groups.
/// groups: Vec of &[f64], one per group.
pub fn anova_oneway(groups: &[&[f64]]) -> Result<AnovaResult, String>

/// F-test for equality of variances.
pub fn f_test(x: &[f64], y: &[f64]) -> Result<(f64, f64), String>
```

### Functions: ml.rs additions

```rust
pub struct ConfusionMatrix {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_count: usize,  // 'fn' is reserved
}

/// Binary confusion matrix from predicted and actual labels.
pub fn confusion_matrix(predicted: &[bool], actual: &[bool]) -> ConfusionMatrix

/// Precision: TP / (TP + FP)
pub fn precision(cm: &ConfusionMatrix) -> f64

/// Recall / sensitivity: TP / (TP + FN)
pub fn recall(cm: &ConfusionMatrix) -> f64

/// F1 score: 2 * (precision * recall) / (precision + recall)
pub fn f1_score(cm: &ConfusionMatrix) -> f64

/// Accuracy: (TP + TN) / total
pub fn accuracy(cm: &ConfusionMatrix) -> f64

/// AUC-ROC via trapezoidal rule.
/// scores: predicted probabilities, labels: true binary labels.
/// DETERMINISM: sort by score with stable sort + index tie-breaking.
pub fn auc_roc(scores: &[f64], labels: &[bool]) -> Result<f64, String>

/// K-fold cross-validation indices.
/// Returns Vec of (train_indices, test_indices) tuples.
/// DETERMINISM: uses seeded RNG for shuffling.
pub fn kfold_indices(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)>

/// Train/test split indices.
pub fn train_test_split(n: usize, test_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>)
```

### Hardening Tests

`tests/hardening_tests/test_h18_advanced.rs` — 20+ tests

---

## Final Validation (after all sprints)

```bash
# Full workspace must pass with 0 failures:
cargo test --workspace

# Run ALL hardening tests:
cargo test --test test_hardening

# Expected: H1-H18 all green, ~200+ hardening tests total
# Expected: ~2600+ total workspace tests, 0 failures
```

### Post-Sprint Checklist

After all 6 sprints, verify these are true:
- [ ] Every new builtin appears in is_known_builtin() in BOTH executors
- [ ] Every new builtin has an effect classification in effect_registry.rs
- [ ] Every new function has Rust unit tests AND MIR-executor hardening tests
- [ ] All floating-point reductions use Kahan or Binned summation
- [ ] No HashMap with iteration — only BTreeMap or Vec
- [ ] SVD and eigenvalue decompositions produce bit-identical results on double-run
- [ ] `cargo test --workspace` passes with 0 failures
- [ ] Linear regression (lm) coefficients match R/NumPy on test datasets
- [ ] Normal CDF matches known values (z=1.96 → 0.975, z=0 → 0.5)
- [ ] t-test p-values match R's t.test() on test datasets
