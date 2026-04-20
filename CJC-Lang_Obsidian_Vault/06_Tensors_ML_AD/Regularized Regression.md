---
title: Regularized Regression
tags: [stats, ml, runtime, implemented]
status: Implemented (v0.1.7, 2026-04-19)
---

# Regularized Regression

**Source**: `crates/cjc-runtime/src/builtins.rs` (lines ~4735-5289).

Three regularized linear regression estimators exposed as deterministic builtins:

- `ridge_regression` — L2-penalized least squares
- `lasso_regression` — L1-penalized least squares (coordinate descent)
- `elastic_net` — convex combination of L1 and L2 penalties

All three use **coordinate descent** with a fixed feature-iteration order (`0..n_features`) and [[Kahan Summation]] for the inner-product reductions, so the fit is bit-identical across runs. See [[Determinism Contract]].

## Signatures

```cjcl
ridge_regression(X: Tensor, y: Tensor, alpha: f64) -> RegressionResult
lasso_regression(X: Tensor, y: Tensor, alpha: f64) -> RegressionResult
elastic_net(X: Tensor, y: Tensor, alpha: f64, l1_ratio: f64) -> RegressionResult
```

- `X` — design matrix, shape `(n_samples, n_features)`
- `y` — target vector, shape `(n_samples,)`
- `alpha` — regularization strength (≥ 0)
- `l1_ratio` — mixing parameter for `elastic_net`; `0.0` → pure ridge, `1.0` → pure lasso

## Return struct

All three return a struct with the following fields:

| Field | Type | Meaning |
|---|---|---|
| `.coefficients` | Tensor | fitted coefficient vector, shape `(n_features,)` |
| `.intercept` | f64 | fitted intercept |
| `.r_squared` | f64 | coefficient of determination on the training data |
| `.converged` | Bool | `true` iff the coordinate descent loop hit the tolerance before the iteration cap |
| `.n_iter` | i64 | number of outer coordinate-descent sweeps performed |
| `.alpha` | f64 | echoes back the `alpha` used |
| `.l1_ratio` | f64 | (only on `elastic_net`) echoes back the `l1_ratio` used |

## Example

```cjcl
let X: Tensor = [| 1.0, 2.0; 2.0, 3.0; 3.0, 5.0; 4.0, 7.0 |];
let y: Tensor = [1.0, 2.0, 3.0, 4.0];

let fit = ridge_regression(X, y, 0.1);
print(fit.coefficients);
print(fit.r_squared);
print(fit.converged);
```

## Determinism notes

- **Fixed feature order**: the coordinate descent loop iterates `0..n_features` in index order on every sweep. No randomized shuffling.
- **Kahan / binned accumulation**: all inner products (`X[:, j] · r`) are accumulated with stable summation.
- **No FMA**: consistent with the broader [[Numerical Truth]] policy, SIMD paths that feed these loops avoid FMA to preserve bit-identical output across CPUs.
- **Thread-independent**: coordinate descent is inherently sequential; no parallel reduction is introduced here.

## Test coverage

`tests/test_regularized_regression.rs` — **32 tests, all passing**. Covers:

- Closed-form ridge sanity checks against small by-hand examples
- Lasso sparsity: high `alpha` drives coefficients to exactly zero
- ElasticNet boundary cases: `l1_ratio = 0.0` matches ridge, `l1_ratio = 1.0` matches lasso
- Convergence flag toggles correctly when the iteration cap is hit
- Determinism: two fits on the same inputs produce bit-identical `.coefficients` tensors
- Parity between [[cjc-eval]] and [[cjc-mir-exec]]

## Related

- [[Builtins Catalog]]
- [[Statistics and Distributions]]
- [[Linear Algebra]]
- [[ML Primitives]]
- [[Numerical Truth]]
- [[Determinism Contract]]
- [[Version History]]
