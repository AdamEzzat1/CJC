---
title: Linear Algebra
tags: [runtime, linalg, numerics]
status: Implemented
---

# Linear Algebra

**Source**: `crates/cjc-runtime/src/linalg.rs` (~63K), `sparse.rs`, `sparse_eigen.rs`, `sparse_solvers.rs`.

## Summary

A self-contained deterministic linear algebra library — no BLAS, no LAPACK, no vendor kernels.

## Dense operations

| Operation | Notes |
|---|---|
| `matmul(a, b)` | Tiled (64×64) for cache, Kahan-stable, post-hardening allocation-free |
| `dot(a, b)` | Inner product |
| `cross(a, b)` | Cross product |
| `norm(a, p)` | Lp norms |
| `det(a)` | Determinant via LU |
| `solve(a, b)` | Solve `Ax = b` via LU |
| `lstsq(a, b)` | Least squares |
| `eigh(a)` | Symmetric eigendecomposition |
| `svd(a)` | Singular value decomposition |
| `qr(a)` | QR decomposition |
| `cholesky(a)` | Cholesky for SPD matrices |
| `schur(a)` | Schur decomposition |
| `inv(a)` | Matrix inverse |

## Sparse operations

From `sparse.rs`, `sparse_eigen.rs`, `sparse_solvers.rs`:
- CSR and CSC formats
- `sparse_matmul`
- Sparse solvers
- Sparse eigensolvers (Lanczos, Arnoldi — **Needs verification** whether both are wired as user builtins; the CLAUDE.md roadmap lists them for future expansion)

## Determinism

- Tiled matmul uses a fixed block order, not a work-stealing schedule.
- Reductions use Kahan or binned accumulation.
- Sorting (e.g., eigenvalue output ordering) uses `f64::total_cmp` — see [[Total-Cmp and NaN Ordering]].
- No FMA — see [[Float Reassociation Policy]].

## Complex BLAS

`tests/test_complex_blas.rs` suggests a complex-valued BLAS path exists. Used by [[Quantum Simulation]] and some signal processing paths.

## Related

- [[Tensor Runtime]]
- [[Numerical Truth]]
- [[Kahan Summation]]
- [[Float Reassociation Policy]]
- [[ML Primitives]]
- [[Quantum Simulation]]
- [[Sparse Linear Algebra]]
