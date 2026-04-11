---
title: Sparse Linear Algebra
tags: [advanced, linalg]
status: Implemented (dense paths solid, sparse expansion on Roadmap)
---

# Sparse Linear Algebra

**Source**: `crates/cjc-runtime/src/sparse.rs`, `sparse_eigen.rs`, `sparse_solvers.rs`.

## Summary

Sparse matrix support in CJC-Lang. Includes compressed storage (CSR / CSC), sparse matmul, sparse linear solves, and sparse eigenvalue solvers.

## What exists

- **CSR / CSC** formats in `sparse.rs`
- **Sparse matmul** — builtin `sparse_matmul`
- **Sparse solvers** — Lanczos / Arnoldi style iterative solvers in `sparse_eigen.rs` / `sparse_solvers.rs`. **Needs verification** on exact algorithms.

## Roadmap

From CLAUDE.md:
> **5. SPARSE LINEAR ALGEBRA EXPANSION**
>
> Add support for sparse eigenvalue solvers (Lanczos, Arnoldi).
>
> **Constraints:** Deterministic iteration ordering, stable floating-point reductions (BinnedAccumulator).

This suggests the sparse eigensolvers either exist but are being extended, or the expansion is in progress. **Needs verification**.

## Related

- [[Linear Algebra]]
- [[Advanced Computing in CJC-Lang]]
- [[Kahan Summation]]
- [[Binned Accumulator]]
- [[Roadmap]]
