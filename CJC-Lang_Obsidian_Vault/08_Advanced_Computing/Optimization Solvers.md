---
title: Optimization Solvers
tags: [advanced, numerics]
status: Implemented
---

# Optimization Solvers

**Source**: `crates/cjc-runtime/src/optimize.rs`.

## Summary

Numerical optimization routines (not to be confused with the compiler's [[MIR Optimizer]]). Used by VQE, QAOA, ML training, and general minimization problems.

## Likely contents (from survey)

- Gradient descent family (SGD, Adam, momentum)
- Possibly line search, BFGS, L-BFGS (**Needs verification**)
- Quadratic and linear solvers

## Related

- [[Autodiff]]
- [[ML Primitives]]
- [[Linear Algebra]]
- [[Quantum Simulation]] — VQE / QAOA use optimizers
