---
title: "ADR-0015: PINN PDE Problem Suite"
tags: [adr, physics, ml]
status: Accepted
date: 2026-04-11
---

# ADR-0015: PINN PDE Problem Suite

## Status

Accepted

## Context

CJC-Lang's PINN infrastructure (`cjc-ad/src/pinn.rs`) had a working harmonic oscillator and polynomial-regression heat equation, but lacked the canonical PDE benchmarks used by the research community (Raissi 2019, PINNacle, DeepXDE). To validate CJC-Lang as a serious scientific computing platform, we needed Burgers, Poisson, and Heat equation solvers with proper test coverage.

## Decision

### Finite differences vs analytical second-order AD

We chose **central finite differences** for PDE residuals rather than implementing graph-of-graphs second-order autodiff. Rationale:

- CJC-Lang's GradGraph is a first-order reverse-mode AD engine
- Analytical Hessians would require differentiating through the graph itself — a major architectural change affecting `GradOp`, backward traversal, and memory model
- FD is O(eps^2) accurate with eps=1e-4, sufficient for PINN training
- The same FD approach is used by early PINN papers and many production codes
- Trade-off: 3 forward passes per collocation point per spatial dimension

### Graph rebuild vs reforward for multi-dimensional inputs

For 2D-input problems (Burgers, Poisson, Heat with space-time), we **rebuild the graph each epoch** rather than using `set_tensor` + `reforward`. Rationale:

- 2D problems have ~5x more nodes per collocation point (5 FD stencil evaluations)
- Reforward indexing for variable-count collocation points is error-prone
- Rebuild is simpler and correct by construction
- Performance is acceptable for training with 50-100 collocation points
- The 1D harmonic oscillator retains the efficient reforward path as a reference

### Hard vs soft boundary enforcement

We provide **both**: soft BCs (loss-based, default) and hard BCs (`hard_bc_1d`). Rationale:

- Soft BCs are simpler and work with the existing loss framework
- Hard BCs (distance function approach) guarantee exact satisfaction
- Users choose based on problem requirements

### Builtins in executors, not in builtins.rs

PINN builtins live in `cjc-eval` and `cjc-mir-exec` rather than the shared `dispatch_builtin()` in `cjc-runtime/builtins.rs`. Rationale:

- `cjc-runtime` cannot depend on `cjc-ad` (circular dependency)
- Both executors already depend on `cjc-ad`
- The training functions themselves are shared (both executors call the same `cjc_ad::pinn::*` functions)

## Consequences

- 11 PDE solvers: Burgers, Poisson, Heat, Wave, Helmholtz, Diffusion-Reaction, Allen-Cahn, KdV, Schrödinger (NLS), Navier-Stokes 2D, Burgers 2D (all deterministic)
- 11 CJC-Lang builtins wired in both executors + inverse problem infrastructure
- `PinnDomain` enum: 8 domain geometries (Interval1D, Rectangle2D, SpaceTime1D, Disk, LShape, Polygon, Cuboid3D, SpaceTime2D)
- 4 boundary condition types: Dirichlet, Neumann, Robin, Periodic
- 9 activation functions: Tanh, Sigmoid, ReLU, GELU, SiLU, ELU, SELU, SinAct, None
- 3 optimizers: Adam, L-BFGS (two-loop recursion), TwoStageOptimizer
- `hard_bc_1d` for exact Dirichlet BC enforcement
- `adaptive_refine` for residual-based adaptive refinement
- 113 total tests (unit, property, fuzz, parity) across 4 test files + inline
- Future: if second-order AD is added, FD residuals can be replaced without changing the training API

## Related

- [[PINN Support]]
- [[PINN Benchmark Results]]
- [[Autodiff]]
