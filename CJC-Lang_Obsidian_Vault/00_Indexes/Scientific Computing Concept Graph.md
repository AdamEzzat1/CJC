---
title: Scientific Computing Concept Graph
tags: [concept-graph, scientific]
status: Graph hub
---

# Scientific Computing Concept Graph

How CJC-Lang's scientific computing layers compose.

## Bottom layer: tensors

[[Tensor Runtime]] provides the n-dimensional array type, backed by [[COW Buffers]]. Everything sits on this.

## Linear algebra

[[Linear Algebra]] adds QR, LU, Cholesky, SVD, eigensolvers, and matmul kernels. Every reduction routes through [[Kahan Summation]] or [[Binned Accumulator]]. No external BLAS — the kernels are written in [[cjc-runtime]].

## ML primitives

[[ML Primitives]] layer on top of linalg:

- Activations, losses, layers.
- Attention with [[ML Primitives]] KV cache.
- Optimizers (SGD, Adam, RMSprop).

[[Autodiff]] provides gradients via forward dual numbers and reverse-mode tape.

## Signal and statistics

- [[Signal Processing]] — FFT, convolutions.
- [[Statistics and Distributions]] — 24 distributions, sampling via [[SplitMix64]].
- [[Hypothesis Tests]] — 24 tests with exact determinism.

## Data handling

[[DataFrame DSL]] (from `cjc-data`) — tidyverse-style labeled tables with deterministic joins and aggregations.

## Advanced numerical

- [[Sparse Linear Algebra]] — `SparseCsr` / `SparseCoo` + sparse kernels.
- [[ODE Integration]] — solver infrastructure, mostly planned.
- [[Optimization Solvers]] — gradient descent, Adam, …
- [[PINN Support]] — physics-informed networks, example-driven.
- [[Quantum Simulation]] — statevector + MPS + DMRG + density + stabilizer + VQE + QAOA + QEC + QML.

## Visualization

[[Vizor]] — grammar-of-graphics, deterministic SVG/BMP output. Every plot is a reproducible fixture.

## Serialization

[[Binary Serialization]] — checkpoint models, DataFrames, tensors, or any `Value`. SHA-256 + NaN canonicalization guarantee hash-identical snapshots across runs.

## Capstone

[[Chess RL Demo]] — a full RL training pipeline that exercises tensors, ML primitives, autodiff, RNG, optimizers, and serialization end-to-end, with 216 tests proving determinism.

## Meta-library

[[Bastion]] — a doc-only spec for a 15-primitive statistical-computing library that would sit above [[ML Primitives]] and below research code.

## Related

- [[Demonstrated Scientific Computing Capabilities]]
- [[Tensor and Scientific Computing]]
- [[Advanced Computing in CJC-Lang]]
- [[CJC-Lang Knowledge Map]]
