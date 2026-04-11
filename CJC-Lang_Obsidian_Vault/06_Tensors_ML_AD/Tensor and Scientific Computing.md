---
title: Tensor and Scientific Computing
tags: [tensors, ml, hub]
status: Implemented
---

# Tensor and Scientific Computing

This is the hub for the ML / tensor / scientific computing cluster. CJC-Lang's tensor + numerics story is its largest surface area by LOC and the most important feature set for the language's scientific-computing mission.

## Core pieces

| Subsystem | Note |
|---|---|
| The tensor itself | [[Tensor Runtime]] |
| Linear algebra kernels | [[Linear Algebra]] |
| ML primitives | [[ML Primitives]] |
| Signal processing | [[Signal Processing]] |
| Statistics and distributions | [[Statistics and Distributions]] |
| Hypothesis tests | [[Hypothesis Tests]] |
| Automatic differentiation | [[Autodiff]] |
| PINN / physics-informed ML | [[PINN Support]] |

## The three promises

1. **Deterministic kernels** — every reduction, matmul, and convolution produces bit-identical output. See [[Determinism Contract]].
2. **Zero-dependency** — no BLAS vendor, no LAPACK, no libm in the hot path. Everything is in [[cjc-runtime]].
3. **Allocation-free hot paths** — `@nogc`-verified functions can run a neural network forward pass without any heap allocation. See [[NoGC Verifier]] and [[Memory Model]].

## Quick tour of what works

From README, survey, and performance manifesto:

- **Tensor construction**: `zeros`, `ones`, `randn`, `eye`, tensor literals
- **Shape ops**: `reshape`, `slice`, `broadcast`, `transpose`, `squeeze`, `unsqueeze`, `einsum`
- **Linear algebra**: `matmul`, `dot`, `det`, `solve`, `lstsq`, `eigh`, `svd`, `qr`, `cholesky`, sparse matmul
- **Reductions**: `sum`, `mean`, `prod`, `min`, `max`, `cumsum`, `diff` — all Kahan/binned
- **ML layers**: `relu`, `gelu`, `sigmoid`, `softmax`, `tanh`, `attention`, `conv1d`, `conv2d`, `batch_norm`, `dropout_mask`, `embedding`
- **Optimizers**: Adam
- **Losses**: `binary_cross_entropy`, MSE, etc.
- **FFT**: `fft`, `rfft`, `ifft`, window functions
- **Statistics**: 35+ functions
- **Distributions**: 24 (PDF, CDF, PPF for normal, t, chi2, F, beta, gamma, binomial, weibull, ...)
- **Hypothesis tests**: 24 (t-test variants, ANOVA, chi-squared, Wilcoxon, ADF, Tukey HSD)
- **Autodiff**: forward (Dual) and reverse (ComputeGraph)
- **PINN**: full `crates/cjc-ad/src/pinn.rs` module

## What is NOT yet there

- **MIR-level integration of reverse-mode AD.** Forward and reverse AD both work, but the reverse-mode tape runs in the runtime, not through [[MIR]]. Integrating it at MIR is explicitly on [[Roadmap]] per CLAUDE.md.
- **GPU acceleration.** Everything is CPU.
- **Distributed training.** Single-process only.

## Related

- [[Runtime Architecture]]
- [[Scientific Computing Concept Graph]]
- [[Demonstrated Scientific Computing Capabilities]]
