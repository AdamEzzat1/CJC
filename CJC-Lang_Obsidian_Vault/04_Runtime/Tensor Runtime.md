---
title: Tensor Runtime
tags: [runtime, tensors, numerics]
status: Implemented
---

# Tensor Runtime

**Primary source**: `crates/cjc-runtime/src/tensor.rs` (the largest single module in the workspace, ~113K bytes), plus supporting modules: `tensor_dtype.rs`, `tensor_simd.rs`, `tensor_pool.rs`, `tensor_tiled.rs`, `sparse.rs`.

## Summary

N-dimensional arrays with explicit dtype, COW-backed storage, deterministic kernels, and aligned allocation for SIMD paths. The hot path for all of [[Linear Algebra]], [[ML Primitives]], [[Signal Processing]], [[Quantum Simulation]], and [[Autodiff]].

## dtype support

- `f64` (default for scientific workloads)
- `f32`
- `i64`
- `bool`
- `f16`, `bf16` — in `cjc-runtime/src/f16.rs`
- `Complex64` — in `cjc-runtime/src/complex.rs`
- Quantized — `cjc-runtime/src/quantized.rs`

## Construction

```cjcl
let a = Tensor.zeros([3, 3]);
let b = Tensor.ones([3, 3]);
let c = Tensor.randn([2, 4]);      // uses SplitMix64 seed
let d = Tensor.eye(3);
let e = [| 1.0, 2.0; 3.0, 4.0 |];  // tensor literal
```

## Operations

- Shape: `reshape`, `broadcast`, `transpose`, `slice`, `flatten`, `squeeze`, `unsqueeze`
- Element-wise arithmetic with broadcasting
- Reductions: `sum`, `mean`, `prod`, `min`, `max` — all using [[Kahan Summation]] or [[Binned Accumulator]]
- Matmul and friends: see [[Linear Algebra]]
- Einsum
- Tiled matmul for cache friendliness — `tensor_tiled.rs`

## Storage

- Aligned column-major buffers via `aligned_pool.rs`
- COW sharing via `buffer.rs` — see [[COW Buffers]]
- Sparse variants (CSR / CSC) via `sparse.rs` for large zero-heavy matrices
- Tensor-level pools for hot reuse — `tensor_pool.rs`

## Determinism in kernels

From the code survey and manifesto:
- No FMA in SIMD kernels (see `tensor_simd.rs`) — preserves bit-identical results across architectures.
- Reductions use Kahan or binned accumulation, never naive `+`.
- Column-major with fixed memory layout.

See [[Numerical Truth]] and [[Float Reassociation Policy]].

## Hardened matmul

From `docs/CJC_Feature_Capabilities.md`: post-hardening matmul is **allocation-free** on the hot path. This means you can run a neural network forward pass without any arena/Rc allocation once buffers are warmed — the cornerstone of the zero-allocation inference story in the performance manifesto.

## Related

- [[COW Buffers]]
- [[Memory Model]]
- [[Linear Algebra]]
- [[Kahan Summation]]
- [[Binned Accumulator]]
- [[Float Reassociation Policy]]
- [[ML Primitives]]
- [[Autodiff]]
- [[Quantum Simulation]]
