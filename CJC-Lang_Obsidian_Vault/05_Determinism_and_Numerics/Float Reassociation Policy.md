---
title: Float Reassociation Policy
tags: [determinism, compiler, numerics]
status: Implemented
---

# Float Reassociation Policy

## The rule

> **The optimizer may not reassociate floating-point reductions** that carry a reduction contract annotation. No FMA in SIMD kernels. No reordering that could change bit patterns.

IEEE 754 floating-point addition is **not** associative:

```
(a + b) + c   !=   a + (b + c)    // in general, with rounding
```

Any optimizer that treats `+` as associative will produce different bit patterns on different compilations of the same program. That breaks [[Determinism Contract]] and [[Parity Gates]].

## How CJC-Lang enforces it

### In [[MIR Optimizer]]

- `crates/cjc-mir/src/reduction.rs` — annotates reductions that must preserve order.
- `crates/cjc-mir/src/verify.rs` — rejects optimizer output that violates the reduction contract.
- `crates/cjc-mir/src/optimize.rs` and `ssa_optimize.rs` — the passes are written not to reorder floating-point ops.

### In SIMD kernels

- `crates/cjc-runtime/src/tensor_simd.rs` — no FMA (fused multiply-add). FMA does a single rounding for `a * b + c` instead of two separate roundings, which changes the bit pattern. Hardware with and without FMA would produce different results. CJC-Lang always uses the two-rounding version.

### In user code

- Reduction builtins (`sum`, `mean`, `dot`, ...) route through [[Kahan Summation]] or [[Binned Accumulator]], not naive `+=`.

## What this costs

Performance. A compiler allowed to reassociate floats and emit FMA can often double throughput on modern CPUs. CJC-Lang deliberately gives up that throughput in exchange for deterministic output. See [[Language Philosophy]] — *determinism > speed*.

## Related

- [[Determinism Contract]]
- [[Kahan Summation]]
- [[Binned Accumulator]]
- [[MIR Optimizer]]
- [[Tensor Runtime]]
