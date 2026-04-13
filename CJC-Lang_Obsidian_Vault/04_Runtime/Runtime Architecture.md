---
title: Runtime Architecture
tags: [runtime, hub]
status: Implemented
---

# Runtime Architecture

This is the hub for everything that happens after the compiler hands off to an executor.

**Primary crate**: `cjc-runtime` — `crates/cjc-runtime/src/` (~32,000+ LOC across ~40 modules). This is by far the largest crate in the workspace.

## Three-layer mental model

From CLAUDE.md: CJC-Lang has a **three-layer architecture**:

```
+--------------------------------------+
|          High-level (GC)             |   user objects, Rc heap
+--------------------------------------+
|          Dispatch layer              |   cjc-dispatch, builtins.rs
+--------------------------------------+
|   Core (no-GC tensors, numerics)     |   COW buffers, arena, stack
+--------------------------------------+
```

- **Core** is what [[NoGC Verifier]] protects: numeric types, tensor buffers, BLAS kernels. This layer has no runtime GC.
- **Dispatch** is the hinge ([[Dispatch Layer]]) — every call from the high level lands here and gets routed to a concrete implementation.
- **High-level** is user code with Rc-based heap allocation for things that escape local frames.

## Crates that constitute the runtime

| Crate | Role |
|---|---|
| [[cjc-runtime]] | Tensors, numerics, 200+ builtins, memory primitives |
| [[cjc-dispatch]] | Multi-dispatch resolution (shared between executors) |
| [[cjc-repro]] | Deterministic RNG, Kahan/Binned accumulators |
| [[cjc-ad]] | Forward and reverse mode autodiff |
| [[cjc-data]] | DataFrame DSL |
| [[cjc-regex]] | NFA regex |
| [[cjc-snap]] | Binary serialization |
| [[cjc-vizor]] | Visualization |
| [[cjc-quantum]] | Quantum simulation |

## Execution

At runtime, one of two executors walks the program:

- [[cjc-eval]] — tree-walking, v1
- [[cjc-mir-exec]] — register machine, v2

Both call into [[cjc-dispatch]] and `cjc-runtime::builtins` for everything non-trivial. This is what guarantees [[Parity Gates]] pass.

## Subsystems

| Subsystem | Note |
|---|---|
| Value model | [[Value Model]] |
| Tensors | [[Tensor Runtime]] |
| Memory | [[Memory Model]] |
| COW buffers | [[COW Buffers]] |
| Frame arena | [[Frame Arena]] |
| Binned allocator | [[Binned Allocator]] |
| Builtins | [[Builtins Catalog]] |
| Linear algebra | [[Linear Algebra]] |
| FFT / signal | [[Signal Processing]] |
| Statistics | [[Statistics and Distributions]] |
| Hypothesis tests | [[Hypothesis Tests]] |
| ML primitives | [[ML Primitives]] |
| Autodiff | [[Autodiff]] |
| DataFrames | [[DataFrame DSL]] |
| Visualization | [[Vizor]] |
| Quantum | [[Quantum Simulation]] |

## Design constants

- `BTreeMap` over `HashMap` everywhere — [[Deterministic Ordering]]
- No FMA in SIMD kernels — [[Float Reassociation Policy]]
- Kahan / binned summation for reductions — [[Kahan Summation]], [[Binned Accumulator]]
- SplitMix64 for all RNG — [[SplitMix64]]
- `f64::total_cmp` for sorting — [[Total-Cmp and NaN Ordering]]

See [[Determinism Contract]] for the full list.

## Related

- [[Compiler Architecture]]
- [[Determinism Contract]]
- [[Runtime Concept Graph]]
- [[Runtime Source Map]]
