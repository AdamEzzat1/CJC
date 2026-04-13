---
title: Runtime Concept Graph
tags: [concept-graph, runtime]
status: Graph hub
---

# Runtime Concept Graph

The runtime layer and its internal relationships.

## The value stack

[[Value Model]] is the tagged-union at the center of everything. It references:

- Primitive scalars (i64, f64, bool, str) — cheap, heap-free.
- [[Tensor Runtime]] — multi-dim arrays, backed by [[COW Buffers]].
- `Array`, `Tuple`, `Closure`, `Enum` — Rc-backed aggregates.
- Sparse tensors — see [[Sparse Linear Algebra]].

## Memory hierarchy

[[Memory Model]] is the three-tier allocator:

1. **Stack** — primitive scalars, small fixed-size values.
2. **Frame arena** ([[Frame Arena]]) — per-call scratch memory, reset on return.
3. **Rc heap** — shared values; the only form of GC.

[[COW Buffers]] sit inside this hierarchy and decide *when* to materialize a new buffer on mutation.

[[NoGC Verifier]] proves (statically) that a function does not reach the Rc heap.

## Deterministic primitives

[[SplitMix64]] is the only RNG.

[[Kahan Summation]] and [[Binned Accumulator]] are the only permitted reductions over f64.

Both are reused throughout [[Tensor Runtime]], [[Linear Algebra]], [[ML Primitives]], [[Statistics and Distributions]], and [[Autodiff]].

## Dispatch

[[Dispatch Layer]] is the bridge from operator calls to typed kernels. It's what keeps [[cjc-eval]] and [[cjc-mir-exec]] bit-identical — both route through the same kernels.

## Builtins surface

[[Builtins Catalog]] lists the ~334 native functions. Every one obeys the [[Wiring Pattern]].

## Related

- [[Runtime Architecture]]
- [[Runtime Source Map]]
- [[Determinism Concept Graph]]
- [[CJC-Lang Knowledge Map]]
