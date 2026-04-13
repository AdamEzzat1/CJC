---
title: NoGC Verifier
tags: [compiler, determinism, memory]
status: Implemented
---

# NoGC Verifier

Source: `crates/cjc-mir/src/nogc_verify.rs`.

## Summary

A static verifier that proves a function marked `@nogc` (or `@no_gc`) performs **no** GC-triggering allocation. This is a call-graph analysis that fixpoints over a `may_gc` flag.

```rust
let result = cjc_mir_exec::verify_nogc(&program);
```

## Why

Determinism and tight latency bounds (e.g., neural network inference) require that hot paths never allocate in a way that could trigger tracing, compaction, or unpredictable pauses. [[Memory Model]] defines a three-tier system (stack / arena / Rc), and NoGC says *"this function stays entirely in stack or arena"*.

The performance manifesto (`docs/spec/CJC_PERFORMANCE_MANIFESTO.md`) calls this out as the enabler for zero-allocation neural network inference: 10K-step RNN sequences, transformer forward passes, and CNN operations all run inside NoGC-verified regions.

## Algorithm

1. Start with a set of known "may GC" primitives (e.g., things that grow a `Rc` heap).
2. For every function, propagate: if it calls any `may_gc` function, it's `may_gc` itself.
3. Iterate to fixpoint over the call graph.
4. For each `@nogc` function, assert `may_gc == false`. If not, emit a verifier error with a trace through the call graph.

**Conservative handling**: unknown or foreign functions are treated as `may_gc = true`. This means the set of provably NoGC functions is a sound underapproximation.

## Relationship to Escape Analysis

[[Escape Analysis]] classifies individual allocations as `Stack`, `Arena`, or `Rc`. The NoGC verifier uses the output: if any allocation in the function is `Rc` (and reaches GC), the function is `may_gc`.

## Related

- [[Escape Analysis]]
- [[Memory Model]]
- [[Frame Arena]]
- [[MIR]]
- [[Determinism Contract]]
