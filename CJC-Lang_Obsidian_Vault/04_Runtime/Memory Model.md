---
title: Memory Model
tags: [runtime, memory]
status: Implemented
---

# Memory Model

Documented in `docs/memory_model_2_0.md`. This is one of the more load-bearing design notes in the project.

## Three tiers

```
┌────────────────────┐
│   Rc heap          │   escaping values, shared via reference counting
├────────────────────┤
│   Frame arena      │   per-call bump allocator, freed on return
├────────────────────┤
│   Stack            │   immediate scalars, fixed-size locals
└────────────────────┘
```

Every allocation is tagged with an [[Escape Analysis]] hint — `Stack`, `Arena`, or `Rc` — at compile time in [[MIR]]. [[cjc-mir-exec]] honors the hint at runtime.

## No mark-sweep GC

Policy A (from `memory_model_2_0.md`): **no cycles are allowed**. The language relies on reference counting without a tracing cycle collector. Back-references must use `Weak<T>`. This is a hard rule because a tracing GC would make pauses unpredictable and break determinism guarantees about latency.

## Components

| Module | Role |
|---|---|
| `crates/cjc-runtime/src/object_slab.rs` | Slab allocator for small fixed-size objects |
| `crates/cjc-runtime/src/frame_arena.rs` | Per-call bump allocator — see [[Frame Arena]] |
| `crates/cjc-runtime/src/binned_alloc.rs` | Size-class bins for large allocations — see [[Binned Allocator]] |
| `crates/cjc-runtime/src/aligned_pool.rs` | SIMD-friendly aligned buffers |
| `crates/cjc-runtime/src/tensor_pool.rs` | Hot-path tensor buffer reuse |
| `crates/cjc-runtime/src/buffer.rs` | COW buffer — see [[COW Buffers]] |
| `crates/cjc-runtime/src/scratchpad.rs` | Scratch regions for temporary computations |
| `crates/cjc-runtime/src/paged_kv.rs` | vLLM-style paged KV cache for transformer inference |

## @nogc enforcement

The `@nogc` annotation is not just a hint — it is checked statically by the [[NoGC Verifier]] via call-graph fixpoint. Any path that could lead to an `Rc` allocation fails verification.

This is what enables the performance manifesto's zero-allocation neural network inference: RNN forward passes, transformer decoder steps, and 1D/2D CNNs all run inside NoGC regions with their buffers preallocated and reused from pools.

## Tiled matmul

`crates/cjc-runtime/src/tensor_tiled.rs` provides an L2-friendly 64×64 tiled matmul for performance. The manifesto also discusses L2 tiling as a key enabler for throughput at zero allocation cost.

## Related

- [[Tensor Runtime]]
- [[COW Buffers]]
- [[Frame Arena]]
- [[Binned Allocator]]
- [[NoGC Verifier]]
- [[Escape Analysis]]
- [[Performance Profile]]
