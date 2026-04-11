---
title: Performance Profile
tags: [showcase, performance]
status: Historical / needs re-verification
---

# Performance Profile

CJC-Lang's honest performance positioning, from `docs/CJC_Optimization_and_Roadmap.md`, the performance manifesto, and general observations.

## Where it's slow

- **Tree-walking interpreter** ([[cjc-eval]]): 10–100× slower than native compiled code.
- **No LLVM backend**: [[cjc-mir-exec]] is faster than tree-walk but still a register machine walking MIR, not compiled to machine code.
- **No GPU**: everything is CPU.
- **Single process**: no distributed training.

## Where it's fast enough

- **Zero-allocation inference**: with `@nogc` verification, forward passes do not hit the heap once warm. This gives deterministic microsecond latencies.
- **Tiled matmul**: `tensor_tiled.rs` provides L2-friendly blocked matmul.
- **vLLM-style KV cache**: `paged_kv.rs` for transformer inference.
- **No vendor BLAS overhead**: no library loading, no cold-start penalty.

## Reported numbers

From `docs/spec/CJC_PERFORMANCE_MANIFESTO.md` (historical, **needs verification**):

| Workload | Throughput |
|---|---|
| RNN 10K steps | ~2.5 seconds (~3,995 steps/sec) |
| Transformer | ~562 tokens/sec |
| Binary size | ~1.8 MB |

These are from a specific point in time and should be re-measured before citing externally. They illustrate the *shape* of the performance profile — acceptable for deterministic inference workloads, not competitive with PyTorch for bulk training.

## Why you'd accept the performance trade

- You need bit-identical reproducibility.
- You need deterministic latency (no GC pauses).
- You want to minimize toolchain churn (zero-dep).
- The model you're running is small/medium, not frontier-scale.

## Known optimization gaps

From the optimization/roadmap doc:

- No LLVM / native backend (P1).
- TCO not everywhere (currently only some tail positions — conditional branches on roadmap).
- Some mutable-binding validation is still being strengthened.
- Generic monomorphization completion is P0.
- Parallel matmul via optional rayon is P1.
- Vec COW completion is P1.

See [[Roadmap]].

## Related

- [[ML Primitives]]
- [[Memory Model]]
- [[NoGC Verifier]]
- [[Roadmap]]
