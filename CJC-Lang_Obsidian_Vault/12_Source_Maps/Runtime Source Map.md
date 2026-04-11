---
title: Runtime Source Map
tags: [source-map, runtime]
status: Grounded in crate layout
---

# Runtime Source Map

Where runtime concepts live in the source tree. Paths relative to `C:\Users\adame\CJC`.

## Core runtime (`cjc-runtime`)

| Concept | File | Hub note |
|---|---|---|
| `Value` enum | `crates/cjc-runtime/src/lib.rs` (re-exports from `value.rs`) | [[Value Model]] |
| `Buffer<T>` (COW) | `crates/cjc-runtime/src/buffer.rs` or `value.rs` | [[COW Buffers]] |
| `Tensor` + matmul + linalg | `crates/cjc-runtime/src/tensor.rs` | [[Tensor Runtime]] |
| `Scratchpad` / `AlignedPool` | `crates/cjc-runtime/src/scratchpad.rs` | [[Frame Arena]] |
| Raw kernels (`kernel` mod) | `crates/cjc-runtime/src/kernel.rs` | [[Tensor Runtime]] |
| Paged KV cache | `crates/cjc-runtime/src/paged_kv.rs` | [[ML Primitives]] |
| `GcRef` / `GcHeap` | `crates/cjc-runtime/src/gc.rs` | [[Memory Model]] |
| Sparse (`SparseCsr` / `SparseCoo`) | `crates/cjc-runtime/src/sparse.rs` | [[Sparse Linear Algebra]] |
| `DetMap` (deterministic hashmap) | `crates/cjc-runtime/src/det_map.rs` | [[Deterministic Ordering]] |
| `Bf16` / `Value` / `FnValue` | `crates/cjc-runtime/src/value.rs` | [[Value Model]] |
| `RuntimeError` | `crates/cjc-runtime/src/error.rs` | [[Diagnostics]] |
| `accumulator.rs` | Kahan / Binned runtime helpers | [[Kahan Summation]], [[Binned Accumulator]] |
| `complex.rs` | Complex number support | [[Linear Algebra]] |
| `dispatch.rs` | Operator routing | [[Dispatch Layer]] |
| `f16.rs` / `quantized.rs` | Reduced-precision types | [[Roadmap]] (S3-P1-03, S3-P2-02) |
| `builtins.rs` | All ~334 native functions | [[Builtins Catalog]], [[Wiring Pattern]] |

> **Note:** Per `docs/spec/stage3_roadmap.md` S3-P0-05, `cjc-runtime/src/lib.rs` is being split from ~4,000 lines into ≤80 lines of re-exports. The above list reflects the **target** layout; some paths may still live inline in `lib.rs` today.

## Reproducibility primitives (`cjc-repro`)

| Concept | File | Hub note |
|---|---|---|
| `SplitMix64` RNG | `crates/cjc-repro/src/rng.rs` | [[SplitMix64]] |
| `KahanAccumulatorF64` | `crates/cjc-repro/src/accumulator.rs` | [[Kahan Summation]] |
| `BinnedAccumulatorF64` | `crates/cjc-repro/src/accumulator.rs` | [[Binned Accumulator]] |

## Autodiff (`cjc-ad`)

| Concept | File | Hub note |
|---|---|---|
| Forward dual numbers | `crates/cjc-ad/src/forward.rs` | [[Autodiff]] |
| Reverse-mode tape | `crates/cjc-ad/src/reverse.rs` | [[Autodiff]] |

## Binary serialization (`cjc-snap`)

| Concept | File | Hub note |
|---|---|---|
| Serializer / deserializer | `crates/cjc-snap/src/lib.rs` | [[Binary Serialization]] |
| SHA-256 hashing | `crates/cjc-snap/src/hash.rs` (or inline) | [[Binary Serialization]] |
| NaN canonicalization | `crates/cjc-snap/src/lib.rs` | [[Binary Serialization]] |

## Dispatch layer (`cjc-dispatch`)

- `crates/cjc-dispatch/src/lib.rs` — operator dispatch used by both executors.
- See [[Dispatch Layer]].

## Regex engine (`cjc-regex`)

- `crates/cjc-regex/src/lib.rs` — Thompson NFA construction.
- See [[Regex Engine]].

## Runtime-adjacent: the NoGC boundary

The runtime is the side that makes the [[NoGC Verifier]] meaningful. The verifier proves that **within a `@nogc` function**, no path reaches a Value variant that uses `Rc`. That proof depends on:

- Which `Value` variants are GC-free (primitive scalars, stack-only tensors, arena-allocated buffers).
- Which `Value` variants are GC-backed (`Array`, `Tuple`, `Closure`, `Enum` — anything `Rc<...>`).

The list of GC-backed variants is defined by the `Value` enum in `cjc-runtime/src/value.rs`. When you add a variant, you **must** update the NoGC verifier's escape classifier in `cjc-mir/src/nogc_verify.rs`.

## Related

- [[Runtime Architecture]]
- [[Wiring Pattern]]
- [[Runtime Concept Graph]]
