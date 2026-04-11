---
title: ADR-0005 Binned Accumulator
tags: [adr, accepted, determinism, numerics, parallelism]
status: Accepted
date: 2024-01-20
source: docs/adr/ADR-0005-exponent-binned-accumulator.md
---

# ADR-0005 — Exponent-Binned Accumulator for Order-Invariant Summation

**Status:** Accepted · **Date:** 2024-01-20

## The decision

For any reduction that might be **parallel or reordered**, use `BinnedAccumulatorF64` (exponent-binned) instead of Kahan. Lives in `cjc-runtime/src/accumulator.rs`.

Mechanism: a `[f64; 64]` bucket array indexed by each value's IEEE 754 exponent bits. Values with the same exponent are summed within a bucket (no catastrophic cancellation possible); buckets are then combined at the end. Because each element's bucket is determined by its own magnitude — not by the state of the accumulator — **addition is commutative and associative**.

## Why this matters

- **Order invariance.** `add(a); add(b)` produces the same `total()` as `add(b); add(a)`. This is the property that lets parallel workers accumulate partial sums in arbitrary thread order and still produce a bit-identical result.
- **Unlocks parallel determinism.** Without this, [[ADR-0011 Parallel Matmul]] could not honor the reproducibility contract. This ADR is its *prerequisite*.
- **Validated.** `test_stress_parallel_invariance.rs` proves bit-identity of serial vs parallel accumulation.

## Known limits

- **~3× slower than Kahan** for strictly serial reductions, which is why [[ADR-0002 Kahan Accumulator]] remains the default on serial paths.
- **Serial Kahan and parallel Binned diverge** by at most `ε · sqrt(k)` — documented and accepted. The determinism contract is "same feature flag + same input → bit-identical output", not "every accumulator agrees".

## What this constrains

- Any reduction that crosses a thread boundary must use `BinnedAccumulatorF64`.
- Tests comparing serial-vs-parallel output use approximate equality, not bit-equality.

## Related

- [[Binned Accumulator]]
- [[ADR-0002 Kahan Accumulator]] — the serial companion
- [[ADR-0011 Parallel Matmul]] — the consumer that needs this property
- [[Determinism Contract]]
- [[ADR Index]]
