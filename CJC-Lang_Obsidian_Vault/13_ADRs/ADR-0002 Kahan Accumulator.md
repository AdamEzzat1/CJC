---
title: ADR-0002 Kahan Accumulator
tags: [adr, accepted, determinism, numerics]
status: Accepted
date: 2024-01-15
source: docs/adr/ADR-0002-kahan-accumulator-f64.md
---

# ADR-0002 — `KahanAccumulatorF64` for Serial Reductions

**Status:** Accepted · **Date:** 2024-01-15

## The decision

Use **Kahan compensated summation** (`KahanAccumulatorF64` in `cjc-repro`) as the standard accumulator for every serial floating-point reduction in CJC-Lang. Error bound: O(ε), independent of input length.

## Why this matters

- **Serial determinism.** Given a fixed traversal order, Kahan produces a unique, reproducible result on every run and platform.
- **Speed.** Much cheaper than exact arithmetic or double-double; adds a single compensation subtraction per element.
- **Used everywhere serial.** `sum`, `mean`, `variance`, dot products on small vectors, serial matmul paths.

## Known limit

Kahan is **order-dependent**. Two threads that add the same elements in a different sequence may diverge by the compensation term, which breaks reproducibility for parallel reductions. This limitation is what motivates [[ADR-0005 Binned Accumulator]] — a commutative variant that the parallel paths use instead.

## What this constrains

- Any reduction that may become parallel later must be written against an `Accumulator` trait so the backend can swap in `BinnedAccumulatorF64`.
- FMA (fused multiply-add) is forbidden on reduction hot paths because it can reorder rounding — see [[Determinism Contract]].

## Related

- [[Kahan Summation]]
- [[ADR-0005 Binned Accumulator]] — the commutative companion
- [[ADR-0011 Parallel Matmul]] — switches to Binned for parallel reductions
- [[Determinism Contract]]
- [[ADR Index]]
