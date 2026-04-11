---
title: ADR-0011 Parallel Matmul
tags: [adr, proposed, runtime, parallelism, determinism]
status: Proposed
date: 2025-01-01
source: docs/adr/ADR-0011-parallel-matmul-rayon.md
---

# ADR-0011 — Parallel Matmul via Rayon (Feature-Gated)

**Status:** Proposed · **Date:** 2025-01-01

## The decision (proposed)

Add `rayon` as an **optional** dependency of `cjc-runtime` behind a `parallel` feature flag. When the feature is enabled and the matrix is large enough (`m | n | k ≥ 64` by default, tunable via `CJC_MATMUL_THRESHOLD`), `Tensor::matmul` parallelizes across output cells:

```rust
result.par_iter_mut().enumerate().for_each(|(idx, cell)| {
    let mut acc = BinnedAccumulatorF64::new();
    for p in 0..k { acc.add(a[i*k + p] * b[p*n + j]); }
    *cell = acc.total();
});
```

**Serial path unchanged** — it keeps `KahanAccumulatorF64` because the overhead of Binned is unnecessary when order is fixed.

## Why this matters

- **4×–8× speedup** on typical 4–8 core workstations for 256×256 through 4096×4096 matmul.
- **Determinism preserved via [[ADR-0005 Binned Accumulator]].** Because `BinnedAccumulatorF64` is commutative, the result is bit-identical regardless of thread count or scheduling order.
- **Opt-in.** The default build has zero external deps; `--features cjc-runtime/parallel` is required to pull `rayon`.

## The accumulator divergence

Serial (Kahan) and parallel (Binned) paths produce results that differ by up to `ε · sqrt(k)` — because they are literally different algorithms. The determinism contract is therefore stated precisely:

> **Same feature flag + same input → bit-identical output across runs and thread counts.**

Tests comparing serial-vs-parallel must use approximate equality, not bit-equality.

## What this constrains

- `test_audit_parallelism_absence.rs` must become `#[cfg(not(feature = "cjc-runtime/parallel"))]` once this lands.
- Regression gate: both `cargo test --workspace` and `cargo test --workspace --features cjc-runtime/parallel` must pass.
- Future embedded / WASM targets must stay on the serial path — `rayon` is not `no_std`.

## Related

- [[ADR-0002 Kahan Accumulator]] — serial path
- [[ADR-0005 Binned Accumulator]] — the mathematical prerequisite for this ADR
- [[Tensor Runtime]]
- [[Determinism Contract]]
- [[ADR Index]]
