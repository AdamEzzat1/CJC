---
title: Kahan Summation
tags: [determinism, numerics]
status: Implemented
---

# Kahan Summation

**Source**: `crates/cjc-repro/src/` — `KahanAccumulatorF64`, `KahanAccumulatorF32`, `kahan_sum_f64`, `pairwise_sum_f64`.

## Summary

Compensated summation: alongside the running sum, carry a small error term that tracks low-order bits lost to rounding. Each add reclaims those bits in the next step.

```
sum := 0
comp := 0
for x in xs {
    y = x - comp
    t = sum + y
    comp = (t - sum) - y
    sum = t
}
```

## Why

Naive `+=` loses bits when you add a small number to a large sum — the small addend rounds off. Kahan compensation keeps the running error so the *sequence* of adds has bounded error regardless of order magnitude imbalances.

## When to use which

| Situation | Tool |
|---|---|
| Sequential sum with known order | [[Kahan Summation]] |
| Parallel / unordered sum | [[Binned Accumulator]] |
| Large vector with mixed magnitudes | Pairwise + Kahan (hybrid) |
| FFT / convolution reductions | Hybrid |

See [[Numerical Truth]] for the five stability strategies.

## In practice

Every reduction builtin in `crates/cjc-runtime/src/stats.rs`, `linalg.rs`, `ml.rs` uses either Kahan or binned accumulation. `tensor.sum()` is Kahan-stable. A plain `for x in xs { total += x }` in user code is also Kahan under the hood if it routes through the `sum` builtin.

## Related

- [[Binned Accumulator]]
- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Float Reassociation Policy]]
- [[Tensor Runtime]]
