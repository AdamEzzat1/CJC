---
title: Numerical Truth
tags: [determinism, numerics]
status: Implemented
---

# Numerical Truth

"Numerical truth" is the project's shorthand for *numerical correctness under the determinism contract*. It is more than "correct within epsilon" — it means **exactly the same bits, every time, everywhere, for any valid program**.

## The stack

| Layer | Tool |
|---|---|
| RNG | [[SplitMix64]] |
| Summation (ordered) | [[Kahan Summation]] |
| Summation (unordered) | [[Binned Accumulator]] |
| Sorting | [[Total-Cmp and NaN Ordering]] |
| Dispatch of non-determinism out of hot paths | [[Deterministic Ordering]] (BTreeMap) |
| Compiler safeguards | [[Float Reassociation Policy]] |
| Verification | [[Parity Gates]], `crates/cjc-mir/src/verify.rs` |

## Five numerical stability strategies

From `docs/CJC_Feature_Capabilities.md`, CJC-Lang uses five families of numerical stability:

1. **Kahan compensation** — for sequential floating-point sums (small compensation term tracks lost low-order bits).
2. **Pairwise summation** — tree-structured add for log(n) error growth.
3. **Binned summation** — bucket floats by exponent to avoid catastrophic cancellation.
4. **Hybrid** — combine strategies based on input size/shape (e.g., leaf-32 Kahan + pairwise upper tree).
5. **Two-pass softmax** — first pass finds max, second pass subtracts and exponentiates for numerically stable softmax.

## Why deterministic numerics matter for ML

- Training reproducibility: same seed → same weights after N episodes. This is verified by [[Chess RL Demo]] determinism tests (216 tests).
- Regression testing: if a commit changes a bit in the 47th decimal place, you know immediately.
- Cross-platform portability: identical runs on Windows, Linux, macOS, with or without SIMD.

## What CJC-Lang does NOT claim

- It does **not** claim to be the most numerically accurate language. Kahan and binned accumulators improve accuracy *as a side effect* of enforcing order-independence — that is not their primary purpose.
- It does **not** claim zero bit drift for transcendentals across platforms if the underlying `sin`/`cos` implementation differs. **Needs verification** of the exact boundary.

## Related

- [[Determinism Contract]]
- [[SplitMix64]]
- [[Kahan Summation]]
- [[Binned Accumulator]]
- [[Float Reassociation Policy]]
- [[Tensor Runtime]]
- [[Statistics and Distributions]]
