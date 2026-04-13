---
title: Binned Accumulator
tags: [determinism, numerics]
status: Implemented
---

# Binned Accumulator

**Source**: `crates/cjc-repro/src/` — `BinnedAccumulatorF64`. Referenced in `docs/architecture/byte_first_vm_strategy.md` and `docs/spec/milestone_2_7_hybrid_summation.md`.

## Summary

A superaccumulator that bucketizes floats by their IEEE 754 exponent. Each bin receives floats of approximately the same magnitude; within a bin, addition is approximately associative; across bins, the final combine is a Kahan-stable cascade.

Per the survey, the default is **2048 bins**.

## Why

The key property: **order-invariant summation**. No matter what order you add the values in, the result is bit-identical. This is what makes parallel reductions deterministic — each worker maintains its own binned accumulator, and the final merge adds the bins together in fixed order.

## Contrast with Kahan

| Aspect | [[Kahan Summation]] | [[Binned Accumulator]] |
|---|---|---|
| Sum model | Running sum + compensation | Per-exponent buckets |
| Order sensitivity | Sensitive to reassociation (but bounded error) | Order-invariant |
| Use case | Sequential | Parallel, reorderable |
| Overhead | Small | Larger (2048 f64 slots + ancillary) |
| Accuracy | Good | Better for adversarial inputs |

## Used by

- Large-scale reductions in `crates/cjc-runtime/src/stats.rs` and `linalg.rs`
- Any reduction that might become parallel
- The [[Quantum Simulation]] amplitude accumulations (per `crates/cjc-quantum/src/lib.rs` header)

## Related

- [[Kahan Summation]]
- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Float Reassociation Policy]]
