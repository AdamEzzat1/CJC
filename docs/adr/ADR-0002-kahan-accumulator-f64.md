# ADR-0002: KahanAccumulatorF64 for Deterministic Serial Summation

**Status:** Accepted
**Date:** 2024-01-15
**Deciders:** Applied Scientist, Technical Lead
**Supersedes:** none

## Context

Floating-point summation is inherently non-associative: `(a + b) + c ≠ a + (b + c)` in general. For a scientific computing language, this creates reproducibility problems — the same program may produce different results depending on the order of evaluation, compiler optimizations, or hardware instruction selection.

Three strategies were evaluated:
1. **Naive summation**: Simple `sum += x`. Fast but non-deterministic across platforms.
2. **Kahan compensated summation**: Two-variable algorithm that tracks rounding error compensation. O(1) per element, deterministic for a fixed traversal order.
3. **Pairwise/tree summation**: O(log n) extra memory, but deterministic regardless of input order.

## Decision

Use **Kahan compensated summation** (`KahanAccumulatorF64` in `cjc-repro`) as the standard serial accumulation strategy for tensor `.sum()`, `.mean()`, and matmul inner products.

The implementation lives in `crates/cjc-repro/src/lib.rs` and is re-exported through `cjc-runtime`.

## Rationale

- **Determinism for fixed traversal order**: Given the same input array traversed left-to-right, Kahan always produces the same result. This satisfies the CJC determinism contract for single-threaded execution.
- **Error bound**: Kahan reduces summation error from O(n·ε) to O(ε) where ε is machine epsilon. For the 100,000-element test in `test_stable_summation_via_tensor`, the result is within 1e-10 of the exact value.
- **Simplicity**: Two extra registers. No memory allocation. Compatible with the zero-external-dependency constraint.

## Consequences

**Positive:**
- Single-threaded tensor sums are reproducible across platforms.
- The `test_stable_summation_via_tensor` test validates correctness.

**Known limitations:**
- Kahan is **order-dependent**: parallel reduction with Kahan is non-deterministic because different orderings produce different compensations. For parallel matmul (see ADR-0011), `BinnedAccumulatorF64` must be used instead.
- Kahan adds ~2x arithmetic operations per accumulation step vs naive summation.

## Implementation Notes

- Crates affected: `cjc-repro`, `cjc-runtime`
- Files: `crates/cjc-repro/src/lib.rs` (`KahanAccumulatorF64`), `crates/cjc-runtime/src/lib.rs` (`Tensor::sum`, `Tensor::matmul`)
- Regression gate: `cargo test --workspace` must pass with 0 failures
