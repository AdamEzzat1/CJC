# ADR-0005: Exponent-Binned Accumulator for Order-Invariant Summation

**Status:** Accepted
**Date:** 2024-01-20
**Deciders:** Applied Scientist, Technical Lead
**Supersedes:** none

## Context

Kahan summation (ADR-0002) is deterministic only for a **fixed traversal order**. Parallel reduction (see ADR-0011) breaks Kahan determinism because worker threads accumulate partial sums in arbitrary order, and Kahan's error compensation term depends on the addition sequence.

For parallel matmul and future vectorized reductions, a **commutative and associative** accumulation strategy is required: one that produces bit-identical results regardless of the order partial sums are combined.

Three options evaluated:
1. **Reduce to exact arithmetic**: Use 128-bit integers or arbitrary-precision floats. Correct but 10–100x slower.
2. **Exponent-binned accumulator**: Bucket numbers by their IEEE 754 exponent; sum within each bucket (same exponent → no catastrophic cancellation); then sum across buckets. Deterministic regardless of combination order.
3. **Faith-based**: Accept small non-determinism in parallel reductions. Unacceptable for a reproducibility-focused language.

## Decision

Use **`BinnedAccumulatorF64`** (exponent-binned accumulator) in `cjc-runtime/src/accumulator.rs` as the accumulation strategy for parallel reductions.

The accumulator maintains a `[f64; 64]` bucket array indexed by the exponent bits of each input. Addition is commutative because each element lands in a fixed bucket determined by its own magnitude, not by previous elements.

## Rationale

- **Commutativity**: `acc.add(a); acc.add(b)` produces the same `total()` as `acc.add(b); acc.add(a)` — the bucket assignment depends only on the element, not on order.
- **Performance**: 64-bucket lookup is O(1) per element. ~3x overhead vs Kahan, but parallelism recovers the cost on large matrices.
- **Test coverage**: `test_stress_parallel_invariance.rs` verifies bit-identical output from parallel vs serial accumulation.

## Consequences

**Positive:**
- Parallel matmul (ADR-0011) can use `BinnedAccumulatorF64` and guarantee bit-identical results across different thread counts.
- The determinism contract extends to all reduction operations that use this accumulator.

**Known limitations:**
- `BinnedAccumulatorF64` is ~3x slower than Kahan for single-threaded use, so the serial matmul path retains `KahanAccumulatorF64`.
- Numbers requiring more than 64 distinct exponents in a single reduction may lose precision at bucket boundaries (documented as acceptable).

## Implementation Notes

- Crates affected: `cjc-runtime`
- Files: `crates/cjc-runtime/src/accumulator.rs`
- Regression gate: `cargo test --workspace` must pass with 0 failures
- See also: ADR-0011 (parallel matmul uses this accumulator)
