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

## Amendment (2026-11-10, bug P5): exact superaccumulator

The binned implementation was **not** order-invariant: within a bin it did
plain IEEE `bins[e] += v`, which rounds, so e.g. with `b = 1 + 2^-52`,
`[1, b, b]` summed to `3.0` but `[b, b, 1]` to `3.0000000000000004`; the
2Sum merge and the cancelling-overflow paths (`[MAX, MAX, -MAX]` → `Inf` or
`NaN`) were order-dependent as well.

`BinnedAccumulatorF64` (same public API) is now a fixed-point
superaccumulator (option 1 above, Kulisch / ExBLAS style): every finite f64
is an integer multiple of 2^-1074, and the accumulator keeps the exact
integer sum in 68 `i64` limbs of radix 2^32 with deferred carry
propagation; `merge` is limb-wise integer addition; `finalize` rounds the
exact sum once to nearest-even. Integer addition is exactly commutative and
associative, so the result is a function of the multiset of inputs only
(proof in the module docs of `crates/cjc-runtime/src/accumulator.rs`). The
state shrank from ~40 KiB to under 600 bytes. Special values: NaN → NaN, +Inf with
−Inf → NaN, a lone signed Inf wins, an exactly-zero sum is `+0.0`, overflow
is decided on the exact sum. Results change versus the old implementation
wherever the old one was not correctly rounded — usually by a few ulps, but
the old in-bin rounding error grew with n (on 10^6 uniform values in
[-1, 1) it was ~17 ulps off the exact sum), and cancelling-overflow inputs
change from Inf/NaN to a finite value. Cost: about 3–4x slower per `add`
than the old in-bin `+=` on 10^6-element sums (release build, noisy
machine), while `new()` + `finalize()` of a small accumulator is ~10x
cheaper (no 40 KiB zeroing). Oracle and
property tests: `tests/test_binned_order_invariance.rs`,
`tests/refs/binned_exact.py`.

`BinnedAccumulatorF32` is unchanged; its bins are exact (hence
order-invariant) only while a bin holds fewer than 2^29 values.

## Implementation Notes

- Crates affected: `cjc-runtime`
- Files: `crates/cjc-runtime/src/accumulator.rs`
- Regression gate: `cargo test --workspace` must pass with 0 failures
- See also: ADR-0011 (parallel matmul uses this accumulator)
