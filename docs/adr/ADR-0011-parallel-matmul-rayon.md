# ADR-0011: Parallel Matmul via Rayon (Optional Feature Gate)

**Status:** Proposed
**Date:** 2025-01-01
**Deciders:** Applied Scientist, Technical Lead
**Supersedes:** none

## Context

`Tensor::matmul` in `cjc-runtime/src/lib.rs` (tensor section) uses a single-threaded triple-nested loop with `KahanAccumulatorF64` for each inner product. For matrix dimensions used in ML workloads (256×256 to 4096×4096), this is the primary performance bottleneck.

**Benchmarks (estimated, pre-implementation):**
- 256×256 matmul: ~50ms single-threaded
- 4096×4096 matmul: ~3200ms single-threaded (blocking inference)

**Determinism constraint:** CJC's reproducibility contract requires that `matmul(A, B)` produces bit-identical results regardless of:
1. Number of threads used
2. Thread scheduling order
3. Execution runs

`KahanAccumulatorF64` violates this for parallel reduction because the compensation term depends on addition order. `BinnedAccumulatorF64` (ADR-0005) is commutative and provides the required guarantee.

## Decision

Add `rayon` as an **optional feature gate** (`parallel`) to `cjc-runtime`:

```toml
# crates/cjc-runtime/Cargo.toml
[dependencies.rayon]
version = "1"
optional = true

[features]
parallel = ["rayon"]
```

In `Tensor::matmul`, activate parallel outer-loop reduction for matrices where `m * n * k >= THRESHOLD^3` (default: 64):

```rust
#[cfg(feature = "parallel")]
if m >= 64 || n >= 64 || k >= 64 {
    use rayon::prelude::*;
    use crate::accumulator::BinnedAccumulatorF64;

    result.par_iter_mut().enumerate().for_each(|(idx, cell)| {
        let i = idx / n;
        let j = idx % n;
        let mut acc = BinnedAccumulatorF64::new();
        for p in 0..k {
            acc.add(a[i * k + p] * b[p * n + j]);
        }
        *cell = acc.total();
    });
    return Tensor::from_vec(result, &[m, n]);
}
```

The threshold is tunable via the `CJC_MATMUL_THRESHOLD` environment variable.

**Accumulator switch:** Serial path retains `KahanAccumulatorF64` (lower overhead, correct for serial order). Parallel path uses `BinnedAccumulatorF64` (commutative, deterministic regardless of thread order).

**Note:** The result of serial (Kahan) and parallel (Binned) matmul for the same inputs may differ by up to `f64::EPSILON * sqrt(k)` due to different accumulation algorithms. This is documented and accepted. The determinism guarantee is: **same feature flag + same input → bit-identical output across runs and thread counts**.

## Rationale

- **Optional feature**: Tests without `--features cjc-runtime/parallel` compile and pass using the serial path. This preserves the zero-external-dependency default build.
- **Threshold-based activation**: Small matrices (< 64×64) have negligible parallelism benefit; the thread spawn overhead would dominate.
- **BinnedAccumulator for parallel**: The commutativity proof in `test_stress_parallel_invariance.rs` validates that `BinnedAccumulatorF64` produces bit-identical results with any combination order.

## Consequences

**Positive:**
- 4×–8× speedup on large matmul for common 4–8 core workstations.
- The determinism contract is maintained within each feature flag.
- Serial path is unchanged; all existing tests continue to pass.

**Known limitations:**
- Serial and parallel results differ by up to machine epsilon × sqrt(k) (different accumulation algorithms). Tests comparing serial vs parallel must use approximate equality, not bit-equality.
- `rayon` is not `no_std` compatible. Future embedded/WASM targets must use the serial path.
- `test_audit_parallelism_absence.rs` must be updated to be conditional on `not(feature = "parallel")`.

## Implementation Notes

- Crates affected: `cjc-runtime`
- Files: `crates/cjc-runtime/Cargo.toml` (add rayon dep), `crates/cjc-runtime/src/lib.rs` or `src/tensor.rs` (Tensor::matmul)
- New test: `tests/audit_tests/test_audit_parallel_matmul.rs`
- Update: `tests/audit_tests/test_audit_parallelism_absence.rs` — add `#[cfg(not(feature = "cjc-runtime/parallel"))]`
- Regression gate: `cargo test --workspace` passes (serial); `cargo test --workspace --features cjc-runtime/parallel` passes (parallel)
