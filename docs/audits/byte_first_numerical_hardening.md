# Byte-First Numerical Hardening Audit

## Audit Date: 2026-03-15

## Scope

Systematic review of all floating-point reduction operations across the CJC runtime to identify and fix order-dependent summation that could produce non-deterministic results.

## Methodology

1. Grep for `.sum()` and `sum +=` patterns across all crate source files
2. Classify each site as: (a) already using BinnedAccumulator, (b) needs upgrade, (c) acceptable as-is
3. Fix category (b) sites
4. Verify with tests

## Findings

### Production Code - Fixed (2 bugs)

#### 1. `builtins.rs::dot()` (line ~835)

**Before:** Naive `.sum()` on iterator of products
```rust
let sum: f64 = av.iter().zip(bv.iter()).map(|(x, y)| x * y).sum();
```

**After:** BinnedAccumulator via `binned_sum_f64`
```rust
let products: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x * y).collect();
let sum = crate::accumulator::binned_sum_f64(&products);
```

**Impact:** Dot products with catastrophic cancellation (e.g., `[1e16, 1.0, -1e16] . [1, 1, 1]`) now correctly return `1.0` instead of `0.0`.

#### 2. `builtins.rs::norm()` (lines ~900-908)

**Before:** Naive `.sum()` for L1, L2, and Lp norms
**After:** All three norm variants use `binned_sum_f64`

**Impact:** Norm computations on large tensors with mixed magnitudes are now order-invariant.

### Production Code - Already Hardened

| File | Function | Status | Notes |
|------|----------|--------|-------|
| `tensor.rs::sum()` | Tensor reduction | OK | Uses `binned_sum_f64` |
| `tensor.rs::matmul_parallel_mode_a()` | Matrix multiply | OK | BinnedAccumulator per element |
| `accumulator.rs` | Core accumulators | OK | 2048-bin design, stack-allocated |
| `stats.rs` | All 13 statistical functions | OK | Uses KahanAccumulator (sequential order) |
| `linalg.rs` | GEMM inner products | OK | Fixed iteration order per element |

### Production Code - Acceptable As-Is

| File | Function | Reason |
|------|----------|--------|
| `tensor_tiled.rs` | Tiled matmul inner loop | Single thread, fixed iteration order |
| `linalg.rs` | Cholesky `sum +=` | Small dimension, fixed order |
| `stats.rs` | Kahan-based stats | Sequential operations where order IS the semantics |

### Test Code - Not Changed

Multiple test files use `.sum()` for verification. These are acceptable because test assertions use the same summation both times (deterministic within a single run).

## Accumulator Architecture

### BinnedAccumulatorF64

- **2048 bins** indexed by IEEE 754 exponent field
- **Stack-allocated**: `[f64; 2048]` - no heap allocation
- **Order-invariant**: Same result regardless of input order
- **Handles special values**: NaN propagated, Inf tracked separately, subnormals in bin 0
- **Merge operation**: Allocation-free, uses 2Sum error-free transformation

### BinnedAccumulatorF32

- **256 bins** for f32 exponent range
- Same design as F64 variant

### KahanAccumulatorF64

- **Compensated summation**: Tracks error term
- **Order-dependent**: Result depends on input order (by design)
- **Used for**: Sequential operations (cumsum, scan, rolling windows)

## Verification Tests

All tests in `tests/byte_first/test_numerical_hardening.rs`:

| Test | What It Verifies |
|------|-----------------|
| `dot_product_binned` | Dot product with catastrophic cancellation returns exact result |
| `dot_product_parity` | Eval and MIR produce identical dot product |
| `norm_l2_basic` | L2 norm of [3, 4] = 5 |
| `float_nan_propagation` | 0.0/0.0 produces NaN |
| `float_infinity` | 1.0/0.0 produces +Inf |
| `tensor_sum_deterministic` | Same seed -> identical tensor sum |
| `matmul_deterministic` | Same seed -> identical matmul result |
| `binned_accumulator_signed_zero` | 0.0 + (-0.0) = 0.0 |
| `binned_accumulator_extreme_magnitudes` | 1e308 + 1e-308 + (-1e308) = 1e-308 |
| `binned_accumulator_mixed_inf_nan` | Inf + NaN = NaN |
| `snap_nan_canonical` | All NaN bit patterns canonicalize to same encoding |

## Conclusion

Two production bugs fixed. All floating-point reductions in the runtime now use appropriate accumulation strategies. The binned accumulator provides order-invariant results for unordered reductions, while Kahan summation is retained for sequential operations.
