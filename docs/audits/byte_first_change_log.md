# Byte-First Audit Change Log

## Audit Date: 2026-03-15

## Production Code Changes

### 1. `crates/cjc-runtime/src/builtins.rs` - dot() builtin

**Change:** Replaced naive `.sum()` with `binned_sum_f64()` for order-invariant dot product.

**Before:**
```rust
let sum: f64 = av.iter().zip(bv.iter()).map(|(x, y)| x * y).sum();
```

**After:**
```rust
let products: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x * y).collect();
let sum = crate::accumulator::binned_sum_f64(&products);
```

**Risk:** LOW. Changes floating-point results at ULP level for inputs with catastrophic cancellation. All existing tests pass.

### 2. `crates/cjc-runtime/src/builtins.rs` - norm() builtin

**Change:** Replaced naive `.sum()` with `binned_sum_f64()` for L1, L2, and Lp norms.

**Before:**
```rust
1 => data.iter().map(|x| x.abs()).sum(),
2 => data.iter().map(|x| x * x).sum::<f64>().sqrt(),
_ => { let p = ord as f64; data.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p) }
```

**After:**
```rust
1 => { let abs_vals: Vec<f64> = data.iter().map(|x| x.abs()).collect();
       crate::accumulator::binned_sum_f64(&abs_vals) }
2 => { let sq_vals: Vec<f64> = data.iter().map(|x| x * x).collect();
       crate::accumulator::binned_sum_f64(&sq_vals).sqrt() }
_ => { let p = ord as f64;
       let pow_vals: Vec<f64> = data.iter().map(|x| x.abs().powf(p)).collect();
       crate::accumulator::binned_sum_f64(&pow_vals).powf(1.0 / p) }
```

**Risk:** LOW. Same ULP-level changes as dot().

## New Test Files

### `tests/test_byte_first.rs`
Entry point for byte_first test module.

### `tests/byte_first/mod.rs`
Module declarations for 6 test submodules.

### `tests/byte_first/test_determinism.rs` (10 tests)
- RNG determinism (SplitMix64, normal distribution, fork)
- DetMap iteration order stability
- Program-level determinism (same seed = same output)
- Different seeds produce different outputs
- BTreeMap ordering
- Kahan and binned sum determinism

### `tests/byte_first/test_numerical_hardening.rs` (12 tests)
- Dot product with binned accumulation
- Dot product eval/MIR parity
- Norm L2 correctness
- NaN/Inf propagation
- Tensor sum and matmul determinism
- BinnedAccumulator edge cases (signed zero, extreme magnitudes, Inf+NaN)
- Snap NaN canonicalization

### `tests/byte_first/test_parallel_determinism.rs` (7 tests)
- Matmul repeated-run identity
- Element-wise ops determinism
- SIMD binop large tensor determinism
- Eval/MIR tensor parity
- Tiled matmul determinism
- No-HashMap verification

### `tests/byte_first/test_vm_runtime_parity.rs` (14 tests)
- Eval vs MIR parity for: int arithmetic, float arithmetic, string ops, booleans, comparisons, arrays, structs, functions, nested calls, while loops, for loops, match, tensors, recursion

### `tests/byte_first/test_type_representations.rs` (16 tests)
- Value variant layouts (Int, Float, Bool, Bf16)
- COW semantics (Array clone, Buffer)
- DetMap ordering, hashing, growth
- BTreeMap ordering for structs
- Enum hashing determinism
- Tensor row-major layout and COW

### `tests/byte_first/test_serialization.rs` (15 tests)
- Snap encode/decode roundtrip for all major Value types
- NaN canonicalization
- Struct field sorting
- Content hash determinism and differentiation

## New Documentation Files

| File | Description |
|------|-------------|
| `docs/architecture/byte_first_type_inventory.md` | Complete Value/Type inventory with byte-level details |
| `docs/architecture/byte_first_vm_strategy.md` | VM architecture and determinism mechanisms |
| `docs/audits/byte_first_numerical_hardening.md` | Numerical audit findings and fixes |
| `docs/audits/byte_first_parallel_async_determinism.md` | Parallel/async determinism analysis |
| `docs/audits/byte_first_verification_checklist.md` | Complete verification checklist |
| `docs/audits/byte_first_change_log.md` | This file |

## Regression Summary

- **Total workspace tests:** 4,898 passed, 0 failed, 48 ignored
- **Failures after changes:** 0
- **New tests added:** 74 (byte_first suite)
- **Production code changes:** 2 files (builtins.rs), ~20 lines changed
