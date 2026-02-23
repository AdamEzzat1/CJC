# Milestone 2.7 BLAS Expansion — "Numerical Fortress"

**Status:** Complete
**Date:** 2025-02-21
**Test Count:** 1,107 (target was 1,100+)
**Failures:** 0

---

## 1. Overview

The Numerical Fortress expansion extends the base Milestone 2.7 Hybrid Deterministic
Summation system with industrial-grade numeric type support. Every new code path
feeds results through the existing BinnedAccumulator infrastructure, preserving the
determinism guarantees established in the base milestone.

### Determinism Contract

| Guarantee | Mechanism |
|-----------|-----------|
| **Merge-order invariance** | Knuth 2Sum error-free transformation in `BinnedAccumulatorF64::merge()` with per-bin compensation array |
| **Cross-run bit-identity** | Fixed accumulation sequence within each chunk; merge order does not affect result |
| **NaN canonicalization** | All NaN payloads normalized during merge; NaN takes priority over Inf |
| **Signed-zero preservation** | IEEE 754 signed-zero semantics preserved through all arithmetic paths |

---

## 2. New Subsystems

### 2.1 Quantized BLAS (i8/i4)

**File:** `crates/cjc-runtime/src/quantized.rs`

| Component | Description |
|-----------|-------------|
| `QuantParamsI8` | Scale + zero_point for INT8 quantization; `dequantize()` promotes to f64 |
| `QuantParamsI4` | Scale + zero_point for INT4; `unpack_byte()` extracts two signed 4-bit values via shift trick `(nibble as i8) << 4 >> 4` |
| `saturating_mul_i8()` | Overflow-safe i8 multiply using `i16` intermediate |
| `saturating_dot_i8()` | Element-wise saturating dot product with `saturating_add` |
| `quantized_matmul_i8()` | Full GEMM: integer products computed in `i64`, multiplied by combined scale, fed into BinnedAccumulator |
| `quantized_dot_i8()` | Dequantize-and-accumulate dot product through BinnedAccumulator |
| `quantized_sum_i8()` | Dequantize-and-accumulate sum through BinnedAccumulator |
| `quantized_sum_i4()` | Unpacks nibble pairs, dequantizes, accumulates through BinnedAccumulator |

**Key design decision:** Integer products are computed in `i64` to avoid intermediate `f32`
rounding. The combined scale `(scale_a * scale_b)` is applied once per product before
feeding into the BinnedAccumulator. This eliminates a systematic rounding error source
present in typical quantized inference engines.

### 2.2 Complex BLAS (ComplexF64)

**File:** `crates/cjc-runtime/src/complex.rs`

| Component | Description |
|-----------|-------------|
| `ComplexF64` | 16-byte struct with `re: f64, im: f64` |
| `mul_fixed()` | Fixed-sequence multiplication: `t1=a*c, t2=b*d, t3=a*d, t4=b*c, re=t1-t2, im=t3+t4` — prevents LLVM FMA contraction |
| `complex_dot()` | Hermitian inner product with real/imaginary parts accumulated in separate BinnedAccumulators |
| `complex_sum()` | Element-wise sum through dual BinnedAccumulators |
| `complex_matmul()` | Full complex GEMM using `mul_fixed()` per element, accumulated through BinnedAccumulators |

**Key design decision:** `mul_fixed()` uses explicit temporary variables for each of the
4 multiplications and 2 additions. By never combining multiplications and additions in a
single expression, we prevent LLVM from fusing them into FMA instructions, which would
produce different results across CPU architectures (x86 vs ARM vs RISC-V).

### 2.3 f16 (Half Precision)

**File:** `crates/cjc-runtime/src/f16.rs`

| Component | Description |
|-----------|-------------|
| `F16(u16)` | 2-byte newtype implementing IEEE 754 binary16 (1 sign + 5 exponent + 10 mantissa) |
| `to_f64()` | Lossless promotion handling normals, subnormals (`mant * 2^(-24)`), zeros, inf, NaN |
| `from_f64()` | Rounding conversion with overflow→inf, underflow→zero, subnormal preservation |
| `f16_binned_sum()` | Promote to f64, accumulate via BinnedAccumulator |
| `f16_binned_dot()` | Promote pairs to f64, multiply, accumulate via BinnedAccumulator |
| `f16_matmul()` | Full GEMM with f16→f64 promotion at inner loop |

**Key design decision:** All f16 arithmetic is performed in f64 precision. The F16 type
serves purely as a storage format. Promotion happens at the point of load, ensuring that
the BinnedAccumulator always receives full-precision f64 values.

### 2.4 Type System Integration

All three new types are wired into the CJC runtime pipeline:

| Layer | Changes |
|-------|---------|
| **Value enum** (`lib.rs`) | Added `Value::F16(F16)` and `Value::Complex(ComplexF64)` variants |
| **eval** (`cjc-eval/src/lib.rs`) | Method dispatch for Complex (`re, im, abs, conj, norm_sq, add, mul`) and F16 (`to_f64, to_f32`) |
| **mir-exec** (`cjc-mir-exec/src/lib.rs`) | Identical method dispatch as eval |
| **NoGC verifier** (`nogc_verify.rs`) | Added all new builtins to `is_safe_builtin()` |

---

## 3. Test Suite Breakdown

### By File

| Test File | Tests | Category |
|-----------|-------|----------|
| `quantized.rs` (inline) | 13 | i8/i4 unit tests |
| `complex.rs` (inline) | 18 | Complex unit tests |
| `f16.rs` (inline) | 20 | f16 unit tests |
| `accumulator.rs` (inline) | 54 | Core accumulator unit tests |
| `test_quantized_blas.rs` | 36 | i8/i4 integration |
| `test_complex_blas.rs` | 33 | Complex integration |
| `test_f16_precision.rs` | 31 | f16 integration |
| `test_u64_boundary.rs` | 33 | u64 indexing |
| `test_edge_cases.rs` | 35 | NaN/Inf/signed-zero/degenerate |
| `test_numerical_fortress.rs` | 55 | Cross-cutting fortress tests |
| All other pre-existing tests | ~779 | Phases 1-6, Milestone 2.4-2.7 base |
| **Total** | **1,107** | |

### By Category

| Category | Count | Description |
|----------|-------|-------------|
| Determinism | ~80 | Bit-identical across runs, merge-order invariance, chunk-size independence |
| Precision | ~60 | Catastrophic cancellation, extreme scales, subnormal handling |
| Edge Cases | ~70 | NaN/Inf propagation, signed-zero, empty inputs, ragged rows |
| Integration | ~50 | Cross-type pipelines, dispatch context verification, Value coherence |
| Core Runtime | ~779 | Pre-existing compiler phases 1-6 |
| Performance | 1 | 10M element benchmark (naive/kahan/binned/merge) |

---

## 4. Gap Coverage

The spec identified three gap areas. All are now covered:

### Empty Ragged Rows
- `test_ragged_rows_empty_first_row` — mixed empty/non-empty rows
- `test_ragged_rows_all_empty` — all-empty matrix
- `test_ragged_rows_merge_determinism` — merge-order invariance with empty rows
- `test_ragged_complex_rows_empty` — complex-valued ragged rows

### Extreme Quantization Scales
- `test_quantized_extreme_tiny_scale` — scale = 1e-30
- `test_quantized_extreme_huge_scale` — scale = 1e20
- `test_quantized_scale_preserves_sign` — negative scale
- `test_i4_extreme_values_sum` — i4 range extremes (-8 to 7)
- `test_quantized_zero_point_extreme` — zero_point = 127 (max)

### Complex Signed-Zero Math
- `test_complex_signed_zero_add` — 0.0 + (-0.0) = 0.0
- `test_complex_signed_zero_mul_fixed` — zero × nonzero via fixed-sequence
- `test_complex_neg_zero_real_part` — negative zero in real component
- `test_complex_conj_preserves_signed_zero` — conjugate on signed zeros

---

## 5. File Manifest

### New Files (this expansion)

| File | Purpose |
|------|---------|
| `crates/cjc-runtime/src/quantized.rs` | Quantized BLAS (i8/i4) with BinnedAccumulator integration |
| `crates/cjc-runtime/src/complex.rs` | Complex BLAS with fixed-sequence arithmetic |
| `crates/cjc-runtime/src/f16.rs` | IEEE 754 binary16 with f64 promotion path |
| `tests/test_quantized_blas.rs` | Quantized integration tests |
| `tests/test_complex_blas.rs` | Complex integration tests |
| `tests/test_f16_precision.rs` | f16 integration tests |
| `tests/test_u64_boundary.rs` | u64 indexing boundary tests |
| `tests/test_edge_cases.rs` | NaN/Inf/signed-zero/degenerate edge cases |
| `tests/test_numerical_fortress.rs` | Cross-cutting fortress tests |
| `docs/spec/milestone_2_7_blas_expansion.md` | This document |

### Modified Files (this expansion)

| File | Changes |
|------|---------|
| `crates/cjc-runtime/src/lib.rs` | Added 3 module declarations, 2 Value variants |
| `crates/cjc-runtime/src/accumulator.rs` | Added per-bin comp array, Knuth 2Sum in merge (base milestone) |
| `crates/cjc-eval/src/lib.rs` | Complex and F16 method dispatch |
| `crates/cjc-mir-exec/src/lib.rs` | Complex and F16 method dispatch |
| `crates/cjc-mir/src/nogc_verify.rs` | Added 9 safe builtins |

---

## 6. Performance Notes

From `tests/bench_accumulator.rs` (10M elements):

| Strategy | Time | Relative |
|----------|------|----------|
| Naive `f64` sum | 12.3ms | 1.00× |
| Kahan compensated | 43.8ms | 3.56× |
| **Binned accumulator** | **19.2ms** | **1.56×** |
| Binned + merge (100 chunks) | 20.9ms | 1.70× |

The BinnedAccumulator delivers deterministic, order-invariant results at only 1.56×
the cost of naive summation — substantially faster than Kahan compensation while
providing stronger guarantees.

---

## 7. Architectural Invariants

1. **All accumulation paths funnel through BinnedAccumulator** — quantized, complex,
   f16, and f64 types all use the same exponent-binning infrastructure.

2. **No intermediate f32 rounding in quantized path** — integer products computed in
   i64, combined scale applied once, result fed directly to f64 BinnedAccumulator.

3. **Fixed-sequence complex arithmetic prevents FMA drift** — cross-platform
   bit-identical results for complex operations.

4. **f16 is storage-only** — all computation happens in f64 precision after promotion.

5. **Merge is exact** — Knuth 2Sum error-free transformation makes multi-way merge
   fully order-invariant. Single-accumulator add-order is near-invariant (sub-10 ULPs).
