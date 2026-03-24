# CJC Performance V2 Optimization Changelog

**Date:** 2026-03-23
**Scope:** Second pass — reduce allocations, fuse NN operations, optimize batch operations

---

## Summary

Six additional optimizations building on the V1 fixes (matmul regression, eigendecomposition, activation COW). Focused on reducing tensor allocations in hot loops and fusing multi-step neural network operations.

---

## Optimization 1: Sequential Matmul — BinnedAccumulator → Kahan

**Root cause:** Small matrices (< 64) used `BinnedAccumulatorF64` (32KB stack per accumulator) even though iteration order is fixed. No need for order-invariance.

**Fix:** Replaced with `KahanAccumulatorF64` (16 bytes). 2000x less memory per accumulator.

**File:** `crates/cjc-runtime/src/tensor.rs` — `matmul_sequential()`

| Size | Before | After | Note |
|------|--------|-------|------|
| 32×32 | ~150µs | **136µs** | Minor win (small matrices already fast) |

---

## Optimization 2: Batched Matmul — Tiled Path + Parallel Over Batches

**Root cause:** `bmm()` used BinnedAccumulator per element AND processed batches sequentially.

**Fix:**
1. Each batch now dispatches to tiled matmul engine for medium/large matrices (>=64)
2. Falls back to Kahan for small matrices
3. Parallel over batches via rayon when batch_size > 1 and matrix is large enough

**File:** `crates/cjc-runtime/src/tensor.rs` — `bmm()`

| Config | Before | After | Speedup |
|--------|--------|-------|---------|
| 4×32×32 | ~4ms (est.) | **739µs** | **~5x** |
| 8×64×64 | ~20ms (est.) | **3.7ms** | **~5x** |

---

## Optimization 3: Softmax — Zero-Copy Input

**Fix:** Use `borrow_data()` instead of `to_vec()` when tensor is contiguous, avoiding an allocation for the input buffer.

**File:** `crates/cjc-runtime/src/tensor.rs` — `softmax()`

---

## Optimization 4: Tensor::scale_add — Fused Alpha*A + Beta*B

**New method:** `scale_add(alpha, other, beta)` computes `alpha*self + beta*other` in a single pass with one allocation instead of three (mul + mul + add = 3 temporaries).

**File:** `crates/cjc-runtime/src/tensor.rs`

---

## Optimization 5: Fused LSTM Cell

**Root cause:** The original `lstm_cell()` creates 13 intermediate tensors per step: 2 linear transforms + 1 add + 4 chunks + 4 activations + 3 element-wise ops.

**Fix:** `lstm_cell_fused()` computes gates in two matmuls, then does all activations and cell updates in a single scalar loop — no intermediate tensors for the gate/activation phase.

**File:** `crates/cjc-runtime/src/ml.rs` — `lstm_cell_fused()`

| Hidden Size | Original | Fused | Speedup |
|-------------|----------|-------|---------|
| 64 | 1.45ms | **634µs** | **2.3x** |

---

## Optimization 6: Fused GRU Cell

Same pattern as LSTM. `gru_cell_fused()` does gates + activations in a single scalar loop.

**File:** `crates/cjc-runtime/src/ml.rs` — `gru_cell_fused()`

| Hidden Size | Original | Fused | Speedup |
|-------------|----------|-------|---------|
| 64 | 1.36ms | **448µs** | **3.0x** |

---

## Complete Performance Summary (V1 + V2 Combined)

### Matmul

| Size | Original | After V1 | After V2 | Total Speedup |
|------|----------|----------|----------|---------------|
| 32×32 | ~200µs | ~200µs | **136µs** | ~1.5x |
| 64×64 | ~140µs | ~132µs | ~132µs | ~1x |
| 128×128 | ~1.0ms | ~871µs | ~871µs | ~1.2x |
| 256×256 | **191ms** | **19ms** | **19ms** | **~10x** |
| 512×512 | **400ms** | **14ms** | **14ms** | **~29x** |
| 1024×1024 | **6.28s** | **121ms** | **121ms** | **~52x** |

### Neural Networks

| Operation | Original | After V1 | After V2 (Fused) | Total Speedup |
|-----------|----------|----------|-------------------|---------------|
| LSTM hidden=64 | 1.3ms | 1.3ms | **634µs** | **~2x** |
| LSTM hidden=256 | 17ms | 9ms | **~4ms** (est.) | **~4x** |
| GRU hidden=64 | 1.2ms | 1.2ms | **448µs** | **~2.7x** |
| MHA seq=64 dim=128 | 659ms | 80ms | 80ms | **~8.2x** |
| BMM 8×64×64 | ~20ms | ~20ms | **3.7ms** | **~5x** |

### Linear Algebra

| Operation | Original | After V1+V2 | Total Speedup |
|-----------|----------|-------------|---------------|
| SVD 64×64 | 132ms | **879µs** | **~150x** |
| eigh 64×64 | 66ms | **1.8ms** | **~37x** |
| QR 256×256 | 557ms | **48ms** | **~12x** |
| Cholesky 256×256 | 248ms | **13.5ms** | **~18x** |

### Activations

| Operation | Original | After V1+V2 | Total Speedup |
|-----------|----------|-------------|---------------|
| relu 100K | 1.2ms | **308µs** | **~4x** |
| sigmoid 100K | 5.2ms | **1.7ms** | **~3x** |
| tanh 100K | 10ms | **3.4ms** | **~3x** |

---

## Determinism Status

**All tests produce bit-identical results across multiple runs:**
- 23 new V2 tests — all passing
- 33 V1 benchmark tests — all passing
- Full workspace: 5,616 tests, 0 failures

---

## Future Recommendations

### High Impact
1. **Wire fused LSTM/GRU into builtins** — Currently `lstm_cell_fused` and `gru_cell_fused` are Rust API only. Wire them as `lstm_cell_fast` / `gru_cell_fast` builtins so CJC programs benefit.
2. **Fused MHA** — Same pattern: compute Q/K/V projections then do SDPA in a single fused loop.
3. **Register-blocked micro-kernel for matmul** — Current tiled matmul achieves ~19 GFLOPS at 512×512. A 6×16 register-blocked micro-kernel could reach ~35-40 GFLOPS.

### Medium Impact
4. **In-place element-wise binary ops** — COW pattern from activations can apply to add/sub/mul/div.
5. **Column-major B storage for small matmul** — Sequential matmul accesses B column-wise (stride=n). Transposing B first would improve cache hit rate.
6. **Batched softmax** — Fuse max-finding and exp-summing passes for multi-row softmax.

### Low Impact (Polish)
7. **Optional BLAS feature flag** — `feature = "blas"` to link OpenBLAS for users who need >50 GFLOPS.
8. **SIMD activation kernels** — Use AVX2 for relu/sigmoid/tanh on contiguous tensors.
9. **Memory pool for temporary tensors** — Reduce allocator pressure in training loops.

### Architecture Notes
- The fused cell pattern (matmul for gates, scalar loop for activations) is the right approach for a zero-dep language. The matmul dominates at hidden>=128; the scalar loop dominates at hidden<64.
- BinnedAccumulator should only be used for parallel reductions where summation order varies. For fixed-order loops, KahanAccumulator is sufficient and 2000x lighter.
- The COW `map_elementwise` pattern should be extended to all unary tensor operations.
