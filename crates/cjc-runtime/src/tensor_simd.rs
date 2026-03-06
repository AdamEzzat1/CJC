//! SIMD acceleration for tensor operations (AVX2, 4-wide f64).
//!
//! Provides AVX2-accelerated kernels for:
//! - Element-wise binary operations (add, sub, mul, div)
//! - Element-wise unary operations (relu, abs, neg, sqrt)
//! - Inner loop of tiled matrix multiplication (axpy: c += a * b)
//!
//! # Determinism
//!
//! All SIMD paths produce **bit-identical** results to scalar paths because:
//! - IEEE 754 mandates identical rounding for scalar and SIMD add/sub/mul/div/sqrt.
//! - **No FMA** instructions are used (`_mm256_fmadd_pd` changes rounding vs
//!   separate mul+add — we explicitly avoid it).
//! - Element-wise ops are independent — no cross-lane reductions.
//! - Tiled matmul SIMD processes multiple j-columns simultaneously but each
//!   `C[i,j]` accumulates the same values in the same order.
//!
//! # Fallback
//!
//! On non-x86_64 platforms or CPUs without AVX2, all functions fall back to
//! scalar implementations that produce identical results.

/// Runtime check for AVX2 support.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    // `is_x86_feature_detected!` caches the CPUID result after the first call.
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_avx2() -> bool {
    false
}

// ── Element-wise binary operations ──────────────────────────────────────────

/// Dispatch tag for SIMD-able binary operations.
#[derive(Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Threshold above which element-wise operations are parallelized.
/// Below this, thread creation overhead dominates.
const PARALLEL_THRESHOLD: usize = 100_000;

/// SIMD-accelerated element-wise binary operation on equal-length slices.
///
/// Returns a new Vec with `out[i] = a[i] ⊕ b[i]` for the chosen operation.
/// Bit-identical to the scalar loop `a.iter().zip(b).map(|(&x, &y)| op(x, y))`.
///
/// For tensors > 100K elements (when the `parallel` feature is enabled),
/// splits work across threads with each thread using SIMD on its chunk.
/// Deterministic because each element is independent (no cross-element reduction).
pub fn simd_binop(a: &[f64], b: &[f64], op: BinOp) -> Vec<f64> {
    let n = a.len();
    debug_assert_eq!(n, b.len());

    // Parallel path for large tensors.
    #[cfg(feature = "parallel")]
    {
        if n >= PARALLEL_THRESHOLD {
            return simd_binop_parallel(a, b, op);
        }
    }

    simd_binop_sequential(a, b, op)
}

/// Sequential SIMD binop (used for small/medium tensors or as fallback).
fn simd_binop_sequential(a: &[f64], b: &[f64], op: BinOp) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![0.0f64; n];

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                match op {
                    BinOp::Add => avx2_binop::<ADD_TAG>(a, b, &mut out),
                    BinOp::Sub => avx2_binop::<SUB_TAG>(a, b, &mut out),
                    BinOp::Mul => avx2_binop::<MUL_TAG>(a, b, &mut out),
                    BinOp::Div => avx2_binop::<DIV_TAG>(a, b, &mut out),
                }
            }
            return out;
        }
    }

    // Scalar fallback
    match op {
        BinOp::Add => { for i in 0..n { out[i] = a[i] + b[i]; } }
        BinOp::Sub => { for i in 0..n { out[i] = a[i] - b[i]; } }
        BinOp::Mul => { for i in 0..n { out[i] = a[i] * b[i]; } }
        BinOp::Div => { for i in 0..n { out[i] = a[i] / b[i]; } }
    }
    out
}

/// Parallel SIMD binop for large tensors.
///
/// Splits the data into chunks, each processed by a thread using SIMD.
/// Deterministic because each element `out[i] = a[i] ⊕ b[i]` is independent.
#[cfg(feature = "parallel")]
fn simd_binop_parallel(a: &[f64], b: &[f64], op: BinOp) -> Vec<f64> {
    use rayon::prelude::*;

    let n = a.len();
    let mut out = vec![0.0f64; n];
    let chunk_size = 4096; // ~32 KB per chunk (good L1 cache fit)

    out.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * chunk_size;
            let len = out_chunk.len();
            let a_chunk = &a[start..start + len];
            let b_chunk = &b[start..start + len];

            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2() {
                    unsafe {
                        match op {
                            BinOp::Add => avx2_binop::<ADD_TAG>(a_chunk, b_chunk, out_chunk),
                            BinOp::Sub => avx2_binop::<SUB_TAG>(a_chunk, b_chunk, out_chunk),
                            BinOp::Mul => avx2_binop::<MUL_TAG>(a_chunk, b_chunk, out_chunk),
                            BinOp::Div => avx2_binop::<DIV_TAG>(a_chunk, b_chunk, out_chunk),
                        }
                    }
                    return;
                }
            }

            match op {
                BinOp::Add => { for i in 0..len { out_chunk[i] = a_chunk[i] + b_chunk[i]; } }
                BinOp::Sub => { for i in 0..len { out_chunk[i] = a_chunk[i] - b_chunk[i]; } }
                BinOp::Mul => { for i in 0..len { out_chunk[i] = a_chunk[i] * b_chunk[i]; } }
                BinOp::Div => { for i in 0..len { out_chunk[i] = a_chunk[i] / b_chunk[i]; } }
            }
        });

    out
}

// Const tags for the generic AVX2 binop function.
const ADD_TAG: u8 = 0;
const SUB_TAG: u8 = 1;
const MUL_TAG: u8 = 2;
const DIV_TAG: u8 = 3;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_binop<const OP: u8>(a: &[f64], b: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;

    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let vr = match OP {
            ADD_TAG => _mm256_add_pd(va, vb),
            SUB_TAG => _mm256_sub_pd(va, vb),
            MUL_TAG => _mm256_mul_pd(va, vb),
            _       => _mm256_div_pd(va, vb), // DIV_TAG
        };
        _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    // Scalar tail (0-3 elements)
    while i < n {
        out[i] = match OP {
            ADD_TAG => a[i] + b[i],
            SUB_TAG => a[i] - b[i],
            MUL_TAG => a[i] * b[i],
            _       => a[i] / b[i],
        };
        i += 1;
    }
}

// ── Element-wise unary operations ───────────────────────────────────────────

/// Dispatch tag for SIMD-able unary operations.
#[derive(Clone, Copy)]
pub enum UnaryOp {
    Sqrt,
    Abs,
    Neg,
    Relu,
}

/// SIMD-accelerated element-wise unary operation.
///
/// Returns a new Vec with `out[i] = f(a[i])`.
/// Bit-identical to scalar for all supported operations:
/// - `sqrt`: IEEE 754 mandates correctly-rounded sqrt.
/// - `abs`: Bit mask operation (clear sign bit).
/// - `neg`: Bit flip operation (toggle sign bit).
/// - `relu`: max(0, x) via compare + blend.
pub fn simd_unary(a: &[f64], op: UnaryOp) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![0.0f64; n];

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                match op {
                    UnaryOp::Sqrt => avx2_sqrt(a, &mut out),
                    UnaryOp::Abs  => avx2_abs(a, &mut out),
                    UnaryOp::Neg  => avx2_neg(a, &mut out),
                    UnaryOp::Relu => avx2_relu(a, &mut out),
                }
            }
            return out;
        }
    }

    // Scalar fallback
    match op {
        UnaryOp::Sqrt => { for i in 0..n { out[i] = a[i].sqrt(); } }
        UnaryOp::Abs  => { for i in 0..n { out[i] = a[i].abs(); } }
        UnaryOp::Neg  => { for i in 0..n { out[i] = -a[i]; } }
        UnaryOp::Relu => { for i in 0..n { out[i] = if a[i] > 0.0 { a[i] } else { 0.0 }; } }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_sqrt(a: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vr = _mm256_sqrt_pd(va);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
        i += 4;
    }
    while i < n { out[i] = a[i].sqrt(); i += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_abs(a: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = a.len();
    // Clear sign bit: AND with 0x7FFF_FFFF_FFFF_FFFF
    let mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFFu64));
    let mut i = 0;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vr = _mm256_and_pd(va, mask);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
        i += 4;
    }
    while i < n { out[i] = a[i].abs(); i += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_neg(a: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = a.len();
    // Toggle sign bit: XOR with 0x8000_0000_0000_0000
    let sign_bit = _mm256_set1_pd(f64::from_bits(0x8000_0000_0000_0000u64));
    let mut i = 0;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vr = _mm256_xor_pd(va, sign_bit);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
        i += 4;
    }
    while i < n { out[i] = -a[i]; i += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_relu(a: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let zero = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vr = _mm256_max_pd(va, zero);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
        i += 4;
    }
    while i < n { out[i] = if a[i] > 0.0 { a[i] } else { 0.0 }; i += 1; }
}

// ── Tiled matmul AXPY kernel ────────────────────────────────────────────────

/// SIMD-accelerated AXPY: `c[0..len] += scalar * b[0..len]`.
///
/// Used in the inner loop of tiled matrix multiplication where `scalar = A[i,p]`
/// and `b` is a row segment of B. Processes 4 elements per iteration with AVX2.
///
/// Deterministic because each `c[j]` accumulates the same `scalar * b[j]`
/// contribution using separate mul + add (no FMA), matching scalar behavior.
pub fn simd_axpy(c: &mut [f64], b: &[f64], scalar: f64, len: usize) {
    debug_assert!(c.len() >= len);
    debug_assert!(b.len() >= len);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { avx2_axpy(c, b, scalar, len); }
            return;
        }
    }

    // Scalar fallback
    for j in 0..len {
        c[j] += scalar * b[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_axpy(c: &mut [f64], b: &[f64], scalar: f64, len: usize) {
    use std::arch::x86_64::*;
    let a_vec = _mm256_set1_pd(scalar);
    let mut j = 0;

    while j + 4 <= len {
        let c_ptr = c.as_mut_ptr().add(j);
        let b_ptr = b.as_ptr().add(j);
        let c_val = _mm256_loadu_pd(c_ptr);
        let b_val = _mm256_loadu_pd(b_ptr);
        // Separate mul + add (NOT FMA) — preserves bit-identity with scalar path.
        let prod = _mm256_mul_pd(a_vec, b_val);
        let result = _mm256_add_pd(c_val, prod);
        _mm256_storeu_pd(c_ptr, result);
        j += 4;
    }

    // Scalar tail
    while j < len {
        *c.get_unchecked_mut(j) += scalar * *b.get_unchecked(j);
        j += 1;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_matches_scalar() {
        let a: Vec<f64> = (0..17).map(|i| i as f64 * 0.3).collect();
        let b: Vec<f64> = (0..17).map(|i| (17 - i) as f64 * 0.7).collect();
        let result = simd_binop(&a, &b, BinOp::Add);
        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        assert_eq!(result, expected, "SIMD add must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_sub_matches_scalar() {
        let a: Vec<f64> = (0..17).map(|i| i as f64 * 1.1).collect();
        let b: Vec<f64> = (0..17).map(|i| (17 - i) as f64 * 0.9).collect();
        let result = simd_binop(&a, &b, BinOp::Sub);
        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
        assert_eq!(result, expected, "SIMD sub must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_mul_matches_scalar() {
        let a: Vec<f64> = (0..17).map(|i| i as f64 * 0.1 + 0.01).collect();
        let b: Vec<f64> = (0..17).map(|i| (17 - i) as f64 * 0.2 + 0.03).collect();
        let result = simd_binop(&a, &b, BinOp::Mul);
        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
        assert_eq!(result, expected, "SIMD mul must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_div_matches_scalar() {
        let a: Vec<f64> = (0..17).map(|i| i as f64 * 0.5 + 1.0).collect();
        let b: Vec<f64> = (0..17).map(|i| (i + 1) as f64 * 0.3).collect();
        let result = simd_binop(&a, &b, BinOp::Div);
        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect();
        assert_eq!(result, expected, "SIMD div must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_sqrt_matches_scalar() {
        let a: Vec<f64> = (0..17).map(|i| i as f64 * 2.5 + 0.1).collect();
        let result = simd_unary(&a, UnaryOp::Sqrt);
        let expected: Vec<f64> = a.iter().map(|&x| x.sqrt()).collect();
        assert_eq!(result, expected, "SIMD sqrt must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_abs_matches_scalar() {
        let a: Vec<f64> = (-8..9).map(|i| i as f64 * 1.5).collect();
        let result = simd_unary(&a, UnaryOp::Abs);
        let expected: Vec<f64> = a.iter().map(|&x| x.abs()).collect();
        assert_eq!(result, expected, "SIMD abs must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_neg_matches_scalar() {
        let a: Vec<f64> = (-8..9).map(|i| i as f64 * 1.5).collect();
        let result = simd_unary(&a, UnaryOp::Neg);
        let expected: Vec<f64> = a.iter().map(|&x| -x).collect();
        assert_eq!(result, expected, "SIMD neg must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_relu_matches_scalar() {
        let a: Vec<f64> = (-8..9).map(|i| i as f64 * 1.5).collect();
        let result = simd_unary(&a, UnaryOp::Relu);
        let expected: Vec<f64> = a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        assert_eq!(result, expected, "SIMD relu must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_axpy_matches_scalar() {
        let b: Vec<f64> = (0..17).map(|i| i as f64 * 0.3).collect();
        let scalar = 2.5;
        let mut c_simd: Vec<f64> = (0..17).map(|i| i as f64 * 0.1).collect();
        let mut c_scalar = c_simd.clone();

        simd_axpy(&mut c_simd, &b, scalar, 17);
        for j in 0..17 {
            c_scalar[j] += scalar * b[j];
        }
        assert_eq!(c_simd, c_scalar, "SIMD axpy must be bit-identical to scalar");
    }

    #[test]
    fn test_simd_empty_input() {
        let empty: Vec<f64> = vec![];
        assert_eq!(simd_binop(&empty, &empty, BinOp::Add), Vec::<f64>::new());
        assert_eq!(simd_unary(&empty, UnaryOp::Sqrt), Vec::<f64>::new());
    }

    #[test]
    fn test_simd_single_element() {
        let a = vec![3.0];
        let b = vec![4.0];
        assert_eq!(simd_binop(&a, &b, BinOp::Add), vec![7.0]);
        assert_eq!(simd_unary(&a, UnaryOp::Sqrt), vec![3.0f64.sqrt()]);
    }

    #[test]
    fn test_simd_exactly_four_elements() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert_eq!(simd_binop(&a, &b, BinOp::Add), vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(simd_binop(&a, &b, BinOp::Mul), vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_avx2_detection() {
        // Just verify the function doesn't panic.
        let _has = has_avx2();
    }
}
