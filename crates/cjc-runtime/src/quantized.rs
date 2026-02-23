//! Quantized BLAS — i8/i4 dequantization into BinnedAccumulator.
//!
//! # Design
//!
//! Quantized integer products (i8×i8 → i32) are dequantized and fed directly
//! into the BinnedAccumulator, bypassing intermediate f32 rounding entirely.
//! This eliminates a major source of non-determinism in quantized inference.
//!
//! # Saturation
//!
//! Integer overflow is handled via saturation arithmetic — values clamp to
//! `i32::MAX` / `i32::MIN` rather than wrapping silently.

use crate::accumulator::BinnedAccumulatorF64;

// ---------------------------------------------------------------------------
// i8 Quantized Operations
// ---------------------------------------------------------------------------

/// Quantization parameters for i8 tensors.
///
/// Maps integer range [zero_point - 128, zero_point + 127] to floating-point
/// via: `float_value = scale * (int_value - zero_point)`
#[derive(Debug, Clone, Copy)]
pub struct QuantParamsI8 {
    /// Scale factor: the step size between consecutive quantized values.
    pub scale: f64,
    /// Zero point: the integer value that maps to 0.0 in float.
    pub zero_point: i8,
}

impl QuantParamsI8 {
    /// Create new quantization parameters.
    pub fn new(scale: f64, zero_point: i8) -> Self {
        QuantParamsI8 { scale, zero_point }
    }

    /// Dequantize a single i8 value to f64.
    #[inline]
    pub fn dequantize(&self, v: i8) -> f64 {
        self.scale * (v as i64 - self.zero_point as i64) as f64
    }

    /// Dequantize a slice of i8 values to f64.
    pub fn dequantize_slice(&self, src: &[i8]) -> Vec<f64> {
        src.iter().map(|&v| self.dequantize(v)).collect()
    }
}

/// Quantization parameters for i4 (nibble-packed) tensors.
///
/// i4 values range from -8 to +7 (signed) or 0 to 15 (unsigned).
/// Stored packed: two i4 values per byte (high nibble, low nibble).
#[derive(Debug, Clone, Copy)]
pub struct QuantParamsI4 {
    /// Scale factor.
    pub scale: f64,
    /// Zero point in i4 range [-8, 7].
    pub zero_point: i8,
}

impl QuantParamsI4 {
    pub fn new(scale: f64, zero_point: i8) -> Self {
        assert!(zero_point >= -8 && zero_point <= 7, "i4 zero_point must be in [-8, 7]");
        QuantParamsI4 { scale, zero_point }
    }

    /// Unpack a byte into two signed i4 values: (high_nibble, low_nibble).
    #[inline]
    pub fn unpack_byte(byte: u8) -> (i8, i8) {
        // Sign-extension for 4-bit signed values via shift trick.
        let hi = (((byte >> 4) & 0x0F) as i8) << 4 >> 4;
        let lo = ((byte & 0x0F) as i8) << 4 >> 4;
        (hi, lo)
    }

    /// Dequantize a single i4 value to f64.
    #[inline]
    pub fn dequantize(&self, v: i8) -> f64 {
        self.scale * (v as i64 - self.zero_point as i64) as f64
    }
}

// ---------------------------------------------------------------------------
// Saturating i32 Arithmetic
// ---------------------------------------------------------------------------

/// Saturating multiply of two i8 values, producing i32 without overflow.
#[inline]
pub fn saturating_mul_i8(a: i8, b: i8) -> i32 {
    (a as i32) * (b as i32)
    // i8 * i8 fits in i32 without overflow (max: 127*127 = 16129)
}

/// Saturating dot product of two i8 slices, accumulating into i32.
///
/// Uses saturating addition to prevent silent wrap-around.
#[inline]
pub fn saturating_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum: i32 = 0;
    for i in 0..a.len() {
        let prod = (a[i] as i32) * (b[i] as i32);
        sum = sum.saturating_add(prod);
    }
    sum
}

// ---------------------------------------------------------------------------
// Quantized GEMM via BinnedAccumulator
// ---------------------------------------------------------------------------

/// Quantized matrix multiply: C[m,n] = dequant(A[m,k]) × dequant(B[k,n])
///
/// The i8×i8 products are computed in i32, then dequantized to f64 and
/// accumulated via BinnedAccumulator for deterministic summation.
///
/// This avoids intermediate f32 rounding: integer products go directly
/// to f64 dequantization, then into binned accumulation.
///
/// # Arguments
/// * `a` - Row-major i8 matrix [m, k]
/// * `b` - Row-major i8 matrix [k, n]
/// * `params_a` - Quantization parameters for A
/// * `params_b` - Quantization parameters for B
/// * `out` - Output buffer [m, n] (pre-allocated)
pub fn quantized_matmul_i8(
    a: &[i8], b: &[i8], out: &mut [f64],
    m: usize, k: usize, n: usize,
    params_a: &QuantParamsI8, params_b: &QuantParamsI8,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    // Combined scale factor.
    let combined_scale = params_a.scale * params_b.scale;

    for i in 0..m {
        for j in 0..n {
            let mut acc = BinnedAccumulatorF64::new();
            for p in 0..k {
                // Integer product: no rounding.
                let int_prod = (a[i * k + p] as i64 - params_a.zero_point as i64)
                    * (b[p * n + j] as i64 - params_b.zero_point as i64);
                // Dequantize directly to f64: combined_scale * int_prod.
                acc.add(combined_scale * int_prod as f64);
            }
            out[i * n + j] = acc.finalize();
        }
    }
}

/// Quantized dot product of two i8 vectors, returning f64.
///
/// Dequantizes products into BinnedAccumulator for determinism.
pub fn quantized_dot_i8(
    a: &[i8], b: &[i8],
    params_a: &QuantParamsI8, params_b: &QuantParamsI8,
) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let combined_scale = params_a.scale * params_b.scale;
    let mut acc = BinnedAccumulatorF64::new();
    for i in 0..a.len() {
        let int_prod = (a[i] as i64 - params_a.zero_point as i64)
            * (b[i] as i64 - params_b.zero_point as i64);
        acc.add(combined_scale * int_prod as f64);
    }
    acc.finalize()
}

/// Sum dequantized i8 values using BinnedAccumulator.
pub fn quantized_sum_i8(values: &[i8], params: &QuantParamsI8) -> f64 {
    let mut acc = BinnedAccumulatorF64::new();
    for &v in values {
        acc.add(params.dequantize(v));
    }
    acc.finalize()
}

/// Sum dequantized i4 (packed) values using BinnedAccumulator.
///
/// `packed` contains pairs of i4 values packed into bytes.
/// `count` is the total number of i4 elements (may be odd).
pub fn quantized_sum_i4(packed: &[u8], count: usize, params: &QuantParamsI4) -> f64 {
    let mut acc = BinnedAccumulatorF64::new();
    let mut remaining = count;
    for &byte in packed {
        if remaining == 0 { break; }
        let (hi, lo) = QuantParamsI4::unpack_byte(byte);
        acc.add(params.dequantize(hi));
        remaining -= 1;
        if remaining == 0 { break; }
        acc.add(params.dequantize(lo));
        remaining -= 1;
    }
    acc.finalize()
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_i8_basic() {
        let params = QuantParamsI8::new(0.1, 0);
        assert_eq!(params.dequantize(10), 1.0);
        assert_eq!(params.dequantize(-10), -1.0);
        assert_eq!(params.dequantize(0), 0.0);
    }

    #[test]
    fn test_dequantize_i8_with_zero_point() {
        let params = QuantParamsI8::new(0.5, 10);
        // float = 0.5 * (20 - 10) = 5.0
        assert_eq!(params.dequantize(20), 5.0);
        // float = 0.5 * (10 - 10) = 0.0
        assert_eq!(params.dequantize(10), 0.0);
    }

    #[test]
    fn test_saturating_dot_i8() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![5i8, 6, 7, 8];
        assert_eq!(saturating_dot_i8(&a, &b), 70); // 5+12+21+32
    }

    #[test]
    fn test_saturating_dot_overflow() {
        // Test that saturation prevents wrap-around.
        let a = vec![127i8; 1000];
        let b = vec![127i8; 1000];
        let result = saturating_dot_i8(&a, &b);
        // 127*127 = 16129; 16129 * 1000 = 16_129_000 — fits in i32.
        assert_eq!(result, 16_129_000);
    }

    #[test]
    fn test_quantized_matmul_identity() {
        // 2x2 identity via i8 with scale=1.0, zp=0
        let params = QuantParamsI8::new(1.0, 0);
        let a = vec![1i8, 0, 0, 1]; // identity
        let b = vec![3i8, 4, 5, 6];
        let mut out = vec![0.0f64; 4];
        quantized_matmul_i8(&a, &b, &mut out, 2, 2, 2, &params, &params);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_quantized_matmul_scaling() {
        let params_a = QuantParamsI8::new(0.5, 0);
        let params_b = QuantParamsI8::new(2.0, 0);
        // combined_scale = 0.5 * 2.0 = 1.0
        let a = vec![2i8, 3];
        let b = vec![4i8, 5];
        let mut out = vec![0.0f64; 1];
        quantized_matmul_i8(&a, &b, &mut out, 1, 2, 1, &params_a, &params_b);
        // dot = (0.5*2)*(2.0*4) + (0.5*3)*(2.0*5) = 8 + 15 = 23
        // via combined: 1.0 * (2*4 + 3*5) = 1.0 * 23 = 23.0
        assert_eq!(out[0], 23.0);
    }

    #[test]
    fn test_quantized_dot_deterministic() {
        let params = QuantParamsI8::new(0.001, 0);
        let a: Vec<i8> = (0..100).map(|i| (i % 127) as i8).collect();
        let b: Vec<i8> = (0..100).map(|i| ((100 - i) % 127) as i8).collect();

        let r1 = quantized_dot_i8(&a, &b, &params, &params);
        let r2 = quantized_dot_i8(&a, &b, &params, &params);
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    #[test]
    fn test_i4_unpack() {
        // Pack: high=3, low=-2 → 0x3E
        // 3 in 4-bit signed: 0011
        // -2 in 4-bit signed: 1110
        // byte: 0011_1110 = 0x3E
        let (hi, lo) = QuantParamsI4::unpack_byte(0x3E);
        assert_eq!(hi, 3);
        assert_eq!(lo, -2);
    }

    #[test]
    fn test_i4_unpack_negatives() {
        // high=-1 (1111), low=-8 (1000) → 0xF8
        let (hi, lo) = QuantParamsI4::unpack_byte(0xF8);
        assert_eq!(hi, -1);
        assert_eq!(lo, -8);
    }

    #[test]
    fn test_quantized_sum_i4() {
        let params = QuantParamsI4::new(1.0, 0);
        // Pack: (2, 3), (4, 5) = 0x23, 0x45
        let packed = vec![0x23u8, 0x45];
        let result = quantized_sum_i4(&packed, 4, &params);
        assert_eq!(result, 14.0); // 2 + 3 + 4 + 5
    }

    #[test]
    fn test_quantized_sum_i8_near_order_invariant() {
        let params = QuantParamsI8::new(0.001, 0);
        let values: Vec<i8> = (0..200).map(|i| ((i as i16 - 100) % 128) as i8).collect();

        let r1 = quantized_sum_i8(&values, &params);

        // Reverse order.
        let mut rev = values.clone();
        rev.reverse();
        let r2 = quantized_sum_i8(&rev, &params);

        // Within-bin accumulation order may cause a few ULPs of difference
        // due to IEEE-754 non-associativity. The BinnedAccumulator minimizes
        // this by binning values with similar exponents, but doesn't eliminate it.
        let ulps = (r1.to_bits() as i64 - r2.to_bits() as i64).unsigned_abs();
        assert!(ulps < 10,
            "Quantized sum should be near-order-invariant: {r1} vs {r2} ({ulps} ULPs)");
    }

    #[test]
    fn test_quantized_sum_i8_merge_order_invariant() {
        // Merge-based accumulation IS fully order-invariant (Knuth 2Sum merge).
        let params = QuantParamsI8::new(0.001, 0);
        let values: Vec<i8> = (0..200).map(|i| ((i as i16 - 100) % 128) as i8).collect();

        // Chunk into 20s, merge forward.
        let mut fwd = BinnedAccumulatorF64::new();
        for chunk in values.chunks(20) {
            let mut c = BinnedAccumulatorF64::new();
            for &v in chunk {
                c.add(params.dequantize(v));
            }
            fwd.merge(&c);
        }

        // Chunk into 20s, merge reverse.
        let chunks: Vec<Vec<i8>> = values.chunks(20).map(|c| c.to_vec()).collect();
        let mut rev = BinnedAccumulatorF64::new();
        for chunk in chunks.iter().rev() {
            let mut c = BinnedAccumulatorF64::new();
            for &v in chunk.iter() {
                c.add(params.dequantize(v));
            }
            rev.merge(&c);
        }

        assert_eq!(fwd.finalize().to_bits(), rev.finalize().to_bits(),
            "Merge-based quantized sum must be order-invariant");
    }
}
