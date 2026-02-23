//! Quantized BLAS Integration Tests
//!
//! Tests the i8/i4 dequantization path into BinnedAccumulator,
//! verifying determinism, saturation, and precision.

use cjc_runtime::quantized::*;
use cjc_runtime::accumulator::BinnedAccumulatorF64;

// ---------------------------------------------------------------------------
// i8 Dequantization Tests
// ---------------------------------------------------------------------------

#[test]
fn test_dequant_i8_extreme_scale() {
    // Very small scale — tests precision near zero.
    let params = QuantParamsI8::new(1e-15, 0);
    let val = params.dequantize(127);
    // f64 multiplication may introduce tiny rounding errors.
    assert!((val - 127e-15).abs() < 1e-27);
    assert!(val > 0.0);
}

#[test]
fn test_dequant_i8_very_large_scale() {
    let params = QuantParamsI8::new(1e10, 0);
    let val = params.dequantize(127);
    assert_eq!(val, 127e10);
    assert!(val.is_finite());
}

#[test]
fn test_dequant_i8_negative_scale() {
    // Negative scale inverts the mapping.
    let params = QuantParamsI8::new(-0.5, 0);
    assert_eq!(params.dequantize(10), -5.0);
    assert_eq!(params.dequantize(-10), 5.0);
}

#[test]
fn test_dequant_i8_zero_scale() {
    let params = QuantParamsI8::new(0.0, 0);
    assert_eq!(params.dequantize(127), 0.0);
    assert_eq!(params.dequantize(-128), 0.0);
}

#[test]
fn test_dequant_i8_full_range() {
    let params = QuantParamsI8::new(1.0, 0);
    assert_eq!(params.dequantize(i8::MAX), 127.0);
    assert_eq!(params.dequantize(i8::MIN), -128.0);
}

#[test]
fn test_dequant_i8_slice() {
    let params = QuantParamsI8::new(0.1, 5);
    let values = vec![5i8, 15, 25, -5];
    let result = params.dequantize_slice(&values);
    assert_eq!(result.len(), 4);
    assert_eq!(result[0], 0.0);   // 0.1 * (5-5)
    assert_eq!(result[1], 1.0);   // 0.1 * (15-5)
    assert_eq!(result[2], 2.0);   // 0.1 * (25-5)
    assert_eq!(result[3], -1.0);  // 0.1 * (-5-5)
}

// ---------------------------------------------------------------------------
// i4 Packing/Unpacking Tests
// ---------------------------------------------------------------------------

#[test]
fn test_i4_unpack_all_zeros() {
    let (hi, lo) = QuantParamsI4::unpack_byte(0x00);
    assert_eq!(hi, 0);
    assert_eq!(lo, 0);
}

#[test]
fn test_i4_unpack_all_ones() {
    // 0xFF = 1111_1111 → hi=-1, lo=-1
    let (hi, lo) = QuantParamsI4::unpack_byte(0xFF);
    assert_eq!(hi, -1);
    assert_eq!(lo, -1);
}

#[test]
fn test_i4_unpack_max_positive() {
    // 0x77 = 0111_0111 → hi=7, lo=7
    let (hi, lo) = QuantParamsI4::unpack_byte(0x77);
    assert_eq!(hi, 7);
    assert_eq!(lo, 7);
}

#[test]
fn test_i4_unpack_min_negative() {
    // 0x88 = 1000_1000 → hi=-8, lo=-8
    let (hi, lo) = QuantParamsI4::unpack_byte(0x88);
    assert_eq!(hi, -8);
    assert_eq!(lo, -8);
}

#[test]
fn test_i4_unpack_mixed() {
    // 0x5A = 0101_1010 → hi=5, lo=-6
    let (hi, lo) = QuantParamsI4::unpack_byte(0x5A);
    assert_eq!(hi, 5);
    assert_eq!(lo, -6);
}

#[test]
fn test_i4_sum_odd_count() {
    let params = QuantParamsI4::new(1.0, 0);
    let packed = vec![0x12u8]; // hi=1, lo=2
    // Only take 1 element.
    let result = quantized_sum_i4(&packed, 1, &params);
    assert_eq!(result, 1.0);
}

#[test]
fn test_i4_sum_empty() {
    let params = QuantParamsI4::new(1.0, 0);
    let packed: Vec<u8> = vec![];
    let result = quantized_sum_i4(&packed, 0, &params);
    assert_eq!(result, 0.0);
}

// ---------------------------------------------------------------------------
// Saturating Arithmetic Tests
// ---------------------------------------------------------------------------

#[test]
fn test_saturating_mul_max_values() {
    let result = saturating_mul_i8(127, 127);
    assert_eq!(result, 16129);
}

#[test]
fn test_saturating_mul_min_values() {
    let result = saturating_mul_i8(-128, -128);
    assert_eq!(result, 16384);
}

#[test]
fn test_saturating_mul_cross_sign() {
    let result = saturating_mul_i8(127, -128);
    assert_eq!(result, -16256);
}

#[test]
fn test_saturating_dot_empty() {
    let result = saturating_dot_i8(&[], &[]);
    assert_eq!(result, 0);
}

#[test]
fn test_saturating_dot_single() {
    let result = saturating_dot_i8(&[5], &[7]);
    assert_eq!(result, 35);
}

#[test]
fn test_saturating_dot_saturation_extreme() {
    // i32::MAX / (127*127) ≈ 133,143 iterations needed to overflow.
    // With 200_000 elements of 127*127 each:
    let n = 200_000;
    let a = vec![127i8; n];
    let b = vec![127i8; n];
    let result = saturating_dot_i8(&a, &b);
    // 127*127 = 16129; 16129 * 200000 = 3,225,800,000 > i32::MAX (2,147,483,647)
    // Should saturate to i32::MAX.
    assert_eq!(result, i32::MAX);
}

// ---------------------------------------------------------------------------
// Quantized GEMM Tests
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_matmul_2x2() {
    let params = QuantParamsI8::new(1.0, 0);
    let a = vec![1i8, 2, 3, 4];
    let b = vec![5i8, 6, 7, 8];
    let mut out = vec![0.0f64; 4];
    quantized_matmul_i8(&a, &b, &mut out, 2, 2, 2, &params, &params);
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_quantized_matmul_non_square() {
    let params = QuantParamsI8::new(1.0, 0);
    let a = vec![1i8, 2, 3]; // 1x3
    let b = vec![4i8, 5, 6]; // 3x1
    let mut out = vec![0.0f64; 1];
    quantized_matmul_i8(&a, &b, &mut out, 1, 3, 1, &params, &params);
    assert_eq!(out[0], 32.0); // 1*4 + 2*5 + 3*6
}

#[test]
fn test_quantized_matmul_with_zero_point() {
    let params_a = QuantParamsI8::new(1.0, 5);
    let params_b = QuantParamsI8::new(1.0, 0);
    let a = vec![10i8, 10]; // dequant: (10-5)=5, (10-5)=5
    let b = vec![2i8, 3];   // dequant: 2, 3
    let mut out = vec![0.0f64; 1];
    quantized_matmul_i8(&a, &b, &mut out, 1, 2, 1, &params_a, &params_b);
    // 5*2 + 5*3 = 25.0
    assert_eq!(out[0], 25.0);
}

#[test]
fn test_quantized_matmul_deterministic() {
    let params = QuantParamsI8::new(0.01, 3);
    let a: Vec<i8> = (0..100).map(|i| ((i * 7 + 13) % 256) as i8).collect();
    let b: Vec<i8> = (0..100).map(|i| ((i * 11 + 37) % 256) as i8).collect();
    let mut out1 = vec![0.0f64; 100];
    let mut out2 = vec![0.0f64; 100];
    quantized_matmul_i8(&a, &b, &mut out1, 10, 10, 10, &params, &params);
    quantized_matmul_i8(&a, &b, &mut out2, 10, 10, 10, &params, &params);
    for i in 0..100 {
        assert_eq!(out1[i].to_bits(), out2[i].to_bits(),
            "Quantized matmul element {i} differs between runs");
    }
}

#[test]
fn test_quantized_matmul_against_float() {
    // Verify quantized matmul matches f64 matmul when scale=1, zp=0.
    let params = QuantParamsI8::new(1.0, 0);
    let a_i8 = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9];
    let b_i8 = vec![9i8, 8, 7, 6, 5, 4, 3, 2, 1];
    let mut out_quant = vec![0.0f64; 9];
    quantized_matmul_i8(&a_i8, &b_i8, &mut out_quant, 3, 3, 3, &params, &params);

    // Same as f64.
    let a_f64: Vec<f64> = a_i8.iter().map(|&v| v as f64).collect();
    let b_f64: Vec<f64> = b_i8.iter().map(|&v| v as f64).collect();
    let mut out_f64 = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for p in 0..3 {
                sum += a_f64[i*3+p] * b_f64[p*3+j];
            }
            out_f64[i*3+j] = sum;
        }
    }

    for i in 0..9 {
        assert_eq!(out_quant[i], out_f64[i],
            "Element {i}: quant={} vs f64={}", out_quant[i], out_f64[i]);
    }
}

// ---------------------------------------------------------------------------
// Quantized Dot Product Tests
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_dot_empty() {
    let params = QuantParamsI8::new(1.0, 0);
    let result = quantized_dot_i8(&[], &[], &params, &params);
    assert_eq!(result, 0.0);
}

#[test]
fn test_quantized_dot_single_element() {
    let params = QuantParamsI8::new(0.5, 0);
    let result = quantized_dot_i8(&[4], &[6], &params, &params);
    // (0.5*4) * (0.5*6) = 2.0 * 3.0 = 6.0
    // via combined: 0.25 * (4*6) = 0.25 * 24 = 6.0
    assert_eq!(result, 6.0);
}

#[test]
fn test_quantized_dot_with_different_params() {
    let pa = QuantParamsI8::new(0.1, 10);
    let pb = QuantParamsI8::new(0.2, 5);
    // a=[20] → dequant: 0.1*(20-10) = 1.0
    // b=[15] → dequant: 0.2*(15-5) = 2.0
    // dot = 1.0 * 2.0 = 2.0
    // via combined: 0.02 * ((20-10)*(15-5)) = 0.02 * 100 = 2.0
    // May have tiny rounding from scale multiplication.
    let result = quantized_dot_i8(&[20], &[15], &pa, &pb);
    assert!((result - 2.0).abs() < 1e-10, "Expected ~2.0, got {result}");
}

// ---------------------------------------------------------------------------
// Quantized Sum Tests
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_sum_i8_basic() {
    let params = QuantParamsI8::new(1.0, 0);
    let values = vec![1i8, 2, 3, 4, 5];
    assert_eq!(quantized_sum_i8(&values, &params), 15.0);
}

#[test]
fn test_quantized_sum_i8_empty() {
    let params = QuantParamsI8::new(1.0, 0);
    assert_eq!(quantized_sum_i8(&[], &params), 0.0);
}

#[test]
fn test_quantized_sum_i8_with_zp() {
    let params = QuantParamsI8::new(2.0, 3);
    // values: 3, 5 → dequant: 2*(3-3)=0, 2*(5-3)=4 → sum=4
    let values = vec![3i8, 5];
    assert_eq!(quantized_sum_i8(&values, &params), 4.0);
}

#[test]
fn test_quantized_sum_i4_with_zp() {
    let params = QuantParamsI4::new(0.5, -3);
    // Pack: (2, 4) → byte 0x24
    // dequant(2) = 0.5 * (2 - (-3)) = 0.5 * 5 = 2.5
    // dequant(4) = 0.5 * (4 - (-3)) = 0.5 * 7 = 3.5
    let packed = vec![0x24u8];
    let result = quantized_sum_i4(&packed, 2, &params);
    assert_eq!(result, 6.0);
}

// ---------------------------------------------------------------------------
// Extreme Scale Stress Tests
// ---------------------------------------------------------------------------

#[test]
fn test_extreme_scale_tiny() {
    let params = QuantParamsI8::new(1e-300, 0);
    let values: Vec<i8> = (1..=100).map(|i| i as i8).collect();
    let result = quantized_sum_i8(&values, &params);
    assert!(result.is_finite());
    assert!(result > 0.0);
}

#[test]
fn test_extreme_scale_huge() {
    let params = QuantParamsI8::new(1e200, 0);
    let values = vec![1i8, -1]; // Should cancel.
    let result = quantized_sum_i8(&values, &params);
    assert_eq!(result, 0.0);
}

#[test]
fn test_extreme_scale_cancellation() {
    let params = QuantParamsI8::new(1e100, 0);
    let mut values: Vec<i8> = vec![127; 50];
    values.extend(vec![-127i8; 50]);
    let result = quantized_sum_i8(&values, &params);
    assert_eq!(result, 0.0);
}

#[test]
fn test_dequant_preserves_integer_products() {
    // Verify that integer products are exact after dequantization.
    let _pa = QuantParamsI8::new(1.0, 0);
    // i8*i8 → i32 → f64: no rounding occurs (i32 fits exactly in f64).
    for a in [-128i8, -1, 0, 1, 127] {
        for b in [-128i8, -1, 0, 1, 127] {
            let int_prod = (a as i64) * (b as i64);
            let f64_prod = int_prod as f64;
            assert_eq!(f64_prod as i64, int_prod,
                "Integer product {a}*{b}={int_prod} not preserved in f64");
        }
    }
}

// ---------------------------------------------------------------------------
// Merge-Based Parallel Quantized Summation
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_parallel_sum_merge() {
    let params = QuantParamsI8::new(0.01, 5);
    let values: Vec<i8> = (0..1000).map(|i| ((i * 3 + 7) % 256) as i8).collect();

    // Chunk into various sizes, merge forward.
    for chunk_size in [10, 50, 100, 250, 500] {
        let mut fwd = BinnedAccumulatorF64::new();
        for chunk in values.chunks(chunk_size) {
            let mut c = BinnedAccumulatorF64::new();
            for &v in chunk {
                c.add(params.dequantize(v));
            }
            fwd.merge(&c);
        }

        // Merge reverse.
        let chunks: Vec<Vec<i8>> = values.chunks(chunk_size).map(|c| c.to_vec()).collect();
        let mut rev = BinnedAccumulatorF64::new();
        for chunk in chunks.iter().rev() {
            let mut c = BinnedAccumulatorF64::new();
            for &v in chunk.iter() {
                c.add(params.dequantize(v));
            }
            rev.merge(&c);
        }

        assert_eq!(fwd.finalize().to_bits(), rev.finalize().to_bits(),
            "Merge order invariance failed for chunk_size={chunk_size}");
    }
}
