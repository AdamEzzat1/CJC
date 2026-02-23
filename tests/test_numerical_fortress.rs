//! Numerical Fortress Tests
//!
//! Cross-cutting determinism and precision tests that exercise multiple
//! subsystems together, verifying the "Numerical Fortress" guarantees
//! of Milestone 2.7 Expansion.

use cjc_runtime::Tensor;
use cjc_runtime::accumulator::{BinnedAccumulatorF64, binned_sum_f64};
use cjc_runtime::complex::ComplexF64;
use cjc_runtime::f16::F16;
use cjc_runtime::quantized::{QuantParamsI8, QuantParamsI4, quantized_sum_i8, quantized_sum_i4, quantized_matmul_i8};
use cjc_runtime::dispatch::{ReductionContext, dispatch_sum_f64, dispatch_dot_f64};

// ---------------------------------------------------------------------------
// Cross-Type Consistency
// ---------------------------------------------------------------------------

#[test]
fn test_all_sum_strategies_agree_simple() {
    let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let kahan = cjc_repro::kahan_sum_f64(&data);
    let binned = binned_sum_f64(&data);
    let ctx_serial = ReductionContext::default_serial();
    let ctx_strict = ReductionContext::strict_parallel();
    let dispatch_kahan = dispatch_sum_f64(&data, &ctx_serial);
    let dispatch_binned = dispatch_sum_f64(&data, &ctx_strict);
    assert_eq!(kahan, 5050.0);
    assert_eq!(binned, 5050.0);
    assert_eq!(dispatch_kahan, 5050.0);
    assert_eq!(dispatch_binned, 5050.0);
}

#[test]
fn test_dispatch_dot_agrees_on_integers() {
    let a: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let b: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let ctx_k = ReductionContext::default_serial();
    let ctx_b = ReductionContext::strict_parallel();
    let dk = dispatch_dot_f64(&a, &b, &ctx_k);
    let db = dispatch_dot_f64(&a, &b, &ctx_b);
    assert_eq!(dk, 385.0);
    assert_eq!(db, 385.0);
}

#[test]
fn test_f16_through_binned_matches_f64() {
    // Simple integers in f16 should exactly match f64 binned sum.
    let f16_vals: Vec<F16> = (1..=50).map(|i| F16::from_f64(i as f64)).collect();
    let f64_vals: Vec<f64> = (1..=50).map(|i| i as f64).collect();
    let f16_result = cjc_runtime::f16::f16_binned_sum(&f16_vals);
    let f64_result = binned_sum_f64(&f64_vals);
    assert_eq!(f16_result, f64_result);
}

#[test]
fn test_quantized_through_binned_matches_f64() {
    let params = QuantParamsI8::new(1.0, 0);
    let i8_vals: Vec<i8> = (1..=50).map(|i| i as i8).collect();
    let f64_vals: Vec<f64> = (1..=50).map(|i| i as f64).collect();
    let quant_result = quantized_sum_i8(&i8_vals, &params);
    let f64_result = binned_sum_f64(&f64_vals);
    assert_eq!(quant_result, f64_result);
}

// ---------------------------------------------------------------------------
// Determinism Under Stress
// ---------------------------------------------------------------------------

#[test]
fn test_50_run_tensor_sum_determinism() {
    let mut rng = cjc_repro::Rng::seeded(77);
    let data: Vec<f64> = (0..500).map(|_| rng.next_f64() * 2000.0 - 1000.0).collect();
    let t = Tensor::from_vec(data, &[500]).unwrap();
    let reference = t.binned_sum();
    for _ in 0..50 {
        assert_eq!(t.binned_sum().to_bits(), reference.to_bits());
    }
}

#[test]
fn test_50_run_complex_dot_determinism() {
    let n = 200;
    let a: Vec<ComplexF64> = (0..n)
        .map(|i| ComplexF64::new(i as f64 * 0.7 - 70.0, -(i as f64) * 0.3))
        .collect();
    let b: Vec<ComplexF64> = (0..n)
        .map(|i| ComplexF64::new(-(i as f64) * 0.5, (n - i) as f64 * 0.2))
        .collect();
    let reference = cjc_runtime::complex::complex_dot(&a, &b);
    for _ in 0..50 {
        let r = cjc_runtime::complex::complex_dot(&a, &b);
        assert_eq!(r.re.to_bits(), reference.re.to_bits());
        assert_eq!(r.im.to_bits(), reference.im.to_bits());
    }
}

#[test]
fn test_50_run_quantized_matmul_determinism() {
    let params = QuantParamsI8::new(0.01, 5);
    let a: Vec<i8> = (0..25).map(|i| ((i * 7 + 3) % 200 - 100) as i8).collect();
    let b: Vec<i8> = (0..25).map(|i| ((i * 11 + 17) % 200 - 100) as i8).collect();
    let mut ref_out = vec![0.0f64; 25];
    quantized_matmul_i8(&a, &b, &mut ref_out, 5, 5, 5, &params, &params);
    for _ in 0..50 {
        let mut out = vec![0.0f64; 25];
        quantized_matmul_i8(&a, &b, &mut out, 5, 5, 5, &params, &params);
        assert_eq!(out, ref_out);
    }
}

#[test]
fn test_50_run_f16_matmul_determinism() {
    let n = 4;
    let a: Vec<F16> = (0..n*n).map(|i| F16::from_f64(i as f64 * 0.5 - 4.0)).collect();
    let b: Vec<F16> = (0..n*n).map(|i| F16::from_f64(-(i as f64) * 0.3 + 2.0)).collect();
    let mut ref_out = vec![0.0f64; n*n];
    cjc_runtime::f16::f16_matmul(&a, &b, &mut ref_out, n, n, n);
    for _ in 0..50 {
        let mut out = vec![0.0f64; n*n];
        cjc_runtime::f16::f16_matmul(&a, &b, &mut out, n, n, n);
        for i in 0..n*n {
            assert_eq!(out[i].to_bits(), ref_out[i].to_bits());
        }
    }
}

// ---------------------------------------------------------------------------
// Catastrophic Cancellation Resistance
// ---------------------------------------------------------------------------

#[test]
fn test_binned_cancellation_1e16() {
    let values = vec![1e16, 1.0, -1e16];
    let result = binned_sum_f64(&values);
    assert_eq!(result, 1.0);
}

#[test]
fn test_binned_cancellation_alternating() {
    let mut values = Vec::new();
    for _ in 0..100 {
        values.push(1e15);
        values.push(-1e15);
    }
    values.push(42.0);
    let result = binned_sum_f64(&values);
    assert_eq!(result, 42.0);
}

#[test]
fn test_complex_cancellation() {
    let values = vec![
        ComplexF64::new(1e15, 1e15),
        ComplexF64::new(-1e15, -1e15),
        ComplexF64::new(7.0, -3.0),
    ];
    let result = cjc_runtime::complex::complex_sum(&values);
    assert_eq!(result.re, 7.0);
    assert_eq!(result.im, -3.0);
}

#[test]
fn test_f16_cancellation() {
    let values = vec![
        F16::from_f64(1000.0),
        F16::from_f64(-1000.0),
        F16::from_f64(0.5),
    ];
    let result = cjc_runtime::f16::f16_binned_sum(&values);
    assert_eq!(result, F16::from_f64(0.5).to_f64());
}

// ---------------------------------------------------------------------------
// Merge-Based Parallel Reduction Consistency
// ---------------------------------------------------------------------------

#[test]
fn test_merge_forward_reverse_100_chunks() {
    let mut rng = cjc_repro::Rng::seeded(42);
    let data: Vec<f64> = (0..10_000).map(|_| rng.next_normal_f64() * 100.0).collect();

    let mut fwd = BinnedAccumulatorF64::new();
    for chunk in data.chunks(100) {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        fwd.merge(&c);
    }

    let chunks: Vec<Vec<f64>> = data.chunks(100).map(|c| c.to_vec()).collect();
    let mut rev = BinnedAccumulatorF64::new();
    for chunk in chunks.iter().rev() {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        rev.merge(&c);
    }

    assert_eq!(fwd.finalize().to_bits(), rev.finalize().to_bits(),
        "Forward vs reverse merge must be bit-identical");
}

#[test]
fn test_merge_interleaved_order() {
    let mut rng = cjc_repro::Rng::seeded(99);
    let data: Vec<f64> = (0..1000).map(|_| rng.next_f64() * 50.0 - 25.0).collect();

    let chunk_accs: Vec<BinnedAccumulatorF64> = data.chunks(50)
        .map(|chunk| {
            let mut c = BinnedAccumulatorF64::new();
            c.add_slice(chunk);
            c
        })
        .collect();

    // Forward merge.
    let mut fwd = BinnedAccumulatorF64::new();
    for c in &chunk_accs { fwd.merge(c); }

    // Interleaved merge (even first, then odd).
    let mut interleaved = BinnedAccumulatorF64::new();
    for c in chunk_accs.iter().step_by(2) { interleaved.merge(c); }
    for c in chunk_accs.iter().skip(1).step_by(2) { interleaved.merge(c); }

    assert_eq!(fwd.finalize().to_bits(), interleaved.finalize().to_bits(),
        "Interleaved merge must match forward merge");
}

// ---------------------------------------------------------------------------
// Dispatch Context Tests
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_nogc_uses_binned() {
    let ctx = ReductionContext::nogc();
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let result = dispatch_sum_f64(&data, &ctx);
    let binned = binned_sum_f64(&data);
    // nogc forces binned, so results must be identical.
    assert_eq!(result.to_bits(), binned.to_bits());
}

#[test]
fn test_dispatch_linalg_uses_binned() {
    let ctx = ReductionContext::linalg();
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let result = dispatch_sum_f64(&data, &ctx);
    let binned = binned_sum_f64(&data);
    assert_eq!(result.to_bits(), binned.to_bits());
}

#[test]
fn test_dispatch_strict_parallel_uses_binned() {
    let ctx = ReductionContext::strict_parallel();
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let result = dispatch_sum_f64(&data, &ctx);
    let binned = binned_sum_f64(&data);
    assert_eq!(result.to_bits(), binned.to_bits());
}

// ---------------------------------------------------------------------------
// Mixed-Precision Pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_f16_to_tensor_pipeline() {
    // f16 values → promote to f64 → create Tensor → compute sum.
    let f16_vals: Vec<F16> = (0..100).map(|i| F16::from_f64(i as f64 * 0.5)).collect();
    let f64_vals: Vec<f64> = f16_vals.iter().map(|v| v.to_f64()).collect();
    let t = Tensor::from_vec(f64_vals, &[100]).unwrap();
    let tensor_sum = t.binned_sum();
    let f16_sum = cjc_runtime::f16::f16_binned_sum(&f16_vals);
    assert_eq!(tensor_sum.to_bits(), f16_sum.to_bits());
}

#[test]
fn test_quantized_to_tensor_pipeline() {
    let params = QuantParamsI8::new(0.5, 0);
    let i8_vals: Vec<i8> = (0..50).map(|i| i as i8).collect();
    let f64_vals: Vec<f64> = i8_vals.iter().map(|&v| params.dequantize(v)).collect();
    let t = Tensor::from_vec(f64_vals, &[50]).unwrap();
    let tensor_sum = t.binned_sum();
    let quant_sum = quantized_sum_i8(&i8_vals, &params);
    assert_eq!(tensor_sum.to_bits(), quant_sum.to_bits());
}

#[test]
fn test_complex_real_part_matches_tensor() {
    // Sum of real parts of complex values should match Tensor sum.
    let values: Vec<ComplexF64> = (0..100)
        .map(|i| ComplexF64::new(i as f64, 0.0))
        .collect();
    let complex_result = cjc_runtime::complex::complex_sum(&values);
    let reals: Vec<f64> = values.iter().map(|z| z.re).collect();
    let t = Tensor::from_vec(reals, &[100]).unwrap();
    assert_eq!(complex_result.re.to_bits(), t.binned_sum().to_bits());
}

// ---------------------------------------------------------------------------
// Stack Size Verification
// ---------------------------------------------------------------------------

#[test]
fn test_accumulator_stack_size_bound() {
    let size = std::mem::size_of::<BinnedAccumulatorF64>();
    // 2048 bins * 8 bytes + 2048 comp * 8 bytes + 2048 counts * 4 bytes + overhead
    // ≈ 16384 + 16384 + 8192 + ~20 ≈ 41000 bytes
    assert!(size < 48_000, "BinnedAccumulatorF64 too large: {size} bytes");
}

#[test]
fn test_f16_size() {
    assert_eq!(std::mem::size_of::<F16>(), 2);
}

#[test]
fn test_complex_size() {
    assert_eq!(std::mem::size_of::<ComplexF64>(), 16);
}

#[test]
fn test_quant_params_size() {
    assert!(std::mem::size_of::<QuantParamsI8>() <= 16);
}

// ---------------------------------------------------------------------------
// Type System Coherence
// ---------------------------------------------------------------------------

#[test]
fn test_value_complex_type_name() {
    use cjc_runtime::Value;
    let v = Value::Complex(ComplexF64::new(1.0, 2.0));
    assert_eq!(v.type_name(), "Complex");
}

#[test]
fn test_value_f16_type_name() {
    use cjc_runtime::Value;
    let v = Value::F16(F16::from_f64(1.0));
    assert_eq!(v.type_name(), "F16");
}

#[test]
fn test_value_complex_display() {
    use cjc_runtime::Value;
    let v = Value::Complex(ComplexF64::new(3.0, -4.0));
    let s = format!("{v}");
    assert_eq!(s, "3-4i");
}

#[test]
fn test_value_f16_display() {
    use cjc_runtime::Value;
    let v = Value::F16(F16::from_f64(2.5));
    let s = format!("{v}");
    assert_eq!(s, "2.5");
}

// ---------------------------------------------------------------------------
// Edge: All-NaN Inputs
// ---------------------------------------------------------------------------

#[test]
fn test_all_nan_sum() {
    let values = vec![f64::NAN, f64::NAN, f64::NAN];
    let result = binned_sum_f64(&values);
    assert!(result.is_nan());
}

#[test]
fn test_all_inf_sum() {
    let values = vec![f64::INFINITY, f64::INFINITY];
    let result = binned_sum_f64(&values);
    assert!(result.is_infinite());
}

#[test]
fn test_mixed_inf_nan_sum() {
    let values = vec![f64::INFINITY, f64::NAN, 1.0];
    let result = binned_sum_f64(&values);
    assert!(result.is_nan()); // NaN takes precedence
}

// ---------------------------------------------------------------------------
// Empty Vectors
// ---------------------------------------------------------------------------

#[test]
fn test_empty_binned_sum() {
    assert_eq!(binned_sum_f64(&[]), 0.0);
}

#[test]
fn test_empty_dispatch_sum() {
    let ctx = ReductionContext::default_serial();
    assert_eq!(dispatch_sum_f64(&[], &ctx), 0.0);
}

#[test]
fn test_empty_complex_dot() {
    let result = cjc_runtime::complex::complex_dot(&[], &[]);
    assert_eq!(result.re, 0.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_empty_f16_sum() {
    assert_eq!(cjc_runtime::f16::f16_binned_sum(&[]), 0.0);
}

#[test]
fn test_empty_quantized_sum() {
    let params = QuantParamsI8::new(1.0, 0);
    assert_eq!(quantized_sum_i8(&[], &params), 0.0);
}

// ---------------------------------------------------------------------------
// Gap Coverage: Empty Ragged Rows
// ---------------------------------------------------------------------------

#[test]
fn test_ragged_rows_empty_first_row() {
    // Simulate ragged matrix: row lengths [0, 5, 3, 0, 2]
    let row_data: Vec<Vec<f64>> = vec![
        vec![],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0, 20.0, 30.0],
        vec![],
        vec![100.0, 200.0],
    ];
    let row_sums: Vec<f64> = row_data.iter().map(|row| binned_sum_f64(row)).collect();
    assert_eq!(row_sums, vec![0.0, 15.0, 60.0, 0.0, 300.0]);
}

#[test]
fn test_ragged_rows_all_empty() {
    let row_data: Vec<Vec<f64>> = vec![vec![], vec![], vec![]];
    let row_sums: Vec<f64> = row_data.iter().map(|row| binned_sum_f64(row)).collect();
    assert_eq!(row_sums, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_ragged_rows_merge_determinism() {
    let rows: Vec<Vec<f64>> = vec![
        vec![],
        vec![1e15, -1e15, 0.001],
        vec![],
        vec![1e-300, 1e300, -1e300],
        vec![],
    ];
    // Merge accumulators for all rows in forward and reverse order.
    let mut fwd = BinnedAccumulatorF64::new();
    for row in &rows {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add_slice(row);
        fwd.merge(&acc);
    }
    let mut rev = BinnedAccumulatorF64::new();
    for row in rows.iter().rev() {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add_slice(row);
        rev.merge(&acc);
    }
    assert_eq!(fwd.finalize().to_bits(), rev.finalize().to_bits());
}

#[test]
fn test_ragged_complex_rows_empty() {
    let rows: Vec<Vec<ComplexF64>> = vec![
        vec![],
        vec![ComplexF64::new(1.0, 2.0), ComplexF64::new(3.0, 4.0)],
        vec![],
    ];
    let sums: Vec<ComplexF64> = rows.iter()
        .map(|r| cjc_runtime::complex::complex_sum(r))
        .collect();
    assert_eq!(sums[0].re, 0.0);
    assert_eq!(sums[0].im, 0.0);
    assert_eq!(sums[1].re, 4.0);
    assert_eq!(sums[1].im, 6.0);
    assert_eq!(sums[2].re, 0.0);
    assert_eq!(sums[2].im, 0.0);
}

// ---------------------------------------------------------------------------
// Gap Coverage: Extreme Quantization Scales
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_extreme_tiny_scale() {
    let params = QuantParamsI8::new(1e-30, 0);
    let vals: Vec<i8> = vec![127, -128, 1, -1];
    let result = quantized_sum_i8(&vals, &params);
    // (127 - 128 + 1 - 1) * 1e-30 = -1e-30
    assert!((result - (-1e-30)).abs() < 1e-43);
}

#[test]
fn test_quantized_extreme_huge_scale() {
    let params = QuantParamsI8::new(1e20, 0);
    let vals: Vec<i8> = vec![10, 20, 30];
    let result = quantized_sum_i8(&vals, &params);
    assert_eq!(result, 60.0 * 1e20);
}

#[test]
fn test_quantized_scale_preserves_sign() {
    let params = QuantParamsI8::new(-0.5, 0);
    let vals: Vec<i8> = vec![10, -10];
    let result = quantized_sum_i8(&vals, &params);
    // scale is -0.5, so dequantize(10) = -5.0, dequantize(-10) = 5.0
    // sum = 0.0
    assert_eq!(result, 0.0);
}

#[test]
fn test_i4_extreme_values_sum() {
    // i4 range: -8 to 7.  All extremes.
    let params = QuantParamsI4::new(1.0, 0);
    // Packed: [-8, 7] per byte → 0x87 packs hi=-8 (0x8), lo=7 (0x7)
    // Actually pack: hi nibble = (-8 & 0xF) = 0x8, lo nibble = (7 & 0xF) = 0x7 → byte 0x87
    let packed: Vec<u8> = vec![0x87, 0x87, 0x87, 0x87];
    let result = quantized_sum_i4(&packed, 8, &params);
    // Each byte unpacks to [-8, 7], sum per byte = -1, total = -4
    assert_eq!(result, -4.0);
}

#[test]
fn test_quantized_zero_point_extreme() {
    let params = QuantParamsI8::new(1.0, 127);
    let vals: Vec<i8> = vec![127]; // dequantize: (127 - 127) * 1.0 = 0
    let result = quantized_sum_i8(&vals, &params);
    assert_eq!(result, 0.0);
}

// ---------------------------------------------------------------------------
// Gap Coverage: Complex Signed-Zero Math
// ---------------------------------------------------------------------------

#[test]
fn test_complex_signed_zero_add() {
    let a = ComplexF64::new(0.0, 0.0);
    let b = ComplexF64::new(-0.0, -0.0);
    let c = a.add(b);
    // IEEE 754: 0.0 + (-0.0) = 0.0 (positive zero)
    assert_eq!(c.re, 0.0);
    assert_eq!(c.im, 0.0);
}

#[test]
fn test_complex_signed_zero_mul_fixed() {
    // (0+0i) * (1+1i) = 0+0i using fixed-sequence mul
    let a = ComplexF64::new(0.0, 0.0);
    let b = ComplexF64::new(1.0, 1.0);
    let c = a.mul_fixed(b);
    assert_eq!(c.re, 0.0);
    assert_eq!(c.im, 0.0);
}

#[test]
fn test_complex_neg_zero_real_part() {
    let z = ComplexF64::new(-0.0, 3.0);
    assert!(z.re.is_sign_negative());
    assert_eq!(z.im, 3.0);
}

#[test]
fn test_complex_conj_preserves_signed_zero() {
    let z = ComplexF64::new(-0.0, 0.0);
    let c = z.conj();
    // conj: (re, -im) → (-0.0, -0.0)
    assert!(c.re.is_sign_negative());
    assert!(c.im.is_sign_negative());
}

// ---------------------------------------------------------------------------
// Gap Coverage: f16 Subnormal Round-Trip & Boundary
// ---------------------------------------------------------------------------

#[test]
fn test_f16_subnormal_smallest() {
    // Smallest positive f16 subnormal: 2^(-24) ≈ 5.96e-8
    let smallest = F16(0x0001);
    let val = smallest.to_f64();
    assert!(val > 0.0);
    assert!(val < 1e-7);
    let back = F16::from_f64(val);
    assert_eq!(back.0, 0x0001);
}

#[test]
fn test_f16_max_value_round_trip() {
    let max = F16::from_f64(65504.0);
    assert_eq!(max.to_f64(), 65504.0);
}

#[test]
fn test_f16_overflow_to_inf() {
    let inf = F16::from_f64(100000.0);
    assert!(inf.to_f64().is_infinite());
}

// ---------------------------------------------------------------------------
// Gap Coverage: Accumulator Merge with Special Values
// ---------------------------------------------------------------------------

#[test]
fn test_merge_nan_propagation() {
    let mut a = BinnedAccumulatorF64::new();
    a.add_slice(&[1.0, 2.0, f64::NAN]);
    let mut b = BinnedAccumulatorF64::new();
    b.add_slice(&[3.0, 4.0]);
    a.merge(&b);
    assert!(a.finalize().is_nan());
}

#[test]
fn test_merge_inf_propagation() {
    let mut a = BinnedAccumulatorF64::new();
    a.add_slice(&[1.0, 2.0]);
    let mut b = BinnedAccumulatorF64::new();
    b.add_slice(&[f64::INFINITY]);
    a.merge(&b);
    assert!(a.finalize().is_infinite());
}

#[test]
fn test_merge_inf_nan_priority() {
    // NaN should take priority over Inf in merge
    let mut a = BinnedAccumulatorF64::new();
    a.add_slice(&[f64::INFINITY]);
    let mut b = BinnedAccumulatorF64::new();
    b.add_slice(&[f64::NAN]);
    a.merge(&b);
    assert!(a.finalize().is_nan());
}
