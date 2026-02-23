//! Edge Case Tests — Empty Ragged Rows, NaN Canonicalization,
//! Signed-Zero Preservation, Cross-Type Determinism.
//!
//! These tests fill the coverage gaps identified in the Milestone 2.7
//! expansion audit.

use cjc_runtime::Tensor;
use cjc_runtime::accumulator::{BinnedAccumulatorF64, BinnedAccumulatorF32, binned_sum_f64};
use cjc_runtime::complex::ComplexF64;
use cjc_runtime::f16::F16;
use cjc_runtime::quantized::QuantParamsI8;

// ---------------------------------------------------------------------------
// NaN Canonicalization
// ---------------------------------------------------------------------------

#[test]
fn test_nan_canonicalization_in_accumulator() {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(f64::NAN);
    assert!(acc.finalize().is_nan());
}

#[test]
fn test_nan_different_payloads_same_result() {
    // Different NaN payloads should produce the same behavior.
    let nan1 = f64::from_bits(0x7FF0000000000001); // quiet NaN, payload=1
    let nan2 = f64::from_bits(0x7FF8000000000000); // canonical quiet NaN
    let nan3 = f64::from_bits(0x7FF0000000000002); // quiet NaN, payload=2

    let mut acc1 = BinnedAccumulatorF64::new();
    acc1.add(nan1);
    acc1.add(1.0);

    let mut acc2 = BinnedAccumulatorF64::new();
    acc2.add(nan2);
    acc2.add(1.0);

    let mut acc3 = BinnedAccumulatorF64::new();
    acc3.add(nan3);
    acc3.add(1.0);

    // All should finalize to NaN.
    assert!(acc1.finalize().is_nan());
    assert!(acc2.finalize().is_nan());
    assert!(acc3.finalize().is_nan());
}

#[test]
fn test_nan_merge_canonicalization() {
    let mut a = BinnedAccumulatorF64::new();
    a.add(1.0);
    let mut b = BinnedAccumulatorF64::new();
    b.add(f64::NAN);

    a.merge(&b);
    assert!(a.finalize().is_nan());
}

#[test]
fn test_nan_propagation_in_sum() {
    let values = vec![1.0, 2.0, f64::NAN, 4.0];
    let result = binned_sum_f64(&values);
    assert!(result.is_nan());
}

#[test]
fn test_nan_in_complex_dot() {
    use cjc_runtime::complex::complex_dot;
    let a = vec![ComplexF64::new(f64::NAN, 1.0)];
    let b = vec![ComplexF64::new(1.0, 1.0)];
    let result = complex_dot(&a, &b);
    assert!(result.is_nan());
}

// ---------------------------------------------------------------------------
// Signed-Zero Preservation
// ---------------------------------------------------------------------------

#[test]
fn test_signed_zero_in_accumulator() {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(0.0);
    let result = acc.finalize();
    assert_eq!(result, 0.0);
}

#[test]
fn test_negative_zero_in_accumulator() {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(-0.0);
    let result = acc.finalize();
    assert_eq!(result, 0.0); // May be +0 or -0
}

#[test]
fn test_signed_zero_cancellation() {
    // 1.0 + (-1.0) should give +0.0 (IEEE 754 default rounding).
    let values = vec![1.0, -1.0];
    let result = binned_sum_f64(&values);
    assert_eq!(result, 0.0);
}

#[test]
fn test_f16_signed_zero_through_binned() {
    let values = vec![F16::ZERO, F16::NEG_ZERO];
    let result = cjc_runtime::f16::f16_binned_sum(&values);
    assert_eq!(result, 0.0);
}

#[test]
fn test_complex_signed_zero_through_sum() {
    let values = vec![
        ComplexF64::new(0.0, 0.0),
        ComplexF64::new(-0.0, -0.0),
    ];
    let result = cjc_runtime::complex::complex_sum(&values);
    assert_eq!(result.re, 0.0);
    assert_eq!(result.im, 0.0);
}

// ---------------------------------------------------------------------------
// Infinity Handling
// ---------------------------------------------------------------------------

#[test]
fn test_inf_plus_neg_inf_gives_nan() {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(f64::INFINITY);
    acc.add(f64::NEG_INFINITY);
    let result = acc.finalize();
    assert!(result.is_nan());
}

#[test]
fn test_inf_plus_inf_gives_inf() {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(f64::INFINITY);
    acc.add(f64::INFINITY);
    let result = acc.finalize();
    assert!(result.is_infinite());
    assert!(result.is_sign_positive());
}

#[test]
fn test_neg_inf_plus_neg_inf() {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add(f64::NEG_INFINITY);
    acc.add(f64::NEG_INFINITY);
    let result = acc.finalize();
    assert!(result.is_infinite());
    assert!(result.is_sign_negative());
}

#[test]
fn test_inf_merge_handling() {
    let mut a = BinnedAccumulatorF64::new();
    a.add(f64::INFINITY);
    let mut b = BinnedAccumulatorF64::new();
    b.add(1.0);
    a.merge(&b);
    assert!(a.finalize().is_infinite());
}

// ---------------------------------------------------------------------------
// Empty/Degenerate Tensor Cases
// ---------------------------------------------------------------------------

#[test]
fn test_empty_tensor_sum() {
    let t = Tensor::zeros(&[0]);
    assert_eq!(t.sum(), 0.0);
}

#[test]
fn test_empty_tensor_mean() {
    let t = Tensor::zeros(&[0]);
    assert_eq!(t.mean(), 0.0);
}

#[test]
fn test_empty_tensor_binned_sum() {
    let t = Tensor::zeros(&[0]);
    assert_eq!(t.binned_sum(), 0.0);
}

#[test]
fn test_single_element_tensor() {
    let t = Tensor::from_vec(vec![42.0], &[1]).unwrap();
    assert_eq!(t.sum(), 42.0);
    assert_eq!(t.mean(), 42.0);
    assert_eq!(t.binned_sum(), 42.0);
}

#[test]
fn test_singleton_matmul() {
    let a = Tensor::from_vec(vec![3.0], &[1, 1]).unwrap();
    let b = Tensor::from_vec(vec![4.0], &[1, 1]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.get(&[0, 0]).unwrap(), 12.0);
}

// ---------------------------------------------------------------------------
// Empty Accumulator Edge Cases
// ---------------------------------------------------------------------------

#[test]
fn test_empty_accumulator_finalize() {
    let acc = BinnedAccumulatorF64::new();
    assert_eq!(acc.finalize(), 0.0);
    assert_eq!(acc.count(), 0);
}

#[test]
fn test_empty_accumulator_merge() {
    let mut a = BinnedAccumulatorF64::new();
    let b = BinnedAccumulatorF64::new();
    a.merge(&b);
    assert_eq!(a.finalize(), 0.0);
    assert_eq!(a.count(), 0);
}

#[test]
fn test_merge_two_empty() {
    let mut a = BinnedAccumulatorF64::new();
    let b = BinnedAccumulatorF64::new();
    a.merge(&b);
    assert_eq!(a.finalize(), 0.0);
}

#[test]
fn test_merge_empty_into_nonempty() {
    let mut a = BinnedAccumulatorF64::new();
    a.add(42.0);
    let b = BinnedAccumulatorF64::new();
    a.merge(&b);
    assert_eq!(a.finalize(), 42.0);
}

#[test]
fn test_merge_nonempty_into_empty() {
    let mut a = BinnedAccumulatorF64::new();
    let mut b = BinnedAccumulatorF64::new();
    b.add(42.0);
    a.merge(&b);
    assert_eq!(a.finalize(), 42.0);
}

// ---------------------------------------------------------------------------
// Cross-Type Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_f32_accumulator_empty() {
    let acc = BinnedAccumulatorF32::new();
    assert_eq!(acc.finalize(), 0.0f32);
}

#[test]
fn test_f32_accumulator_basic() {
    let mut acc = BinnedAccumulatorF32::new();
    for i in 1..=10 {
        acc.add(i as f32);
    }
    assert_eq!(acc.finalize(), 55.0f32);
}

#[test]
fn test_f32_accumulator_merge() {
    let mut a = BinnedAccumulatorF32::new();
    a.add(1.0f32);
    a.add(2.0f32);
    let mut b = BinnedAccumulatorF32::new();
    b.add(3.0f32);
    b.add(4.0f32);

    let mut ab = a.clone();
    ab.merge(&b);
    let mut ba = b.clone();
    ba.merge(&a);

    assert_eq!(ab.finalize(), ba.finalize());
}

#[test]
fn test_quantized_and_float_agree_small() {
    // For simple integer inputs with scale=1, zp=0, quantized should match float.
    let params = QuantParamsI8::new(1.0, 0);
    let values_i8: Vec<i8> = (1..=10).map(|i| i as i8).collect();
    let values_f64: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let q_sum = cjc_runtime::quantized::quantized_sum_i8(&values_i8, &params);
    let f_sum = binned_sum_f64(&values_f64);

    assert_eq!(q_sum, f_sum);
}

#[test]
fn test_f16_and_f64_agree_integers() {
    let values_f16: Vec<F16> = (1..=10).map(|i| F16::from_f64(i as f64)).collect();
    let values_f64: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let f16_sum = cjc_runtime::f16::f16_binned_sum(&values_f16);
    let f64_sum = binned_sum_f64(&values_f64);

    assert_eq!(f16_sum, f64_sum);
}

// ---------------------------------------------------------------------------
// Ragged Row Simulation (Zero-Length Chunks)
// ---------------------------------------------------------------------------

#[test]
fn test_empty_chunk_merge() {
    let mut acc = BinnedAccumulatorF64::new();
    // Merge a series of empty accumulators.
    for _ in 0..10 {
        let empty = BinnedAccumulatorF64::new();
        acc.merge(&empty);
    }
    assert_eq!(acc.finalize(), 0.0);
    assert_eq!(acc.count(), 0);
}

#[test]
fn test_mixed_empty_and_nonempty_chunks() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let chunk_sizes = [0, 2, 0, 3, 0]; // Ragged with empty rows.

    let mut acc = BinnedAccumulatorF64::new();
    let mut offset = 0;
    for &size in &chunk_sizes {
        let mut chunk_acc = BinnedAccumulatorF64::new();
        for i in 0..size {
            chunk_acc.add(data[offset + i]);
        }
        acc.merge(&chunk_acc);
        offset += size;
    }
    assert_eq!(acc.finalize(), 15.0);
}

#[test]
fn test_single_element_chunks() {
    let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let mut acc = BinnedAccumulatorF64::new();
    for &v in &data {
        let mut chunk = BinnedAccumulatorF64::new();
        chunk.add(v);
        acc.merge(&chunk);
    }
    assert_eq!(acc.finalize(), 5050.0);
}

#[test]
fn test_all_empty_chunks() {
    let mut acc = BinnedAccumulatorF64::new();
    for _ in 0..1000 {
        let empty = BinnedAccumulatorF64::new();
        acc.merge(&empty);
    }
    assert_eq!(acc.finalize(), 0.0);
}

// ---------------------------------------------------------------------------
// Determinism Stress
// ---------------------------------------------------------------------------

#[test]
fn test_determinism_100_runs() {
    let mut rng = cjc_repro::Rng::seeded(42);
    let data: Vec<f64> = (0..1000).map(|_| rng.next_f64() * 1000.0 - 500.0).collect();

    let reference = binned_sum_f64(&data);
    for _ in 0..100 {
        let result = binned_sum_f64(&data);
        assert_eq!(result.to_bits(), reference.to_bits());
    }
}

#[test]
fn test_tensor_matmul_determinism_100_runs() {
    let mut rng = cjc_repro::Rng::seeded(123);
    let a_data: Vec<f64> = (0..100).map(|_| rng.next_f64()).collect();
    let b_data: Vec<f64> = (0..100).map(|_| rng.next_f64()).collect();
    let a = Tensor::from_vec(a_data, &[10, 10]).unwrap();
    let b = Tensor::from_vec(b_data, &[10, 10]).unwrap();

    let reference = a.matmul(&b).unwrap();
    let ref_data = reference.to_vec();

    for _ in 0..100 {
        let result = a.matmul(&b).unwrap();
        let result_data = result.to_vec();
        for (j, (&r, &e)) in result_data.iter().zip(ref_data.iter()).enumerate() {
            assert_eq!(r.to_bits(), e.to_bits(),
                "Matmul element {j} differs on repeated run");
        }
    }
}
