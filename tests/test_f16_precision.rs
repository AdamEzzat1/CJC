//! F16 Half-Precision Integration Tests
//!
//! Tests IEEE 754 binary16 conversion, accumulation via BinnedAccumulator,
//! subnormal handling, and precision guarantees.

use cjc_runtime::f16::*;

// ---------------------------------------------------------------------------
// Conversion Correctness
// ---------------------------------------------------------------------------

#[test]
fn test_f16_small_integers() {
    for i in -100..=100 {
        let f16 = F16::from_f64(i as f64);
        assert_eq!(f16.to_f64(), i as f64, "Integer {i} roundtrip failed");
    }
}

#[test]
fn test_f16_powers_of_two() {
    for exp in -14..=15 {
        let val = 2.0f64.powi(exp);
        let f16 = F16::from_f64(val);
        assert_eq!(f16.to_f64(), val, "2^{exp} roundtrip failed");
    }
}

#[test]
fn test_f16_common_values() {
    let values = [0.5, 0.25, 0.125, 1.5, 3.0, 7.0, 15.0, 31.0, 255.0, 1024.0];
    for &v in &values {
        let f16 = F16::from_f64(v);
        assert_eq!(f16.to_f64(), v, "Value {v} roundtrip failed");
    }
}

#[test]
fn test_f16_precision_near_one() {
    // f16 has 10 mantissa bits → precision ≈ 2^(-10) = 1/1024 near 1.0.
    let eps = 1.0 / 1024.0;
    let f16 = F16::from_f64(1.0 + eps);
    let back = f16.to_f64();
    assert!((back - (1.0 + eps)).abs() < eps,
        "Near-one precision: {back} vs {}", 1.0 + eps);
}

#[test]
fn test_f16_pi_approximation() {
    let f16 = F16::from_f64(std::f64::consts::PI);
    let back = f16.to_f64();
    // f16 can represent ≈ 3.14 to 3 significant digits.
    assert!((back - std::f64::consts::PI).abs() < 0.002,
        "Pi: {back} vs {}", std::f64::consts::PI);
}

// ---------------------------------------------------------------------------
// Subnormal Handling
// ---------------------------------------------------------------------------

#[test]
fn test_f16_subnormal_range() {
    // All subnormals: exponent=0, mantissa 1..1023.
    // value = mantissa * 2^(-24)
    for m in [1, 2, 100, 512, 1023] {
        let bits = m as u16; // sign=0, exp=0, mant=m
        let f16 = F16(bits);
        let val = f16.to_f64();
        let expected = (m as f64) * 2.0f64.powi(-24);
        assert!((val - expected).abs() < 1e-15,
            "Subnormal m={m}: {val} vs {expected}");
        assert!(f16.is_subnormal());
    }
}

#[test]
fn test_f16_subnormal_roundtrip() {
    let sub = F16::MIN_POSITIVE_SUBNORMAL;
    let val = sub.to_f64();
    let back = F16::from_f64(val);
    assert_eq!(back.0, sub.0, "Subnormal roundtrip failed");
}

#[test]
fn test_f16_subnormal_addition() {
    let sub = F16::MIN_POSITIVE_SUBNORMAL;
    let result = sub.add(sub);
    // Should be 2 * MIN_POSITIVE_SUBNORMAL.
    let expected = F16(0x0002);
    assert_eq!(result.0, expected.0);
}

#[test]
fn test_f16_subnormal_accumulation_precision() {
    // Accumulate 10000 smallest subnormals via binned path.
    let sub = F16::MIN_POSITIVE_SUBNORMAL;
    let n = 10_000;
    let values = vec![sub; n];
    let result = f16_binned_sum(&values);
    let expected = sub.to_f64() * n as f64;
    // Should be very close (exact for this case since all values are identical).
    assert!((result - expected).abs() < 1e-15,
        "Subnormal accumulation: {result} vs {expected}");
}

// ---------------------------------------------------------------------------
// Special Values
// ---------------------------------------------------------------------------

#[test]
fn test_f16_nan_variants() {
    // All NaN encodings: exp=31, mant!=0.
    for mant in [1, 512, 1023] {
        let bits = 0x7C00 | mant;
        let f16 = F16(bits);
        assert!(f16.is_nan());
        assert!(f16.to_f64().is_nan());
    }
}

#[test]
fn test_f16_nan_from_nan() {
    let f16 = F16::from_f64(f64::NAN);
    assert!(f16.is_nan());
}

#[test]
fn test_f16_inf_arithmetic() {
    let inf = F16::INFINITY;
    let one = F16::from_f64(1.0);
    let result = inf.add(one);
    assert!(result.is_infinite());
}

#[test]
fn test_f16_neg_inf_from_f64() {
    let f16 = F16::from_f64(f64::NEG_INFINITY);
    assert!(f16.is_infinite());
    assert!(f16.to_f64().is_sign_negative());
}

#[test]
fn test_f16_zero_variations() {
    // +0 and -0 should be distinct in bits but equal in value.
    let pz = F16::ZERO;
    let nz = F16::NEG_ZERO;
    assert_ne!(pz.0, nz.0);
    assert_eq!(pz.to_f64(), nz.to_f64()); // Both == 0.0
}

// ---------------------------------------------------------------------------
// f16 Binned Accumulation Tests
// ---------------------------------------------------------------------------

#[test]
fn test_f16_binned_sum_large() {
    let n = 5000;
    let values: Vec<F16> = (0..n)
        .map(|i| F16::from_f64((i as f64 - 2500.0) * 0.1))
        .collect();
    let result = f16_binned_sum(&values);
    // Sum of arithmetic progression, but with f16 quantization.
    // Each value may lose precision.
    assert!(result.is_finite());
}

#[test]
fn test_f16_binned_sum_all_same() {
    let val = F16::from_f64(1.5);
    let n = 1000;
    let values = vec![val; n];
    let result = f16_binned_sum(&values);
    let expected = val.to_f64() * n as f64;
    assert!((result - expected).abs() < 1e-10,
        "All-same sum: {result} vs {expected}");
}

#[test]
fn test_f16_binned_sum_cancellation() {
    // Large positive + large negative + small positive.
    let values = vec![
        F16::from_f64(65504.0),  // f16 max
        F16::from_f64(-65504.0), // f16 min
        F16::from_f64(1.0),
    ];
    let result = f16_binned_sum(&values);
    assert_eq!(result, 1.0);
}

#[test]
fn test_f16_binned_sum_mixed_magnitudes() {
    let values = vec![
        F16::from_f64(65504.0),
        F16::from_f64(0.001),
        F16::from_f64(-65504.0),
    ];
    let result = f16_binned_sum(&values);
    // 65504 - 65504 + 0.001 ≈ 0.001 (but f16 0.001 rounds)
    let expected = F16::from_f64(0.001).to_f64();
    assert!((result - expected).abs() < 0.01,
        "Mixed magnitudes: {result} vs {expected}");
}

// ---------------------------------------------------------------------------
// f16 Dot Product Tests
// ---------------------------------------------------------------------------

#[test]
fn test_f16_dot_orthogonal() {
    let a = vec![F16::from_f64(1.0), F16::from_f64(0.0)];
    let b = vec![F16::from_f64(0.0), F16::from_f64(1.0)];
    assert_eq!(f16_binned_dot(&a, &b), 0.0);
}

#[test]
fn test_f16_dot_parallel() {
    let a = vec![F16::from_f64(3.0), F16::from_f64(4.0)];
    let b = vec![F16::from_f64(3.0), F16::from_f64(4.0)];
    assert_eq!(f16_binned_dot(&a, &b), 25.0);
}

#[test]
fn test_f16_dot_large() {
    let n = 1000;
    let a: Vec<F16> = (0..n).map(|i| F16::from_f64((i as f64) * 0.01)).collect();
    let b: Vec<F16> = (0..n).map(|i| F16::from_f64(1.0 - (i as f64) * 0.001)).collect();
    let result = f16_binned_dot(&a, &b);
    assert!(result.is_finite());
}

#[test]
fn test_f16_dot_deterministic() {
    let n = 500;
    let a: Vec<F16> = (0..n).map(|i| F16::from_f64(i as f64 * 0.1)).collect();
    let b: Vec<F16> = (0..n).map(|i| F16::from_f64(-(i as f64) * 0.05 + 10.0)).collect();
    let r1 = f16_binned_dot(&a, &b);
    let r2 = f16_binned_dot(&a, &b);
    assert_eq!(r1.to_bits(), r2.to_bits());
}

// ---------------------------------------------------------------------------
// f16 Matmul Tests
// ---------------------------------------------------------------------------

#[test]
fn test_f16_matmul_2x2() {
    let a = vec![
        F16::from_f64(1.0), F16::from_f64(2.0),
        F16::from_f64(3.0), F16::from_f64(4.0),
    ];
    let b = vec![
        F16::from_f64(5.0), F16::from_f64(6.0),
        F16::from_f64(7.0), F16::from_f64(8.0),
    ];
    let mut out = vec![0.0f64; 4];
    f16_matmul(&a, &b, &mut out, 2, 2, 2);
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_f16_matmul_deterministic() {
    let n = 4;
    let a: Vec<F16> = (0..n*n).map(|i| F16::from_f64(i as f64 * 0.5)).collect();
    let b: Vec<F16> = (0..n*n).map(|i| F16::from_f64(-(i as f64) * 0.3 + 2.0)).collect();
    let mut out1 = vec![0.0f64; n*n];
    let mut out2 = vec![0.0f64; n*n];
    f16_matmul(&a, &b, &mut out1, n, n, n);
    f16_matmul(&a, &b, &mut out2, n, n, n);
    for i in 0..n*n {
        assert_eq!(out1[i].to_bits(), out2[i].to_bits(),
            "f16 matmul element {i} differs");
    }
}

#[test]
fn test_f16_matmul_non_square() {
    // 2x3 × 3x1 = 2x1
    let a = vec![
        F16::from_f64(1.0), F16::from_f64(2.0), F16::from_f64(3.0),
        F16::from_f64(4.0), F16::from_f64(5.0), F16::from_f64(6.0),
    ];
    let b = vec![F16::from_f64(1.0), F16::from_f64(1.0), F16::from_f64(1.0)];
    let mut out = vec![0.0f64; 2];
    f16_matmul(&a, &b, &mut out, 2, 3, 1);
    assert_eq!(out, vec![6.0, 15.0]);
}

// ---------------------------------------------------------------------------
// f16 Arithmetic Edge Cases
// ---------------------------------------------------------------------------

#[test]
fn test_f16_div_by_zero() {
    let a = F16::from_f64(1.0);
    let zero = F16::ZERO;
    let result = a.div(zero);
    assert!(result.is_infinite());
}

#[test]
fn test_f16_zero_div_zero() {
    let zero = F16::ZERO;
    let result = zero.div(zero);
    assert!(result.is_nan());
}

#[test]
fn test_f16_max_plus_one() {
    // MAX (65504) + smallest increment should overflow to inf.
    let max = F16::MAX;
    let small = F16::from_f64(32.0); // Smallest increment at this range
    let result = max.add(small);
    // 65504 + 32 = 65536, which overflows f16.
    assert!(result.is_infinite() || result.to_f64() == 65504.0);
}

#[test]
fn test_f16_negative_max() {
    let neg_max = F16::from_f64(-65504.0);
    assert_eq!(neg_max.to_f64(), -65504.0);
}

#[test]
fn test_f16_denorm_boundary() {
    // The boundary between subnormal and normal: 2^(-14) = 6.103515625e-5
    let boundary = 6.103515625e-5;
    let f16 = F16::from_f64(boundary);
    let back = f16.to_f64();
    assert!((back - boundary).abs() < 1e-10);
    assert!(!f16.is_subnormal());
}

#[test]
fn test_f16_just_below_denorm_boundary() {
    let below = 6.0e-5;
    let f16 = F16::from_f64(below);
    let back = f16.to_f64();
    // Should be close to below, possibly subnormal.
    assert!((back - below).abs() / below < 0.1);
}
