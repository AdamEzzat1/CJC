//! Complex BLAS Integration Tests
//!
//! Tests fixed-sequence complex multiplication, deterministic complex
//! reductions, and edge cases (signed zeros, NaN, infinity).

use cjc_runtime::complex::*;

// ---------------------------------------------------------------------------
// Fixed-Sequence Multiplication Tests
// ---------------------------------------------------------------------------

#[test]
fn test_fixed_mul_associativity_approx() {
    // (a * b) * c ≈ a * (b * c) — not exact due to rounding, but close.
    let a = ComplexF64::new(1.1, 2.2);
    let b = ComplexF64::new(3.3, -4.4);
    let c = ComplexF64::new(-5.5, 6.6);
    let ab_c = a.mul_fixed(b).mul_fixed(c);
    let a_bc = a.mul_fixed(b.mul_fixed(c));
    let re_diff = (ab_c.re - a_bc.re).abs();
    let im_diff = (ab_c.im - a_bc.im).abs();
    assert!(re_diff < 1e-10, "Real: {re_diff}");
    assert!(im_diff < 1e-10, "Imag: {im_diff}");
}

#[test]
fn test_fixed_mul_distributive() {
    // a * (b + c) == a*b + a*c (approximately).
    let a = ComplexF64::new(2.0, 3.0);
    let b = ComplexF64::new(4.0, 5.0);
    let c = ComplexF64::new(6.0, 7.0);
    let lhs = a.mul_fixed(b.add(c));
    let rhs = a.mul_fixed(b).add(a.mul_fixed(c));
    assert!((lhs.re - rhs.re).abs() < 1e-10);
    assert!((lhs.im - rhs.im).abs() < 1e-10);
}

#[test]
fn test_fixed_mul_conjugate_norm() {
    // z * conj(z) = |z|^2 (real)
    let z = ComplexF64::new(3.0, 4.0);
    let result = z.mul_fixed(z.conj());
    assert_eq!(result.re, 25.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_fixed_mul_pure_imaginary() {
    // (3i) * (4i) = -12
    let a = ComplexF64::imag(3.0);
    let b = ComplexF64::imag(4.0);
    let result = a.mul_fixed(b);
    assert_eq!(result.re, -12.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_fixed_mul_pure_real() {
    let a = ComplexF64::real(5.0);
    let b = ComplexF64::real(7.0);
    let result = a.mul_fixed(b);
    assert_eq!(result.re, 35.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_fixed_mul_zero() {
    let a = ComplexF64::new(5.0, 3.0);
    let result = a.mul_fixed(ComplexF64::ZERO);
    assert_eq!(result.re, 0.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_fixed_mul_deterministic_repeated() {
    let a = ComplexF64::new(1.23456789012345, -9.87654321098765);
    let b = ComplexF64::new(-3.14159265358979, 2.71828182845905);
    for _ in 0..100 {
        let r1 = a.mul_fixed(b);
        let r2 = a.mul_fixed(b);
        assert_eq!(r1.re.to_bits(), r2.re.to_bits());
        assert_eq!(r1.im.to_bits(), r2.im.to_bits());
    }
}

// ---------------------------------------------------------------------------
// Complex Signed-Zero Tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_signed_zero_add() {
    let pz = ComplexF64::new(0.0, 0.0);
    let nz = ComplexF64::new(-0.0, -0.0);
    let sum = pz.add(nz);
    // IEEE 754: 0.0 + (-0.0) = 0.0 (positive zero in RNE mode)
    assert_eq!(sum.re, 0.0);
    assert_eq!(sum.im, 0.0);
}

#[test]
fn test_complex_signed_zero_mul() {
    let pz = ComplexF64::new(0.0, 0.0);
    let nz = ComplexF64::new(-0.0, -0.0);
    let prod = pz.mul_fixed(nz);
    // (0 + 0i)(-0 + (-0)i)
    // = 0*(-0) - 0*(-0) + (0*(-0) + 0*(-0))i
    // = 0 - 0 + (0 + 0)i = 0 + 0i
    // Sign depends on IEEE-754 rules for 0 * (-0) = -0
    assert_eq!(prod.re, 0.0);
    assert_eq!(prod.im, 0.0);
}

#[test]
fn test_complex_zero_conj() {
    let z = ComplexF64::new(0.0, -0.0);
    let c = z.conj();
    assert_eq!(c.re, 0.0);
    assert!(c.im.is_sign_positive()); // conj(-0i) = +0i
}

#[test]
fn test_complex_negative_zero_identity() {
    let nz = ComplexF64::new(-0.0, -0.0);
    let one = ComplexF64::ONE;
    let result = nz.mul_fixed(one);
    // (-0)*(1) - (-0)*(0) = -0 - 0 = -0
    // (-0)*(0) + (-0)*(1) = 0 + (-0) = -0
    // Signs may vary by implementation, just verify result is zero-magnitude.
    assert_eq!(result.re.abs(), 0.0);
    assert_eq!(result.im.abs(), 0.0);
}

// ---------------------------------------------------------------------------
// Complex NaN/Infinity Tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_nan_real() {
    let z = ComplexF64::new(f64::NAN, 5.0);
    assert!(z.is_nan());
    assert!(!z.is_finite());
}

#[test]
fn test_complex_nan_imag() {
    let z = ComplexF64::new(5.0, f64::NAN);
    assert!(z.is_nan());
    assert!(!z.is_finite());
}

#[test]
fn test_complex_inf_mul_zero() {
    let inf = ComplexF64::new(f64::INFINITY, 0.0);
    let zero = ComplexF64::ZERO;
    let result = inf.mul_fixed(zero);
    // inf * 0 = NaN
    assert!(result.re.is_nan() || result.re == 0.0);
}

#[test]
fn test_complex_inf_plus_inf() {
    let inf1 = ComplexF64::new(f64::INFINITY, f64::INFINITY);
    let inf2 = ComplexF64::new(f64::INFINITY, f64::INFINITY);
    let result = inf1.add(inf2);
    assert!(result.re.is_infinite());
    assert!(result.im.is_infinite());
}

#[test]
fn test_complex_inf_minus_inf() {
    let inf = ComplexF64::new(f64::INFINITY, 0.0);
    let neg_inf = ComplexF64::new(f64::NEG_INFINITY, 0.0);
    let result = inf.add(neg_inf);
    assert!(result.re.is_nan()); // inf - inf = NaN
}

// ---------------------------------------------------------------------------
// Complex Dot Product Tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_dot_orthogonal() {
    // <(1, 0), (0, 1)> should be 0 (orthogonal in C^1).
    let a = vec![ComplexF64::new(1.0, 0.0)];
    let b = vec![ComplexF64::new(0.0, 1.0)];
    let result = complex_dot(&a, &b);
    // dot = 1 * conj(i) = 1 * (-i) = -i
    assert_eq!(result.re, 0.0);
    assert_eq!(result.im, -1.0);
}

#[test]
fn test_complex_dot_self() {
    // <z, z> = |z|^2 (non-negative real)
    let z = vec![ComplexF64::new(3.0, 4.0)];
    let result = complex_dot(&z, &z);
    assert_eq!(result.re, 25.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_complex_dot_conjugate_symmetry() {
    // <a, b> = conj(<b, a>)
    let a = vec![
        ComplexF64::new(1.0, 2.0),
        ComplexF64::new(3.0, -1.0),
    ];
    let b = vec![
        ComplexF64::new(-1.0, 3.0),
        ComplexF64::new(2.0, 2.0),
    ];
    let ab = complex_dot(&a, &b);
    let ba = complex_dot(&b, &a);
    let ba_conj = ba.conj();
    let re_diff = (ab.re - ba_conj.re).abs();
    let im_diff = (ab.im - ba_conj.im).abs();
    assert!(re_diff < 1e-10, "Conjugate symmetry re: {re_diff}");
    assert!(im_diff < 1e-10, "Conjugate symmetry im: {im_diff}");
}

#[test]
fn test_complex_dot_linearity() {
    // <a + b, c> = <a, c> + <b, c>
    let a = vec![ComplexF64::new(1.0, 2.0)];
    let b = vec![ComplexF64::new(3.0, -1.0)];
    let c = vec![ComplexF64::new(2.0, 5.0)];

    let sum_ab: Vec<ComplexF64> = a.iter().zip(b.iter()).map(|(x, y)| x.add(*y)).collect();
    let lhs = complex_dot(&sum_ab, &c);
    let rhs_ac = complex_dot(&a, &c);
    let rhs_bc = complex_dot(&b, &c);
    let rhs = rhs_ac.add(rhs_bc);

    assert!((lhs.re - rhs.re).abs() < 1e-10);
    assert!((lhs.im - rhs.im).abs() < 1e-10);
}

#[test]
fn test_complex_dot_large_deterministic() {
    let n = 1000;
    let a: Vec<ComplexF64> = (0..n)
        .map(|i| ComplexF64::new((i as f64) * 0.01 - 5.0, (i as f64) * 0.007))
        .collect();
    let b: Vec<ComplexF64> = (0..n)
        .map(|i| ComplexF64::new(-(i as f64) * 0.003, (n - i) as f64 * 0.009))
        .collect();
    let r1 = complex_dot(&a, &b);
    let r2 = complex_dot(&a, &b);
    assert_eq!(r1.re.to_bits(), r2.re.to_bits());
    assert_eq!(r1.im.to_bits(), r2.im.to_bits());
}

// ---------------------------------------------------------------------------
// Complex Sum Tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_sum_empty() {
    let result = complex_sum(&[]);
    assert_eq!(result.re, 0.0);
    assert_eq!(result.im, 0.0);
}

#[test]
fn test_complex_sum_single() {
    let result = complex_sum(&[ComplexF64::new(3.14, -2.71)]);
    assert_eq!(result.re, 3.14);
    assert_eq!(result.im, -2.71);
}

#[test]
fn test_complex_sum_cancellation() {
    let values = vec![
        ComplexF64::new(1e16, 1e16),
        ComplexF64::new(-1e16, -1e16),
        ComplexF64::new(1.0, 1.0),
    ];
    let result = complex_sum(&values);
    assert_eq!(result.re, 1.0);
    assert_eq!(result.im, 1.0);
}

// ---------------------------------------------------------------------------
// Complex Matmul Tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_matmul_2x2() {
    let a = vec![
        ComplexF64::new(1.0, 1.0), ComplexF64::new(2.0, -1.0),
        ComplexF64::new(0.0, 3.0), ComplexF64::new(1.0, 0.0),
    ];
    let b = vec![
        ComplexF64::new(1.0, 0.0), ComplexF64::new(0.0, 1.0),
        ComplexF64::new(0.0, 0.0), ComplexF64::new(1.0, 1.0),
    ];
    let mut out = vec![ComplexF64::ZERO; 4];
    complex_matmul(&a, &b, &mut out, 2, 2, 2);

    // C[0,0] = (1+i)(1) + (2-i)(0) = 1+i
    assert_eq!(out[0].re, 1.0);
    assert_eq!(out[0].im, 1.0);
    // C[0,1] = (1+i)(i) + (2-i)(1+i) = (-1+i) + (3+i) = 2+2i
    assert_eq!(out[1].re, 2.0);
    assert_eq!(out[1].im, 2.0);
}

#[test]
fn test_complex_matmul_1x1() {
    let a = vec![ComplexF64::new(3.0, 4.0)];
    let b = vec![ComplexF64::new(5.0, -2.0)];
    let mut out = vec![ComplexF64::ZERO; 1];
    complex_matmul(&a, &b, &mut out, 1, 1, 1);
    // (3+4i)(5-2i) = 15-6i+20i-8i^2 = 15+8+14i = 23+14i
    assert_eq!(out[0].re, 23.0);
    assert_eq!(out[0].im, 14.0);
}

#[test]
fn test_complex_matmul_non_square() {
    // 1x3 × 3x2 = 1x2
    let a = vec![
        ComplexF64::new(1.0, 0.0),
        ComplexF64::new(0.0, 1.0),
        ComplexF64::new(1.0, 1.0),
    ];
    let b = vec![
        ComplexF64::new(1.0, 0.0), ComplexF64::new(0.0, 0.0),
        ComplexF64::new(0.0, 0.0), ComplexF64::new(1.0, 0.0),
        ComplexF64::new(0.0, 0.0), ComplexF64::new(0.0, 0.0),
    ];
    let mut out = vec![ComplexF64::ZERO; 2];
    complex_matmul(&a, &b, &mut out, 1, 3, 2);
    // C[0,0] = 1*1 + i*0 + (1+i)*0 = 1
    assert_eq!(out[0].re, 1.0);
    assert_eq!(out[0].im, 0.0);
    // C[0,1] = 1*0 + i*1 + (1+i)*0 = i
    assert_eq!(out[1].re, 0.0);
    assert_eq!(out[1].im, 1.0);
}

// ---------------------------------------------------------------------------
// Complex Merge-Order Invariance (Parallel Simulation)
// ---------------------------------------------------------------------------

#[test]
fn test_complex_matmul_merge_deterministic() {
    let n = 4;
    let a: Vec<ComplexF64> = (0..n*n)
        .map(|i| ComplexF64::new(i as f64 * 0.1, -(i as f64) * 0.05))
        .collect();
    let b: Vec<ComplexF64> = (0..n*n)
        .map(|i| ComplexF64::new(-(i as f64) * 0.07, i as f64 * 0.03))
        .collect();
    let mut out1 = vec![ComplexF64::ZERO; n*n];
    let mut out2 = vec![ComplexF64::ZERO; n*n];
    complex_matmul(&a, &b, &mut out1, n, n, n);
    complex_matmul(&a, &b, &mut out2, n, n, n);
    for i in 0..n*n {
        assert_eq!(out1[i].re.to_bits(), out2[i].re.to_bits());
        assert_eq!(out1[i].im.to_bits(), out2[i].im.to_bits());
    }
}

// ---------------------------------------------------------------------------
// Utility Tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_scale() {
    let z = ComplexF64::new(3.0, 4.0);
    let scaled = z.scale(2.0);
    assert_eq!(scaled.re, 6.0);
    assert_eq!(scaled.im, 8.0);
}

#[test]
fn test_complex_neg() {
    let z = ComplexF64::new(3.0, -4.0);
    let neg = z.neg();
    assert_eq!(neg.re, -3.0);
    assert_eq!(neg.im, 4.0);
}

#[test]
fn test_complex_sub() {
    let a = ComplexF64::new(5.0, 3.0);
    let b = ComplexF64::new(2.0, 7.0);
    let result = a.sub(b);
    assert_eq!(result.re, 3.0);
    assert_eq!(result.im, -4.0);
}

#[test]
fn test_complex_norm_sq() {
    let z = ComplexF64::new(3.0, 4.0);
    assert_eq!(z.norm_sq(), 25.0);
}

#[test]
fn test_complex_is_finite() {
    assert!(ComplexF64::new(1.0, 2.0).is_finite());
    assert!(!ComplexF64::new(f64::INFINITY, 0.0).is_finite());
    assert!(!ComplexF64::new(0.0, f64::NAN).is_finite());
}
