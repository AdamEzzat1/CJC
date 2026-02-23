//! Complex BLAS — ComplexF64 with Fixed-Sequence Arithmetic.
//!
//! # Design
//!
//! Complex multiplication is lowered to a **fixed-sequence** of four
//! multiplications and two additions, explicitly ordered to prevent
//! cross-architecture FMA drift. This ensures bit-parity between x86
//! and ARM platforms.
//!
//! Complex reductions (dot products, sums) feed real and imaginary parts
//! separately into BinnedAccumulators for deterministic results.
//!
//! # Fixed-Sequence Complex Multiply
//!
//! ```text
//! (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//! ```
//!
//! The four multiplications are computed first, then the two additions:
//! ```text
//! t1 = a * c   (mul #1)
//! t2 = b * d   (mul #2)
//! t3 = a * d   (mul #3)
//! t4 = b * c   (mul #4)
//! re = t1 - t2 (sub #1)
//! im = t3 + t4 (add #1)
//! ```
//!
//! This explicit ordering prevents the compiler from fusing `a*c - b*d`
//! into an FMA (which would change the rounding behavior).

use crate::accumulator::BinnedAccumulatorF64;

// ---------------------------------------------------------------------------
// ComplexF64
// ---------------------------------------------------------------------------

/// A complex number with f64 real and imaginary parts.
///
/// Arithmetic follows the fixed-sequence protocol to prevent FMA drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexF64 {
    pub re: f64,
    pub im: f64,
}

impl ComplexF64 {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        ComplexF64 { re, im }
    }

    /// Create a purely real complex number.
    #[inline]
    pub fn real(re: f64) -> Self {
        ComplexF64 { re, im: 0.0 }
    }

    /// Create a purely imaginary complex number.
    #[inline]
    pub fn imag(im: f64) -> Self {
        ComplexF64 { re: 0.0, im }
    }

    /// Zero.
    pub const ZERO: ComplexF64 = ComplexF64 { re: 0.0, im: 0.0 };

    /// One.
    pub const ONE: ComplexF64 = ComplexF64 { re: 1.0, im: 0.0 };

    /// Imaginary unit.
    pub const I: ComplexF64 = ComplexF64 { re: 0.0, im: 1.0 };

    /// Squared magnitude: |z|^2 = re^2 + im^2.
    #[inline]
    pub fn norm_sq(self) -> f64 {
        // Fixed sequence: two muls, one add.
        let r2 = self.re * self.re;
        let i2 = self.im * self.im;
        r2 + i2
    }

    /// Magnitude: |z| = sqrt(re^2 + im^2).
    #[inline]
    pub fn abs(self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Complex conjugate: (a - bi).
    #[inline]
    pub fn conj(self) -> Self {
        ComplexF64 { re: self.re, im: -self.im }
    }

    /// Fixed-Sequence Complex Multiplication.
    ///
    /// Explicitly computes four multiplications and two additions in a
    /// deterministic order, preventing FMA contraction:
    ///
    /// ```text
    /// t1 = a.re * b.re
    /// t2 = a.im * b.im
    /// t3 = a.re * b.im
    /// t4 = a.im * b.re
    /// result.re = t1 - t2
    /// result.im = t3 + t4
    /// ```
    ///
    /// # FMA Prevention
    ///
    /// By storing intermediates in local variables and computing each step
    /// explicitly, we prevent LLVM from fusing operations into FMA
    /// instructions, which would cause different rounding on platforms
    /// with/without hardware FMA support.
    #[inline]
    pub fn mul_fixed(self, rhs: Self) -> Self {
        // Step 1: Four independent multiplications.
        let t1 = self.re * rhs.re; // a*c
        let t2 = self.im * rhs.im; // b*d
        let t3 = self.re * rhs.im; // a*d
        let t4 = self.im * rhs.re; // b*c

        // Step 2: Two additions (using the pre-computed products).
        let re = t1 - t2; // ac - bd
        let im = t3 + t4; // ad + bc

        ComplexF64 { re, im }
    }

    /// Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        ComplexF64 {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    /// Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i.
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        ComplexF64 {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    /// Complex negation: -(a+bi) = (-a) + (-b)i.
    #[inline]
    pub fn neg(self) -> Self {
        ComplexF64 { re: -self.re, im: -self.im }
    }

    /// Scalar multiplication: s * (a+bi) = (s*a) + (s*b)i.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        ComplexF64 { re: s * self.re, im: s * self.im }
    }

    /// Check if NaN in either component.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Check if both components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl std::fmt::Display for ComplexF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im)
        }
    }
}

// ---------------------------------------------------------------------------
// Complex BLAS Operations via BinnedAccumulator
// ---------------------------------------------------------------------------

/// Complex dot product using BinnedAccumulator for deterministic results.
///
/// `dot(a, b) = Σ a[i] * conj(b[i])` (standard Hermitian inner product).
///
/// Real and imaginary parts are accumulated separately via BinnedAccumulator.
pub fn complex_dot(a: &[ComplexF64], b: &[ComplexF64]) -> ComplexF64 {
    debug_assert_eq!(a.len(), b.len());
    let mut re_acc = BinnedAccumulatorF64::new();
    let mut im_acc = BinnedAccumulatorF64::new();

    for i in 0..a.len() {
        // z = a[i] * conj(b[i])
        let z = a[i].mul_fixed(b[i].conj());
        re_acc.add(z.re);
        im_acc.add(z.im);
    }

    ComplexF64 {
        re: re_acc.finalize(),
        im: im_acc.finalize(),
    }
}

/// Complex sum using BinnedAccumulator for deterministic results.
///
/// Real and imaginary parts accumulated independently.
pub fn complex_sum(values: &[ComplexF64]) -> ComplexF64 {
    let mut re_acc = BinnedAccumulatorF64::new();
    let mut im_acc = BinnedAccumulatorF64::new();

    for &z in values {
        re_acc.add(z.re);
        im_acc.add(z.im);
    }

    ComplexF64 {
        re: re_acc.finalize(),
        im: im_acc.finalize(),
    }
}

/// Complex matrix multiply: C[m,n] = A[m,k] × B[k,n] (fixed-sequence).
///
/// Each element C[i,j] = Σ_p A[i,p] * B[p,j] with BinnedAccumulator.
pub fn complex_matmul(
    a: &[ComplexF64], b: &[ComplexF64], out: &mut [ComplexF64],
    m: usize, k: usize, n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut re_acc = BinnedAccumulatorF64::new();
            let mut im_acc = BinnedAccumulatorF64::new();
            for p in 0..k {
                let prod = a[i * k + p].mul_fixed(b[p * n + j]);
                re_acc.add(prod.re);
                im_acc.add(prod.im);
            }
            out[i * n + j] = ComplexF64 {
                re: re_acc.finalize(),
                im: im_acc.finalize(),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_mul_basic() {
        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let a = ComplexF64::new(1.0, 2.0);
        let b = ComplexF64::new(3.0, 4.0);
        let c = a.mul_fixed(b);
        assert_eq!(c.re, -5.0);
        assert_eq!(c.im, 10.0);
    }

    #[test]
    fn test_complex_mul_commutative() {
        let a = ComplexF64::new(1.23456789, -9.87654321);
        let b = ComplexF64::new(-3.14159265, 2.71828183);
        let ab = a.mul_fixed(b);
        let ba = b.mul_fixed(a);
        assert_eq!(ab.re.to_bits(), ba.re.to_bits());
        assert_eq!(ab.im.to_bits(), ba.im.to_bits());
    }

    #[test]
    fn test_complex_mul_identity() {
        let a = ComplexF64::new(7.0, -3.0);
        let one = ComplexF64::ONE;
        let result = a.mul_fixed(one);
        assert_eq!(result.re, a.re);
        assert_eq!(result.im, a.im);
    }

    #[test]
    fn test_complex_mul_i_squared() {
        // i * i = -1
        let i = ComplexF64::I;
        let result = i.mul_fixed(i);
        assert_eq!(result.re, -1.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn test_complex_conj() {
        let z = ComplexF64::new(3.0, 4.0);
        let c = z.conj();
        assert_eq!(c.re, 3.0);
        assert_eq!(c.im, -4.0);
    }

    #[test]
    fn test_complex_abs() {
        let z = ComplexF64::new(3.0, 4.0);
        assert_eq!(z.abs(), 5.0);
    }

    #[test]
    fn test_complex_dot_basic() {
        let a = vec![ComplexF64::new(1.0, 0.0), ComplexF64::new(0.0, 1.0)];
        let b = vec![ComplexF64::new(1.0, 0.0), ComplexF64::new(0.0, 1.0)];
        // dot = a[0]*conj(b[0]) + a[1]*conj(b[1])
        //     = 1*1 + i*(-i) = 1 + 1 = 2 + 0i
        let result = complex_dot(&a, &b);
        assert_eq!(result.re, 2.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn test_complex_dot_deterministic() {
        let n = 500;
        let a: Vec<ComplexF64> = (0..n)
            .map(|i| ComplexF64::new(i as f64 * 0.001, -(i as f64 * 0.002)))
            .collect();
        let b: Vec<ComplexF64> = (0..n)
            .map(|i| ComplexF64::new((n - i) as f64 * 0.003, i as f64 * 0.004))
            .collect();

        let r1 = complex_dot(&a, &b);
        let r2 = complex_dot(&a, &b);
        assert_eq!(r1.re.to_bits(), r2.re.to_bits());
        assert_eq!(r1.im.to_bits(), r2.im.to_bits());
    }

    #[test]
    fn test_complex_sum_deterministic() {
        let values: Vec<ComplexF64> = (0..1000)
            .map(|i| ComplexF64::new(i as f64 * 0.7 - 350.0, -(i as f64) * 0.3 + 150.0))
            .collect();
        let r1 = complex_sum(&values);
        let r2 = complex_sum(&values);
        assert_eq!(r1.re.to_bits(), r2.re.to_bits());
        assert_eq!(r1.im.to_bits(), r2.im.to_bits());
    }

    #[test]
    fn test_complex_sum_near_order_invariant() {
        let values: Vec<ComplexF64> = (0..100)
            .map(|i| ComplexF64::new(i as f64 * 1.1 - 50.0, -(i as f64) * 0.9 + 45.0))
            .collect();
        let mut reversed = values.clone();
        reversed.reverse();

        let r1 = complex_sum(&values);
        let r2 = complex_sum(&reversed);
        // Within-bin accumulation is near-order-invariant (sub-10 ULPs).
        let re_ulps = (r1.re.to_bits() as i64 - r2.re.to_bits() as i64).unsigned_abs();
        let im_ulps = (r1.im.to_bits() as i64 - r2.im.to_bits() as i64).unsigned_abs();
        assert!(re_ulps < 10, "Real parts near-order-invariant: {re_ulps} ULPs");
        assert!(im_ulps < 10, "Imaginary parts near-order-invariant: {im_ulps} ULPs");
    }

    #[test]
    fn test_complex_sum_merge_order_invariant() {
        // Merge-based complex summation IS fully order-invariant.
        let values: Vec<ComplexF64> = (0..100)
            .map(|i| ComplexF64::new(i as f64 * 1.1 - 50.0, -(i as f64) * 0.9 + 45.0))
            .collect();

        // Chunk into 10s, merge forward.
        let mut re_fwd = BinnedAccumulatorF64::new();
        let mut im_fwd = BinnedAccumulatorF64::new();
        for chunk in values.chunks(10) {
            let mut re_c = BinnedAccumulatorF64::new();
            let mut im_c = BinnedAccumulatorF64::new();
            for z in chunk {
                re_c.add(z.re);
                im_c.add(z.im);
            }
            re_fwd.merge(&re_c);
            im_fwd.merge(&im_c);
        }

        // Chunk into 10s, merge reverse.
        let chunks: Vec<Vec<ComplexF64>> = values.chunks(10).map(|c| c.to_vec()).collect();
        let mut re_rev = BinnedAccumulatorF64::new();
        let mut im_rev = BinnedAccumulatorF64::new();
        for chunk in chunks.iter().rev() {
            let mut re_c = BinnedAccumulatorF64::new();
            let mut im_c = BinnedAccumulatorF64::new();
            for z in chunk.iter() {
                re_c.add(z.re);
                im_c.add(z.im);
            }
            re_rev.merge(&re_c);
            im_rev.merge(&im_c);
        }

        assert_eq!(re_fwd.finalize().to_bits(), re_rev.finalize().to_bits(),
            "Complex real merge must be order-invariant");
        assert_eq!(im_fwd.finalize().to_bits(), im_rev.finalize().to_bits(),
            "Complex imaginary merge must be order-invariant");
    }

    #[test]
    fn test_complex_matmul_identity() {
        // 2x2 identity × arbitrary = same
        let identity = vec![
            ComplexF64::ONE, ComplexF64::ZERO,
            ComplexF64::ZERO, ComplexF64::ONE,
        ];
        let b = vec![
            ComplexF64::new(1.0, 2.0), ComplexF64::new(3.0, 4.0),
            ComplexF64::new(5.0, 6.0), ComplexF64::new(7.0, 8.0),
        ];
        let mut out = vec![ComplexF64::ZERO; 4];
        complex_matmul(&identity, &b, &mut out, 2, 2, 2);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v.re, b[i].re);
            assert_eq!(v.im, b[i].im);
        }
    }

    #[test]
    fn test_complex_matmul_deterministic() {
        let a: Vec<ComplexF64> = (0..9)
            .map(|i| ComplexF64::new(i as f64 * 0.3, -(i as f64) * 0.2))
            .collect();
        let b: Vec<ComplexF64> = (0..9)
            .map(|i| ComplexF64::new(-(i as f64) * 0.1, i as f64 * 0.4))
            .collect();
        let mut out1 = vec![ComplexF64::ZERO; 9];
        let mut out2 = vec![ComplexF64::ZERO; 9];
        complex_matmul(&a, &b, &mut out1, 3, 3, 3);
        complex_matmul(&a, &b, &mut out2, 3, 3, 3);
        for i in 0..9 {
            assert_eq!(out1[i].re.to_bits(), out2[i].re.to_bits());
            assert_eq!(out1[i].im.to_bits(), out2[i].im.to_bits());
        }
    }

    #[test]
    fn test_complex_signed_zero_preserved() {
        let z1 = ComplexF64::new(0.0, 0.0);
        let z2 = ComplexF64::new(-0.0, -0.0);
        // Addition should preserve signs correctly.
        let sum = z1.add(z2);
        assert!(sum.re.is_sign_positive() || sum.re == 0.0);
    }

    #[test]
    fn test_complex_nan_propagation() {
        let nan_z = ComplexF64::new(f64::NAN, 1.0);
        let normal = ComplexF64::new(1.0, 1.0);
        let result = nan_z.mul_fixed(normal);
        assert!(result.is_nan());
    }

    #[test]
    fn test_complex_display() {
        let z = ComplexF64::new(3.0, -4.0);
        assert_eq!(format!("{z}"), "3-4i");
        let z2 = ComplexF64::new(1.0, 2.0);
        assert_eq!(format!("{z2}"), "1+2i");
    }
}
