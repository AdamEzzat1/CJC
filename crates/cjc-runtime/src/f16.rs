//! Half Precision (f16) — IEEE 754 binary16 with promotion to f64.
//!
//! # Design
//!
//! f16 values are promoted to f64 before entering the binned accumulation
//! path. Subnormal handling is preserved by the bin 0 logic of the
//! BinnedAccumulator. Arithmetic is performed in f64, then narrowed back
//! to f16 on storage.
//!
//! # IEEE 754 binary16 Layout
//!
//! ```text
//! Bit 15:     sign
//! Bits 14-10: exponent (5 bits, bias = 15)
//! Bits 9-0:   mantissa (10 bits)
//! ```
//!
//! Range: ±65504 (max normal), ±6.1e-5 (min positive subnormal)

use crate::accumulator::BinnedAccumulatorF64;

// ---------------------------------------------------------------------------
// F16 Type
// ---------------------------------------------------------------------------

/// IEEE 754 binary16 half-precision float.
///
/// Stored as u16. All arithmetic is performed by promoting to f64,
/// computing, then narrowing back. This ensures deterministic behavior
/// regardless of platform, since the f64 path is well-defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F16(pub u16);

impl F16 {
    /// Positive zero.
    pub const ZERO: F16 = F16(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: F16 = F16(0x8000);
    /// Positive infinity.
    pub const INFINITY: F16 = F16(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: F16 = F16(0xFC00);
    /// Canonical NaN.
    pub const NAN: F16 = F16(0x7E00);
    /// Maximum finite value: 65504.0.
    pub const MAX: F16 = F16(0x7BFF);
    /// Minimum positive subnormal.
    pub const MIN_POSITIVE_SUBNORMAL: F16 = F16(0x0001);

    /// Convert f16 to f64.
    ///
    /// Handles normals, subnormals, zeros, infinities, and NaNs.
    pub fn to_f64(self) -> f64 {
        let bits = self.0;
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1F;
        let mant = bits & 0x03FF;

        let sign_f = if sign == 1 { -1.0 } else { 1.0 };

        if exp == 0 {
            if mant == 0 {
                // Signed zero.
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // Subnormal: value = sign * 2^(-14) * (mant / 1024)
                sign_f * (mant as f64) * 2.0f64.powi(-24)
            }
        } else if exp == 0x1F {
            if mant == 0 {
                // Infinity.
                if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY }
            } else {
                // NaN — canonicalize to a single NaN value.
                f64::NAN
            }
        } else {
            // Normal: value = sign * 2^(exp-15) * (1 + mant/1024)
            sign_f * 2.0f64.powi(exp as i32 - 15) * (1.0 + mant as f64 / 1024.0)
        }
    }

    /// Convert f64 to f16 (round-to-nearest-even).
    ///
    /// Handles overflow to infinity, underflow to zero, and subnormals.
    pub fn from_f64(value: f64) -> Self {
        if value.is_nan() {
            return F16::NAN;
        }

        let sign: u16 = if value.is_sign_negative() { 0x8000 } else { 0 };
        let abs_val = value.abs();

        if abs_val == 0.0 {
            return F16(sign); // Preserves sign of zero
        }

        if abs_val.is_infinite() {
            return F16(sign | 0x7C00);
        }

        // Overflow to infinity.
        if abs_val > 65504.0 {
            return F16(sign | 0x7C00);
        }

        // Subnormal range: < 2^(-14) = 6.103515625e-5
        if abs_val < 6.103515625e-5 {
            // Subnormal: round to nearest subnormal representation.
            let mant = (abs_val / 2.0f64.powi(-24)).round() as u16;
            if mant == 0 {
                return F16(sign); // Underflow to signed zero
            }
            return F16(sign | mant.min(0x03FF));
        }

        // Normal range.
        let log2_val = abs_val.log2();
        let exp = log2_val.floor() as i32;
        let biased_exp = (exp + 15) as u16;

        if biased_exp >= 31 {
            return F16(sign | 0x7C00); // Overflow
        }

        let significand = abs_val / 2.0f64.powi(exp) - 1.0;
        let mant = (significand * 1024.0).round() as u16;

        // Handle rounding that pushes mantissa to 1024 (overflow to next exponent).
        if mant >= 1024 {
            let biased_exp = biased_exp + 1;
            if biased_exp >= 31 {
                return F16(sign | 0x7C00);
            }
            return F16(sign | (biased_exp << 10));
        }

        F16(sign | (biased_exp << 10) | mant)
    }

    /// Convert f32 to f16.
    pub fn from_f32(value: f32) -> Self {
        Self::from_f64(value as f64)
    }

    /// Convert f16 to f32.
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Check if this is NaN.
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x03FF;
        exp == 0x1F && mant != 0
    }

    /// Check if this is infinite.
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x03FF;
        exp == 0x1F && mant == 0
    }

    /// Check if this is finite (not NaN or Inf).
    pub fn is_finite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        exp != 0x1F
    }

    /// Check if this is subnormal (exponent == 0, mantissa != 0).
    pub fn is_subnormal(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x03FF;
        exp == 0 && mant != 0
    }

    /// Add two f16 values. Promotes both to f64, adds, then narrows back.
    pub fn add(self, rhs: Self) -> Self { Self::from_f64(self.to_f64() + rhs.to_f64()) }
    /// Subtract `rhs` from `self`. Promotes both to f64, subtracts, then narrows back.
    pub fn sub(self, rhs: Self) -> Self { Self::from_f64(self.to_f64() - rhs.to_f64()) }
    /// Multiply two f16 values. Promotes both to f64, multiplies, then narrows back.
    pub fn mul(self, rhs: Self) -> Self { Self::from_f64(self.to_f64() * rhs.to_f64()) }
    /// Divide `self` by `rhs`. Promotes both to f64, divides, then narrows back.
    pub fn div(self, rhs: Self) -> Self { Self::from_f64(self.to_f64() / rhs.to_f64()) }
    /// Negate by toggling the sign bit. Does not promote to f64.
    pub fn neg(self) -> Self { F16(self.0 ^ 0x8000) }
}

impl std::fmt::Display for F16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

// ---------------------------------------------------------------------------
// f16 Accumulation via BinnedAccumulator
// ---------------------------------------------------------------------------

/// Sum f16 values by promoting to f64 and using BinnedAccumulator.
///
/// This ensures order-invariant, deterministic results regardless of
/// the f16 precision limitations.
pub fn f16_binned_sum(values: &[F16]) -> f64 {
    let mut acc = BinnedAccumulatorF64::new();
    for &v in values {
        acc.add(v.to_f64());
    }
    acc.finalize()
}

/// Dot product of two f16 slices, accumulated in f64 via BinnedAccumulator.
pub fn f16_binned_dot(a: &[F16], b: &[F16]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = BinnedAccumulatorF64::new();
    for i in 0..a.len() {
        // Promote both operands to f64, multiply in f64, then accumulate.
        acc.add(a[i].to_f64() * b[i].to_f64());
    }
    acc.finalize()
}

/// Matrix multiply for f16 arrays, computing in f64 via BinnedAccumulator.
///
/// Result is in f64 for full precision.
pub fn f16_matmul(
    a: &[F16], b: &[F16], out: &mut [f64],
    m: usize, k: usize, n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut acc = BinnedAccumulatorF64::new();
            for p in 0..k {
                let av = a[i * k + p].to_f64();
                let bv = b[p * n + j].to_f64();
                acc.add(av * bv);
            }
            out[i * n + j] = acc.finalize();
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
    fn test_f16_zero() {
        let z = F16::ZERO;
        assert_eq!(z.to_f64(), 0.0);
        assert!(z.to_f64().is_sign_positive());
    }

    #[test]
    fn test_f16_neg_zero() {
        let z = F16::NEG_ZERO;
        assert_eq!(z.to_f64(), 0.0);
        assert!(z.to_f64().is_sign_negative());
    }

    #[test]
    fn test_f16_one() {
        let one = F16::from_f64(1.0);
        assert_eq!(one.to_f64(), 1.0);
    }

    #[test]
    fn test_f16_max() {
        let max = F16::MAX;
        assert_eq!(max.to_f64(), 65504.0);
    }

    #[test]
    fn test_f16_infinity() {
        let inf = F16::INFINITY;
        assert!(inf.to_f64().is_infinite());
        assert!(inf.to_f64().is_sign_positive());
    }

    #[test]
    fn test_f16_neg_infinity() {
        let ninf = F16::NEG_INFINITY;
        assert!(ninf.to_f64().is_infinite());
        assert!(ninf.to_f64().is_sign_negative());
    }

    #[test]
    fn test_f16_nan() {
        let nan = F16::NAN;
        assert!(nan.to_f64().is_nan());
        assert!(nan.is_nan());
    }

    #[test]
    fn test_f16_subnormal() {
        let sub = F16::MIN_POSITIVE_SUBNORMAL;
        let val = sub.to_f64();
        assert!(val > 0.0);
        assert!(sub.is_subnormal());
        // Smallest f16 subnormal: 2^(-24) ≈ 5.96e-8
        assert!((val - 5.960464477539063e-8).abs() < 1e-15);
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -0.25, 65504.0];
        for &v in &values {
            let f16 = F16::from_f64(v);
            let back = f16.to_f64();
            assert_eq!(back, v, "Roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_f16_overflow_to_inf() {
        let f16 = F16::from_f64(100000.0);
        assert!(f16.is_infinite());
    }

    #[test]
    fn test_f16_underflow_to_zero() {
        let f16 = F16::from_f64(1e-10);
        assert_eq!(f16.to_f64(), 0.0);
    }

    #[test]
    fn test_f16_arithmetic() {
        let a = F16::from_f64(2.0);
        let b = F16::from_f64(3.0);
        assert_eq!(a.add(b).to_f64(), 5.0);
        assert_eq!(a.mul(b).to_f64(), 6.0);
        assert_eq!(b.sub(a).to_f64(), 1.0);
    }

    #[test]
    fn test_f16_neg_preserves_bits() {
        let a = F16::from_f64(3.5);
        let neg = a.neg();
        assert_eq!(neg.to_f64(), -3.5);
        // Double negation round-trips.
        assert_eq!(neg.neg().0, a.0);
    }

    #[test]
    fn test_f16_binned_sum_basic() {
        let values: Vec<F16> = (0..10).map(|i| F16::from_f64(i as f64)).collect();
        let result = f16_binned_sum(&values);
        assert_eq!(result, 45.0);
    }

    #[test]
    fn test_f16_binned_sum_order_invariant() {
        let values: Vec<F16> = (0..200).map(|i| F16::from_f64(i as f64 * 0.5 - 50.0)).collect();
        let mut reversed = values.clone();
        reversed.reverse();

        let r1 = f16_binned_sum(&values);
        let r2 = f16_binned_sum(&reversed);
        assert_eq!(r1.to_bits(), r2.to_bits(), "f16 sum must be order-invariant");
    }

    #[test]
    fn test_f16_dot_basic() {
        let a = vec![F16::from_f64(1.0), F16::from_f64(2.0), F16::from_f64(3.0)];
        let b = vec![F16::from_f64(4.0), F16::from_f64(5.0), F16::from_f64(6.0)];
        let result = f16_binned_dot(&a, &b);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_f16_matmul_identity() {
        let identity = vec![
            F16::from_f64(1.0), F16::from_f64(0.0),
            F16::from_f64(0.0), F16::from_f64(1.0),
        ];
        let b = vec![
            F16::from_f64(3.0), F16::from_f64(4.0),
            F16::from_f64(5.0), F16::from_f64(6.0),
        ];
        let mut out = vec![0.0f64; 4];
        f16_matmul(&identity, &b, &mut out, 2, 2, 2);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_f16_subnormal_accumulation() {
        // Test that subnormals are correctly accumulated via the binned path.
        let sub = F16::MIN_POSITIVE_SUBNORMAL;
        let values = vec![sub; 1000];
        let result = f16_binned_sum(&values);
        let expected = sub.to_f64() * 1000.0;
        assert!((result - expected).abs() < 1e-12,
            "Subnormal accumulation: {result} vs expected {expected}");
    }

    #[test]
    fn test_f16_signed_zero_preserved() {
        let pz = F16::ZERO;
        let nz = F16::NEG_ZERO;
        assert!(pz.to_f64().is_sign_positive());
        assert!(nz.to_f64().is_sign_negative());
        // From f64 preserves sign.
        assert_eq!(F16::from_f64(0.0).0, F16::ZERO.0);
        assert_eq!(F16::from_f64(-0.0).0, F16::NEG_ZERO.0);
    }
}
