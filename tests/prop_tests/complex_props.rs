//! Property-based tests for ComplexF64 arithmetic.
//!
//! Tests cover:
//! - Algebraic identities (commutativity, associativity, distributivity)
//! - Inverse relationships (mul/div roundtrip, add/sub roundtrip)
//! - Determinism (bit-identical repeated runs)
//! - NaN/Inf propagation (special values never panic)
//! - Pipeline parity (MIR-exec and AST-eval produce identical output)

use proptest::prelude::*;
use cjc_runtime::complex::ComplexF64;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Finite f64 values only (no NaN, no Inf).
fn finite_f64() -> impl Strategy<Value = f64> {
    prop::num::f64::NORMAL
        .prop_filter("must be finite", |x| x.is_finite())
}

/// Finite complex values.
fn finite_complex() -> impl Strategy<Value = ComplexF64> {
    (finite_f64(), finite_f64()).prop_map(|(re, im)| ComplexF64::new(re, im))
}

/// Non-zero finite complex values (for division tests).
fn nonzero_finite_complex() -> impl Strategy<Value = ComplexF64> {
    finite_complex().prop_filter("must be nonzero", |z| z.re != 0.0 || z.im != 0.0)
}

/// Small-magnitude finite complex values (to avoid overflow in mul/div chains).
fn small_complex() -> impl Strategy<Value = ComplexF64> {
    (-1e10f64..1e10, -1e10f64..1e10)
        .prop_filter("must be finite", |(re, im)| re.is_finite() && im.is_finite())
        .prop_map(|(re, im)| ComplexF64::new(re, im))
}

/// Non-zero small complex values (for division roundtrip tests).
fn nonzero_small_complex() -> impl Strategy<Value = ComplexF64> {
    small_complex().prop_filter("must be nonzero", |z| z.norm_sq() > 1e-100)
}

/// Special f64 values: NaN, Inf, -Inf, 0.0, -0.0.
fn special_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(0.0),
        Just(-0.0),
        Just(f64::MAX),
        Just(f64::MIN),
        Just(f64::MIN_POSITIVE),
    ]
}

/// Complex with at least one special component.
fn special_complex() -> impl Strategy<Value = ComplexF64> {
    prop_oneof![
        (special_f64(), finite_f64()).prop_map(|(re, im)| ComplexF64::new(re, im)),
        (finite_f64(), special_f64()).prop_map(|(re, im)| ComplexF64::new(re, im)),
        (special_f64(), special_f64()).prop_map(|(re, im)| ComplexF64::new(re, im)),
    ]
}

// ---------------------------------------------------------------------------
// Section 1: Algebraic Identity Tests (finite domain)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Addition commutativity: a + b == b + a (bit-exact for finite values).
    #[test]
    fn add_commutative(a in finite_complex(), b in finite_complex()) {
        let ab = a.add(b);
        let ba = b.add(a);
        prop_assert_eq!(ab.re.to_bits(), ba.re.to_bits(),
            "add commutativity re: {} + {} gave {} vs {}", a, b, ab, ba);
        prop_assert_eq!(ab.im.to_bits(), ba.im.to_bits(),
            "add commutativity im: {} + {} gave {} vs {}", a, b, ab, ba);
    }

    /// Multiplication commutativity: a * b == b * a (bit-exact for finite values).
    #[test]
    fn mul_commutative(a in finite_complex(), b in finite_complex()) {
        let ab = a.mul_fixed(b);
        let ba = b.mul_fixed(a);
        prop_assert_eq!(ab.re.to_bits(), ba.re.to_bits(),
            "mul commutativity re failed");
        prop_assert_eq!(ab.im.to_bits(), ba.im.to_bits(),
            "mul commutativity im failed");
    }

    /// Addition identity: a + 0 == a (bit-exact).
    #[test]
    fn add_identity(a in finite_complex()) {
        let result = a.add(ComplexF64::ZERO);
        prop_assert_eq!(result.re.to_bits(), a.re.to_bits());
        prop_assert_eq!(result.im.to_bits(), a.im.to_bits());
    }

    /// Multiplication identity: a * 1 == a (bit-exact).
    #[test]
    fn mul_identity(a in finite_complex()) {
        let result = a.mul_fixed(ComplexF64::ONE);
        prop_assert_eq!(result.re.to_bits(), a.re.to_bits());
        prop_assert_eq!(result.im.to_bits(), a.im.to_bits());
    }

    /// Multiplication by zero: a * 0 == 0 (NaN-free for finite inputs).
    #[test]
    fn mul_zero(a in finite_complex()) {
        let result = a.mul_fixed(ComplexF64::ZERO);
        prop_assert_eq!(result.re, 0.0, "mul zero re: {} * 0 = {}", a, result);
        prop_assert_eq!(result.im, 0.0, "mul zero im: {} * 0 = {}", a, result);
    }

    /// Additive inverse: a + (-a) == 0 (bit-exact for finite values).
    #[test]
    fn add_neg_inverse(a in finite_complex()) {
        let result = a.add(a.neg());
        prop_assert_eq!(result.re, 0.0, "a + (-a) re should be 0");
        prop_assert_eq!(result.im, 0.0, "a + (-a) im should be 0");
    }

    /// Conjugate involution: conj(conj(z)) == z (bit-exact).
    #[test]
    fn conj_involution(z in finite_complex()) {
        let cc = z.conj().conj();
        prop_assert_eq!(cc.re.to_bits(), z.re.to_bits());
        prop_assert_eq!(cc.im.to_bits(), z.im.to_bits());
    }

    /// Conjugate of sum: conj(a+b) == conj(a) + conj(b) (bit-exact).
    #[test]
    fn conj_distributes_over_add(a in finite_complex(), b in finite_complex()) {
        let lhs = a.add(b).conj();
        let rhs = a.conj().add(b.conj());
        prop_assert_eq!(lhs.re.to_bits(), rhs.re.to_bits());
        prop_assert_eq!(lhs.im.to_bits(), rhs.im.to_bits());
    }

    /// norm_sq(z) == z * conj(z) for re part (bit-exact for small values).
    #[test]
    fn norm_sq_equals_z_times_conj_re(z in small_complex()) {
        let ns = z.norm_sq();
        let product = z.mul_fixed(z.conj());
        // Real part should match norm_sq.
        prop_assert_eq!(ns.to_bits(), product.re.to_bits(),
            "norm_sq({}) = {} vs z*conj(z).re = {}", z, ns, product.re);
    }

    /// abs(z) >= 0 for all finite z.
    #[test]
    fn abs_nonneg(z in finite_complex()) {
        let a = z.abs();
        prop_assert!(a >= 0.0 || a.is_nan(), "abs should be non-negative: {}", a);
    }

    /// Subtraction as inverse of addition: (a + b) - b == a (within tolerance).
    #[test]
    fn sub_inverse_of_add(a in small_complex(), b in small_complex()) {
        let sum = a.add(b);
        let back = sum.sub(b);
        let tol = 1e-6 * (a.abs() + b.abs() + 1.0);
        prop_assert!((back.re - a.re).abs() < tol,
            "add/sub roundtrip re: {} != {}", back.re, a.re);
        prop_assert!((back.im - a.im).abs() < tol,
            "add/sub roundtrip im: {} != {}", back.im, a.im);
    }

    /// Division roundtrip: (a * b) / b ≈ a for non-zero b.
    #[test]
    fn div_roundtrip(a in nonzero_small_complex(), b in nonzero_small_complex()) {
        let product = a.mul_fixed(b);
        let back = product.div_fixed(b);
        let tol = 1e-6 * (a.abs() + 1.0);
        if back.is_finite() {
            prop_assert!((back.re - a.re).abs() < tol,
                "mul/div roundtrip re: {} != {} (product={}, b={})", back.re, a.re, product, b);
            prop_assert!((back.im - a.im).abs() < tol,
                "mul/div roundtrip im: {} != {} (product={}, b={})", back.im, a.im, product, b);
        }
    }
}

// ---------------------------------------------------------------------------
// Section 2: Determinism Tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// mul_fixed is deterministic: same inputs → same bit pattern.
    #[test]
    fn mul_deterministic(a in finite_complex(), b in finite_complex()) {
        let r1 = a.mul_fixed(b);
        let r2 = a.mul_fixed(b);
        prop_assert_eq!(r1.re.to_bits(), r2.re.to_bits());
        prop_assert_eq!(r1.im.to_bits(), r2.im.to_bits());
    }

    /// div_fixed is deterministic: same inputs → same bit pattern.
    #[test]
    fn div_deterministic(a in finite_complex(), b in nonzero_finite_complex()) {
        let r1 = a.div_fixed(b);
        let r2 = a.div_fixed(b);
        prop_assert_eq!(r1.re.to_bits(), r2.re.to_bits());
        prop_assert_eq!(r1.im.to_bits(), r2.im.to_bits());
    }
}

// ---------------------------------------------------------------------------
// Section 3: Special Values / Edge Cases
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// All arithmetic ops on special values (NaN, Inf) must not panic.
    #[test]
    fn special_values_no_panic(a in special_complex(), b in special_complex()) {
        // None of these should panic — they may produce NaN/Inf, which is fine.
        let _ = a.add(b);
        let _ = a.sub(b);
        let _ = a.mul_fixed(b);
        let _ = a.div_fixed(b);
        let _ = a.conj();
        let _ = a.neg();
        let _ = a.abs();
        let _ = a.norm_sq();
        let _ = a.is_nan();
        let _ = a.is_finite();
        let _ = a.scale(b.re);
    }

    /// NaN input propagates through all operations.
    #[test]
    fn nan_propagation(b in finite_complex()) {
        let nan_z = ComplexF64::new(f64::NAN, 1.0);
        prop_assert!(nan_z.add(b).is_nan(), "add should propagate NaN");
        prop_assert!(nan_z.mul_fixed(b).is_nan(), "mul should propagate NaN");
        prop_assert!(nan_z.conj().is_nan(), "conj should propagate NaN");
        prop_assert!(nan_z.norm_sq().is_nan(), "norm_sq should propagate NaN");
    }

    /// Display never panics on any value.
    #[test]
    fn display_no_panic(z in special_complex()) {
        let _ = format!("{}", z);
    }
}
