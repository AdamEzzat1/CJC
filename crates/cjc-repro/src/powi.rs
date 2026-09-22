//! `powi_f64`: integer powers of a double by binary exponentiation, the algorithm of
//! compiler-builtins' `__powidf2`, transcribed so that CJC owns it.
//!
//! Why a crate-owned `powi` exists: Rust's `f64::powi` with a *runtime* exponent is
//! `__powidf2` on `x86_64-unknown-linux-gnu`, but on `x86_64-pc-windows-msvc` LLVM has no
//! such libcall and lowers it to the C runtime's `pow`, which rounds differently in about
//! half of all cases (2728 of 5000 probes over bases in `[0.5, 4)` and exponents
//! `-12..=12`, rustc 1.97.1). A constant exponent at `-O` is expanded inline instead, so on
//! that target the same source computed different bits in debug and release, and
//! different bits from Linux. This function computes the Linux bits everywhere, on every
//! platform and in every profile, and every CJC site with a runtime exponent calls it.
//!
//! The algorithm: `r = r * a` on each set bit of `|b|`, `a = a * a` between bits, `1 / r`
//! for a negative exponent. Serial, no FMA, total on its domain (`i32::MIN` included).

/// `a` to the integer power `b`, by binary exponentiation (`__powidf2`'s multiplications
/// in `__powidf2`'s order). Bit-identical on every platform; **not** `f64::powi` on
/// MSVC targets, which is the point.
#[inline]
pub fn powi_f64(mut a: f64, b: i32) -> f64 {
    let recip = b < 0;
    let mut e = b.unsigned_abs();
    let mut r = 1.0f64;
    loop {
        if e & 1 != 0 {
            r *= a;
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        a *= a;
    }
    if recip { 1.0 / r } else { r }
}

#[cfg(test)]
mod tests {
    use super::powi_f64;

    /// The spec, not a tautology: hard-coded bits computed by the same multiplications in
    /// the same order outside Rust (a Python double is an IEEE double), including the
    /// `(3.8511975033176533, -11)` witness where MSVC's `f64::powi` gives
    /// `4510433485740570284` and `__powidf2` gives `4510433485740570282`.
    const SPEC: &[(f64, i32, u64)] = &[
        (1.1, 5, 0x3ff9c4a6223e186c),
        (0.7, 3, 0x3fd5f3b645a1cabf),
        (2.0, 62, 0x43d0000000000000),
        (2.0, -3, 0x3fc0000000000000),
        (f64::NAN, 0, 0x3ff0000000000000),
        (0.0, 0, 0x3ff0000000000000),
        (-0.0, 3, 0x8000000000000000),
        (f64::INFINITY, -1, 0x0000000000000000),
        (3.8511975033176533, -11, 0x3e98475ba497f6aa),
        (10.0, -7, 0x3e7ad7f29abcaf48),
        (0.5, 13, 0x3f20000000000000),
        (-1.5, 9, 0xc04338c000000000),
    ];

    #[test]
    fn powi_f64_computes_the_powidf2_bits_everywhere() {
        for &(a, b, want) in SPEC {
            assert_eq!(powi_f64(a, b).to_bits(), want, "a = {a:e}, b = {b}");
        }
    }

    /// The witness, asserted unconditionally: on MSVC this is exactly the value
    /// `f64::powi` does not compute.
    #[test]
    fn the_msvc_witness_takes_the_linux_bits() {
        assert_eq!(powi_f64(3.8511975033176533, -11).to_bits(), 4510433485740570282);
        assert_ne!(powi_f64(3.8511975033176533, -11).to_bits(), 4510433485740570284);
    }

    #[test]
    fn exponent_zero_is_one_for_every_base_and_extremes_do_not_panic() {
        for &a in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, 1e308, 5e-324] {
            assert_eq!(powi_f64(a, 0).to_bits(), 1.0f64.to_bits(), "a = {a:e}");
        }
        assert_eq!(powi_f64(2.0, -3).to_bits(), 0.125f64.to_bits());
        assert_eq!(powi_f64(-0.0, 3).to_bits(), (-0.0f64).to_bits(), "an odd power keeps the sign of zero");
        // `i32::MIN` has no positive counterpart; `unsigned_abs` makes it 2^31 and the loop
        // runs 32 doublings: the answer for |a| > 1 is `1 / inf = +0`, for |a| < 1 is `+inf`.
        assert_eq!(powi_f64(2.0, i32::MIN), 0.0);
        assert_eq!(powi_f64(0.5, i32::MIN), f64::INFINITY);
        assert_eq!(powi_f64(1.0, i32::MIN), 1.0);
    }

    /// Where the exponent is small and the base is an exact power of two, `f64::powi` and
    /// `powi_f64` agree on every target (both answers are exactly representable); this
    /// pins that the crate-owned function is a superset, not a different function.
    #[test]
    fn agrees_with_f64_powi_on_exactly_representable_cases() {
        for e in -60..=60 {
            let e = std::hint::black_box(e);
            assert_eq!(powi_f64(2.0, e).to_bits(), 2.0f64.powi(e).to_bits(), "e = {e}");
            assert_eq!(powi_f64(0.5, e).to_bits(), 0.5f64.powi(e).to_bits(), "e = {e}");
        }
    }
}
