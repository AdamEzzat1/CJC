//! Dump the `cjc_repro::dmath` functions added for the runtime migration
//! (tan, asin, acos, atan, atan2, sinh, cosh, tanh, atanh, exp_m1, ln_1p,
//! log2, log10, pow, hypot) as hex, for accuracy checking against mpmath
//! (`dmath_ext_check.py`) and cross-platform hash comparison.
//!
//!     cargo run --release > dmath_ext_out.txt
//!
//! One-argument lines: `fn x_bits y_bits`. Two-argument lines (atan2, pow,
//! hypot): `fn a_bits b_bits y_bits`, in the function's own argument order.
//! Inputs concentrate on each algorithm's branch thresholds, where
//! transcription errors hide.
use cjc_repro::dmath;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn unit(s: &mut u64) -> f64 {
    (splitmix(s) >> 11) as f64 / (1u64 << 53) as f64
}

/// Finite double with a uniformly random exponent (both signs).
fn any_finite(s: &mut u64) -> f64 {
    let r = splitmix(s);
    f64::from_bits((r & !(0x7ffu64 << 52)) | ((splitmix(s) % 0x7ff) << 52))
}

/// `n` points uniform in [lo, hi].
fn uniform(s: &mut u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n).map(|_| lo + (hi - lo) * unit(s)).collect()
}

/// Points within a few ulps (and a small relative window) of each threshold.
fn around(s: &mut u64, points: &[f64]) -> Vec<f64> {
    let mut v = Vec::new();
    for &p in points {
        let b = p.to_bits() as i64;
        for d in -8i64..=8 {
            v.push(f64::from_bits((b + d) as u64));
        }
        v.extend(uniform(s, 200, p * (1.0 - 1e-3), p * (1.0 + 1e-3)));
    }
    v
}

fn one(name: &str, xs: &[f64], f: fn(f64) -> f64) {
    for &x in xs {
        println!("{name} {:016x} {:016x}", x.to_bits(), f(x).to_bits());
    }
}

fn two(name: &str, pairs: &[(f64, f64)], f: fn(f64, f64) -> f64) {
    for &(a, b) in pairs {
        println!("{name} {:016x} {:016x} {:016x}", a.to_bits(), b.to_bits(), f(a, b).to_bits());
    }
}

fn main() {
    let s = &mut 0x0e47_d0e5_u64;
    let pi = std::f64::consts::PI;

    // tan: gate range, kernel range, big-branch threshold, near the poles, every exponent.
    let mut xs = uniform(s, 20_000, -8.0 * pi, 8.0 * pi);
    xs.extend(uniform(s, 10_000, -pi / 4.0, pi / 4.0));
    xs.extend(around(s, &[0.6744, pi / 4.0, pi / 2.0, 3.0 * pi / 2.0, 1e6]));
    xs.extend((0..20_000).map(|_| any_finite(s)));
    one("tan", &xs, dmath::tan);

    // asin / acos: all of [-1, 1], near ±1, at the 0.5 and 0.975 switches, tiny.
    let mut xs = uniform(s, 20_000, -1.0, 1.0);
    xs.extend((1..=60).map(|k| 1.0 - 2f64.powi(-k)));
    xs.extend((1..=60).map(|k| -1.0 + 2f64.powi(-k)));
    xs.extend(around(s, &[0.5, -0.5, 0.975, -0.975]));
    xs.extend((0..2_000).map(|_| any_finite(s) * 1e-300).filter(|x| x.abs() < 1.0));
    one("asin", &xs, dmath::asin);
    one("acos", &xs, dmath::acos);
    one("atanh", &xs.iter().copied().filter(|x| x.abs() < 1.0).collect::<Vec<_>>(), dmath::atanh);

    // atan: every exponent plus the reduction breakpoints.
    let mut xs: Vec<f64> = (0..30_000).map(|_| any_finite(s)).collect();
    xs.extend(uniform(s, 10_000, -4.0, 4.0));
    xs.extend(around(s, &[0.4375, 0.6875, 1.1875, 2.4375, -1.1875]));
    one("atan", &xs, dmath::atan);

    // atan2: random pairs over exponents and signs, plus axes-adjacent pairs.
    let mut pairs: Vec<(f64, f64)> = (0..30_000).map(|_| (any_finite(s), any_finite(s))).collect();
    pairs.extend((0..10_000).map(|_| (unit(s) * 4.0 - 2.0, unit(s) * 4.0 - 2.0)));
    two("atan2", &pairs, dmath::atan2);

    // sinh / cosh / tanh / exp_m1: small, medium, thresholds, expo2 window.
    let mut xs = uniform(s, 20_000, -1.0, 1.0);
    xs.extend(uniform(s, 20_000, -30.0, 30.0));
    xs.extend(uniform(s, 5_000, -720.0, 720.0));
    xs.extend(around(s, &[0.2554, 0.5493, 0.25, 20.0, 709.78, 710.4, -0.2554, -20.0]));
    xs.extend((0..2_000).map(|_| any_finite(s) * 1e-300));
    one("sinh", &xs, dmath::sinh);
    one("cosh", &xs, dmath::cosh);
    one("tanh", &xs, dmath::tanh);
    let mut em = xs.clone();
    em.extend(uniform(s, 5_000, -60.0, 709.0));
    em.extend(around(s, &[0.3466, 1.0397, 38.8]));
    one("exp_m1", &em, dmath::exp_m1);

    // ln_1p: (-1, inf) over exponents, near 0, near -1, the √2 switches.
    let mut xs: Vec<f64> = (0..20_000).map(|_| any_finite(s).abs()).collect();
    xs.extend(uniform(s, 20_000, -0.999, 2.0));
    xs.extend((1..=50).map(|k| -1.0 + 2f64.powi(-k)));
    xs.extend(around(s, &[0.4142135623730950, -0.2928932188134524, 1e-8, -1e-8]));
    one("ln_1p", &xs, dmath::ln_1p);

    // log2 / log10: every exponent incl. subnormals, and near 1.
    let mut xs: Vec<f64> = (0..30_000)
        .map(|_| f64::from_bits((splitmix(s) & !(1u64 << 63)) % 0x7ff0_0000_0000_0000))
        .collect();
    xs.extend(uniform(s, 10_000, 0.5, 2.0));
    xs.extend(around(s, &[1.0, std::f64::consts::SQRT_2, std::f64::consts::FRAC_1_SQRT_2]));
    one("log2", &xs, dmath::log2);
    one("log10", &xs, dmath::log10);

    // pow: positive bases over wide ranges with results in range, integer
    // exponents of negative bases, near 1 with huge y, subnormal results.
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for _ in 0..20_000 {
        let x = f64::from_bits((splitmix(s) & !(1u64 << 63)) % 0x7ff0_0000_0000_0000);
        let lx = x.ln().abs().max(1e-3);
        let y = (unit(s) * 2.0 - 1.0) * (700.0 / lx); // keeps |y·ln x| < 700
        pairs.push((x, y));
    }
    for _ in 0..10_000 {
        pairs.push((unit(s) * 20.0, unit(s) * 40.0 - 20.0));
    }
    for _ in 0..5_000 {
        let n = (splitmix(s) % 61) as f64 - 30.0;
        pairs.push((-(unit(s) * 10.0 + 0.1), n)); // negative base, integer y
    }
    for _ in 0..5_000 {
        pairs.push((1.0 + (unit(s) - 0.5) * 1e-7, (unit(s) - 0.5) * 1e10)); // |1-x| tiny, |y| > 2^31
    }
    for _ in 0..2_000 {
        pairs.push((unit(s) * 0.5 + 0.25, 1000.0 + unit(s) * 60.0)); // subnormal results
    }
    two("pow", &pairs, dmath::pow);

    // hypot: exponent pairs incl. extreme ratios and scaling boundaries.
    let mut pairs: Vec<(f64, f64)> = (0..30_000).map(|_| (any_finite(s), any_finite(s))).collect();
    pairs.extend((0..10_000).map(|_| (unit(s) * 10.0, unit(s) * 10.0)));
    pairs.extend((0..2_000).map(|_| (1e300 * (1.0 + unit(s)), 1e300 * unit(s))));
    pairs.extend((0..2_000).map(|_| (1e-300 * (1.0 + unit(s)), 1e-300 * unit(s))));
    two("hypot", &pairs, dmath::hypot);
}
