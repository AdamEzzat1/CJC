//! Deterministic elementary functions: [`sin`], [`cos`], [`sin_cos`], [`exp`],
//! [`ln`], [`pow`], and [`powi`].
//!
//! `f64::sin` and friends call the platform C library, and different
//! libraries return different last bits: Windows (UCRT) and Linux (glibc)
//! disagree on 6.0% of rotation-gate angles, including `sin(π/2 · k)` for
//! some `k` (see `docs/quantum_simulation_research_stack/verification/`).
//! That breaks CJC-Lang's "same seed ⇒ same bits" guarantee across operating
//! systems.
//!
//! The functions here use only IEEE-754 `+ - * /` (correctly rounded on every
//! conforming platform), integer bit manipulation, and no fused multiply-add
//! (Rust never contracts `a * b + c` implicitly). The result is a function of
//! the input bits alone, identical on every OS, CPU, and compiler version.
//!
//! # Accuracy
//!
//! [`sin`], [`cos`], [`exp`], and [`ln`] follow fdlibm and are accurate to
//! < 1 ulp for all finite inputs (verified against mpmath at 200 bits; see
//! `verification/dmath_check/`). [`pow`] is `exp(y · ln x)` and loses about
//! `|y · ln x|` ulps; use it only where that is documented as acceptable.
//! [`powi`] is exact repeated squaring (same result as a multiply chain).
//!
//! # Provenance
//!
//! Polynomial kernels, the Cody–Waite reduction, `exp`, and `ln` are
//! transcribed from fdlibm as ported by musl:
//!
//! > Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! > Developed at SunPro, a Sun Microsystems, Inc. business.
//! > Permission to use, copy, modify, and distribute this software is freely
//! > granted, provided that this notice is preserved.
//!
//! The large-argument (Payne–Hanek) reduction is new code using `u128`
//! arithmetic; its 2/π table was generated with mpmath.

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------

#[inline]
fn high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

/// `x * 2^n`, exact unless the result over- or underflows (musl `scalbn`).
fn scalbn(x: f64, mut n: i32) -> f64 {
    let mut y = x;
    if n > 1023 {
        y *= f64::from_bits(0x7FE0_0000_0000_0000); // 2^1023
        n -= 1023;
        if n > 1023 {
            y *= f64::from_bits(0x7FE0_0000_0000_0000);
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // 2^-1022 * 2^53: keep the final n below -53 so the subnormal
        // result is rounded once, not twice.
        let down = f64::from_bits(0x0010_0000_0000_0000) * f64::from_bits(0x4340_0000_0000_0000);
        y *= down;
        n += 1022 - 53;
        if n < -1022 {
            y *= down;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

// ---------------------------------------------------------------------------
// sin / cos kernels on [-π/4, π/4]  (fdlibm k_sin.c, k_cos.c)
// ---------------------------------------------------------------------------

const S1: f64 = -1.66666666666666324348e-01;
const S2: f64 = 8.33333333332248946124e-03;
const S3: f64 = -1.98412698298579493134e-04;
const S4: f64 = 2.75573137070700676789e-06;
const S5: f64 = -2.50507602534068634195e-08;
const S6: f64 = 1.58969099521155010221e-10;

/// sin(x + y) for |x| ≤ ~π/4, where y is the tail of x. `has_tail` false
/// means y is known to be zero.
fn k_sin(x: f64, y: f64, has_tail: bool) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    if !has_tail {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

const C1: f64 = 4.16666666666666019037e-02;
const C2: f64 = -1.38888888888741095749e-03;
const C3: f64 = 2.48015872894767294178e-05;
const C4: f64 = -2.75573143513906633035e-07;
const C5: f64 = 2.08757232129817482790e-09;
const C6: f64 = -1.13596475577881948265e-11;

/// cos(x + y) for |x| ≤ ~π/4, where y is the tail of x.
fn k_cos(x: f64, y: f64) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

// ---------------------------------------------------------------------------
// Argument reduction: x = n·(π/2) + (y0 + y1),  |y0 + y1| ≤ ~π/4
// ---------------------------------------------------------------------------

const TOINT: f64 = 1.5 / f64::EPSILON;
const PIO4: f64 = 7.85398163397448278999e-01; // 0x3FE921FB54442D18
const INVPIO2: f64 = 6.36619772367581382433e-01;
const PIO2_1: f64 = 1.57079632673412561417e+00; // first 33 bits of π/2
const PIO2_1T: f64 = 6.07710050650619224932e-11; // π/2 - PIO2_1
const PIO2_2: f64 = 6.07710050630396597660e-11; // second 33 bits
const PIO2_2T: f64 = 2.02226624879595063154e-21;
const PIO2_3: f64 = 2.02226624871116645580e-21; // third 33 bits
const PIO2_3T: f64 = 8.47842766036889956997e-32;

/// Returns (n, y0, y1). Only `n & 3` is meaningful for large |x|.
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    let ix = high_word(x) & 0x7fff_ffff;
    if ix < 0x4139_21fb {
        return rem_pio2_medium(x, ix);
    }
    // Finite, |x| ≥ 2^20·π/2 (callers have already filtered inf/NaN).
    let (n, y0, y1) = rem_pio2_large(x.abs());
    if x < 0.0 {
        (-n, -y0, -y1)
    } else {
        (n, y0, y1)
    }
}

/// Cody–Waite reduction for |x| < 2^20·π/2 (musl `__rem_pio2` medium case,
/// used here for every |x| in that range).
fn rem_pio2_medium(x: f64, ix: u32) -> (i32, f64, f64) {
    // rint(x / (π/2)) via the 1.5·2^52 trick; each op is a separate rounding.
    let mut f_n = x * INVPIO2 + TOINT - TOINT;
    let mut n = f_n as i32;
    let mut r = x - f_n * PIO2_1; // exact: PIO2_1 has 33 bits, |n| < 2^20
    let mut w = f_n * PIO2_1T;
    if r - w < -PIO4 {
        n -= 1;
        f_n -= 1.0;
        r = x - f_n * PIO2_1;
        w = f_n * PIO2_1T;
    } else if r - w > PIO4 {
        n += 1;
        f_n += 1.0;
        r = x - f_n * PIO2_1;
        w = f_n * PIO2_1T;
    }
    let mut y0 = r - w;
    let ex = (ix >> 20) as i32;
    let ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
    if ex - ey > 16 {
        // Cancellation: second round, good to 118 bits.
        let t = r;
        w = f_n * PIO2_2;
        r = t - w;
        w = f_n * PIO2_2T - ((t - r) - w);
        y0 = r - w;
        let ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
        if ex - ey > 49 {
            // Third round, good to 151 bits: covers every double.
            let t = r;
            w = f_n * PIO2_3;
            r = t - w;
            w = f_n * PIO2_3T - ((t - r) - w);
            y0 = r - w;
        }
    }
    let y1 = (r - y0) - w;
    (n, y0, y1)
}

/// Bits of 2/π after the binary point, most significant first (1280 bits,
/// enough for any finite double: the largest exponent needs bit ~1161).
/// Generated with mpmath: `floor(2/π · 2^1280)`.
const TWO_OVER_PI: [u64; 20] = [
    0xA2F9836E4E441529, 0xFC2757D1F534DDC0, 0xDB6295993C439041, 0xFE5163ABDEBBC561,
    0xB7246E3A424DD2E0, 0x06492EEA09D1921C, 0xFE1DEB1CB129A73E, 0xE88235F52EBB4484,
    0xE99C7026B45F7E41, 0x3991D639835339F4, 0x9C845F8BBDF9283B, 0x1FF897FFDE05980F,
    0xEF2F118B5A0A6D1F, 0x6D367ECF27CB09B7, 0x4F463F669E5FEA2D, 0x7527BAC7EBE5F17B,
    0x3D0739F78A5292EA, 0x6BFB5FB11F8D5D08, 0x56033046FC7B6BAB, 0xF0CFBC209AF4361D,
];

/// 64 bits of 2/π starting at bit offset `g` (0 = first bit after the point).
fn two_over_pi_bits(g: usize) -> u64 {
    let (w, off) = (g / 64, g % 64);
    if off == 0 {
        TWO_OVER_PI[w]
    } else {
        (TWO_OVER_PI[w] << off) | (TWO_OVER_PI[w + 1] >> (64 - off))
    }
}

/// Bits `[lo, lo + 128)` of the 256-bit little-endian integer `p`.
fn bits128(p: &[u64; 4], lo: usize) -> u128 {
    let get = |k: usize| if k < 4 { p[k] } else { 0 };
    let (w, off) = (lo / 64, lo % 64);
    let (lo64, hi64) = if off == 0 {
        (get(w), get(w + 1))
    } else {
        (
            (get(w) >> off) | (get(w + 1) << (64 - off)),
            (get(w + 1) >> off) | (get(w + 2) << (64 - off)),
        )
    };
    ((hi64 as u128) << 64) | lo64 as u128
}

/// Dekker split: a = hi + lo with hi holding the top 26 bits.
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a; // 2^27 + 1
    let hi = c - (c - a);
    (hi, a - hi)
}

/// Exact product a·b = p + e without FMA (Dekker).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

const PIO2_HI: f64 = 1.5707963267948966; // 0x3FF921FB54442D18
const PIO2_LO: f64 = 6.123233995736766e-17; // 0x3C91A62633145C07

/// Payne–Hanek reduction for finite ax ≥ 2^20·π/2.
fn rem_pio2_large(ax: f64) -> (i32, f64, f64) {
    let bits = ax.to_bits();
    let m = (bits & ((1u64 << 52) - 1)) | (1u64 << 52); // ax = m · 2^e
    let e = ((bits >> 52) & 0x7ff) as i64 - 1075;
    // Bit i of 2/π (1-based) contributes m·2^(e-i); for i ≤ e-2 that is a
    // multiple of 4 and cannot change the quadrant, so skip it.
    let i0 = if e >= 2 { e - 1 } else { 1 };
    let g0 = (i0 - 1) as usize;
    let t = [
        two_over_pi_bits(g0 + 128), // least significant
        two_over_pi_bits(g0 + 64),
        two_over_pi_bits(g0),
    ];
    // P = m · T  (T = 192-bit window, P < 2^245).
    let mut p = [0u64; 4];
    let mut carry: u128 = 0;
    for k in 0..3 {
        let prod = (m as u128) * (t[k] as u128) + carry;
        p[k] = prod as u64;
        carry = prod >> 64;
    }
    p[3] = carry as u64;
    // x·2/π ≡ P · 2^-s (mod 4), with s = i0 + 191 - e ∈ [190, 224].
    let s = (i0 + 191 - e) as usize;
    let mut n = (bits128(&p, s) & 3) as i32;
    let frac = bits128(&p, s - 128); // floor(fraction · 2^128)
    // Round to the nearest quadrant: fraction in [-1/2, 1/2).
    let (mag, neg) = if frac >> 127 == 1 {
        n += 1;
        (frac.wrapping_neg(), true) // 2^128 - frac, ≤ 2^127
    } else {
        (frac, false)
    };
    // mag·2^-128 as a double-double a + b, then times π/2.
    let a = mag as f64;
    let b = (mag as i128 - a as u128 as i128) as f64;
    let (hi, err) = two_prod(a, PIO2_HI);
    let lo = err + (a * PIO2_LO + b * PIO2_HI);
    let y0 = hi + lo;
    let y1 = lo - (y0 - hi);
    let scale = f64::from_bits(0x37F0_0000_0000_0000); // 2^-128
    let (y0, y1) = (y0 * scale, y1 * scale);
    if neg {
        (n, -y0, -y1)
    } else {
        (n, y0, y1)
    }
}

// ---------------------------------------------------------------------------
// Public trig
// ---------------------------------------------------------------------------

/// Deterministic sine (< 1 ulp).
pub fn sin(x: f64) -> f64 {
    let ix = high_word(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        // |x| ≲ π/4
        if ix < 0x3e50_0000 {
            return x; // |x| < 2^-26: sin x rounds to x
        }
        return k_sin(x, 0.0, false);
    }
    if ix >= 0x7ff0_0000 {
        return x - x; // NaN for inf and NaN
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_sin(y0, y1, true),
        1 => k_cos(y0, y1),
        2 => -k_sin(y0, y1, true),
        _ => -k_cos(y0, y1),
    }
}

/// Deterministic cosine (< 1 ulp).
pub fn cos(x: f64) -> f64 {
    let ix = high_word(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        if ix < 0x3e46_a09e {
            return 1.0; // |x| < 2^-27·√2: cos x rounds to 1
        }
        return k_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_cos(y0, y1),
        1 => -k_sin(y0, y1, true),
        2 => -k_cos(y0, y1),
        _ => k_sin(y0, y1, true),
    }
}

/// `(sin(x), cos(x))`, bit-identical to calling [`sin`] and [`cos`]
/// separately, with one shared argument reduction.
pub fn sin_cos(x: f64) -> (f64, f64) {
    let ix = high_word(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb || ix >= 0x7ff0_0000 {
        return (sin(x), cos(x));
    }
    let (n, y0, y1) = rem_pio2(x);
    let (s, c) = (k_sin(y0, y1, true), k_cos(y0, y1));
    match n & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

// ---------------------------------------------------------------------------
// exp  (fdlibm e_exp.c)
// ---------------------------------------------------------------------------

const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const INVLN2: f64 = 1.44269504088896338700e+00;
const P1: f64 = 1.66666666666666019037e-01;
const P2: f64 = -2.77777777770155933842e-03;
const P3: f64 = 6.61375632143793436117e-05;
const P4: f64 = -1.65339022054652515390e-06;
const P5: f64 = 4.13813679705723846039e-08;

/// Deterministic e^x (< 1 ulp).
pub fn exp(x: f64) -> f64 {
    let hx_full = high_word(x);
    let negative = hx_full >> 31 == 1;
    let hx = hx_full & 0x7fff_ffff;

    if hx >= 0x4086_232b {
        // |x| ≥ 708.39 or NaN
        if x.is_nan() {
            return x;
        }
        if x > 709.782712893383973096 {
            return f64::INFINITY;
        }
        if x < -745.13321910194110842 {
            return 0.0;
        }
    }

    let (hi, lo, k, xr);
    if hx > 0x3fd6_2e42 {
        // |x| > 0.5·ln2
        k = if hx >= 0x3ff0_a2b2 {
            // |x| ≥ 1.5·ln2
            (INVLN2 * x + if negative { -0.5 } else { 0.5 }) as i32
        } else if negative {
            -1
        } else {
            1
        };
        hi = x - k as f64 * LN2_HI; // exact
        lo = k as f64 * LN2_LO;
        xr = hi - lo;
    } else if hx > 0x3e30_0000 {
        // |x| > 2^-28
        k = 0;
        hi = x;
        lo = 0.0;
        xr = x;
    } else {
        return 1.0 + x;
    }

    let xx = xr * xr;
    let c = xr - xx * (P1 + xx * (P2 + xx * (P3 + xx * (P4 + xx * P5))));
    let y = 1.0 + (xr * c / (2.0 - c) - lo + hi);
    if k == 0 {
        y
    } else {
        scalbn(y, k)
    }
}

// ---------------------------------------------------------------------------
// ln  (fdlibm e_log.c)
// ---------------------------------------------------------------------------

const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

/// Deterministic natural logarithm (< 1 ulp).
pub fn ln(x: f64) -> f64 {
    let mut bits = x.to_bits();
    let mut hx = (bits >> 32) as u32;
    let mut k: i32 = 0;
    let mut x = x;

    if hx < 0x0010_0000 || hx >> 31 == 1 {
        if bits << 1 == 0 {
            return f64::NEG_INFINITY; // ln(±0)
        }
        if hx >> 31 == 1 {
            return f64::NAN; // ln(negative)
        }
        // Subnormal: scale up by 2^54.
        k -= 54;
        x *= f64::from_bits(0x4350_0000_0000_0000);
        bits = x.to_bits();
        hx = (bits >> 32) as u32;
    } else if hx >= 0x7ff0_0000 {
        return x; // +inf or NaN
    } else if hx == 0x3ff0_0000 && bits << 32 == 0 {
        return 0.0; // ln(1)
    }

    // Reduce x into [√2/2, √2].
    hx = hx.wrapping_add(0x3ff0_0000 - 0x3fe6_a09e);
    k += (hx >> 20) as i32 - 0x3ff;
    hx = (hx & 0x000f_ffff) + 0x3fe6_a09e;
    let x = f64::from_bits(((hx as u64) << 32) | (bits & 0xffff_ffff));

    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let dk = k as f64;
    s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
}

// ---------------------------------------------------------------------------
// Powers
// ---------------------------------------------------------------------------

/// `x^n` by binary exponentiation. Deterministic (fixed multiply order), not
/// correctly rounded: about `log2(n)` roundings.
pub fn powi(x: f64, n: i32) -> f64 {
    let mut base = x;
    let mut e = n.unsigned_abs();
    let mut acc = 1.0;
    while e > 0 {
        if e & 1 == 1 {
            acc *= base;
        }
        base *= base;
        e >>= 1;
    }
    if n < 0 {
        1.0 / acc
    } else {
        acc
    }
}

/// `x^y` for x > 0 as `exp(y · ln x)`, plus the exact cases `y == 0`,
/// `x == 1`, `x == 0`, and integer `y` (via [`powi`] when |y| < 2^31).
///
/// Deterministic, but the relative error grows like `|y · ln x| · 2^-53`
/// (not < 1 ulp). Negative `x` with non-integer `y` returns NaN.
pub fn pow(x: f64, y: f64) -> f64 {
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if y == y.trunc() && y.abs() < 2_147_483_648.0 {
        return powi(x, y as i32);
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f64::INFINITY };
    }
    exp(y * ln(x))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        let (ia, ib) = (a.to_bits() as i64, b.to_bits() as i64);
        // Map to a monotone integer line so ±0 and sign changes work.
        let key = |i: i64| if i < 0 { i64::MIN - i } else { i };
        (key(ia) - key(ib)).unsigned_abs()
    }

    /// SplitMix64 input generator (kept local: tests must not depend on Rng).
    fn inputs() -> Vec<f64> {
        let mut s = 0x0D15_EA5E_u64;
        let mut next = || {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut v = Vec::new();
        for _ in 0..20_000 {
            // Uniform in [-8π, 8π]: the rotation-gate range.
            let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
            v.push((u * 2.0 - 1.0) * 8.0 * std::f64::consts::PI);
        }
        for _ in 0..20_000 {
            // Random finite doubles across all exponents (both signs).
            let b = next() & !(0x7ffu64 << 52) | ((next() % 0x7ff) << 52);
            v.push(f64::from_bits(b));
        }
        for k in -64i32..=64 {
            v.push(k as f64 * std::f64::consts::FRAC_PI_2);
            v.push(k as f64 * std::f64::consts::FRAC_PI_4);
        }
        // Known hard case for 2/π reduction (closest double to a multiple of π/2).
        v.push(6381956970095103.0 * 2f64.powi(797));
        v
    }

    #[test]
    fn special_values() {
        assert_eq!(sin(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
        assert!(sin(f64::INFINITY).is_nan() && sin(f64::NAN).is_nan());
        assert_eq!(cos(0.0), 1.0);
        assert!(cos(f64::NEG_INFINITY).is_nan());
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(exp(-746.0), 0.0);
        assert!(exp(f64::NAN).is_nan());
        assert_eq!(ln(1.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert_eq!(ln(-0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan() && ln(f64::NAN).is_nan());
        assert_eq!(ln(f64::INFINITY), f64::INFINITY);
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(pow(-2.0, 3.0), -8.0);
        assert!(pow(-2.0, 0.5).is_nan());
        assert_eq!(powi(3.0, -2), 1.0 / 9.0);
    }

    #[test]
    fn exact_reference_points() {
        // Values checked against mpmath. fdlibm is < 1 ulp, not correctly
        // rounded: sin(π/6) is 0.49999999999999995027..., which rounds to
        // 0.49999999999999994, but the kernel returns 0.5 (0.90 ulp).
        assert_eq!(sin(std::f64::consts::FRAC_PI_6), 0.5);
        assert_eq!(cos(std::f64::consts::FRAC_PI_2), 6.123233995736766e-17);
        assert_eq!(sin(std::f64::consts::PI), 1.2246467991473532e-16);
        // fdlibm's known exp(1) result: one ulp above E (0.67 ulp error).
        assert_eq!(exp(1.0), 2.7182818284590455);
        assert_eq!(ulps(exp(1.0), std::f64::consts::E), 1);
        assert_eq!(ln(std::f64::consts::E), 1.0);
        assert_eq!(ln(2.0), std::f64::consts::LN_2);
        assert_eq!(ln(10.0), std::f64::consts::LN_10);
    }

    #[test]
    fn subnormal_exp_and_ln() {
        // exp into the subnormal range and ln of subnormals.
        let tiny = exp(-740.0);
        assert!(tiny > 0.0 && tiny < f64::MIN_POSITIVE);
        assert!(ulps(ln(f64::from_bits(1)), -744.4400719213812) <= 1);
        assert!(ulps(ln(f64::MIN_POSITIVE), -708.3964185322641) <= 1);
    }

    #[test]
    fn agrees_with_platform_libm_within_two_ulps() {
        // Both implementations are < 1 ulp from the true value, so they are
        // at most 2 ulps apart. A larger gap means a transcription error.
        let mut worst = [0u64; 4];
        for &x in &inputs() {
            if x.abs() < 1e15 {
                worst[0] = worst[0].max(ulps(sin(x), x.sin()));
                worst[1] = worst[1].max(ulps(cos(x), x.cos()));
            }
            let e = x.clamp(-745.0, 709.0);
            if exp(e) >= f64::MIN_POSITIVE {
                worst[2] = worst[2].max(ulps(exp(e), e.exp()));
            }
            if x > 0.0 {
                worst[3] = worst[3].max(ulps(ln(x), x.ln()));
            }
        }
        assert!(worst.iter().all(|&w| w <= 2), "max ulp gap sin/cos/exp/ln = {:?}", worst);
    }

    #[test]
    fn sin_cos_matches_separate_calls() {
        for &x in &inputs() {
            let (s, c) = sin_cos(x);
            assert_eq!(s.to_bits(), sin(x).to_bits(), "sin_cos({x}).0");
            assert_eq!(c.to_bits(), cos(x).to_bits(), "sin_cos({x}).1");
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        for &x in &inputs() {
            let (s, c) = sin_cos(x);
            if s.is_finite() {
                assert!((s * s + c * c - 1.0).abs() < 4.0 * f64::EPSILON, "x = {x}");
            }
        }
    }

    /// Golden hash of every output bit pattern. CI runs this on Linux,
    /// Windows, and macOS: a mismatch on any OS means the functions are no
    /// longer platform-independent.
    #[test]
    fn golden_hash_is_platform_independent() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: f64| {
            for byte in v.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for &x in &inputs() {
            mix(sin(x));
            mix(cos(x));
            mix(exp(x.clamp(-750.0, 710.0)));
            mix(ln(x.abs()));
            mix(pow(x.abs(), 0.37));
        }
        assert_eq!(h, GOLDEN_HASH, "dmath golden hash changed: got {:#018x}", h);
    }

    const GOLDEN_HASH: u64 = 0xa92d_f4d4_e1fb_935e;
}
