//! Deterministic elementary functions: [`sin`], [`cos`], [`sin_cos`], [`tan`],
//! [`asin`], [`acos`], [`atan`], [`atan2`], [`sinh`], [`cosh`], [`tanh`],
//! [`atanh`], [`exp`], [`exp_m1`], [`ln`], [`ln_1p`], [`log2`], [`log10`],
//! [`pow`], [`powi`], and [`hypot`] — plus the [`DetMath`] extension trait,
//! which exposes them as `f64` methods (`x.det_exp()`) so call sites migrate
//! from `f64::exp` by a pure rename.
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
//! Measured against mpmath (`verification/dmath_check*`), max error in ulps:
//!
//! | < 1 ulp | above 1 ulp (inherent to the musl algorithm) |
//! |---|---|
//! | `sin cos tan asin acos atan exp exp_m1 ln ln_1p log2 log10 pow hypot` | `atan2` 1.20, `cosh` 1.24, `sinh` 1.67, `atanh` ~1.7 (|x| < 0.5), `tanh` ~2 ([0.1, 0.2554]) |
//!
//! The right-hand column is not a transcription loss: `dmath` is
//! bit-identical to musl's own C code on every tested input
//! (`verification/musl_bitcompare/`). [`pow`] is fdlibm `e_pow.c` (0.81 ulp
//! measured), not `exp(y · ln x)`. [`powi`] is exact repeated squaring (same
//! result as a multiply chain). `sqrt` needs nothing here: IEEE requires it
//! to be correctly rounded.
//!
//! # Provenance
//!
//! [`tanh`], [`sinh`], [`cosh`], [`atanh`], and [`hypot`] are transcribed
//! from musl's own implementations (MIT license, Copyright © 2005-2020 Rich
//! Felker, et al.). [`log2`] and [`pow`] are fdlibm as kept in FreeBSD
//! `msun` (musl replaced both with table-driven code). Everything else —
//! polynomial kernels, the Cody–Waite reduction, `sin`/`cos`/`tan`,
//! `asin`/`acos`/`atan`/`atan2`, `exp`/`exp_m1`, `ln`/`ln_1p`/`log10` — is
//! fdlibm as ported by musl:
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

#[inline]
fn low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

/// fdlibm `SET_LOW_WORD`.
#[inline]
fn with_low_word(x: f64, lo: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | lo as u64)
}

/// fdlibm `SET_HIGH_WORD`.
#[inline]
fn with_high_word(x: f64, hi: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}

/// fdlibm `INSERT_WORDS`.
#[inline]
fn from_words(hi: u32, lo: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | lo as u64)
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

// ---------------------------------------------------------------------------
// tan  (fdlibm k_tan.c / s_tan.c, as in musl)
// ---------------------------------------------------------------------------

const T: [f64; 13] = [
    3.33333333333334091986e-01,
    1.33333333333201242699e-01,
    5.39682539762260521377e-02,
    2.18694882948595424599e-02,
    8.86323982359930005737e-03,
    3.59207910759131235356e-03,
    1.45620945432529025516e-03,
    5.88041240820264096874e-04,
    2.46463134818469906812e-04,
    7.81794442939557092300e-05,
    7.14072491382608190305e-05,
    -1.85586374855275456654e-05,
    2.59073051863633712884e-05,
];
const PIO4LO: f64 = 3.06161699786838301793e-17;

/// tan(x + y) for |x| ≤ ~π/4; `odd` gives -1/tan instead.
fn k_tan(x: f64, y: f64, odd: bool) -> f64 {
    let hx = high_word(x);
    let big = (hx & 0x7fff_ffff) >= 0x3fe5_9428; // |x| >= 0.6744
    let (mut x, mut y) = (x, y);
    let sign = hx >> 31 == 1;
    if big {
        if sign {
            x = -x;
            y = -y;
        }
        x = (PIO4 - x) + (PIO4LO - y);
        y = 0.0;
    }
    let z = x * x;
    let w = z * z;
    let r = T[1] + w * (T[3] + w * (T[5] + w * (T[7] + w * (T[9] + w * T[11]))));
    let v = z * (T[2] + w * (T[4] + w * (T[6] + w * (T[8] + w * (T[10] + w * T[12])))));
    let s = z * x;
    let r = y + z * (s * (r + v) + y) + s * T[0];
    let w = x + r;
    if big {
        let s = if odd { -1.0 } else { 1.0 };
        let v = s - 2.0 * (x + (r - w * w / (w + s)));
        return if sign { -v } else { v };
    }
    if !odd {
        return w;
    }
    // -1/(x + r) has up to 2 ulp error, so compute it accurately.
    let w0 = with_low_word(w, 0);
    let v = r - (w0 - x); // w0 + v = r + x
    let a = -1.0 / w;
    let a0 = with_low_word(a, 0);
    a0 + a * (1.0 + a0 * w0 + a0 * v)
}

/// Deterministic tangent (< 1 ulp).
pub fn tan(x: f64) -> f64 {
    let ix = high_word(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        if ix < 0x3e40_0000 {
            return x; // |x| < 2^-27
        }
        return k_tan(x, 0.0, false);
    }
    if ix >= 0x7ff0_0000 {
        return x - x; // NaN for inf and NaN
    }
    let (n, y0, y1) = rem_pio2(x);
    k_tan(y0, y1, n & 1 == 1)
}

// ---------------------------------------------------------------------------
// asin / acos  (fdlibm e_asin.c / e_acos.c, as in musl)
// ---------------------------------------------------------------------------

const PS0: f64 = 1.66666666666666657415e-01;
const PS1: f64 = -3.25565818622400915405e-01;
const PS2: f64 = 2.01212532134862925881e-01;
const PS3: f64 = -4.00555345006794114027e-02;
const PS4: f64 = 7.91534994289814532176e-04;
const PS5: f64 = 3.47933107596021167570e-05;
const QS1: f64 = -2.40339491173441421878e+00;
const QS2: f64 = 2.02094576023350569471e+00;
const QS3: f64 = -6.88283971605453293030e-01;
const QS4: f64 = 7.70381505559019352791e-02;

/// Rational approximation shared by asin and acos.
fn asin_r(z: f64) -> f64 {
    let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
    let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
    p / q
}

/// Deterministic arcsine (< 1 ulp). NaN outside [-1, 1].
pub fn asin(x: f64) -> f64 {
    let hx = high_word(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        if (ix.wrapping_sub(0x3ff0_0000) | low_word(x)) == 0 {
            return x * PIO2_HI; // asin(±1) = ±π/2
        }
        return f64::NAN; // |x| > 1 or NaN
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix < 0x3e50_0000 && ix >= 0x0010_0000 {
            return x;
        }
        return x + x * asin_r(x * x);
    }
    // 0.5 <= |x| < 1
    let z = (1.0 - x.abs()) * 0.5;
    let s = z.sqrt();
    let r = asin_r(z);
    let y = if ix >= 0x3fef_3333 {
        PIO2_HI - (2.0 * (s + s * r) - PIO2_LO)
    } else {
        let f = with_low_word(s, 0);
        let c = (z - f * f) / (s + f);
        0.5 * PIO2_HI - (2.0 * s * r - (PIO2_LO - 2.0 * c) - (0.5 * PIO2_HI - 2.0 * f))
    };
    if hx >> 31 == 1 {
        -y
    } else {
        y
    }
}

/// Deterministic arccosine (< 1 ulp). NaN outside [-1, 1].
pub fn acos(x: f64) -> f64 {
    let hx = high_word(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        if (ix.wrapping_sub(0x3ff0_0000) | low_word(x)) == 0 {
            // acos(1) = 0, acos(-1) = π
            return if hx >> 31 == 1 { 2.0 * PIO2_HI } else { 0.0 };
        }
        return f64::NAN;
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix <= 0x3c60_0000 {
            return PIO2_HI; // |x| < 2^-57
        }
        return PIO2_HI - (x - (PIO2_LO - x * asin_r(x * x)));
    }
    if hx >> 31 == 1 {
        // x < -0.5
        let z = (1.0 + x) * 0.5;
        let s = z.sqrt();
        let w = asin_r(z) * s - PIO2_LO;
        return 2.0 * (PIO2_HI - (s + w));
    }
    // x > 0.5
    let z = (1.0 - x) * 0.5;
    let s = z.sqrt();
    let df = with_low_word(s, 0);
    let c = (z - df * df) / (s + df);
    let w = asin_r(z) * s + c;
    2.0 * (df + w)
}

// ---------------------------------------------------------------------------
// atan / atan2  (fdlibm s_atan.c / e_atan2.c, as in musl)
// ---------------------------------------------------------------------------

const ATANHI: [f64; 4] = [
    4.63647609000806093515e-01,
    7.85398163397448278999e-01,
    9.82793723247329054082e-01,
    1.57079632679489655800e+00,
];
const ATANLO: [f64; 4] = [
    2.26987774529616870924e-17,
    3.06161699786838301793e-17,
    1.39033110312309984516e-17,
    6.12323399573676603587e-17,
];
const AT: [f64; 11] = [
    3.33333333333329318027e-01,
    -1.99999999998764832476e-01,
    1.42857142725034663711e-01,
    -1.11111104054623557880e-01,
    9.09088713343650656196e-02,
    -7.69187620504482999495e-02,
    6.66107313738753120669e-02,
    -5.83357013379057348645e-02,
    4.97687799461593236017e-02,
    -3.65315727442169155270e-02,
    1.62858201153657823623e-02,
];

/// Deterministic arctangent (< 1 ulp).
pub fn atan(x: f64) -> f64 {
    let hx = high_word(x);
    let sign = hx >> 31 == 1;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x4410_0000 {
        // |x| >= 2^66
        if x.is_nan() {
            return x;
        }
        return if sign { -ATANHI[3] } else { ATANHI[3] };
    }
    let (id, x) = if ix < 0x3fdc_0000 {
        // |x| < 0.4375
        if ix < 0x3e40_0000 {
            return x; // |x| < 2^-27
        }
        (-1, x)
    } else {
        let x = x.abs();
        if ix < 0x3ff3_0000 {
            if ix < 0x3fe6_0000 {
                (0, (2.0 * x - 1.0) / (2.0 + x)) // 7/16 <= |x| < 11/16
            } else {
                (1, (x - 1.0) / (x + 1.0)) // 11/16 <= |x| < 19/16
            }
        } else if ix < 0x4003_8000 {
            (2, (x - 1.5) / (1.0 + 1.5 * x)) // |x| < 2.4375
        } else {
            (3, -1.0 / x) // 2.4375 <= |x| < 2^66
        }
    };
    let z = x * x;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        return x - x * (s1 + s2);
    }
    let id = id as usize;
    let z = ATANHI[id] - (x * (s1 + s2) - ATANLO[id] - x);
    if sign {
        -z
    } else {
        z
    }
}

const PI: f64 = 3.1415926535897931160E+00;
const PI_LO: f64 = 1.2246467991473531772E-16;

/// Deterministic `atan2(y, x)`, the angle of the point `(x, y)` (≤ 1.20 ulp
/// measured: `atan` of the rounded quotient `y/x`, as in fdlibm/musl).
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    let (ix, lx) = (high_word(x), low_word(x));
    let (iy, ly) = (high_word(y), low_word(y));
    if (ix.wrapping_sub(0x3ff0_0000) | lx) == 0 {
        return atan(y); // x = 1.0
    }
    let m = ((iy >> 31) & 1) | ((ix >> 30) & 2); // 2·sign(x) + sign(y)
    let ix = ix & 0x7fff_ffff;
    let iy = iy & 0x7fff_ffff;

    if (iy | ly) == 0 {
        // y = 0
        return match m {
            0 | 1 => y,
            2 => PI,
            _ => -PI,
        };
    }
    if (ix | lx) == 0 {
        // x = 0
        return if m & 1 == 1 { -PI / 2.0 } else { PI / 2.0 };
    }
    if ix == 0x7ff0_0000 {
        // x = ±inf
        return if iy == 0x7ff0_0000 {
            match m {
                0 => PI / 4.0,
                1 => -PI / 4.0,
                2 => 3.0 * PI / 4.0,
                _ => -3.0 * PI / 4.0,
            }
        } else {
            match m {
                0 => 0.0,
                1 => -0.0,
                2 => PI,
                _ => -PI,
            }
        };
    }
    if ix + (64 << 20) < iy || iy == 0x7ff0_0000 {
        // |y/x| > 2^64
        return if m & 1 == 1 { -PI / 2.0 } else { PI / 2.0 };
    }
    let z = if m & 2 != 0 && iy + (64 << 20) < ix {
        0.0 // |y/x| < 2^-64, x < 0
    } else {
        atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
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
// exp_m1  (fdlibm s_expm1.c, as in musl)
// ---------------------------------------------------------------------------

const EXPM1_O_THRESHOLD: f64 = 7.09782712893383973096e+02;
// Scaled Q's: Qn here = 2^n · Qn in the fdlibm comment, for R(2z), z = x²/2.
const EQ1: f64 = -3.33333333333331316428e-02;
const EQ2: f64 = 1.58730158725481460165e-03;
const EQ3: f64 = -7.93650757867487942473e-05;
const EQ4: f64 = 4.00821782732936239552e-06;
const EQ5: f64 = -2.01099218183624371326e-07;

/// Deterministic `e^x - 1` (< 1 ulp), accurate near 0 where `exp(x) - 1`
/// cancels.
pub fn exp_m1(x: f64) -> f64 {
    let bits = x.to_bits();
    let hx = (bits >> 32) as u32 & 0x7fff_ffff;
    let sign = bits >> 63 == 1;
    let mut x = x;

    if hx >= 0x4043_687a {
        // |x| >= 56·ln2
        if x.is_nan() {
            return x;
        }
        if sign {
            return -1.0;
        }
        if x > EXPM1_O_THRESHOLD {
            return f64::INFINITY;
        }
    }

    let (k, c): (i32, f64);
    if hx > 0x3fd6_2e42 {
        // |x| > 0.5·ln2
        let (hi, lo);
        if hx < 0x3ff0_a2b2 {
            // and |x| < 1.5·ln2
            if !sign {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if sign { -0.5 } else { 0.5 }) as i32;
            let t = k as f64;
            hi = x - t * LN2_HI; // exact
            lo = t * LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3c90_0000 {
        return x; // |x| < 2^-54
    } else {
        k = 0;
        c = 0.0;
    }

    // x is now in the primary range.
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0 + hxs * (EQ1 + hxs * (EQ2 + hxs * (EQ3 + hxs * (EQ4 + hxs * EQ5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs); // c is 0
    }
    e = x * (e - c) - c;
    e -= hxs;
    // exp(x) ~ 2^k (x_reduced - e + 1)
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return 1.0 + 2.0 * (x - e);
    }
    let twopk = f64::from_bits(((0x3ff + k) as u64) << 52); // 2^k (inf for k = 1024, unused)
    if !(0..=56).contains(&k) {
        // Suffices to return exp(x) - 1.
        let mut y = x - e + 1.0;
        if k == 1024 {
            y = y * 2.0 * f64::from_bits(0x7FE0_0000_0000_0000);
        } else {
            y *= twopk;
        }
        return y - 1.0;
    }
    let tk = f64::from_bits(((0x3ff - k) as u64) << 52); // 2^-k
    if k < 20 {
        (x - e + (1.0 - tk)) * twopk
    } else {
        (x - (e + tk) + 1.0) * twopk
    }
}

// ---------------------------------------------------------------------------
// Hyperbolic functions  (musl tanh.c, sinh.c, cosh.c, __expo2.c)
// ---------------------------------------------------------------------------

/// `exp(x)/2` for `x >= ln(f64::MAX)`, without the intermediate overflow of
/// `0.5 * exp(x)` (musl `__expo2`).
fn expo2(x: f64, sign: f64) -> f64 {
    let kln2 = f64::from_bits(0x4096_2066_151a_dd8b); // 2043·ln2 (0x1.62066151add8bp+10)
    let scale = f64::from_bits(0x7FC0_0000_0000_0000); // 2^1021 = 2^(2043/2)
    exp(x - kln2) * (sign * scale) * scale
}

/// Deterministic hyperbolic tangent (< 1 ulp, up to ~2 ulp on [0.1, 0.2554]).
pub fn tanh(x: f64) -> f64 {
    let bits = x.to_bits();
    let sign = bits >> 63 == 1;
    let ax = f64::from_bits(bits & (u64::MAX >> 1));
    let w = (ax.to_bits() >> 32) as u32;
    let t = if w > 0x3fe1_93ea {
        // |x| > ln(3)/2 ≈ 0.5493, or NaN
        if w > 0x4034_0000 {
            1.0 - 0.0 / ax // |x| > 20: ±1 (NaN stays NaN)
        } else {
            let t = exp_m1(2.0 * ax);
            1.0 - 2.0 / (t + 2.0)
        }
    } else if w > 0x3fd0_58ae {
        // |x| > ln(5/3)/2 ≈ 0.2554
        let t = exp_m1(2.0 * ax);
        t / (t + 2.0)
    } else if w >= 0x0010_0000 {
        let t = exp_m1(-2.0 * ax);
        -t / (t + 2.0)
    } else {
        ax // subnormal
    };
    if sign {
        -t
    } else {
        t
    }
}

/// Deterministic hyperbolic sine (≤ 1.67 ulp measured, as musl's own code).
pub fn sinh(x: f64) -> f64 {
    let bits = x.to_bits();
    let h = if bits >> 63 == 1 { -0.5 } else { 0.5 };
    let absx = f64::from_bits(bits & (u64::MAX >> 1));
    let w = (absx.to_bits() >> 32) as u32;
    if w < 0x4086_2e42 {
        // |x| < ln(f64::MAX)
        let t = exp_m1(absx);
        if w < 0x3ff0_0000 {
            if w < 0x3ff0_0000 - (26 << 20) {
                return x;
            }
            return h * (2.0 * t - t * t / (t + 1.0));
        }
        return h * (t + t / (t + 1.0));
    }
    expo2(absx, 2.0 * h) // |x| >= ln(f64::MAX), or NaN
}

/// Deterministic hyperbolic cosine (≤ 1.24 ulp measured, as musl's own code).
pub fn cosh(x: f64) -> f64 {
    let ax = f64::from_bits(x.to_bits() & (u64::MAX >> 1));
    let w = (ax.to_bits() >> 32) as u32;
    if w < 0x3fe6_2e42 {
        // |x| < ln2
        if w < 0x3ff0_0000 - (26 << 20) {
            return 1.0;
        }
        let t = exp_m1(ax);
        return 1.0 + t * t / (2.0 * (1.0 + t));
    }
    if w < 0x4086_2e42 {
        // |x| < ln(f64::MAX)
        let t = exp(ax);
        return 0.5 * (t + 1.0 / t);
    }
    expo2(ax, 1.0) // |x| >= ln(f64::MAX), or NaN
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

/// fdlibm `k_log1p`: `log(1 + f) - f + f²/2` for `f` in [√2/2 - 1, √2 - 1].
fn k_log1p(f: f64) -> f64 {
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    s * (hfsq + r)
}

/// Deterministic `ln(1 + x)` (< 1 ulp), accurate near 0 where `ln(1 + x)`
/// loses the low bits of `x`.
pub fn ln_1p(x: f64) -> f64 {
    let hx = high_word(x);
    let mut k: i32 = 1;
    let (mut c, mut f) = (0.0, 0.0);
    if hx < 0x3fda_827a || hx >> 31 == 1 {
        // 1 + x < √2+
        if hx >= 0xbff0_0000 {
            // x <= -1
            if x == -1.0 {
                return f64::NEG_INFINITY;
            }
            return f64::NAN;
        }
        if hx << 1 < 0x3ca0_0000 << 1 {
            return x; // |x| < 2^-53
        }
        if hx <= 0xbfd2_bec4 {
            // √2/2- <= 1 + x < √2+
            k = 0;
            c = 0.0;
            f = x;
        }
    } else if hx >= 0x7ff0_0000 {
        return x; // +inf or NaN
    }
    if k != 0 {
        let u = 1.0 + x;
        let ubits = u.to_bits();
        let hu = ((ubits >> 32) as u32).wrapping_add(0x3ff0_0000 - 0x3fe6_a09e);
        k = (hu >> 20) as i32 - 0x3ff;
        // Correction term ~ log(1+x) - log(u); avoid underflow in c/u.
        if k < 54 {
            c = if k >= 2 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            c = 0.0;
        }
        // Reduce u into [√2/2, √2].
        let hu = (hu & 0x000f_ffff) + 0x3fe6_a09e;
        let u = f64::from_bits(((hu as u64) << 32) | (ubits & 0xffff_ffff));
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let dk = k as f64;
    s * (hfsq + r) + (dk * LN2_LO + c) - hfsq + f + dk * LN2_HI
}

/// Deterministic inverse hyperbolic tangent (< 1 ulp; up to ~1.7 ulp for
/// |x| < 0.5, as in musl). NaN outside [-1, 1], ±inf at ±1.
pub fn atanh(x: f64) -> f64 {
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as u32;
    let negative = bits >> 63 == 1;
    let mut y = f64::from_bits(bits & (u64::MAX >> 1));
    if e < 0x3ff - 1 {
        if e >= 0x3ff - 32 {
            // |x| < 0.5
            y = 0.5 * ln_1p(2.0 * y + 2.0 * y * y / (1.0 - y));
        }
        // else |x| < 2^-32: atanh(x) rounds to x
    } else {
        y = 0.5 * ln_1p(2.0 * (y / (1.0 - y))); // avoids overflow
    }
    if negative {
        -y
    } else {
        y
    }
}

const IVLN2HI: f64 = 1.44269504072144627571e+00; // 0x3ff71547 65200000
const IVLN2LO: f64 = 1.67517131648865118353e-10; // 0x3de705fc 2eefa200

/// Deterministic base-2 logarithm (< 1 ulp; FreeBSD msun `e_log2.c`). Exact
/// at powers of two.
pub fn log2(x: f64) -> f64 {
    let (mut hx, lx) = (high_word(x) as i32, low_word(x));
    let mut x = x;
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        // x < 2^-1022, or negative
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return f64::NEG_INFINITY; // log2(±0)
        }
        if hx < 0 {
            return f64::NAN; // log2(negative)
        }
        k -= 54; // subnormal: scale up
        x *= f64::from_bits(0x4350_0000_0000_0000); // 2^54
        hx = high_word(x) as i32;
    }
    if hx >= 0x7ff0_0000 {
        return x + x; // +inf or NaN
    }
    if hx == 0x3ff0_0000 && lx == 0 {
        return 0.0; // log2(1) = +0
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x0010_0000;
    let x = with_high_word(x, (hx | (i ^ 0x3ff0_0000)) as u32); // normalize x or x/2
    k += i >> 20;
    let y = k as f64;
    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let r = k_log1p(f);

    // f - hfsq in extra precision (hi + lo), then y added in extra precision.
    let hi = with_low_word(f - hfsq, 0);
    let lo = (f - hi) - hfsq + r;
    let mut val_hi = hi * IVLN2HI;
    let mut val_lo = (lo + hi) * IVLN2LO + lo * IVLN2HI;
    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;
    val_lo + val_hi
}

const IVLN10HI: f64 = 4.34294481878168880939e-01; // 0x3fdbcb7b 15200000
const IVLN10LO: f64 = 2.50829467116452752298e-11; // 0x3dbb9438 ca9aadd5
const LOG10_2HI: f64 = 3.01029995663611771306e-01; // 0x3FD34413 509F6000
const LOG10_2LO: f64 = 3.69423907715893078616e-13; // 0x3D59FEF3 11F12B36

/// Deterministic base-10 logarithm (< 1 ulp).
pub fn log10(x: f64) -> f64 {
    let mut bits = x.to_bits();
    let mut hx = (bits >> 32) as u32;
    let mut k: i32 = 0;
    let mut x = x;
    if hx < 0x0010_0000 || hx >> 31 == 1 {
        if bits << 1 == 0 {
            return f64::NEG_INFINITY; // log10(±0)
        }
        if hx >> 31 == 1 {
            return f64::NAN; // log10(negative)
        }
        k -= 54; // subnormal: scale up
        x *= f64::from_bits(0x4350_0000_0000_0000); // 2^54
        bits = x.to_bits();
        hx = (bits >> 32) as u32;
    } else if hx >= 0x7ff0_0000 {
        return x;
    } else if hx == 0x3ff0_0000 && bits << 32 == 0 {
        return 0.0;
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

    // hi + lo = f - hfsq + s·(hfsq + r) ≈ log(1 + f)
    let hi = f64::from_bits((f - hfsq).to_bits() & (u64::MAX << 32));
    let lo = f - hi - hfsq + s * (hfsq + r);
    // val_hi + val_lo ≈ log10(1 + f) + k·log10(2)
    let mut val_hi = hi * IVLN10HI;
    let dk = k as f64;
    let y = dk * LOG10_2HI;
    let mut val_lo = dk * LOG10_2LO + (lo + hi) * IVLN10LO + lo * IVLN10HI;
    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;
    val_lo + val_hi
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

const BP: [f64; 2] = [1.0, 1.5];
const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01]; // 0x3FE2B803 40000000
const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08]; // 0x3E4CFDEB 43CFD006
const TWO53: f64 = 9007199254740992.0;
const HUGE: f64 = 1.0e300;
const TINY: f64 = 1.0e-300;
const THRD: f64 = 3.3333333333333331e-01;
// Poly coefs for (3/2)·(log(x) - 2s - 2/3·s³).
const PL1: f64 = 5.99999999999994648725e-01;
const PL2: f64 = 4.28571428578550184252e-01;
const PL3: f64 = 3.33333329818377432918e-01;
const PL4: f64 = 2.72728123808534006489e-01;
const PL5: f64 = 2.30660745775561754067e-01;
const PL6: f64 = 2.06975017800338417784e-01;
const LG2_FULL: f64 = 6.93147180559945286227e-01; // 0x3FE62E42 FEFA39EF
const LG2_H: f64 = 6.93147182464599609375e-01; // 0x3FE62E43 00000000
const LG2_L: f64 = -1.90465429995776804525e-09; // 0xBE205C61 0CA86C39
const OVT: f64 = 8.0085662595372944372e-17; // -(1024 - log2(ovfl + .5ulp))
const CP: f64 = 9.61796693925975554329e-01; // 2/(3·ln2)
const CP_H: f64 = 9.61796700954437255859e-01; // (float)CP
const CP_L: f64 = -7.02846165095275826516e-09; // tail of CP_H
const IVLN2: f64 = 1.44269504088896338700e+00;
const IVLN2_H: f64 = 1.44269502162933349609e+00; // 24-bit 1/ln2
const IVLN2_L: f64 = 1.92596299112661746887e-08; // 1/ln2 tail

/// Deterministic `x^y` (< 1 ulp; fdlibm `e_pow.c` as kept in FreeBSD
/// `msun`). Follows C99 Annex F for special values: `x^0 = 1` and `1^y = 1`
/// even for NaN, and negative `x` with non-integer `y` is NaN.
pub fn pow(x: f64, y: f64) -> f64 {
    let (hx, lx) = (high_word(x) as i32, low_word(x));
    let (hy, ly) = (high_word(y) as i32, low_word(y));
    let ix = hx & 0x7fff_ffff;
    let iy = hy & 0x7fff_ffff;

    if (iy as u32 | ly) == 0 {
        return 1.0; // x^0 = 1
    }
    if hx == 0x3ff0_0000 && lx == 0 {
        return 1.0; // 1^y = 1, even for NaN y
    }
    if ix > 0x7ff0_0000
        || (ix == 0x7ff0_0000 && lx != 0)
        || iy > 0x7ff0_0000
        || (iy == 0x7ff0_0000 && ly != 0)
    {
        return x + y; // NaN
    }

    // yisint: 0 = y not an integer, 1 = odd integer, 2 = even integer
    // (only computed for x < 0).
    let mut yisint: i32 = 0;
    if hx < 0 {
        if iy >= 0x4340_0000 {
            yisint = 2; // |y| >= 2^53: even
        } else if iy >= 0x3ff0_0000 {
            let k = (iy >> 20) - 0x3ff; // exponent
            if k > 20 {
                let j = ly >> (52 - k);
                if (j << (52 - k)) == ly {
                    yisint = 2 - (j & 1) as i32;
                }
            } else if ly == 0 {
                let j = iy >> (20 - k);
                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    // Special values of y.
    if ly == 0 {
        if iy == 0x7ff0_0000 {
            // y = ±inf
            if ((ix.wrapping_sub(0x3ff0_0000)) as u32 | lx) == 0 {
                return 1.0; // (-1)^±inf = 1
            } else if ix >= 0x3ff0_0000 {
                return if hy >= 0 { y } else { 0.0 }; // (|x| > 1)^±inf = inf, 0
            } else {
                return if hy < 0 { -y } else { 0.0 }; // (|x| < 1)^∓inf = inf, 0
            }
        }
        if iy == 0x3ff0_0000 {
            return if hy < 0 { 1.0 / x } else { x }; // y = ±1
        }
        if hy == 0x4000_0000 {
            return x * x; // y = 2
        }
        if hy == 0x3fe0_0000 && hx >= 0 {
            return x.sqrt(); // y = 0.5, x >= +0
        }
    }

    let mut ax = x.abs();
    // Special values of x: ±0, ±inf, ±1.
    if lx == 0 && (ix == 0x7ff0_0000 || ix == 0 || ix == 0x3ff0_0000) {
        let mut z = ax;
        if hy < 0 {
            z = 1.0 / z;
        }
        if hx < 0 {
            if (ix.wrapping_sub(0x3ff0_0000) | yisint) == 0 {
                z = (z - z) / (z - z); // (-1)^non-int is NaN
            } else if yisint == 1 {
                z = -z; // (x < 0)^odd = -(|x|^odd)
            }
        }
        return z;
    }

    // n = 0 for x < 0, -1 for x > 0.
    let n = ((hx as u32) >> 31) as i32 - 1;
    if (n | yisint) == 0 {
        return (x - x) / (x - x); // (x < 0)^(non-int) is NaN
    }
    let s = if (n | (yisint - 1)) == 0 { -1.0 } else { 1.0 }; // (-ve)^(odd int)

    let (t1, t2);
    if iy > 0x41e0_0000 {
        // |y| > 2^31
        if iy > 0x43f0_0000 {
            // |y| > 2^64: must over/underflow
            if ix <= 0x3fef_ffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }
            if ix >= 0x3ff0_0000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }
        // Over/underflow if x is not close to one.
        if ix < 0x3fef_ffff {
            return if hy < 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        if ix > 0x3ff0_0000 {
            return if hy > 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        // |1 - x| <= 2^-20: log(x) by x - x²/2 + x³/3 - x⁴/4.
        let t = ax - 1.0; // 20 trailing zeros
        let w = (t * t) * (0.5 - t * (THRD - t * 0.25));
        let u = IVLN2_H * t; // IVLN2_H has 21 significant bits
        let v = t * IVLN2_L - w * IVLN2;
        t1 = with_low_word(u + v, 0);
        t2 = v - (t1 - u);
    } else {
        let mut n: i32 = 0;
        let mut ix = ix;
        if ix < 0x0010_0000 {
            // subnormal x
            ax *= TWO53;
            n -= 53;
            ix = high_word(ax) as i32;
        }
        n += (ix >> 20) - 0x3ff;
        let j = ix & 0x000f_ffff;
        // Determine interval.
        ix = j | 0x3ff0_0000; // normalize ix
        let k: usize;
        if j <= 0x3988E {
            k = 0; // |x| < √(3/2)
        } else if j < 0xBB67A {
            k = 1; // |x| < √3
        } else {
            k = 0;
            n += 1;
            ix -= 0x0010_0000;
        }
        ax = with_high_word(ax, ix as u32);

        // ss = s_h + s_l = (x - 1)/(x + 1) or (x - 1.5)/(x + 1.5)
        let u = ax - BP[k];
        let v = 1.0 / (ax + BP[k]);
        let ss = u * v;
        let s_h = with_low_word(ss, 0);
        // t_h = ax + BP[k], high part
        let t_h = from_words(((ix >> 1) as u32 | 0x2000_0000) + 0x0008_0000 + ((k as u32) << 18), 0);
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // log(ax)
        let mut s2 = ss * ss;
        let mut r = s2 * s2 * (PL1 + s2 * (PL2 + s2 * (PL3 + s2 * (PL4 + s2 * (PL5 + s2 * PL6)))));
        r += s_l * (s_h + ss);
        s2 = s_h * s_h;
        let t_h = with_low_word(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        // u + v = ss·(1 + ...)
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        // 2/(3·ln2)·(ss + ...)
        let p_h = with_low_word(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h; // CP_H + CP_L = 2/(3·ln2)
        let z_l = CP_L * p_h + p_l * CP + DP_L[k];
        // log2(ax) = (ss + ..)·2/(3·ln2) = n + DP_H + z_h + z_l
        let t = n as f64;
        t1 = with_low_word(((z_h + z_l) + DP_H[k]) + t, 0);
        t2 = z_l - (((t1 - t) - DP_H[k]) - z_h);
    }

    // Split y into y1 + y2 and compute (y1 + y2)·(t1 + t2).
    let y1 = with_low_word(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let (j, i) = (high_word(z) as i32, low_word(z));
    if j >= 0x4090_0000 {
        // z >= 1024
        if (j.wrapping_sub(0x4090_0000) as u32 | i) != 0 || p_l + OVT > z - p_h {
            return s * HUGE * HUGE; // overflow
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        // z <= -1075
        if ((j as u32).wrapping_sub(0xc090_cc00) | i) != 0 || p_l <= z - p_h {
            return s * TINY * TINY; // underflow
        }
    }

    // 2^(p_h + p_l)
    let i = j & 0x7fff_ffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n: i32 = 0;
    if i > 0x3fe0_0000 {
        // |z| > 0.5: n = [z + 0.5]
        n = j.wrapping_add(0x0010_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 20) - 0x3ff; // new k for n
        let t = from_words((n & !(0x000f_ffff >> k)) as u32, 0);
        n = ((n & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_low_word(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2_FULL + t * LG2_L;
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    let j = (high_word(z) as i32).wrapping_add(((n as u32) << 20) as i32);
    if (j >> 20) <= 0 {
        z = scalbn(z, n); // subnormal output
    } else {
        z = with_high_word(z, j as u32);
    }
    s * z
}

// ---------------------------------------------------------------------------
// hypot  (musl hypot.c)
// ---------------------------------------------------------------------------

/// Exact square: x² = hi + lo (Dekker, no FMA).
fn sq(x: f64) -> (f64, f64) {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let xc = x * SPLIT;
    let xh = x - xc + xc;
    let xl = x - xh;
    let hi = x * x;
    let lo = xh * xh - hi + 2.0 * xh * xl + xl * xl;
    (hi, lo)
}

/// Deterministic `sqrt(x² + y²)` without undue overflow or underflow
/// (< 1 ulp). `hypot(±inf, NaN)` is `+inf`, per C99.
pub fn hypot(x: f64, y: f64) -> f64 {
    let mut ux = x.to_bits() & (u64::MAX >> 1);
    let mut uy = y.to_bits() & (u64::MAX >> 1);
    if ux < uy {
        std::mem::swap(&mut ux, &mut uy); // arrange |x| >= |y|
    }
    let ex = (ux >> 52) as i32;
    let ey = (uy >> 52) as i32;
    let (mut x, mut y) = (f64::from_bits(ux), f64::from_bits(uy));
    if ey == 0x7ff {
        return y; // hypot(inf, NaN) == inf
    }
    if ex == 0x7ff || uy == 0 {
        return x;
    }
    if ex - ey > 64 {
        return x + y;
    }
    // Scale so the exact squares neither overflow nor underflow.
    let mut z = 1.0;
    if ex > 0x3ff + 510 {
        z = f64::from_bits(0x6BB0_0000_0000_0000); // 2^700
        x *= f64::from_bits(0x1430_0000_0000_0000); // 2^-700
        y *= f64::from_bits(0x1430_0000_0000_0000);
    } else if ey < 0x3ff - 450 {
        z = f64::from_bits(0x1430_0000_0000_0000); // 2^-700
        x *= f64::from_bits(0x6BB0_0000_0000_0000); // 2^700
        y *= f64::from_bits(0x6BB0_0000_0000_0000);
    }
    let (hx, lx) = sq(x);
    let (hy, ly) = sq(y);
    z * (ly + lx + hy + hx).sqrt()
}

// ---------------------------------------------------------------------------
// Method-call form
// ---------------------------------------------------------------------------

/// The functions of this module as `f64` methods, so call sites migrate from
/// the platform libm by a pure rename (`x.exp()` → `x.det_exp()`) with the
/// receiver expression untouched. Argument order matches `std`:
/// `y.det_atan2(x)` is `atan2(y, x)`, like `y.atan2(x)`.
///
/// The `det_` prefix is required, not cosmetic: inherent `f64` methods take
/// precedence over trait methods, so a trait method named `exp` could never
/// be called with method syntax.
pub trait DetMath {
    fn det_sin(self) -> f64;
    fn det_cos(self) -> f64;
    fn det_sin_cos(self) -> (f64, f64);
    fn det_tan(self) -> f64;
    fn det_asin(self) -> f64;
    fn det_acos(self) -> f64;
    fn det_atan(self) -> f64;
    fn det_atan2(self, x: f64) -> f64;
    fn det_sinh(self) -> f64;
    fn det_cosh(self) -> f64;
    fn det_tanh(self) -> f64;
    fn det_atanh(self) -> f64;
    fn det_exp(self) -> f64;
    fn det_exp_m1(self) -> f64;
    fn det_ln(self) -> f64;
    fn det_ln_1p(self) -> f64;
    fn det_log2(self) -> f64;
    fn det_log10(self) -> f64;
    fn det_powf(self, y: f64) -> f64;
    fn det_powi(self, n: i32) -> f64;
    fn det_hypot(self, y: f64) -> f64;
}

impl DetMath for f64 {
    #[inline]
    fn det_sin(self) -> f64 {
        sin(self)
    }
    #[inline]
    fn det_cos(self) -> f64 {
        cos(self)
    }
    #[inline]
    fn det_sin_cos(self) -> (f64, f64) {
        sin_cos(self)
    }
    #[inline]
    fn det_tan(self) -> f64 {
        tan(self)
    }
    #[inline]
    fn det_asin(self) -> f64 {
        asin(self)
    }
    #[inline]
    fn det_acos(self) -> f64 {
        acos(self)
    }
    #[inline]
    fn det_atan(self) -> f64 {
        atan(self)
    }
    #[inline]
    fn det_atan2(self, x: f64) -> f64 {
        atan2(self, x)
    }
    #[inline]
    fn det_sinh(self) -> f64 {
        sinh(self)
    }
    #[inline]
    fn det_cosh(self) -> f64 {
        cosh(self)
    }
    #[inline]
    fn det_tanh(self) -> f64 {
        tanh(self)
    }
    #[inline]
    fn det_atanh(self) -> f64 {
        atanh(self)
    }
    #[inline]
    fn det_exp(self) -> f64 {
        exp(self)
    }
    #[inline]
    fn det_exp_m1(self) -> f64 {
        exp_m1(self)
    }
    #[inline]
    fn det_ln(self) -> f64 {
        ln(self)
    }
    #[inline]
    fn det_ln_1p(self) -> f64 {
        ln_1p(self)
    }
    #[inline]
    fn det_log2(self) -> f64 {
        log2(self)
    }
    #[inline]
    fn det_log10(self) -> f64 {
        log10(self)
    }
    #[inline]
    fn det_powf(self, y: f64) -> f64 {
        pow(self, y)
    }
    #[inline]
    fn det_powi(self, n: i32) -> f64 {
        powi(self, n)
    }
    #[inline]
    fn det_hypot(self, y: f64) -> f64 {
        hypot(self, y)
    }
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

    #[test]
    fn special_values_of_extended_functions() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let pos0 = |v: f64| v == 0.0 && v.is_sign_positive();
        let neg0 = |v: f64| v == 0.0 && v.is_sign_negative();
        // Signed zeros pass through odd functions.
        let odd: [fn(f64) -> f64; 8] = [tan, asin, atan, sinh, tanh, atanh, exp_m1, ln_1p];
        for f in odd {
            assert!(pos0(f(0.0)) && neg0(f(-0.0)));
        }
        assert!(tan(inf).is_nan() && tan(nan).is_nan());
        assert!(asin(1.5).is_nan() && acos(-1.5).is_nan() && asin(nan).is_nan());
        assert_eq!(asin(1.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(asin(-1.0), -std::f64::consts::FRAC_PI_2);
        assert!(pos0(acos(1.0)));
        assert_eq!(acos(-1.0), std::f64::consts::PI);
        assert_eq!(atan(inf), std::f64::consts::FRAC_PI_2);
        assert_eq!(atan(-inf), -std::f64::consts::FRAC_PI_2);
        assert_eq!(atan2(0.0, -1.0), std::f64::consts::PI);
        assert_eq!(atan2(-0.0, -1.0), -std::f64::consts::PI);
        assert_eq!(atan2(1.0, 0.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(atan2(inf, inf), std::f64::consts::FRAC_PI_4);
        assert!(atan2(nan, 1.0).is_nan() && atan2(1.0, nan).is_nan());
        assert_eq!(tanh(inf), 1.0);
        assert_eq!(tanh(-inf), -1.0);
        assert_eq!(tanh(30.0), 1.0);
        assert!(tanh(nan).is_nan());
        assert_eq!(sinh(inf), inf);
        assert_eq!(sinh(-inf), -inf);
        assert_eq!(cosh(-inf), inf);
        assert_eq!(cosh(0.0), 1.0);
        assert_eq!(sinh(1000.0), inf); // overflow
        assert!(sinh(710.0).is_finite()); // expo2 range: exp(710)/2 is finite
        assert_eq!(atanh(1.0), inf);
        assert_eq!(atanh(-1.0), -inf);
        assert!(atanh(1.5).is_nan());
        assert_eq!(exp_m1(inf), inf);
        assert_eq!(exp_m1(-inf), -1.0);
        assert_eq!(exp_m1(1000.0), inf);
        assert_eq!(ln_1p(-1.0), -inf);
        assert!(ln_1p(-2.0).is_nan());
        assert_eq!(ln_1p(inf), inf);
        for k in -1074i32..=1023 {
            // Exact 2^k from bits (powi would overflow computing 1/2^1074).
            let x = if k >= -1022 {
                f64::from_bits(((k + 1023) as u64) << 52)
            } else {
                f64::from_bits(1u64 << (k + 1074))
            };
            assert_eq!(log2(x), k as f64, "log2(2^{k})");
        }
        assert_eq!(log2(0.0), -inf);
        assert!(log2(-1.0).is_nan());
        for k in 0..=22 {
            assert_eq!(log10(10f64.powi(k)), k as f64, "log10(1e{k})");
        }
        assert_eq!(log10(0.0), -inf);
        assert!(log10(-1.0).is_nan());
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(inf, nan), inf); // C99
        assert_eq!(hypot(nan, -inf), inf);
        assert!(hypot(nan, 1.0).is_nan());
        assert_eq!(hypot(1e300, 1e300), 1.4142135623730951e300); // no overflow
        // pow special cases (C99 Annex F).
        assert_eq!(pow(nan, 0.0), 1.0);
        assert_eq!(pow(1.0, nan), 1.0);
        assert_eq!(pow(-1.0, inf), 1.0);
        assert_eq!(pow(0.5, inf), 0.0);
        assert_eq!(pow(2.0, -inf), 0.0);
        assert_eq!(pow(-8.0, 1.0 / 3.0).is_nan(), true);
        assert!(neg0(pow(-0.0, 3.0)));
        assert_eq!(pow(-0.0, -3.0), -inf);
        assert_eq!(pow(10.0, 308.0), 1e308);
        assert_eq!(pow(10.0, 400.0), inf);
        assert_eq!(pow(10.0, -400.0), 0.0);
        assert_eq!(pow(2.0, -1074.0), f64::from_bits(1)); // smallest subnormal
        assert_eq!(pow(4.0, 0.5), 2.0);
    }

    #[test]
    fn extended_functions_agree_with_platform_libm() {
        // Each side is ~< 1 ulp from the true value (tanh/atanh ~2), so a gap
        // beyond 2-3 ulps means a transcription error, not a rounding choice.
        let mut worst: Vec<(&str, u64, f64)> = Vec::new();
        let mut check = |name: &'static str, cap: u64, ours: f64, theirs: f64, x: f64| {
            if ours.is_nan() && theirs.is_nan() {
                return;
            }
            let d = ulps(ours, theirs);
            assert!(d <= cap, "{name}({x:e}): dmath {ours:e} vs libm {theirs:e} ({d} ulps)");
            if let Some(w) = worst.iter_mut().find(|w| w.0 == name) {
                if d > w.1 {
                    *w = (name, d, x);
                }
            } else {
                worst.push((name, d, x));
            }
        };
        for &x in &inputs() {
            if x.abs() < 1e15 {
                check("tan", 2, tan(x), x.tan(), x);
            }
            let u = (x / (1.0 + x.abs())).clamp(-1.0, 1.0); // squashed into [-1, 1]
            check("asin", 2, asin(u), u.asin(), u);
            check("acos", 2, acos(u), u.acos(), u);
            // atanh is NOT compared here: Windows UCRT's atanh is inaccurate
            // near ±1 (mpmath: at x = -0.94815958780866 dmath is 0.41 ulp off,
            // UCRT 8.41 ulp; at -0.94725921928561, 0.03 vs 5.03), so it is no
            // reference. atanh is covered by dmath_ext_check.py (mpmath, max
            // 1.49 ulp) and verification/musl_bitcompare (bit-identical).
            check("atan", 2, atan(x), x.atan(), x);
            check("atan2", 2, atan2(x, u), x.atan2(u), x);
            let h = x.clamp(-720.0, 720.0);
            check("sinh", 2, sinh(h), h.sinh(), h);
            check("cosh", 2, cosh(h), h.cosh(), h);
            check("tanh", 3, tanh(h), h.tanh(), h);
            check("exp_m1", 2, exp_m1(h), h.exp_m1(), h);
            if x > -1.0 {
                check("ln_1p", 2, ln_1p(x), x.ln_1p(), x);
            }
            if x > 0.0 {
                check("log2", 2, log2(x), x.log2(), x);
                check("log10", 2, log10(x), x.log10(), x);
                // pow over a range that neither overflows nor underflows.
                let b = x.clamp(1e-3, 1e3);
                for e in [0.37, -2.5, 7.0, 31.3, -64.0] {
                    check("pow", 2, pow(b, e), b.powf(e), b);
                }
            }
            check("hypot", 2, hypot(x, u), x.hypot(u), x);
        }
        // Keep the observed maxima visible in `cargo test -- --nocapture`.
        eprintln!("max ulp gap vs platform libm: {worst:?}");
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

    /// Golden hash for the functions added for the runtime migration.
    /// Kept separate from [`GOLDEN_HASH`] so a change is attributable.
    #[test]
    fn golden_hash_extended_is_platform_independent() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: f64| {
            for byte in v.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for &x in &inputs() {
            let u = (x / (1.0 + x.abs())).clamp(-1.0, 1.0);
            let hx = x.clamp(-720.0, 720.0);
            mix(tan(x));
            mix(asin(u));
            mix(acos(u));
            mix(atan(x));
            mix(atan2(x, u));
            mix(sinh(hx));
            mix(cosh(hx));
            mix(tanh(hx));
            mix(atanh(u));
            mix(exp_m1(hx));
            mix(ln_1p(x.abs()));
            mix(log2(x.abs()));
            mix(log10(x.abs()));
            mix(hypot(x, u));
            mix(pow(x.abs().clamp(1e-3, 1e3), 31.3));
        }
        assert_eq!(h, GOLDEN_HASH_EXTENDED, "extended golden hash changed: got {:#018x}", h);
    }

    // Regenerate only after re-verification (mpmath: dmath_check*.py; musl
    // bit-compare: verification/musl_bitcompare/).
    // 0xa92df4d4e1fb935e -> 0x8ae8ded20ae0ffde (2026-09-26): `pow` replaced
    // by fdlibm e_pow.c (< 1 ulp) instead of exp(y·ln x); sin/cos/exp/ln
    // bits are unchanged.
    const GOLDEN_HASH: u64 = 0x8ae8_ded2_0ae0_ffde;
    const GOLDEN_HASH_EXTENDED: u64 = 0xa7a1_3cc9_5309_466a;
}
