//! BinnedAccumulator — Deterministic, order-invariant floating-point summation.
//!
//! # Determinism Contract
//!
//! [`BinnedAccumulatorF64`] returns the **exact** real sum of its finite
//! inputs, rounded **once** to the nearest f64 (ties to even).  Its result is
//! therefore a function of the *multiset* of inputs only, and is bit-identical
//! regardless of:
//! - Addition order (commutative + associative within the accumulator)
//! - Thread count
//! - Chunk size
//! - Merge order / merge-tree shape
//! - Scheduling order and CPU core topology
//!
//! # Architecture (f64): fixed-point superaccumulator
//!
//! (Kulisch-style long accumulator with deferred carries, as in ExBLAS /
//! Demmel–Nguyen; the type keeps its historical "Binned" name for API
//! stability.)
//!
//! Every finite f64 is an integer multiple of 2^-1074 (the smallest
//! subnormal): `x = M · 2^-1074` with `M` a signed integer, `|M| < 2^2098`.
//! The accumulator stores one signed integer
//!
//! ```text
//!     S = Σ_k limbs[k] · 2^(32·k),   k = 0..F64_LIMBS
//! ```
//!
//! in 68 `i64` limbs of radix 2^32 (2176 bits: 2098 for the f64 range plus 64
//! bits of headroom for up to 2^64 summands, plus the sign carried by the
//! signed top limb).
//!
//! - `add(x)` splits `M`'s 53-bit significand into three 32-bit digits at
//!   the right offset and adds (or subtracts) them to three limbs.  This is
//!   integer arithmetic: exact, no rounding.
//! - Carries are **deferred**: each `add` changes any limb by less than
//!   2^32, so after `p` un-normalised contributions `|limb| < (p+1)·2^32`.
//!   `pending` counts them and carry propagation (`normalize`) runs before
//!   `p` can reach 2^30, so no `i64` limb can overflow (2^30·2^32 ≪ 2^63).
//!   `normalize` moves `floor(limb / 2^32)` into the next limb, which leaves
//!   `S` unchanged exactly.
//! - `merge(other)` adds `other`'s limbs element-wise: `S ← S + S_other`,
//!   again exact integer addition (normalising first if the pending bound
//!   would be exceeded).
//! - `finalize()` copies the limbs, normalises to the canonical form
//!   (limbs `0..N-1` in `[0, 2^32)`, signed top limb — unique for a given
//!   `S`), takes sign and magnitude, and rounds `|S| · 2^-1074` to the
//!   nearest f64 with round / sticky bits, ties to even.
//!
//! **Why this is order-invariant.**  After any sequence of `add`/`merge`/
//! `normalize` calls over a multiset `X` of finite inputs, the represented
//! integer is exactly `S = Σ_{x∈X} M(x)`: every step is an exact integer
//! addition (no limb overflows, by the `pending` bound), and integer addition
//! is commutative and associative.  `finalize` is a pure function of `S`
//! (via its unique canonical form) and of the special-value flags (an OR and
//! two counts, also commutative and associative).  Hence the result depends
//! only on the multiset — not on order, chunking or merge tree — and it is the
//! correctly rounded exact sum.  (Contrast the previous design, plain IEEE
//! `+=` inside each exponent bin: that rounds, so `fl(fl(a+b)+c) ≠
//! fl(fl(c+b)+a)` in general — bug P5.)
//!
//! # Memory Rules
//!
//! - **Stack-allocated only.** No `Vec`, `Box`, or heap allocation.
//! - **Fixed-size.** 68 limbs (544 bytes) for f64.
//! - **Merge is allocation-free.**
//!
//! # IEEE-754 Special Value Policy (f64)
//!
//! | Inputs                        | Result                                   |
//! |-------------------------------|------------------------------------------|
//! | any NaN                       | NaN (`f64::NAN`)                         |
//! | +Inf and -Inf (no NaN)        | NaN (indeterminate)                      |
//! | +Inf only (no NaN)            | +Inf (finite inputs ignored)             |
//! | -Inf only (no NaN)            | -Inf (finite inputs ignored)             |
//! | finite only, exact sum == 0   | +0.0 (also `[-0.0]`, `[-0.0, -0.0]`, empty) |
//! | finite only, otherwise        | exact sum rounded to nearest, ties-to-even |
//! | exact sum ≥ 2^1024 − 2^970    | ±Inf (IEEE round-to-nearest overflow)    |
//!
//! Overflow is decided on the **exact** sum, so an intermediate overflow that
//! cancels (`[MAX, MAX, -MAX]`) yields `MAX` in every order.  Subnormals are
//! never flushed.  This matches the previous policy except that the previous
//! code returned `Inf`/`NaN` for such cancelling overflows depending on order.
//!
//! # Platform Control
//!
//! - **FMA / SIMD / rounding mode:** irrelevant to the f64 path — it uses only
//!   integer arithmetic plus bit manipulation of the IEEE encoding.
//! - **FTZ/DAZ:** Not enabled. Subnormals are preserved.
//!
//! # Supported Types
//!
//! - `f64`: exact fixed-point superaccumulator (above).
//! - `f32`: 256 exponent bins accumulated in f64 (see [`BinnedAccumulatorF32`]).

use cjc_repro::dmath::DetMath;
// ---------------------------------------------------------------------------
// f64 BinnedAccumulator (exact fixed-point superaccumulator)
// ---------------------------------------------------------------------------

/// Number of 32-bit-radix limbs. Bit `b` of the fixed-point integer has
/// weight `2^(b - 1074)`; 68 × 32 = 2176 bits ≥ 2098 (f64 range) + 64
/// (headroom for 2^64 summands) + sign.
const F64_LIMBS: usize = 68;

/// Carry propagation must run before this many un-normalised contributions
/// accumulate in any limb (each contributes < 2^32; i64 holds < 2^63).
const F64_PENDING_LIMIT: u32 = 1 << 30;

const F64_FRAC_MASK: u64 = (1u64 << 52) - 1;

/// Deterministic, exact superaccumulator for f64.
///
/// Stack-allocated, fixed-size. No heap allocation on any path.
/// See the module docs for the order-invariance argument.
///
/// # Usage
///
/// ```ignore
/// let mut acc = BinnedAccumulatorF64::new();
/// for &x in data {
///     acc.add(x);
/// }
/// let result = acc.finalize();
/// ```
#[derive(Clone)]
pub struct BinnedAccumulatorF64 {
    /// Fixed-point integer `S = Σ limbs[k]·2^(32k)` in units of 2^-1074.
    /// Between normalisations limbs may be any i64 within the pending bound.
    limbs: [i64; F64_LIMBS],
    /// Contributions added to the limbs since the last carry propagation.
    pending: u32,
    /// Positive infinity count.
    pos_inf_count: u32,
    /// Negative infinity count.
    neg_inf_count: u32,
    /// Whether any NaN was observed.
    has_nan: bool,
    /// Total number of values added.
    total_count: u64,
}

/// Propagate carries so limbs `0..N-1` lie in `[0, 2^32)`; the top limb keeps
/// the (signed) remainder. Preserves `Σ limbs[k]·2^(32k)` exactly.
#[inline]
fn f64_normalize_limbs(limbs: &mut [i64; F64_LIMBS]) {
    for k in 0..F64_LIMBS - 1 {
        let carry = limbs[k] >> 32; // arithmetic shift = floor(limb / 2^32)
        limbs[k] -= carry << 32;
        limbs[k + 1] += carry;
    }
}

/// Bits `[lo, lo + n)` (n ≤ 53) of a canonical non-negative limb array.
#[inline]
fn f64_extract_bits(limbs: &[i64; F64_LIMBS], lo: usize, n: u32) -> u64 {
    let idx = lo / 32;
    let off = (lo % 32) as u32;
    let get = |i: usize| -> u128 {
        if i < F64_LIMBS {
            limbs[i] as u64 as u128
        } else {
            0
        }
    };
    let w = get(idx) | (get(idx + 1) << 32) | (get(idx + 2) << 64);
    ((w >> off) as u64) & ((1u64 << n) - 1)
}

/// Whether any bit strictly below position `pos` is set (canonical limbs).
#[inline]
fn f64_any_below(limbs: &[i64; F64_LIMBS], pos: usize) -> bool {
    let idx = pos / 32;
    let off = (pos % 32) as u32;
    if limbs[..idx].iter().any(|&l| l != 0) {
        return true;
    }
    off > 0 && (limbs[idx] as u64 & ((1u64 << off) - 1)) != 0
}

impl BinnedAccumulatorF64 {
    /// Create a new, empty accumulator.
    #[inline]
    pub fn new() -> Self {
        BinnedAccumulatorF64 {
            limbs: [0; F64_LIMBS],
            pending: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            has_nan: false,
            total_count: 0,
        }
    }

    /// Add a single f64 value to the accumulator (exactly).
    ///
    /// Special values are handled according to the policy:
    /// - NaN: sets `has_nan` flag, value is not accumulated.
    /// - ±Inf: counted separately, not accumulated.
    /// - ±0.0: contributes nothing.
    /// - Subnormals: accumulated exactly (no flush).
    #[inline]
    pub fn add(&mut self, value: f64) {
        self.total_count += 1;

        let bits = value.to_bits();
        let biased_exp = ((bits >> 52) & 0x7FF) as u32;
        let frac = bits & F64_FRAC_MASK;

        if biased_exp == 0x7FF {
            if frac != 0 {
                self.has_nan = true;
            } else if bits >> 63 == 0 {
                self.pos_inf_count += 1;
            } else {
                self.neg_inf_count += 1;
            }
            return;
        }

        // |value| = mant · 2^(pos - 1074).
        let (mant, pos) = if biased_exp == 0 {
            (frac, 0u32)
        } else {
            (frac | (1u64 << 52), biased_exp - 1)
        };
        if mant == 0 {
            return; // ±0.0
        }

        let k = (pos >> 5) as usize; // pos ≤ 2045 → k ≤ 63, k + 2 ≤ 65
        let w = (mant as u128) << (pos & 31); // < 2^84
        let d0 = (w as u64 & 0xFFFF_FFFF) as i64;
        let d1 = ((w >> 32) as u64 & 0xFFFF_FFFF) as i64;
        let d2 = (w >> 64) as i64;
        if bits >> 63 == 0 {
            self.limbs[k] += d0;
            self.limbs[k + 1] += d1;
            self.limbs[k + 2] += d2;
        } else {
            self.limbs[k] -= d0;
            self.limbs[k + 1] -= d1;
            self.limbs[k + 2] -= d2;
        }

        self.pending += 1;
        if self.pending >= F64_PENDING_LIMIT {
            self.normalize();
        }
    }

    /// Add a slice of f64 values.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Carry-propagate the limbs. Does not change the represented value.
    #[inline]
    fn normalize(&mut self) {
        f64_normalize_limbs(&mut self.limbs);
        self.pending = 0;
    }

    /// Deterministic, exact merge of two accumulators: `S ← S + S_other`.
    ///
    /// Integer addition of the fixed-point limbs, so the merge is exactly
    /// commutative and associative; any merge tree over any chunking of the
    /// inputs yields the same `S` and hence the same `finalize()` bits.
    ///
    /// No heap allocation.
    #[inline]
    pub fn merge(&mut self, other: &BinnedAccumulatorF64) {
        // After adding, |limb| < (p_self + 1 + p_other + 1) · 2^32.
        if self.pending as u64 + other.pending as u64 + 1 >= F64_PENDING_LIMIT as u64 {
            self.normalize();
            let mut o = other.limbs;
            f64_normalize_limbs(&mut o);
            for k in 0..F64_LIMBS {
                self.limbs[k] += o[k];
            }
            self.pending = 1;
        } else {
            for k in 0..F64_LIMBS {
                self.limbs[k] += other.limbs[k];
            }
            self.pending += other.pending + 1;
        }
        self.pos_inf_count += other.pos_inf_count;
        self.neg_inf_count += other.neg_inf_count;
        self.has_nan = self.has_nan || other.has_nan;
        self.total_count += other.total_count;
    }

    /// Finalize: the exact sum rounded once to nearest f64 (ties to even).
    ///
    /// # Special value semantics
    ///
    /// - If `has_nan` is true, returns NaN.
    /// - If both +Inf and -Inf were seen, returns NaN (indeterminate).
    /// - If only +Inf was seen, returns +Inf; if only -Inf, returns -Inf.
    /// - An exactly-zero finite sum returns +0.0.
    /// - A finite sum whose rounding overflows returns ±Inf.
    #[inline]
    pub fn finalize(&self) -> f64 {
        if self.has_nan {
            return f64::NAN;
        }
        if self.pos_inf_count > 0 && self.neg_inf_count > 0 {
            return f64::NAN; // +Inf + -Inf is indeterminate
        }
        if self.pos_inf_count > 0 {
            return f64::INFINITY;
        }
        if self.neg_inf_count > 0 {
            return f64::NEG_INFINITY;
        }

        // Canonical form, then sign/magnitude.
        let mut l = self.limbs;
        f64_normalize_limbs(&mut l);
        let negative = l[F64_LIMBS - 1] < 0;
        if negative {
            for x in l.iter_mut() {
                *x = -*x;
            }
            f64_normalize_limbs(&mut l);
        }
        let hi = match l.iter().rposition(|&x| x != 0) {
            Some(i) => i,
            None => return 0.0,
        };
        // Highest set bit of |S|.
        let h = hi * 32 + (63 - (l[hi] as u64).leading_zeros()) as usize;
        let sign = if negative { 1u64 << 63 } else { 0 };

        let mag = if h <= 52 {
            // |S| < 2^53: subnormal or smallest binade, exactly representable;
            // its bit pattern is |S| itself.
            f64_extract_bits(&l, 0, 53)
        } else {
            let lo = h - 52;
            let mut mant = f64_extract_bits(&l, lo, 53);
            let round = f64_extract_bits(&l, lo - 1, 1) == 1;
            let sticky = f64_any_below(&l, lo - 1);
            if round && (sticky || mant & 1 == 1) {
                mant += 1;
            }
            // value = mant · 2^(lo - 1074) = mant · 2^(E - 1075) with E = lo + 1.
            let mut e = (lo + 1) as u64;
            if mant == 1u64 << 53 {
                mant >>= 1;
                e += 1;
            }
            if e >= 0x7FF {
                return f64::from_bits(sign | 0x7FF0_0000_0000_0000); // ±Inf
            }
            (e << 52) | (mant & F64_FRAC_MASK)
        };
        f64::from_bits(sign | mag)
    }

    /// Total number of values added.
    #[inline]
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Whether any NaN was encountered.
    #[inline]
    pub fn has_nan(&self) -> bool {
        self.has_nan
    }
}

impl Default for BinnedAccumulatorF64 {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BinnedAccumulatorF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinnedAccumulatorF64")
            .field("total_count", &self.total_count)
            .field("has_nan", &self.has_nan)
            .field("pos_inf_count", &self.pos_inf_count)
            .field("neg_inf_count", &self.neg_inf_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// f32 BinnedAccumulator
// ---------------------------------------------------------------------------

/// Number of exponent bins for f32 (2^8 = 256 distinct biased exponents).
const F32_BIN_COUNT: usize = 256;

/// Deterministic superaccumulator for f32.
///
/// Stack-allocated, fixed-size. No heap allocation on any path.
/// Internally accumulates in f64 for precision, converts to f32 on finalize.
///
/// **Order-invariance scope:** f32 values sharing a biased exponent are
/// multiples of a common power of two and need 24 + log2(n) bits, so each
/// f64 bin sum is *exact* (hence order-free) while a bin holds fewer than
/// 2^29 values; the final fixed-order fold is then a deterministic function
/// of the bins.  The result is not correctly rounded (fold in f64, then
/// narrowed to f32), and this type is not an exact superaccumulator like
/// [`BinnedAccumulatorF64`].
#[derive(Clone)]
pub struct BinnedAccumulatorF32 {
    /// One bin per biased exponent (f32 has 8-bit exponent field, 256 values).
    /// Accumulated in f64 to avoid precision loss during binned reduction.
    bins: [f64; F32_BIN_COUNT],
    /// Compensation terms per bin (two-sum merge errors).
    comp: [f64; F32_BIN_COUNT],
    /// Count of values per bin.
    counts: [u32; F32_BIN_COUNT],
    /// Positive infinity count.
    pos_inf_count: u32,
    /// Negative infinity count.
    neg_inf_count: u32,
    /// Whether any NaN was observed.
    has_nan: bool,
    /// Total number of values added.
    total_count: u64,
}

impl BinnedAccumulatorF32 {
    /// Create a new, empty accumulator.
    #[inline]
    pub fn new() -> Self {
        BinnedAccumulatorF32 {
            bins: [0.0; F32_BIN_COUNT],
            comp: [0.0; F32_BIN_COUNT],
            counts: [0; F32_BIN_COUNT],
            pos_inf_count: 0,
            neg_inf_count: 0,
            has_nan: false,
            total_count: 0,
        }
    }

    /// Add a single f32 value.
    #[inline]
    pub fn add(&mut self, value: f32) {
        self.total_count += 1;

        if value.is_nan() {
            self.has_nan = true;
            return;
        }
        if value.is_infinite() {
            if value > 0.0 {
                self.pos_inf_count += 1;
            } else {
                self.neg_inf_count += 1;
            }
            return;
        }

        let bits = value.to_bits();
        let biased_exp = ((bits >> 23) & 0xFF) as usize;

        // Promote to f64 for accumulation precision, then add directly.
        self.bins[biased_exp] += value as f64;
        self.counts[biased_exp] += 1;
    }

    /// Add a slice of f32 values.
    #[inline]
    pub fn add_slice(&mut self, values: &[f32]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Deterministic merge. Same commutativity/associativity guarantees as f64.
    /// Uses Knuth 2Sum error-free transformation for merge-order invariance.
    #[inline]
    pub fn merge(&mut self, other: &BinnedAccumulatorF32) {
        for i in 0..F32_BIN_COUNT {
            let a = self.bins[i];
            let b = other.bins[i];
            let s = a + b;
            let v = s - a;
            let e = (a - (s - v)) + (b - v);
            self.bins[i] = s;
            self.comp[i] += e + other.comp[i];
            self.counts[i] += other.counts[i];
        }
        self.pos_inf_count += other.pos_inf_count;
        self.neg_inf_count += other.neg_inf_count;
        self.has_nan = self.has_nan || other.has_nan;
        self.total_count += other.total_count;
    }

    /// Finalize to f32 result.
    #[inline]
    pub fn finalize(&self) -> f32 {
        if self.has_nan {
            return f32::NAN;
        }
        if self.pos_inf_count > 0 && self.neg_inf_count > 0 {
            return f32::NAN;
        }
        if self.pos_inf_count > 0 {
            return f32::INFINITY;
        }
        if self.neg_inf_count > 0 {
            return f32::NEG_INFINITY;
        }

        let mut sum = 0.0f64;
        let mut kahan_comp = 0.0f64;
        for i in 0..F32_BIN_COUNT {
            if self.counts[i] == 0 {
                continue;
            }
            let val = self.bins[i] + self.comp[i];
            let y = val - kahan_comp;
            let t = sum + y;
            kahan_comp = (t - sum) - y;
            sum = t;
        }

        sum as f32
    }

    /// Total count of values added.
    #[inline]
    pub fn count(&self) -> u64 {
        self.total_count
    }
}

impl Default for BinnedAccumulatorF32 {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BinnedAccumulatorF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let active_bins = self.counts.iter().filter(|&&c| c > 0).count();
        f.debug_struct("BinnedAccumulatorF32")
            .field("total_count", &self.total_count)
            .field("active_bins", &active_bins)
            .field("has_nan", &self.has_nan)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience functions (parallel to kahan_sum_f64 / kahan_sum_f32 in cjc-repro)
// ---------------------------------------------------------------------------

/// Deterministic, order-invariant summation of f64 values using binned accumulation.
///
/// Bit-identical result regardless of input order.
#[inline]
pub fn binned_sum_f64(values: &[f64]) -> f64 {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add_slice(values);
    acc.finalize()
}

/// Fused matmul + matmul kernel: computes `(A @ B) @ C` in a single pass.
///
/// Inputs are flat row-major slices: `a` length `m*n` (row-major `[m, n]`),
/// `b` length `n*k` (row-major `[n, k]`), `c` length `k*p` (row-major `[k, p]`).
///
/// Math is **left-associative** — `(A @ B) @ C`, matching the unfused chain
/// `matmul(matmul(A, B), C)`. The other associativity `A @ (B @ C)` gives a
/// different bit-pattern in IEEE-754; this kernel picks the same order as
/// the unfused chain by construction.
///
/// Equivalent to `matmul(matmul(A, B), C)` and bit-identical when both
/// matmul ops take the sequential path (every dim < 64).
///
/// **Allocation:** the named intermediate Tensor wrapper is gone (we still
/// hold a `Vec<f64>` for the [m, k] partial; that's the cheapest
/// allocation savings). Real fusion-with-tiling would skip even that —
/// future work.
///
/// **Determinism:** per-element Kahan in both matmul steps. Order matches
/// `matmul_sequential` exactly.
#[inline]
pub fn fused_matmul_matmul_kernel(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    m: usize,
    n: usize,
    k: usize,
    p: usize,
) -> Vec<f64> {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), k * p);

    // Step 1: intermediate[i, q] = Σₗ a[i, l] * b[l, q]   shape [m, k]
    let mut intermediate = vec![0.0f64; m * k];
    for i in 0..m {
        for q in 0..k {
            let mut acc = cjc_repro::KahanAccumulatorF64::new();
            for l in 0..n {
                acc.add(a[i * n + l] * b[l * k + q]);
            }
            intermediate[i * k + q] = acc.finalize();
        }
    }

    // Step 2: result[i, j] = Σ_q intermediate[i, q] * c[q, j]   shape [m, p]
    let mut result = vec![0.0f64; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut acc = cjc_repro::KahanAccumulatorF64::new();
            for q in 0..k {
                acc.add(intermediate[i * k + q] * c[q * p + j]);
            }
            result[i * p + j] = acc.finalize();
        }
    }
    result
}

/// Fused matmul + norm kernel: computes `||A @ W||_p` in a single pass.
///
/// Inputs are flat row-major slices: `a` length `m*k` (row-major `[m, k]`),
/// `w` length `k*n` (row-major `[k, n]`).
///
/// Equivalent to `norm(matmul(A, W), ord)` and bit-identical to the unfused
/// chain when matmul takes the sequential path (every dim < 64 in
/// [`crate::tensor::Tensor::matmul`]).
///
/// **`ord` semantics:** matches the `norm` builtin —
///   - `1` → Σᵢⱼ |h[i,j]| (L1 norm, no final root)
///   - `2` → √(Σᵢⱼ h[i,j]²) (L2 / Frobenius norm)
///   - `p` → (Σᵢⱼ |h[i,j]|^p)^(1/p) (general L_p)
///
/// **Math:** for each output element `h[i,j]`, accumulate `a[i,k] · w[k,j]`
/// over the inner dim with a Kahan accumulator (matches `matmul_sequential`),
/// transform via the ord-specific function, then accumulate across all
/// `(i, j)` with a binned accumulator (matches `norm`).
///
/// **Allocation:** stack-only — no `Vec` for the `[m, n]` intermediate.
///
/// **Determinism:** order-invariant binned reduction over (i, j) matches
/// `binned_sum_f64`; the per-element Kahan order matches `matmul_sequential`.
#[inline]
pub fn fused_matmul_norm_kernel(
    a: &[f64],
    w: &[f64],
    ord: i64,
    m: usize,
    k: usize,
    n: usize,
) -> f64 {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(w.len(), k * n);

    let mut outer = BinnedAccumulatorF64::new();
    for i in 0..m {
        for j in 0..n {
            let mut inner = cjc_repro::KahanAccumulatorF64::new();
            for kk in 0..k {
                inner.add(a[i * k + kk] * w[kk * n + j]);
            }
            let h_ij = inner.finalize();
            // Apply ord-specific transform (matches `norm` builtin exactly).
            let transformed = match ord {
                1 => h_ij.abs(),
                2 => h_ij * h_ij,
                p => h_ij.abs().det_powf(p as f64),
            };
            outer.add(transformed);
        }
    }
    let raw_sum = outer.finalize();
    // Final root (matches `norm` builtin).
    match ord {
        1 => raw_sum,
        2 => raw_sum.sqrt(),
        p => raw_sum.det_powf(1.0 / (p as f64)),
    }
}

/// Fused matmul + dot kernel: computes aᵀ · W · v in a single pass.
///
/// Inputs are flat row-major slices: `a` length `m`, `w` length `m*n`
/// (row-major `[m, n]`), `v` length `n`.
///
/// Equivalent to `dot(matmul(a_as_row_vec, w).flatten(), v)` and bit-identical
/// to the unfused chain when matmul takes the sequential path (every dim < 64
/// in [`crate::tensor::Tensor::matmul`]).
///
/// **Math:** for each output column `j`, accumulate `a[i] · W[i, j]` in
/// column-major order with a Kahan accumulator (matching matmul_sequential),
/// then accumulate `(intermediate[j] · v[j])` across `j` with a binned
/// accumulator (matching `dot`).
///
/// **Allocation:** stack-only — no `Vec` for the `[1, n]` intermediate. This
/// is the win that makes Phase 3.5 fusion worthwhile.
///
/// **Determinism:** order-invariant binned reduction over the j-axis matches
/// `binned_sum_f64`; the per-column Kahan order matches `matmul_sequential`.
#[inline]
pub fn fused_matmul_dot_kernel(a: &[f64], w: &[f64], v: &[f64], m: usize, n: usize) -> f64 {
    debug_assert_eq!(a.len(), m);
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(v.len(), n);

    let mut outer = BinnedAccumulatorF64::new();
    for j in 0..n {
        let mut inner = cjc_repro::KahanAccumulatorF64::new();
        for i in 0..m {
            inner.add(a[i] * w[i * n + j]);
        }
        outer.add(inner.finalize() * v[j]);
    }
    outer.finalize()
}

/// Deterministic, order-invariant summation of f32 values using binned accumulation.
#[inline]
pub fn binned_sum_f32(values: &[f32]) -> f32 {
    let mut acc = BinnedAccumulatorF32::new();
    acc.add_slice(values);
    acc.finalize()
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_accumulator() {
        let acc = BinnedAccumulatorF64::new();
        assert_eq!(acc.finalize(), 0.0);
        assert_eq!(acc.count(), 0);
    }

    #[test]
    fn test_single_value() {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add(42.0);
        assert_eq!(acc.finalize(), 42.0);
    }

    #[test]
    fn test_simple_sum() {
        let mut acc = BinnedAccumulatorF64::new();
        for i in 1..=10 {
            acc.add(i as f64);
        }
        assert_eq!(acc.finalize(), 55.0);
    }

    #[test]
    fn test_order_invariance() {
        let values = vec![1e16, 1.0, -1e16, 0.5, -0.5, 1e-16, -1e-16];

        // Forward order
        let mut acc1 = BinnedAccumulatorF64::new();
        acc1.add_slice(&values);

        // Reverse order
        let rev: Vec<f64> = values.iter().rev().copied().collect();
        let mut acc2 = BinnedAccumulatorF64::new();
        acc2.add_slice(&rev);

        assert_eq!(acc1.finalize().to_bits(), acc2.finalize().to_bits());
    }

    #[test]
    fn test_merge_commutativity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let mut acc_a = BinnedAccumulatorF64::new();
        acc_a.add_slice(&a);
        let mut acc_b = BinnedAccumulatorF64::new();
        acc_b.add_slice(&b);

        // a.merge(b)
        let mut merged_ab = acc_a.clone();
        merged_ab.merge(&acc_b);

        // b.merge(a)
        let mut merged_ba = acc_b.clone();
        merged_ba.merge(&acc_a);

        assert_eq!(merged_ab.finalize().to_bits(), merged_ba.finalize().to_bits());
        assert_eq!(merged_ab.finalize(), 21.0);
    }

    #[test]
    fn test_nan_propagation() {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add(1.0);
        acc.add(f64::NAN);
        acc.add(2.0);
        assert!(acc.finalize().is_nan());
        assert!(acc.has_nan());
    }

    #[test]
    fn test_inf_handling() {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add(f64::INFINITY);
        assert_eq!(acc.finalize(), f64::INFINITY);

        let mut acc2 = BinnedAccumulatorF64::new();
        acc2.add(f64::NEG_INFINITY);
        assert_eq!(acc2.finalize(), f64::NEG_INFINITY);

        // +Inf + -Inf = NaN
        let mut acc3 = BinnedAccumulatorF64::new();
        acc3.add(f64::INFINITY);
        acc3.add(f64::NEG_INFINITY);
        assert!(acc3.finalize().is_nan());
    }

    #[test]
    fn test_subnormal_preservation() {
        let subnormal = 5e-324_f64; // Smallest positive subnormal
        let mut acc = BinnedAccumulatorF64::new();
        acc.add(subnormal);
        // Subnormal goes to bin 0 and is preserved.
        assert_eq!(acc.finalize(), subnormal);
    }

    #[test]
    fn test_signed_zero() {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add(0.0);
        acc.add(-0.0);
        // Both zeros accumulate normally; result should be 0.0.
        let result = acc.finalize();
        assert_eq!(result, 0.0);
    }

    // -- P5: exact superaccumulator -------------------------------------

    #[test]
    fn test_p5_same_bin_order_invariance() {
        let b = 1.0000000000000002_f64; // 1 + 2^-52
        let exact = 3.0000000000000004_f64.to_bits(); // 3 + 2^-51
        assert_eq!(binned_sum_f64(&[1.0, b, b]).to_bits(), exact);
        assert_eq!(binned_sum_f64(&[b, b, 1.0]).to_bits(), exact);
        assert_eq!(binned_sum_f64(&[b, 1.0, b]).to_bits(), exact);
    }

    #[test]
    fn test_negative_and_subnormal_exact() {
        assert_eq!(binned_sum_f64(&[-1.0, -2.0, -0.5]), -3.5);
        assert_eq!(binned_sum_f64(&[5e-324, 5e-324]).to_bits(), 2);
        assert_eq!(binned_sum_f64(&[-5e-324]).to_bits(), (-5e-324f64).to_bits());
        assert_eq!(binned_sum_f64(&[f64::MAX, -f64::MAX, 1e-300]), 1e-300);
        assert_eq!(binned_sum_f64(&[f64::MAX, f64::MAX]), f64::INFINITY);
        assert_eq!(binned_sum_f64(&[-f64::MAX, -f64::MAX]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_signed_zero_result_is_positive() {
        assert_eq!(binned_sum_f64(&[-0.0]).to_bits(), 0);
        assert_eq!(binned_sum_f64(&[-0.0, -0.0]).to_bits(), 0);
        assert_eq!(binned_sum_f64(&[2.5, -2.5]).to_bits(), 0);
    }

    /// Force the deferred-carry paths (add-triggered and merge-triggered
    /// normalisation) and check they do not change the result.
    #[test]
    fn test_pending_limit_normalisation_paths() {
        let xs: Vec<f64> = (0..5000)
            .map(|i| ((i * 7919) % 1000) as f64 * 1.37e-3 - 0.61 + (i as f64) * 1e-9)
            .collect();
        let reference = binned_sum_f64(&xs).to_bits();

        // add() crossing the limit.
        let mut a = BinnedAccumulatorF64::new();
        a.add_slice(&xs[..2500]);
        a.pending = F64_PENDING_LIMIT - 3;
        a.add_slice(&xs[2500..]);
        assert!(a.pending < 2500, "normalisation did not run");
        assert_eq!(a.finalize().to_bits(), reference);

        // merge() crossing the limit, both directions.
        let mut l = BinnedAccumulatorF64::new();
        l.add_slice(&xs[..1234]);
        let mut r = BinnedAccumulatorF64::new();
        r.add_slice(&xs[1234..]);
        let mut lr = l.clone();
        lr.pending = F64_PENDING_LIMIT / 2;
        let mut r_hi = r.clone();
        r_hi.pending = F64_PENDING_LIMIT / 2;
        lr.merge(&r_hi);
        assert_eq!(lr.pending, 1);
        assert_eq!(lr.finalize().to_bits(), reference);
        let mut rl = r.clone();
        rl.merge(&l);
        assert_eq!(rl.finalize().to_bits(), reference);
    }

    #[test]
    fn test_size_is_small() {
        // 68 limbs + flags; the old layout was ~40 KiB.
        assert!(std::mem::size_of::<BinnedAccumulatorF64>() <= 600);
    }

    #[test]
    fn test_catastrophic_cancellation() {
        // 1e16 + 1.0 - 1e16 should be 1.0 (naive sum loses this).
        let mut acc = BinnedAccumulatorF64::new();
        acc.add(1e16);
        acc.add(1.0);
        acc.add(-1e16);
        let result = acc.finalize();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_many_small_values() {
        let mut acc = BinnedAccumulatorF64::new();
        for _ in 0..10000 {
            acc.add(0.0001);
        }
        let result = acc.finalize();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f32_accumulator() {
        let mut acc = BinnedAccumulatorF32::new();
        for i in 1..=100 {
            acc.add(i as f32);
        }
        // 100*101/2 = 5050
        assert_eq!(acc.finalize(), 5050.0);
    }

    #[test]
    fn test_f32_order_invariance() {
        let values: Vec<f32> = (1..=1000).map(|i| i as f32 * 0.001).collect();
        let reversed: Vec<f32> = values.iter().rev().copied().collect();

        let r1 = binned_sum_f32(&values);
        let r2 = binned_sum_f32(&reversed);
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    #[test]
    fn test_convenience_function() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(binned_sum_f64(&values), 5050.0);
    }

    #[test]
    fn test_merge_associativity() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let c = vec![5.0, 6.0];

        let mut acc_a = BinnedAccumulatorF64::new();
        acc_a.add_slice(&a);
        let mut acc_b = BinnedAccumulatorF64::new();
        acc_b.add_slice(&b);
        let mut acc_c = BinnedAccumulatorF64::new();
        acc_c.add_slice(&c);

        // (a + b) + c
        let mut ab = acc_a.clone();
        ab.merge(&acc_b);
        let mut abc1 = ab;
        abc1.merge(&acc_c);

        // a + (b + c)
        let mut bc = acc_b.clone();
        bc.merge(&acc_c);
        let mut abc2 = acc_a.clone();
        abc2.merge(&bc);

        assert_eq!(abc1.finalize().to_bits(), abc2.finalize().to_bits());
        assert_eq!(abc1.finalize(), 21.0);
    }

    #[test]
    fn test_deterministic_across_chunk_sizes() {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.7) - 350.0).collect();

        // Single accumulator
        let r1 = binned_sum_f64(&values);

        // Chunked into 7s, then merged
        let mut final_acc = BinnedAccumulatorF64::new();
        for chunk in values.chunks(7) {
            let mut chunk_acc = BinnedAccumulatorF64::new();
            chunk_acc.add_slice(chunk);
            final_acc.merge(&chunk_acc);
        }
        let r2 = final_acc.finalize();

        // Chunked into 13s, then merged
        let mut final_acc2 = BinnedAccumulatorF64::new();
        for chunk in values.chunks(13) {
            let mut chunk_acc = BinnedAccumulatorF64::new();
            chunk_acc.add_slice(chunk);
            final_acc2.merge(&chunk_acc);
        }
        let r3 = final_acc2.finalize();

        // Single-add vs chunk-merge may differ by a few ULPs because the
        // within-bin accumulation grouping is different. The key guarantee is
        // that any *fixed* chunking scheme produces the same result regardless
        // of merge order (tested in test_merge_associativity).
        let ulp_12 = (r1.to_bits() as i64 - r2.to_bits() as i64).unsigned_abs();
        let ulp_13 = (r1.to_bits() as i64 - r3.to_bits() as i64).unsigned_abs();
        let ulp_23 = (r2.to_bits() as i64 - r3.to_bits() as i64).unsigned_abs();
        assert!(ulp_12 < 1000,
            "Single vs chunk-7 differ by {ulp_12} ULPs: {r1} vs {r2}");
        assert!(ulp_13 < 1000,
            "Single vs chunk-13 differ by {ulp_13} ULPs: {r1} vs {r3}");
        assert!(ulp_23 < 1000,
            "Chunk-7 vs chunk-13 differ by {ulp_23} ULPs: {r2} vs {r3}");
    }

    // -----------------------------------------------------------------------
    // fused_matmul_dot_kernel tests
    // -----------------------------------------------------------------------

    /// Reference: unfused chain — sequential Kahan matmul, then binned dot.
    /// This is what the fused kernel must match bit-for-bit.
    fn unfused_matmul_dot_reference(a: &[f64], w: &[f64], v: &[f64], m: usize, n: usize) -> f64 {
        // 1) intermediate[j] = sum_i a[i] * w[i*n+j] via Kahan (matches matmul_sequential)
        let mut intermediate = Vec::with_capacity(n);
        for j in 0..n {
            let mut k = cjc_repro::KahanAccumulatorF64::new();
            for i in 0..m {
                k.add(a[i] * w[i * n + j]);
            }
            intermediate.push(k.finalize());
        }
        // 2) dot(intermediate, v) via binned (matches the `dot` builtin)
        let products: Vec<f64> = intermediate.iter().zip(v.iter()).map(|(x, y)| x * y).collect();
        binned_sum_f64(&products)
    }

    #[test]
    fn fused_matmul_dot_empty_n_yields_zero() {
        // n=0: no columns, so result is the empty binned sum = 0.0.
        let result = fused_matmul_dot_kernel(&[1.0, 2.0], &[], &[], 2, 0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn fused_matmul_dot_1x1_matches_reference() {
        // [3.0] @ [[2.0]] · [5.0] = (3*2) * 5 = 30
        let a = [3.0];
        let w = [2.0];
        let v = [5.0];
        let result = fused_matmul_dot_kernel(&a, &w, &v, 1, 1);
        assert_eq!(result, 30.0);
        assert_eq!(result.to_bits(), unfused_matmul_dot_reference(&a, &w, &v, 1, 1).to_bits());
    }

    #[test]
    fn fused_matmul_dot_2x3_matches_reference() {
        // a = [1, 2], w = [[10, 20, 30], [40, 50, 60]], v = [100, 200, 300]
        // intermediate = [1*10 + 2*40, 1*20 + 2*50, 1*30 + 2*60] = [90, 120, 150]
        // dot = 90*100 + 120*200 + 150*300 = 9000 + 24000 + 45000 = 78000
        let a = [1.0, 2.0];
        let w = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let v = [100.0, 200.0, 300.0];
        let result = fused_matmul_dot_kernel(&a, &w, &v, 2, 3);
        assert_eq!(result, 78000.0);
        assert_eq!(result.to_bits(), unfused_matmul_dot_reference(&a, &w, &v, 2, 3).to_bits());
    }

    #[test]
    fn fused_matmul_dot_catastrophic_cancellation_stays_bit_identical() {
        // Pathological case: mixes 1e16 with 1.0, designed to break naive
        // summation. Binned + Kahan must still match the reference bit-exactly.
        let a = vec![1e16, 1.0, -1e16];
        let w = vec![
            1.0, 2.0,    // row 0
            1.0, 2.0,    // row 1
            1.0, 2.0,    // row 2
        ];
        let v = vec![1.0, 0.5];
        let result = fused_matmul_dot_kernel(&a, &w, &v, 3, 2);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_dot_reference(&a, &w, &v, 3, 2).to_bits(),
            "fused kernel must be bit-identical to unfused reference"
        );
    }

    #[test]
    fn fused_matmul_dot_determinism_100_runs() {
        let a = vec![0.1, 0.2, 0.3, 0.4];
        let w = vec![
            0.5, 0.6, 0.7,
            0.8, 0.9, 1.0,
            1.1, 1.2, 1.3,
            1.4, 1.5, 1.6,
        ];
        let v = vec![2.0, 3.0, 4.0];
        let first = fused_matmul_dot_kernel(&a, &w, &v, 4, 3);
        for _ in 0..100 {
            let again = fused_matmul_dot_kernel(&a, &w, &v, 4, 3);
            assert_eq!(first.to_bits(), again.to_bits());
        }
    }

    #[test]
    fn fused_matmul_dot_negative_values() {
        let a = vec![-1.0, 2.0, -3.0];
        let w = vec![
            1.0, -1.0,
            -2.0, 2.0,
            3.0, -3.0,
        ];
        let v = vec![-1.0, 1.0];
        let result = fused_matmul_dot_kernel(&a, &w, &v, 3, 2);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_dot_reference(&a, &w, &v, 3, 2).to_bits()
        );
    }

    // -----------------------------------------------------------------------
    // fused_matmul_norm_kernel tests
    // -----------------------------------------------------------------------

    /// Reference: sequential Kahan matmul, then norm (mirrors the unfused chain).
    fn unfused_matmul_norm_reference(
        a: &[f64], w: &[f64], ord: i64, m: usize, k: usize, n: usize,
    ) -> f64 {
        let mut intermediate = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut acc = cjc_repro::KahanAccumulatorF64::new();
                for kk in 0..k {
                    acc.add(a[i * k + kk] * w[kk * n + j]);
                }
                intermediate.push(acc.finalize());
            }
        }
        match ord {
            1 => {
                let abs_vals: Vec<f64> = intermediate.iter().map(|x| x.abs()).collect();
                binned_sum_f64(&abs_vals)
            }
            2 => {
                let sq_vals: Vec<f64> = intermediate.iter().map(|x| x * x).collect();
                binned_sum_f64(&sq_vals).sqrt()
            }
            p => {
                let pf = p as f64;
                let pv: Vec<f64> = intermediate.iter().map(|x| x.abs().powf(pf)).collect();
                binned_sum_f64(&pv).powf(1.0 / pf)
            }
        }
    }

    #[test]
    fn fused_matmul_norm_1x1_l2_matches_reference() {
        // [3.0] @ [[4.0]] = [12.0], ||·||_2 = 12.0
        let a = [3.0];
        let w = [4.0];
        let result = fused_matmul_norm_kernel(&a, &w, 2, 1, 1, 1);
        assert_eq!(result, 12.0);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_norm_reference(&a, &w, 2, 1, 1, 1).to_bits()
        );
    }

    #[test]
    fn fused_matmul_norm_l1_matches_reference() {
        // a = [[1, 2], [3, 4]] (2x2), w = [[1, 0], [0, 1]] (identity, 2x2)
        // h = a, |a| = [1, 2, 3, 4], sum = 10
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let result = fused_matmul_norm_kernel(&a, &w, 1, 2, 2, 2);
        assert_eq!(result, 10.0);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_norm_reference(&a, &w, 1, 2, 2, 2).to_bits()
        );
    }

    #[test]
    fn fused_matmul_norm_l2_matches_reference() {
        // a = [[3, 0], [0, 4]] (2x2), w = identity
        // h = a, ||·||_2 = sqrt(9 + 0 + 0 + 16) = sqrt(25) = 5
        let a = vec![3.0, 0.0, 0.0, 4.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let result = fused_matmul_norm_kernel(&a, &w, 2, 2, 2, 2);
        assert_eq!(result, 5.0);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_norm_reference(&a, &w, 2, 2, 2, 2).to_bits()
        );
    }

    #[test]
    fn fused_matmul_norm_l3_matches_reference() {
        // L3 norm exercises the general-p branch with .powf().
        let a = vec![2.0, 0.0, 0.0, 0.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let result = fused_matmul_norm_kernel(&a, &w, 3, 2, 2, 2);
        // h = a, L3 = (8 + 0 + 0 + 0)^(1/3) = 2.0
        assert!((result - 2.0).abs() < 1e-12);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_norm_reference(&a, &w, 3, 2, 2, 2).to_bits()
        );
    }

    #[test]
    fn fused_matmul_norm_negative_values_l2() {
        // Negative values exercise the |x|² (squaring kills sign).
        let a = vec![-1.0, 2.0, -3.0, 4.0];
        let w = vec![1.0, -1.0, -2.0, 2.0];
        let result = fused_matmul_norm_kernel(&a, &w, 2, 2, 2, 2);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_norm_reference(&a, &w, 2, 2, 2, 2).to_bits()
        );
    }

    #[test]
    fn fused_matmul_norm_catastrophic_cancellation_bit_identical() {
        let a = vec![1e16, 1.0, -1e16, 0.5];
        let w = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let result = fused_matmul_norm_kernel(&a, &w, 2, 2, 2, 4);
        assert_eq!(
            result.to_bits(),
            unfused_matmul_norm_reference(&a, &w, 2, 2, 2, 4).to_bits(),
            "fused norm must be bit-identical to unfused reference"
        );
    }

    #[test]
    fn fused_matmul_norm_determinism_100_runs() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w = vec![1.5, 2.5, 3.5, 4.5];
        let first = fused_matmul_norm_kernel(&a, &w, 2, 3, 2, 2);
        for _ in 0..100 {
            let again = fused_matmul_norm_kernel(&a, &w, 2, 3, 2, 2);
            assert_eq!(first.to_bits(), again.to_bits());
        }
    }

    // -----------------------------------------------------------------------
    // fused_matmul_matmul_kernel tests
    // -----------------------------------------------------------------------

    /// Reference: two sequential Kahan matmuls (matches matmul_sequential).
    fn unfused_matmul_matmul_reference(
        a: &[f64], b: &[f64], c: &[f64], m: usize, n: usize, k: usize, p: usize,
    ) -> Vec<f64> {
        let mut intermediate = vec![0.0f64; m * k];
        for i in 0..m {
            for q in 0..k {
                let mut acc = cjc_repro::KahanAccumulatorF64::new();
                for l in 0..n {
                    acc.add(a[i * n + l] * b[l * k + q]);
                }
                intermediate[i * k + q] = acc.finalize();
            }
        }
        let mut result = vec![0.0f64; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut acc = cjc_repro::KahanAccumulatorF64::new();
                for q in 0..k {
                    acc.add(intermediate[i * k + q] * c[q * p + j]);
                }
                result[i * p + j] = acc.finalize();
            }
        }
        result
    }

    fn assert_bits_equal(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: {x:e} vs {y:e}");
        }
    }

    #[test]
    fn fused_matmul_matmul_identity_chain() {
        // I @ I @ V = V (all 1x1)
        let a = vec![1.0];
        let b = vec![1.0];
        let c = vec![5.0];
        let r = fused_matmul_matmul_kernel(&a, &b, &c, 1, 1, 1, 1);
        assert_eq!(r, vec![5.0]);
    }

    #[test]
    fn fused_matmul_matmul_2x2_chain_matches_reference() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [2, 2]
        let c = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity
        let fused = fused_matmul_matmul_kernel(&a, &b, &c, 2, 2, 2, 2);
        let unfused = unfused_matmul_matmul_reference(&a, &b, &c, 2, 2, 2, 2);
        assert_bits_equal(&fused, &unfused);
    }

    #[test]
    fn fused_matmul_matmul_2x3_matches_reference() {
        // a: [2, 3], b: [3, 4], c: [4, 2]
        let a: Vec<f64> = (1..=6).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (1..=12).map(|i| i as f64 * 0.2).collect();
        let c: Vec<f64> = (1..=8).map(|i| i as f64 * 0.3).collect();
        let fused = fused_matmul_matmul_kernel(&a, &b, &c, 2, 3, 4, 2);
        let unfused = unfused_matmul_matmul_reference(&a, &b, &c, 2, 3, 4, 2);
        assert_bits_equal(&fused, &unfused);
    }

    #[test]
    fn fused_matmul_matmul_catastrophic_cancellation_bit_identical() {
        // Pathological: mixes 1e16 with 1.0, designed to test that the
        // per-element Kahan accumulator in both matmul steps matches the
        // unfused reference's accumulator exactly.
        let a = vec![1e16, 1.0, -1e16, 1.0]; // [2, 2]
        let b = vec![1.0, 2.0, 1.0, 2.0];     // [2, 2]
        let c = vec![1.0, -1.0, 0.5, 0.25];   // [2, 2]
        let fused = fused_matmul_matmul_kernel(&a, &b, &c, 2, 2, 2, 2);
        let unfused = unfused_matmul_matmul_reference(&a, &b, &c, 2, 2, 2, 2);
        assert_bits_equal(&fused, &unfused);
    }

    #[test]
    fn fused_matmul_matmul_determinism_100_runs() {
        let a: Vec<f64> = (1..=4).map(|i| i as f64 * 0.3).collect();
        let b: Vec<f64> = (1..=4).map(|i| i as f64 * 0.5).collect();
        let c: Vec<f64> = (1..=4).map(|i| i as f64 * 0.7).collect();
        let first = fused_matmul_matmul_kernel(&a, &b, &c, 2, 2, 2, 2);
        for _ in 0..100 {
            let again = fused_matmul_matmul_kernel(&a, &b, &c, 2, 2, 2, 2);
            assert_bits_equal(&first, &again);
        }
    }

    #[test]
    fn fused_matmul_matmul_3x3_chain_matches_reference() {
        let a: Vec<f64> = (1..=9).map(|i| i as f64 * 0.11).collect(); // [3, 3]
        let b: Vec<f64> = (1..=9).map(|i| i as f64 * 0.13).collect(); // [3, 3]
        let c: Vec<f64> = (1..=9).map(|i| i as f64 * 0.17).collect(); // [3, 3]
        let fused = fused_matmul_matmul_kernel(&a, &b, &c, 3, 3, 3, 3);
        let unfused = unfused_matmul_matmul_reference(&a, &b, &c, 3, 3, 3, 3);
        assert_bits_equal(&fused, &unfused);
    }
}
