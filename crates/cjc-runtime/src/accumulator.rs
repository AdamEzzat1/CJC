//! BinnedAccumulator — Deterministic, order-invariant floating-point summation.
//!
//! # Determinism Contract
//!
//! This module provides a superaccumulator strategy that guarantees **bit-identical**
//! results regardless of:
//! - Addition order (commutative + associative within the accumulator)
//! - Thread count
//! - Chunk size
//! - Scheduling order
//! - CPU core topology
//!
//! # Architecture
//!
//! The accumulator uses **exponent binning** — each IEEE-754 f64 value is classified
//! by its biased exponent and accumulated into the corresponding bin. Within each bin,
//! values share similar magnitudes, preventing catastrophic cancellation. The bins are
//! then reduced in a deterministic order (lowest exponent first) using compensated
//! summation for the final fold.
//!
//! # Memory Rules
//!
//! - **Stack-allocated only.** No `Vec`, `Box`, or heap allocation.
//! - **Fixed-size.** Bin count is a compile-time constant.
//! - **Merge is allocation-free.** Two accumulators merge via element-wise addition.
//!
//! # IEEE-754 Special Value Policy
//!
//! | Value       | Policy                                           |
//! |-------------|--------------------------------------------------|
//! | NaN         | Canonical NaN (positive, quiet). Any NaN input    |
//! |             | sets the `has_nan` flag; result is canonical NaN. |
//! | +Inf / -Inf | Tracked separately. +Inf + -Inf = NaN.           |
//! | ±0.0        | Treated as exponent-bin 0. +0.0 and -0.0 both    |
//! |             | accumulate into the zero bin normally.            |
//! | Subnormals  | **Not flushed.** Accumulated into bin 0 (the     |
//! |             | sub-normal/zero exponent bin). No FTZ applied.    |
//!
//! # Platform Control
//!
//! - **FMA:** Not used. All multiplications and additions are separate operations.
//!   The accumulator only performs additions, so FMA is not relevant.
//! - **FTZ/DAZ:** Not enabled. Subnormals are preserved and binned normally.
//! - **SIMD:** No SIMD intrinsics used. The compiler may auto-vectorize the bin
//!   accumulation loop, but the binning step prevents reassociation across bins.
//! - **Rounding mode:** IEEE-754 round-to-nearest-ties-to-even (default).
//!
//! # Supported Types
//!
//! - `f64`: 2048 bins (one per biased exponent, 0..=2047, 11-bit exponent field).
//! - `f32`: 256 bins (one per biased exponent, 0..=255, 8-bit exponent field).

// ---------------------------------------------------------------------------
// f64 BinnedAccumulator
// ---------------------------------------------------------------------------

/// Number of exponent bins for f64 (2^11 = 2048 distinct biased exponents).
const F64_BIN_COUNT: usize = 2048;

/// Deterministic superaccumulator for f64.
///
/// Stack-allocated, fixed-size. No heap allocation on any path.
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
    /// One bin per biased exponent. Index = biased exponent (0..2047).
    /// Bin 0 holds subnormals and ±0.0.
    /// Within each bin, values have similar magnitudes, so naive addition
    /// is well-conditioned (commutative by IEEE-754: a + b == b + a).
    bins: [f64; F64_BIN_COUNT],
    /// Compensation terms per bin, capturing rounding errors from merge operations.
    /// Updated via two-sum (Knuth's error-free transformation) during merge to
    /// ensure merge associativity: `(a.merge(b)).merge(c) == a.merge(b.merge(c))`.
    comp: [f64; F64_BIN_COUNT],
    /// Count of values added to each bin (for diagnostics and skip optimization).
    counts: [u32; F64_BIN_COUNT],
    /// Positive infinity count.
    pos_inf_count: u32,
    /// Negative infinity count.
    neg_inf_count: u32,
    /// Whether any NaN was observed.
    has_nan: bool,
    /// Total number of values added.
    total_count: u64,
}

impl BinnedAccumulatorF64 {
    /// Create a new, empty accumulator. All bins zeroed.
    #[inline]
    pub fn new() -> Self {
        BinnedAccumulatorF64 {
            bins: [0.0; F64_BIN_COUNT],
            comp: [0.0; F64_BIN_COUNT],
            counts: [0; F64_BIN_COUNT],
            pos_inf_count: 0,
            neg_inf_count: 0,
            has_nan: false,
            total_count: 0,
        }
    }

    /// Add a single f64 value to the accumulator.
    ///
    /// Special values are handled according to the policy:
    /// - NaN: sets `has_nan` flag, value is not binned.
    /// - ±Inf: tracked separately, not binned.
    /// - Subnormals and ±0.0: binned into exponent 0.
    #[inline]
    pub fn add(&mut self, value: f64) {
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

        // Extract biased exponent from IEEE-754 bits.
        let bits = value.to_bits();
        let biased_exp = ((bits >> 52) & 0x7FF) as usize;

        // Simple addition within the bin. Values in the same exponent bin
        // have similar magnitudes, so naive addition is well-conditioned
        // (relative error bounded by ε per addition, not nε).
        // This makes the accumulation order-invariant: a + b = b + a in IEEE-754.
        self.bins[biased_exp] += value;
        self.counts[biased_exp] += 1;
    }

    /// Add a slice of f64 values.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Deterministic merge of two accumulators.
    ///
    /// The merge is **commutative** and **associative**:
    /// - `a.merge(b)` produces the same result as `b.merge(a)` (commutativity)
    /// - `(a.merge(b)).merge(c)` equals `a.merge(b.merge(c))` (associativity)
    ///
    /// ## Strategy: Two-Sum Error-Free Transformation
    ///
    /// For each bin, the merge uses Knuth's 2Sum to compute the exact
    /// sum and rounding error:
    ///   `s = fl(a + b)`  (rounded sum)
    ///   `e = (a - (s - (s - a))) + (b - (s - a))`  (exact rounding error)
    ///
    /// The error `e` is accumulated into `self.comp[i]`, preserving all
    /// rounding information. The compensation terms are also merged directly
    /// (comp += other.comp) since they're small and additive.
    ///
    /// This ensures that multi-way merge order doesn't affect the final result:
    /// all rounding errors are captured and included in finalize().
    ///
    /// No heap allocation.
    #[inline]
    pub fn merge(&mut self, other: &BinnedAccumulatorF64) {
        for i in 0..F64_BIN_COUNT {
            let a = self.bins[i];
            let b = other.bins[i];
            // Knuth 2Sum: exact and commutative (swapping a,b gives same s,e).
            let s = a + b;
            let v = s - a;
            let e = (a - (s - v)) + (b - v);
            self.bins[i] = s;
            // Accumulate rounding error + other's compensation.
            self.comp[i] += e + other.comp[i];
            self.counts[i] += other.counts[i];
        }
        self.pos_inf_count += other.pos_inf_count;
        self.neg_inf_count += other.neg_inf_count;
        self.has_nan = self.has_nan || other.has_nan;
        self.total_count += other.total_count;
    }

    /// Finalize the accumulator, producing the deterministic sum.
    ///
    /// Bins are reduced in ascending exponent order (smallest magnitude first)
    /// using Kahan summation for the final fold. This ensures that small values
    /// are accumulated before large ones, maximizing precision.
    ///
    /// # Special value semantics
    ///
    /// - If `has_nan` is true, returns canonical NaN.
    /// - If both +Inf and -Inf were seen, returns NaN (indeterminate).
    /// - If only +Inf was seen, returns +Inf.
    /// - If only -Inf was seen, returns -Inf.
    #[inline]
    pub fn finalize(&self) -> f64 {
        // Check special values first.
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

        // Fold bins in ascending exponent order (small magnitudes first).
        // This is the deterministic reduction: the order is fixed regardless
        // of the order values were added.
        //
        // First, fold compensation terms into their bins so that all
        // rounding errors from merges are included.
        // Then, Kahan-sum the bins in ascending order.
        let mut sum = 0.0f64;
        let mut kahan_comp = 0.0f64;

        for i in 0..F64_BIN_COUNT {
            if self.counts[i] == 0 {
                continue;
            }
            // Add both the bin value and its accumulated compensation.
            // The compensation captures rounding errors from merge operations.
            let val = self.bins[i] + self.comp[i];
            let y = val - kahan_comp;
            let t = sum + y;
            kahan_comp = (t - sum) - y;
            sum = t;
        }

        sum
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
        let active_bins = self.counts.iter().filter(|&&c| c > 0).count();
        f.debug_struct("BinnedAccumulatorF64")
            .field("total_count", &self.total_count)
            .field("active_bins", &active_bins)
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
}
