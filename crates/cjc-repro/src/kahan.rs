//! Kahan Summation — Scalar-tier deterministic reduction.
//!
//! # Determinism Contract
//!
//! Kahan compensated summation is deterministic in **serial execution** when
//! values are processed in a fixed order. It is NOT order-invariant — different
//! input orderings may produce different (but numerically stable) results.
//!
//! # When Used
//!
//! - Serial execution with `ReproMode::On`
//! - Non-vectorized loops
//! - Not inside `@nogc` or forced strict mode
//!
//! # Properties
//!
//! - **Error bound:** O(ε) for n summands (vs O(nε) for naive).
//! - **Heap allocation:** None. Two f64 registers (sum + compensation).
//! - **Branching:** No branches inside inner loop except compensation update.
//!
//! # Implementation Note
//!
//! The core implementations live in `cjc_repro::kahan_sum_f64` and
//! `cjc_repro::kahan_sum_f32` (the parent crate's `lib.rs`). This module
//! provides the `KahanAccumulator` struct for incremental accumulation.

/// Incremental Kahan compensated-summation accumulator for `f64`.
///
/// Maintains a running sum and a compensation term on the stack -- no heap
/// allocation is ever performed.  The error bound is O(epsilon) for *n*
/// additions, compared to O(*n* * epsilon) for naive summation.
///
/// Use this when values arrive one at a time or in variable-length batches.
/// For a one-shot slice reduction, see [`crate::kahan_sum_f64`].
///
/// # Determinism
///
/// The accumulator is deterministic for a given sequence of [`add`](Self::add)
/// calls in a fixed order.  It is **not** order-invariant -- permuting the
/// input may produce a different (but equally stable) result.
///
/// # Examples
///
/// ```
/// use cjc_repro::KahanAccumulatorF64;
///
/// let mut acc = KahanAccumulatorF64::new();
/// for _ in 0..10_000 {
///     acc.add(0.0001);
/// }
/// assert!((acc.finalize() - 1.0).abs() < 1e-10);
/// assert_eq!(acc.count(), 10_000);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF64 {
    sum: f64,
    compensation: f64,
    count: u64,
}

impl KahanAccumulatorF64 {
    /// Creates a new accumulator initialized to zero.
    ///
    /// Both the running sum and the compensation term start at `0.0`, and the
    /// count starts at `0`.
    #[inline]
    pub fn new() -> Self {
        KahanAccumulatorF64 {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }

    /// Adds a single `f64` value to the running sum.
    ///
    /// If `value` is exactly `0.0`, the compensation term is left untouched so
    /// that the accumulated sum remains bit-identical to what it would be
    /// without the zero.  The count is still incremented.
    ///
    /// # Arguments
    ///
    /// * `value` -- The value to accumulate.
    #[inline]
    pub fn add(&mut self, value: f64) {
        if value == 0.0 {
            self.count += 1;
            return;
        }
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
        self.count += 1;
    }

    /// Adds every element of `values` to the running sum in order.
    ///
    /// Equivalent to calling [`add`](Self::add) for each element.
    ///
    /// # Arguments
    ///
    /// * `values` -- The slice of `f64` values to accumulate.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Returns the accumulated compensated sum.
    ///
    /// This does **not** consume the accumulator -- you may continue adding
    /// values after calling `finalize`.
    ///
    /// # Returns
    ///
    /// The current compensated sum as `f64`.
    #[inline]
    pub fn finalize(&self) -> f64 {
        self.sum
    }

    /// Returns the number of values that have been added so far.
    ///
    /// Includes zero-valued additions.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF64 {
    /// Returns a zero-initialized accumulator (equivalent to [`KahanAccumulatorF64::new`]).
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental Kahan compensated-summation accumulator for `f32`.
///
/// Single-precision counterpart to [`KahanAccumulatorF64`].  Maintains a
/// running sum and compensation term on the stack with no heap allocation.
/// The error bound is O(epsilon) relative to `f32` machine epsilon.
///
/// # Determinism
///
/// Deterministic for a given sequence of additions in a fixed order.
///
/// # Examples
///
/// ```
/// use cjc_repro::KahanAccumulatorF32;
///
/// let mut acc = KahanAccumulatorF32::new();
/// for _ in 0..10_000 {
///     acc.add(0.0001_f32);
/// }
/// assert!((acc.finalize() - 1.0).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF32 {
    sum: f32,
    compensation: f32,
    count: u64,
}

impl KahanAccumulatorF32 {
    /// Creates a new accumulator initialized to zero.
    #[inline]
    pub fn new() -> Self {
        KahanAccumulatorF32 {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }

    /// Adds a single `f32` value to the running sum.
    ///
    /// If `value` is exactly `0.0`, the compensation term is left untouched
    /// to preserve bit-identical results.  The count is still incremented.
    ///
    /// # Arguments
    ///
    /// * `value` -- The value to accumulate.
    #[inline]
    pub fn add(&mut self, value: f32) {
        if value == 0.0 {
            self.count += 1;
            return;
        }
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
        self.count += 1;
    }

    /// Adds every element of `values` to the running sum in order.
    ///
    /// Equivalent to calling [`add`](Self::add) for each element.
    ///
    /// # Arguments
    ///
    /// * `values` -- The slice of `f32` values to accumulate.
    #[inline]
    pub fn add_slice(&mut self, values: &[f32]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Returns the accumulated compensated sum.
    ///
    /// Does **not** consume the accumulator.
    ///
    /// # Returns
    ///
    /// The current compensated sum as `f32`.
    #[inline]
    pub fn finalize(&self) -> f32 {
        self.sum
    }

    /// Returns the number of values that have been added so far.
    ///
    /// Includes zero-valued additions.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF32 {
    /// Returns a zero-initialized accumulator (equivalent to [`KahanAccumulatorF32::new`]).
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_accumulator_simple() {
        let mut acc = KahanAccumulatorF64::new();
        for i in 1..=10 {
            acc.add(i as f64);
        }
        assert_eq!(acc.finalize(), 55.0);
        assert_eq!(acc.count(), 10);
    }

    #[test]
    fn test_kahan_accumulator_stability() {
        let mut acc = KahanAccumulatorF64::new();
        for _ in 0..10000 {
            acc.add(0.0001);
        }
        assert!((acc.finalize() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kahan_accumulator_f32() {
        let mut acc = KahanAccumulatorF32::new();
        for _ in 0..10000 {
            acc.add(0.0001f32);
        }
        assert!((acc.finalize() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_kahan_matches_existing_function() {
        let values: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.001).collect();
        let mut acc = KahanAccumulatorF64::new();
        acc.add_slice(&values);

        let func_result = crate::kahan_sum_f64(&values);
        assert_eq!(acc.finalize().to_bits(), func_result.to_bits());
    }
}
