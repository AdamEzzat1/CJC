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

/// Incremental Kahan accumulator for f64.
///
/// Stack-allocated. No heap allocation.
///
/// # Usage
/// ```ignore
/// let mut k = KahanAccumulatorF64::new();
/// for &x in data {
///     k.add(x);
/// }
/// let result = k.finalize();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF64 {
    sum: f64,
    compensation: f64,
    count: u64,
}

impl KahanAccumulatorF64 {
    /// Create a new zero accumulator.
    #[inline]
    pub fn new() -> Self {
        KahanAccumulatorF64 {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }

    /// Add a single value.
    ///
    /// Adding zero is a no-op (preserves bit-identical sum).
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

    /// Add a slice of values.
    #[inline]
    pub fn add_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Return the accumulated sum.
    #[inline]
    pub fn finalize(&self) -> f64 {
        self.sum
    }

    /// Number of values added.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF64 {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental Kahan accumulator for f32.
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulatorF32 {
    sum: f32,
    compensation: f32,
    count: u64,
}

impl KahanAccumulatorF32 {
    #[inline]
    pub fn new() -> Self {
        KahanAccumulatorF32 {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }

    /// Adding zero is a no-op (preserves bit-identical sum).
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

    #[inline]
    pub fn add_slice(&mut self, values: &[f32]) {
        for &v in values {
            self.add(v);
        }
    }

    #[inline]
    pub fn finalize(&self) -> f32 {
        self.sum
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KahanAccumulatorF32 {
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
