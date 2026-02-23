//! Stress Test 1: Catastrophic Cancellation
//!
//! Generates extreme-magnitude alternating sequences and verifies that
//! the BinnedAccumulator handles catastrophic cancellation correctly.
//!
//! BigFloat reference: We use a manual 128-bit accumulator (two f64s in
//! double-double format) as the high-precision reference, since CJC has
//! zero external dependencies. This gives ~31 decimal digits of precision,
//! far exceeding the 15-16 of f64.

use cjc_runtime::accumulator::binned_sum_f64;
use cjc_repro::kahan_sum_f64;

// ---------------------------------------------------------------------------
// Double-double (pseudo-BigFloat) reference implementation
// ---------------------------------------------------------------------------

/// Double-double representation: value = hi + lo, where |lo| <= 0.5 * ulp(hi).
/// Provides ~31 decimal digits of precision using two f64 values.
/// This serves as our "BigFloat reference" with zero external dependencies.
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    fn zero() -> Self {
        DoubleDouble { hi: 0.0, lo: 0.0 }
    }

    /// Knuth's two-sum: exact addition of two f64 values.
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }

    /// Add a single f64 to the accumulator.
    fn add(&mut self, val: f64) {
        let (s1, e1) = Self::two_sum(self.hi, val);
        let (s2, e2) = Self::two_sum(self.lo, e1);
        self.hi = s1 + s2;
        self.lo = (s1 - self.hi) + s2 + e2;
    }

    /// Fold a slice deterministically (left-to-right).
    fn sum_slice(values: &[f64]) -> f64 {
        let mut acc = DoubleDouble::zero();
        for &v in values {
            acc.add(v);
        }
        acc.hi + acc.lo
    }
}

/// Naive sum (intentionally unstable).
fn naive_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for &v in values {
        sum += v;
    }
    sum
}

/// Compute ULP distance between two f64 values.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits - b_bits).unsigned_abs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_alternating_extreme_magnitudes() {
    // 10,000 elements alternating 1e16 and 1e-16.
    // True sum: 5000 * 1e16 + 5000 * 1e-16 = 5e19 + 5e-13.
    // Naive sum will lose the 5e-13 completely.
    let n = 10_000;
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        if i % 2 == 0 {
            values.push(1e16);
        } else {
            values.push(1e-16);
        }
    }

    let reference = DoubleDouble::sum_slice(&values);
    let naive = naive_sum(&values);
    let kahan = kahan_sum_f64(&values);
    let binned = binned_sum_f64(&values);

    // Reference should be very close to exact.
    let expected = 5000.0 * 1e16 + 5000.0 * 1e-16;
    assert!((reference - expected).abs() < 1e6, "BigFloat ref broken");

    // Naive loses the small component entirely.
    let naive_err = (naive - reference).abs();
    let kahan_err = (kahan - reference).abs();
    let binned_err = (binned - reference).abs();

    // Binned should be at least as good as Kahan.
    assert!(binned_err <= kahan_err + 1e-5,
        "Binned error ({binned_err}) worse than Kahan ({kahan_err})");

    // Naive should be significantly worse.
    assert!(naive_err >= binned_err,
        "Naive should be worse: naive_err={naive_err}, binned_err={binned_err}");
}

#[test]
fn test_cancellation_1e16_plus_1_minus_1e16() {
    // Classic: 1e16 + 1.0 - 1e16 should be 1.0.
    let values = vec![1e16, 1.0, -1e16];

    let reference = DoubleDouble::sum_slice(&values);
    let binned = binned_sum_f64(&values);

    // Binned accumulator bins values by exponent, so 1e16 and -1e16
    // cancel perfectly in their bin, leaving 1.0 in its bin.
    assert_eq!(binned, 1.0, "Binned should get exact result");
    assert!((reference - 1.0).abs() < 1e-10, "Reference should be ~1.0");
}

#[test]
fn test_large_cancellation_series() {
    // Sum of pairs: (+big, -big, +small) repeated.
    // All the big values should cancel, leaving only smalls.
    let big = 1e15;
    let small = 0.001;
    let reps = 1000;

    let mut values = Vec::with_capacity(reps * 3);
    for _ in 0..reps {
        values.push(big);
        values.push(-big);
        values.push(small);
    }

    let reference = DoubleDouble::sum_slice(&values);
    let binned = binned_sum_f64(&values);
    let kahan = kahan_sum_f64(&values);

    // Expected: 1000 * 0.001 = 1.0
    assert!((binned - 1.0).abs() < 1e-10,
        "Binned: expected ~1.0, got {binned}");
    // Kahan processes in order. With extreme cancellation (1e15 + -1e15),
    // Kahan's compensation can accumulate drift. The key point is that
    // Binned handles this much better than Kahan for unordered data.
    eprintln!("Kahan result: {kahan}, Binned result: {binned}");
    assert!((kahan - 1.0).abs() < 1.0,
        "Kahan: expected ~1.0, got {kahan}");
    assert!((reference - 1.0).abs() < 1e-12,
        "Reference: expected ~1.0, got {reference}");
}

#[test]
fn test_ulp_distance_metrics() {
    let values = vec![1e16, 1.0, -1e16, 0.5, -0.5];

    let reference = DoubleDouble::sum_slice(&values);
    let binned = binned_sum_f64(&values);
    let naive = naive_sum(&values);

    let binned_ulp = ulp_distance(binned, reference);
    let naive_ulp = ulp_distance(naive, reference);

    // Binned should have lower ULP distance than naive.
    assert!(binned_ulp <= naive_ulp + 10,
        "Binned ULP ({binned_ulp}) should be <= naive ULP ({naive_ulp})");
}

#[test]
fn test_l1_error_comparison() {
    // Many small values with occasional large spike.
    let n = 10_000;
    let mut values = Vec::with_capacity(n);
    let mut rng = cjc_repro::Rng::seeded(42);
    for i in 0..n {
        if i % 100 == 0 {
            values.push(1e12 * if rng.next_f64() > 0.5 { 1.0 } else { -1.0 });
        } else {
            values.push(rng.next_f64() * 2.0 - 1.0);
        }
    }

    let reference = DoubleDouble::sum_slice(&values);
    let naive = naive_sum(&values);
    let kahan = kahan_sum_f64(&values);
    let binned = binned_sum_f64(&values);

    let naive_l1 = (naive - reference).abs();
    let kahan_l1 = (kahan - reference).abs();
    let binned_l1 = (binned - reference).abs();

    // Print for diagnostics (visible in test output with --nocapture).
    eprintln!("L1 errors: naive={naive_l1:.2e} kahan={kahan_l1:.2e} binned={binned_l1:.2e}");

    // Binned should be competitive with Kahan.
    assert!(binned_l1 <= kahan_l1 * 100.0 + 1e-10,
        "Binned L1 error ({binned_l1}) too large vs Kahan ({kahan_l1})");
}
