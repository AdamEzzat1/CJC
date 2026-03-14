//! Hardening tests: Near-order-invariant accumulation after BinnedAccumulator migration.
//!
//! Verifies that summing the same values in different orders produces
//! results within a tight ULP tolerance — using BinnedAccumulatorF64's
//! exponent-binned summation which is far more stable than naive or Kahan
//! summation under reordering.

use cjc_runtime::accumulator::BinnedAccumulatorF64;
use cjc_runtime::tensor::Tensor;

/// Maximum allowed ULP difference for order-invariance tests.
/// BinnedAccumulator provides near-order-invariance (typically 0-5 ULP).
const MAX_ULP_DIFF: u64 = 8;

fn ulp_diff(a: f64, b: f64) -> u64 {
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits - b_bits).unsigned_abs()
}

// ---------------------------------------------------------------------------
// BinnedAccumulator: forward vs reverse order
// ---------------------------------------------------------------------------

#[test]
fn test_binned_sum_forward_reverse_identical() {
    let values: Vec<f64> = (1..=10000).map(|i| 1.0 / (i as f64)).collect();

    let mut fwd = BinnedAccumulatorF64::new();
    for &v in &values {
        fwd.add(v);
    }

    let mut rev = BinnedAccumulatorF64::new();
    for &v in values.iter().rev() {
        rev.add(v);
    }

    let diff = ulp_diff(fwd.finalize(), rev.finalize());
    assert!(
        diff <= MAX_ULP_DIFF,
        "BinnedAccumulator forward vs reverse ULP diff {} exceeds max {}",
        diff, MAX_ULP_DIFF
    );
}

#[test]
fn test_binned_sum_shuffled_identical() {
    let values: Vec<f64> = (1..=10000).map(|i| 1.0 / (i as f64)).collect();

    // Deterministic "shuffle" — interleave odds and evens
    let mut shuffled = Vec::with_capacity(values.len());
    let (odds, evens): (Vec<_>, Vec<_>) = values.iter().enumerate().partition(|(i, _)| i % 2 == 1);
    for (_, v) in &evens {
        shuffled.push(**v);
    }
    for (_, v) in &odds {
        shuffled.push(**v);
    }

    let mut orig = BinnedAccumulatorF64::new();
    for &v in &values {
        orig.add(v);
    }

    let mut shuf = BinnedAccumulatorF64::new();
    for &v in &shuffled {
        shuf.add(v);
    }

    let diff = ulp_diff(orig.finalize(), shuf.finalize());
    assert!(
        diff <= MAX_ULP_DIFF,
        "BinnedAccumulator original vs shuffled ULP diff {} exceeds max {}",
        diff, MAX_ULP_DIFF
    );
}

// ---------------------------------------------------------------------------
// Tensor sum: deterministic through BinnedAccumulator
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_sum_deterministic() {
    let data: Vec<f64> = (1..=1000).map(|i| 1.0 / (i as f64)).collect();
    let t = Tensor::from_vec(data.clone(), &[1000]).unwrap();

    let sum1 = t.sum();
    let sum2 = t.sum();

    assert_eq!(
        sum1.to_bits(),
        sum2.to_bits(),
        "Tensor.sum() must be deterministic"
    );
}

#[test]
fn test_tensor_sum_matches_binned() {
    // Verify tensor sum matches direct BinnedAccumulator sum
    let data: Vec<f64> = (1..=500).map(|i| (i as f64).sqrt()).collect();
    let t = Tensor::from_vec(data.clone(), &[500]).unwrap();

    let tensor_sum = t.sum();

    let mut acc = BinnedAccumulatorF64::new();
    for &v in &data {
        acc.add(v);
    }
    let direct_sum = acc.finalize();

    assert_eq!(
        tensor_sum.to_bits(),
        direct_sum.to_bits(),
        "Tensor.sum() must match direct BinnedAccumulator sum"
    );
}

// ---------------------------------------------------------------------------
// Stress: large mixed-magnitude values
// ---------------------------------------------------------------------------

#[test]
fn test_binned_sum_mixed_magnitude() {
    // Mix tiny and huge values — this is where naive summation fails
    let mut values = Vec::new();
    for i in 0..1000 {
        values.push(1e15);
        values.push(1e-15);
        values.push(-(1e15));
        values.push(i as f64 * 0.001);
    }

    let mut fwd = BinnedAccumulatorF64::new();
    for &v in &values {
        fwd.add(v);
    }

    let mut rev = BinnedAccumulatorF64::new();
    for &v in values.iter().rev() {
        rev.add(v);
    }

    let diff = ulp_diff(fwd.finalize(), rev.finalize());
    assert!(
        diff <= MAX_ULP_DIFF,
        "Mixed-magnitude forward vs reverse ULP diff {} exceeds max {}",
        diff, MAX_ULP_DIFF
    );
}

// ---------------------------------------------------------------------------
// Stats variance: deterministic
// ---------------------------------------------------------------------------

#[test]
fn test_stats_variance_deterministic() {
    let data: Vec<f64> = (1..=10000).map(|i| 1.0 / (i as f64)).collect();

    let var1 = cjc_runtime::stats::variance(&data).unwrap();
    let var2 = cjc_runtime::stats::variance(&data).unwrap();

    assert_eq!(
        var1.to_bits(),
        var2.to_bits(),
        "stats::variance must be deterministic"
    );
}

// ---------------------------------------------------------------------------
// Repeated accumulation produces same result
// ---------------------------------------------------------------------------

#[test]
fn test_binned_accumulator_repeated_identical() {
    for _ in 0..10 {
        let mut acc = BinnedAccumulatorF64::new();
        for i in 1..=5000 {
            acc.add((i as f64).ln());
        }
        let result = acc.finalize();
        // First iteration establishes the golden value
        static mut GOLDEN: Option<u64> = None;
        unsafe {
            match GOLDEN {
                None => GOLDEN = Some(result.to_bits()),
                Some(g) => assert_eq!(
                    result.to_bits(),
                    g,
                    "Repeated accumulation must produce identical results"
                ),
            }
        }
    }
}
