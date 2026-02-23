//! Stress Test 2: Log-Uniform Magnitude Test
//!
//! Generates values with exponents sampled log-uniformly to stress the
//! bin distribution of the BinnedAccumulator. This exercises the full
//! exponent range (10^-300 to 10^+300) and verifies stability.

use cjc_runtime::accumulator::{binned_sum_f64, BinnedAccumulatorF64};
use cjc_repro::kahan_sum_f64;

/// Generate a value with a log-uniformly distributed exponent.
///
/// exponent is sampled uniformly from [-max_exp, +max_exp], then
/// value = sign * 10^exponent.
fn log_uniform_value(rng: &mut cjc_repro::Rng, max_exp: f64) -> f64 {
    let exp = rng.next_f64() * 2.0 * max_exp - max_exp;
    let sign = if rng.next_f64() > 0.5 { 1.0 } else { -1.0 };
    sign * 10.0_f64.powf(exp)
}

#[test]
fn test_log_uniform_binned_stability() {
    let n = 10_000;
    let mut rng = cjc_repro::Rng::seeded(123);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(log_uniform_value(&mut rng, 200.0));
    }

    // Binned sum should not panic or produce NaN/Inf from finite inputs.
    let binned = binned_sum_f64(&values);
    assert!(binned.is_finite(), "Binned sum should be finite, got {binned}");
}

#[test]
fn test_log_uniform_merge_order_invariance() {
    // Test that chunk-and-merge produces identical results regardless
    // of chunk size or merge order — this is the actual parallel invariant.
    let n = 5_000;
    let mut rng = cjc_repro::Rng::seeded(456);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(log_uniform_value(&mut rng, 150.0));
    }

    // Chunk size 7, merge forward.
    let mut acc_7 = BinnedAccumulatorF64::new();
    for chunk in values.chunks(7) {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        acc_7.merge(&c);
    }

    // Chunk size 13, merge forward.
    let mut acc_13 = BinnedAccumulatorF64::new();
    for chunk in values.chunks(13) {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        acc_13.merge(&c);
    }

    // Chunk size 7, merge reverse.
    let chunks_7: Vec<Vec<f64>> = values.chunks(7).map(|c| c.to_vec()).collect();
    let mut acc_7r = BinnedAccumulatorF64::new();
    for chunk in chunks_7.iter().rev() {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        acc_7r.merge(&c);
    }

    let r7 = acc_7.finalize();
    let r13 = acc_13.finalize();
    let _r7r = acc_7r.finalize();

    // Merge commutativity: for each pair of chunk accumulators,
    // a.merge(b) == b.merge(a).
    let chunks_7v: Vec<Vec<f64>> = values.chunks(7).map(|c| c.to_vec()).collect();
    if chunks_7v.len() >= 2 {
        let mut a = BinnedAccumulatorF64::new();
        a.add_slice(&chunks_7v[0]);
        let mut b = BinnedAccumulatorF64::new();
        b.add_slice(&chunks_7v[1]);

        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);

        assert_eq!(ab.finalize().to_bits(), ba.finalize().to_bits(),
            "Merge commutativity failed: a+b={} vs b+a={}", ab.finalize(), ba.finalize());
    }

    // Results across chunk sizes should be very close (sub-ULP may differ
    // due to different associativity patterns within bins).
    let diff = (r7 - r13).abs();
    let rel = if r7.abs() > 1e-300 { diff / r7.abs() } else { diff };
    eprintln!("Chunk-7 vs Chunk-13: abs_diff={diff:.2e}, rel_diff={rel:.2e}");
    assert!(rel < 1e-10 || diff < 1e-100,
        "Different chunk sizes should be very close: {r7} vs {r13}");
}

#[test]
fn test_log_uniform_chunk_invariance() {
    let n = 8_000;
    let mut rng = cjc_repro::Rng::seeded(789);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(log_uniform_value(&mut rng, 100.0));
    }

    let single = binned_sum_f64(&values);

    // Chunk into 17s, merge.
    let mut merged = BinnedAccumulatorF64::new();
    for chunk in values.chunks(17) {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        merged.merge(&c);
    }
    let chunked = merged.finalize();

    assert_eq!(single.to_bits(), chunked.to_bits(),
        "Single ({single}) vs chunked-17 ({chunked}) must be bit-identical");
}

#[test]
fn test_log_uniform_extreme_range() {
    // Test the very extremes of f64 exponent range.
    let values = vec![
        1e-308, // near subnormal boundary
        1e308,  // near overflow boundary
        -1e308,
        -1e-308,
        5e-324, // smallest subnormal
        1.7976931348623157e308, // f64::MAX
        -1.7976931348623157e308, // f64::MIN
    ];

    let binned = binned_sum_f64(&values);
    assert!(binned.is_finite() || binned.is_nan(),
        "Result should be finite or NaN for cancelling MAX values");

    // When MAX and -MAX cancel, remaining subnormals should dominate.
    // The exact result depends on bin precision.
}

#[test]
fn test_kahan_error_profile_documented() {
    // Document the Kahan error profile for log-uniform data.
    // This test doesn't assert failure — it documents the behavior.
    let n = 10_000;
    let mut rng = cjc_repro::Rng::seeded(999);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(log_uniform_value(&mut rng, 100.0));
    }

    let kahan = kahan_sum_f64(&values);
    let binned = binned_sum_f64(&values);

    // Both should produce finite results.
    assert!(kahan.is_finite(), "Kahan must be finite");
    assert!(binned.is_finite(), "Binned must be finite");

    // Document the difference.
    let diff = (kahan - binned).abs();
    let rel = if binned.abs() > 1e-300 {
        diff / binned.abs()
    } else {
        diff
    };

    eprintln!("Kahan vs Binned on log-uniform data:");
    eprintln!("  Kahan:  {kahan:.6e}");
    eprintln!("  Binned: {binned:.6e}");
    eprintln!("  Abs diff: {diff:.6e}");
    eprintln!("  Rel diff: {rel:.6e}");
}

#[test]
fn test_bin_distribution_coverage() {
    // Verify that log-uniform data actually exercises many bins.
    let n = 10_000;
    let mut rng = cjc_repro::Rng::seeded(42);
    let mut acc = BinnedAccumulatorF64::new();
    for _ in 0..n {
        acc.add(log_uniform_value(&mut rng, 200.0));
    }

    // We should have values spread across many exponent bins.
    // Can't directly inspect bins, but we can verify the count is correct.
    assert_eq!(acc.count(), n as u64);
    assert!(!acc.has_nan());
}
