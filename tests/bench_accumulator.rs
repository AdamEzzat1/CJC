//! Performance Benchmark: Naive vs Kahan vs Binned Summation
//!
//! Measures wall-clock time for 10M-element summation across strategies.
//! Run with: cargo test --test bench_accumulator --release -- --nocapture

use cjc_runtime::accumulator::{binned_sum_f64, BinnedAccumulatorF64};
use cjc_repro::kahan_sum_f64;
use std::time::Instant;

/// Naive summation (no compensation).
fn naive_sum_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for &v in values {
        sum += v;
    }
    sum
}

/// Measure a function N times, return (min_ns, mean_ns, result).
fn bench<F: Fn() -> f64>(f: F, iters: usize) -> (u64, u64, f64) {
    let mut times = Vec::with_capacity(iters);
    let mut result = 0.0;
    for _ in 0..iters {
        let start = Instant::now();
        result = f();
        let elapsed = start.elapsed().as_nanos() as u64;
        times.push(elapsed);
    }
    let min = *times.iter().min().unwrap();
    let mean = times.iter().sum::<u64>() / iters as u64;
    (min, mean, result)
}

#[test]
fn bench_10m_elements() {
    let n = 10_000_000;
    let mut rng = cjc_repro::Rng::seeded(42);
    let values: Vec<f64> = (0..n).map(|_| rng.next_normal_f64() * 1000.0).collect();
    let iters = 5;

    eprintln!("\n=== Accumulator Benchmark: {n} elements, {iters} iterations ===\n");

    // Naive
    let (naive_min, naive_mean, naive_result) = bench(|| naive_sum_f64(&values), iters);
    eprintln!("Naive:  min={:.2}ms  mean={:.2}ms  result={naive_result:.6e}",
        naive_min as f64 / 1e6, naive_mean as f64 / 1e6);

    // Kahan
    let (kahan_min, kahan_mean, kahan_result) = bench(|| kahan_sum_f64(&values), iters);
    eprintln!("Kahan:  min={:.2}ms  mean={:.2}ms  result={kahan_result:.6e}",
        kahan_min as f64 / 1e6, kahan_mean as f64 / 1e6);

    // Binned
    let (binned_min, binned_mean, binned_result) = bench(|| binned_sum_f64(&values), iters);
    eprintln!("Binned: min={:.2}ms  mean={:.2}ms  result={binned_result:.6e}",
        binned_min as f64 / 1e6, binned_mean as f64 / 1e6);

    // Binned with chunk-merge (simulating parallel)
    let (merge_min, merge_mean, merge_result) = bench(|| {
        let mut acc = BinnedAccumulatorF64::new();
        for chunk in values.chunks(10_000) {
            let mut c = BinnedAccumulatorF64::new();
            c.add_slice(chunk);
            acc.merge(&c);
        }
        acc.finalize()
    }, iters);
    eprintln!("Binned+Merge(10K chunks): min={:.2}ms  mean={:.2}ms  result={merge_result:.6e}",
        merge_min as f64 / 1e6, merge_mean as f64 / 1e6);

    // Ratios
    eprintln!("\n--- Slowdown vs Naive ---");
    eprintln!("Kahan:        {:.2}x", kahan_min as f64 / naive_min as f64);
    eprintln!("Binned:       {:.2}x", binned_min as f64 / naive_min as f64);
    eprintln!("Binned+Merge: {:.2}x", merge_min as f64 / naive_min as f64);

    // ULP distance from Binned (reference)
    let naive_ulps = (naive_result.to_bits() as i64 - binned_result.to_bits() as i64).unsigned_abs();
    let kahan_ulps = (kahan_result.to_bits() as i64 - binned_result.to_bits() as i64).unsigned_abs();
    let merge_ulps = (merge_result.to_bits() as i64 - binned_result.to_bits() as i64).unsigned_abs();

    eprintln!("\n--- ULP Distance from Binned (reference) ---");
    eprintln!("Naive:        {naive_ulps} ULPs");
    eprintln!("Kahan:        {kahan_ulps} ULPs");
    eprintln!("Binned+Merge: {merge_ulps} ULPs");

    // Determinism check
    eprintln!("\n--- Determinism ---");
    let r1 = binned_sum_f64(&values);
    let r2 = binned_sum_f64(&values);
    eprintln!("Binned run 1 == run 2: {}", r1.to_bits() == r2.to_bits());

    // Merge order invariance
    let mut fwd = BinnedAccumulatorF64::new();
    for chunk in values.chunks(10_000) {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        fwd.merge(&c);
    }

    let chunks: Vec<Vec<f64>> = values.chunks(10_000).map(|c| c.to_vec()).collect();
    let mut rev = BinnedAccumulatorF64::new();
    for chunk in chunks.iter().rev() {
        let mut c = BinnedAccumulatorF64::new();
        c.add_slice(chunk);
        rev.merge(&c);
    }
    eprintln!("Merge fwd == rev: {}", fwd.finalize().to_bits() == rev.finalize().to_bits());

    eprintln!("\n=== Benchmark Complete ===");
}
