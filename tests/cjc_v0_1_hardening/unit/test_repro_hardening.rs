//! Reproducibility module (cjc-repro) hardening tests — RNG, Kahan accumulator.

use cjc_repro::{Rng, KahanAccumulatorF64};

/// Same seed produces identical sequence.
#[test]
fn rng_deterministic() {
    let mut r1 = Rng::seeded(12345);
    let mut r2 = Rng::seeded(12345);
    for _ in 0..100 {
        assert_eq!(r1.next_u64(), r2.next_u64());
    }
}

/// Different seeds produce different sequences.
#[test]
fn rng_different_seeds() {
    let mut r1 = Rng::seeded(1);
    let mut r2 = Rng::seeded(2);
    let v1: Vec<u64> = (0..10).map(|_| r1.next_u64()).collect();
    let v2: Vec<u64> = (0..10).map(|_| r2.next_u64()).collect();
    assert_ne!(v1, v2, "Different seeds should produce different sequences");
}

/// Zero seed works.
#[test]
fn rng_zero_seed() {
    let mut r = Rng::seeded(0);
    // Should not all be zero
    let vals: Vec<u64> = (0..10).map(|_| r.next_u64()).collect();
    assert!(vals.iter().any(|&v| v != 0), "Zero seed should still generate non-zero values");
}

/// Max seed works.
#[test]
fn rng_max_seed() {
    let mut r = Rng::seeded(u64::MAX);
    let _ = r.next_u64(); // Should not panic
}

/// next_f64 produces values in [0, 1) range.
#[test]
fn rng_f64_range() {
    let mut r = Rng::seeded(42);
    for _ in 0..1000 {
        let v = r.next_f64();
        assert!(v >= 0.0 && v < 1.0, "next_f64 should be in [0, 1), got {v}");
    }
}

/// Fork produces different but deterministic child RNG.
#[test]
fn rng_fork_deterministic() {
    let mut r1 = Rng::seeded(42);
    let child1 = r1.fork();
    let mut r2 = Rng::seeded(42);
    let child2 = r2.fork();

    let mut c1 = child1;
    let mut c2 = child2;
    for _ in 0..50 {
        assert_eq!(c1.next_u64(), c2.next_u64(), "Forked RNGs should be identical");
    }
}

/// Kahan accumulator — basic sum.
#[test]
fn kahan_basic_sum() {
    let mut acc = KahanAccumulatorF64::new();
    acc.add(1.0);
    acc.add(2.0);
    acc.add(3.0);
    assert!((acc.finalize() - 6.0).abs() < 1e-15);
}

/// Kahan accumulator — compensated summation is more accurate than naive.
#[test]
fn kahan_compensated_accuracy() {
    let n = 1_000_000;
    let val = 0.1;

    // Naive sum
    let mut naive = 0.0f64;
    for _ in 0..n {
        naive += val;
    }

    // Kahan sum
    let mut kahan = KahanAccumulatorF64::new();
    for _ in 0..n {
        kahan.add(val);
    }

    let expected = val * (n as f64);
    let naive_err = (naive - expected).abs();
    let kahan_err = (kahan.finalize() - expected).abs();

    // Kahan should be at least as accurate as naive (usually much better)
    assert!(
        kahan_err <= naive_err + 1e-10,
        "Kahan error ({kahan_err}) should not be worse than naive error ({naive_err})"
    );
}

/// Kahan accumulator — empty sum is zero.
#[test]
fn kahan_empty_sum() {
    let acc = KahanAccumulatorF64::new();
    assert_eq!(acc.finalize(), 0.0);
}

/// Kahan accumulator — single element.
#[test]
fn kahan_single_element() {
    let mut acc = KahanAccumulatorF64::new();
    acc.add(42.0);
    assert_eq!(acc.finalize(), 42.0);
}

/// Kahan accumulator — negative values.
#[test]
fn kahan_negative_values() {
    let mut acc = KahanAccumulatorF64::new();
    acc.add(10.0);
    acc.add(-3.0);
    acc.add(-7.0);
    assert!((acc.finalize() - 0.0).abs() < 1e-15, "Sum of 10 - 3 - 7 should be 0");
}

/// Kahan accumulator determinism — same inputs same output.
#[test]
fn kahan_deterministic() {
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.001).collect();

    let mut acc1 = KahanAccumulatorF64::new();
    let mut acc2 = KahanAccumulatorF64::new();
    for &v in &values {
        acc1.add(v);
        acc2.add(v);
    }

    assert_eq!(
        acc1.finalize().to_bits(),
        acc2.finalize().to_bits(),
        "Kahan sum must be bit-identical for same inputs"
    );
}
