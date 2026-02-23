// CJC Test Suite — cjc-repro (6 tests)
// Source: crates/cjc-repro/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_repro::*;

#[test]
fn test_rng_deterministic() {
    let mut rng1 = Rng::seeded(42);
    let mut rng2 = Rng::seeded(42);

    for _ in 0..100 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
}

#[test]
fn test_rng_f64_range() {
    let mut rng = Rng::seeded(123);
    for _ in 0..1000 {
        let v = rng.next_f64();
        assert!((0.0..1.0).contains(&v));
    }
}

#[test]
fn test_rng_fork_deterministic() {
    let mut rng1 = Rng::seeded(42);
    let mut rng2 = Rng::seeded(42);

    let mut fork1 = rng1.fork();
    let mut fork2 = rng2.fork();

    for _ in 0..50 {
        assert_eq!(fork1.next_u64(), fork2.next_u64());
    }
}

#[test]
fn test_kahan_sum() {
    // Sum of many small values where naive sum would lose precision
    let values: Vec<f64> = (0..10000).map(|_| 0.0001).collect();
    let result = kahan_sum_f64(&values);
    assert!((result - 1.0).abs() < 1e-10);
}

#[test]
fn test_kahan_sum_f32() {
    let values: Vec<f32> = (0..10000).map(|_| 0.0001f32).collect();
    let result = kahan_sum_f32(&values);
    assert!((result - 1.0).abs() < 1e-4);
}

#[test]
fn test_pairwise_sum() {
    let values: Vec<f64> = (0..10000).map(|_| 0.0001).collect();
    let result = pairwise_sum_f64(&values);
    assert!((result - 1.0).abs() < 1e-10);
}
