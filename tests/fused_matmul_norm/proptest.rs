//! Property tests — `fused_matmul_norm` equals the unfused chain bit-for-bit.
//!
//! Tests the kernel-level helper directly across random shapes and orders.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::accumulator::{binned_sum_f64, fused_matmul_norm_kernel};
use proptest::prelude::*;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn det_floats(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| (splitmix64(&mut s) as f64 / u64::MAX as f64) * 4.0 - 2.0)
        .collect()
}

/// Unfused reference at the kernel level.
fn unfused_reference(a: &[f64], w: &[f64], ord: i64, m: usize, k: usize, n: usize) -> f64 {
    let mut intermediate = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = KahanAccumulatorF64::new();
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

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// Headline: fused == unfused, bit-for-bit, for L2 norm.
    #[test]
    fn fused_l2_equals_unfused(
        m in 1usize..16,
        k in 1usize..16,
        n in 1usize..16,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m * k);
        let w = det_floats(seed.wrapping_add(1), k * n);
        let fused = fused_matmul_norm_kernel(&a, &w, 2, m, k, n);
        let unfused = unfused_reference(&a, &w, 2, m, k, n);
        prop_assert_eq!(
            fused.to_bits(),
            unfused.to_bits(),
            "L2 fused vs unfused diverged at ({}, {}, {}) seed {}",
            m, k, n, seed
        );
    }

    /// Fused == unfused for L1 norm.
    #[test]
    fn fused_l1_equals_unfused(
        m in 1usize..12,
        k in 1usize..12,
        n in 1usize..12,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m * k);
        let w = det_floats(seed.wrapping_add(1), k * n);
        let fused = fused_matmul_norm_kernel(&a, &w, 1, m, k, n);
        let unfused = unfused_reference(&a, &w, 1, m, k, n);
        prop_assert_eq!(fused.to_bits(), unfused.to_bits());
    }

    /// Determinism — same input, same output, 100% of the time.
    #[test]
    fn fused_norm_is_deterministic(
        m in 1usize..8,
        k in 1usize..8,
        n in 1usize..8,
        ord in prop::sample::select(vec![1i64, 2, 3]),
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m * k);
        let w = det_floats(seed.wrapping_add(1), k * n);
        let first = fused_matmul_norm_kernel(&a, &w, ord, m, k, n);
        for _ in 0..4 {
            let again = fused_matmul_norm_kernel(&a, &w, ord, m, k, n);
            prop_assert_eq!(first.to_bits(), again.to_bits());
        }
    }

    /// Norm is always non-negative (mathematical invariant).
    #[test]
    fn fused_norm_is_nonnegative(
        m in 1usize..8,
        k in 1usize..8,
        n in 1usize..8,
        ord in prop::sample::select(vec![1i64, 2]),
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m * k);
        let w = det_floats(seed.wrapping_add(1), k * n);
        let result = fused_matmul_norm_kernel(&a, &w, ord, m, k, n);
        prop_assert!(result >= 0.0, "L{ord} norm = {result} (negative!)");
        prop_assert!(result.is_finite(), "L{ord} norm = {result} (non-finite)");
    }
}
