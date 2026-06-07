//! Property tests — `fused_matmul_dot` equals the unfused chain bit-for-bit.
//!
//! The core determinism guarantee: for ANY (a, W, v) tuple where matmul takes
//! the sequential path (every dim < 64 in `Tensor::matmul`), the fused kernel
//! produces byte-identical output to `dot(matmul(a_as_row, W).flatten(), v)`.
//!
//! We test the kernel-level helper directly (not the dispatch surface) so the
//! property holds even when we vary shapes a lot.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::accumulator::{binned_sum_f64, fused_matmul_dot_kernel};
use proptest::prelude::*;

/// SplitMix64 (same mix as `cjc_repro::Rng`) for deterministic test data.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic finite floats in [-4, 4] (well-conditioned for the kernels).
fn det_floats(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| (splitmix64(&mut s) as f64 / u64::MAX as f64) * 8.0 - 4.0)
        .collect()
}

/// Unfused reference: sequential Kahan matmul (matches matmul_sequential),
/// then binned dot (matches the `dot` builtin).
fn unfused_reference(a: &[f64], w: &[f64], v: &[f64], m: usize, n: usize) -> f64 {
    let mut intermediate = Vec::with_capacity(n);
    for j in 0..n {
        let mut k = KahanAccumulatorF64::new();
        for i in 0..m {
            k.add(a[i] * w[i * n + j]);
        }
        intermediate.push(k.finalize());
    }
    let products: Vec<f64> = intermediate
        .iter()
        .zip(v.iter())
        .map(|(x, y)| x * y)
        .collect();
    binned_sum_f64(&products)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// The headline guarantee: fused == unfused, bit-for-bit, across any
    /// well-conditioned (a, W, v) shape we throw at it.
    #[test]
    fn fused_equals_unfused_bit_identical(
        m in 1usize..32,
        n in 1usize..32,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m);
        let w = det_floats(seed.wrapping_add(1), m * n);
        let v = det_floats(seed.wrapping_add(2), n);

        let fused = fused_matmul_dot_kernel(&a, &w, &v, m, n);
        let unfused = unfused_reference(&a, &w, &v, m, n);

        prop_assert_eq!(
            fused.to_bits(),
            unfused.to_bits(),
            "fused vs unfused diverged at shape ({}, {}) seed {}: fused={:e}, unfused={:e}",
            m, n, seed, fused, unfused
        );
    }

    /// The fused kernel is deterministic — same input → same output, across
    /// any number of calls.
    #[test]
    fn fused_is_deterministic(
        m in 1usize..16,
        n in 1usize..16,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m);
        let w = det_floats(seed.wrapping_add(1), m * n);
        let v = det_floats(seed.wrapping_add(2), n);

        let first = fused_matmul_dot_kernel(&a, &w, &v, m, n);
        for _ in 0..4 {
            let again = fused_matmul_dot_kernel(&a, &w, &v, m, n);
            prop_assert_eq!(first.to_bits(), again.to_bits());
        }
    }

    /// Result is finite when all inputs are finite (no spurious NaN/Inf).
    #[test]
    fn fused_is_finite_for_well_conditioned_inputs(
        m in 1usize..16,
        n in 1usize..16,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m);
        let w = det_floats(seed.wrapping_add(1), m * n);
        let v = det_floats(seed.wrapping_add(2), n);
        let result = fused_matmul_dot_kernel(&a, &w, &v, m, n);
        prop_assert!(result.is_finite(), "got {result} for shape ({m}, {n})");
    }
}
