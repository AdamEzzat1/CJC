//! Property tests — `fused_matmul_matmul` equals the unfused chain bit-for-bit
//! across random shapes and inputs.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::accumulator::fused_matmul_matmul_kernel;
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

fn unfused_reference(a: &[f64], b: &[f64], c: &[f64], m: usize, n: usize, k: usize, p: usize) -> Vec<f64> {
    let mut intermediate = vec![0.0f64; m * k];
    for i in 0..m {
        for q in 0..k {
            let mut acc = KahanAccumulatorF64::new();
            for l in 0..n {
                acc.add(a[i * n + l] * b[l * k + q]);
            }
            intermediate[i * k + q] = acc.finalize();
        }
    }
    let mut result = vec![0.0f64; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut acc = KahanAccumulatorF64::new();
            for q in 0..k {
                acc.add(intermediate[i * k + q] * c[q * p + j]);
            }
            result[i * p + j] = acc.finalize();
        }
    }
    result
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    /// Headline: fused == unfused, bit-for-bit, across random shapes.
    #[test]
    fn fused_equals_unfused_bit_identical(
        m in 1usize..12,
        n in 1usize..12,
        k in 1usize..12,
        p in 1usize..12,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m * n);
        let b = det_floats(seed.wrapping_add(1), n * k);
        let c = det_floats(seed.wrapping_add(2), k * p);
        let fused = fused_matmul_matmul_kernel(&a, &b, &c, m, n, k, p);
        let unfused = unfused_reference(&a, &b, &c, m, n, k, p);
        prop_assert_eq!(fused.len(), unfused.len());
        for (i, (x, y)) in fused.iter().zip(unfused.iter()).enumerate() {
            prop_assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "shape ({}, {}, {}, {}) seed {} element {} differs",
                m, n, k, p, seed, i
            );
        }
    }

    /// Determinism — repeated calls produce identical output.
    #[test]
    fn fused_is_deterministic(
        m in 1usize..8,
        n in 1usize..8,
        k in 1usize..8,
        p in 1usize..8,
        seed in any::<u64>(),
    ) {
        let a = det_floats(seed, m * n);
        let b = det_floats(seed.wrapping_add(1), n * k);
        let c = det_floats(seed.wrapping_add(2), k * p);
        let first = fused_matmul_matmul_kernel(&a, &b, &c, m, n, k, p);
        for _ in 0..4 {
            let again = fused_matmul_matmul_kernel(&a, &b, &c, m, n, k, p);
            prop_assert_eq!(first.len(), again.len());
            for (x, y) in first.iter().zip(again.iter()) {
                prop_assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}
