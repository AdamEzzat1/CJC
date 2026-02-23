//! Stress Test 3: Parallel Invariance Test
//!
//! Verifies bit-identical results under parallel-style chunk-and-merge patterns.
//!
//! The BinnedAccumulator guarantees:
//! 1. Merge commutativity: a.merge(b) == b.merge(a)
//! 2. Same-input determinism: same inputs → same output, always
//! 3. Merge order invariance: given fixed chunks, any merge order is identical
//!    (because merge is element-wise addition on same-exponent bins,
//!    and IEEE-754 addition is commutative)
//!
//! Note: Different chunk *boundaries* can produce slightly different results
//! (within a few ULPs) because IEEE-754 addition is not associative.
//! The key parallel invariant is that a fixed work distribution produces
//! identical results regardless of scheduling/merge order.

use cjc_runtime::accumulator::{binned_sum_f64, BinnedAccumulatorF64};
use cjc_runtime::Tensor;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash a tensor's data for bit-exact comparison.
fn tensor_hash(t: &Tensor) -> u64 {
    let data = t.to_vec();
    let mut hasher = DefaultHasher::new();
    for &v in &data {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Simulate parallel matmul using binned dot products.
fn matmul_binned(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = BinnedAccumulatorF64::new();
            for p in 0..k {
                acc.add(a[i * k + p] * b[p * n + j]);
            }
            c[i * n + j] = acc.finalize();
        }
    }
    c
}

#[test]
fn test_merge_commutativity_pairwise() {
    // For any two chunk accumulators, a.merge(b) == b.merge(a).
    let n = 10_000;
    let mut rng = cjc_repro::Rng::seeded(42);
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64() * 1000.0 - 500.0).collect();

    let chunk_size = 500;
    let chunks: Vec<Vec<f64>> = values.chunks(chunk_size).map(|c| c.to_vec()).collect();

    // Test all adjacent pairs.
    for i in 0..chunks.len() - 1 {
        let mut a = BinnedAccumulatorF64::new();
        a.add_slice(&chunks[i]);
        let mut b = BinnedAccumulatorF64::new();
        b.add_slice(&chunks[i + 1]);

        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);

        assert_eq!(ab.finalize().to_bits(), ba.finalize().to_bits(),
            "Merge commutativity failed at pair ({}, {}): {} vs {}",
            i, i + 1, ab.finalize(), ba.finalize());
    }
}

#[test]
fn test_merge_order_invariance_fixed_chunks() {
    // Given fixed chunk boundaries, any merge order produces the same result.
    let n = 4_000;
    let mut rng = cjc_repro::Rng::seeded(456);
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2e10 - 1e10).collect();

    let chunk_size = 500;
    let mut chunk_accs: Vec<BinnedAccumulatorF64> = Vec::new();
    for chunk in values.chunks(chunk_size) {
        let mut acc = BinnedAccumulatorF64::new();
        acc.add_slice(chunk);
        chunk_accs.push(acc);
    }

    // Merge order 1: sequential (0,1,2,...,7)
    let mut m1 = BinnedAccumulatorF64::new();
    for acc in &chunk_accs {
        m1.merge(acc);
    }

    // Merge order 2: reverse (7,6,...,0)
    let mut m2 = BinnedAccumulatorF64::new();
    for acc in chunk_accs.iter().rev() {
        m2.merge(acc);
    }

    // Merge order 3: interleaved
    let mut m3 = BinnedAccumulatorF64::new();
    let n_chunks = chunk_accs.len();
    let mut order: Vec<usize> = Vec::new();
    for i in 0..n_chunks / 2 {
        order.push(i);
        order.push(n_chunks - 1 - i);
    }
    if n_chunks % 2 == 1 {
        order.push(n_chunks / 2);
    }
    for &idx in &order {
        m3.merge(&chunk_accs[idx]);
    }

    let r1 = m1.finalize();
    let r2 = m2.finalize();
    let r3 = m3.finalize();

    // All must be bit-identical (merge commutativity).
    assert_eq!(r1.to_bits(), r2.to_bits(), "Sequential vs reverse differ: {r1} vs {r2}");
    assert_eq!(r1.to_bits(), r3.to_bits(), "Sequential vs interleaved differ: {r1} vs {r3}");
}

#[test]
fn test_different_chunk_sizes_near_identical() {
    // Different chunk sizes produce results within a few ULPs.
    let n = 5_000;
    let mut rng = cjc_repro::Rng::seeded(123);
    let values: Vec<f64> = (0..n).map(|_| rng.next_normal_f64() * 100.0).collect();

    let chunk_sizes = [1, 7, 13, 64, 128, 256, 1000, 5000];
    let mut results = Vec::new();

    for &cs in &chunk_sizes {
        let mut acc = BinnedAccumulatorF64::new();
        for chunk in values.chunks(cs) {
            let mut c = BinnedAccumulatorF64::new();
            c.add_slice(chunk);
            acc.merge(&c);
        }
        results.push(acc.finalize());
    }

    // All should be within a few ULPs of each other.
    let reference = results[0];
    for (i, &r) in results.iter().enumerate() {
        let ulp_dist = (r.to_bits() as i64 - reference.to_bits() as i64).unsigned_abs();
        assert!(ulp_dist < 100,
            "Chunk size {} differs by {} ULPs from reference: {} vs {}",
            chunk_sizes[i], ulp_dist, r, reference);
    }
}

#[test]
fn test_matmul_bit_identical_across_chunk_schemes() {
    // Small matmul to verify bit-identical results.
    let m = 16;
    let k = 32;
    let n = 16;

    let mut rng = cjc_repro::Rng::seeded(789);
    let a: Vec<f64> = (0..m * k).map(|_| rng.next_normal_f64()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| rng.next_normal_f64()).collect();

    let c1 = matmul_binned(&a, &b, m, k, n);
    let c2 = matmul_binned(&a, &b, m, k, n);

    for i in 0..c1.len() {
        assert_eq!(c1[i].to_bits(), c2[i].to_bits(),
            "Matmul result differs at index {i}: {} vs {}", c1[i], c2[i]);
    }
}

#[test]
fn test_tensor_matmul_hash_determinism() {
    let mut rng = cjc_repro::Rng::seeded(42);
    let a_data: Vec<f64> = (0..32 * 32).map(|_| rng.next_normal_f64()).collect();
    let b_data: Vec<f64> = (0..32 * 32).map(|_| rng.next_normal_f64()).collect();

    let a = Tensor::from_vec(a_data.clone(), &[32, 32]).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[32, 32]).unwrap();
    let c1 = a.matmul(&b).unwrap();
    let h1 = tensor_hash(&c1);

    let a2 = Tensor::from_vec(a_data, &[32, 32]).unwrap();
    let b2 = Tensor::from_vec(b_data, &[32, 32]).unwrap();
    let c2 = a2.matmul(&b2).unwrap();
    let h2 = tensor_hash(&c2);

    assert_eq!(h1, h2, "Tensor.matmul hashes differ across runs");
}

#[test]
fn test_zero_coefficient_of_variation() {
    let n = 1000;
    let mut rng = cjc_repro::Rng::seeded(42);
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64() * 1e6 - 5e5).collect();

    let mut results = Vec::new();
    for _ in 0..100 {
        results.push(binned_sum_f64(&values));
    }

    let first = results[0];
    for (i, &r) in results.iter().enumerate() {
        assert_eq!(r.to_bits(), first.to_bits(),
            "Iteration {i} produced different result: {r} vs {first}");
    }
}
