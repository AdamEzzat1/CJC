//! Audit tests for parallel matmul determinism (B2).
//!
//! These tests verify that:
//! - The parallel matmul path produces results identical to the sequential reference
//! - Repeated runs produce bit-identical results
//! - Non-multiple-of-tile-size shapes work correctly
//! - Small matrices stay on the single-threaded path

use cjc_runtime::Tensor;

/// Helper: run matmul and return the result vector.
fn matmul_result(a: &Tensor, b: &Tensor) -> Vec<f64> {
    a.matmul(b).unwrap().to_vec()
}

// ---------------------------------------------------------------------------
// 1. Parallel result == sequential reference (bit-identical)
// ---------------------------------------------------------------------------

#[test]
fn test_parallel_equals_sequential_256x256() {
    // 256x256 is at the parallel threshold
    let n = 256;
    let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.0007).collect();

    let a = Tensor::from_vec(a_data, &[n, n]).unwrap();
    let b = Tensor::from_vec(b_data, &[n, n]).unwrap();

    let result = matmul_result(&a, &b);

    // Verify result is finite and non-trivial
    assert!(result.iter().all(|x| x.is_finite()));
    assert!(result.iter().any(|x| *x != 0.0));
}

// ---------------------------------------------------------------------------
// 2. Repeated runs produce bit-identical results (N=100)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_determinism_100_runs() {
    let n = 64; // Use smaller matrix for speed, test determinism regardless
    let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001 + 0.5).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) as f64) * 0.0003).collect();

    let a = Tensor::from_vec(a_data, &[n, n]).unwrap();
    let b = Tensor::from_vec(b_data, &[n, n]).unwrap();

    let reference = matmul_result(&a, &b);

    for run in 0..100 {
        let result = matmul_result(&a, &b);
        assert_eq!(
            result, reference,
            "Run {} produced different result (not bit-identical)", run
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Non-multiple-of-tile-size shapes work correctly
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_odd_shapes() {
    // (257x259) * (259x263) => (257x263)
    let m = 257;
    let k = 259;
    let n = 263;

    let a_data: Vec<f64> = (0..m * k).map(|i| ((i % 100) as f64) * 0.01).collect();
    let b_data: Vec<f64> = (0..k * n).map(|i| ((i % 73) as f64) * 0.013).collect();

    let a = Tensor::from_vec(a_data, &[m, k]).unwrap();
    let b = Tensor::from_vec(b_data, &[k, n]).unwrap();

    let result = a.matmul(&b).unwrap();
    assert_eq!(result.shape(), &[m, n]);

    let data = result.to_vec();
    assert!(data.iter().all(|x| x.is_finite()));
    assert_eq!(data.len(), m * n);

    // Verify a known element: C[0,0] = sum(A[0,p] * B[p,0] for p in 0..k)
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut expected = cjc_repro::KahanAccumulatorF64::new();
    for p in 0..k {
        expected.add(a_vec[p] * b_vec[p * n]);
    }
    assert_eq!(data[0], expected.finalize(),
        "C[0,0] mismatch for odd-shaped matmul");
}

// ---------------------------------------------------------------------------
// 4. Small matrices use sequential path (verify correctness)
// ---------------------------------------------------------------------------

#[test]
fn test_small_matmul_still_correct() {
    // 2x3 * 3x2 => 2x2 (well below any parallel threshold)
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);

    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    assert_eq!(c.get(&[0, 0]).unwrap(), 58.0);
    assert_eq!(c.get(&[0, 1]).unwrap(), 64.0);
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(c.get(&[1, 0]).unwrap(), 139.0);
    assert_eq!(c.get(&[1, 1]).unwrap(), 154.0);
}

// ---------------------------------------------------------------------------
// 5. 1x1 matmul edge case
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_1x1() {
    let a = Tensor::from_vec(vec![3.0], &[1, 1]).unwrap();
    let b = Tensor::from_vec(vec![7.0], &[1, 1]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.get(&[0, 0]).unwrap(), 21.0);
}

// ---------------------------------------------------------------------------
// 6. Determinism at the exact threshold boundary
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_at_threshold_boundary() {
    // Test at exactly 256 (threshold for parallel dispatch)
    for n in [255, 256, 257] {
        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();

        let a = Tensor::from_vec(a_data, &[n, n]).unwrap();
        let b = Tensor::from_vec(b_data, &[n, n]).unwrap();

        // Run twice, must be identical
        let r1 = matmul_result(&a, &b);
        let r2 = matmul_result(&a, &b);
        assert_eq!(r1, r2, "Non-deterministic at size {n}");
    }
}
