//! Hardening test H10: TiledMatmul integration into Tensor::matmul.
//!
//! Verifies that:
//! - Matrices with any dimension >= 64 route through the tiled path
//! - Small matrices stay on the sequential (Kahan) path
//! - Tiled results match sequential within acceptable tolerance
//! - The integration is deterministic
//! - Tensor::matmul still works correctly after the routing change

use cjc_runtime::tensor::Tensor;

// ── Threshold routing tests ─────────────────────────────────────────

#[test]
fn h10_small_matrix_sequential_path() {
    // 4x4 matmul — should use sequential Kahan path (all dims < 64).
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[4, 4],
    ).unwrap();
    let eye = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0],
        &[4, 4],
    ).unwrap();
    let c = a.matmul(&eye).unwrap();
    assert_eq!(c.to_vec(), a.to_vec(), "A * I = A for small matrix");
}

#[test]
fn h10_tiled_threshold_triggers() {
    // 64x64 matmul — should route through tiled path (m == 64).
    let n = 64;
    let a_data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.01).collect();
    let eye_data: Vec<f64> = (0..n * n)
        .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
        .collect();

    let a = Tensor::from_vec(a_data.clone(), &[n, n]).unwrap();
    let eye = Tensor::from_vec(eye_data, &[n, n]).unwrap();
    let c = a.matmul(&eye).unwrap();

    // A * I should equal A within floating-point tolerance.
    for (i, (got, expected)) in c.to_vec().iter().zip(a_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-10,
            "mismatch at [{}, {}]: got={got}, expected={expected}",
            i / n, i % n,
        );
    }
}

#[test]
fn h10_tiled_triggers_on_single_large_dim() {
    // 2x64 * 64x2 — should trigger tiled path (k == 64).
    let m = 2;
    let k = 64;
    let n = 2;

    let a_data: Vec<f64> = (0..m * k).map(|i| (i + 1) as f64).collect();
    let b_data: Vec<f64> = (0..k * n).map(|i| (i + 1) as f64 * 0.01).collect();

    let a = Tensor::from_vec(a_data, &[m, k]).unwrap();
    let b = Tensor::from_vec(b_data, &[k, n]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.shape(), &[m, n]);
    // Verify the result is finite and non-zero.
    let result = c.to_vec();
    assert!(result.iter().all(|v| v.is_finite()), "all results finite");
    assert!(result.iter().any(|v| *v != 0.0), "results non-zero");
}

// ── Correctness tests ───────────────────────────────────────────────

#[test]
fn h10_tiled_vs_sequential_within_tolerance() {
    // Compare the tiled path result against a manually computed sequential result
    // for a 64x64 matrix multiplication.
    let n = 64;
    let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.01).collect();

    let a = Tensor::from_vec(a_data.clone(), &[n, n]).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[n, n]).unwrap();
    let c = a.matmul(&b).unwrap();

    // Compute reference result using naive triple-loop.
    let mut expected = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..n {
                sum += a_data[i * n + p] * b_data[p * n + j];
            }
            expected[i * n + j] = sum;
        }
    }

    let result = c.to_vec();
    for i in 0..n * n {
        assert!(
            (result[i] - expected[i]).abs() < 1e-8,
            "mismatch at [{}, {}]: tiled={}, naive={}",
            i / n, i % n, result[i], expected[i],
        );
    }
}

// ── Determinism tests ───────────────────────────────────────────────

#[test]
fn h10_tiled_deterministic() {
    // Same inputs must produce bit-identical outputs.
    let n = 64;
    let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.1).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.1).collect();

    let a1 = Tensor::from_vec(a_data.clone(), &[n, n]).unwrap();
    let b1 = Tensor::from_vec(b_data.clone(), &[n, n]).unwrap();
    let c1 = a1.matmul(&b1).unwrap();

    let a2 = Tensor::from_vec(a_data, &[n, n]).unwrap();
    let b2 = Tensor::from_vec(b_data, &[n, n]).unwrap();
    let c2 = a2.matmul(&b2).unwrap();

    assert_eq!(c1.to_vec(), c2.to_vec(), "tiled matmul must be deterministic");
}

// ── Regression tests ────────────────────────────────────────────────

#[test]
fn h10_small_matmul_unchanged() {
    // Ensure small matrix results are exactly the same as before the tiled path.
    // 2x3 * 3x2 — well below the 64 threshold.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    // Expected: [1*7+2*9+3*11  1*8+2*10+3*12] = [58  64]
    //           [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
    assert_eq!(c.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
}
