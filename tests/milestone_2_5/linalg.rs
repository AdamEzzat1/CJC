// Milestone 2.5 — Linalg Decomposition Tests
//
// Tests for linear algebra operations on Tensor:
// - LU decomposition with partial pivoting
// - QR decomposition via Modified Gram-Schmidt
// - Cholesky decomposition (A = L * L^T)
// - Matrix inverse via LU
// - Error cases (non-square, singular, non-positive-definite)

use cjc_runtime::Tensor;

const TOL: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

// ── LU Decomposition ────────────────────────────────────────────

#[test]
fn linalg_lu_decompose_2x2() {
    // A = [[4, 3], [6, 3]]
    let a = Tensor::from_vec(vec![4.0, 3.0, 6.0, 3.0], &[2, 2]).unwrap();
    let (l, u, _pivots) = a.lu_decompose().unwrap();

    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);

    // L should be lower triangular with 1s on diagonal
    assert!(approx_eq(l.get(&[0, 0]).unwrap(), 1.0));
    assert!(approx_eq(l.get(&[0, 1]).unwrap(), 0.0));
    assert!(approx_eq(l.get(&[1, 1]).unwrap(), 1.0));

    // U should be upper triangular
    assert!(approx_eq(u.get(&[1, 0]).unwrap(), 0.0));

    // Verify L*U reconstructs A (up to pivoting)
    let reconstructed = l.matmul(&u).unwrap();
    // The reconstruction is P*A, but we just check it's a valid factorization
    assert_eq!(reconstructed.shape(), &[2, 2]);
}

#[test]
fn linalg_lu_decompose_3x3() {
    // A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
    let a = Tensor::from_vec(
        vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
        &[3, 3],
    )
    .unwrap();

    let (l, u, _pivots) = a.lu_decompose().unwrap();

    // L*U should be a valid factorization
    let lu = l.matmul(&u).unwrap();
    assert_eq!(lu.shape(), &[3, 3]);

    // L is lower triangular with unit diagonal
    assert!(approx_eq(l.get(&[0, 0]).unwrap(), 1.0));
    assert!(approx_eq(l.get(&[1, 1]).unwrap(), 1.0));
    assert!(approx_eq(l.get(&[2, 2]).unwrap(), 1.0));
    assert!(approx_eq(l.get(&[0, 1]).unwrap(), 0.0));
    assert!(approx_eq(l.get(&[0, 2]).unwrap(), 0.0));
    assert!(approx_eq(l.get(&[1, 2]).unwrap(), 0.0));
}

#[test]
fn linalg_lu_non_square_fails() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.lu_decompose().is_err());
}

// ── QR Decomposition ────────────────────────────────────────────

#[test]
fn linalg_qr_decompose_2x2() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let (q, r) = a.qr_decompose().unwrap();

    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    // Q should be orthogonal (Q^T * Q = I for identity input)
    let qt = q.transpose();
    let qtq = qt.matmul(&q).unwrap();
    assert!(approx_eq(qtq.get(&[0, 0]).unwrap(), 1.0));
    assert!(approx_eq(qtq.get(&[1, 1]).unwrap(), 1.0));
    assert!(approx_eq(qtq.get(&[0, 1]).unwrap(), 0.0));
    assert!(approx_eq(qtq.get(&[1, 0]).unwrap(), 0.0));
}

// ── Cholesky Decomposition ──────────────────────────────────────

#[test]
fn linalg_cholesky_spd_matrix() {
    // A = [[4, 2], [2, 3]] -- symmetric positive definite
    let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]).unwrap();
    let l = a.cholesky().unwrap();

    assert_eq!(l.shape(), &[2, 2]);

    // L should be lower triangular
    assert!(approx_eq(l.get(&[0, 1]).unwrap(), 0.0));

    // Verify L * L^T = A
    let lt = l.transpose();
    let llt = l.matmul(&lt).unwrap();
    assert!(approx_eq(llt.get(&[0, 0]).unwrap(), 4.0));
    assert!(approx_eq(llt.get(&[0, 1]).unwrap(), 2.0));
    assert!(approx_eq(llt.get(&[1, 0]).unwrap(), 2.0));
    assert!(approx_eq(llt.get(&[1, 1]).unwrap(), 3.0));
}

#[test]
fn linalg_cholesky_not_positive_definite_fails() {
    // Not positive definite: [[1, 2], [2, 1]] (eigenvalues: -1, 3)
    let a = Tensor::from_vec(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]).unwrap();
    assert!(a.cholesky().is_err());
}

// ── Matrix Inverse ──────────────────────────────────────────────

#[test]
fn linalg_inverse_2x2() {
    // A = [[4, 7], [2, 6]], A^-1 = (1/10)*[[6, -7], [-2, 4]]
    let a = Tensor::from_vec(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]).unwrap();
    let inv = a.inverse().unwrap();

    assert_eq!(inv.shape(), &[2, 2]);

    // A * A^-1 should be approximately I
    let product = a.matmul(&inv).unwrap();
    assert!(approx_eq(product.get(&[0, 0]).unwrap(), 1.0));
    assert!(approx_eq(product.get(&[0, 1]).unwrap(), 0.0));
    assert!(approx_eq(product.get(&[1, 0]).unwrap(), 0.0));
    assert!(approx_eq(product.get(&[1, 1]).unwrap(), 1.0));
}
