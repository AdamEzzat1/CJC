// Milestone 2.5 — Sparse Tensor Tests
//
// Tests for the SparseCoo and SparseCsr representations:
// - COO construction and sum
// - COO -> CSR conversion
// - CSR element access and nnz
// - CSR matvec (sparse matrix-vector multiply)
// - CSR to_dense round-trip
// - Empty sparse matrix edge case

use cjc_runtime::{SparseCoo, SparseCsr};

#[test]
fn sparse_coo_construction_and_sum() {
    // 3x3 matrix with 3 non-zeros: (0,1)=2.0, (1,0)=3.0, (2,2)=5.0
    let coo = SparseCoo::new(
        vec![2.0, 3.0, 5.0],
        vec![0, 1, 2],
        vec![1, 0, 2],
        3,
        3,
    );

    assert_eq!(coo.nnz(), 3);
    assert!((coo.sum() - 10.0).abs() < 1e-12);
}

#[test]
fn sparse_coo_to_csr_conversion() {
    let coo = SparseCoo::new(
        vec![1.0, 2.0, 3.0],
        vec![0, 1, 2],     // rows
        vec![0, 1, 2],     // cols (diagonal)
        3,
        3,
    );

    let csr = coo.to_csr();
    assert_eq!(csr.nnz(), 3);
    assert_eq!(csr.nrows, 3);
    assert_eq!(csr.ncols, 3);

    // Check diagonal values
    assert_eq!(csr.get(0, 0), 1.0);
    assert_eq!(csr.get(1, 1), 2.0);
    assert_eq!(csr.get(2, 2), 3.0);

    // Zero entries
    assert_eq!(csr.get(0, 1), 0.0);
    assert_eq!(csr.get(1, 0), 0.0);
}

#[test]
fn sparse_csr_matvec() {
    // Identity-like sparse matrix: diag(2, 3, 4)
    let coo = SparseCoo::new(
        vec![2.0, 3.0, 4.0],
        vec![0, 1, 2],
        vec![0, 1, 2],
        3,
        3,
    );
    let csr = SparseCsr::from_coo(&coo);

    let x = vec![1.0, 2.0, 3.0];
    let y = csr.matvec(&x).unwrap();

    assert_eq!(y.len(), 3);
    assert!((y[0] - 2.0).abs() < 1e-12);  // 2*1
    assert!((y[1] - 6.0).abs() < 1e-12);  // 3*2
    assert!((y[2] - 12.0).abs() < 1e-12); // 4*3
}

#[test]
fn sparse_csr_matvec_dimension_mismatch() {
    let coo = SparseCoo::new(
        vec![1.0],
        vec![0],
        vec![0],
        2,
        3,
    );
    let csr = SparseCsr::from_coo(&coo);

    // x has wrong length (2 instead of 3)
    let result = csr.matvec(&vec![1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn sparse_csr_to_dense_round_trip() {
    // Build a 3x4 sparse matrix with a few entries
    let coo = SparseCoo::new(
        vec![1.0, 5.0, 9.0],
        vec![0, 1, 2],
        vec![0, 2, 3],
        3,
        4,
    );
    let csr = SparseCsr::from_coo(&coo);

    let dense = csr.to_dense();
    assert_eq!(dense.shape(), &[3, 4]);
    assert_eq!(dense.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(dense.get(&[0, 1]).unwrap(), 0.0);
    assert_eq!(dense.get(&[1, 2]).unwrap(), 5.0);
    assert_eq!(dense.get(&[2, 3]).unwrap(), 9.0);
    assert_eq!(dense.get(&[2, 0]).unwrap(), 0.0);
}

#[test]
fn sparse_empty_matrix() {
    let coo = SparseCoo::new(vec![], vec![], vec![], 5, 5);
    assert_eq!(coo.nnz(), 0);
    assert!((coo.sum() - 0.0).abs() < 1e-15);

    let csr = coo.to_csr();
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.get(0, 0), 0.0);
    assert_eq!(csr.get(4, 4), 0.0);
}
