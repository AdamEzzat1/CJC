use std::collections::BTreeMap;

use cjc_repro::kahan_sum_f64;

use crate::accumulator::binned_sum_f64;
use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 4. Sparse Tensor Representations (CSR + COO)
// ---------------------------------------------------------------------------

/// Compressed Sparse Row (CSR) matrix representation.
#[derive(Debug, Clone)]
pub struct SparseCsr {
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_offsets: Vec<usize>, // length = nrows + 1
    pub nrows: usize,
    pub ncols: usize,
}

impl SparseCsr {
    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Access element at (row, col). Returns 0.0 for zero entries.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        for idx in start..end {
            if self.col_indices[idx] == col {
                return self.values[idx];
            }
        }
        0.0
    }

    /// Sparse matrix-vector multiplication: y = A * x.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, RuntimeError> {
        if x.len() != self.ncols {
            return Err(RuntimeError::DimensionMismatch {
                expected: self.ncols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0f64; self.nrows];
        for row in 0..self.nrows {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            let products: Vec<f64> = (start..end)
                .map(|idx| self.values[idx] * x[self.col_indices[idx]])
                .collect();
            y[row] = kahan_sum_f64(&products);
        }
        Ok(y)
    }

    /// Convert to dense Tensor.
    pub fn to_dense(&self) -> Tensor {
        let mut data = vec![0.0f64; self.nrows * self.ncols];
        for row in 0..self.nrows {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for idx in start..end {
                data[row * self.ncols + self.col_indices[idx]] = self.values[idx];
            }
        }
        Tensor::from_vec(data, &[self.nrows, self.ncols]).unwrap()
    }

    /// Construct CSR from COO data.
    pub fn from_coo(coo: &SparseCoo) -> Self {
        // Sort by row, then by column
        let nnz = coo.values.len();
        let mut order: Vec<usize> = (0..nnz).collect();
        order.sort_by_key(|&i| (coo.row_indices[i], coo.col_indices[i]));

        let mut values = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut row_offsets = vec![0usize; coo.nrows + 1];

        for &i in &order {
            values.push(coo.values[i]);
            col_indices.push(coo.col_indices[i]);
            row_offsets[coo.row_indices[i] + 1] += 1;
        }

        // Cumulative sum for row_offsets
        for i in 1..=coo.nrows {
            row_offsets[i] += row_offsets[i - 1];
        }

        SparseCsr {
            values,
            col_indices,
            row_offsets,
            nrows: coo.nrows,
            ncols: coo.ncols,
        }
    }
}

/// Coordinate (COO) sparse matrix representation.
#[derive(Debug, Clone)]
pub struct SparseCoo {
    pub values: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

impl SparseCoo {
    pub fn new(
        values: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        SparseCoo {
            values,
            row_indices,
            col_indices,
            nrows,
            ncols,
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn to_csr(&self) -> SparseCsr {
        SparseCsr::from_coo(self)
    }

    pub fn sum(&self) -> f64 {
        kahan_sum_f64(&self.values)
    }
}

// ---------------------------------------------------------------------------
// Sparse Arithmetic Operations
// ---------------------------------------------------------------------------

/// Helper: merge two sorted CSR rows element-wise using a combiner function.
/// Returns (values, col_indices) for the merged row, dropping exact zeros.
fn merge_rows(
    a_vals: &[f64],
    a_cols: &[usize],
    b_vals: &[f64],
    b_cols: &[usize],
    combine: fn(f64, f64) -> f64,
    default_a: f64,
    default_b: f64,
) -> (Vec<f64>, Vec<usize>) {
    let mut values = Vec::new();
    let mut cols = Vec::new();
    let mut ia = 0;
    let mut ib = 0;

    while ia < a_cols.len() && ib < b_cols.len() {
        match a_cols[ia].cmp(&b_cols[ib]) {
            std::cmp::Ordering::Less => {
                let v = combine(a_vals[ia], default_b);
                if v != 0.0 {
                    values.push(v);
                    cols.push(a_cols[ia]);
                }
                ia += 1;
            }
            std::cmp::Ordering::Greater => {
                let v = combine(default_a, b_vals[ib]);
                if v != 0.0 {
                    values.push(v);
                    cols.push(b_cols[ib]);
                }
                ib += 1;
            }
            std::cmp::Ordering::Equal => {
                let v = combine(a_vals[ia], b_vals[ib]);
                if v != 0.0 {
                    values.push(v);
                    cols.push(a_cols[ia]);
                }
                ia += 1;
                ib += 1;
            }
        }
    }
    while ia < a_cols.len() {
        let v = combine(a_vals[ia], default_b);
        if v != 0.0 {
            values.push(v);
            cols.push(a_cols[ia]);
        }
        ia += 1;
    }
    while ib < b_cols.len() {
        let v = combine(default_a, b_vals[ib]);
        if v != 0.0 {
            values.push(v);
            cols.push(b_cols[ib]);
        }
        ib += 1;
    }
    (values, cols)
}

/// Apply an element-wise binary operation on two CSR matrices of the same dimensions.
fn sparse_binop(
    a: &SparseCsr,
    b: &SparseCsr,
    combine: fn(f64, f64) -> f64,
    default_a: f64,
    default_b: f64,
    op_name: &str,
) -> Result<SparseCsr, String> {
    if a.nrows != b.nrows || a.ncols != b.ncols {
        return Err(format!(
            "sparse_{}: dimension mismatch: ({}, {}) vs ({}, {})",
            op_name, a.nrows, a.ncols, b.nrows, b.ncols
        ));
    }

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::with_capacity(a.nrows + 1);
    row_offsets.push(0);

    for row in 0..a.nrows {
        let a_start = a.row_offsets[row];
        let a_end = a.row_offsets[row + 1];
        let b_start = b.row_offsets[row];
        let b_end = b.row_offsets[row + 1];

        let (rv, rc) = merge_rows(
            &a.values[a_start..a_end],
            &a.col_indices[a_start..a_end],
            &b.values[b_start..b_end],
            &b.col_indices[b_start..b_end],
            combine,
            default_a,
            default_b,
        );
        values.extend_from_slice(&rv);
        col_indices.extend_from_slice(&rc);
        row_offsets.push(values.len());
    }

    Ok(SparseCsr {
        values,
        col_indices,
        row_offsets,
        nrows: a.nrows,
        ncols: a.ncols,
    })
}

/// Element-wise addition of two CSR matrices (same dimensions).
pub fn sparse_add(a: &SparseCsr, b: &SparseCsr) -> Result<SparseCsr, String> {
    sparse_binop(a, b, |x, y| x + y, 0.0, 0.0, "add")
}

/// Element-wise subtraction of two CSR matrices (same dimensions).
pub fn sparse_sub(a: &SparseCsr, b: &SparseCsr) -> Result<SparseCsr, String> {
    sparse_binop(a, b, |x, y| x - y, 0.0, 0.0, "sub")
}

/// Element-wise multiplication (Hadamard product) of two CSR matrices.
/// Only positions where BOTH matrices have non-zeros produce non-zeros.
pub fn sparse_mul(a: &SparseCsr, b: &SparseCsr) -> Result<SparseCsr, String> {
    if a.nrows != b.nrows || a.ncols != b.ncols {
        return Err(format!(
            "sparse_mul: dimension mismatch: ({}, {}) vs ({}, {})",
            a.nrows, a.ncols, b.nrows, b.ncols
        ));
    }

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::with_capacity(a.nrows + 1);
    row_offsets.push(0);

    for row in 0..a.nrows {
        let a_start = a.row_offsets[row];
        let a_end = a.row_offsets[row + 1];
        let b_start = b.row_offsets[row];
        let b_end = b.row_offsets[row + 1];

        let mut ia = a_start;
        let mut ib = b_start;

        // Only emit where both have entries (Hadamard)
        while ia < a_end && ib < b_end {
            match a.col_indices[ia].cmp(&b.col_indices[ib]) {
                std::cmp::Ordering::Less => ia += 1,
                std::cmp::Ordering::Greater => ib += 1,
                std::cmp::Ordering::Equal => {
                    let v = a.values[ia] * b.values[ib];
                    if v != 0.0 {
                        values.push(v);
                        col_indices.push(a.col_indices[ia]);
                    }
                    ia += 1;
                    ib += 1;
                }
            }
        }
        row_offsets.push(values.len());
    }

    Ok(SparseCsr {
        values,
        col_indices,
        row_offsets,
        nrows: a.nrows,
        ncols: a.ncols,
    })
}

/// Sparse matrix-matrix multiplication (SpGEMM): C = A * B.
/// Uses row-wise accumulation with BTreeMap for deterministic column ordering.
/// All floating-point reductions use binned summation.
pub fn sparse_matmul(a: &SparseCsr, b: &SparseCsr) -> Result<SparseCsr, String> {
    if a.ncols != b.nrows {
        return Err(format!(
            "sparse_matmul: inner dimension mismatch: A is ({}, {}), B is ({}, {})",
            a.nrows, a.ncols, b.nrows, b.ncols
        ));
    }

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::with_capacity(a.nrows + 1);
    row_offsets.push(0);

    for row in 0..a.nrows {
        // Accumulate contributions into a BTreeMap for deterministic column order.
        let mut accum: BTreeMap<usize, Vec<f64>> = BTreeMap::new();

        let a_start = a.row_offsets[row];
        let a_end = a.row_offsets[row + 1];

        for a_idx in a_start..a_end {
            let k = a.col_indices[a_idx];
            let a_val = a.values[a_idx];

            let b_start = b.row_offsets[k];
            let b_end = b.row_offsets[k + 1];

            for b_idx in b_start..b_end {
                let j = b.col_indices[b_idx];
                accum.entry(j).or_default().push(a_val * b.values[b_idx]);
            }
        }

        // BTreeMap iterates in sorted column order (deterministic)
        for (col, terms) in &accum {
            let v = binned_sum_f64(&terms);
            if v != 0.0 {
                col_indices.push(*col);
                values.push(v);
            }
        }
        row_offsets.push(values.len());
    }

    Ok(SparseCsr {
        values,
        col_indices,
        row_offsets,
        nrows: a.nrows,
        ncols: b.ncols,
    })
}

/// Scalar multiplication: every non-zero element is multiplied by `s`.
pub fn sparse_scalar_mul(a: &SparseCsr, s: f64) -> SparseCsr {
    let values: Vec<f64> = a.values.iter().map(|&v| v * s).collect();
    SparseCsr {
        values,
        col_indices: a.col_indices.clone(),
        row_offsets: a.row_offsets.clone(),
        nrows: a.nrows,
        ncols: a.ncols,
    }
}

/// Transpose a CSR matrix. Returns a new CSR where rows and columns are swapped.
pub fn sparse_transpose(a: &SparseCsr) -> SparseCsr {
    // Build COO in (col, row) order, then convert to CSR of transposed shape.
    let mut row_counts = vec![0usize; a.ncols + 1];

    // Count entries per column of A (= per row of A^T)
    for &c in &a.col_indices {
        row_counts[c + 1] += 1;
    }
    // Prefix sum
    for i in 1..=a.ncols {
        row_counts[i] += row_counts[i - 1];
    }

    let nnz = a.values.len();
    let mut new_values = vec![0.0f64; nnz];
    let mut new_col_indices = vec![0usize; nnz];
    let mut cursor = row_counts.clone();

    for row in 0..a.nrows {
        let start = a.row_offsets[row];
        let end = a.row_offsets[row + 1];
        for idx in start..end {
            let col = a.col_indices[idx];
            let dest = cursor[col];
            new_values[dest] = a.values[idx];
            new_col_indices[dest] = row;
            cursor[col] += 1;
        }
    }

    SparseCsr {
        values: new_values,
        col_indices: new_col_indices,
        row_offsets: row_counts,
        nrows: a.ncols,
        ncols: a.nrows,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small CSR matrix from dense data.
    fn csr_from_dense(data: &[f64], nrows: usize, ncols: usize) -> SparseCsr {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_offsets = vec![0usize];

        for r in 0..nrows {
            for c in 0..ncols {
                let v = data[r * ncols + c];
                if v != 0.0 {
                    values.push(v);
                    col_indices.push(c);
                }
            }
            row_offsets.push(values.len());
        }

        SparseCsr { values, col_indices, row_offsets, nrows, ncols }
    }

    // -- sparse_add --

    #[test]
    fn test_sparse_add_basic() {
        let a = csr_from_dense(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 5.0], 3, 3);
        let b = csr_from_dense(&[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0], 3, 3);
        let c = sparse_add(&a, &b).unwrap();
        // Dense result: [1,1,2, 2,3,3, 4,4,10]
        for r in 0..3 {
            for col in 0..3 {
                let expected = a.get(r, col) + b.get(r, col);
                assert_eq!(c.get(r, col), expected, "mismatch at ({}, {})", r, col);
            }
        }
    }

    #[test]
    fn test_sparse_add_a_plus_a_eq_2a() {
        let a = csr_from_dense(&[1.0, 2.0, 0.0, 3.0], 2, 2);
        let sum = sparse_add(&a, &a).unwrap();
        let doubled = sparse_scalar_mul(&a, 2.0);
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(sum.get(r, c), doubled.get(r, c));
            }
        }
    }

    #[test]
    fn test_sparse_add_dimension_mismatch() {
        let a = csr_from_dense(&[1.0, 2.0], 1, 2);
        let b = csr_from_dense(&[1.0, 2.0, 3.0], 1, 3);
        assert!(sparse_add(&a, &b).is_err());
    }

    // -- sparse_sub --

    #[test]
    fn test_sparse_sub_basic() {
        let a = csr_from_dense(&[5.0, 3.0, 0.0, 1.0], 2, 2);
        let b = csr_from_dense(&[2.0, 3.0, 1.0, 0.0], 2, 2);
        let c = sparse_sub(&a, &b).unwrap();
        assert_eq!(c.get(0, 0), 3.0);
        assert_eq!(c.get(0, 1), 0.0); // 3 - 3 = 0, should be dropped
        assert_eq!(c.get(1, 0), -1.0);
        assert_eq!(c.get(1, 1), 1.0);
    }

    #[test]
    fn test_sparse_sub_self_is_zero() {
        let a = csr_from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let c = sparse_sub(&a, &a).unwrap();
        assert_eq!(c.nnz(), 0);
    }

    // -- sparse_mul (Hadamard) --

    #[test]
    fn test_sparse_mul_hadamard() {
        let a = csr_from_dense(&[1.0, 0.0, 3.0, 4.0], 2, 2);
        let b = csr_from_dense(&[2.0, 5.0, 0.0, 3.0], 2, 2);
        let c = sparse_mul(&a, &b).unwrap();
        assert_eq!(c.get(0, 0), 2.0);  // 1*2
        assert_eq!(c.get(0, 1), 0.0);  // one is zero
        assert_eq!(c.get(1, 0), 0.0);  // one is zero
        assert_eq!(c.get(1, 1), 12.0); // 4*3
    }

    // -- sparse_matmul --

    #[test]
    fn test_sparse_matmul_identity() {
        // A * I = A
        let a = csr_from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let eye = csr_from_dense(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let c = sparse_matmul(&a, &eye).unwrap();
        for r in 0..2 {
            for col in 0..2 {
                assert_eq!(c.get(r, col), a.get(r, col));
            }
        }
    }

    #[test]
    fn test_sparse_matmul_vs_dense() {
        // Compare sparse matmul against dense result for a small case
        let a_data = [1.0, 2.0, 0.0, 0.0, 3.0, 4.0];
        let b_data = [5.0, 0.0, 6.0, 7.0, 0.0, 8.0];
        let a = csr_from_dense(&a_data, 2, 3);
        let b = csr_from_dense(&b_data, 3, 2);

        let c = sparse_matmul(&a, &b).unwrap();

        // Dense: A(2x3) * B(3x2)
        // C[0,0] = 1*5 + 2*6 + 0*0 = 17
        // C[0,1] = 1*0 + 2*7 + 0*8 = 14
        // C[1,0] = 0*5 + 3*6 + 4*0 = 18
        // C[1,1] = 0*0 + 3*7 + 4*8 = 53
        assert_eq!(c.get(0, 0), 17.0);
        assert_eq!(c.get(0, 1), 14.0);
        assert_eq!(c.get(1, 0), 18.0);
        assert_eq!(c.get(1, 1), 53.0);
    }

    #[test]
    fn test_sparse_matmul_dimension_mismatch() {
        let a = csr_from_dense(&[1.0, 2.0], 1, 2);
        let b = csr_from_dense(&[1.0, 2.0, 3.0], 1, 3);
        assert!(sparse_matmul(&a, &b).is_err());
    }

    // -- sparse_scalar_mul --

    #[test]
    fn test_sparse_scalar_mul_basic() {
        let a = csr_from_dense(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let c = sparse_scalar_mul(&a, 3.0);
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(1, 1), 12.0);
        assert_eq!(c.nnz(), 2);
    }

    // -- sparse_transpose --

    #[test]
    fn test_sparse_transpose_square() {
        let a = csr_from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let at = sparse_transpose(&a);
        assert_eq!(at.get(0, 0), 1.0);
        assert_eq!(at.get(0, 1), 3.0);
        assert_eq!(at.get(1, 0), 2.0);
        assert_eq!(at.get(1, 1), 4.0);
    }

    #[test]
    fn test_sparse_transpose_rect() {
        let a = csr_from_dense(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let at = sparse_transpose(&a);
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(at.get(c, r), a.get(r, c), "mismatch at transpose({}, {})", c, r);
            }
        }
    }

    #[test]
    fn test_sparse_transpose_double_is_identity() {
        let a = csr_from_dense(&[1.0, 0.0, 2.0, 3.0, 0.0, 4.0], 2, 3);
        let att = sparse_transpose(&sparse_transpose(&a));
        assert_eq!(att.nrows, a.nrows);
        assert_eq!(att.ncols, a.ncols);
        for r in 0..a.nrows {
            for c in 0..a.ncols {
                assert_eq!(att.get(r, c), a.get(r, c));
            }
        }
    }

    // -- Determinism --

    #[test]
    fn test_sparse_matmul_determinism() {
        let a = csr_from_dense(&[1.0, 2.0, 0.0, 0.0, 3.0, 4.0], 2, 3);
        let b = csr_from_dense(&[5.0, 0.0, 6.0, 7.0, 0.0, 8.0], 3, 2);

        let c1 = sparse_matmul(&a, &b).unwrap();
        let c2 = sparse_matmul(&a, &b).unwrap();

        assert_eq!(c1.values, c2.values);
        assert_eq!(c1.col_indices, c2.col_indices);
        assert_eq!(c1.row_offsets, c2.row_offsets);
    }

    #[test]
    fn test_sparse_add_determinism() {
        let a = csr_from_dense(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 5.0], 3, 3);
        let b = csr_from_dense(&[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0], 3, 3);

        let c1 = sparse_add(&a, &b).unwrap();
        let c2 = sparse_add(&a, &b).unwrap();

        assert_eq!(c1.values, c2.values);
        assert_eq!(c1.col_indices, c2.col_indices);
    }
}

