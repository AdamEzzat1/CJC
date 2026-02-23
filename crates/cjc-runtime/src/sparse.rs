use cjc_repro::kahan_sum_f64;

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

