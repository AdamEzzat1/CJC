use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 6. Linalg Operations on Tensor
// ---------------------------------------------------------------------------

impl Tensor {
    /// LU decomposition with partial pivoting. Returns (L, U, pivot_indices).
    /// Input must be square 2D.
    pub fn lu_decompose(&self) -> Result<(Tensor, Tensor, Vec<usize>), RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "LU decomposition requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        let mut a = self.to_vec();
        let mut pivots: Vec<usize> = (0..n).collect();

        for k in 0..n {
            // Find pivot
            let mut max_val = a[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = a[i * n + k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_val < 1e-15 {
                return Err(RuntimeError::InvalidOperation(
                    "LU decomposition: singular matrix".to_string(),
                ));
            }
            if max_row != k {
                pivots.swap(k, max_row);
                for j in 0..n {
                    let tmp = a[k * n + j];
                    a[k * n + j] = a[max_row * n + j];
                    a[max_row * n + j] = tmp;
                }
            }
            for i in (k + 1)..n {
                a[i * n + k] /= a[k * n + k];
                for j in (k + 1)..n {
                    a[i * n + j] -= a[i * n + k] * a[k * n + j];
                }
            }
        }

        // Extract L and U
        let mut l_data = vec![0.0f64; n * n];
        let mut u_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    l_data[i * n + j] = 1.0;
                    u_data[i * n + j] = a[i * n + j];
                } else if i > j {
                    l_data[i * n + j] = a[i * n + j];
                } else {
                    u_data[i * n + j] = a[i * n + j];
                }
            }
        }

        Ok((
            Tensor::from_vec(l_data, &[n, n])?,
            Tensor::from_vec(u_data, &[n, n])?,
            pivots,
        ))
    }

    /// QR decomposition via Modified Gram-Schmidt. Returns (Q, R).
    /// Input must be 2D with rows >= cols.
    pub fn qr_decompose(&self) -> Result<(Tensor, Tensor), RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "QR decomposition requires a 2D matrix".to_string(),
            ));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let a = self.to_vec();

        // Work column-major for convenience
        let mut q_cols: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..m).map(|i| a[i * n + j]).collect())
            .collect();
        let mut r_data = vec![0.0f64; n * n];

        for j in 0..n {
            // Orthogonalize against previous columns
            for i in 0..j {
                let dot: f64 = (0..m).map(|k| q_cols[i][k] * q_cols[j][k]).collect::<Vec<f64>>().iter().sum();
                r_data[i * n + j] = dot;
                for k in 0..m {
                    q_cols[j][k] -= dot * q_cols[i][k];
                }
            }
            // Normalize
            let norm: f64 = (0..m).map(|k| q_cols[j][k] * q_cols[j][k]).collect::<Vec<f64>>().iter().sum::<f64>().sqrt();
            r_data[j * n + j] = norm;
            if norm > 1e-15 {
                for k in 0..m {
                    q_cols[j][k] /= norm;
                }
            }
        }

        // Assemble Q (m x n)
        let mut q_data = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                q_data[i * n + j] = q_cols[j][i];
            }
        }

        Ok((
            Tensor::from_vec(q_data, &[m, n])?,
            Tensor::from_vec(r_data, &[n, n])?,
        ))
    }

    /// Cholesky decomposition: A = L * L^T.
    /// Input must be symmetric positive definite 2D.
    pub fn cholesky(&self) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "Cholesky decomposition requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        let a = self.to_vec();
        let mut l = vec![0.0f64; n * n];

        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[j * n + k] * l[j * n + k];
            }
            let diag = a[j * n + j] - sum;
            if diag <= 0.0 {
                return Err(RuntimeError::InvalidOperation(
                    "Cholesky: matrix is not positive definite".to_string(),
                ));
            }
            l[j * n + j] = diag.sqrt();

            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }

        Tensor::from_vec(l, &[n, n])
    }

    /// Matrix inverse via LU decomposition + back-substitution.
    pub fn inverse(&self) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "Matrix inverse requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        let (l, u, pivots) = self.lu_decompose()?;
        let l_data = l.to_vec();
        let u_data = u.to_vec();

        let mut inv = vec![0.0f64; n * n];

        // Solve for each column of the inverse
        for col in 0..n {
            // Create permuted identity column
            let mut b = vec![0.0f64; n];
            b[pivots[col]] = 1.0;

            // Forward substitution: L * y = b
            let mut y = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = b[i];
                for j in 0..i {
                    sum -= l_data[i * n + j] * y[j];
                }
                y[i] = sum; // L has 1s on diagonal
            }

            // Back substitution: U * x = y
            let mut x = vec![0.0f64; n];
            for i in (0..n).rev() {
                let mut sum = y[i];
                for j in (i + 1)..n {
                    sum -= u_data[i * n + j] * x[j];
                }
                x[i] = sum / u_data[i * n + i];
            }

            for i in 0..n {
                inv[i * n + col] = x[i];
            }
        }

        Tensor::from_vec(inv, &[n, n])
    }
}

