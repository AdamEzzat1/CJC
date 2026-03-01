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

    /// Determinant via LU decomposition: product of U diagonal * parity.
    pub fn det(&self) -> Result<f64, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "det requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        let mut a = self.to_vec();
        let mut sign = 1.0f64;
        for k in 0..n {
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
                return Ok(0.0); // singular
            }
            if max_row != k {
                sign = -sign;
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
        let mut det = sign;
        for i in 0..n {
            det *= a[i * n + i];
        }
        Ok(det)
    }

    /// Solve Ax = b via LU decomposition.
    /// self = A (n x n), b = vector (n).
    pub fn solve(&self, b: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "solve requires a square 2D matrix A".to_string(),
            ));
        }
        let n = self.shape[0];
        if b.len() != n {
            return Err(RuntimeError::InvalidOperation(
                format!("solve: b length {} != n = {n}", b.len()),
            ));
        }
        let (l, u, pivots) = self.lu_decompose()?;
        let l_data = l.to_vec();
        let u_data = u.to_vec();
        let b_data = b.to_vec();

        // Permute b
        let mut pb = vec![0.0; n];
        for i in 0..n {
            pb[i] = b_data[pivots[i]];
        }

        // Forward substitution: L * y = pb
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = pb[i];
            for j in 0..i {
                s -= l_data[i * n + j] * y[j];
            }
            y[i] = s;
        }

        // Back substitution: U * x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n {
                s -= u_data[i * n + j] * x[j];
            }
            x[i] = s / u_data[i * n + i];
        }

        Tensor::from_vec(x, &[n])
    }

    /// Least squares solution: min ||Ax - b||_2 via QR decomposition.
    /// self = A (m x n, m >= n), b = vector (m).
    pub fn lstsq(&self, b: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "lstsq requires a 2D matrix".to_string(),
            ));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        if m < n {
            return Err(RuntimeError::InvalidOperation(
                "lstsq requires m >= n".to_string(),
            ));
        }
        if b.len() != m {
            return Err(RuntimeError::InvalidOperation(
                format!("lstsq: b length {} != m = {m}", b.len()),
            ));
        }
        let (q, r) = self.qr_decompose()?;
        let q_data = q.to_vec();
        let r_data = r.to_vec();
        let b_data = b.to_vec();

        // Q^T * b (Q is m x n, so Q^T * b gives n-vector)
        let mut qtb = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                qtb[j] += q_data[i * n + j] * b_data[i];
            }
        }

        // Back substitution: R * x = Q^T * b
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = qtb[i];
            for j in (i + 1)..n {
                s -= r_data[i * n + j] * x[j];
            }
            if r_data[i * n + i].abs() < 1e-15 {
                return Err(RuntimeError::InvalidOperation(
                    "lstsq: rank-deficient matrix".to_string(),
                ));
            }
            x[i] = s / r_data[i * n + i];
        }

        Tensor::from_vec(x, &[n])
    }

    /// Matrix trace: sum of diagonal elements.
    pub fn trace(&self) -> Result<f64, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "trace requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        let data = self.to_vec();
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for i in 0..n {
            acc.add(data[i * n + i]);
        }
        Ok(acc.finalize())
    }

    /// Frobenius norm: sqrt(sum(aij^2)).
    pub fn norm_frobenius(&self) -> Result<f64, RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "norm_frobenius requires a 2D matrix".to_string(),
            ));
        }
        let data = self.to_vec();
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for &v in &data {
            acc.add(v * v);
        }
        Ok(acc.finalize().sqrt())
    }

    /// Eigenvalue decomposition for symmetric matrices (Jacobi method).
    /// Returns (eigenvalues sorted ascending, eigenvectors n x n).
    /// DETERMINISM: fixed row-major sweep order, smallest (i,j) tie-breaking.
    pub fn eigh(&self) -> Result<(Vec<f64>, Tensor), RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "eigh requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        let mut a = self.to_vec();
        // V = identity (eigenvectors)
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        let max_iter = 100 * n * n;
        for _ in 0..max_iter {
            // Find largest off-diagonal element (row-major for determinism)
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v_abs = a[i * n + j].abs();
                    if v_abs > max_val {
                        max_val = v_abs;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-14 {
                break; // converged
            }
            // Compute rotation angle
            let app = a[p * n + p];
            let aqq = a[q * n + q];
            let apq = a[p * n + q];
            let theta = if (app - aqq).abs() < 1e-15 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * apq / (app - aqq)).atan()
            };
            let c = theta.cos();
            let s = theta.sin();
            // Apply Givens rotation
            for i in 0..n {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                a[i * n + p] = c * aip + s * aiq;
                a[i * n + q] = -s * aip + c * aiq;
            }
            for j in 0..n {
                let apj = a[p * n + j];
                let aqj = a[q * n + j];
                a[p * n + j] = c * apj + s * aqj;
                a[q * n + j] = -s * apj + c * aqj;
            }
            // Fix diagonal after double rotation
            a[p * n + q] = 0.0;
            a[q * n + p] = 0.0;
            // Accumulate eigenvectors
            for i in 0..n {
                let vip = v[i * n + p];
                let viq = v[i * n + q];
                v[i * n + p] = c * vip + s * viq;
                v[i * n + q] = -s * vip + c * viq;
            }
        }
        // Extract eigenvalues (diagonal of a)
        let mut eigenvalues: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
        eigenvalues.sort_by(|a, b| a.0.total_cmp(&b.0));
        let vals: Vec<f64> = eigenvalues.iter().map(|&(v, _)| v).collect();
        // Reorder eigenvector columns
        let mut v_sorted = vec![0.0; n * n];
        for (new_col, &(_, old_col)) in eigenvalues.iter().enumerate() {
            for row in 0..n {
                v_sorted[row * n + new_col] = v[row * n + old_col];
            }
        }
        // Sign-canonical: first nonzero component positive
        for col in 0..n {
            let mut first_nonzero = 0.0;
            for row in 0..n {
                if v_sorted[row * n + col].abs() > 1e-15 {
                    first_nonzero = v_sorted[row * n + col];
                    break;
                }
            }
            if first_nonzero < 0.0 {
                for row in 0..n {
                    v_sorted[row * n + col] = -v_sorted[row * n + col];
                }
            }
        }

        Ok((vals, Tensor::from_vec(v_sorted, &[n, n])?))
    }

    /// Matrix rank via SVD: count singular values > tolerance.
    pub fn matrix_rank(&self) -> Result<usize, RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "matrix_rank requires a 2D matrix".to_string(),
            ));
        }
        // Simple approach: use QR and count non-zero diagonal
        let (_q, r) = self.qr_decompose()?;
        let r_data = r.to_vec();
        let n = r.shape()[0].min(r.shape()[1]);
        let cols = r.shape()[1];
        let mut rank = 0;
        for i in 0..n {
            if r_data[i * cols + i].abs() > 1e-10 {
                rank += 1;
            }
        }
        Ok(rank)
    }

    /// Kronecker product: A ⊗ B.
    pub fn kron(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "kron requires two 2D matrices".to_string(),
            ));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let (p, q) = (other.shape()[0], other.shape()[1]);
        let a = self.to_vec();
        let b = other.to_vec();
        let mut result = vec![0.0; m * p * n * q];
        let out_cols = n * q;
        for i in 0..m {
            for j in 0..n {
                let aij = a[i * n + j];
                for k in 0..p {
                    for l in 0..q {
                        result[(i * p + k) * out_cols + (j * q + l)] = aij * b[k * q + l];
                    }
                }
            }
        }
        Tensor::from_vec(result, &[m * p, n * q])
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

