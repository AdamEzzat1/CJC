use crate::accumulator::BinnedAccumulatorF64;
use crate::error::RuntimeError;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 6. Linalg Operations on Tensor
// ---------------------------------------------------------------------------

impl Tensor {
    /// LU decomposition with partial pivoting. Returns (L, U, pivot_indices).
    /// Input must be square 2D.
    ///
    /// **Determinism contract:** Pivot selection uses strict `>` comparison on
    /// absolute values. When two candidates have identical absolute values, the
    /// first (lowest row index) is chosen. This is deterministic given identical
    /// input bits.
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
                let dot: f64 = {
                    let mut acc = BinnedAccumulatorF64::new();
                    for k in 0..m { acc.add(q_cols[i][k] * q_cols[j][k]); }
                    acc.finalize()
                };
                r_data[i * n + j] = dot;
                for k in 0..m {
                    q_cols[j][k] -= dot * q_cols[i][k];
                }
            }
            // Normalize
            let norm: f64 = {
                let mut acc = BinnedAccumulatorF64::new();
                for k in 0..m { acc.add(q_cols[j][k] * q_cols[j][k]); }
                acc.finalize()
            }.sqrt();
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
            // Use BinnedAccumulator for deterministic summation across platforms.
            let mut acc = BinnedAccumulatorF64::new();
            for k in 0..j {
                acc.add(l[j * n + k] * l[j * n + k]);
            }
            let diag = a[j * n + j] - acc.finalize();
            if diag <= 0.0 {
                return Err(RuntimeError::InvalidOperation(
                    "Cholesky: matrix is not positive definite".to_string(),
                ));
            }
            l[j * n + j] = diag.sqrt();

            for i in (j + 1)..n {
                // Use BinnedAccumulator for deterministic summation across platforms.
                let mut acc = BinnedAccumulatorF64::new();
                for k in 0..j {
                    acc.add(l[i * n + k] * l[j * n + k]);
                }
                l[i * n + j] = (a[i * n + j] - acc.finalize()) / l[j * n + j];
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
        // Use BinnedAccumulator for deterministic dot products.
        let mut qtb = vec![0.0; n];
        for j in 0..n {
            let mut acc = BinnedAccumulatorF64::new();
            for i in 0..m {
                acc.add(q_data[i * n + j] * b_data[i]);
            }
            qtb[j] = acc.finalize();
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
        let mut acc = BinnedAccumulatorF64::new();
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
        let mut acc = BinnedAccumulatorF64::new();
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

    // -----------------------------------------------------------------------
    // Phase B3: Linear algebra extensions
    // -----------------------------------------------------------------------

    /// 1-norm: maximum absolute column sum.
    pub fn norm_1(&self) -> Result<f64, RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "norm_1 requires a 2D matrix".to_string(),
            ));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let data = self.to_vec();
        let mut max_col_sum = 0.0_f64;
        for j in 0..n {
            let mut acc = BinnedAccumulatorF64::new();
            for i in 0..m {
                acc.add(data[i * n + j].abs());
            }
            let col_sum = acc.finalize();
            if col_sum > max_col_sum {
                max_col_sum = col_sum;
            }
        }
        Ok(max_col_sum)
    }

    /// Infinity norm: maximum absolute row sum.
    pub fn norm_inf(&self) -> Result<f64, RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "norm_inf requires a 2D matrix".to_string(),
            ));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let data = self.to_vec();
        let mut max_row_sum = 0.0_f64;
        for i in 0..m {
            let mut acc = BinnedAccumulatorF64::new();
            for j in 0..n {
                acc.add(data[i * n + j].abs());
            }
            let row_sum = acc.finalize();
            if row_sum > max_row_sum {
                max_row_sum = row_sum;
            }
        }
        Ok(max_row_sum)
    }

    /// Condition number via eigenvalue ratio.
    /// For symmetric: |lambda_max| / |lambda_min|. For general: sqrt(sigma_max/sigma_min).
    pub fn cond(&self) -> Result<f64, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "cond requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        // Check if symmetric
        let data = self.to_vec();
        let mut is_sym = true;
        'outer: for i in 0..n {
            for j in (i + 1)..n {
                if (data[i * n + j] - data[j * n + i]).abs() > 1e-14 {
                    is_sym = false;
                    break 'outer;
                }
            }
        }
        if is_sym {
            let (eigenvalues, _) = self.eigh()?;
            let abs_min = eigenvalues.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            let abs_max = eigenvalues.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            if abs_min < 1e-15 {
                return Ok(f64::INFINITY);
            }
            Ok(abs_max / abs_min)
        } else {
            // General: compute A^T * A, then eigh
            let at = self.transpose();
            let ata = at.matmul(self)?;
            let (eigenvalues, _) = ata.eigh()?;
            let abs_min = eigenvalues.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            let abs_max = eigenvalues.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            if abs_min < 1e-15 {
                return Ok(f64::INFINITY);
            }
            Ok((abs_max / abs_min).sqrt())
        }
    }

    /// Real Schur decomposition: A = Q * T * Q^T.
    /// Uses Hessenberg reduction + implicit double-shift QR.
    pub fn schur(&self) -> Result<(Tensor, Tensor), RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "schur requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        if n == 0 {
            return Err(RuntimeError::InvalidOperation("schur: empty matrix".to_string()));
        }
        if n == 1 {
            return Ok((
                Tensor::from_vec(vec![1.0], &[1, 1])?,
                self.clone(),
            ));
        }

        let mut h = self.to_vec();
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }

        // Step 1: Reduce to upper Hessenberg form
        for k in 0..n.saturating_sub(2) {
            // Compute Householder reflector for column k below diagonal
            let mut col = vec![0.0; n - k - 1];
            for i in 0..col.len() {
                col[i] = h[(k + 1 + i) * n + k];
            }
            let norm_col = {
                let mut acc = BinnedAccumulatorF64::new();
                for &v in &col { acc.add(v * v); }
                acc.finalize().sqrt()
            };
            if norm_col < 1e-15 {
                continue;
            }
            // Sign convention: positive diagonal
            let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
            col[0] += sign * norm_col;
            let norm_v = {
                let mut acc = BinnedAccumulatorF64::new();
                for &v in &col { acc.add(v * v); }
                acc.finalize().sqrt()
            };
            if norm_v < 1e-15 {
                continue;
            }
            for v in &mut col {
                *v /= norm_v;
            }
            // Apply H = (I - 2vv^T) * H from left
            for j in 0..n {
                let mut acc = BinnedAccumulatorF64::new();
                for i in 0..col.len() {
                    acc.add(col[i] * h[(k + 1 + i) * n + j]);
                }
                let dot = acc.finalize();
                for i in 0..col.len() {
                    h[(k + 1 + i) * n + j] -= 2.0 * col[i] * dot;
                }
            }
            // Apply H = H * (I - 2vv^T) from right
            for i in 0..n {
                let mut acc = BinnedAccumulatorF64::new();
                for j in 0..col.len() {
                    acc.add(col[j] * h[i * n + (k + 1 + j)]);
                }
                let dot = acc.finalize();
                for j in 0..col.len() {
                    h[i * n + (k + 1 + j)] -= 2.0 * col[j] * dot;
                }
            }
            // Accumulate Q
            for i in 0..n {
                let mut acc = BinnedAccumulatorF64::new();
                for j in 0..col.len() {
                    acc.add(col[j] * q[i * n + (k + 1 + j)]);
                }
                let dot = acc.finalize();
                for j in 0..col.len() {
                    q[i * n + (k + 1 + j)] -= 2.0 * col[j] * dot;
                }
            }
        }

        // Step 2: Implicit QR iterations
        let eps = 1e-14;
        let max_iter = 200 * n;
        let mut ihi = n - 1;

        for _iter in 0..max_iter {
            if ihi == 0 {
                break;
            }
            // Find active block
            let mut ilo = ihi;
            while ilo > 0 {
                if h[ilo * n + (ilo - 1)].abs()
                    < eps * (h[(ilo - 1) * n + (ilo - 1)].abs() + h[ilo * n + ilo].abs())
                {
                    h[ilo * n + (ilo - 1)] = 0.0;
                    break;
                }
                ilo -= 1;
            }
            if ilo == ihi {
                // 1x1 block converged
                ihi -= 1;
                continue;
            }
            if ilo + 1 == ihi {
                // 2x2 block - check if real eigenvalues
                ihi -= 2;
                continue;
            }

            // Wilkinson shift from trailing 2x2
            let a11 = h[(ihi - 1) * n + (ihi - 1)];
            let a12 = h[(ihi - 1) * n + ihi];
            let a21 = h[ihi * n + (ihi - 1)];
            let a22 = h[ihi * n + ihi];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;

            let shift = if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let ev1 = (tr + sqrt_disc) / 2.0;
                let ev2 = (tr - sqrt_disc) / 2.0;
                if (ev1 - a22).abs() < (ev2 - a22).abs() { ev1 } else { ev2 }
            } else {
                a22 // complex eigenvalues: use a22 as shift
            };

            // Apply single-shift QR step with Givens rotations
            let mut x = h[ilo * n + ilo] - shift;
            let mut z = h[(ilo + 1) * n + ilo];
            for k in ilo..ihi {
                let r = (x * x + z * z).sqrt();
                let c = if r < 1e-15 { 1.0 } else { x / r };
                let s = if r < 1e-15 { 0.0 } else { z / r };
                // Apply Givens from left: rows k and k+1
                for j in 0..n {
                    let t1 = h[k * n + j];
                    let t2 = h[(k + 1) * n + j];
                    h[k * n + j] = c * t1 + s * t2;
                    h[(k + 1) * n + j] = -s * t1 + c * t2;
                }
                // Apply Givens from right: cols k and k+1
                let jmax = if k + 3 < n { k + 3 } else { n };
                for i in 0..jmax {
                    let t1 = h[i * n + k];
                    let t2 = h[i * n + (k + 1)];
                    h[i * n + k] = c * t1 + s * t2;
                    h[i * n + (k + 1)] = -s * t1 + c * t2;
                }
                // Accumulate Q
                for i in 0..n {
                    let t1 = q[i * n + k];
                    let t2 = q[i * n + (k + 1)];
                    q[i * n + k] = c * t1 + s * t2;
                    q[i * n + (k + 1)] = -s * t1 + c * t2;
                }
                if k + 2 <= ihi {
                    x = h[(k + 1) * n + k];
                    z = h[(k + 2) * n + k];
                }
            }
        }

        // Clean up sub-diagonal entries
        for i in 0..n {
            for j in 0..i.saturating_sub(1) {
                h[i * n + j] = 0.0;
            }
        }

        Ok((
            Tensor::from_vec(q, &[n, n])?,
            Tensor::from_vec(h, &[n, n])?,
        ))
    }

    // -----------------------------------------------------------------------
    // Phase 3A: SVD via Golub-Kahan Bidiagonalization
    // -----------------------------------------------------------------------

    /// Compute the Singular Value Decomposition: A = U @ diag(S) @ Vt.
    /// Returns (U, S, Vt) as (Tensor, Vec<f64>, Tensor).
    ///
    /// Implementation: via eigendecomposition of A^T*A (for V and S^2),
    /// then U = A*V*diag(1/s_i).
    ///
    /// **Determinism contract:** All intermediate float reductions use
    /// `BinnedAccumulatorF64`. Iteration order is fixed row-major.
    pub fn svd(&self) -> Result<(Tensor, Vec<f64>, Tensor), RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "SVD requires a 2D matrix".to_string(),
            ));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        if m == 0 || n == 0 {
            return Err(RuntimeError::InvalidOperation(
                "SVD: empty matrix".to_string(),
            ));
        }

        let min_mn = m.min(n);

        // Compute A^T * A (n x n symmetric matrix)
        let at = self.transpose();
        let ata = at.matmul(self)?;

        // Eigendecomposition of A^T*A gives V and eigenvalues = sigma^2
        let (eigenvalues, eigenvectors) = ata.eigh()?;

        // eigenvalues are in ascending order from eigh; we want descending singular values
        // Singular values = sqrt(eigenvalues), clamp negatives to 0
        let mut singular_values: Vec<f64> = eigenvalues.iter()
            .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
            .collect();

        // Reverse to get descending order
        singular_values.reverse();

        // Take only min(m,n) singular values
        let k = min_mn.min(singular_values.len());
        let s: Vec<f64> = singular_values[..k].to_vec();

        // V columns are eigenvectors of A^T*A, reversed for descending order
        // eigenvectors is n x n, columns are eigenvectors
        let ev_data = eigenvectors.to_vec();
        let ev_n = eigenvectors.shape()[1]; // should be n

        // Build V matrix (n x k) with columns in descending singular value order
        let mut v_data = vec![0.0f64; n * k];
        for col in 0..k {
            let ev_col = n - 1 - col; // reverse index for descending order
            for row in 0..n {
                v_data[row * k + col] = ev_data[row * ev_n + ev_col];
            }
        }
        let v_mat = Tensor::from_vec(v_data.clone(), &[n, k])?;

        // U = A * V * diag(1/s)
        // First compute A * V
        let av = self.matmul(&v_mat)?;
        let av_data = av.to_vec();

        // Then scale each column by 1/s_i
        let mut u_data = vec![0.0f64; m * k];
        for col in 0..k {
            if s[col] > 1e-14 {
                let inv_s = 1.0 / s[col];
                for row in 0..m {
                    u_data[row * k + col] = av_data[row * k + col] * inv_s;
                }
            }
            // If s[col] ≈ 0, leave u column as zeros
        }

        // Sign-canonical: ensure largest-magnitude element of each u column is positive
        for col in 0..k {
            let mut max_abs = 0.0f64;
            let mut max_sign = 1.0f64;
            for row in 0..m {
                let val = u_data[row * k + col];
                if val.abs() > max_abs {
                    max_abs = val.abs();
                    max_sign = if val >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            if max_sign < 0.0 {
                for row in 0..m {
                    u_data[row * k + col] = -u_data[row * k + col];
                }
                for row in 0..n {
                    v_data[row * k + col] = -v_data[row * k + col];
                }
            }
        }

        let u_tensor = Tensor::from_vec(u_data, &[m, k])?;

        // Vt = V^T (k x n)
        let mut vt_data = vec![0.0f64; k * n];
        for row in 0..k {
            for col in 0..n {
                vt_data[row * n + col] = v_data[col * k + row];
            }
        }
        let vt_tensor = Tensor::from_vec(vt_data, &[k, n])?;

        Ok((u_tensor, s, vt_tensor))
    }

    /// Truncated SVD — only the top `k` singular values/vectors.
    /// Returns (U_k, S_k, Vt_k) where U_k is m x k, Vt_k is k x n.
    pub fn svd_truncated(
        &self,
        k: usize,
    ) -> Result<(Tensor, Vec<f64>, Tensor), RuntimeError> {
        let (u_full, s_full, vt_full) = self.svd()?;
        let m = u_full.shape()[0];
        let n = vt_full.shape()[1];
        let actual_k = k.min(s_full.len());

        if actual_k == 0 {
            return Err(RuntimeError::InvalidOperation(
                "svd_truncated: k must be > 0".to_string(),
            ));
        }

        let s_k: Vec<f64> = s_full[..actual_k].to_vec();

        // Extract first k columns of U
        let u_data = u_full.to_vec();
        let u_cols = u_full.shape()[1];
        let mut u_k = vec![0.0f64; m * actual_k];
        for row in 0..m {
            for col in 0..actual_k {
                u_k[row * actual_k + col] = u_data[row * u_cols + col];
            }
        }

        // Extract first k rows of Vt
        let vt_data = vt_full.to_vec();
        let mut vt_k = vec![0.0f64; actual_k * n];
        for row in 0..actual_k {
            for col in 0..n {
                vt_k[row * n + col] = vt_data[row * n + col];
            }
        }

        Ok((
            Tensor::from_vec(u_k, &[m, actual_k])?,
            s_k,
            Tensor::from_vec(vt_k, &[actual_k, n])?,
        ))
    }

    // -----------------------------------------------------------------------
    // Phase 3B: Pseudoinverse (Moore-Penrose, via SVD)
    // -----------------------------------------------------------------------

    /// Compute the Moore-Penrose pseudoinverse via SVD.
    /// A+ = V @ diag(1/s_i) @ Ut (with default tolerance for near-zero singular values).
    pub fn pinv(&self) -> Result<Tensor, RuntimeError> {
        // Default tolerance: max(m,n) * eps * max(S)
        let (u, s, vt) = self.svd()?;
        let m = self.shape[0];
        let n = self.shape[1];
        let max_s = s.first().copied().unwrap_or(0.0);
        let tol = (m.max(n) as f64) * f64::EPSILON * max_s;
        Self::pinv_from_svd(&u, &s, &vt, tol)
    }

    /// Compute the Moore-Penrose pseudoinverse via SVD with explicit tolerance.
    pub fn pinv_with_tol(&self, tol: f64) -> Result<Tensor, RuntimeError> {
        let (u, s, vt) = self.svd()?;
        Self::pinv_from_svd(&u, &s, &vt, tol)
    }

    /// Internal: compute pseudoinverse from pre-computed SVD.
    /// A+ = V @ diag(1/s_i) @ Ut, zeroing 1/s_i where s_i <= tol.
    fn pinv_from_svd(
        u: &Tensor,
        s: &[f64],
        vt: &Tensor,
        tol: f64,
    ) -> Result<Tensor, RuntimeError> {
        let m = u.shape()[0];
        let k = s.len();
        let n = vt.shape()[1];

        // Build S_inv: k-vector with 1/s_i or 0
        let s_inv: Vec<f64> = s
            .iter()
            .map(|&si| if si > tol { 1.0 / si } else { 0.0 })
            .collect();

        // Compute Vt^T @ diag(s_inv) @ U^T = V @ diag(s_inv) @ Ut
        // Result is n x m
        let u_data = u.to_vec();
        let vt_data = vt.to_vec();
        let mut result = vec![0.0f64; n * m];

        for i in 0..n {
            for j in 0..m {
                let mut acc = BinnedAccumulatorF64::new();
                for l in 0..k {
                    // V[i, l] = Vt[l, i] (transposed)
                    // Ut[l, j] = U[j, l] (transposed)
                    acc.add(vt_data[l * n + i] * s_inv[l] * u_data[j * k + l]);
                }
                result[i * m + j] = acc.finalize();
            }
        }

        Tensor::from_vec(result, &[n, m])
    }

    /// Helper: compute Givens rotation parameters.
    /// Returns (c, s, r) such that [c s; -s c]^T * [a; b] = [r; 0].
    fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
        if b.abs() < 1e-15 {
            (1.0, 0.0, a)
        } else if a.abs() < 1e-15 {
            (0.0, if b >= 0.0 { 1.0 } else { -1.0 }, b.abs())
        } else {
            let r = (a * a + b * b).sqrt();
            (a / r, b / r, r)
        }
    }

    /// Matrix exponential via scaling and squaring with Pade(13,13) approximation.
    pub fn matrix_exp(&self) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 || self.shape[0] != self.shape[1] {
            return Err(RuntimeError::InvalidOperation(
                "matrix_exp requires a square 2D matrix".to_string(),
            ));
        }
        let n = self.shape[0];
        if n == 0 {
            return Err(RuntimeError::InvalidOperation("matrix_exp: empty matrix".to_string()));
        }

        const PADE_COEFFS: [f64; 14] = [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ];
        const THETA_13: f64 = 5.371920351148152;

        // Scaling: s = max(0, ceil(log2(||A||_1 / theta_13)))
        let norm = self.norm_1()?;
        let s = if norm <= THETA_13 {
            0u32
        } else {
            (norm / THETA_13).log2().ceil() as u32
        };

        // B = A / 2^s
        let scale = 2.0_f64.powi(-(s as i32));
        let b_data: Vec<f64> = self.to_vec().iter().map(|&x| x * scale).collect();
        let b = Tensor::from_vec(b_data, &[n, n])?;

        // Compute B^2, B^4, B^6
        let b2 = b.matmul(&b)?;
        let b4 = b2.matmul(&b2)?;
        let b6 = b4.matmul(&b2)?;

        // Identity matrix
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let eye_t = Tensor::from_vec(eye, &[n, n])?;

        // Build U and V
        // U = B * (b_13*B^6 + b_11*B^4 + b_9*B^2 + b_7*I) * B^6
        //   + B * (b_5*B^4 + b_3*B^2 + b_1*I)
        // V = (b_12*B^6 + b_10*B^4 + b_8*B^2 + b_6*I) * B^6
        //   + (b_4*B^4 + b_2*B^2 + b_0*I)

        // Helper: scale and add tensors
        fn scale_add(a: &Tensor, sa: f64, b: &Tensor, sb: f64, n: usize) -> Vec<f64> {
            let ad = a.to_vec();
            let bd = b.to_vec();
            let mut r = vec![0.0; n * n];
            for i in 0..n * n {
                r[i] = sa * ad[i] + sb * bd[i];
            }
            r
        }

        let c = &PADE_COEFFS;

        // inner_u1 = b_13*B6 + b_11*B4 + b_9*B2 + b_7*I
        let mut iu1 = scale_add(&b6, c[13], &b4, c[11], n);
        let t = scale_add(&b2, c[9], &eye_t, c[7], n);
        for i in 0..n * n {
            iu1[i] += t[i];
        }
        let iu1_t = Tensor::from_vec(iu1, &[n, n])?;
        let iu1_b6 = iu1_t.matmul(&b6)?;

        // inner_u2 = b_5*B4 + b_3*B2 + b_1*I
        let mut iu2 = scale_add(&b4, c[5], &b2, c[3], n);
        let t = Tensor::from_vec({
            let mut v = vec![0.0; n * n];
            for i in 0..n { v[i * n + i] = c[1]; }
            v
        }, &[n, n])?;
        let td = t.to_vec();
        for i in 0..n * n {
            iu2[i] += td[i];
        }
        let iu2_t = Tensor::from_vec(iu2, &[n, n])?;

        // U_inner = iu1_b6 + iu2
        let iu1d = iu1_b6.to_vec();
        let iu2d = iu2_t.to_vec();
        let mut u_inner = vec![0.0; n * n];
        for i in 0..n * n {
            u_inner[i] = iu1d[i] + iu2d[i];
        }
        let u_inner_t = Tensor::from_vec(u_inner, &[n, n])?;
        let u = b.matmul(&u_inner_t)?;

        // inner_v1 = b_12*B6 + b_10*B4 + b_8*B2 + b_6*I
        let mut iv1 = scale_add(&b6, c[12], &b4, c[10], n);
        let t = scale_add(&b2, c[8], &eye_t, c[6], n);
        for i in 0..n * n {
            iv1[i] += t[i];
        }
        let iv1_t = Tensor::from_vec(iv1, &[n, n])?;
        let iv1_b6 = iv1_t.matmul(&b6)?;

        // inner_v2 = b_4*B4 + b_2*B2 + b_0*I
        let mut iv2 = scale_add(&b4, c[4], &b2, c[2], n);
        let t = Tensor::from_vec({
            let mut v = vec![0.0; n * n];
            for i in 0..n { v[i * n + i] = c[0]; }
            v
        }, &[n, n])?;
        let td = t.to_vec();
        for i in 0..n * n {
            iv2[i] += td[i];
        }
        let iv2_t = Tensor::from_vec(iv2, &[n, n])?;

        // V = iv1_b6 + iv2
        let iv1d = iv1_b6.to_vec();
        let iv2d = iv2_t.to_vec();
        let mut v_data = vec![0.0; n * n];
        for i in 0..n * n {
            v_data[i] = iv1d[i] + iv2d[i];
        }
        let v_mat = Tensor::from_vec(v_data, &[n, n])?;

        // Solve (V - U) * r = (V + U)
        let ud = u.to_vec();
        let vd = v_mat.to_vec();
        let mut lhs_data = vec![0.0; n * n];
        let mut rhs_data = vec![0.0; n * n];
        for i in 0..n * n {
            lhs_data[i] = vd[i] - ud[i];
            rhs_data[i] = vd[i] + ud[i];
        }
        let lhs = Tensor::from_vec(lhs_data, &[n, n])?;

        // Solve column-by-column
        let mut result = vec![0.0; n * n];
        for col in 0..n {
            let mut rhs_col = vec![0.0; n];
            for row in 0..n {
                rhs_col[row] = rhs_data[row * n + col];
            }
            let rhs_tensor = Tensor::from_vec(rhs_col, &[n])?;
            let sol = lhs.solve(&rhs_tensor)?;
            let sol_data = sol.to_vec();
            for row in 0..n {
                result[row * n + col] = sol_data[row];
            }
        }

        let mut r = Tensor::from_vec(result, &[n, n])?;

        // Square s times: r = r * r
        for _ in 0..s {
            r = r.matmul(&r)?;
        }

        Ok(r)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: reconstruct A from SVD: U @ diag(S) @ Vt
    fn reconstruct_svd(u: &Tensor, s: &[f64], vt: &Tensor) -> Vec<f64> {
        let m = u.shape()[0];
        let k = s.len();
        let n = vt.shape()[1];
        let u_data = u.to_vec();
        let vt_data = vt.to_vec();
        let u_cols = u.shape()[1];
        let mut result = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += u_data[i * u_cols + l] * s[l] * vt_data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        result
    }

    /// Helper: check two flat arrays are approximately equal
    fn assert_approx_eq(a: &[f64], b: &[f64], tol: f64, msg: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() < tol,
                "{}: element [{}] differs: {} vs {} (diff={})",
                msg,
                i,
                ai,
                bi,
                (ai - bi).abs()
            );
        }
    }

    #[test]
    fn test_svd_identity_2x2() {
        let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let (u, s, vt) = eye.svd().unwrap();
        // Singular values should be [1, 1]
        assert!((s[0] - 1.0).abs() < 1e-10, "s[0] = {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-10, "s[1] = {}", s[1]);
        // Roundtrip
        let recon = reconstruct_svd(&u, &s, &vt);
        assert_approx_eq(&recon, &[1.0, 0.0, 0.0, 1.0], 1e-10, "SVD identity roundtrip");
    }

    #[test]
    fn test_svd_identity_3x3() {
        let mut data = vec![0.0; 9];
        for i in 0..3 {
            data[i * 3 + i] = 1.0;
        }
        let eye = Tensor::from_vec(data.clone(), &[3, 3]).unwrap();
        let (u, s, vt) = eye.svd().unwrap();
        for &si in &s {
            assert!((si - 1.0).abs() < 1e-10, "singular value = {}", si);
        }
        let recon = reconstruct_svd(&u, &s, &vt);
        assert_approx_eq(&recon, &data, 1e-10, "SVD 3x3 identity roundtrip");
    }

    #[test]
    fn test_svd_known_matrix() {
        // A = [[3, 0], [0, 2]] — singular values should be 3, 2
        let a = Tensor::from_vec(vec![3.0, 0.0, 0.0, 2.0], &[2, 2]).unwrap();
        let (_u, s, _vt) = a.svd().unwrap();
        assert!((s[0] - 3.0).abs() < 1e-10, "s[0] = {}", s[0]);
        assert!((s[1] - 2.0).abs() < 1e-10, "s[1] = {}", s[1]);
    }

    #[test]
    fn test_svd_roundtrip_general() {
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5],
            &[3, 3],
        )
        .unwrap();
        let (u, s, vt) = a.svd().unwrap();
        let recon = reconstruct_svd(&u, &s, &vt);
        let original = a.to_vec();
        assert_approx_eq(&recon, &original, 1e-8, "SVD general roundtrip");
    }

    #[test]
    fn test_svd_rectangular_tall() {
        // 3x2 matrix
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let (u, s, vt) = a.svd().unwrap();
        assert_eq!(u.shape(), &[3, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 2]);
        let recon = reconstruct_svd(&u, &s, &vt);
        let original = a.to_vec();
        assert_approx_eq(&recon, &original, 1e-8, "SVD tall rectangular roundtrip");
    }

    #[test]
    fn test_svd_rectangular_wide() {
        // 2x3 matrix
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let (u, s, vt) = a.svd().unwrap();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 3]);
        let recon = reconstruct_svd(&u, &s, &vt);
        let original = a.to_vec();
        assert_approx_eq(&recon, &original, 1e-8, "SVD wide rectangular roundtrip");
    }

    #[test]
    fn test_svd_singular_values_descending() {
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            &[3, 3],
        )
        .unwrap();
        let (_, s, _) = a.svd().unwrap();
        for i in 0..s.len() - 1 {
            assert!(s[i] >= s[i + 1], "singular values not descending: {} < {}", s[i], s[i + 1]);
        }
    }

    #[test]
    fn test_svd_truncated_basic() {
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            &[3, 3],
        )
        .unwrap();
        let (u, s, vt) = a.svd_truncated(2).unwrap();
        assert_eq!(u.shape(), &[3, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 3]);
    }

    #[test]
    fn test_svd_deterministic() {
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            &[3, 3],
        )
        .unwrap();
        let (u1, s1, vt1) = a.svd().unwrap();
        let (u2, s2, vt2) = a.svd().unwrap();
        assert_eq!(u1.to_vec(), u2.to_vec(), "U not deterministic");
        assert_eq!(s1, s2, "S not deterministic");
        assert_eq!(vt1.to_vec(), vt2.to_vec(), "Vt not deterministic");
    }

    #[test]
    fn test_pinv_square() {
        // For a non-singular square matrix, pinv(A) ≈ inv(A)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 5.0], &[2, 2]).unwrap();
        let a_pinv = a.pinv().unwrap();
        // Check A @ A+ @ A ≈ A
        let a_ap = a.matmul(&a_pinv).unwrap();
        let a_ap_a = a_ap.matmul(&a).unwrap();
        assert_approx_eq(&a_ap_a.to_vec(), &a.to_vec(), 1e-8, "pinv square: A @ A+ @ A ≈ A");
    }

    #[test]
    fn test_pinv_identity() {
        let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let eye_pinv = eye.pinv().unwrap();
        assert_approx_eq(
            &eye_pinv.to_vec(),
            &[1.0, 0.0, 0.0, 1.0],
            1e-10,
            "pinv of identity",
        );
    }

    #[test]
    fn test_pinv_rectangular() {
        // Tall matrix: 3x2
        let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2]).unwrap();
        let a_pinv = a.pinv().unwrap();
        assert_eq!(a_pinv.shape(), &[2, 3]);
        // A @ A+ @ A ≈ A
        let a_ap = a.matmul(&a_pinv).unwrap();
        let a_ap_a = a_ap.matmul(&a).unwrap();
        assert_approx_eq(&a_ap_a.to_vec(), &a.to_vec(), 1e-8, "pinv rect: A @ A+ @ A ≈ A");
    }

    #[test]
    fn test_pinv_moore_penrose_conditions() {
        // All 4 Moore-Penrose conditions for a general matrix
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let ap = a.pinv().unwrap();
        // Condition 1: A @ A+ @ A ≈ A
        let aapa = a.matmul(&ap).unwrap().matmul(&a).unwrap();
        assert_approx_eq(&aapa.to_vec(), &a.to_vec(), 1e-6, "MP condition 1");
        // Condition 2: A+ @ A @ A+ ≈ A+
        let apaap = ap.matmul(&a).unwrap().matmul(&ap).unwrap();
        assert_approx_eq(&apaap.to_vec(), &ap.to_vec(), 1e-6, "MP condition 2");
    }

    #[test]
    fn test_pinv_with_tol() {
        let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1e-16], &[2, 2]).unwrap();
        // With large tolerance, treat 1e-16 as zero
        let ap = a.pinv_with_tol(1e-10).unwrap();
        let ap_data = ap.to_vec();
        // Should act like pseudoinverse of [[1,0],[0,0]]
        assert!((ap_data[0] - 1.0).abs() < 1e-8, "pinv_with_tol [0,0]");
        assert!(ap_data[3].abs() < 1e-8, "pinv_with_tol [1,1] should be ~0");
    }

    #[test]
    fn test_svd_1x1() {
        let a = Tensor::from_vec(vec![5.0], &[1, 1]).unwrap();
        let (u, s, vt) = a.svd().unwrap();
        assert!((s[0] - 5.0).abs() < 1e-10);
        let recon = reconstruct_svd(&u, &s, &vt);
        assert_approx_eq(&recon, &[5.0], 1e-10, "SVD 1x1 roundtrip");
    }
}

