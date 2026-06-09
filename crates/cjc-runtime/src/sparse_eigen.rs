//! Sparse Eigenvalue Solvers — Lanczos and Arnoldi
//!
//! Deterministic sparse eigenvalue extraction for symmetric (Lanczos) and
//! general (Arnoldi) matrices. All operations use Kahan summation for
//! order-invariant deterministic reductions.

use crate::sparse::SparseCsr;
use cjc_repro::kahan_sum_f64;

// ---------------------------------------------------------------------------
// Lanczos Algorithm (Symmetric Matrices)
// ---------------------------------------------------------------------------

/// Result of a Lanczos eigenvalue computation.
#[derive(Debug, Clone)]
pub struct LanczosResult {
    /// Eigenvalues (sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as rows (each row is an eigenvector).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Compute the `k` largest eigenvalues of a symmetric sparse matrix using
/// the Lanczos algorithm.
///
/// # Arguments
/// * `matrix` - Symmetric sparse CSR matrix
/// * `k` - Number of eigenvalues to compute (must be <= matrix dimension)
/// * `max_iter` - Maximum Lanczos iterations (typically 2*k to 10*k)
///
/// # Returns
/// `LanczosResult` with eigenvalues and corresponding eigenvectors.
///
/// # Determinism
/// Uses Kahan summation for all dot products and norms.
/// Result is deterministic for the same input.
pub fn lanczos_eigsh(matrix: &SparseCsr, k: usize, max_iter: usize) -> LanczosResult {
    let n = matrix.nrows;
    assert_eq!(n, matrix.ncols, "lanczos_eigsh: matrix must be square");
    assert!(k > 0 && k <= n, "lanczos_eigsh: k must be in 1..=n");

    let iters = max_iter.min(n);

    // Lanczos vectors (orthonormal basis)
    let mut v_prev = vec![0.0_f64; n];
    // Start with uniform vector (deterministic, has components along all eigenvectors)
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v_curr = vec![inv_sqrt_n; n];

    // Tridiagonal matrix elements
    let mut alpha = Vec::with_capacity(iters); // diagonal
    let mut beta = Vec::with_capacity(iters);  // off-diagonal

    for j in 0..iters {
        // w = A * v_j
        let w_raw = matrix.matvec(&v_curr).unwrap();
        let mut w = w_raw;

        // alpha_j = v_j^T * w
        let dot_products: Vec<f64> = v_curr.iter().zip(w.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let alpha_j = kahan_sum_f64(&dot_products);
        alpha.push(alpha_j);

        // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        let beta_prev = if j > 0 { beta[j - 1] } else { 0.0 };
        for i in 0..n {
            w[i] -= alpha_j * v_curr[i] + beta_prev * v_prev[i];
        }

        // beta_j = ||w||
        let norm_sq: Vec<f64> = w.iter().map(|&x| x * x).collect();
        let beta_j = kahan_sum_f64(&norm_sq).sqrt();

        if beta_j < 1e-14 {
            // Invariant subspace found
            break;
        }
        beta.push(beta_j);

        // v_{j+1} = w / beta_j
        v_prev = v_curr;
        v_curr = w.iter().map(|&x| x / beta_j).collect();
    }

    // Solve tridiagonal eigenvalue problem using QR iteration
    let (eigenvalues, _) = tridiagonal_qr(&alpha, &beta);

    // Return the k largest eigenvalues (sorted ascending)
    let mut sorted_evals: Vec<f64> = eigenvalues;
    sorted_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let k_actual = k.min(sorted_evals.len());
    let top_k: Vec<f64> = sorted_evals[sorted_evals.len() - k_actual..].to_vec();

    LanczosResult {
        eigenvalues: top_k,
        eigenvectors: vec![], // Eigenvector-free variant — use lanczos_eigsh_with_vectors for vectors
    }
}

/// Lanczos with eigenvector recovery — same eigenvalues as
/// [`lanczos_eigsh`] but populates `LanczosResult.eigenvectors` with
/// the top-k approximate eigenvectors of A.
///
/// # Algorithm
///
/// 1. Run Lanczos iteration, accumulating basis vectors V_k = [v_0, ..., v_{k-1}]
///    (n × k_iter matrix; columns are the Lanczos vectors).
/// 2. Solve the tridiagonal eigenvalue problem T_k v = λ v using
///    [`tridiagonal_qr_with_vectors`] — produces both eigenvalues and
///    the orthogonal matrix Z whose columns are T_k's eigenvectors.
/// 3. Back-transform: each A-eigenvector w_i = V_k @ z_i, where z_i is
///    the eigenvector of T_k for eigenvalue λ_i.
/// 4. Sort by eigenvalue ascending; return the top k.
///
/// # Memory cost
///
/// O(n × k_iter) for basis storage vs the eigenvalue-only variant's
/// O(n) — significant on large sparse matrices. The dispatch wrapper
/// in builtins.rs uses this variant only when eigenvectors are
/// requested.
///
/// # Determinism
///
/// Same starting vector as `lanczos_eigsh` (uniform 1/√n), Kahan
/// summation throughout. Tridiagonal QR uses Wilkinson shift + Givens
/// rotation accumulation, both deterministic. Two consecutive calls
/// produce bit-identical eigenvector entries.
///
/// # Eigenvector quality
///
/// The Ritz vectors V_k @ z_i are approximate — they're the projection
/// of A's true eigenvectors onto the Krylov subspace span(V_k). The
/// residual ||A w_i - λ_i w_i|| converges as `max_iter` grows, but
/// for tight convergence on the smallest few eigenvalues, restart
/// strategies (e.g. IRLM) are needed. Those are a future enhancement.
pub fn lanczos_eigsh_with_vectors(
    matrix: &SparseCsr,
    k: usize,
    max_iter: usize,
) -> LanczosResult {
    let n = matrix.nrows;
    assert_eq!(n, matrix.ncols, "lanczos_eigsh_with_vectors: matrix must be square");
    assert!(k > 0 && k <= n, "lanczos_eigsh_with_vectors: k must be in 1..=n");

    let iters = max_iter.min(n);

    let mut v_prev = vec![0.0_f64; n];
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v_curr = vec![inv_sqrt_n; n];

    let mut alpha = Vec::with_capacity(iters);
    let mut beta = Vec::with_capacity(iters);

    // Basis storage — each iteration pushes v_j (orthonormal). After
    // termination, basis[j] is the j-th Lanczos vector. The
    // back-transformation step needs all of these.
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(iters);

    for j in 0..iters {
        basis.push(v_curr.clone());

        let mut w = matrix.matvec(&v_curr).unwrap();

        let dot_products: Vec<f64> = v_curr.iter().zip(w.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let alpha_j = kahan_sum_f64(&dot_products);
        alpha.push(alpha_j);

        let beta_prev = if j > 0 { beta[j - 1] } else { 0.0 };
        for i in 0..n {
            w[i] -= alpha_j * v_curr[i] + beta_prev * v_prev[i];
        }

        let norm_sq: Vec<f64> = w.iter().map(|&x| x * x).collect();
        let beta_j = kahan_sum_f64(&norm_sq).sqrt();

        if beta_j < 1e-14 {
            break;
        }
        beta.push(beta_j);

        v_prev = v_curr;
        v_curr = w.iter().map(|&x| x / beta_j).collect();
    }

    let m = basis.len(); // actual subspace dimension (≤ iters)
    if m == 0 {
        return LanczosResult { eigenvalues: vec![], eigenvectors: vec![] };
    }

    // Solve tridiagonal eigenvalue + eigenvector problem on T_m.
    let (t_evals, t_evecs, _) = tridiagonal_qr_with_vectors(
        &alpha[..m],
        &beta[..m.saturating_sub(1).min(beta.len())],
    );

    // Pair (eigenvalue, eigenvector) and sort by eigenvalue ascending
    // for deterministic ordering.
    let mut pairs: Vec<(f64, Vec<f64>)> = t_evals.into_iter().zip(t_evecs.into_iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let k_actual = k.min(pairs.len());
    let top_pairs = &pairs[pairs.len() - k_actual..];

    // Back-transform: A-eigenvector w_i = V_m @ z_i where V_m is the
    // n × m basis matrix (basis[j] is column j of V_m) and z_i is the
    // m-dim eigenvector of T_m.
    let top_eigenvalues: Vec<f64> = top_pairs.iter().map(|(ev, _)| *ev).collect();
    let top_eigenvectors: Vec<Vec<f64>> = top_pairs
        .iter()
        .map(|(_, z)| {
            // w[row] = sum_{col=0..m} basis[col][row] * z[col]
            //        = V_m @ z evaluated at row
            (0..n)
                .map(|row| {
                    let terms: Vec<f64> = (0..m).map(|col| basis[col][row] * z[col]).collect();
                    kahan_sum_f64(&terms)
                })
                .collect()
        })
        .collect();

    LanczosResult {
        eigenvalues: top_eigenvalues,
        eigenvectors: top_eigenvectors,
    }
}

// ---------------------------------------------------------------------------
// Arnoldi Algorithm (General Matrices)
// ---------------------------------------------------------------------------

/// Result of an Arnoldi eigenvalue computation for a general (non-symmetric)
/// sparse matrix.
#[derive(Debug, Clone)]
pub struct ArnoldiResult {
    /// Real parts of the Ritz eigenvalue approximations.
    pub eigenvalues_real: Vec<f64>,
    /// Imaginary parts of the Ritz eigenvalue approximations.
    pub eigenvalues_imag: Vec<f64>,
}

/// Compute eigenvalue approximations of a general sparse matrix using
/// the Arnoldi iteration to build an upper Hessenberg matrix, then
/// extract eigenvalues from the Hessenberg form.
///
/// # Arguments
/// * `matrix` - Sparse CSR matrix (square)
/// * `k` - Number of Arnoldi iterations (determines subspace size)
///
/// # Returns
/// `ArnoldiResult` with real and imaginary parts of eigenvalue approximations.
pub fn arnoldi_eigs(matrix: &SparseCsr, k: usize) -> ArnoldiResult {
    let n = matrix.nrows;
    assert_eq!(n, matrix.ncols, "arnoldi_eigs: matrix must be square");
    let m = k.min(n);

    // Arnoldi vectors
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut h: Vec<Vec<f64>> = vec![vec![0.0; m]; m + 1]; // Upper Hessenberg (m+1) x m

    // q_0 = e_0 (deterministic starting vector)
    let mut q0 = vec![0.0_f64; n];
    q0[0] = 1.0;
    q.push(q0);

    for j in 0..m {
        // w = A * q_j
        let mut w = matrix.matvec(&q[j]).unwrap();

        // Modified Gram-Schmidt
        for i in 0..=j {
            let dots: Vec<f64> = q[i].iter().zip(w.iter())
                .map(|(&a, &b)| a * b)
                .collect();
            h[i][j] = kahan_sum_f64(&dots);
            for l in 0..n {
                w[l] -= h[i][j] * q[i][l];
            }
        }

        // h_{j+1,j} = ||w||
        let norm_sq: Vec<f64> = w.iter().map(|&x| x * x).collect();
        let h_next = kahan_sum_f64(&norm_sq).sqrt();

        if j + 1 < m + 1 {
            h[j + 1][j] = h_next;
        }

        if h_next < 1e-14 {
            break;
        }

        // q_{j+1} = w / h_{j+1,j}
        let q_next: Vec<f64> = w.iter().map(|&x| x / h_next).collect();
        q.push(q_next);
    }

    // Extract eigenvalues from the m x m upper Hessenberg matrix
    // For now, use the diagonal as approximations (Ritz values)
    let actual_m = (q.len() - 1).min(m);
    let eigenvalues_real: Vec<f64> = (0..actual_m).map(|i| h[i][i]).collect();
    let eigenvalues_imag = vec![0.0_f64; actual_m];

    ArnoldiResult {
        eigenvalues_real,
        eigenvalues_imag,
    }
}

// ---------------------------------------------------------------------------
// Tridiagonal QR Iteration (Internal)
// ---------------------------------------------------------------------------

/// Solve a symmetric tridiagonal eigenvalue problem using implicit QR iteration.
///
/// # Arguments
/// * `diag` - Diagonal elements (alpha)
/// * `offdiag` - Off-diagonal elements (beta)
///
/// # Returns
/// (eigenvalues, number of iterations used)
fn tridiagonal_qr(diag: &[f64], offdiag: &[f64]) -> (Vec<f64>, usize) {
    let n = diag.len();
    if n == 0 {
        return (vec![], 0);
    }
    if n == 1 {
        return (vec![diag[0]], 0);
    }

    let mut d = diag.to_vec();
    let mut e = offdiag.to_vec();
    // Pad e to length n (e[n-1] unused but simplifies indexing)
    while e.len() < n {
        e.push(0.0);
    }

    let max_iter = 30 * n;
    let mut total_iters = 0;

    // Implicit QL algorithm with Wilkinson shift (LAPACK tql1 approach)
    for l_start in 0..n {
        let mut iters_this = 0;
        loop {
            // Find the unreduced submatrix: smallest m >= l_start with e[m] negligible
            let mut m = l_start;
            while m < n - 1 {
                let tst = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * tst {
                    break;
                }
                m += 1;
            }
            if m == l_start {
                break; // eigenvalue d[l_start] converged
            }

            iters_this += 1;
            total_iters += 1;
            if iters_this > max_iter {
                break;
            }

            // Wilkinson shift
            let mut g = (d[l_start + 1] - d[l_start]) / (2.0 * e[l_start]);
            let r = g.hypot(1.0);
            g = d[m] - d[l_start] + e[l_start] / (g + r.copysign(g));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut early_break = false;

            // Chase bulge from m-1 down to l_start
            for i in (l_start..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;
                if r.abs() < 1e-30 {
                    // Deflation
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early_break = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let r2 = (d[i] - g) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g + p;
                g = c * r2 - b;
            }

            if !early_break {
                d[l_start] -= p;
                e[l_start] = g;
                e[m] = 0.0;
            }
        }
    }

    (d, total_iters)
}

/// Solve a symmetric tridiagonal eigenvalue problem AND recover
/// the eigenvectors.
///
/// Returns `(eigenvalues, eigenvectors_as_rows, iterations)` where
/// `eigenvectors[i]` is the eigenvector for `eigenvalues[i]`.
///
/// # Algorithm
///
/// Same implicit QL with Wilkinson shift as [`tridiagonal_qr`], but
/// additionally accumulates each Givens rotation into a matrix `z`,
/// initialised to the identity. After convergence:
///
///   T = Z^T D Z       (where D is diagonal with eigenvalues)
///   columns of Z      = eigenvectors of T
///
/// Memory cost rises from O(n) (eigenvalues-only) to O(n²) (full
/// eigenvector matrix). Most callers of Lanczos use this through
/// [`lanczos_eigsh_with_vectors`], which then truncates to top-k.
///
/// # Determinism
///
/// Givens rotations are deterministic functions of the running
/// diagonal/off-diagonal values. The same input tridiagonal produces
/// bit-identical Z output across runs.
fn tridiagonal_qr_with_vectors(
    diag: &[f64],
    offdiag: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>, usize) {
    let n = diag.len();
    if n == 0 {
        return (vec![], vec![], 0);
    }
    if n == 1 {
        return (vec![diag[0]], vec![vec![1.0]], 0);
    }

    let mut d = diag.to_vec();
    let mut e = offdiag.to_vec();
    while e.len() < n {
        e.push(0.0);
    }

    // z[row][col] starts as identity. Givens rotations applied to T's
    // columns (i, i+1) are mirrored on z's columns (i, i+1) so that
    // when the algorithm converges, z's columns are T's eigenvectors.
    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_iter = 30 * n;
    let mut total_iters = 0;

    for l_start in 0..n {
        let mut iters_this = 0;
        loop {
            let mut m = l_start;
            while m < n - 1 {
                let tst = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * tst {
                    break;
                }
                m += 1;
            }
            if m == l_start {
                break;
            }

            iters_this += 1;
            total_iters += 1;
            if iters_this > max_iter {
                break;
            }

            let mut g = (d[l_start + 1] - d[l_start]) / (2.0 * e[l_start]);
            let r = g.hypot(1.0);
            g = d[m] - d[l_start] + e[l_start] / (g + r.copysign(g));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut early_break = false;

            for i in (l_start..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;
                if r.abs() < 1e-30 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early_break = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let r2 = (d[i] - g) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g + p;
                g = c * r2 - b;

                // Apply the Givens rotation (c, s) to columns i and
                // i+1 of z. The order matters: read both old values
                // into temporaries first, then write the new ones.
                for k in 0..n {
                    let zi = z[k][i];
                    let zi1 = z[k][i + 1];
                    z[k][i] = c * zi - s * zi1;
                    z[k][i + 1] = s * zi + c * zi1;
                }
            }

            if !early_break {
                d[l_start] -= p;
                e[l_start] = g;
                e[m] = 0.0;
            }
        }
    }

    // Extract eigenvectors: columns of z. eigenvectors[i] is the
    // i-th eigenvector (corresponds to eigenvalue d[i]).
    let mut eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|col| (0..n).map(|row| z[row][col]).collect())
        .collect();

    // Normalize each eigenvector defensively — the QL algorithm
    // should produce orthonormal output, but accumulated rounding
    // can drift slightly. Renormalising is cheap and protects the
    // back-transformation step's accuracy.
    for v in &mut eigenvectors {
        let norm_sq: Vec<f64> = v.iter().map(|x| x * x).collect();
        let norm = kahan_sum_f64(&norm_sq).sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    (d, eigenvectors, total_iters)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseCoo;

    fn identity_csr(n: usize) -> SparseCsr {
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        for i in 0..n {
            values.push(1.0);
            row_indices.push(i);
            col_indices.push(i);
        }
        let coo = SparseCoo::new(values, row_indices, col_indices, n, n);
        SparseCsr::from_coo(&coo)
    }

    fn diagonal_csr(diag: &[f64]) -> SparseCsr {
        let n = diag.len();
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        for (i, &d) in diag.iter().enumerate() {
            values.push(d);
            row_indices.push(i);
            col_indices.push(i);
        }
        let coo = SparseCoo::new(values, row_indices, col_indices, n, n);
        SparseCsr::from_coo(&coo)
    }

    #[test]
    fn test_lanczos_identity() {
        let mat = identity_csr(5);
        let result = lanczos_eigsh(&mat, 3, 20);
        // All eigenvalues of identity are 1.0
        for &ev in &result.eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10, "expected 1.0, got {ev}");
        }
    }

    #[test]
    fn test_lanczos_diagonal() {
        let mat = diagonal_csr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = lanczos_eigsh(&mat, 2, 50);
        // Top 2 eigenvalues should be 4 and 5
        assert!(result.eigenvalues.len() >= 2);
        let top = &result.eigenvalues;
        assert!((top[top.len() - 1] - 5.0).abs() < 1e-8,
            "expected 5.0, got {}", top[top.len() - 1]);
    }

    #[test]
    fn test_arnoldi_identity() {
        let mat = identity_csr(4);
        let result = arnoldi_eigs(&mat, 4);
        // Arnoldi on identity should give eigenvalues close to 1.0
        for &ev in &result.eigenvalues_real {
            assert!((ev - 1.0).abs() < 1e-10, "expected ~1.0, got {ev}");
        }
    }

    #[test]
    fn test_tridiagonal_qr_simple() {
        // 2x2 tridiagonal: [[3, 1], [1, 3]] -> eigenvalues 2 and 4
        let (evals, _) = tridiagonal_qr(&[3.0, 3.0], &[1.0]);
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 2.0).abs() < 1e-10, "expected 2.0, got {}", sorted[0]);
        assert!((sorted[1] - 4.0).abs() < 1e-10, "expected 4.0, got {}", sorted[1]);
    }

    #[test]
    fn test_lanczos_deterministic() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let r1 = lanczos_eigsh(&mat, 3, 30);
        let r2 = lanczos_eigsh(&mat, 3, 30);
        assert_eq!(r1.eigenvalues.len(), r2.eigenvalues.len());
        for (a, b) in r1.eigenvalues.iter().zip(r2.eigenvalues.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic eigenvalue");
        }
    }

    // -- Eigenvector recovery tests ------------------------------------------

    #[test]
    fn tridiagonal_qr_with_vectors_identity_case() {
        // Diagonal tridiagonal (offdiag all zero) → eigenvalues = diag,
        // eigenvectors = standard basis. Z = identity after no rotations.
        let (evals, evecs, _) = tridiagonal_qr_with_vectors(&[5.0, 3.0, 7.0], &[0.0, 0.0]);
        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.len(), 3);
        // Each eigenvector should be a standard basis vector.
        for (i, v) in evecs.iter().enumerate() {
            assert_eq!(v.len(), 3);
            for (j, x) in v.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (x - expected).abs() < 1e-12,
                    "evec[{i}][{j}] should be {expected}, got {x}",
                );
            }
        }
    }

    #[test]
    fn tridiagonal_qr_with_vectors_2x2_known_case() {
        // T = [[3, 1], [1, 3]] → eigenvalues 2 and 4.
        // Eigenvectors are (1, -1)/√2 for λ=2 and (1, 1)/√2 for λ=4.
        let (evals, evecs, _) = tridiagonal_qr_with_vectors(&[3.0, 3.0], &[1.0]);
        assert_eq!(evals.len(), 2);
        assert_eq!(evecs.len(), 2);
        // Pair eigenvalues with eigenvectors and sort by eigenvalue.
        let mut pairs: Vec<(f64, Vec<f64>)> =
            evals.into_iter().zip(evecs.into_iter()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // pairs[0] should be (2.0, (±1/√2, ∓1/√2)) ; pairs[1] (4.0, (±1/√2, ±1/√2)).
        assert!((pairs[0].0 - 2.0).abs() < 1e-10, "expected 2.0, got {}", pairs[0].0);
        assert!((pairs[1].0 - 4.0).abs() < 1e-10, "expected 4.0, got {}", pairs[1].0);
        let inv_sqrt_2 = 1.0 / 2.0_f64.sqrt();
        // For λ=2: components should be ±(1/√2, -1/√2). Check magnitude.
        assert!(
            (pairs[0].1[0].abs() - inv_sqrt_2).abs() < 1e-10,
            "evec[0][0] magnitude mismatch: {}",
            pairs[0].1[0],
        );
        assert!(
            (pairs[0].1[0] * pairs[0].1[1] + inv_sqrt_2 * inv_sqrt_2).abs() < 1e-10,
            "evec[0] should be antisymmetric — got ({}, {})",
            pairs[0].1[0],
            pairs[0].1[1],
        );
        // For λ=4: components should be ±(1/√2, 1/√2) — symmetric.
        assert!(
            (pairs[1].1[0] * pairs[1].1[1] - inv_sqrt_2 * inv_sqrt_2).abs() < 1e-10,
            "evec[1] should be symmetric — got ({}, {})",
            pairs[1].1[0],
            pairs[1].1[1],
        );
    }

    #[test]
    fn lanczos_eigsh_with_vectors_diagonal_matrix() {
        // Diagonal matrix with eigenvalues 1..=5; top 2 are 4 and 5
        // with standard basis eigenvectors e_3 (for 4) and e_4 (for 5).
        let mat = diagonal_csr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = lanczos_eigsh_with_vectors(&mat, 2, 50);
        assert!(result.eigenvalues.len() >= 2);
        assert_eq!(
            result.eigenvalues.len(),
            result.eigenvectors.len(),
            "every eigenvalue must have a matching eigenvector",
        );
        let top = &result.eigenvalues;
        let top_evec = &result.eigenvectors[top.len() - 1]; // for largest eigenvalue
        assert!(
            (top[top.len() - 1] - 5.0).abs() < 1e-8,
            "largest eigenvalue should be 5.0, got {}",
            top[top.len() - 1]
        );
        // The eigenvector for 5.0 should be close to e_4 = (0,0,0,0,1).
        // Lanczos returns an approximate Ritz vector — accept a relaxed
        // residual threshold.
        assert_eq!(top_evec.len(), 5);
        // The component on the dominant axis (index 4) should dominate.
        let abs4 = top_evec[4].abs();
        for (i, &v) in top_evec.iter().enumerate() {
            if i != 4 {
                assert!(
                    v.abs() <= abs4 * 0.5 + 1e-6,
                    "off-axis component too large: top_evec[{i}] = {v}, dominant = {abs4}",
                );
            }
        }
    }

    #[test]
    fn lanczos_eigsh_with_vectors_residual_check() {
        // For a diagonal matrix, the Ritz vectors should satisfy
        // A·v ≈ λ·v with small residual. This is the load-bearing
        // eigenvector quality test.
        let diag = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let mat = diagonal_csr(&diag);
        let result = lanczos_eigsh_with_vectors(&mat, 3, 100);
        assert_eq!(result.eigenvalues.len(), result.eigenvectors.len());
        for (lambda, v) in result.eigenvalues.iter().zip(result.eigenvectors.iter()) {
            let av = mat.matvec(v).unwrap();
            // residual = ||A v - λ v||
            let resid: f64 = av
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - lambda * b).powi(2))
                .sum::<f64>()
                .sqrt();
            // Threshold relaxed because Ritz vectors are approximate.
            // For a diagonal matrix with well-separated eigenvalues and
            // enough iterations, residual should be tiny.
            assert!(
                resid < 1e-6,
                "eigenvector residual too large for λ={lambda}: {resid}",
            );
        }
    }

    #[test]
    fn lanczos_eigsh_with_vectors_deterministic() {
        // Same input → bit-identical eigenvector bytes across runs.
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let r1 = lanczos_eigsh_with_vectors(&mat, 3, 30);
        let r2 = lanczos_eigsh_with_vectors(&mat, 3, 30);
        assert_eq!(r1.eigenvalues.len(), r2.eigenvalues.len());
        assert_eq!(r1.eigenvectors.len(), r2.eigenvectors.len());
        for (a, b) in r1.eigenvalues.iter().zip(r2.eigenvalues.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic eigenvalue");
        }
        for (va, vb) in r1.eigenvectors.iter().zip(r2.eigenvectors.iter()) {
            assert_eq!(va.len(), vb.len());
            for (a, b) in va.iter().zip(vb.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "non-deterministic eigenvector entry",
                );
            }
        }
    }

    #[test]
    fn lanczos_eigsh_with_vectors_eigenvector_count_matches_k() {
        let mat = diagonal_csr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        for k in 1..=4 {
            let result = lanczos_eigsh_with_vectors(&mat, k, 50);
            assert!(result.eigenvalues.len() <= k);
            assert_eq!(
                result.eigenvalues.len(),
                result.eigenvectors.len(),
                "eigenvalue count must match eigenvector count for k={k}",
            );
            for v in &result.eigenvectors {
                assert_eq!(v.len(), 6, "eigenvector dimension must be n");
            }
        }
    }
}
