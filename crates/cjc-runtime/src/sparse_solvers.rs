//! Iterative solvers for sparse linear systems.
//!
//! All solvers operate on CSR matrices and use deterministic floating-point
//! reductions via `binned_sum_f64` to guarantee bit-identical results across
//! runs and platforms.

use crate::accumulator::binned_sum_f64;
use crate::sparse::SparseCsr;

/// Result of an iterative solver.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual norm (L2).
    pub residual: f64,
    /// Whether the solver converged to within the requested tolerance.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Vector helpers — all reductions use binned_sum_f64 for determinism.
// ---------------------------------------------------------------------------

/// Deterministic dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let products: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    binned_sum_f64(&products)
}

/// Deterministic L2 norm.
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// y = a*x + y  (in-place AXPY).
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// r = b - A*x
fn compute_residual(a: &SparseCsr, x: &[f64], b: &[f64]) -> Vec<f64> {
    let ax = spmv(a, x);
    b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect()
}

/// Sparse matrix-vector product using binned summation for each row.
fn spmv(a: &SparseCsr, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; a.nrows];
    for row in 0..a.nrows {
        let start = a.row_offsets[row];
        let end = a.row_offsets[row + 1];
        let products: Vec<f64> = (start..end)
            .map(|idx| a.values[idx] * x[a.col_indices[idx]])
            .collect();
        y[row] = binned_sum_f64(&products);
    }
    y
}

// ---------------------------------------------------------------------------
// Conjugate Gradient (CG)
// ---------------------------------------------------------------------------

/// Conjugate Gradient solver for symmetric positive-definite systems Ax = b.
///
/// # Arguments
/// * `a` — SPD sparse matrix (must be square, n x n)
/// * `b` — right-hand side vector (length n)
/// * `tol` — convergence tolerance on the relative residual norm
/// * `max_iter` — maximum number of iterations
///
/// # Determinism
/// All inner products and norms use `binned_sum_f64`.
pub fn cg_solve(a: &SparseCsr, b: &[f64], tol: f64, max_iter: usize) -> SolverResult {
    let n = b.len();
    assert_eq!(a.nrows, n, "cg_solve: matrix rows must match rhs length");
    assert_eq!(a.ncols, n, "cg_solve: matrix must be square");

    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec(); // r = b - A*0 = b
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);
    let b_norm = norm2(b);

    if b_norm == 0.0 {
        return SolverResult {
            x,
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    for iter in 0..max_iter {
        let ap = spmv(a, &p);
        let p_ap = dot(&p, &ap);

        if p_ap == 0.0 {
            // Breakdown — p is in the null space
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: norm2(&r) / b_norm,
                converged: false,
            };
        }

        let alpha = rs_old / p_ap;

        // x = x + alpha * p
        axpy(alpha, &p, &mut x);
        // r = r - alpha * A*p
        axpy(-alpha, &ap, &mut r);

        let rs_new = dot(&r, &r);
        let res_norm = rs_new.sqrt() / b_norm;

        if res_norm < tol {
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: res_norm,
                converged: true,
            };
        }

        let beta = rs_new / rs_old;
        // p = r + beta * p
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    SolverResult {
        x,
        iterations: max_iter,
        residual: norm2(&r) / b_norm,
        converged: false,
    }
}

// ---------------------------------------------------------------------------
// GMRES (Generalized Minimum Residual)
// ---------------------------------------------------------------------------

/// GMRES solver for general (non-symmetric) systems Ax = b with restarts.
///
/// # Arguments
/// * `a` — sparse matrix (must be square, n x n)
/// * `b` — right-hand side vector (length n)
/// * `tol` — convergence tolerance on the relative residual norm
/// * `max_iter` — maximum total number of iterations
/// * `restart` — restart dimension (Krylov subspace size before restart)
///
/// # Determinism
/// All inner products, norms, and Givens rotations use deterministic arithmetic.
pub fn gmres_solve(
    a: &SparseCsr,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> SolverResult {
    let n = b.len();
    assert_eq!(a.nrows, n, "gmres_solve: matrix rows must match rhs length");
    assert_eq!(a.ncols, n, "gmres_solve: matrix must be square");

    let b_norm = norm2(b);
    if b_norm == 0.0 {
        return SolverResult {
            x: vec![0.0; n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut x = vec![0.0f64; n];
    let mut total_iter = 0;

    while total_iter < max_iter {
        let r = compute_residual(a, &x, b);
        let r_norm = norm2(&r);

        if r_norm / b_norm < tol {
            return SolverResult {
                x,
                iterations: total_iter,
                residual: r_norm / b_norm,
                converged: true,
            };
        }

        // Arnoldi + Givens rotation based GMRES
        let m = restart.min(max_iter - total_iter);

        // Krylov basis vectors V[0..m+1], each of length n
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        // v[0] = r / ||r||
        let v0: Vec<f64> = r.iter().map(|&ri| ri / r_norm).collect();
        v.push(v0);

        // Upper Hessenberg matrix H (stored as (m+1) x m)
        let mut h = vec![vec![0.0f64; m]; m + 1];

        // Givens rotation parameters
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];

        // Right-hand side of the least-squares problem
        let mut g = vec![0.0f64; m + 1];
        g[0] = r_norm;

        let mut last_res = r_norm / b_norm;

        for j in 0..m {
            total_iter += 1;

            // w = A * v[j]
            let mut w = spmv(a, &v[j]);

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                // w = w - h[i][j] * v[i]
                axpy(-h[i][j], &v[i], &mut w);
            }

            h[j + 1][j] = norm2(&w);

            if h[j + 1][j] > 1e-14 {
                let inv = 1.0 / h[j + 1][j];
                let vn: Vec<f64> = w.iter().map(|&wi| wi * inv).collect();
                v.push(vn);
            } else {
                // Lucky breakdown — w is zero, push zero vector
                v.push(vec![0.0; n]);
            }

            // Apply previous Givens rotations to column j of H
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = tmp;
            }

            // Compute new Givens rotation for row (j, j+1)
            let (c, s) = givens_rotation(h[j][j], h[j + 1][j]);
            cs[j] = c;
            sn[j] = s;

            h[j][j] = c * h[j][j] + s * h[j + 1][j];
            h[j + 1][j] = 0.0;

            // Update g
            let tmp = cs[j] * g[j] + sn[j] * g[j + 1];
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
            g[j] = tmp;

            last_res = g[j + 1].abs() / b_norm;

            if last_res < tol {
                // Solve the upper triangular system H * y = g
                let y = solve_upper_triangular(&h, &g, j + 1);
                // x = x + V * y
                update_solution(&mut x, &v, &y, j + 1);
                return SolverResult {
                    x,
                    iterations: total_iter,
                    residual: last_res,
                    converged: true,
                };
            }
        }

        // End of restart cycle — solve and update
        let y = solve_upper_triangular(&h, &g, m);
        update_solution(&mut x, &v, &y, m);

        if last_res < tol {
            return SolverResult {
                x,
                iterations: total_iter,
                residual: last_res,
                converged: true,
            };
        }
    }

    let r = compute_residual(a, &x, b);
    SolverResult {
        x,
        iterations: total_iter,
        residual: norm2(&r) / b_norm,
        converged: false,
    }
}

/// Compute Givens rotation parameters (c, s) such that
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if a.abs() > b.abs() {
        let t = b / a;
        let r = (1.0 + t * t).sqrt();
        let c = 1.0 / r;
        (c, c * t)
    } else {
        let t = a / b;
        let r = (1.0 + t * t).sqrt();
        let s = 1.0 / r;
        (s * t, s)
    }
}

/// Solve upper triangular system H[0..k, 0..k] * y = g[0..k] by back substitution.
fn solve_upper_triangular(h: &[Vec<f64>], g: &[f64], k: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut sum_terms: Vec<f64> = Vec::with_capacity(k - i);
        sum_terms.push(g[i]);
        for j in (i + 1)..k {
            sum_terms.push(-h[i][j] * y[j]);
        }
        let s = binned_sum_f64(&sum_terms);
        if h[i][i].abs() > 1e-30 {
            y[i] = s / h[i][i];
        }
    }
    y
}

/// x += V[0..k] * y[0..k]
fn update_solution(x: &mut [f64], v: &[Vec<f64>], y: &[f64], k: usize) {
    for i in 0..k {
        axpy(y[i], &v[i], x);
    }
}

// ---------------------------------------------------------------------------
// BiCGSTAB (Biconjugate Gradient Stabilized)
// ---------------------------------------------------------------------------

/// BiCGSTAB solver for general (non-symmetric) systems Ax = b.
///
/// # Arguments
/// * `a` — sparse matrix (must be square, n x n)
/// * `b` — right-hand side vector (length n)
/// * `tol` — convergence tolerance on the relative residual norm
/// * `max_iter` — maximum number of iterations
///
/// # Determinism
/// All inner products and reductions use `binned_sum_f64`.
pub fn bicgstab_solve(
    a: &SparseCsr,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> SolverResult {
    let n = b.len();
    assert_eq!(a.nrows, n, "bicgstab_solve: matrix rows must match rhs length");
    assert_eq!(a.ncols, n, "bicgstab_solve: matrix must be square");

    let b_norm = norm2(b);
    if b_norm == 0.0 {
        return SolverResult {
            x: vec![0.0; n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec(); // r = b - A*0 = b
    let r0_hat = r.clone();  // shadow residual, kept fixed

    let mut rho = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;

    let mut v = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];

    for iter in 0..max_iter {
        let rho_new = dot(&r0_hat, &r);

        if rho_new.abs() < 1e-30 {
            // Breakdown
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: norm2(&r) / b_norm,
                converged: false,
            };
        }

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A * p
        v = spmv(a, &p);

        let r0_v = dot(&r0_hat, &v);
        if r0_v.abs() < 1e-30 {
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: norm2(&r) / b_norm,
                converged: false,
            };
        }

        alpha = rho / r0_v;

        // s = r - alpha * v
        let s: Vec<f64> = r.iter().zip(v.iter()).map(|(&ri, &vi)| ri - alpha * vi).collect();

        let s_norm = norm2(&s) / b_norm;
        if s_norm < tol {
            // x = x + alpha * p
            axpy(alpha, &p, &mut x);
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: s_norm,
                converged: true,
            };
        }

        // t = A * s
        let t = spmv(a, &s);

        let t_t = dot(&t, &t);
        if t_t.abs() < 1e-30 {
            axpy(alpha, &p, &mut x);
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: norm2(&s) / b_norm,
                converged: false,
            };
        }

        omega = dot(&t, &s) / t_t;

        // x = x + alpha * p + omega * s
        axpy(alpha, &p, &mut x);
        axpy(omega, &s, &mut x);

        // r = s - omega * t
        r = s.iter().zip(t.iter()).map(|(&si, &ti)| si - omega * ti).collect();

        let res_norm = norm2(&r) / b_norm;
        if res_norm < tol {
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: res_norm,
                converged: true,
            };
        }

        if omega.abs() < 1e-30 {
            return SolverResult {
                x,
                iterations: iter + 1,
                residual: res_norm,
                converged: false,
            };
        }
    }

    let res = norm2(&r) / b_norm;
    SolverResult {
        x,
        iterations: max_iter,
        residual: res,
        converged: false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseCsr;

    /// Build a CSR matrix from dense data.
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

    /// Build a tridiagonal SPD matrix: diag=4, off-diag=-1.
    fn tridiag_spd(n: usize) -> SparseCsr {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 4.0;
            if i > 0 {
                data[i * n + (i - 1)] = -1.0;
            }
            if i + 1 < n {
                data[i * n + (i + 1)] = -1.0;
            }
        }
        csr_from_dense(&data, n, n)
    }

    // -- CG tests --

    #[test]
    fn test_cg_tridiag() {
        let n = 10;
        let a = tridiag_spd(n);
        let b: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();

        let result = cg_solve(&a, &b, 1e-10, 100);
        assert!(result.converged, "CG did not converge: residual={}", result.residual);
        assert!(result.residual < 1e-10);

        // Verify A*x ≈ b
        let ax = spmv(&a, &result.x);
        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "CG solution mismatch at i={}: ax={} b={}",
                i, ax[i], b[i]
            );
        }
    }

    #[test]
    fn test_cg_identity() {
        let a = csr_from_dense(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = vec![3.0, 7.0];
        let result = cg_solve(&a, &b, 1e-12, 10);
        assert!(result.converged);
        assert!((result.x[0] - 3.0).abs() < 1e-10);
        assert!((result.x[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_zero_rhs() {
        let a = tridiag_spd(5);
        let b = vec![0.0; 5];
        let result = cg_solve(&a, &b, 1e-10, 100);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for &xi in &result.x {
            assert_eq!(xi, 0.0);
        }
    }

    #[test]
    fn test_cg_determinism() {
        let a = tridiag_spd(20);
        let b: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();

        let r1 = cg_solve(&a, &b, 1e-10, 200);
        let r2 = cg_solve(&a, &b, 1e-10, 200);

        assert_eq!(r1.x, r2.x, "CG is not deterministic");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.residual, r2.residual);
    }

    // -- GMRES tests --

    #[test]
    fn test_gmres_nonsymmetric() {
        // Non-symmetric system
        let a = csr_from_dense(
            &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 2.0],
            3, 3,
        );
        let b = vec![1.0, 2.0, 3.0];

        let result = gmres_solve(&a, &b, 1e-10, 100, 30);
        assert!(result.converged, "GMRES did not converge: residual={}", result.residual);

        let ax = spmv(&a, &result.x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "GMRES mismatch at i={}: ax={} b={}",
                i, ax[i], b[i]
            );
        }
    }

    #[test]
    fn test_gmres_identity() {
        let a = csr_from_dense(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
        let b = vec![5.0, 6.0, 7.0];
        let result = gmres_solve(&a, &b, 1e-12, 10, 10);
        assert!(result.converged, "GMRES did not converge: residual={}", result.residual);
        for i in 0..3 {
            assert!((result.x[i] - b[i]).abs() < 1e-10,
                "GMRES identity mismatch at i={}: x={} b={}", i, result.x[i], b[i]);
        }
    }

    #[test]
    fn test_gmres_zero_rhs() {
        let a = csr_from_dense(&[2.0, 1.0, 0.0, 3.0], 2, 2);
        let b = vec![0.0, 0.0];
        let result = gmres_solve(&a, &b, 1e-10, 100, 10);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_gmres_determinism() {
        let a = csr_from_dense(
            &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 2.0],
            3, 3,
        );
        let b = vec![1.0, 2.0, 3.0];

        let r1 = gmres_solve(&a, &b, 1e-10, 100, 30);
        let r2 = gmres_solve(&a, &b, 1e-10, 100, 30);

        assert_eq!(r1.x, r2.x, "GMRES is not deterministic");
        assert_eq!(r1.iterations, r2.iterations);
    }

    // -- BiCGSTAB tests --

    #[test]
    fn test_bicgstab_nonsymmetric() {
        let a = csr_from_dense(
            &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 2.0],
            3, 3,
        );
        let b = vec![1.0, 2.0, 3.0];

        let result = bicgstab_solve(&a, &b, 1e-10, 100);
        assert!(result.converged, "BiCGSTAB did not converge: residual={}", result.residual);

        let ax = spmv(&a, &result.x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "BiCGSTAB mismatch at i={}: ax={} b={}",
                i, ax[i], b[i]
            );
        }
    }

    #[test]
    fn test_bicgstab_spd() {
        // BiCGSTAB should also work for SPD systems
        let a = tridiag_spd(10);
        let b: Vec<f64> = (1..=10).map(|i| i as f64).collect();

        let result = bicgstab_solve(&a, &b, 1e-10, 200);
        assert!(result.converged, "BiCGSTAB did not converge: residual={}", result.residual);

        let ax = spmv(&a, &result.x);
        for i in 0..10 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "BiCGSTAB SPD mismatch at i={}",
                i
            );
        }
    }

    #[test]
    fn test_bicgstab_identity() {
        let a = csr_from_dense(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = vec![3.0, 7.0];
        let result = bicgstab_solve(&a, &b, 1e-12, 10);
        assert!(result.converged);
        assert!((result.x[0] - 3.0).abs() < 1e-10);
        assert!((result.x[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_bicgstab_zero_rhs() {
        let a = csr_from_dense(&[2.0, 1.0, 0.0, 3.0], 2, 2);
        let b = vec![0.0, 0.0];
        let result = bicgstab_solve(&a, &b, 1e-10, 100);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_bicgstab_determinism() {
        let a = tridiag_spd(15);
        let b: Vec<f64> = (0..15).map(|i| (i as f64 * 0.7).cos()).collect();

        let r1 = bicgstab_solve(&a, &b, 1e-10, 200);
        let r2 = bicgstab_solve(&a, &b, 1e-10, 200);

        assert_eq!(r1.x, r2.x, "BiCGSTAB is not deterministic");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.residual, r2.residual);
    }

    // -- Cross-solver agreement --

    #[test]
    fn test_solvers_agree_on_spd_system() {
        let a = tridiag_spd(8);
        let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let cg = cg_solve(&a, &b, 1e-12, 200);
        let gmres = gmres_solve(&a, &b, 1e-12, 200, 20);
        let bicg = bicgstab_solve(&a, &b, 1e-12, 200);

        assert!(cg.converged);
        assert!(gmres.converged);
        assert!(bicg.converged);

        // All three should produce the same solution (within tolerance)
        for i in 0..8 {
            assert!(
                (cg.x[i] - gmres.x[i]).abs() < 1e-8,
                "CG vs GMRES disagree at i={}: {} vs {}",
                i, cg.x[i], gmres.x[i]
            );
            assert!(
                (cg.x[i] - bicg.x[i]).abs() < 1e-8,
                "CG vs BiCGSTAB disagree at i={}: {} vs {}",
                i, cg.x[i], bicg.x[i]
            );
        }
    }
}
