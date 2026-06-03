//! Small dense linear algebra helpers for cjc-causal.
//!
//! Used exclusively by [`super::iv_regression`] for the HC1 sandwich SE
//! computation. We do not reuse `cjc_runtime::linalg::Tensor` because:
//!
//! 1. Our matrices are small (k × k where k = covariates + 2, typically < 20).
//!    The `Tensor` machinery's COW buffer allocation overhead would dominate.
//! 2. Inline Kahan summation is easier to audit per ADR-0043 §determinism.
//! 3. Keeping the dependency surface small means a future Determinism Auditor
//!    can read this file in one pass and confirm zero non-Kahan reductions.
//!
//! All matrices are row-major `Vec<f64>` of length `n * m`. Element `(i, j)`
//! lives at index `i * m + j`. Symmetric matrices store both triangles
//! (no packed storage) so callers don't need to remember whether to read
//! upper or lower.

use crate::error::CausalError;
use cjc_repro::KahanAccumulatorF64;

/// Compute the Cholesky factor `L` of an n×n symmetric positive-definite
/// matrix `A`, such that `L · Lᵀ = A`.
///
/// `L` is lower-triangular; the upper triangle of the returned buffer is
/// zero. Returns `CausalError::Numerical` if `A` is not positive-definite
/// (a non-positive diagonal pivot is the deterministic indicator).
pub fn cholesky_factor(a: &[f64], n: usize) -> Result<Vec<f64>, CausalError> {
    if a.len() != n * n {
        return Err(CausalError::Numerical {
            detail: format!("cholesky_factor: matrix has {} elements, expected {}", a.len(), n * n),
        });
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut acc = KahanAccumulatorF64::new();
            acc.add(a[i * n + j]);
            for k in 0..j {
                acc.add(-l[i * n + k] * l[j * n + k]);
            }
            let s = acc.finalize();
            if i == j {
                if !(s > 0.0) {
                    return Err(CausalError::Numerical {
                        detail: format!(
                            "cholesky_factor: matrix not positive-definite at diagonal {} (pivot = {})",
                            i, s
                        ),
                    });
                }
                l[i * n + i] = s.sqrt();
            } else {
                let pivot = l[j * n + j];
                if pivot == 0.0 {
                    return Err(CausalError::Numerical {
                        detail: format!("cholesky_factor: zero pivot at column {}", j),
                    });
                }
                l[i * n + j] = s / pivot;
            }
        }
    }
    Ok(l)
}

/// Invert a lower-triangular matrix `L` in place-style: returns `L⁻¹`.
///
/// The result is also lower-triangular. Used as the second half of
/// [`cholesky_invert`].
pub fn invert_lower_triangular(l: &[f64], n: usize) -> Result<Vec<f64>, CausalError> {
    if l.len() != n * n {
        return Err(CausalError::Numerical {
            detail: format!("invert_lower_triangular: matrix has {} elements, expected {}", l.len(), n * n),
        });
    }
    let mut linv = vec![0.0; n * n];
    for i in 0..n {
        let diag = l[i * n + i];
        if diag == 0.0 {
            return Err(CausalError::Numerical {
                detail: format!("invert_lower_triangular: zero diagonal at {}", i),
            });
        }
        linv[i * n + i] = 1.0 / diag;
        for j in 0..i {
            let mut acc = KahanAccumulatorF64::new();
            for k in j..i {
                acc.add(l[i * n + k] * linv[k * n + j]);
            }
            linv[i * n + j] = -acc.finalize() / diag;
        }
    }
    Ok(linv)
}

/// Invert a symmetric positive-definite matrix `A` via Cholesky decomposition.
///
/// Algorithm: `A = L Lᵀ`, so `A⁻¹ = L⁻ᵀ L⁻¹`. Both factors are lower-triangular
/// (or transposes thereof), so the matrix-product reduces to a sum over indices
/// `k ≥ max(i, j)` only.
///
/// All reductions are Kahan-summed.
pub fn cholesky_invert(a: &[f64], n: usize) -> Result<Vec<f64>, CausalError> {
    let l = cholesky_factor(a, n)?;
    let linv = invert_lower_triangular(&l, n)?;
    let mut a_inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = KahanAccumulatorF64::new();
            // (L⁻ᵀ)[i,k] = (L⁻¹)[k,i], non-zero only when k ≥ i.
            // (L⁻¹)[k,j] is non-zero only when k ≥ j.
            // Intersection: k ≥ max(i, j).
            for k in i.max(j)..n {
                acc.add(linv[k * n + i] * linv[k * n + j]);
            }
            a_inv[i * n + j] = acc.finalize();
        }
    }
    Ok(a_inv)
}

/// Gram matrix `XᵀX` where `X` is n × k row-major. Output is k × k symmetric.
///
/// Exploits symmetry: only the upper triangle is computed, then mirrored
/// to the lower. All inner products are Kahan-summed.
pub fn gram(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    debug_assert_eq!(x.len(), n * k, "gram: x must have n*k = {} elements", n * k);
    let mut g = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let mut acc = KahanAccumulatorF64::new();
            for row in 0..n {
                acc.add(x[row * k + i] * x[row * k + j]);
            }
            let v = acc.finalize();
            g[i * k + j] = v;
            g[j * k + i] = v;
        }
    }
    g
}

/// Weighted Gram matrix `Xᵀ diag(w) X`, where `X` is n × k and `w` has length n.
///
/// Used inside the HC1 sandwich SE computation, with `w[i] = e_i²`.
/// All inner products are Kahan-summed.
pub fn gram_weighted(x: &[f64], n: usize, k: usize, w: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), n * k, "gram_weighted: x must have n*k elements");
    debug_assert_eq!(w.len(), n, "gram_weighted: w must have n elements");
    let mut g = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let mut acc = KahanAccumulatorF64::new();
            for row in 0..n {
                acc.add(w[row] * x[row * k + i] * x[row * k + j]);
            }
            let v = acc.finalize();
            g[i * k + j] = v;
            g[j * k + i] = v;
        }
    }
    g
}

/// Square-matrix multiply `C = A · B` for `n × n` matrices. All inner products
/// are Kahan-summed.
pub fn matmul_square(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n, "matmul_square: a must be n*n");
    debug_assert_eq!(b.len(), n * n, "matmul_square: b must be n*n");
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = KahanAccumulatorF64::new();
            for kk in 0..n {
                acc.add(a[i * n + kk] * b[kk * n + j]);
            }
            c[i * n + j] = acc.finalize();
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_n(n: usize) -> Vec<f64> {
        let mut i = vec![0.0; n * n];
        for k in 0..n {
            i[k * n + k] = 1.0;
        }
        i
    }

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn cholesky_factor_of_identity_is_identity() {
        let i = identity_n(3);
        let l = cholesky_factor(&i, 3).unwrap();
        assert!(approx_eq(&l, &i, 1e-12));
    }

    #[test]
    fn cholesky_factor_of_known_2x2_matches_textbook() {
        // A = [[4, 2], [2, 5]] → L = [[2, 0], [1, 2]]
        let a = vec![4.0, 2.0, 2.0, 5.0];
        let l = cholesky_factor(&a, 2).unwrap();
        // L[0,0] = 2, L[1,0] = 1, L[1,1] = 2.
        assert!((l[0] - 2.0).abs() < 1e-12);
        assert_eq!(l[1], 0.0);
        assert!((l[2] - 1.0).abs() < 1e-12);
        assert!((l[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_factor_rejects_non_positive_definite() {
        // Diagonal of zero pivots → not PSD.
        let a = vec![0.0, 0.0, 0.0, 1.0];
        assert!(cholesky_factor(&a, 2).is_err());
    }

    #[test]
    fn cholesky_factor_rejects_negative_pivot() {
        let a = vec![-1.0, 0.0, 0.0, 1.0];
        assert!(cholesky_factor(&a, 2).is_err());
    }

    #[test]
    fn invert_lower_triangular_of_identity_is_identity() {
        let i = identity_n(3);
        let linv = invert_lower_triangular(&i, 3).unwrap();
        assert!(approx_eq(&linv, &i, 1e-12));
    }

    #[test]
    fn cholesky_invert_yields_identity_when_multiplied_back() {
        // A = [[4, 2], [2, 5]]; expect A * A^-1 = I.
        let a = vec![4.0, 2.0, 2.0, 5.0];
        let a_inv = cholesky_invert(&a, 2).unwrap();
        let product = matmul_square(&a, &a_inv, 2);
        assert!(approx_eq(&product, &identity_n(2), 1e-10));
    }

    #[test]
    fn cholesky_invert_3x3_matches_known_inverse() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] (textbook SPD).
        // Compute A^-1, then verify A * A^-1 = I.
        let a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let a_inv = cholesky_invert(&a, 3).unwrap();
        let product = matmul_square(&a, &a_inv, 3);
        assert!(approx_eq(&product, &identity_n(3), 1e-8));
    }

    #[test]
    fn gram_of_single_column_is_sum_of_squares() {
        // X = [[1], [2], [3]] → X'X = [[14]]
        let x = vec![1.0, 2.0, 3.0];
        let g = gram(&x, 3, 1);
        assert!((g[0] - 14.0).abs() < 1e-12);
    }

    #[test]
    fn gram_of_orthonormal_rows_is_diagonal() {
        // X = [[1, 0], [0, 1], [0, 0]] → X'X = [[1, 0], [0, 1]]
        let x = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let g = gram(&x, 3, 2);
        assert!(approx_eq(&g, &identity_n(2), 1e-12));
    }

    #[test]
    fn gram_weighted_with_unit_weights_equals_gram() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0];
        let g_plain = gram(&x, 2, 2);
        let g_w = gram_weighted(&x, 2, 2, &w);
        assert!(approx_eq(&g_plain, &g_w, 1e-12));
    }

    #[test]
    fn gram_weighted_scales_with_weights() {
        // X = [[1], [2]], w = [3, 4] → X' W X = 3*1 + 4*4 = 19
        let x = vec![1.0, 2.0];
        let w = vec![3.0, 4.0];
        let g = gram_weighted(&x, 2, 1, &w);
        assert!((g[0] - 19.0).abs() < 1e-12);
    }

    #[test]
    fn matmul_square_with_identity_is_identity() {
        let i = identity_n(3);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let product = matmul_square(&a, &i, 3);
        assert!(approx_eq(&product, &a, 1e-12));
        let product = matmul_square(&i, &a, 3);
        assert!(approx_eq(&product, &a, 1e-12));
    }

    #[test]
    fn matmul_square_known_2x2() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_square(&a, &b, 2);
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert!(approx_eq(&c, &expected, 1e-12));
    }

    #[test]
    fn cholesky_factor_size_mismatch_returns_error() {
        let a = vec![1.0, 2.0, 3.0]; // 3 elements but n=2 needs 4
        assert!(cholesky_factor(&a, 2).is_err());
    }
}
