//! Advisory low-rank summary via deterministic power iteration with
//! deflation.
//!
//! `LowRankAdvisory` compresses a tall-skinny f64 matrix `M ∈ ℝ^(m × n)`
//! into the rank-K approximation
//!
//! ```text
//!     M ≈ Σ_{k=0}^{K-1} σ_k u_k v_k^T
//! ```
//!
//! where `σ_0 ≥ σ_1 ≥ ... ≥ σ_{K-1} ≥ 0` and `(u_k, v_k)` are normalized
//! left/right singular vectors. The summary is **advisory only** — the
//! constructor refuses to operate on
//! [`Criticality::SemanticCritical`](crate::candidate::Criticality)
//! inputs at the [`crate::candidate::CompressionCandidate`] layer.
//!
//! ## Algorithm
//!
//! For each rank `k = 0..K`:
//!
//! 1. Start with a deterministic unit vector `v` (alternating ±1 pattern,
//!    normalized — no RNG).
//! 2. Power-iterate on `B = M^T M`: `v ← B v / ||B v||`, with Gram-Schmidt
//!    deflation against all previously-found `v_0..v_{k-1}`.
//! 3. Stop when either the change between iterates falls below
//!    [`POWER_TOL`] or [`POWER_MAX_ITER`] is reached.
//! 4. `σ_k = √(v^T B v)`, `u_k = M v_k / σ_k`.
//! 5. **Sign stabilization**: find the argmax-magnitude index in `v_k`
//!    (smallest index wins ties — deterministic); if that element is
//!    negative, flip the sign of both `u_k` AND `v_k`.
//!
//! ## Why not a full Jacobi SVD?
//!
//! Jacobi SVD is more robust on tightly-clustered singular values, but
//! advisory CANA features rarely have pathological spectra (they're
//! summaries of compiler decisions, not signal-processing data). Power
//! iteration with deflation is ~150 LOC, deterministic by construction
//! (fixed seed vector, fixed iteration count cap), and converges on the
//! shapes we actually see. The full Jacobi pattern is reserved for
//! [`crate::tensor_train`] where it's already imported via
//! [`cjc_quantum::mps`].
//!
//! ## Determinism notes
//!
//! - Every reduction uses [`cjc_repro::KahanAccumulatorF64`]. No naive
//!   `iter().sum()`.
//! - Initial `v` is computed without RNG; the seed pattern depends only
//!   on `n`.
//! - Sign stabilization picks the smallest argmax index on ties (NOT the
//!   last-seen — argmax otherwise depends on iteration order).
//! - Output bytes are stamped with a magic header + input/summary hashes
//!   so corruption is detected before reconstruction is consumed.
//! - We *report* observed Frobenius error in [`LowRankPayload`] but do
//!   NOT compare it to a tolerance here — that comparison lives at the
//!   [`crate::candidate::CompressionCandidate`] layer once the candidate
//!   is wrapped into a plan.

use cjc_cana::hash::hash_bytes;
use cjc_repro::KahanAccumulatorF64;

use crate::candidate::CompressionError;

const POWER_MAX_ITER: usize = 256;
const POWER_TOL: f64 = 1e-12;
const MAGIC: &[u8; 4] = b"CLR0";

/// Output of [`compress_low_rank`].
///
/// `u` is stored row-major as `(rows × rank)`, `v` row-major as
/// `(cols × rank)`. `singular_values` has length `rank`. The reconstructed
/// matrix at index `(r, c)` is
///
/// ```text
///   Σ_k singular_values[k] * u[r * rank + k] * v[c * rank + k]
/// ```
#[derive(Debug, Clone)]
pub struct LowRankPayload {
    /// Rows of the original matrix.
    pub rows: usize,
    /// Columns of the original matrix.
    pub cols: usize,
    /// Selected rank `K`. Always `≤ min(rows, cols)`.
    pub rank: usize,
    /// Singular values in descending order.
    pub singular_values: Vec<f64>,
    /// Left singular vectors (column-major within rows, but flattened
    /// row-major for storage; see struct docs for the index formula).
    pub u: Vec<f64>,
    /// Right singular vectors (same layout convention as `u`).
    pub v: Vec<f64>,
    /// FNV-1a hash of the original matrix's bit pattern.
    pub input_hash: u64,
    /// FNV-1a hash of the summary's canonical bytes (see
    /// [`Self::canonical_bytes`]).
    pub summary_hash: u64,
    /// Observed relative Frobenius error
    /// `||M - M̂||_F / ||M||_F` ∈ `[0, 1]`. `0.0` for a zero-norm input
    /// (in which case the summary is the zero matrix).
    pub frobenius_error: f64,
}

impl LowRankPayload {
    /// Canonical bytes: magic + shape + singular values + U + V + hashes.
    /// Deterministic for byte-identical input.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MAGIC.len() + 16 + self.u.len() * 8 + self.v.len() * 8);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.rows as u32).to_le_bytes());
        out.extend_from_slice(&(self.cols as u32).to_le_bytes());
        out.extend_from_slice(&(self.rank as u32).to_le_bytes());
        for s in &self.singular_values {
            out.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        for x in &self.u {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        for x in &self.v {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&self.input_hash.to_le_bytes());
        out.extend_from_slice(&self.frobenius_error.to_bits().to_le_bytes());
        out
    }

    /// Reconstruct the rank-K approximation as a row-major `rows × cols`
    /// matrix. Useful for validation tests and downstream consumers.
    pub fn reconstruct(&self) -> Vec<f64> {
        let mut m_hat = vec![0.0f64; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut acc = KahanAccumulatorF64::new();
                for k in 0..self.rank {
                    acc.add(
                        self.singular_values[k]
                            * self.u[r * self.rank + k]
                            * self.v[c * self.rank + k],
                    );
                }
                m_hat[r * self.cols + c] = acc.finalize();
            }
        }
        m_hat
    }
}

/// Compress an `m × n` row-major f64 matrix into the rank-`max_rank`
/// approximation (capped at `min(m, n)`).
///
/// The caller decides `max_rank`; a typical CANA feature summary would
/// request `max_rank = 4` on a 32×16 matrix. The function does NOT
/// enforce a tolerance — the candidate wrapper layer compares
/// `frobenius_error` against the declared
/// [`Criticality::AdvisoryOnly::tolerance_f`](crate::candidate::Criticality::AdvisoryOnly)
/// budget and surfaces [`CompressionError::ToleranceExceeded`] if needed.
///
/// Returns [`CompressionError::UnsupportedShape`] if the matrix is empty
/// or `max_rank == 0`.
pub fn compress_low_rank(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    max_rank: usize,
) -> Result<LowRankPayload, CompressionError> {
    if rows == 0 || cols == 0 {
        return Err(CompressionError::UnsupportedShape {
            reason: "empty matrix",
        });
    }
    if matrix.len() != rows * cols {
        return Err(CompressionError::UnsupportedShape {
            reason: "matrix length mismatches rows*cols",
        });
    }
    if max_rank == 0 {
        return Err(CompressionError::UnsupportedShape {
            reason: "max_rank must be >= 1",
        });
    }
    if !matrix.iter().all(|x| x.is_finite()) {
        return Err(CompressionError::UnsupportedShape {
            reason: "matrix contains non-finite entries",
        });
    }

    let rank = max_rank.min(rows.min(cols));
    let input_hash = hash_bytes_f64(matrix);

    // Compute B = M^T M (cols × cols).
    let b = mt_m(matrix, rows, cols);

    // Run power iteration with deflation for `rank` eigenpairs.
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(rank);
    let mut sigmas: Vec<f64> = Vec::with_capacity(rank);

    for k in 0..rank {
        // Deterministic initial vector — alternating ±1 normalized to unit
        // length. No RNG.
        let mut v = initial_unit_vector(cols, k);
        gram_schmidt_against(&mut v, &vs);

        // If after GS the vector collapsed to ~zero, rank is exhausted.
        if vector_norm(&v) < POWER_TOL {
            break;
        }
        // Renormalize after GS.
        let inv_norm = 1.0 / vector_norm(&v);
        for x in v.iter_mut() {
            *x *= inv_norm;
        }

        let mut prev_v = v.clone();
        let mut eigenvalue = 0.0f64;
        for _ in 0..POWER_MAX_ITER {
            // w = B * v
            let mut w = matvec(&b, cols, &v);
            // Deflate against previous eigenvectors.
            gram_schmidt_against(&mut w, &vs);
            let norm = vector_norm(&w);
            if norm < POWER_TOL {
                break;
            }
            eigenvalue = norm;
            // Normalize.
            for x in w.iter_mut() {
                *x /= norm;
            }
            // Convergence check: ||v_new - v_prev|| (with sign-flip
            // tolerance because eigenvectors are defined up to sign).
            let direct = diff_norm(&w, &prev_v);
            let flipped = sum_norm(&w, &prev_v);
            let change = direct.min(flipped);
            v = w;
            if change < POWER_TOL {
                break;
            }
            prev_v.clone_from(&v);
        }

        // σ = sqrt(v^T B v) — same as `eigenvalue` for a converged power
        // iteration, but we recompute explicitly to be robust to early
        // termination.
        let bv = matvec(&b, cols, &v);
        let vtbv = dot_product(&v, &bv).max(0.0);
        let sigma = vtbv.sqrt();
        // Don't keep numerically-degenerate eigenpairs.
        if sigma < POWER_TOL {
            // Rank exhausted; remaining eigenvalues are below numerical
            // precision.
            break;
        }
        // Sign stabilization on v.
        let v_signed = sign_stabilize(v);
        vs.push(v_signed);
        sigmas.push(sigma);
        let _ = eigenvalue; // silence unused-on-some-paths
    }

    // If we found zero eigenpairs (input was all-zero), produce a degenerate
    // empty summary so the report stays well-formed.
    if sigmas.is_empty() {
        let payload = LowRankPayload {
            rows,
            cols,
            rank: 0,
            singular_values: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            input_hash,
            summary_hash: 0,
            frobenius_error: 0.0,
        };
        let mut p = payload;
        p.summary_hash = hash_bytes(&p.canonical_bytes());
        return Ok(p);
    }

    // Build U columns: u_k = M v_k / σ_k.
    let mut u_flat = vec![0.0f64; rows * sigmas.len()];
    for (k, (sigma, v_vec)) in sigmas.iter().zip(vs.iter()).enumerate() {
        let mv = matvec_m(matrix, rows, cols, v_vec);
        for (r, val) in mv.iter().enumerate() {
            u_flat[r * sigmas.len() + k] = val / sigma;
        }
    }
    // Reflow V into the same column-of-rank layout.
    let mut v_flat = vec![0.0f64; cols * sigmas.len()];
    for (k, v_vec) in vs.iter().enumerate() {
        for (r, val) in v_vec.iter().enumerate() {
            v_flat[r * sigmas.len() + k] = *val;
        }
    }
    // Compute Frobenius error.
    let m_norm_sq = frobenius_norm_sq(matrix);
    let m_norm = m_norm_sq.sqrt();
    // Reconstruct and measure error.
    let payload_initial = LowRankPayload {
        rows,
        cols,
        rank: sigmas.len(),
        singular_values: sigmas,
        u: u_flat,
        v: v_flat,
        input_hash,
        summary_hash: 0,
        frobenius_error: 0.0,
    };
    let m_hat = payload_initial.reconstruct();
    let err_sq = frobenius_diff_sq(matrix, &m_hat);
    let err_norm = err_sq.sqrt();
    let rel_err = if m_norm > 0.0 {
        (err_norm / m_norm).min(1.0)
    } else {
        0.0
    };

    let mut final_payload = LowRankPayload {
        frobenius_error: rel_err,
        ..payload_initial
    };
    final_payload.summary_hash = hash_bytes(&final_payload.canonical_bytes());
    Ok(final_payload)
}

// ---------------------------------------------------------------------------
// Internal linear-algebra helpers
// ---------------------------------------------------------------------------

fn hash_bytes_f64(matrix: &[f64]) -> u64 {
    let mut bytes = Vec::with_capacity(matrix.len() * 8);
    for x in matrix {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    hash_bytes(&bytes)
}

fn mt_m(matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut acc = KahanAccumulatorF64::new();
            for r in 0..rows {
                acc.add(matrix[r * cols + i] * matrix[r * cols + j]);
            }
            b[i * cols + j] = acc.finalize();
        }
    }
    b
}

fn matvec(b: &[f64], n: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = KahanAccumulatorF64::new();
        for j in 0..n {
            acc.add(b[i * n + j] * v[j]);
        }
        out[i] = acc.finalize();
    }
    out
}

fn matvec_m(matrix: &[f64], rows: usize, cols: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows];
    for r in 0..rows {
        let mut acc = KahanAccumulatorF64::new();
        for c in 0..cols {
            acc.add(matrix[r * cols + c] * v[c]);
        }
        out[r] = acc.finalize();
    }
    out
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..a.len() {
        acc.add(a[i] * b[i]);
    }
    acc.finalize()
}

fn vector_norm(v: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &x in v {
        acc.add(x * x);
    }
    acc.finalize().max(0.0).sqrt()
}

fn diff_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc.add(d * d);
    }
    acc.finalize().max(0.0).sqrt()
}

fn sum_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..a.len() {
        let s = a[i] + b[i];
        acc.add(s * s);
    }
    acc.finalize().max(0.0).sqrt()
}

fn gram_schmidt_against(v: &mut [f64], basis: &[Vec<f64>]) {
    for u in basis {
        let coeff = dot_product(v, u);
        for i in 0..v.len() {
            v[i] -= coeff * u[i];
        }
    }
}

fn initial_unit_vector(n: usize, k: usize) -> Vec<f64> {
    // Alternating ±1 with a phase offset per rank-index so different
    // ranks don't trivially collide before Gram-Schmidt. Normalized at
    // construction. Deterministic — depends only on (n, k).
    let mut v = vec![0.0f64; n];
    for i in 0..n {
        // Phase pattern: flip every (k+1) entries.
        let phase = (i / (k + 1)) % 2;
        v[i] = if phase == 0 { 1.0 } else { -1.0 };
    }
    let norm = (n as f64).sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

fn sign_stabilize(v: Vec<f64>) -> Vec<f64> {
    // Largest-magnitude entry forced positive. On ties, smallest index
    // wins (so the function is total even when all entries have equal
    // magnitude).
    let mut argmax = 0usize;
    let mut best = v[0].abs();
    for (i, &x) in v.iter().enumerate().skip(1) {
        let ax = x.abs();
        if ax > best {
            best = ax;
            argmax = i;
        }
    }
    if v[argmax] < 0.0 {
        v.into_iter().map(|x| -x).collect()
    } else {
        v
    }
}

fn frobenius_norm_sq(matrix: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &x in matrix {
        acc.add(x * x);
    }
    acc.finalize()
}

fn frobenius_diff_sq(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc.add(d * d);
    }
    acc.finalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_3x3() -> Vec<f64> {
        // A non-symmetric matrix; useful for sanity-checking SVD.
        vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 10.0, // row 2 (broken to ensure rank 3)
        ]
    }

    #[test]
    fn rejects_empty_matrix() {
        assert!(matches!(
            compress_low_rank(&[], 0, 0, 1),
            Err(CompressionError::UnsupportedShape { .. })
        ));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        // 3 elements but claims 3x3 = 9.
        assert!(matches!(
            compress_low_rank(&[1.0, 2.0, 3.0], 3, 3, 1),
            Err(CompressionError::UnsupportedShape { .. })
        ));
    }

    #[test]
    fn rejects_zero_max_rank() {
        assert!(matches!(
            compress_low_rank(&matrix_3x3(), 3, 3, 0),
            Err(CompressionError::UnsupportedShape { .. })
        ));
    }

    #[test]
    fn rejects_non_finite() {
        let m = vec![1.0, f64::NAN, 3.0, 4.0];
        assert!(matches!(
            compress_low_rank(&m, 2, 2, 1),
            Err(CompressionError::UnsupportedShape { .. })
        ));
    }

    #[test]
    fn full_rank_reconstruction_is_near_exact() {
        let m = matrix_3x3();
        let payload = compress_low_rank(&m, 3, 3, 3).unwrap();
        assert_eq!(payload.rows, 3);
        assert_eq!(payload.cols, 3);
        assert!(payload.rank <= 3);
        assert!(
            payload.frobenius_error < 1e-6,
            "full-rank SVD should reconstruct exactly, got error {}",
            payload.frobenius_error
        );
    }

    #[test]
    fn rank_one_has_finite_error_on_general_matrix() {
        let m = matrix_3x3();
        let payload = compress_low_rank(&m, 3, 3, 1).unwrap();
        assert_eq!(payload.rank, 1);
        assert!(payload.frobenius_error.is_finite());
        assert!(payload.frobenius_error > 0.0);
        assert!(payload.frobenius_error <= 1.0);
    }

    #[test]
    fn singular_values_are_non_increasing() {
        let m = matrix_3x3();
        let payload = compress_low_rank(&m, 3, 3, 3).unwrap();
        for w in payload.singular_values.windows(2) {
            assert!(
                w[0] + 1e-9 >= w[1],
                "singular values not non-increasing: {:?}",
                payload.singular_values
            );
        }
    }

    #[test]
    fn deterministic_across_repeated_calls() {
        let m = matrix_3x3();
        let p1 = compress_low_rank(&m, 3, 3, 2).unwrap();
        for _ in 0..20 {
            let p2 = compress_low_rank(&m, 3, 3, 2).unwrap();
            assert_eq!(p1.singular_values, p2.singular_values);
            assert_eq!(p1.u, p2.u);
            assert_eq!(p1.v, p2.v);
            assert_eq!(p1.summary_hash, p2.summary_hash);
            assert_eq!(p1.frobenius_error.to_bits(), p2.frobenius_error.to_bits());
        }
    }

    #[test]
    fn sign_stabilization_is_consistent() {
        // The sign-stabilized convention says max-magnitude entry of v
        // is positive. Verify the property holds on both u and v columns
        // (only v's argmax-positive is enforced; u inherits its sign from
        // u = M v / σ, but we still expect at least v to satisfy it).
        let m = matrix_3x3();
        let payload = compress_low_rank(&m, 3, 3, 2).unwrap();
        for k in 0..payload.rank {
            let v_col: Vec<f64> = (0..payload.cols)
                .map(|c| payload.v[c * payload.rank + k])
                .collect();
            let max_abs = v_col.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let argmax = v_col.iter().position(|x| x.abs() == max_abs).unwrap();
            assert!(
                v_col[argmax] >= 0.0,
                "rank {}: max-magnitude entry of v was negative",
                k
            );
        }
    }

    #[test]
    fn reconstruct_matches_internal_use() {
        let m = matrix_3x3();
        let payload = compress_low_rank(&m, 3, 3, 3).unwrap();
        let m_hat = payload.reconstruct();
        assert_eq!(m_hat.len(), 9);
        // Each entry close to original.
        for (orig, recon) in m.iter().zip(m_hat.iter()) {
            assert!((orig - recon).abs() < 1e-6, "{} vs {}", orig, recon);
        }
    }

    #[test]
    fn zero_input_produces_zero_summary() {
        let m = vec![0.0f64; 9];
        let payload = compress_low_rank(&m, 3, 3, 3).unwrap();
        // All-zero input: every power iteration finds a zero eigenvalue,
        // rank collapses to 0 → empty rank-zero summary.
        assert_eq!(payload.rank, 0);
        assert_eq!(payload.frobenius_error, 0.0);
    }

    #[test]
    fn canonical_bytes_deterministic_under_repeat_calls() {
        let m = matrix_3x3();
        let p = compress_low_rank(&m, 3, 3, 2).unwrap();
        let b1 = p.canonical_bytes();
        for _ in 0..20 {
            let b2 = p.canonical_bytes();
            assert_eq!(b1, b2);
        }
    }

    #[test]
    fn rank_one_rank_matrix_reconstructs_exactly() {
        // M = u * v^T where u = [1,2,3], v = [4,5,6]. M is exactly rank 1.
        let u = [1.0f64, 2.0, 3.0];
        let v = [4.0f64, 5.0, 6.0];
        let mut m = vec![0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                m[r * 3 + c] = u[r] * v[c];
            }
        }
        let payload = compress_low_rank(&m, 3, 3, 1).unwrap();
        assert!(
            payload.frobenius_error < 1e-9,
            "rank-1 matrix should reconstruct exactly, got error {}",
            payload.frobenius_error
        );
    }

    #[test]
    fn larger_matrix_summary_hash_distinguishes_input() {
        let mut m1 = vec![0.0; 16];
        let mut m2 = vec![0.0; 16];
        for i in 0..16 {
            m1[i] = i as f64;
            m2[i] = (i * 2) as f64;
        }
        let p1 = compress_low_rank(&m1, 4, 4, 2).unwrap();
        let p2 = compress_low_rank(&m2, 4, 4, 2).unwrap();
        assert_ne!(p1.summary_hash, p2.summary_hash);
        assert_ne!(p1.input_hash, p2.input_hash);
    }
}
