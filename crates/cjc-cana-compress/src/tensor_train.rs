//! Advisory tensor-train (MPS) compression for multi-axis feature tensors.
//!
//! This module reuses [`cjc_quantum::mps::svd_sign_stabilized`] + the
//! `DenseMatrix` substrate to implement the **TT-SVD** decomposition of
//! Oseledets, adapted as a CANA compression primitive:
//!
//! ```text
//!   T[i_0, i_1, ..., i_{n-1}]
//!     ≈ Σ_{α_0, ..., α_{n-2}}
//!         G_0[1, i_0, α_0] · G_1[α_0, i_1, α_1] · ... · G_{n-1}[α_{n-2}, i_{n-1}, 1]
//! ```
//!
//! Each core `G_k` has shape `(r_k × d_k × r_{k+1})`, with boundary bond
//! dimensions `r_0 = r_n = 1`. The decomposition is **lossy** unless every
//! intermediate SVD is kept at full rank — when the bond is truncated to
//! `max_bond < min(m, n)`, the truncated singular values' Frobenius
//! contribution becomes the reconstruction error.
//!
//! ## Why this reuses the quantum stack
//!
//! `cjc_quantum::mps` already ships:
//!
//! - **One-sided Jacobi SVD** with no random pivoting (deterministic
//!   convergence path).
//! - **Sign stabilization** so the largest-magnitude element of each
//!   column of U is positive (defeats SVD's ±1 sign ambiguity).
//! - **Kahan + fixed-sequence multiplication** in `ComplexF64::mul_fixed`
//!   (no FMA contraction).
//!
//! Replicating that discipline in `cjc-cana-compress` would duplicate
//! ~300 LOC of carefully-tested numerical code. We instead **lift** real
//! f64 inputs into [`ComplexF64`] with `im = 0.0`, run the existing SVD,
//! and drop the (numerically zero) imaginary parts on the way out.
//!
//! ## Determinism notes
//!
//! - The same `(tensor, shape, max_bond, tol)` always produces
//!   byte-identical [`TensorTrainPayload::canonical_bytes`].
//! - Bond truncation rule: keep singular values with
//!   `σ_k ≥ σ_0 · tol` AND `k < max_bond`. Tie-break is by index
//!   (smaller index wins) because `s` is already non-increasing.
//! - Sign stabilization is delegated to `svd_sign_stabilized`.
//! - All reductions in this module use [`KahanAccumulatorF64`].

use cjc_quantum::mps::{svd_sign_stabilized, DenseMatrix, SvdResult};
use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::complex::ComplexF64;

use cjc_cana::hash::hash_bytes;

use crate::candidate::CompressionError;

const MAGIC: &[u8; 4] = b"CTT0";

/// Output of [`compress_tensor_train`].
///
/// Each entry in `cores` is one TT core, flattened row-major as
/// `r_k × d_k × r_{k+1}`. The boundary bonds `r_0 = r_n = 1` are part of
/// the shape and **not** separately stored. Length of `bond_dims` is
/// always `shape.len() - 1`.
#[derive(Debug, Clone)]
pub struct TensorTrainPayload {
    /// Shape of the original tensor `(d_0, d_1, ..., d_{n-1})`.
    pub shape: Vec<usize>,
    /// Selected bond dimensions `(r_1, r_2, ..., r_{n-1})` — boundary
    /// `r_0 = r_n = 1` are implicit.
    pub bond_dims: Vec<usize>,
    /// One core per axis. Each core is row-major
    /// `r_k * d_k * r_{k+1}` length.
    pub cores: Vec<Vec<f64>>,
    /// FNV-1a hash of the original tensor's bit pattern.
    pub input_hash: u64,
    /// FNV-1a hash of [`Self::canonical_bytes`].
    pub summary_hash: u64,
    /// Relative Frobenius reconstruction error
    /// `||T - T̂||_F / ||T||_F`, clamped to `[0, 1]`.
    pub frobenius_error: f64,
}

impl TensorTrainPayload {
    /// Canonical bytes — used for content-addressed report hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MAGIC.len() + 64);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
        for &d in &self.shape {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        out.extend_from_slice(&(self.bond_dims.len() as u32).to_le_bytes());
        for &b in &self.bond_dims {
            out.extend_from_slice(&(b as u32).to_le_bytes());
        }
        // Cores in order; each core prefixed with its byte length so a
        // future schema change (e.g. boundary bonds becoming explicit)
        // can be detected.
        for core in &self.cores {
            out.extend_from_slice(&(core.len() as u32).to_le_bytes());
            for &x in core {
                out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
        out.extend_from_slice(&self.input_hash.to_le_bytes());
        out.extend_from_slice(&self.frobenius_error.to_bits().to_le_bytes());
        out
    }

    /// Reconstruct the rank-truncated tensor by contracting cores in
    /// sequence. Output is flat row-major `prod(shape)` length.
    ///
    /// Per-entry contraction: for each tensor index `(i_0, ..., i_{n-1})`,
    /// chain-multiply the per-axis slices
    /// `G_0[0, i_0, :] @ G_1[:, i_1, :] @ ... @ G_{n-1}[:, i_{n-1}, 0]`.
    /// O(N · Σ r_k · r_{k+1}) total; simple and obviously correct (the
    /// optimized "scan one axis at a time" variant tripped a subtle
    /// indexing bug on 3+-axis tensors).
    pub fn reconstruct(&self) -> Vec<f64> {
        let n = self.shape.len();
        if n == 0 {
            return Vec::new();
        }
        let total: usize = self.shape.iter().product();
        if self.cores.is_empty() {
            return vec![0.0; total];
        }
        let mut out = Vec::with_capacity(total);
        let mut coords = vec![0usize; n];
        for _ in 0..total {
            out.push(self.contract_at(&coords));
            // Increment coords (row-major: rightmost varies fastest).
            for axis in (0..n).rev() {
                coords[axis] += 1;
                if coords[axis] < self.shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        out
    }

    fn contract_at(&self, coords: &[usize]) -> f64 {
        let n = self.shape.len();
        // Running row vector of length r_k (bond after axis k-1).
        // Starts at (1, 1) = [1.0], representing the boundary bond.
        let mut vec_left = vec![1.0f64];
        for k in 0..n {
            let r_prev = if k == 0 { 1 } else { self.bond_dims[k - 1] };
            let r_next = if k == n - 1 { 1 } else { self.bond_dims[k] };
            let d_k = self.shape[k];
            let d_idx = coords[k];
            // Slice of core_k at axis-coordinate d_idx is
            // (r_prev × r_next); we contract vec_left (r_prev) against
            // it on r_prev → produces new vec_left of length r_next.
            let core_k = &self.cores[k];
            let mut next = vec![0.0f64; r_next];
            for jr in 0..r_next {
                let mut acc = KahanAccumulatorF64::new();
                for ip in 0..r_prev {
                    let core_val = core_k[ip * (d_k * r_next) + d_idx * r_next + jr];
                    acc.add(vec_left[ip] * core_val);
                }
                next[jr] = acc.finalize();
            }
            vec_left = next;
        }
        debug_assert_eq!(vec_left.len(), 1);
        vec_left[0]
    }
}

/// Compress a real f64 tensor by TT-SVD, truncating each bond to at most
/// `max_bond` and discarding singular values below `σ_0 · tol`.
///
/// Edge cases:
/// - **1-D tensor** (`shape.len() == 1`): returned as a single core
///   `(1, d_0, 1)` with `bond_dims = []`. Error is `0`.
/// - **All-zero tensor**: every SVD finds zero singular values; bonds
///   collapse to 0; reconstruction is the zero tensor with error `0`.
///
/// Returns [`CompressionError::UnsupportedShape`] if the tensor length
/// mismatches `prod(shape)`, if `shape` is empty, or if any axis has
/// dimension `0`.
pub fn compress_tensor_train(
    tensor: &[f64],
    shape: &[usize],
    max_bond: usize,
    tol: f64,
) -> Result<TensorTrainPayload, CompressionError> {
    if shape.is_empty() {
        return Err(CompressionError::UnsupportedShape {
            reason: "tensor shape is empty",
        });
    }
    let total: usize = shape.iter().product();
    if total == 0 {
        return Err(CompressionError::UnsupportedShape {
            reason: "tensor shape has a zero axis",
        });
    }
    if tensor.len() != total {
        return Err(CompressionError::UnsupportedShape {
            reason: "tensor length mismatches product(shape)",
        });
    }
    if max_bond == 0 {
        return Err(CompressionError::UnsupportedShape {
            reason: "max_bond must be >= 1",
        });
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(CompressionError::UnsupportedShape {
            reason: "tol must be finite and >= 0",
        });
    }
    if !tensor.iter().all(|x| x.is_finite()) {
        return Err(CompressionError::UnsupportedShape {
            reason: "tensor contains non-finite entries",
        });
    }

    let input_hash = hash_bytes_f64(tensor);

    // ----- 1D fast path -------------------------------------------------
    if shape.len() == 1 {
        let mut payload = TensorTrainPayload {
            shape: shape.to_vec(),
            bond_dims: Vec::new(),
            cores: vec![tensor.to_vec()],
            input_hash,
            summary_hash: 0,
            frobenius_error: 0.0,
        };
        payload.summary_hash = hash_bytes(&payload.canonical_bytes());
        return Ok(payload);
    }

    // ----- N-D TT-SVD ---------------------------------------------------
    let mut cores: Vec<Vec<f64>> = Vec::with_capacity(shape.len());
    let mut bond_dims: Vec<usize> = Vec::with_capacity(shape.len() - 1);

    // Running matrix M is reshaped between iterations.
    // Initial: shape M = (d_0, prod(d_1..d_{n-1})).
    let mut r_prev = 1usize;
    let mut m_rows = shape[0]; // = r_prev * d_0 = 1 * d_0
    let mut m_cols: usize = shape[1..].iter().product();
    let mut m_flat: Vec<f64> = tensor.to_vec();

    for k in 0..(shape.len() - 1) {
        let d_k = shape[k];
        // SVD of (m_rows × m_cols).
        let svd = real_svd(&m_flat, m_rows, m_cols);
        // Truncate bond.
        let r_next = truncation_size(&svd.s, max_bond, tol);
        let r_next = r_next.max(1).min(svd.s.len());
        bond_dims.push(r_next);
        // Core_k = U[:, :r_next], reshape to (r_prev, d_k, r_next).
        let mut core = vec![0.0f64; r_prev * d_k * r_next];
        for r in 0..m_rows {
            // r decomposes as (r_prev_idx, d_idx) row-major because the
            // matrix's rows were laid out (r_prev, d_k) flat.
            let r_prev_idx = r / d_k;
            let d_idx = r % d_k;
            for kk in 0..r_next {
                core[r_prev_idx * (d_k * r_next) + d_idx * r_next + kk] =
                    svd.u_real[r * svd.k + kk];
            }
        }
        cores.push(core);

        // New M = Σ[:r_next] @ V^T[:r_next, :], shape (r_next, m_cols).
        // Reshape conceptually as (r_next * d_{k+1}, prod(d_{k+2}..)).
        if k + 1 < shape.len() - 1 {
            let d_next = shape[k + 1];
            let rest: usize = shape[(k + 2)..].iter().product();
            let mut next_m = vec![0.0f64; r_next * m_cols];
            for kk in 0..r_next {
                let sigma = svd.s[kk];
                for c in 0..m_cols {
                    next_m[kk * m_cols + c] = sigma * svd.vt_real[kk * m_cols + c];
                }
            }
            // Reshape (r_next, d_next * rest) → (r_next * d_next, rest).
            // Layout is already correct because indexing was row-major.
            r_prev = r_next;
            m_rows = r_next * d_next;
            m_cols = rest;
            m_flat = next_m;
        } else {
            // We're about to enter the last-core branch.
            let d_last = shape[shape.len() - 1];
            let mut last_core = vec![0.0f64; r_next * d_last];
            for kk in 0..r_next {
                let sigma = svd.s[kk];
                for c in 0..d_last {
                    // V^T row kk, column c.
                    last_core[kk * d_last + c] = sigma * svd.vt_real[kk * m_cols + c];
                }
            }
            cores.push(last_core);
            r_prev = r_next;
            m_rows = 0;
            m_cols = 0;
        }
        let _ = m_rows;
        let _ = m_cols;
        let _ = r_prev;
    }

    // Compute Frobenius error by full reconstruction.
    let payload_initial = TensorTrainPayload {
        shape: shape.to_vec(),
        bond_dims,
        cores,
        input_hash,
        summary_hash: 0,
        frobenius_error: 0.0,
    };
    let t_hat = payload_initial.reconstruct();
    let m_norm = frobenius_norm_sq(tensor).sqrt();
    let err_norm = frobenius_diff_sq(tensor, &t_hat).sqrt();
    let rel_err = if m_norm > 0.0 {
        (err_norm / m_norm).min(1.0)
    } else {
        0.0
    };
    let mut final_payload = TensorTrainPayload {
        frobenius_error: rel_err,
        ..payload_initial
    };
    final_payload.summary_hash = hash_bytes(&final_payload.canonical_bytes());
    Ok(final_payload)
}

// ---------------------------------------------------------------------------
// Internal: real SVD via cjc-quantum complex SVD
// ---------------------------------------------------------------------------

/// Output shape: U is (m × k), Σ is k, V^T is (k × n). All values real.
struct RealSvd {
    u_real: Vec<f64>,
    s: Vec<f64>,
    vt_real: Vec<f64>,
    k: usize,
}

fn real_svd(matrix: &[f64], rows: usize, cols: usize) -> RealSvd {
    // `cjc_quantum::mps::svd_sign_stabilized` handles both tall and wide
    // matrices natively — the wide-matrix (m < n) routing was patched
    // upstream after this crate first shipped (see that function's
    // docstring for the conjugate-transpose identity used by the
    // routing). We can therefore just lift real f64 entries into
    // `ComplexF64` (im = 0) and call the function directly. The
    // imaginary parts of U and V^H come back as numerically zero (within
    // f64 epsilon) for real inputs, so we drop them on the way out.
    let mut a = DenseMatrix::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            a.set(r, c, ComplexF64::real(matrix[r * cols + c]));
        }
    }
    let SvdResult { u, s, vh } = svd_sign_stabilized(&a);
    let k = s.len();
    let mut u_real = vec![0.0f64; rows * k];
    for r in 0..rows {
        for c in 0..k {
            u_real[r * k + c] = u.get(r, c).re;
        }
    }
    let mut vt_real = vec![0.0f64; k * cols];
    for r in 0..k {
        for c in 0..cols {
            vt_real[r * cols + c] = vh.get(r, c).re;
        }
    }
    RealSvd {
        u_real,
        s,
        vt_real,
        k,
    }
}

/// Pick how many singular values to retain.
///
/// Rule: keep `σ_i` while `i < max_bond` AND
/// `σ_i ≥ σ_0 · tol` (relative threshold against largest singular value).
/// Always keep at least 1 singular value if `s` is non-empty and `σ_0 > 0`.
fn truncation_size(s: &[f64], max_bond: usize, tol: f64) -> usize {
    if s.is_empty() {
        return 0;
    }
    let s0 = s[0];
    if s0 <= 0.0 {
        return 0;
    }
    let threshold = s0 * tol;
    let mut keep = 0;
    for (i, &si) in s.iter().enumerate() {
        if i >= max_bond {
            break;
        }
        if si >= threshold {
            keep = i + 1;
        } else {
            break;
        }
    }
    keep.max(1)
}

fn hash_bytes_f64(matrix: &[f64]) -> u64 {
    let mut bytes = Vec::with_capacity(matrix.len() * 8);
    for x in matrix {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    hash_bytes(&bytes)
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

    #[test]
    fn rejects_empty_shape() {
        let r = compress_tensor_train(&[1.0, 2.0], &[], 4, 1e-9);
        assert!(matches!(
            r,
            Err(CompressionError::UnsupportedShape {
                reason: "tensor shape is empty",
            })
        ));
    }

    #[test]
    fn rejects_zero_axis() {
        let r = compress_tensor_train(&[], &[3, 0, 2], 4, 1e-9);
        assert!(matches!(
            r,
            Err(CompressionError::UnsupportedShape {
                reason: "tensor shape has a zero axis",
            })
        ));
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = compress_tensor_train(&[1.0, 2.0, 3.0], &[2, 2], 4, 1e-9);
        assert!(matches!(
            r,
            Err(CompressionError::UnsupportedShape {
                reason: "tensor length mismatches product(shape)",
            })
        ));
    }

    #[test]
    fn rejects_invalid_tol() {
        let r = compress_tensor_train(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 4, f64::NAN);
        assert!(matches!(
            r,
            Err(CompressionError::UnsupportedShape {
                reason: "tol must be finite and >= 0",
            })
        ));
        let r = compress_tensor_train(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 4, -0.1);
        assert!(matches!(
            r,
            Err(CompressionError::UnsupportedShape {
                reason: "tol must be finite and >= 0",
            })
        ));
    }

    #[test]
    fn rejects_zero_max_bond() {
        let r = compress_tensor_train(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 0, 1e-9);
        assert!(matches!(
            r,
            Err(CompressionError::UnsupportedShape {
                reason: "max_bond must be >= 1",
            })
        ));
    }

    #[test]
    fn one_d_tensor_passes_through() {
        let t = vec![1.0, 2.0, 3.0, 4.0];
        let payload = compress_tensor_train(&t, &[4], 8, 1e-9).unwrap();
        assert_eq!(payload.cores.len(), 1);
        assert_eq!(payload.cores[0], t);
        assert_eq!(payload.bond_dims, Vec::<usize>::new());
        assert_eq!(payload.frobenius_error, 0.0);
    }

    #[test]
    fn full_bond_decomposition_round_trips() {
        // 2×2 matrix viewed as a tensor of shape [2, 2].
        let t = vec![1.0, 2.0, 3.0, 4.0];
        let payload = compress_tensor_train(&t, &[2, 2], 8, 1e-12).unwrap();
        let back = payload.reconstruct();
        assert_eq!(back.len(), 4);
        for (i, (a, b)) in t.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "entry {} reconstruction error: {} vs {}",
                i,
                a,
                b
            );
        }
        assert!(payload.frobenius_error < 1e-9);
    }

    #[test]
    fn full_bond_3d_decomposition_round_trips() {
        // 2×2×2 tensor of distinct entries.
        let t: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let payload = compress_tensor_train(&t, &[2, 2, 2], 8, 1e-12).unwrap();
        let back = payload.reconstruct();
        assert_eq!(back.len(), 8);
        for (i, (a, b)) in t.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "entry {} reconstruction error: {} vs {}",
                i,
                a,
                b
            );
        }
        assert!(payload.frobenius_error < 1e-9);
        // Bond dims should be present for 3-axis tensor.
        assert_eq!(payload.bond_dims.len(), 2);
    }

    #[test]
    fn truncation_to_bond_one_produces_finite_error() {
        // A non-rank-1 matrix; bond=1 forces lossy compression.
        let t = vec![1.0, 2.0, 3.0, 4.0];
        let payload = compress_tensor_train(&t, &[2, 2], 1, 1e-12).unwrap();
        assert_eq!(payload.bond_dims, vec![1]);
        assert!(payload.frobenius_error.is_finite());
        assert!(payload.frobenius_error > 0.0);
        assert!(payload.frobenius_error <= 1.0);
    }

    #[test]
    fn zero_tensor_compresses_to_zero() {
        let t = vec![0.0f64; 8];
        let payload = compress_tensor_train(&t, &[2, 2, 2], 4, 1e-9).unwrap();
        assert_eq!(payload.frobenius_error, 0.0);
        let back = payload.reconstruct();
        for x in back {
            assert_eq!(x.abs(), 0.0);
        }
    }

    #[test]
    fn deterministic_across_repeated_calls() {
        let t: Vec<f64> = (1..=16).map(|x| x as f64 * 0.5).collect();
        let p1 = compress_tensor_train(&t, &[2, 2, 2, 2], 4, 1e-9).unwrap();
        for _ in 0..20 {
            let p2 = compress_tensor_train(&t, &[2, 2, 2, 2], 4, 1e-9).unwrap();
            assert_eq!(p1.cores, p2.cores);
            assert_eq!(p1.bond_dims, p2.bond_dims);
            assert_eq!(p1.summary_hash, p2.summary_hash);
            assert_eq!(p1.frobenius_error.to_bits(), p2.frobenius_error.to_bits());
        }
    }

    #[test]
    fn canonical_bytes_distinguishes_input() {
        let t1: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let t2: Vec<f64> = (1..=8).map(|x| x as f64 + 0.5).collect();
        let p1 = compress_tensor_train(&t1, &[2, 2, 2], 4, 1e-9).unwrap();
        let p2 = compress_tensor_train(&t2, &[2, 2, 2], 4, 1e-9).unwrap();
        assert_ne!(p1.canonical_bytes(), p2.canonical_bytes());
        assert_ne!(p1.summary_hash, p2.summary_hash);
        assert_ne!(p1.input_hash, p2.input_hash);
    }

    #[test]
    fn truncation_size_keeps_at_least_one_if_nonzero() {
        let s = vec![1.0, 0.5, 1e-20];
        assert_eq!(truncation_size(&s, 4, 1e-9), 2); // s[2] / s[0] = 1e-20 < 1e-9
        assert_eq!(truncation_size(&s, 1, 1e-9), 1); // capped by max_bond
        assert_eq!(truncation_size(&[1.0], 4, 1e-9), 1);
        assert_eq!(truncation_size(&[0.0], 4, 1e-9), 0); // σ_0 = 0
    }
}
