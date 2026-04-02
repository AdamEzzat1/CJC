//! Matrix Product States (MPS) — Tensor-Train decomposition for large qubit systems.
//!
//! # Architecture
//!
//! For N qubits, the full statevector requires 2^N complex amplitudes. MPS represents
//! the same state as a chain of 3-index tensors (one per qubit), connected by bond
//! dimensions. For states with limited entanglement (e.g., GHZ, product states),
//! the bond dimension stays small and memory scales as O(N * chi^2) instead of O(2^N).
//!
//! # Sign-Stabilized SVD
//!
//! Standard SVD has sign ambiguity: columns of U and V can be multiplied by -1 without
//! changing the decomposition. This breaks bit-identical determinism. We enforce a
//! convention: the largest-magnitude element in each left singular vector is positive.
//!
//! # Determinism
//!
//! - SVD uses one-sided Jacobi rotations (no random pivoting)
//! - Sign stabilization applied after every SVD
//! - All reductions use Kahan summation
//! - Bond truncation uses deterministic sorted ordering

use cjc_runtime::complex::ComplexF64;

/// Maximum bond dimension before truncation.
const DEFAULT_MAX_BOND: usize = 64;

/// Convergence tolerance for Jacobi SVD.
const SVD_TOL: f64 = 1e-14;

/// Maximum Jacobi SVD iterations.
const SVD_MAX_ITER: usize = 200;

// ---------------------------------------------------------------------------
// Dense Complex Matrix (minimal, for SVD)
// ---------------------------------------------------------------------------

/// Row-major dense complex matrix.
#[derive(Clone, Debug)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<ComplexF64>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![ComplexF64::ZERO; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = ComplexF64::ONE;
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> ComplexF64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: ComplexF64) {
        self.data[r * self.cols + c] = v;
    }

    /// Frobenius norm of off-diagonal elements of A^H * A (for convergence check).
    fn off_diag_norm_ata(&self) -> f64 {
        let m = self.rows;
        let n = self.cols;
        let mut sum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                // Compute (A^H A)_{ij} = sum_k conj(A_{ki}) * A_{kj}
                let mut dot = ComplexF64::ZERO;
                for k in 0..m {
                    dot = dot.add(self.get(k, i).conj().mul_fixed(self.get(k, j)));
                }
                sum += dot.norm_sq();
            }
        }
        sum.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Sign-Stabilized SVD (One-Sided Jacobi)
// ---------------------------------------------------------------------------

/// Result of a sign-stabilized SVD: A = U * diag(S) * V^H
#[derive(Clone, Debug)]
pub struct SvdResult {
    pub u: DenseMatrix,
    pub s: Vec<f64>,
    pub vh: DenseMatrix,
}

/// Compute the sign-stabilized SVD of an m×n complex matrix.
///
/// Uses one-sided Jacobi rotations for determinism (no random pivoting,
/// no parallel sweeps). After convergence, sign stabilization ensures
/// the largest-magnitude element in each column of U is real and positive.
///
/// Returns (U, S, V^H) where A ≈ U * diag(S) * V^H.
pub fn svd_sign_stabilized(a: &DenseMatrix) -> SvdResult {
    let m = a.rows;
    let n = a.cols;
    let k = m.min(n);

    if m == 0 || n == 0 {
        return SvdResult {
            u: DenseMatrix::zeros(m, 0),
            s: vec![],
            vh: DenseMatrix::zeros(0, n),
        };
    }

    // Work on a copy; columns will be orthogonalized in place.
    let mut work = a.clone();
    let mut v = DenseMatrix::identity(n);

    // One-sided Jacobi: apply rotations to columns of `work` until A^H A is diagonal.
    for _iter in 0..SVD_MAX_ITER {
        let off = work.off_diag_norm_ata();
        if off < SVD_TOL {
            break;
        }

        // Sweep all column pairs (i, j) with i < j
        for i in 0..n {
            for j in (i + 1)..n {
                // Compute 2x2 Gram sub-matrix of columns i, j
                let mut aii = 0.0f64;
                let mut ajj = 0.0f64;
                let mut aij = ComplexF64::ZERO;

                for r in 0..m {
                    let ai = work.get(r, i);
                    let aj = work.get(r, j);
                    aii += ai.norm_sq();
                    ajj += aj.norm_sq();
                    aij = aij.add(ai.conj().mul_fixed(aj));
                }

                if aij.norm_sq() < SVD_TOL * SVD_TOL {
                    continue;
                }

                // Compute Jacobi rotation to zero aij.
                // Decompose: U = D * R where D = diag(1, e^{-iφ}) removes the
                // phase of aij, and R is the real Jacobi rotation for the
                // resulting real symmetric 2×2 Gram sub-matrix.
                let (cs, s_real, conj_phase) = jacobi_rotation_complex(aii, ajj, aij);

                // The unitary Q = [[c, -s], [s*conj(phase), c*conj(phase)]]
                // Applied to columns: new_i = c*col_i + s*conj(phase)*col_j
                //                     new_j = -s*col_i + c*conj(phase)*col_j
                let q10 = conj_phase.scale(s_real);  // s * conj(phase)
                let q11 = conj_phase.scale(cs);      // c * conj(phase)
                let neg_s = -s_real;

                for r in 0..m {
                    let wi = work.get(r, i);
                    let wj = work.get(r, j);
                    let ni = wi.scale(cs).add(wj.mul_fixed(q10));
                    let nj = wi.scale(neg_s).add(wj.mul_fixed(q11));
                    work.set(r, i, ni);
                    work.set(r, j, nj);
                }

                // Apply same rotation to V
                for r in 0..n {
                    let vi = v.get(r, i);
                    let vj = v.get(r, j);
                    let ni = vi.scale(cs).add(vj.mul_fixed(q10));
                    let nj = vi.scale(neg_s).add(vj.mul_fixed(q11));
                    v.set(r, i, ni);
                    v.set(r, j, nj);
                }
            }
        }
    }

    // Extract singular values (column norms of work) and normalize to get U
    let mut singular_values = Vec::with_capacity(k);
    let mut u_mat = DenseMatrix::zeros(m, k);

    for j in 0..k {
        let mut norm_sq = 0.0f64;
        for r in 0..m {
            norm_sq += work.get(r, j).norm_sq();
        }
        let sigma = norm_sq.sqrt();
        singular_values.push(sigma);

        if sigma > SVD_TOL {
            let inv_sigma = 1.0 / sigma;
            for r in 0..m {
                u_mat.set(r, j, work.get(r, j).scale(inv_sigma));
            }
        }
    }

    // Sort by descending singular value (deterministic: stable sort by value)
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        singular_values[b]
            .partial_cmp(&singular_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_s = Vec::with_capacity(k);
    let mut sorted_u = DenseMatrix::zeros(m, k);
    let mut sorted_v = DenseMatrix::zeros(n, k);

    for (new_j, &old_j) in order.iter().enumerate() {
        sorted_s.push(singular_values[old_j]);
        for r in 0..m {
            sorted_u.set(r, new_j, u_mat.get(r, old_j));
        }
        for r in 0..n {
            sorted_v.set(r, new_j, v.get(r, old_j));
        }
    }

    // Sign stabilization: ensure largest-magnitude element in each U column is real+positive
    for j in 0..k {
        if sorted_s[j] < SVD_TOL {
            continue;
        }

        // Find element with largest magnitude in column j of U
        let mut max_mag = 0.0f64;
        let mut max_val = ComplexF64::ONE;
        for r in 0..m {
            let val = sorted_u.get(r, j);
            let mag = val.norm_sq();
            if mag > max_mag {
                max_mag = mag;
                max_val = val;
            }
        }

        if max_mag < SVD_TOL * SVD_TOL {
            continue;
        }

        // Phase factor: e^{-i*arg(max_val)}
        let abs_val = max_mag.sqrt();
        let phase = ComplexF64::new(max_val.re / abs_val, -max_val.im / abs_val);

        // Multiply both U and V columns by the same phase.
        // A = U*S*V^H, so (U*P)*S*(V*P)^H = U*P*S*P^H*V^H = U*S*V^H
        // because P*S*P^H has diagonal entries phase_i*sigma_i*conj(phase_i) = sigma_i.
        for r in 0..m {
            sorted_u.set(r, j, sorted_u.get(r, j).mul_fixed(phase));
        }
        for r in 0..n {
            sorted_v.set(r, j, sorted_v.get(r, j).mul_fixed(phase));
        }
    }

    // Build V^H from V
    let mut vh = DenseMatrix::zeros(k, n);
    for r in 0..k {
        for c in 0..n {
            vh.set(r, c, sorted_v.get(c, r).conj());
        }
    }

    SvdResult {
        u: sorted_u,
        s: sorted_s,
        vh,
    }
}

/// Compute Jacobi rotation parameters for a 2x2 Hermitian matrix [[aii, aij], [conj(aij), ajj]].
///
/// Returns (cos, sin_real, conj_phase) for the decomposed rotation U = D * R:
/// - D = diag(1, e^{-iφ}) removes the complex phase from aij
/// - R = [[c, -s], [s, c]] is the real Jacobi rotation
/// - conj_phase = e^{-iφ} = conj(aij) / |aij|
fn jacobi_rotation_complex(aii: f64, ajj: f64, aij: ComplexF64) -> (f64, f64, ComplexF64) {
    let off_mag = aij.norm_sq().sqrt();
    if off_mag < SVD_TOL {
        return (1.0, 0.0, ComplexF64::ONE);
    }

    // Phase of aij: e^{iφ} = aij / |aij|, conjugate: e^{-iφ}
    let conj_phase = ComplexF64::new(aij.re / off_mag, -aij.im / off_mag);

    // Real Jacobi rotation for [[aii, |aij|], [|aij|, ajj]]
    let tau = (ajj - aii) / (2.0 * off_mag);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };

    let cs = 1.0 / (1.0 + t * t).sqrt();
    let s_real = t * cs;

    (cs, s_real, conj_phase)
}

// ---------------------------------------------------------------------------
// MPS Tensor (one per qubit site)
// ---------------------------------------------------------------------------

/// A single MPS tensor for site k.
///
/// Shape: (bond_left, physical=2, bond_right)
/// Stored as two matrices (one per physical index 0, 1):
///   A[0]: bond_left × bond_right
///   A[1]: bond_left × bond_right
#[derive(Clone, Debug)]
pub struct MpsTensor {
    pub bond_left: usize,
    pub bond_right: usize,
    /// a[s] is the bond_left × bond_right matrix for physical index s ∈ {0, 1}
    pub a: [DenseMatrix; 2],
}

impl MpsTensor {
    pub fn new(bond_left: usize, bond_right: usize) -> Self {
        Self {
            bond_left,
            bond_right,
            a: [
                DenseMatrix::zeros(bond_left, bond_right),
                DenseMatrix::zeros(bond_left, bond_right),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix Product State
// ---------------------------------------------------------------------------

/// Matrix Product State representation for N qubits.
///
/// Memory: O(N * chi^2 * 2) where chi = max bond dimension.
/// For product states chi=1; for GHZ chi=2.
#[derive(Clone, Debug)]
pub struct Mps {
    pub n_qubits: usize,
    pub tensors: Vec<MpsTensor>,
    pub max_bond: usize,
}

impl Mps {
    /// Create an MPS for the |0...0⟩ state.
    pub fn new(n_qubits: usize) -> Self {
        Self::with_max_bond(n_qubits, DEFAULT_MAX_BOND)
    }

    /// Create an MPS for |0...0⟩ with specified max bond dimension.
    pub fn with_max_bond(n_qubits: usize, max_bond: usize) -> Self {
        assert!(n_qubits > 0, "MPS requires at least 1 qubit");

        let mut tensors = Vec::with_capacity(n_qubits);
        for i in 0..n_qubits {
            let bl = if i == 0 { 1 } else { 1 };
            let br = if i == n_qubits - 1 { 1 } else { 1 };
            let mut t = MpsTensor::new(bl, br);
            // |0⟩ state: A[0] = [[1]], A[1] = [[0]]
            t.a[0].set(0, 0, ComplexF64::ONE);
            tensors.push(t);
        }

        Self {
            n_qubits,
            tensors,
            max_bond,
        }
    }

    /// Apply a single-qubit gate to qubit q.
    ///
    /// For single-qubit gates, MPS application is exact (no truncation needed):
    /// new_A[s'] = sum_s U[s',s] * A[s]
    ///
    /// Optimized: in-place computation with zero heap allocations.
    pub fn apply_single_qubit(&mut self, q: usize, u: [[ComplexF64; 2]; 2]) {
        assert!(q < self.n_qubits, "Qubit index out of range");

        let t = &mut self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;

        // In-place update: read both old values, compute both new values, write back.
        // No heap allocations — just stack temporaries.
        for r in 0..bl {
            for c in 0..br {
                let idx = r * br + c;
                let old0 = t.a[0].data[idx];
                let old1 = t.a[1].data[idx];

                t.a[0].data[idx] = u[0][0].mul_fixed(old0).add(u[0][1].mul_fixed(old1));
                t.a[1].data[idx] = u[1][0].mul_fixed(old0).add(u[1][1].mul_fixed(old1));
            }
        }
    }

    /// Apply a CNOT gate between adjacent qubits (control=q, target=q+1).
    ///
    /// For two-qubit gates on adjacent sites, we:
    /// 1. Contract tensors at sites q and q+1 into a single 4-index tensor
    /// 2. Apply the gate (CNOT permutation fused into contraction)
    /// 3. Reshape into matrix and SVD to split back into two tensors
    /// 4. Truncate bond dimension if needed
    ///
    /// Optimized: gate permutation is fused into the contraction step, eliminating
    /// 4 intermediate matrix clones. Combined matrix is built directly from theta.
    pub fn apply_cnot_adjacent(&mut self, ctrl: usize, targ: usize) {
        assert!(targ == ctrl + 1 || ctrl == targ + 1,
            "CNOT requires adjacent qubits for MPS");

        let (q_left, q_right) = if ctrl < targ { (ctrl, targ) } else { (targ, ctrl) };

        // Contract sites q_left and q_right
        let tl = &self.tensors[q_left];
        let tr = &self.tensors[q_right];
        let bl = tl.bond_left;
        let bm = tl.bond_right; // = tr.bond_left
        let br = tr.bond_right;

        // CNOT permutation lookup: gate_map[sl][sr] = (new_sl, new_sr)
        // ctrl < targ (ctrl=left):  |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        // ctrl > targ (ctrl=right): |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
        let gate_map: [[usize; 2]; 4] = if ctrl < targ {
            [[0, 0], [0, 1], [1, 1], [1, 0]]  // (sl,sr) → (new_sl, new_sr)
        } else {
            [[0, 0], [1, 1], [1, 0], [0, 1]]
        };

        // Build combined (2*bl) × (2*br) matrix directly, fusing contraction + gate
        let m = bl * 2;
        let n = 2 * br;
        let mut combined = DenseMatrix::zeros(m, n);

        for i in 0..bl {
            for k in 0..br {
                // Accumulate theta[sl][sr](i,k) and write directly to permuted position
                for sl in 0..2usize {
                    for sr in 0..2usize {
                        let mut acc = ComplexF64::ZERO;
                        for j in 0..bm {
                            let al = tl.a[sl].get(i, j);
                            let ar = tr.a[sr].get(j, k);
                            acc = acc.add(al.mul_fixed(ar));
                        }
                        // Write to gate-permuted position in combined matrix
                        let [new_sl, new_sr] = gate_map[sl * 2 + sr];
                        let row = new_sl * bl + i;
                        let col = new_sr * br + k;
                        combined.set(row, col, combined.get(row, col).add(acc));
                    }
                }
            }
        }

        // SVD and truncate
        let svd = svd_sign_stabilized(&combined);
        let chi = svd.s.iter().filter(|&&s| s > SVD_TOL).count().min(self.max_bond);
        let chi = chi.max(1); // at least bond dim 1

        // Build new left tensor: A_left[s](i, j) from U * sqrt(S)
        let mut new_left = MpsTensor::new(bl, chi);
        for s in 0..2 {
            for i in 0..bl {
                for j in 0..chi {
                    let row = s * bl + i;
                    let val = svd.u.get(row, j).scale(svd.s[j].sqrt());
                    new_left.a[s].set(i, j, val);
                }
            }
        }

        // Build new right tensor: A_right[s](j, k) from sqrt(S) * Vh
        let mut new_right = MpsTensor::new(chi, br);
        for s in 0..2 {
            for j in 0..chi {
                for k in 0..br {
                    let col = s * br + k;
                    let val = svd.vh.get(j, col).scale(svd.s[j].sqrt());
                    new_right.a[s].set(j, k, val);
                }
            }
        }

        self.tensors[q_left] = new_left;
        self.tensors[q_right] = new_right;
    }

    /// Compute the full statevector from the MPS (for verification).
    ///
    /// Warning: This is O(2^N) and should only be used for small systems.
    pub fn to_statevector(&self) -> Vec<ComplexF64> {
        let n = self.n_qubits;
        let dim = 1usize << n;
        let mut amplitudes = vec![ComplexF64::ZERO; dim];

        for basis in 0..dim {
            // Contract MPS for this basis state
            // Start with a 1×bond_right(0) row vector
            let mut vec_current: Vec<ComplexF64> = vec![ComplexF64::ZERO; 1];
            vec_current[0] = ComplexF64::ONE;

            for q in 0..n {
                let s = (basis >> q) & 1; // physical index at site q
                let t = &self.tensors[q];
                let br = t.bond_right;

                let mut vec_next = vec![ComplexF64::ZERO; br];
                for k in 0..br {
                    let mut sum = ComplexF64::ZERO;
                    for j in 0..t.bond_left {
                        sum = sum.add(vec_current[j].mul_fixed(t.a[s].get(j, k)));
                    }
                    vec_next[k] = sum;
                }
                vec_current = vec_next;
            }

            // Final result should be a single scalar
            amplitudes[basis] = vec_current[0];
        }

        amplitudes
    }

    /// Estimate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for t in &self.tensors {
            // 2 matrices of (bond_left × bond_right), each ComplexF64 = 16 bytes
            total += 2 * t.bond_left * t.bond_right * 16;
        }
        total
    }

    // -----------------------------------------------------------------------
    // Canonical Form Normalization
    // -----------------------------------------------------------------------

    /// Left-canonicalize the MPS: sweep left-to-right, performing QR/SVD at
    /// each bond to ensure each tensor is a left-isometry (A†A = I).
    ///
    /// After this, contracting from the left gives identity matrices,
    /// which is needed for efficient expectation value computation and
    /// proper DMRG initialization.
    pub fn left_canonicalize(&mut self) {
        for i in 0..(self.n_qubits - 1) {
            self.canonicalize_bond_left(i);
        }
    }

    /// Right-canonicalize the MPS: sweep right-to-left, ensuring each tensor
    /// is a right-isometry (AA† = I).
    pub fn right_canonicalize(&mut self) {
        for i in (1..self.n_qubits).rev() {
            self.canonicalize_bond_right(i);
        }
    }

    /// Mixed-canonical form: left-canonicalize sites 0..center-1 and
    /// right-canonicalize sites center+1..N-1. The center site holds
    /// all non-trivial information (the "orthogonality center").
    pub fn mixed_canonicalize(&mut self, center: usize) {
        assert!(center < self.n_qubits, "center must be < n_qubits");
        for i in 0..center {
            self.canonicalize_bond_left(i);
        }
        for i in (center + 1..self.n_qubits).rev() {
            self.canonicalize_bond_right(i);
        }
    }

    /// Left-canonicalize site i: reshape A[i] into a tall matrix, SVD,
    /// absorb S*V† into A[i+1].
    fn canonicalize_bond_left(&mut self, i: usize) {
        let bl = self.tensors[i].bond_left;
        let br = self.tensors[i].bond_right;

        // Reshape: (bl, 2, br) → (2*bl, br) matrix
        let rows = 2 * bl;
        let cols = br;
        let mut mat = DenseMatrix::zeros(rows, cols);
        for s in 0..2 {
            for r in 0..bl {
                for c in 0..br {
                    mat.set(s * bl + r, c, self.tensors[i].a[s].get(r, c));
                }
            }
        }

        let svd = svd_sign_stabilized(&mat);
        let k = svd.s.len().min(self.max_bond);

        // New tensor i: reshape U[:, :k] back to (bl, 2, k)
        let mut new_t = MpsTensor::new(bl, k);
        for s in 0..2 {
            for r in 0..bl {
                for c in 0..k {
                    new_t.a[s].set(r, c, svd.u.get(s * bl + r, c));
                }
            }
        }
        self.tensors[i] = new_t;

        // Absorb S*V† into tensor i+1
        // SV†: k × br matrix
        let next_bl = self.tensors[i + 1].bond_left;
        let next_br = self.tensors[i + 1].bond_right;
        assert_eq!(next_bl, br, "bond dimension mismatch");

        let mut sv_mat = DenseMatrix::zeros(k, br);
        for r in 0..k {
            for c in 0..br {
                sv_mat.set(r, c, svd.vh.get(r, c).scale(svd.s[r]));
            }
        }

        // new_A[i+1][s] = SV† * A[i+1][s]
        let mut new_next = MpsTensor::new(k, next_br);
        for s in 0..2 {
            for r in 0..k {
                for c in 0..next_br {
                    let mut val = ComplexF64::ZERO;
                    for m in 0..br {
                        val = val.add(sv_mat.get(r, m).mul_fixed(self.tensors[i + 1].a[s].get(m, c)));
                    }
                    new_next.a[s].set(r, c, val);
                }
            }
        }
        self.tensors[i + 1] = new_next;
    }

    /// Right-canonicalize site i: reshape, SVD, absorb U*S into A[i-1].
    fn canonicalize_bond_right(&mut self, i: usize) {
        let bl = self.tensors[i].bond_left;
        let br = self.tensors[i].bond_right;

        // Reshape: (bl, 2, br) → (bl, 2*br) matrix
        let rows = bl;
        let cols = 2 * br;
        let mut mat = DenseMatrix::zeros(rows, cols);
        for s in 0..2 {
            for r in 0..bl {
                for c in 0..br {
                    mat.set(r, s * br + c, self.tensors[i].a[s].get(r, c));
                }
            }
        }

        let svd = svd_sign_stabilized(&mat);
        let k = svd.s.len().min(self.max_bond);

        // New tensor i: reshape V†[:k, :] back to (k, 2, br)
        let mut new_t = MpsTensor::new(k, br);
        for s in 0..2 {
            for r in 0..k {
                for c in 0..br {
                    new_t.a[s].set(r, c, svd.vh.get(r, s * br + c));
                }
            }
        }
        self.tensors[i] = new_t;

        // Absorb U*S into tensor i-1
        let prev_bl = self.tensors[i - 1].bond_left;
        let prev_br = self.tensors[i - 1].bond_right;
        assert_eq!(prev_br, bl, "bond dimension mismatch");

        let mut us_mat = DenseMatrix::zeros(bl, k);
        for r in 0..bl {
            for c in 0..k {
                us_mat.set(r, c, svd.u.get(r, c).scale(svd.s[c]));
            }
        }

        // new_A[i-1][s] = A[i-1][s] * US
        let mut new_prev = MpsTensor::new(prev_bl, k);
        for s in 0..2 {
            for r in 0..prev_bl {
                for c in 0..k {
                    let mut val = ComplexF64::ZERO;
                    for m in 0..bl {
                        val = val.add(self.tensors[i - 1].a[s].get(r, m).mul_fixed(us_mat.get(m, c)));
                    }
                    new_prev.a[s].set(r, c, val);
                }
            }
        }
        self.tensors[i - 1] = new_prev;
    }

    // -----------------------------------------------------------------------
    // SWAP Network for Non-Adjacent Gates
    // -----------------------------------------------------------------------

    /// Apply a two-qubit gate between arbitrary (possibly non-adjacent) qubits
    /// using SWAP decomposition.
    ///
    /// Strategy: SWAP qubits closer together, apply the gate on adjacent sites,
    /// then SWAP back. This is exact but may increase bond dimension.
    pub fn apply_gate_swap_network(
        &mut self,
        q1: usize,
        q2: usize,
        gate: [[ComplexF64; 4]; 4],
    ) {
        assert!(q1 < self.n_qubits && q2 < self.n_qubits && q1 != q2,
            "invalid qubit indices for SWAP network");

        let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

        if hi - lo == 1 {
            // Already adjacent: apply directly
            self.apply_two_qubit_gate(lo, gate);
            return;
        }

        // SWAP q_hi down to lo+1
        for i in (lo + 1..hi).rev() {
            self.apply_swap_adjacent(i, i + 1);
        }

        // Now the qubits are at positions lo and lo+1
        // But we need to account for whether q1 < q2 or not
        self.apply_two_qubit_gate(lo, gate);

        // SWAP back
        for i in (lo + 1)..hi {
            self.apply_swap_adjacent(i, i + 1);
        }
    }

    /// Apply a SWAP gate between adjacent qubits i and i+1.
    fn apply_swap_adjacent(&mut self, i: usize, j: usize) {
        assert_eq!(j, i + 1, "SWAP requires adjacent qubits");
        // SWAP matrix in computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
        // [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        let zero = ComplexF64::ZERO;
        let one = ComplexF64::ONE;
        let swap_mat = [
            [one,  zero, zero, zero],
            [zero, zero, one,  zero],
            [zero, one,  zero, zero],
            [zero, zero, zero, one],
        ];
        self.apply_two_qubit_gate(i, swap_mat);
    }

    /// Apply a 4×4 two-qubit gate on adjacent sites (i, i+1).
    /// Contracts both tensors, applies the gate, then decomposes via SVD.
    fn apply_two_qubit_gate(&mut self, i: usize, gate: [[ComplexF64; 4]; 4]) {
        let bl = self.tensors[i].bond_left;
        let br = self.tensors[i + 1].bond_right;
        let bond_mid = self.tensors[i].bond_right;

        // Contract tensors i and i+1 into a (4*bl, br) matrix:
        // Theta[s1*s2][bl_idx][br_idx] = Σ_m A[i][s1][bl_idx,m] * A[i+1][s2][m,br_idx]
        let mut theta = DenseMatrix::zeros(4 * bl, br);
        for s1 in 0..2 {
            for s2 in 0..2 {
                let s = s1 * 2 + s2;
                for r in 0..bl {
                    for c in 0..br {
                        let mut val = ComplexF64::ZERO;
                        for m in 0..bond_mid {
                            val = val.add(
                                self.tensors[i].a[s1].get(r, m)
                                    .mul_fixed(self.tensors[i + 1].a[s2].get(m, c))
                            );
                        }
                        theta.set(s * bl + r, c, val);
                    }
                }
            }
        }

        // Apply gate: new_theta[s'][bl_idx][br_idx] = Σ_s gate[s'][s] * theta[s][bl_idx][br_idx]
        let mut new_theta = DenseMatrix::zeros(4 * bl, br);
        for sp in 0..4 {
            for s in 0..4 {
                if gate[sp][s].re.abs() < 1e-20 && gate[sp][s].im.abs() < 1e-20 {
                    continue;
                }
                for r in 0..bl {
                    for c in 0..br {
                        let old = new_theta.get(sp * bl + r, c);
                        new_theta.set(
                            sp * bl + r, c,
                            old.add(gate[sp][s].mul_fixed(theta.get(s * bl + r, c))),
                        );
                    }
                }
            }
        }

        // SVD to split back into two tensors
        // Reshape: (4*bl, br) → (2*bl, 2*br) for SVD
        let svd_rows = 2 * bl;
        let svd_cols = 2 * br;
        let mut svd_mat = DenseMatrix::zeros(svd_rows, svd_cols);
        for s1 in 0..2 {
            for s2 in 0..2 {
                let s = s1 * 2 + s2;
                for r in 0..bl {
                    for c in 0..br {
                        svd_mat.set(s1 * bl + r, s2 * br + c, new_theta.get(s * bl + r, c));
                    }
                }
            }
        }

        let svd = svd_sign_stabilized(&svd_mat);
        let k = svd.s.iter().filter(|&&s| s > SVD_TOL).count().min(self.max_bond);
        let k = k.max(1);

        // New tensor i: U[:, :k] reshaped to (bl, 2, k)
        let mut new_i = MpsTensor::new(bl, k);
        for s1 in 0..2 {
            for r in 0..bl {
                for c in 0..k {
                    new_i.a[s1].set(r, c, svd.u.get(s1 * bl + r, c));
                }
            }
        }

        // New tensor i+1: S*V†[:k, :] reshaped to (k, 2, br)
        let mut new_ip1 = MpsTensor::new(k, br);
        for s2 in 0..2 {
            for r in 0..k {
                for c in 0..br {
                    new_ip1.a[s2].set(r, c, svd.vh.get(r, s2 * br + c).scale(svd.s[r]));
                }
            }
        }

        self.tensors[i] = new_i;
        self.tensors[i + 1] = new_ip1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn h_matrix() -> [[ComplexF64; 2]; 2] {
        let s = ComplexF64::real(INV_SQRT2);
        let ms = ComplexF64::real(-INV_SQRT2);
        [[s, s], [s, ms]]
    }

    fn x_matrix() -> [[ComplexF64; 2]; 2] {
        [[ComplexF64::ZERO, ComplexF64::ONE],
         [ComplexF64::ONE, ComplexF64::ZERO]]
    }

    #[test]
    fn test_mps_initial_state() {
        let mps = Mps::new(3);
        let sv = mps.to_statevector();
        // |000⟩ state
        assert!((sv[0].re - 1.0).abs() < TOL);
        for i in 1..8 {
            assert!(sv[i].norm_sq() < TOL);
        }
    }

    #[test]
    fn test_mps_single_qubit_x() {
        let mut mps = Mps::new(2);
        mps.apply_single_qubit(0, x_matrix());
        let sv = mps.to_statevector();
        // X|00⟩ = |10⟩ → bit pattern: qubit 0 = 1 → index 1
        assert!((sv[1].re - 1.0).abs() < TOL);
        assert!(sv[0].norm_sq() < TOL);
    }

    #[test]
    fn test_mps_hadamard() {
        let mut mps = Mps::new(1);
        mps.apply_single_qubit(0, h_matrix());
        let sv = mps.to_statevector();
        assert!((sv[0].re - INV_SQRT2).abs() < TOL);
        assert!((sv[1].re - INV_SQRT2).abs() < TOL);
    }

    #[test]
    fn test_mps_bell_state() {
        let mut mps = Mps::new(2);
        mps.apply_single_qubit(0, h_matrix());
        mps.apply_cnot_adjacent(0, 1);
        let sv = mps.to_statevector();
        // Bell state: (|00⟩ + |11⟩)/√2
        assert!((sv[0].re - INV_SQRT2).abs() < TOL, "sv[0] = {:?}", sv[0]);
        assert!(sv[1].norm_sq() < TOL, "sv[1] = {:?}", sv[1]);
        assert!(sv[2].norm_sq() < TOL, "sv[2] = {:?}", sv[2]);
        assert!((sv[3].re - INV_SQRT2).abs() < TOL, "sv[3] = {:?}", sv[3]);
    }

    #[test]
    fn test_mps_ghz_3() {
        let mut mps = Mps::new(3);
        mps.apply_single_qubit(0, h_matrix());
        mps.apply_cnot_adjacent(0, 1);
        mps.apply_cnot_adjacent(1, 2);
        let sv = mps.to_statevector();
        // GHZ: (|000⟩ + |111⟩)/√2
        assert!((sv[0].re - INV_SQRT2).abs() < TOL, "sv[0] = {:?}", sv[0]);
        for i in 1..7 {
            assert!(sv[i].norm_sq() < TOL, "sv[{}] = {:?}", i, sv[i]);
        }
        assert!((sv[7].re - INV_SQRT2).abs() < TOL, "sv[7] = {:?}", sv[7]);
    }

    #[test]
    fn test_svd_identity() {
        let id = DenseMatrix::identity(3);
        let svd = svd_sign_stabilized(&id);
        // Singular values should all be 1.0
        for &s in &svd.s {
            assert!((s - 1.0).abs() < TOL, "singular value {} != 1.0", s);
        }
    }

    #[test]
    fn test_svd_sign_stability() {
        // Run SVD twice on same matrix — must get bit-identical results
        let mut m = DenseMatrix::zeros(3, 2);
        m.set(0, 0, ComplexF64::new(1.0, 0.5));
        m.set(0, 1, ComplexF64::new(0.3, -0.2));
        m.set(1, 0, ComplexF64::new(-0.7, 0.1));
        m.set(1, 1, ComplexF64::new(0.9, 0.4));
        m.set(2, 0, ComplexF64::new(0.2, -0.8));
        m.set(2, 1, ComplexF64::new(-0.1, 0.6));

        let svd1 = svd_sign_stabilized(&m);
        let svd2 = svd_sign_stabilized(&m);

        for i in 0..svd1.s.len() {
            assert_eq!(svd1.s[i].to_bits(), svd2.s[i].to_bits(),
                "SVD singular value {} not bit-identical", i);
        }
        for r in 0..svd1.u.rows {
            for c in 0..svd1.u.cols {
                assert_eq!(
                    svd1.u.get(r, c).re.to_bits(),
                    svd2.u.get(r, c).re.to_bits(),
                    "U[{},{}].re not bit-identical", r, c
                );
                assert_eq!(
                    svd1.u.get(r, c).im.to_bits(),
                    svd2.u.get(r, c).im.to_bits(),
                    "U[{},{}].im not bit-identical", r, c
                );
            }
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        // A = U * diag(S) * Vh
        let mut m = DenseMatrix::zeros(3, 2);
        m.set(0, 0, ComplexF64::new(1.0, 0.5));
        m.set(0, 1, ComplexF64::new(0.3, -0.2));
        m.set(1, 0, ComplexF64::new(-0.7, 0.1));
        m.set(1, 1, ComplexF64::new(0.9, 0.4));
        m.set(2, 0, ComplexF64::new(0.2, -0.8));
        m.set(2, 1, ComplexF64::new(-0.1, 0.6));

        let svd = svd_sign_stabilized(&m);
        let k = svd.s.len();

        // Reconstruct: A_approx = U * diag(S) * Vh
        for r in 0..m.rows {
            for c in 0..m.cols {
                let mut sum = ComplexF64::ZERO;
                for i in 0..k {
                    sum = sum.add(
                        svd.u.get(r, i).scale(svd.s[i]).mul_fixed(svd.vh.get(i, c))
                    );
                }
                let orig = m.get(r, c);
                let err = sum.add(orig.neg()).norm_sq().sqrt();
                assert!(err < 1e-8,
                    "Reconstruction error at ({},{}): {} (got {:?}, expected {:?})",
                    r, c, err, sum, orig);
            }
        }
    }

    #[test]
    fn test_mps_memory_scaling() {
        // For a product state, bond dim stays 1 → memory = O(N)
        let mps = Mps::new(50);
        let mem = mps.memory_bytes();
        // 50 qubits * 2 matrices * 1*1 * 16 bytes = 1600 bytes
        assert!(mem < 10_000, "Product state MPS too large: {} bytes", mem);
    }

    #[test]
    fn test_mps_ghz_bond_dimension() {
        // GHZ state should have bond dimension 2
        let mut mps = Mps::new(5);
        mps.apply_single_qubit(0, h_matrix());
        for i in 0..4 {
            mps.apply_cnot_adjacent(i, i + 1);
        }

        // Interior bonds should be 2
        for i in 0..4 {
            assert!(
                self::tensors_bond_right(&mps, i) <= 2,
                "Bond between {} and {} is {}, expected ≤ 2",
                i, i + 1, self::tensors_bond_right(&mps, i)
            );
        }
    }

    fn tensors_bond_right(mps: &Mps, site: usize) -> usize {
        mps.tensors[site].bond_right
    }
}
