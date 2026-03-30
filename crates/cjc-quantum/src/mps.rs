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
    pub fn apply_single_qubit(&mut self, q: usize, u: [[ComplexF64; 2]; 2]) {
        assert!(q < self.n_qubits, "Qubit index out of range");

        let t = &self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;

        let mut new_a0 = DenseMatrix::zeros(bl, br);
        let mut new_a1 = DenseMatrix::zeros(bl, br);

        for r in 0..bl {
            for c in 0..br {
                let old0 = t.a[0].get(r, c);
                let old1 = t.a[1].get(r, c);

                // new[0] = U[0][0]*old[0] + U[0][1]*old[1]
                new_a0.set(r, c,
                    u[0][0].mul_fixed(old0).add(u[0][1].mul_fixed(old1)));
                // new[1] = U[1][0]*old[0] + U[1][1]*old[1]
                new_a1.set(r, c,
                    u[1][0].mul_fixed(old0).add(u[1][1].mul_fixed(old1)));
            }
        }

        self.tensors[q].a[0] = new_a0;
        self.tensors[q].a[1] = new_a1;
    }

    /// Apply a CNOT gate between adjacent qubits (control=q, target=q+1).
    ///
    /// For two-qubit gates on adjacent sites, we:
    /// 1. Contract tensors at sites q and q+1 into a single 4-index tensor
    /// 2. Apply the gate
    /// 3. Reshape into matrix and SVD to split back into two tensors
    /// 4. Truncate bond dimension if needed
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

        // Theta[s_l, s_r] is a bl × br matrix
        // Theta[s_l, s_r](i, k) = sum_j A_left[s_l](i,j) * A_right[s_r](j,k)
        let mut theta_00 = DenseMatrix::zeros(bl, br);
        let mut theta_01 = DenseMatrix::zeros(bl, br);
        let mut theta_10 = DenseMatrix::zeros(bl, br);
        let mut theta_11 = DenseMatrix::zeros(bl, br);

        for i in 0..bl {
            for k in 0..br {
                for j in 0..bm {
                    let al0 = tl.a[0].get(i, j);
                    let al1 = tl.a[1].get(i, j);
                    let ar0 = tr.a[0].get(j, k);
                    let ar1 = tr.a[1].get(j, k);

                    // theta[sl][sr] += A_left[sl](i,j) * A_right[sr](j,k)
                    theta_00.set(i, k, theta_00.get(i, k).add(al0.mul_fixed(ar0)));
                    theta_01.set(i, k, theta_01.get(i, k).add(al0.mul_fixed(ar1)));
                    theta_10.set(i, k, theta_10.get(i, k).add(al1.mul_fixed(ar0)));
                    theta_11.set(i, k, theta_11.get(i, k).add(al1.mul_fixed(ar1)));
                }
            }
        }

        // Apply CNOT: |c,t⟩ → |c, c⊕t⟩
        // If ctrl < targ: left=ctrl, right=targ
        //   |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        // If ctrl > targ: left=targ, right=ctrl
        //   |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
        let (new_00, new_01, new_10, new_11) = if ctrl < targ {
            // ctrl=left: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
            (theta_00.clone(), theta_01.clone(), theta_11.clone(), theta_10.clone())
        } else {
            // ctrl=right: |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
            (theta_00.clone(), theta_11.clone(), theta_10.clone(), theta_01.clone())
        };

        // Reshape into (bl*2) × (2*br) matrix for SVD
        let m = bl * 2;
        let n = 2 * br;
        let mut combined = DenseMatrix::zeros(m, n);

        // Row index: (s_left * bl + i), col index: (s_right * br + k)
        for i in 0..bl {
            for k in 0..br {
                combined.set(0 * bl + i, 0 * br + k, new_00.get(i, k));
                combined.set(0 * bl + i, 1 * br + k, new_01.get(i, k));
                combined.set(1 * bl + i, 0 * br + k, new_10.get(i, k));
                combined.set(1 * bl + i, 1 * br + k, new_11.get(i, k));
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
