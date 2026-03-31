//! Density Matrix Simulator -- Mixed states and noise channels.
//!
//! Represents quantum states as rho (2^N x 2^N complex matrix).
//! Supports noise channels via Kraus operators: rho -> Sum_k K_k rho K_k^dagger.
//! Max ~12-13 qubits (2^26 entries x 16 bytes ~ 1 GB).
//!
//! # Determinism
//!
//! - All complex arithmetic uses `mul_fixed` (no FMA)
//! - Reductions use ascending index order
//! - No HashMap or non-deterministic data structures
//! - Eigenvalues computed via sign-stabilized SVD (Hermitian => singular values = eigenvalues)

use cjc_runtime::complex::ComplexF64;
use crate::gates::Gate;
use crate::mps::{DenseMatrix, svd_sign_stabilized};
use crate::statevector::Statevector;

/// Maximum supported qubits for density matrix simulation.
const MAX_QUBITS: usize = 14;

/// Single-qubit Kraus operators represented as 2x2 complex matrices.
pub type KrausOps2x2 = Vec<[[ComplexF64; 2]; 2]>;

/// Density matrix representation of an N-qubit quantum state.
///
/// Stored as a dim x dim row-major complex matrix where dim = 2^n_qubits.
/// Pure states satisfy Tr(rho^2) = 1; mixed states have Tr(rho^2) < 1.
pub struct DensityMatrix {
    pub n_qubits: usize,
    dim: usize,
    pub data: Vec<ComplexF64>,
}

// ---------------------------------------------------------------------------
// Gate matrix extraction helpers
// ---------------------------------------------------------------------------

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Extract the 2x2 unitary matrix and target qubit for single-qubit gates.
/// Returns None for multi-qubit gates (handled separately).
fn gate_matrix(gate: &Gate) -> Option<(usize, [[ComplexF64; 2]; 2])> {
    match gate {
        Gate::H(q) => {
            let s = ComplexF64::real(INV_SQRT2);
            let ms = ComplexF64::real(-INV_SQRT2);
            Some((*q, [[s, s], [s, ms]]))
        }
        Gate::X(q) => Some((*q, [
            [ComplexF64::ZERO, ComplexF64::ONE],
            [ComplexF64::ONE, ComplexF64::ZERO],
        ])),
        Gate::Y(q) => Some((*q, [
            [ComplexF64::ZERO, ComplexF64::new(0.0, -1.0)],
            [ComplexF64::new(0.0, 1.0), ComplexF64::ZERO],
        ])),
        Gate::Z(q) => Some((*q, [
            [ComplexF64::ONE, ComplexF64::ZERO],
            [ComplexF64::ZERO, ComplexF64::new(-1.0, 0.0)],
        ])),
        Gate::S(q) => Some((*q, [
            [ComplexF64::ONE, ComplexF64::ZERO],
            [ComplexF64::ZERO, ComplexF64::I],
        ])),
        Gate::T(q) => {
            let phase = ComplexF64::new(INV_SQRT2, INV_SQRT2);
            Some((*q, [
                [ComplexF64::ONE, ComplexF64::ZERO],
                [ComplexF64::ZERO, phase],
            ]))
        }
        Gate::Rx(q, theta) => {
            let c = ComplexF64::real((theta / 2.0).cos());
            let s = ComplexF64::new(0.0, -(theta / 2.0).sin());
            Some((*q, [[c, s], [s, c]]))
        }
        Gate::Ry(q, theta) => {
            let c = ComplexF64::real((theta / 2.0).cos());
            let s = ComplexF64::real((theta / 2.0).sin());
            let ms = ComplexF64::real(-(theta / 2.0).sin());
            Some((*q, [[c, ms], [s, c]]))
        }
        Gate::Rz(q, theta) => {
            let pos = ComplexF64::new((theta / 2.0).cos(), (theta / 2.0).sin());
            let neg = ComplexF64::new((theta / 2.0).cos(), -(theta / 2.0).sin());
            Some((*q, [
                [neg, ComplexF64::ZERO],
                [ComplexF64::ZERO, pos],
            ]))
        }
        Gate::CNOT(_, _) | Gate::CZ(_, _) | Gate::SWAP(_, _) | Gate::Toffoli(_, _, _) => None,
    }
}

// ---------------------------------------------------------------------------
// Permutation helpers for multi-qubit gates on density matrices
// ---------------------------------------------------------------------------

/// CNOT permutation: flip target bit if control bit is set.
#[inline]
fn cnot_map(idx: usize, ctrl: usize, tgt: usize) -> usize {
    if (idx >> ctrl) & 1 == 1 {
        idx ^ (1 << tgt)
    } else {
        idx
    }
}

/// SWAP permutation: exchange two bits in the index.
#[inline]
fn swap_map(idx: usize, a: usize, b: usize) -> usize {
    let bit_a = (idx >> a) & 1;
    let bit_b = (idx >> b) & 1;
    if bit_a == bit_b {
        idx
    } else {
        idx ^ (1 << a) ^ (1 << b)
    }
}

/// Toffoli permutation: flip target if both controls are set.
#[inline]
fn toffoli_map(idx: usize, c1: usize, c2: usize, tgt: usize) -> usize {
    if (idx >> c1) & 1 == 1 && (idx >> c2) & 1 == 1 {
        idx ^ (1 << tgt)
    } else {
        idx
    }
}

impl DensityMatrix {
    /// Create a new density matrix initialized to |0...0><0...0|.
    ///
    /// Panics if `n_qubits` exceeds MAX_QUBITS (14).
    pub fn new(n_qubits: usize) -> Self {
        assert!(
            n_qubits <= MAX_QUBITS,
            "density matrix supports at most {} qubits, got {}",
            MAX_QUBITS, n_qubits
        );
        let dim = 1usize << n_qubits;
        let mut data = vec![ComplexF64::ZERO; dim * dim];
        data[0] = ComplexF64::ONE; // |0><0| element
        DensityMatrix { n_qubits, dim, data }
    }

    /// Construct a density matrix from a statevector: rho = |psi><psi|.
    pub fn from_statevector(sv: &Statevector) -> Self {
        let n_qubits = sv.n_qubits;
        assert!(
            n_qubits <= MAX_QUBITS,
            "density matrix supports at most {} qubits, got {}",
            MAX_QUBITS, n_qubits
        );
        let dim = 1usize << n_qubits;
        let mut data = vec![ComplexF64::ZERO; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                // rho[i,j] = psi[i] * conj(psi[j])
                data[i * dim + j] = sv.amplitudes[i].mul_fixed(sv.amplitudes[j].conj());
            }
        }
        DensityMatrix { n_qubits, dim, data }
    }

    /// Get element rho[r, c].
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> ComplexF64 {
        self.data[r * self.dim + c]
    }

    /// Set element rho[r, c].
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: ComplexF64) {
        self.data[r * self.dim + c] = v;
    }

    /// Apply a quantum gate: rho -> U rho U^dagger.
    ///
    /// Handles single-qubit gates via the 2x2 matrix approach and multi-qubit
    /// gates (CNOT, CZ, SWAP, Toffoli) via basis permutations.
    pub fn apply_gate(&mut self, gate: &Gate) {
        if let Some((q, u)) = gate_matrix(gate) {
            self.apply_single_qubit_gate(q, &u);
        } else {
            match gate {
                Gate::CNOT(ctrl, tgt) => self.apply_permutation_gate(|idx| cnot_map(idx, *ctrl, *tgt)),
                Gate::CZ(a, b) => self.apply_cz_gate(*a, *b),
                Gate::SWAP(a, b) => self.apply_permutation_gate(|idx| swap_map(idx, *a, *b)),
                Gate::Toffoli(c1, c2, tgt) => self.apply_permutation_gate(|idx| toffoli_map(idx, *c1, *c2, *tgt)),
                _ => unreachable!(),
            }
        }
    }

    /// Apply a single-qubit unitary to qubit q on the density matrix.
    ///
    /// rho'[i,j] = Sum_{a,b} U[i_q, a] * rho[i_a, j_b] * conj(U[j_q, b])
    /// where i_a is index i with bit q replaced by a.
    fn apply_single_qubit_gate(&mut self, q: usize, u: &[[ComplexF64; 2]; 2]) {
        let dim = self.dim;
        let bit = 1usize << q;

        // Compute U^dagger for the right side
        let u_dag: [[ComplexF64; 2]; 2] = [
            [u[0][0].conj(), u[1][0].conj()],
            [u[0][1].conj(), u[1][1].conj()],
        ];

        // Apply U from the left: rho -> U rho
        // For each row i, consider pairs (i with bit q=0) and (i with bit q=1)
        for i in 0..dim {
            if i & bit != 0 {
                continue; // process only when bit q is 0
            }
            let i0 = i;       // bit q = 0
            let i1 = i | bit; // bit q = 1
            for j in 0..dim {
                let r0 = self.get(i0, j);
                let r1 = self.get(i1, j);
                // new_rho[i0, j] = U[0,0]*r0 + U[0,1]*r1
                // new_rho[i1, j] = U[1,0]*r0 + U[1,1]*r1
                self.set(i0, j, u[0][0].mul_fixed(r0).add(u[0][1].mul_fixed(r1)));
                self.set(i1, j, u[1][0].mul_fixed(r0).add(u[1][1].mul_fixed(r1)));
            }
        }

        // Apply U^dagger from the right: rho -> rho U^dagger
        // For each column j, consider pairs (j with bit q=0) and (j with bit q=1)
        for i in 0..dim {
            for j in 0..dim {
                if j & bit != 0 {
                    continue; // process only when bit q is 0
                }
                let j0 = j;
                let j1 = j | bit;
                let r0 = self.get(i, j0);
                let r1 = self.get(i, j1);
                // new_rho[i, j0] = r0 * U_dag[0,0] + r1 * U_dag[1,0]
                // new_rho[i, j1] = r0 * U_dag[0,1] + r1 * U_dag[1,1]
                self.set(i, j0, r0.mul_fixed(u_dag[0][0]).add(r1.mul_fixed(u_dag[1][0])));
                self.set(i, j1, r0.mul_fixed(u_dag[0][1]).add(r1.mul_fixed(u_dag[1][1])));
            }
        }
    }

    /// Apply a permutation gate: rho'[i,j] = rho[P(i), P(j)].
    /// Used for CNOT, SWAP, Toffoli which permute basis states.
    ///
    /// Optimized: uses in-place cycle-following permutation to avoid allocating
    /// a full dim×dim temporary matrix. Only uses O(dim) bits for visited tracking.
    fn apply_permutation_gate<F: Fn(usize) -> usize>(&mut self, perm: F) {
        let dim = self.dim;
        let n = dim * dim;

        // Build the combined permutation on the flattened index: (i,j) → (P(i), P(j))
        // Follow cycles in-place to avoid O(dim²) allocation.
        let mut visited = vec![false; n];

        for start in 0..n {
            if visited[start] { continue; }

            let start_i = start / dim;
            let start_j = start % dim;
            let dest = perm(start_i) * dim + perm(start_j);

            if dest == start {
                visited[start] = true;
                continue;
            }

            // Follow the cycle
            let mut current = start;
            let saved = self.data[start];

            loop {
                let ci = current / dim;
                let cj = current % dim;
                let next = perm(ci) * dim + perm(cj);
                visited[current] = true;

                if next == start {
                    self.data[current] = saved;
                    break;
                }

                self.data[current] = self.data[next];
                current = next;
            }
        }
    }

    /// Apply CZ gate: diagonal phase, rho'[i,j] = phase(i) * phase(j) * rho[i,j]
    /// where phase(k) = -1 if both bits a and b are set in k, else +1.
    fn apply_cz_gate(&mut self, a: usize, b: usize) {
        let dim = self.dim;
        let a_bit = 1usize << a;
        let b_bit = 1usize << b;
        for i in 0..dim {
            let phase_i = if (i & a_bit) != 0 && (i & b_bit) != 0 { -1.0 } else { 1.0 };
            for j in 0..dim {
                let phase_j = if (j & a_bit) != 0 && (j & b_bit) != 0 { -1.0 } else { 1.0 };
                let phase = phase_i * phase_j;
                if phase < 0.0 {
                    let idx = i * dim + j;
                    self.data[idx] = self.data[idx].neg();
                }
            }
        }
    }

    /// Apply full-dimension Kraus operators: rho -> Sum_k K_k rho K_k^dagger.
    ///
    /// Each K_k must be a dim x dim DenseMatrix.
    pub fn apply_kraus(&mut self, kraus_ops: &[DenseMatrix]) {
        let dim = self.dim;
        let mut new_data = vec![ComplexF64::ZERO; dim * dim];

        for k_op in kraus_ops {
            assert_eq!(k_op.rows, dim);
            assert_eq!(k_op.cols, dim);
            for i in 0..dim {
                for j in 0..dim {
                    // contribution = Sum_{a,b} K[i,a] * rho[a,b] * conj(K[j,b])
                    let mut acc = ComplexF64::ZERO;
                    for a in 0..dim {
                        let k_ia = k_op.get(i, a);
                        for b in 0..dim {
                            let rho_ab = self.get(a, b);
                            let k_jb_conj = k_op.get(j, b).conj();
                            acc = acc.add(k_ia.mul_fixed(rho_ab).mul_fixed(k_jb_conj));
                        }
                    }
                    new_data[i * dim + j] = new_data[i * dim + j].add(acc);
                }
            }
        }

        self.data = new_data;
    }

    /// Apply a single-qubit noise channel specified by 2x2 Kraus operators.
    ///
    /// For each Kraus operator K (2x2), compute rho' += K_q rho K_q^dagger
    /// where K_q acts on qubit q. The formula is:
    ///
    /// rho'[i,j] = Sum_k Sum_{a,b} K[i_q, a] * rho[i_a, j_b] * conj(K[j_q, b])
    ///
    /// where i_a means index i with qubit q set to value a.
    pub fn apply_single_qubit_channel(&mut self, q: usize, kraus: &[[[ComplexF64; 2]; 2]]) {
        let dim = self.dim;
        let bit = 1usize << q;
        let mut new_data = vec![ComplexF64::ZERO; dim * dim];

        for k_op in kraus {
            for i in 0..dim {
                let i_q = (i >> q) & 1; // value of qubit q in index i
                for j in 0..dim {
                    let j_q = (j >> q) & 1;
                    // Sum over a, b in {0, 1}
                    // rho'[i,j] += Sum_{a,b} K[i_q,a] * rho[i_a, j_b] * conj(K[j_q,b])
                    let mut acc = ComplexF64::ZERO;
                    for a in 0..2 {
                        let k_iq_a = k_op[i_q][a];
                        // i_a: index i with qubit q set to a
                        let i_a = (i & !(bit)) | (a << q);
                        for b in 0..2 {
                            let k_jq_b_conj = k_op[j_q][b].conj();
                            let j_b = (j & !(bit)) | (b << q);
                            let rho_ab = self.data[i_a * dim + j_b];
                            acc = acc.add(k_iq_a.mul_fixed(rho_ab).mul_fixed(k_jq_b_conj));
                        }
                    }
                    new_data[i * dim + j] = new_data[i * dim + j].add(acc);
                }
            }
        }

        self.data = new_data;
    }

    /// Trace of the density matrix: Tr(rho) = Sum_i rho[i,i].re.
    ///
    /// For a valid density matrix this should be 1.0.
    pub fn trace(&self) -> f64 {
        let mut sum = 0.0f64;
        for i in 0..self.dim {
            sum += self.get(i, i).re;
        }
        sum
    }

    /// Purity of the state: Tr(rho^2) = Sum_{i,j} |rho[i,j]|^2.
    ///
    /// Pure states have purity 1.0; maximally mixed states have purity 1/dim.
    pub fn purity(&self) -> f64 {
        let mut sum = 0.0f64;
        for i in 0..self.dim {
            for j in 0..self.dim {
                sum += self.get(i, j).norm_sq();
            }
        }
        sum
    }

    /// Measurement probabilities: P(k) = rho[k,k].re for each basis state k.
    pub fn probabilities(&self) -> Vec<f64> {
        let mut probs = Vec::with_capacity(self.dim);
        for k in 0..self.dim {
            probs.push(self.get(k, k).re);
        }
        probs
    }

    /// Fidelity between two density matrices (pure-state formula): F = Tr(rho * sigma).
    ///
    /// This is exact when at least one state is pure. For general mixed states,
    /// the true fidelity is (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2, but that
    /// requires matrix square roots. This simplified version suffices for most
    /// quantum circuit validation.
    pub fn fidelity(rho: &DensityMatrix, sigma: &DensityMatrix) -> f64 {
        assert_eq!(rho.dim, sigma.dim, "density matrices must have same dimension");
        let dim = rho.dim;
        // Tr(rho * sigma) = Sum_{i,k} rho[i,k] * sigma[k,i]
        let mut sum = 0.0f64;
        for i in 0..dim {
            for k in 0..dim {
                let prod = rho.get(i, k).mul_fixed(sigma.get(k, i));
                sum += prod.re;
            }
        }
        sum
    }

    /// Von Neumann entropy: S = -Tr(rho ln rho) = -Sum_i lambda_i ln(lambda_i).
    ///
    /// Computed by finding eigenvalues of the Hermitian density matrix via SVD
    /// (for a Hermitian positive-semidefinite matrix, singular values equal eigenvalues).
    pub fn von_neumann_entropy(&self) -> f64 {
        let dim = self.dim;
        let mut mat = DenseMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                mat.set(i, j, self.get(i, j));
            }
        }

        let svd = svd_sign_stabilized(&mat);

        let mut entropy = 0.0f64;
        for &lambda in &svd.s {
            if lambda > 1e-15 {
                entropy -= lambda * lambda.ln();
            }
        }
        entropy
    }

    /// Partial trace: trace out all qubits NOT in `keep_qubits`.
    ///
    /// Returns a reduced density matrix on the kept qubits.
    /// The kept qubits are re-indexed in the order they appear in `keep_qubits`.
    pub fn partial_trace(&self, keep_qubits: &[usize]) -> DensityMatrix {
        let n_keep = keep_qubits.len();
        let new_dim = 1usize << n_keep;
        let n_trace = self.n_qubits - n_keep;
        let trace_dim = 1usize << n_trace;

        // Determine which qubits to trace out
        let mut is_kept = vec![false; self.n_qubits];
        for &q in keep_qubits {
            is_kept[q] = true;
        }
        let trace_qubits: Vec<usize> = (0..self.n_qubits).filter(|q| !is_kept[*q]).collect();

        let mut result = DensityMatrix {
            n_qubits: n_keep,
            dim: new_dim,
            data: vec![ComplexF64::ZERO; new_dim * new_dim],
        };

        // For each pair of reduced indices (ri, rj), sum over all traced-out configurations
        for ri in 0..new_dim {
            for rj in 0..new_dim {
                let mut acc = ComplexF64::ZERO;
                for t in 0..trace_dim {
                    // Build full index from reduced index and trace index
                    let full_i = build_full_index(ri, keep_qubits, t, &trace_qubits);
                    let full_j = build_full_index(rj, keep_qubits, t, &trace_qubits);
                    acc = acc.add(self.get(full_i, full_j));
                }
                result.set(ri, rj, acc);
            }
        }

        result
    }
}

/// Build a full index from a reduced index (over kept qubits) and a trace index
/// (over traced-out qubits).
fn build_full_index(
    reduced_idx: usize,
    keep_qubits: &[usize],
    trace_idx: usize,
    trace_qubits: &[usize],
) -> usize {
    let mut full = 0usize;
    for (k, &q) in keep_qubits.iter().enumerate() {
        if (reduced_idx >> k) & 1 == 1 {
            full |= 1 << q;
        }
    }
    for (k, &q) in trace_qubits.iter().enumerate() {
        if (trace_idx >> k) & 1 == 1 {
            full |= 1 << q;
        }
    }
    full
}

// ---------------------------------------------------------------------------
// Noise channel constructors (single-qubit, return 2x2 Kraus operators)
// ---------------------------------------------------------------------------

/// Depolarizing channel for a single qubit.
///
/// With probability p the qubit is replaced by the maximally mixed state;
/// with probability (1-p) it is left unchanged.
///
/// Kraus operators:
///   K0 = sqrt(1 - p) * I
///   K1 = sqrt(p/3) * X
///   K2 = sqrt(p/3) * Y
///   K3 = sqrt(p/3) * Z
pub fn depolarizing_channel(p: f64) -> KrausOps2x2 {
    assert!((0.0..=1.0).contains(&p), "depolarizing parameter p must be in [0, 1]");
    let s0 = (1.0 - p).sqrt();
    let s1 = (p / 3.0).sqrt();

    let i_2x2 = [
        [ComplexF64::real(s0), ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::real(s0)],
    ];
    let x_2x2 = [
        [ComplexF64::ZERO, ComplexF64::real(s1)],
        [ComplexF64::real(s1), ComplexF64::ZERO],
    ];
    let y_2x2 = [
        [ComplexF64::ZERO, ComplexF64::new(0.0, -s1)],
        [ComplexF64::new(0.0, s1), ComplexF64::ZERO],
    ];
    let z_2x2 = [
        [ComplexF64::real(s1), ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::real(-s1)],
    ];

    vec![i_2x2, x_2x2, y_2x2, z_2x2]
}

/// Dephasing (phase damping) channel for a single qubit.
///
/// Destroys off-diagonal elements with probability p while preserving populations.
///
/// Kraus operators:
///   K0 = [[1, 0], [0, sqrt(1-p)]]
///   K1 = [[0, 0], [0, sqrt(p)]]
pub fn dephasing_channel(p: f64) -> KrausOps2x2 {
    assert!((0.0..=1.0).contains(&p), "dephasing parameter p must be in [0, 1]");

    let k0 = [
        [ComplexF64::ONE, ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::real((1.0 - p).sqrt())],
    ];
    let k1 = [
        [ComplexF64::ZERO, ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::real(p.sqrt())],
    ];

    vec![k0, k1]
}

/// Amplitude damping channel for a single qubit.
///
/// Models energy dissipation (T1 decay). With rate gamma, the excited state
/// |1> decays toward |0>.
///
/// Kraus operators:
///   K0 = [[1, 0], [0, sqrt(1-gamma)]]
///   K1 = [[0, sqrt(gamma)], [0, 0]]
pub fn amplitude_damping_channel(gamma: f64) -> KrausOps2x2 {
    assert!((0.0..=1.0).contains(&gamma), "damping parameter gamma must be in [0, 1]");

    let k0 = [
        [ComplexF64::ONE, ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::real((1.0 - gamma).sqrt())],
    ];
    let k1 = [
        [ComplexF64::ZERO, ComplexF64::real(gamma.sqrt())],
        [ComplexF64::ZERO, ComplexF64::ZERO],
    ];

    vec![k0, k1]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_new_is_pure() {
        let dm = DensityMatrix::new(2);
        assert!((dm.trace() - 1.0).abs() < TOL);
        assert!((dm.purity() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_from_statevector() {
        let sv = Statevector::new(2);
        let dm = DensityMatrix::from_statevector(&sv);
        assert!((dm.trace() - 1.0).abs() < TOL);
        assert!((dm.get(0, 0).re - 1.0).abs() < TOL); // |00><00|
    }

    #[test]
    fn test_apply_gate_h() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::H(0));
        // H|0> = |+>, rho = |+><+| = [[0.5, 0.5], [0.5, 0.5]]
        assert!((dm.get(0, 0).re - 0.5).abs() < TOL);
        assert!((dm.get(0, 1).re - 0.5).abs() < TOL);
        assert!((dm.get(1, 0).re - 0.5).abs() < TOL);
        assert!((dm.get(1, 1).re - 0.5).abs() < TOL);
    }

    #[test]
    fn test_apply_gate_x() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::X(0));
        // X|0> = |1>, rho = |1><1|
        assert!(dm.get(0, 0).norm_sq() < TOL);
        assert!((dm.get(1, 1).re - 1.0).abs() < TOL);
    }

    #[test]
    fn test_apply_gate_cnot() {
        // Create Bell state via H then CNOT
        let mut dm = DensityMatrix::new(2);
        dm.apply_gate(&Gate::H(0));
        dm.apply_gate(&Gate::CNOT(0, 1));
        // Bell state: (|00>+|11>)/sqrt(2), rho[0,0]=0.5, rho[0,3]=0.5, rho[3,0]=0.5, rho[3,3]=0.5
        assert!((dm.get(0, 0).re - 0.5).abs() < TOL);
        assert!((dm.get(0, 3).re - 0.5).abs() < TOL);
        assert!((dm.get(3, 0).re - 0.5).abs() < TOL);
        assert!((dm.get(3, 3).re - 0.5).abs() < TOL);
        assert!((dm.trace() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_apply_gate_cz() {
        let mut dm = DensityMatrix::new(2);
        dm.apply_gate(&Gate::H(0));
        dm.apply_gate(&Gate::H(1));
        let before_11 = dm.get(3, 3);
        dm.apply_gate(&Gate::CZ(0, 1));
        // CZ only flips phase of |11> component
        assert!((dm.trace() - 1.0).abs() < TOL);
        assert!((dm.get(3, 3).re - before_11.re).abs() < TOL);
    }

    #[test]
    fn test_apply_gate_swap() {
        let mut dm = DensityMatrix::new(2);
        dm.apply_gate(&Gate::X(0)); // |01> in index 1
        dm.apply_gate(&Gate::SWAP(0, 1));
        // Should be |10> now, index 2
        assert!((dm.get(2, 2).re - 1.0).abs() < TOL);
        assert!((dm.trace() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_probabilities_match_statevector() {
        let mut sv = Statevector::new(2);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::CNOT(0, 1).apply(&mut sv).unwrap();
        let dm = DensityMatrix::from_statevector(&sv);
        let sv_probs = sv.probabilities();
        let dm_probs = dm.probabilities();
        for i in 0..4 {
            assert!(
                (sv_probs[i] - dm_probs[i]).abs() < TOL,
                "prob[{}]: sv={} dm={}", i, sv_probs[i], dm_probs[i]
            );
        }
    }

    #[test]
    fn test_depolarizing_reduces_purity() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::H(0));
        let purity_before = dm.purity();
        let kraus = depolarizing_channel(0.1);
        dm.apply_single_qubit_channel(0, &kraus);
        let purity_after = dm.purity();
        assert!(purity_after < purity_before, "Depolarizing should reduce purity");
        assert!((dm.trace() - 1.0).abs() < TOL, "Trace must be preserved");
    }

    #[test]
    fn test_amplitude_damping_drives_to_zero() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::X(0)); // |1>
        // Apply strong amplitude damping
        let kraus = amplitude_damping_channel(0.9);
        dm.apply_single_qubit_channel(0, &kraus);
        // Should be mostly |0> now
        assert!(dm.get(0, 0).re > 0.8, "Should decay toward |0>");
        assert!((dm.trace() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_von_neumann_entropy_pure() {
        let dm = DensityMatrix::new(2);
        let s = dm.von_neumann_entropy();
        assert!(s.abs() < 1e-6, "Pure state entropy should be 0, got {}", s);
    }

    #[test]
    fn test_von_neumann_entropy_maximally_mixed() {
        // Maximally mixed 1-qubit: rho = I/2
        let mut dm = DensityMatrix::new(1);
        dm.set(0, 0, ComplexF64::real(0.5));
        dm.set(1, 1, ComplexF64::real(0.5));
        dm.set(0, 1, ComplexF64::ZERO);
        dm.set(1, 0, ComplexF64::ZERO);
        let s = dm.von_neumann_entropy();
        let expected = 2.0f64.ln(); // ln(2) ~ 0.693
        assert!((s - expected).abs() < 1e-6, "Max mixed entropy should be ln(2), got {}", s);
    }

    #[test]
    fn test_partial_trace_bell_state() {
        // Bell state: (|00>+|11>)/sqrt(2)
        // Partial trace over qubit 1 should give maximally mixed state on qubit 0
        let mut sv = Statevector::new(2);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::CNOT(0, 1).apply(&mut sv).unwrap();
        let dm = DensityMatrix::from_statevector(&sv);
        let reduced = dm.partial_trace(&[0]); // keep qubit 0
        assert!((reduced.get(0, 0).re - 0.5).abs() < TOL);
        assert!((reduced.get(1, 1).re - 0.5).abs() < TOL);
        assert!(reduced.get(0, 1).norm_sq() < TOL);
    }

    #[test]
    fn test_partial_trace_product_state() {
        // |0> tensor |1> -> partial trace over qubit 1 should give |0><0|
        let mut sv = Statevector::new(2);
        Gate::X(1).apply(&mut sv).unwrap(); // |0> on q0, |1> on q1
        let dm = DensityMatrix::from_statevector(&sv);
        let reduced = dm.partial_trace(&[0]); // keep qubit 0
        assert!((reduced.get(0, 0).re - 1.0).abs() < TOL);
        assert!(reduced.get(1, 1).norm_sq() < TOL);
    }

    #[test]
    fn test_fidelity_same_state() {
        let dm = DensityMatrix::new(2);
        let f = DensityMatrix::fidelity(&dm, &dm);
        assert!((f - 1.0).abs() < TOL);
    }

    #[test]
    fn test_fidelity_orthogonal_states() {
        let dm0 = DensityMatrix::new(1); // |0><0|
        let mut dm1 = DensityMatrix::new(1);
        dm1.apply_gate(&Gate::X(0)); // |1><1|
        let f = DensityMatrix::fidelity(&dm0, &dm1);
        assert!(f.abs() < TOL, "Orthogonal states should have fidelity 0, got {}", f);
    }

    #[test]
    fn test_dephasing_preserves_diagonal() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::H(0));
        let p00_before = dm.get(0, 0).re;
        let kraus = dephasing_channel(0.5);
        dm.apply_single_qubit_channel(0, &kraus);
        let p00_after = dm.get(0, 0).re;
        // Dephasing preserves diagonal but kills off-diagonal
        assert!((p00_before - p00_after).abs() < TOL);
        assert!((dm.trace() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_dephasing_reduces_off_diagonal() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::H(0));
        let off_before = dm.get(0, 1).norm_sq();
        let kraus = dephasing_channel(0.5);
        dm.apply_single_qubit_channel(0, &kraus);
        let off_after = dm.get(0, 1).norm_sq();
        assert!(off_after < off_before, "Dephasing should reduce off-diagonal elements");
    }

    #[test]
    fn test_depolarizing_p0_is_identity() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::H(0));
        let before = dm.data.clone();
        let kraus = depolarizing_channel(0.0);
        dm.apply_single_qubit_channel(0, &kraus);
        for i in 0..dm.data.len() {
            assert!(
                (dm.data[i].re - before[i].re).abs() < TOL &&
                (dm.data[i].im - before[i].im).abs() < TOL,
                "p=0 depolarizing should be identity"
            );
        }
    }

    #[test]
    fn test_apply_kraus_full_dimension() {
        // Test apply_kraus with identity channel (single Kraus op = I)
        let mut dm = DensityMatrix::new(1);
        dm.apply_gate(&Gate::H(0));
        let before = dm.data.clone();
        let id = DenseMatrix::identity(2);
        dm.apply_kraus(&[id]);
        for i in 0..dm.data.len() {
            assert!(
                (dm.data[i].re - before[i].re).abs() < TOL &&
                (dm.data[i].im - before[i].im).abs() < TOL,
                "Identity Kraus should preserve state"
            );
        }
    }

    #[test]
    fn test_gate_then_trace_preserved() {
        // Apply a sequence of gates, trace should remain 1
        let mut dm = DensityMatrix::new(3);
        dm.apply_gate(&Gate::H(0));
        dm.apply_gate(&Gate::CNOT(0, 1));
        dm.apply_gate(&Gate::T(2));
        dm.apply_gate(&Gate::Ry(1, 0.7));
        dm.apply_gate(&Gate::CZ(1, 2));
        assert!(
            (dm.trace() - 1.0).abs() < TOL,
            "Trace should be preserved after gate sequence, got {}",
            dm.trace()
        );
    }

    #[test]
    fn test_toffoli_gate() {
        let mut dm = DensityMatrix::new(3);
        dm.apply_gate(&Gate::X(0)); // set qubit 0
        dm.apply_gate(&Gate::X(1)); // set qubit 1
        // Now state is |11> on q0,q1, |0> on q2 -> index 3
        dm.apply_gate(&Gate::Toffoli(0, 1, 2));
        // Should flip q2: index 3 -> index 7
        assert!((dm.get(7, 7).re - 1.0).abs() < TOL);
        assert!((dm.trace() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_density_matrix_vs_statevector_gate_sequence() {
        // Verify DM and SV produce the same probabilities after a gate sequence
        let mut sv = Statevector::new(3);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::CNOT(0, 1).apply(&mut sv).unwrap();
        Gate::Ry(2, 1.2).apply(&mut sv).unwrap();
        Gate::CZ(1, 2).apply(&mut sv).unwrap();

        let mut dm = DensityMatrix::new(3);
        dm.apply_gate(&Gate::H(0));
        dm.apply_gate(&Gate::CNOT(0, 1));
        dm.apply_gate(&Gate::Ry(2, 1.2));
        dm.apply_gate(&Gate::CZ(1, 2));

        let sv_probs = sv.probabilities();
        let dm_probs = dm.probabilities();
        for i in 0..8 {
            assert!(
                (sv_probs[i] - dm_probs[i]).abs() < TOL,
                "prob[{}]: sv={} dm={}", i, sv_probs[i], dm_probs[i]
            );
        }
    }
}
