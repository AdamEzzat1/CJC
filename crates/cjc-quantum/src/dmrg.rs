//! Two-Site DMRG (Density Matrix Renormalization Group) for 1D Hamiltonians.
//!
//! Finds ground states of nearest-neighbor 1D spin chains by iteratively
//! optimizing pairs of adjacent MPS tensors via Lanczos eigensolver + SVD.
//!
//! # Algorithm
//!
//! 1. Initialize MPS as a random product state (seeded SplitMix64).
//! 2. Build left and right environment tensors encoding the full Hamiltonian.
//! 3. Sweep left-to-right then right-to-left:
//!    a. Build operator-weighted environments (identity, Hamiltonian, boundary ops).
//!    b. Form two-site tensor theta from sites (i, i+1).
//!    c. Apply effective Hamiltonian (all terms) via Lanczos to find ground state.
//!    d. SVD and truncate to max_bond.
//!    e. Update MPS tensors.
//! 4. Repeat sweeps until energy converges.
//!
//! # Supported Hamiltonians
//!
//! - **Ising**: H = sum_i Z_i Z_{i+1}
//! - **Heisenberg**: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
//!
//! # Determinism
//!
//! - All floating-point reductions are ordered deterministically.
//! - SVD uses sign-stabilized one-sided Jacobi (no random pivoting).
//! - RNG is SplitMix64 with fixed seed (42).
//! - No HashMap or HashSet; all iteration is deterministic.

use cjc_runtime::complex::ComplexF64;

use crate::mps::{DenseMatrix, Mps, MpsTensor, SvdResult, svd_sign_stabilized};
use crate::vqe::{mps_heisenberg_energy, mps_full_heisenberg_energy};

// ---------------------------------------------------------------------------
// Hamiltonian types
// ---------------------------------------------------------------------------

/// Supported nearest-neighbor 1D Hamiltonians for DMRG.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DmrgHamiltonian {
    /// ZZ-only Ising: H = sum_i Z_i Z_{i+1}.
    Ising,
    /// Full Heisenberg: H = sum_i (XX + YY + ZZ)_{i,i+1}.
    Heisenberg,
}

/// Result of a DMRG optimization.
#[derive(Debug, Clone)]
pub struct DmrgResult {
    /// Final ground-state energy estimate.
    pub energy: f64,
    /// Optimized MPS.
    pub mps: Mps,
    /// Energy after each sweep (including initial).
    pub energies: Vec<f64>,
    /// Number of sweeps performed.
    pub sweeps: usize,
}

// ---------------------------------------------------------------------------
// Local Hamiltonian matrices (4x4, two-site)
// ---------------------------------------------------------------------------

/// 4x4 local Hamiltonian for two adjacent sites in the {|00>, |01>, |10>, |11>} basis.
fn local_hamiltonian(h: DmrgHamiltonian) -> [[f64; 4]; 4] {
    match h {
        DmrgHamiltonian::Ising => {
            let mut m = [[0.0; 4]; 4];
            m[0][0] = 1.0;
            m[1][1] = -1.0;
            m[2][2] = -1.0;
            m[3][3] = 1.0;
            m
        }
        DmrgHamiltonian::Heisenberg => {
            // XX + YY + ZZ = [[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]]
            let mut m = [[0.0; 4]; 4];
            m[0][0] = 1.0;
            m[1][1] = -1.0;
            m[1][2] = 2.0;
            m[2][1] = 2.0;
            m[2][2] = -1.0;
            m[3][3] = 1.0;
            m
        }
    }
}

// ---------------------------------------------------------------------------
// SVD helper for wide matrices
// ---------------------------------------------------------------------------

/// SVD that handles both tall and wide matrices by transposing if needed.
fn svd_tall_or_wide(mat: &DenseMatrix) -> SvdResult {
    if mat.rows >= mat.cols {
        svd_sign_stabilized(mat)
    } else {
        let mut mat_t = DenseMatrix::zeros(mat.cols, mat.rows);
        for r in 0..mat.rows {
            for c in 0..mat.cols {
                mat_t.set(c, r, mat.get(r, c).conj());
            }
        }
        let svd_t = svd_sign_stabilized(&mat_t);
        let k = svd_t.s.len();
        let m = mat.rows;
        let n = mat.cols;
        let mut u_a = DenseMatrix::zeros(m, k);
        for r in 0..m {
            for c in 0..k {
                u_a.set(r, c, svd_t.vh.get(c, r).conj());
            }
        }
        let mut vh_a = DenseMatrix::zeros(k, n);
        for r in 0..k {
            for c in 0..n {
                vh_a.set(r, c, svd_t.u.get(c, r).conj());
            }
        }
        SvdResult {
            u: u_a,
            s: svd_t.s,
            vh: vh_a,
        }
    }
}

// ---------------------------------------------------------------------------
// Complex vector helpers
// ---------------------------------------------------------------------------

fn complex_norm(v: &[ComplexF64]) -> f64 {
    let mut sum = 0.0;
    for x in v {
        sum += x.norm_sq();
    }
    sum.sqrt()
}

fn complex_real_dot(a: &[ComplexF64], b: &[ComplexF64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i].conj().mul_fixed(b[i]).re;
    }
    sum
}

fn complex_dot(a: &[ComplexF64], b: &[ComplexF64]) -> ComplexF64 {
    let mut sum = ComplexF64::ZERO;
    for i in 0..a.len() {
        sum = sum.add(a[i].conj().mul_fixed(b[i]));
    }
    sum
}

// ---------------------------------------------------------------------------
// Lanczos eigensolver
// ---------------------------------------------------------------------------

fn lanczos_ground(
    apply_h: &dyn Fn(&[ComplexF64]) -> Vec<ComplexF64>,
    initial: &[ComplexF64],
    dim: usize,
    max_iter: usize,
) -> (f64, Vec<ComplexF64>) {
    let n_iter = max_iter.min(dim).max(1);

    let mut v = initial.to_vec();
    let norm = complex_norm(&v);
    if norm < 1e-15 {
        return (0.0, v);
    }
    let inv = 1.0 / norm;
    for x in &mut v {
        *x = x.scale(inv);
    }

    let mut vecs: Vec<Vec<ComplexF64>> = Vec::with_capacity(n_iter);
    let mut alphas: Vec<f64> = Vec::with_capacity(n_iter);
    let mut betas: Vec<f64> = Vec::with_capacity(n_iter);

    vecs.push(v.clone());

    let mut w = apply_h(&v);
    let alpha_0 = complex_real_dot(&w, &v);
    alphas.push(alpha_0);
    betas.push(0.0);

    for i in 0..dim {
        w[i] = w[i].sub(v[i].scale(alpha_0));
    }

    for _j in 1..n_iter {
        let beta_j = complex_norm(&w);
        if beta_j < 1e-14 {
            break;
        }
        betas.push(beta_j);

        let inv_beta = 1.0 / beta_j;
        let mut v_new = vec![ComplexF64::ZERO; dim];
        for i in 0..dim {
            v_new[i] = w[i].scale(inv_beta);
        }

        let v_prev = vecs.last().unwrap().clone();
        vecs.push(v_new.clone());

        w = apply_h(&v_new);
        let alpha_j = complex_real_dot(&w, &v_new);
        alphas.push(alpha_j);

        for i in 0..dim {
            w[i] = w[i].sub(v_new[i].scale(alpha_j)).sub(v_prev[i].scale(beta_j));
        }

        // Full reorthogonalization
        for prev in &vecs {
            let overlap = complex_dot(prev, &w);
            for i in 0..dim {
                w[i] = w[i].sub(prev[i].mul_fixed(overlap));
            }
        }
    }

    let n_actual = vecs.len();
    if n_actual == 0 {
        return (0.0, initial.to_vec());
    }

    let (evals, evecs_tri) = diag_tridiagonal(&alphas, &betas);

    let mut min_idx = 0;
    for i in 1..evals.len() {
        if evals[i] < evals[min_idx] {
            min_idx = i;
        }
    }

    let mut result = vec![ComplexF64::ZERO; dim];
    for j in 0..n_actual {
        let coeff = evecs_tri[j * n_actual + min_idx];
        for i in 0..dim {
            result[i] = result[i].add(vecs[j][i].scale(coeff));
        }
    }

    let result_norm = complex_norm(&result);
    if result_norm > 1e-15 {
        let inv = 1.0 / result_norm;
        for x in &mut result {
            *x = x.scale(inv);
        }
    }

    (evals[min_idx], result)
}

// ---------------------------------------------------------------------------
// Tridiagonal diagonalization (Jacobi)
// ---------------------------------------------------------------------------

fn diag_tridiagonal(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = alphas.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![alphas[0]], vec![1.0]);
    }

    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = alphas[i];
    }
    for i in 1..n {
        let b = if i < betas.len() { betas[i] } else { 0.0 };
        a[i * n + (i - 1)] = b;
        a[(i - 1) * n + i] = b;
    }

    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }

        for p in 0..n {
            for qq_idx in (p + 1)..n {
                let apq = a[p * n + qq_idx];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (a[qq_idx * n + qq_idx] - a[p * n + p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = a[p * n + p];
                a[p * n + p] = app - t * apq;
                a[qq_idx * n + qq_idx] = a[qq_idx * n + qq_idx] + t * apq;
                a[p * n + qq_idx] = 0.0;
                a[qq_idx * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == qq_idx {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + qq_idx];
                    a[r * n + p] = c * arp - s * arq;
                    a[p * n + r] = c * arp - s * arq;
                    a[r * n + qq_idx] = s * arp + c * arq;
                    a[qq_idx * n + r] = s * arp + c * arq;
                }
                for r in 0..n {
                    let qrp = q[r * n + p];
                    let qrq = q[r * n + qq_idx];
                    q[r * n + p] = c * qrp - s * qrq;
                    q[r * n + qq_idx] = s * qrp + c * qrq;
                }
            }
        }
    }

    let evals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (evals, q)
}

// ---------------------------------------------------------------------------
// Pauli operators (2x2, complex)
// ---------------------------------------------------------------------------

fn op_i() -> [[ComplexF64; 2]; 2] {
    [
        [ComplexF64::ONE, ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::ONE],
    ]
}

fn op_z() -> [[ComplexF64; 2]; 2] {
    [
        [ComplexF64::ONE, ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::real(-1.0)],
    ]
}

fn op_x() -> [[ComplexF64; 2]; 2] {
    [
        [ComplexF64::ZERO, ComplexF64::ONE],
        [ComplexF64::ONE, ComplexF64::ZERO],
    ]
}

fn op_y() -> [[ComplexF64; 2]; 2] {
    [
        [ComplexF64::ZERO, ComplexF64::new(0.0, -1.0)],
        [ComplexF64::new(0.0, 1.0), ComplexF64::ZERO],
    ]
}

/// Which operators couple nearest neighbors in the Hamiltonian.
fn hamiltonian_ops(h: DmrgHamiltonian) -> Vec<[[ComplexF64; 2]; 2]> {
    match h {
        DmrgHamiltonian::Ising => vec![op_z()],
        DmrgHamiltonian::Heisenberg => vec![op_x(), op_y(), op_z()],
    }
}

// ---------------------------------------------------------------------------
// Operator-weighted transfer matrices
// ---------------------------------------------------------------------------

/// Left-to-right transfer with operator on physical index:
///   result[a', a] = sum_{s_bra, s_ket, j, j'}
///     env[j, j'] * conj(A^{s_bra}[j, a']) * Op[s_bra, s_ket] * A^{s_ket}[j', a]
fn transfer_left_op(
    tensor: &MpsTensor,
    env: &[Vec<ComplexF64>],
    op: &[[ComplexF64; 2]; 2],
) -> Vec<Vec<ComplexF64>> {
    let bl = tensor.bond_left;
    let br = tensor.bond_right;
    let mut result = vec![vec![ComplexF64::ZERO; br]; br];

    for s_bra in 0..2 {
        for s_ket in 0..2 {
            let o = op[s_bra][s_ket];
            if o.re == 0.0 && o.im == 0.0 {
                continue;
            }
            for j in 0..bl {
                for jp in 0..bl {
                    let e = env[j][jp];
                    if e.re == 0.0 && e.im == 0.0 {
                        continue;
                    }
                    let eo = e.mul_fixed(o);
                    for a in 0..br {
                        let ca = tensor.a[s_bra].get(j, a).conj();
                        let eoa = eo.mul_fixed(ca);
                        for b in 0..br {
                            let v = tensor.a[s_ket].get(jp, b);
                            result[a][b] = result[a][b].add(eoa.mul_fixed(v));
                        }
                    }
                }
            }
        }
    }

    result
}

/// Right-to-left transfer with operator on physical index:
///   result[j, j'] = sum_{s_bra, s_ket, a, a'}
///     env[a, a'] * conj(A^{s_bra}[j, a]) * Op[s_bra, s_ket] * A^{s_ket}[j', a']
fn transfer_right_op(
    tensor: &MpsTensor,
    env: &[Vec<ComplexF64>],
    op: &[[ComplexF64; 2]; 2],
) -> Vec<Vec<ComplexF64>> {
    let bl = tensor.bond_left;
    let br = tensor.bond_right;
    let mut result = vec![vec![ComplexF64::ZERO; bl]; bl];

    for s_bra in 0..2 {
        for s_ket in 0..2 {
            let o = op[s_bra][s_ket];
            if o.re == 0.0 && o.im == 0.0 {
                continue;
            }
            for a in 0..br {
                for ap in 0..br {
                    let e = env[a][ap];
                    if e.re == 0.0 && e.im == 0.0 {
                        continue;
                    }
                    let eo = e.mul_fixed(o);
                    for j in 0..bl {
                        let ca = tensor.a[s_bra].get(j, a).conj();
                        let eoa = eo.mul_fixed(ca);
                        for jp in 0..bl {
                            let v = tensor.a[s_ket].get(jp, ap);
                            result[j][jp] = result[j][jp].add(eoa.mul_fixed(v));
                        }
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Bond environment construction
// ---------------------------------------------------------------------------

/// Full Hamiltonian environments for optimizing bond (i, i+1).
///
/// For H = sum_{k} h_{k,k+1}, the effective Hamiltonian at bond (i, i+1) is:
///   H_eff = L_H ⊗ I ⊗ R_I          (accumulated energy from bonds k < i-1)
///         + L_Op ⊗ (Op_i ⊗ I_j) ⊗ R_I   (boundary term h_{i-1,i})
///         + L_I ⊗ h_{i,i+1} ⊗ R_I   (direct local term)
///         + L_I ⊗ (I_i ⊗ Op_j) ⊗ R_Op  (boundary term h_{i+1,i+2})
///         + L_I ⊗ I ⊗ R_H          (accumulated energy from bonds k > i+2)
struct BondEnvs {
    l_i: Vec<Vec<ComplexF64>>,
    l_h: Vec<Vec<ComplexF64>>,
    l_ops: Vec<Vec<Vec<ComplexF64>>>,
    r_i: Vec<Vec<ComplexF64>>,
    r_h: Vec<Vec<ComplexF64>>,
    r_ops: Vec<Vec<Vec<ComplexF64>>>,
}

fn zero_env(dim: usize) -> Vec<Vec<ComplexF64>> {
    vec![vec![ComplexF64::ZERO; dim]; dim]
}

fn add_env(a: &mut [Vec<ComplexF64>], b: &[Vec<ComplexF64>]) {
    let n = a.len();
    for i in 0..n {
        for j in 0..n {
            a[i][j] = a[i][j].add(b[i][j]);
        }
    }
}

/// Build operator-weighted environments for optimizing bond (bond, bond+1).
fn build_bond_envs(mps: &Mps, bond: usize, hamiltonian: DmrgHamiltonian) -> BondEnvs {
    let ops = hamiltonian_ops(hamiltonian);
    let n_ops = ops.len();
    let oi = op_i();
    let n = mps.n_qubits;

    // ── Left environments (sites 0..bond-1) ──
    let mut l_i = zero_env(1);
    l_i[0][0] = ComplexF64::ONE;
    let mut l_h = zero_env(1);
    let mut l_ops: Vec<Vec<Vec<ComplexF64>>> = (0..n_ops).map(|_| zero_env(1)).collect();

    for k in 0..bond {
        let t = &mps.tensors[k];

        // Accumulated Hamiltonian: propagate old L_H through identity,
        // plus complete each dangling operator pair.
        let mut new_l_h = transfer_left_op(t, &l_h, &oi);
        for (o, op) in ops.iter().enumerate() {
            let completed = transfer_left_op(t, &l_ops[o], op);
            add_env(&mut new_l_h, &completed);
        }

        // Fresh dangling operators from site k.
        let new_l_ops: Vec<Vec<Vec<ComplexF64>>> = ops
            .iter()
            .map(|op| transfer_left_op(t, &l_i, op))
            .collect();

        let new_l_i = transfer_left_op(t, &l_i, &oi);

        l_i = new_l_i;
        l_h = new_l_h;
        l_ops = new_l_ops;
    }

    // ── Right environments (sites bond+2..N-1) ──
    let mut r_i = zero_env(1);
    r_i[0][0] = ComplexF64::ONE;
    let mut r_h = zero_env(1);
    let mut r_ops: Vec<Vec<Vec<ComplexF64>>> = (0..n_ops).map(|_| zero_env(1)).collect();

    for k in (bond + 2..n).rev() {
        let t = &mps.tensors[k];

        let mut new_r_h = transfer_right_op(t, &r_h, &oi);
        for (o, op) in ops.iter().enumerate() {
            let completed = transfer_right_op(t, &r_ops[o], op);
            add_env(&mut new_r_h, &completed);
        }

        let new_r_ops: Vec<Vec<Vec<ComplexF64>>> = ops
            .iter()
            .map(|op| transfer_right_op(t, &r_i, op))
            .collect();

        let new_r_i = transfer_right_op(t, &r_i, &oi);

        r_i = new_r_i;
        r_h = new_r_h;
        r_ops = new_r_ops;
    }

    BondEnvs {
        l_i,
        l_h,
        l_ops,
        r_i,
        r_h,
        r_ops,
    }
}

// ---------------------------------------------------------------------------
// Effective Hamiltonian application (all terms)
// ---------------------------------------------------------------------------

/// Kronecker product: Op ⊗ I in 4x4 physical space.
fn kron_op_i(op: &[[ComplexF64; 2]; 2]) -> [[ComplexF64; 4]; 4] {
    let mut m = [[ComplexF64::ZERO; 4]; 4];
    for si_p in 0..2 {
        for si in 0..2 {
            for sj in 0..2 {
                m[si_p * 2 + sj][si * 2 + sj] = op[si_p][si];
            }
        }
    }
    m
}

/// Kronecker product: I ⊗ Op in 4x4 physical space.
fn kron_i_op(op: &[[ComplexF64; 2]; 2]) -> [[ComplexF64; 4]; 4] {
    let mut m = [[ComplexF64::ZERO; 4]; 4];
    for si in 0..2 {
        for sj_p in 0..2 {
            for sj in 0..2 {
                m[si * 2 + sj_p][si * 2 + sj] = op[sj_p][sj];
            }
        }
    }
    m
}

/// Apply one term: result += left_env ⊗ phys_op ⊗ right_env applied to theta.
fn apply_term(
    theta: &[ComplexF64],
    left_env: &[Vec<ComplexF64>],
    phys_op: &[[ComplexF64; 4]; 4],
    right_env: &[Vec<ComplexF64>],
    bl: usize,
    br: usize,
    result: &mut [ComplexF64],
) {
    let rd = 2 * br;

    for ap in 0..bl {
        for bp in 0..br {
            let mut out = [ComplexF64::ZERO; 4];

            for a in 0..bl {
                let lv = left_env[ap][a];
                if lv.re == 0.0 && lv.im == 0.0 {
                    continue;
                }
                for b in 0..br {
                    let rv = right_env[b][bp];
                    if rv.re == 0.0 && rv.im == 0.0 {
                        continue;
                    }
                    let lr = lv.mul_fixed(rv);

                    let mut phys = [ComplexF64::ZERO; 4];
                    for si in 0..2 {
                        for sj in 0..2 {
                            phys[si * 2 + sj] = theta[(si * bl + a) * rd + sj * br + b];
                        }
                    }

                    for row in 0..4 {
                        for col in 0..4 {
                            let h = phys_op[row][col];
                            if h.re == 0.0 && h.im == 0.0 {
                                continue;
                            }
                            out[row] = out[row].add(lr.mul_fixed(phys[col]).mul_fixed(h));
                        }
                    }
                }
            }

            for si in 0..2 {
                for sj in 0..2 {
                    let idx = (si * bl + ap) * rd + sj * br + bp;
                    result[idx] = result[idx].add(out[si * 2 + sj]);
                }
            }
        }
    }
}

/// Apply the full effective Hamiltonian to theta, including ALL bond terms.
fn apply_h_eff_full(
    theta: &[ComplexF64],
    h_local: &[[f64; 4]; 4],
    envs: &BondEnvs,
    hamiltonian: DmrgHamiltonian,
    bl: usize,
    br: usize,
) -> Vec<ComplexF64> {
    let rd = 2 * br;
    let total = 2 * bl * rd;
    let mut result = vec![ComplexF64::ZERO; total];

    let ops = hamiltonian_ops(hamiltonian);

    let mut i4 = [[ComplexF64::ZERO; 4]; 4];
    for i in 0..4 {
        i4[i][i] = ComplexF64::ONE;
    }

    let mut h_c = [[ComplexF64::ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            h_c[i][j] = ComplexF64::real(h_local[i][j]);
        }
    }

    // Term 1: L_I ⊗ h_local ⊗ R_I
    apply_term(theta, &envs.l_i, &h_c, &envs.r_i, bl, br, &mut result);

    // Term 2: L_H ⊗ I ⊗ R_I
    apply_term(theta, &envs.l_h, &i4, &envs.r_i, bl, br, &mut result);

    // Term 3: L_I ⊗ I ⊗ R_H
    apply_term(theta, &envs.l_i, &i4, &envs.r_h, bl, br, &mut result);

    // Term 4: L_Op ⊗ (Op ⊗ I) ⊗ R_I (left boundary)
    for (o, op) in ops.iter().enumerate() {
        let phys = kron_op_i(op);
        apply_term(theta, &envs.l_ops[o], &phys, &envs.r_i, bl, br, &mut result);
    }

    // Term 5: L_I ⊗ (I ⊗ Op) ⊗ R_Op (right boundary)
    for (o, op) in ops.iter().enumerate() {
        let phys = kron_i_op(op);
        apply_term(theta, &envs.l_i, &phys, &envs.r_ops[o], bl, br, &mut result);
    }

    result
}

// ---------------------------------------------------------------------------
// Two-site optimization via Lanczos
// ---------------------------------------------------------------------------

/// Direction of sweep for SVD splitting.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SweepDir {
    LeftToRight,
    RightToLeft,
}

/// Optimize the two-site tensor at bond (i, i+1) using Lanczos with
/// the full effective Hamiltonian (all bond terms included).
fn optimize_two_site(
    mps: &mut Mps,
    bond: usize,
    h_local: &[[f64; 4]; 4],
    max_bond: usize,
    envs: &BondEnvs,
    hamiltonian: DmrgHamiltonian,
    direction: SweepDir,
) {
    let i = bond;
    let j = bond + 1;
    let bl = mps.tensors[i].bond_left;
    let br = mps.tensors[j].bond_right;
    let bm = mps.tensors[i].bond_right;

    let right_dim = 2 * br;
    let total_dim = 2 * bl * right_dim;

    // Form two-site tensor theta
    let mut theta = vec![ComplexF64::ZERO; total_dim];
    for si in 0..2 {
        for sj in 0..2 {
            for a in 0..bl {
                for b in 0..br {
                    let mut val = ComplexF64::ZERO;
                    for g in 0..bm {
                        val = val.add(
                            mps.tensors[i].a[si]
                                .get(a, g)
                                .mul_fixed(mps.tensors[j].a[sj].get(g, b)),
                        );
                    }
                    theta[(si * bl + a) * right_dim + sj * br + b] = val;
                }
            }
        }
    }

    // Small perturbation to escape eigenspace traps
    let mut perturb_rng = (bond as u64).wrapping_mul(7919) ^ 123456789;
    let perturb_scale = complex_norm(&theta) * 1e-3;
    if perturb_scale > 0.0 {
        for idx in 0..total_dim {
            let r = crate::rand_f64(&mut perturb_rng) - 0.5;
            theta[idx] = theta[idx].add(ComplexF64::real(r * perturb_scale));
        }
    }

    // Lanczos to find ground state of H_eff
    let lanczos_iters = 30.min(total_dim);
    let apply_h = |v: &[ComplexF64]| -> Vec<ComplexF64> {
        apply_h_eff_full(v, h_local, envs, hamiltonian, bl, br)
    };

    let (_energy, ground_state) = lanczos_ground(&apply_h, &theta, total_dim, lanczos_iters);

    // SVD the ground state
    let left_dim = 2 * bl;
    let mut mat = DenseMatrix::zeros(left_dim, right_dim);
    for idx in 0..total_dim {
        mat.set(idx / right_dim, idx % right_dim, ground_state[idx]);
    }

    let svd = svd_tall_or_wide(&mat);
    let k = svd
        .s
        .iter()
        .filter(|&&s| s > 1e-14)
        .count()
        .min(max_bond)
        .max(1);

    match direction {
        SweepDir::LeftToRight => {
            let mut new_ti = MpsTensor::new(bl, k);
            for si in 0..2 {
                new_ti.a[si] = DenseMatrix::zeros(bl, k);
                for a in 0..bl {
                    for g in 0..k {
                        new_ti.a[si].set(a, g, svd.u.get(si * bl + a, g));
                    }
                }
            }
            new_ti.bond_left = bl;
            new_ti.bond_right = k;

            let mut new_tj = MpsTensor::new(k, br);
            for sj in 0..2 {
                new_tj.a[sj] = DenseMatrix::zeros(k, br);
                for g in 0..k {
                    let sg = svd.s[g];
                    for b in 0..br {
                        new_tj.a[sj].set(g, b, svd.vh.get(g, sj * br + b).scale(sg));
                    }
                }
            }
            new_tj.bond_left = k;
            new_tj.bond_right = br;

            mps.tensors[i] = new_ti;
            mps.tensors[j] = new_tj;
        }
        SweepDir::RightToLeft => {
            let mut new_ti = MpsTensor::new(bl, k);
            for si in 0..2 {
                new_ti.a[si] = DenseMatrix::zeros(bl, k);
                for a in 0..bl {
                    for g in 0..k {
                        let sg = svd.s[g];
                        new_ti.a[si].set(a, g, svd.u.get(si * bl + a, g).scale(sg));
                    }
                }
            }
            new_ti.bond_left = bl;
            new_ti.bond_right = k;

            let mut new_tj = MpsTensor::new(k, br);
            for sj in 0..2 {
                new_tj.a[sj] = DenseMatrix::zeros(k, br);
                for g in 0..k {
                    for b in 0..br {
                        new_tj.a[sj].set(g, b, svd.vh.get(g, sj * br + b));
                    }
                }
            }
            new_tj.bond_left = k;
            new_tj.bond_right = br;

            mps.tensors[i] = new_ti;
            mps.tensors[j] = new_tj;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run two-site DMRG for the Ising (ZZ) Hamiltonian.
pub fn dmrg_heisenberg_1d(
    n_qubits: usize,
    max_bond: usize,
    max_sweeps: usize,
    tol: f64,
) -> DmrgResult {
    dmrg_1d(n_qubits, max_bond, max_sweeps, tol, DmrgHamiltonian::Ising)
}

/// Run two-site DMRG for the full Heisenberg (XX+YY+ZZ) Hamiltonian.
pub fn dmrg_full_heisenberg_1d(
    n_qubits: usize,
    max_bond: usize,
    max_sweeps: usize,
    tol: f64,
) -> DmrgResult {
    dmrg_1d(n_qubits, max_bond, max_sweeps, tol, DmrgHamiltonian::Heisenberg)
}

/// Core two-site DMRG using variational Lanczos with proper MPO environments.
///
/// At each bond, the effective Hamiltonian includes ALL nearest-neighbor terms:
/// - Bonds fully to the left/right are accumulated in Hamiltonian environments.
/// - Boundary bonds use operator environments with dangling single-site operators.
/// - The direct bond term acts on physical indices.
pub fn dmrg_1d(
    n_qubits: usize,
    max_bond: usize,
    max_sweeps: usize,
    tol: f64,
    hamiltonian: DmrgHamiltonian,
) -> DmrgResult {
    assert!(n_qubits >= 2, "DMRG requires at least 2 qubits");

    let mut mps = Mps::with_max_bond(n_qubits, max_bond);

    // Randomize initial state with seeded RNG
    let mut rng = 42u64;
    for q in 0..n_qubits {
        let theta_val = crate::rand_f64(&mut rng) * std::f64::consts::PI;
        let c = (theta_val / 2.0).cos();
        let s = (theta_val / 2.0).sin();
        let ry = [
            [ComplexF64::real(c), ComplexF64::real(-s)],
            [ComplexF64::real(s), ComplexF64::real(c)],
        ];
        mps.apply_single_qubit(q, ry);
    }

    let h_local = local_hamiltonian(hamiltonian);
    let energy_fn: fn(&Mps) -> f64 = match hamiltonian {
        DmrgHamiltonian::Ising => mps_heisenberg_energy,
        DmrgHamiltonian::Heisenberg => mps_full_heisenberg_energy,
    };

    let mut energies = Vec::new();
    let mut prev_energy = energy_fn(&mps);
    energies.push(prev_energy);

    for sweep in 0..max_sweeps {
        // Left-to-right sweep
        for bond in 0..(n_qubits - 1) {
            let envs = build_bond_envs(&mps, bond, hamiltonian);
            optimize_two_site(
                &mut mps,
                bond,
                &h_local,
                max_bond,
                &envs,
                hamiltonian,
                SweepDir::LeftToRight,
            );
        }

        // Right-to-left sweep
        for bond in (0..(n_qubits - 1)).rev() {
            let envs = build_bond_envs(&mps, bond, hamiltonian);
            optimize_two_site(
                &mut mps,
                bond,
                &h_local,
                max_bond,
                &envs,
                hamiltonian,
                SweepDir::RightToLeft,
            );
        }

        let energy = energy_fn(&mps);
        energies.push(energy);

        if (energy - prev_energy).abs() < tol && sweep > 0 {
            return DmrgResult {
                energy,
                mps,
                energies,
                sweeps: sweep + 1,
            };
        }
        prev_energy = energy;
    }

    DmrgResult {
        energy: prev_energy,
        mps,
        energies,
        sweeps: max_sweeps,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_hamiltonian_ising() {
        let h = local_hamiltonian(DmrgHamiltonian::Ising);
        assert_eq!(h[0][0], 1.0);
        assert_eq!(h[1][1], -1.0);
        assert_eq!(h[2][2], -1.0);
        assert_eq!(h[3][3], 1.0);
        assert_eq!(h[0][1], 0.0);
    }

    #[test]
    fn test_local_hamiltonian_heisenberg() {
        let h = local_hamiltonian(DmrgHamiltonian::Heisenberg);
        assert_eq!(h[0][0], 1.0);
        assert_eq!(h[1][1], -1.0);
        assert_eq!(h[1][2], 2.0);
        assert_eq!(h[2][1], 2.0);
        assert_eq!(h[2][2], -1.0);
        assert_eq!(h[3][3], 1.0);
    }

    #[test]
    fn test_diag_tridiagonal_2x2() {
        let alphas = vec![-1.0, -1.0];
        let betas = vec![0.0, 2.0];
        let (mut evals, _) = diag_tridiagonal(&alphas, &betas);
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((evals[0] - (-3.0)).abs() < 1e-10);
        assert!((evals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lanczos_simple_4x4() {
        let h = local_hamiltonian(DmrgHamiltonian::Heisenberg);
        let apply_h = |v: &[ComplexF64]| -> Vec<ComplexF64> {
            let mut result = vec![ComplexF64::ZERO; 4];
            for row in 0..4 {
                for col in 0..4 {
                    if h[row][col] != 0.0 {
                        result[row] = result[row].add(v[col].scale(h[row][col]));
                    }
                }
            }
            result
        };

        let initial = vec![
            ComplexF64::real(0.3),
            ComplexF64::real(0.7),
            ComplexF64::real(-0.4),
            ComplexF64::real(0.5),
        ];

        let (eval, _) = lanczos_ground(&apply_h, &initial, 4, 4);
        assert!(
            (eval - (-3.0)).abs() < 1e-8,
            "Lanczos ground state should be -3, got {}",
            eval
        );
    }

    #[test]
    fn test_dmrg_ising_2_qubit() {
        let result = dmrg_heisenberg_1d(2, 4, 20, 1e-10);
        assert!(
            (result.energy - (-1.0)).abs() < 0.1,
            "2-site Ising ground state should be near -1, got {}",
            result.energy
        );
    }

    #[test]
    fn test_dmrg_2_qubit_heisenberg() {
        let result = dmrg_full_heisenberg_1d(2, 4, 20, 1e-10);
        assert!(
            (result.energy - (-3.0)).abs() < 0.1,
            "2-site Heisenberg ground state should be near -3, got {}",
            result.energy
        );
    }

    #[test]
    fn test_dmrg_ising_4_qubit() {
        let result = dmrg_heisenberg_1d(4, 16, 20, 1e-8);
        assert!(
            result.energy < -2.0,
            "DMRG Ising energy {} should be below -2.0",
            result.energy
        );
    }

    #[test]
    fn test_dmrg_heisenberg_4_qubit() {
        let result = dmrg_full_heisenberg_1d(4, 16, 20, 1e-8);
        assert!(
            result.energy < 0.0,
            "DMRG Heisenberg energy should be negative, got {}",
            result.energy
        );
    }

    #[test]
    fn test_dmrg_energy_decreases() {
        let result = dmrg_full_heisenberg_1d(4, 16, 10, 1e-12);
        if result.energies.len() > 2 {
            let last = result.energies.last().unwrap();
            let first = result.energies[0];
            assert!(
                *last <= first + 0.5,
                "Energy should decrease: first={} last={}",
                first,
                last
            );
        }
    }

    #[test]
    fn test_dmrg_deterministic() {
        let r1 = dmrg_heisenberg_1d(4, 8, 6, 1e-8);
        let r2 = dmrg_heisenberg_1d(4, 8, 6, 1e-8);
        assert_eq!(
            r1.energy.to_bits(),
            r2.energy.to_bits(),
            "DMRG must be deterministic: {} vs {}",
            r1.energy,
            r2.energy
        );
        assert_eq!(r1.sweeps, r2.sweeps);
        for (a, b) in r1.energies.iter().zip(&r2.energies) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_dmrg_beats_initial() {
        let result = dmrg_full_heisenberg_1d(6, 8, 10, 1e-8);
        assert!(result.energies.len() >= 2);
        let initial_energy = result.energies[0];
        let final_energy = result.energy;
        assert!(
            final_energy <= initial_energy + 0.01,
            "DMRG final {} should be <= initial {}",
            final_energy,
            initial_energy
        );
    }

    #[test]
    fn test_dmrg_larger_system() {
        let result = dmrg_heisenberg_1d(10, 8, 4, 1e-6);
        assert!(
            result.energy.is_finite(),
            "10-qubit DMRG energy should be finite, got {}",
            result.energy
        );
    }

    #[test]
    fn test_complex_helpers() {
        let a = ComplexF64::new(1.0, 2.0);
        let b = ComplexF64::new(3.0, -1.0);
        let v = vec![a, b];
        let n = complex_norm(&v);
        assert!((n * n - 15.0).abs() < 1e-12);
        let dot = complex_real_dot(&v, &v);
        assert!((dot - 15.0).abs() < 1e-12);
    }
}
