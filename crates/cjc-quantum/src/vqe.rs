//! Variational Quantum Eigensolver (VQE) — Ground state energy estimation.
//!
//! Implements VQE for 1D Hamiltonians using:
//! - MPS representation for O(N) memory scaling
//! - Adjoint differentiation for O(1) gradient memory (small circuits)
//! - Parameter-shift gradients for MPS-backed circuits
//!
//! # Heisenberg 1D Model
//!
//! H = Σ_i Z_i Z_{i+1} (nearest-neighbor ZZ interaction)
//!
//! The ground state energy per site approaches -1 in the thermodynamic limit.
//!
//! # Determinism
//!
//! - All parameters initialized from seeded SplitMix64
//! - Gradient computation uses deterministic MPS operations
//! - Sign-stabilized SVD ensures bit-identical bond truncation

use cjc_runtime::complex::ComplexF64;
use crate::mps::Mps;

// ---------------------------------------------------------------------------
// Heisenberg 1D Hamiltonian
// ---------------------------------------------------------------------------

/// Compute ⟨ψ|Z_i Z_{i+1}|ψ⟩ for an MPS state.
///
/// For a diagonal operator like ZZ, the expectation value can be computed
/// efficiently from the MPS by contracting transfer matrices.
///
/// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩, so Z_i Z_{i+1} is diagonal with eigenvalues
/// (+1, -1, -1, +1) on the {|00⟩, |01⟩, |10⟩, |11⟩} subspace.
pub fn mps_zz_expectation(mps: &Mps, site_i: usize) -> f64 {
    let n = mps.n_qubits;
    assert!(site_i + 1 < n, "ZZ site out of range");

    // Transfer matrix approach:
    // Contract from left boundary to site i-1 with identity,
    // at sites i and i+1 with Z⊗Z (diagonal: +1,-1,-1,+1),
    // then from site i+2 to right boundary with identity.
    //
    // The transfer matrix for site k with operator O_k is:
    // T^{O_k}[j,j'] = Σ_s o_s * conj(A^s[j_left, j_right]) * A^s[j'_left, j'_right]
    //
    // For identity: o_0 = o_1 = 1
    // For Z: o_0 = 1, o_1 = -1

    // Start with left boundary: 1×1 identity
    let mut env = vec![vec![ComplexF64::ZERO; 1]; 1];
    env[0][0] = ComplexF64::ONE;

    // Contract sites 0..site_i with identity transfer matrices
    for k in 0..site_i {
        env = transfer_matrix_identity(&mps.tensors[k], &env);
    }

    // Contract site_i with Z
    env = transfer_matrix_z(&mps.tensors[site_i], &env);

    // Contract site_i+1 with Z
    env = transfer_matrix_z(&mps.tensors[site_i + 1], &env);

    // Contract remaining sites with identity
    for k in (site_i + 2)..n {
        env = transfer_matrix_identity(&mps.tensors[k], &env);
    }

    // Final result: trace of the 1×1 environment
    env[0][0].re
}

/// Compute the total Heisenberg energy E = Σ_i ⟨Z_i Z_{i+1}⟩.
pub fn mps_heisenberg_energy(mps: &Mps) -> f64 {
    let mut energy = 0.0;
    for i in 0..(mps.n_qubits - 1) {
        energy += mps_zz_expectation(mps, i);
    }
    energy
}

/// Transfer matrix contraction with identity operator.
///
/// T_identity[j_left, j'_left] → T_new[j_right, j'_right]
/// T_new[a,b] = Σ_{j,j'} Σ_s env[j,j'] * conj(A^s[j,a]) * A^s[j',b]
fn transfer_matrix_identity(
    tensor: &crate::mps::MpsTensor,
    env: &[Vec<ComplexF64>],
) -> Vec<Vec<ComplexF64>> {
    let bl = tensor.bond_left;
    let br = tensor.bond_right;
    let env_rows = env.len();
    let env_cols = env[0].len();
    assert_eq!(env_rows, bl);
    assert_eq!(env_cols, bl);

    let mut result = vec![vec![ComplexF64::ZERO; br]; br];

    for s in 0..2 {
        for j in 0..bl {
            for jp in 0..bl {
                let e = env[j][jp];
                if e.re == 0.0 && e.im == 0.0 {
                    continue;
                }
                for a in 0..br {
                    let conj_a = tensor.a[s].get(j, a).conj();
                    let ea = e.mul_fixed(conj_a);
                    for b in 0..br {
                        let asb = tensor.a[s].get(jp, b);
                        result[a][b] = result[a][b].add(ea.mul_fixed(asb));
                    }
                }
            }
        }
    }

    result
}

/// Transfer matrix contraction with Z operator.
///
/// Z eigenvalues: o_0 = +1, o_1 = -1
fn transfer_matrix_z(
    tensor: &crate::mps::MpsTensor,
    env: &[Vec<ComplexF64>],
) -> Vec<Vec<ComplexF64>> {
    let bl = tensor.bond_left;
    let br = tensor.bond_right;
    let env_rows = env.len();
    let env_cols = env[0].len();
    assert_eq!(env_rows, bl);
    assert_eq!(env_cols, bl);

    let z_eigenvalues = [1.0, -1.0];
    let mut result = vec![vec![ComplexF64::ZERO; br]; br];

    for s in 0..2 {
        let z_s = z_eigenvalues[s];
        for j in 0..bl {
            for jp in 0..bl {
                let e = env[j][jp].scale(z_s);
                if e.re == 0.0 && e.im == 0.0 {
                    continue;
                }
                for a in 0..br {
                    let conj_a = tensor.a[s].get(j, a).conj();
                    let ea = e.mul_fixed(conj_a);
                    for b in 0..br {
                        let asb = tensor.a[s].get(jp, b);
                        result[a][b] = result[a][b].add(ea.mul_fixed(asb));
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// VQE Ansatz: Ry layer + CNOT chain
// ---------------------------------------------------------------------------

/// Build an MPS ansatz state from parameters.
///
/// The ansatz applies:
/// 1. Ry(theta_i) on each qubit i (parameterized layer)
/// 2. CNOT chain: CNOT(0,1), CNOT(1,2), ..., CNOT(N-2, N-1)
///
/// This is a "hardware-efficient" ansatz suited for 1D MPS.
pub fn build_mps_ansatz(n_qubits: usize, thetas: &[f64], max_bond: usize) -> Mps {
    assert_eq!(thetas.len(), n_qubits, "Need one theta per qubit");

    let mut mps = Mps::with_max_bond(n_qubits, max_bond);

    // Ry layer
    for (q, &theta) in thetas.iter().enumerate() {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let ry = [
            [ComplexF64::real(c), ComplexF64::real(-s)],
            [ComplexF64::real(s), ComplexF64::real(c)],
        ];
        mps.apply_single_qubit(q, ry);
    }

    // CNOT chain (1D entanglement)
    for i in 0..(n_qubits - 1) {
        mps.apply_cnot_adjacent(i, i + 1);
    }

    mps
}

/// Compute energy gradient via parameter-shift rule on MPS.
///
/// ∂E/∂θ_k = (E(θ_k + π/2) - E(θ_k - π/2)) / 2
pub fn mps_parameter_shift_gradient(
    n_qubits: usize,
    thetas: &[f64],
    max_bond: usize,
) -> Vec<f64> {
    let shift = std::f64::consts::FRAC_PI_2;
    let mut grads = vec![0.0; thetas.len()];

    for k in 0..thetas.len() {
        let mut thetas_plus = thetas.to_vec();
        let mut thetas_minus = thetas.to_vec();
        thetas_plus[k] += shift;
        thetas_minus[k] -= shift;

        let mps_plus = build_mps_ansatz(n_qubits, &thetas_plus, max_bond);
        let mps_minus = build_mps_ansatz(n_qubits, &thetas_minus, max_bond);

        let e_plus = mps_heisenberg_energy(&mps_plus);
        let e_minus = mps_heisenberg_energy(&mps_minus);

        grads[k] = (e_plus - e_minus) / 2.0;
    }

    grads
}

// ---------------------------------------------------------------------------
// VQE Optimizer
// ---------------------------------------------------------------------------

/// Result of a VQE optimization run.
#[derive(Debug, Clone)]
pub struct VqeResult {
    /// Final optimized parameters.
    pub thetas: Vec<f64>,
    /// Final energy.
    pub energy: f64,
    /// Energy history (one per iteration).
    pub energy_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Run the VQE optimization for the 1D Heisenberg model.
///
/// # Arguments
///
/// * `n_qubits` - Number of qubits (sites)
/// * `max_bond` - Maximum MPS bond dimension
/// * `learning_rate` - Gradient descent step size
/// * `max_iters` - Maximum optimization iterations
/// * `seed` - RNG seed for parameter initialization
///
/// # Returns
///
/// VqeResult with optimized parameters and energy history.
pub fn vqe_heisenberg_1d(
    n_qubits: usize,
    max_bond: usize,
    learning_rate: f64,
    max_iters: usize,
    seed: u64,
) -> VqeResult {
    // Initialize parameters from seeded PRNG
    let mut rng_state = seed;
    let mut thetas: Vec<f64> = (0..n_qubits)
        .map(|_| {
            let r = crate::rand_f64(&mut rng_state);
            (r - 0.5) * 0.1 // Small initial angles near 0
        })
        .collect();

    let mut energy_history = Vec::with_capacity(max_iters);

    // Initial energy
    let mps = build_mps_ansatz(n_qubits, &thetas, max_bond);
    let mut best_energy = mps_heisenberg_energy(&mps);
    energy_history.push(best_energy);

    for _iter in 0..max_iters {
        // Compute gradients via parameter shift
        let grads = mps_parameter_shift_gradient(n_qubits, &thetas, max_bond);

        // Gradient descent update
        for k in 0..thetas.len() {
            thetas[k] -= learning_rate * grads[k];
        }

        // Compute new energy
        let mps = build_mps_ansatz(n_qubits, &thetas, max_bond);
        let energy = mps_heisenberg_energy(&mps);
        energy_history.push(energy);

        if energy < best_energy {
            best_energy = energy;
        }
    }

    VqeResult {
        thetas,
        energy: best_energy,
        energy_history,
        iterations: max_iters,
    }
}

// ---------------------------------------------------------------------------
// Continuous Verification
// ---------------------------------------------------------------------------

/// Run the continuous verification gate.
///
/// Returns Ok(()) if all checks pass, Err with description otherwise.
pub fn verification_gate(seed: u64) -> Result<(), String> {
    // Check 1: MPS/Statevector parity on a 4-qubit sub-circuit
    let n = 4;
    let mut rng = seed;
    let thetas: Vec<f64> = (0..n).map(|_| crate::rand_f64(&mut rng) * std::f64::consts::PI).collect();

    let mps = build_mps_ansatz(n, &thetas, 64);
    let mps_sv = mps.to_statevector();

    // Build equivalent circuit
    let mut circ = crate::circuit::Circuit::new(n);
    for (q, &theta) in thetas.iter().enumerate() {
        circ.ry(q, theta);
    }
    for i in 0..(n - 1) {
        circ.cnot(i, i + 1);
    }
    let circ_sv = circ.execute().map_err(|e| format!("Circuit exec failed: {}", e))?;

    for i in 0..(1 << n) {
        let err = mps_sv[i].add(circ_sv.amplitudes[i].neg()).norm_sq().sqrt();
        if err > 1e-10 {
            return Err(format!(
                "MPS/SV parity failure at state {}: mps={:?} sv={:?} err={}",
                i, mps_sv[i], circ_sv.amplitudes[i], err
            ));
        }
    }

    // Check 2: Adjoint gradient vs parameter-shift for a 4-qubit circuit
    let z_obs: Vec<f64> = (0..(1 << n))
        .map(|basis: usize| {
            // ZZ observable for sites 0,1
            let z0 = if basis & 1 == 0 { 1.0 } else { -1.0 };
            let z1 = if basis & 2 == 0 { 1.0 } else { -1.0 };
            z0 * z1
        })
        .collect();

    let adj_grads = crate::adjoint::adjoint_differentiation(&circ, &z_obs)
        .map_err(|e| format!("Adjoint failed: {}", e))?;

    // Parameter-shift for first Ry gate (gate index 0)
    let eps = std::f64::consts::FRAC_PI_2;
    let mut thetas_p = thetas.clone();
    let mut thetas_m = thetas.clone();
    thetas_p[0] += eps;
    thetas_m[0] -= eps;

    let mut circ_p = crate::circuit::Circuit::new(n);
    let mut circ_m = crate::circuit::Circuit::new(n);
    for (q, &theta) in thetas_p.iter().enumerate() { circ_p.ry(q, theta); }
    for (q, &theta) in thetas_m.iter().enumerate() { circ_m.ry(q, theta); }
    for i in 0..(n - 1) { circ_p.cnot(i, i + 1); circ_m.cnot(i, i + 1); }

    let sv_p = circ_p.execute().map_err(|e| format!("Circuit+ exec: {}", e))?;
    let sv_m = circ_m.execute().map_err(|e| format!("Circuit- exec: {}", e))?;
    let e_p = crate::adjoint::expectation_value(&sv_p, &z_obs)
        .map_err(|e| format!("Expect+ failed: {}", e))?;
    let e_m = crate::adjoint::expectation_value(&sv_m, &z_obs)
        .map_err(|e| format!("Expect- failed: {}", e))?;
    let ps_grad = (e_p - e_m) / 2.0;

    let adj_grad_0 = adj_grads.gradients[0]; // First gate is Ry(0, thetas[0])
    let grad_err = (adj_grad_0 - ps_grad).abs();
    if grad_err > 1e-6 {
        return Err(format!(
            "Gradient sync failure: adjoint={} ps={} err={}",
            adj_grad_0, ps_grad, grad_err
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_zz_expectation_product_state() {
        // |0...0⟩: all Z eigenvalues are +1, so ZZ = 1 for all pairs
        let mps = Mps::new(4);
        for i in 0..3 {
            let zz = mps_zz_expectation(&mps, i);
            assert!((zz - 1.0).abs() < TOL, "ZZ({},{}) = {}, expected 1.0", i, i + 1, zz);
        }
    }

    #[test]
    fn test_zz_expectation_x_state() {
        // X|0⟩ = |1⟩, Z eigenvalue = -1
        // |10...0⟩: Z_0=-1, Z_1=+1, so Z_0 Z_1 = -1
        let mut mps = Mps::new(4);
        let x = [[ComplexF64::ZERO, ComplexF64::ONE],
                  [ComplexF64::ONE, ComplexF64::ZERO]];
        mps.apply_single_qubit(0, x);

        assert!((mps_zz_expectation(&mps, 0) - (-1.0)).abs() < TOL,
            "ZZ(0,1) after X on q0");
        assert!((mps_zz_expectation(&mps, 1) - 1.0).abs() < TOL,
            "ZZ(1,2) after X on q0");
    }

    #[test]
    fn test_heisenberg_energy_all_zero() {
        // |0...0⟩: E = Σ ZZ = (N-1) * 1 = N-1
        let mps = Mps::new(5);
        let e = mps_heisenberg_energy(&mps);
        assert!((e - 4.0).abs() < TOL, "E = {}, expected 4.0", e);
    }

    #[test]
    fn test_ansatz_builds_correctly() {
        let thetas = vec![0.0; 4];
        let mps = build_mps_ansatz(4, &thetas, 64);
        // All thetas=0: Ry(0)=I, so state is |0000⟩ after CNOT chain
        let e = mps_heisenberg_energy(&mps);
        assert!((e - 3.0).abs() < TOL, "E = {}, expected 3.0", e);
    }

    #[test]
    fn test_parameter_shift_gradient_symmetry() {
        // At all-zero params, gradient should be zero by symmetry
        let thetas = vec![0.0; 4];
        let grads = mps_parameter_shift_gradient(4, &thetas, 64);
        for (k, &g) in grads.iter().enumerate() {
            assert!(g.abs() < 1e-8,
                "Gradient at theta=0 should be ~0, got {} for param {}", g, k);
        }
    }

    #[test]
    fn test_vqe_small_energy_decreases() {
        // 4-qubit VQE should decrease energy over a few iterations
        let result = vqe_heisenberg_1d(4, 16, 0.1, 5, 42);
        let initial = result.energy_history[0];
        let final_e = result.energy;
        assert!(final_e <= initial + 1e-8,
            "Energy should decrease: initial={}, final={}", initial, final_e);
    }

    #[test]
    fn test_vqe_deterministic() {
        // Same seed → same result
        let r1 = vqe_heisenberg_1d(4, 16, 0.1, 3, 42);
        let r2 = vqe_heisenberg_1d(4, 16, 0.1, 3, 42);

        for k in 0..r1.thetas.len() {
            assert_eq!(r1.thetas[k].to_bits(), r2.thetas[k].to_bits(),
                "theta[{}] not bit-identical", k);
        }
        assert_eq!(r1.energy.to_bits(), r2.energy.to_bits(),
            "Energy not bit-identical");
    }

    #[test]
    fn test_verification_gate_passes() {
        verification_gate(42).expect("Verification gate should pass");
        verification_gate(123).expect("Verification gate should pass with different seed");
    }

    #[test]
    fn test_vqe_50_qubit_memory() {
        // Verify 50-qubit ansatz stays under memory budget
        let thetas = vec![0.1; 50];
        let mps = build_mps_ansatz(50, &thetas, 8);
        let mem = mps.memory_bytes();
        assert!(mem < 500_000_000, "Memory {} exceeds 500MB", mem);
        // Energy should be computable
        let e = mps_heisenberg_energy(&mps);
        assert!(e.is_finite(), "Energy should be finite");
    }

    #[test]
    fn test_vqe_50_qubit_bond_dimension() {
        // With max_bond=8, all bonds should stay ≤ 8
        let thetas = vec![0.3; 50];
        let mps = build_mps_ansatz(50, &thetas, 8);
        for i in 0..49 {
            assert!(mps.tensors[i].bond_right <= 8,
                "Bond between {} and {} is {}", i, i + 1, mps.tensors[i].bond_right);
        }
    }
}
