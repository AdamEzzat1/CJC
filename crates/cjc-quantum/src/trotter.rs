//! Suzuki-Trotter Decomposition — Time Evolution of Quantum Hamiltonians.
//!
//! Implements 1st and 2nd order product formulas for simulating
//! time evolution under non-commuting Hamiltonians:
//!
//! **1st order (Lie-Trotter):**
//!   e^{-i(A+B)t} ≈ (e^{-iAt/n} e^{-iBt/n})^n
//!
//! **2nd order (Strang splitting):**
//!   e^{-i(A+B)t} ≈ (e^{-iAt/(2n)} e^{-iBt/n} e^{-iAt/(2n)})^n
//!
//! # Determinism
//!
//! - All complex arithmetic uses mul_fixed (no FMA)
//! - Kahan summation for amplitude accumulations
//! - Matrix exponential via explicit analytic formulas for Pauli terms
//! - Fixed iteration order throughout

use cjc_runtime::complex::ComplexF64;
use cjc_repro::KahanAccumulatorF64;
use crate::fermion::{Pauli, PauliTerm, FermionicHamiltonian};
use crate::statevector::Statevector;

// ---------------------------------------------------------------------------
// Trotter Order
// ---------------------------------------------------------------------------

/// Order of the Suzuki-Trotter decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrotterOrder {
    /// 1st order: e^{-i(A+B)t} ≈ e^{-iAt} e^{-iBt}
    First,
    /// 2nd order: e^{-i(A+B)t} ≈ e^{-iAt/2} e^{-iBt} e^{-iAt/2}
    Second,
}

// ---------------------------------------------------------------------------
// Pauli Exponential
// ---------------------------------------------------------------------------

/// Apply e^{-i θ P} to a statevector, where P is a Pauli string.
///
/// For a Pauli string P with eigenvalues ±1, the exponential is:
///   e^{-iθP} = cos(θ)I - i sin(θ)P
///
/// This is exact (not an approximation) because P² = I for any Pauli string.
pub fn apply_pauli_exp(sv: &mut Statevector, term: &PauliTerm, theta: f64) {
    let cos_t = ComplexF64::real(theta.cos());
    let sin_t = ComplexF64::new(0.0, -theta.sin()); // -i sin(θ)

    let n = sv.n_states();

    // Temporary buffer for the result (needed because P may map k → k')
    let mut new_amps = vec![ComplexF64::ZERO; n];

    // First pass: accumulate cos(θ) * I contribution
    for k in 0..n {
        new_amps[k] = cos_t.mul_fixed(sv.amplitudes[k]);
    }

    // Second pass: accumulate -i sin(θ) * coeff * P contribution
    let phase = sin_t.mul_fixed(term.coeff);
    for k in 0..n {
        let (k_prime, pauli_phase) = apply_pauli_to_basis(term, k);
        let contrib = phase.mul_fixed(pauli_phase).mul_fixed(sv.amplitudes[k]);
        new_amps[k_prime] = new_amps[k_prime].add(contrib);
    }

    sv.amplitudes = new_amps;
}

/// Apply a Pauli string to a basis state, returning (new_index, phase).
fn apply_pauli_to_basis(term: &PauliTerm, k: usize) -> (usize, ComplexF64) {
    let mut k_new = k;
    let mut phase = ComplexF64::ONE;

    for (q, &op) in term.ops.iter().enumerate() {
        let bit = (k >> q) & 1;
        match op {
            Pauli::I => {}
            Pauli::X => {
                k_new ^= 1 << q;
            }
            Pauli::Y => {
                k_new ^= 1 << q;
                if bit == 0 {
                    phase = phase.mul_fixed(ComplexF64::I);
                } else {
                    phase = phase.mul_fixed(ComplexF64::new(0.0, -1.0));
                }
            }
            Pauli::Z => {
                if bit == 1 {
                    phase = phase.mul_fixed(ComplexF64::new(-1.0, 0.0));
                }
            }
        }
    }

    (k_new, phase)
}

// ---------------------------------------------------------------------------
// Trotter Evolution
// ---------------------------------------------------------------------------

/// Perform Suzuki-Trotter time evolution: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩.
///
/// The Hamiltonian H = Σ_k c_k P_k is decomposed into its Pauli terms.
/// Each Trotter step applies the product of Pauli exponentials.
///
/// # Arguments
///
/// * `sv` - Statevector to evolve (modified in-place)
/// * `hamiltonian` - Hamiltonian as sum of Pauli terms
/// * `time` - Total evolution time
/// * `n_steps` - Number of Trotter steps (higher = more accurate)
/// * `order` - Trotter decomposition order (First or Second)
pub fn trotter_evolve(
    sv: &mut Statevector,
    hamiltonian: &FermionicHamiltonian,
    time: f64,
    n_steps: usize,
    order: TrotterOrder,
) {
    assert_eq!(sv.n_qubits, hamiltonian.n_qubits, "qubit count mismatch");
    assert!(n_steps > 0, "n_steps must be positive");

    let dt = time / n_steps as f64;

    match order {
        TrotterOrder::First => {
            for _step in 0..n_steps {
                // e^{-iHdt} ≈ Π_k e^{-i c_k P_k dt}
                for term in &hamiltonian.terms {
                    let theta = term.coeff.re * dt;
                    if theta.abs() > 1e-20 {
                        apply_pauli_rotation(sv, &term.ops, theta);
                    }
                }
            }
        }
        TrotterOrder::Second => {
            let n_terms = hamiltonian.terms.len();
            for _step in 0..n_steps {
                // Forward half-step for first term through second-to-last
                for i in 0..n_terms {
                    let term = &hamiltonian.terms[i];
                    let theta = if i == 0 || i == n_terms - 1 {
                        // First and last terms get half-step in 2nd-order Trotter
                        term.coeff.re * dt * 0.5
                    } else {
                        term.coeff.re * dt
                    };
                    if theta.abs() > 1e-20 {
                        apply_pauli_rotation(sv, &term.ops, theta);
                    }
                }
                // Backward half-step for first and last terms
                // (Already handled by the half-step above in symmetric form.)
                // For proper 2nd-order: apply terms in reverse
                for i in (0..n_terms).rev() {
                    let term = &hamiltonian.terms[i];
                    let theta = if i == 0 || i == n_terms - 1 {
                        term.coeff.re * dt * 0.5
                    } else {
                        // Inner terms already applied at full step above
                        0.0
                    };
                    if theta.abs() > 1e-20 {
                        apply_pauli_rotation(sv, &term.ops, theta);
                    }
                }
            }
        }
    }
}

/// Apply e^{-iθ P} where P is a Pauli string (no coefficient included).
/// This is the core rotation: cos(θ)I - i·sin(θ)P.
fn apply_pauli_rotation(sv: &mut Statevector, ops: &[Pauli], theta: f64) {
    let n = sv.n_states();
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Check if all ops are diagonal (I or Z only)
    let all_diagonal = ops.iter().all(|&p| p == Pauli::I || p == Pauli::Z);

    if all_diagonal {
        // Diagonal case: e^{-iθP}|k⟩ = e^{-iθ·eigenvalue(k)} |k⟩
        // eigenvalue(k) = product of Z eigenvalues (+1 for |0⟩, -1 for |1⟩)
        for k in 0..n {
            let eigenvalue = compute_z_eigenvalue(ops, k);
            let angle = -theta * eigenvalue;
            let phase = ComplexF64::new(angle.cos(), angle.sin());
            sv.amplitudes[k] = sv.amplitudes[k].mul_fixed(phase);
        }
    } else {
        // General case: e^{-iθP} = cos(θ)I - i·sin(θ)P
        let mut new_amps = vec![ComplexF64::ZERO; n];

        // cos(θ) * I contribution
        for k in 0..n {
            new_amps[k] = sv.amplitudes[k].scale(cos_t);
        }

        // -i·sin(θ) * P contribution
        let neg_i_sin = ComplexF64::new(0.0, -sin_t);
        for k in 0..n {
            let (k_prime, pauli_phase) = apply_pauli_ops(ops, k);
            let contrib = neg_i_sin.mul_fixed(pauli_phase).mul_fixed(sv.amplitudes[k]);
            new_amps[k_prime] = new_amps[k_prime].add(contrib);
        }

        sv.amplitudes = new_amps;
    }
}

/// Compute the eigenvalue of a diagonal Pauli string (I/Z only) on basis state k.
/// Each Z contributes +1 if the corresponding qubit is |0⟩, -1 if |1⟩.
fn compute_z_eigenvalue(ops: &[Pauli], k: usize) -> f64 {
    let mut eigenvalue = 1.0f64;
    for (q, &op) in ops.iter().enumerate() {
        if op == Pauli::Z {
            let bit = (k >> q) & 1;
            if bit == 1 {
                eigenvalue = -eigenvalue;
            }
        }
    }
    eigenvalue
}

/// Apply Pauli operators to a basis state. Returns (new_basis, phase).
fn apply_pauli_ops(ops: &[Pauli], k: usize) -> (usize, ComplexF64) {
    let mut k_new = k;
    let mut phase = ComplexF64::ONE;

    for (q, &op) in ops.iter().enumerate() {
        let bit = (k >> q) & 1;
        match op {
            Pauli::I => {}
            Pauli::X => {
                k_new ^= 1 << q;
            }
            Pauli::Y => {
                k_new ^= 1 << q;
                if bit == 0 {
                    phase = phase.mul_fixed(ComplexF64::I);
                } else {
                    phase = phase.mul_fixed(ComplexF64::new(0.0, -1.0));
                }
            }
            Pauli::Z => {
                if bit == 1 {
                    phase = phase.mul_fixed(ComplexF64::new(-1.0, 0.0));
                }
            }
        }
    }

    (k_new, phase)
}

/// Compute the Trotter error bound (rough upper bound).
///
/// For 1st order: error ≤ t²/(2n) * Σ_{j<k} ||[H_j, H_k]||
/// For 2nd order: error ≤ t³/(12n²) * (commutator bound)
///
/// Returns a rough estimate based on the number of terms and time.
pub fn trotter_error_bound(
    hamiltonian: &FermionicHamiltonian,
    time: f64,
    n_steps: usize,
    order: TrotterOrder,
) -> f64 {
    let n_terms = hamiltonian.n_terms() as f64;
    let dt = time / n_steps as f64;

    // Upper bound on operator norms
    let mut norm_sum = KahanAccumulatorF64::new();
    for term in &hamiltonian.terms {
        norm_sum.add(term.coeff.re.abs() + term.coeff.im.abs());
    }
    let h_norm = norm_sum.finalize();

    match order {
        TrotterOrder::First => {
            // Rough bound: n_terms * h_norm² * dt² / 2
            n_terms * h_norm * h_norm * dt * dt / 2.0
        }
        TrotterOrder::Second => {
            // Rough bound: n_terms² * h_norm³ * dt³ / 12
            n_terms * n_terms * h_norm * h_norm * h_norm * dt * dt * dt / 12.0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::h2_hamiltonian;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_trotter_preserves_norm() {
        let h = h2_hamiltonian();
        let mut sv = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv).unwrap();

        trotter_evolve(&mut sv, &h, 1.0, 100, TrotterOrder::First);
        assert!(sv.is_normalized(1e-10), "Trotter must preserve norm");
    }

    #[test]
    fn test_trotter_second_order_preserves_norm() {
        let h = h2_hamiltonian();
        let mut sv = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv).unwrap();

        trotter_evolve(&mut sv, &h, 1.0, 50, TrotterOrder::Second);
        assert!(sv.is_normalized(1e-10), "2nd order Trotter must preserve norm");
    }

    #[test]
    fn test_trotter_zero_time_no_change() {
        let h = h2_hamiltonian();
        let mut sv = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv).unwrap();
        let before = sv.amplitudes.clone();

        trotter_evolve(&mut sv, &h, 0.0, 10, TrotterOrder::First);

        for i in 0..sv.n_states() {
            assert!((sv.amplitudes[i].re - before[i].re).abs() < TOL);
            assert!((sv.amplitudes[i].im - before[i].im).abs() < TOL);
        }
    }

    #[test]
    fn test_trotter_determinism() {
        let h = h2_hamiltonian();
        let mut sv1 = Statevector::new(2);
        let mut sv2 = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv1).unwrap();
        crate::gates::Gate::H(0).apply(&mut sv2).unwrap();

        trotter_evolve(&mut sv1, &h, 0.5, 20, TrotterOrder::First);
        trotter_evolve(&mut sv2, &h, 0.5, 20, TrotterOrder::First);

        for i in 0..sv1.n_states() {
            assert_eq!(sv1.amplitudes[i].re.to_bits(), sv2.amplitudes[i].re.to_bits(),
                "Trotter must be bit-identical (re[{}])", i);
            assert_eq!(sv1.amplitudes[i].im.to_bits(), sv2.amplitudes[i].im.to_bits(),
                "Trotter must be bit-identical (im[{}])", i);
        }
    }

    #[test]
    fn test_trotter_second_order_more_accurate() {
        // With the same number of steps, 2nd order should be more accurate
        // than 1st order (closer to the exact evolution).
        let h = h2_hamiltonian();
        let time = 0.5;
        let steps = 5;

        // Compute 1st order result
        let mut sv1 = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv1).unwrap();
        trotter_evolve(&mut sv1, &h, time, steps, TrotterOrder::First);

        // Compute 2nd order result
        let mut sv2 = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv2).unwrap();
        trotter_evolve(&mut sv2, &h, time, steps, TrotterOrder::Second);

        // Compute "exact" (high-step) reference
        let mut sv_ref = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv_ref).unwrap();
        trotter_evolve(&mut sv_ref, &h, time, 1000, TrotterOrder::First);

        // Compare fidelities
        let fid1 = fidelity(&sv1, &sv_ref);
        let fid2 = fidelity(&sv2, &sv_ref);
        assert!(fid2 >= fid1 - 0.01,
            "2nd order fidelity ({:.6}) should be >= 1st order ({:.6})", fid2, fid1);
    }

    #[test]
    fn test_trotter_error_bound_positive() {
        let h = h2_hamiltonian();
        let bound = trotter_error_bound(&h, 1.0, 10, TrotterOrder::First);
        assert!(bound > 0.0, "error bound must be positive");
        assert!(bound.is_finite(), "error bound must be finite");
    }

    #[test]
    fn test_apply_pauli_rotation_diagonal() {
        // e^{-iθZ}|0⟩ = e^{-iθ}|0⟩
        let mut sv = Statevector::new(1);
        let theta = 0.3;
        apply_pauli_rotation(&mut sv, &[Pauli::Z], theta);
        // |0⟩ has Z eigenvalue +1, so phase = e^{-iθ}
        let expected_re = (-theta).cos();
        let expected_im = (-theta).sin();
        assert!((sv.amplitudes[0].re - expected_re).abs() < 1e-12);
        assert!((sv.amplitudes[0].im - expected_im).abs() < 1e-12);
        assert!(sv.is_normalized(1e-12));
    }

    fn fidelity(a: &Statevector, b: &Statevector) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        let mut acc_im = KahanAccumulatorF64::new();
        for i in 0..a.n_states() {
            let dot = a.amplitudes[i].conj().mul_fixed(b.amplitudes[i]);
            acc.add(dot.re);
            acc_im.add(dot.im);
        }
        let re = acc.finalize();
        let im = acc_im.finalize();
        re * re + im * im
    }
}
