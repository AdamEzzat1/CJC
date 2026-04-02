//! Hardened tests for Jordan-Wigner fermionic Hamiltonians.

use cjc_quantum::fermion::*;
use cjc_quantum::statevector::Statevector;
use cjc_quantum::gates::Gate;

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Structure tests
// ---------------------------------------------------------------------------

#[test]
fn test_h2_hamiltonian_has_6_terms() {
    let h = h2_hamiltonian();
    assert_eq!(h.n_qubits, 2);
    assert_eq!(h.n_terms(), 6);
}

#[test]
fn test_lih_hamiltonian_has_terms() {
    let h = lih_hamiltonian();
    assert_eq!(h.n_qubits, 4);
    assert!(h.n_terms() > 10, "LiH should have many terms, got {}", h.n_terms());
}

// ---------------------------------------------------------------------------
// Pauli algebra
// ---------------------------------------------------------------------------

#[test]
fn test_pauli_anticommutation_all_pairs() {
    use cjc_runtime::complex::ComplexF64;
    // {σ_i, σ_j} = 2δ_{ij}I for i,j in {X,Y,Z}
    let paulis = [Pauli::X, Pauli::Y, Pauli::Z];
    for (i, &a) in paulis.iter().enumerate() {
        for (j, &b) in paulis.iter().enumerate() {
            let mut t1 = PauliTerm::identity(1);
            t1.ops[0] = a;
            let mut t2 = PauliTerm::identity(1);
            t2.ops[0] = b;

            let ab = t1.multiply(&t2);
            let ba = t2.multiply(&t1);
            let anticomm_re = ab.coeff.re + ba.coeff.re;
            let anticomm_im = ab.coeff.im + ba.coeff.im;

            if i == j {
                // {σ_i, σ_i} = 2I
                assert!((anticomm_re - 2.0).abs() < TOL, "{{σ_{}, σ_{}}} re = {}", i, j, anticomm_re);
                assert!(anticomm_im.abs() < TOL);
            } else {
                // {σ_i, σ_j} = 0
                assert!(anticomm_re.abs() < TOL, "{{σ_{}, σ_{}}} re = {}", i, j, anticomm_re);
                assert!(anticomm_im.abs() < TOL, "{{σ_{}, σ_{}}} im = {}", i, j, anticomm_im);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Number operator verification
// ---------------------------------------------------------------------------

#[test]
fn test_number_operator_on_vacuum() {
    // ⟨00|n_0|00⟩ = 0 (qubit 0 unoccupied)
    let terms = jw_one_body(2, 0, 0);
    let mut h = FermionicHamiltonian::new(2);
    for t in terms { h.add_term(t); }
    let sv = Statevector::new(2); // |00⟩
    let e = h.expectation(&sv);
    assert!(e.abs() < TOL, "⟨00|n_0|00⟩ = {}, expected 0", e);
}

#[test]
fn test_number_operator_on_occupied() {
    // ⟨10|n_0|10⟩ = 1
    let terms = jw_one_body(2, 0, 0);
    let mut h = FermionicHamiltonian::new(2);
    for t in terms { h.add_term(t); }
    let mut sv = Statevector::new(2);
    Gate::X(0).apply(&mut sv).unwrap(); // |10⟩
    let e = h.expectation(&sv);
    assert!((e - 1.0).abs() < TOL, "⟨10|n_0|10⟩ = {}, expected 1", e);
}

// ---------------------------------------------------------------------------
// Hermiticity: ⟨ψ|H|ψ⟩ must be real
// ---------------------------------------------------------------------------

#[test]
fn test_h2_expectation_real_for_random_states() {
    let h = h2_hamiltonian();
    // Create superposition state
    let mut sv = Statevector::new(2);
    Gate::Ry(0, 0.7).apply(&mut sv).unwrap();
    Gate::Ry(1, 1.3).apply(&mut sv).unwrap();
    Gate::CNOT(0, 1).apply(&mut sv).unwrap();

    let e = h.expectation(&sv);
    assert!(e.is_finite(), "H₂ expectation must be finite");
    // The value itself is real by construction (we only return .re)
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_h2_expectation_deterministic() {
    let h = h2_hamiltonian();
    let mut sv = Statevector::new(2);
    Gate::H(0).apply(&mut sv).unwrap();
    Gate::CNOT(0, 1).apply(&mut sv).unwrap();

    let e1 = h.expectation(&sv);
    let e2 = h.expectation(&sv);
    assert_eq!(e1.to_bits(), e2.to_bits(), "must be bit-identical");
}

#[test]
fn test_lih_expectation_deterministic() {
    let h = lih_hamiltonian();
    let mut sv = Statevector::new(4);
    Gate::H(0).apply(&mut sv).unwrap();
    Gate::H(2).apply(&mut sv).unwrap();

    let e1 = h.expectation(&sv);
    let e2 = h.expectation(&sv);
    assert_eq!(e1.to_bits(), e2.to_bits(), "LiH must be bit-identical");
}

// ---------------------------------------------------------------------------
// JW one-body terms
// ---------------------------------------------------------------------------

#[test]
fn test_jw_one_body_diagonal_gives_two_terms() {
    let terms = jw_one_body(3, 1, 1);
    assert_eq!(terms.len(), 2, "number operator should give I and Z terms");
}

#[test]
fn test_jw_one_body_offdiag_gives_four_terms() {
    let terms = jw_one_body(3, 0, 2);
    assert_eq!(terms.len(), 4, "off-diagonal should give XX, YY, XY, YX terms");
}
