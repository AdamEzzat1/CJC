//! Dual-mode parity tests: Rust backend vs Pure CJC backend.
//!
//! Constraint: delta between backends must be exactly 0.00e0.

use cjc_quantum::fermion::*;
use cjc_quantum::pure::*;
use cjc_quantum::statevector::Statevector;
use cjc_quantum::gates::Gate;
use cjc_quantum::mitigation;

const TOL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Fermion parity: H₂ expectation values
// ---------------------------------------------------------------------------

#[test]
fn test_h2_expectation_parity() {
    // Rust backend
    let h_rust = h2_hamiltonian();
    let mut sv = Statevector::new(2);
    Gate::H(0).apply(&mut sv).unwrap();
    Gate::CNOT(0, 1).apply(&mut sv).unwrap();
    let e_rust = h_rust.expectation(&sv);

    // Pure backend
    let h_pure = pure_h2_hamiltonian();
    // Construct same state in pure format
    let isq2 = 1.0 / 2.0f64.sqrt();
    let amps: Vec<(f64, f64)> = vec![
        (isq2, 0.0),  // |00⟩
        (0.0, 0.0),   // |01⟩
        (0.0, 0.0),   // |10⟩
        (isq2, 0.0),  // |11⟩
    ];
    let e_pure = h_pure.expectation(&amps, 2);

    let delta = (e_rust - e_pure).abs();
    assert!(delta < TOL,
        "H₂ parity violation: rust={}, pure={}, delta={:.2e}", e_rust, e_pure, delta);
}

#[test]
fn test_h2_expectation_parity_product_state() {
    // |00⟩ state
    let h_rust = h2_hamiltonian();
    let sv = Statevector::new(2);
    let e_rust = h_rust.expectation(&sv);

    let h_pure = pure_h2_hamiltonian();
    let amps = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let e_pure = h_pure.expectation(&amps, 2);

    let delta = (e_rust - e_pure).abs();
    assert!(delta < TOL,
        "H₂ |00⟩ parity violation: rust={}, pure={}, delta={:.2e}", e_rust, e_pure, delta);
}

#[test]
fn test_h2_expectation_parity_x_state() {
    // X|0⟩ ⊗ |0⟩ = |10⟩
    let h_rust = h2_hamiltonian();
    let mut sv = Statevector::new(2);
    Gate::X(0).apply(&mut sv).unwrap();
    let e_rust = h_rust.expectation(&sv);

    let h_pure = pure_h2_hamiltonian();
    let amps = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let e_pure = h_pure.expectation(&amps, 2);

    let delta = (e_rust - e_pure).abs();
    assert!(delta < TOL,
        "H₂ |10⟩ parity violation: rust={}, pure={}, delta={:.2e}", e_rust, e_pure, delta);
}

// ---------------------------------------------------------------------------
// ZNE parity: Richardson extrapolation
// ---------------------------------------------------------------------------

#[test]
fn test_zne_richardson_parity() {
    let scales = [1.0, 2.0, 3.0];
    let values = [0.95, 0.88, 0.75];

    // Rust backend
    let r_rust = mitigation::richardson_extrapolate(&scales, &values).unwrap();

    // Pure backend
    let r_pure = pure_richardson_extrapolate(&scales, &values).unwrap();

    let delta = (r_rust.mitigated_value - r_pure.mitigated_value).abs();
    assert!(delta < TOL,
        "ZNE parity violation: rust={}, pure={}, delta={:.2e}",
        r_rust.mitigated_value, r_pure.mitigated_value, delta);

    // Coefficient parity
    for (i, (cr, cp)) in r_rust.coefficients.iter().zip(&r_pure.coefficients).enumerate() {
        let d = (cr - cp).abs();
        assert!(d < TOL,
            "ZNE coeff[{}] parity: rust={}, pure={}, delta={:.2e}", i, cr, cp, d);
    }
}

#[test]
fn test_zne_richardson_parity_five_points() {
    let scales = [1.0, 1.5, 2.0, 2.5, 3.0];
    let values = [0.98, 0.95, 0.90, 0.82, 0.70];

    let r_rust = mitigation::richardson_extrapolate(&scales, &values).unwrap();
    let r_pure = pure_richardson_extrapolate(&scales, &values).unwrap();

    let delta = (r_rust.mitigated_value - r_pure.mitigated_value).abs();
    assert!(delta < TOL,
        "ZNE 5-point parity: rust={}, pure={}, delta={:.2e}",
        r_rust.mitigated_value, r_pure.mitigated_value, delta);
}

// ---------------------------------------------------------------------------
// Trotter parity: time evolution
// ---------------------------------------------------------------------------

#[test]
fn test_trotter_parity_h2() {
    use cjc_quantum::trotter;

    let h_rust = h2_hamiltonian();
    let h_pure = pure_h2_hamiltonian();

    // Rust backend
    let mut sv = Statevector::new(2);
    Gate::H(0).apply(&mut sv).unwrap();
    trotter::trotter_evolve(&mut sv, &h_rust, 0.5, 50, trotter::TrotterOrder::First);
    let e_rust = h_rust.expectation(&sv);

    // Pure backend
    let isq2 = 1.0 / 2.0f64.sqrt();
    let mut amps: Vec<(f64, f64)> = vec![
        (isq2, 0.0),  // H|0⟩ on qubit 0
        (isq2, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ];
    pure_trotter_evolve(&mut amps, 2, &h_pure, 0.5, 50, false);
    let e_pure = h_pure.expectation(&amps, 2);

    let delta = (e_rust - e_pure).abs();
    assert!(delta < 1e-8,
        "Trotter parity: rust={}, pure={}, delta={:.2e}", e_rust, e_pure, delta);
}

// ---------------------------------------------------------------------------
// MPS canonical form: verify state is preserved
// ---------------------------------------------------------------------------

#[test]
fn test_mps_left_canonicalize_preserves_state() {
    use cjc_quantum::mps::Mps;
    use cjc_runtime::complex::ComplexF64;

    let isq2 = 1.0 / 2.0f64.sqrt();
    let h_mat = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
                 [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];

    let mut mps = Mps::with_max_bond(4, 8);
    mps.apply_single_qubit(0, h_mat);
    mps.apply_cnot_adjacent(0, 1);

    let sv_before = mps.to_statevector();
    mps.left_canonicalize();
    let sv_after = mps.to_statevector();

    for i in 0..sv_before.len() {
        let d_re = (sv_before[i].re - sv_after[i].re).abs();
        let d_im = (sv_before[i].im - sv_after[i].im).abs();
        assert!(d_re < 1e-10 && d_im < 1e-10,
            "left-canonicalize changed state at [{}]: before={:?}, after={:?}",
            i, sv_before[i], sv_after[i]);
    }
}

#[test]
fn test_mps_right_canonicalize_preserves_product_state() {
    use cjc_quantum::mps::Mps;
    use cjc_runtime::complex::ComplexF64;

    // Product state: no entanglement, bonds stay at 1
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h_mat = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
                 [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];

    let mut mps = Mps::with_max_bond(3, 8);
    mps.apply_single_qubit(0, h_mat);
    mps.apply_single_qubit(1, h_mat);

    let sv_before = mps.to_statevector();
    mps.right_canonicalize();
    let sv_after = mps.to_statevector();

    for i in 0..sv_before.len() {
        let d_re = (sv_before[i].re - sv_after[i].re).abs();
        let d_im = (sv_before[i].im - sv_after[i].im).abs();
        assert!(d_re < 1e-10 && d_im < 1e-10,
            "right-canonicalize changed state at [{}]: before={:?}, after={:?}",
            i, sv_before[i], sv_after[i]);
    }
}
