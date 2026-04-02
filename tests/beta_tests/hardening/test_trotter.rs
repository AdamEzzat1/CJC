//! Hardened tests for Suzuki-Trotter time evolution.

use cjc_quantum::fermion::h2_hamiltonian;
use cjc_quantum::trotter::*;
use cjc_quantum::statevector::Statevector;
use cjc_quantum::gates::Gate;
use cjc_repro::KahanAccumulatorF64;

const TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Unitarity: Trotter must preserve norm
// ---------------------------------------------------------------------------

#[test]
fn test_trotter_first_order_preserves_norm() {
    let h = h2_hamiltonian();
    let mut sv = Statevector::new(2);
    Gate::Ry(0, 1.2).apply(&mut sv).unwrap();
    Gate::Ry(1, 0.8).apply(&mut sv).unwrap();

    trotter_evolve(&mut sv, &h, 2.0, 100, TrotterOrder::First);
    assert!(sv.is_normalized(1e-10), "1st order Trotter must preserve norm");
}

#[test]
fn test_trotter_second_order_preserves_norm() {
    let h = h2_hamiltonian();
    let mut sv = Statevector::new(2);
    Gate::Ry(0, 0.5).apply(&mut sv).unwrap();
    Gate::CNOT(0, 1).apply(&mut sv).unwrap();

    trotter_evolve(&mut sv, &h, 3.0, 50, TrotterOrder::Second);
    assert!(sv.is_normalized(1e-10), "2nd order Trotter must preserve norm");
}

// ---------------------------------------------------------------------------
// Convergence: more steps → more accurate
// ---------------------------------------------------------------------------

#[test]
fn test_trotter_converges_with_steps() {
    let h = h2_hamiltonian();
    let time = 1.0;

    // Reference: high-step 1st order
    let mut sv_ref = Statevector::new(2);
    Gate::H(0).apply(&mut sv_ref).unwrap();
    trotter_evolve(&mut sv_ref, &h, time, 2000, TrotterOrder::First);

    // Low-step
    let mut sv_lo = Statevector::new(2);
    Gate::H(0).apply(&mut sv_lo).unwrap();
    trotter_evolve(&mut sv_lo, &h, time, 10, TrotterOrder::First);

    // Mid-step
    let mut sv_mid = Statevector::new(2);
    Gate::H(0).apply(&mut sv_mid).unwrap();
    trotter_evolve(&mut sv_mid, &h, time, 100, TrotterOrder::First);

    let fid_lo = state_fidelity(&sv_lo, &sv_ref);
    let fid_mid = state_fidelity(&sv_mid, &sv_ref);

    assert!(fid_mid >= fid_lo - 0.001,
        "more steps should give higher fidelity: lo={}, mid={}", fid_lo, fid_mid);
}

// ---------------------------------------------------------------------------
// Energy conservation
// ---------------------------------------------------------------------------

#[test]
fn test_trotter_energy_approximately_conserved() {
    let h = h2_hamiltonian();
    let mut sv = Statevector::new(2);
    Gate::Ry(0, 1.0).apply(&mut sv).unwrap();
    Gate::Ry(1, 0.5).apply(&mut sv).unwrap();

    let e_before = h.expectation(&sv);
    trotter_evolve(&mut sv, &h, 0.5, 200, TrotterOrder::First);
    let e_after = h.expectation(&sv);

    // Energy should be approximately conserved (within Trotter error)
    assert!((e_before - e_after).abs() < 0.01,
        "energy drift too large: before={}, after={}, delta={}",
        e_before, e_after, (e_before - e_after).abs());
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_trotter_evolve_bitwise_deterministic() {
    let h = h2_hamiltonian();

    let mut sv1 = Statevector::new(2);
    let mut sv2 = Statevector::new(2);
    Gate::Ry(0, 0.7).apply(&mut sv1).unwrap();
    Gate::Ry(0, 0.7).apply(&mut sv2).unwrap();

    trotter_evolve(&mut sv1, &h, 1.5, 30, TrotterOrder::First);
    trotter_evolve(&mut sv2, &h, 1.5, 30, TrotterOrder::First);

    for i in 0..sv1.n_states() {
        assert_eq!(sv1.amplitudes[i].re.to_bits(), sv2.amplitudes[i].re.to_bits(),
            "re[{}] not bit-identical", i);
        assert_eq!(sv1.amplitudes[i].im.to_bits(), sv2.amplitudes[i].im.to_bits(),
            "im[{}] not bit-identical", i);
    }
}

#[test]
fn test_trotter_second_order_deterministic() {
    let h = h2_hamiltonian();

    let mut sv1 = Statevector::new(2);
    let mut sv2 = Statevector::new(2);
    Gate::H(0).apply(&mut sv1).unwrap();
    Gate::H(0).apply(&mut sv2).unwrap();

    trotter_evolve(&mut sv1, &h, 1.0, 20, TrotterOrder::Second);
    trotter_evolve(&mut sv2, &h, 1.0, 20, TrotterOrder::Second);

    for i in 0..sv1.n_states() {
        assert_eq!(sv1.amplitudes[i].re.to_bits(), sv2.amplitudes[i].re.to_bits());
        assert_eq!(sv1.amplitudes[i].im.to_bits(), sv2.amplitudes[i].im.to_bits());
    }
}

// ---------------------------------------------------------------------------
// Error bound
// ---------------------------------------------------------------------------

#[test]
fn test_trotter_error_bound_decreases_with_steps() {
    let h = h2_hamiltonian();
    let time = 1.0;

    let e10 = trotter_error_bound(&h, time, 10, TrotterOrder::First);
    let e100 = trotter_error_bound(&h, time, 100, TrotterOrder::First);

    assert!(e100 < e10, "more steps should reduce error bound: e10={}, e100={}", e10, e100);
}

#[test]
fn test_trotter_second_order_bound_smaller() {
    let h = h2_hamiltonian();
    let time = 0.5;
    let steps = 20;

    let e1 = trotter_error_bound(&h, time, steps, TrotterOrder::First);
    let e2 = trotter_error_bound(&h, time, steps, TrotterOrder::Second);

    assert!(e2 < e1, "2nd order should have smaller error bound: first={}, second={}", e1, e2);
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn state_fidelity(a: &Statevector, b: &Statevector) -> f64 {
    let mut acc_re = KahanAccumulatorF64::new();
    let mut acc_im = KahanAccumulatorF64::new();
    for i in 0..a.n_states() {
        let dot = a.amplitudes[i].conj().mul_fixed(b.amplitudes[i]);
        acc_re.add(dot.re);
        acc_im.add(dot.im);
    }
    let re = acc_re.finalize();
    let im = acc_im.finalize();
    re * re + im * im
}
