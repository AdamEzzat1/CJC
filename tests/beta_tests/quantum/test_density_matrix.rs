//! Integration tests for the density matrix simulator module.
//!
//! Covers: initial state properties, gate application, noise channels,
//! entropy, partial trace, and fidelity.

use cjc_quantum::density::*;
use cjc_quantum::Complex as ComplexF64;
use cjc_quantum::gates::Gate;
use cjc_quantum::Statevector;

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// 1. Initial state: trace = 1, purity = 1
// ---------------------------------------------------------------------------

#[test]
fn initial_state_trace_is_one() {
    for n in 1..=3 {
        let dm = DensityMatrix::new(n);
        assert!(
            (dm.trace() - 1.0).abs() < TOL,
            "Trace of {}-qubit initial state should be 1.0, got {}",
            n,
            dm.trace()
        );
    }
}

#[test]
fn initial_state_purity_is_one() {
    for n in 1..=3 {
        let dm = DensityMatrix::new(n);
        assert!(
            (dm.purity() - 1.0).abs() < TOL,
            "Purity of {}-qubit initial state should be 1.0, got {}",
            n,
            dm.purity()
        );
    }
}

#[test]
fn initial_state_is_zero_ket() {
    let dm = DensityMatrix::new(2);
    // |00><00| => only rho[0,0] = 1, rest = 0
    assert!(
        (dm.get(0, 0).re - 1.0).abs() < TOL,
        "rho[0,0] of initial state should be 1.0, got {}",
        dm.get(0, 0).re
    );
    for i in 1..(1 << 2) {
        assert!(
            dm.get(i, i).norm_sq() < TOL,
            "rho[{},{}] of initial state should be 0, got {:?}",
            i,
            i,
            dm.get(i, i)
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Apply H gate: probabilities should be 0.5/0.5
// ---------------------------------------------------------------------------

#[test]
fn hadamard_gives_equal_probabilities() {
    let mut dm = DensityMatrix::new(1);
    dm.apply_gate(&Gate::H(0));
    let probs = dm.probabilities();

    assert!(
        (probs[0] - 0.5).abs() < TOL,
        "P(|0>) after H should be 0.5, got {}",
        probs[0]
    );
    assert!(
        (probs[1] - 0.5).abs() < TOL,
        "P(|1>) after H should be 0.5, got {}",
        probs[1]
    );
}

#[test]
fn hadamard_preserves_trace_and_purity() {
    let mut dm = DensityMatrix::new(1);
    dm.apply_gate(&Gate::H(0));

    assert!(
        (dm.trace() - 1.0).abs() < TOL,
        "Trace after H should remain 1.0, got {}",
        dm.trace()
    );
    assert!(
        (dm.purity() - 1.0).abs() < TOL,
        "Purity after H should remain 1.0 (still pure), got {}",
        dm.purity()
    );
}

#[test]
fn hadamard_produces_plus_state_density_matrix() {
    let mut dm = DensityMatrix::new(1);
    dm.apply_gate(&Gate::H(0));

    // |+><+| = [[0.5, 0.5], [0.5, 0.5]]
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (dm.get(i, j).re - 0.5).abs() < TOL,
                "rho[{},{}].re should be 0.5, got {}",
                i,
                j,
                dm.get(i, j).re
            );
            assert!(
                dm.get(i, j).im.abs() < TOL,
                "rho[{},{}].im should be 0, got {}",
                i,
                j,
                dm.get(i, j).im
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Depolarizing noise reduces purity
// ---------------------------------------------------------------------------

#[test]
fn depolarizing_noise_reduces_purity() {
    let mut dm = DensityMatrix::new(1);
    // Start with a pure state |0>
    let purity_before = dm.purity();
    assert!(
        (purity_before - 1.0).abs() < TOL,
        "Purity before noise should be 1.0"
    );

    let channel = depolarizing_channel(0.3);
    dm.apply_single_qubit_channel(0, &channel);

    let purity_after = dm.purity();
    assert!(
        purity_after < purity_before - TOL,
        "Purity after depolarizing noise (p=0.3) should be less than before: {} >= {}",
        purity_after,
        purity_before
    );
    assert!(
        (dm.trace() - 1.0).abs() < TOL,
        "Trace should remain 1.0 after depolarizing noise, got {}",
        dm.trace()
    );
}

#[test]
fn depolarizing_full_destroys_coherence_and_reduces_purity() {
    // p=1.0 depolarizing: rho -> (X rho X + Y rho Y + Z rho Z) / 3
    // For |0><0|: rho' = (|1><1| + |1><1| + |0><0|) / 3 = diag(1/3, 2/3)
    let mut dm = DensityMatrix::new(1);
    let channel = depolarizing_channel(1.0);
    dm.apply_single_qubit_channel(0, &channel);

    // Off-diagonals should be zero
    assert!(
        dm.get(0, 1).norm_sq() < TOL,
        "Off-diagonal after full depolarizing should be 0"
    );
    // Purity should be less than 1 (mixed state)
    assert!(
        dm.purity() < 1.0 - TOL,
        "Full depolarizing should produce a mixed state, purity = {}",
        dm.purity()
    );
    // Trace preserved
    assert!(
        (dm.trace() - 1.0).abs() < TOL,
        "Trace must remain 1.0, got {}",
        dm.trace()
    );
}

#[test]
fn depolarizing_noise_on_multiqubit_state() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_gate(&Gate::H(0));
    dm.apply_gate(&Gate::CNOT(0, 1));
    // Now in Bell state, purity = 1.0
    let purity_before = dm.purity();
    assert!(
        (purity_before - 1.0).abs() < TOL,
        "Bell state should be pure"
    );

    let channel = depolarizing_channel(0.1);
    dm.apply_single_qubit_channel(0, &channel);

    assert!(
        dm.purity() < purity_before - TOL,
        "Depolarizing qubit 0 of Bell state should reduce purity"
    );
    assert!(
        (dm.trace() - 1.0).abs() < TOL,
        "Trace must remain 1.0 after noise"
    );
}

// ---------------------------------------------------------------------------
// 4. Amplitude damping: excited state decays toward ground
// ---------------------------------------------------------------------------

#[test]
fn amplitude_damping_decays_excited_state() {
    // Start in |1> state
    let mut dm = DensityMatrix::new(1);
    dm.apply_gate(&Gate::X(0));
    assert!(
        (dm.get(1, 1).re - 1.0).abs() < TOL,
        "Should start in |1> state"
    );

    let channel = amplitude_damping_channel(0.5);
    dm.apply_single_qubit_channel(0, &channel);

    // After damping with gamma=0.5:
    // P(|0>) should increase, P(|1>) should decrease
    let probs = dm.probabilities();
    assert!(
        probs[0] > TOL,
        "P(|0>) should be positive after amplitude damping, got {}",
        probs[0]
    );
    assert!(
        probs[1] < 1.0 - TOL,
        "P(|1>) should decrease after amplitude damping, got {}",
        probs[1]
    );
    assert!(
        (dm.trace() - 1.0).abs() < TOL,
        "Trace must remain 1.0, got {}",
        dm.trace()
    );
}

#[test]
fn amplitude_damping_full_resets_to_ground() {
    // Start in |1>, apply full damping (gamma=1.0)
    let mut dm = DensityMatrix::new(1);
    dm.apply_gate(&Gate::X(0));

    let channel = amplitude_damping_channel(1.0);
    dm.apply_single_qubit_channel(0, &channel);

    // gamma=1.0 => |1> fully decays to |0>
    assert!(
        (dm.get(0, 0).re - 1.0).abs() < TOL,
        "After full amplitude damping, rho[0,0] should be 1.0, got {}",
        dm.get(0, 0).re
    );
    assert!(
        dm.get(1, 1).re.abs() < TOL,
        "After full amplitude damping, rho[1,1] should be 0, got {}",
        dm.get(1, 1).re
    );
    assert!(
        (dm.purity() - 1.0).abs() < TOL,
        "Full damping of |1> to |0> should yield a pure state"
    );
}

#[test]
fn amplitude_damping_ground_state_unchanged() {
    // |0> should be unaffected by amplitude damping
    let mut dm = DensityMatrix::new(1);
    let channel = amplitude_damping_channel(0.7);
    dm.apply_single_qubit_channel(0, &channel);

    assert!(
        (dm.get(0, 0).re - 1.0).abs() < TOL,
        "Ground state should be unaffected by amplitude damping"
    );
    assert!(
        dm.get(1, 1).re.abs() < TOL,
        "rho[1,1] should remain 0 for ground state"
    );
}

// ---------------------------------------------------------------------------
// 5. Von Neumann entropy of maximally mixed state
// ---------------------------------------------------------------------------

#[test]
fn entropy_of_pure_state_is_zero() {
    let dm = DensityMatrix::new(1);
    let s = dm.von_neumann_entropy();
    assert!(
        s.abs() < 1e-6,
        "Von Neumann entropy of a pure state should be 0, got {}",
        s
    );
}

#[test]
fn entropy_of_maximally_mixed_single_qubit() {
    // Maximally mixed 1-qubit: rho = I/2, built directly
    // S = ln(2) ~ 0.6931...
    let mut dm = DensityMatrix::new(1);
    // Manually set to I/2
    dm.set(0, 0, ComplexF64::real(0.5));
    dm.set(0, 1, ComplexF64::ZERO);
    dm.set(1, 0, ComplexF64::ZERO);
    dm.set(1, 1, ComplexF64::real(0.5));

    let s = dm.von_neumann_entropy();
    let expected = 2.0f64.ln(); // ln(2)
    assert!(
        (s - expected).abs() < 1e-6,
        "Entropy of maximally mixed 1-qubit state should be ln(2) = {:.6}, got {:.6}",
        expected,
        s
    );
}

#[test]
fn entropy_of_maximally_mixed_two_qubit() {
    // Maximally mixed 2-qubit: rho = I/4, built directly
    // S = ln(4) = 2*ln(2) ~ 1.3863...
    let mut dm = DensityMatrix::new(2);
    let dim = 4;
    for i in 0..dim {
        for j in 0..dim {
            if i == j {
                dm.set(i, j, ComplexF64::real(0.25));
            } else {
                dm.set(i, j, ComplexF64::ZERO);
            }
        }
    }

    let s = dm.von_neumann_entropy();
    let expected = 4.0f64.ln(); // ln(4) = 2*ln(2)
    assert!(
        (s - expected).abs() < 1e-4,
        "Entropy of maximally mixed 2-qubit state should be ln(4) = {:.6}, got {:.6}",
        expected,
        s
    );
}

#[test]
fn entropy_increases_with_noise() {
    let dm_pure = DensityMatrix::new(1);
    let s_pure = dm_pure.von_neumann_entropy();

    let mut dm_noisy = DensityMatrix::new(1);
    let channel = depolarizing_channel(0.5);
    dm_noisy.apply_single_qubit_channel(0, &channel);
    let s_noisy = dm_noisy.von_neumann_entropy();

    assert!(
        s_noisy > s_pure + 1e-8,
        "Entropy should increase after noise: pure={}, noisy={}",
        s_pure,
        s_noisy
    );
}

// ---------------------------------------------------------------------------
// 6. Partial trace of Bell state gives maximally mixed single qubit
// ---------------------------------------------------------------------------

#[test]
fn partial_trace_bell_state_gives_maximally_mixed() {
    // Create Bell state (|00> + |11>)/sqrt(2)
    let mut dm = DensityMatrix::new(2);
    dm.apply_gate(&Gate::H(0));
    dm.apply_gate(&Gate::CNOT(0, 1));

    // Trace out qubit 1, keep qubit 0
    let reduced = dm.partial_trace(&[0]);
    assert!(
        (reduced.trace() - 1.0).abs() < TOL,
        "Trace of reduced state should be 1.0, got {}",
        reduced.trace()
    );

    // Should be maximally mixed: rho_reduced = I/2
    assert!(
        (reduced.get(0, 0).re - 0.5).abs() < TOL,
        "Reduced rho[0,0] should be 0.5, got {}",
        reduced.get(0, 0).re
    );
    assert!(
        (reduced.get(1, 1).re - 0.5).abs() < TOL,
        "Reduced rho[1,1] should be 0.5, got {}",
        reduced.get(1, 1).re
    );
    assert!(
        reduced.get(0, 1).norm_sq() < TOL,
        "Off-diagonal of reduced Bell state should be 0, got {:?}",
        reduced.get(0, 1)
    );
    assert!(
        reduced.get(1, 0).norm_sq() < TOL,
        "Off-diagonal of reduced Bell state should be 0, got {:?}",
        reduced.get(1, 0)
    );
}

#[test]
fn partial_trace_bell_state_symmetric() {
    // Both reductions of a Bell state should give the same maximally mixed state
    let mut dm = DensityMatrix::new(2);
    dm.apply_gate(&Gate::H(0));
    dm.apply_gate(&Gate::CNOT(0, 1));

    let reduced_keep0 = dm.partial_trace(&[0]);
    let reduced_keep1 = dm.partial_trace(&[1]);

    // Both should be I/2
    for i in 0..2 {
        for j in 0..2 {
            let diff = (reduced_keep0.get(i, j).re - reduced_keep1.get(i, j).re).abs()
                + (reduced_keep0.get(i, j).im - reduced_keep1.get(i, j).im).abs();
            assert!(
                diff < TOL,
                "Reduced states should match at [{},{}]: keep0={:?}, keep1={:?}",
                i,
                j,
                reduced_keep0.get(i, j),
                reduced_keep1.get(i, j)
            );
        }
    }
}

#[test]
fn partial_trace_purity_of_bell_reduction() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_gate(&Gate::H(0));
    dm.apply_gate(&Gate::CNOT(0, 1));

    let reduced = dm.partial_trace(&[0]);
    let purity = reduced.purity();
    // Maximally mixed 1-qubit has purity = 0.5
    assert!(
        (purity - 0.5).abs() < TOL,
        "Purity of reduced Bell state should be 0.5, got {}",
        purity
    );
}

#[test]
fn partial_trace_product_state_stays_pure() {
    // |0> tensor |1>: tracing out one qubit should leave a pure state
    let mut dm = DensityMatrix::new(2);
    dm.apply_gate(&Gate::X(1));

    let reduced = dm.partial_trace(&[0]);
    assert!(
        (reduced.purity() - 1.0).abs() < TOL,
        "Partial trace of product state should yield pure state, purity = {}",
        reduced.purity()
    );
    // Should be |0><0|
    assert!(
        (reduced.get(0, 0).re - 1.0).abs() < TOL,
        "Keeping qubit 0 of |0>|1> should give |0><0|"
    );
}

#[test]
fn partial_trace_three_qubit_ghz() {
    // GHZ state: (|000> + |111>)/sqrt(2)
    let mut dm = DensityMatrix::new(3);
    dm.apply_gate(&Gate::H(0));
    dm.apply_gate(&Gate::CNOT(0, 1));
    dm.apply_gate(&Gate::CNOT(1, 2));

    // Trace out qubit 2, keep qubits 0 and 1
    let reduced = dm.partial_trace(&[0, 1]);
    assert!(
        (reduced.trace() - 1.0).abs() < TOL,
        "Trace of reduced GHZ state should be 1.0"
    );

    // Reduced state of GHZ tracing one qubit:
    // rho = 0.5 * |00><00| + 0.5 * |11><11|  (classically correlated, not entangled)
    assert!(
        (reduced.get(0, 0).re - 0.5).abs() < TOL,
        "GHZ reduced rho[00,00] should be 0.5, got {}",
        reduced.get(0, 0).re
    );
    assert!(
        (reduced.get(3, 3).re - 0.5).abs() < TOL,
        "GHZ reduced rho[11,11] should be 0.5, got {}",
        reduced.get(3, 3).re
    );
    // Off-diagonal coherences between |00> and |11> should vanish
    assert!(
        reduced.get(0, 3).norm_sq() < TOL,
        "GHZ reduced rho[00,11] should be 0 after tracing, got {:?}",
        reduced.get(0, 3)
    );
}

// ---------------------------------------------------------------------------
// 7. Fidelity between identical states = 1
// ---------------------------------------------------------------------------

#[test]
fn fidelity_identical_states_is_one() {
    let dm = DensityMatrix::new(1);
    let f = DensityMatrix::fidelity(&dm, &dm);
    assert!(
        (f - 1.0).abs() < TOL,
        "Fidelity of state with itself should be 1.0, got {}",
        f
    );
}

#[test]
fn fidelity_identical_after_gate() {
    let mut dm1 = DensityMatrix::new(2);
    dm1.apply_gate(&Gate::H(0));
    dm1.apply_gate(&Gate::CNOT(0, 1));

    let mut dm2 = DensityMatrix::new(2);
    dm2.apply_gate(&Gate::H(0));
    dm2.apply_gate(&Gate::CNOT(0, 1));

    let f = DensityMatrix::fidelity(&dm1, &dm2);
    assert!(
        (f - 1.0).abs() < TOL,
        "Fidelity of identically-prepared Bell states should be 1.0, got {}",
        f
    );
}

#[test]
fn fidelity_orthogonal_states_is_zero() {
    // |0><0| and |1><1| are orthogonal
    let dm0 = DensityMatrix::new(1);
    let mut dm1 = DensityMatrix::new(1);
    dm1.apply_gate(&Gate::X(0));

    let f = DensityMatrix::fidelity(&dm0, &dm1);
    assert!(
        f.abs() < TOL,
        "Fidelity of orthogonal states should be 0, got {}",
        f
    );
}

#[test]
fn fidelity_symmetric() {
    let dm0 = DensityMatrix::new(1);
    let mut dm1 = DensityMatrix::new(1);
    dm1.apply_gate(&Gate::H(0));

    let f01 = DensityMatrix::fidelity(&dm0, &dm1);
    let f10 = DensityMatrix::fidelity(&dm1, &dm0);
    assert!(
        (f01 - f10).abs() < TOL,
        "Fidelity should be symmetric: F(rho,sigma)={} != F(sigma,rho)={}",
        f01,
        f10
    );
}

#[test]
fn fidelity_between_zero_and_plus() {
    // F(|0><0|, |+><+|) = <0|+><+|0> = 0.5
    let dm0 = DensityMatrix::new(1);
    let mut dm_plus = DensityMatrix::new(1);
    dm_plus.apply_gate(&Gate::H(0));

    let f = DensityMatrix::fidelity(&dm0, &dm_plus);
    assert!(
        (f - 0.5).abs() < TOL,
        "Fidelity between |0> and |+> should be 0.5, got {}",
        f
    );
}

// ---------------------------------------------------------------------------
// Additional: from_statevector consistency
// ---------------------------------------------------------------------------

#[test]
fn from_statevector_matches_gate_application() {
    // Build the same state two ways and compare
    // Way 1: DensityMatrix with gate application
    let mut dm1 = DensityMatrix::new(2);
    dm1.apply_gate(&Gate::H(0));
    dm1.apply_gate(&Gate::CNOT(0, 1));

    // Way 2: Statevector then convert
    let mut sv = Statevector::new(2);
    Gate::H(0).apply(&mut sv).unwrap();
    Gate::CNOT(0, 1).apply(&mut sv).unwrap();
    let dm2 = DensityMatrix::from_statevector(&sv);

    let f = DensityMatrix::fidelity(&dm1, &dm2);
    assert!(
        (f - 1.0).abs() < TOL,
        "Density matrix built via gates vs from_statevector should have fidelity 1.0, got {}",
        f
    );
}

// ---------------------------------------------------------------------------
// Additional: dephasing channel
// ---------------------------------------------------------------------------

#[test]
fn dephasing_destroys_coherence() {
    // Start in |+> = H|0>, which has off-diagonal elements
    let mut dm = DensityMatrix::new(1);
    dm.apply_gate(&Gate::H(0));

    assert!(
        dm.get(0, 1).re.abs() > 0.1,
        "Should have coherence before dephasing"
    );

    let channel = dephasing_channel(1.0);
    dm.apply_single_qubit_channel(0, &channel);

    // Full dephasing should kill off-diagonal elements
    assert!(
        dm.get(0, 1).norm_sq() < TOL,
        "Full dephasing should destroy off-diagonal coherence, got {:?}",
        dm.get(0, 1)
    );
    // Populations unchanged
    assert!(
        (dm.get(0, 0).re - 0.5).abs() < TOL,
        "Dephasing should not change populations"
    );
    assert!(
        (dm.get(1, 1).re - 0.5).abs() < TOL,
        "Dephasing should not change populations"
    );
}
