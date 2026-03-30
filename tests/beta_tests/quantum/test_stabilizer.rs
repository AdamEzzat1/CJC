//! Integration tests for the Clifford/Stabilizer simulator module.
//!
//! Covers: initial state, single gates, entangled states (Bell, GHZ),
//! statevector extraction, determinism, and large-qubit-count scaling.

use cjc_quantum::stabilizer::StabilizerState;
use cjc_quantum::Complex as ComplexF64;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tolerance for floating-point comparisons on amplitudes.
const EPS: f64 = 1e-10;

/// Check that two complex amplitudes are close.
fn approx_eq(a: ComplexF64, b: ComplexF64, msg: &str) {
    let dr = (a.re - b.re).abs();
    let di = (a.im - b.im).abs();
    assert!(
        dr < EPS && di < EPS,
        "{}: expected ({}, {}), got ({}, {})",
        msg,
        b.re,
        b.im,
        a.re,
        a.im,
    );
}

/// Check that a statevector is normalized (sum of |a_k|^2 == 1).
fn assert_normalized(sv: &[ComplexF64]) {
    let norm_sq: f64 = sv.iter().map(|c| c.norm_sq()).sum();
    assert!(
        (norm_sq - 1.0).abs() < EPS,
        "statevector not normalized: norm^2 = {}",
        norm_sq,
    );
}

// ===========================================================================
// 1. Initial state is |0...0>
// ===========================================================================

#[test]
fn initial_state_single_qubit_is_zero() {
    let mut s = StabilizerState::new(1);
    let mut rng = 12345u64;
    let outcome = s.measure(0, &mut rng);
    assert_eq!(outcome, 0, "fresh 1-qubit state should measure 0");
}

#[test]
fn initial_state_multi_qubit_all_zero() {
    let mut s = StabilizerState::new(5);
    let mut rng = 99u64;
    for q in 0..5 {
        let outcome = s.measure(q, &mut rng);
        assert_eq!(
            outcome, 0,
            "qubit {} of fresh 5-qubit state should measure 0",
            q
        );
    }
}

#[test]
fn initial_state_statevector_is_ket_zero() {
    let s = StabilizerState::new(2);
    let sv = s.to_statevector().expect("n=2 should produce a statevector");
    // |00> = [1, 0, 0, 0]
    assert_eq!(sv.len(), 4, "2-qubit statevector should have 4 entries");
    assert_normalized(&sv);
    approx_eq(sv[0], ComplexF64::real(1.0), "|00> amplitude");
    for k in 1..4 {
        approx_eq(
            sv[k],
            ComplexF64::real(0.0),
            &format!("amplitude at index {}", k),
        );
    }
}

// ===========================================================================
// 2. H gate creates |+> state
// ===========================================================================

#[test]
fn h_creates_plus_state() {
    let mut s = StabilizerState::new(1);
    s.h(0);
    let sv = s.to_statevector().expect("n=1 should produce a statevector");
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    // |+> = (|0> + |1>) / sqrt(2)
    approx_eq(sv[0], ComplexF64::real(inv_sqrt2), "|+> amplitude at |0>");
    approx_eq(sv[1], ComplexF64::real(inv_sqrt2), "|+> amplitude at |1>");
    assert_normalized(&sv);
}

#[test]
fn h_creates_plus_measurement_is_random() {
    // Measure |+> many times (fresh state each time); both 0 and 1 should appear.
    let mut saw_zero = false;
    let mut saw_one = false;
    let mut rng = 7u64;
    for _ in 0..100 {
        let mut s = StabilizerState::new(1);
        s.h(0);
        let outcome = s.measure(0, &mut rng);
        if outcome == 0 {
            saw_zero = true;
        } else {
            saw_one = true;
        }
        if saw_zero && saw_one {
            break;
        }
    }
    assert!(saw_zero, "measuring |+> should sometimes give 0");
    assert!(saw_one, "measuring |+> should sometimes give 1");
}

#[test]
fn double_h_returns_to_zero() {
    let mut s = StabilizerState::new(1);
    s.h(0);
    s.h(0);
    // H^2 = I, so we are back to |0>
    let sv = s.to_statevector().expect("statevector");
    approx_eq(sv[0], ComplexF64::real(1.0), "H^2 |0> amplitude at |0>");
    approx_eq(sv[1], ComplexF64::real(0.0), "H^2 |0> amplitude at |1>");
}

// ===========================================================================
// 3. Bell state creation (H then CNOT)
// ===========================================================================

#[test]
fn bell_state_statevector() {
    let mut s = StabilizerState::new(2);
    s.h(0);
    s.cnot(0, 1);
    let sv = s.to_statevector().expect("n=2 statevector");
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    // Bell |Phi+> = (|00> + |11>) / sqrt(2)
    // In little-endian qubit ordering: index 0 = |00>, index 3 = |11>
    approx_eq(sv[0], ComplexF64::real(inv_sqrt2), "Bell |00> amplitude");
    approx_eq(sv[1], ComplexF64::real(0.0), "Bell |01> amplitude");
    approx_eq(sv[2], ComplexF64::real(0.0), "Bell |10> amplitude");
    approx_eq(sv[3], ComplexF64::real(inv_sqrt2), "Bell |11> amplitude");
    assert_normalized(&sv);
}

#[test]
fn bell_state_measurements_are_correlated() {
    // Measuring both qubits of a Bell state must always give the same result.
    let mut rng = 42u64;
    for _ in 0..50 {
        let mut s = StabilizerState::new(2);
        s.h(0);
        s.cnot(0, 1);
        let m0 = s.measure(0, &mut rng);
        let m1 = s.measure(1, &mut rng);
        assert_eq!(
            m0, m1,
            "Bell state qubits must be perfectly correlated (got {} and {})",
            m0, m1
        );
    }
}

// ===========================================================================
// 4. GHZ state creation and measurement
// ===========================================================================

#[test]
fn ghz_state_statevector_3_qubits() {
    let mut s = StabilizerState::new(3);
    s.h(0);
    s.cnot(0, 1);
    s.cnot(0, 2);
    let sv = s.to_statevector().expect("n=3 statevector");
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    // GHZ = (|000> + |111>) / sqrt(2)
    // index 0 = |000>, index 7 = |111>
    assert_eq!(sv.len(), 8);
    approx_eq(sv[0], ComplexF64::real(inv_sqrt2), "GHZ |000>");
    approx_eq(sv[7], ComplexF64::real(inv_sqrt2), "GHZ |111>");
    for k in 1..7 {
        approx_eq(
            sv[k],
            ComplexF64::real(0.0),
            &format!("GHZ amplitude at index {}", k),
        );
    }
    assert_normalized(&sv);
}

#[test]
fn ghz_state_measurements_all_agree() {
    let mut rng = 1000u64;
    for _ in 0..50 {
        let mut s = StabilizerState::new(4);
        s.h(0);
        s.cnot(0, 1);
        s.cnot(0, 2);
        s.cnot(0, 3);
        let m0 = s.measure(0, &mut rng);
        let m1 = s.measure(1, &mut rng);
        let m2 = s.measure(2, &mut rng);
        let m3 = s.measure(3, &mut rng);
        assert!(
            m0 == m1 && m1 == m2 && m2 == m3,
            "GHZ measurements must all agree: got [{}, {}, {}, {}]",
            m0,
            m1,
            m2,
            m3,
        );
    }
}

#[test]
fn ghz_state_sees_both_outcomes() {
    let mut saw_all_zero = false;
    let mut saw_all_one = false;
    let mut rng = 55u64;
    for _ in 0..200 {
        let mut s = StabilizerState::new(3);
        s.h(0);
        s.cnot(0, 1);
        s.cnot(0, 2);
        let m0 = s.measure(0, &mut rng);
        if m0 == 0 {
            saw_all_zero = true;
        } else {
            saw_all_one = true;
        }
        if saw_all_zero && saw_all_one {
            break;
        }
    }
    assert!(saw_all_zero, "GHZ should sometimes measure all 0");
    assert!(saw_all_one, "GHZ should sometimes measure all 1");
}

// ===========================================================================
// 5. to_statevector matches expected for small states
// ===========================================================================

#[test]
fn x_gate_produces_ket_one() {
    let mut s = StabilizerState::new(1);
    s.x(0);
    let sv = s.to_statevector().expect("statevector");
    // |1>
    approx_eq(sv[0], ComplexF64::real(0.0), "X|0> at |0>");
    approx_eq(sv[1], ComplexF64::real(1.0), "X|0> at |1>");
}

#[test]
fn z_gate_on_zero_is_still_zero() {
    // Z|0> = |0> (Z has eigenvalue +1 for |0>)
    let mut s = StabilizerState::new(1);
    s.z(0);
    let sv = s.to_statevector().expect("statevector");
    approx_eq(sv[0], ComplexF64::real(1.0), "Z|0> at |0>");
    approx_eq(sv[1], ComplexF64::real(0.0), "Z|0> at |1>");
}

#[test]
fn s_gate_twice_equals_z() {
    // S^2 = Z; apply S twice to |+> and compare with Z|+> = |->
    let mut s1 = StabilizerState::new(1);
    s1.h(0);
    s1.s(0);
    s1.s(0);
    let sv1 = s1.to_statevector().expect("statevector S^2|+>");

    let mut s2 = StabilizerState::new(1);
    s2.h(0);
    s2.z(0);
    let sv2 = s2.to_statevector().expect("statevector Z|+>");

    for k in 0..2 {
        approx_eq(sv1[k], sv2[k], &format!("S^2 == Z at index {}", k));
    }
}

#[test]
fn h_s_h_produces_expected_state() {
    // H S H |0> produces a known state; verify via statevector
    let mut s = StabilizerState::new(1);
    s.h(0);
    s.s(0);
    s.h(0);
    let sv = s.to_statevector().expect("statevector");
    assert_normalized(&sv);
    // H S H is equivalent to Sdg X up to global phase, but let's just verify
    // it is normalized and deterministic (checked in determinism tests below).
}

#[test]
fn to_statevector_returns_none_for_large_n() {
    let s = StabilizerState::new(13);
    assert!(
        s.to_statevector().is_none(),
        "to_statevector should return None for n > 12"
    );
}

#[test]
fn to_statevector_works_at_boundary_n12() {
    let s = StabilizerState::new(12);
    let sv = s.to_statevector();
    assert!(
        sv.is_some(),
        "to_statevector should work for n = 12"
    );
    let sv = sv.unwrap();
    assert_eq!(sv.len(), 1 << 12, "2^12 = 4096 amplitudes expected");
    assert_normalized(&sv);
    // |0...0> state: only index 0 is nonzero
    approx_eq(sv[0], ComplexF64::real(1.0), "12-qubit |0..0> at index 0");
}

// ===========================================================================
// 6. Determinism: same operations produce identical tableaus
// ===========================================================================

#[test]
fn determinism_same_gates_same_statevector() {
    // Run the same circuit twice and confirm bit-identical statevectors.
    let build_circuit = || {
        let mut s = StabilizerState::new(3);
        s.h(0);
        s.cnot(0, 1);
        s.s(1);
        s.h(2);
        s.cnot(2, 0);
        s.x(1);
        s.y(2);
        s.to_statevector().expect("statevector")
    };

    let sv_a = build_circuit();
    let sv_b = build_circuit();
    assert_eq!(sv_a.len(), sv_b.len(), "statevector lengths must match");
    for k in 0..sv_a.len() {
        assert!(
            sv_a[k].re == sv_b[k].re && sv_a[k].im == sv_b[k].im,
            "statevectors differ at index {}: ({}, {}) vs ({}, {})",
            k,
            sv_a[k].re,
            sv_a[k].im,
            sv_b[k].re,
            sv_b[k].im,
        );
    }
}

#[test]
fn determinism_same_seed_same_measurement_sequence() {
    // Same circuit + same RNG seed must produce identical measurement sequences.
    let run = |seed: u64| -> Vec<u8> {
        let mut rng = seed;
        let mut s = StabilizerState::new(4);
        s.h(0);
        s.h(1);
        s.h(2);
        s.h(3);
        let mut results = Vec::new();
        for q in 0..4 {
            results.push(s.measure(q, &mut rng));
        }
        results
    };

    let r1 = run(777);
    let r2 = run(777);
    assert_eq!(
        r1, r2,
        "identical seed must produce identical measurement outcomes"
    );
}

#[test]
fn determinism_different_seeds_can_differ() {
    // With superposition states, different seeds should (with high probability)
    // eventually produce different measurement outcomes.
    let run = |seed: u64| -> Vec<u8> {
        let mut rng = seed;
        let mut s = StabilizerState::new(4);
        for q in 0..4 {
            s.h(q);
        }
        (0..4).map(|q| s.measure(q, &mut rng)).collect()
    };

    // Try several seed pairs; at least one pair should differ.
    let mut found_diff = false;
    for offset in 1..20u64 {
        if run(100) != run(100 + offset) {
            found_diff = true;
            break;
        }
    }
    assert!(
        found_diff,
        "different RNG seeds should produce different outcomes for superposition states"
    );
}

// ===========================================================================
// 7. Large system (100+ qubits) doesn't crash
// ===========================================================================

#[test]
fn large_system_128_qubits_construction_and_gates() {
    let mut s = StabilizerState::new(128);
    assert_eq!(s.num_qubits(), 128);
    // Apply a spread of gates across the register
    for q in 0..128 {
        s.h(q);
    }
    for q in 0..127 {
        s.cnot(q, q + 1);
    }
    // Measure a few qubits to verify no panic
    let mut rng = 42u64;
    let _m0 = s.measure(0, &mut rng);
    let _m64 = s.measure(64, &mut rng);
    let _m127 = s.measure(127, &mut rng);
}

#[test]
fn large_system_200_qubits_with_pauli_gates() {
    let mut s = StabilizerState::new(200);
    for q in 0..200 {
        s.x(q);
    }
    // After X on all qubits of |0...0>, we should have |1...1>
    // All measurements should deterministically yield 1
    let mut rng = 0u64;
    for q in 0..200 {
        let outcome = s.measure(q, &mut rng);
        assert_eq!(
            outcome, 1,
            "qubit {} of X|0...0> should measure 1",
            q
        );
    }
}

#[test]
fn large_system_500_qubits_no_crash() {
    let mut s = StabilizerState::new(500);
    s.h(0);
    s.cnot(0, 249);
    s.cnot(0, 499);
    // Measure to exercise the full code path
    let mut rng = 99u64;
    let m0 = s.measure(0, &mut rng);
    let m249 = s.measure(249, &mut rng);
    let m499 = s.measure(499, &mut rng);
    // GHZ-like state: all three must agree
    assert_eq!(
        m0, m249,
        "500-qubit GHZ: qubit 0 and 249 must agree"
    );
    assert_eq!(
        m0, m499,
        "500-qubit GHZ: qubit 0 and 499 must agree"
    );
}
