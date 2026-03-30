// CJC v0.2 Beta — Quantum Property-Based Tests
//
// These tests verify determinism, unitarity, and statistical correctness
// of the quantum simulation primitives.

use cjc_quantum::{Circuit, Statevector, Gate};
use cjc_quantum::measure::{measure_qubit, measure_all, sample_basis_state};
use cjc_repro::KahanAccumulatorF64;

const TOL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Property: Determinism — same seed = identical results
// ---------------------------------------------------------------------------

#[test]
fn prop_quantum_circuit_deterministic_10_runs() {
    // Execute the same circuit 10 times, verify bit-identical amplitudes
    for run in 0..10 {
        let mut circ = Circuit::new(3);
        circ.h(0).cnot(0, 1).t(2).ry(1, 0.7).cz(1, 2);
        let sv = circ.execute().unwrap();

        let mut circ2 = Circuit::new(3);
        circ2.h(0).cnot(0, 1).t(2).ry(1, 0.7).cz(1, 2);
        let sv2 = circ2.execute().unwrap();

        for i in 0..sv.n_states() {
            assert_eq!(
                sv.amplitudes[i].re.to_bits(),
                sv2.amplitudes[i].re.to_bits(),
                "Determinism failure at run {}, state {}, re", run, i
            );
            assert_eq!(
                sv.amplitudes[i].im.to_bits(),
                sv2.amplitudes[i].im.to_bits(),
                "Determinism failure at run {}, state {}, im", run, i
            );
        }
    }
}

#[test]
fn prop_measurement_deterministic_same_seed() {
    // Same seed must produce identical measurement outcomes
    for seed in 0..50u64 {
        let mut circ = Circuit::new(3);
        circ.h(0).h(1).h(2); // equal superposition over all 8 states

        let mut rng1 = seed;
        let mut rng2 = seed;
        let (o1, _) = circ.execute_and_measure(&mut rng1).unwrap();
        let (o2, _) = circ.execute_and_measure(&mut rng2).unwrap();
        assert_eq!(o1, o2, "Measurement determinism failure at seed {}", seed);
    }
}

#[test]
fn prop_sampling_deterministic_same_seed() {
    let mut circ = Circuit::new(2);
    circ.h(0).cnot(0, 1);

    for seed in 0..20u64 {
        let mut rng1 = seed;
        let mut rng2 = seed;
        let s1 = circ.sample(100, &mut rng1).unwrap();
        let s2 = circ.sample(100, &mut rng2).unwrap();
        assert_eq!(s1, s2, "Sampling determinism failure at seed {}", seed);
    }
}

// ---------------------------------------------------------------------------
// Property: Unitarity — all gates preserve normalization
// ---------------------------------------------------------------------------

#[test]
fn prop_unitarity_random_gate_sequences() {
    // Generate various gate sequences, verify norm is preserved
    let gate_sequences: Vec<Vec<Gate>> = vec![
        vec![Gate::H(0), Gate::X(1), Gate::CNOT(0, 1)],
        vec![Gate::Ry(0, 0.3), Gate::Rz(1, 1.7), Gate::CZ(0, 1)],
        vec![Gate::H(0), Gate::H(1), Gate::H(2), Gate::Toffoli(0, 1, 2)],
        vec![Gate::X(0), Gate::Y(1), Gate::Z(2), Gate::SWAP(0, 2)],
        vec![Gate::S(0), Gate::T(1), Gate::Rx(2, 2.1), Gate::CNOT(1, 2)],
        vec![Gate::H(0), Gate::CNOT(0, 1), Gate::CNOT(1, 2), Gate::H(2)],
        // 10 Hadamard gates in sequence
        vec![Gate::H(0); 10],
        // Deep entangling circuit
        vec![
            Gate::H(0), Gate::H(1), Gate::H(2), Gate::H(3),
            Gate::CNOT(0, 1), Gate::CNOT(1, 2), Gate::CNOT(2, 3),
            Gate::Ry(0, 0.5), Gate::Rz(3, 1.2),
        ],
    ];

    for (idx, gates) in gate_sequences.iter().enumerate() {
        let n_qubits = gates.iter()
            .flat_map(|g| g.qubits())
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        let mut sv = Statevector::new(n_qubits);
        for gate in gates {
            gate.apply(&mut sv).unwrap();
        }
        assert!(sv.is_normalized(TOL),
            "Unitarity violation in sequence {}: norm deviation", idx);
    }
}

// ---------------------------------------------------------------------------
// Property: Involutions — self-inverse gates
// ---------------------------------------------------------------------------

#[test]
fn prop_involution_gates() {
    // X², Y², Z², H², SWAP² should all be identity
    let involution_pairs: Vec<(Gate, Gate, usize)> = vec![
        (Gate::X(0), Gate::X(0), 1),
        (Gate::Y(0), Gate::Y(0), 1),
        (Gate::Z(0), Gate::Z(0), 1),
        (Gate::H(0), Gate::H(0), 1),
        (Gate::SWAP(0, 1), Gate::SWAP(0, 1), 2),
    ];

    for (g1, g2, n_qubits) in involution_pairs {
        let mut sv = Statevector::new(n_qubits);
        // Start with non-trivial state
        Gate::H(0).apply(&mut sv).unwrap();

        let before = sv.amplitudes.clone();
        g1.apply(&mut sv).unwrap();
        g2.apply(&mut sv).unwrap();

        for i in 0..sv.n_states() {
            assert!(
                (sv.amplitudes[i].re - before[i].re).abs() < TOL &&
                (sv.amplitudes[i].im - before[i].im).abs() < TOL,
                "Involution failure for {:?} at state {}", g1, i
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property: Bell state correlations
// ---------------------------------------------------------------------------

#[test]
fn prop_bell_state_correlations_1000_shots() {
    let mut circ = Circuit::new(2);
    circ.h(0).cnot(0, 1);

    let mut correlated = 0;
    let n_shots = 1000;

    for seed in 0..n_shots as u64 {
        let mut rng = seed;
        let (outcomes, _) = circ.execute_and_measure(&mut rng).unwrap();
        if outcomes[0] == outcomes[1] {
            correlated += 1;
        }
    }

    // Bell state: qubits MUST always be correlated
    assert_eq!(correlated, n_shots,
        "Bell state: all {} measurements must be correlated, got {}", n_shots, correlated);
}

// ---------------------------------------------------------------------------
// Property: Probability normalization via Kahan summation
// ---------------------------------------------------------------------------

#[test]
fn prop_probability_sum_equals_one() {
    // For various circuits, verify sum of probabilities = 1.0 (Kahan)
    let circuits: Vec<Circuit> = {
        let mut v = Vec::new();

        let mut c1 = Circuit::new(4);
        c1.h(0).h(1).h(2).h(3); // equal superposition over 16 states
        v.push(c1);

        let mut c2 = Circuit::new(3);
        c2.h(0).cnot(0, 1).cnot(0, 2); // GHZ
        v.push(c2);

        let mut c3 = Circuit::new(5);
        c3.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3).cnot(3, 4);
        c3.ry(0, 0.3).rz(2, 1.7).rx(4, 0.9);
        v.push(c3);

        v
    };

    for (idx, circ) in circuits.iter().enumerate() {
        let sv = circ.execute().unwrap();
        let probs = sv.probabilities();

        let mut acc = KahanAccumulatorF64::new();
        for &p in &probs {
            acc.add(p);
        }
        let total = acc.finalize();
        assert!((total - 1.0).abs() < TOL,
            "Probability sum not 1.0 for circuit {}: got {}", idx, total);
    }
}

// ---------------------------------------------------------------------------
// Property: H|0⟩ measurement statistics (χ² test)
// ---------------------------------------------------------------------------

#[test]
fn prop_h_gate_statistics_chi_squared() {
    let mut count = [0usize; 2]; // count[0] = times measured 0, count[1] = times measured 1
    let n = 2000;

    for seed in 0..n as u64 {
        let mut sv = Statevector::new(1);
        Gate::H(0).apply(&mut sv).unwrap();
        let mut rng = seed;
        let outcome = measure_qubit(&mut sv, 0, &mut rng).unwrap();
        count[outcome as usize] += 1;
    }

    // Chi-squared test: expected 50/50
    let expected = n as f64 / 2.0;
    let chi2 = (count[0] as f64 - expected).powi(2) / expected
             + (count[1] as f64 - expected).powi(2) / expected;

    // Chi-squared critical value at p=0.001 with df=1 is 10.83
    assert!(chi2 < 10.83,
        "H|0⟩ measurement statistics failed chi² test: χ²={:.2}, counts=[{}, {}]",
        chi2, count[0], count[1]);
}

// ---------------------------------------------------------------------------
// Property: Different seeds produce different results (non-degeneracy)
// ---------------------------------------------------------------------------

#[test]
fn prop_different_seeds_differ() {
    let mut circ = Circuit::new(2);
    circ.h(0).cnot(0, 1);

    let mut results: Vec<Vec<u8>> = Vec::new();
    for seed in 0..100u64 {
        let mut rng = seed;
        let (outcomes, _) = circ.execute_and_measure(&mut rng).unwrap();
        results.push(outcomes);
    }

    // Not all results should be identical (with overwhelming probability)
    let first = &results[0];
    let all_same = results.iter().all(|r| r == first);
    assert!(!all_same, "100 different seeds should not all produce the same outcome");
}

// ---------------------------------------------------------------------------
// Property: GHZ measurement — all qubits agree
// ---------------------------------------------------------------------------

#[test]
fn prop_ghz_all_qubits_agree() {
    for n in 2..=5 {
        let mut circ = Circuit::new(n);
        circ.h(0);
        for i in 1..n {
            circ.cnot(0, i);
        }

        for seed in 0..100u64 {
            let mut rng = seed;
            let (outcomes, _) = circ.execute_and_measure(&mut rng).unwrap();
            let first = outcomes[0];
            for (i, &o) in outcomes.iter().enumerate() {
                assert_eq!(o, first,
                    "GHZ({}) qubits must agree: qubit {} = {}, qubit 0 = {} (seed {})",
                    n, i, o, first, seed);
            }
        }
    }
}
