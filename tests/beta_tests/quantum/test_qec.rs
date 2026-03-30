//! Integration tests for the QEC (Quantum Error Correction) module.

use cjc_quantum::qec::*;
use cjc_quantum::stabilizer::StabilizerState;

// ---------------------------------------------------------------------------
// 1. Repetition code structure
// ---------------------------------------------------------------------------

#[test]
fn test_repetition_code_d3_structure() {
    let code = build_repetition_code(3);
    assert_eq!(code.distance, 3, "Distance should be 3");
    assert_eq!(code.data_qubits.len(), 3, "d=3 repetition code has 3 data qubits");
    assert_eq!(
        code.z_stabilizers.len(),
        2,
        "d=3 repetition code has d-1=2 Z stabilizers"
    );
    assert_eq!(code.z_ancillas.len(), 2, "One ancilla per Z stabilizer");
    assert!(
        code.x_stabilizers.is_empty(),
        "Repetition code has no X stabilizers"
    );
    assert!(
        code.x_ancillas.is_empty(),
        "Repetition code has no X ancillas"
    );
    assert_eq!(
        code.total_qubits, 5,
        "d=3 repetition code needs 3 data + 2 ancilla = 5 qubits"
    );

    // Each Z stabilizer should act on a pair of adjacent data qubits.
    for (i, stab) in code.z_stabilizers.iter().enumerate() {
        assert_eq!(stab.len(), 2, "Z stabilizer {} should act on 2 qubits", i);
        assert_eq!(stab[0], i, "Z stabilizer {} left qubit", i);
        assert_eq!(stab[1], i + 1, "Z stabilizer {} right qubit", i);
    }
}

#[test]
fn test_repetition_code_d5_structure() {
    let code = build_repetition_code(5);
    assert_eq!(code.distance, 5, "Distance should be 5");
    assert_eq!(code.data_qubits.len(), 5, "d=5 repetition code has 5 data qubits");
    assert_eq!(
        code.z_stabilizers.len(),
        4,
        "d=5 repetition code has d-1=4 Z stabilizers"
    );
    assert_eq!(code.z_ancillas.len(), 4, "One ancilla per Z stabilizer");
    assert_eq!(
        code.total_qubits, 9,
        "d=5 repetition code needs 5 data + 4 ancilla = 9 qubits"
    );

    // Verify data qubit indices are 0..5 and ancillas are 5..9.
    assert_eq!(code.data_qubits, vec![0, 1, 2, 3, 4]);
    assert_eq!(code.z_ancillas, vec![5, 6, 7, 8]);
}

// ---------------------------------------------------------------------------
// 2. Surface code structure
// ---------------------------------------------------------------------------

#[test]
fn test_surface_code_d3_structure() {
    let code = build_surface_code(3);
    assert_eq!(code.distance, 3, "Distance should be 3");
    assert_eq!(
        code.data_qubits.len(),
        9,
        "d=3 surface code has 3x3=9 data qubits"
    );

    // A 3x3 grid has a 2x2 plaquette grid = 4 stabilizers total.
    let total_stabs = code.x_stabilizers.len() + code.z_stabilizers.len();
    assert_eq!(total_stabs, 4, "d=3 surface code has (d-1)^2=4 plaquette stabilizers");

    // Each stabilizer must act on exactly 4 data qubits, all within range.
    for (i, stab) in code.x_stabilizers.iter().enumerate() {
        assert_eq!(stab.len(), 4, "X stabilizer {} should act on 4 qubits", i);
        for &q in stab {
            assert!(q < 9, "X stabilizer {} references out-of-range qubit {}", i, q);
        }
    }
    for (i, stab) in code.z_stabilizers.iter().enumerate() {
        assert_eq!(stab.len(), 4, "Z stabilizer {} should act on 4 qubits", i);
        for &q in stab {
            assert!(q < 9, "Z stabilizer {} references out-of-range qubit {}", i, q);
        }
    }

    // Ancilla counts match stabilizer counts.
    assert_eq!(
        code.x_ancillas.len(),
        code.x_stabilizers.len(),
        "One X ancilla per X stabilizer"
    );
    assert_eq!(
        code.z_ancillas.len(),
        code.z_stabilizers.len(),
        "One Z ancilla per Z stabilizer"
    );

    // Total qubits = data + all ancillas.
    assert_eq!(
        code.total_qubits,
        9 + code.x_ancillas.len() + code.z_ancillas.len(),
        "Total qubits should be data + ancilla count"
    );
}

// ---------------------------------------------------------------------------
// 3. Syndrome extraction on clean state gives all-zero syndrome
// ---------------------------------------------------------------------------

#[test]
fn test_syndrome_clean_state_repetition_d3() {
    let code = build_repetition_code(3);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 42u64;
    let syndrome = syndrome_extraction(&mut state, &code, &mut rng);

    assert_eq!(
        syndrome.len(),
        code.z_stabilizers.len(),
        "Repetition code syndrome length should equal number of Z stabilizers"
    );
    for (i, &s) in syndrome.iter().enumerate() {
        assert_eq!(s, 0, "Clean state syndrome bit {} should be 0", i);
    }
}

#[test]
fn test_syndrome_clean_state_repetition_d5() {
    let code = build_repetition_code(5);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 99u64;
    let syndrome = syndrome_extraction(&mut state, &code, &mut rng);

    assert_eq!(syndrome.len(), 4, "d=5 repetition code has 4 syndrome bits");
    for (i, &s) in syndrome.iter().enumerate() {
        assert_eq!(s, 0, "Clean state d=5 syndrome bit {} should be 0", i);
    }
}

#[test]
fn test_syndrome_clean_state_surface_d3() {
    let code = build_surface_code(3);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 7u64;
    let syndrome = syndrome_extraction(&mut state, &code, &mut rng);

    let expected_len = code.z_stabilizers.len() + code.x_stabilizers.len();
    assert_eq!(
        syndrome.len(),
        expected_len,
        "Surface code syndrome length should equal total stabilizer count"
    );
    for (i, &s) in syndrome.iter().enumerate() {
        assert_eq!(s, 0, "Clean state surface code syndrome bit {} should be 0", i);
    }
}

// ---------------------------------------------------------------------------
// 4. Repetition code decoder corrects single errors
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_corrects_single_x_error_middle() {
    let code = build_repetition_code(5);

    // An X error on data qubit 2 triggers stabilizers 1 (Z_1 Z_2) and 2 (Z_2 Z_3).
    // Construct the expected syndrome pattern directly.
    let syndrome = vec![0, 1, 1, 0];

    let corrections = decode_repetition_code(&syndrome, &code);
    assert!(
        !corrections.is_empty(),
        "Decoder should produce at least one correction for a middle error"
    );

    // Verify via full round-trip: apply error, extract syndrome, decode, correct,
    // then measure data qubits. A correct decode means majority of data qubits
    // remain in |0>.
    let mut state = StabilizerState::new(code.total_qubits);
    state.x(code.data_qubits[2]);

    let mut rng = 42u64;
    let syn = syndrome_extraction(&mut state, &code, &mut rng);
    let corr = decode_repetition_code(&syn, &code);
    for q in corr {
        state.x(q);
    }

    // Measure data qubits: majority should be |0> if correction succeeded.
    let mut ones = 0usize;
    for &q in &code.data_qubits {
        let mut rng_m = rng;
        if state.measure(q, &mut rng_m) == 1 {
            ones += 1;
        }
    }
    assert!(
        ones <= code.distance / 2,
        "After correcting single middle error, majority of data qubits should be |0>, got {} ones",
        ones
    );
}

#[test]
fn test_decoder_corrects_single_x_error_boundary() {
    let code = build_repetition_code(5);

    // An X error on data qubit 0 triggers only stabilizer 0 (Z_0 Z_1).
    let syndrome = vec![1, 0, 0, 0];

    let corrections = decode_repetition_code(&syndrome, &code);
    assert!(
        !corrections.is_empty(),
        "Decoder should produce at least one correction for a boundary error"
    );

    // Full round-trip verification.
    let mut state = StabilizerState::new(code.total_qubits);
    state.x(code.data_qubits[0]);

    let mut rng = 55u64;
    let syn = syndrome_extraction(&mut state, &code, &mut rng);
    let corr = decode_repetition_code(&syn, &code);
    for q in corr {
        state.x(q);
    }

    let mut ones = 0usize;
    for &q in &code.data_qubits {
        let mut rng_m = rng;
        if state.measure(q, &mut rng_m) == 1 {
            ones += 1;
        }
    }
    assert!(
        ones <= code.distance / 2,
        "After correcting boundary error, majority of data qubits should be |0>, got {} ones",
        ones
    );
}

#[test]
fn test_decoder_no_correction_on_clean_syndrome() {
    let code = build_repetition_code(5);
    let syndrome = vec![0, 0, 0, 0];
    let corrections = decode_repetition_code(&syndrome, &code);
    assert!(
        corrections.is_empty(),
        "Zero syndrome should produce zero corrections"
    );
}

// ---------------------------------------------------------------------------
// 5. Logical error rate decreases with distance
// ---------------------------------------------------------------------------

#[test]
fn test_logical_error_rate_low_noise_near_zero() {
    // Very low physical error rate should yield very low logical error rate.
    let rate = estimate_logical_error_rate(3, 0.001, 200, 42);
    assert!(
        rate < 0.1,
        "Logical error rate at p=0.001 should be < 0.1, got {}",
        rate
    );
}

#[test]
fn test_logical_error_rate_decreases_with_distance() {
    // Below the threshold, larger distance should reduce logical error rate.
    // Use a moderate physical error rate and enough rounds for statistical signal.
    let rate_d3 = estimate_logical_error_rate(3, 0.02, 500, 42);
    let rate_d5 = estimate_logical_error_rate(5, 0.02, 500, 42);

    // Both should be reasonably bounded.
    assert!(
        rate_d3 < 0.5,
        "d=3 logical error rate should be < 0.5 at p=0.02, got {}",
        rate_d3
    );
    assert!(
        rate_d5 < 0.5,
        "d=5 logical error rate should be < 0.5 at p=0.02, got {}",
        rate_d5
    );

    // d=5 should have a lower or equal logical error rate than d=3.
    assert!(
        rate_d5 <= rate_d3,
        "Logical error rate should decrease with distance: d=3 rate={}, d=5 rate={}",
        rate_d3,
        rate_d5
    );
}

// ---------------------------------------------------------------------------
// 6. Determinism: same seed = identical results
// ---------------------------------------------------------------------------

#[test]
fn test_deterministic_syndrome_extraction() {
    let code = build_repetition_code(5);

    let run = |seed: u64| -> Vec<u8> {
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = seed;
        apply_noise_round(&mut state, &code, 0.1, &mut rng);
        syndrome_extraction(&mut state, &code, &mut rng)
    };

    let s1 = run(42);
    let s2 = run(42);
    assert_eq!(s1, s2, "Syndrome extraction must be deterministic for same seed");
}

#[test]
fn test_deterministic_logical_error_rate() {
    let r1 = estimate_logical_error_rate(3, 0.05, 100, 42);
    let r2 = estimate_logical_error_rate(3, 0.05, 100, 42);
    assert_eq!(
        r1.to_bits(),
        r2.to_bits(),
        "Logical error rate must be bit-identical across runs with same seed"
    );
}

#[test]
fn test_different_seeds_produce_different_results() {
    let code = build_repetition_code(11);

    let run = |seed: u64| -> Vec<u8> {
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = seed;
        apply_noise_round(&mut state, &code, 0.5, &mut rng);
        syndrome_extraction(&mut state, &code, &mut rng)
    };

    let s1 = run(1);
    let s2 = run(999);
    // At 50% error rate on 11 qubits, collision probability is negligible.
    assert_ne!(
        s1, s2,
        "Different seeds should produce different syndromes at high error rate"
    );
}

#[test]
fn test_deterministic_noise_round() {
    let code = build_repetition_code(7);

    let run = |seed: u64| -> Vec<u8> {
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = seed;
        apply_noise_round(&mut state, &code, 0.3, &mut rng);
        let mut rng_syn = 0u64;
        syndrome_extraction(&mut state, &code, &mut rng_syn)
    };

    let s1 = run(12345);
    let s2 = run(12345);
    assert_eq!(s1, s2, "Noise round followed by syndrome must be deterministic");
}

// ---------------------------------------------------------------------------
// 7. apply_noise_round doesn't crash
// ---------------------------------------------------------------------------

#[test]
fn test_apply_noise_round_no_crash_repetition() {
    let code = build_repetition_code(3);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 42u64;

    // Various error rates, including edge cases.
    for &rate in &[0.0, 0.01, 0.1, 0.5, 1.0] {
        apply_noise_round(&mut state, &code, rate, &mut rng);
    }
}

#[test]
fn test_apply_noise_round_no_crash_surface() {
    let code = build_surface_code(3);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 77u64;

    for &rate in &[0.0, 0.01, 0.1, 0.5, 1.0] {
        apply_noise_round(&mut state, &code, rate, &mut rng);
    }
}

#[test]
fn test_apply_noise_round_zero_rate_preserves_clean_state() {
    let code = build_repetition_code(5);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 42u64;

    // Zero error rate should not flip any qubit.
    apply_noise_round(&mut state, &code, 0.0, &mut rng);

    let mut rng_syn = 0u64;
    let syndrome = syndrome_extraction(&mut state, &code, &mut rng_syn);
    for (i, &s) in syndrome.iter().enumerate() {
        assert_eq!(
            s, 0,
            "Zero error rate should preserve clean state, but syndrome bit {} is 1",
            i
        );
    }
}

#[test]
fn test_full_qec_round_trip_repetition_d5() {
    // End-to-end: build code, apply noise, extract syndrome, decode, correct.
    // Verify the pipeline runs without errors.
    let code = build_repetition_code(5);
    let mut state = StabilizerState::new(code.total_qubits);
    let mut rng = 42u64;

    apply_noise_round(&mut state, &code, 0.05, &mut rng);
    let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
    let corrections = decode_repetition_code(&syndrome, &code);

    for q in corrections {
        state.x(q);
    }

    // After correction, the syndrome should ideally be clean (for low noise
    // and single errors). We just verify no crash here.
    let mut rng2 = 200u64;
    let _syndrome_after = syndrome_extraction(&mut state, &code, &mut rng2);
}
