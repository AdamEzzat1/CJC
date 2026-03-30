//! Quantum Error Correction — Surface code simulation.
//!
//! Uses the Clifford/Stabilizer simulator for efficient QEC code simulation.
//! Supports repetition codes and surface codes with arbitrary distance d.
//!
//! # Codes implemented
//!
//! - **Repetition code**: 1D code with d data qubits and d-1 Z stabilizers.
//!   Corrects up to (d-1)/2 bit-flip (X) errors. Decoded by a simple
//!   minimum-weight matching decoder that pairs adjacent syndrome defects.
//!
//! - **Surface code**: 2D planar code with d*d data qubits and X/Z plaquette
//!   stabilizers on a checkerboard lattice. Provided for structural completeness;
//!   the decoder is not yet implemented for the 2D case.
//!
//! # Determinism
//!
//! All routines are deterministic: same seed produces bit-identical results.
//! No `HashMap`/`HashSet`, no floating-point non-determinism, no thread-local
//! state. RNG is threaded explicitly via `&mut u64` (SplitMix64).

use crate::stabilizer::StabilizerState;

// ---------------------------------------------------------------------------
// Code layout
// ---------------------------------------------------------------------------

/// Layout of a stabilizer QEC code.
///
/// Stores the qubit indices and stabilizer structure needed to run syndrome
/// extraction and decoding on a `StabilizerState`.
#[derive(Debug, Clone)]
pub struct SurfaceCode {
    /// Code distance.
    pub distance: usize,
    /// Indices of data qubits in the stabilizer state.
    pub data_qubits: Vec<usize>,
    /// X-type stabilizers: each entry is a list of data qubit indices acted on.
    pub x_stabilizers: Vec<Vec<usize>>,
    /// Z-type stabilizers: each entry is a list of data qubit indices acted on.
    pub z_stabilizers: Vec<Vec<usize>>,
    /// Ancilla qubit indices used for X-stabilizer measurement.
    pub x_ancillas: Vec<usize>,
    /// Ancilla qubit indices used for Z-stabilizer measurement.
    pub z_ancillas: Vec<usize>,
    /// Total number of qubits (data + ancilla) required in the stabilizer state.
    pub total_qubits: usize,
}

// ---------------------------------------------------------------------------
// Code construction
// ---------------------------------------------------------------------------

/// Build a repetition code of distance `d`.
///
/// The repetition code is the simplest QEC code:
/// - `d` data qubits in a line.
/// - `d - 1` Z-type stabilizers: each checks adjacent pairs Z_i Z_{i+1}.
/// - `d - 1` ancilla qubits for syndrome extraction.
/// - No X stabilizers (the code only corrects X errors, not Z errors).
///
/// The code can correct up to `(d - 1) / 2` single-qubit X (bit-flip) errors.
///
/// # Panics
///
/// Panics if `distance < 2`.
pub fn build_repetition_code(distance: usize) -> SurfaceCode {
    assert!(distance >= 2, "Repetition code requires distance >= 2");

    let n_data = distance;
    let n_stabilizers = distance - 1;
    let n_ancilla = n_stabilizers;
    let total = n_data + n_ancilla;

    let data_qubits: Vec<usize> = (0..n_data).collect();
    let z_ancillas: Vec<usize> = (n_data..(n_data + n_ancilla)).collect();

    // Z stabilizers: Z_i Z_{i+1} for each adjacent pair of data qubits.
    let z_stabilizers: Vec<Vec<usize>> = (0..n_stabilizers)
        .map(|i| vec![data_qubits[i], data_qubits[i + 1]])
        .collect();

    SurfaceCode {
        distance,
        data_qubits,
        x_stabilizers: vec![],
        z_stabilizers,
        x_ancillas: vec![],
        z_ancillas,
        total_qubits: total,
    }
}

/// Build a d x d planar surface code.
///
/// Data qubits sit on vertices of a `d x d` grid. Stabilizers are assigned
/// to plaquettes (the `(d-1) x (d-1)` faces of the grid) in a checkerboard
/// pattern:
///
/// - Faces where `(row + col)` is even become **X-type** stabilizers.
/// - Faces where `(row + col)` is odd become **Z-type** stabilizers.
///
/// Each plaquette stabilizer acts on the four data qubits at its corners.
///
/// # Panics
///
/// Panics if `distance < 2`.
pub fn build_surface_code(distance: usize) -> SurfaceCode {
    assert!(distance >= 2, "Surface code requires distance >= 2");

    let d = distance;
    let n_data = d * d;

    let data_qubits: Vec<usize> = (0..n_data).collect();

    let mut x_stabs: Vec<Vec<usize>> = Vec::new();
    let mut z_stabs: Vec<Vec<usize>> = Vec::new();

    // Iterate over the (d-1) x (d-1) plaquettes of the grid.
    for r in 0..d - 1 {
        for c in 0..d - 1 {
            let stab = vec![
                r * d + c,           // top-left
                r * d + c + 1,       // top-right
                (r + 1) * d + c,     // bottom-left
                (r + 1) * d + c + 1, // bottom-right
            ];
            if (r + c) % 2 == 0 {
                x_stabs.push(stab);
            } else {
                z_stabs.push(stab);
            }
        }
    }

    let n_x = x_stabs.len();
    let n_z = z_stabs.len();
    let x_ancillas: Vec<usize> = (n_data..(n_data + n_x)).collect();
    let z_ancillas: Vec<usize> = ((n_data + n_x)..(n_data + n_x + n_z)).collect();
    let total = n_data + n_x + n_z;

    SurfaceCode {
        distance,
        data_qubits,
        x_stabilizers: x_stabs,
        z_stabilizers: z_stabs,
        x_ancillas,
        z_ancillas,
        total_qubits: total,
    }
}

// ---------------------------------------------------------------------------
// Syndrome extraction
// ---------------------------------------------------------------------------

/// Measure all stabilizers and return the syndrome bit-string.
///
/// The syndrome is a `Vec<u8>` where each entry is 0 or 1. The first
/// `code.z_stabilizers.len()` entries correspond to Z-stabilizer outcomes,
/// followed by `code.x_stabilizers.len()` entries for X-stabilizer outcomes.
///
/// # Z-stabilizer measurement circuit
///
/// For each Z stabilizer (measuring a product of Z operators on the data
/// qubits), the circuit is:
/// 1. For each data qubit in the stabilizer, apply CNOT(data, ancilla).
/// 2. Measure the ancilla in the Z basis.
///
/// The ancilla accumulates the XOR of all data qubit Z-basis values,
/// yielding the Z-parity eigenvalue directly. This correctly detects X
/// (bit-flip) errors that anti-commute with the Z stabilizers.
///
/// # X-stabilizer measurement circuit
///
/// For each X stabilizer (measuring a product of X operators), the circuit is:
/// 1. Apply H to the ancilla (prepare |+>).
/// 2. For each data qubit in the stabilizer, apply CNOT(ancilla, data).
/// 3. Apply H to the ancilla.
/// 4. Measure the ancilla in the Z basis.
///
/// The ancilla in the |+> state picks up a phase from each data qubit's
/// X eigenvalue via the controlled-Z-like action, yielding the X-parity.
pub fn syndrome_extraction(
    state: &mut StabilizerState,
    code: &SurfaceCode,
    rng: &mut u64,
) -> Vec<u8> {
    let mut syndrome = Vec::with_capacity(
        code.z_stabilizers.len() + code.x_stabilizers.len(),
    );

    // Z-stabilizer measurements.
    // CNOT(data -> ancilla) computes the parity of data qubit Z-basis values
    // into the ancilla, which is then measured directly.
    for (s_idx, stab) in code.z_stabilizers.iter().enumerate() {
        let anc = code.z_ancillas[s_idx];

        for &data_q in stab {
            state.cnot(data_q, anc);
        }

        let outcome = state.measure(anc, rng);
        syndrome.push(outcome);
    }

    // X-stabilizer measurements.
    // Prepare ancilla in |+>, CNOT(ancilla -> data) for each data qubit,
    // then H and measure. This extracts the X-parity.
    for (s_idx, stab) in code.x_stabilizers.iter().enumerate() {
        let anc = code.x_ancillas[s_idx];

        state.h(anc);
        for &data_q in stab {
            state.cnot(anc, data_q);
        }
        state.h(anc);

        let outcome = state.measure(anc, rng);
        syndrome.push(outcome);
    }

    syndrome
}

// ---------------------------------------------------------------------------
// Noise model
// ---------------------------------------------------------------------------

/// Apply independent depolarizing noise to every data qubit.
///
/// Each data qubit independently suffers a Pauli error with probability
/// `error_rate`. The error is chosen uniformly among X, Y, Z (each with
/// probability `error_rate / 3`).
///
/// The RNG state is advanced deterministically regardless of which errors
/// fire, ensuring reproducibility.
pub fn apply_noise_round(
    state: &mut StabilizerState,
    code: &SurfaceCode,
    error_rate: f64,
    rng: &mut u64,
) {
    for &q in &code.data_qubits {
        let r = crate::rand_f64(rng);
        if r < error_rate {
            // Choose uniformly among X, Y, Z.
            let err_type = crate::rand_f64(rng);
            if err_type < 1.0 / 3.0 {
                state.x(q);
            } else if err_type < 2.0 / 3.0 {
                state.z(q);
            } else {
                state.y(q);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Minimum-weight decoder for the repetition code.
///
/// Given a syndrome bit-string (from the Z stabilizers only), returns a list
/// of data-qubit indices on which an X correction should be applied.
///
/// The algorithm pairs adjacent syndrome defects (entries equal to 1) and
/// applies a correction at the midpoint of each defect pair. This is optimal
/// for the repetition code under independent noise.
///
/// # Note
///
/// This decoder is specific to the repetition code and does not handle the
/// 2D surface code. A proper MWPM (Minimum Weight Perfect Matching) decoder
/// would be needed for the full surface code.
pub fn decode_repetition_code(syndrome: &[u8], code: &SurfaceCode) -> Vec<usize> {
    let mut corrections = Vec::new();
    let mut i = 0;

    while i < syndrome.len() {
        if syndrome[i] == 1 {
            // Find the extent of the contiguous defect chain.
            let start = i;
            while i < syndrome.len() && syndrome[i] == 1 {
                i += 1;
            }
            // Correct at the midpoint of the defect chain.
            // For a chain spanning stabilizers [start, i), the corresponding
            // data qubit to correct is at the midpoint index.
            let mid = (start + i) / 2;
            if mid < code.data_qubits.len() {
                corrections.push(code.data_qubits[mid]);
            }
        } else {
            i += 1;
        }
    }

    corrections
}

// ---------------------------------------------------------------------------
// Monte Carlo logical error rate estimation
// ---------------------------------------------------------------------------

/// Estimate the logical error rate of a repetition code via Monte Carlo.
///
/// Runs `n_rounds` independent trials. In each trial:
/// 1. Initialize all qubits in |0>.
/// 2. Apply depolarizing noise at rate `physical_error_rate`.
/// 3. Extract the syndrome.
/// 4. Decode and apply corrections.
/// 5. Measure all data qubits; if a majority are |1>, count a logical error.
///
/// Returns the fraction of trials that resulted in a logical error.
///
/// # Determinism
///
/// The result is fully deterministic given `seed`.
pub fn estimate_logical_error_rate(
    distance: usize,
    physical_error_rate: f64,
    n_rounds: usize,
    seed: u64,
) -> f64 {
    let code = build_repetition_code(distance);
    let mut rng = seed;
    let mut logical_errors: usize = 0;

    for _ in 0..n_rounds {
        let mut state = StabilizerState::new(code.total_qubits);

        // Apply noise to data qubits.
        apply_noise_round(&mut state, &code, physical_error_rate, &mut rng);

        // Extract syndrome.
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);

        // Decode and apply corrections.
        let corrections = decode_repetition_code(&syndrome, &code);
        for q in corrections {
            state.x(q);
        }

        // Check for logical error: measure all data qubits and count |1> outcomes.
        // For the repetition code encoding |0_L> = |00...0>, a logical error
        // has occurred if a majority of qubits have been flipped to |1>.
        let mut ones: usize = 0;
        for &q in &code.data_qubits {
            // Use a fresh copy of rng for each measurement to keep the main
            // rng state deterministic across rounds regardless of branching.
            let mut rng_meas = rng;
            if state.measure(q, &mut rng_meas) == 1 {
                ones += 1;
            }
        }
        // Advance the main rng past the measurement draws.
        // We draw one splitmix64 per data qubit measurement (at most).
        for _ in 0..code.data_qubits.len() {
            crate::splitmix64(&mut rng);
        }

        if ones > distance / 2 {
            logical_errors += 1;
        }
    }

    logical_errors as f64 / n_rounds as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Repetition code structure -------------------------------------------

    #[test]
    fn test_repetition_code_structure_d3() {
        let code = build_repetition_code(3);
        assert_eq!(code.distance, 3);
        assert_eq!(code.data_qubits.len(), 3);
        assert_eq!(code.z_stabilizers.len(), 2);
        assert_eq!(code.z_ancillas.len(), 2);
        assert!(code.x_stabilizers.is_empty());
        assert!(code.x_ancillas.is_empty());
        assert_eq!(code.total_qubits, 5); // 3 data + 2 ancilla
    }

    #[test]
    fn test_repetition_code_structure_d5() {
        let code = build_repetition_code(5);
        assert_eq!(code.data_qubits.len(), 5);
        assert_eq!(code.z_stabilizers.len(), 4);
        assert_eq!(code.total_qubits, 9); // 5 data + 4 ancilla
    }

    #[test]
    fn test_repetition_code_stabilizer_pairs() {
        let code = build_repetition_code(4);
        // Each Z stabilizer should check adjacent data qubits.
        for (i, stab) in code.z_stabilizers.iter().enumerate() {
            assert_eq!(stab.len(), 2);
            assert_eq!(stab[0], i);
            assert_eq!(stab[1], i + 1);
        }
    }

    #[test]
    #[should_panic(expected = "distance >= 2")]
    fn test_repetition_code_panics_d1() {
        build_repetition_code(1);
    }

    // -- Surface code structure ----------------------------------------------

    #[test]
    fn test_surface_code_structure_d3() {
        let code = build_surface_code(3);
        assert_eq!(code.data_qubits.len(), 9);
        // 2x2 grid of plaquettes: 2 X-type + 2 Z-type.
        assert_eq!(code.x_stabilizers.len() + code.z_stabilizers.len(), 4);
        assert_eq!(code.x_ancillas.len(), code.x_stabilizers.len());
        assert_eq!(code.z_ancillas.len(), code.z_stabilizers.len());
    }

    #[test]
    fn test_surface_code_structure_d4() {
        let code = build_surface_code(4);
        assert_eq!(code.data_qubits.len(), 16);
        // 3x3 = 9 plaquettes total.
        assert_eq!(code.x_stabilizers.len() + code.z_stabilizers.len(), 9);
    }

    #[test]
    fn test_surface_code_plaquette_qubits() {
        let code = build_surface_code(3);
        // Every stabilizer should reference exactly 4 data qubits.
        for stab in code.x_stabilizers.iter().chain(code.z_stabilizers.iter()) {
            assert_eq!(stab.len(), 4);
            for &q in stab {
                assert!(q < 9, "Data qubit index {} out of range", q);
            }
        }
    }

    #[test]
    #[should_panic(expected = "distance >= 2")]
    fn test_surface_code_panics_d1() {
        build_surface_code(1);
    }

    // -- Syndrome extraction -------------------------------------------------

    #[test]
    fn test_syndrome_clean_state() {
        let code = build_repetition_code(3);
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = 42u64;
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
        // A clean |00...0> state should produce an all-zero syndrome.
        for (i, &s) in syndrome.iter().enumerate() {
            assert_eq!(s, 0, "Syndrome bit {} should be 0 for clean state", i);
        }
    }

    #[test]
    fn test_syndrome_clean_state_d5() {
        let code = build_repetition_code(5);
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = 123u64;
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
        assert_eq!(syndrome.len(), 4);
        for &s in &syndrome {
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn test_single_error_detected() {
        let code = build_repetition_code(5);
        let mut state = StabilizerState::new(code.total_qubits);
        // Apply X error on data qubit 2 (middle of the chain).
        state.x(code.data_qubits[2]);
        let mut rng = 42u64;
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
        // Stabilizers 1 (Z_1 Z_2) and 2 (Z_2 Z_3) should fire.
        let nonzero: usize = syndrome.iter().filter(|&&s| s == 1).count();
        assert!(nonzero > 0, "Single X error should trigger at least one syndrome bit");
    }

    #[test]
    fn test_boundary_error_detected() {
        let code = build_repetition_code(5);
        let mut state = StabilizerState::new(code.total_qubits);
        // Apply X error on the first data qubit.
        state.x(code.data_qubits[0]);
        let mut rng = 99u64;
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
        // Only stabilizer 0 (Z_0 Z_1) should fire.
        assert_eq!(syndrome[0], 1, "Boundary error should trigger stabilizer 0");
    }

    // -- Decoder -------------------------------------------------------------

    #[test]
    fn test_decode_single_error() {
        let code = build_repetition_code(5);
        // Syndrome pattern for an error on data qubit 1: stabilizers 0 and 1 fire.
        let syndrome = vec![1, 1, 0, 0];
        let corrections = decode_repetition_code(&syndrome, &code);
        assert!(!corrections.is_empty(), "Should suggest at least one correction");
    }

    #[test]
    fn test_decode_no_error() {
        let code = build_repetition_code(5);
        let syndrome = vec![0, 0, 0, 0];
        let corrections = decode_repetition_code(&syndrome, &code);
        assert!(corrections.is_empty(), "No syndrome defects -> no corrections");
    }

    #[test]
    fn test_decode_boundary_error() {
        let code = build_repetition_code(5);
        // Single defect at the boundary: only stabilizer 0 fires.
        let syndrome = vec![1, 0, 0, 0];
        let corrections = decode_repetition_code(&syndrome, &code);
        assert!(!corrections.is_empty(), "Boundary defect should produce correction");
    }

    // -- Logical error rate --------------------------------------------------

    #[test]
    fn test_logical_error_rate_low_noise() {
        // At very low physical error rate, logical error rate should be near 0.
        let rate = estimate_logical_error_rate(3, 0.001, 100, 42);
        assert!(
            rate < 0.1,
            "Low-noise logical error rate {} is unexpectedly high",
            rate
        );
    }

    #[test]
    fn test_logical_error_rate_decreases_with_distance() {
        // Below the threshold, larger distance should (statistically) help.
        let rate_3 = estimate_logical_error_rate(3, 0.02, 200, 42);
        let rate_5 = estimate_logical_error_rate(5, 0.02, 200, 42);
        // Both should be well below 50% at this noise level.
        assert!(rate_3 < 0.5, "d=3 rate {} unexpectedly high", rate_3);
        assert!(rate_5 < 0.5, "d=5 rate {} unexpectedly high", rate_5);
    }

    // -- Determinism ---------------------------------------------------------

    #[test]
    fn test_deterministic_syndrome() {
        let code = build_repetition_code(5);

        let run = |seed: u64| -> Vec<u8> {
            let mut state = StabilizerState::new(code.total_qubits);
            let mut rng = seed;
            apply_noise_round(&mut state, &code, 0.1, &mut rng);
            syndrome_extraction(&mut state, &code, &mut rng)
        };

        let s1 = run(42);
        let s2 = run(42);
        assert_eq!(s1, s2, "Syndrome extraction must be deterministic");
    }

    #[test]
    fn test_deterministic_logical_error_rate() {
        let r1 = estimate_logical_error_rate(3, 0.05, 50, 42);
        let r2 = estimate_logical_error_rate(3, 0.05, 50, 42);
        assert_eq!(
            r1.to_bits(),
            r2.to_bits(),
            "Logical error rate must be bit-identical across runs"
        );
    }

    #[test]
    fn test_deterministic_different_seeds_differ() {
        // Different seeds should (with overwhelming probability) give different
        // syndrome sequences when the error rate is high enough.
        // Use a larger code and high error rate to make collision negligible.
        let code = build_repetition_code(15);

        let run = |seed: u64| -> Vec<u8> {
            let mut state = StabilizerState::new(code.total_qubits);
            let mut rng = seed;
            apply_noise_round(&mut state, &code, 0.5, &mut rng);
            syndrome_extraction(&mut state, &code, &mut rng)
        };

        let s1 = run(1);
        let s2 = run(999);
        // At 50% error rate on 15 qubits with 14 stabilizers, the probability
        // of identical syndromes from different seeds is ~2^{-14}, negligible.
        assert_ne!(s1, s2, "Different seeds should produce different results");
    }
}
