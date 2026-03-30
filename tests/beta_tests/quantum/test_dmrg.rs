//! DMRG Integration Tests — correctness, determinism, and convergence.

use cjc_quantum::dmrg::*;

// ---------------------------------------------------------------------------
// 2-Qubit Ising: exact ground-state energy is -1
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_2_qubit_ising_energy() {
    let result = dmrg_heisenberg_1d(2, 4, 10, 1e-8);
    assert!(
        (result.energy - (-1.0)).abs() < 0.1,
        "2-qubit Ising energy should be near -1, got {}",
        result.energy
    );
}

// ---------------------------------------------------------------------------
// 2-Qubit Heisenberg: exact ground-state energy is -3
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_2_qubit_heisenberg_energy() {
    let result = dmrg_full_heisenberg_1d(2, 4, 10, 1e-8);
    assert!(
        (result.energy - (-3.0)).abs() < 0.5,
        "2-qubit Heisenberg energy should be near -3, got {}",
        result.energy
    );
}

// ---------------------------------------------------------------------------
// 4-Qubit Ising: energy below -2
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_4_qubit_ising_energy_below_threshold() {
    let result = dmrg_1d(4, 8, 10, 1e-8, DmrgHamiltonian::Ising);
    assert!(
        result.energy < -2.0,
        "4-qubit Ising energy should be below -2, got {}",
        result.energy
    );
}

// ---------------------------------------------------------------------------
// 4-Qubit Heisenberg: energy is negative
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_4_qubit_heisenberg_energy_negative() {
    let result = dmrg_1d(4, 8, 10, 1e-8, DmrgHamiltonian::Heisenberg);
    assert!(
        result.energy < 0.0,
        "4-qubit Heisenberg energy should be negative, got {}",
        result.energy
    );
}

// ---------------------------------------------------------------------------
// Determinism: same parameters produce bit-identical energies
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_determinism_ising() {
    let a = dmrg_heisenberg_1d(2, 4, 5, 1e-8);
    let b = dmrg_heisenberg_1d(2, 4, 5, 1e-8);
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "Ising DMRG must be bit-identical across runs: {} vs {}",
        a.energy,
        b.energy
    );
    assert_eq!(
        a.energies.len(),
        b.energies.len(),
        "Energy history length must match across runs"
    );
    for (i, (ea, eb)) in a.energies.iter().zip(b.energies.iter()).enumerate() {
        assert_eq!(
            ea.to_bits(),
            eb.to_bits(),
            "Energy at sweep {} must be bit-identical: {} vs {}",
            i,
            ea,
            eb
        );
    }
}

#[test]
fn test_dmrg_determinism_heisenberg() {
    let a = dmrg_full_heisenberg_1d(2, 4, 5, 1e-8);
    let b = dmrg_full_heisenberg_1d(2, 4, 5, 1e-8);
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "Heisenberg DMRG must be bit-identical across runs: {} vs {}",
        a.energy,
        b.energy
    );
    for (i, (ea, eb)) in a.energies.iter().zip(b.energies.iter()).enumerate() {
        assert_eq!(
            ea.to_bits(),
            eb.to_bits(),
            "Energy at sweep {} must be bit-identical: {} vs {}",
            i,
            ea,
            eb
        );
    }
}

// ---------------------------------------------------------------------------
// Energy decreases across sweeps
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_energy_decreases_ising() {
    let result = dmrg_heisenberg_1d(4, 8, 8, 1e-10);
    assert!(
        result.energies.len() >= 2,
        "Need at least 2 energy entries to check decrease, got {}",
        result.energies.len()
    );
    let first = result.energies[0];
    let last = *result.energies.last().unwrap();
    assert!(
        last <= first + 1e-6,
        "Final energy ({}) should not exceed initial energy ({}) by more than 1e-6",
        last,
        first
    );
}

#[test]
fn test_dmrg_energy_decreases_heisenberg() {
    let result = dmrg_full_heisenberg_1d(4, 8, 8, 1e-10);
    assert!(
        result.energies.len() >= 2,
        "Need at least 2 energy entries to check decrease, got {}",
        result.energies.len()
    );
    let first = result.energies[0];
    let last = *result.energies.last().unwrap();
    assert!(
        last <= first + 1e-6,
        "Final energy ({}) should not exceed initial energy ({}) by more than 1e-6",
        last,
        first
    );
}

// ---------------------------------------------------------------------------
// DmrgResult has correct structure
// ---------------------------------------------------------------------------

#[test]
fn test_dmrg_result_structure_ising() {
    let result = dmrg_heisenberg_1d(2, 4, 3, 1e-4);

    // energies vector should have at least 1 entry (initial + sweeps)
    assert!(
        !result.energies.is_empty(),
        "DmrgResult.energies should not be empty"
    );

    // sweeps should be non-zero (at least one sweep was performed)
    assert!(
        result.sweeps >= 1,
        "DmrgResult.sweeps should be at least 1, got {}",
        result.sweeps
    );

    // mps should have the correct number of qubits
    assert_eq!(
        result.mps.n_qubits, 2,
        "MPS should have 2 qubits, got {}",
        result.mps.n_qubits
    );

    // final energy should match the last entry in energies
    let last_energy = *result.energies.last().unwrap();
    assert!(
        (result.energy - last_energy).abs() < 1e-12,
        "DmrgResult.energy ({}) should match last entry in energies ({})",
        result.energy,
        last_energy
    );
}

#[test]
fn test_dmrg_result_structure_heisenberg() {
    let result = dmrg_full_heisenberg_1d(4, 8, 3, 1e-4);

    assert!(
        !result.energies.is_empty(),
        "DmrgResult.energies should not be empty"
    );

    assert!(
        result.sweeps >= 1,
        "DmrgResult.sweeps should be at least 1, got {}",
        result.sweeps
    );

    assert_eq!(
        result.mps.n_qubits, 4,
        "MPS should have 4 qubits, got {}",
        result.mps.n_qubits
    );

    assert_eq!(
        result.mps.tensors.len(),
        4,
        "MPS should have 4 tensors, got {}",
        result.mps.tensors.len()
    );

    let last_energy = *result.energies.last().unwrap();
    assert!(
        (result.energy - last_energy).abs() < 1e-12,
        "DmrgResult.energy ({}) should match last entry in energies ({})",
        result.energy,
        last_energy
    );
}
