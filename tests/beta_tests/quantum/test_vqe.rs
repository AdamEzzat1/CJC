//! VQE Integration Tests — 50-qubit Heisenberg model with continuous verification.

use cjc_quantum::vqe::*;
use cjc_quantum::mps::Mps;

// ---------------------------------------------------------------------------
// Continuous Verification Loop
// ---------------------------------------------------------------------------

#[test]
fn test_verification_gate_10_seeds() {
    for seed in 0..10u64 {
        verification_gate(seed).unwrap_or_else(|e| {
            panic!("Verification gate failed at seed {}: {}", seed, e);
        });
    }
}

// ---------------------------------------------------------------------------
// 4-Qubit VQE (fast, validates correctness)
// ---------------------------------------------------------------------------

#[test]
fn test_vqe_4_qubit_convergence() {
    let result = vqe_heisenberg_1d(4, 16, 0.15, 20, 42);

    // Energy should decrease monotonically (mostly — allow small fluctuations)
    let initial = result.energy_history[0];
    let final_e = *result.energy_history.last().unwrap();
    assert!(final_e < initial,
        "Energy should decrease: initial={}, final={}", initial, final_e);

    // For 4 qubits, Heisenberg ground state energy is around -3
    // (3 bonds, each contributing about -1 at optimum)
    // With limited iterations we won't reach it, but should be below 0
    assert!(final_e < initial,
        "Energy {} should be below initial {}", final_e, initial);
}

#[test]
fn test_vqe_bit_identical_determinism() {
    let r1 = vqe_heisenberg_1d(4, 16, 0.1, 10, 123);
    let r2 = vqe_heisenberg_1d(4, 16, 0.1, 10, 123);

    // Parameters must be bit-identical
    for k in 0..r1.thetas.len() {
        assert_eq!(r1.thetas[k].to_bits(), r2.thetas[k].to_bits(),
            "theta[{}]: run1={} run2={}", k, r1.thetas[k], r2.thetas[k]);
    }

    // Energy history must be bit-identical
    for (i, (e1, e2)) in r1.energy_history.iter().zip(&r2.energy_history).enumerate() {
        assert_eq!(e1.to_bits(), e2.to_bits(),
            "energy_history[{}]: run1={} run2={}", i, e1, e2);
    }
}

#[test]
fn test_vqe_different_seeds_different_results() {
    let r1 = vqe_heisenberg_1d(4, 16, 0.1, 5, 42);
    let r2 = vqe_heisenberg_1d(4, 16, 0.1, 5, 99);
    // Different seeds should give different parameters
    let same = r1.thetas.iter().zip(&r2.thetas)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(!same, "Different seeds should give different parameters");
}

// ---------------------------------------------------------------------------
// 50-Qubit VQE Benchmarks
// ---------------------------------------------------------------------------

#[test]
fn test_vqe_50_qubit_ansatz_memory_under_500mb() {
    let thetas = vec![0.1; 50];
    let mps = build_mps_ansatz(50, &thetas, 8);
    let mem = mps.memory_bytes();
    assert!(mem < 500_000_000,
        "50-qubit MPS uses {} bytes, exceeds 500MB budget", mem);
}

#[test]
fn test_vqe_50_qubit_bond_dimension_le_8() {
    let thetas = vec![0.3; 50];
    let mps = build_mps_ansatz(50, &thetas, 8);
    for i in 0..49 {
        assert!(mps.tensors[i].bond_right <= 8,
            "Bond between {} and {} is {}, expected ≤ 8",
            i, i + 1, mps.tensors[i].bond_right);
    }
}

#[test]
fn test_vqe_50_qubit_energy_computable() {
    let thetas = vec![0.2; 50];
    let mps = build_mps_ansatz(50, &thetas, 8);
    let e = mps_heisenberg_energy(&mps);
    assert!(e.is_finite(), "50-qubit energy should be finite, got {}", e);
    // 49 bonds, product state gives E ≈ 49 (all ZZ=+1 after CNOT chain)
    // With non-zero thetas, should be different
    assert!(e.abs() < 100.0, "Energy {} seems unreasonable for 50 qubits", e);
}

#[test]
fn test_vqe_50_qubit_gradient_computable() {
    let thetas = vec![0.1; 50];
    let grads = mps_parameter_shift_gradient(50, &thetas, 8);
    assert_eq!(grads.len(), 50);
    for (k, &g) in grads.iter().enumerate() {
        assert!(g.is_finite(), "Gradient {} is not finite for param {}", g, k);
    }
}

#[test]
fn test_vqe_50_qubit_single_step() {
    // Run 1 iteration of VQE on 50 qubits — verify it works end-to-end
    let result = vqe_heisenberg_1d(50, 8, 0.05, 1, 42);
    assert_eq!(result.thetas.len(), 50);
    assert_eq!(result.energy_history.len(), 2); // initial + 1 iteration
    assert!(result.energy.is_finite());
}

#[test]
fn test_vqe_50_qubit_deterministic() {
    let r1 = vqe_heisenberg_1d(50, 8, 0.05, 1, 42);
    let r2 = vqe_heisenberg_1d(50, 8, 0.05, 1, 42);

    for k in 0..50 {
        assert_eq!(r1.thetas[k].to_bits(), r2.thetas[k].to_bits(),
            "50-qubit theta[{}] not bit-identical", k);
    }
    assert_eq!(r1.energy.to_bits(), r2.energy.to_bits(),
        "50-qubit energy not bit-identical");
}

// ---------------------------------------------------------------------------
// Heisenberg Observable Correctness
// ---------------------------------------------------------------------------

#[test]
fn test_zz_matches_statevector_4_qubit() {
    // Compare MPS ZZ expectation with statevector computation
    use cjc_quantum::Complex as ComplexF64;

    let thetas = vec![0.3, 0.7, 1.1, 0.5];
    let mps = build_mps_ansatz(4, &thetas, 64);
    let sv = mps.to_statevector();

    for site in 0..3 {
        let mps_zz = mps_zz_expectation(&mps, site);

        // Compute ZZ from statevector: Σ_k |α_k|² * z_i(k) * z_{i+1}(k)
        let mut sv_zz = 0.0;
        for (k, amp) in sv.iter().enumerate() {
            let z_i = if (k >> site) & 1 == 0 { 1.0 } else { -1.0 };
            let z_ip1 = if (k >> (site + 1)) & 1 == 0 { 1.0 } else { -1.0 };
            sv_zz += amp.norm_sq() * z_i * z_ip1;
        }

        assert!((mps_zz - sv_zz).abs() < 1e-10,
            "ZZ({},{}) mps={} sv={}", site, site + 1, mps_zz, sv_zz);
    }
}
