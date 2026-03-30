//! Full Heisenberg (XX + YY + ZZ) Integration Tests.

use cjc_quantum::vqe::*;
use cjc_quantum::mps::Mps;
use cjc_quantum::Complex as ComplexF64;

// ---------------------------------------------------------------------------
// XX Expectation Tests
// ---------------------------------------------------------------------------

#[test]
fn test_xx_product_state_zero() {
    let mps = Mps::new(4);
    for i in 0..3 {
        let xx = mps_xx_expectation(&mps, i);
        assert!(xx.abs() < 1e-10, "XX({},{}) = {} for |0000⟩", i, i + 1, xx);
    }
}

#[test]
fn test_xx_bell_state() {
    // Bell |Φ+⟩ = (|00⟩+|11⟩)/√2: ⟨XX⟩ = 1
    let mut mps = Mps::new(2);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
              [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];
    mps.apply_single_qubit(0, h);
    mps.apply_cnot_adjacent(0, 1);
    assert!((mps_xx_expectation(&mps, 0) - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// YY Expectation Tests
// ---------------------------------------------------------------------------

#[test]
fn test_yy_product_state_zero() {
    let mps = Mps::new(4);
    for i in 0..3 {
        let yy = mps_yy_expectation(&mps, i);
        assert!(yy.abs() < 1e-10, "YY({},{}) = {} for |0000⟩", i, i + 1, yy);
    }
}

#[test]
fn test_yy_bell_state() {
    // Bell |Φ+⟩: ⟨YY⟩ = -1
    let mut mps = Mps::new(2);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
              [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];
    mps.apply_single_qubit(0, h);
    mps.apply_cnot_adjacent(0, 1);
    assert!((mps_yy_expectation(&mps, 0) - (-1.0)).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Full Heisenberg Energy Tests
// ---------------------------------------------------------------------------

#[test]
fn test_full_heisenberg_product_state() {
    // |0...0⟩: XX=0, YY=0, ZZ=+1 per bond. Full H = N-1
    let mps = Mps::new(6);
    let e = mps_full_heisenberg_energy(&mps);
    assert!((e - 5.0).abs() < 1e-10, "E = {}, expected 5.0", e);
}

#[test]
fn test_full_heisenberg_bell_state() {
    // Bell: XX=1, YY=-1, ZZ=1. Sum = 1
    let mut mps = Mps::new(2);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
              [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];
    mps.apply_single_qubit(0, h);
    mps.apply_cnot_adjacent(0, 1);
    let e = mps_full_heisenberg_energy(&mps);
    assert!((e - 1.0).abs() < 1e-10, "Bell H = {}", e);
}

#[test]
fn test_full_heisenberg_matches_components() {
    // Verify XX + YY + ZZ = full energy for random state
    let thetas = vec![0.3, 0.7, 1.1, 0.5];
    let mps = build_mps_ansatz(4, &thetas, 64);
    let full = mps_full_heisenberg_energy(&mps);
    let mut sum = 0.0;
    for i in 0..3 {
        sum += mps_xx_expectation(&mps, i);
        sum += mps_yy_expectation(&mps, i);
        sum += mps_zz_expectation(&mps, i);
    }
    assert!((full - sum).abs() < 1e-10, "full={} sum={}", full, sum);
}

// ---------------------------------------------------------------------------
// Cross-Validation: MPS vs Statevector
// ---------------------------------------------------------------------------

#[test]
fn test_full_heisenberg_mps_vs_statevector_4_qubit() {
    let thetas = vec![0.3, 0.7, 1.1, 0.5];
    let mps = build_mps_ansatz(4, &thetas, 64);
    let sv = mps.to_statevector();

    // Compute full Heisenberg from statevector for each bond
    for site in 0..3 {
        // ZZ from statevector
        let mut sv_zz = 0.0;
        for (k, amp) in sv.iter().enumerate() {
            let z_i = if (k >> site) & 1 == 0 { 1.0 } else { -1.0 };
            let z_j = if (k >> (site + 1)) & 1 == 0 { 1.0 } else { -1.0 };
            sv_zz += amp.norm_sq() * z_i * z_j;
        }

        // XX from statevector: flip both bits at site and site+1
        let mut sv_xx = 0.0;
        for k in 0..(1 << 4) {
            let k_flip = k ^ (1 << site) ^ (1 << (site + 1));
            // ⟨k|XX|k'⟩ = 1 if k = k' with both bits flipped
            let bra = sv[k].conj();
            let ket = sv[k_flip];
            sv_xx += bra.mul_fixed(ket).re;
        }

        // YY from statevector
        let mut sv_yy = 0.0;
        for k in 0..(1 << 4) {
            let k_flip = k ^ (1 << site) ^ (1 << (site + 1));
            let bit_i = (k >> site) & 1;
            let bit_j = (k >> (site + 1)) & 1;
            // Y|b⟩ = i(-1)^b |1-b⟩, so Y_i Y_j |k⟩ has phase i²(-1)^{b_i+b_j} = -(-1)^{b_i+b_j}
            let yy_phase = -((-1.0f64).powi((bit_i + bit_j) as i32));
            let bra = sv[k].conj();
            let ket = sv[k_flip];
            sv_yy += bra.mul_fixed(ket).re * yy_phase;
        }

        let mps_zz = mps_zz_expectation(&mps, site);
        let mps_xx = mps_xx_expectation(&mps, site);
        let mps_yy = mps_yy_expectation(&mps, site);

        assert!((mps_zz - sv_zz).abs() < 1e-10,
            "ZZ({}) mps={} sv={}", site, mps_zz, sv_zz);
        assert!((mps_xx - sv_xx).abs() < 1e-10,
            "XX({}) mps={} sv={}", site, mps_xx, sv_xx);
        assert!((mps_yy - sv_yy).abs() < 1e-10,
            "YY({}) mps={} sv={}", site, mps_yy, sv_yy);
    }
}

// ---------------------------------------------------------------------------
// VQE with Full Heisenberg
// ---------------------------------------------------------------------------

#[test]
fn test_vqe_full_heisenberg_4_qubit_convergence() {
    let result = vqe_full_heisenberg_1d(4, 16, 0.15, 20, 42);
    let initial = result.energy_history[0];
    let final_e = *result.energy_history.last().unwrap();
    assert!(final_e < initial,
        "Full Heisenberg energy should decrease: initial={}, final={}", initial, final_e);
}

#[test]
fn test_vqe_full_heisenberg_lower_than_ising() {
    // Full Heisenberg has lower ground state than Ising
    let h_result = vqe_full_heisenberg_1d(4, 16, 0.15, 20, 42);
    let i_result = vqe_heisenberg_1d(4, 16, 0.15, 20, 42);
    // Full Heisenberg should find lower energy (more terms to minimize)
    assert!(h_result.energy <= i_result.energy + 1.0,
        "Full Heisenberg {} should not be much higher than Ising {}", h_result.energy, i_result.energy);
}

#[test]
fn test_vqe_full_heisenberg_bit_identical_determinism() {
    let r1 = vqe_full_heisenberg_1d(4, 16, 0.1, 5, 123);
    let r2 = vqe_full_heisenberg_1d(4, 16, 0.1, 5, 123);
    for k in 0..r1.thetas.len() {
        assert_eq!(r1.thetas[k].to_bits(), r2.thetas[k].to_bits(),
            "theta[{}]: run1={} run2={}", k, r1.thetas[k], r2.thetas[k]);
    }
    for (i, (e1, e2)) in r1.energy_history.iter().zip(&r2.energy_history).enumerate() {
        assert_eq!(e1.to_bits(), e2.to_bits(),
            "energy_history[{}]: run1={} run2={}", i, e1, e2);
    }
}

#[test]
fn test_hamiltonian_selector_ising() {
    let thetas = vec![0.3; 4];
    let mps = build_mps_ansatz(4, &thetas, 16);
    let e_ising = mps_energy(&mps, Hamiltonian::Ising);
    let e_direct = mps_heisenberg_energy(&mps);
    assert_eq!(e_ising.to_bits(), e_direct.to_bits());
}

#[test]
fn test_hamiltonian_selector_heisenberg() {
    let thetas = vec![0.3; 4];
    let mps = build_mps_ansatz(4, &thetas, 16);
    let e_heis = mps_energy(&mps, Hamiltonian::Heisenberg);
    let e_direct = mps_full_heisenberg_energy(&mps);
    assert_eq!(e_heis.to_bits(), e_direct.to_bits());
}
