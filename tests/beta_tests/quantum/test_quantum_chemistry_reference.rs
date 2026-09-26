//! Chemistry reference tests: cjc-quantum's molecular Hamiltonians against
//! PySCF/OpenFermion reference energies.
//!
//! References come from
//! `docs/quantum_simulation_research_stack/verification/pyscf_openfermion_check.py`
//! and `gen_lih_hamiltonian.py` (PySCF 2.14.0, OpenFermion 1.8.1, STO-3G).
//!
//! The Hamiltonian matrix is recovered from CJC's own
//! `FermionicHamiltonian::expectation` (the `q_fermion_expectation` code path)
//! by polarization, so these tests check what `.cjcl` programs actually see.

use cjc_quantum::fermion::{h2_hamiltonian, lih_hamiltonian, FermionicHamiltonian};
use cjc_quantum::statevector::Statevector;
use cjc_runtime::complex::ComplexF64;

/// Real symmetric matrix of `h` (all four Hamiltonians here are real).
fn matrix(h: &FermionicHamiltonian) -> Vec<Vec<f64>> {
    let d = 1usize << h.n_qubits;
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let e = |amps: Vec<ComplexF64>| h.expectation(&Statevector::from_amplitudes(amps).unwrap());
    let basis = |k: usize| {
        let mut v = vec![ComplexF64::ZERO; d];
        v[k] = ComplexF64::ONE;
        v
    };
    let diag: Vec<f64> = (0..d).map(|k| e(basis(k))).collect();
    let mut m = vec![vec![0.0; d]; d];
    for i in 0..d {
        m[i][i] = diag[i];
        for j in (i + 1)..d {
            let mut a = vec![ComplexF64::ZERO; d];
            a[i] = ComplexF64::real(r);
            a[j] = ComplexF64::real(r);
            let re = e(a) - 0.5 * (diag[i] + diag[j]);
            let mut b = vec![ComplexF64::ZERO; d];
            b[i] = ComplexF64::real(r);
            b[j] = ComplexF64::new(0.0, r);
            let im = e(b) - 0.5 * (diag[i] + diag[j]);
            assert!(im.abs() < 1e-12, "H[{i}][{j}] has imaginary part {im}");
            m[i][j] = re;
            m[j][i] = re;
        }
    }
    m
}

/// Eigenvalues of a small real symmetric matrix (cyclic Jacobi), ascending.
fn eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ev
}

/// Ground energy restricted to basis states with `n_elec` occupied qubits (JW).
fn sector_ground(h: &FermionicHamiltonian, n_elec: u32) -> f64 {
    let m = matrix(h);
    let idx: Vec<usize> = (0..m.len()).filter(|k| k.count_ones() == n_elec).collect();
    let sub: Vec<Vec<f64>> = idx.iter().map(|&i| idx.iter().map(|&j| m[i][j]).collect()).collect();
    eigenvalues(sub)[0]
}

#[test]
fn lih_hartree_fock_energy_matches_pyscf_rhf() {
    // HF determinant = qubits 0 and 1 occupied = basis index 0b0011.
    let h = lih_hamiltonian();
    let mut amps = vec![ComplexF64::ZERO; 16];
    amps[0b0011] = ComplexF64::ONE;
    let e = h.expectation(&Statevector::from_amplitudes(amps).unwrap());
    assert!((e - (-7.8631336887)).abs() < 1e-9, "LiH <HF|H|HF> = {e}, PySCF E_RHF = -7.8631336887");
}

#[test]
fn lih_ground_state_matches_pyscf_casci() {
    let e = sector_ground(&lih_hamiltonian(), 2);
    assert!((e - (-7.8633736643)).abs() < 1e-9, "LiH CAS(2,2) ground = {e}, PySCF = -7.8633736643");
    // Variational sanity: an active-space energy cannot go below full FCI.
    assert!(e > -7.8827618487, "LiH active-space energy {e} is below the full-space FCI bound");
}

#[test]
fn h2_ground_state_matches_fci_electronic_energy() {
    // The shipped H2 matrix is electronic-only; FCI electronic = -1.851024 Ha.
    // Tolerance 5e-4 covers the documented 4-decimal coefficient rounding
    // (observed difference 1.75e-4 Ha).
    let ev = eigenvalues(matrix(&h2_hamiltonian()));
    assert!((ev[0] - (-1.851024)).abs() < 5e-4, "H2 ground = {}, FCI electronic = -1.851024", ev[0]);
    // Total energy after adding nuclear repulsion 1/R (R = 0.7414 Å).
    let e_nuc = 0.713754;
    assert!((ev[0] + e_nuc - (-1.137270)).abs() < 5e-4);
}
