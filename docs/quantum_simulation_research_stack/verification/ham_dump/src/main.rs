// Recover the full Hermitian matrix of cjc-quantum's H2 / LiH Hamiltonians by
// polarization over CJC's own FermionicHamiltonian::expectation (the code path
// q_fermion_expectation uses). Output: "name n i j re im" lines.
use cjc_quantum::fermion::{h2_hamiltonian, lih_hamiltonian, FermionicHamiltonian};
use cjc_quantum::statevector::Statevector;
use cjc_runtime::complex::ComplexF64;

fn expval(h: &FermionicHamiltonian, amps: Vec<ComplexF64>) -> f64 {
    h.expectation(&Statevector::from_amplitudes(amps).unwrap())
}
fn dump(name: &str, h: &FermionicHamiltonian) {
    let n = h.n_qubits;
    let d = 1usize << n;
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let basis = |k: usize| { let mut v = vec![ComplexF64::ZERO; d]; v[k] = ComplexF64::ONE; v };
    let diag: Vec<f64> = (0..d).map(|k| expval(h, basis(k))).collect();
    println!("# {name}: n_qubits={n} n_terms={}", h.n_terms());
    for i in 0..d {
        for j in 0..d {
            let (re, im) = if i == j { (diag[i], 0.0) } else {
                let mut a = vec![ComplexF64::ZERO; d]; a[i] = ComplexF64::real(r); a[j] = ComplexF64::real(r);
                let mut b = vec![ComplexF64::ZERO; d]; b[i] = ComplexF64::real(r); b[j] = ComplexF64::new(0.0, r);
                let base = 0.5 * (diag[i] + diag[j]);
                // <a|H|a> = base + Re(H_ij); <b|H|b> = base + Im(H_ij) with b = (|i> + i|j>)/sqrt2
                (expval(h, a) - base, expval(h, b) - base)
            };
            println!("{name} {n} {i} {j} {re:.17e} {im:.17e}");
        }
    }
}
fn main() { dump("h2", &h2_hamiltonian()); dump("lih", &lih_hamiltonian()); }
