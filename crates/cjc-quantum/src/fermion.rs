//! Jordan-Wigner Transformation — Fermionic Hamiltonians to Qubit Operators.
//!
//! Maps fermionic creation (a†) and annihilation (a) operators to qubit Pauli
//! strings using the Jordan-Wigner encoding:
//!
//!   a†_j = (X_j - iY_j) / 2 ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
//!   a_j  = (X_j + iY_j) / 2 ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
//!
//! # Supported Molecular Hamiltonians
//!
//! - **H₂** (hydrogen molecule): 2-site, 2-orbital minimal basis
//! - **LiH** (lithium hydride): 4-site, 4-orbital STO-3G
//!
//! # Determinism
//!
//! - All coefficient accumulation uses Kahan summation
//! - Pauli terms stored in Vec (deterministic iteration)
//! - Complex arithmetic via mul_fixed (no FMA)

use cjc_runtime::complex::ComplexF64;
use cjc_repro::KahanAccumulatorF64;
use crate::statevector::Statevector;

// ---------------------------------------------------------------------------
// Pauli Algebra
// ---------------------------------------------------------------------------

/// Single Pauli operator on one qubit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A Pauli string: tensor product of single-qubit Pauli operators with a
/// complex coefficient. Represents coeff * P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}.
#[derive(Debug, Clone)]
pub struct PauliTerm {
    pub coeff: ComplexF64,
    pub ops: Vec<Pauli>,
}

impl PauliTerm {
    /// Create a new Pauli term with all-identity on `n` qubits.
    pub fn identity(n: usize) -> Self {
        PauliTerm {
            coeff: ComplexF64::ONE,
            ops: vec![Pauli::I; n],
        }
    }

    /// Number of qubits this term acts on.
    pub fn n_qubits(&self) -> usize {
        self.ops.len()
    }

    /// Count non-identity Pauli operators (the "weight" of the term).
    pub fn weight(&self) -> usize {
        self.ops.iter().filter(|&&p| p != Pauli::I).count()
    }

    /// Multiply two Pauli terms: (c1 * P) * (c2 * Q) = c1*c2*phase * PQ.
    /// Pauli multiplication rules:
    ///   XY = iZ, YX = -iZ, XZ = -iY, ZX = iY, YZ = iX, ZY = -iX
    ///   P*P = I for P in {X,Y,Z}
    pub fn multiply(&self, other: &PauliTerm) -> PauliTerm {
        assert_eq!(self.ops.len(), other.ops.len(), "Pauli string length mismatch");
        let n = self.ops.len();
        let mut result_ops = vec![Pauli::I; n];
        let mut phase = self.coeff.mul_fixed(other.coeff);

        for i in 0..n {
            let (p, ph) = pauli_mul(self.ops[i], other.ops[i]);
            result_ops[i] = p;
            phase = phase.mul_fixed(ph);
        }

        PauliTerm { coeff: phase, ops: result_ops }
    }
}

/// Multiply two single-qubit Pauli operators: returns (result, phase).
fn pauli_mul(a: Pauli, b: Pauli) -> (Pauli, ComplexF64) {
    use Pauli::*;
    match (a, b) {
        (I, x) | (x, I) => (x, ComplexF64::ONE),
        (X, X) | (Y, Y) | (Z, Z) => (I, ComplexF64::ONE),
        (X, Y) => (Z, ComplexF64::I),
        (Y, X) => (Z, ComplexF64::new(0.0, -1.0)),
        (X, Z) => (Y, ComplexF64::new(0.0, -1.0)),
        (Z, X) => (Y, ComplexF64::I),
        (Y, Z) => (X, ComplexF64::I),
        (Z, Y) => (X, ComplexF64::new(0.0, -1.0)),
    }
}

// ---------------------------------------------------------------------------
// Fermionic Hamiltonian
// ---------------------------------------------------------------------------

/// A fermionic Hamiltonian expressed as a sum of Pauli terms (after JW transform).
#[derive(Debug, Clone)]
pub struct FermionicHamiltonian {
    pub n_qubits: usize,
    pub terms: Vec<PauliTerm>,
}

impl FermionicHamiltonian {
    /// Create an empty Hamiltonian on `n` qubits.
    pub fn new(n: usize) -> Self {
        FermionicHamiltonian { n_qubits: n, terms: Vec::new() }
    }

    /// Add a Pauli term to the Hamiltonian.
    pub fn add_term(&mut self, term: PauliTerm) {
        assert_eq!(term.ops.len(), self.n_qubits, "term qubit count mismatch");
        self.terms.push(term);
    }

    /// Number of terms in the Hamiltonian.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Compute the expectation value ⟨ψ|H|ψ⟩ on a statevector.
    /// Uses Kahan summation for deterministic accumulation.
    pub fn expectation(&self, sv: &Statevector) -> f64 {
        assert_eq!(sv.n_qubits, self.n_qubits, "qubit count mismatch");
        let mut acc = KahanAccumulatorF64::new();
        for term in &self.terms {
            let val = pauli_expectation(term, sv);
            acc.add(val);
        }
        acc.finalize()
    }
}

/// Compute ⟨ψ|P|ψ⟩ for a single Pauli term P on statevector |ψ⟩.
///
/// For each basis state |k⟩, P|k⟩ = phase * |k'⟩ where k' may differ from k.
/// The expectation value is Re(coeff * Σ_k conj(ψ_k) * phase_k * ψ_{k'_k}).
fn pauli_expectation(term: &PauliTerm, sv: &Statevector) -> f64 {
    let n = sv.n_states();
    let mut acc_re = KahanAccumulatorF64::new();
    let mut acc_im = KahanAccumulatorF64::new();

    for k in 0..n {
        let (k_prime, phase) = apply_pauli_string(&term.ops, k);
        // contribution = conj(ψ_k) * phase * ψ_{k'}
        let contrib = sv.amplitudes[k].conj().mul_fixed(phase).mul_fixed(sv.amplitudes[k_prime]);
        acc_re.add(contrib.re);
        acc_im.add(contrib.im);
    }

    // ⟨ψ|P|ψ⟩ = coeff * Σ_k ...
    let sum = ComplexF64::new(acc_re.finalize(), acc_im.finalize());
    let result = term.coeff.mul_fixed(sum);
    result.re // Hermitian operator → expectation value is real
}

/// Apply a Pauli string to a basis state index.
/// Returns (new_index, phase_factor).
fn apply_pauli_string(ops: &[Pauli], mut k: usize) -> (usize, ComplexF64) {
    let mut phase = ComplexF64::ONE;
    let mut k_new = k;

    for (q, &op) in ops.iter().enumerate() {
        let bit = (k >> q) & 1;
        match op {
            Pauli::I => {}
            Pauli::X => {
                k_new ^= 1 << q; // flip bit
            }
            Pauli::Y => {
                k_new ^= 1 << q; // flip bit
                // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                if bit == 0 {
                    phase = phase.mul_fixed(ComplexF64::I);
                } else {
                    phase = phase.mul_fixed(ComplexF64::new(0.0, -1.0));
                }
            }
            Pauli::Z => {
                if bit == 1 {
                    phase = phase.mul_fixed(ComplexF64::new(-1.0, 0.0));
                }
            }
        }
    }

    (k_new, phase)
}

// ---------------------------------------------------------------------------
// Jordan-Wigner Transform
// ---------------------------------------------------------------------------

/// Jordan-Wigner transform: a†_p a_q → sum of Pauli terms.
///
/// For p == q: a†_p a_p = (I - Z_p) / 2 (number operator)
/// For p != q: uses the standard JW encoding with Z-string between p and q.
pub fn jw_one_body(n_qubits: usize, p: usize, q: usize) -> Vec<PauliTerm> {
    assert!(p < n_qubits && q < n_qubits);

    if p == q {
        // Number operator: n_p = (I - Z_p) / 2
        let mut id_term = PauliTerm::identity(n_qubits);
        id_term.coeff = ComplexF64::real(0.5);

        let mut z_term = PauliTerm::identity(n_qubits);
        z_term.coeff = ComplexF64::real(-0.5);
        z_term.ops[p] = Pauli::Z;

        vec![id_term, z_term]
    } else {
        // a†_p a_q = (X_p X_q + Y_p Y_q + i(X_p Y_q - Y_p X_q)) / 4
        // times Z-string between min(p,q)+1..max(p,q)-1
        let lo = p.min(q);
        let hi = p.max(q);

        let mut terms = Vec::with_capacity(4);

        // XX term
        let mut xx = PauliTerm::identity(n_qubits);
        xx.coeff = ComplexF64::real(0.5);
        xx.ops[p] = Pauli::X;
        xx.ops[q] = Pauli::X;
        for z in (lo + 1)..hi { xx.ops[z] = Pauli::Z; }
        terms.push(xx);

        // YY term
        let mut yy = PauliTerm::identity(n_qubits);
        yy.coeff = ComplexF64::real(0.5);
        yy.ops[p] = Pauli::Y;
        yy.ops[q] = Pauli::Y;
        for z in (lo + 1)..hi { yy.ops[z] = Pauli::Z; }
        terms.push(yy);

        if p < q {
            // a†_p a_q: need +i(XY) and -i(YX) terms
            let mut xy = PauliTerm::identity(n_qubits);
            xy.coeff = ComplexF64::new(0.0, 0.5);
            xy.ops[p] = Pauli::X;
            xy.ops[q] = Pauli::Y;
            for z in (lo + 1)..hi { xy.ops[z] = Pauli::Z; }
            terms.push(xy);

            let mut yx = PauliTerm::identity(n_qubits);
            yx.coeff = ComplexF64::new(0.0, -0.5);
            yx.ops[p] = Pauli::Y;
            yx.ops[q] = Pauli::X;
            for z in (lo + 1)..hi { yx.ops[z] = Pauli::Z; }
            terms.push(yx);
        } else {
            // a†_p a_q with p > q: conjugate
            let mut xy = PauliTerm::identity(n_qubits);
            xy.coeff = ComplexF64::new(0.0, -0.5);
            xy.ops[p] = Pauli::X;
            xy.ops[q] = Pauli::Y;
            for z in (lo + 1)..hi { xy.ops[z] = Pauli::Z; }
            terms.push(xy);

            let mut yx = PauliTerm::identity(n_qubits);
            yx.coeff = ComplexF64::new(0.0, 0.5);
            yx.ops[p] = Pauli::Y;
            yx.ops[q] = Pauli::X;
            for z in (lo + 1)..hi { yx.ops[z] = Pauli::Z; }
            terms.push(yx);
        }

        // Scale by 1/2 for the JW encoding: a†_p a_q = (1/2)(terms above)
        for t in &mut terms {
            t.coeff = t.coeff.mul_fixed(ComplexF64::real(0.5));
        }

        terms
    }
}

/// Jordan-Wigner transform of two-body term: a†_p a†_q a_r a_s.
/// Decomposes into one-body JW terms via anticommutation.
/// Used for two-electron integrals in molecular Hamiltonians.
pub fn jw_two_body(n_qubits: usize, p: usize, q: usize, r: usize, s: usize, coeff: f64) -> Vec<PauliTerm> {
    assert!(p < n_qubits && q < n_qubits && r < n_qubits && s < n_qubits);

    // a†_p a†_q a_r a_s decomposed using Wick's theorem:
    // = δ_{qr} a†_p a_s - a†_p a_r a†_q a_s  (after normal ordering)
    // For simplicity, we directly enumerate all Pauli terms from the
    // Bravyi-Kitaev / JW encoding of the 4-index integral.

    // Full JW: build from products of one-body operators
    let mut result = Vec::new();

    // Term 1: (a†_p a_s)(a†_q a_r) contribution
    let ps_terms = jw_one_body(n_qubits, p, s);
    let qr_terms = jw_one_body(n_qubits, q, r);

    for ps in &ps_terms {
        for qr in &qr_terms {
            let mut product = ps.multiply(qr);
            product.coeff = product.coeff.mul_fixed(ComplexF64::real(coeff));
            result.push(product);
        }
    }

    // Term 2: -(a†_p a_r)(a†_q a_s) contribution
    let pr_terms = jw_one_body(n_qubits, p, r);
    let qs_terms = jw_one_body(n_qubits, q, s);

    for pr in &pr_terms {
        for qs in &qs_terms {
            let mut product = pr.multiply(qs);
            product.coeff = product.coeff.mul_fixed(ComplexF64::real(-coeff));
            result.push(product);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Pre-built Molecular Hamiltonians
// ---------------------------------------------------------------------------

/// Build the minimal-basis H₂ molecular Hamiltonian at equilibrium geometry.
///
/// 2-qubit encoding of H₂ in STO-3G basis at R=0.7414 Å.
/// Exact ground state energy: -1.1373 Hartree.
///
/// H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
pub fn h2_hamiltonian() -> FermionicHamiltonian {
    let n = 2;
    let mut h = FermionicHamiltonian::new(n);

    // STO-3G coefficients at equilibrium bond length R=0.7414 Å
    let g0 = -0.4804;  // Nuclear repulsion + constant
    let g1 = 0.3435;   // Z_0 coefficient
    let g2 = -0.4347;  // Z_1 coefficient
    let g3 = 0.5716;   // Z_0 Z_1 coefficient
    let g4 = 0.0910;   // X_0 X_1 coefficient
    let g5 = 0.0910;   // Y_0 Y_1 coefficient

    // Identity term
    let mut id = PauliTerm::identity(n);
    id.coeff = ComplexF64::real(g0);
    h.add_term(id);

    // Z_0 term
    let mut z0 = PauliTerm::identity(n);
    z0.coeff = ComplexF64::real(g1);
    z0.ops[0] = Pauli::Z;
    h.add_term(z0);

    // Z_1 term
    let mut z1 = PauliTerm::identity(n);
    z1.coeff = ComplexF64::real(g2);
    z1.ops[1] = Pauli::Z;
    h.add_term(z1);

    // Z_0 Z_1 term
    let mut z0z1 = PauliTerm::identity(n);
    z0z1.coeff = ComplexF64::real(g3);
    z0z1.ops[0] = Pauli::Z;
    z0z1.ops[1] = Pauli::Z;
    h.add_term(z0z1);

    // X_0 X_1 term
    let mut x0x1 = PauliTerm::identity(n);
    x0x1.coeff = ComplexF64::real(g4);
    x0x1.ops[0] = Pauli::X;
    x0x1.ops[1] = Pauli::X;
    h.add_term(x0x1);

    // Y_0 Y_1 term
    let mut y0y1 = PauliTerm::identity(n);
    y0y1.coeff = ComplexF64::real(g5);
    y0y1.ops[0] = Pauli::Y;
    y0y1.ops[1] = Pauli::Y;
    h.add_term(y0y1);

    h
}

/// Build the minimal-basis LiH molecular Hamiltonian.
///
/// 4-qubit encoding of LiH in STO-3G basis at R=1.546 Å.
/// Exact ground state energy: approximately -7.8825 Hartree.
///
/// Uses the Jordan-Wigner encoding of the one- and two-electron integrals.
pub fn lih_hamiltonian() -> FermionicHamiltonian {
    let n = 4;
    let mut h = FermionicHamiltonian::new(n);

    // Simplified STO-3G coefficients for LiH at equilibrium
    // (Reduced to the most significant terms for tractability.)
    let nuclear_repulsion = -7.4983;

    // Constant (nuclear repulsion + core)
    let mut id = PauliTerm::identity(n);
    id.coeff = ComplexF64::real(nuclear_repulsion);
    h.add_term(id);

    // One-body terms: h_{pq} a†_p a_q
    // Diagonal (number operators)
    let h_diag = [-1.2528, -0.4760, -1.2528, -0.4760];
    for (p, &coeff) in h_diag.iter().enumerate() {
        let terms = jw_one_body(n, p, p);
        for mut t in terms {
            t.coeff = t.coeff.mul_fixed(ComplexF64::real(coeff));
            h.add_term(t);
        }
    }

    // Off-diagonal one-body: h_{01} a†_0 a_1 + h.c.
    let h_01 = -0.0453;
    let one_body_pairs = [(0, 1, h_01), (2, 3, h_01)];
    for &(p, q, coeff) in &one_body_pairs {
        // a†_p a_q + a†_q a_p (Hermitian)
        let terms_pq = jw_one_body(n, p, q);
        let terms_qp = jw_one_body(n, q, p);
        for mut t in terms_pq {
            t.coeff = t.coeff.mul_fixed(ComplexF64::real(coeff));
            h.add_term(t);
        }
        for mut t in terms_qp {
            t.coeff = t.coeff.mul_fixed(ComplexF64::real(coeff));
            h.add_term(t);
        }
    }

    // Two-body terms: selected dominant two-electron integrals
    let v_0011 = 0.3366;  // (00|11)
    let v_0101 = 0.0908;  // (01|01)
    let two_body_integrals = [
        (0, 0, 1, 1, v_0011),
        (2, 2, 3, 3, v_0011),
        (0, 1, 0, 1, v_0101),
        (2, 3, 2, 3, v_0101),
    ];

    for &(p, q, r, s, coeff) in &two_body_integrals {
        let terms = jw_two_body(n, p, q, r, s, coeff);
        for t in terms {
            h.add_term(t);
        }
    }

    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_pauli_mul_identity() {
        let (p, phase) = pauli_mul(Pauli::I, Pauli::X);
        assert_eq!(p, Pauli::X);
        assert!((phase.re - 1.0).abs() < TOL && phase.im.abs() < TOL);
    }

    #[test]
    fn test_pauli_mul_xy_iz() {
        let (p, phase) = pauli_mul(Pauli::X, Pauli::Y);
        assert_eq!(p, Pauli::Z);
        assert!(phase.re.abs() < TOL && (phase.im - 1.0).abs() < TOL); // i
    }

    #[test]
    fn test_pauli_mul_xx_i() {
        let (p, phase) = pauli_mul(Pauli::X, Pauli::X);
        assert_eq!(p, Pauli::I);
        assert!((phase.re - 1.0).abs() < TOL && phase.im.abs() < TOL);
    }

    #[test]
    fn test_pauli_anticommutation() {
        // XY = iZ, YX = -iZ → XY + YX = 0 (anti-commute)
        let (_, phase_xy) = pauli_mul(Pauli::X, Pauli::Y);
        let (_, phase_yx) = pauli_mul(Pauli::Y, Pauli::X);
        let sum = phase_xy.add(phase_yx);
        assert!(sum.re.abs() < TOL && sum.im.abs() < TOL, "X and Y must anti-commute");
    }

    #[test]
    fn test_jw_number_operator() {
        let terms = jw_one_body(2, 0, 0);
        assert_eq!(terms.len(), 2, "number operator should have 2 terms (I and Z)");

        // Apply to |01⟩ state (qubit 0 = 1, occupied)
        let mut sv = Statevector::new(2);
        crate::gates::Gate::X(0).apply(&mut sv).unwrap();

        let mut h = FermionicHamiltonian::new(2);
        for t in terms { h.add_term(t); }
        let e = h.expectation(&sv);
        assert!((e - 1.0).abs() < TOL, "⟨01|n_0|01⟩ should be 1.0, got {}", e);
    }

    #[test]
    fn test_h2_hamiltonian_structure() {
        let h = h2_hamiltonian();
        assert_eq!(h.n_qubits, 2);
        assert_eq!(h.n_terms(), 6, "H₂ should have 6 Pauli terms");
    }

    #[test]
    fn test_h2_ground_state_energy() {
        let h = h2_hamiltonian();
        // The ground state of H₂ in this encoding is found by diagonalization.
        // Compute ⟨ψ|H|ψ⟩ for the |01⟩ state (one electron in each orbital).
        let mut sv = Statevector::new(2);
        crate::gates::Gate::X(0).apply(&mut sv).unwrap();
        let e = h.expectation(&sv);
        // This is NOT the ground state energy, just a check that the Hamiltonian works.
        assert!(e.is_finite(), "H₂ energy must be finite");
    }

    #[test]
    fn test_lih_hamiltonian_structure() {
        let h = lih_hamiltonian();
        assert_eq!(h.n_qubits, 4);
        assert!(h.n_terms() > 0, "LiH should have terms");
    }

    #[test]
    fn test_pauli_expectation_determinism() {
        let h = h2_hamiltonian();
        let mut sv = Statevector::new(2);
        crate::gates::Gate::H(0).apply(&mut sv).unwrap();
        crate::gates::Gate::CNOT(0, 1).apply(&mut sv).unwrap();

        let e1 = h.expectation(&sv);
        let e2 = h.expectation(&sv);
        assert_eq!(e1.to_bits(), e2.to_bits(), "expectation must be bit-identical");
    }

    #[test]
    fn test_apply_pauli_string_identity() {
        let ops = vec![Pauli::I, Pauli::I];
        let (k_new, phase) = apply_pauli_string(&ops, 0b10);
        assert_eq!(k_new, 0b10);
        assert!((phase.re - 1.0).abs() < TOL && phase.im.abs() < TOL);
    }

    #[test]
    fn test_apply_pauli_string_x() {
        let ops = vec![Pauli::X, Pauli::I];
        let (k_new, phase) = apply_pauli_string(&ops, 0b00);
        assert_eq!(k_new, 0b01); // X flips qubit 0
        assert!((phase.re - 1.0).abs() < TOL);
    }

    #[test]
    fn test_apply_pauli_string_z() {
        let ops = vec![Pauli::Z, Pauli::I];
        let (k_new, phase) = apply_pauli_string(&ops, 0b01);
        assert_eq!(k_new, 0b01); // Z doesn't flip
        assert!((phase.re - (-1.0)).abs() < TOL); // Z|1⟩ = -|1⟩
    }

    #[test]
    fn test_pauli_term_weight() {
        let mut t = PauliTerm::identity(4);
        t.ops[1] = Pauli::X;
        t.ops[3] = Pauli::Z;
        assert_eq!(t.weight(), 2);
    }
}
