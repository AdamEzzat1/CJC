//! Quantum Gates — unitary operations on statevectors.
//!
//! Each gate is represented as a 2×2 or 4×4 complex matrix (for 1- and 2-qubit
//! gates respectively). Gate application uses bit-manipulation of basis state
//! indices to identify affected amplitude pairs, then applies the unitary
//! matrix in ascending index order for determinism.
//!
//! # Determinism
//!
//! - All complex arithmetic uses `mul_fixed` (no FMA)
//! - Basis states are processed in ascending index order
//! - No HashMap or non-deterministic data structures

use cjc_runtime::complex::ComplexF64;
use crate::statevector::Statevector;

/// A quantum gate represented by its unitary matrix and target qubits.
#[derive(Debug, Clone)]
pub enum Gate {
    // --- Single-qubit gates (2×2 matrix) ---
    /// Hadamard gate: creates equal superposition.
    H(usize),
    /// Pauli-X (NOT) gate: bit flip.
    X(usize),
    /// Pauli-Y gate.
    Y(usize),
    /// Pauli-Z gate: phase flip.
    Z(usize),
    /// S gate (√Z): π/2 phase.
    S(usize),
    /// T gate (√S): π/4 phase.
    T(usize),
    /// Rx(θ): rotation around X axis.
    Rx(usize, f64),
    /// Ry(θ): rotation around Y axis.
    Ry(usize, f64),
    /// Rz(θ): rotation around Z axis.
    Rz(usize, f64),

    // --- Two-qubit gates ---
    /// CNOT (CX): controlled-NOT. First arg = control, second = target.
    CNOT(usize, usize),
    /// CZ: controlled-Z. Symmetric.
    CZ(usize, usize),
    /// SWAP: exchange two qubits.
    SWAP(usize, usize),

    // --- Three-qubit gates ---
    /// Toffoli (CCX): doubly-controlled NOT.
    Toffoli(usize, usize, usize),
}

impl Gate {
    /// Return the qubit indices this gate acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Gate::H(q) | Gate::X(q) | Gate::Y(q) | Gate::Z(q)
            | Gate::S(q) | Gate::T(q)
            | Gate::Rx(q, _) | Gate::Ry(q, _) | Gate::Rz(q, _) => vec![*q],
            Gate::CNOT(a, b) | Gate::CZ(a, b) | Gate::SWAP(a, b) => vec![*a, *b],
            Gate::Toffoli(a, b, c) => vec![*a, *b, *c],
        }
    }

    /// Apply this gate to a statevector in-place.
    /// Returns an error if any qubit index is out of range.
    pub fn apply(&self, sv: &mut Statevector) -> Result<(), String> {
        // Validate all qubit indices
        for &q in &self.qubits() {
            sv.validate_qubit(q)?;
        }

        match self {
            // --- Single-qubit gates ---
            Gate::H(q) => apply_single_qubit(sv, *q, h_matrix()),
            Gate::X(q) => apply_single_qubit(sv, *q, x_matrix()),
            Gate::Y(q) => apply_single_qubit(sv, *q, y_matrix()),
            Gate::Z(q) => apply_single_qubit(sv, *q, z_matrix()),
            Gate::S(q) => apply_single_qubit(sv, *q, s_matrix()),
            Gate::T(q) => apply_single_qubit(sv, *q, t_matrix()),
            Gate::Rx(q, theta) => apply_single_qubit(sv, *q, rx_matrix(*theta)),
            Gate::Ry(q, theta) => apply_single_qubit(sv, *q, ry_matrix(*theta)),
            Gate::Rz(q, theta) => apply_single_qubit(sv, *q, rz_matrix(*theta)),

            // --- Two-qubit gates ---
            Gate::CNOT(ctrl, tgt) => apply_cnot(sv, *ctrl, *tgt),
            Gate::CZ(a, b) => apply_cz(sv, *a, *b),
            Gate::SWAP(a, b) => apply_swap(sv, *a, *b),

            // --- Three-qubit gates ---
            Gate::Toffoli(c1, c2, tgt) => apply_toffoli(sv, *c1, *c2, *tgt),
        }

        Ok(())
    }
}

impl std::fmt::Display for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gate::H(q) => write!(f, "H({})", q),
            Gate::X(q) => write!(f, "X({})", q),
            Gate::Y(q) => write!(f, "Y({})", q),
            Gate::Z(q) => write!(f, "Z({})", q),
            Gate::S(q) => write!(f, "S({})", q),
            Gate::T(q) => write!(f, "T({})", q),
            Gate::Rx(q, t) => write!(f, "Rx({}, {:.4})", q, t),
            Gate::Ry(q, t) => write!(f, "Ry({}, {:.4})", q, t),
            Gate::Rz(q, t) => write!(f, "Rz({}, {:.4})", q, t),
            Gate::CNOT(c, t) => write!(f, "CNOT({}, {})", c, t),
            Gate::CZ(a, b) => write!(f, "CZ({}, {})", a, b),
            Gate::SWAP(a, b) => write!(f, "SWAP({}, {})", a, b),
            Gate::Toffoli(a, b, c) => write!(f, "Toffoli({}, {}, {})", a, b, c),
        }
    }
}

// ---------------------------------------------------------------------------
// Gate matrices (2×2 represented as [[ComplexF64; 2]; 2])
// ---------------------------------------------------------------------------

type Mat2x2 = [[ComplexF64; 2]; 2];

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn h_matrix() -> Mat2x2 {
    let s = ComplexF64::real(INV_SQRT2);
    let ms = ComplexF64::real(-INV_SQRT2);
    [[s, s], [s, ms]]
}

fn x_matrix() -> Mat2x2 {
    [[ComplexF64::ZERO, ComplexF64::ONE],
     [ComplexF64::ONE, ComplexF64::ZERO]]
}

fn y_matrix() -> Mat2x2 {
    [[ComplexF64::ZERO, ComplexF64::new(0.0, -1.0)],
     [ComplexF64::new(0.0, 1.0), ComplexF64::ZERO]]
}

fn z_matrix() -> Mat2x2 {
    [[ComplexF64::ONE, ComplexF64::ZERO],
     [ComplexF64::ZERO, ComplexF64::new(-1.0, 0.0)]]
}

fn s_matrix() -> Mat2x2 {
    [[ComplexF64::ONE, ComplexF64::ZERO],
     [ComplexF64::ZERO, ComplexF64::I]]
}

fn t_matrix() -> Mat2x2 {
    let phase = ComplexF64::new(INV_SQRT2, INV_SQRT2); // e^(iπ/4)
    [[ComplexF64::ONE, ComplexF64::ZERO],
     [ComplexF64::ZERO, phase]]
}

fn rx_matrix(theta: f64) -> Mat2x2 {
    let c = ComplexF64::real((theta / 2.0).cos());
    let s = ComplexF64::new(0.0, -(theta / 2.0).sin());
    [[c, s], [s, c]]
}

fn ry_matrix(theta: f64) -> Mat2x2 {
    let c = ComplexF64::real((theta / 2.0).cos());
    let s = ComplexF64::real((theta / 2.0).sin());
    let ms = ComplexF64::real(-(theta / 2.0).sin());
    [[c, ms], [s, c]]
}

fn rz_matrix(theta: f64) -> Mat2x2 {
    let pos = ComplexF64::new((theta / 2.0).cos(), (theta / 2.0).sin());
    let neg = ComplexF64::new((theta / 2.0).cos(), -(theta / 2.0).sin());
    [[neg, ComplexF64::ZERO],
     [ComplexF64::ZERO, pos]]
}

// ---------------------------------------------------------------------------
// Single-qubit gate application
// ---------------------------------------------------------------------------

/// Apply a 2×2 unitary matrix to qubit `q` of the statevector.
///
/// For each pair of basis states that differ only in bit `q`, apply:
///   |0⟩ component → U[0][0]*α₀ + U[0][1]*α₁
///   |1⟩ component → U[1][0]*α₀ + U[1][1]*α₁
///
/// Processes pairs in ascending index order for determinism.
fn apply_single_qubit(sv: &mut Statevector, q: usize, u: Mat2x2) {
    let n = sv.n_states();
    let bit = 1usize << q;

    // Iterate over basis states in ascending order.
    // For each state where bit `q` is 0, compute the pair.
    let mut i = 0;
    while i < n {
        if i & bit == 0 {
            let j = i | bit; // j has bit q set to 1
            let a0 = sv.amplitudes[i];
            let a1 = sv.amplitudes[j];

            // Fixed-sequence complex arithmetic (no FMA):
            // new_a0 = U[0][0]*a0 + U[0][1]*a1
            // new_a1 = U[1][0]*a0 + U[1][1]*a1
            sv.amplitudes[i] = u[0][0].mul_fixed(a0).add(u[0][1].mul_fixed(a1));
            sv.amplitudes[j] = u[1][0].mul_fixed(a0).add(u[1][1].mul_fixed(a1));
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Two-qubit gate application
// ---------------------------------------------------------------------------

/// CNOT: flip target qubit when control qubit is |1⟩.
fn apply_cnot(sv: &mut Statevector, ctrl: usize, tgt: usize) {
    let n = sv.n_states();
    let ctrl_bit = 1usize << ctrl;
    let tgt_bit = 1usize << tgt;

    for i in 0..n {
        // Only process each pair once: when control is 1 and target is 0
        if (i & ctrl_bit) != 0 && (i & tgt_bit) == 0 {
            let j = i | tgt_bit;
            sv.amplitudes.swap(i, j);
        }
    }
}

/// CZ: apply phase flip (-1) when both qubits are |1⟩.
fn apply_cz(sv: &mut Statevector, a: usize, b: usize) {
    let n = sv.n_states();
    let a_bit = 1usize << a;
    let b_bit = 1usize << b;

    for i in 0..n {
        if (i & a_bit) != 0 && (i & b_bit) != 0 {
            sv.amplitudes[i] = sv.amplitudes[i].neg();
        }
    }
}

/// SWAP: exchange amplitudes of two qubits.
fn apply_swap(sv: &mut Statevector, a: usize, b: usize) {
    let n = sv.n_states();
    let a_bit = 1usize << a;
    let b_bit = 1usize << b;

    for i in 0..n {
        // Only swap when qubit a=0, qubit b=1 (to process each pair once)
        if (i & a_bit) == 0 && (i & b_bit) != 0 {
            let j = (i | a_bit) & !b_bit; // flip: a→1, b→0
            sv.amplitudes.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Three-qubit gate application
// ---------------------------------------------------------------------------

/// Toffoli (CCX): flip target when both controls are |1⟩.
fn apply_toffoli(sv: &mut Statevector, c1: usize, c2: usize, tgt: usize) {
    let n = sv.n_states();
    let c1_bit = 1usize << c1;
    let c2_bit = 1usize << c2;
    let tgt_bit = 1usize << tgt;

    for i in 0..n {
        // Both controls are 1, target is 0 → swap with target=1
        if (i & c1_bit) != 0 && (i & c2_bit) != 0 && (i & tgt_bit) == 0 {
            let j = i | tgt_bit;
            sv.amplitudes.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn assert_approx(a: ComplexF64, b: ComplexF64, msg: &str) {
        assert!((a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL,
            "{}: got ({}, {}) expected ({}, {})", msg, a.re, a.im, b.re, b.im);
    }

    #[test]
    fn test_h_creates_superposition() {
        let mut sv = Statevector::new(1);
        Gate::H(0).apply(&mut sv).unwrap();
        let inv = INV_SQRT2;
        assert_approx(sv.amplitudes[0], ComplexF64::real(inv), "H|0⟩[0]");
        assert_approx(sv.amplitudes[1], ComplexF64::real(inv), "H|0⟩[1]");
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_x_bit_flip() {
        let mut sv = Statevector::new(1);
        Gate::X(0).apply(&mut sv).unwrap();
        assert_approx(sv.amplitudes[0], ComplexF64::ZERO, "X|0⟩[0]");
        assert_approx(sv.amplitudes[1], ComplexF64::ONE, "X|0⟩[1]");
    }

    #[test]
    fn test_x_involution() {
        let mut sv = Statevector::new(1);
        Gate::X(0).apply(&mut sv).unwrap();
        Gate::X(0).apply(&mut sv).unwrap();
        assert_approx(sv.amplitudes[0], ComplexF64::ONE, "X²|0⟩[0]");
        assert_approx(sv.amplitudes[1], ComplexF64::ZERO, "X²|0⟩[1]");
    }

    #[test]
    fn test_z_phase_flip() {
        // Z|+⟩ = |−⟩
        let mut sv = Statevector::new(1);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::Z(0).apply(&mut sv).unwrap();
        let inv = INV_SQRT2;
        assert_approx(sv.amplitudes[0], ComplexF64::real(inv), "Z|+⟩[0]");
        assert_approx(sv.amplitudes[1], ComplexF64::real(-inv), "Z|+⟩[1]");
    }

    #[test]
    fn test_y_gate() {
        let mut sv = Statevector::new(1);
        Gate::Y(0).apply(&mut sv).unwrap();
        assert_approx(sv.amplitudes[0], ComplexF64::ZERO, "Y|0⟩[0]");
        assert_approx(sv.amplitudes[1], ComplexF64::new(0.0, 1.0), "Y|0⟩[1]");
    }

    #[test]
    fn test_s_gate() {
        // S|1⟩ = i|1⟩
        let mut sv = Statevector::new(1);
        Gate::X(0).apply(&mut sv).unwrap(); // |1⟩
        Gate::S(0).apply(&mut sv).unwrap();
        assert_approx(sv.amplitudes[0], ComplexF64::ZERO, "S|1⟩[0]");
        assert_approx(sv.amplitudes[1], ComplexF64::I, "S|1⟩[1]");
    }

    #[test]
    fn test_t_gate() {
        // T|1⟩ = e^(iπ/4)|1⟩
        let mut sv = Statevector::new(1);
        Gate::X(0).apply(&mut sv).unwrap();
        Gate::T(0).apply(&mut sv).unwrap();
        let expected = ComplexF64::new(INV_SQRT2, INV_SQRT2);
        assert_approx(sv.amplitudes[1], expected, "T|1⟩");
    }

    #[test]
    fn test_rx_pi_equals_x() {
        // Rx(π) = -iX (up to global phase)
        let mut sv = Statevector::new(1);
        Gate::Rx(0, std::f64::consts::PI).apply(&mut sv).unwrap();
        // |0⟩ → cos(π/2)|0⟩ - i·sin(π/2)|1⟩ = -i|1⟩
        assert!((sv.amplitudes[0].norm_sq()).abs() < TOL, "Rx(π)|0⟩ should have no |0⟩");
        assert!((sv.amplitudes[1].norm_sq() - 1.0).abs() < TOL, "Rx(π)|0⟩ should be all |1⟩");
    }

    #[test]
    fn test_ry_pi_flips() {
        // Ry(π)|0⟩ = |1⟩
        let mut sv = Statevector::new(1);
        Gate::Ry(0, std::f64::consts::PI).apply(&mut sv).unwrap();
        assert!((sv.amplitudes[0].norm_sq()).abs() < TOL);
        assert!((sv.amplitudes[1].norm_sq() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_rz_preserves_basis() {
        // Rz(θ)|0⟩ = e^(-iθ/2)|0⟩ (stays in |0⟩)
        let mut sv = Statevector::new(1);
        Gate::Rz(0, 1.23).apply(&mut sv).unwrap();
        assert!((sv.amplitudes[0].norm_sq() - 1.0).abs() < TOL);
        assert!((sv.amplitudes[1].norm_sq()).abs() < TOL);
    }

    #[test]
    fn test_cnot_entangles() {
        // H|0⟩ ⊗ |0⟩ → CNOT → Bell state (|00⟩ + |11⟩)/√2
        let mut sv = Statevector::new(2);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::CNOT(0, 1).apply(&mut sv).unwrap();
        let inv = INV_SQRT2;
        assert_approx(sv.amplitudes[0], ComplexF64::real(inv), "Bell|00⟩");
        assert_approx(sv.amplitudes[1], ComplexF64::ZERO, "Bell|01⟩");
        assert_approx(sv.amplitudes[2], ComplexF64::ZERO, "Bell|10⟩");
        assert_approx(sv.amplitudes[3], ComplexF64::real(inv), "Bell|11⟩");
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_cz_symmetric() {
        // CZ is symmetric in control/target
        let mut sv1 = Statevector::new(2);
        Gate::H(0).apply(&mut sv1).unwrap();
        Gate::H(1).apply(&mut sv1).unwrap();
        let mut sv2 = sv1.clone();

        Gate::CZ(0, 1).apply(&mut sv1).unwrap();
        Gate::CZ(1, 0).apply(&mut sv2).unwrap();

        for i in 0..4 {
            assert_approx(sv1.amplitudes[i], sv2.amplitudes[i],
                &format!("CZ symmetry [{}]", i));
        }
    }

    #[test]
    fn test_swap_exchanges_qubits() {
        // Prepare |10⟩, SWAP → |01⟩
        let mut sv = Statevector::new(2);
        Gate::X(0).apply(&mut sv).unwrap(); // |01⟩ in little-endian = index 1
        // State is now: [0, 1, 0, 0] → qubit 0 is |1⟩, qubit 1 is |0⟩
        Gate::SWAP(0, 1).apply(&mut sv).unwrap();
        // After SWAP: qubit 0 is |0⟩, qubit 1 is |1⟩ → index 2
        assert_approx(sv.amplitudes[2], ComplexF64::ONE, "SWAP|10⟩→|01⟩");
    }

    #[test]
    fn test_swap_involution() {
        let mut sv = Statevector::new(2);
        Gate::H(0).apply(&mut sv).unwrap();
        let before = sv.amplitudes.clone();
        Gate::SWAP(0, 1).apply(&mut sv).unwrap();
        Gate::SWAP(0, 1).apply(&mut sv).unwrap();
        for i in 0..4 {
            assert_approx(sv.amplitudes[i], before[i], &format!("SWAP² [{}]", i));
        }
    }

    #[test]
    fn test_toffoli_flips_only_when_both_controls_set() {
        // |110⟩ → Toffoli(0,1,2) → |111⟩
        let mut sv = Statevector::new(3);
        Gate::X(0).apply(&mut sv).unwrap();
        Gate::X(1).apply(&mut sv).unwrap();
        // State: |11⟩ on qubits 0,1 = index 3 (binary 011)
        Gate::Toffoli(0, 1, 2).apply(&mut sv).unwrap();
        // Should flip qubit 2: index 3 → index 7 (binary 111)
        assert_approx(sv.amplitudes[7], ComplexF64::ONE, "Toffoli|110⟩→|111⟩");
        assert_approx(sv.amplitudes[3], ComplexF64::ZERO, "Toffoli clears |110⟩");
    }

    #[test]
    fn test_toffoli_no_flip_single_control() {
        // |100⟩ → Toffoli(0,1,2) → |100⟩ (no change, only one control set)
        let mut sv = Statevector::new(3);
        Gate::X(0).apply(&mut sv).unwrap();
        Gate::Toffoli(0, 1, 2).apply(&mut sv).unwrap();
        assert_approx(sv.amplitudes[1], ComplexF64::ONE, "Toffoli single ctrl");
    }

    #[test]
    fn test_gate_out_of_range() {
        let mut sv = Statevector::new(2);
        assert!(Gate::H(2).apply(&mut sv).is_err());
        assert!(Gate::CNOT(0, 3).apply(&mut sv).is_err());
    }

    #[test]
    fn test_h_h_identity() {
        // H² = I
        let mut sv = Statevector::new(1);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::H(0).apply(&mut sv).unwrap();
        assert_approx(sv.amplitudes[0], ComplexF64::ONE, "H²|0⟩[0]");
        assert!((sv.amplitudes[1].norm_sq()).abs() < TOL, "H²|0⟩[1]");
    }

    #[test]
    fn test_unitarity_preserves_norm() {
        // Apply a sequence of gates, norm should stay 1.
        let mut sv = Statevector::new(3);
        Gate::H(0).apply(&mut sv).unwrap();
        Gate::CNOT(0, 1).apply(&mut sv).unwrap();
        Gate::T(2).apply(&mut sv).unwrap();
        Gate::Ry(1, 0.7).apply(&mut sv).unwrap();
        Gate::CZ(1, 2).apply(&mut sv).unwrap();
        assert!(sv.is_normalized(TOL), "Norm preserved after gate sequence");
    }
}
