//! Adjoint Differentiation — Memory-efficient gradient computation for quantum circuits.
//!
//! The adjoint method computes gradients of expectation values with respect to
//! circuit parameters using O(1) auxiliary memory instead of O(gates) for
//! standard backpropagation through the circuit.
//!
//! # Algorithm
//!
//! Given a parameterized circuit U(θ) = U_L(θ_L) ... U_2(θ_2) U_1(θ_1):
//!
//! 1. Forward pass: compute |ψ⟩ = U(θ)|0⟩
//! 2. Initialize adjoint state: |λ⟩ = O|ψ⟩ (observable applied to final state)
//! 3. Reverse sweep: for each gate U_k from L down to 1:
//!    a. Unapply gate: |ψ⟩ ← U_k†|ψ⟩
//!    b. Compute gradient: ∂E/∂θ_k = -2 Im(⟨λ|∂U_k/∂θ_k|ψ⟩)
//!    c. Unapply gate from adjoint: |λ⟩ ← U_k†|λ⟩
//!
//! # Determinism
//!
//! - All inner products use Kahan summation
//! - Gate unapplication uses the same fixed-sequence complex arithmetic
//! - Basis states processed in ascending order

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::complex::ComplexF64;
use crate::gates::Gate;
use crate::statevector::Statevector;
use crate::circuit::Circuit;

/// Result of adjoint differentiation: gradients for each parameterized gate.
#[derive(Debug, Clone)]
pub struct AdjointGradients {
    /// Gradient for each gate in the circuit. Non-parameterized gates have
    /// gradient = 0.0. Indexed by gate position in the circuit.
    pub gradients: Vec<f64>,
}

/// Compute gradients of an expectation value using the adjoint method.
///
/// The expectation value is E = ⟨ψ|O|ψ⟩ where O is a diagonal observable
/// specified by its eigenvalues.
///
/// # Arguments
///
/// * `circuit` - The parameterized quantum circuit
/// * `observable` - Diagonal observable eigenvalues (length 2^n_qubits)
///
/// # Returns
///
/// Gradients ∂E/∂θ_k for each parameterized gate.
pub fn adjoint_differentiation(
    circuit: &Circuit,
    observable: &[f64],
) -> Result<AdjointGradients, String> {
    let n_qubits = circuit.n_qubits();
    let n_states = 1usize << n_qubits;

    if observable.len() != n_states {
        return Err(format!(
            "observable length {} != 2^n_qubits = {}",
            observable.len(), n_states
        ));
    }

    // Step 1: Forward pass — compute |ψ⟩ = U(θ)|0⟩
    let psi = circuit.execute()?;

    // Step 2: Initialize adjoint state |λ⟩ = O|ψ⟩
    let mut lambda = psi.clone();
    for i in 0..n_states {
        lambda.amplitudes[i] = lambda.amplitudes[i].scale(observable[i]);
    }

    // Step 3: Reverse sweep through gates
    let gates = circuit.gates();
    let n_gates = gates.len();
    let mut gradients = vec![0.0f64; n_gates];

    // We need the state |ψ⟩ at each intermediate point.
    // We reconstruct it by unapplying gates in reverse.
    let mut psi_state = psi;

    for k in (0..n_gates).rev() {
        let gate = &gates[k];

        // Unapply gate from |ψ⟩: |ψ⟩ ← U_k†|ψ⟩
        unapply_gate(&mut psi_state, gate)?;

        // Compute gradient if this is a parameterized gate
        if let Some(grad) = compute_parameter_gradient(&lambda, &psi_state, gate) {
            gradients[k] = grad;
        }

        // Unapply gate from |λ⟩: |λ⟩ ← U_k†|λ⟩
        unapply_gate(&mut lambda, gate)?;
    }

    Ok(AdjointGradients { gradients })
}

/// Compute the expectation value ⟨ψ|O|ψ⟩ for a diagonal observable.
pub fn expectation_value(sv: &Statevector, observable: &[f64]) -> Result<f64, String> {
    if observable.len() != sv.n_states() {
        return Err(format!(
            "observable length {} != n_states {}",
            observable.len(), sv.n_states()
        ));
    }

    let mut acc = KahanAccumulatorF64::new();
    for (i, &obs) in observable.iter().enumerate() {
        acc.add(sv.amplitudes[i].norm_sq() * obs);
    }
    Ok(acc.finalize())
}

/// Unapply a gate (apply its adjoint/dagger).
fn unapply_gate(sv: &mut Statevector, gate: &Gate) -> Result<(), String> {
    // The adjoint of each gate:
    // H† = H, X† = X, Y† = Y, Z† = Z (self-adjoint)
    // S† = S_dag, T† = T_dag
    // Rx(θ)† = Rx(-θ), Ry(θ)† = Ry(-θ), Rz(θ)† = Rz(-θ)
    // CNOT† = CNOT, CZ† = CZ, SWAP† = SWAP, Toffoli† = Toffoli
    let adjoint = gate_adjoint(gate);
    adjoint.apply(sv)
}

/// Construct the adjoint (dagger) of a gate.
fn gate_adjoint(gate: &Gate) -> Gate {
    match gate {
        // Self-adjoint gates
        Gate::H(q) => Gate::H(*q),
        Gate::X(q) => Gate::X(*q),
        Gate::Y(q) => Gate::Y(*q),
        Gate::Z(q) => Gate::Z(*q),
        Gate::CNOT(c, t) => Gate::CNOT(*c, *t),
        Gate::CZ(a, b) => Gate::CZ(*a, *b),
        Gate::SWAP(a, b) => Gate::SWAP(*a, *b),
        Gate::Toffoli(a, b, c) => Gate::Toffoli(*a, *b, *c),
        // Rotation gates: adjoint negates the angle
        Gate::Rx(q, theta) => Gate::Rx(*q, -theta),
        Gate::Ry(q, theta) => Gate::Ry(*q, -theta),
        Gate::Rz(q, theta) => Gate::Rz(*q, -theta),
        // S† and T† are Rz with negated angles
        Gate::S(q) => Gate::Rz(*q, -std::f64::consts::FRAC_PI_2),
        Gate::T(q) => Gate::Rz(*q, -std::f64::consts::FRAC_PI_4),
    }
}

/// Compute the parameter gradient for a single gate.
///
/// For rotation gates Rx(θ), Ry(θ), Rz(θ):
/// ∂E/∂θ = -2 Im(⟨λ| ∂U/∂θ |ψ⟩)
///
/// where ∂Rx/∂θ = (-i/2) X · Rx(θ), etc.
///
/// For non-parameterized gates, returns None.
fn compute_parameter_gradient(
    lambda: &Statevector,
    psi: &Statevector,
    gate: &Gate,
) -> Option<f64> {
    match gate {
        Gate::Rx(q, theta) => {
            // ∂Rx(θ)/∂θ |ψ⟩ = (-i/2) X Rx(θ) |ψ⟩
            // But we have |ψ⟩ = state BEFORE gate application.
            // So we need: (-i/2) X Rx(θ) |ψ⟩
            let mut shifted = psi.clone();
            Gate::Rx(*q, *theta).apply(&mut shifted).ok()?;
            Gate::X(*q).apply(&mut shifted).ok()?;
            // Scale by -i/2
            let inner = inner_product(&lambda.amplitudes, &shifted.amplitudes);
            let scaled = ComplexF64::new(0.0, -0.5).mul_fixed(inner);
            Some(2.0 * scaled.re)
        }
        Gate::Ry(q, theta) => {
            // ∂Ry(θ)/∂θ |ψ⟩ = (-i/2) Y Ry(θ) |ψ⟩
            let mut shifted = psi.clone();
            Gate::Ry(*q, *theta).apply(&mut shifted).ok()?;
            Gate::Y(*q).apply(&mut shifted).ok()?;
            let inner = inner_product(&lambda.amplitudes, &shifted.amplitudes);
            let scaled = ComplexF64::new(0.0, -0.5).mul_fixed(inner);
            Some(2.0 * scaled.re)
        }
        Gate::Rz(q, theta) => {
            // ∂Rz(θ)/∂θ |ψ⟩ = (-i/2) Z Rz(θ) |ψ⟩
            let mut shifted = psi.clone();
            Gate::Rz(*q, *theta).apply(&mut shifted).ok()?;
            Gate::Z(*q).apply(&mut shifted).ok()?;
            let inner = inner_product(&lambda.amplitudes, &shifted.amplitudes);
            let scaled = ComplexF64::new(0.0, -0.5).mul_fixed(inner);
            Some(2.0 * scaled.re)
        }
        // Non-parameterized gates have no gradient
        _ => None,
    }
}

/// Deterministic inner product ⟨a|b⟩ = Σ a_i* · b_i using Kahan summation.
fn inner_product(a: &[ComplexF64], b: &[ComplexF64]) -> ComplexF64 {
    let mut re_acc = KahanAccumulatorF64::new();
    let mut im_acc = KahanAccumulatorF64::new();
    for i in 0..a.len() {
        let prod = a[i].conj().mul_fixed(b[i]);
        re_acc.add(prod.re);
        im_acc.add(prod.im);
    }
    ComplexF64::new(re_acc.finalize(), im_acc.finalize())
}

// ---------------------------------------------------------------------------
// Mid-Circuit Measurement
// ---------------------------------------------------------------------------

/// A circuit with mid-circuit measurement and classical feed-forward.
///
/// Extends the basic `Circuit` to support measuring qubits at arbitrary
/// points and conditioning subsequent gates on measurement outcomes.
#[derive(Debug, Clone)]
pub enum CircuitOp {
    /// Apply a quantum gate.
    Gate(Gate),
    /// Measure a qubit, storing the result in a classical register.
    Measure { qubit: usize, creg: usize },
    /// Conditional gate: apply only if classical register has the given value.
    IfThen { creg: usize, value: u8, gate: Gate },
}

/// A circuit supporting mid-circuit measurement and classical control.
#[derive(Debug, Clone)]
pub struct HybridCircuit {
    n_qubits: usize,
    n_cregs: usize,
    ops: Vec<CircuitOp>,
}

impl HybridCircuit {
    /// Create a new hybrid circuit.
    pub fn new(n_qubits: usize, n_cregs: usize) -> Self {
        HybridCircuit {
            n_qubits,
            n_cregs,
            ops: Vec::new(),
        }
    }

    /// Add a quantum gate.
    pub fn gate(&mut self, g: Gate) -> &mut Self {
        self.ops.push(CircuitOp::Gate(g));
        self
    }

    /// Add a mid-circuit measurement.
    pub fn measure(&mut self, qubit: usize, creg: usize) -> &mut Self {
        self.ops.push(CircuitOp::Measure { qubit, creg });
        self
    }

    /// Add a classically-controlled gate.
    pub fn if_then(&mut self, creg: usize, value: u8, gate: Gate) -> &mut Self {
        self.ops.push(CircuitOp::IfThen { creg, value, gate });
        self
    }

    /// Execute the hybrid circuit with a given RNG seed.
    ///
    /// Returns (final statevector, classical register values).
    pub fn execute(&self, rng_state: &mut u64) -> Result<(Statevector, Vec<u8>), String> {
        let mut sv = Statevector::new(self.n_qubits);
        let mut cregs = vec![0u8; self.n_cregs];

        for op in &self.ops {
            match op {
                CircuitOp::Gate(g) => {
                    g.apply(&mut sv)?;
                }
                CircuitOp::Measure { qubit, creg } => {
                    if *creg >= self.n_cregs {
                        return Err(format!("classical register {} out of range", creg));
                    }
                    let outcome = crate::measure::measure_qubit(&mut sv, *qubit, rng_state)?;
                    cregs[*creg] = outcome;
                }
                CircuitOp::IfThen { creg, value, gate } => {
                    if *creg >= self.n_cregs {
                        return Err(format!("classical register {} out of range", creg));
                    }
                    if cregs[*creg] == *value {
                        gate.apply(&mut sv)?;
                    }
                }
            }
        }

        Ok((sv, cregs))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_expectation_value_z_observable() {
        // |0⟩ state: E(Z) = 1.0
        let sv = Statevector::new(1);
        let z_obs = vec![1.0, -1.0]; // Z eigenvalues
        let e = expectation_value(&sv, &z_obs).unwrap();
        assert!((e - 1.0).abs() < 1e-12, "⟨0|Z|0⟩ = 1, got {}", e);
    }

    #[test]
    fn test_expectation_value_after_x() {
        // X|0⟩ = |1⟩: E(Z) = -1.0
        let mut sv = Statevector::new(1);
        Gate::X(0).apply(&mut sv).unwrap();
        let z_obs = vec![1.0, -1.0];
        let e = expectation_value(&sv, &z_obs).unwrap();
        assert!((e - (-1.0)).abs() < 1e-12, "⟨1|Z|1⟩ = -1, got {}", e);
    }

    #[test]
    fn test_expectation_value_superposition() {
        // H|0⟩ = |+⟩: E(Z) = 0.0
        let mut sv = Statevector::new(1);
        Gate::H(0).apply(&mut sv).unwrap();
        let z_obs = vec![1.0, -1.0];
        let e = expectation_value(&sv, &z_obs).unwrap();
        assert!(e.abs() < 1e-12, "⟨+|Z|+⟩ = 0, got {}", e);
    }

    #[test]
    fn test_adjoint_gradient_rz() {
        // Circuit: H Rz(θ) H |0⟩, observable: Z
        // H Rz(θ) H maps |0⟩ to cos(θ/2)|0⟩ - i sin(θ/2)|1⟩
        // E(θ) = cos²(θ/2) - sin²(θ/2) = cos(θ)
        // ∂E/∂θ_Rz = -sin(θ)
        let theta = PI / 4.0;
        let mut circ = Circuit::new(1);
        circ.h(0).rz(0, theta).h(0);

        let z_obs = vec![1.0, -1.0];
        let grads = adjoint_differentiation(&circ, &z_obs).unwrap();

        // Gate 0 = H (gradient 0), Gate 1 = Rz (gradient -sin(θ)), Gate 2 = H (gradient 0)
        let expected = -(theta).sin();
        assert!((grads.gradients[1] - expected).abs() < TOL,
            "Rz gradient: got {}, expected {}", grads.gradients[1], expected);
    }

    #[test]
    fn test_adjoint_gradient_ry() {
        // Circuit: Ry(θ)|0⟩, observable: Z
        // E(θ) = cos(θ)
        // ∂E/∂θ = -sin(θ)
        let theta = PI / 3.0;
        let mut circ = Circuit::new(1);
        circ.ry(0, theta);

        let z_obs = vec![1.0, -1.0];
        let grads = adjoint_differentiation(&circ, &z_obs).unwrap();

        let expected = -(theta).sin();
        assert!((grads.gradients[0] - expected).abs() < TOL,
            "Ry gradient: got {}, expected {}", grads.gradients[0], expected);
    }

    #[test]
    fn test_adjoint_non_parameterized_gates_zero_gradient() {
        let mut circ = Circuit::new(1);
        circ.h(0).x(0);
        let z_obs = vec![1.0, -1.0];
        let grads = adjoint_differentiation(&circ, &z_obs).unwrap();
        // H and X are not parameterized, gradients should be 0
        assert_eq!(grads.gradients[0], 0.0, "H gradient");
        assert_eq!(grads.gradients[1], 0.0, "X gradient");
    }

    #[test]
    fn test_adjoint_gradient_matches_finite_diff() {
        // Verify adjoint gradient matches finite differences
        let theta = 0.7;
        let eps = 1e-5;
        let z_obs = vec![1.0, -1.0];

        // E(θ + eps)
        let mut circ_plus = Circuit::new(1);
        circ_plus.ry(0, theta + eps);
        let sv_plus = circ_plus.execute().unwrap();
        let e_plus = expectation_value(&sv_plus, &z_obs).unwrap();

        // E(θ - eps)
        let mut circ_minus = Circuit::new(1);
        circ_minus.ry(0, theta - eps);
        let sv_minus = circ_minus.execute().unwrap();
        let e_minus = expectation_value(&sv_minus, &z_obs).unwrap();

        let fd_grad = (e_plus - e_minus) / (2.0 * eps);

        // Adjoint gradient
        let mut circ = Circuit::new(1);
        circ.ry(0, theta);
        let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

        assert!((adj_grads.gradients[0] - fd_grad).abs() < 1e-4,
            "Adjoint {} vs FD {}", adj_grads.gradients[0], fd_grad);
    }

    #[test]
    fn test_adjoint_multi_parameter() {
        // Circuit: Ry(θ1) Rz(θ2) |0⟩
        let theta1 = 0.3;
        let theta2 = 0.7;
        let z_obs = vec![1.0, -1.0];

        let mut circ = Circuit::new(1);
        circ.ry(0, theta1).rz(0, theta2);
        let grads = adjoint_differentiation(&circ, &z_obs).unwrap();

        // Verify with finite differences
        let eps = 1e-5;
        for (i, theta_i) in [theta1, theta2].iter().enumerate() {
            let mut circ_p = Circuit::new(1);
            let mut circ_m = Circuit::new(1);
            if i == 0 {
                circ_p.ry(0, theta1 + eps).rz(0, theta2);
                circ_m.ry(0, theta1 - eps).rz(0, theta2);
            } else {
                circ_p.ry(0, theta1).rz(0, theta2 + eps);
                circ_m.ry(0, theta1).rz(0, theta2 - eps);
            }
            let e_p = expectation_value(&circ_p.execute().unwrap(), &z_obs).unwrap();
            let e_m = expectation_value(&circ_m.execute().unwrap(), &z_obs).unwrap();
            let fd = (e_p - e_m) / (2.0 * eps);
            assert!((grads.gradients[i] - fd).abs() < 1e-4,
                "Gate {} gradient: adjoint {} vs FD {}", i, grads.gradients[i], fd);
        }
    }

    // -----------------------------------------------------------------------
    // Mid-Circuit Measurement Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hybrid_circuit_basic() {
        let mut hc = HybridCircuit::new(1, 1);
        hc.gate(Gate::X(0));
        hc.measure(0, 0);

        let mut rng = 42u64;
        let (sv, cregs) = hc.execute(&mut rng).unwrap();
        assert_eq!(cregs[0], 1, "X|0⟩ measured should be 1");
        assert!(sv.is_normalized(1e-12));
    }

    #[test]
    fn test_hybrid_teleportation() {
        // Quantum teleportation with mid-circuit measurement:
        // 1. Prepare state |ψ⟩ on qubit 0 (Ry(π/3)|0⟩)
        // 2. Create Bell pair on qubits 1,2
        // 3. Bell measurement on qubits 0,1 (CNOT + H + measure)
        // 4. Classical corrections on qubit 2 based on measurements
        let mut hc = HybridCircuit::new(3, 2);

        // Prepare state on q0
        hc.gate(Gate::Ry(0, std::f64::consts::FRAC_PI_3));
        // Bell pair: q1-q2
        hc.gate(Gate::H(1));
        hc.gate(Gate::CNOT(1, 2));
        // Bell measurement: q0-q1
        hc.gate(Gate::CNOT(0, 1));
        hc.gate(Gate::H(0));
        hc.measure(0, 0); // measure q0 → creg 0
        hc.measure(1, 1); // measure q1 → creg 1
        // Classical corrections
        hc.if_then(1, 1, Gate::X(2)); // if creg[1] == 1, apply X to q2
        hc.if_then(0, 1, Gate::Z(2)); // if creg[0] == 1, apply Z to q2

        let mut rng = 42u64;
        let (sv, cregs) = hc.execute(&mut rng).unwrap();
        assert!(sv.is_normalized(1e-12), "Teleportation preserves norm");
        assert_eq!(cregs.len(), 2);
    }

    #[test]
    fn test_hybrid_deterministic() {
        let mut hc = HybridCircuit::new(2, 1);
        hc.gate(Gate::H(0));
        hc.measure(0, 0);
        hc.if_then(0, 1, Gate::X(1));

        // Same seed → same outcomes
        for seed in 0..20u64 {
            let mut rng1 = seed;
            let mut rng2 = seed;
            let (_, c1) = hc.execute(&mut rng1).unwrap();
            let (_, c2) = hc.execute(&mut rng2).unwrap();
            assert_eq!(c1, c2, "Determinism failure at seed {}", seed);
        }
    }

    #[test]
    fn test_hybrid_feed_forward_correlation() {
        // If we measure qubit 0 and conditionally X qubit 1,
        // qubit 1 should always end up matching the measurement.
        let mut hc = HybridCircuit::new(2, 1);
        hc.gate(Gate::H(0));      // |+⟩ on q0
        hc.measure(0, 0);         // measure → creg[0]
        hc.if_then(0, 1, Gate::X(1)); // if 1, flip q1

        for seed in 0..100u64 {
            let mut rng = seed;
            let (sv, cregs) = hc.execute(&mut rng).unwrap();
            // After this circuit, qubit 1 should match the measurement
            // If creg[0] == 0: state = |00⟩
            // If creg[0] == 1: state = |11⟩
            if cregs[0] == 0 {
                assert!((sv.amplitudes[0].norm_sq() - 1.0).abs() < 1e-12,
                    "seed {}: expected |00⟩", seed);
            } else {
                assert!((sv.amplitudes[3].norm_sq() - 1.0).abs() < 1e-12,
                    "seed {}: expected |11⟩", seed);
            }
        }
    }
}
