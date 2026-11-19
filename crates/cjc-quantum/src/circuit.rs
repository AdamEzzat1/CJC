//! Circuit — quantum circuit builder and executor.
//!
//! A `Circuit` is an ordered sequence of gates applied to a statevector.
//! The builder pattern allows constructing circuits declaratively, then
//! executing them with a specific seed for deterministic results.

use std::cell::RefCell;
use std::rc::Rc;

use crate::gates::Gate;
use crate::measure;
use crate::statevector::Statevector;

/// Largest circuit whose executed statevector is cached (2^24 amplitudes =
/// 256 MiB). Above this, a cache would silently hold gigabytes, so
/// [`Circuit::execute_shared`] recomputes instead.
pub const EXEC_CACHE_MAX_QUBITS: usize = 24;

/// A quantum circuit: an ordered list of gates on N qubits.
#[derive(Debug)]
pub struct Circuit {
    /// Number of qubits in the circuit.
    n_qubits: usize,
    /// Gates in application order.
    gates: Vec<Gate>,
    /// Result of executing `gates`, filled by `execute_shared` and cleared by
    /// `add`. Execution is deterministic, so a cached result is bit-identical
    /// to a fresh one.
    exec_cache: RefCell<Option<Rc<Statevector>>>,
}

impl Clone for Circuit {
    /// Copies the gates; the copy starts without a cached execution.
    fn clone(&self) -> Self {
        Circuit {
            n_qubits: self.n_qubits,
            gates: self.gates.clone(),
            exec_cache: RefCell::new(None),
        }
    }
}

impl Circuit {
    /// Create a new empty circuit for `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Circuit {
            n_qubits,
            gates: Vec::new(),
            exec_cache: RefCell::new(None),
        }
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of gates in the circuit.
    pub fn n_gates(&self) -> usize {
        self.gates.len()
    }

    /// Add a gate to the circuit. Returns `&mut Self` for chaining.
    pub fn add(&mut self, gate: Gate) -> &mut Self {
        self.gates.push(gate);
        *self.exec_cache.get_mut() = None;
        self
    }

    /// Convenience: add Hadamard gate.
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.add(Gate::H(qubit))
    }
    /// Convenience: add Pauli-X gate.
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        self.add(Gate::X(qubit))
    }
    /// Convenience: add Pauli-Y gate.
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        self.add(Gate::Y(qubit))
    }
    /// Convenience: add Pauli-Z gate.
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        self.add(Gate::Z(qubit))
    }
    /// Convenience: add S gate.
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        self.add(Gate::S(qubit))
    }
    /// Convenience: add T gate.
    pub fn t(&mut self, qubit: usize) -> &mut Self {
        self.add(Gate::T(qubit))
    }
    /// Convenience: add Rx gate.
    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.add(Gate::Rx(qubit, theta))
    }
    /// Convenience: add Ry gate.
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.add(Gate::Ry(qubit, theta))
    }
    /// Convenience: add Rz gate.
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.add(Gate::Rz(qubit, theta))
    }
    /// Convenience: add CNOT gate.
    pub fn cnot(&mut self, ctrl: usize, tgt: usize) -> &mut Self {
        self.add(Gate::CNOT(ctrl, tgt))
    }
    /// Convenience: add CZ gate.
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.add(Gate::CZ(a, b))
    }
    /// Convenience: add SWAP gate.
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.add(Gate::SWAP(a, b))
    }
    /// Convenience: add Toffoli gate.
    pub fn toffoli(&mut self, c1: usize, c2: usize, tgt: usize) -> &mut Self {
        self.add(Gate::Toffoli(c1, c2, tgt))
    }

    /// Execute the circuit: initialize |000...0⟩, apply all gates in order.
    /// Returns the final statevector.
    pub fn execute(&self) -> Result<Statevector, String> {
        let mut sv = Statevector::new(self.n_qubits);
        for gate in &self.gates {
            gate.apply(&mut sv)?;
        }
        Ok(sv)
    }

    /// Execute once and share the result: later calls on the same circuit
    /// return the cached statevector (circuits up to
    /// [`EXEC_CACHE_MAX_QUBITS`]). Used by the `.cjcl` observables, so
    /// `q_probs(c)` followed by `q_sample(c, …)` simulates `c` once.
    pub fn execute_shared(&self) -> Result<Rc<Statevector>, String> {
        if self.n_qubits > EXEC_CACHE_MAX_QUBITS {
            return Ok(Rc::new(self.execute()?));
        }
        if let Some(sv) = self.exec_cache.borrow().as_ref() {
            return Ok(Rc::clone(sv));
        }
        let sv = Rc::new(self.execute()?);
        *self.exec_cache.borrow_mut() = Some(Rc::clone(&sv));
        Ok(sv)
    }

    /// Execute the circuit and measure all qubits.
    /// Returns (measurement outcomes, final collapsed statevector).
    pub fn execute_and_measure(
        &self,
        rng_state: &mut u64,
    ) -> Result<(Vec<u8>, Statevector), String> {
        let mut sv = self.execute()?;
        let outcomes = measure::measure_all(&mut sv, rng_state)?;
        Ok((outcomes, sv))
    }

    /// Execute the circuit and sample the output distribution `n_shots` times.
    ///
    /// Returns a vector of `n_shots` basis state indices. The statevector is
    /// not collapsed (sampling without measurement).
    pub fn sample(&self, n_shots: usize, rng_state: &mut u64) -> Result<Vec<usize>, String> {
        let sv = self.execute_shared()?;
        Ok(measure::sample_basis_states(&sv, n_shots, rng_state))
    }

    /// Get the list of gates.
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }
}

impl std::fmt::Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Circuit({} qubits, {} gates):",
            self.n_qubits,
            self.gates.len()
        )?;
        for (i, gate) in self.gates.iter().enumerate() {
            writeln!(f, "  [{}] {}", i, gate)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn execute_shared_caches_until_the_circuit_changes() {
        let bits = |sv: &Statevector| -> Vec<(u64, u64)> {
            sv.amplitudes.iter().map(|a| (a.re.to_bits(), a.im.to_bits())).collect()
        };
        let mut c = Circuit::new(3);
        c.h(0).ry(1, 0.7).cnot(0, 2);
        let a = c.execute_shared().unwrap();
        let b = c.execute_shared().unwrap();
        assert!(std::rc::Rc::ptr_eq(&a, &b), "second call must reuse the cache");
        assert_eq!(bits(&a), bits(&c.execute().unwrap()), "cache is bit-identical");
        let copy = c.clone();
        assert!(!std::rc::Rc::ptr_eq(&a, &copy.execute_shared().unwrap()), "clone starts uncached");
        c.x(1);
        let d = c.execute_shared().unwrap();
        assert!(!std::rc::Rc::ptr_eq(&a, &d), "add must invalidate");
        assert_eq!(bits(&d), bits(&c.execute().unwrap()));
    }

    use super::*;
    use cjc_runtime::complex::ComplexF64;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_empty_circuit() {
        let circ = Circuit::new(2);
        let sv = circ.execute().unwrap();
        assert_eq!(sv.amplitudes[0], ComplexF64::ONE);
        assert_eq!(sv.n_qubits(), 2);
    }

    #[test]
    fn test_bell_circuit() {
        let mut circ = Circuit::new(2);
        circ.h(0).cnot(0, 1);
        let sv = circ.execute().unwrap();

        let inv = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv.amplitudes[0].re - inv).abs() < TOL);
        assert!((sv.amplitudes[3].re - inv).abs() < TOL);
        assert!((sv.amplitudes[1].norm_sq()).abs() < TOL);
        assert!((sv.amplitudes[2].norm_sq()).abs() < TOL);
    }

    #[test]
    fn test_ghz_3_qubit() {
        // GHZ state: (|000⟩ + |111⟩)/√2
        let mut circ = Circuit::new(3);
        circ.h(0).cnot(0, 1).cnot(0, 2);
        let sv = circ.execute().unwrap();

        let inv = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv.amplitudes[0].re - inv).abs() < TOL, "GHZ |000⟩");
        assert!((sv.amplitudes[7].re - inv).abs() < TOL, "GHZ |111⟩");
        // All other amplitudes should be 0
        for i in 1..7 {
            assert!(
                (sv.amplitudes[i].norm_sq()).abs() < TOL,
                "GHZ non-zero at index {}",
                i
            );
        }
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_circuit_execute_and_measure() {
        let mut circ = Circuit::new(2);
        circ.h(0).cnot(0, 1);

        let mut rng = 42u64;
        let (outcomes, sv) = circ.execute_and_measure(&mut rng).unwrap();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0], outcomes[1], "Bell state qubits correlated");
        assert!(sv.is_normalized(TOL));
    }

    #[test]
    fn test_circuit_sample_deterministic() {
        let mut circ = Circuit::new(2);
        circ.h(0).cnot(0, 1);

        let mut rng1 = 42u64;
        let mut rng2 = 42u64;
        let s1 = circ.sample(100, &mut rng1).unwrap();
        let s2 = circ.sample(100, &mut rng2).unwrap();
        assert_eq!(s1, s2, "Same seed must produce same samples");
    }

    #[test]
    fn test_circuit_sample_distribution() {
        let mut circ = Circuit::new(1);
        circ.h(0); // equal superposition

        let mut rng = 42u64;
        let samples = circ.sample(1000, &mut rng).unwrap();
        let count0 = samples.iter().filter(|&&s| s == 0).count();
        let ratio = count0 as f64 / 1000.0;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "H|0⟩ sample ratio: {} (expected ~0.5)",
            ratio
        );
    }

    #[test]
    fn test_circuit_chaining() {
        let mut circ = Circuit::new(3);
        circ.h(0).h(1).h(2).cnot(0, 1).cnot(1, 2).t(0).s(1);
        assert_eq!(circ.n_gates(), 7);
    }

    #[test]
    fn test_circuit_display() {
        let mut circ = Circuit::new(2);
        circ.h(0).cnot(0, 1);
        let s = format!("{}", circ);
        assert!(s.contains("H(0)"));
        assert!(s.contains("CNOT(0, 1)"));
    }

    #[test]
    fn test_circuit_gate_error() {
        let mut circ = Circuit::new(2);
        circ.h(5); // qubit 5 out of range for 2-qubit circuit
        assert!(circ.execute().is_err());
    }

    #[test]
    fn test_quantum_teleportation() {
        // Quantum teleportation protocol:
        // 1. Create Bell pair between qubits 1,2
        // 2. Prepare qubit 0 in state to teleport (Ry(π/3)|0⟩)
        // 3. Bell measurement on qubits 0,1
        // 4. Conditional corrections on qubit 2
        //
        // We can't do classical feed-forward in a circuit, but we can verify
        // the entanglement structure is correct.
        let mut circ = Circuit::new(3);
        // Prepare state on qubit 0
        circ.ry(0, std::f64::consts::FRAC_PI_3);
        // Bell pair on qubits 1,2
        circ.h(1).cnot(1, 2);
        // Bell measurement circuit on qubits 0,1
        circ.cnot(0, 1).h(0);

        let sv = circ.execute().unwrap();
        assert!(
            sv.is_normalized(TOL),
            "Teleportation circuit preserves norm"
        );
    }
}
