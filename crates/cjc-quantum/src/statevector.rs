//! Statevector — quantum state representation as a complex amplitude vector.
//!
//! An N-qubit system is represented by 2^N complex amplitudes. The state
//! |ψ⟩ = Σ αᵢ|i⟩ where Σ|αᵢ|² = 1.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::complex::ComplexF64;

/// A quantum statevector of N qubits (2^N complex amplitudes).
#[derive(Debug, Clone)]
pub struct Statevector {
    /// Complex amplitudes in lexicographic basis state order.
    pub amplitudes: Vec<ComplexF64>,
    /// Number of qubits.
    pub n_qubits: usize,
}

impl Statevector {
    /// Create a new statevector initialized to |000...0⟩.
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits <= 26, "maximum 26 qubits supported (memory limit)");
        let n_states = 1usize << n_qubits;
        let mut amplitudes = vec![ComplexF64::ZERO; n_states];
        if n_states > 0 {
            amplitudes[0] = ComplexF64::ONE;
        }
        Statevector { amplitudes, n_qubits }
    }

    /// Create a statevector from explicit amplitudes.
    /// Returns error if the number of amplitudes is not a power of 2.
    pub fn from_amplitudes(amps: Vec<ComplexF64>) -> Result<Self, String> {
        let n = amps.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(format!("amplitude count must be a power of 2, got {}", n));
        }
        let n_qubits = n.trailing_zeros() as usize;
        Ok(Statevector { amplitudes: amps, n_qubits })
    }

    /// Number of qubits.
    #[inline]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of basis states (2^n_qubits).
    #[inline]
    pub fn n_states(&self) -> usize {
        self.amplitudes.len()
    }

    /// Get the complex amplitude of a basis state.
    pub fn amplitude(&self, index: usize) -> Result<ComplexF64, String> {
        if index >= self.n_states() {
            return Err(format!(
                "basis state index {} out of range (n_states={})", index, self.n_states()
            ));
        }
        Ok(self.amplitudes[index])
    }

    /// Compute the probability of each basis state: |αᵢ|².
    /// Uses Kahan summation internally for normalization verification.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Check if the statevector is normalized (Σ|αᵢ|² ≈ 1.0).
    pub fn is_normalized(&self, tol: f64) -> bool {
        let mut acc = KahanAccumulatorF64::new();
        for a in &self.amplitudes {
            acc.add(a.norm_sq());
        }
        (acc.finalize() - 1.0).abs() < tol
    }

    /// Renormalize the statevector so that Σ|αᵢ|² = 1.0.
    pub fn normalize(&mut self) {
        let mut acc = KahanAccumulatorF64::new();
        for a in &self.amplitudes {
            acc.add(a.norm_sq());
        }
        let norm = acc.finalize().sqrt();
        if norm > 0.0 {
            for a in &mut self.amplitudes {
                a.re /= norm;
                a.im /= norm;
            }
        }
    }

    /// Validate a qubit index.
    pub fn validate_qubit(&self, qubit: usize) -> Result<(), String> {
        if qubit >= self.n_qubits {
            Err(format!("qubit {} out of range (n_qubits={})", qubit, self.n_qubits))
        } else {
            Ok(())
        }
    }
}

impl std::fmt::Display for Statevector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Statevector({} qubits, {} states)", self.n_qubits, self.n_states())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_initial_state() {
        let sv = Statevector::new(2);
        assert_eq!(sv.n_qubits(), 2);
        assert_eq!(sv.n_states(), 4);
        assert_eq!(sv.amplitudes[0], ComplexF64::ONE);
        assert_eq!(sv.amplitudes[1], ComplexF64::ZERO);
        assert_eq!(sv.amplitudes[2], ComplexF64::ZERO);
        assert_eq!(sv.amplitudes[3], ComplexF64::ZERO);
        assert!(sv.is_normalized(1e-12));
    }

    #[test]
    fn test_from_amplitudes() {
        let amps = vec![
            ComplexF64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ComplexF64::new(0.0, 0.0),
            ComplexF64::new(0.0, 0.0),
            ComplexF64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let sv = Statevector::from_amplitudes(amps).unwrap();
        assert_eq!(sv.n_qubits(), 2);
        assert!(sv.is_normalized(1e-12));
    }

    #[test]
    fn test_from_amplitudes_not_power_of_2() {
        let amps = vec![ComplexF64::ONE, ComplexF64::ZERO, ComplexF64::ZERO];
        assert!(Statevector::from_amplitudes(amps).is_err());
    }

    #[test]
    fn test_probabilities() {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let amps = vec![
            ComplexF64::new(inv_sqrt2, 0.0),
            ComplexF64::new(inv_sqrt2, 0.0),
        ];
        let sv = Statevector::from_amplitudes(amps).unwrap();
        let probs = sv.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!((probs[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_normalize() {
        let amps = vec![
            ComplexF64::new(1.0, 0.0),
            ComplexF64::new(1.0, 0.0),
        ];
        let mut sv = Statevector::from_amplitudes(amps).unwrap();
        assert!(!sv.is_normalized(1e-12));
        sv.normalize();
        assert!(sv.is_normalized(1e-12));
    }

    #[test]
    fn test_validate_qubit() {
        let sv = Statevector::new(3);
        assert!(sv.validate_qubit(0).is_ok());
        assert!(sv.validate_qubit(2).is_ok());
        assert!(sv.validate_qubit(3).is_err());
    }
}
