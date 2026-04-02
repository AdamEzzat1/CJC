//! CJC Quantum — Deterministic Quantum Circuit Simulator
//!
//! Classical simulation of quantum circuits using statevector representation.
//! All operations are deterministic: same seed = bit-identical measurement outcomes.
//!
//! # Determinism Guarantees
//!
//! - Amplitude accumulations use Kahan summation
//! - Complex multiplication uses fixed-sequence (no FMA)
//! - Measurement sampling via SplitMix64 with explicit seed threading
//! - Gate application processes basis states in ascending index order
//! - All collections use deterministic ordering (Vec, not HashMap)
//!
//! # Limitations
//!
//! - Classical simulation: ~25-30 qubits max (2^N memory scaling)
//! - No noise model (pure unitary evolution only)
//! - No hardware backend (simulation only)

pub mod gates;
pub mod statevector;
pub mod measure;
pub mod circuit;
pub mod dispatch;
pub mod wirtinger;
pub mod adjoint;
pub mod simd_kernel;
pub mod mps;
pub mod vqe;
pub mod qaoa;
pub mod stabilizer;
pub mod density;
pub mod qec;
pub mod dmrg;
pub mod qml;
pub mod fermion;
pub mod trotter;
pub mod mitigation;
pub mod pure;

pub use gates::Gate;
pub use statevector::Statevector;
pub use measure::{measure_qubit, measure_all};
pub use circuit::Circuit;
pub use dispatch::dispatch_quantum;

// Re-export ComplexF64 for convenience
pub use cjc_runtime::complex::ComplexF64 as Complex;

// ---------------------------------------------------------------------------
// SplitMix64 (local copy to avoid depending on cjc-repro internals)
// ---------------------------------------------------------------------------

/// Deterministic PRNG for measurement sampling.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Convert a u64 to a uniform f64 in [0, 1).
pub fn rand_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitmix64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..100 {
            assert_eq!(splitmix64(&mut s1), splitmix64(&mut s2));
        }
    }
}
