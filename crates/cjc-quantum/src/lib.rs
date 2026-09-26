//! CJC Quantum — Deterministic Quantum Circuit Simulator
//!
//! Classical simulation of quantum circuits: dense statevector, MPS,
//! stabilizer (Clifford), and density-matrix backends, plus VQE, QAOA, QML,
//! QEC, fermion/Trotter, and zero-noise extrapolation on top of them.
//!
//! # Determinism
//!
//! Same inputs and seed give bit-identical results, across runs and across
//! operating systems (Linux, Windows, macOS):
//!
//! - Amplitude accumulations use Kahan summation
//! - Complex multiplication uses fixed-sequence (no FMA)
//! - Measurement sampling via SplitMix64 with explicit seed threading
//! - Gate application processes basis states in ascending index order
//! - All collections use deterministic ordering (Vec, not HashMap)
//! - `sin`, `cos`, `exp`, `ln`, and powers come from `cjc_repro::dmath`, never
//!   the platform libm, which differs between operating systems (ADR-0046)
//!
//! Evidence: `tests/cross_platform_golden.rs` hashes the bits of every
//! transcendental-dependent output and runs in the Linux/Windows/macOS CI
//! matrix. Scope: an angle computed in `.cjcl` with the `cjc-runtime` math
//! builtins (`sin`, `exp`, …) still uses the platform libm *before* it reaches
//! this crate.
//!
//! # Limitations
//!
//! - Dense statevector: at most 26 qubits from `.cjcl` (2^N memory);
//!   density matrix: at most 14 (4^N)
//! - Noise: depolarizing, dephasing, and amplitude damping on the density
//!   backend only; no noisy trajectories on the statevector backend
//! - No hardware backend (classical simulation only, by design)

pub mod adjoint;
pub mod circuit;
pub mod density;
pub mod dispatch;
pub mod dmrg;
pub mod fermion;
pub mod gates;
pub mod kernels;
pub mod measure;
pub mod mitigation;
pub mod mps;
pub mod pure;
pub mod qaoa;
pub mod qasm;
pub mod qec;
pub mod qml;
pub mod simd_kernel;
pub mod stabilizer;
pub mod statevector;
pub mod trotter;
pub mod vqe;
pub mod wirtinger;

pub use circuit::Circuit;
pub use dispatch::dispatch_quantum;
pub use gates::Gate;
pub use measure::{measure_all, measure_qubit};
pub use statevector::Statevector;

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
