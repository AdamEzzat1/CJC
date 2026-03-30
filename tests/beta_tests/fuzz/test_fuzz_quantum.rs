// CJC v0.2 Beta — Fuzz Tests: Quantum Module Robustness
//
// These tests feed random/edge-case inputs to the quantum modules to verify
// they never panic. All errors should be handled gracefully.

use cjc_quantum::stabilizer::StabilizerState;
use cjc_quantum::density::{DensityMatrix, depolarizing_channel, dephasing_channel, amplitude_damping_channel};
use cjc_quantum::gates::Gate;
use cjc_quantum::qec::*;
use cjc_quantum::mps::Mps;
use cjc_quantum::vqe::*;
use cjc_quantum::dmrg::*;
use cjc_quantum::qaoa::*;
use cjc_runtime::complex::ComplexF64;

// ── Helper: deterministic pseudo-random via SplitMix64 ──

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn rand_usize(state: &mut u64, max: usize) -> usize {
    (splitmix64(state) as usize) % max.max(1)
}

fn rand_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ── Fuzz: Stabilizer simulator handles random gate sequences ──

#[test]
fn fuzz_stabilizer_random_circuits() {
    for seed in 0..200u64 {
        let mut rng = seed;
        let n = rand_usize(&mut rng, 10) + 1; // 1-10 qubits
        let mut s = StabilizerState::new(n);
        let n_gates = rand_usize(&mut rng, 50);
        for _ in 0..n_gates {
            let gate_type = rand_usize(&mut rng, 6);
            let q = rand_usize(&mut rng, n);
            match gate_type {
                0 => s.h(q),
                1 => s.s(q),
                2 => s.x(q),
                3 => s.y(q),
                4 => s.z(q),
                5 => {
                    let q2 = rand_usize(&mut rng, n);
                    if q != q2 {
                        s.cnot(q, q2);
                    }
                }
                _ => {}
            }
        }
        // Measure all qubits — must not panic
        let mut meas_rng = seed + 1000;
        for q in 0..n {
            let _ = s.measure(q, &mut meas_rng);
        }
    }
}

#[test]
fn fuzz_stabilizer_statevector_small() {
    for seed in 0..100u64 {
        let mut rng = seed;
        let n = rand_usize(&mut rng, 8) + 1; // 1-8 qubits
        let mut s = StabilizerState::new(n);
        let n_gates = rand_usize(&mut rng, 20);
        for _ in 0..n_gates {
            let q = rand_usize(&mut rng, n);
            match rand_usize(&mut rng, 3) {
                0 => s.h(q),
                1 => s.s(q),
                2 => {
                    let q2 = rand_usize(&mut rng, n);
                    if q != q2 { s.cnot(q, q2); }
                }
                _ => {}
            }
        }
        // to_statevector must not panic for small n
        // Note: some random circuits may produce states where the extraction
        // algorithm returns a zero vector due to phase/sign edge cases in
        // Gaussian elimination — this is a known limitation, not a crash.
        let _ = s.to_statevector();
    }
}

// ── Fuzz: Density matrix handles random gate + noise sequences ──

#[test]
fn fuzz_density_matrix_random_circuits() {
    for seed in 0..100u64 {
        let mut rng = seed;
        let n = rand_usize(&mut rng, 4) + 1; // 1-4 qubits (density is expensive)
        let mut rho = DensityMatrix::new(n);
        let n_ops = rand_usize(&mut rng, 20);
        for _ in 0..n_ops {
            let q = rand_usize(&mut rng, n);
            match rand_usize(&mut rng, 8) {
                0 => rho.apply_gate(&Gate::H(q)),
                1 => rho.apply_gate(&Gate::X(q)),
                2 => rho.apply_gate(&Gate::Y(q)),
                3 => rho.apply_gate(&Gate::Z(q)),
                4 => rho.apply_gate(&Gate::S(q)),
                5 => {
                    let ch = depolarizing_channel(rand_f64(&mut rng) * 0.5);
                    rho.apply_single_qubit_channel(q, &ch);
                }
                6 => {
                    let ch = dephasing_channel(rand_f64(&mut rng) * 0.5);
                    rho.apply_single_qubit_channel(q, &ch);
                }
                7 => {
                    let ch = amplitude_damping_channel(rand_f64(&mut rng) * 0.5);
                    rho.apply_single_qubit_channel(q, &ch);
                }
                _ => {}
            }
        }
        // Must not panic
        let tr = rho.trace();
        assert!((tr - 1.0).abs() < 1e-8, "seed={}: trace={}", seed, tr);
        let _ = rho.purity();
        let _ = rho.probabilities();
    }
}

// ── Fuzz: MPS handles random single-qubit gates ──

#[test]
fn fuzz_mps_random_single_qubit() {
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h_mat = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
                 [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];
    let x_mat = [[ComplexF64::ZERO, ComplexF64::ONE],
                 [ComplexF64::ONE, ComplexF64::ZERO]];

    for seed in 0..100u64 {
        let mut rng = seed;
        let n = rand_usize(&mut rng, 8) + 2; // 2-9 qubits
        let mut mps = Mps::new(n);
        let n_ops = rand_usize(&mut rng, 15);
        for _ in 0..n_ops {
            let q = rand_usize(&mut rng, n);
            match rand_usize(&mut rng, 3) {
                0 => mps.apply_single_qubit(q, h_mat),
                1 => mps.apply_single_qubit(q, x_mat),
                2 => {
                    if q + 1 < n {
                        mps.apply_cnot_adjacent(q, q + 1);
                    }
                }
                _ => {}
            }
        }
        // Energy computation must not panic
        let _ = mps_heisenberg_energy(&mps);
    }
}

// ── Fuzz: VQE handles edge-case parameters ──

#[test]
fn fuzz_vqe_edge_parameters() {
    // Very small/large learning rates, 0 iterations, etc.
    let configs: Vec<(usize, usize, f64, usize, u64)> = vec![
        (2, 4, 0.0, 1, 0),       // zero learning rate
        (2, 4, 1e-15, 1, 42),    // tiny learning rate
        (2, 4, 10.0, 1, 42),     // large learning rate
        (2, 4, 0.1, 0, 42),      // zero iterations
        (3, 2, 0.1, 3, 99),      // small bond dim
    ];
    for (n, chi, lr, iters, seed) in &configs {
        // Must not panic
        let _ = vqe_heisenberg_1d(*n, *chi, *lr, *iters, *seed);
    }
}

#[test]
fn fuzz_vqe_full_heisenberg_edge() {
    let configs: Vec<(usize, usize, f64, usize, u64)> = vec![
        (2, 4, 0.0, 1, 0),
        (2, 4, 0.1, 0, 42),
        (3, 2, 0.1, 2, 99),
    ];
    for (n, chi, lr, iters, seed) in &configs {
        let _ = vqe_full_heisenberg_1d(*n, *chi, *lr, *iters, *seed);
    }
}

// ── Fuzz: DMRG handles edge-case parameters ──

#[test]
fn fuzz_dmrg_edge_parameters() {
    // Must not panic on edge cases
    let _ = dmrg_heisenberg_1d(2, 2, 1, 1e-8);  // minimal system
    let _ = dmrg_heisenberg_1d(2, 4, 0, 1e-8);  // zero sweeps
    let _ = dmrg_full_heisenberg_1d(2, 2, 1, 1e-8);
}

// ── Fuzz: QAOA handles various graph sizes ──

#[test]
fn fuzz_qaoa_various_graphs() {
    for n in 3..8 {
        let g = Graph::cycle(n);
        // Must not panic; very few iterations just to test robustness
        let _ = qaoa_maxcut(&g, 1, 8, 0.1, 3, 42);
    }
}

// ── Fuzz: QEC repetition code handles various distances ──

#[test]
fn fuzz_qec_repetition_various_distances() {
    for d in 3..8 {
        let code = build_repetition_code(d);
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = 42u64;
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
        let _ = decode_repetition_code(&syndrome, &code);
    }
}

#[test]
fn fuzz_qec_surface_code_small() {
    for d in [3, 5] {
        let code = build_surface_code(d);
        let mut state = StabilizerState::new(code.total_qubits);
        let mut rng = 42u64;
        let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
        let _ = decode_repetition_code(&syndrome, &code);
    }
}
