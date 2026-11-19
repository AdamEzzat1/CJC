//! Quantum dispatch — builtin function and method routing for quantum state.
//!
//! This module is the entry point for both cjc-eval and cjc-mir-exec to
//! handle quantum operations. It follows the same pattern as
//! `cjc_vizor::dispatch` and `cjc_data::tidy_dispatch`.

use cjc_repro::dmath;
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cjc_runtime::complex::ComplexF64;
use cjc_runtime::value::Value;

use crate::circuit::Circuit;
use crate::gates::Gate;
use crate::statevector::Statevector;

// Extension imports
use crate::density::DensityMatrix;
use crate::mps::Mps;
use crate::stabilizer::StabilizerState;

// Pure backend imports
use crate::pure::{
    wrap_pure, PureCircuit, PureDensity, PureGate, PureMps, PureStabilizer,
};

/// Dispatch a quantum builtin function call by name.
///
/// Returns `Ok(Some(value))` if handled, `Ok(None)` if not a quantum builtin.
///
/// # Dual-mode backend
///
/// Pass `"pure"` as the last argument to constructor functions to use the
/// pure CJC backend (inspectable state, modifiable algorithms):
///
/// ```cjc
/// let m = mps_new(50, 16, "pure");    // Pure CJC backend
/// let m = mps_new(50, 16);            // Rust backend (default, faster)
/// ```
///
/// Subsequent operations auto-detect the backend from the state value.
pub fn dispatch_quantum(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    // Validate at the language boundary, before any backend indexes `args` or
    // reaches a library `assert!`. Malformed input must be a runtime error,
    // never a process-aborting panic.
    match min_arity(name) {
        None => return Ok(None), // not a quantum builtin
        Some(min) if args.len() < min => {
            return Err(format!(
                "{} requires at least {} argument(s), got {}",
                name,
                min,
                args.len()
            ))
        }
        Some(_) => {}
    }
    prevalidate(name, args)?;

    // Try pure backend dispatch first
    if let Some(result) = dispatch_pure(name, args)? {
        return Ok(Some(result));
    }

    match name {
        // --- Constructor: create quantum circuit ---
        "qubits" => {
            let n = match args.get(0) {
                Some(Value::Int(n)) => {
                    if *n < 1 || *n > 26 {
                        return Err(format!("qubits() requires 1-26 qubits, got {}", n));
                    }
                    *n as usize
                }
                _ => return Err("qubits() requires an integer argument".into()),
            };
            let circuit = Circuit::new(n);
            Ok(Some(wrap_circuit(circuit)))
        }

        // --- Single-qubit gates ---
        "q_h" => apply_gate_1q(args, |q| Gate::H(q)),
        "q_x" => apply_gate_1q(args, |q| Gate::X(q)),
        "q_y" => apply_gate_1q(args, |q| Gate::Y(q)),
        "q_z" => apply_gate_1q(args, |q| Gate::Z(q)),
        "q_s" => apply_gate_1q(args, |q| Gate::S(q)),
        "q_t" => apply_gate_1q(args, |q| Gate::T(q)),

        // --- Parameterized single-qubit gates ---
        "q_rx" => apply_gate_1q_param(args, |q, t| Gate::Rx(q, t)),
        "q_ry" => apply_gate_1q_param(args, |q, t| Gate::Ry(q, t)),
        "q_rz" => apply_gate_1q_param(args, |q, t| Gate::Rz(q, t)),

        // --- Two-qubit gates ---
        "q_cx" | "q_cnot" => apply_gate_2q(args, |a, b| Gate::CNOT(a, b)),
        "q_cz" => apply_gate_2q(args, |a, b| Gate::CZ(a, b)),
        "q_swap" => apply_gate_2q(args, |a, b| Gate::SWAP(a, b)),

        // --- Three-qubit gates ---
        "q_toffoli" | "q_ccx" => {
            if args.len() != 4 {
                return Err(format!(
                    "q_toffoli(circuit, ctrl1, ctrl2, target) requires 4 args, got {}",
                    args.len()
                ));
            }
            let c1 = extract_qubit_index(&args[1], "ctrl1")?;
            let c2 = extract_qubit_index(&args[2], "ctrl2")?;
            let tgt = extract_qubit_index(&args[3], "target")?;
            Ok(Some(circuit_with_gate(&args[0], Gate::Toffoli(c1, c2, tgt))?))
        }

        // --- Execute circuit → get statevector ---
        "q_run" => {
            if args.is_empty() {
                return Err("q_run(circuit) requires a circuit argument".into());
            }
            // An independent copy of the (cached) execution.
            let sv = with_circuit(&args[0], |circ| circ.execute_shared().map(|sv| (*sv).clone()))?;
            Ok(Some(wrap_statevector(sv)))
        }

        // --- Measure all qubits ---
        "q_measure" => {
            if args.len() != 2 {
                return Err("q_measure(circuit, seed) requires 2 args".into());
            }
            let seed = match &args[1] {
                Value::Int(s) => *s as u64,
                _ => return Err("q_measure seed must be an integer".into()),
            };
            let mut rng_state = seed;
            // Measure a private copy: the circuit's cached execution and any
            // statevector argument stay unchanged.
            let mut sv = (*state_vector(&args[0])?).clone();
            let outcomes = crate::measure::measure_all(&mut sv, &mut rng_state)?;
            let result: Vec<Value> = outcomes.iter().map(|&b| Value::Int(b as i64)).collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Get probability distribution ---
        "q_probs" => {
            if args.is_empty() {
                return Err("q_probs(state) requires a circuit or statevector argument".into());
            }
            let sv = state_vector(&args[0])?;
            let probs = sv.probabilities();
            let result: Vec<Value> = probs.iter().map(|&p| Value::Float(p)).collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Sample from circuit distribution ---
        "q_sample" => {
            if args.len() != 3 {
                return Err("q_sample(circuit, n_shots, seed) requires 3 args".into());
            }
            let n_shots = match &args[1] {
                Value::Int(n) => {
                    if *n < 1 {
                        return Err("q_sample n_shots must be positive".into());
                    }
                    *n as usize
                }
                _ => return Err("q_sample n_shots must be an integer".into()),
            };
            let seed = match &args[2] {
                Value::Int(s) => *s as u64,
                _ => return Err("q_sample seed must be an integer".into()),
            };
            let mut rng_state = seed;
            // Same draws as Circuit::sample, so q_sample(c) == q_sample(q_run(c)).
            let sv = state_vector(&args[0])?;
            let samples = crate::measure::sample_basis_states(&sv, n_shots, &mut rng_state);
            let result: Vec<Value> = samples.iter().map(|&s| Value::Int(s as i64)).collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Get amplitudes as array of Complex ---
        "q_amplitudes" => {
            if args.is_empty() {
                return Err("q_amplitudes(state) requires a circuit or statevector argument".into());
            }
            let sv = state_vector(&args[0])?;
            let result: Vec<Value> = sv.amplitudes.iter().map(|&a| Value::Complex(a)).collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Get number of qubits ---
        "q_n_qubits" => {
            if args.is_empty() {
                return Err("q_n_qubits(circuit) requires a circuit argument".into());
            }
            match state_n_qubits(&args[0]) {
                Some(n) => Ok(Some(Value::Int(n as i64))),
                None => Err(format!(
                    "q_n_qubits: expected a quantum state, got {}",
                    args[0].type_name()
                )),
            }
        }

        // --- Get number of gates ---
        "q_n_gates" => {
            if args.is_empty() {
                return Err("q_n_gates(circuit) requires a circuit argument".into());
            }
            with_circuit(&args[0], |circ| Ok(Value::Int(circ.n_gates() as i64))).map(Some)
        }

        // =======================================================================
        // MPS (Matrix Product States) — 50+ qubit simulation
        // =======================================================================
        "mps_new" => {
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_MPS_QUBITS)?;
            let mps = if args.len() > 1 {
                let bond = extract_at_least(args, 1, "max_bond", 1)?;
                Mps::with_max_bond(n, bond)
            } else {
                Mps::new(n)
            };
            Ok(Some(wrap_any(mps)))
        }

        "mps_h" | "mps_x" => {
            let q = extract_usize(args, 1, "qubit")?;
            let isq2 = 1.0 / 2.0f64.sqrt();
            let mat = if name == "mps_h" {
                [
                    [ComplexF64::real(isq2), ComplexF64::real(isq2)],
                    [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
                ]
            } else {
                [
                    [ComplexF64::ZERO, ComplexF64::ONE],
                    [ComplexF64::ONE, ComplexF64::ZERO],
                ]
            };
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_single_qubit(q, mat);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_ry" => {
            let q = extract_usize(args, 1, "qubit")?;
            let theta = extract_f64(args, 2, "theta")?;
            let c = ComplexF64::real(dmath::cos(theta / 2.0));
            let s = ComplexF64::real(dmath::sin(theta / 2.0));
            let mat = [[c, ComplexF64::real(-s.re)], [s, c]];
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_single_qubit(q, mat);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_cnot" => {
            let q1 = extract_usize(args, 1, "control")?;
            let q2 = extract_usize(args, 2, "target")?;
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_cnot_adjacent(q1, q2);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_z_expectation" => {
            let q = extract_usize(args, 1, "qubit")?;
            let z = read_f64::<Mps>(&args[0], "MPS", |mps| {
                crate::qml::mps_single_z_expectation(mps, q)
            })?;
            Ok(Some(Value::Float(z)))
        }

        "mps_energy" => {
            let ham = match args.get(1) {
                None => crate::vqe::Hamiltonian::Ising,
                Some(Value::String(s)) if s.as_ref() == "ising" => crate::vqe::Hamiltonian::Ising,
                Some(Value::String(s)) if s.as_ref() == "heisenberg" => {
                    crate::vqe::Hamiltonian::Heisenberg
                }
                Some(other) => {
                    return Err(format!(
                        "mps_energy: hamiltonian must be \"ising\" or \"heisenberg\", got {}",
                        other
                    ))
                }
            };
            let e = read_f64::<Mps>(&args[0], "MPS", |mps| crate::vqe::mps_energy(mps, ham))?;
            Ok(Some(Value::Float(e)))
        }

        "mps_memory" => {
            let mem = read_i64::<Mps>(&args[0], "MPS", |mps| mps.memory_bytes() as i64)?;
            Ok(Some(Value::Int(mem)))
        }

        // =======================================================================
        // VQE — Variational Quantum Eigensolver
        // =======================================================================
        "vqe_heisenberg" | "vqe_full_heisenberg" => {
            let n = extract_in_range(args, 0, "n_qubits", 2, MAX_MPS_QUBITS)?;
            let chi = extract_at_least(args, 1, "max_bond", 1)?;
            let lr = extract_f64(args, 2, "learning_rate")?;
            let iters = extract_usize(args, 3, "iterations")?;
            let seed = extract_seed(args, 4, "seed")?;
            let result = if name == "vqe_heisenberg" {
                crate::vqe::vqe_heisenberg_1d(n, chi, lr, iters, seed)
            } else {
                crate::vqe::vqe_full_heisenberg_1d(n, chi, lr, iters, seed)
            };
            Ok(Some(Value::Float(result.energy)))
        }

        // =======================================================================
        // QAOA — Quantum Approximate Optimization
        // =======================================================================
        "qaoa_graph_cycle" => {
            let n = extract_in_range(args, 0, "n_vertices", 3, MAX_MPS_QUBITS)?;
            let g = crate::qaoa::Graph::cycle(n);
            Ok(Some(wrap_any(g)))
        }

        "qaoa_maxcut" => {
            let max_bond = extract_at_least(args, 1, "max_bond", 1)?;
            let layers = extract_usize(args, 2, "p_layers")?;
            let lr = extract_f64(args, 3, "learning_rate")?;
            let iters = extract_usize(args, 4, "iterations")?;
            let seed = extract_seed(args, 5, "seed")?;
            let result = match &args[0] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let g = borrow
                        .downcast_ref::<crate::qaoa::Graph>()
                        .ok_or_else(|| "expected Graph".to_string())?;
                    crate::qaoa::qaoa_maxcut(g, layers, max_bond, lr, iters, seed)
                }
                _ => return Err("qaoa_maxcut: first arg must be a Graph".into()),
            };
            let arr = vec![
                Value::Float(result.energy),
                Value::Int(result.cut_value as i64),
            ];
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        // =======================================================================
        // Stabilizer — Clifford/CHP simulator (1000+ qubits)
        // =======================================================================
        "stabilizer_new" => {
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_STABILIZER_QUBITS)?;
            Ok(Some(wrap_any(StabilizerState::new(n))))
        }

        "stabilizer_h" | "stabilizer_s" | "stabilizer_x" | "stabilizer_y" | "stabilizer_z" => {
            let q = extract_usize(args, 1, "qubit")?;
            with_any_mut::<StabilizerState>(&args[0], "StabilizerState", |s| {
                match name {
                    "stabilizer_h" => s.h(q),
                    "stabilizer_s" => s.s(q),
                    "stabilizer_x" => s.x(q),
                    "stabilizer_y" => s.y(q),
                    "stabilizer_z" => s.z(q),
                    _ => unreachable!(),
                }
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_cnot" => {
            let ctrl = extract_usize(args, 1, "control")?;
            let tgt = extract_usize(args, 2, "target")?;
            with_any_mut::<StabilizerState>(&args[0], "StabilizerState", |s| {
                s.cnot(ctrl, tgt);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_measure" => {
            let q = extract_usize(args, 1, "qubit")?;
            let seed = extract_seed(args, 2, "seed")?;
            let mut rng = seed;
            let outcome = match &args[0] {
                Value::QuantumState(rc) => {
                    let mut borrow = rc.borrow_mut();
                    let s = borrow
                        .downcast_mut::<StabilizerState>()
                        .ok_or_else(|| "expected StabilizerState".to_string())?;
                    s.measure(q, &mut rng) as i64
                }
                _ => return Err("stabilizer_measure: expected StabilizerState".into()),
            };
            Ok(Some(Value::Int(outcome)))
        }

        "stabilizer_n_qubits" => {
            let n = read_i64::<StabilizerState>(&args[0], "StabilizerState", |s| {
                s.num_qubits() as i64
            })?;
            Ok(Some(Value::Int(n)))
        }

        // =======================================================================
        // Density Matrix — Mixed states + noise
        // =======================================================================
        "density_new" => {
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_DENSITY_QUBITS)?;
            Ok(Some(wrap_any(DensityMatrix::new(n))))
        }

        "density_gate" => {
            let gate_name = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err("density_gate: gate name must be a string".into()),
            };
            let q = extract_usize(args, 2, "qubit")?;
            let gate = match gate_name.as_str() {
                "H" => Gate::H(q),
                "X" => Gate::X(q),
                "Y" => Gate::Y(q),
                "Z" => Gate::Z(q),
                "S" => Gate::S(q),
                "T" => Gate::T(q),
                _ => return Err(format!("density_gate: unknown gate '{}'", gate_name)),
            };
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_gate(&gate);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_cnot" => {
            let ctrl = extract_usize(args, 1, "control")?;
            let tgt = extract_usize(args, 2, "target")?;
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_gate(&Gate::CNOT(ctrl, tgt));
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_depolarize" => {
            let q = extract_usize(args, 1, "qubit")?;
            let p = extract_f64(args, 2, "probability")?;
            let ch = crate::density::depolarizing_channel(p);
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_single_qubit_channel(q, &ch);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_dephase" => {
            let q = extract_usize(args, 1, "qubit")?;
            let p = extract_f64(args, 2, "probability")?;
            let ch = crate::density::dephasing_channel(p);
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_single_qubit_channel(q, &ch);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_amplitude_damp" => {
            let q = extract_usize(args, 1, "qubit")?;
            let gamma = extract_f64(args, 2, "gamma")?;
            let ch = crate::density::amplitude_damping_channel(gamma);
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_single_qubit_channel(q, &ch);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_trace" => {
            let tr = read_f64::<DensityMatrix>(&args[0], "DensityMatrix", |dm| dm.trace())?;
            Ok(Some(Value::Float(tr)))
        }

        "density_purity" => {
            let p = read_f64::<DensityMatrix>(&args[0], "DensityMatrix", |dm| dm.purity())?;
            Ok(Some(Value::Float(p)))
        }

        "density_entropy" => {
            let e = read_f64::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.von_neumann_entropy()
            })?;
            Ok(Some(Value::Float(e)))
        }

        "density_probs" => {
            let probs =
                read_vec_f64::<DensityMatrix>(&args[0], "DensityMatrix", |dm| dm.probabilities())?;
            let arr: Vec<Value> = probs.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        // =======================================================================
        // DMRG — Density Matrix Renormalization Group
        // =======================================================================
        "dmrg_ising" | "dmrg_heisenberg" => {
            let n = extract_in_range(args, 0, "n_qubits", 2, MAX_MPS_QUBITS)?;
            let chi = extract_at_least(args, 1, "max_bond", 1)?;
            let sweeps = extract_usize(args, 2, "sweeps")?;
            let tol = extract_f64(args, 3, "tolerance")?;
            // Note: the Rust fn `dmrg_heisenberg_1d` runs the Ising (ZZ) model.
            let result = if name == "dmrg_ising" {
                crate::dmrg::dmrg_heisenberg_1d(n, chi, sweeps, tol)
            } else {
                crate::dmrg::dmrg_full_heisenberg_1d(n, chi, sweeps, tol)
            };
            Ok(Some(Value::Float(result.energy)))
        }

        // =======================================================================
        // QEC — Quantum Error Correction
        // =======================================================================
        "qec_repetition_code" => {
            let d = extract_in_range(args, 0, "distance", 2, MAX_QEC_DISTANCE)?;
            let code = crate::qec::build_repetition_code(d);
            Ok(Some(wrap_any(code)))
        }

        "qec_surface_code" => {
            let d = extract_in_range(args, 0, "distance", 2, MAX_QEC_DISTANCE)?;
            let code = crate::qec::build_surface_code(d);
            Ok(Some(wrap_any(code)))
        }

        "qec_syndrome" => {
            let seed = extract_seed(args, 2, "seed")?;
            // The state must have room for every data and ancilla qubit the code uses.
            if let (Some(n), Value::QuantumState(code_rc)) = (state_n_qubits(&args[0]), &args[1]) {
                if let Some(code) = code_rc.borrow().downcast_ref::<crate::qec::SurfaceCode>() {
                    if n < code.total_qubits {
                        return Err(format!(
                            "qec_syndrome: code needs {} qubits, state has {}",
                            code.total_qubits, n
                        ));
                    }
                }
            }
            let mut rng = seed;
            // We need both a mutable StabilizerState and an immutable SurfaceCode.
            // Extract both from QuantumState wrappers, being careful with borrows.
            match (&args[0], &args[1]) {
                (Value::QuantumState(state_rc), Value::QuantumState(code_rc)) => {
                    let code_borrow = code_rc.borrow();
                    let code = code_borrow
                        .downcast_ref::<crate::qec::SurfaceCode>()
                        .ok_or_else(|| "expected SurfaceCode".to_string())?;
                    let mut state_borrow = state_rc.borrow_mut();
                    let state = state_borrow
                        .downcast_mut::<StabilizerState>()
                        .ok_or_else(|| "expected StabilizerState".to_string())?;
                    let syndrome = crate::qec::syndrome_extraction(state, code, &mut rng);
                    let arr: Vec<Value> =
                        syndrome.into_iter().map(|b| Value::Int(b as i64)).collect();
                    Ok(Some(Value::Array(Rc::new(arr))))
                }
                _ => Err("qec_syndrome(state, code, seed): expected QuantumState args".into()),
            }
        }

        "qec_decode" => {
            let syndrome = match &args[0] {
                Value::Array(arr) => arr
                    .iter()
                    .enumerate()
                    .map(|(i, v)| match v {
                        Value::Int(b @ (0 | 1)) => Ok(*b as u8),
                        other => Err(format!(
                            "qec_decode: syndrome[{}] must be 0 or 1, got {}",
                            i, other
                        )),
                    })
                    .collect::<Result<Vec<u8>, String>>()?,
                _ => return Err("qec_decode: first arg must be syndrome array".into()),
            };
            let corrections = match &args[1] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let code = borrow
                        .downcast_ref::<crate::qec::SurfaceCode>()
                        .ok_or_else(|| "expected SurfaceCode".to_string())?;
                    // Only the repetition-code decoder exists (qec.rs). Applying it
                    // to a surface code returns plausible-looking but meaningless
                    // corrections, so refuse instead.
                    if !code.x_stabilizers.is_empty() {
                        return Err(
                            "qec_decode: surface-code decoding is not implemented; only repetition codes can be decoded"
                                .into(),
                        );
                    }
                    if syndrome.len() != code.z_stabilizers.len() {
                        return Err(format!(
                            "qec_decode: syndrome length {} != {} stabilizers",
                            syndrome.len(),
                            code.z_stabilizers.len()
                        ));
                    }
                    crate::qec::decode_repetition_code(&syndrome, code)
                }
                _ => return Err("qec_decode: second arg must be SurfaceCode".into()),
            };
            let arr: Vec<Value> = corrections
                .into_iter()
                .map(|c| Value::Int(c as i64))
                .collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        "qec_logical_error_rate" => {
            let distance = extract_in_range(args, 0, "distance", 2, MAX_QEC_DISTANCE)?;
            let p = extract_prob(args, 1, "error_rate")?;
            let rounds = extract_usize(args, 2, "rounds")?;
            let seed = extract_seed(args, 3, "seed")?;
            let rate = crate::qec::estimate_logical_error_rate(distance, p, rounds, seed);
            Ok(Some(Value::Float(rate)))
        }

        // =======================================================================
        // QML — Quantum Machine Learning
        // =======================================================================
        "qml_train" => {
            let n_qubits = extract_in_range(args, 0, "n_qubits", 1, MAX_MPS_QUBITS)?;
            let layers = extract_usize(args, 1, "layers")?;
            let n_classes = extract_in_range(args, 2, "n_classes", 1, n_qubits)?;
            let chi = extract_at_least(args, 3, "max_bond", 1)?;
            let lr = extract_f64(args, 4, "learning_rate")?;
            let epochs = extract_usize(args, 5, "epochs")?;
            let seed = extract_seed(args, 6, "seed")?;

            // Build config with readout qubits = first n_classes qubits
            let config = crate::qml::QmlConfig {
                n_qubits,
                n_reupload_passes: layers,
                n_classes,
                max_bond: chi,
                readout_qubits: (0..n_classes).collect(),
                learning_rate: lr,
                epochs,
                batch_size: 4,
                loss: crate::qml::QmlLoss::CrossEntropy,
                seed,
            };

            // Build dataset from args[7] (samples array) and args[8] (labels array).
            // Every element is validated: non-numeric features and out-of-range
            // labels are errors, not silent zeros.
            let samples = match args.get(7) {
                Some(Value::Array(arr)) => arr
                    .iter()
                    .enumerate()
                    .map(|(r, row)| match row {
                        Value::Array(inner) => {
                            if inner.len() != n_qubits {
                                return Err(format!(
                                    "qml_train: samples[{}] has {} features, expected n_qubits={}",
                                    r,
                                    inner.len(),
                                    n_qubits
                                ));
                            }
                            inner
                                .iter()
                                .enumerate()
                                .map(|(c, v)| extract_angle(v, &format!("samples[{}][{}]", r, c)))
                                .collect::<Result<Vec<f64>, String>>()
                        }
                        other => Err(format!(
                            "qml_train: samples[{}] must be an array, got {}",
                            r,
                            other.type_name()
                        )),
                    })
                    .collect::<Result<Vec<Vec<f64>>, String>>()?,
                _ => return Err("qml_train: arg 7 must be samples array".into()),
            };
            let labels = match args.get(8) {
                Some(Value::Array(arr)) => arr
                    .iter()
                    .enumerate()
                    .map(|(i, v)| match v {
                        Value::Int(l) if *l >= 0 && (*l as usize) < n_classes => Ok(*l as usize),
                        other => Err(format!(
                            "qml_train: labels[{}] must be an integer in 0..{}, got {}",
                            i, n_classes, other
                        )),
                    })
                    .collect::<Result<Vec<usize>, String>>()?,
                _ => return Err("qml_train: arg 8 must be labels array".into()),
            };
            if samples.len() != labels.len() {
                return Err(format!(
                    "qml_train: {} samples but {} labels",
                    samples.len(),
                    labels.len()
                ));
            }
            if samples.is_empty() {
                return Err("qml_train: dataset must not be empty".into());
            }

            let dataset = crate::qml::QmlDataset {
                samples,
                labels,
                n_classes,
            };
            let result = crate::qml::qml_train(&config, &dataset);

            let loss_arr: Vec<Value> = result.loss_history.into_iter().map(Value::Float).collect();
            let out = vec![
                Value::Float(result.final_accuracy),
                Value::Array(Rc::new(loss_arr)),
            ];
            Ok(Some(Value::Array(Rc::new(out))))
        }

        "qml_predict" => {
            let n_qubits = extract_in_range(args, 0, "n_qubits", 1, MAX_MPS_QUBITS)?;
            let layers = extract_usize(args, 1, "layers")?;
            let n_classes = extract_in_range(args, 2, "n_classes", 1, n_qubits)?;
            let chi = extract_at_least(args, 3, "max_bond", 1)?;

            let config = crate::qml::QmlConfig {
                n_qubits,
                n_reupload_passes: layers,
                n_classes,
                max_bond: chi,
                readout_qubits: (0..n_classes).collect(),
                learning_rate: 0.0,
                epochs: 0,
                batch_size: 1,
                loss: crate::qml::QmlLoss::CrossEntropy,
                seed: 0,
            };

            let params = extract_f64_array(args, 4, "params")?;
            let input = extract_f64_array(args, 5, "input")?;
            let expected = crate::qml::total_params(&config);
            if params.len() != expected {
                return Err(format!(
                    "qml_predict: params has {} values, expected {}",
                    params.len(),
                    expected
                ));
            }
            if input.len() != n_qubits {
                return Err(format!(
                    "qml_predict: input has {} values, expected n_qubits={}",
                    input.len(),
                    n_qubits
                ));
            }

            let class = crate::qml::predict(&config, &params, &input);
            Ok(Some(Value::Int(class as i64)))
        }

        // --- Inspect pure backend state ---
        "quantum_inspect" => {
            if args.is_empty() {
                return Err("quantum_inspect(state) requires 1 arg".into());
            }
            let map = crate::pure::quantum_inspect(&args[0])?;
            Ok(Some(map))
        }

        // =======================================================================
        // Fermion — Jordan-Wigner Hamiltonians
        // =======================================================================
        "q_fermion_h2" => {
            let h = crate::fermion::h2_hamiltonian();
            Ok(Some(wrap_any(h)))
        }

        "q_fermion_lih" => {
            let h = crate::fermion::lih_hamiltonian();
            Ok(Some(wrap_any(h)))
        }

        "q_fermion_new" => {
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_FERMION_QUBITS)?;
            Ok(Some(wrap_any(crate::fermion::FermionicHamiltonian::new(n))))
        }

        "q_fermion_n_terms" => {
            let n = read_i64::<crate::fermion::FermionicHamiltonian>(
                &args[0],
                "FermionicHamiltonian",
                |h| h.n_terms() as i64,
            )?;
            Ok(Some(Value::Int(n)))
        }

        "q_fermion_expectation" => {
            // q_fermion_expectation(hamiltonian, state): ⟨ψ|H|ψ⟩ where the state
            // is a circuit (executed) or a statevector (ADR-0045).
            let sv = state_vector(&args[1])?;
            let e = match &args[0] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let h = borrow
                        .downcast_ref::<crate::fermion::FermionicHamiltonian>()
                        .ok_or_else(|| "expected FermionicHamiltonian".to_string())?;
                    check_same_size(h.n_qubits, sv.n_qubits, "q_fermion_expectation")?;
                    h.expectation(&sv)
                }
                _ => {
                    return Err(
                        "q_fermion_expectation: first arg must be a FermionicHamiltonian".into(),
                    )
                }
            };
            Ok(Some(Value::Float(e)))
        }

        // =======================================================================
        // Trotter — Time Evolution
        // =======================================================================
        "q_trotter_evolve" => {
            // q_trotter_evolve(hamiltonian, state, time, n_steps, order)
            // order: 1 or 2
            let time = extract_f64(args, 2, "time")?;
            let n_steps = extract_at_least(args, 3, "n_steps", 1)?;
            let order = extract_trotter_order(args, 4)?;

            let mut sv = (*state_vector(&args[1])?).clone();

            match &args[0] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let h = borrow
                        .downcast_ref::<crate::fermion::FermionicHamiltonian>()
                        .ok_or_else(|| "expected FermionicHamiltonian".to_string())?;
                    check_same_size(h.n_qubits, sv.n_qubits, "q_trotter_evolve")?;
                    crate::trotter::trotter_evolve(&mut sv, h, time, n_steps, order);
                }
                _ => {
                    return Err("q_trotter_evolve: first arg must be a FermionicHamiltonian".into())
                }
            }

            Ok(Some(wrap_statevector(sv)))
        }

        "q_trotter_error" => {
            // q_trotter_error(hamiltonian, time, n_steps, order)
            let time = extract_f64(args, 1, "time")?;
            let n_steps = extract_at_least(args, 2, "n_steps", 1)?;
            let order = extract_trotter_order(args, 3)?;
            let bound = match &args[0] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let h = borrow
                        .downcast_ref::<crate::fermion::FermionicHamiltonian>()
                        .ok_or_else(|| "expected FermionicHamiltonian".to_string())?;
                    crate::trotter::trotter_error_bound(h, time, n_steps, order)
                }
                _ => return Err("q_trotter_error: first arg must be a FermionicHamiltonian".into()),
            };
            Ok(Some(Value::Float(bound)))
        }

        // =======================================================================
        // State interop (ADR-0045) and explicit copies (ADR-0044)
        // =======================================================================
        "q_expect_pauli" => {
            // q_expect_pauli(state, "XZIY"): ⟨ψ|P|ψ⟩, character k acts on qubit k.
            let sv = state_vector(&args[0])?;
            let ops = extract_pauli_string(args, 1, sv.n_qubits, "q_expect_pauli")?;
            let mut h = crate::fermion::FermionicHamiltonian::new(sv.n_qubits);
            h.add_term(crate::fermion::PauliTerm {
                coeff: ComplexF64::ONE,
                ops,
            });
            Ok(Some(Value::Float(h.expectation(&sv))))
        }

        "q_fermion_add_term" => {
            // q_fermion_add_term(h, "XZ", coeff) -> new Hamiltonian; h is unchanged.
            let coeff = extract_f64(args, 2, "coeff")?;
            let b = match &args[0] {
                Value::QuantumState(rc) => rc.borrow(),
                other => {
                    return Err(format!(
                        "q_fermion_add_term: first arg must be a FermionicHamiltonian, got {}",
                        other.type_name()
                    ))
                }
            };
            if let Some(h) = b.downcast_ref::<crate::fermion::FermionicHamiltonian>() {
                let ops = extract_pauli_string(args, 1, h.n_qubits, "q_fermion_add_term")?;
                let mut next = h.clone();
                next.add_term(crate::fermion::PauliTerm {
                    coeff: ComplexF64::real(coeff),
                    ops,
                });
                Ok(Some(wrap_any(next)))
            } else if let Some(h) = b.downcast_ref::<crate::pure::PureFermionicHamiltonian>() {
                let ops = extract_pauli_string(args, 1, h.n_qubits, "q_fermion_add_term")?
                    .into_iter()
                    .map(|p| match p {
                        crate::fermion::Pauli::I => crate::pure::PurePauli::I,
                        crate::fermion::Pauli::X => crate::pure::PurePauli::X,
                        crate::fermion::Pauli::Y => crate::pure::PurePauli::Y,
                        crate::fermion::Pauli::Z => crate::pure::PurePauli::Z,
                    })
                    .collect();
                let mut next = h.clone();
                next.terms.push(crate::pure::PurePauliTerm {
                    coeff_re: coeff,
                    coeff_im: 0.0,
                    ops,
                });
                Ok(Some(wrap_pure(next)))
            } else {
                Err("q_fermion_add_term: first arg must be a FermionicHamiltonian".into())
            }
        }

        "density_from_state" => {
            // density_from_state(state) -> ρ = |ψ⟩⟨ψ|. Size is checked before
            // a circuit is executed.
            if let Some(n) = state_n_qubits(&args[0]) {
                if n > MAX_DENSITY_QUBITS {
                    return Err(format!(
                        "density_from_state: density matrices support at most {} qubits, got {}",
                        MAX_DENSITY_QUBITS, n
                    ));
                }
            }
            let pure_input = match &args[0] {
                Value::QuantumState(rc) => {
                    let b = rc.borrow();
                    b.is::<PureCircuit>() || b.is::<crate::pure::PureStatevector>()
                }
                _ => false,
            };
            if pure_input {
                let psv = pure_state_vector(&args[0])?;
                Ok(Some(wrap_pure(PureDensity::from_statevector(&psv))))
            } else {
                let sv = state_vector(&args[0])?;
                Ok(Some(wrap_any(DensityMatrix::from_statevector(&sv))))
            }
        }

        "q_copy" => Ok(Some(deep_copy(&args[0])?)),

        // =======================================================================
        // OpenQASM 2.0 interchange
        // =======================================================================
        "q_to_qasm" => {
            // q_to_qasm(circuit) -> String (either backend).
            let text = match &args[0] {
                Value::QuantumState(rc) => {
                    let b = rc.borrow();
                    if let Some(c) = b.downcast_ref::<Circuit>() {
                        crate::qasm::to_qasm(c)
                    } else if let Some(c) = b.downcast_ref::<PureCircuit>() {
                        crate::qasm::pure_to_qasm(c)
                    } else {
                        return Err(format!("q_to_qasm: {}", not_a_circuit(&*b)));
                    }
                }
                other => {
                    return Err(format!(
                        "q_to_qasm: expected a quantum circuit, got {}",
                        other.type_name()
                    ))
                }
            };
            Ok(Some(Value::String(Rc::new(text))))
        }

        "q_from_qasm" => {
            // q_from_qasm(text[, "pure"]) -> circuit.
            let text = match &args[0] {
                Value::String(s) => s.clone(),
                other => {
                    return Err(format!(
                        "q_from_qasm: expected OpenQASM 2.0 text (a string), got {}",
                        other.type_name()
                    ))
                }
            };
            match args.get(1) {
                None => Ok(Some(wrap_circuit(crate::qasm::from_qasm(&text)?))),
                Some(Value::String(s)) if s.as_str() == "pure" => {
                    Ok(Some(wrap_pure(crate::qasm::pure_from_qasm(&text)?)))
                }
                Some(other) => Err(format!(
                    "q_from_qasm: second argument must be \"pure\", got {}",
                    other
                )),
            }
        }

        // =======================================================================
        // ZNE — Zero-Noise Extrapolation
        // =======================================================================
        "q_zne_mitigate" => {
            // q_zne_mitigate(scale_factors_array, measured_values_array)
            let scales = extract_f64_array(args, 0, "scale_factors")?;
            let values = extract_f64_array(args, 1, "measured_values")?;

            let result = crate::mitigation::richardson_extrapolate(&scales, &values)?;
            let out = vec![
                Value::Float(result.mitigated_value),
                Value::Array(Rc::new(
                    result.coefficients.into_iter().map(Value::Float).collect(),
                )),
            ];
            Ok(Some(Value::Array(Rc::new(out))))
        }

        "q_zne_linear" => {
            // q_zne_linear(lambda1, value1, lambda2, value2)
            let l1 = extract_f64(args, 0, "lambda1")?;
            let v1 = extract_f64(args, 1, "value1")?;
            let l2 = extract_f64(args, 2, "lambda2")?;
            let v2 = extract_f64(args, 3, "value2")?;
            let result = crate::mitigation::linear_extrapolate(l1, v1, l2, v2)?;
            Ok(Some(Value::Float(result)))
        }

        "q_scale_noise" => {
            // q_scale_noise(base_p, scale_factor, noise_type)
            let base_p = extract_prob(args, 0, "base_p")?;
            let scale = extract_f64(args, 1, "scale_factor")?;
            let result = match args.get(2) {
                None => crate::mitigation::scale_depolarizing_noise(base_p, scale),
                Some(Value::String(s)) => match s.as_str() {
                    "depolarizing" => crate::mitigation::scale_depolarizing_noise(base_p, scale),
                    "dephasing" => crate::mitigation::scale_dephasing_noise(base_p, scale),
                    "amplitude_damping" => {
                        crate::mitigation::scale_amplitude_damping(base_p, scale)
                    }
                    other => {
                        return Err(format!(
                            "q_scale_noise: noise_type must be \"depolarizing\", \"dephasing\", or \"amplitude_damping\", got \"{}\"",
                            other
                        ))
                    }
                },
                Some(other) => {
                    return Err(format!(
                        "q_scale_noise: noise_type must be a string, got {}",
                        other.type_name()
                    ))
                }
            };
            Ok(Some(Value::Float(result)))
        }

        // =======================================================================
        // MPS — Canonical Form & SWAP Network
        // =======================================================================
        "mps_left_canonicalize" => {
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.left_canonicalize();
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_right_canonicalize" => {
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.right_canonicalize();
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_mixed_canonicalize" => {
            let center = extract_usize(args, 1, "center")?;
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.mixed_canonicalize(center);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_swap" => {
            let q1 = extract_usize(args, 1, "qubit1")?;
            let q2 = extract_usize(args, 2, "qubit2")?;
            let zero = ComplexF64::ZERO;
            let one = ComplexF64::ONE;
            let swap_gate = [
                [one, zero, zero, zero],
                [zero, zero, one, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
            ];
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_gate_swap_network(q1, q2, swap_gate);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        _ => Ok(None),
    }
}

/// Qubit-index, adjacency, and probability checks shared by both backends.
fn prevalidate(name: &str, args: &[Value]) -> Result<(), String> {
    match name {
        "q_h" | "q_x" | "q_y" | "q_z" | "q_s" | "q_t" | "q_rx" | "q_ry" | "q_rz" | "mps_h"
        | "mps_x" | "mps_ry" | "mps_z_expectation" | "stabilizer_h" | "stabilizer_s"
        | "stabilizer_x" | "stabilizer_y" | "stabilizer_z" | "stabilizer_measure" => {
            check_state_qubits(args, &[(1, "qubit")], name)
        }
        "mps_mixed_canonicalize" => check_state_qubits(args, &[(1, "center")], name),
        "q_cx" | "q_cnot" | "q_cz" | "q_swap" | "mps_swap" | "stabilizer_cnot"
        | "density_cnot" => check_state_qubits(args, &[(1, "qubit_a"), (2, "qubit_b")], name),
        "mps_cnot" => {
            check_state_qubits(args, &[(1, "control"), (2, "target")], name)?;
            let (c, t) = (extract_usize(args, 1, "control")?, extract_usize(args, 2, "target")?);
            if c.abs_diff(t) != 1 {
                return Err(format!(
                    "mps_cnot requires adjacent qubits, got ({}, {}); use mps_swap to move qubits",
                    c, t
                ));
            }
            Ok(())
        }
        "q_toffoli" | "q_ccx" => check_state_qubits(
            args,
            &[(1, "ctrl1"), (2, "ctrl2"), (3, "target")],
            name,
        ),
        "density_gate" => check_state_qubits(args, &[(2, "qubit")], name),
        "density_depolarize" | "density_dephase" | "density_amplitude_damp" => {
            check_state_qubits(args, &[(1, "qubit")], name)?;
            extract_prob(args, 2, "probability").map(|_| ())
        }
        _ => Ok(()),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pure backend dispatch
// ═══════════════════════════════════════════════════════════════════
//
// Routes to pure CJC implementations when:
// 1. Constructor called with "pure" flag: mps_new(50, 16, "pure")
// 2. Operation called on pure-backend state: mps_h(pure_mps, 0)

fn dispatch_pure(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    use crate::pure::*;

    match name {
        // === Pure constructors (triggered by "pure" flag) ===
        "qubits" if has_pure_flag(args) => {
            let n = extract_usize(args, 0, "n_qubits")?;
            if n < 1 || n > 26 {
                return Err(format!("qubits() requires 1-26 qubits, got {}", n));
            }
            Ok(Some(wrap_pure(PureCircuit::new(n))))
        }

        "mps_new" if has_pure_flag(args) => {
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_MPS_QUBITS)?;
            let chi = if args.len() > 2 {
                extract_at_least(args, 1, "max_bond", 1)?
            } else {
                32
            };
            Ok(Some(wrap_pure(PureMps::new(n, chi))))
        }

        "stabilizer_new" if has_pure_flag(args) => {
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_STABILIZER_QUBITS)?;
            Ok(Some(wrap_pure(PureStabilizer::new(n))))
        }

        "density_new" if has_pure_flag(args) => {
            // Same cap as the Rust backend: 4^n complex entries.
            let n = extract_in_range(args, 0, "n_qubits", 1, MAX_DENSITY_QUBITS)?;
            Ok(Some(wrap_pure(PureDensity::new(n))))
        }

        // === Pure MPS operations (auto-detected) ===
        "mps_h" | "mps_x" if is_pure::<PureMps>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let mat = if name == "mps_h" {
                h_matrix()
            } else {
                x_matrix()
            };
            pure_mps_mut(&args[0], |mps| mps.apply_single_qubit(q, mat))?;
            Ok(Some(args[0].clone()))
        }

        "mps_ry" if is_pure::<PureMps>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let theta = extract_f64(args, 2, "theta")?;
            pure_mps_mut(&args[0], |mps| mps.apply_single_qubit(q, ry_matrix(theta)))?;
            Ok(Some(args[0].clone()))
        }

        "mps_cnot" if is_pure::<PureMps>(&args[0]) => {
            let ctrl = extract_usize(args, 1, "control")?;
            let targ = extract_usize(args, 2, "target")?;
            pure_mps_mut(&args[0], |mps| mps.apply_cnot(ctrl, targ))?;
            Ok(Some(args[0].clone()))
        }

        "mps_z_expectation" if is_pure::<PureMps>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let z = pure_mps_ref(&args[0], |mps| mps.z_expectation(q))?;
            Ok(Some(Value::Float(z)))
        }

        "mps_memory" if is_pure::<PureMps>(&args[0]) => {
            let mem = pure_mps_ref(&args[0], |mps| mps.memory_bytes() as f64)?;
            Ok(Some(Value::Int(mem as i64)))
        }

        // === Pure Stabilizer operations ===
        "stabilizer_h" | "stabilizer_s" | "stabilizer_x" | "stabilizer_y" | "stabilizer_z"
            if is_pure::<PureStabilizer>(&args[0]) =>
        {
            let q = extract_usize(args, 1, "qubit")?;
            pure_stab_mut(&args[0], |s| match name {
                "stabilizer_h" => s.h(q),
                "stabilizer_s" => s.s(q),
                "stabilizer_x" => s.x(q),
                "stabilizer_y" => s.y(q),
                "stabilizer_z" => s.z(q),
                _ => unreachable!(),
            })?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_cnot" if is_pure::<PureStabilizer>(&args[0]) => {
            let ctrl = extract_usize(args, 1, "control")?;
            let tgt = extract_usize(args, 2, "target")?;
            pure_stab_mut(&args[0], |s| s.cnot(ctrl, tgt))?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_measure" if is_pure::<PureStabilizer>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let seed = extract_seed(args, 2, "seed")?;
            let outcome = match &args[0] {
                Value::QuantumState(rc) => {
                    let mut borrow = rc.borrow_mut();
                    let s = borrow
                        .downcast_mut::<PureStabilizer>()
                        .ok_or_else(|| "expected PureStabilizer".to_string())?;
                    let mut rng = seed;
                    s.measure(q, &mut rng) as i64
                }
                _ => return Err("stabilizer_measure: expected PureStabilizer".into()),
            };
            Ok(Some(Value::Int(outcome)))
        }

        "stabilizer_n_qubits" if is_pure::<PureStabilizer>(&args[0]) => {
            let n = pure_stab_ref(&args[0], |s| s.num_qubits() as f64)?;
            Ok(Some(Value::Int(n as i64)))
        }

        // === Pure Density Matrix operations ===
        "density_gate" if is_pure::<PureDensity>(&args[0]) => {
            let gate_name = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err("density_gate: gate name must be a string".into()),
            };
            let q = extract_usize(args, 2, "qubit")?;
            let mat = match gate_name.as_str() {
                "H" => h_matrix(),
                "X" => x_matrix(),
                "Y" => y_matrix(),
                "Z" => z_matrix(),
                "S" => s_matrix(),
                "T" => t_matrix(),
                _ => return Err(format!("density_gate: unknown gate '{}'", gate_name)),
            };
            pure_density_mut(&args[0], |dm| dm.apply_gate_2x2(q, mat))?;
            Ok(Some(args[0].clone()))
        }

        "density_cnot" if is_pure::<PureDensity>(&args[0]) => {
            let ctrl = extract_usize(args, 1, "control")?;
            let tgt = extract_usize(args, 2, "target")?;
            pure_density_mut(&args[0], |dm| dm.apply_cnot(ctrl, tgt))?;
            Ok(Some(args[0].clone()))
        }

        "density_depolarize" if is_pure::<PureDensity>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let p = extract_f64(args, 2, "probability")?;
            pure_density_mut(&args[0], |dm| dm.apply_depolarize(q, p))?;
            Ok(Some(args[0].clone()))
        }

        "density_dephase" if is_pure::<PureDensity>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let p = extract_f64(args, 2, "probability")?;
            pure_density_mut(&args[0], |dm| dm.apply_dephase(q, p))?;
            Ok(Some(args[0].clone()))
        }

        "density_amplitude_damp" if is_pure::<PureDensity>(&args[0]) => {
            let q = extract_usize(args, 1, "qubit")?;
            let gamma = extract_f64(args, 2, "gamma")?;
            pure_density_mut(&args[0], |dm| dm.apply_amplitude_damp(q, gamma))?;
            Ok(Some(args[0].clone()))
        }

        "density_trace" if is_pure::<PureDensity>(&args[0]) => {
            let tr = pure_density_ref(&args[0], |dm| dm.trace())?;
            Ok(Some(Value::Float(tr)))
        }

        "density_purity" if is_pure::<PureDensity>(&args[0]) => {
            let p = pure_density_ref(&args[0], |dm| dm.purity())?;
            Ok(Some(Value::Float(p)))
        }

        "density_entropy" if is_pure::<PureDensity>(&args[0]) => {
            let e = pure_density_ref(&args[0], |dm| dm.von_neumann_entropy())?;
            Ok(Some(Value::Float(e)))
        }

        "density_probs" if is_pure::<PureDensity>(&args[0]) => {
            let probs = pure_density_ref_vec(&args[0], |dm| dm.probabilities())?;
            let arr: Vec<Value> = probs.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        // === Pure Circuit operations ===
        "q_h" | "q_x" | "q_y" | "q_z" | "q_s" | "q_t" if is_pure::<PureCircuit>(&args[0]) => {
            let q = extract_qubit_index(&args[1], "qubit")?;
            let gate = match name {
                "q_h" => PureGate::H(q),
                "q_x" => PureGate::X(q),
                "q_y" => PureGate::Y(q),
                "q_z" => PureGate::Z(q),
                "q_s" => PureGate::S(q),
                "q_t" => PureGate::T(q),
                _ => unreachable!(),
            };
            Ok(Some(pure_circuit_with_gate(&args[0], gate)?))
        }

        "q_rx" | "q_ry" | "q_rz" if is_pure::<PureCircuit>(&args[0]) => {
            let q = extract_qubit_index(&args[1], "qubit")?;
            let theta = extract_f64(args, 2, "angle")?;
            let gate = match name {
                "q_rx" => PureGate::Rx(q, theta),
                "q_ry" => PureGate::Ry(q, theta),
                "q_rz" => PureGate::Rz(q, theta),
                _ => unreachable!(),
            };
            Ok(Some(pure_circuit_with_gate(&args[0], gate)?))
        }

        "q_cx" | "q_cnot" if is_pure::<PureCircuit>(&args[0]) => {
            let a = extract_qubit_index(&args[1], "a")?;
            let b = extract_qubit_index(&args[2], "b")?;
            Ok(Some(pure_circuit_with_gate(&args[0], PureGate::CNOT(a, b))?))
        }

        "q_cz" if is_pure::<PureCircuit>(&args[0]) => {
            let a = extract_qubit_index(&args[1], "a")?;
            let b = extract_qubit_index(&args[2], "b")?;
            Ok(Some(pure_circuit_with_gate(&args[0], PureGate::CZ(a, b))?))
        }

        "q_swap" if is_pure::<PureCircuit>(&args[0]) => {
            let a = extract_qubit_index(&args[1], "a")?;
            let b = extract_qubit_index(&args[2], "b")?;
            Ok(Some(pure_circuit_with_gate(&args[0], PureGate::SWAP(a, b))?))
        }

        "q_run" if is_pure::<PureCircuit>(&args[0]) => {
            let sv = pure_circuit_ref(&args[0], |c| c.execute())?;
            Ok(Some(wrap_pure(sv)))
        }

        "q_probs" if is_pure::<PureCircuit>(&args[0]) => {
            let sv = pure_circuit_ref(&args[0], |c| c.execute())?;
            let probs = sv.probabilities();
            let arr: Vec<Value> = probs.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        "q_measure" if is_pure::<PureCircuit>(&args[0]) => {
            let seed = match &args[1] {
                Value::Int(s) => *s as u64,
                _ => return Err("q_measure seed must be an integer".into()),
            };
            let mut rng = seed;
            let outcomes = pure_circuit_ref(&args[0], |c| c.execute_and_measure(&mut rng))?;
            let arr: Vec<Value> = outcomes.into_iter().map(|b| Value::Int(b as i64)).collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        "q_n_qubits" if is_pure::<PureCircuit>(&args[0]) => {
            let n = pure_circuit_ref(&args[0], |c| c.n_qubits)?;
            Ok(Some(Value::Int(n as i64)))
        }

        "q_n_gates" if is_pure::<PureCircuit>(&args[0]) => {
            let n = pure_circuit_ref(&args[0], |c| c.n_gates())?;
            Ok(Some(Value::Int(n as i64)))
        }

        // === Pure Fermion/Trotter/ZNE (triggered by "pure" flag) ===
        "q_fermion_h2" if has_pure_flag(args) => Ok(Some(wrap_pure(pure_h2_hamiltonian()))),

        "q_fermion_expectation" if is_pure::<PureFermionicHamiltonian>(&args[0]) => {
            // Second arg: any state argument (circuit or statevector, ADR-0045).
            let psv = pure_state_vector(&args[1])?;
            let h_borrow = match &args[0] {
                Value::QuantumState(hrc) => hrc.borrow(),
                _ => return Err("expected PureFermionicHamiltonian".into()),
            };
            let h = h_borrow
                .downcast_ref::<PureFermionicHamiltonian>()
                .ok_or_else(|| "expected PureFermionicHamiltonian".to_string())?;
            check_same_size(h.n_qubits, psv.n_qubits, "q_fermion_expectation")?;
            let e = h.expectation(&psv.amplitudes, psv.n_qubits);
            Ok(Some(Value::Float(e)))
        }

        "q_zne_mitigate" if has_pure_flag(args) => {
            let scales = extract_f64_array(args, 0, "scale_factors")?;
            let values = extract_f64_array(args, 1, "measured_values")?;
            let result = pure_richardson_extrapolate(&scales, &values)?;
            let out = vec![
                Value::Float(result.mitigated_value),
                Value::Array(Rc::new(
                    result.coefficients.into_iter().map(Value::Float).collect(),
                )),
            ];
            Ok(Some(Value::Array(Rc::new(out))))
        }

        _ => Ok(None),
    }
}

// Pure backend helper functions
fn pure_mps_mut(val: &Value, f: impl FnOnce(&mut PureMps)) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut b = rc.borrow_mut();
            let mps = b.downcast_mut::<PureMps>().ok_or("expected PureMps")?;
            f(mps);
            Ok(())
        }
        _ => Err("expected PureMps".into()),
    }
}

fn pure_mps_ref(val: &Value, f: impl FnOnce(&PureMps) -> f64) -> Result<f64, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let mps = b.downcast_ref::<PureMps>().ok_or("expected PureMps")?;
            Ok(f(mps))
        }
        _ => Err("expected PureMps".into()),
    }
}

fn pure_stab_mut(val: &Value, f: impl FnOnce(&mut PureStabilizer)) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut b = rc.borrow_mut();
            let s = b
                .downcast_mut::<PureStabilizer>()
                .ok_or("expected PureStabilizer")?;
            f(s);
            Ok(())
        }
        _ => Err("expected PureStabilizer".into()),
    }
}

fn pure_stab_ref(val: &Value, f: impl FnOnce(&PureStabilizer) -> f64) -> Result<f64, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let s = b
                .downcast_ref::<PureStabilizer>()
                .ok_or("expected PureStabilizer")?;
            Ok(f(s))
        }
        _ => Err("expected PureStabilizer".into()),
    }
}

fn pure_density_mut(val: &Value, f: impl FnOnce(&mut PureDensity)) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut b = rc.borrow_mut();
            let dm = b
                .downcast_mut::<PureDensity>()
                .ok_or("expected PureDensity")?;
            f(dm);
            Ok(())
        }
        _ => Err("expected PureDensity".into()),
    }
}

fn pure_density_ref(val: &Value, f: impl FnOnce(&PureDensity) -> f64) -> Result<f64, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let dm = b
                .downcast_ref::<PureDensity>()
                .ok_or("expected PureDensity")?;
            Ok(f(dm))
        }
        _ => Err("expected PureDensity".into()),
    }
}

fn pure_density_ref_vec(
    val: &Value,
    f: impl FnOnce(&PureDensity) -> Vec<f64>,
) -> Result<Vec<f64>, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let dm = b
                .downcast_ref::<PureDensity>()
                .ok_or("expected PureDensity")?;
            Ok(f(dm))
        }
        _ => Err("expected PureDensity".into()),
    }
}

fn pure_circuit_ref<T>(val: &Value, f: impl FnOnce(&PureCircuit) -> T) -> Result<T, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let c = b
                .downcast_ref::<PureCircuit>()
                .ok_or("expected PureCircuit")?;
            Ok(f(c))
        }
        _ => Err("expected PureCircuit".into()),
    }
}

fn wrap_circuit(circ: Circuit) -> Value {
    Value::QuantumState(Rc::new(RefCell::new(circ)))
}

fn wrap_statevector(sv: Statevector) -> Value {
    Value::QuantumState(Rc::new(RefCell::new(sv)))
}

/// Wrap any type as a QuantumState value.
fn wrap_any<T: Any + 'static>(val: T) -> Value {
    Value::QuantumState(Rc::new(RefCell::new(val)))
}

/// Borrow a typed value immutably and extract a float.
fn read_f64<T: Any + 'static>(
    val: &Value,
    type_name: &str,
    f: impl FnOnce(&T) -> f64,
) -> Result<f64, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let obj = borrow
                .downcast_ref::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            Ok(f(obj))
        }
        _ => Err(format!(
            "expected QuantumState({}), got {}",
            type_name,
            val.type_name()
        )),
    }
}

/// Borrow a typed value immutably and extract an i64.
fn read_i64<T: Any + 'static>(
    val: &Value,
    type_name: &str,
    f: impl FnOnce(&T) -> i64,
) -> Result<i64, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let obj = borrow
                .downcast_ref::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            Ok(f(obj))
        }
        _ => Err(format!(
            "expected QuantumState({}), got {}",
            type_name,
            val.type_name()
        )),
    }
}

/// Borrow a typed value immutably and extract a Vec<f64>.
fn read_vec_f64<T: Any + 'static>(
    val: &Value,
    type_name: &str,
    f: impl FnOnce(&T) -> Vec<f64>,
) -> Result<Vec<f64>, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let obj = borrow
                .downcast_ref::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            Ok(f(obj))
        }
        _ => Err(format!(
            "expected QuantumState({}), got {}",
            type_name,
            val.type_name()
        )),
    }
}

/// Borrow a typed value mutably from a QuantumState.
fn with_any_mut<T: Any + 'static>(
    val: &Value,
    type_name: &str,
    f: impl FnOnce(&mut T) -> Result<(), String>,
) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut borrow = rc.borrow_mut();
            let obj = borrow
                .downcast_mut::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            f(obj)
        }
        _ => Err(format!(
            "expected QuantumState({}), got {}",
            type_name,
            val.type_name()
        )),
    }
}

/// Integer argument. A `Float` is accepted only if it is finite and integral
/// (e.g. `4.0`); `1.9` is rejected rather than silently truncated.
fn extract_int(args: &[Value], idx: usize, name: &str) -> Result<i64, String> {
    match args.get(idx) {
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Float(f)) if f.is_finite() && f.fract() == 0.0 => Ok(*f as i64),
        Some(Value::Float(f)) => Err(format!("{} must be an integer, got {}", name, f)),
        Some(other) => Err(format!(
            "{} must be an integer, got {}",
            name,
            other.type_name()
        )),
        None => Err(format!("missing argument: {}", name)),
    }
}

/// Non-negative integer argument (sizes, indices, counts).
fn extract_usize(args: &[Value], idx: usize, name: &str) -> Result<usize, String> {
    let v = extract_int(args, idx, name)?;
    if v < 0 {
        return Err(format!("{} must be non-negative, got {}", name, v));
    }
    Ok(v as usize)
}

/// Integer argument constrained to `lo..=hi`.
fn extract_in_range(
    args: &[Value],
    idx: usize,
    name: &str,
    lo: usize,
    hi: usize,
) -> Result<usize, String> {
    let v = extract_int(args, idx, name)?;
    if v < lo as i64 || v > hi as i64 {
        return Err(format!("{} must be in {}..={}, got {}", name, lo, hi, v));
    }
    Ok(v as usize)
}

/// Integer argument with only a lower bound (e.g. bond dimension, step count).
fn extract_at_least(args: &[Value], idx: usize, name: &str, lo: usize) -> Result<usize, String> {
    let v = extract_int(args, idx, name)?;
    if v < lo as i64 {
        return Err(format!("{} must be at least {}, got {}", name, lo, v));
    }
    Ok(v as usize)
}

/// Finite numeric argument by position (angles, rates, times, probabilities).
fn extract_f64(args: &[Value], idx: usize, name: &str) -> Result<f64, String> {
    match args.get(idx) {
        Some(v) => extract_angle(v, name),
        None => Err(format!("missing argument: {}", name)),
    }
}

/// Probability-like argument in `[0, 1]`.
fn extract_prob(args: &[Value], idx: usize, name: &str) -> Result<f64, String> {
    let p = extract_f64(args, idx, name)?;
    if !(0.0..=1.0).contains(&p) {
        return Err(format!("{} must be in [0, 1], got {}", name, p));
    }
    Ok(p)
}

/// Seed argument: any integer, reinterpreted as `u64` (so `-1` → `u64::MAX`).
fn extract_seed(args: &[Value], idx: usize, name: &str) -> Result<u64, String> {
    Ok(extract_int(args, idx, name)? as u64)
}

/// Array of numbers (Int or finite Float). Non-numeric elements are an error,
/// not a silent 0.0.
fn extract_f64_array(args: &[Value], idx: usize, name: &str) -> Result<Vec<f64>, String> {
    match args.get(idx) {
        Some(Value::Array(arr)) => arr
            .iter()
            .enumerate()
            .map(|(i, v)| extract_angle(v, &format!("{}[{}]", name, i)))
            .collect(),
        Some(other) => Err(format!("{} must be an array, got {}", name, other.type_name())),
        None => Err(format!("missing argument: {}", name)),
    }
}

fn check_same_size(h_qubits: usize, state_qubits: usize, what: &str) -> Result<(), String> {
    if h_qubits != state_qubits {
        return Err(format!(
            "{}: Hamiltonian has {} qubits but the state has {}",
            what, h_qubits, state_qubits
        ));
    }
    Ok(())
}

/// Trotter order: omitted → 1st order; otherwise exactly 1 or 2.
fn extract_trotter_order(args: &[Value], idx: usize) -> Result<crate::trotter::TrotterOrder, String> {
    match args.get(idx) {
        None => Ok(crate::trotter::TrotterOrder::First),
        Some(_) => match extract_int(args, idx, "order")? {
            1 => Ok(crate::trotter::TrotterOrder::First),
            2 => Ok(crate::trotter::TrotterOrder::Second),
            other => Err(format!("Trotter order must be 1 or 2, got {}", other)),
        },
    }
}

fn check_qubit(q: usize, n: usize, what: &str) -> Result<(), String> {
    if q >= n {
        return Err(format!("{}: qubit {} out of range (n_qubits={})", what, q, n));
    }
    Ok(())
}

fn check_distinct(qs: &[usize], what: &str) -> Result<(), String> {
    for i in 0..qs.len() {
        for j in (i + 1)..qs.len() {
            if qs[i] == qs[j] {
                return Err(format!("{}: qubit operands must be distinct, got {:?}", what, qs));
            }
        }
    }
    Ok(())
}

/// Largest qubit counts accepted from `.cjcl`. Dense and density caps come from
/// memory (2^n and 4^n complex numbers); MPS and stabilizer caps keep a single
/// constructor call bounded (tableau memory is ~ n²/4 bytes).
const MAX_DENSITY_QUBITS: usize = 14;
const MAX_MPS_QUBITS: usize = 100_000;
const MAX_STABILIZER_QUBITS: usize = 32_768;
const MAX_FERMION_QUBITS: usize = 26;
const MAX_QEC_DISTANCE: usize = 1024;

/// Minimum number of arguments each builtin needs. Checked before any arm
/// indexes `args`, so a short argument list is an error rather than a panic.
fn min_arity(name: &str) -> Option<usize> {
    Some(match name {
        "q_fermion_h2" | "q_fermion_lih" => 0,
        "qubits" | "q_run" | "q_probs" | "q_amplitudes" | "q_n_qubits" | "q_n_gates"
        | "mps_new" | "mps_energy" | "mps_memory" | "mps_left_canonicalize"
        | "mps_right_canonicalize" | "qaoa_graph_cycle" | "stabilizer_new"
        | "stabilizer_n_qubits" | "density_new" | "density_trace" | "density_purity"
        | "density_entropy" | "density_probs" | "qec_repetition_code" | "qec_surface_code"
        | "quantum_inspect" | "q_fermion_new" | "q_fermion_n_terms" | "q_copy"
        | "density_from_state" | "q_to_qasm" | "q_from_qasm" => 1,
        "q_h" | "q_x" | "q_y" | "q_z" | "q_s" | "q_t" | "q_measure" | "mps_h" | "mps_x"
        | "mps_z_expectation" | "stabilizer_h" | "stabilizer_s" | "stabilizer_x"
        | "stabilizer_y" | "stabilizer_z" | "qec_decode" | "q_fermion_expectation"
        | "q_zne_mitigate" | "q_scale_noise" | "mps_mixed_canonicalize" | "q_expect_pauli" => 2,
        "q_rx" | "q_ry" | "q_rz" | "q_cx" | "q_cnot" | "q_cz" | "q_swap" | "q_sample"
        | "mps_ry" | "mps_cnot" | "mps_swap" | "stabilizer_cnot" | "stabilizer_measure"
        | "density_gate" | "density_cnot" | "density_depolarize" | "density_dephase"
        | "density_amplitude_damp" | "qec_syndrome" | "q_trotter_error" | "q_fermion_add_term" => 3,
        "q_toffoli" | "q_ccx" | "dmrg_ising" | "dmrg_heisenberg" | "qec_logical_error_rate"
        | "q_trotter_evolve" | "q_zne_linear" => 4,
        "vqe_heisenberg" | "vqe_full_heisenberg" => 5,
        "qaoa_maxcut" | "qml_predict" => 6,
        "qml_train" => 9,
        _ => return None,
    })
}

/// Number of qubits in any quantum state value that has a qubit count.
fn state_n_qubits(val: &Value) -> Option<usize> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            if let Some(c) = b.downcast_ref::<Circuit>() {
                Some(c.n_qubits())
            } else if let Some(m) = b.downcast_ref::<Mps>() {
                Some(m.n_qubits)
            } else if let Some(s) = b.downcast_ref::<StabilizerState>() {
                Some(s.n)
            } else if let Some(d) = b.downcast_ref::<DensityMatrix>() {
                Some(d.n_qubits)
            } else if let Some(c) = b.downcast_ref::<PureCircuit>() {
                Some(c.n_qubits)
            } else if let Some(m) = b.downcast_ref::<PureMps>() {
                Some(m.n_qubits)
            } else if let Some(s) = b.downcast_ref::<PureStabilizer>() {
                Some(s.n)
            } else if let Some(d) = b.downcast_ref::<PureDensity>() {
                Some(d.n_qubits)
            } else if let Some(sv) = b.downcast_ref::<Statevector>() {
                Some(sv.n_qubits)
            } else if let Some(sv) = b.downcast_ref::<crate::pure::PureStatevector>() {
                Some(sv.n_qubits)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Validate qubit-index arguments `idxs` against the state in `args[0]`, and
/// require them to be pairwise distinct. Runs before any arm touches the state.
fn check_state_qubits(args: &[Value], idxs: &[(usize, &str)], what: &str) -> Result<(), String> {
    let n = match args.first().and_then(state_n_qubits) {
        Some(n) => n,
        None => return Ok(()), // wrong type: the arm reports a typed error
    };
    let mut qs = Vec::with_capacity(idxs.len());
    for &(i, label) in idxs {
        let q = extract_usize(args, i, label)?;
        check_qubit(q, n, what)?;
        qs.push(q);
    }
    check_distinct(&qs, what)
}

fn extract_qubit_index(val: &Value, name: &str) -> Result<usize, String> {
    match val {
        Value::Int(i) => {
            if *i < 0 {
                Err(format!("{} must be non-negative, got {}", name, i))
            } else {
                Ok(*i as usize)
            }
        }
        _ => Err(format!("{} must be an integer", name)),
    }
}

/// Finite numeric value. NaN and ±∞ are rejected: no quantum builtin has a
/// meaningful result for a non-finite angle, rate, time, or probability.
fn extract_angle(val: &Value, name: &str) -> Result<f64, String> {
    let x = match val {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err(format!("{} must be a number", name)),
    };
    if !x.is_finite() {
        return Err(format!("{} must be finite, got {}", name, x));
    }
    Ok(x)
}

/// Error for a circuit builtin that received some other quantum value. Names
/// the backend mismatch or statevector case instead of a generic message.
fn not_a_circuit(obj: &dyn Any) -> String {
    if obj.is::<PureCircuit>() {
        "this operation is not supported by the \"pure\" backend; build the circuit with qubits(n) instead of qubits(n, \"pure\")".into()
    } else if obj.is::<Statevector>() || obj.is::<crate::pure::PureStatevector>() {
        "expected a quantum circuit, got a statevector (from q_run/q_trotter_evolve); gates apply to circuits, observables accept either".into()
    } else {
        "expected a quantum circuit".into()
    }
}

/// The statevector a state argument denotes (ADR-0045): a circuit is executed
/// (once: the result is cached on the circuit value, see
/// `Circuit::execute_shared`), a statevector is used as is. Pure-backend values
/// convert losslessly (their amplitudes are the same (re, im) pairs).
fn state_vector(val: &Value) -> Result<Rc<Statevector>, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            if let Some(c) = b.downcast_ref::<Circuit>() {
                c.execute_shared()
            } else if let Some(sv) = b.downcast_ref::<Statevector>() {
                Ok(Rc::new(sv.clone()))
            } else if let Some(c) = b.downcast_ref::<PureCircuit>() {
                pure_to_statevector(&c.execute()).map(Rc::new)
            } else if let Some(sv) = b.downcast_ref::<crate::pure::PureStatevector>() {
                pure_to_statevector(sv).map(Rc::new)
            } else {
                Err("expected a quantum circuit or statevector".into())
            }
        }
        other => Err(format!(
            "expected a quantum circuit or statevector, got {}",
            other.type_name()
        )),
    }
}

fn pure_to_statevector(sv: &crate::pure::PureStatevector) -> Result<Statevector, String> {
    Statevector::from_amplitudes(
        sv.amplitudes
            .iter()
            .map(|&(re, im)| ComplexF64::new(re, im))
            .collect(),
    )
}

/// Pure-backend counterpart of [`state_vector`].
fn pure_state_vector(val: &Value) -> Result<crate::pure::PureStatevector, String> {
    if let Value::QuantumState(rc) = val {
        let b = rc.borrow();
        if let Some(sv) = b.downcast_ref::<crate::pure::PureStatevector>() {
            return Ok(sv.clone());
        }
        if let Some(c) = b.downcast_ref::<PureCircuit>() {
            return Ok(c.execute());
        }
    }
    let sv = state_vector(val)?;
    Ok(crate::pure::PureStatevector {
        n_qubits: sv.n_qubits,
        amplitudes: sv.amplitudes.iter().map(|a| (a.re, a.im)).collect(),
    })
}

/// A Pauli string such as "XZIY": one character from {I, X, Y, Z} per qubit,
/// character k acting on qubit k.
fn extract_pauli_string(
    args: &[Value],
    idx: usize,
    n_qubits: usize,
    what: &str,
) -> Result<Vec<crate::fermion::Pauli>, String> {
    use crate::fermion::Pauli;
    let s = match args.get(idx) {
        Some(Value::String(s)) => s,
        Some(other) => {
            return Err(format!(
                "{}: Pauli string must be a string, got {}",
                what,
                other.type_name()
            ))
        }
        None => return Err(format!("{}: missing Pauli string", what)),
    };
    let ops = s
        .chars()
        .map(|c| match c {
            'I' => Ok(Pauli::I),
            'X' => Ok(Pauli::X),
            'Y' => Ok(Pauli::Y),
            'Z' => Ok(Pauli::Z),
            other => Err(format!(
                "{}: invalid Pauli character '{}' in \"{}\" (expected I, X, Y, or Z)",
                what, other, s
            )),
        })
        .collect::<Result<Vec<Pauli>, String>>()?;
    if ops.len() != n_qubits {
        return Err(format!(
            "{}: Pauli string \"{}\" has {} characters, expected one per qubit ({})",
            what,
            s,
            ops.len(),
            n_qubits
        ));
    }
    Ok(ops)
}

/// `q_copy`: an independent deep copy of any quantum value (ADR-0044). This is
/// how a program forks a simulator state (MPS, stabilizer, density), which are
/// mutable handles.
fn deep_copy(val: &Value) -> Result<Value, String> {
    let b = match val {
        Value::QuantumState(rc) => rc.borrow(),
        other => {
            return Err(format!(
                "q_copy expects a quantum value, got {}",
                other.type_name()
            ))
        }
    };
    macro_rules! try_copy {
        ($($t:ty),* $(,)?) => {
            $(if let Some(x) = b.downcast_ref::<$t>() {
                return Ok(wrap_any(x.clone()));
            })*
        };
    }
    try_copy!(
        Circuit,
        Statevector,
        Mps,
        StabilizerState,
        DensityMatrix,
        crate::qaoa::Graph,
        crate::qec::SurfaceCode,
        crate::fermion::FermionicHamiltonian,
        PureCircuit,
        crate::pure::PureStatevector,
        PureMps,
        PureStabilizer,
        PureDensity,
        crate::pure::PureFermionicHamiltonian,
    );
    Err("q_copy: unsupported quantum value".into())
}

/// Borrow the circuit immutably from a QuantumState value.
fn with_circuit<T>(
    val: &Value,
    f: impl FnOnce(&Circuit) -> Result<T, String>,
) -> Result<T, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let circ = borrow
                .downcast_ref::<Circuit>()
                .ok_or_else(|| not_a_circuit(&*borrow))?;
            f(circ)
        }
        _ => Err(format!("expected QuantumState, got {}", val.type_name())),
    }
}

/// A new circuit equal to `val` plus `gate` (ADR-0044: circuits are values).
/// The argument is never modified, so other bindings of the same circuit keep
/// their gates. Cost: one copy of the gate list, O(gates).
fn circuit_with_gate(val: &Value, gate: Gate) -> Result<Value, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let circ = borrow
                .downcast_ref::<Circuit>()
                .ok_or_else(|| not_a_circuit(&*borrow))?;
            let mut next = circ.clone();
            next.add(gate);
            Ok(wrap_circuit(next))
        }
        _ => Err(format!("expected QuantumState, got {}", val.type_name())),
    }
}

/// Pure-backend counterpart of [`circuit_with_gate`].
fn pure_circuit_with_gate(val: &Value, gate: PureGate) -> Result<Value, String> {
    let mut next = pure_circuit_ref(val, |c| c.clone())?;
    next.add(gate);
    Ok(wrap_pure(next))
}

/// Apply a single-qubit gate: fn(circuit, qubit) -> circuit
fn apply_gate_1q(
    args: &[Value],
    make_gate: impl Fn(usize) -> Gate,
) -> Result<Option<Value>, String> {
    if args.len() != 2 {
        return Err(format!(
            "gate requires (circuit, qubit), got {} args",
            args.len()
        ));
    }
    let q = extract_qubit_index(&args[1], "qubit")?;
    Ok(Some(circuit_with_gate(&args[0], make_gate(q))?))
}

/// Apply a parameterized single-qubit gate: fn(circuit, qubit, angle) -> circuit
fn apply_gate_1q_param(
    args: &[Value],
    make_gate: impl Fn(usize, f64) -> Gate,
) -> Result<Option<Value>, String> {
    if args.len() != 3 {
        return Err(format!(
            "gate requires (circuit, qubit, angle), got {} args",
            args.len()
        ));
    }
    let q = extract_qubit_index(&args[1], "qubit")?;
    let theta = extract_f64(args, 2, "angle")?;
    Ok(Some(circuit_with_gate(&args[0], make_gate(q, theta))?))
}

/// Apply a two-qubit gate: fn(circuit, qubit_a, qubit_b) -> circuit
fn apply_gate_2q(
    args: &[Value],
    make_gate: impl Fn(usize, usize) -> Gate,
) -> Result<Option<Value>, String> {
    if args.len() != 3 {
        return Err(format!(
            "gate requires (circuit, qubit_a, qubit_b), got {} args",
            args.len()
        ));
    }
    let a = extract_qubit_index(&args[1], "qubit_a")?;
    let b = extract_qubit_index(&args[2], "qubit_b")?;
    Ok(Some(circuit_with_gate(&args[0], make_gate(a, b))?))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubits_constructor() {
        let result = dispatch_quantum("qubits", &[Value::Int(2)]).unwrap();
        assert!(result.is_some());
        let val = result.unwrap();
        assert_eq!(val.type_name(), "QuantumState");
    }

    #[test]
    fn test_qubits_out_of_range() {
        assert!(dispatch_quantum("qubits", &[Value::Int(0)]).is_err());
        assert!(dispatch_quantum("qubits", &[Value::Int(27)]).is_err());
    }

    #[test]
    fn test_gate_chain() {
        let circ = dispatch_quantum("qubits", &[Value::Int(2)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_cx", &[circ.clone(), Value::Int(0), Value::Int(1)])
            .unwrap()
            .unwrap();

        // Check n_gates
        let n = dispatch_quantum("q_n_gates", &[circ.clone()])
            .unwrap()
            .unwrap();
        match n {
            Value::Int(2) => {}
            other => panic!("Expected Int(2), got {}", other),
        }
    }

    #[test]
    fn test_q_probs_bell_state() {
        let circ = dispatch_quantum("qubits", &[Value::Int(2)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_cx", &[circ.clone(), Value::Int(0), Value::Int(1)])
            .unwrap()
            .unwrap();

        let probs = dispatch_quantum("q_probs", &[circ.clone()])
            .unwrap()
            .unwrap();
        if let Value::Array(arr) = probs {
            assert_eq!(arr.len(), 4);
            // Bell state: P(|00⟩) ≈ 0.5, P(|01⟩) ≈ 0, P(|10⟩) ≈ 0, P(|11⟩) ≈ 0.5
            if let (Value::Float(p00), Value::Float(p11)) = (&arr[0], &arr[3]) {
                assert!((p00 - 0.5).abs() < 1e-12);
                assert!((p11 - 0.5).abs() < 1e-12);
            }
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_q_measure_deterministic() {
        let circ = dispatch_quantum("qubits", &[Value::Int(2)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_cx", &[circ.clone(), Value::Int(0), Value::Int(1)])
            .unwrap()
            .unwrap();

        let r1 = dispatch_quantum("q_measure", &[circ.clone(), Value::Int(42)])
            .unwrap()
            .unwrap();
        let r2 = dispatch_quantum("q_measure", &[circ.clone(), Value::Int(42)])
            .unwrap()
            .unwrap();
        // Compare string representations since Value doesn't implement PartialEq
        assert_eq!(
            format!("{}", r1),
            format!("{}", r2),
            "Same seed must produce same measurement"
        );
    }

    #[test]
    fn test_q_sample_deterministic() {
        let circ = dispatch_quantum("qubits", &[Value::Int(1)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)])
            .unwrap()
            .unwrap();

        let s1 = dispatch_quantum("q_sample", &[circ.clone(), Value::Int(100), Value::Int(42)])
            .unwrap()
            .unwrap();
        let s2 = dispatch_quantum("q_sample", &[circ.clone(), Value::Int(100), Value::Int(42)])
            .unwrap()
            .unwrap();
        assert_eq!(
            format!("{}", s1),
            format!("{}", s2),
            "Same seed must produce same samples"
        );
    }

    #[test]
    fn test_q_amplitudes() {
        let circ = dispatch_quantum("qubits", &[Value::Int(1)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)])
            .unwrap()
            .unwrap();

        let amps = dispatch_quantum("q_amplitudes", &[circ.clone()])
            .unwrap()
            .unwrap();
        if let Value::Array(arr) = amps {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_unknown_returns_none() {
        let result = dispatch_quantum("not_a_quantum_fn", &[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_rotation_gates() {
        let circ = dispatch_quantum("qubits", &[Value::Int(1)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_rx", &[circ.clone(), Value::Int(0), Value::Float(1.57)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_ry", &[circ.clone(), Value::Int(0), Value::Float(1.57)])
            .unwrap()
            .unwrap();
        let circ = dispatch_quantum("q_rz", &[circ.clone(), Value::Int(0), Value::Float(1.57)])
            .unwrap()
            .unwrap();
        let n = dispatch_quantum("q_n_gates", &[circ]).unwrap().unwrap();
        match n {
            Value::Int(3) => {}
            other => panic!("Expected Int(3), got {}", other),
        }
    }
}
