//! Quantum dispatch — builtin function and method routing for quantum state.
//!
//! This module is the entry point for both cjc-eval and cjc-mir-exec to
//! handle quantum operations. It follows the same pattern as
//! `cjc_vizor::dispatch` and `cjc_data::tidy_dispatch`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cjc_runtime::complex::ComplexF64;
use cjc_runtime::value::Value;

use crate::circuit::Circuit;
use crate::gates::Gate;
use crate::statevector::Statevector;

// Extension imports
use crate::mps::Mps;
use crate::stabilizer::StabilizerState;
use crate::density::DensityMatrix;

// Pure backend imports
use crate::pure::{
    PureMps, PureStabilizer, PureDensity, PureCircuit, PureGate,
    has_pure_flag, is_pure, wrap_pure,
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
                return Err(format!("q_toffoli(circuit, ctrl1, ctrl2, target) requires 4 args, got {}", args.len()));
            }
            let c1 = extract_qubit_index(&args[1], "ctrl1")?;
            let c2 = extract_qubit_index(&args[2], "ctrl2")?;
            let tgt = extract_qubit_index(&args[3], "target")?;
            with_circuit_mut(&args[0], |circ| {
                circ.add(Gate::Toffoli(c1, c2, tgt));
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        // --- Execute circuit → get statevector ---
        "q_run" => {
            if args.is_empty() {
                return Err("q_run(circuit) requires a circuit argument".into());
            }
            let sv = with_circuit(&args[0], |circ| circ.execute())?;
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
            let (outcomes, _sv) = with_circuit(&args[0], |circ| {
                circ.execute_and_measure(&mut rng_state)
            })?;
            let result: Vec<Value> = outcomes.iter().map(|&b| Value::Int(b as i64)).collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Get probability distribution ---
        "q_probs" => {
            if args.is_empty() {
                return Err("q_probs(circuit) requires a circuit argument".into());
            }
            let sv = with_circuit(&args[0], |circ| circ.execute())?;
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
            let samples = with_circuit(&args[0], |circ| circ.sample(n_shots, &mut rng_state))?;
            let result: Vec<Value> = samples.iter().map(|&s| Value::Int(s as i64)).collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Get amplitudes as array of Complex ---
        "q_amplitudes" => {
            if args.is_empty() {
                return Err("q_amplitudes(circuit) requires a circuit argument".into());
            }
            let sv = with_circuit(&args[0], |circ| circ.execute())?;
            let result: Vec<Value> = sv.amplitudes.iter()
                .map(|&a| Value::Complex(a))
                .collect();
            Ok(Some(Value::Array(Rc::new(result))))
        }

        // --- Get number of qubits ---
        "q_n_qubits" => {
            if args.is_empty() {
                return Err("q_n_qubits(circuit) requires a circuit argument".into());
            }
            with_circuit(&args[0], |circ| Ok(Value::Int(circ.n_qubits() as i64)))
                .map(Some)
        }

        // --- Get number of gates ---
        "q_n_gates" => {
            if args.is_empty() {
                return Err("q_n_gates(circuit) requires a circuit argument".into());
            }
            with_circuit(&args[0], |circ| Ok(Value::Int(circ.n_gates() as i64)))
                .map(Some)
        }

        // =======================================================================
        // MPS (Matrix Product States) — 50+ qubit simulation
        // =======================================================================

        "mps_new" => {
            let n = extract_int(&args, 0, "n_qubits")?;
            let mps = if args.len() > 1 {
                let bond = extract_int(&args, 1, "max_bond")?;
                Mps::with_max_bond(n as usize, bond as usize)
            } else {
                Mps::new(n as usize)
            };
            Ok(Some(wrap_any(mps)))
        }

        "mps_h" | "mps_x" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let isq2 = 1.0 / 2.0f64.sqrt();
            let mat = if name == "mps_h" {
                [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
                 [ComplexF64::real(isq2), ComplexF64::real(-isq2)]]
            } else {
                [[ComplexF64::ZERO, ComplexF64::ONE],
                 [ComplexF64::ONE, ComplexF64::ZERO]]
            };
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_single_qubit(q, mat);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_ry" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let theta = extract_angle(&args[2], "theta")?;
            let c = ComplexF64::real((theta / 2.0).cos());
            let s = ComplexF64::real((theta / 2.0).sin());
            let mat = [[c, ComplexF64::real(-s.re)], [s, c]];
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_single_qubit(q, mat);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_cnot" => {
            let q1 = extract_int(&args, 1, "control")? as usize;
            let q2 = extract_int(&args, 2, "target")? as usize;
            with_any_mut::<Mps>(&args[0], "MPS", |mps| {
                mps.apply_cnot_adjacent(q1, q2);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "mps_z_expectation" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let z = read_f64::<Mps>(&args[0], "MPS", |mps| {
                crate::qml::mps_single_z_expectation(mps, q)
            })?;
            Ok(Some(Value::Float(z)))
        }

        "mps_energy" => {
            let ham = match args.get(1) {
                Some(Value::String(s)) if s.as_ref() == "heisenberg" =>
                    crate::vqe::Hamiltonian::Heisenberg,
                _ => crate::vqe::Hamiltonian::Ising,
            };
            let e = read_f64::<Mps>(&args[0], "MPS", |mps| {
                crate::vqe::mps_energy(mps, ham)
            })?;
            Ok(Some(Value::Float(e)))
        }

        "mps_memory" => {
            let mem = read_i64::<Mps>(&args[0], "MPS", |mps| {
                mps.memory_bytes() as i64
            })?;
            Ok(Some(Value::Int(mem)))
        }

        // =======================================================================
        // VQE — Variational Quantum Eigensolver
        // =======================================================================

        "vqe_heisenberg" => {
            let n = extract_int(&args, 0, "n_qubits")? as usize;
            let chi = extract_int(&args, 1, "max_bond")? as usize;
            let lr = extract_angle(&args[2], "learning_rate")?;
            let iters = extract_int(&args, 3, "iterations")? as usize;
            let seed = extract_int(&args, 4, "seed")? as u64;
            let result = crate::vqe::vqe_heisenberg_1d(n, chi, lr, iters, seed);
            Ok(Some(Value::Float(result.energy)))
        }

        "vqe_full_heisenberg" => {
            let n = extract_int(&args, 0, "n_qubits")? as usize;
            let chi = extract_int(&args, 1, "max_bond")? as usize;
            let lr = extract_angle(&args[2], "learning_rate")?;
            let iters = extract_int(&args, 3, "iterations")? as usize;
            let seed = extract_int(&args, 4, "seed")? as u64;
            let result = crate::vqe::vqe_full_heisenberg_1d(n, chi, lr, iters, seed);
            Ok(Some(Value::Float(result.energy)))
        }

        // =======================================================================
        // QAOA — Quantum Approximate Optimization
        // =======================================================================

        "qaoa_graph_cycle" => {
            let n = extract_int(&args, 0, "n_vertices")? as usize;
            let g = crate::qaoa::Graph::cycle(n);
            Ok(Some(wrap_any(g)))
        }

        "qaoa_maxcut" => {
            let max_bond = extract_int(&args, 1, "max_bond")? as usize;
            let layers = extract_int(&args, 2, "p_layers")? as usize;
            let lr = extract_angle(&args[3], "learning_rate")?;
            let iters = extract_int(&args, 4, "iterations")? as usize;
            let seed = extract_int(&args, 5, "seed")? as u64;
            let result = match &args[0] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let g = borrow.downcast_ref::<crate::qaoa::Graph>()
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
            let n = extract_int(&args, 0, "n_qubits")? as usize;
            Ok(Some(wrap_any(StabilizerState::new(n))))
        }

        "stabilizer_h" | "stabilizer_s" | "stabilizer_x" |
        "stabilizer_y" | "stabilizer_z" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
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
            let ctrl = extract_int(&args, 1, "control")? as usize;
            let tgt = extract_int(&args, 2, "target")? as usize;
            with_any_mut::<StabilizerState>(&args[0], "StabilizerState", |s| {
                s.cnot(ctrl, tgt);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_measure" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let seed = extract_int(&args, 2, "seed")? as u64;
            let mut rng = seed;
            let outcome = match &args[0] {
                Value::QuantumState(rc) => {
                    let mut borrow = rc.borrow_mut();
                    let s = borrow.downcast_mut::<StabilizerState>()
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
            let n = extract_int(&args, 0, "n_qubits")? as usize;
            Ok(Some(wrap_any(DensityMatrix::new(n))))
        }

        "density_gate" => {
            let gate_name = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err("density_gate: gate name must be a string".into()),
            };
            let q = extract_int(&args, 2, "qubit")? as usize;
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
            let ctrl = extract_int(&args, 1, "control")? as usize;
            let tgt = extract_int(&args, 2, "target")? as usize;
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_gate(&Gate::CNOT(ctrl, tgt));
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_depolarize" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let p = extract_angle(&args[2], "probability")?;
            let ch = crate::density::depolarizing_channel(p);
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_single_qubit_channel(q, &ch);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_dephase" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let p = extract_angle(&args[2], "probability")?;
            let ch = crate::density::dephasing_channel(p);
            with_any_mut::<DensityMatrix>(&args[0], "DensityMatrix", |dm| {
                dm.apply_single_qubit_channel(q, &ch);
                Ok(())
            })?;
            Ok(Some(args[0].clone()))
        }

        "density_amplitude_damp" => {
            let q = extract_int(&args, 1, "qubit")? as usize;
            let gamma = extract_angle(&args[2], "gamma")?;
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
            let e = read_f64::<DensityMatrix>(&args[0], "DensityMatrix", |dm| dm.von_neumann_entropy())?;
            Ok(Some(Value::Float(e)))
        }

        "density_probs" => {
            let probs = read_vec_f64::<DensityMatrix>(&args[0], "DensityMatrix", |dm| dm.probabilities())?;
            let arr: Vec<Value> = probs.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        // =======================================================================
        // DMRG — Density Matrix Renormalization Group
        // =======================================================================

        "dmrg_ising" => {
            let n = extract_int(&args, 0, "n_qubits")? as usize;
            let chi = extract_int(&args, 1, "max_bond")? as usize;
            let sweeps = extract_int(&args, 2, "sweeps")? as usize;
            let tol = extract_angle(&args[3], "tolerance")?;
            let result = crate::dmrg::dmrg_heisenberg_1d(n, chi, sweeps, tol);
            Ok(Some(Value::Float(result.energy)))
        }

        "dmrg_heisenberg" => {
            let n = extract_int(&args, 0, "n_qubits")? as usize;
            let chi = extract_int(&args, 1, "max_bond")? as usize;
            let sweeps = extract_int(&args, 2, "sweeps")? as usize;
            let tol = extract_angle(&args[3], "tolerance")?;
            let result = crate::dmrg::dmrg_full_heisenberg_1d(n, chi, sweeps, tol);
            Ok(Some(Value::Float(result.energy)))
        }

        // =======================================================================
        // QEC — Quantum Error Correction
        // =======================================================================

        "qec_repetition_code" => {
            let d = extract_int(&args, 0, "distance")? as usize;
            let code = crate::qec::build_repetition_code(d);
            Ok(Some(wrap_any(code)))
        }

        "qec_surface_code" => {
            let d = extract_int(&args, 0, "distance")? as usize;
            let code = crate::qec::build_surface_code(d);
            Ok(Some(wrap_any(code)))
        }

        "qec_syndrome" => {
            let seed = extract_int(&args, 2, "seed")? as u64;
            let mut rng = seed;
            // We need both a mutable StabilizerState and an immutable SurfaceCode.
            // Extract both from QuantumState wrappers, being careful with borrows.
            match (&args[0], &args[1]) {
                (Value::QuantumState(state_rc), Value::QuantumState(code_rc)) => {
                    let code_borrow = code_rc.borrow();
                    let code = code_borrow.downcast_ref::<crate::qec::SurfaceCode>()
                        .ok_or_else(|| "expected SurfaceCode".to_string())?;
                    let mut state_borrow = state_rc.borrow_mut();
                    let state = state_borrow.downcast_mut::<StabilizerState>()
                        .ok_or_else(|| "expected StabilizerState".to_string())?;
                    let syndrome = crate::qec::syndrome_extraction(state, code, &mut rng);
                    let arr: Vec<Value> = syndrome.into_iter().map(|b| Value::Int(b as i64)).collect();
                    Ok(Some(Value::Array(Rc::new(arr))))
                }
                _ => Err("qec_syndrome(state, code, seed): expected QuantumState args".into()),
            }
        }

        "qec_decode" => {
            let syndrome = match &args[0] {
                Value::Array(arr) => arr.iter().map(|v| match v {
                    Value::Int(i) => *i as u8,
                    _ => 0,
                }).collect::<Vec<u8>>(),
                _ => return Err("qec_decode: first arg must be syndrome array".into()),
            };
            let corrections = match &args[1] {
                Value::QuantumState(rc) => {
                    let borrow = rc.borrow();
                    let code = borrow.downcast_ref::<crate::qec::SurfaceCode>()
                        .ok_or_else(|| "expected SurfaceCode".to_string())?;
                    crate::qec::decode_repetition_code(&syndrome, code)
                }
                _ => return Err("qec_decode: second arg must be SurfaceCode".into()),
            };
            let arr: Vec<Value> = corrections.into_iter().map(|c| Value::Int(c as i64)).collect();
            Ok(Some(Value::Array(Rc::new(arr))))
        }

        "qec_logical_error_rate" => {
            let distance = extract_int(&args, 0, "distance")? as usize;
            let p = extract_angle(&args[1], "error_rate")?;
            let rounds = extract_int(&args, 2, "rounds")? as usize;
            let seed = extract_int(&args, 3, "seed")? as u64;
            let rate = crate::qec::estimate_logical_error_rate(distance, p, rounds, seed);
            Ok(Some(Value::Float(rate)))
        }

        // =======================================================================
        // QML — Quantum Machine Learning
        // =======================================================================

        "qml_train" => {
            let n_qubits = extract_int(&args, 0, "n_qubits")? as usize;
            let layers = extract_int(&args, 1, "layers")? as usize;
            let n_classes = extract_int(&args, 2, "n_classes")? as usize;
            let chi = extract_int(&args, 3, "max_bond")? as usize;
            let lr = extract_angle(&args[4], "learning_rate")?;
            let epochs = extract_int(&args, 5, "epochs")? as usize;
            let seed = extract_int(&args, 6, "seed")? as u64;

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

            // Build dataset from args[7] (samples array) and args[8] (labels array)
            let samples = match args.get(7) {
                Some(Value::Array(arr)) => {
                    arr.iter().map(|row| match row {
                        Value::Array(inner) => inner.iter().map(|v| match v {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f64,
                            _ => 0.0,
                        }).collect(),
                        _ => vec![],
                    }).collect::<Vec<Vec<f64>>>()
                }
                _ => return Err("qml_train: arg 7 must be samples array".into()),
            };
            let labels = match args.get(8) {
                Some(Value::Array(arr)) => {
                    arr.iter().map(|v| match v {
                        Value::Int(i) => *i as usize,
                        _ => 0,
                    }).collect::<Vec<usize>>()
                }
                _ => return Err("qml_train: arg 8 must be labels array".into()),
            };

            let dataset = crate::qml::QmlDataset { samples, labels, n_classes };
            let result = crate::qml::qml_train(&config, &dataset);

            let loss_arr: Vec<Value> = result.loss_history.into_iter().map(Value::Float).collect();
            let out = vec![
                Value::Float(result.final_accuracy),
                Value::Array(Rc::new(loss_arr)),
            ];
            Ok(Some(Value::Array(Rc::new(out))))
        }

        "qml_predict" => {
            let n_qubits = extract_int(&args, 0, "n_qubits")? as usize;
            let layers = extract_int(&args, 1, "layers")? as usize;
            let n_classes = extract_int(&args, 2, "n_classes")? as usize;
            let chi = extract_int(&args, 3, "max_bond")? as usize;

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

            let params = match args.get(4) {
                Some(Value::Array(arr)) => arr.iter().map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => 0.0,
                }).collect::<Vec<f64>>(),
                _ => return Err("qml_predict: arg 4 must be params array".into()),
            };
            let input = match args.get(5) {
                Some(Value::Array(arr)) => arr.iter().map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => 0.0,
                }).collect::<Vec<f64>>(),
                _ => return Err("qml_predict: arg 5 must be input array".into()),
            };

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

        _ => Ok(None),
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
            let n = extract_int(args, 0, "n_qubits")? as usize;
            if n < 1 || n > 26 {
                return Err(format!("qubits() requires 1-26 qubits, got {}", n));
            }
            Ok(Some(wrap_pure(PureCircuit::new(n))))
        }

        "mps_new" if has_pure_flag(args) => {
            let n = extract_int(args, 0, "n_qubits")? as usize;
            let chi = if args.len() > 2 { extract_int(args, 1, "max_bond")? as usize } else { 32 };
            Ok(Some(wrap_pure(PureMps::new(n, chi))))
        }

        "stabilizer_new" if has_pure_flag(args) => {
            let n = extract_int(args, 0, "n_qubits")? as usize;
            Ok(Some(wrap_pure(PureStabilizer::new(n))))
        }

        "density_new" if has_pure_flag(args) => {
            let n = extract_int(args, 0, "n_qubits")? as usize;
            Ok(Some(wrap_pure(PureDensity::new(n))))
        }

        // === Pure MPS operations (auto-detected) ===

        "mps_h" | "mps_x" if is_pure::<PureMps>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let mat = if name == "mps_h" { h_matrix() } else { x_matrix() };
            pure_mps_mut(&args[0], |mps| mps.apply_single_qubit(q, mat))?;
            Ok(Some(args[0].clone()))
        }

        "mps_ry" if is_pure::<PureMps>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let theta = extract_angle(&args[2], "theta")?;
            pure_mps_mut(&args[0], |mps| mps.apply_single_qubit(q, ry_matrix(theta)))?;
            Ok(Some(args[0].clone()))
        }

        "mps_cnot" if is_pure::<PureMps>(&args[0]) => {
            let ctrl = extract_int(args, 1, "control")? as usize;
            let targ = extract_int(args, 2, "target")? as usize;
            pure_mps_mut(&args[0], |mps| mps.apply_cnot(ctrl, targ))?;
            Ok(Some(args[0].clone()))
        }

        "mps_z_expectation" if is_pure::<PureMps>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let z = pure_mps_ref(&args[0], |mps| mps.z_expectation(q))?;
            Ok(Some(Value::Float(z)))
        }

        "mps_memory" if is_pure::<PureMps>(&args[0]) => {
            let mem = pure_mps_ref(&args[0], |mps| mps.memory_bytes() as f64)?;
            Ok(Some(Value::Int(mem as i64)))
        }

        // === Pure Stabilizer operations ===

        "stabilizer_h" | "stabilizer_s" | "stabilizer_x" |
        "stabilizer_y" | "stabilizer_z" if is_pure::<PureStabilizer>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            pure_stab_mut(&args[0], |s| {
                match name {
                    "stabilizer_h" => s.h(q),
                    "stabilizer_s" => s.s(q),
                    "stabilizer_x" => s.x(q),
                    "stabilizer_y" => s.y(q),
                    "stabilizer_z" => s.z(q),
                    _ => unreachable!(),
                }
            })?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_cnot" if is_pure::<PureStabilizer>(&args[0]) => {
            let ctrl = extract_int(args, 1, "control")? as usize;
            let tgt = extract_int(args, 2, "target")? as usize;
            pure_stab_mut(&args[0], |s| s.cnot(ctrl, tgt))?;
            Ok(Some(args[0].clone()))
        }

        "stabilizer_measure" if is_pure::<PureStabilizer>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let seed = extract_int(args, 2, "seed")? as u64;
            let outcome = match &args[0] {
                Value::QuantumState(rc) => {
                    let mut borrow = rc.borrow_mut();
                    let s = borrow.downcast_mut::<PureStabilizer>()
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
            let q = extract_int(args, 2, "qubit")? as usize;
            let mat = match gate_name.as_str() {
                "H" => h_matrix(), "X" => x_matrix(), "Y" => y_matrix(),
                "Z" => z_matrix(), "S" => s_matrix(), "T" => t_matrix(),
                _ => return Err(format!("density_gate: unknown gate '{}'", gate_name)),
            };
            pure_density_mut(&args[0], |dm| dm.apply_gate_2x2(q, mat))?;
            Ok(Some(args[0].clone()))
        }

        "density_cnot" if is_pure::<PureDensity>(&args[0]) => {
            let ctrl = extract_int(args, 1, "control")? as usize;
            let tgt = extract_int(args, 2, "target")? as usize;
            pure_density_mut(&args[0], |dm| dm.apply_cnot(ctrl, tgt))?;
            Ok(Some(args[0].clone()))
        }

        "density_depolarize" if is_pure::<PureDensity>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let p = extract_angle(&args[2], "probability")?;
            pure_density_mut(&args[0], |dm| dm.apply_depolarize(q, p))?;
            Ok(Some(args[0].clone()))
        }

        "density_dephase" if is_pure::<PureDensity>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let p = extract_angle(&args[2], "probability")?;
            pure_density_mut(&args[0], |dm| dm.apply_dephase(q, p))?;
            Ok(Some(args[0].clone()))
        }

        "density_amplitude_damp" if is_pure::<PureDensity>(&args[0]) => {
            let q = extract_int(args, 1, "qubit")? as usize;
            let gamma = extract_angle(&args[2], "gamma")?;
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
                "q_h" => PureGate::H(q), "q_x" => PureGate::X(q),
                "q_y" => PureGate::Y(q), "q_z" => PureGate::Z(q),
                "q_s" => PureGate::S(q), "q_t" => PureGate::T(q),
                _ => unreachable!(),
            };
            pure_circuit_mut(&args[0], |c| c.add(gate))?;
            Ok(Some(args[0].clone()))
        }

        "q_rx" | "q_ry" | "q_rz" if is_pure::<PureCircuit>(&args[0]) => {
            let q = extract_qubit_index(&args[1], "qubit")?;
            let theta = extract_angle(&args[2], "angle")?;
            let gate = match name {
                "q_rx" => PureGate::Rx(q, theta),
                "q_ry" => PureGate::Ry(q, theta),
                "q_rz" => PureGate::Rz(q, theta),
                _ => unreachable!(),
            };
            pure_circuit_mut(&args[0], |c| c.add(gate))?;
            Ok(Some(args[0].clone()))
        }

        "q_cx" | "q_cnot" if is_pure::<PureCircuit>(&args[0]) => {
            let a = extract_qubit_index(&args[1], "a")?;
            let b = extract_qubit_index(&args[2], "b")?;
            pure_circuit_mut(&args[0], |c| c.add(PureGate::CNOT(a, b)))?;
            Ok(Some(args[0].clone()))
        }

        "q_cz" if is_pure::<PureCircuit>(&args[0]) => {
            let a = extract_qubit_index(&args[1], "a")?;
            let b = extract_qubit_index(&args[2], "b")?;
            pure_circuit_mut(&args[0], |c| c.add(PureGate::CZ(a, b)))?;
            Ok(Some(args[0].clone()))
        }

        "q_swap" if is_pure::<PureCircuit>(&args[0]) => {
            let a = extract_qubit_index(&args[1], "a")?;
            let b = extract_qubit_index(&args[2], "b")?;
            pure_circuit_mut(&args[0], |c| c.add(PureGate::SWAP(a, b)))?;
            Ok(Some(args[0].clone()))
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
            let s = b.downcast_mut::<PureStabilizer>().ok_or("expected PureStabilizer")?;
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
            let s = b.downcast_ref::<PureStabilizer>().ok_or("expected PureStabilizer")?;
            Ok(f(s))
        }
        _ => Err("expected PureStabilizer".into()),
    }
}

fn pure_density_mut(val: &Value, f: impl FnOnce(&mut PureDensity)) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut b = rc.borrow_mut();
            let dm = b.downcast_mut::<PureDensity>().ok_or("expected PureDensity")?;
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
            let dm = b.downcast_ref::<PureDensity>().ok_or("expected PureDensity")?;
            Ok(f(dm))
        }
        _ => Err("expected PureDensity".into()),
    }
}

fn pure_density_ref_vec(val: &Value, f: impl FnOnce(&PureDensity) -> Vec<f64>) -> Result<Vec<f64>, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let dm = b.downcast_ref::<PureDensity>().ok_or("expected PureDensity")?;
            Ok(f(dm))
        }
        _ => Err("expected PureDensity".into()),
    }
}

fn pure_circuit_mut(val: &Value, f: impl FnOnce(&mut PureCircuit)) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut b = rc.borrow_mut();
            let c = b.downcast_mut::<PureCircuit>().ok_or("expected PureCircuit")?;
            f(c);
            Ok(())
        }
        _ => Err("expected PureCircuit".into()),
    }
}

fn pure_circuit_ref<T>(val: &Value, f: impl FnOnce(&PureCircuit) -> T) -> Result<T, String> {
    match val {
        Value::QuantumState(rc) => {
            let b = rc.borrow();
            let c = b.downcast_ref::<PureCircuit>().ok_or("expected PureCircuit")?;
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
fn read_f64<T: Any + 'static>(val: &Value, type_name: &str, f: impl FnOnce(&T) -> f64) -> Result<f64, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let obj = borrow.downcast_ref::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            Ok(f(obj))
        }
        _ => Err(format!("expected QuantumState({}), got {}", type_name, val.type_name())),
    }
}

/// Borrow a typed value immutably and extract an i64.
fn read_i64<T: Any + 'static>(val: &Value, type_name: &str, f: impl FnOnce(&T) -> i64) -> Result<i64, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let obj = borrow.downcast_ref::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            Ok(f(obj))
        }
        _ => Err(format!("expected QuantumState({}), got {}", type_name, val.type_name())),
    }
}

/// Borrow a typed value immutably and extract a Vec<f64>.
fn read_vec_f64<T: Any + 'static>(val: &Value, type_name: &str, f: impl FnOnce(&T) -> Vec<f64>) -> Result<Vec<f64>, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let obj = borrow.downcast_ref::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            Ok(f(obj))
        }
        _ => Err(format!("expected QuantumState({}), got {}", type_name, val.type_name())),
    }
}

/// Borrow a typed value mutably from a QuantumState.
fn with_any_mut<T: Any + 'static>(val: &Value, type_name: &str, f: impl FnOnce(&mut T) -> Result<(), String>) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut borrow = rc.borrow_mut();
            let obj = borrow.downcast_mut::<T>()
                .ok_or_else(|| format!("expected {}", type_name))?;
            f(obj)
        }
        _ => Err(format!("expected QuantumState({}), got {}", type_name, val.type_name())),
    }
}

fn extract_int(args: &[Value], idx: usize, name: &str) -> Result<i64, String> {
    match args.get(idx) {
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Float(f)) => Ok(*f as i64),
        Some(other) => Err(format!("{} must be an integer, got {}", name, other.type_name())),
        None => Err(format!("missing argument: {}", name)),
    }
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

fn extract_angle(val: &Value, name: &str) -> Result<f64, String> {
    match val {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("{} must be a number", name)),
    }
}

/// Borrow the circuit immutably from a QuantumState value.
fn with_circuit<T>(val: &Value, f: impl FnOnce(&Circuit) -> Result<T, String>) -> Result<T, String> {
    match val {
        Value::QuantumState(rc) => {
            let borrow = rc.borrow();
            let circ = borrow.downcast_ref::<Circuit>()
                .ok_or_else(|| "expected a quantum circuit".to_string())?;
            f(circ)
        }
        _ => Err(format!("expected QuantumState, got {}", val.type_name())),
    }
}

/// Borrow the circuit mutably from a QuantumState value.
fn with_circuit_mut(val: &Value, f: impl FnOnce(&mut Circuit) -> Result<(), String>) -> Result<(), String> {
    match val {
        Value::QuantumState(rc) => {
            let mut borrow = rc.borrow_mut();
            let circ = borrow.downcast_mut::<Circuit>()
                .ok_or_else(|| "expected a quantum circuit".to_string())?;
            f(circ)
        }
        _ => Err(format!("expected QuantumState, got {}", val.type_name())),
    }
}

/// Apply a single-qubit gate: fn(circuit, qubit) -> circuit
fn apply_gate_1q(args: &[Value], make_gate: impl Fn(usize) -> Gate) -> Result<Option<Value>, String> {
    if args.len() != 2 {
        return Err(format!("gate requires (circuit, qubit), got {} args", args.len()));
    }
    let q = extract_qubit_index(&args[1], "qubit")?;
    with_circuit_mut(&args[0], |circ| {
        circ.add(make_gate(q));
        Ok(())
    })?;
    Ok(Some(args[0].clone()))
}

/// Apply a parameterized single-qubit gate: fn(circuit, qubit, angle) -> circuit
fn apply_gate_1q_param(args: &[Value], make_gate: impl Fn(usize, f64) -> Gate) -> Result<Option<Value>, String> {
    if args.len() != 3 {
        return Err(format!("gate requires (circuit, qubit, angle), got {} args", args.len()));
    }
    let q = extract_qubit_index(&args[1], "qubit")?;
    let theta = extract_angle(&args[2], "angle")?;
    with_circuit_mut(&args[0], |circ| {
        circ.add(make_gate(q, theta));
        Ok(())
    })?;
    Ok(Some(args[0].clone()))
}

/// Apply a two-qubit gate: fn(circuit, qubit_a, qubit_b) -> circuit
fn apply_gate_2q(args: &[Value], make_gate: impl Fn(usize, usize) -> Gate) -> Result<Option<Value>, String> {
    if args.len() != 3 {
        return Err(format!("gate requires (circuit, qubit_a, qubit_b), got {} args", args.len()));
    }
    let a = extract_qubit_index(&args[1], "qubit_a")?;
    let b = extract_qubit_index(&args[2], "qubit_b")?;
    with_circuit_mut(&args[0], |circ| {
        circ.add(make_gate(a, b));
        Ok(())
    })?;
    Ok(Some(args[0].clone()))
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
        let circ = dispatch_quantum("qubits", &[Value::Int(2)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_cx", &[circ.clone(), Value::Int(0), Value::Int(1)]).unwrap().unwrap();

        // Check n_gates
        let n = dispatch_quantum("q_n_gates", &[circ.clone()]).unwrap().unwrap();
        match n {
            Value::Int(2) => {}
            other => panic!("Expected Int(2), got {}", other),
        }
    }

    #[test]
    fn test_q_probs_bell_state() {
        let circ = dispatch_quantum("qubits", &[Value::Int(2)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_cx", &[circ.clone(), Value::Int(0), Value::Int(1)]).unwrap().unwrap();

        let probs = dispatch_quantum("q_probs", &[circ.clone()]).unwrap().unwrap();
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
        let circ = dispatch_quantum("qubits", &[Value::Int(2)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_cx", &[circ.clone(), Value::Int(0), Value::Int(1)]).unwrap().unwrap();

        let r1 = dispatch_quantum("q_measure", &[circ.clone(), Value::Int(42)]).unwrap().unwrap();
        let r2 = dispatch_quantum("q_measure", &[circ.clone(), Value::Int(42)]).unwrap().unwrap();
        // Compare string representations since Value doesn't implement PartialEq
        assert_eq!(format!("{}", r1), format!("{}", r2), "Same seed must produce same measurement");
    }

    #[test]
    fn test_q_sample_deterministic() {
        let circ = dispatch_quantum("qubits", &[Value::Int(1)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)]).unwrap().unwrap();

        let s1 = dispatch_quantum("q_sample", &[circ.clone(), Value::Int(100), Value::Int(42)]).unwrap().unwrap();
        let s2 = dispatch_quantum("q_sample", &[circ.clone(), Value::Int(100), Value::Int(42)]).unwrap().unwrap();
        assert_eq!(format!("{}", s1), format!("{}", s2), "Same seed must produce same samples");
    }

    #[test]
    fn test_q_amplitudes() {
        let circ = dispatch_quantum("qubits", &[Value::Int(1)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_h", &[circ.clone(), Value::Int(0)]).unwrap().unwrap();

        let amps = dispatch_quantum("q_amplitudes", &[circ.clone()]).unwrap().unwrap();
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
        let circ = dispatch_quantum("qubits", &[Value::Int(1)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_rx", &[circ.clone(), Value::Int(0), Value::Float(1.57)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_ry", &[circ.clone(), Value::Int(0), Value::Float(1.57)]).unwrap().unwrap();
        let circ = dispatch_quantum("q_rz", &[circ.clone(), Value::Int(0), Value::Float(1.57)]).unwrap().unwrap();
        let n = dispatch_quantum("q_n_gates", &[circ]).unwrap().unwrap();
        match n {
            Value::Int(3) => {}
            other => panic!("Expected Int(3), got {}", other),
        }
    }
}
