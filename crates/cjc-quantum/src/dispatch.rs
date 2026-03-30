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

/// Dispatch a quantum builtin function call by name.
///
/// Returns `Ok(Some(value))` if handled, `Ok(None)` if not a quantum builtin.
pub fn dispatch_quantum(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
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

        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn wrap_circuit(circ: Circuit) -> Value {
    Value::QuantumState(Rc::new(RefCell::new(circ)))
}

fn wrap_statevector(sv: Statevector) -> Value {
    Value::QuantumState(Rc::new(RefCell::new(sv)))
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
