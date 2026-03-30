// CJC v0.2 Beta — Quantum Integration Tests
//
// These tests verify quantum builtins work through the CJC interpreter
// and MIR executor pipelines (eval + mir-exec parity).

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn eval_program(src: &str) -> Result<cjc_runtime::value::Value, String> {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        return Err(format!("Parse errors: {}", diags.error_count()));
    }
    match cjc_eval::Interpreter::new(42).exec(&program) {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("{:?}", e)),
    }
}

fn mir_program(src: &str) -> Result<cjc_runtime::value::Value, String> {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        return Err(format!("Parse errors: {}", diags.error_count()));
    }
    match cjc_mir_exec::run_program_with_executor(&program, 42) {
        Ok((v, _)) => Ok(v),
        Err(e) => Err(format!("{:?}", e)),
    }
}

fn format_val(v: &cjc_runtime::value::Value) -> String {
    format!("{}", v)
}

// ---------------------------------------------------------------------------
// Integration: qubits constructor
// ---------------------------------------------------------------------------

#[test]
fn test_eval_qubits_constructor() {
    let src = "fn main() -> Any { let q = qubits(2); q_n_qubits(q) }";
    let result = eval_program(src).unwrap();
    assert_eq!(format_val(&result), "2");
}

#[test]
fn test_mir_qubits_constructor() {
    let src = "fn main() -> Any { let q = qubits(2); q_n_qubits(q) }";
    let result = mir_program(src).unwrap();
    assert_eq!(format_val(&result), "2");
}

// ---------------------------------------------------------------------------
// Integration: gate application + probabilities
// ---------------------------------------------------------------------------

#[test]
fn test_eval_bell_state_probs() {
    let src = "fn main() -> Any { let q = qubits(2); let q = q_h(q, 0); let q = q_cx(q, 0, 1); q_probs(q) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("0.5"), "Bell state should have 0.5 probability, got: {}", s);
}

#[test]
fn test_mir_bell_state_probs() {
    let src = "fn main() -> Any { let q = qubits(2); let q = q_h(q, 0); let q = q_cx(q, 0, 1); q_probs(q) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("0.5"), "Bell state should have 0.5 probability, got: {}", s);
}

// ---------------------------------------------------------------------------
// Integration: measurement
// ---------------------------------------------------------------------------

#[test]
fn test_eval_measurement() {
    let src = "fn main() -> Any { let q = qubits(1); let q = q_x(q, 0); q_measure(q, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("1"), "X|0⟩ should measure 1, got: {}", s);
}

#[test]
fn test_mir_measurement() {
    let src = "fn main() -> Any { let q = qubits(1); let q = q_x(q, 0); q_measure(q, 42) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("1"), "X|0⟩ should measure 1, got: {}", s);
}

// ---------------------------------------------------------------------------
// Integration: sampling
// ---------------------------------------------------------------------------

#[test]
fn test_eval_sampling() {
    let src = "fn main() -> Any { let q = qubits(1); let q = q_h(q, 0); q_sample(q, 10, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.starts_with("["), "Expected array, got: {}", s);
}

#[test]
fn test_mir_sampling() {
    let src = "fn main() -> Any { let q = qubits(1); let q = q_h(q, 0); q_sample(q, 10, 42) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.starts_with("["), "Expected array, got: {}", s);
}

// ---------------------------------------------------------------------------
// Parity: eval vs mir-exec produce identical results
// ---------------------------------------------------------------------------

#[test]
fn test_parity_qubits_n_gates() {
    let src = "fn main() -> Any { let q = qubits(3); let q = q_h(q, 0); let q = q_h(q, 1); let q = q_h(q, 2); let q = q_cx(q, 0, 1); let q = q_cz(q, 1, 2); q_n_gates(q) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for n_gates");
}

#[test]
fn test_parity_bell_probs() {
    let src = "fn main() -> Any { let q = qubits(2); let q = q_h(q, 0); let q = q_cx(q, 0, 1); q_probs(q) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for Bell probs");
}

#[test]
fn test_parity_measurement_determinism() {
    let src = "fn main() -> Any { let q = qubits(2); let q = q_h(q, 0); let q = q_cx(q, 0, 1); q_measure(q, 123) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for measurement");
}

#[test]
fn test_parity_sampling() {
    let src = "fn main() -> Any { let q = qubits(1); let q = q_h(q, 0); q_sample(q, 50, 999) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for sampling");
}

// ---------------------------------------------------------------------------
// Integration: GHZ state
// ---------------------------------------------------------------------------

#[test]
fn test_eval_ghz_3qubit() {
    let src = "fn main() -> Any { let q = qubits(3); let q = q_h(q, 0); let q = q_cx(q, 0, 1); let q = q_cx(q, 0, 2); q_probs(q) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("0.5"), "GHZ state probs, got: {}", s);
}

// ---------------------------------------------------------------------------
// Integration: rotation gates
// ---------------------------------------------------------------------------

#[test]
fn test_eval_rotation_gates() {
    let src = "fn main() -> Any { let q = qubits(1); let q = q_rx(q, 0, 3.14159265358979); q_probs(q) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("1"), "Rx(pi)|0> probs, got: {}", s);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_eval_qubit_out_of_range() {
    // Error surfaces when executing the circuit (q_probs), not when adding the gate
    let src = "fn main() -> Any { let q = qubits(2); let q = q_h(q, 5); q_probs(q) }";
    let result = eval_program(src);
    assert!(result.is_err(), "Should error on out-of-range qubit");
}
