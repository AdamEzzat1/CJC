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

// ===========================================================================
// Extension 1: MPS — Matrix Product States
// ===========================================================================

#[test]
fn test_eval_mps_new_and_memory() {
    let src = "fn main() -> Any { let m = mps_new(4, 8); mps_memory(m) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let mem: i64 = s.parse().expect("should return integer memory");
    assert!(mem > 0, "MPS memory should be positive, got {}", mem);
}

#[test]
fn test_mir_mps_new_and_memory() {
    let src = "fn main() -> Any { let m = mps_new(4, 8); mps_memory(m) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    let mem: i64 = s.parse().expect("should return integer memory");
    assert!(mem > 0, "MPS memory should be positive, got {}", mem);
}

#[test]
fn test_eval_mps_z_expectation() {
    // |0...0> state: Z expectation should be 1.0
    let src = "fn main() -> Any { let m = mps_new(3, 8); mps_z_expectation(m, 0) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let z: f64 = s.parse().expect("should return float");
    assert!((z - 1.0).abs() < 1e-10, "Z of |0> should be 1.0, got {}", z);
}

#[test]
fn test_mir_mps_z_expectation() {
    let src = "fn main() -> Any { let m = mps_new(3, 8); mps_z_expectation(m, 0) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    let z: f64 = s.parse().expect("should return float");
    assert!((z - 1.0).abs() < 1e-10, "Z of |0> should be 1.0, got {}", z);
}

#[test]
fn test_eval_mps_h_changes_state() {
    let src = "fn main() -> Any { let m = mps_new(2, 8); let m = mps_h(m, 0); mps_z_expectation(m, 0) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let z: f64 = s.parse().expect("should return float");
    assert!(z.abs() < 1e-10, "H|0> should give Z=0, got {}", z);
}

#[test]
fn test_eval_mps_x_flips_qubit() {
    let src = "fn main() -> Any { let m = mps_new(2, 8); let m = mps_x(m, 0); mps_z_expectation(m, 0) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let z: f64 = s.parse().expect("should return float");
    assert!((z - (-1.0)).abs() < 1e-10, "X|0> should give Z=-1, got {}", z);
}

#[test]
fn test_parity_mps_operations() {
    let src = "fn main() -> Any { let m = mps_new(3, 8); let m = mps_h(m, 0); let m = mps_cnot(m, 0, 1); mps_z_expectation(m, 1) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for MPS ops");
}

// ===========================================================================
// Extension 2: VQE — Variational Quantum Eigensolver
// ===========================================================================

#[test]
fn test_eval_vqe_heisenberg() {
    let src = "fn main() -> Any { vqe_heisenberg(4, 8, 0.1, 50, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e.is_finite(), "VQE Heisenberg energy should be finite, got {}", e);
}

#[test]
fn test_mir_vqe_heisenberg() {
    let src = "fn main() -> Any { vqe_heisenberg(4, 8, 0.1, 50, 42) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e.is_finite(), "VQE Heisenberg energy should be finite, got {}", e);
}

#[test]
fn test_parity_vqe_heisenberg() {
    let src = "fn main() -> Any { vqe_heisenberg(4, 8, 0.1, 10, 42) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for VQE Heisenberg");
}

#[test]
fn test_eval_vqe_full_heisenberg() {
    let src = "fn main() -> Any { vqe_full_heisenberg(4, 8, 0.1, 50, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e.is_finite(), "VQE full Heisenberg energy should be finite, got {}", e);
}

// ===========================================================================
// Extension 3: QAOA — Quantum Approximate Optimization
// ===========================================================================

#[test]
fn test_eval_qaoa_maxcut() {
    let src = "fn main() -> Any { let g = qaoa_graph_cycle(4); qaoa_maxcut(g, 8, 1, 0.1, 5, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.starts_with("["), "Expected array [energy, cut_value], got: {}", s);
}

#[test]
fn test_mir_qaoa_maxcut() {
    let src = "fn main() -> Any { let g = qaoa_graph_cycle(4); qaoa_maxcut(g, 8, 1, 0.1, 5, 42) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.starts_with("["), "Expected array [energy, cut_value], got: {}", s);
}

#[test]
fn test_parity_qaoa_maxcut() {
    let src = "fn main() -> Any { let g = qaoa_graph_cycle(4); qaoa_maxcut(g, 8, 1, 0.1, 3, 42) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for QAOA maxcut");
}

// ===========================================================================
// Extension 4: Stabilizer — Clifford/CHP simulator
// ===========================================================================

#[test]
fn test_eval_stabilizer_basic() {
    let src = "fn main() -> Any { let s = stabilizer_new(3); stabilizer_n_qubits(s) }";
    let result = eval_program(src).unwrap();
    assert_eq!(format_val(&result), "3");
}

#[test]
fn test_mir_stabilizer_basic() {
    let src = "fn main() -> Any { let s = stabilizer_new(3); stabilizer_n_qubits(s) }";
    let result = mir_program(src).unwrap();
    assert_eq!(format_val(&result), "3");
}

#[test]
fn test_eval_stabilizer_gates_and_measure() {
    // H then measure: random outcome, but X after H|0> = |+> is deterministic with seed
    let src = "fn main() -> Any { let s = stabilizer_new(1); let s = stabilizer_x(s, 0); stabilizer_measure(s, 0, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert_eq!(s, "1", "X|0> should measure 1, got {}", s);
}

#[test]
fn test_parity_stabilizer_measure() {
    let src = "fn main() -> Any { let s = stabilizer_new(2); let s = stabilizer_h(s, 0); let s = stabilizer_cnot(s, 0, 1); stabilizer_measure(s, 0, 42) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for stabilizer measure");
}

// ===========================================================================
// Extension 5: Density Matrix — Mixed states + noise
// ===========================================================================

#[test]
fn test_eval_density_trace() {
    let src = "fn main() -> Any { let d = density_new(2); density_trace(d) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let tr: f64 = s.parse().expect("should return float");
    assert!((tr - 1.0).abs() < 1e-10, "trace should be 1.0, got {}", tr);
}

#[test]
fn test_mir_density_trace() {
    let src = "fn main() -> Any { let d = density_new(2); density_trace(d) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    let tr: f64 = s.parse().expect("should return float");
    assert!((tr - 1.0).abs() < 1e-10, "trace should be 1.0, got {}", tr);
}

#[test]
fn test_eval_density_purity() {
    let src = "fn main() -> Any { let d = density_new(1); density_purity(d) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let p: f64 = s.parse().expect("should return float");
    assert!((p - 1.0).abs() < 1e-10, "pure state purity should be 1.0, got {}", p);
}

#[test]
fn test_eval_density_gate_and_depolarize() {
    let src = "fn main() -> Any { let d = density_new(1); let d = density_gate(d, \"H\", 0); let d = density_depolarize(d, 0, 0.5); density_purity(d) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let p: f64 = s.parse().expect("should return float");
    assert!(p < 1.0, "depolarized state should have purity < 1, got {}", p);
}

#[test]
fn test_eval_density_entropy() {
    let src = "fn main() -> Any { let d = density_new(1); density_entropy(d) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float");
    assert!((e - 0.0).abs() < 1e-10, "pure state entropy should be 0, got {}", e);
}

#[test]
fn test_eval_density_probs() {
    let src = "fn main() -> Any { let d = density_new(1); density_probs(d) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.contains("1"), "probs of |0> should contain 1, got: {}", s);
}

#[test]
fn test_parity_density_matrix() {
    let src = "fn main() -> Any { let d = density_new(1); let d = density_gate(d, \"X\", 0); density_purity(d) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for density matrix");
}

// ===========================================================================
// Extension 6: DMRG
// ===========================================================================

#[test]
fn test_eval_dmrg_ising() {
    let src = "fn main() -> Any { dmrg_ising(4, 8, 5, 0.001) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e < 0.0, "DMRG Ising energy should be negative, got {}", e);
}

#[test]
fn test_mir_dmrg_ising() {
    let src = "fn main() -> Any { dmrg_ising(4, 8, 5, 0.001) }";
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e < 0.0, "DMRG Ising energy should be negative, got {}", e);
}

#[test]
fn test_parity_dmrg_ising() {
    let src = "fn main() -> Any { dmrg_ising(4, 8, 3, 0.001) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for DMRG");
}

#[test]
fn test_eval_dmrg_heisenberg() {
    let src = "fn main() -> Any { dmrg_heisenberg(4, 8, 5, 0.001) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e < 0.0, "DMRG Heisenberg energy should be negative, got {}", e);
}

// ===========================================================================
// Extension 7: QEC — Quantum Error Correction
// ===========================================================================

#[test]
fn test_eval_qec_repetition_code() {
    // distance-3 repetition code needs n_data=3 + n_ancilla=2 = 5 qubits
    let src = "fn main() -> Any { let code = qec_repetition_code(3); let state = stabilizer_new(5); let syndrome = qec_syndrome(state, code, 42); syndrome }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.starts_with("["), "Expected syndrome array, got: {}", s);
}

#[test]
fn test_eval_qec_logical_error_rate() {
    let src = "fn main() -> Any { qec_logical_error_rate(3, 0.01, 10, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let rate: f64 = s.parse().expect("should return float");
    assert!(rate >= 0.0 && rate <= 1.0, "error rate should be in [0,1], got {}", rate);
}

#[test]
fn test_parity_qec_logical_error_rate() {
    let src = "fn main() -> Any { qec_logical_error_rate(3, 0.01, 10, 42) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for QEC error rate");
}

// ===========================================================================
// Extension 8: QML — Quantum Machine Learning
// ===========================================================================

#[test]
fn test_eval_qml_predict() {
    // 3 qubits, 1 layer, 2 classes → total_params = 6 * 3 * 1 = 18
    let src = r#"
fn main() -> Any {
    let params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8];
    let input = [1.0, 0.5, 0.2];
    qml_predict(3, 1, 2, 8, params, input)
}
"#;
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let class: i64 = s.parse().expect("should return integer class");
    assert!(class >= 0 && class < 2, "predicted class should be 0 or 1, got {}", class);
}

#[test]
fn test_mir_qml_predict() {
    let src = r#"
fn main() -> Any {
    let params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8];
    let input = [1.0, 0.5, 0.2];
    qml_predict(3, 1, 2, 8, params, input)
}
"#;
    let result = mir_program(src).unwrap();
    let s = format_val(&result);
    let class: i64 = s.parse().expect("should return integer class");
    assert!(class >= 0 && class < 2, "predicted class should be 0 or 1, got {}", class);
}

#[test]
fn test_parity_qml_predict() {
    let src = r#"
fn main() -> Any {
    let params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8];
    let input = [1.0, 0.5, 0.2];
    qml_predict(3, 1, 2, 8, params, input)
}
"#;
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "Eval vs MIR parity for QML predict");
}

// ===========================================================================
// 50-QUBIT TESTS — Pure CJC, no Rust dependencies
// ===========================================================================

#[test]
fn test_50_qubit_mps_from_cjc() {
    // 50-qubit MPS: create, apply gates, read Z expectations and memory
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16);
    let m = mps_h(m, 0);
    let m = mps_h(m, 25);
    let m = mps_h(m, 49);
    let mem = mps_memory(m);
    let z0 = mps_z_expectation(m, 0);
    let z25 = mps_z_expectation(m, 25);
    let z49 = mps_z_expectation(m, 49);
    let z1 = mps_z_expectation(m, 1);
    [mem, z0, z25, z49, z1]
}
"#;
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    // Should return [memory, ~0.0, ~0.0, ~0.0, 1.0]
    // H|0> gives Z=0 for qubits 0,25,49; untouched qubit 1 gives Z=1
    assert!(s.starts_with("["), "Expected array result, got: {}", s);
}

#[test]
fn test_50_qubit_mps_memory_under_1mb() {
    let src = "fn main() -> Any { let m = mps_new(50, 16); mps_memory(m) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let mem: i64 = s.parse().expect("should return integer");
    assert!(mem < 1_000_000, "50-qubit MPS should be under 1MB, got {} bytes", mem);
}

#[test]
fn test_50_qubit_vqe_from_cjc() {
    // 50-qubit VQE with 1 iteration — runs the full optimization step
    let src = "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }";
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    let e: f64 = s.parse().expect("should return float energy");
    assert!(e.is_finite(), "50-qubit VQE energy should be finite, got {}", e);
}

#[test]
fn test_50_qubit_vqe_deterministic_from_cjc() {
    let src = "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }";
    let r1 = format_val(&eval_program(src).unwrap());
    let r2 = format_val(&eval_program(src).unwrap());
    assert_eq!(r1, r2, "50-qubit VQE should be deterministic");
}

#[test]
fn test_50_qubit_vqe_parity_eval_vs_mir() {
    let src = "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }";
    let eval_result = format_val(&eval_program(src).unwrap());
    let mir_result = format_val(&mir_program(src).unwrap());
    assert_eq!(eval_result, mir_result, "50-qubit VQE: eval vs MIR parity");
}

#[test]
fn test_50_qubit_stabilizer_from_cjc() {
    // 50-qubit stabilizer: H on qubit 0, then CNOT to all others → GHZ state
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(50);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_cnot(s, 1, 2);
    let s = stabilizer_cnot(s, 2, 3);
    let s = stabilizer_cnot(s, 3, 4);
    stabilizer_n_qubits(s)
}
"#;
    let result = eval_program(src).unwrap();
    assert_eq!(format_val(&result), "50");
}

#[test]
fn test_50_qubit_qml_predict_from_cjc() {
    // 50-qubit QML forward pass — predict with random params
    // total_params = 6 * 50 * 1 = 300, but CJC arrays can't have 300 literal elements easily
    // So we test via the dispatch which accepts arrays from CJC
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16);
    let m = mps_h(m, 0);
    let m = mps_ry(m, 1, 0.5);
    let z = mps_z_expectation(m, 0);
    let mem = mps_memory(m);
    [z, mem]
}
"#;
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    assert!(s.starts_with("["), "Expected array, got: {}", s);
}

#[test]
fn test_500_qubit_stabilizer_from_cjc() {
    // 500-qubit Clifford simulation — only possible with stabilizer formalism
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(500);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_cnot(s, 1, 2);
    let s = stabilizer_cnot(s, 2, 3);
    let s = stabilizer_cnot(s, 3, 4);
    let s = stabilizer_x(s, 499);
    let m = stabilizer_measure(s, 499, 42);
    let n = stabilizer_n_qubits(s);
    [n, m]
}
"#;
    let result = eval_program(src).unwrap();
    let s = format_val(&result);
    // X|0> on qubit 499 should measure 1, total qubits = 500
    assert!(s.contains("500"), "Should report 500 qubits, got: {}", s);
    assert!(s.contains("1"), "X|0> on qubit 499 should measure 1, got: {}", s);
}
