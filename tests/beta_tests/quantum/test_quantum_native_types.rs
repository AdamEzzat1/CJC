// CJC v0.2 — Quantum Native Type System Tests
//
// These tests verify that quantum types (QuantumCircuit, QuantumMps,
// QuantumStabilizer, QuantumDensity, QuantumGraph, QuantumSurfaceCode)
// are first-class primitives in CJC's type system, resolving through
// the type checker and executing correctly via both eval and mir-exec.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn eval(src: &str) -> String {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Parse errors in source");
    format!("{}", cjc_eval::Interpreter::new(42).exec(&prog).unwrap())
}

fn mir(src: &str) -> String {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Parse errors in source");
    let (v, _) = cjc_mir_exec::run_program_with_executor(&prog, 42).unwrap();
    format!("{}", v)
}

/// Run through both eval and mir, assert they match.
fn parity(src: &str) -> String {
    let e = eval(src);
    let m = mir(src);
    assert_eq!(e, m, "Eval vs MIR parity failure");
    e
}

// ===========================================================================
// Type Registration: quantum types are recognized by the type checker
// ===========================================================================

#[test]
fn native_type_quantum_circuit_resolves() {
    use cjc_types::TypeEnv;
    let env = TypeEnv::new();
    assert!(env.resolve_type_name("QuantumCircuit").is_some());
    assert!(env.resolve_type_name("QuantumStatevector").is_some());
    assert!(env.resolve_type_name("QuantumMps").is_some());
    assert!(env.resolve_type_name("QuantumStabilizer").is_some());
    assert!(env.resolve_type_name("QuantumDensity").is_some());
    assert!(env.resolve_type_name("QuantumGraph").is_some());
    assert!(env.resolve_type_name("QuantumSurfaceCode").is_some());
}

#[test]
fn native_type_quantum_fn_sigs_registered() {
    use cjc_types::TypeEnv;
    let env = TypeEnv::new();
    // Core circuit builtins
    assert!(env.fn_sigs.contains_key("qubits"), "qubits not registered");
    assert!(env.fn_sigs.contains_key("q_h"), "q_h not registered");
    assert!(env.fn_sigs.contains_key("q_cnot"), "q_cnot not registered");
    assert!(env.fn_sigs.contains_key("q_run"), "q_run not registered");
    assert!(env.fn_sigs.contains_key("q_measure"), "q_measure not registered");
    // MPS builtins
    assert!(env.fn_sigs.contains_key("mps_new"), "mps_new not registered");
    assert!(env.fn_sigs.contains_key("mps_h"), "mps_h not registered");
    assert!(env.fn_sigs.contains_key("mps_cnot"), "mps_cnot not registered");
    assert!(env.fn_sigs.contains_key("mps_z_expectation"), "mps_z_expectation not registered");
    // VQE
    assert!(env.fn_sigs.contains_key("vqe_heisenberg"), "vqe_heisenberg not registered");
    // Stabilizer
    assert!(env.fn_sigs.contains_key("stabilizer_new"), "stabilizer_new not registered");
    assert!(env.fn_sigs.contains_key("stabilizer_h"), "stabilizer_h not registered");
    assert!(env.fn_sigs.contains_key("stabilizer_measure"), "stabilizer_measure not registered");
    // Density
    assert!(env.fn_sigs.contains_key("density_new"), "density_new not registered");
    assert!(env.fn_sigs.contains_key("density_gate"), "density_gate not registered");
    assert!(env.fn_sigs.contains_key("density_purity"), "density_purity not registered");
    // DMRG
    assert!(env.fn_sigs.contains_key("dmrg_heisenberg"), "dmrg_heisenberg not registered");
    // QAOA
    assert!(env.fn_sigs.contains_key("qaoa_graph_cycle"), "qaoa_graph_cycle not registered");
    assert!(env.fn_sigs.contains_key("qaoa_maxcut"), "qaoa_maxcut not registered");
    // QEC
    assert!(env.fn_sigs.contains_key("qec_repetition_code"), "qec_repetition_code not registered");
    assert!(env.fn_sigs.contains_key("qec_surface_code"), "qec_surface_code not registered");
    assert!(env.fn_sigs.contains_key("qec_logical_error_rate"), "qec_logical_error_rate not registered");
    // QML
    assert!(env.fn_sigs.contains_key("qml_predict"), "qml_predict not registered");
}

#[test]
fn native_type_display_names() {
    use cjc_types::Type;
    assert_eq!(format!("{}", Type::QuantumCircuit), "QuantumCircuit");
    assert_eq!(format!("{}", Type::QuantumStatevector), "QuantumStatevector");
    assert_eq!(format!("{}", Type::QuantumMps), "QuantumMps");
    assert_eq!(format!("{}", Type::QuantumStabilizer), "QuantumStabilizer");
    assert_eq!(format!("{}", Type::QuantumDensity), "QuantumDensity");
    assert_eq!(format!("{}", Type::QuantumGraph), "QuantumGraph");
    assert_eq!(format!("{}", Type::QuantumSurfaceCode), "QuantumSurfaceCode");
}

#[test]
fn native_type_unification() {
    use cjc_types::{Type, TypeSubst, unify};
    let mut subst = TypeSubst::new();

    // Same quantum types unify
    assert!(unify(&Type::QuantumCircuit, &Type::QuantumCircuit, &mut subst).is_ok());
    assert!(unify(&Type::QuantumMps, &Type::QuantumMps, &mut subst).is_ok());
    assert!(unify(&Type::QuantumStabilizer, &Type::QuantumStabilizer, &mut subst).is_ok());
    assert!(unify(&Type::QuantumDensity, &Type::QuantumDensity, &mut subst).is_ok());
    assert!(unify(&Type::QuantumGraph, &Type::QuantumGraph, &mut subst).is_ok());
    assert!(unify(&Type::QuantumSurfaceCode, &Type::QuantumSurfaceCode, &mut subst).is_ok());

    // Different quantum types don't unify
    assert!(unify(&Type::QuantumCircuit, &Type::QuantumMps, &mut subst).is_err());
    assert!(unify(&Type::QuantumMps, &Type::QuantumStabilizer, &mut subst).is_err());
    assert!(unify(&Type::QuantumDensity, &Type::F64, &mut subst).is_err());
}

#[test]
fn native_type_types_match() {
    use cjc_types::{Type, TypeEnv};
    let env = TypeEnv::new();

    assert!(env.types_match(&Type::QuantumCircuit, &Type::QuantumCircuit));
    assert!(env.types_match(&Type::QuantumMps, &Type::QuantumMps));
    assert!(!env.types_match(&Type::QuantumCircuit, &Type::QuantumMps));
    assert!(!env.types_match(&Type::QuantumMps, &Type::I64));
}

// ===========================================================================
// Circuit primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_circuit_create_and_gates() {
    let src = r#"
fn main() -> Any {
    let c = qubits(2);
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    q_n_gates(c)
}
"#;
    assert_eq!(parity(src), "2");
}

#[test]
fn native_circuit_bell_probs() {
    let src = r#"
fn main() -> Any {
    let c = qubits(2);
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    q_probs(c)
}
"#;
    let result = parity(src);
    assert!(result.contains("0.5"), "Bell state should have ~0.5 probs, got {}", result);
}

#[test]
fn native_circuit_measure() {
    let src = r#"
fn main() -> Any {
    let c = qubits(1);
    let c = q_x(c, 0);
    q_measure(c, 42)
}
"#;
    let result = parity(src);
    assert!(result.contains("1"), "X|0> should measure 1, got {}", result);
}

// ===========================================================================
// MPS primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_mps_create() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(10, 8);
    mps_memory(m)
}
"#;
    let result = parity(src);
    let mem: i64 = result.parse().unwrap();
    assert!(mem > 0, "MPS should use memory, got {}", mem);
}

#[test]
fn native_mps_hadamard_z_exp() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(10, 8);
    let m = mps_h(m, 0);
    mps_z_expectation(m, 0)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!(z.abs() < 1e-10, "H|0> Z-exp should be ~0, got {}", z);
}

#[test]
fn native_mps_x_gate() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(5, 4);
    let m = mps_x(m, 2);
    mps_z_expectation(m, 2)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!((z - (-1.0)).abs() < 1e-10, "X|0> Z-exp should be -1, got {}", z);
}

#[test]
fn native_mps_cnot_entanglement() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(3, 4);
    let m = mps_h(m, 0);
    let m = mps_cnot(m, 0, 1);
    mps_z_expectation(m, 0)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!(z.abs() < 1e-10, "Bell state Z-exp should be ~0, got {}", z);
}

#[test]
fn native_mps_50q() {
    // 50-qubit MPS — this is the signature CJC capability
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16);
    let m = mps_h(m, 0);
    let m = mps_h(m, 25);
    let m = mps_h(m, 49);
    let m = mps_x(m, 10);
    mps_z_expectation(m, 0)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!(z.abs() < 1e-10, "H|0> should give Z~0, got {}", z);
}

// ===========================================================================
// Stabilizer primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_stabilizer_create() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(100);
    stabilizer_n_qubits(s)
}
"#;
    assert_eq!(parity(src), "100");
}

#[test]
fn native_stabilizer_gates_and_measure() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(5);
    let s = stabilizer_x(s, 0);
    stabilizer_measure(s, 0, 42)
}
"#;
    assert_eq!(parity(src), "1");
}

#[test]
fn native_stabilizer_bell_state() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(2);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    stabilizer_measure(s, 0, 42)
}
"#;
    let result = parity(src);
    let m: i64 = result.parse().unwrap();
    assert!(m == 0 || m == 1, "Bell measurement should be 0 or 1, got {}", m);
}

#[test]
fn native_stabilizer_1000q() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(1000);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_x(s, 999);
    stabilizer_measure(s, 999, 42)
}
"#;
    assert_eq!(parity(src), "1");
}

// ===========================================================================
// Density matrix primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_density_create_trace() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2);
    density_trace(d)
}
"#;
    let result = parity(src);
    let tr: f64 = result.parse().unwrap();
    assert!((tr - 1.0).abs() < 1e-10, "Trace should be 1.0, got {}", tr);
}

#[test]
fn native_density_gate_purity() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2);
    let d = density_gate(d, "H", 0);
    density_purity(d)
}
"#;
    let result = parity(src);
    let p: f64 = result.parse().unwrap();
    assert!((p - 1.0).abs() < 1e-10, "Pure state purity should be 1.0, got {}", p);
}

#[test]
fn native_density_noise() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2);
    let d = density_gate(d, "H", 0);
    let d = density_depolarize(d, 0, 0.1);
    density_purity(d)
}
"#;
    let result = parity(src);
    let p: f64 = result.parse().unwrap();
    assert!(p < 1.0 && p > 0.0, "Noisy state purity should be < 1, got {}", p);
}

#[test]
fn native_density_entropy() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2);
    let d = density_gate(d, "H", 0);
    let d = density_cnot(d, 0, 1);
    let d = density_depolarize(d, 0, 0.05);
    density_entropy(d)
}
"#;
    let result = parity(src);
    let s: f64 = result.parse().unwrap();
    assert!(s >= 0.0, "Entropy should be non-negative, got {}", s);
}

// ===========================================================================
// VQE primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_vqe_small() {
    let src = r#"
fn main() -> Any {
    vqe_heisenberg(4, 4, 0.05, 1, 42)
}
"#;
    let result = parity(src);
    let e: f64 = result.parse().unwrap();
    assert!(e.is_finite(), "VQE energy should be finite, got {}", e);
}

// ===========================================================================
// QAOA primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_qaoa_graph() {
    let src = r#"
fn main() -> Any {
    let g = qaoa_graph_cycle(4);
    qaoa_maxcut(g, 4, 1, 0.1, 2, 42)
}
"#;
    let result = parity(src);
    assert!(result.contains(",") || result.contains("["), "QAOA should return array, got {}", result);
}

// ===========================================================================
// QEC primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_qec_logical_error_rate() {
    let src = r#"
fn main() -> Any {
    qec_logical_error_rate(3, 0.01, 10, 42)
}
"#;
    let result = parity(src);
    let rate: f64 = result.parse().unwrap();
    assert!(rate >= 0.0 && rate <= 1.0, "Error rate should be in [0,1], got {}", rate);
}

#[test]
fn native_qec_repetition_code() {
    let src = r#"
fn main() -> Any {
    let code = qec_repetition_code(3);
    let s = stabilizer_new(5);
    let s = stabilizer_h(s, 0);
    qec_syndrome(s, code, 42)
}
"#;
    let result = eval(src);
    assert!(result.starts_with("["), "Syndrome should be array, got {}", result);
}

// ===========================================================================
// DMRG primitives — eval + mir parity
// ===========================================================================

#[test]
fn native_dmrg_small() {
    let src = r#"
fn main() -> Any {
    dmrg_heisenberg(4, 4, 2, 0.1)
}
"#;
    let result = parity(src);
    let e: f64 = result.parse().unwrap();
    assert!(e < 0.0, "DMRG ground state energy should be negative, got {}", e);
}

// ===========================================================================
// Determinism: same seed = bit-identical output
// ===========================================================================

#[test]
fn native_determinism_circuit() {
    let src = r#"
fn main() -> Any {
    let c = qubits(3);
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    let c = q_cnot(c, 1, 2);
    q_probs(c)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Circuit must be deterministic");
}

#[test]
fn native_determinism_mps() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(20, 8);
    let m = mps_h(m, 0);
    let m = mps_ry(m, 5, 0.7);
    let m = mps_x(m, 10);
    mps_z_expectation(m, 5)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "MPS must be deterministic");
}

#[test]
fn native_determinism_stabilizer() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(50);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_x(s, 49);
    stabilizer_measure(s, 0, 42)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Stabilizer must be deterministic");
}

#[test]
fn native_determinism_density() {
    let src = r#"
fn main() -> Any {
    let d = density_new(3);
    let d = density_gate(d, "H", 0);
    let d = density_cnot(d, 0, 1);
    let d = density_depolarize(d, 0, 0.01);
    density_purity(d)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Density must be deterministic");
}

// ===========================================================================
// Cross-type operations: composing different quantum primitives
// ===========================================================================

#[test]
fn native_mps_vqe_compose() {
    // Create MPS → compute energy → run VQE
    let src = r#"
fn main() -> Any {
    let m = mps_new(6, 4);
    let m = mps_h(m, 0);
    let m = mps_cnot(m, 0, 1);
    let e1 = mps_z_expectation(m, 0);
    let e2 = vqe_heisenberg(6, 4, 0.05, 1, 42);
    e1 + e2
}
"#;
    let result = parity(src);
    let v: f64 = result.parse().unwrap();
    assert!(v.is_finite());
}

#[test]
fn native_stabilizer_qec_compose() {
    // Stabilizer state + QEC code + syndrome
    let src = r#"
fn main() -> Any {
    let code = qec_repetition_code(3);
    let s = stabilizer_new(5);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let syndrome = qec_syndrome(s, code, 42);
    qec_decode(syndrome, code)
}
"#;
    let result = eval(src);
    assert!(result.starts_with("["), "QEC decode should return array, got {}", result);
}

// ===========================================================================
// Fn sig return types are correct
// ===========================================================================

#[test]
fn native_fn_sig_return_types() {
    use cjc_types::{Type, TypeEnv};
    let env = TypeEnv::new();

    // qubits returns QuantumCircuit
    let qubits_sig = &env.fn_sigs["qubits"][0];
    assert_eq!(qubits_sig.ret, Type::QuantumCircuit);

    // mps_new returns QuantumMps
    let mps_sig = &env.fn_sigs["mps_new"][0];
    assert_eq!(mps_sig.ret, Type::QuantumMps);

    // stabilizer_new returns QuantumStabilizer
    let stab_sig = &env.fn_sigs["stabilizer_new"][0];
    assert_eq!(stab_sig.ret, Type::QuantumStabilizer);

    // density_new returns QuantumDensity
    let den_sig = &env.fn_sigs["density_new"][0];
    assert_eq!(den_sig.ret, Type::QuantumDensity);

    // qaoa_graph_cycle returns QuantumGraph
    let graph_sig = &env.fn_sigs["qaoa_graph_cycle"][0];
    assert_eq!(graph_sig.ret, Type::QuantumGraph);

    // qec_repetition_code returns QuantumSurfaceCode
    let qec_sig = &env.fn_sigs["qec_repetition_code"][0];
    assert_eq!(qec_sig.ret, Type::QuantumSurfaceCode);

    // mps_z_expectation returns f64
    let z_sig = &env.fn_sigs["mps_z_expectation"][0];
    assert_eq!(z_sig.ret, Type::F64);

    // stabilizer_measure returns i64
    let m_sig = &env.fn_sigs["stabilizer_measure"][0];
    assert_eq!(m_sig.ret, Type::I64);
}

// ===========================================================================
// Fn sig param types are correct
// ===========================================================================

#[test]
fn native_fn_sig_param_types() {
    use cjc_types::{Type, TypeEnv};
    let env = TypeEnv::new();

    // mps_h(mps: QuantumMps, qubit: i64) -> QuantumMps
    let mps_h_sig = &env.fn_sigs["mps_h"][0];
    assert_eq!(mps_h_sig.params[0].1, Type::QuantumMps);
    assert_eq!(mps_h_sig.params[1].1, Type::I64);
    assert_eq!(mps_h_sig.ret, Type::QuantumMps);

    // density_gate(dm: QuantumDensity, gate: String, qubit: i64) -> QuantumDensity
    let dg_sig = &env.fn_sigs["density_gate"][0];
    assert_eq!(dg_sig.params[0].1, Type::QuantumDensity);
    assert_eq!(dg_sig.params[1].1, Type::Str);
    assert_eq!(dg_sig.params[2].1, Type::I64);
    assert_eq!(dg_sig.ret, Type::QuantumDensity);

    // stabilizer_cnot(state: QuantumStabilizer, ctrl: i64, tgt: i64) -> QuantumStabilizer
    let sc_sig = &env.fn_sigs["stabilizer_cnot"][0];
    assert_eq!(sc_sig.params[0].1, Type::QuantumStabilizer);
    assert_eq!(sc_sig.params[1].1, Type::I64);
    assert_eq!(sc_sig.params[2].1, Type::I64);
    assert_eq!(sc_sig.ret, Type::QuantumStabilizer);
}

// ===========================================================================
// NoGC safety: all quantum builtins are marked nogc-safe
// ===========================================================================

#[test]
fn native_quantum_builtins_nogc_safe() {
    use cjc_types::TypeEnv;
    let env = TypeEnv::new();

    let quantum_fns = [
        "qubits", "q_h", "q_x", "q_cnot", "q_run", "q_measure",
        "mps_new", "mps_h", "mps_cnot", "mps_z_expectation",
        "stabilizer_new", "stabilizer_h", "stabilizer_cnot", "stabilizer_measure",
        "density_new", "density_gate", "density_purity",
        "vqe_heisenberg", "dmrg_heisenberg",
        "qaoa_graph_cycle", "qaoa_maxcut",
        "qec_repetition_code", "qec_logical_error_rate",
    ];

    for name in &quantum_fns {
        let sigs = env.fn_sigs.get(*name)
            .unwrap_or_else(|| panic!("{} not registered", name));
        assert!(sigs[0].is_nogc, "{} should be marked nogc", name);
    }
}
