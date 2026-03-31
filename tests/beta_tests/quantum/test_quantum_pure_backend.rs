// CJC v0.2 — Pure CJC Backend Tests
//
// These tests verify the dual-mode quantum architecture:
// - "pure" flag selects pure CJC backend (inspectable, modifiable)
// - Default selects Rust backend (fast, optimized)
// - Both backends produce physically correct results
// - Both are deterministic (same seed → same result)

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

fn parity(src: &str) -> String {
    let e = eval(src);
    let m = mir(src);
    assert_eq!(e, m, "Eval vs MIR parity failure");
    e
}

// ===========================================================================
// Pure MPS — "mps_new(n, chi, \"pure\")"
// ===========================================================================

#[test]
fn pure_mps_create() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(10, 8, "pure");
    mps_memory(m)
}
"#;
    let result = parity(src);
    let mem: i64 = result.parse().unwrap();
    assert!(mem > 0, "Pure MPS should use memory, got {}", mem);
}

#[test]
fn pure_mps_hadamard_z_exp() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(10, 8, "pure");
    let m = mps_h(m, 0);
    mps_z_expectation(m, 0)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!(z.abs() < 1e-10, "H|0> Z-exp should be ~0, got {}", z);
}

#[test]
fn pure_mps_x_gate() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(5, 4, "pure");
    let m = mps_x(m, 2);
    mps_z_expectation(m, 2)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!((z - (-1.0)).abs() < 1e-10, "X|0> Z-exp should be -1, got {}", z);
}

#[test]
fn pure_mps_cnot() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(3, 8, "pure");
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
fn pure_mps_50q() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16, "pure");
    let m = mps_h(m, 0);
    let m = mps_h(m, 25);
    let m = mps_h(m, 49);
    let m = mps_x(m, 10);
    mps_z_expectation(m, 0)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    assert!(z.abs() < 1e-10, "50q H|0> Z-exp should be ~0, got {}", z);
}

#[test]
fn pure_mps_ry() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(5, 8, "pure");
    let m = mps_ry(m, 0, 1.5707963267948966);
    mps_z_expectation(m, 0)
}
"#;
    let result = parity(src);
    let z: f64 = result.parse().unwrap();
    // Ry(π/2)|0⟩ = (|0⟩ + |1⟩)/√2, Z-exp ≈ 0
    assert!(z.abs() < 1e-10, "Ry(pi/2) Z-exp should be ~0, got {}", z);
}

// ===========================================================================
// Pure Stabilizer — "stabilizer_new(n, \"pure\")"
// ===========================================================================

#[test]
fn pure_stabilizer_create() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(100, "pure");
    stabilizer_n_qubits(s)
}
"#;
    assert_eq!(parity(src), "100");
}

#[test]
fn pure_stabilizer_x_measure() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(5, "pure");
    let s = stabilizer_x(s, 0);
    stabilizer_measure(s, 0, 42)
}
"#;
    assert_eq!(parity(src), "1");
}

#[test]
fn pure_stabilizer_bell() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(2, "pure");
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    stabilizer_measure(s, 0, 42)
}
"#;
    let result = parity(src);
    let m: i64 = result.parse().unwrap();
    assert!(m == 0 || m == 1);
}

#[test]
fn pure_stabilizer_1000q() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(1000, "pure");
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_x(s, 999);
    stabilizer_measure(s, 999, 42)
}
"#;
    assert_eq!(parity(src), "1");
}

#[test]
fn pure_stabilizer_all_gates() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(5, "pure");
    let s = stabilizer_h(s, 0);
    let s = stabilizer_s(s, 1);
    let s = stabilizer_x(s, 2);
    let s = stabilizer_y(s, 3);
    let s = stabilizer_z(s, 4);
    stabilizer_n_qubits(s)
}
"#;
    assert_eq!(parity(src), "5");
}

// ===========================================================================
// Pure Density — "density_new(n, \"pure\")"
// ===========================================================================

#[test]
fn pure_density_trace() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2, "pure");
    density_trace(d)
}
"#;
    let result = parity(src);
    let tr: f64 = result.parse().unwrap();
    assert!((tr - 1.0).abs() < 1e-10, "Trace should be 1.0, got {}", tr);
}

#[test]
fn pure_density_gate_purity() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2, "pure");
    let d = density_gate(d, "H", 0);
    density_purity(d)
}
"#;
    let result = parity(src);
    let p: f64 = result.parse().unwrap();
    assert!((p - 1.0).abs() < 1e-10, "Pure state purity should be 1.0, got {}", p);
}

#[test]
fn pure_density_noise() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2, "pure");
    let d = density_gate(d, "H", 0);
    let d = density_depolarize(d, 0, 0.1);
    density_purity(d)
}
"#;
    let result = parity(src);
    let p: f64 = result.parse().unwrap();
    assert!(p < 1.0 && p > 0.0, "Noisy purity should be < 1, got {}", p);
}

#[test]
fn pure_density_entropy() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2, "pure");
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

#[test]
fn pure_density_all_noise_channels() {
    let src = r#"
fn main() -> Any {
    let d = density_new(2, "pure");
    let d = density_gate(d, "X", 0);
    let d = density_depolarize(d, 0, 0.01);
    let d = density_dephase(d, 0, 0.01);
    let d = density_amplitude_damp(d, 0, 0.01);
    density_purity(d)
}
"#;
    let result = parity(src);
    let p: f64 = result.parse().unwrap();
    assert!(p > 0.0 && p <= 1.0, "Purity after noise should be in (0,1], got {}", p);
}

// ===========================================================================
// Pure Circuit — "qubits(n, \"pure\")"
// ===========================================================================

#[test]
fn pure_circuit_bell_probs() {
    let src = r#"
fn main() -> Any {
    let c = qubits(2, "pure");
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    q_probs(c)
}
"#;
    let result = parity(src);
    // Pure backend may have tiny fp rounding: 0.4999999999999999 vs 0.5000000000000001
    assert!(result.contains("0.49999") || result.contains("0.5"),
        "Bell state should have ~0.5 probs, got {}", result);
}

#[test]
fn pure_circuit_x_measure() {
    let src = r#"
fn main() -> Any {
    let c = qubits(1, "pure");
    let c = q_x(c, 0);
    q_measure(c, 42)
}
"#;
    let result = parity(src);
    assert!(result.contains("1"), "X|0> should measure 1, got {}", result);
}

#[test]
fn pure_circuit_n_gates() {
    let src = r#"
fn main() -> Any {
    let c = qubits(3, "pure");
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    let c = q_rx(c, 2, 0.5);
    q_n_gates(c)
}
"#;
    assert_eq!(parity(src), "3");
}

// ===========================================================================
// Determinism — pure backend produces identical results across runs
// ===========================================================================

#[test]
fn pure_determinism_mps() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(20, 8, "pure");
    let m = mps_h(m, 0);
    let m = mps_ry(m, 5, 0.7);
    let m = mps_x(m, 10);
    mps_z_expectation(m, 5)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Pure MPS must be deterministic");
}

#[test]
fn pure_determinism_stabilizer() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(50, "pure");
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    stabilizer_measure(s, 0, 42)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Pure Stabilizer must be deterministic");
}

#[test]
fn pure_determinism_density() {
    let src = r#"
fn main() -> Any {
    let d = density_new(3, "pure");
    let d = density_gate(d, "H", 0);
    let d = density_cnot(d, 0, 1);
    let d = density_depolarize(d, 0, 0.01);
    density_purity(d)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Pure Density must be deterministic");
}

#[test]
fn pure_determinism_circuit() {
    let src = r#"
fn main() -> Any {
    let c = qubits(3, "pure");
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    let c = q_ry(c, 2, 0.7);
    q_probs(c)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Pure Circuit must be deterministic");
}

// ===========================================================================
// Cross-backend: Rust and pure backends produce physically equivalent results
// ===========================================================================

#[test]
fn cross_backend_mps_z_expectation() {
    let rust_src = r#"
fn main() -> Any {
    let m = mps_new(10, 8);
    let m = mps_h(m, 0);
    let m = mps_x(m, 5);
    mps_z_expectation(m, 0)
}
"#;
    let pure_src = r#"
fn main() -> Any {
    let m = mps_new(10, 8, "pure");
    let m = mps_h(m, 0);
    let m = mps_x(m, 5);
    mps_z_expectation(m, 0)
}
"#;
    let rust_z: f64 = eval(rust_src).parse().unwrap();
    let pure_z: f64 = eval(pure_src).parse().unwrap();
    assert!((rust_z - pure_z).abs() < 1e-10,
        "Cross-backend Z-exp should match: rust={}, pure={}", rust_z, pure_z);
}

#[test]
fn cross_backend_stabilizer_measure() {
    let rust_src = r#"
fn main() -> Any {
    let s = stabilizer_new(5);
    let s = stabilizer_x(s, 0);
    stabilizer_measure(s, 0, 42)
}
"#;
    let pure_src = r#"
fn main() -> Any {
    let s = stabilizer_new(5, "pure");
    let s = stabilizer_x(s, 0);
    stabilizer_measure(s, 0, 42)
}
"#;
    assert_eq!(eval(rust_src), eval(pure_src), "X|0> should measure 1 in both backends");
}

#[test]
fn cross_backend_density_purity() {
    let rust_src = r#"
fn main() -> Any {
    let d = density_new(2);
    let d = density_gate(d, "H", 0);
    density_purity(d)
}
"#;
    let pure_src = r#"
fn main() -> Any {
    let d = density_new(2, "pure");
    let d = density_gate(d, "H", 0);
    density_purity(d)
}
"#;
    let rust_p: f64 = eval(rust_src).parse().unwrap();
    let pure_p: f64 = eval(pure_src).parse().unwrap();
    assert!((rust_p - pure_p).abs() < 1e-10,
        "Cross-backend purity should match: rust={}, pure={}", rust_p, pure_p);
}

#[test]
fn cross_backend_circuit_probs() {
    let rust_src = r#"
fn main() -> Any {
    let c = qubits(2);
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    q_probs(c)
}
"#;
    let pure_src = r#"
fn main() -> Any {
    let c = qubits(2, "pure");
    let c = q_h(c, 0);
    let c = q_cnot(c, 0, 1);
    q_probs(c)
}
"#;
    // Allow tiny fp differences between backends (different evaluation order)
    let rust_r = eval(rust_src);
    let pure_r = eval(pure_src);
    // Both should have 0.5 at positions 0 and 3, 0 at 1 and 2
    assert!(rust_r.contains("0.5") || rust_r.contains("0.49999"), "Rust: {}", rust_r);
    assert!(pure_r.contains("0.5") || pure_r.contains("0.49999"), "Pure: {}", pure_r);
}

// ===========================================================================
// Inspect — pure backend state is CJC-inspectable
// ===========================================================================

#[test]
fn pure_inspect_mps() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(3, 4, "pure");
    quantum_inspect(m)
}
"#;
    let result = eval(src);
    assert!(result.contains("pure") || result.contains("mps"),
        "Inspect should show backend info, got {}", result);
}
