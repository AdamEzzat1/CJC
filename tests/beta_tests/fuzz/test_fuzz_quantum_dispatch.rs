// Fuzz: every quantum builtin is panic-free at the language boundary.
//
// `dispatch_quantum` is the single entry point both executors use (cjc-eval and
// cjc-mir-exec call it with the evaluated argument vector). This sweep calls
// every builtin with seeded random argument vectors drawn from:
// - bad integers, floats (incl. NaN/inf), strings, and malformed arrays;
// - freshly built quantum states of every kind, on both backends.
//
// Property: the call returns `Ok(..)` or `Err(..)`, and never panics. A panic
// aborts the user's whole `cjcl` process (audit probes p01–p07, p12, p17).
//
// Integer arguments are kept small (≤ 5) so every accidentally valid call stays
// cheap. The size caps themselves are exercised separately with explicit calls.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use cjc_quantum::dispatch_quantum;
use cjc_runtime::value::Value;

/// Every builtin name `dispatch_quantum` handles (84, incl. aliases).
const BUILTINS: &[&str] = &[
    "qubits", "q_h", "q_x", "q_y", "q_z", "q_s", "q_t", "q_rx", "q_ry", "q_rz", "q_cx",
    "q_cnot", "q_cz", "q_swap", "q_toffoli", "q_ccx", "q_run", "q_measure", "q_probs",
    "q_sample", "q_amplitudes", "q_n_qubits", "q_n_gates", "mps_new", "mps_h", "mps_x",
    "mps_ry", "mps_cnot", "mps_z_expectation", "mps_energy", "mps_memory",
    "mps_left_canonicalize", "mps_right_canonicalize", "mps_mixed_canonicalize", "mps_swap",
    "vqe_heisenberg", "vqe_full_heisenberg", "qaoa_graph_cycle", "qaoa_maxcut",
    "stabilizer_new", "stabilizer_h", "stabilizer_s", "stabilizer_x", "stabilizer_y",
    "stabilizer_z", "stabilizer_cnot", "stabilizer_measure", "stabilizer_n_qubits",
    "density_new", "density_gate", "density_cnot", "density_depolarize", "density_dephase",
    "density_amplitude_damp", "density_trace", "density_purity", "density_entropy",
    "density_probs", "dmrg_ising", "dmrg_heisenberg", "qec_repetition_code",
    "qec_surface_code", "qec_syndrome", "qec_decode", "qec_logical_error_rate", "qml_train",
    "qml_predict", "quantum_inspect", "q_fermion_h2", "q_fermion_lih", "q_fermion_new",
    "q_fermion_n_terms", "q_fermion_expectation", "q_trotter_evolve", "q_trotter_error",
    "q_zne_mitigate", "q_zne_linear", "q_scale_noise", "q_expect_pauli", "q_fermion_add_term",
    "density_from_state", "q_copy", "q_to_qasm", "q_from_qasm",
];

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_quantum(name, args)
        .unwrap_or_else(|e| panic!("setup call {}({:?}) failed: {}", name, args, e))
        .unwrap_or_else(|| panic!("setup call {} not handled", name))
}

fn s(x: &str) -> Value {
    Value::String(Rc::new(x.to_string()))
}

fn arr(v: Vec<Value>) -> Value {
    Value::Array(Rc::new(v))
}

/// A fresh value of kind `k`. States are rebuilt per use because gate builtins
/// mutate their input in place.
fn make_value(k: u64) -> Value {
    match k % 35 {
        0 => Value::Int(-3),
        1 => Value::Int(-1),
        2 => Value::Int(0),
        3 => Value::Int(1),
        4 => Value::Int(2),
        5 => Value::Int(3),
        6 => Value::Int(5),
        7 => Value::Float(0.5),
        8 => Value::Float(1.9),
        9 => Value::Float(-0.25),
        10 => Value::Float(f64::NAN),
        11 => Value::Float(f64::INFINITY),
        12 => Value::Float(2.0),
        13 => s("pure"),
        14 => s("H"),
        15 => s("heisenberg"),
        16 => s("x"),
        17 => arr(vec![]),
        18 => arr(vec![Value::Int(0), Value::Int(1)]),
        19 => arr(vec![Value::Float(1.0), s("x")]),
        20 => arr(vec![
            arr(vec![Value::Float(0.1), Value::Float(0.2)]),
            arr(vec![Value::Float(0.3), Value::Float(0.4)]),
        ]),
        21 => call("q_h", &[call("qubits", &[Value::Int(3)]), Value::Int(0)]),
        22 => call("qubits", &[Value::Int(3), s("pure")]),
        23 => call("mps_new", &[Value::Int(4), Value::Int(4)]),
        24 => call("mps_new", &[Value::Int(4), Value::Int(4), s("pure")]),
        25 => call("stabilizer_new", &[Value::Int(4)]),
        26 => call("stabilizer_new", &[Value::Int(4), s("pure")]),
        27 => call("density_new", &[Value::Int(2)]),
        28 => call("density_new", &[Value::Int(2), s("pure")]),
        29 => call("qaoa_graph_cycle", &[Value::Int(4)]),
        30 => call("qec_repetition_code", &[Value::Int(3)]),
        31 => call("qec_surface_code", &[Value::Int(3)]),
        32 => call("q_fermion_h2", &[]),
        33 => call("q_run", &[call("qubits", &[Value::Int(2), s("pure")])]),
        _ => call("q_run", &[call("qubits", &[Value::Int(2)])]),
    }
}

#[test]
fn fuzz_dispatch_every_builtin_is_panic_free() {
    let mut rng = 0x5eed_u64;
    let mut panics: Vec<String> = Vec::new();
    let mut calls = 0usize;
    for &name in BUILTINS {
        for _ in 0..250 {
            let arity = (splitmix64(&mut rng) % 10) as usize;
            let args: Vec<Value> = (0..arity).map(|_| make_value(splitmix64(&mut rng))).collect();
            let shown = format!("{}({:?})", name, args);
            calls += 1;
            if catch_unwind(AssertUnwindSafe(|| dispatch_quantum(name, &args))).is_err() {
                panics.push(shown);
            }
        }
    }
    assert!(
        panics.is_empty(),
        "{} of {} dispatch calls panicked; first few:\n{}",
        panics.len(),
        calls,
        panics.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn fuzz_dispatch_every_builtin_is_recognised() {
    // An unknown name returns Ok(None); every listed builtin must be handled
    // (Err for missing args is fine, Ok(None) is not).
    for &name in BUILTINS {
        let r = dispatch_quantum(name, &[]);
        assert!(!matches!(r, Ok(None)), "{} is not recognised by dispatch_quantum", name);
    }
    assert!(matches!(dispatch_quantum("not_a_quantum_builtin", &[]), Ok(None)));
}

#[test]
fn fuzz_dispatch_builtin_list_matches_source() {
    // Guards against a new dispatch arm that the arity table, the sweep, and the
    // docs don't know about.
    let src = include_str!("../../../crates/cjc-quantum/src/dispatch.rs");
    let mut found: Vec<String> = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        if !t.starts_with('"') || !t.contains("=>") {
            continue;
        }
        let head = t.split("=>").next().unwrap();
        let head = head.split(" if ").next().unwrap();
        if !head.split('|').all(|p| p.trim().starts_with('"')) {
            continue;
        }
        for part in head.split('|') {
            let n = part.trim().trim_matches('"');
            // Skip string-literal match arms inside builtins (gate names etc.).
            let is_builtin_name = n
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                && !matches!(
                    n,
                    "ising" | "heisenberg" | "depolarizing" | "dephasing" | "amplitude_damping"
                );
            if is_builtin_name && !found.iter().any(|f| f == n) {
                found.push(n.to_string());
            }
        }
    }
    let mut expected: Vec<String> = BUILTINS.iter().map(|s| s.to_string()).collect();
    found.sort();
    expected.sort();
    assert_eq!(found, expected, "dispatch.rs match arms and BUILTINS list differ");
}

#[test]
fn fuzz_dispatch_size_caps_are_errors_not_aborts() {
    // Each of these used to panic or abort the process (probes p04, p05, p17, p22).
    let cases: Vec<(&str, Vec<Value>)> = vec![
        ("qubits", vec![Value::Int(27)]),
        ("qubits", vec![Value::Int(-1), s("pure")]),
        ("mps_new", vec![Value::Int(0), Value::Int(4)]),
        ("mps_new", vec![Value::Int(-1), Value::Int(4)]),
        ("mps_new", vec![Value::Int(1_000_000_000), Value::Int(4)]),
        ("mps_new", vec![Value::Int(4), Value::Int(0)]),
        ("stabilizer_new", vec![Value::Int(0)]),
        ("stabilizer_new", vec![Value::Int(1_000_000), s("pure")]),
        ("density_new", vec![Value::Int(15)]),
        ("density_new", vec![Value::Int(20), s("pure")]),
        ("qaoa_graph_cycle", vec![Value::Int(2)]),
        ("qec_surface_code", vec![Value::Int(1_000_000)]),
        ("q_fermion_new", vec![Value::Int(40)]),
    ];
    for (name, args) in cases {
        let r = catch_unwind(AssertUnwindSafe(|| dispatch_quantum(name, &args)));
        match r {
            Ok(Err(_)) => {}
            other => panic!("{}({:?}) should be Err, got {:?}", name, args, other.map(|r| r.map(|v| v.map(|v| v.type_name().to_string())))),
        }
    }
}

#[test]
fn fuzz_dispatch_reference_documents_every_builtin() {
    // docs/QUANTUM_SIMULATION.md's "Builtin Reference" must have a row for
    // every builtin, written as `name(`. Adding a dispatch arm without
    // documenting it fails here (the list test above ties BUILTINS to
    // dispatch.rs).
    let doc = include_str!("../../../docs/QUANTUM_SIMULATION.md");
    let start = doc
        .find("## Builtin Reference")
        .expect("QUANTUM_SIMULATION.md has no \"## Builtin Reference\" section");
    let section = &doc[start..];
    let end = section[3..].find("
## ").map(|i| i + 3).unwrap_or(section.len());
    let section = &section[..end];
    let missing: Vec<&str> = BUILTINS
        .iter()
        .copied()
        .filter(|n| !section.contains(&format!("`{}(", n)) && !section.contains(&format!(", `{}(", n)))
        .collect();
    assert!(missing.is_empty(), "builtins missing from the reference: {:?}", missing);
    let header = format!("all {} quantum builtins", BUILTINS.len());
    assert!(section.contains(&header), "reference header should say \"{}\"", header);
}
