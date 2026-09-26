// ADR-0045 — passing quantum states between builtins.
//
// Observables take a "state argument": a circuit (executed) or a statevector
// (from q_run / q_trotter_evolve). New builtins: q_expect_pauli,
// q_fermion_add_term, density_from_state.
//
// Key property: taking an observable of `q_run(c)` gives the same bits as
// taking it of `c`, including seeded sampling and measurement.

use std::rc::Rc;

use cjc_quantum::dispatch_quantum;
use cjc_runtime::value::Value;

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_quantum(name, args)
        .unwrap_or_else(|e| panic!("{name} failed: {e}"))
        .unwrap_or_else(|| panic!("{name} not handled"))
}

fn err(name: &str, args: &[Value]) -> String {
    match dispatch_quantum(name, args) {
        Err(e) => e,
        Ok(v) => panic!("{name} should fail, got {:?}", v.map(|v| v.to_string())),
    }
}

fn s(x: &str) -> Value {
    Value::String(Rc::new(x.to_string()))
}

fn f(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        other => panic!("expected Float, got {other}"),
    }
}

fn i(v: &Value) -> i64 {
    match v {
        Value::Int(x) => *x,
        other => panic!("expected Int, got {other}"),
    }
}

/// A 3-qubit circuit with superposition, entanglement, and a rotation.
fn test_circuit(backend: Option<&str>) -> Value {
    let mut c = match backend {
        Some(b) => call("qubits", &[Value::Int(3), s(b)]),
        None => call("qubits", &[Value::Int(3)]),
    };
    c = call("q_h", &[c, Value::Int(0)]);
    c = call("q_cx", &[c, Value::Int(0), Value::Int(1)]);
    c = call("q_ry", &[c, Value::Int(2), Value::Float(0.9)]);
    c = call("q_rz", &[c, Value::Int(1), Value::Float(0.3)]);
    c
}

fn bell() -> Value {
    let c = call("qubits", &[Value::Int(2)]);
    let c = call("q_h", &[c, Value::Int(0)]);
    call("q_cx", &[c, Value::Int(0), Value::Int(1)])
}

// ---------------------------------------------------------------------------
// Observables accept a statevector, bit-identically
// ---------------------------------------------------------------------------

#[test]
fn observables_on_statevector_equal_observables_on_circuit() {
    let c = test_circuit(None);
    let psi = call("q_run", &[c.clone()]);
    for name in ["q_probs", "q_amplitudes", "q_n_qubits"] {
        assert_eq!(
            call(name, &[c.clone()]).to_string(),
            call(name, &[psi.clone()]).to_string(),
            "{name}"
        );
    }
    for seed in [0i64, 7, -3] {
        let a = call("q_sample", &[c.clone(), Value::Int(64), Value::Int(seed)]);
        let b = call("q_sample", &[psi.clone(), Value::Int(64), Value::Int(seed)]);
        assert_eq!(a.to_string(), b.to_string(), "q_sample seed {seed}");
        let a = call("q_measure", &[c.clone(), Value::Int(seed)]);
        let b = call("q_measure", &[psi.clone(), Value::Int(seed)]);
        assert_eq!(a.to_string(), b.to_string(), "q_measure seed {seed}");
    }
}

#[test]
fn measuring_a_statevector_does_not_collapse_it() {
    // q_measure works on a copy; the statevector value is unchanged.
    let psi = call("q_run", &[bell()]);
    let before = call("q_amplitudes", &[psi.clone()]).to_string();
    call("q_measure", &[psi.clone(), Value::Int(1)]);
    assert_eq!(before, call("q_amplitudes", &[psi]).to_string());
}

#[test]
fn fermion_expectation_on_statevector_equals_on_circuit() {
    let h = call("q_fermion_h2", &[]); // 2-qubit reduced H2
    let mut c = call("qubits", &[Value::Int(2)]);
    c = call("q_x", &[c, Value::Int(0)]);
    c = call("q_ry", &[c, Value::Int(1), Value::Float(0.2)]);
    c = call("q_cx", &[c, Value::Int(1), Value::Int(0)]);
    let psi = call("q_run", &[c.clone()]);
    let a = f(&call("q_fermion_expectation", &[h.clone(), c]));
    let b = f(&call("q_fermion_expectation", &[h, psi]));
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn trotter_output_can_be_measured_and_conserves_energy() {
    // exp(-iHt) conserves <H>; second-order Trotter with small steps comes close.
    let h = call("q_fermion_h2", &[]);
    let mut c = call("qubits", &[Value::Int(2)]);
    c = call("q_x", &[c, Value::Int(0)]);
    c = call("q_ry", &[c, Value::Int(1), Value::Float(0.8)]);
    c = call("q_cx", &[c, Value::Int(1), Value::Int(0)]);
    let e0 = f(&call("q_fermion_expectation", &[h.clone(), c.clone()]));
    let evolved = call(
        "q_trotter_evolve",
        &[h.clone(), c, Value::Float(0.5), Value::Int(50), Value::Int(2)],
    );
    let e1 = f(&call("q_fermion_expectation", &[h.clone(), evolved.clone()]));
    assert!((e1 - e0).abs() < 1e-3, "energy drift {} -> {}", e0, e1);
    // Evolving an already-evolved statevector is allowed too.
    let twice = call(
        "q_trotter_evolve",
        &[h, evolved.clone(), Value::Float(0.1), Value::Int(10), Value::Int(2)],
    );
    let p: f64 = match call("q_probs", &[twice]) {
        Value::Array(a) => a.iter().map(f).sum(),
        _ => unreachable!(),
    };
    assert!((p - 1.0).abs() < 1e-12);
    // And measured.
    assert_eq!(match call("q_measure", &[evolved, Value::Int(5)]) {
        Value::Array(a) => a.len(),
        _ => 0,
    }, 2);
}

// ---------------------------------------------------------------------------
// q_expect_pauli
// ---------------------------------------------------------------------------

#[test]
fn pauli_expectations_of_bell_state() {
    let psi = call("q_run", &[bell()]);
    for (p, want) in [("ZZ", 1.0), ("XX", 1.0), ("YY", -1.0), ("ZI", 0.0), ("IZ", 0.0), ("II", 1.0)] {
        let got = f(&call("q_expect_pauli", &[psi.clone(), s(p)]));
        assert!((got - want).abs() < 1e-12, "<{p}> = {got}, want {want}");
    }
}

#[test]
fn pauli_string_character_k_acts_on_qubit_k() {
    // X on qubit 0 only: <Z> on qubit 0 is -1, on qubit 1 is +1.
    let c = call("q_x", &[call("qubits", &[Value::Int(2)]), Value::Int(0)]);
    assert_eq!(f(&call("q_expect_pauli", &[c.clone(), s("ZI")])), -1.0);
    assert_eq!(f(&call("q_expect_pauli", &[c, s("IZ")])), 1.0);
}

#[test]
fn hamiltonian_equals_sum_of_its_pauli_terms() {
    // Build H = 0.5·ZZ − 0.25·XI + 0.125·YY with q_fermion_new + add_term and
    // compare against the weighted sum of q_expect_pauli.
    let terms = [("ZZI", 0.5), ("XII", -0.25), ("IYY", 0.125)];
    let mut h = call("q_fermion_new", &[Value::Int(3)]);
    for (p, c) in terms {
        h = call("q_fermion_add_term", &[h, s(p), Value::Float(c)]);
    }
    assert_eq!(i(&call("q_fermion_n_terms", &[h.clone()])), 3);
    let psi = call("q_run", &[test_circuit(None)]);
    let whole = f(&call("q_fermion_expectation", &[h, psi.clone()]));
    let parts: f64 = terms
        .iter()
        .map(|(p, c)| c * f(&call("q_expect_pauli", &[psi.clone(), s(p)])))
        .sum();
    assert!((whole - parts).abs() < 1e-14, "{whole} vs {parts}");
}

#[test]
fn add_term_returns_new_hamiltonian_and_keeps_original() {
    for pure in [false, true] {
        let h0 = if pure {
            call("q_fermion_h2", &[s("pure")])
        } else {
            call("q_fermion_new", &[Value::Int(2)])
        };
        let n0 = if pure { None } else { Some(i(&call("q_fermion_n_terms", &[h0.clone()]))) };
        let h1 = call("q_fermion_add_term", &[h0.clone(), s("ZZ"), Value::Float(1.0)]);
        if let Some(n0) = n0 {
            assert_eq!(i(&call("q_fermion_n_terms", &[h0.clone()])), n0);
            assert_eq!(i(&call("q_fermion_n_terms", &[h1.clone()])), n0 + 1);
        }
        // Adding ZZ with coefficient 1 shifts <H> on |00> by exactly +1.
        let zero = if pure {
            call("qubits", &[Value::Int(2), s("pure")])
        } else {
            call("qubits", &[Value::Int(2)])
        };
        let e0 = f(&call("q_fermion_expectation", &[h0, zero.clone()]));
        let e1 = f(&call("q_fermion_expectation", &[h1, zero]));
        assert!((e1 - e0 - 1.0).abs() < 1e-12, "pure={pure}: {e0} -> {e1}");
    }
}

#[test]
fn bad_pauli_strings_are_errors() {
    let psi = call("q_run", &[bell()]);
    assert!(err("q_expect_pauli", &[psi.clone(), s("ZQ")]).contains("invalid Pauli character"));
    assert!(err("q_expect_pauli", &[psi.clone(), s("ZZZ")]).contains("expected one per qubit"));
    assert!(err("q_expect_pauli", &[psi.clone(), Value::Int(3)]).contains("must be a string"));
    let h = call("q_fermion_new", &[Value::Int(2)]);
    assert!(err("q_fermion_add_term", &[h.clone(), s("zz"), Value::Float(1.0)]).contains("invalid"));
    assert!(err("q_fermion_add_term", &[h, s("ZZ"), Value::Float(f64::NAN)]).contains("finite"));
    assert!(err("q_expect_pauli", &[Value::Int(1), s("Z")]).contains("circuit or statevector"));
}

// ---------------------------------------------------------------------------
// density_from_state
// ---------------------------------------------------------------------------

#[test]
fn density_from_state_is_the_pure_state_projector() {
    for input in [test_circuit(None), call("q_run", &[test_circuit(None)])] {
        let rho = call("density_from_state", &[input.clone()]);
        assert!((f(&call("density_trace", &[rho.clone()])) - 1.0).abs() < 1e-14);
        assert!((f(&call("density_purity", &[rho.clone()])) - 1.0).abs() < 1e-12);
        let (dp, sp) = match (call("density_probs", &[rho]), call("q_probs", &[input])) {
            (Value::Array(a), Value::Array(b)) => (a, b),
            _ => unreachable!(),
        };
        for (a, b) in dp.iter().zip(sp.iter()) {
            assert!((f(a) - f(b)).abs() < 1e-15);
        }
    }
}

#[test]
fn noise_can_be_applied_to_a_prepared_state() {
    let rho = call("density_from_state", &[bell()]);
    let rho = call("density_depolarize", &[rho, Value::Int(0), Value::Float(0.2)]);
    let purity = f(&call("density_purity", &[rho]));
    assert!(purity < 0.99 && purity > 0.25, "purity {purity}");
}

#[test]
fn density_from_pure_backend_state_stays_pure_backend() {
    let rho = call("density_from_state", &[test_circuit(Some("pure"))]);
    // quantum_inspect errors on Rust-backend states, so success proves the
    // result stayed on the pure backend.
    call("quantum_inspect", &[rho.clone()]);
    assert!((f(&call("density_purity", &[rho])) - 1.0).abs() < 1e-12);
}

#[test]
fn density_from_state_respects_the_size_cap() {
    // Rejected before executing the 15-qubit circuit.
    let c = call("qubits", &[Value::Int(15)]);
    assert!(err("density_from_state", &[c]).contains("at most 14"));
}

// ---------------------------------------------------------------------------
// Pure backend
// ---------------------------------------------------------------------------

#[test]
fn pure_fermion_expectation_accepts_circuit_and_statevector() {
    let h = call("q_fermion_h2", &[s("pure")]);
    let mut c = call("qubits", &[Value::Int(2), s("pure")]);
    c = call("q_x", &[c, Value::Int(0)]);
    c = call("q_ry", &[c, Value::Int(1), Value::Float(0.4)]);
    let psi = call("q_run", &[c.clone()]);
    let a = f(&call("q_fermion_expectation", &[h.clone(), c]));
    let b = f(&call("q_fermion_expectation", &[h, psi]));
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn backends_agree_on_pauli_expectations() {
    let rust = f(&call("q_expect_pauli", &[test_circuit(None), s("XZY")]));
    let pure = f(&call("q_expect_pauli", &[test_circuit(Some("pure")), s("XZY")]));
    assert!((rust - pure).abs() < 1e-12, "{rust} vs {pure}");
}

// ---------------------------------------------------------------------------
// Executor parity for the whole surface
// ---------------------------------------------------------------------------

#[test]
fn interop_program_runs_identically_in_both_executors() {
    let src = r#"
        fn main() -> Any {
            let c = qubits(2);
            c = q_h(c, 0);
            c = q_cx(c, 0, 1);
            let psi = q_run(c);
            let h = q_fermion_new(2);
            h = q_fermion_add_term(h, "ZZ", 0.5);
            h = q_fermion_add_term(h, "XX", -0.25);
            let e = q_fermion_expectation(h, psi);
            let zz = q_expect_pauli(psi, "ZZ");
            let rho = density_from_state(psi);
            rho = density_dephase(rho, 1, 0.3);
            let evolved = q_trotter_evolve(h, psi, 0.4, 8, 2);
            [e, zz, density_purity(rho), q_sample(evolved, 6, 11), q_measure(psi, 3)]
        }
    "#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let a = cjc_eval::Interpreter::new(42).exec(&program).expect("eval");
    let (b, _) = cjc_mir_exec::run_program_with_executor(&program, 42).expect("mir");
    assert_eq!(a.to_string(), b.to_string());
    // <H> = 0.5<ZZ> - 0.25<XX> = 0.25 on the Bell state.
    match &a {
        Value::Array(items) => assert!((f(&items[0]) - 0.25).abs() < 1e-12, "{}", a),
        other => panic!("expected array, got {other}"),
    }
}
