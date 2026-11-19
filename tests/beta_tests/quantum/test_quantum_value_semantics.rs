// ADR-0044 — quantum value semantics.
//
// Circuits (`qubits`, `q_*` gates) are values: a gate returns a new circuit
// and never changes its argument. Before ADR-0044, `let b = a; b = q_x(b, 1);`
// also added the X to `a`, because both names shared one Rc<RefCell<Circuit>>.
//
// Simulator states (MPS, stabilizer, density) are documented mutable handles.
// `q_copy` makes an independent deep copy of any quantum value.
//
// Every .cjcl case runs through both executors and must agree.

use std::rc::Rc;

use cjc_quantum::dispatch_quantum;
use cjc_runtime::value::Value;

fn run_eval(src: &str) -> String {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in:\n{src}");
    let v = cjc_eval::Interpreter::new(42).exec(&program).expect("eval failed");
    format!("{}", v)
}

fn run_mir(src: &str) -> String {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in:\n{src}");
    let (v, _) = cjc_mir_exec::run_program_with_executor(&program, 42).expect("mir failed");
    format!("{}", v)
}

/// Run in both executors, require identical output, return it.
fn run_both(src: &str) -> String {
    let (a, b) = (run_eval(src), run_mir(src));
    assert_eq!(a, b, "executor divergence for:\n{src}");
    a
}

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_quantum(name, args)
        .unwrap_or_else(|e| panic!("{name} failed: {e}"))
        .unwrap_or_else(|| panic!("{name} not handled"))
}

fn s(x: &str) -> Value {
    Value::String(Rc::new(x.to_string()))
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Int(i) => *i,
        other => panic!("expected Int, got {other}"),
    }
}

// ---------------------------------------------------------------------------
// Circuits are values
// ---------------------------------------------------------------------------

#[test]
fn gate_on_alias_does_not_change_original_circuit() {
    for backend in ["", ", \"pure\""] {
        let src = format!(
            "fn main() -> Any {{
                let a = qubits(2{backend});
                a = q_x(a, 0);
                let b = a;
                b = q_x(b, 1);
                [q_n_gates(a), q_n_gates(b)]
            }}"
        );
        assert_eq!(run_both(&src), "[1, 2]", "backend {backend:?}");
    }
}

#[test]
fn gate_inside_function_does_not_change_callers_circuit() {
    let src = "
        fn with_h(c: Any) -> Any {
            c = q_h(c, 0);
            return c;
        }
        fn main() -> Any {
            let base = qubits(2);
            let grown = with_h(base);
            [q_n_gates(base), q_n_gates(grown)]
        }";
    assert_eq!(run_both(src), "[0, 1]");
}

#[test]
fn parameter_sweep_from_shared_base_circuit() {
    // The pattern the old aliasing broke: every variant is base + one rotation.
    let src = "
        fn main() -> Any {
            let base = q_h(qubits(1), 0);
            let counts = [];
            let theta = 0.0;
            while theta < 0.35 {
                let variant = q_ry(base, 0, theta);
                counts = array_push(counts, q_n_gates(variant));
                theta = theta + 0.1;
            }
            counts = array_push(counts, q_n_gates(base));
            counts
        }";
    assert_eq!(run_both(src), "[2, 2, 2, 2, 1]");
}

#[test]
fn discarded_gate_result_is_a_no_op() {
    // Documented consequence: a gate call whose result is dropped does nothing.
    let src = "
        fn main() -> Any {
            let c = qubits(1);
            q_x(c, 0);
            q_probs(c)
        }";
    assert_eq!(run_both(src), "[1, 0]");
}

#[test]
fn circuit_value_semantics_preserve_simulation_results() {
    // Same gates, same bits as before: only aliasing changed.
    let src = "
        fn main() -> Any {
            let c = qubits(3);
            c = q_h(c, 0);
            c = q_cx(c, 0, 1);
            c = q_ry(c, 2, 0.7);
            c = q_toffoli(c, 0, 1, 2);
            q_probs(c)
        }";
    let out = run_both(src);
    let c = call("qubits", &[Value::Int(3)]);
    let c = call("q_h", &[c, Value::Int(0)]);
    let c = call("q_cx", &[c, Value::Int(0), Value::Int(1)]);
    let c = call("q_ry", &[c, Value::Int(2), Value::Float(0.7)]);
    let c = call("q_toffoli", &[c, Value::Int(0), Value::Int(1), Value::Int(2)]);
    assert_eq!(out, format!("{}", call("q_probs", &[c])));
}

// ---------------------------------------------------------------------------
// Simulator states are handles; q_copy forks them
// ---------------------------------------------------------------------------

#[test]
fn stabilizer_is_a_handle_and_q_copy_forks_it() {
    let src = "
        fn main() -> Any {
            let s = stabilizer_new(2);
            let fork = q_copy(s);
            stabilizer_x(s, 0);
            [stabilizer_measure(s, 0, 1), stabilizer_measure(fork, 0, 1)]
        }";
    // The bare call mutated the handle `s`; the fork kept |00>.
    assert_eq!(run_both(src), "[1, 0]");
}

#[test]
fn mps_q_copy_is_independent() {
    for backend in ["", ", \"pure\""] {
        let src = format!(
            "fn main() -> Any {{
                let m = mps_new(3, 4{backend});
                let fork = q_copy(m);
                fork = mps_x(fork, 1);
                [mps_z_expectation(m, 1), mps_z_expectation(fork, 1)]
            }}"
        );
        assert_eq!(run_both(&src), "[1, -1]", "backend {backend:?}");
    }
}

#[test]
fn density_q_copy_is_independent() {
    let src = "
        fn main() -> Any {
            let d = density_new(1);
            let fork = q_copy(d);
            fork = density_gate(fork, \"X\", 0);
            [density_probs(d), density_probs(fork)]
        }";
    assert_eq!(run_both(src), "[[1, 0], [0, 1]]");
}

#[test]
fn q_copy_supports_every_quantum_value() {
    let values = [
        call("qubits", &[Value::Int(2)]),
        call("q_run", &[call("qubits", &[Value::Int(2)])]),
        call("mps_new", &[Value::Int(3), Value::Int(4)]),
        call("stabilizer_new", &[Value::Int(3)]),
        call("density_new", &[Value::Int(2)]),
        call("qaoa_graph_cycle", &[Value::Int(4)]),
        call("qec_repetition_code", &[Value::Int(3)]),
        call("q_fermion_h2", &[]),
        call("qubits", &[Value::Int(2), s("pure")]),
        call("q_run", &[call("qubits", &[Value::Int(2), s("pure")])]),
        call("mps_new", &[Value::Int(3), Value::Int(4), s("pure")]),
        call("stabilizer_new", &[Value::Int(3), s("pure")]),
        call("density_new", &[Value::Int(2), s("pure")]),
        call("q_fermion_h2", &[s("pure")]),
    ];
    for v in &values {
        let copy = call("q_copy", &[v.clone()]);
        match (v, &copy) {
            (Value::QuantumState(a), Value::QuantumState(b)) => {
                assert!(!Rc::ptr_eq(a, b), "q_copy returned the same object");
            }
            _ => panic!("q_copy did not return a QuantumState"),
        }
    }
}

#[test]
fn q_copy_rejects_non_quantum_values() {
    assert!(dispatch_quantum("q_copy", &[Value::Int(3)]).is_err());
    assert!(dispatch_quantum("q_copy", &[]).is_err());
}

#[test]
fn stabilizer_copy_matches_original_measurements() {
    // A copy is a faithful snapshot: same seed, same outcomes.
    let st = call("stabilizer_new", &[Value::Int(3)]);
    call("stabilizer_h", &[st.clone(), Value::Int(0)]);
    call("stabilizer_cnot", &[st.clone(), Value::Int(0), Value::Int(1)]);
    let copy = call("q_copy", &[st.clone()]);
    for q in 0..3 {
        let a = int(&call("stabilizer_measure", &[st.clone(), Value::Int(q), Value::Int(9)]));
        let b = int(&call("stabilizer_measure", &[copy.clone(), Value::Int(q), Value::Int(9)]));
        assert_eq!(a, b, "qubit {q}");
    }
}
