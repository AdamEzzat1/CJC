//! Builtin-level tests via [`cjc_abng::dispatch_abng`].
//!
//! These tests exercise the user-facing builtin surface — the same code path
//! that `.cjcl` source hits when it calls `abng_*`. Both happy-path returns
//! and Err-path messages are asserted so the language-side error-handling
//! contract is part of the test suite.

use std::cell::RefCell;
use std::rc::Rc;

use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_runtime::value::Value;

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_abng(name, args).unwrap().unwrap()
}

fn try_call(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    dispatch_abng(name, args)
}

fn new_graph(seed: i64) -> i64 {
    match call("abng_new", &[Value::Int(seed)]) {
        Value::Int(i) => i,
        _ => panic!("abng_new should return Int"),
    }
}

#[test]
fn unknown_name_returns_none() {
    let r = dispatch_abng("not_an_abng_builtin", &[]).unwrap();
    assert!(r.is_none());
}

#[test]
fn arg_count_mismatch_errs() {
    let err = try_call("abng_new", &[]).unwrap_err();
    assert!(err.contains("expected 1 arguments"));
}

#[test]
fn graphs_are_independent() {
    reset_arena();
    let g1 = new_graph(1);
    let g2 = new_graph(2);
    assert_ne!(g1, g2);
    let _ = call("abng_observe", &[Value::Int(g1), Value::Int(0), Value::Float(1.0)]);
    // g2 should still be untouched.
    let stats = match call("abng_node_stats", &[Value::Int(g2), Value::Int(0)]) {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    assert_eq!(stats[0], 0.0); // n_seen == 0
}

#[test]
fn observe_batch_matches_individual() {
    reset_arena();
    let g_indiv = new_graph(0);
    let g_batch = new_graph(0);
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
        let _ = call(
            "abng_observe",
            &[Value::Int(g_indiv), Value::Int(0), Value::Float(v)],
        );
    }
    let batch = cjc_runtime::tensor::Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        &[5],
    )
    .unwrap();
    let _ = call(
        "abng_observe_batch",
        &[Value::Int(g_batch), Value::Int(0), Value::Tensor(batch)],
    );
    let head_a = match call("abng_chain_head", &[Value::Int(g_indiv)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    let head_b = match call("abng_chain_head", &[Value::Int(g_batch)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    assert_eq!(head_a, head_b);
}

fn expect_int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected Int, got {}", other.type_name()),
    }
}

fn expect_bool(v: Value) -> bool {
    match v {
        Value::Bool(b) => b,
        other => panic!("expected Bool, got {}", other.type_name()),
    }
}

fn expect_string(v: Value) -> String {
    match v {
        Value::String(s) => (*s).clone(),
        other => panic!("expected String, got {}", other.type_name()),
    }
}

#[test]
fn root_returns_zero() {
    reset_arena();
    let g = new_graph(0);
    assert_eq!(expect_int(call("abng_root", &[Value::Int(g)])), 0);
}

#[test]
fn node_count_is_one_in_phase_0_1() {
    reset_arena();
    let g = new_graph(0);
    assert_eq!(expect_int(call("abng_node_count", &[Value::Int(g)])), 1);
}

#[test]
fn audit_len_starts_at_one() {
    reset_arena();
    let g = new_graph(0);
    // The Created event is always present.
    assert_eq!(expect_int(call("abng_audit_len", &[Value::Int(g)])), 1);
    let _ = call("abng_observe", &[Value::Int(g), Value::Int(0), Value::Float(1.0)]);
    assert_eq!(expect_int(call("abng_audit_len", &[Value::Int(g)])), 2);
}

#[test]
fn chain_head_changes_on_observe() {
    reset_arena();
    let g = new_graph(0);
    let h0 = match call("abng_chain_head", &[Value::Int(g)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    let _ = call("abng_observe", &[Value::Int(g), Value::Int(0), Value::Float(1.0)]);
    let h1 = match call("abng_chain_head", &[Value::Int(g)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    assert_ne!(h0, h1);
    // Hex form is 64 chars (256-bit hash).
    assert_eq!(h0.len(), 64);
    assert_eq!(h1.len(), 64);
}

#[test]
fn verify_chain_returns_true_when_intact() {
    reset_arena();
    let g = new_graph(0);
    for i in 0..10 {
        let _ = call(
            "abng_observe",
            &[Value::Int(g), Value::Int(0), Value::Float(i as f64)],
        );
    }
    assert!(expect_bool(call("abng_verify_chain", &[Value::Int(g)])));
}

#[test]
fn missing_graph_id_errs_clearly() {
    reset_arena();
    let err = try_call("abng_chain_head", &[Value::Int(99999)]).unwrap_err();
    assert!(err.contains("no graph"));
}

#[test]
fn out_of_range_node_id_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = try_call("abng_observe", &[Value::Int(g), Value::Int(99), Value::Float(0.0)])
        .unwrap_err();
    assert!(err.contains("out of range"));
}

#[test]
fn negative_node_id_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = try_call("abng_observe", &[Value::Int(g), Value::Int(-1), Value::Float(0.0)])
        .unwrap_err();
    assert!(err.contains("non-negative"));
}

#[test]
fn serialize_replay_round_trip_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    for v in [1.0, 2.5, 7.0, 0.1, -3.0] {
        let _ = call("abng_observe", &[Value::Int(g), Value::Int(0), Value::Float(v)]);
    }
    let head_before = match call("abng_chain_head", &[Value::Int(g)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    let blob = match call("abng_serialize", &[Value::Int(g)]) {
        Value::Bytes(b) => b.borrow().clone(),
        _ => panic!(),
    };
    let g2 = match call(
        "abng_replay",
        &[Value::Bytes(Rc::new(RefCell::new(blob)))],
    ) {
        Value::Int(i) => i,
        _ => panic!(),
    };
    let head_after = match call("abng_chain_head", &[Value::Int(g2)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    assert_eq!(head_before, head_after);
}

#[test]
fn replay_rejects_corrupted_bytes() {
    reset_arena();
    let g = new_graph(0);
    let _ = call("abng_observe", &[Value::Int(g), Value::Int(0), Value::Float(1.0)]);
    let mut blob = match call("abng_serialize", &[Value::Int(g)]) {
        Value::Bytes(b) => b.borrow().clone(),
        _ => panic!(),
    };
    blob[0] = b'X'; // bust the magic
    let err = try_call(
        "abng_replay",
        &[Value::Bytes(Rc::new(RefCell::new(blob)))],
    )
    .unwrap_err();
    assert!(err.contains("magic"));
}

#[test]
fn drop_frees_the_id() {
    reset_arena();
    let g = new_graph(0);
    let _ = call("abng_drop", &[Value::Int(g)]);
    let err = try_call("abng_chain_head", &[Value::Int(g)]).unwrap_err();
    assert!(err.contains("no graph"));
}

#[test]
fn determinism_double_run_chain_head() {
    let make = || {
        reset_arena();
        let g = new_graph(123);
        for i in 0..30 {
            let _ = call(
                "abng_observe",
                &[Value::Int(g), Value::Int(0), Value::Float((i as f64) * 0.5)],
            );
        }
        match call("abng_chain_head", &[Value::Int(g)]) {
            Value::String(s) => (*s).clone(),
            _ => panic!(),
        }
    };
    let a = make();
    let b = make();
    assert_eq!(a, b);
}
