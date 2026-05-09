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

// ── Phase 0.6 Item 7: abng_route_to_leaf native kernel ─────────────

#[test]
fn route_to_leaf_matches_three_call_sequence() {
    // Phase 0.6 Item 7 — the fused native kernel must produce the
    // same leaf id as the encode_prefix + descend + extract_leaf
    // sequence it replaces.
    reset_arena();
    let g = new_graph(7);
    let codebook = cjc_runtime::tensor::Tensor::from_vec(
        vec![0.25, 0.5, 0.75],
        &[1, 3],
    )
    .unwrap();
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(codebook)],
    );
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(0)],
    );
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    );
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(2)],
    );

    // Run the 3-call sequence for a probe point, then the fused
    // single-call kernel. They must produce the same leaf id.
    for &x in &[0.10_f64, 0.30, 0.45, 0.62, 0.85] {
        let x_t = cjc_runtime::tensor::Tensor::from_vec(vec![x], &[1]).unwrap();

        // Sequence path: encode_prefix → descend → extract index 1.
        let prefix = call(
            "abng_encode_prefix",
            &[Value::Int(g), Value::Tensor(x_t.clone())],
        );
        let evidence = call("abng_descend", &[Value::Int(g), prefix]);
        let leaf_seq = match evidence {
            Value::Tensor(t) => t.to_vec()[1] as i64,
            _ => panic!("expected tensor"),
        };

        // Fused path: abng_route_to_leaf.
        let leaf_fused = match call(
            "abng_route_to_leaf",
            &[Value::Int(g), Value::Tensor(x_t)],
        ) {
            Value::Int(i) => i,
            _ => panic!("expected int"),
        };

        assert_eq!(
            leaf_seq, leaf_fused,
            "fused route_to_leaf must match the 3-call sequence at x={x}"
        );
    }
}

#[test]
fn route_to_leaf_arg_count_validation() {
    reset_arena();
    let _ = new_graph(0);
    // Missing args → error.
    let err = try_call("abng_route_to_leaf", &[Value::Int(0)]).unwrap_err();
    assert!(err.contains("expected 2 arguments"));
}

// ── Phase 0.6 Item 8: abng_route_to_leaf_batch (chunked dispatch) ──

#[test]
fn route_to_leaf_batch_matches_per_row() {
    // Phase 0.6 Item 8 — the batched native kernel must produce the
    // same per-row leaf ids as N invocations of abng_route_to_leaf.
    reset_arena();
    let g = new_graph(7);
    let codebook = cjc_runtime::tensor::Tensor::from_vec(
        vec![0.25, 0.5, 0.75],
        &[1, 3],
    )
    .unwrap();
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(codebook)],
    );
    for byte in 0..4 {
        let _ = call(
            "abng_add_node",
            &[Value::Int(g), Value::Int(0), Value::Int(byte)],
        );
    }

    let xs = vec![0.10_f64, 0.30, 0.45, 0.62, 0.85, 0.05, 0.55, 0.95];
    let n = xs.len();

    // Per-row leaf ids via abng_route_to_leaf.
    let mut per_row: Vec<i64> = Vec::with_capacity(n);
    for &x in &xs {
        let x_t = cjc_runtime::tensor::Tensor::from_vec(vec![x], &[1]).unwrap();
        let leaf = match call(
            "abng_route_to_leaf",
            &[Value::Int(g), Value::Tensor(x_t)],
        ) {
            Value::Int(i) => i,
            _ => panic!(),
        };
        per_row.push(leaf);
    }

    // Batched leaf ids via abng_route_to_leaf_batch.
    let xs_t = cjc_runtime::tensor::Tensor::from_vec(xs, &[n, 1]).unwrap();
    let batch_t = match call(
        "abng_route_to_leaf_batch",
        &[Value::Int(g), Value::Tensor(xs_t)],
    ) {
        Value::Tensor(t) => t,
        _ => panic!(),
    };
    let batch: Vec<i64> = batch_t.to_vec().iter().map(|f| *f as i64).collect();

    assert_eq!(per_row, batch);
}

#[test]
fn route_to_leaf_batch_rejects_non_2d_input() {
    reset_arena();
    let g = new_graph(0);
    let codebook = cjc_runtime::tensor::Tensor::from_vec(
        vec![0.25, 0.5, 0.75],
        &[1, 3],
    )
    .unwrap();
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(codebook)],
    );
    // 1-D input — must be 2-D [n, d].
    let bad = cjc_runtime::tensor::Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
    let err = try_call(
        "abng_route_to_leaf_batch",
        &[Value::Int(g), Value::Tensor(bad)],
    )
    .unwrap_err();
    assert!(err.contains("must be 2-D"), "got: {err}");
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
fn observe_batch_stats_match_individual_but_chain_differs() {
    // Phase 0.6 Item 4 (v13) — `abng_observe_batch` now emits ONE
    // BeliefUpdateBatch event instead of N BeliefUpdate events.
    // Post-batch stats are bit-identical to N per-row observes
    // (Welford in row order with Kahan compensation), but the chain
    // head differs because they're different audit histories. Use
    // `abng_observe_slice` for the legacy loop-observe semantics.
    reset_arena();
    let g_indiv = new_graph(0);
    let g_batch = new_graph(0);
    let g_slice = new_graph(0);
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
        let _ = call(
            "abng_observe",
            &[Value::Int(g_indiv), Value::Int(0), Value::Float(v)],
        );
    }
    let batch_t = cjc_runtime::tensor::Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        &[5],
    )
    .unwrap();
    let _ = call(
        "abng_observe_batch",
        &[
            Value::Int(g_batch),
            Value::Int(0),
            Value::Tensor(batch_t.clone()),
        ],
    );
    let _ = call(
        "abng_observe_slice",
        &[Value::Int(g_slice), Value::Int(0), Value::Tensor(batch_t)],
    );

    let head_indiv = match call("abng_chain_head", &[Value::Int(g_indiv)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    let head_batch = match call("abng_chain_head", &[Value::Int(g_batch)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    let head_slice = match call("abng_chain_head", &[Value::Int(g_slice)]) {
        Value::String(s) => (*s).clone(),
        _ => panic!(),
    };
    // Per-row and slice produce the same chain (slice is N per-row
    // events under the hood).
    assert_eq!(head_indiv, head_slice);
    // Batch produces a different chain (one batch event vs N).
    assert_ne!(head_batch, head_indiv);

    // But all three end up with bit-identical NodeStats canonical
    // bytes — Welford folds in the same row order regardless of
    // chain accounting.
    let stats_indiv = match call("abng_node_stats", &[Value::Int(g_indiv), Value::Int(0)]) {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    let stats_batch = match call("abng_node_stats", &[Value::Int(g_batch), Value::Int(0)]) {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    let stats_slice = match call("abng_node_stats", &[Value::Int(g_slice), Value::Int(0)]) {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!(),
    };
    assert_eq!(stats_indiv, stats_batch);
    assert_eq!(stats_indiv, stats_slice);
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

// ── Phase 0.7 Item 4: abng_train_step (fused per-row training) ────────

fn install_full_training_setup(seed: i64) -> i64 {
    // Codebook over 1-D input with 4 bins.
    let g = new_graph(seed);
    let codebook = cjc_runtime::tensor::Tensor::from_vec(
        vec![0.25, 0.5, 0.75],
        &[1, 3],
    )
    .unwrap();
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(codebook)],
    );
    // Leaf head: input_dim=1, hidden=[4], output_dim=1, activation="tanh".
    let hidden = cjc_runtime::tensor::Tensor::from_vec(vec![4.0_f64], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(1),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(Rc::new("tanh".to_string())),
        ],
    );
    // BLR prior: precision=2.0, a=1.0, b=0.5.
    let _ = call(
        "abng_set_blr_prior",
        &[Value::Int(g), Value::Float(2.0), Value::Float(1.0), Value::Float(0.5)],
    );
    // Add 4 child nodes so descend can land on a non-root leaf.
    for byte in 0..4 {
        let _ = call(
            "abng_add_node",
            &[Value::Int(g), Value::Int(0), Value::Int(byte)],
        );
    }
    g
}

#[test]
fn train_step_chain_head_matches_three_call_sequence() {
    // Phase 0.7 Item 4 — `abng_train_step` collapses route + blr_update +
    // observe into a single dispatch, but emits the EXACT SAME audit
    // event sequence (BlrUpdated, then BeliefUpdate). The chain_head
    // after `abng_train_step(g, x, phi, y)` MUST equal the chain_head
    // after the 3-call sequence on identical pre-state.
    //
    // Two independent graphs are built with identical setup and seed.
    // One runs the 3-call path, the other the fused builtin. Their
    // chain_heads must hex-equal byte-for-byte.
    reset_arena();
    let g_three = install_full_training_setup(42);
    let g_fused = install_full_training_setup(42);

    let xs: &[(f64, [f64; 4], f64)] = &[
        (0.10, [1.0, 0.5, 0.25, 0.125], 0.7),
        (0.45, [0.3, 0.6, 0.9, 1.2], 1.1),
        (0.80, [0.8, 0.4, 0.2, 0.1], 0.4),
        (0.55, [0.5, 0.5, 0.5, 0.5], 0.5),
    ];

    for &(x_val, phi, y_val) in xs {
        // ── Graph A: 3-call sequence ──────────────────────────────────
        let x_t = cjc_runtime::tensor::Tensor::from_vec(vec![x_val], &[1]).unwrap();
        let leaf_three = match call(
            "abng_route_to_leaf",
            &[Value::Int(g_three), Value::Tensor(x_t)],
        ) {
            Value::Int(i) => i,
            _ => panic!("expected int leaf"),
        };
        let phi_2d = cjc_runtime::tensor::Tensor::from_vec(phi.to_vec(), &[1, 4]).unwrap();
        let y_1d = cjc_runtime::tensor::Tensor::from_vec(vec![y_val], &[1]).unwrap();
        let _ = call(
            "abng_blr_update",
            &[
                Value::Int(g_three),
                Value::Int(leaf_three),
                Value::Tensor(phi_2d),
                Value::Tensor(y_1d),
            ],
        );
        let _ = call(
            "abng_observe",
            &[Value::Int(g_three), Value::Int(leaf_three), Value::Float(y_val)],
        );

        // ── Graph B: fused train_step ─────────────────────────────────
        let x_t2 = cjc_runtime::tensor::Tensor::from_vec(vec![x_val], &[1]).unwrap();
        let phi_1d = cjc_runtime::tensor::Tensor::from_vec(phi.to_vec(), &[4]).unwrap();
        let leaf_fused = match call(
            "abng_train_step",
            &[
                Value::Int(g_fused),
                Value::Tensor(x_t2),
                Value::Tensor(phi_1d),
                Value::Float(y_val),
            ],
        ) {
            Value::Int(i) => i,
            _ => panic!("expected int leaf"),
        };

        // Same leaf id and same chain head after each row.
        assert_eq!(
            leaf_three, leaf_fused,
            "leaf id divergence at x={x_val}"
        );
        let head_three = match call("abng_chain_head", &[Value::Int(g_three)]) {
            Value::String(s) => (*s).clone(),
            _ => panic!(),
        };
        let head_fused = match call("abng_chain_head", &[Value::Int(g_fused)]) {
            Value::String(s) => (*s).clone(),
            _ => panic!(),
        };
        assert_eq!(
            head_three, head_fused,
            "chain_head divergence after row x={x_val}, y={y_val}"
        );
    }
}

#[test]
fn train_step_arg_count_validation() {
    reset_arena();
    let _ = new_graph(0);
    let err = try_call("abng_train_step", &[Value::Int(0)]).unwrap_err();
    assert!(
        err.contains("expected 4 arguments"),
        "unexpected error: {err}"
    );
}

#[test]
fn train_step_dimension_errors_propagate() {
    reset_arena();
    let g = install_full_training_setup(0);
    // Wrong x dim (codebook expects 1-D length 1; pass length 2).
    let bad_x = cjc_runtime::tensor::Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
    let phi = cjc_runtime::tensor::Tensor::from_vec(vec![1.0, 0.5, 0.25, 0.125], &[4]).unwrap();
    let err = try_call(
        "abng_train_step",
        &[
            Value::Int(g),
            Value::Tensor(bad_x),
            Value::Tensor(phi),
            Value::Float(0.5),
        ],
    )
    .unwrap_err();
    assert!(
        err.contains("expected 1") || err.contains("arity") || err.contains("got 2"),
        "expected arity/dim error, got: {err}"
    );
}
