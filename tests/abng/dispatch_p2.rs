//! Phase 0.2 — dispatch-level tests for the new `abng_*` builtins
//! (`abng_add_node`, `abng_node_parent`, `abng_node_kind`,
//! `abng_node_child_count`, `abng_node_child`, `abng_set_codebook`,
//! `abng_codebook_dims`, `abng_codebook_hash`, `abng_encode_prefix`,
//! `abng_descend`, `abng_route_path`, `abng_node_stats_chain_head`).

use std::cell::RefCell;
use std::rc::Rc;

use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_runtime::tensor::Tensor;
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
        _ => panic!(),
    }
}

fn expect_int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected Int, got {}", other.type_name()),
    }
}

fn expect_string(v: Value) -> String {
    match v {
        Value::String(s) => (*s).clone(),
        other => panic!("expected String, got {}", other.type_name()),
    }
}

fn expect_tensor(v: Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.to_vec(),
        other => panic!("expected Tensor, got {}", other.type_name()),
    }
}

#[test]
fn add_node_returns_new_id() {
    reset_arena();
    let g = new_graph(0);
    let n = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    ));
    assert_eq!(n, 1);
    assert_eq!(expect_int(call("abng_node_count", &[Value::Int(g)])), 2);
}

#[test]
fn add_node_duplicate_key_errs() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    );
    let err = try_call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    )
    .unwrap_err();
    assert!(err.contains("already has a child"));
}

#[test]
fn add_node_rejects_out_of_range_byte() {
    reset_arena();
    let g = new_graph(0);
    let err = try_call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(256)],
    )
    .unwrap_err();
    assert!(err.contains("[0, 255]"));
}

#[test]
fn node_parent_returns_negative_one_for_root() {
    reset_arena();
    let g = new_graph(0);
    assert_eq!(
        expect_int(call("abng_node_parent", &[Value::Int(g), Value::Int(0)])),
        -1
    );
}

#[test]
fn node_parent_returns_parent_id_for_child() {
    reset_arena();
    let g = new_graph(0);
    let n = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(5)],
    ));
    assert_eq!(
        expect_int(call("abng_node_parent", &[Value::Int(g), Value::Int(n)])),
        0
    );
}

#[test]
fn node_kind_progression_through_promotions() {
    reset_arena();
    let g = new_graph(0);
    // No children → kind 0 (None).
    assert_eq!(
        expect_int(call("abng_node_kind", &[Value::Int(g), Value::Int(0)])),
        0
    );
    // 1 child → kind 1 (Node4).
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(0)],
    );
    assert_eq!(
        expect_int(call("abng_node_kind", &[Value::Int(g), Value::Int(0)])),
        1
    );
    // 17 children → kind 3 (Node48). 5th promotes to Node16, 17th to Node48.
    for k in 1..17i64 {
        let _ = call(
            "abng_add_node",
            &[Value::Int(g), Value::Int(0), Value::Int(k)],
        );
    }
    assert_eq!(
        expect_int(call("abng_node_kind", &[Value::Int(g), Value::Int(0)])),
        3
    );
}

#[test]
fn node_child_returns_minus_one_for_unbound() {
    reset_arena();
    let g = new_graph(0);
    assert_eq!(
        expect_int(call(
            "abng_node_child",
            &[Value::Int(g), Value::Int(0), Value::Int(99)]
        )),
        -1
    );
}

#[test]
fn node_child_returns_id_for_bound() {
    reset_arena();
    let g = new_graph(0);
    let n = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    ));
    assert_eq!(
        expect_int(call(
            "abng_node_child",
            &[Value::Int(g), Value::Int(0), Value::Int(7)]
        )),
        n
    );
}

#[test]
fn node_child_count_matches_inserts() {
    reset_arena();
    let g = new_graph(0);
    for k in 0i64..7 {
        let _ = call(
            "abng_add_node",
            &[Value::Int(g), Value::Int(0), Value::Int(k)],
        );
    }
    assert_eq!(
        expect_int(call(
            "abng_node_child_count",
            &[Value::Int(g), Value::Int(0)]
        )),
        7
    );
}

// ─── Codebook ──────────────────────────────────────────────────────

fn boundaries_2d(n_dims: usize, n_bins: u16) -> Tensor {
    // boundaries = [0.5, 1.5, ..., n_bins-1.5] per dim
    let per_dim = (n_bins - 1) as usize;
    let mut data = Vec::new();
    for _ in 0..n_dims {
        for k in 1..n_bins {
            data.push(k as f64 - 0.5);
        }
    }
    Tensor::from_vec(data, &[n_dims, per_dim]).unwrap()
}

#[test]
fn codebook_dims_zero_when_unset() {
    reset_arena();
    let g = new_graph(0);
    assert_eq!(
        expect_int(call("abng_codebook_dims", &[Value::Int(g)])),
        0
    );
}

#[test]
fn codebook_hash_empty_when_unset() {
    reset_arena();
    let g = new_graph(0);
    let h = expect_string(call("abng_codebook_hash", &[Value::Int(g)]));
    assert!(h.is_empty());
}

#[test]
fn set_codebook_updates_dims_and_hash() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(boundaries_2d(3, 4))],
    );
    assert_eq!(
        expect_int(call("abng_codebook_dims", &[Value::Int(g)])),
        3
    );
    let h = expect_string(call("abng_codebook_hash", &[Value::Int(g)]));
    assert_eq!(h.len(), 64);
}

#[test]
fn codebook_install_is_one_shot() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(boundaries_2d(2, 4))],
    );
    let err = try_call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(boundaries_2d(2, 4))],
    )
    .unwrap_err();
    assert!(err.contains("already frozen"));
}

#[test]
fn encode_prefix_returns_expected_bin_indices() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_set_codebook",
        &[Value::Int(g), Value::Tensor(boundaries_2d(2, 4))],
    );
    let x = Tensor::from_vec(vec![0.0, 2.0], &[2]).unwrap();
    let p = expect_tensor(call(
        "abng_encode_prefix",
        &[Value::Int(g), Value::Tensor(x)],
    ));
    assert_eq!(p, vec![0.0, 2.0]);
}

#[test]
fn encode_prefix_without_codebook_errs() {
    reset_arena();
    let g = new_graph(0);
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let err = try_call(
        "abng_encode_prefix",
        &[Value::Int(g), Value::Tensor(x)],
    )
    .unwrap_err();
    assert!(err.contains("no codebook"));
}

// ─── Descend / route_path ─────────────────────────────────────────

#[test]
fn descend_root_only_when_first_byte_unbound() {
    reset_arena();
    let g = new_graph(0);
    let prefix = Tensor::from_vec(vec![5.0, 1.0], &[2]).unwrap();
    let r = expect_tensor(call("abng_descend", &[Value::Int(g), Value::Tensor(prefix)]));
    assert_eq!(r, vec![0.0, 0.0]); // matched=0, leaf=root
}

#[test]
fn descend_two_hops_match() {
    reset_arena();
    let g = new_graph(0);
    let n1 = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    ));
    let n2 = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(n1), Value::Int(11)],
    ));
    let prefix = Tensor::from_vec(vec![7.0, 11.0], &[2]).unwrap();
    let r = expect_tensor(call("abng_descend", &[Value::Int(g), Value::Tensor(prefix)]));
    assert_eq!(r, vec![2.0, n2 as f64]);
}

#[test]
fn route_path_returns_full_traversal() {
    reset_arena();
    let g = new_graph(0);
    let n1 = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(2)],
    ));
    let n2 = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(n1), Value::Int(4)],
    ));
    let prefix = Tensor::from_vec(vec![2.0, 4.0], &[2]).unwrap();
    let path = expect_tensor(call(
        "abng_route_path",
        &[Value::Int(g), Value::Tensor(prefix)],
    ));
    assert_eq!(path, vec![0.0, n1 as f64, n2 as f64]);
}

#[test]
fn descend_rejects_non_byte_prefix() {
    reset_arena();
    let g = new_graph(0);
    let prefix = Tensor::from_vec(vec![1.5, 2.0], &[2]).unwrap();
    let err = try_call(
        "abng_descend",
        &[Value::Int(g), Value::Tensor(prefix)],
    )
    .unwrap_err();
    assert!(err.contains("not an integer byte"));
}

// ─── Per-node stats chain ─────────────────────────────────────────

#[test]
fn per_node_chain_advances_only_on_observation() {
    reset_arena();
    let g = new_graph(0);
    let n1 = expect_int(call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    ));
    let h_n0_before = expect_string(call(
        "abng_node_stats_chain_head",
        &[Value::Int(g), Value::Int(0)],
    ));
    let h_n1_before = expect_string(call(
        "abng_node_stats_chain_head",
        &[Value::Int(g), Value::Int(n1)],
    ));
    // Observe only node 1.
    let _ = call(
        "abng_observe",
        &[Value::Int(g), Value::Int(n1), Value::Float(5.0)],
    );
    let h_n0_after = expect_string(call(
        "abng_node_stats_chain_head",
        &[Value::Int(g), Value::Int(0)],
    ));
    let h_n1_after = expect_string(call(
        "abng_node_stats_chain_head",
        &[Value::Int(g), Value::Int(n1)],
    ));
    assert_eq!(h_n0_before, h_n0_after, "node 0 chain advanced unexpectedly");
    assert_ne!(h_n1_before, h_n1_after, "node 1 chain didn't advance");
}

#[test]
fn serialize_replay_round_trip_multinode_via_dispatch() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(1)],
    );
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(2)],
    );
    let _ = call(
        "abng_observe",
        &[Value::Int(g), Value::Int(0), Value::Float(1.0)],
    );
    let head_before = expect_string(call("abng_chain_head", &[Value::Int(g)]));
    let blob = match call("abng_serialize", &[Value::Int(g)]) {
        Value::Bytes(b) => b.borrow().clone(),
        _ => panic!(),
    };
    let g2 = expect_int(call(
        "abng_replay",
        &[Value::Bytes(Rc::new(RefCell::new(blob)))],
    ));
    let head_after = expect_string(call("abng_chain_head", &[Value::Int(g2)]));
    assert_eq!(head_before, head_after);
}
