//! Phase 0.3a — dispatch-level tests for the new leaf-head builtins
//! (`abng_set_leaf_head`, `abng_leaf_head_dims`, `abng_leaf_param_count`,
//! `abng_leaf_param`, `abng_leaf_set_param`, `abng_leaf_params_hash`,
//! `abng_leaf_forward`).

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

fn expect_tensor(v: Value) -> (Vec<f64>, Vec<usize>) {
    match v {
        Value::Tensor(t) => (t.to_vec(), t.shape().to_vec()),
        other => panic!("expected Tensor, got {}", other.type_name()),
    }
}

fn expect_array(v: Value) -> Vec<Value> {
    match v {
        Value::Array(a) => (*a).clone(),
        other => panic!("expected Array, got {}", other.type_name()),
    }
}

// ─── Configuration ────────────────────────────────────────────────

#[test]
fn set_leaf_head_basic() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    // Param count = 4 (2 layers × {W, b}).
    let n = expect_int(call(
        "abng_leaf_param_count",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert_eq!(n, 4);
}

#[test]
fn set_leaf_head_unknown_activation_errs() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let err = try_call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("not_a_real_activation".to_string())),
        ],
    )
    .unwrap_err();
    assert!(err.contains("unknown activation"));
}

#[test]
fn set_leaf_head_twice_errs() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden.clone()),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    let err = try_call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    )
    .unwrap_err();
    assert!(err.contains("already frozen"));
}

#[test]
fn set_leaf_head_after_add_node_errs() {
    reset_arena();
    let g = new_graph(0);
    let _ = call(
        "abng_add_node",
        &[Value::Int(g), Value::Int(0), Value::Int(7)],
    );
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let err = try_call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    )
    .unwrap_err();
    assert!(err.contains("must be installed before any add_node"));
}

#[test]
fn leaf_head_dims_empty_when_unset() {
    reset_arena();
    let g = new_graph(0);
    let (data, _shape) = expect_tensor(call(
        "abng_leaf_head_dims",
        &[Value::Int(g)],
    ));
    assert_eq!(data.len(), 0);
}

#[test]
fn leaf_head_dims_layout() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0, 8.0], &[2]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(3),
            Value::Tensor(hidden),
            Value::Int(2),
            Value::String(std::rc::Rc::new("relu".to_string())),
        ],
    );
    let (data, _shape) = expect_tensor(call(
        "abng_leaf_head_dims",
        &[Value::Int(g)],
    ));
    // Layout: [input_dim=3, n_hidden=2, hidden_dims..=4,8, output_dim=2, activation_tag]
    assert_eq!(data, vec![3.0, 2.0, 4.0, 8.0, 2.0, 0x02 as f64]);
}

// ─── Param read/write ─────────────────────────────────────────────

#[test]
fn leaf_param_count_zero_when_no_head() {
    reset_arena();
    let g = new_graph(0);
    let n = expect_int(call(
        "abng_leaf_param_count",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert_eq!(n, 0);
}

#[test]
fn leaf_param_returns_initialized_weight() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    let (data, shape) = expect_tensor(call(
        "abng_leaf_param",
        &[Value::Int(g), Value::Int(0), Value::Int(0)],
    ));
    // W_1 shape is [4, 2]; data is non-trivial after Xavier init.
    assert_eq!(shape, vec![4, 2]);
    assert_eq!(data.len(), 8);
    let nonzero_count = data.iter().filter(|&&x| x != 0.0).count();
    assert!(nonzero_count > 0, "Xavier init should produce non-zero weights");
}

#[test]
fn leaf_set_param_writes_back() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    let new_w = Tensor::from_vec(vec![0.5; 8], &[4, 2]).unwrap();
    let _ = call(
        "abng_leaf_set_param",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Int(0),
            Value::Tensor(new_w),
        ],
    );
    let (data, _) = expect_tensor(call(
        "abng_leaf_param",
        &[Value::Int(g), Value::Int(0), Value::Int(0)],
    ));
    assert!(data.iter().all(|&x| x == 0.5));
}

#[test]
fn leaf_set_param_shape_mismatch_errs() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    let bad = Tensor::from_vec(vec![0.0; 7], &[7]).unwrap();
    let err = try_call(
        "abng_leaf_set_param",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Int(0),
            Value::Tensor(bad),
        ],
    )
    .unwrap_err();
    assert!(err.contains("expected shape"));
}

#[test]
fn leaf_params_hash_changes_after_update() {
    reset_arena();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );
    let h_before = expect_string(call(
        "abng_leaf_params_hash",
        &[Value::Int(g), Value::Int(0)],
    ));
    let new_w = Tensor::from_vec(vec![0.42; 8], &[4, 2]).unwrap();
    let _ = call(
        "abng_leaf_set_param",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Int(0),
            Value::Tensor(new_w),
        ],
    );
    let h_after = expect_string(call(
        "abng_leaf_params_hash",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert_ne!(h_before, h_after);
    assert_eq!(h_before.len(), 64);
    assert_eq!(h_after.len(), 64);
}

// ─── Forward into ambient GradGraph ───────────────────────────────

#[test]
fn leaf_forward_returns_y_plus_param_indices() {
    use cjc_ad::dispatch::{reset_ambient, with_ambient};
    reset_arena();
    reset_ambient();
    let g = new_graph(0);
    let hidden = Tensor::from_vec(vec![4.0], &[1]).unwrap();
    let _ = call(
        "abng_set_leaf_head",
        &[
            Value::Int(g),
            Value::Int(2),
            Value::Tensor(hidden),
            Value::Int(1),
            Value::String(std::rc::Rc::new("tanh".to_string())),
        ],
    );

    // Add an input on the ambient graph.
    let x_idx = with_ambient(|gg| {
        gg.input(Tensor::from_vec(vec![0.5, -0.5], &[1, 2]).unwrap())
    }) as i64;

    let result = expect_array(call(
        "abng_leaf_forward",
        &[Value::Int(g), Value::Int(0), Value::Int(x_idx)],
    ));
    // Length: 1 (y) + 4 params = 5
    assert_eq!(result.len(), 5);
    let y_idx = match result[0] {
        Value::Int(i) => i,
        _ => panic!(),
    };
    // Output tensor on ambient graph has shape [1, 1]
    let y_shape = with_ambient(|gg| gg.tensor(y_idx as usize).shape().to_vec());
    assert_eq!(y_shape, vec![1, 1]);
}
