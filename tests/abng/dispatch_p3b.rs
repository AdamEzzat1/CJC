//! Phase 0.3b — dispatch-level tests for the BLR builtins.

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

fn install_head(g: i64) {
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
}

fn install_prior(g: i64) {
    let _ = call(
        "abng_set_blr_prior",
        &[
            Value::Int(g),
            Value::Float(1.0),
            Value::Float(1.5),
            Value::Float(1.0),
        ],
    );
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

fn expect_tensor_shape_data(v: Value) -> (Vec<usize>, Vec<f64>) {
    match v {
        Value::Tensor(t) => (t.shape().to_vec(), t.to_vec()),
        other => panic!("expected Tensor, got {}", other.type_name()),
    }
}

#[test]
fn set_blr_prior_basic() {
    reset_arena();
    let g = new_graph(0);
    install_head(g);
    install_prior(g);
    assert_eq!(expect_int(call("abng_blr_n_seen", &[Value::Int(g), Value::Int(0)])), 0);
}

#[test]
fn set_blr_prior_without_head_errs() {
    reset_arena();
    let g = new_graph(0);
    let err = try_call(
        "abng_set_blr_prior",
        &[
            Value::Int(g),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(1.0),
        ],
    )
    .unwrap_err();
    assert!(err.contains("must be installed *after* the leaf head"));
}

#[test]
fn set_blr_prior_invalid_params_err() {
    reset_arena();
    let g = new_graph(0);
    install_head(g);
    let err = try_call(
        "abng_set_blr_prior",
        &[
            Value::Int(g),
            Value::Float(0.0), // precision must be > 0
            Value::Float(1.0),
            Value::Float(1.0),
        ],
    )
    .unwrap_err();
    assert!(err.contains("precision > 0"));
}

#[test]
fn blr_update_increments_n_seen() {
    reset_arena();
    let g = new_graph(0);
    install_head(g);
    install_prior(g);
    // hidden=[4] means d=4. Pass two samples of dim-4 features.
    let features = Tensor::from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &[2, 4],
    )
    .unwrap();
    let y = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let _ = call(
        "abng_blr_update",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(features),
            Value::Tensor(y),
        ],
    );
    let n = expect_int(call("abng_blr_n_seen", &[Value::Int(g), Value::Int(0)]));
    assert_eq!(n, 2);
}

#[test]
fn blr_update_dim_mismatch_errs() {
    reset_arena();
    let g = new_graph(0);
    install_head(g);
    install_prior(g);
    // d=4 expected (last hidden layer width); pass d=2.
    let features = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let y = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let err = try_call(
        "abng_blr_update",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(features),
            Value::Tensor(y),
        ],
    )
    .unwrap_err();
    assert!(err.contains("doesn't match prior d"));
}

#[test]
fn blr_predict_returns_three_element_tensor() {
    reset_arena();
    let g = new_graph(0);
    install_head(g);
    install_prior(g);
    let phi = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();
    let (shape, data) = expect_tensor_shape_data(call(
        "abng_blr_predict",
        &[Value::Int(g), Value::Int(0), Value::Tensor(phi)],
    ));
    assert_eq!(shape, vec![3]);
    assert_eq!(data.len(), 3);
    let (mean, epi, ale) = (data[0], data[1], data[2]);
    assert!(mean.is_finite());
    assert!(epi >= 0.0);
    assert!(ale > 0.0);
}

#[test]
fn blr_state_hash_changes_after_update() {
    reset_arena();
    let g = new_graph(0);
    install_head(g);
    install_prior(g);
    let h0 = expect_string(call(
        "abng_blr_state_hash",
        &[Value::Int(g), Value::Int(0)],
    ));
    let features = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let y = Tensor::from_vec(vec![5.0], &[1]).unwrap();
    let _ = call(
        "abng_blr_update",
        &[
            Value::Int(g),
            Value::Int(0),
            Value::Tensor(features),
            Value::Tensor(y),
        ],
    );
    let h1 = expect_string(call(
        "abng_blr_state_hash",
        &[Value::Int(g), Value::Int(0)],
    ));
    assert_ne!(h0, h1);
    assert_eq!(h0.len(), 64);
}

#[test]
fn blr_features_returns_grad_idx() {
    use cjc_ad::dispatch::{reset_ambient, with_ambient};
    reset_arena();
    reset_ambient();
    let g = new_graph(0);
    install_head(g);
    install_prior(g);
    let x_idx = with_ambient(|gg| {
        gg.input(Tensor::from_vec(vec![0.5, -0.5], &[1, 2]).unwrap())
    }) as i64;
    let phi_idx = expect_int(call(
        "abng_blr_features",
        &[Value::Int(g), Value::Int(0), Value::Int(x_idx)],
    ));
    // phi_idx should produce a [1, 4] tensor on the ambient graph.
    let phi_shape = with_ambient(|gg| gg.tensor(phi_idx as usize).shape().to_vec());
    assert_eq!(phi_shape, vec![1, 4]);
}
