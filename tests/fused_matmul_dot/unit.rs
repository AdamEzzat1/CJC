//! Unit tests — explicit shapes, arity errors, type errors.
//!
//! Kernel-level math correctness lives in
//! `crates/cjc-runtime/src/accumulator.rs` (the `fused_matmul_dot_*` tests).
//! These tests cover the dispatch layer above the kernel: error messages,
//! shape validation, type rejection.

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn t1(data: &[f64]) -> Value {
    Value::Tensor(Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap())
}

fn t2(data: &[f64], rows: usize, cols: usize) -> Value {
    assert_eq!(data.len(), rows * cols);
    Value::Tensor(Tensor::from_vec(data.to_vec(), &[rows, cols]).unwrap())
}

#[test]
fn fused_matmul_dot_dispatches_with_correct_result() {
    // a = [1, 2], W = [[1, 2, 3], [4, 5, 6]], v = [1, 1, 1]
    // a @ W = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    // sum(. * v) = 9 + 12 + 15 = 36
    let args = [
        t1(&[1.0, 2.0]),
        t2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
        t1(&[1.0, 1.0, 1.0]),
    ];
    let result = dispatch_builtin("fused_matmul_dot", &args).unwrap().unwrap();
    match result {
        Value::Float(f) => assert_eq!(f, 36.0),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn fused_matmul_dot_rejects_wrong_arity() {
    let args = [t1(&[1.0, 2.0]), t2(&[3.0], 1, 1)];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("3 arguments"), "got: {err}");
}

#[test]
fn fused_matmul_dot_rejects_non_tensor_arg() {
    let args = [Value::Int(5), t2(&[1.0], 1, 1), t1(&[1.0])];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("tensors"), "got: {err}");
}

#[test]
fn fused_matmul_dot_rejects_a_not_1d() {
    // a passed as 2D
    let args = [
        t2(&[1.0, 2.0], 1, 2),
        t2(&[1.0, 2.0], 2, 1),
        t1(&[1.0]),
    ];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("a must be 1D"), "got: {err}");
}

#[test]
fn fused_matmul_dot_rejects_w_not_2d() {
    let args = [
        t1(&[1.0, 2.0]),
        t1(&[1.0, 2.0]), // w is 1D, should be 2D
        t1(&[1.0]),
    ];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("w must be 2D"), "got: {err}");
}

#[test]
fn fused_matmul_dot_rejects_v_not_1d() {
    let args = [
        t1(&[1.0]),
        t2(&[1.0], 1, 1),
        t2(&[1.0], 1, 1), // v is 2D, should be 1D
    ];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("v must be 1D"), "got: {err}");
}

#[test]
fn fused_matmul_dot_rejects_shape_mismatch_a_w() {
    // a length 3, W rows 2 → mismatch
    let args = [
        t1(&[1.0, 2.0, 3.0]),
        t2(&[1.0, 2.0], 2, 1),
        t1(&[1.0]),
    ];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("a length"), "got: {err}");
}

#[test]
fn fused_matmul_dot_rejects_shape_mismatch_w_v() {
    // W cols 2, v length 3 → mismatch
    let args = [
        t1(&[1.0]),
        t2(&[1.0, 2.0], 1, 2),
        t1(&[1.0, 2.0, 3.0]),
    ];
    let err = dispatch_builtin("fused_matmul_dot", &args).unwrap_err();
    assert!(err.contains("w cols"), "got: {err}");
}
