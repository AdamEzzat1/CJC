//! Unit tests — explicit shapes, arity errors, type errors, default ord.

use std::rc::Rc;

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn t2(data: &[f64], rows: usize, cols: usize) -> Value {
    assert_eq!(data.len(), rows * cols);
    Value::Tensor(Tensor::from_vec(data.to_vec(), &[rows, cols]).unwrap())
}

#[test]
fn fused_matmul_norm_dispatches_l2_default() {
    // a = identity 2x2, w = [[3, 0], [0, 4]]. matmul(a, w) = w, ||w||_2 = 5.
    let args = [
        t2(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        t2(&[3.0, 0.0, 0.0, 4.0], 2, 2),
    ];
    let result = dispatch_builtin("fused_matmul_norm", &args).unwrap().unwrap();
    match result {
        Value::Float(f) => assert_eq!(f, 5.0),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn fused_matmul_norm_l1_dispatches_with_explicit_ord() {
    // a = identity, w = [[1, 2], [3, 4]]. matmul = w. L1 = 1+2+3+4 = 10.
    let args = [
        t2(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        t2(&[1.0, 2.0, 3.0, 4.0], 2, 2),
        Value::Int(1),
    ];
    let result = dispatch_builtin("fused_matmul_norm", &args).unwrap().unwrap();
    match result {
        Value::Float(f) => assert_eq!(f, 10.0),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn fused_matmul_norm_accepts_ord_as_float() {
    // Float ord coerces to i64 (matches the `norm` builtin's behaviour).
    let args = [
        t2(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        t2(&[3.0, 0.0, 0.0, 4.0], 2, 2),
        Value::Float(2.0),
    ];
    let result = dispatch_builtin("fused_matmul_norm", &args).unwrap().unwrap();
    match result {
        Value::Float(f) => assert_eq!(f, 5.0),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn fused_matmul_norm_rejects_zero_args() {
    let args: [Value; 0] = [];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("2-3 arguments"), "got: {err}");
}

#[test]
fn fused_matmul_norm_rejects_too_many_args() {
    let args = [
        t2(&[1.0], 1, 1),
        t2(&[1.0], 1, 1),
        Value::Int(2),
        Value::Int(99),
    ];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("2-3 arguments"), "got: {err}");
}

#[test]
fn fused_matmul_norm_rejects_non_tensor_arg() {
    let args = [Value::Int(5), t2(&[1.0], 1, 1)];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("tensors"), "got: {err}");
}

#[test]
fn fused_matmul_norm_rejects_a_not_2d() {
    let args = [
        Value::Tensor(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()),
        t2(&[1.0, 2.0], 1, 2),
    ];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("a must be 2D"), "got: {err}");
}

#[test]
fn fused_matmul_norm_rejects_w_not_2d() {
    let args = [
        t2(&[1.0, 2.0], 1, 2),
        Value::Tensor(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()),
    ];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("w must be 2D"), "got: {err}");
}

#[test]
fn fused_matmul_norm_rejects_shape_mismatch() {
    // a cols (3) != w rows (2)
    let args = [
        t2(&[1.0, 2.0, 3.0], 1, 3),
        t2(&[1.0, 2.0], 2, 1),
    ];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("a cols"), "got: {err}");
}

#[test]
fn fused_matmul_norm_rejects_non_numeric_ord() {
    let args = [
        t2(&[1.0], 1, 1),
        t2(&[1.0], 1, 1),
        Value::String(Rc::new("two".to_string())),
    ];
    let err = dispatch_builtin("fused_matmul_norm", &args).unwrap_err();
    assert!(err.contains("ord must be"), "got: {err}");
}
