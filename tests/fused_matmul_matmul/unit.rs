//! Unit tests — explicit shapes + arity/shape error paths.

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn t2(data: &[f64], rows: usize, cols: usize) -> Value {
    assert_eq!(data.len(), rows * cols);
    Value::Tensor(Tensor::from_vec(data.to_vec(), &[rows, cols]).unwrap())
}

#[test]
fn fused_matmul_matmul_dispatches_with_identity_chain() {
    // I @ I @ M = M
    let args = [
        t2(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        t2(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        t2(&[3.0, 4.0, 5.0, 6.0], 2, 2),
    ];
    let result = dispatch_builtin("fused_matmul_matmul", &args).unwrap().unwrap();
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.to_vec(), vec![3.0, 4.0, 5.0, 6.0]);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn fused_matmul_matmul_rejects_wrong_arity() {
    let args = [t2(&[1.0], 1, 1), t2(&[1.0], 1, 1)];
    let err = dispatch_builtin("fused_matmul_matmul", &args).unwrap_err();
    assert!(err.contains("3 arguments"), "got: {err}");
}

#[test]
fn fused_matmul_matmul_rejects_a_not_2d() {
    let args = [
        Value::Tensor(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()),
        t2(&[1.0], 1, 1),
        t2(&[1.0], 1, 1),
    ];
    let err = dispatch_builtin("fused_matmul_matmul", &args).unwrap_err();
    assert!(err.contains("a must be 2D"), "got: {err}");
}

#[test]
fn fused_matmul_matmul_rejects_b_not_2d() {
    let args = [
        t2(&[1.0], 1, 1),
        Value::Tensor(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()),
        t2(&[1.0], 1, 1),
    ];
    let err = dispatch_builtin("fused_matmul_matmul", &args).unwrap_err();
    assert!(err.contains("b must be 2D"), "got: {err}");
}

#[test]
fn fused_matmul_matmul_rejects_c_not_2d() {
    let args = [
        t2(&[1.0], 1, 1),
        t2(&[1.0], 1, 1),
        Value::Tensor(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap()),
    ];
    let err = dispatch_builtin("fused_matmul_matmul", &args).unwrap_err();
    assert!(err.contains("c must be 2D"), "got: {err}");
}

#[test]
fn fused_matmul_matmul_rejects_ab_shape_mismatch() {
    // a is [1, 3], b is [2, 1] — inner dim 3 vs 2 mismatch
    let args = [
        t2(&[1.0, 2.0, 3.0], 1, 3),
        t2(&[1.0, 2.0], 2, 1),
        t2(&[1.0], 1, 1),
    ];
    let err = dispatch_builtin("fused_matmul_matmul", &args).unwrap_err();
    assert!(err.contains("a cols"), "got: {err}");
}

#[test]
fn fused_matmul_matmul_rejects_bc_shape_mismatch() {
    // b is [2, 3], c is [2, 1] — inner dim 3 vs 2 mismatch
    let args = [
        t2(&[1.0, 2.0], 1, 2),
        t2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
        t2(&[1.0, 2.0], 2, 1),
    ];
    let err = dispatch_builtin("fused_matmul_matmul", &args).unwrap_err();
    assert!(err.contains("b cols"), "got: {err}");
}

#[test]
fn fused_matmul_matmul_2x3x4x2_shape() {
    // a: [2, 3], b: [3, 4], c: [4, 2]. Result: [2, 2].
    let args = [
        t2(&[1.0; 6], 2, 3),
        t2(&[2.0; 12], 3, 4),
        t2(&[0.5; 8], 4, 2),
    ];
    let result = dispatch_builtin("fused_matmul_matmul", &args).unwrap().unwrap();
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}
