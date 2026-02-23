// CJC Test Suite — cjc-ad (12 tests)
// Source: crates/cjc-ad/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_ad::*;
use cjc_runtime::Tensor;

// ── Forward Mode Tests ──────────────────────────────────

#[test]
fn test_dual_add() {
    let a = Dual::variable(3.0);
    let b = Dual::constant(2.0);
    let c = a + b;
    assert_eq!(c.value, 5.0);
    assert_eq!(c.deriv, 1.0);
}

#[test]
fn test_dual_mul() {
    let a = Dual::variable(3.0);
    let b = Dual::constant(2.0);
    let c = a * b;
    assert_eq!(c.value, 6.0);
    assert_eq!(c.deriv, 2.0); // d/dx (x * 2) = 2
}

#[test]
fn test_dual_chain_rule() {
    // f(x) = x^2 + 2x + 1, f'(x) = 2x + 2, f'(3) = 8
    let x = Dual::variable(3.0);
    let result = x.clone() * x.clone() + Dual::constant(2.0) * x + Dual::one();
    assert_eq!(result.value, 16.0);
    assert_eq!(result.deriv, 8.0);
}

#[test]
fn test_dual_exp() {
    let x = Dual::variable(1.0);
    let result = x.exp();
    assert!((result.value - std::f64::consts::E).abs() < 1e-10);
    assert!((result.deriv - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_dual_sin_cos() {
    let x = Dual::variable(0.0);
    let sin_x = x.clone().sin();
    let cos_x = x.cos();
    assert!((sin_x.value - 0.0).abs() < 1e-10);
    assert!((sin_x.deriv - 1.0).abs() < 1e-10); // d/dx sin(x) at 0 = cos(0) = 1
    assert!((cos_x.value - 1.0).abs() < 1e-10);
    assert!((cos_x.deriv - 0.0).abs() < 1e-10); // d/dx cos(x) at 0 = -sin(0) = 0
}

#[test]
fn test_dual_div() {
    let a = Dual::variable(6.0);
    let b = Dual::constant(3.0);
    let c = a / b;
    assert_eq!(c.value, 2.0);
    assert!((c.deriv - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_finite_diff_validation() {
    // f(x) = x^2, f'(3) = 6
    let f = |x: f64| x * x;
    assert!(check_grad_finite_diff(f, 3.0, 6.0, 1e-7, 1e-5));
}

// ── Reverse Mode Tests ──────────────────────────────────

#[test]
fn test_reverse_add() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
    let b = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
    let c = g.add(a, b);

    g.backward(c);

    let ga = g.grad(a).unwrap();
    let gb = g.grad(b).unwrap();
    assert!((ga.to_vec()[0] - 1.0).abs() < 1e-10);
    assert!((gb.to_vec()[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_reverse_mul() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![3.0], &[1]));
    let b = g.parameter(Tensor::from_vec_unchecked(vec![2.0], &[1]));
    let c = g.mul(a, b);

    g.backward(c);

    let ga = g.grad(a).unwrap();
    let gb = g.grad(b).unwrap();
    assert!((ga.to_vec()[0] - 2.0).abs() < 1e-10); // d/da (a*b) = b = 2
    assert!((gb.to_vec()[0] - 3.0).abs() < 1e-10); // d/db (a*b) = a = 3
}

#[test]
fn test_reverse_matmul_gradient() {
    let mut g = GradGraph::new();

    // Simple 2x2 matmul
    let a = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = g.parameter(Tensor::from_vec_unchecked(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
    let c = g.matmul(a, b);
    let loss = g.sum(c);

    g.backward(loss);

    // Gradient of sum(A @ B) w.r.t. A = ones @ B^T
    let ga = g.grad(a).unwrap();
    let ga_data = ga.to_vec();
    assert!((ga_data[0] - 11.0).abs() < 1e-10);
    assert!((ga_data[1] - 15.0).abs() < 1e-10);
}

#[test]
fn test_reverse_mean_gradient() {
    let mut g = GradGraph::new();
    let a = g.parameter(Tensor::from_vec_unchecked(vec![2.0, 4.0, 6.0, 8.0], &[4]));
    let loss = g.mean(a);

    g.backward(loss);

    let ga = g.grad(a).unwrap();
    let ga_data = ga.to_vec();
    // d/da mean(a) = 1/N for each element
    for &v in &ga_data {
        assert!((v - 0.25).abs() < 1e-10);
    }
}

#[test]
fn test_reverse_mse_loss() {
    // MSE = mean((pred - target)^2)
    let mut g = GradGraph::new();

    let w = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 1.0], &[2, 1]));
    let x = g.input(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let target = g.input(Tensor::from_vec_unchecked(vec![3.0, 7.0], &[2, 1]));

    let pred = g.matmul(x, w);
    let diff = g.sub(pred, target);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq);

    let loss_val = g.value(loss);
    g.backward(loss);

    let gw = g.grad(w).unwrap();

    // Verify loss is finite and gradient exists
    assert!(loss_val.is_finite());
    assert_eq!(gw.to_vec().len(), 2);
    for &v in &gw.to_vec() {
        assert!(v.is_finite());
    }
}
