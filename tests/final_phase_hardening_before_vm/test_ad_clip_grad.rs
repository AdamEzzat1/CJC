//! AD gradient clipping tests.

use cjc_runtime::tensor::Tensor;
use cjc_ad::GradGraph;

#[test]
fn test_clip_grad_basic() {
    let mut graph = GradGraph::new();
    let a = graph.variable(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0], &[3]));
    let b = graph.variable(Tensor::from_vec_unchecked(vec![10.0, 20.0, 30.0], &[3]));
    let c = graph.mul(a, b);
    let loss = graph.sum(c);
    graph.backward(loss);

    // Before clipping, gradients should be the original values
    let grad_a = graph.grad(a).unwrap();
    let grad_a_data = grad_a.to_vec();
    assert!((grad_a_data[0] - 10.0).abs() < 1e-10);
    assert!((grad_a_data[1] - 20.0).abs() < 1e-10);
    assert!((grad_a_data[2] - 30.0).abs() < 1e-10);

    // Clip to max_norm=15.0
    graph.clip_grad(15.0);

    let clipped = graph.grad(a).unwrap();
    let clipped_data = clipped.to_vec();
    // 10.0 should stay as-is (within range)
    assert!((clipped_data[0] - 10.0).abs() < 1e-10);
    // 20.0 should be clipped to 15.0
    assert!((clipped_data[1] - 15.0).abs() < 1e-10);
    // 30.0 should be clipped to 15.0
    assert!((clipped_data[2] - 15.0).abs() < 1e-10);
}

#[test]
fn test_clip_grad_norm() {
    let mut graph = GradGraph::new();
    let a = graph.variable(Tensor::from_vec_unchecked(vec![3.0, 4.0], &[2]));
    let b = graph.variable(Tensor::from_vec_unchecked(vec![100.0, 200.0], &[2]));
    let c = graph.mul(a, b);
    let loss = graph.sum(c);
    graph.backward(loss);

    // grad_a = [100, 200], norm = sqrt(10000 + 40000) = sqrt(50000) ≈ 223.6
    let global_norm = graph.clip_grad_norm(10.0);
    assert!(global_norm > 200.0, "global norm should be large: {}", global_norm);

    // After clipping, the global norm should be approximately 10.0
    let grad_a = graph.grad(a).unwrap();
    let data = grad_a.to_vec();
    let norm_after: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    // Allow some tolerance due to other nodes also having gradients
    // that contribute to the global norm
    assert!(norm_after <= 10.0 + 1e-6, "clipped norm should be <= 10: {}", norm_after);
}

#[test]
fn test_clip_grad_no_clip_needed() {
    let mut graph = GradGraph::new();
    let a = graph.variable(Tensor::from_vec_unchecked(vec![1.0, 2.0], &[2]));
    let loss = graph.sum(a);
    graph.backward(loss);

    // grad = [1, 1], norm = sqrt(2) ≈ 1.41
    let global_norm = graph.clip_grad_norm(100.0);
    assert!(global_norm < 2.0, "norm should be small: {}", global_norm);

    let grad_a = graph.grad(a).unwrap();
    let data = grad_a.to_vec();
    // Should remain unchanged
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[1] - 1.0).abs() < 1e-10);
}
