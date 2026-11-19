//! The gradient through `transpose_op` reaches a parameter as a transposed view.
//! It must land element for element, on both backward paths.

use cjc_ad::GradGraph;
use cjc_runtime::Tensor;

fn graph() -> (GradGraph, usize, usize) {
    let mut g = GradGraph::new();
    let p = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    // w has shape [3, 2]; loss = sum(transpose(p) * w), so d loss / d p = transpose(w).
    let w = g.input(Tensor::from_vec_unchecked(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]));
    let pt = g.transpose_op(p);
    let prod = g.mul(pt, w);
    let loss = g.sum(prod);
    (g, p, loss)
}

#[test]
fn transpose_gradient_is_the_transposed_weight() {
    let (mut g, p, loss) = graph();
    g.zero_grad();
    g.backward(loss);
    let grad = g.grad(p).expect("parameter has a gradient");
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(grad.to_vec(), vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0]);
}

#[test]
fn backward_with_seed_agrees_bit_for_bit() {
    let (mut g, p, loss) = graph();
    g.zero_grad();
    g.backward(loss);
    let main: Vec<u64> = g.grad(p).unwrap().to_vec().iter().map(|v| v.to_bits()).collect();
    g.zero_grad();
    g.backward_with_seed(loss, &Tensor::from_vec_unchecked(vec![1.0], &[1]));
    let seeded: Vec<u64> = g.grad(p).unwrap().to_vec().iter().map(|v| v.to_bits()).collect();
    assert_eq!(main, seeded);
}
