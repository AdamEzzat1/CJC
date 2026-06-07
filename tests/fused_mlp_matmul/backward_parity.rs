//! Backward parity: gradients from `mlp_layer_matmul` match those from the
//! unfused `matmul(mlp_layer(...), w2)` chain bit-for-bit, across all
//! parameters (input, w1, b1, w2).
//!
//! This is the load-bearing correctness test for Phase 3.5b. The hand-
//! derived backward must yield identical gradients to the chain of
//! existing per-op backward implementations.

use cjc_ad::pinn::Activation;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

/// Identical setup helper as forward_parity (kept duplicated so the two
/// files stay self-contained — a future refactor could move both into a
/// shared `helpers.rs` once the suite grows).
fn setup(
    graph: &mut GradGraph,
    batch: usize,
    in_f: usize,
    hidden: usize,
    out_f: usize,
) -> (usize, usize, usize, usize) {
    let input_data: Vec<f64> = (0..batch * in_f).map(|i| (i as f64) * 0.1).collect();
    let w1_data: Vec<f64> = (0..hidden * in_f).map(|i| (i as f64) * 0.05).collect();
    let b1_data: Vec<f64> = (0..hidden).map(|i| (i as f64) * 0.01).collect();
    let w2_data: Vec<f64> = (0..hidden * out_f).map(|i| (i as f64) * 0.03).collect();

    let input = graph.input(Tensor::from_vec_unchecked(input_data, &[batch, in_f]));
    let w1 = graph.parameter(Tensor::from_vec_unchecked(w1_data, &[hidden, in_f]));
    let b1 = graph.parameter(Tensor::from_vec_unchecked(b1_data, &[hidden]));
    let w2 = graph.parameter(Tensor::from_vec_unchecked(w2_data, &[hidden, out_f]));
    (input, w1, b1, w2)
}

/// Run forward + sum-reduction + backward through BOTH paths and return
/// (grad_w1, grad_b1, grad_w2) for parity comparison. We use sum as the
/// loss so the seed gradient is well-defined (ones_like(output)).
fn fused_vs_unfused_grads(
    batch: usize,
    in_f: usize,
    hidden: usize,
    out_f: usize,
    act: Activation,
) -> ((Vec<f64>, Vec<f64>, Vec<f64>), (Vec<f64>, Vec<f64>, Vec<f64>)) {
    // Fused
    let (gw1_f, gb1_f, gw2_f) = {
        let mut g = GradGraph::new();
        let (input, w1, b1, w2) = setup(&mut g, batch, in_f, hidden, out_f);
        let out = g.mlp_layer_matmul(input, w1, b1, act, w2);
        let loss = g.sum(out);
        g.zero_grad();
        g.backward(loss);
        (
            g.grad(w1).map(|t| t.to_vec()).unwrap_or_default(),
            g.grad(b1).map(|t| t.to_vec()).unwrap_or_default(),
            g.grad(w2).map(|t| t.to_vec()).unwrap_or_default(),
        )
    };

    // Unfused
    let (gw1_u, gb1_u, gw2_u) = {
        let mut g = GradGraph::new();
        let (input, w1, b1, w2) = setup(&mut g, batch, in_f, hidden, out_f);
        let h = g.mlp_layer(input, w1, b1, act);
        let out = g.matmul(h, w2);
        let loss = g.sum(out);
        g.zero_grad();
        g.backward(loss);
        (
            g.grad(w1).map(|t| t.to_vec()).unwrap_or_default(),
            g.grad(b1).map(|t| t.to_vec()).unwrap_or_default(),
            g.grad(w2).map(|t| t.to_vec()).unwrap_or_default(),
        )
    };

    ((gw1_f, gb1_f, gw2_f), (gw1_u, gb1_u, gw2_u))
}

fn assert_bit_identical(a: &[f64], b: &[f64], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch ({} vs {})", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: fused={x:e}, unfused={y:e}"
        );
    }
}

#[test]
fn backward_grads_match_tanh() {
    let (fused, unfused) = fused_vs_unfused_grads(2, 3, 4, 2, Activation::Tanh);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 tanh");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 tanh");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 tanh");
}

#[test]
fn backward_grads_match_sigmoid() {
    let (fused, unfused) = fused_vs_unfused_grads(1, 2, 3, 1, Activation::Sigmoid);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 sigmoid");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 sigmoid");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 sigmoid");
}

#[test]
fn backward_grads_match_relu() {
    let (fused, unfused) = fused_vs_unfused_grads(3, 2, 4, 2, Activation::Relu);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 relu");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 relu");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 relu");
}

#[test]
fn backward_grads_match_none() {
    let (fused, unfused) = fused_vs_unfused_grads(2, 2, 3, 2, Activation::None);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 none");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 none");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 none");
}

#[test]
fn backward_grads_match_gelu() {
    let (fused, unfused) = fused_vs_unfused_grads(2, 3, 4, 1, Activation::Gelu);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 gelu");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 gelu");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 gelu");
}

#[test]
fn backward_grads_match_silu() {
    let (fused, unfused) = fused_vs_unfused_grads(1, 4, 5, 2, Activation::Silu);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 silu");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 silu");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 silu");
}

#[test]
fn backward_grads_match_elu() {
    let (fused, unfused) = fused_vs_unfused_grads(2, 3, 3, 2, Activation::Elu);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 elu");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 elu");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 elu");
}

#[test]
fn backward_grads_match_selu() {
    let (fused, unfused) = fused_vs_unfused_grads(2, 2, 4, 2, Activation::Selu);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 selu");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 selu");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 selu");
}

#[test]
fn backward_grads_match_sin() {
    let (fused, unfused) = fused_vs_unfused_grads(1, 2, 2, 1, Activation::SinAct);
    assert_bit_identical(&fused.0, &unfused.0, "grad_w1 sin");
    assert_bit_identical(&fused.1, &unfused.1, "grad_b1 sin");
    assert_bit_identical(&fused.2, &unfused.2, "grad_w2 sin");
}
