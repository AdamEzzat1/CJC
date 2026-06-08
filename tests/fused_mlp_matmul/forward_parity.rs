//! Forward parity tests: `mlp_layer_matmul(...)` output == `matmul(mlp_layer(...), w2)`
//! bit-for-bit. Verifies the fused op produces the same intermediate
//! mathematics as the unfused 2-op chain.

use cjc_ad::pinn::Activation;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

/// Build a small (batch, in, hidden, out) test setup and return the
/// (input_idx, w1_idx, b1_idx, w2_idx) into a fresh graph.
fn setup(
    graph: &mut GradGraph,
    batch: usize,
    in_f: usize,
    hidden: usize,
    out_f: usize,
) -> (usize, usize, usize, usize) {
    // Deterministic seed-based initialisation so tests are reproducible.
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

/// Run both fused and unfused paths and return their output tensors as
/// flat Vec<f64> for bit-identical comparison.
fn fused_vs_unfused(
    batch: usize,
    in_f: usize,
    hidden: usize,
    out_f: usize,
    act: Activation,
) -> (Vec<f64>, Vec<f64>) {
    // Fused
    let mut g_fused = GradGraph::new();
    let (input, w1, b1, w2) = setup(&mut g_fused, batch, in_f, hidden, out_f);
    let fused_out = g_fused.mlp_layer_matmul(input, w1, b1, act, w2);
    let fused_data = g_fused.tensor(fused_out).to_vec();

    // Unfused
    let mut g_un = GradGraph::new();
    let (input_u, w1_u, b1_u, w2_u) = setup(&mut g_un, batch, in_f, hidden, out_f);
    let h_u = g_un.mlp_layer(input_u, w1_u, b1_u, act);
    let unfused_out = g_un.matmul(h_u, w2_u);
    let unfused_data = g_un.tensor(unfused_out).to_vec();

    (fused_data, unfused_data)
}

fn assert_bit_identical(a: &[f64], b: &[f64], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: element {i} differs: fused={x:e}, unfused={y:e}"
        );
    }
}

#[test]
fn fused_forward_matches_unfused_tanh() {
    let (fused, unfused) = fused_vs_unfused(2, 3, 4, 2, Activation::Tanh);
    assert_bit_identical(&fused, &unfused, "tanh");
}

#[test]
fn fused_forward_matches_unfused_sigmoid() {
    let (fused, unfused) = fused_vs_unfused(1, 2, 3, 1, Activation::Sigmoid);
    assert_bit_identical(&fused, &unfused, "sigmoid");
}

#[test]
fn fused_forward_matches_unfused_relu() {
    let (fused, unfused) = fused_vs_unfused(3, 2, 4, 2, Activation::Relu);
    assert_bit_identical(&fused, &unfused, "relu");
}

#[test]
fn fused_forward_matches_unfused_none() {
    let (fused, unfused) = fused_vs_unfused(2, 2, 3, 2, Activation::None);
    assert_bit_identical(&fused, &unfused, "none");
}

#[test]
fn fused_forward_matches_unfused_gelu() {
    let (fused, unfused) = fused_vs_unfused(2, 3, 4, 1, Activation::Gelu);
    assert_bit_identical(&fused, &unfused, "gelu");
}

#[test]
fn fused_forward_matches_unfused_silu() {
    let (fused, unfused) = fused_vs_unfused(1, 4, 5, 2, Activation::Silu);
    assert_bit_identical(&fused, &unfused, "silu");
}

#[test]
fn fused_forward_matches_unfused_elu() {
    let (fused, unfused) = fused_vs_unfused(2, 3, 3, 2, Activation::Elu);
    assert_bit_identical(&fused, &unfused, "elu");
}

#[test]
fn fused_forward_matches_unfused_selu() {
    let (fused, unfused) = fused_vs_unfused(2, 2, 4, 2, Activation::Selu);
    assert_bit_identical(&fused, &unfused, "selu");
}

#[test]
fn fused_forward_matches_unfused_sin() {
    let (fused, unfused) = fused_vs_unfused(1, 2, 2, 1, Activation::SinAct);
    assert_bit_identical(&fused, &unfused, "sin");
}

#[test]
fn fused_forward_shape_is_batch_out() {
    let mut graph = GradGraph::new();
    let (input, w1, b1, w2) = setup(&mut graph, 5, 3, 7, 4);
    let out = graph.mlp_layer_matmul(input, w1, b1, Activation::Tanh, w2);
    let shape = graph.tensor(out).shape().to_vec();
    assert_eq!(shape, vec![5, 4], "expected [batch=5, out=4]");
}
