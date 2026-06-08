//! Determinism: the fused op produces bit-identical forward and backward
//! results across many runs with the same inputs.

use cjc_ad::pinn::Activation;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

fn build_graph_and_run(seed_offset: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut g = GradGraph::new();
    let input_data: Vec<f64> = (0..6).map(|i| (i as f64 + seed_offset as f64) * 0.1).collect();
    let w1_data: Vec<f64> = (0..12).map(|i| (i as f64 + seed_offset as f64) * 0.05).collect();
    let b1_data: Vec<f64> = (0..4).map(|i| (i as f64 + seed_offset as f64) * 0.01).collect();
    let w2_data: Vec<f64> = (0..8).map(|i| (i as f64 + seed_offset as f64) * 0.03).collect();

    let input = g.input(Tensor::from_vec_unchecked(input_data, &[2, 3]));
    let w1 = g.parameter(Tensor::from_vec_unchecked(w1_data, &[4, 3]));
    let b1 = g.parameter(Tensor::from_vec_unchecked(b1_data, &[4]));
    let w2 = g.parameter(Tensor::from_vec_unchecked(w2_data, &[4, 2]));

    let out = g.mlp_layer_matmul(input, w1, b1, Activation::Tanh, w2);
    let loss = g.sum(out);
    g.zero_grad();
    g.backward(loss);
    (
        g.tensor(out).to_vec(),
        g.grad(w1).map(|t| t.to_vec()).unwrap_or_default(),
        g.grad(b1).map(|t| t.to_vec()).unwrap_or_default(),
        g.grad(w2).map(|t| t.to_vec()).unwrap_or_default(),
    )
}

#[test]
fn forward_and_backward_bit_identical_across_100_runs() {
    let first = build_graph_and_run(0);
    for run in 1..=100 {
        let again = build_graph_and_run(0);
        for (label, a, b) in [
            ("out", &first.0, &again.0),
            ("grad_w1", &first.1, &again.1),
            ("grad_b1", &first.2, &again.2),
            ("grad_w2", &first.3, &again.3),
        ] {
            assert_eq!(a.len(), b.len(), "run {run} {label}: length differs");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "run {run} {label}[{i}]: {x:e} vs {y:e}"
                );
            }
        }
    }
}

#[test]
fn different_seeds_yield_different_results() {
    // Sanity: distinct inputs DO produce distinct outputs (no hidden
    // constant-folding regression).
    let a = build_graph_and_run(0);
    let b = build_graph_and_run(7);
    assert_ne!(a.0, b.0, "outputs must depend on inputs");
    assert_ne!(a.1, b.1, "grad_w1 must depend on inputs");
}
