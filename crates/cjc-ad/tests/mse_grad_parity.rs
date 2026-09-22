//! The fused `mse_loss_grad` against the graph it stands in for: `GradGraph`'s
//! `sub → mul(diff, diff) → mean → backward`, bit for bit — the loss (the binned mean)
//! and every gradient element — over several sizes, plus the assertion that the textbook
//! `2·d/n` is NOT what the graph computes, so the test cannot pass by comparing a value
//! with itself. Under `--features bruchion-kernels` the same comparison runs with the
//! switch on, so the Bruchion kernel is held to the graph's bits too.

use cjc_ad::GradGraph;
use cjc_runtime::ml::mse_loss_grad;
use cjc_runtime::tensor::Tensor;

fn inputs(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = cjc_repro::Rng::seeded(seed);
    (0..n)
        .map(|i| {
            if i % 23 == 22 {
                return -0.0;
            }
            if i % 17 == 16 {
                return 0.0;
            }
            (rng.next_f64() * 2.0 - 1.0) * cjc_repro::powi_f64(2.0, (rng.next_u64() % 16) as i32 - 8)
        })
        .collect()
}

/// The graph's loss and gradient with respect to `pred`.
fn graph_loss_and_grad(pred: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
    let n = pred.len();
    let mut g = GradGraph::new();
    let p = g.parameter(Tensor::from_vec(pred.to_vec(), &[n]).unwrap());
    let t = g.parameter(Tensor::from_vec(target.to_vec(), &[n]).unwrap());
    let diff = g.sub(p, t);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq);
    g.backward(loss);
    let l = g.tensor(loss).to_vec()[0];
    let grad = g.grad(p).expect("a gradient for pred").to_vec();
    (l, grad)
}

fn check(switch_on: bool) {
    cjc_runtime::runtime_policy::set_bruchion_kernels(switch_on);
    for &n in &[1usize, 3, 7, 64, 1001, 4093] {
        let p = inputs(n, 100 + n as u64);
        let t = inputs(n, 200 + n as u64);
        let (gl, gg) = graph_loss_and_grad(&p, &t);
        let mut fg = vec![7.0; n];
        let fl = mse_loss_grad(&p, &t, &mut fg).unwrap();
        assert_eq!(fl.to_bits(), gl.to_bits(), "loss at n = {n} (switch {switch_on})");
        let gb: Vec<u64> = gg.iter().map(|v| v.to_bits()).collect();
        let fb: Vec<u64> = fg.iter().map(|v| v.to_bits()).collect();
        assert_eq!(fb, gb, "gradient at n = {n} (switch {switch_on})");
    }
    // The witness: the graph's gradient is (1/n)·d + (1/n)·d, which at n = 3, d = 2.9 is
    // not (2·d)/n; both the graph and the fused function give the former.
    let p = [2.9, 0.2, -0.4];
    let t = [0.0; 3];
    let (_, gg) = graph_loss_and_grad(&p, &t);
    let mut fg = [0.0; 3];
    mse_loss_grad(&p, &t, &mut fg).unwrap();
    let h = (1.0f64 / 3.0) * 2.9;
    assert_eq!(gg[0].to_bits(), (h + h).to_bits());
    assert_eq!(fg[0].to_bits(), (h + h).to_bits());
    assert_ne!(fg[0].to_bits(), ((2.0f64 * 2.9) / 3.0).to_bits(), "the textbook spelling is a different value here");
    cjc_runtime::runtime_policy::set_bruchion_kernels(false);
}

#[test]
fn the_fused_mse_loss_grad_is_the_graphs_bits() {
    check(false);
}

#[cfg(feature = "bruchion-kernels")]
#[test]
fn the_bruchion_mse_grad_kernel_is_the_graphs_bits_through_the_switch() {
    check(true);
}
