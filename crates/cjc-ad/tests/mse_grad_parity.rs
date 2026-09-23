//! The fused forms of `mean((pred − target)²)` against the graph chain they stand in for,
//! `sub → mul(diff, diff) → mean → backward`, bit for bit:
//!
//! * `ml::mse_loss_grad`, the one-pass loss and gradient — the loss (the binned mean) and
//!   every gradient element, over several sizes;
//! * the fused graph node `GradOp::MseMean`, which `GradGraph::mean` builds for that chain
//!   — its loss, the gradient of `pred` AND of `target`, at the unit seed `backward`
//!   plants and at two non-unit seeds through `backward_with_seed`; and the fall-back when
//!   the chain is not exclusive (`diff` or `sq` consumed elsewhere), which must be the
//!   plain arms' bits too.
//!
//! The reference is the unfused chain, built with `mean_unfused`; the assertion that the
//! textbook `2·d/n` is NOT what the graph computes keeps the test from passing by
//! comparing a value with itself. Under `--features bruchion-kernels` the same comparisons
//! run with the switch on and every operation in the routing set, so the two Bruchion
//! kernels are held to the graph's bits too.

use cjc_ad::{GradGraph, GradOp};
use cjc_runtime::ml::mse_loss_grad;
use cjc_runtime::tensor::Tensor;

const SIZES: [usize; 6] = [1, 3, 7, 64, 1001, 4093];

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

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The chain's loss and the gradients of `pred` and `target`, through `mean` (fused) or
/// `mean_unfused`, seeded with `seed` through `backward_with_seed` or with `backward`'s
/// own unit seed when `seed` is `None`.
fn chain(pred: &[f64], target: &[f64], fused: bool, seed: Option<f64>) -> (f64, Vec<f64>, Vec<f64>) {
    let n = pred.len();
    let mut g = GradGraph::new();
    let p = g.parameter(Tensor::from_vec(pred.to_vec(), &[n]).unwrap());
    let t = g.parameter(Tensor::from_vec(target.to_vec(), &[n]).unwrap());
    let diff = g.sub(p, t);
    let sq = g.mul(diff, diff);
    let loss = if fused { g.mean(sq) } else { g.mean_unfused(sq) };
    assert!(matches!(g.op(loss), GradOp::MseMean { .. }) == fused, "the node `mean` built");
    match seed {
        None => g.backward(loss),
        Some(s) => g.backward_with_seed(loss, &Tensor::from_vec(vec![s], &[1]).unwrap()),
    }
    let l = g.tensor(loss).to_vec()[0];
    let gp = g.grad(p).expect("a gradient for pred").to_vec();
    let gt = g.grad(t).expect("a gradient for target").to_vec();
    (l, gp, gt)
}

fn check_fused_function(switch_on: bool) {
    cjc_runtime::bruchion::dispatch::route_everything(switch_on);
    for &n in &SIZES {
        let p = inputs(n, 100 + n as u64);
        let t = inputs(n, 200 + n as u64);
        let (gl, gg, _) = chain(&p, &t, false, None);
        let mut fg = vec![7.0; n];
        let fl = mse_loss_grad(&p, &t, &mut fg).unwrap();
        assert_eq!(fl.to_bits(), gl.to_bits(), "loss at n = {n} (switch {switch_on})");
        assert_eq!(bits(&fg), bits(&gg), "gradient at n = {n} (switch {switch_on})");
    }
    // The witness: the graph's gradient is (1/n)·d + (1/n)·d, which at n = 3, d = 2.9 is
    // not (2·d)/n; both the graph and the fused function give the former.
    let p = [2.9, 0.2, -0.4];
    let t = [0.0; 3];
    let (_, gg, _) = chain(&p, &t, false, None);
    let mut fg = [0.0; 3];
    mse_loss_grad(&p, &t, &mut fg).unwrap();
    let h = (1.0f64 / 3.0) * 2.9;
    assert_eq!(gg[0].to_bits(), (h + h).to_bits());
    assert_eq!(fg[0].to_bits(), (h + h).to_bits());
    assert_ne!(fg[0].to_bits(), ((2.0f64 * 2.9) / 3.0).to_bits(), "the textbook spelling is a different value here");
    cjc_runtime::bruchion::dispatch::route_everything(false);
}

fn check_fused_node(switch_on: bool) {
    cjc_runtime::bruchion::dispatch::route_everything(switch_on);
    for &n in &SIZES {
        let p = inputs(n, 300 + n as u64);
        let t = inputs(n, 400 + n as u64);
        for &seed in &[None, Some(1.0), Some(-0.37), Some(3.0)] {
            let (ul, up, ut) = chain(&p, &t, false, seed);
            let (fl, fp, ft) = chain(&p, &t, true, seed);
            assert_eq!(fl.to_bits(), ul.to_bits(), "loss at n = {n}, seed {seed:?} (switch {switch_on})");
            assert_eq!(bits(&fp), bits(&up), "pred gradient at n = {n}, seed {seed:?} (switch {switch_on})");
            assert_eq!(bits(&ft), bits(&ut), "target gradient at n = {n}, seed {seed:?} (switch {switch_on})");
        }
    }
    // The witness again, through the node: h + h, not (2·d)/n.
    let (_, fp, ft) = chain(&[2.9, 0.2, -0.4], &[0.0; 3], true, None);
    let h = (1.0f64 / 3.0) * 2.9;
    assert_eq!(fp[0].to_bits(), (h + h).to_bits());
    assert_eq!(ft[0].to_bits(), (-(h + h)).to_bits());
    assert_ne!(fp[0].to_bits(), ((2.0f64 * 2.9) / 3.0).to_bits());
    cjc_runtime::bruchion::dispatch::route_everything(false);
}

/// A chain that is not the node's own: `diff` also feeds a `sum`, or `sq` does, and the
/// loss is the fused mean plus that sum. The node is still built (the shape matches at
/// construction), its backward must take the plain arms, and every gradient must be the
/// unfused graph's bits — including the order the two contributions reach each parameter.
fn check_not_exclusive(switch_on: bool) {
    cjc_runtime::bruchion::dispatch::route_everything(switch_on);
    for &n in &[3usize, 64, 1001] {
        let p = inputs(n, 500 + n as u64);
        let t = inputs(n, 600 + n as u64);
        for share_diff in [true, false] {
            let run = |fused: bool| {
                let mut g = GradGraph::new();
                let pi = g.parameter(Tensor::from_vec(p.clone(), &[n]).unwrap());
                let ti = g.parameter(Tensor::from_vec(t.clone(), &[n]).unwrap());
                let diff = g.sub(pi, ti);
                let sq = g.mul(diff, diff);
                let m = if fused { g.mean(sq) } else { g.mean_unfused(sq) };
                let other = g.sum(if share_diff { diff } else { sq });
                let loss = g.add(m, other);
                g.backward(loss);
                (g.tensor(loss).to_vec()[0], g.grad(pi).unwrap().to_vec(), g.grad(ti).unwrap().to_vec())
            };
            let (ul, up, ut) = run(false);
            let (fl, fp, ft) = run(true);
            assert_eq!(fl.to_bits(), ul.to_bits(), "loss, n = {n}, shared diff {share_diff}");
            assert_eq!(bits(&fp), bits(&up), "pred gradient, n = {n}, shared diff {share_diff}");
            assert_eq!(bits(&ft), bits(&ut), "target gradient, n = {n}, shared diff {share_diff}");
        }
    }
    cjc_runtime::bruchion::dispatch::route_everything(false);
}

/// `mean` fuses only the exact shape: `mul(d, d)` of a `sub`. A `pow(d, 2)`, a
/// `mul(d, e)` of two different subs, or a `mul(x, x)` of a non-sub stays a plain `Mean`.
#[test]
fn mean_fuses_the_exact_shape_only() {
    let mut g = GradGraph::new();
    let p = g.parameter(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap());
    let t = g.parameter(Tensor::from_vec(vec![0.5, 0.25], &[2]).unwrap());
    let d = g.sub(p, t);
    let e = g.sub(t, p);
    let sq = g.mul(d, d);
    let fused = g.mean(sq);
    assert!(matches!(g.op(fused), GradOp::MseMean { sq: s, diff } if *s == sq && *diff == d));
    let pw = g.pow(d, 2.0);
    let m1 = g.mean(pw);
    assert!(matches!(g.op(m1), GradOp::Mean(_)));
    let de = g.mul(d, e);
    let m2 = g.mean(de);
    assert!(matches!(g.op(m2), GradOp::Mean(_)));
    let pp = g.mul(p, p);
    let m3 = g.mean(pp);
    assert!(matches!(g.op(m3), GradOp::Mean(_)));
    let m4 = g.mean_unfused(sq);
    assert!(matches!(g.op(m4), GradOp::Mean(_)));
}

#[test]
fn the_fused_mse_loss_grad_is_the_graphs_bits() {
    check_fused_function(false);
}

#[test]
fn the_fused_mean_node_is_the_unfused_chains_bits_at_any_seed() {
    check_fused_node(false);
}

#[test]
fn a_shared_chain_takes_the_plain_arms_and_keeps_the_bits() {
    check_not_exclusive(false);
}

#[cfg(feature = "bruchion-kernels")]
#[test]
fn the_bruchion_mse_grad_kernel_is_the_graphs_bits_through_the_switch() {
    check_fused_function(true);
}

#[cfg(feature = "bruchion-kernels")]
#[test]
fn the_bruchion_scaled_kernel_is_the_unfused_chains_bits_through_the_switch() {
    check_fused_node(true);
    check_not_exclusive(true);
}
