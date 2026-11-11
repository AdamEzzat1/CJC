//! Regression tests for `GradGraph::vmap_forward` silently keeping stale
//! values.
//!
//! `vmap_forward` replays the graph once per batch element. Its replay loop
//! ended in a catch-all `_ => self.tensors[node_i].clone()`, so for Log2,
//! Softmax, CrossEntropy, LayerNorm, BatchNorm, Where, Cat, Gather, MlpLayer
//! and MlpLayerMatMul it kept the value from graph construction. Every batch
//! element then produced the same output as the original input.
//!
//! Oracle: for each batch element, a *fresh* graph is built with that element
//! as its input. Graph construction runs each op's forward kernel directly,
//! with no replay, so it is the definition `vmap_forward` must reproduce
//! bit for bit. Softmax and log2 are also checked against plain f64 code that
//! does not touch `GradGraph`.

use cjc_ad::{GradGraph, GradOp};
use cjc_ad::pinn::Activation;
use cjc_runtime::Tensor;

fn t(data: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_unchecked(data.to_vec(), shape)
}

/// Single call site for the API under test.
fn vmap(g: &mut GradGraph, x: usize, batch: &[Tensor]) -> Vec<usize> {
    g.vmap_forward(x, batch).expect("vmap_forward")
}

/// A graph-building recipe: given the graph and the input node, append the
/// subgraph under test. The last node appended is the output `vmap_forward`
/// records.
type Build = fn(&mut GradGraph, usize);

/// Runs `build` under `vmap_forward` over `batch`, and checks every batch
/// output bit for bit against a fresh graph built on that element.
fn check_against_fresh_graphs(label: &str, init: Tensor, batch: &[Tensor], build: Build) {
    let mut g = GradGraph::new();
    let x = g.input(init);
    build(&mut g, x);
    let results = vmap(&mut g, x, batch);
    assert_eq!(results.len(), batch.len(), "{label}: one result per element");

    for (k, elem) in batch.iter().enumerate() {
        let mut fresh = GradGraph::new();
        let fx = fresh.input(elem.clone());
        build(&mut fresh, fx);
        let want = fresh.tensor(fresh.len() - 1);
        let got = g.tensor(results[k]);
        assert_eq!(got.shape(), want.shape(), "{label}[{k}]: shape");
        let (gv, wv) = (got.to_vec(), want.to_vec());
        let same_bits = gv.iter().zip(wv.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same_bits,
            "{label}[{k}]: vmap gave {gv:?}, fresh graph gives {wv:?}"
        );
    }
}

// Three batch elements that are pairwise different and different from the
// construction-time input, so a stale value cannot match by accident.
fn batch3() -> (Tensor, Vec<Tensor>) {
    (
        t(&[0.5, 1.5, 2.5], &[3]),
        vec![
            t(&[1.0, 2.0, 3.0], &[3]),
            t(&[4.0, 0.25, 7.0], &[3]),
            t(&[0.125, 9.0, 2.0], &[3]),
        ],
    )
}

fn weights(g: &mut GradGraph) -> usize {
    g.input(t(&[1.0, -2.0, 0.5], &[3]))
}

// ── one test per op the catch-all skipped ──────────────────────────────

#[test]
fn vmap_log2() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("log2", init, &batch, |g, x| {
        let y = g.log2(x);
        g.sum(y);
    });
}

#[test]
fn vmap_softmax() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("softmax", init, &batch, |g, x| {
        let s = g.softmax(x);
        let w = weights(g);
        let p = g.mul(s, w);
        g.sum(p);
    });
}

#[test]
fn vmap_cross_entropy() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("cross_entropy", init, &batch, |g, x| {
        let targets = g.input(t(&[0.0, 1.0, 0.0], &[3]));
        g.cross_entropy(x, targets);
    });
}

#[test]
fn vmap_layer_norm() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("layer_norm", init, &batch, |g, x| {
        let n = g.layer_norm(x);
        let w = weights(g);
        let p = g.mul(n, w);
        g.sum(p);
    });
}

#[test]
fn vmap_batch_norm() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("batch_norm", init, &batch, |g, x| {
        let n = g.batch_norm(x);
        let w = weights(g);
        let p = g.mul(n, w);
        g.sum(p);
    });
}

#[test]
fn vmap_where() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("where", init, &batch, |g, x| {
        let cond = g.input(t(&[1.0, 0.0, 1.0], &[3]));
        let neg = g.neg(x);
        g.where_cond(cond, x, neg);
    });
}

#[test]
fn vmap_cat() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("cat", init, &batch, |g, x| {
        let sq = g.mul(x, x);
        g.cat(&[x, sq], 0);
    });
}

#[test]
fn vmap_gather() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("gather", init, &batch, |g, x| {
        g.gather(x, &[2, 0, 2], 0);
    });
}

#[test]
fn vmap_mlp_layer() {
    let init = t(&[0.5, 1.5, 2.5, -1.0, 0.0, 1.0], &[2, 3]);
    let batch = vec![
        t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        t(&[-1.0, 0.5, 0.25, 2.0, -3.0, 1.0], &[2, 3]),
    ];
    check_against_fresh_graphs("mlp_layer", init, &batch, |g, x| {
        let w = g.input(t(&[0.1, 0.2, 0.3, -0.4, 0.5, -0.6], &[2, 3]));
        let b = g.input(t(&[0.05, -0.05], &[2]));
        let h = g.mlp_layer(x, w, b, Activation::Tanh);
        g.sum(h);
    });
}

/// A chain mixing ops the old match handled (mul, exp) with ones it did not
/// (softmax, layer_norm): a stale node in the middle must not leak forward.
#[test]
fn vmap_mixed_chain() {
    let (init, batch) = batch3();
    check_against_fresh_graphs("mixed", init, &batch, |g, x| {
        let a = g.scalar_mul(x, 0.5);
        let s = g.softmax(a);
        let e = g.exp(s);
        let n = g.layer_norm(e);
        let w = weights(g);
        let p = g.mul(n, w);
        g.sum(p);
    });
}

// ── oracles that do not use GradGraph at all ───────────────────────────

fn plain_softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

#[test]
fn vmap_softmax_matches_plain_f64() {
    let (init, batch) = batch3();
    let mut g = GradGraph::new();
    let x = g.input(init);
    g.softmax(x);
    let results = vmap(&mut g, x, &batch);
    for (k, elem) in batch.iter().enumerate() {
        let want = plain_softmax(&elem.to_vec());
        let got = g.tensor(results[k]).to_vec();
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-15, "softmax[{k}]: {got:?} vs {want:?}");
        }
    }
}

#[test]
fn vmap_log2_matches_plain_f64() {
    let (init, batch) = batch3();
    let mut g = GradGraph::new();
    let x = g.input(init);
    g.log2(x);
    let results = vmap(&mut g, x, &batch);
    for (k, elem) in batch.iter().enumerate() {
        let got = g.tensor(results[k]).to_vec();
        for (a, v) in got.iter().zip(elem.to_vec()) {
            assert!((a - v.log2()).abs() <= 1e-15, "log2[{k}]: {a} vs {}", v.log2());
        }
    }
}

// ── nodes that cannot be recomputed ────────────────────────────────────

/// A StructField that depends on the input cannot be recomputed. vmap must
/// say so instead of returning the construction-time value for every element.
#[test]
fn vmap_rejects_unrecomputable_dependent() {
    let (init, batch) = batch3();
    let mut g = GradGraph::new();
    let x = g.input(init.clone());
    let sf = g.push_node(
        GradOp::StructField { parent: x, field_index: 0, total_fields: 1 },
        init,
        None,
    );
    g.sum(sf);
    let before = g.len();
    let err = g.vmap_forward(x, &batch).expect_err("must reject StructField");
    assert!(err.contains("StructField"), "error names the op: {err}");
    assert_eq!(g.len(), before, "no snapshot nodes pushed on error");
}

/// A StructField that does NOT depend on the vmapped input holds a value
/// that stays correct for every element, so vmap must not reject it.
#[test]
fn vmap_allows_unrecomputable_independent_node() {
    let (init, batch) = batch3();
    let mut g = GradGraph::new();
    let x = g.input(init);
    let other = g.input(t(&[10.0, 20.0, 30.0], &[3]));
    let sf = g.push_node(
        GradOp::StructField { parent: other, field_index: 0, total_fields: 1 },
        t(&[10.0, 20.0, 30.0], &[3]),
        None,
    );
    let y = g.add(x, sf);
    g.sum(y);
    let results = vmap(&mut g, x, &batch);
    for (k, elem) in batch.iter().enumerate() {
        let want: f64 = elem.to_vec().iter().sum::<f64>() + 60.0;
        assert_eq!(g.value(results[k]), want, "independent StructField [{k}]");
    }
}

#[test]
fn vmap_is_deterministic() {
    let run = || {
        let (init, batch) = batch3();
        let mut g = GradGraph::new();
        let x = g.input(init);
        let s = g.softmax(x);
        let n = g.layer_norm(s);
        g.sum(n);
        let r = vmap(&mut g, x, &batch);
        r.iter().map(|&i| g.value(i).to_bits()).collect::<Vec<u64>>()
    };
    let first = run();
    for _ in 0..20 {
        assert_eq!(run(), first);
    }
}
