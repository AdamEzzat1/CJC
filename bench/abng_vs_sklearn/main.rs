//! ABNG vs sklearn-style GP regression at scale (Phase 0.6 Item 2)
//! ================================================================
//!
//! Establishes ABNG's wall-clock + RMSE baseline on a tabular regression
//! problem at n in {10^3, 10^4, 10^5} so the Phase 0.6 Item 4 batch
//! observe perf win has a comparable to cite.
//!
//! Truth function (deterministic, reproducible seeded noise):
//!     y = 2 * x_1 + 3 * x_2 + 0.5 * x_1 * x_2 + N(0, 0.05)
//!
//! Train/test split: 80/20. Features for the BLR head: `[1, x1, x2, x1*x2]`.
//!
//! The companion sklearn comparison harness (Python) is intentionally
//! deferred to Item 4 — Phase 0.6 Item 2's deliverable is ABNG-side
//! baseline numbers. A Python harness can be added in
//! `bench/abng_vs_sklearn/sklearn_compare.py` later and should consume
//! this binary's JSONL output directly.
//!
//! Invocation:
//!     cargo run -p abng-vs-sklearn --release > abng_vs_sklearn.jsonl

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_ad::pinn::Activation;
use cjc_repro::Rng;
use std::time::Instant;

/// Synthetic dataset generator. Deterministic given (seed, n).
fn make_dataset(seed: u64, n: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut rng = Rng::seeded(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = rng.next_f64();
        let x2 = rng.next_f64();
        let noise = (rng.next_f64() - 0.5) * 0.1; // uniform in (-0.05, 0.05)
        let y = 2.0 * x1 + 3.0 * x2 + 0.5 * x1 * x2 + noise;
        xs.push([x1, x2]);
        ys.push(y);
    }
    (xs, ys)
}

fn features(x: &[f64; 2]) -> [f64; 4] {
    [1.0, x[0], x[1], x[0] * x[1]]
}

fn build_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    // 2D codebook with 4 bins per dim ⇒ 3 boundaries per dim, flattened across both.
    g.set_codebook(2, 4, &[0.25, 0.5, 0.75, 0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(2, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

fn rmse(preds: &[f64], truths: &[f64]) -> f64 {
    assert_eq!(preds.len(), truths.len());
    let mut sum_sq = 0.0_f64;
    for (p, t) in preds.iter().zip(truths.iter()) {
        let e = p - t;
        sum_sq += e * e;
    }
    (sum_sq / preds.len() as f64).sqrt()
}

fn run_at_scale(n_total: usize, seed: u64) {
    let (xs, ys) = make_dataset(seed, n_total);
    let n_train = n_total * 4 / 5;
    let xs_train = &xs[..n_train];
    let ys_train = &ys[..n_train];
    let xs_test = &xs[n_train..];
    let ys_test = &ys[n_train..];

    let mut g = build_graph(seed);

    // Train
    let train_start = Instant::now();
    for (x, y) in xs_train.iter().zip(ys_train.iter()) {
        let prefix = g.encode_prefix(x).unwrap();
        let leaf = g.descend(&prefix).leaf_id;
        let phi = features(x);
        g.blr_update(leaf, &phi, &[*y]).unwrap();
        g.observe(leaf, *y).unwrap();
    }
    let train_elapsed = train_start.elapsed();

    // Predict
    let predict_start = Instant::now();
    let mut preds = Vec::with_capacity(xs_test.len());
    for x in xs_test {
        let prefix = g.encode_prefix(x).unwrap();
        let leaf = g.descend(&prefix).leaf_id;
        let phi = features(x);
        let (mean, _lev, _ale) = g.blr_predict(leaf, &phi).unwrap();
        preds.push(mean);
    }
    let predict_elapsed = predict_start.elapsed();

    let test_rmse = rmse(&preds, ys_test);
    let train_ns_per_row = train_elapsed.as_nanos() as f64 / n_train as f64;
    let predict_ns_per_row = predict_elapsed.as_nanos() as f64 / xs_test.len() as f64;

    println!(
        r#"{{"scale":{n_total},"n_train":{n_train},"n_test":{},"train_total_ms":{:.3},"predict_total_ms":{:.3},"train_per_row_ns":{:.1},"predict_per_row_ns":{:.1},"test_rmse":{:.6}}}"#,
        xs_test.len(),
        train_elapsed.as_secs_f64() * 1000.0,
        predict_elapsed.as_secs_f64() * 1000.0,
        train_ns_per_row,
        predict_ns_per_row,
        test_rmse,
    );
    eprintln!(
        "  n={n_total:>6}  train={:>8.1}ms  predict={:>7.1}ms  rmse={:.5}",
        train_elapsed.as_secs_f64() * 1000.0,
        predict_elapsed.as_secs_f64() * 1000.0,
        test_rmse,
    );
}

fn main() {
    eprintln!("=== ABNG vs sklearn baseline (Phase 0.6 Item 2) ===");
    eprintln!("Truth: y = 2*x1 + 3*x2 + 0.5*x1*x2 + N(0, 0.05)");
    let seed = 7u64;
    for &n in &[1_000usize, 10_000, 100_000] {
        run_at_scale(n, seed);
    }
    eprintln!("=== Done ===");
}
