//! ABNG PINN-shaped regression at scale (Phase 0.6 Item 2)
//! ========================================================
//!
//! Establishes the wall-clock + L2-error baseline for ABNG when used
//! as the regression head of a physics-informed network at realistic
//! collocation density (10^4 points). The Phase 0.6 Item 4 batch
//! observe perf win is expected to dominate the per-point hot path
//! exercised by this benchmark.
//!
//! Truth function (separable, smooth, well-conditioned for BLR):
//!     u(x) = sin(pi * x)  on x in [0, 1]
//!
//! Feature basis for BLR: `[1, x, x^2, x^3, sin(pi*x)]` (the truth
//! sits exactly in span — letting us measure the conditioning of the
//! per-point training, not the basis adequacy).
//!
//! The companion comparison harness against PyTorch MLP-PINN is
//! intentionally deferred to Phase 0.7+ scope. Phase 0.6 Item 2
//! ships ABNG-side baseline numbers only.
//!
//! Invocation:
//!     cargo run -p abng-pinn-scale --release > abng_pinn_scale.jsonl

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_ad::pinn::Activation;
use cjc_repro::Rng;
use std::f64::consts::PI;
use std::time::Instant;

fn truth(x: f64) -> f64 {
    (PI * x).sin()
}

fn features(x: f64) -> [f64; 5] {
    [1.0, x, x * x, x * x * x, (PI * x).sin()]
}

fn build_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 8, &[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]).unwrap();
    g.set_leaf_head(1, vec![5], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for byte in 0u8..8 {
        g.add_node(0, byte).unwrap();
    }
    g
}

fn run_at_scale(n_collocation: usize, seed: u64) {
    // Deterministic Latin-hypercube-style collocation points: shuffle
    // a uniform grid via SplitMix64.
    let mut xs: Vec<f64> = (0..n_collocation)
        .map(|i| (i as f64 + 0.5) / n_collocation as f64)
        .collect();
    let mut rng = Rng::seeded(seed);
    for i in (1..n_collocation).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        xs.swap(i, j);
    }

    let mut g = build_graph(seed);

    let train_start = Instant::now();
    for &x in &xs {
        let prefix = g.encode_prefix(&[x]).unwrap();
        let leaf = g.descend(&prefix).leaf_id;
        let phi = features(x);
        let y = truth(x);
        g.blr_update(leaf, &phi, &[y]).unwrap();
        g.observe(leaf, y).unwrap();
    }
    let train_elapsed = train_start.elapsed();

    // Evaluate on a dense uniform grid (independent of training).
    let n_eval = 1024usize;
    let predict_start = Instant::now();
    let mut sum_sq_err = 0.0_f64;
    for i in 0..n_eval {
        let x = (i as f64 + 0.5) / n_eval as f64;
        let prefix = g.encode_prefix(&[x]).unwrap();
        let leaf = g.descend(&prefix).leaf_id;
        let phi = features(x);
        let (mean, _lev, _ale) = g.blr_predict(leaf, &phi).unwrap();
        let e = mean - truth(x);
        sum_sq_err += e * e;
    }
    let predict_elapsed = predict_start.elapsed();
    let l2_err = (sum_sq_err / n_eval as f64).sqrt();

    let train_per_pt_ns = train_elapsed.as_nanos() as f64 / n_collocation as f64;
    let predict_per_pt_ns = predict_elapsed.as_nanos() as f64 / n_eval as f64;

    println!(
        r#"{{"n_collocation":{n_collocation},"n_eval":{n_eval},"train_total_ms":{:.3},"predict_total_ms":{:.3},"train_per_pt_ns":{:.1},"predict_per_pt_ns":{:.1},"l2_err":{:.6e}}}"#,
        train_elapsed.as_secs_f64() * 1000.0,
        predict_elapsed.as_secs_f64() * 1000.0,
        train_per_pt_ns,
        predict_per_pt_ns,
        l2_err,
    );
    eprintln!(
        "  n_coll={n_collocation:>6}  train={:>8.1}ms  predict={:>7.1}ms  L2={:.3e}",
        train_elapsed.as_secs_f64() * 1000.0,
        predict_elapsed.as_secs_f64() * 1000.0,
        l2_err,
    );
}

fn main() {
    eprintln!("=== ABNG PINN-shaped regression baseline (Phase 0.6 Item 2) ===");
    eprintln!("Truth: u(x) = sin(pi * x) on [0, 1]");
    let seed = 11u64;
    for &n in &[1_000usize, 10_000, 100_000] {
        run_at_scale(n, seed);
    }
    eprintln!("=== Done ===");
}
