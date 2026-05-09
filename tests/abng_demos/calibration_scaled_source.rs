//! Phase 0.6 Item 6 — CJC-Lang source for the **scaled** calibration
//! demo.
//!
//! Differences vs Phase 0.5's `calibration_source.rs`:
//!   - 1000 (predicted_prob, observed_outcome) pairs per graph
//!     (vs 30) — production-realistic prediction volume
//!   - Predicted probabilities sampled from a deterministic
//!     pseudo-uniform stream (Tensor.uniform), producing a more
//!     diverse coverage of the [0, 1] interval than Phase 0.5's
//!     three-bin {0.1, 0.5, 0.9} sequence
//!   - Outcomes for Graph A track the predicted probability via a
//!     deterministic threshold (well-calibrated by construction);
//!     Graph B flips them (severely miscalibrated)
//!
//! Headline: at n=1000, `ece_a` < `ece_b` with a wide margin. The
//! ECE estimate's variance shrinks at √n, so this is more honest
//! than the Phase 0.5 small-sample version.

pub const SOURCE: &str = r#"
fn build_calib_scaled_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.5], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.0, 0.5);
    abng_set_calibration(g, 15);
    g
}

// Run n observations through the calibration head:
//   - predicted_prob: deterministic pseudo-uniform stream (the
//     Tensor.uniform tensor, indexed by i).
//   - outcome:
//       well_calibrated = true  → outcome flips at threshold p
//       well_calibrated = false → outcome flips at threshold 1-p
//   The threshold uses a SECOND deterministic stream so the test
//   is not tautological (otherwise outcome would be deterministic
//   given p, which calibrates trivially).
fn run_calib(g: i64, node: i64, well_calibrated: bool, n: i64) {
    let probs = Tensor.uniform([n]);
    let thresh = Tensor.uniform([n]);
    let i = 0;
    while i < n {
        let p = probs.get([i]);
        let t = thresh.get([i]);
        let target_threshold = p;
        if !well_calibrated {
            target_threshold = 1.0 - p;
        }
        let y = t < target_threshold;
        abng_calibration_observe(g, node, p, y);
        i = i + 1;
    }
}

fn main() {
    let g_a = build_calib_scaled_graph(7);
    let g_b = build_calib_scaled_graph(7);

    // Graph A: well-calibrated — observed-positive frequency
    // matches predicted probability.
    run_calib(g_a, 0, true, 1000);

    // Graph B: severely miscalibrated — observed-positive frequency
    // is the COMPLEMENT of the predicted probability.
    run_calib(g_b, 0, false, 1000);

    let ece_a = abng_calibration_ece(g_a, 0);
    let ece_b = abng_calibration_ece(g_b, 0);
    let n_a = abng_calibration_n_seen(g_a, 0);
    let n_b = abng_calibration_n_seen(g_b, 0);

    print("ece_a: " + to_string(ece_a));
    print("ece_b: " + to_string(ece_b));
    print("n_a: " + to_string(n_a));
    print("n_b: " + to_string(n_b));

    let well_calib_below_miscal = ece_a < ece_b;
    print("well_below_miscal: " + to_string(well_calib_below_miscal));

    // Headline: at n=1000, the gap is wide. ECE for A (well-cal)
    // should be near zero; for B (miscal) should be near 1.0.
    let ece_a_low = ece_a < 0.15;
    let ece_b_high = ece_b > 0.4;
    print("ece_a_low: " + to_string(ece_a_low));
    print("ece_b_high: " + to_string(ece_b_high));

    // Both heads ingested 1000 observations.
    let both_full = n_a == 1000 && n_b == 1000;
    print("both_full: " + to_string(both_full));

    print("verify_chain: " + to_string(abng_verify_chain(g_a)));
    // Use Graph A as the canary anchor.
    print("chain_head: " + abng_chain_head(g_a));
}
"#;
