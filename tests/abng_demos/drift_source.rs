//! CJC-Lang source for the ABNG drift-detection demo.
//!
//! Workload: train density on a baseline distribution, freeze the
//! drift baseline, then observe under two scenarios:
//!
//!   * Phase A: more samples from the same distribution
//!     → drift_score stays low.
//!   * Phase B: samples from a *shifted* distribution
//!     → drift_score increases substantially.
//!
//! Headline assertion: post-shift `drift_score` strictly exceeds
//! same-distribution `drift_score`. The "model knows when its
//! input distribution has changed" story — exactly what
//! production ML systems need to detect when retraining is due.

pub const SOURCE: &str = r#"
fn build_drift_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.5], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 2.0, 1.0, 0.5);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    g
}

// Generate a deterministic batch of `n` 4-D samples drawn from a
// distribution centered at `center`. The samples form a small
// jittered grid around `center` so the density tracker accumulates
// non-degenerate covariance.
fn density_batch(center: f64, n: i64) -> Tensor {
    let buf = [];
    let i = 0;
    while i < n {
        let off = 0.01 * float(i - n / 2);
        let x = center + off;
        buf = array_push(buf, 1.0);
        buf = array_push(buf, x);
        buf = array_push(buf, x * x);
        buf = array_push(buf, x * x * x);
        i = i + 1;
    }
    Tensor.from_vec(buf, [n, 4])
}

fn main() {
    let g = build_drift_graph(7);

    // Phase 0: train baseline density on distribution centered at 0.
    let baseline_batch = density_batch(0.0, 32);
    abng_density_observe(g, 0, baseline_batch);

    // Freeze drift baseline at this snapshot.
    abng_freeze_drift_baseline(g, 0);

    // Phase A: continue with same distribution (centered at 0).
    let same_batch = density_batch(0.0, 16);
    abng_density_observe(g, 0, same_batch);
    let drift_low = abng_drift_score(g, 0);

    // Phase B: shift distribution to be centered at 0.8.
    let shifted_batch = density_batch(0.8, 16);
    abng_density_observe(g, 0, shifted_batch);
    let drift_high = abng_drift_score(g, 0);

    print("drift_low: " + to_string(drift_low));
    print("drift_high: " + to_string(drift_high));

    // Headline: post-shift drift strictly higher than same-dist.
    let drift_grew = drift_high > drift_low;
    print("drift_grew: " + to_string(drift_grew));

    // Drift score is non-negative.
    let nonneg = drift_low >= 0.0 && drift_high >= 0.0;
    print("drift_nonneg: " + to_string(nonneg));

    // The shift produces a meaningful effect: drift_high is at
    // least 2x drift_low. (Loose bound — real production thresholds
    // are workload-dependent.)
    let meaningful_shift = drift_high > 2.0 * drift_low;
    print("meaningful_shift: " + to_string(meaningful_shift));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
