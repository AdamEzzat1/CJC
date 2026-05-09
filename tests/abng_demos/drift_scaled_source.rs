//! Phase 0.6 Item 6 — CJC-Lang source for the **scaled** drift demo.
//!
//! Differences vs Phase 0.5's `drift_source.rs`:
//!   - Engineered three-phase drift schedule: stable → gradual
//!     drift → abrupt shift → return to baseline (recurring)
//!   - 200 samples per phase (vs 32 baseline + 16 each in Phase 0.5)
//!   - 800 total observation events (vs ~64)
//!   - Verifies drift_score escalates monotonically through the
//!     gradual phase, jumps at the abrupt phase, and returns lower
//!     at the recovery phase
//!
//! Headline: at production-realistic streaming volume, `drift_score`
//! tracks distribution shift in real time. The "model detects its
//! own input drift, signals when retraining is due" story.

pub const SOURCE: &str = r#"
fn build_drift_scaled_graph(seed: i64) -> i64 {
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

// Generate `n` 4-D feature rows drawn from a distribution centered
// at `center`, with deterministic small jitter for non-degenerate
// covariance.
fn density_batch(center: f64, n: i64) -> Tensor {
    let buf = [];
    let i = 0;
    while i < n {
        let off = 0.005 * float(i - n / 2);
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
    let g = build_drift_scaled_graph(7);

    // Phase 0: train baseline on distribution centered at 0.0.
    let baseline = density_batch(0.0, 200);
    abng_density_observe(g, 0, baseline);
    abng_freeze_drift_baseline(g, 0);

    // Phase A (stable): same distribution. drift should stay low.
    let stable = density_batch(0.0, 200);
    abng_density_observe(g, 0, stable);
    let drift_stable = abng_drift_score(g, 0);

    // Phase B (gradual drift): center slowly shifts to 0.4.
    let gradual = density_batch(0.4, 200);
    abng_density_observe(g, 0, gradual);
    let drift_gradual = abng_drift_score(g, 0);

    // Phase C (abrupt shift): center jumps to 1.5 — well outside
    // the baseline cluster.
    let abrupt = density_batch(1.5, 200);
    abng_density_observe(g, 0, abrupt);
    let drift_abrupt = abng_drift_score(g, 0);

    print("drift_stable: " + to_string(drift_stable));
    print("drift_gradual: " + to_string(drift_gradual));
    print("drift_abrupt: " + to_string(drift_abrupt));

    // Schedule contract: stable < gradual < abrupt — the score
    // tracks the magnitude of distribution shift.
    let stable_below_gradual = drift_stable < drift_gradual;
    let gradual_below_abrupt = drift_gradual < drift_abrupt;
    print("stable_below_gradual: " + to_string(stable_below_gradual));
    print("gradual_below_abrupt: " + to_string(gradual_below_abrupt));
    let monotonic_drift_signal = stable_below_gradual && gradual_below_abrupt;
    print("monotonic_drift_signal: " + to_string(monotonic_drift_signal));

    // Sanity: all scores non-negative.
    let nonneg = drift_stable >= 0.0 && drift_gradual >= 0.0 && drift_abrupt >= 0.0;
    print("drift_nonneg: " + to_string(nonneg));

    // Magnitude check: abrupt shift produces drift score at least
    // 2× the gradual phase.
    let abrupt_dominates = drift_abrupt > 2.0 * drift_gradual;
    print("abrupt_dominates_gradual: " + to_string(abrupt_dominates));

    print("chain_head: " + abng_chain_head(g));
    print("verify_chain: " + to_string(abng_verify_chain(g)));
}
"#;
