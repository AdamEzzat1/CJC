//! CJC-Lang source for the ABNG calibration / ECE demo.
//!
//! Workload: two graphs, each fed 32 (predicted_prob, observed_outcome)
//! pairs.
//!
//! Graph A — "well calibrated": predictions distributed across
//! the [0, 1] range; outcomes track the predictions (bin 0.7
//! receives ~70% positive outcomes, etc.). ECE should approach 0.
//!
//! Graph B — "miscalibrated": same predictions, but every
//! prediction's outcome is FLIPPED. Bin 0.7 receives ~30% positive
//! outcomes — opposite of what the prediction claimed. ECE should
//! be high (close to the maximum possible).
//!
//! Headline assertion: `ece_a` strictly less than `ece_b`. The
//! calibration head distinguishes well-calibrated vs miscalibrated
//! models — exactly the diagnostic an MLP doesn't give natively.

pub const SOURCE: &str = r#"
fn build_calib_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.5], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.0, 0.5);
    abng_set_calibration(g, 15);
    g
}

// Helper: feed (p, y_correct) into the calibration head on the
// given node.
fn calib_obs(g: i64, node: i64, p: f64, y: bool) {
    abng_calibration_observe(g, node, p, y);
}

// Build a deterministic sequence of (p, y) pairs where outcomes
// track predictions. Each row repeats `count` times so the bin
// statistics converge.
fn well_calibrated_run(g: i64, node: i64) {
    // Bin around 0.1: predict 0.1, outcome positive ~10% of time
    // (1 of 10 observations). With small samples, exact-match is
    // hard; we use enough repetitions that bin frequency
    // approaches the predicted probability.
    let i = 0;
    while i < 10 {
        // 1 positive, 9 negative for predicted_prob = 0.1.
        if i < 1 {
            calib_obs(g, node, 0.1, true);
        }
        if i >= 1 {
            calib_obs(g, node, 0.1, false);
        }
        i = i + 1;
    }
    // Bin 0.5: 5 positive, 5 negative.
    let j = 0;
    while j < 10 {
        if j < 5 {
            calib_obs(g, node, 0.5, true);
        }
        if j >= 5 {
            calib_obs(g, node, 0.5, false);
        }
        j = j + 1;
    }
    // Bin 0.9: 9 positive, 1 negative.
    let k = 0;
    while k < 10 {
        if k < 9 {
            calib_obs(g, node, 0.9, true);
        }
        if k >= 9 {
            calib_obs(g, node, 0.9, false);
        }
        k = k + 1;
    }
}

// Same predictions, but outcomes FLIPPED. Predicted 0.9 → only
// 1 of 10 positive (instead of 9). ECE should be huge.
fn miscalibrated_run(g: i64, node: i64) {
    let i = 0;
    while i < 10 {
        // Predicted 0.1 but outcome positive 90% of time (FLIPPED).
        if i < 9 {
            calib_obs(g, node, 0.1, true);
        }
        if i >= 9 {
            calib_obs(g, node, 0.1, false);
        }
        i = i + 1;
    }
    let j = 0;
    while j < 10 {
        if j < 5 {
            calib_obs(g, node, 0.5, true);
        }
        if j >= 5 {
            calib_obs(g, node, 0.5, false);
        }
        j = j + 1;
    }
    let k = 0;
    while k < 10 {
        // Predicted 0.9 but outcome positive 10% of time (FLIPPED).
        if k < 1 {
            calib_obs(g, node, 0.9, true);
        }
        if k >= 1 {
            calib_obs(g, node, 0.9, false);
        }
        k = k + 1;
    }
}

fn main() {
    let g_a = build_calib_graph(7);
    let g_b = build_calib_graph(7);

    well_calibrated_run(g_a, 0);
    miscalibrated_run(g_b, 0);

    let ece_a = abng_calibration_ece(g_a, 0);
    let ece_b = abng_calibration_ece(g_b, 0);
    let n_a = abng_calibration_n_seen(g_a, 0);
    let n_b = abng_calibration_n_seen(g_b, 0);

    print("ece_a: " + to_string(ece_a));
    print("ece_b: " + to_string(ece_b));
    print("n_a: " + to_string(n_a));
    print("n_b: " + to_string(n_b));

    // Range invariants: ECE in [0, 1].
    let in_range = ece_a >= 0.0 && ece_a <= 1.0
        && ece_b >= 0.0 && ece_b <= 1.0;
    print("ece_in_range: " + to_string(in_range));

    // Headline: well-calibrated ECE strictly below miscalibrated.
    let well_below_mis = ece_a < ece_b;
    print("well_below_mis: " + to_string(well_below_mis));

    // Specific claim: well-calibrated ECE is "small" (< 0.05) AND
    // miscalibrated ECE is "large" (> 0.5). The 0.5 threshold for
    // miscalibrated is conservative — flipping all outcomes gives
    // ECE close to 0.8 in the limit.
    let well_low = ece_a < 0.05;
    let mis_high = ece_b > 0.5;
    print("well_low: " + to_string(well_low));
    print("mis_high: " + to_string(mis_high));

    // n_seen is the same for both runs (30 observations each).
    let n_match = n_a == n_b;
    print("n_match: " + to_string(n_match));

    print("verify_a: " + to_string(abng_verify_chain(g_a)));
    print("verify_b: " + to_string(abng_verify_chain(g_b)));
    print("chain_a: " + abng_chain_head(g_a));
    print("chain_b: " + abng_chain_head(g_b));
}
"#;
