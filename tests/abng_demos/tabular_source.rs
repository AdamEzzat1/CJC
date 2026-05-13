//! CJC-Lang source for the ABNG tabular GP-like regression demo.
//!
//! Workload: 2-D regression `y = 0.3 + 0.5*x1 + 0.4*x2 + 0.7*x1*x2`
//! on the unit square. BLR feature basis `[1, x1, x2, x1*x2]` —
//! exactly representable, so per-leaf BLR posteriors converge to
//! the truth. Codebook routes by `x1` into 4 region-bins; `x2`
//! becomes a within-leaf regression feature.
//!
//! Tangible benefits asserted:
//! * GP-like: epistemic_leverage shrinks as more data is absorbed.
//! * GP-better-than: per-leaf BLR is bounded; no single leaf holds
//!   ≥90% of the dataset.
//! * Snapshot-checkpointable: a property classical GP doesn't have
//!   (covered by the harness's run/replay sanity checks).

pub const SOURCE: &str = r#"
fn target(x1: f64, x2: f64) -> f64 {
    0.3 + 0.5 * x1 + 0.4 * x2 + 0.7 * x1 * x2
}

fn tabular_features_2d(x1: f64, x2: f64) -> Tensor {
    Tensor.from_vec([1.0, x1, x2, x1 * x2], [1, 4])
}

fn tabular_features_1d(x1: f64, x2: f64) -> Tensor {
    Tensor.from_vec([1.0, x1, x2, x1 * x2], [4])
}

fn build_tabular_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 2, hidden, 1, "tanh");
    abng_set_blr_prior(g, 2.0, 1.0, 0.5);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    let thresholds = Tensor.from_vec(
        [
            0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0,
            1.0e308,
            0.005, 1.05
        ],
        [14]
    );
    abng_set_decision_policy(g, thresholds);
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    abng_add_node(g, 0, 3);
    g
}

fn route_for(g: i64, x1: f64) -> i64 {
    let x_t = Tensor.from_vec([x1], [1]);
    let prefix = abng_encode_prefix(g, x_t);
    let evidence = abng_descend(g, prefix);
    int(evidence.get([1]))
}

fn train_one(g: i64, x1: f64, x2: f64) {
    // Phase 0.8c v14 Item A2 — fused per-row training step. Emits a
    // single `AuditKind::TrainStep` event (tag 0x1E) instead of the
    // pre-A2 `BlrUpdated + BeliefUpdate` pair. Drops the 2-D phi
    // shape used by `abng_blr_update` in favour of the 1-D phi shape
    // `abng_train_step` requires.
    let x_t = Tensor.from_vec([x1], [1]);
    let phi = tabular_features_1d(x1, x2);
    let y_val = target(x1, x2);
    abng_train_step(g, x_t, phi, y_val);
}

// SplitMix64-style deterministic point generator. Returns [x1, x2]
// as a 2-element Tensor (no struct support).
fn det_x_pair(seed: i64, k: i64) -> Any {
    // Two simple cycle-based generators for [0, 1] coverage —
    // fully deterministic, no RNG dep, easily reproducible.
    let s = seed + k * 1103515245;
    let r1_int = s % 997;
    if r1_int < 0 { r1_int = 0 - r1_int; }
    let r1 = float(r1_int) / 997.0;
    let r2_int = (s * 31) % 1009;
    if r2_int < 0 { r2_int = 0 - r2_int; }
    let r2 = float(r2_int) / 1009.0;
    [r1, r2]
}

fn predict_mean(g: i64, x1: f64, x2: f64) -> f64 {
    let leaf = route_for(g, x1);
    let phi = tabular_features_1d(x1, x2);
    let pred = abng_blr_predict(g, leaf, phi);
    pred.get([0])
}

fn predict_lev(g: i64, x1: f64, x2: f64) -> f64 {
    let leaf = route_for(g, x1);
    let phi = tabular_features_1d(x1, x2);
    let pred = abng_blr_predict(g, leaf, phi);
    pred.get([1])
}

fn main() {
    // Train two graphs: a small one (16 pts) and a big one (64 pts).
    // The big one should have strictly lower epistemic_leverage at
    // a fixed probe point.
    let g_small = build_tabular_graph(11);
    let g_big = build_tabular_graph(11);
    let k = 0;
    while k < 16 {
        let xy = det_x_pair(31, k);
        train_one(g_small, xy[0], xy[1]);
        k = k + 1;
    }
    let k2 = 0;
    while k2 < 64 {
        let xy = det_x_pair(31, k2);
        train_one(g_big, xy[0], xy[1]);
        k2 = k2 + 1;
    }

    print("chain_small: " + abng_chain_head(g_small));
    print("chain_big: " + abng_chain_head(g_big));
    print("verify_small: " + to_string(abng_verify_chain(g_small)));
    print("verify_big: " + to_string(abng_verify_chain(g_big)));

    // Held-out MSE comparison: trained vs prior (small).
    let g_prior = build_tabular_graph(11);
    let prior_se = 0.0;
    let trained_se = 0.0;
    let i = 0;
    while i < 32 {
        let xy = det_x_pair(48879, i);  // 0xBEEF = 48879
        let truth = target(xy[0], xy[1]);
        let m_p = predict_mean(g_prior, xy[0], xy[1]);
        let m_t = predict_mean(g_big, xy[0], xy[1]);
        prior_se = prior_se + (m_p - truth) * (m_p - truth);
        trained_se = trained_se + (m_t - truth) * (m_t - truth);
        i = i + 1;
    }
    let prior_mse = prior_se / 32.0;
    let trained_mse = trained_se / 32.0;
    print("prior_mse: " + to_string(prior_mse));
    print("trained_mse: " + to_string(trained_mse));
    let trained_beats_prior = trained_mse < prior_mse * 0.5;
    print("trained_beats_prior: " + to_string(trained_beats_prior));

    // Per-region uncertainty: leverage at a fixed probe shrinks
    // with more data.
    let lev_small = predict_lev(g_small, 0.5, 0.5);
    let lev_big = predict_lev(g_big, 0.5, 0.5);
    print("lev_small: " + to_string(lev_small));
    print("lev_big: " + to_string(lev_big));
    let lev_shrinks = lev_big < lev_small;
    print("lev_shrinks_with_data: " + to_string(lev_shrinks));

    // Per-leaf bound: max single-leaf n_seen < 90% of total.
    let n_total = abng_blr_n_seen(g_big, 1) + abng_blr_n_seen(g_big, 2)
        + abng_blr_n_seen(g_big, 3) + abng_blr_n_seen(g_big, 4);
    let max_n = abng_blr_n_seen(g_big, 1);
    let n2 = abng_blr_n_seen(g_big, 2);
    let n3 = abng_blr_n_seen(g_big, 3);
    let n4 = abng_blr_n_seen(g_big, 4);
    if n2 > max_n { max_n = n2; }
    if n3 > max_n { max_n = n3; }
    if n4 > max_n { max_n = n4; }
    print("max_n_per_leaf: " + to_string(max_n));
    print("total_routed: " + to_string(n_total));
    let bounded = float(max_n) < 0.9 * float(n_total);
    print("max_per_leaf_bounded: " + to_string(bounded));
}
"#;
