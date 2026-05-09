//! Phase 0.6 Item 6 — CJC-Lang source for the **scaled** OOD demo.
//!
//! Differences vs Phase 0.5's `ood_source.rs`:
//!   - 320 training samples per dense leaf (vs 32)
//!   - 32 training samples on the sparse leaf (vs 4)
//!   - Same 4-D feature basis: [1, x, x², x³] — proven to give a
//!     non-degenerate density-tracker covariance at small sample
//!     count
//!
//! Scope reduction vs the handoff's "32-D embedding-space input":
//! at 8-D with mixed polynomial+trig features the density tracker's
//! covariance becomes near-singular (off-diagonal correlations are
//! large), which pushes density_score to saturation for all probe
//! points. The Phase 0.5 4-D feature set is the smallest dim that
//! produces a numerically stable density signal at this sample
//! count. The scaling axis is therefore observation count, not
//! feature dim.
//!
//! Headline: at 10× the sample count, the composite ood_score
//! STILL cleanly partitions (dense < sparse < routing-fall-off).
//! Calibration of the OOD signal scales with training volume.

pub const SOURCE: &str = r#"
fn build_ood_scaled_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 2.0, 1.0, 0.5);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    g
}

fn features_4d(x: f64) -> Tensor {
    Tensor.from_vec([1.0, x, x * x, x * x * x], [4])
}

fn train_leaf_density_4d(g: i64, leaf: i64, x_center: f64, n: i64) {
    let i = 0;
    let buf = [];
    while i < n {
        let xi = x_center + 0.001 * float(i - n / 2);
        buf = array_push(buf, 1.0);
        buf = array_push(buf, xi);
        buf = array_push(buf, xi * xi);
        buf = array_push(buf, xi * xi * xi);
        i = i + 1;
    }
    let batch = Tensor.from_vec(buf, [n, 4]);
    abng_density_observe(g, leaf, batch);
}

fn probe_ood(g: i64, x: f64) -> f64 {
    let x_t = Tensor.from_vec([x], [1]);
    let prefix = abng_encode_prefix(g, x_t);
    let evidence = abng_descend(g, prefix);
    let matched = int(evidence.get([0]));
    let leaf = int(evidence.get([1]));
    let phi = features_4d(x);
    abng_ood_score(g, leaf, phi, matched, 1)
}

fn main() {
    let g = build_ood_scaled_graph(7);

    // 10× more dense training than Phase 0.5.
    train_leaf_density_4d(g, 1, 0.125, 320);
    train_leaf_density_4d(g, 2, 0.375, 320);
    // 8× more sparse training, but still well below dense.
    train_leaf_density_4d(g, 3, 0.625, 32);

    let ood_bin0 = probe_ood(g, 0.10);
    let ood_bin1 = probe_ood(g, 0.40);
    let ood_bin2 = probe_ood(g, 0.62);
    let ood_bin3 = probe_ood(g, 0.90);

    print("ood_bin0: " + to_string(ood_bin0));
    print("ood_bin1: " + to_string(ood_bin1));
    print("ood_bin2: " + to_string(ood_bin2));
    print("ood_bin3: " + to_string(ood_bin3));

    let in_range = ood_bin0 >= 0.0 && ood_bin0 <= 1.0
        && ood_bin1 >= 0.0 && ood_bin1 <= 1.0
        && ood_bin2 >= 0.0 && ood_bin2 <= 1.0
        && ood_bin3 >= 0.0 && ood_bin3 <= 1.0;
    print("scores_in_range: " + to_string(in_range));

    let dense_max = ood_bin0;
    if ood_bin1 > dense_max { dense_max = ood_bin1; }
    let dense_below_sparse = dense_max < ood_bin2;
    let sparse_below_falloff = ood_bin2 < ood_bin3;
    print("dense_below_sparse: " + to_string(dense_below_sparse));
    print("sparse_below_falloff: " + to_string(sparse_below_falloff));
    let three_tier_separation = dense_below_sparse && sparse_below_falloff;
    print("three_tier_separation: " + to_string(three_tier_separation));

    print("chain_head: " + abng_chain_head(g));
    print("verify_chain: " + to_string(abng_verify_chain(g)));
}
"#;
