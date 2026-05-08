//! CJC-Lang source for the ABNG OOD-detection demo.
//!
//! Workload: a graph that has children for bins 0, 1, 2 but NOT
//! bin 3. Train the density tracker heavily on bins 0 and 1, lightly
//! on bin 2. Then probe all four bins and assert the composite
//! `ood_score` cleanly partitions:
//!
//!   in-distribution dense  <  in-distribution sparse
//!     <  unseen  <  routing-fall-off
//!
//! This demonstrates ALL THREE ABNG OOD signals composed:
//!   * density_score: high when far from training cluster
//!   * epistemic_z: BLR's leverage at the query point
//!   * prefix_distance: high when descend fails to match a child
//!
//! The "model abstains when it shouldn't trust itself" story.

pub const SOURCE: &str = r#"
fn build_ood_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
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
    // Add children for bins 0, 1, 2 ONLY. Bin 3 has no child —
    // any input falling in bin 3 will descend with matched_prefix=0
    // (the routing-fall-off signal).
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    g
}

// Training: density_observe wants 2-D [n, d] features. We feed
// each leaf a batch of `n` 4-D samples drawn from an x-region.
fn train_leaf_density(g: i64, leaf: i64, x_center: f64, n: i64) {
    let i = 0;
    let buf = [];
    while i < n {
        let xi = x_center + 0.01 * float(i - n / 2);
        buf = array_push(buf, 1.0);
        buf = array_push(buf, xi);
        buf = array_push(buf, xi * xi);
        buf = array_push(buf, xi * xi * xi);
        i = i + 1;
    }
    let batch = Tensor.from_vec(buf, [n, 4]);
    abng_density_observe(g, leaf, batch);
}

// Compute features [1, x, x^2, x^3] (1-D for predict / ood_score).
fn features_1d(x: f64) -> Tensor {
    Tensor.from_vec([1.0, x, x * x, x * x * x], [4])
}

// Probe a query point: descend, then call ood_score.
fn probe_ood(g: i64, x: f64) -> f64 {
    let x_t = Tensor.from_vec([x], [1]);
    let prefix = abng_encode_prefix(g, x_t);
    let evidence = abng_descend(g, prefix);
    let matched = int(evidence.get([0]));
    let leaf = int(evidence.get([1]));
    let phi = features_1d(x);
    abng_ood_score(g, leaf, phi, matched, 1)
}

fn main() {
    let g = build_ood_graph(7);

    // Dense training on bin 0 (x in [0, 0.25)).
    train_leaf_density(g, 1, 0.125, 32);
    // Dense training on bin 1 (x in [0.25, 0.5)).
    train_leaf_density(g, 2, 0.375, 32);
    // Sparse training on bin 2 (x in [0.5, 0.75)) — only 4 samples.
    train_leaf_density(g, 3, 0.625, 4);
    // Bin 3 (x in [0.75, 1]): no child node, so descent will
    // matched_prefix=0 (routing-fall-off).

    // Probe one in-distribution dense point per bin 0 & 1, one
    // sparse point in bin 2, and one routing-fall-off point in bin 3.
    let ood_bin0 = probe_ood(g, 0.10);
    let ood_bin1 = probe_ood(g, 0.40);
    let ood_bin2 = probe_ood(g, 0.62);
    let ood_bin3 = probe_ood(g, 0.90);

    print("ood_bin0: " + to_string(ood_bin0));
    print("ood_bin1: " + to_string(ood_bin1));
    print("ood_bin2: " + to_string(ood_bin2));
    print("ood_bin3: " + to_string(ood_bin3));

    // Range invariant: every score in [0, 1].
    let in_range = ood_bin0 >= 0.0 && ood_bin0 <= 1.0
        && ood_bin1 >= 0.0 && ood_bin1 <= 1.0
        && ood_bin2 >= 0.0 && ood_bin2 <= 1.0
        && ood_bin3 >= 0.0 && ood_bin3 <= 1.0;
    print("scores_in_range: " + to_string(in_range));

    // Headline benefit: dense < sparse < routing-fall-off.
    // (We don't enforce ood_bin0 < ood_bin1 because both are
    //  dense regions and may tie; we DO enforce dense < sparse
    //  and sparse < routing-fall-off as strict separations.)
    let dense_max = ood_bin0;
    if ood_bin1 > dense_max { dense_max = ood_bin1; }
    let sparse_below_falloff = ood_bin2 < ood_bin3;
    let dense_below_sparse = dense_max < ood_bin2;
    print("dense_max: " + to_string(dense_max));
    print("dense_below_sparse: " + to_string(dense_below_sparse));
    print("sparse_below_falloff: " + to_string(sparse_below_falloff));
    let three_tier_separation = dense_below_sparse && sparse_below_falloff;
    print("three_tier_separation: " + to_string(three_tier_separation));

    // Lineage hashes for canary.
    print("chain_head: " + abng_chain_head(g));
    print("verify_chain: " + to_string(abng_verify_chain(g)));
}
"#;
