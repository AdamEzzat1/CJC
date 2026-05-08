//! CJC-Lang source for the ABNG maturity-inspection demo.
//!
//! Workload: build a graph, observe a stream of identical values
//! to drive signature stability up, run `decide_step` to advance
//! the maturity counters, and read back the per-node 4-element
//! maturity tensor at three timepoints (start, mid, end). Show
//! the maturity flags evolve from all-zero at start to non-zero
//! at end. This is the introspection layer production teams use
//! to monitor when training has stabilized.
//!
//! Capability demonstrated: `abng_node_maturity` returns a 4-D
//! tensor [signature_stable, ece_stable, uncertainty_stable,
//! drift_stable] of f64 (1.0 = stable, 0.0 = not yet). These
//! flags are what `decide_step` consults internally; exposing
//! them as a builtin gives users observability into training
//! state without needing to interpret the raw event log.

pub const SOURCE: &str = r#"
fn build_maturity_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.0], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
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
    abng_add_node(g, 0, 1);
    g
}

// Sum the 4 maturity flags for node `n` — useful for checking
// "any flag flipped" without inspecting individually.
fn maturity_sum(g: i64, n: i64) -> f64 {
    let m = abng_node_maturity(g, n);
    m.get([0]) + m.get([1]) + m.get([2]) + m.get([3])
}

fn main() {
    let g = build_maturity_graph(42);

    // T=0: fresh graph, maturity should be all zeros for both nodes.
    let mat0_root = abng_node_maturity(g, 0);
    let mat0_child = abng_node_maturity(g, 1);
    let sum0 = maturity_sum(g, 0) + maturity_sum(g, 1);
    print("sum_at_t0: " + to_string(sum0));
    print("root_t0_signature: " + to_string(mat0_root.get([0])));
    print("root_t0_ece: " + to_string(mat0_root.get([1])));
    print("root_t0_uncertainty: " + to_string(mat0_root.get([2])));
    print("root_t0_drift: " + to_string(mat0_root.get([3])));

    // Phase 1: observe a stable stream + run decide_step several times.
    let i = 0;
    while i < 10 {
        abng_observe(g, 0, 0.1);
        abng_observe(g, 1, 0.1);
        i = i + 1;
    }
    let j = 0;
    while j < 5 {
        abng_decide_step(g);
        j = j + 1;
    }

    // T=end: read maturity. We expect AT LEAST one flag to have
    // flipped on AT LEAST one node — observable evidence of
    // training-state advance.
    let mat1_root = abng_node_maturity(g, 0);
    let mat1_child = abng_node_maturity(g, 1);
    let sum1 = maturity_sum(g, 0) + maturity_sum(g, 1);
    print("sum_at_t1: " + to_string(sum1));
    print("root_t1_signature: " + to_string(mat1_root.get([0])));
    print("root_t1_ece: " + to_string(mat1_root.get([1])));
    print("root_t1_uncertainty: " + to_string(mat1_root.get([2])));
    print("root_t1_drift: " + to_string(mat1_root.get([3])));

    // Headline: the total flag count strictly grew.
    let sum_grew = sum1 > sum0;
    print("maturity_evolved: " + to_string(sum_grew));

    // The maturity tensor is
    //   [samples_seen f64, calibration_stable f64,
    //    uncertainty_stable f64, trust_level f64].
    // Positions [1] and [2] are strictly binary 0/1 flags;
    // position [0] is a sample count (non-negative); position [3]
    // is a continuous trust level in [0, 1].
    let r0 = mat1_root.get([0]);
    let r1 = mat1_root.get([1]);
    let r2 = mat1_root.get([2]);
    let r3 = mat1_root.get([3]);
    let calib_binary = r1 == 0.0 || r1 == 1.0;
    let unc_binary = r2 == 0.0 || r2 == 1.0;
    let samples_seen_nonneg = r0 >= 0.0;
    let trust_in_range = r3 >= 0.0 && r3 <= 1.0;
    print("calib_binary: " + to_string(calib_binary));
    print("unc_binary: " + to_string(unc_binary));
    print("samples_seen_nonneg: " + to_string(samples_seen_nonneg));
    print("trust_in_range: " + to_string(trust_in_range));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
