//! Phase 0.6 Item 5 — CJC-Lang source for the **Split** trigger demo.
//!
//! Workload: a root-only graph trained on a bimodal observation
//! distribution (8 values near -0.7, 8 values near +0.7). A single
//! linear model fits poorly; a split at the median would give a
//! large held-out NLL gain. Run `decide_step` once → Split fires.
//!
//! Trigger fall-through: Compress, Merge can't fire (no children);
//! Split's gate is `samples_seen >= split_min` AND bootstrap
//! held-out ΔNLL gain >= `nll_split_gain`. Both met by construction.
//! Prune / Grow / Freeze sit later in the fall-through, so Split
//! pre-empts them.
//!
//! Headline: `action_counts[Split] >= 1` and `node_count` grew by
//! exactly 2 (Split appends two new children to the parent).

pub const SOURCE: &str = r#"
fn build_split_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    // 1-D quantile codebook with 4 bins straddling the bimodal
    // partition point at 0.
    let codebook = Tensor.from_vec([-0.5, 0.0, 0.5], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    // Thresholds engineered to fire **Split** specifically:
    //   [0]  H_grow                   = 1.0e308 (effectively disable Grow)
    //   [1]  grow_min                 = 1000000 (Grow won't fire)
    //   [2]  split_min                = 16     (matches our 16 samples)
    //   [3]  nll_split_gain           = 0.001  (low — easy NLL gate)
    //   [4]  impurity_min             = 0.001
    //   [5]  tau_merge                = 0      (no sibling, but defensive)
    //   [6]  kl_merge                 = 0.0
    //   [7]  prune_floor              = 4
    //   [8]  prune_grace_epochs       = 1000
    //   [9]  tau_compress             = 0
    //   [10] freeze_after             = 1000
    //   [11] drift_unfreeze           = 1.0e308
    //   [12] ece_stability_max_delta  = 0.005
    //   [13] sigma_stability_ratio    = 1.05
    let thresholds = Tensor.from_vec(
        [
            1.0e308, 1000000.0, 16.0, 0.001,
            0.001, 0.0, 0.0,
            4.0, 1000.0,
            0.0, 1000.0,
            1.0e308,
            0.005, 1.05
        ],
        [14]
    );
    abng_set_decision_policy(g, thresholds);
    g
}

fn main() {
    let g = build_split_graph(42);
    let n_nodes_pre = abng_node_count(g);
    print("n_nodes_pre: " + to_string(n_nodes_pre));

    // Bimodal: 8 in low cluster around -0.7, 8 in high cluster
    // around +0.7. A held-out partition at 0 reduces NLL sharply.
    let xs = Tensor.from_vec(
        [
            -0.80, -0.75, -0.72, -0.70, -0.68, -0.65, -0.62, -0.60,
             0.60,  0.62,  0.65,  0.68,  0.70,  0.72,  0.75,  0.80
        ],
        [16]
    );
    abng_observe_slice(g, 0, xs);

    let counts = abng_decide_step(g);

    let grow_count = abng_action_count(g, 0);
    let split_count = abng_action_count(g, 1);
    let merge_count = abng_action_count(g, 2);
    let prune_count = abng_action_count(g, 3);
    let compress_count = abng_action_count(g, 4);
    let freeze_count = abng_action_count(g, 5);
    print("grow_count: " + to_string(grow_count));
    print("split_count: " + to_string(split_count));
    print("merge_count: " + to_string(merge_count));
    print("prune_count: " + to_string(prune_count));
    print("compress_count: " + to_string(compress_count));
    print("freeze_count: " + to_string(freeze_count));

    let n_nodes_post = abng_node_count(g);
    print("n_nodes_post: " + to_string(n_nodes_post));

    let only_split_fired = grow_count == 0 && split_count > 0
        && merge_count == 0 && prune_count == 0
        && compress_count == 0 && freeze_count == 0;
    print("only_split_fired: " + to_string(only_split_fired));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
