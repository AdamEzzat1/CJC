//! Phase 0.6 Item 5 — CJC-Lang source for the **Prune** trigger demo.
//!
//! Workload: root + one child. The child gets a small fixed number
//! of observations (below `prune_floor`). Run `decide_step` in a
//! loop; signature_stable_calls accumulates each pass until it
//! reaches `prune_grace_epochs`, then Prune fires on the child.
//!
//! Trigger fall-through: Compress can't fire (no grandchildren),
//! Merge can't fire (only one child, no sibling), Split / Grow gates
//! engineered to fail (huge thresholds), Prune fires when its
//! evidence + grace conditions both hold. Root is never pruned by
//! contract.
//!
//! Headline: `action_counts[Prune] >= 1` and the child node's
//! `is_active` flag flipped to false (we don't have a builtin to
//! query that directly here, so we infer via the trigger count).

pub const SOURCE: &str = r#"
fn build_prune_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.0], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    // Thresholds engineered to fire **Prune** specifically:
    //   [0]  H_grow                   = 1.0e308 (Grow disabled)
    //   [1]  grow_min                 = 1000000
    //   [2]  split_min                = 1000000 (Split disabled)
    //   [3]  nll_split_gain           = 1.0e308
    //   [4]  impurity_min             = 0.001
    //   [5]  tau_merge                = 0      (no Merge)
    //   [6]  kl_merge                 = 0.0
    //   [7]  prune_floor              = 8      (child has 2 samples; 2<8)
    //   [8]  prune_grace_epochs       = 2      (Prune after 2 passes)
    //   [9]  tau_compress             = 0      (no Compress)
    //   [10] freeze_after             = 1000   (Freeze won't fire)
    //   [11] drift_unfreeze           = 1.0e308
    //   [12] ece_stability_max_delta  = 0.005
    //   [13] sigma_stability_ratio    = 1.05
    let thresholds = Tensor.from_vec(
        [
            1.0e308, 1000000.0, 1000000.0, 1.0e308,
            0.001, 0.0, 0.0,
            8.0, 2.0,
            0.0, 1000.0,
            1.0e308,
            0.005, 1.05
        ],
        [14]
    );
    abng_set_decision_policy(g, thresholds);
    abng_add_node(g, 0, 1);
    g
}

fn main() {
    let g = build_prune_graph(42);
    let n_nodes_pre = abng_node_count(g);
    print("n_nodes_pre: " + to_string(n_nodes_pre));

    // A few observations on the child — well below prune_floor=8.
    abng_observe(g, 1, 0.10);
    abng_observe(g, 1, 0.20);

    // Run decide_step several times. Signature stability accumulates
    // each pass; Prune fires once stable_calls >= prune_grace_epochs.
    let i = 0;
    while i < 5 {
        let counts = abng_decide_step(g);
        i = i + 1;
    }

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

    // Note: prune doesn't shrink the arena — node count stays the
    // same; the child just becomes is_active=false. We assert via
    // the action_counts.
    let n_nodes_post = abng_node_count(g);
    print("n_nodes_post: " + to_string(n_nodes_post));

    let only_prune_fired = grow_count == 0 && split_count == 0
        && merge_count == 0 && prune_count > 0
        && compress_count == 0 && freeze_count == 0;
    print("only_prune_fired: " + to_string(only_prune_fired));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
