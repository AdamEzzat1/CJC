//! Phase 0.6 Item 5 — CJC-Lang source for the **Freeze** trigger demo.
//!
//! Workload: a root-only graph observed once with a small stable
//! batch. Run `decide_step` repeatedly; with no further observations,
//! the Welford-folded `NodeSignature` doesn't move, so
//! `signature_stable_calls` accumulates each pass. Once
//! `signature_stable_calls >= freeze_after`, Freeze fires.
//!
//! Trigger fall-through: Compress can't fire (no children), Merge
//! can't fire (no sibling), Split / Grow / Prune all gated off via
//! impossible thresholds. Freeze sits last in the ladder, so it
//! fires once its specific gate is met.
//!
//! Headline: `action_counts[Freeze] >= 1`. Once frozen, a node's
//! `is_frozen` flag flips to `true`; subsequent `decide_step`
//! passes skip structural mutations on it.

pub const SOURCE: &str = r#"
fn build_freeze_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.0], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    // Thresholds engineered to fire **Freeze** specifically:
    //   [0]  H_grow                   = 1.0e308 (Grow disabled)
    //   [1]  grow_min                 = 1000000
    //   [2]  split_min                = 1000000 (Split disabled)
    //   [3]  nll_split_gain           = 1.0e308
    //   [4]  impurity_min             = 0.001
    //   [5]  tau_merge                = 0       (no Merge)
    //   [6]  kl_merge                 = 0.0
    //   [7]  prune_floor              = 100     (root never pruned anyway)
    //   [8]  prune_grace_epochs       = 1000
    //   [9]  tau_compress             = 0       (no children, but defensive)
    //   [10] freeze_after             = 1       (fire after ONE stable pass)
    //   [11] drift_unfreeze           = 1.0e308
    //   [12] ece_stability_max_delta  = 0.005
    //   [13] sigma_stability_ratio    = 1.05
    let thresholds = Tensor.from_vec(
        [
            1.0e308, 1000000.0, 1000000.0, 1.0e308,
            0.001, 0.0, 0.0,
            100.0, 1000.0,
            0.0, 1.0,
            1.0e308,
            0.005, 1.05
        ],
        [14]
    );
    abng_set_decision_policy(g, thresholds);
    g
}

fn main() {
    let g = build_freeze_graph(42);

    // Single batch of stable observations on the root. Welford fold
    // converges; subsequent decide_step calls don't shift the
    // signature, so signature_stable_calls accumulates.
    let xs = Tensor.from_vec(
        [0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.50],
        [8]
    );
    abng_observe_slice(g, 0, xs);

    // Run decide_step a handful of times. The signature stabilizes
    // after the first pass; Freeze fires once stable_calls >= freeze_after.
    let i = 0;
    while i < 4 {
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

    let only_freeze_fired = grow_count == 0 && split_count == 0
        && merge_count == 0 && prune_count == 0
        && compress_count == 0 && freeze_count > 0;
    print("only_freeze_fired: " + to_string(only_freeze_fired));

    // Once frozen, the root's is_frozen flag is set.
    print("is_frozen: " + to_string(abng_is_frozen(g, 0)));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
