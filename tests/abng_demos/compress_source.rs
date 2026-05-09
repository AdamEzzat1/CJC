//! Phase 0.6 Item 5 — CJC-Lang source for the **Compress** trigger demo.
//!
//! Workload: root with 2 children. All three nodes are observed on
//! the same distribution so their `NodeSignature` values converge.
//! Run `decide_step` → Compress fires on the root because all child
//! signatures sit within `tau_compress` Hamming of the root's
//! signature.
//!
//! Trigger fall-through: Compress is FIRST in the ladder
//! (Compress → Merge → Split → Prune → Grow → Freeze), so once its
//! gate (children present + all child sigs within tau_compress Hamming)
//! is met, it pre-empts everything else for that node. We disable
//! Merge by setting `tau_merge = 0` (impossible Hamming match) and
//! `kl_merge = 0` (impossible posterior closeness).
//!
//! Headline: `action_counts[Compress] >= 1`. Compress replaces the
//! parent's `AdaptiveChildren` with a `Dense` container — the
//! sub-tree becomes signature-routed instead of byte-keyed.

pub const SOURCE: &str = r#"
fn build_compress_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.0], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    // Thresholds engineered to fire **Compress** specifically:
    //   [0]  H_grow                   = 1.0e308 (Grow disabled)
    //   [1]  grow_min                 = 1000000
    //   [2]  split_min                = 1000000 (Split disabled)
    //   [3]  nll_split_gain           = 1.0e308
    //   [4]  impurity_min             = 0.001
    //   [5]  tau_merge                = 0      (Merge impossible)
    //   [6]  kl_merge                 = 0.0
    //   [7]  prune_floor              = 100    (children have many samples)
    //   [8]  prune_grace_epochs       = 1000
    //   [9]  tau_compress             = 32     (max possible Hamming — always passes)
    //   [10] freeze_after             = 1000
    //   [11] drift_unfreeze           = 1.0e308
    //   [12] ece_stability_max_delta  = 0.005
    //   [13] sigma_stability_ratio    = 1.05
    let thresholds = Tensor.from_vec(
        [
            1.0e308, 1000000.0, 1000000.0, 1.0e308,
            0.001, 0.0, 0.0,
            100.0, 1000.0,
            32.0, 1000.0,
            1.0e308,
            0.005, 1.05
        ],
        [14]
    );
    abng_set_decision_policy(g, thresholds);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    g
}

fn main() {
    let g = build_compress_graph(42);
    let n_nodes_pre = abng_node_count(g);
    print("n_nodes_pre: " + to_string(n_nodes_pre));

    // Same training distribution applied to root + both children:
    // their Welford-folded signatures converge as we observe.
    let xs = Tensor.from_vec(
        [0.10, 0.20, 0.30, 0.15, 0.25, 0.05, 0.18, 0.22],
        [8]
    );
    abng_observe_slice(g, 0, xs);
    let ys = Tensor.from_vec(
        [0.12, 0.18, 0.22, 0.28, 0.10, 0.20, 0.15, 0.25],
        [8]
    );
    abng_observe_slice(g, 1, ys);
    let zs = Tensor.from_vec(
        [0.14, 0.16, 0.24, 0.26, 0.11, 0.21, 0.19, 0.23],
        [8]
    );
    abng_observe_slice(g, 2, zs);

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

    let only_compress_fired = grow_count == 0 && split_count == 0
        && merge_count == 0 && prune_count == 0
        && compress_count > 0 && freeze_count == 0;
    print("only_compress_fired: " + to_string(only_compress_fired));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
