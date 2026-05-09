//! Phase 0.6 Item 5 — CJC-Lang source for the ABNG **Grow** trigger demo.
//!
//! Workload: a root-only graph, observe enough values across the
//! codebook's range to give the route-key entropy at the candidate
//! depth a comfortable margin over `H_grow`. Run `decide_step` once.
//!
//! Trigger fall-through (architecture doc §3.7):
//!     Compress → Merge → Split → Prune → Grow → Freeze
//! To fire **Grow** specifically:
//!   - no children at root → Compress and Merge cannot fire on root
//!   - leaf with samples_seen >= grow_min
//!   - samples_seen < split_min so Split's `n >= split_min` gate
//!     fails and Split doesn't pre-empt
//!   - samples_seen >= prune_floor so Prune's "few samples" gate fails
//!   - leaf still has key-bytes available at the candidate depth (no
//!     children yet, so all 256 bytes are unbound)
//!   - route-key entropy at candidate depth > H_grow
//!
//! Headline: `action_counts[Grow] >= 1` and `node_count` grew from 1
//! to 2+ after one `decide_step`. This proves the engine can grow
//! topology under live evidence.

pub const SOURCE: &str = r#"
fn build_grow_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    // 1-D quantile codebook with 4 bins (3 boundaries) — gives the
    // route-key 2 bits of entropy if observations cover all bins.
    let codebook = Tensor.from_vec([-0.5, 0.0, 0.5], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    // Thresholds engineered to fire **Grow** specifically:
    //   [0]  H_grow                   = 0.05  (low — easy entropy gate)
    //   [1]  grow_min                 = 8     (low — easy evidence gate)
    //   [2]  split_min                = 10000 (huge — Split won't preempt)
    //   [3]  nll_split_gain           = 0.5   (high — Split won't preempt)
    //   [4]  impurity_min             = 0.001 (low — easy)
    //   [5]  tau_merge                = 0     (impossible Hamming match)
    //   [6]  kl_merge                 = 0.0   (impossible)
    //   [7]  prune_floor              = 4     (we'll have >= 8 samples)
    //   [8]  prune_grace_epochs       = 1000  (Prune won't fire in 1 step)
    //   [9]  tau_compress             = 0     (no children to compress)
    //   [10] freeze_after             = 1000  (Freeze won't fire)
    //   [11] drift_unfreeze           = 1.0e308 (disabled)
    //   [12] ece_stability_max_delta  = 0.005
    //   [13] sigma_stability_ratio    = 1.05
    let thresholds = Tensor.from_vec(
        [
            0.05, 8.0, 10000.0, 0.5,
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
    let g = build_grow_graph(42);

    let n_nodes_pre = abng_node_count(g);
    print("n_nodes_pre: " + to_string(n_nodes_pre));

    // Observe 16 values spread evenly across the 4 codebook bins
    // — gives the route-key high entropy at the candidate depth.
    let xs = Tensor.from_vec(
        [
            -0.9, -0.7, -0.6, -0.55,
            -0.3, -0.2, -0.15, -0.05,
             0.05,  0.15,  0.2,  0.3,
             0.55,  0.6,   0.7,  0.9
        ],
        [16]
    );
    abng_observe_slice(g, 0, xs);

    let n_seen = abng_node_count(g);
    print("samples_observed: 16");

    // One decide_step pass.
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
    let nodes_grew = n_nodes_post > n_nodes_pre;
    print("nodes_grew: " + to_string(nodes_grew));

    let only_grow_fired = grow_count > 0
        && split_count == 0 && merge_count == 0
        && prune_count == 0 && compress_count == 0
        && freeze_count == 0;
    print("only_grow_fired: " + to_string(only_grow_fired));

    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
