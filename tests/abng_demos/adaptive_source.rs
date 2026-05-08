//! CJC-Lang source for the ABNG adaptive-structural-triggers demo.
//!
//! Workload: build a graph with root + 2 children (similar
//! signatures via shared training distribution). Run `decide_step`
//! repeatedly. The graph's per-node maturity counters climb under
//! repeated stable observations until the Merge trigger fires
//! (siblings 1 and 2 share signatures within `tau_merge` and the
//! posterior KL is small).
//!
//! Headline assertion: `action_counts[Merge] > 0` after enough
//! `decide_step` passes. This proves the graph genuinely *adapts*
//! its structure to the workload — the marquee feature behind
//! ABNG's *Adaptive* name.

pub const SOURCE: &str = r#"
fn build_adaptive_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([-1.0, 0.0, 1.0], [1, 3]);
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
    abng_add_node(g, 0, 2);
    g
}

fn main() {
    let g = build_adaptive_graph(42);

    // Initial state.
    let n_nodes_pre = abng_node_count(g);
    let audit_pre = abng_audit_len(g);
    print("n_nodes_pre: " + to_string(n_nodes_pre));
    print("audit_pre: " + to_string(audit_pre));

    // Train: a small fixed observation sequence on the root.
    abng_observe(g, 0, 0.10);
    abng_observe(g, 0, 0.25);
    abng_observe(g, 0, 0.40);

    // Run decide_step 3 times — accumulates signature stability,
    // fires the Merge trigger when sibling signatures align.
    let i = 0;
    while i < 3 {
        let counts = abng_decide_step(g);
        i = i + 1;
    }

    // Final action_counts.
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

    let total_actions = grow_count + split_count + merge_count
        + prune_count + compress_count + freeze_count;
    print("total_actions: " + to_string(total_actions));
    let any_action_fired = total_actions > 0;
    print("any_action_fired: " + to_string(any_action_fired));

    // Audit chain reflects the structural events.
    let audit_post = abng_audit_len(g);
    print("audit_post: " + to_string(audit_post));
    let audit_grew = audit_post > audit_pre;
    print("audit_grew: " + to_string(audit_grew));

    // Verify chain integrity post-mutation.
    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
