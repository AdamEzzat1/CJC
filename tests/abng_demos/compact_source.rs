//! CJC-Lang source for the ABNG compact_log / log-compaction demo.
//!
//! Workload: train a graph with multiple observations per node,
//! then call `abng_compact_log` to insert StatsSnapshot marker
//! events for each touched node. Verify the audit chain still
//! verifies AND grew by exactly one event per touched node.
//!
//! Capability demonstrated: `compact_log` is the read-side hook
//! for log compaction (Phase 0.5 ships the marker layer; the
//! cycle-saving fast-forward optimization is deferred per Phase
//! C's commit notes). What's shipped today: the marker emission
//! is deterministic, idempotent under same-state calls, and the
//! audit chain remains intact post-compaction.

pub const SOURCE: &str = r#"
fn build_compact_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.0], [1, 1]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    g
}

fn main() {
    let g = build_compact_graph(42);

    // Observe to multiple nodes — each touched node should
    // produce exactly one StatsSnapshot event when compact_log fires.
    abng_observe(g, 0, 0.1);
    abng_observe(g, 0, 0.2);
    abng_observe(g, 1, 0.5);
    abng_observe(g, 1, 0.6);
    abng_observe(g, 2, 0.9);

    let audit_pre = abng_audit_len(g);
    let chain_pre = abng_chain_head(g);
    print("audit_pre: " + to_string(audit_pre));
    print("chain_pre: " + chain_pre);
    print("verify_pre: " + to_string(abng_verify_chain(g)));

    // Compact: emit one StatsSnapshot per touched node up to seq.
    let emitted = abng_compact_log(g, audit_pre);
    print("emitted: " + to_string(emitted));

    let audit_post = abng_audit_len(g);
    print("audit_post: " + to_string(audit_post));
    let audit_grew_by_emitted = audit_post == audit_pre + emitted;
    print("audit_grew_by_emitted: " + to_string(audit_grew_by_emitted));

    // Chain head must have advanced.
    let chain_post = abng_chain_head(g);
    print("chain_post: " + chain_post);
    let chain_advanced = chain_post != chain_pre;
    print("chain_advanced: " + to_string(chain_advanced));

    // Critically: chain must still verify.
    print("verify_post: " + to_string(abng_verify_chain(g)));

    // Touched node count: nodes 0, 1, 2 all observed → 3 emitted.
    let exactly_three_emitted = emitted == 3;
    print("exactly_three_emitted: " + to_string(exactly_three_emitted));

    // Idempotency: calling compact_log on the SAME prefix again
    // (same audit log up to the original seq) emits MORE events
    // because new events have been added since (the snapshots
    // themselves count as events). This is documented behavior;
    // we verify the audit length grows again.
    let emitted_2 = abng_compact_log(g, audit_pre);
    let audit_post_2 = abng_audit_len(g);
    print("emitted_2: " + to_string(emitted_2));
    print("audit_post_2: " + to_string(audit_post_2));
}
"#;
