//! Phase 0.6 Item 6 — CJC-Lang source for the **scaled** compact_log
//! demo.
//!
//! Differences vs Phase 0.5's `compact_source.rs`:
//!   - 10^4 BeliefUpdate events (vs 5)
//!   - Spread across multiple nodes — exercises compact_log's
//!     per-touched-node deterministic ordering at scale
//!   - Verifies the chain still validates after compaction over a
//!     production-realistic event count
//!
//! Note: smart-replay speedup is measured Rust-side in
//! `bench/abng_micro/`; the CJC-Lang interface only exposes
//! `abng_replay` (= naive). This demo proves compact_log's
//! functional contract (marker emission + chain integrity) holds
//! at scale.

pub const SOURCE: &str = r#"
fn build_compact_scaled_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([2.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.5, 1.0);
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    abng_add_node(g, 0, 3);
    g
}

fn main() {
    let g = build_compact_scaled_graph(42);

    // Generate 10^4 observations spread across 4 leaves via the
    // codebook. Use abng_observe_slice for explicit per-row events
    // (vs abng_observe_batch which would collapse all into one).
    let n: i64 = 10000;
    let xs = Tensor.linspace(0.001, 0.999, n);

    // Bucket per-leaf observations.
    let leaf_obs_0 = [];
    let leaf_obs_1 = [];
    let leaf_obs_2 = [];
    let leaf_obs_3 = [];

    let i = 0;
    while i < n {
        let x = xs.get([i]);
        let prefix = abng_encode_prefix(g, Tensor.from_vec([x], [1]));
        let evidence = abng_descend(g, prefix);
        let leaf = int(evidence.get([1]));
        let y = sin(3.141592653589793 * x);
        if leaf == 1 {
            leaf_obs_0 = array_push(leaf_obs_0, y);
        } else if leaf == 2 {
            leaf_obs_1 = array_push(leaf_obs_1, y);
        } else if leaf == 3 {
            leaf_obs_2 = array_push(leaf_obs_2, y);
        } else if leaf == 4 {
            leaf_obs_3 = array_push(leaf_obs_3, y);
        }
        i = i + 1;
    }

    // Per-leaf slice observation: emits exactly len(leaf_obs) per-row
    // BeliefUpdate events (matching the legacy semantics).
    let n0 = array_len(leaf_obs_0);
    let n1 = array_len(leaf_obs_1);
    let n2 = array_len(leaf_obs_2);
    let n3 = array_len(leaf_obs_3);
    if n0 > 0 { abng_observe_slice(g, 1, Tensor.from_vec(leaf_obs_0, [n0])); }
    if n1 > 0 { abng_observe_slice(g, 2, Tensor.from_vec(leaf_obs_1, [n1])); }
    if n2 > 0 { abng_observe_slice(g, 3, Tensor.from_vec(leaf_obs_2, [n2])); }
    if n3 > 0 { abng_observe_slice(g, 4, Tensor.from_vec(leaf_obs_3, [n3])); }

    let audit_pre = abng_audit_len(g);
    print("audit_pre: " + to_string(audit_pre));

    // Compact: emit one StatsSnapshot per touched node up to seq.
    let emitted = abng_compact_log(g, audit_pre);
    print("emitted: " + to_string(emitted));

    let audit_post = abng_audit_len(g);
    print("audit_post: " + to_string(audit_post));

    // 5 distinct nodes touched in the prefix: root (via the
    // ChildrenPromoted events fired during add_node), plus the 4
    // children (their own NodeAdded + observe events).
    let exactly_five_emitted = emitted == 5;
    print("exactly_five_emitted: " + to_string(exactly_five_emitted));

    // The chain must still verify after compaction at scale.
    print("verify_post: " + to_string(abng_verify_chain(g)));

    print("audit_grew_correctly: "
        + to_string(audit_post == audit_pre + emitted));

    print("chain_head: " + abng_chain_head(g));
}
"#;
