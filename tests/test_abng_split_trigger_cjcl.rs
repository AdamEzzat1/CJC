//! Phase 0.6 Item 5 — ABNG **Split** trigger demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn split_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::split_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("split_count:")));
    assert!(out.iter().any(|l| l.starts_with("only_split_fired:")));
}

#[test]
fn split_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::split_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("split_count:")));
}

#[test]
fn split_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::split_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("split_count:")));
}

#[test]
fn split_cjcl_split_count_is_positive() {
    let out = run_parity(abng_demos::split_source::SOURCE, SEED);
    let split: u64 = extract_value(&out, "split_count").parse().unwrap();
    assert!(split >= 1, "expected Split to fire at least once, got {split}");
}

#[test]
fn split_cjcl_no_other_trigger_fired() {
    let out = run_parity(abng_demos::split_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "only_split_fired"), "true");
}

#[test]
fn split_cjcl_node_count_grew_by_two() {
    // Split appends exactly 2 child nodes to the parent (the
    // bisected partitions). Verify node_count went 1 → 3.
    let out = run_parity(abng_demos::split_source::SOURCE, SEED);
    let pre: u64 = extract_value(&out, "n_nodes_pre").parse().unwrap();
    let post: u64 = extract_value(&out, "n_nodes_post").parse().unwrap();
    assert_eq!(
        post,
        pre + 2,
        "Split appends two new children, so node_count should grow by 2"
    );
}

#[test]
fn split_cjcl_audit_chain_verifies_post_split() {
    let out = run_parity(abng_demos::split_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn split_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::split_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("split cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "5dc83627f71267701eb8187fbb7c9a7ba757480788693a47def8cb7b8cc159e0";
    assert_eq!(
        chain, CANARY_HEX,
        "split cjcl chain_head canary mismatch — see comment"
    );
}
