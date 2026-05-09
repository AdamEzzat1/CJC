//! Phase 0.6 Item 5 — ABNG **Prune** trigger demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn prune_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::prune_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("prune_count:")));
    assert!(out.iter().any(|l| l.starts_with("only_prune_fired:")));
}

#[test]
fn prune_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::prune_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("prune_count:")));
}

#[test]
fn prune_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::prune_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("prune_count:")));
}

#[test]
fn prune_cjcl_prune_count_is_positive() {
    let out = run_parity(abng_demos::prune_source::SOURCE, SEED);
    let prune: u64 = extract_value(&out, "prune_count").parse().unwrap();
    assert!(prune >= 1, "expected Prune to fire at least once, got {prune}");
}

#[test]
fn prune_cjcl_no_other_trigger_fired() {
    let out = run_parity(abng_demos::prune_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "only_prune_fired"), "true");
}

#[test]
fn prune_cjcl_node_count_unchanged() {
    // Prune marks a node inactive but does not shrink the arena —
    // node_count stays the same.
    let out = run_parity(abng_demos::prune_source::SOURCE, SEED);
    let pre: u64 = extract_value(&out, "n_nodes_pre").parse().unwrap();
    let post: u64 = extract_value(&out, "n_nodes_post").parse().unwrap();
    assert_eq!(post, pre, "Prune doesn't shrink the arena");
}

#[test]
fn prune_cjcl_audit_chain_verifies_post_prune() {
    let out = run_parity(abng_demos::prune_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn prune_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::prune_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("prune cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "e7279f0999bbac3cb97c5765d4bb0433cd610de28db740d064bdcd928d824711";
    assert_eq!(
        chain, CANARY_HEX,
        "prune cjcl chain_head canary mismatch — see comment"
    );
}
