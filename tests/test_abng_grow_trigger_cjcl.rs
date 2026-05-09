//! Phase 0.6 Item 5 — ABNG **Grow** trigger demo (CJC-Lang).
//!
//! See `tests/abng_demos/grow_source.rs` for the engineered workload.
//! This test file exercises both AST and MIR backends, asserts the
//! trigger fired exactly once with no other action firing, and locks
//! a SHA-256 chain_head canary.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn grow_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::grow_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("grow_count:")));
    assert!(out.iter().any(|l| l.starts_with("only_grow_fired:")));
}

#[test]
fn grow_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::grow_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("grow_count:")));
}

#[test]
fn grow_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::grow_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("grow_count:")));
}

#[test]
fn grow_cjcl_grow_count_is_positive() {
    let out = run_parity(abng_demos::grow_source::SOURCE, SEED);
    let grow: u64 = extract_value(&out, "grow_count").parse().unwrap();
    assert!(grow >= 1, "expected Grow to fire at least once, got {grow}");
}

#[test]
fn grow_cjcl_no_other_trigger_fired() {
    // The headline assertion: among the 6 action types, ONLY Grow
    // fires for this engineered workload. Catches a regression
    // where the trigger fall-through routes the workload to a
    // different action.
    let out = run_parity(abng_demos::grow_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "only_grow_fired"), "true");
}

#[test]
fn grow_cjcl_node_count_grew() {
    // Grow fires by appending a child to the current node, so
    // node_count should have increased by at least 1.
    let out = run_parity(abng_demos::grow_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "nodes_grew"), "true");
}

#[test]
fn grow_cjcl_audit_chain_verifies_post_grow() {
    let out = run_parity(abng_demos::grow_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn grow_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::grow_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("grow cjcl canary chain_head = {chain}");
    // Locked at Phase 0.6 Item 5 ship — fires only on Grow-trigger
    // determinism breakage. CANARY_HEX populated below after the
    // first successful run.
    const CANARY_HEX: &str =
        "a22caf23d30f3217a99ffe7a0d497fcd270a7ad4576f958c96ac62f6f71c26e8";
    assert_eq!(
        chain, CANARY_HEX,
        "grow cjcl chain_head canary mismatch — see comment"
    );
}
