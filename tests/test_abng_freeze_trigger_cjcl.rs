//! Phase 0.6 Item 5 — ABNG **Freeze** trigger demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn freeze_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::freeze_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("freeze_count:")));
    assert!(out.iter().any(|l| l.starts_with("only_freeze_fired:")));
}

#[test]
fn freeze_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::freeze_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("freeze_count:")));
}

#[test]
fn freeze_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::freeze_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("freeze_count:")));
}

#[test]
fn freeze_cjcl_freeze_count_is_positive() {
    let out = run_parity(abng_demos::freeze_source::SOURCE, SEED);
    let freeze: u64 = extract_value(&out, "freeze_count").parse().unwrap();
    assert!(freeze >= 1, "expected Freeze to fire at least once, got {freeze}");
}

#[test]
fn freeze_cjcl_no_other_trigger_fired() {
    let out = run_parity(abng_demos::freeze_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "only_freeze_fired"), "true");
}

#[test]
fn freeze_cjcl_root_is_frozen_post_decide_step() {
    // Headline: the engineered workload flips root.is_frozen from
    // false → true. After Freeze fires, the live graph reports
    // is_frozen=true via the abng_is_frozen builtin.
    let out = run_parity(abng_demos::freeze_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "is_frozen"), "true");
}

#[test]
fn freeze_cjcl_audit_chain_verifies_post_freeze() {
    let out = run_parity(abng_demos::freeze_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn freeze_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::freeze_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("freeze cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "921ec636b218f06a1378555eeeb44d84375fba32d404e5d2fa5a61a69cb7124b";
    assert_eq!(
        chain, CANARY_HEX,
        "freeze cjcl chain_head canary mismatch — see comment"
    );
}
