//! Phase 0.6 Item 6 — ABNG **scaled OOD** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn ood_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval, abng_demos::ood_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("three_tier_separation:")));
}

#[test]
fn ood_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir, abng_demos::ood_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("three_tier_separation:")));
}

#[test]
fn ood_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::ood_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn ood_scaled_three_tier_holds_at_8d() {
    let out = run_parity(abng_demos::ood_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "three_tier_separation"), "true");
}

#[test]
fn ood_scaled_scores_in_range() {
    let out = run_parity(abng_demos::ood_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "scores_in_range"), "true");
}

#[test]
fn ood_scaled_audit_chain_verifies() {
    let out = run_parity(abng_demos::ood_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn ood_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::ood_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("ood_scaled cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "7fef4e4fddaa980e58c7e4ba5d9e007bfa6e44bf3f3ff2dab67b2906df91bb2f";
    assert_eq!(
        chain, CANARY_HEX,
        "ood_scaled cjcl chain_head canary mismatch — see comment"
    );
}
