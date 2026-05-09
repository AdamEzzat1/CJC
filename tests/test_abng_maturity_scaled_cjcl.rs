//! Phase 0.6 Item 6 — ABNG **scaled maturity** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn maturity_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval, abng_demos::maturity_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("final_sum:")));
}

#[test]
fn maturity_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir, abng_demos::maturity_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("final_sum:")));
}

#[test]
fn maturity_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::maturity_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn maturity_scaled_flags_evolved() {
    // Headline: across 1000 decide_step calls on a stable workload,
    // the maturity flags (signature/ece/uncertainty/drift) climbed
    // from all-zero to a strictly positive sum.
    let out = run_parity(abng_demos::maturity_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "maturity_increased"), "true");
}

#[test]
fn maturity_scaled_audit_chain_verifies_after_1000_decide_steps() {
    // Critical: the audit chain must remain valid after a long
    // decide_step loop — proves no state drift or chain corruption
    // accumulates over production-realistic loop counts.
    let out = run_parity(abng_demos::maturity_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn maturity_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::maturity_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("maturity_scaled cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "01eb4040fbc6b8130eb04b28ba556555a6dd35c8670b04bdc11aaf3674fea7e7";
    assert_eq!(
        chain, CANARY_HEX,
        "maturity_scaled cjcl chain_head canary mismatch — see comment"
    );
}
