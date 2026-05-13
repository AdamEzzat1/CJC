//! Phase 0.6 Item 6 — ABNG **scaled lineage** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn lineage_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval,
        abng_demos::lineage_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("err_low:")));
}

#[test]
fn lineage_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir,
        abng_demos::lineage_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("err_low:")));
}

#[test]
fn lineage_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::lineage_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn lineage_scaled_recovers_truth_at_n_1000() {
    let out = run_parity(abng_demos::lineage_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "recovers_truth"), "true");
}

#[test]
fn lineage_scaled_audit_chain_verifies() {
    let out = run_parity(abng_demos::lineage_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn lineage_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::lineage_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("lineage_scaled cjcl canary chain_head = {chain}");
    // Phase 0.8c v14 Item D2b — re-locked after the SIMD-friendly
    // Kahan refactor of `BlrState::update`. Pre-D2b hex was
    // `39c0b1cb3b92815fe87013cb3e25bcd0273ce3917976c833d6daa99eae3c58cf`.
    // Lineage-scaled batches BLR updates by leaf during the
    // attestation-stamp pass; per-leaf n is typically ≥ 5.
    const CANARY_HEX: &str =
        "2c16d6f1026ba0f5e7b66e8e54f5a69eedda3163e04f150cc1cfc8aea07baa03";
    assert_eq!(
        chain, CANARY_HEX,
        "lineage_scaled cjcl chain_head canary mismatch — see comment"
    );
}
