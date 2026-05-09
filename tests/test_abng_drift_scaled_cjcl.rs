//! Phase 0.6 Item 6 — ABNG **scaled drift** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn drift_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval, abng_demos::drift_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("drift_abrupt:")));
}

#[test]
fn drift_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir, abng_demos::drift_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("drift_abrupt:")));
}

#[test]
fn drift_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::drift_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn drift_scaled_monotonic_signal() {
    // Headline: drift_score escalates monotonically through the
    // engineered schedule (stable < gradual < abrupt).
    let out = run_parity(abng_demos::drift_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "monotonic_drift_signal"), "true");
}

#[test]
fn drift_scaled_abrupt_dominates_gradual() {
    // The abrupt phase produces drift_score > 2× the gradual phase.
    // This rules out a degenerate case where stable, gradual, and
    // abrupt all evaluate to similar small numbers.
    let out = run_parity(abng_demos::drift_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "abrupt_dominates_gradual"), "true");
}

#[test]
fn drift_scaled_scores_nonneg() {
    let out = run_parity(abng_demos::drift_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "drift_nonneg"), "true");
}

#[test]
fn drift_scaled_audit_chain_verifies() {
    let out = run_parity(abng_demos::drift_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn drift_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::drift_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("drift_scaled cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "68ac1fb7e3d53c64e0819e569f980cadd97bb96b3227a0bea7899c5e594befb4";
    assert_eq!(
        chain, CANARY_HEX,
        "drift_scaled cjcl chain_head canary mismatch — see comment"
    );
}
