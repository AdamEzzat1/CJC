//! Phase 0.6 Item 6 — ABNG **scaled compact_log** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn compact_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval,
        abng_demos::compact_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("emitted:")));
}

#[test]
fn compact_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir,
        abng_demos::compact_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("emitted:")));
}

#[test]
fn compact_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::compact_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn compact_scaled_emits_one_per_touched_node() {
    // Five distinct touched nodes: the root (via ChildrenPromoted
    // events emitted while building out the codebook leaf set) +
    // 4 leaves (each with NodeAdded + observe events).
    let out = run_parity(abng_demos::compact_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "exactly_five_emitted"), "true");
}

#[test]
fn compact_scaled_chain_verifies_at_scale() {
    let out = run_parity(abng_demos::compact_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_post"), "true");
}

#[test]
fn compact_scaled_audit_log_grew_correctly() {
    let out = run_parity(abng_demos::compact_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "audit_grew_correctly"), "true");
}

#[test]
fn compact_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::compact_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("compact_scaled cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "00cba297e7532b877fe6507910e96d7ee0f50c72e672e83947f162e436a1e9f6";
    assert_eq!(
        chain, CANARY_HEX,
        "compact_scaled cjcl chain_head canary mismatch — see comment"
    );
}
