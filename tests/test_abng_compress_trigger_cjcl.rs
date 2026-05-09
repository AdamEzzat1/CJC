//! Phase 0.6 Item 5 — ABNG **Compress** trigger demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn compress_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::compress_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("compress_count:")));
    assert!(out.iter().any(|l| l.starts_with("only_compress_fired:")));
}

#[test]
fn compress_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::compress_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("compress_count:")));
}

#[test]
fn compress_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::compress_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("compress_count:")));
}

#[test]
fn compress_cjcl_compress_count_is_positive() {
    let out = run_parity(abng_demos::compress_source::SOURCE, SEED);
    let compress: u64 = extract_value(&out, "compress_count").parse().unwrap();
    assert!(
        compress >= 1,
        "expected Compress to fire at least once, got {compress}"
    );
}

#[test]
fn compress_cjcl_no_other_trigger_fired() {
    let out = run_parity(abng_demos::compress_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "only_compress_fired"), "true");
}

#[test]
fn compress_cjcl_audit_chain_verifies_post_compress() {
    let out = run_parity(abng_demos::compress_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn compress_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::compress_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("compress cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "e4d9eeeae3c9e8f47aa727ffedb318badbf6781123e5c354f8aa176bfb0a4aef";
    assert_eq!(
        chain, CANARY_HEX,
        "compress cjcl chain_head canary mismatch — see comment"
    );
}
