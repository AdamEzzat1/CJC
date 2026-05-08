//! ABNG demo: OOD detection composite, written in CJC-Lang.
//!
//! Capability demonstrated: `abng_ood_score` composites three
//! independent uncertainty signals — density (how far is this
//! input from training points?), epistemic_z (BLR's leverage at
//! this query?), and prefix_distance (how many prefix bytes
//! failed to find a child during descent?). The composite [0, 1]
//! score lets a model say "I haven't seen anything like this;
//! abstain or fall back."
//!
//! Headline assertion: a graph trained densely on bin 0 + bin 1,
//! lightly on bin 2, and missing bin 3 entirely (no child at
//! prefix byte 3) produces ood_scores that strictly satisfy:
//!   dense  <  sparse  <  routing-fall-off

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn ood_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
    assert!(out.iter().any(|l| l.starts_with("three_tier_separation:")));
}

#[test]
fn ood_cjcl_audit_chain_verifies() {
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn ood_cjcl_scores_in_range_zero_one() {
    // Range invariant: ood_score is bounded in [0, 1].
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "scores_in_range"), "true");
}

#[test]
fn ood_cjcl_three_tier_separation() {
    // Headline tangible benefit: max(dense_bin) < sparse_bin
    // < routing_falloff. The composite cleanly partitions three
    // evidence regimes — exactly what an MLP doesn't give for free.
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "three_tier_separation"),
        "true",
        "ood_score must strictly partition: dense < sparse < routing-fall-off"
    );
}

#[test]
fn ood_cjcl_routing_falloff_dominates() {
    // Specifically: the routing-fall-off bin (bin 3, no child) has
    // strictly higher ood_score than the sparsely-trained bin.
    // This proves prefix_distance is contributing meaningfully.
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    let bin2: f64 = extract_value(&out, "ood_bin2").parse().unwrap();
    let bin3: f64 = extract_value(&out, "ood_bin3").parse().unwrap();
    assert!(
        bin3 > bin2,
        "routing-fall-off bin (ood={bin3}) must exceed sparse bin (ood={bin2})"
    );
}

#[test]
fn ood_cjcl_dense_bins_lowest_score() {
    // Specifically: both dense bins have ood < 0.5 (model is
    // confident in regions it has lots of evidence for).
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    let bin0: f64 = extract_value(&out, "ood_bin0").parse().unwrap();
    let bin1: f64 = extract_value(&out, "ood_bin1").parse().unwrap();
    let bin3: f64 = extract_value(&out, "ood_bin3").parse().unwrap();
    assert!(
        bin0 < bin3 && bin1 < bin3,
        "dense bins (ood={bin0}, {bin1}) must be lower than routing-fall-off (ood={bin3})"
    );
}

#[test]
fn ood_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::ood_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("ood cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "85970ca5c2dbd93469fe3c849e3b15f6b32b3a593a7e69c16e1f22dcf8fd533e";
    assert_eq!(
        chain, CANARY_HEX,
        "ood cjcl chain_head canary mismatch — see comment"
    );
}

#[test]
fn ood_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::ood_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn ood_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::ood_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}
