//! ABNG demo: maturity inspection (training-state observability),
//! written in CJC-Lang.
//!
//! Capability demonstrated: `abng_node_maturity` returns a 4-D
//! tensor [signature_stable, ece_stable, uncertainty_stable,
//! drift_stable] per node. The flags evolve from 0 → 1 as
//! training stabilizes — the introspection layer for monitoring
//! production deployments.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 42;

#[test]
fn maturity_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("sum_at_t0:")));
    assert!(out.iter().any(|l| l.starts_with("maturity_evolved:")));
}

#[test]
fn maturity_cjcl_audit_chain_verifies() {
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn maturity_cjcl_starts_at_zero() {
    // At t=0, all flags must be 0.
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    let sum0: f64 = extract_value(&out, "sum_at_t0").parse().unwrap();
    assert_eq!(sum0, 0.0, "fresh graph maturity must be all zeros");
}

#[test]
fn maturity_cjcl_evolves_under_training() {
    // Headline: after observation + decide_step, at least one
    // maturity flag flipped from 0 to 1.
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "maturity_evolved"),
        "true",
        "maturity flags must evolve under stable training"
    );
}

#[test]
fn maturity_cjcl_calibration_uncertainty_flags_are_binary() {
    // Positions [1] (calibration_stable) and [2]
    // (uncertainty_stable) are strict 0/1 flags.
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "calib_binary"), "true");
    assert_eq!(extract_value(&out, "unc_binary"), "true");
}

#[test]
fn maturity_cjcl_samples_seen_nonneg_trust_in_range() {
    // Position [0] is a sample count (non-negative); position [3]
    // is a continuous trust level in [0, 1].
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "samples_seen_nonneg"), "true");
    assert_eq!(extract_value(&out, "trust_in_range"), "true");
}

#[test]
fn maturity_cjcl_post_train_sum_finite() {
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    let sum1: f64 = extract_value(&out, "sum_at_t1").parse().unwrap();
    assert!(sum1.is_finite() && sum1 >= 0.0,
        "maturity sum must be finite & non-negative, got {sum1}");
}

#[test]
fn maturity_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::maturity_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("maturity cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "c7b92726459c670bd960cd27c1f307e733744bcf420f5457f9a34a1b54a1928b";
    assert_eq!(
        chain, CANARY_HEX,
        "maturity cjcl chain_head canary mismatch"
    );
}

#[test]
fn maturity_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::maturity_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("sum_at_t0:")));
}

#[test]
fn maturity_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::maturity_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("sum_at_t0:")));
}
