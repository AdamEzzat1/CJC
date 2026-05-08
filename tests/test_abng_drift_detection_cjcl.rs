//! ABNG demo: distribution-drift detection, written in CJC-Lang.
//!
//! Capability demonstrated: `abng_freeze_drift_baseline` snapshots
//! the current density tracker; `abng_drift_score` measures
//! deviation from that baseline. Detects when the input
//! distribution has shifted — the diagnostic production ML
//! systems need to know when retraining is due.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn drift_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("drift_low:")));
    assert!(out.iter().any(|l| l.starts_with("drift_grew:")));
}

#[test]
fn drift_cjcl_audit_chain_verifies() {
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn drift_cjcl_score_nonneg() {
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "drift_nonneg"), "true");
}

#[test]
fn drift_cjcl_score_grows_under_distribution_shift() {
    // Headline tangible benefit: the post-shift drift_score
    // strictly exceeds the same-distribution drift_score.
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "drift_grew"),
        "true",
        "drift_score must grow when input distribution shifts"
    );
}

#[test]
fn drift_cjcl_shift_is_meaningful() {
    // Specific claim: distribution-shift drift is at least 2x
    // same-distribution drift. Catches a regression where drift
    // detection becomes noisy / numerically dominated by FP error.
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "meaningful_shift"), "true");
}

#[test]
fn drift_cjcl_score_finite_and_bounded() {
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    let low: f64 = extract_value(&out, "drift_low").parse().unwrap();
    let high: f64 = extract_value(&out, "drift_high").parse().unwrap();
    assert!(low.is_finite() && high.is_finite(),
        "drift scores must be finite (no NaN/Inf): low={low}, high={high}");
    assert!(low >= 0.0 && high >= 0.0,
        "drift scores must be non-negative: low={low}, high={high}");
}

#[test]
fn drift_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::drift_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("drift cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "a3a41c5b282a349d1af38637cc2207f66a9e59a20f839288c4f8e5c817ea8121";
    assert_eq!(
        chain, CANARY_HEX,
        "drift cjcl chain_head canary mismatch"
    );
}

#[test]
fn drift_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::drift_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("drift_low:")));
}

#[test]
fn drift_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::drift_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("drift_low:")));
}
