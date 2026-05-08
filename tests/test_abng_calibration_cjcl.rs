//! ABNG demo: calibration / ECE, written in CJC-Lang.
//!
//! Capability demonstrated: `abng_calibration_observe` records
//! (predicted_prob, observed_outcome) into one of 15 reliability
//! bins; `abng_calibration_ece` computes the expected calibration
//! error. ECE distinguishes well-calibrated from miscalibrated
//! models — exactly the diagnostic an MLP doesn't give natively.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn calibration_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("ece_a:")));
    assert!(out.iter().any(|l| l.starts_with("well_below_mis:")));
}

#[test]
fn calibration_cjcl_audit_chains_verify() {
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_a"), "true");
    assert_eq!(extract_value(&out, "verify_b"), "true");
}

#[test]
fn calibration_cjcl_ece_in_valid_range() {
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "ece_in_range"), "true");
}

#[test]
fn calibration_cjcl_well_calibrated_ece_below_miscalibrated() {
    // Headline tangible benefit: ECE distinguishes the two models.
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "well_below_mis"),
        "true",
        "well-calibrated ECE must be strictly less than miscalibrated ECE"
    );
}

#[test]
fn calibration_cjcl_well_calibrated_low_ece() {
    // Well-calibrated graph achieves ECE < 0.05.
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "well_low"), "true");
}

#[test]
fn calibration_cjcl_miscalibrated_high_ece() {
    // Miscalibrated graph achieves ECE > 0.5.
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "mis_high"), "true");
}

#[test]
fn calibration_cjcl_n_seen_matches_observation_count() {
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    let n_a: u64 = extract_value(&out, "n_a").parse().unwrap();
    let n_b: u64 = extract_value(&out, "n_b").parse().unwrap();
    // Each run is 3 bins × 10 observations = 30 calibration_observe calls.
    assert_eq!(n_a, 30);
    assert_eq!(n_b, 30);
    assert_eq!(extract_value(&out, "n_match"), "true");
}

#[test]
fn calibration_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::calibration_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_a");
    println!("calibration cjcl canary chain_a = {chain}");
    const CANARY_HEX: &str =
        "4c625f088c72da46f756972e89adc934183f27e3ba2fa2bbb2509ba091a15a25";
    assert_eq!(
        chain, CANARY_HEX,
        "calibration cjcl chain_head canary mismatch"
    );
}

#[test]
fn calibration_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::calibration_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("ece_a:")));
}

#[test]
fn calibration_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::calibration_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("ece_a:")));
}
