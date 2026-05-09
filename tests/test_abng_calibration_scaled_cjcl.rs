//! Phase 0.6 Item 6 — ABNG **scaled calibration** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn calibration_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval, abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("ece_a:")));
}

#[test]
fn calibration_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir, abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("ece_a:")));
}

#[test]
fn calibration_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn calibration_scaled_well_below_miscal_at_n_1000() {
    let out = run_parity(abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "well_below_miscal"), "true");
}

#[test]
fn calibration_scaled_ece_magnitudes_sane() {
    // At n=1000, well-calibrated ECE should be < 0.15 and
    // miscalibrated ECE > 0.4. These bounds prove the estimator is
    // not just trivially distinguishing — it's quantifying the gap.
    let out = run_parity(abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "ece_a_low"), "true");
    assert_eq!(extract_value(&out, "ece_b_high"), "true");
}

#[test]
fn calibration_scaled_both_full_n_1000() {
    let out = run_parity(abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "both_full"), "true");
}

#[test]
fn calibration_scaled_audit_chain_verifies() {
    let out = run_parity(abng_demos::calibration_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn calibration_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::calibration_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("calibration_scaled cjcl canary chain_head = {chain}");
    const CANARY_HEX: &str =
        "6d72d9bc43102addb124bd904cce92fa7a87b736bcccd62d80fc252786bf6649";
    assert_eq!(
        chain, CANARY_HEX,
        "calibration_scaled cjcl chain_head canary mismatch — see comment"
    );
}
