//! Phase 0.6 Item 6 — ABNG **scaled tabular** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn tabular_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval,
        abng_demos::tabular_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("err_low_noise_leaf:")));
}

#[test]
fn tabular_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir,
        abng_demos::tabular_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("err_low_noise_leaf:")));
}

#[test]
fn tabular_scaled_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::tabular_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn tabular_scaled_recovers_truth_under_heteroskedastic_noise() {
    let out = run_parity(abng_demos::tabular_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "recovers_truth"), "true");
}

#[test]
fn tabular_scaled_leverage_responds_to_noise_level() {
    // Headline: per-leaf epistemic leverage at the high-noise leaf
    // (x1 ≈ 0.9, σ ≈ 0.046) exceeds leverage at the low-noise leaf
    // (x1 ≈ 0.1, σ ≈ 0.014). This proves Bayesian uncertainty
    // calibration: the model knows it's less certain in regions
    // where the data is noisier.
    let out = run_parity(abng_demos::tabular_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "leverage_responds_to_noise"), "true");
}

#[test]
fn tabular_scaled_audit_chain_verifies() {
    let out = run_parity(abng_demos::tabular_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn tabular_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::tabular_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("tabular_scaled cjcl canary chain_head = {chain}");
    // Phase 0.8c v14 Item D2b — re-locked after the SIMD-friendly
    // Kahan refactor of `BlrState::update`. The pre-D2b hex was
    // `fe172059a2c564f0d9b3270e2482468faf478331a703f37bf428e72a9d4fc32f`.
    // The shift comes from this demo's batched `abng_blr_update`
    // (n ≥ 5 per leaf), which now distributes row-products across
    // 4 lanes of `KahanAccumulatorF64x4` instead of summing them
    // strictly left-to-right. Same Kahan stability, different
    // (still deterministic) f64 bits.
    const CANARY_HEX: &str =
        "e1adbf41024db1fd42a5bfc5e9aae9d2288501dc22bca6a5bf8de27815bec003";
    assert_eq!(
        chain, CANARY_HEX,
        "tabular_scaled cjcl chain_head canary mismatch — see comment"
    );
}
