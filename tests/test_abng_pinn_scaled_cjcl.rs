//! Phase 0.6 Item 6 — ABNG **scaled PINN** demo (CJC-Lang).

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn pinn_scaled_smoke_eval() {
    let out = abng_demos::harness::run(
        Backend::Eval,
        abng_demos::pinn_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("max_err:")));
}

#[test]
fn pinn_scaled_smoke_mir() {
    let out = abng_demos::harness::run(
        Backend::Mir,
        abng_demos::pinn_scaled_source::SOURCE,
        SEED,
    );
    assert!(out.iter().any(|l| l.starts_with("max_err:")));
}

#[test]
fn pinn_scaled_eval_mir_byte_equal() {
    // Strongest determinism gate: AST tree-walk + MIR register-machine
    // must produce the same printed output for a 10^3-sample workload
    // with Tensor.randn-driven Gaussian noise.
    let out = run_parity(abng_demos::pinn_scaled_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn pinn_scaled_recovers_truth_under_noise() {
    // Headline: at n=10^3 with sigma=0.01 noise, the BLR posterior
    // mean recovers the analytical truth to within max_err < 0.05
    // (the noise floor + per-leaf basis-fit error).
    let out = run_parity(abng_demos::pinn_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "recovers_truth"), "true");
}

#[test]
fn pinn_scaled_audit_chain_verifies() {
    let out = run_parity(abng_demos::pinn_scaled_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn pinn_scaled_chain_head_canary_locked() {
    let out = run_parity(abng_demos::pinn_scaled_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("pinn_scaled cjcl canary chain_head = {chain}");
    // Phase 0.8c v14 Item D2b — re-locked after the SIMD-friendly
    // Kahan refactor of `BlrState::update`. Pre-D2b hex was
    // `11191d718c961259d10e108dfd3368d5603d708c8fce34bd0355867f43c1e2f2`.
    // PINN-scaled exercises BLR at d=4 with n≥5 per-leaf batches —
    // hits the new `KahanAccumulatorF64x4::add_lanes` path.
    const CANARY_HEX: &str =
        "2d2ddec36850b487b9b714a584382df306f7730a328edb1b06d11fe7c4d7dcd0";
    assert_eq!(
        chain, CANARY_HEX,
        "pinn_scaled cjcl chain_head canary mismatch — see comment"
    );
}
