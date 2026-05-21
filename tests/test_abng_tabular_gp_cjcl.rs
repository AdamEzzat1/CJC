//! ABNG demo: tabular GP-like regression, written in CJC-Lang.
//!
//! Sibling to `tests/test_abng_tabular_gp.rs` (the pure-Rust
//! correctness layer). Runs the same 2-D regression workload through
//! `.cjcl` source on both executors. Asserts the GP-like properties:
//! trained MSE beats prior MSE, leverage shrinks with more data,
//! per-leaf evidence is bounded.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 11;

#[test]
fn tabular_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_small:")));
    assert!(out.iter().any(|l| l.starts_with("trained_beats_prior:")));
}

#[test]
fn tabular_cjcl_audit_chains_verify() {
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_small"), "true");
    assert_eq!(extract_value(&out, "verify_big"), "true");
}

#[test]
fn tabular_cjcl_trained_mse_beats_prior() {
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "trained_beats_prior"),
        "true",
        "trained MSE must beat half the prior MSE"
    );
}

#[test]
fn tabular_cjcl_lev_shrinks_with_more_data() {
    // GP-like property: epistemic_leverage at a fixed probe point
    // shrinks as the dataset grows.
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "lev_shrinks_with_data"),
        "true",
        "leverage at fixed probe must shrink from small→big training set"
    );
}

#[test]
fn tabular_cjcl_max_per_leaf_bounded() {
    // Per-leaf BLR is bounded — no single leaf holds ≥90% of the
    // dataset (vs 100% for global GP).
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "max_per_leaf_bounded"),
        "true",
        "no single leaf may hold ≥90% of total points"
    );
}

#[test]
fn tabular_cjcl_total_routed_equals_training_size() {
    // Sanity: every point routes to exactly one leaf, summing to N.
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    let total: u64 = extract_value(&out, "total_routed").parse().unwrap();
    assert_eq!(total, 64, "total routed must equal 64 (n_big)");
}

#[test]
fn tabular_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::tabular_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_big");
    println!("tabular cjcl canary chain_big = {chain}");
    // Locked at first-green-run. Independent of the Rust-side
    // tabular canary. Fires on CJC-Lang interpreter determinism
    // breakage, BLR conjugate update arithmetic change, or
    // dispatch-routing change for the abng_blr_n_seen / observe path.
    // Re-locked at Phase 0.8c v14 Item A2 — `tabular_source.cjcl`'s
    // `train_one` flipped from `abng_blr_update + abng_observe`
    // (pre-A2: two events / row, tags 0x0A + 0x01) to
    // `abng_train_step` (post-A2: one TrainStep event / row, tag
    // 0x1E). Pre-A2 hex:
    // `4ffacae41d76f505335218ee0479c656e059024cb7e8d6c95350bbc2af09be54`.
    // V14_MIGRATION.md records the v13 → v14 mapping.
    const CANARY_HEX: &str =
        "6b3374934095965bca39904a48a5ab557f80347910ed555bae6ea4bd762510fe";
    assert_eq!(
        chain, CANARY_HEX,
        "cjcl tabular chain_head canary mismatch — see comment"
    );
}

#[test]
fn tabular_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::tabular_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_big:")));
}

#[test]
fn tabular_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::tabular_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_big:")));
}
