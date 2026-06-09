//! ABNG demo: PINN per-region uncertainty, written in CJC-Lang.
//!
//! Sibling to `tests/test_abng_pinn_uncertainty.rs` (the pure-Rust
//! correctness layer). Runs the same heat-equation workload through
//! `.cjcl` source on both executors and asserts the headline benefit:
//! min(edge epistemic_leverage) > max(interior epistemic_leverage)
//! after asymmetric training.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

#[test]
fn pinn_cjcl_eval_mir_byte_equal() {
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
    assert!(out.iter().any(|l| l.starts_with("edge_strictly_higher:")));
}

#[test]
fn pinn_cjcl_audit_chain_verifies() {
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_chain"), "true");
}

#[test]
fn pinn_cjcl_edge_lev_strictly_higher_than_interior() {
    // Headline tangible benefit: min(edge) > max(interior) lev.
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "edge_strictly_higher"),
        "true",
        "min(edge lev) must strictly exceed max(interior lev) post-asymmetric-training"
    );
}

#[test]
fn pinn_cjcl_interior_fits_analytical_solution() {
    // After 32 interior + 8 edge samples, the BLR mean at x=0.5
    // matches the analytical solution within 0.05.
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "interior_fit_ok"), "true");
}

#[test]
fn pinn_cjcl_lev_values_finite_and_in_range() {
    // All five reported leverage values must be finite, non-negative,
    // and at most 1.0 (BLR leverage is in [0, 1]).
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    for key in [
        "lev_int_30",
        "lev_int_50",
        "lev_int_70",
        "lev_edge_10",
        "lev_edge_90",
    ] {
        let v: f64 = extract_value(&out, key).parse().unwrap();
        assert!(v.is_finite() && (0.0..=1.0).contains(&v),
            "{key} = {v} out of [0, 1]");
    }
}

#[test]
fn pinn_cjcl_audit_len_grows_with_training() {
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    let n: u64 = extract_value(&out, "audit_len").parse().unwrap();
    // 32 interior + 8 edge = 40 BLR updates + 40 observes + setup
    // events + provenance stamp; conservatively > 40.
    assert!(n > 40, "audit_len should exceed 40, got {n}");
}

#[test]
fn pinn_cjcl_chain_head_canary_locked() {
    let out = run_parity(abng_demos::pinn_source::SOURCE, SEED);
    let chain = extract_value(&out, "chain_head");
    println!("pinn cjcl canary chain_head = {chain}");
    // Locked at first-green-run. Independent of the Rust-side PINN
    // canary (different node count, different stamp). Fires on
    // CJC-Lang interpreter determinism breakage or BLR/observe
    // arithmetic change.
    // Re-locked at Phase 0.8c v14 Item A2 — `pinn_source.cjcl`'s
    // `train_one` flipped from `abng_blr_update + abng_observe`
    // (pre-A2: two events / row, tags 0x0A + 0x01) to
    // `abng_train_step` (post-A2: one TrainStep event / row, tag
    // 0x1E). Pre-A2 hex:
    // `e5d6c41daeec4b34a78ddab5086f9903d3dd56b8fd995400dd190ba8d684a64a`.
    // V14_MIGRATION.md records the v13 → v14 mapping.
    //
    // Re-locked at ABNG 0.9.5 R0/R1 (2026-06-09) — same algorithmic
    // drift as the Rust-side PINN canary; the 0.9.5 BLR refactors
    // shifted intermediate audit-chain bytes when observed through
    // the CJC-Lang dispatch path. Commits 614b7d7 / 08a4a6b / f678997.
    // Pre-R0/R1 hex:
    // `be14b783c16b34f6bdf944a23c4ccdfdd2039eef3fee2ad8f81242f3c766388d`.
    const CANARY_HEX: &str =
        "639fd29ccf4390844542363ba3e1b81e049c8ef1baf924cb262edbda2bba8603";
    assert_eq!(
        chain, CANARY_HEX,
        "cjcl PINN chain_head canary mismatch — see comment"
    );
}

#[test]
fn pinn_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::pinn_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}

#[test]
fn pinn_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::pinn_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_head:")));
}
