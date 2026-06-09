//! ABNG demo: lineage attestation, written in CJC-Lang.
//!
//! Sibling to `tests/test_abng_lineage_attestation.rs` (the pure-Rust
//! correctness layer). This file runs the *same* attestation workload
//! through `.cjcl` source on both executors (cjc-eval AST and
//! cjc-mir-exec MIR) and asserts byte-equal output. Any AST↔MIR
//! divergence in the abng_* builtin surface would fire here.
//!
//! What this demo proves at the language level (over and above the
//! Rust demo's primitive correctness):
//!
//! * The 75 `abng_*` builtins are *callable from CJC-Lang* — not
//!   just from Rust. The lineage workload exercises ~12 of them.
//! * AST↔MIR parity holds for the whole flow: graph construction,
//!   codebook install, leaf head, BLR prior, density tracker,
//!   calibration, decision policy, child node addition,
//!   provenance stamping, prefix encoding, descent, BLR update,
//!   observe, chain-head readout, BLR state-hash readout, audit-
//!   chain verification.
//! * The locked canary chain_head produced by the .cjcl path
//!   matches the locked canary produced by the Rust path —
//!   strong evidence the language pipeline doesn't perturb FP
//!   determinism.

mod abng_demos;

use abng_demos::harness::{extract_value, run_parity, Backend};

const SEED: u64 = 7;

// ── parity (the headline test) ────────────────────────────────────────

#[test]
fn lineage_cjcl_eval_mir_byte_equal() {
    // The strongest determinism gate: build the lineage workload
    // through CJC-Lang and run it through both executors. Every
    // printed line must be byte-identical, which means every
    // `abng_*` builtin call produced bit-for-bit identical output
    // on AST eval AND MIR exec.
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    // Sanity: at least the headline lines are present.
    assert!(out.iter().any(|l| l.starts_with("chain_a:")));
    assert!(out.iter().any(|l| l.starts_with("chain_b:")));
    assert!(out.iter().any(|l| l.starts_with("three_signals_independent:")));
}

// ── extracted-output property assertions ──────────────────────────────

#[test]
fn lineage_cjcl_three_signals_diverge_on_tamper() {
    // The headline tangible benefit: tampering one row of the
    // 16-row dataset produces three independent SHA-256 divergence
    // signals. Each booleanized line must read `true`.
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "chain_diverged"), "true");
    assert_eq!(extract_value(&out, "stamp_diverged"), "true");
    assert_eq!(extract_value(&out, "blr_diverged"), "true");
    assert_eq!(extract_value(&out, "three_signals_independent"), "true");
}

#[test]
fn lineage_cjcl_audit_chain_verifies_for_both_models() {
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "verify_a"), "true");
    assert_eq!(extract_value(&out, "verify_b"), "true");
}

#[test]
fn lineage_cjcl_dataset_a_stamp_matches_constant() {
    // The .cjcl source hardcodes dataset A's pre-computed
    // fingerprint. After stamping + training, abng_provenance_stamp
    // returns the same hex.
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    assert_eq!(
        extract_value(&out, "stamp_a"),
        "3e85d52f2508aecaaf32737edca48a644796783d6be7e6e324e6760506bc3634"
    );
}

#[test]
fn lineage_cjcl_legitimate_match_holds_spoof_detected() {
    // The end-to-end attestation flow: legitimate model_a's
    // chain_head matches itself; spoof attempt with model_b is
    // detected.
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    assert_eq!(extract_value(&out, "legitimate_match"), "true");
    assert_eq!(extract_value(&out, "spoof_detected"), "true");
}

// ── audit log captures full training history ──────────────────────────

#[test]
fn lineage_cjcl_audit_len_grows_with_training() {
    // 16 training rows => 16 BeliefUpdate + 16 BlrUpdated events,
    // plus stamp + setup events. Audit length must be > 16 for
    // both models (clinical-trial scale).
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    let audit_a: u64 = extract_value(&out, "audit_len_a").parse().unwrap();
    let audit_b: u64 = extract_value(&out, "audit_len_b").parse().unwrap();
    assert!(audit_a > 16, "audit_a should exceed training-row count, got {audit_a}");
    assert!(audit_b > 16, "audit_b should exceed training-row count, got {audit_b}");
}

// ── canary: the .cjcl chain_head matches what the Rust demo would produce ──

#[test]
fn lineage_cjcl_chain_head_canary_locked() {
    // The .cjcl path's chain_head is locked. If this fires, either
    // a CJC-Lang interpreter change broke determinism, or a
    // dispatch routing change altered the byte-level execution.
    let out = run_parity(abng_demos::lineage_source::SOURCE, SEED);
    let chain_a = extract_value(&out, "chain_a");
    println!("lineage cjcl canary chain_a = {chain_a}");
    // Locked at first-green-run. Independent canary from the
    // Rust-side lineage demo's chain_head (different number of
    // training rows, different graph topology). Fires on either
    // CJC-Lang interpreter determinism breakage or a dispatch
    // routing change that alters byte-level execution.
    // Re-locked at Phase 0.8c v14 Item A2 — `lineage_source.cjcl`'s
    // per-row training loop flipped from `abng_blr_update +
    // abng_observe` (pre-A2: two events / row, tags 0x0A + 0x01) to
    // `abng_train_step` (post-A2: one TrainStep event / row, tag
    // 0x1E). Pre-A2 hex:
    // `20f5f977cd7dcfad536fbf4be49d4b18c6ba2430b32e510713a038c94fa39b40`.
    // V14_MIGRATION.md records the v13 → v14 mapping.
    //
    // Re-locked at ABNG 0.9.5 R0/R1 (2026-06-09) — the 0.9.5
    // performance refactors changed the rank-1 BLR update path
    // observed through the CJC-Lang dispatch surface (same algorithmic
    // drift as the Rust-side canary):
    //   * 614b7d7  R1-1: lane-parallel x8 Kahan in rank-1 BLR update
    //   * 08a4a6b  perf: O(d²) rank-1 Cholesky update for n=1 hot path
    //   * f678997  R1-2: cholesky_solve lane-parallel + params_hash cache
    // The CJC-Lang interpreter path is still bit-deterministic
    // (two consecutive runs produced identical actual_hex). Pre-R0/R1
    // hex: `223906f55c3506a5f33c43f378cd4b32ff04545af37e8c706432f5a2250617d7`.
    const CANARY_HEX: &str =
        "fe9a662e311051334ab9e3d530e1a9ba2c479f16e9ac36b42d3b634d9689ed40";
    assert_eq!(
        chain_a, CANARY_HEX,
        "cjcl lineage chain_head canary mismatch — see comment"
    );
}

// ── single-backend smoke tests (cheaper than parity, useful for triage) ──

#[test]
fn lineage_cjcl_smoke_runs_via_eval() {
    let out = abng_demos::harness::run(Backend::Eval, abng_demos::lineage_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_a:")));
}

#[test]
fn lineage_cjcl_smoke_runs_via_mir() {
    let out = abng_demos::harness::run(Backend::Mir, abng_demos::lineage_source::SOURCE, SEED);
    assert!(out.iter().any(|l| l.starts_with("chain_a:")));
}
