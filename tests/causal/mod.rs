//! cjc-causal integration / proptest / fuzz tests.
//!
//! **Status:** SCAFFOLDING. Only a single smoke test lives here today; the
//! per-estimator submodules (`propensity_score_tests`, `iv_regression_tests`,
//! `double_ml_tests`, `causal_proptest`, `causal_fuzz`, `causal_determinism`,
//! `causal_locke_parity`) land in the implementation session per the handoff
//! at `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md` §6.2.
//!
//! Required minimums before v0.1 ships (handoff §6.1):
//!
//! | Bucket            | Count |
//! | ----------------- | ----- |
//! | Unit              | ≥ 25  |
//! | Integration       | ≥ 12  |
//! | Proptest          | ≥ 5   |
//! | Bolero fuzz       | ≥ 3   |
//!
//! Wired into the workspace's `[[test]]` table in the root `Cargo.toml`
//! so `cargo test --test causal` runs everything here.

use cjc_causal::{CausalError, EffectEstimate, FingerprintId, IdentificationAssumption};

#[test]
fn scaffold_reaches_crate() {
    // Foundational re-exports resolve from the integration-test boundary.
    let assumptions = vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::Positivity,
        IdentificationAssumption::NoInterference,
    ];
    assert_eq!(assumptions.len(), 3);

    let id = FingerprintId(0xCAFE_BABE);
    assert_eq!(format!("{}", id), "00000000cafebabe");
}

#[test]
fn causal_error_display_is_stable() {
    // The error Display impl must not depend on HashMap iteration or any
    // other source of run-to-run variance. This is a one-line smoke test
    // for the determinism contract; the implementation session expands it
    // into a full determinism suite at `tests/causal/causal_determinism.rs`.
    let err = CausalError::UnknownColumn { name: "treatment".to_string() };
    assert_eq!(err.to_string(), "unknown column: treatment");
}

#[test]
fn effect_estimate_is_constructible() {
    // Smoke test: the v0.1 public output type assembles from its fields.
    // The implementation session replaces this with real estimator end-to-end
    // tests at `tests/causal/{propensity_score,iv_regression,double_ml}_tests.rs`.
    let estimate = EffectEstimate {
        point: 0.42,
        std_error: 0.05,
        ci_lower: 0.32,
        ci_upper: 0.52,
        confidence_level: 0.95,
        n_treated: 500,
        n_control: 500,
        assumptions_declared: vec![IdentificationAssumption::Unconfoundedness],
        balance_diagnostics: None,
        identifier: FingerprintId(0),
    };
    assert!(estimate.ci_lower <= estimate.point);
    assert!(estimate.point <= estimate.ci_upper);
}
