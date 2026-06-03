//! cjc-causal integration / proptest / fuzz tests.
//!
//! Submodule layout (handoff §6.2):
//! - `common` — shared DataFrame + LockeReport fixtures
//! - `propensity_score_tests` — end-to-end estimator tests
//! - `causal_proptest` — proptest properties (256 cases each by default)
//! - `causal_fuzz` — bolero structural fuzz targets
//!
//! Wired into the workspace's `[[test]]` table in the root `Cargo.toml`
//! so `cargo test --test causal` runs everything here.
//!
//! Required minimums (handoff §6.1): unit ≥ 25, integration ≥ 12,
//! proptest ≥ 5, bolero fuzz ≥ 3. As of this session:
//! - Unit: **40** in cjc-causal lib (`cargo test -p cjc-causal --lib`)
//! - Integration: **17** in this file's `propensity_score_tests` + 3 scaffold smoke tests below
//! - Proptest: **5** in `causal_proptest`
//! - Bolero: **3** in `causal_fuzz`

mod common;
mod propensity_score_tests;
mod iv_regression_tests;
mod dml_tests;
mod causal_proptest;
mod iv_proptest;
mod dml_proptest;
mod causal_fuzz;
mod iv_fuzz;
mod dml_fuzz;

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
    // other source of run-to-run variance.
    let err = CausalError::UnknownColumn { name: "treatment".to_string() };
    assert_eq!(err.to_string(), "unknown column: treatment");
}

#[test]
fn effect_estimate_is_constructible() {
    // Smoke test: the v0.1 public output type assembles from its fields.
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
        iv_first_stage_f: None,
        identifier: FingerprintId(0),
    };
    assert!(estimate.ci_lower <= estimate.point);
    assert!(estimate.point <= estimate.ci_upper);
}
