//! cjc-tempest integration / proptest / fuzz tests.
//!
//! **Status:** SCAFFOLDING. Only smoke tests live here today; sampler
//! integration tests (Metropolis, HMC, NUTS) land across subsequent
//! implementation sessions per the handoff at
//! `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md` §6.2.
//!
//! Required minimums before v0.1 ships (handoff §6.1):
//!
//! | Bucket            | Count |
//! | ----------------- | ----- |
//! | Unit              | ≥ 25  |
//! | Integration       | ≥ 10  |
//! | Proptest          | ≥ 5   |
//! | Bolero fuzz       | ≥ 3   |
//!
//! **Headline determinism test (cross-platform parity)** lives at
//! `tests/tempest/tempest_determinism.rs` once implementation lands —
//! same model + same seed → same `content_hash` on Linux + macOS + Windows.
//! If this test fails on any platform, the release does NOT ship.

use cjc_tempest::{AcceptResult, ConvergenceDiagnostics, FingerprintId, PosteriorSamples, TempestError};

#[test]
fn scaffold_reaches_crate() {
    // Foundational re-exports resolve from the integration-test boundary.
    let r = AcceptResult::Accept { new_log_posterior: -1.5 };
    assert!(matches!(r, AcceptResult::Accept { .. }));

    let id = FingerprintId(0xCAFE_BABE);
    assert_eq!(format!("{}", id), "00000000cafebabe");
}

#[test]
fn tempest_error_display_is_stable() {
    let err = TempestError::InvalidInitialState { detail: "NaN in dim 0".to_string() };
    assert_eq!(err.to_string(), "invalid initial state: NaN in dim 0");
}

#[test]
fn convergence_diagnostics_default_is_zero() {
    let d = ConvergenceDiagnostics::default();
    assert_eq!(d.divergences, 0);
    assert_eq!(d.n_max_treedepth, 0);
    assert!(d.r_hat.is_empty());
}

#[test]
fn posterior_samples_struct_is_constructible() {
    let s = PosteriorSamples {
        chains: vec![vec![vec![0.0, 1.0]; 100]; 4],
        n_chains: 4,
        n_samples_per_chain: 100,
        n_dim: 2,
        diagnostics: ConvergenceDiagnostics::default(),
        content_hash: FingerprintId(0),
    };
    assert_eq!(s.chains.len(), 4);
    assert_eq!(s.chains[0].len(), 100);
    assert_eq!(s.chains[0][0].len(), 2);
}
