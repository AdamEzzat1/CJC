//! # cjc-tempest — probabilistic programming with byte-identical MCMC chains
//!
//! **Status:** v0.1 SCAFFOLDING. The foundational types are defined; the
//! sampler implementations land across subsequent sessions. See the handoff
//! at `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md`
//! and `ADR-0045 cjc-tempest v0.1`.
//!
//! ## Headline value claim
//!
//! Posterior chains that are **byte-identically reproducible** across runs,
//! seeds, and platforms. Two runs over the same model + same data + same
//! seed produce the same [`PosteriorSamples`] including its content-addressed
//! [`content_hash`]. **No existing PPL provides this** — PyMC, Stan, Turing.jl,
//! NumPyro all derive their default RNG from system entropy or rely on
//! platform-dependent BLAS implementations.
//!
//! The publishability of Bayesian model selection, posterior predictive
//! checks, and sensitivity analysis all depend on reproducible posteriors.
//! cjc-tempest's contract is exactly that.
//!
//! ## Determinism contract (the whole reason this crate exists)
//!
//! 1. Every RNG draw routes through [`cjc_repro::Rng`] (SplitMix64) with
//!    the seed threaded explicitly from the caller. Never `rand::thread_rng()`.
//! 2. All floating-point reductions go through `cjc_repro::KahanAccumulatorF64`.
//! 3. All map iteration uses `BTreeMap` / `BTreeSet`.
//! 4. No FMA. `RUSTFLAGS` must not enable `target-feature=+fma`.
//! 5. R-hat and ESS computation use the Vehtari et al. 2021 split-rank-
//!    normalised formulation — no `HashMap`, no platform-dependent sort
//!    tie-breaking.
//! 6. Leapfrog integrator (HMC, NUTS) accumulates energy errors with Kahan.
//!
//! ## Seed-flow diagram (to be expanded in implementation sessions)
//!
//! Every site that consumes randomness must be enumerated for the
//! Determinism Auditor:
//!
//! 1. Initial state per chain (one [`Rng`](cjc_repro::Rng) seeded per chain
//!    from the caller's base seed via SplitMix64 stretch).
//! 2. Momentum draw per leapfrog kick (HMC, NUTS).
//! 3. Acceptance/rejection coin flip per proposal.
//! 4. Direction choice per NUTS tree expansion (left vs right).
//! 5. Slice sample for NUTS (Hoffman & Gelman 2014 §3.1).
//! 6. Adaptation-phase RNG (warmup) — SEPARATE seed from sampling-phase RNG.
//!
//! ## What v0.1 will ship (across implementation sessions)
//!
//! - **Metropolis-Hastings** (Session 1) — symmetric proposal kernel with
//!   adaptive covariance (Welford during warmup). Serves as the
//!   determinism-warmup deliverable before tackling HMC.
//! - **Hamiltonian Monte Carlo** (Session 2-3) — leapfrog integrator,
//!   identity mass matrix in v0.1 (mass adaptation is v0.2), reverse-mode
//!   AD via [`cjc_ad::GradGraph`] for the log-posterior gradient.
//! - **No-U-Turn Sampler (NUTS)** (Session 3-4) — Hoffman & Gelman 2014
//!   Algorithm 6 with dual averaging for step-size adaptation.
//! - **Convergence diagnostics** (Session 4) — Vehtari et al. 2021 R-hat
//!   + bulk-ESS + tail-ESS via Geyer's initial monotone sequence estimator.
//!
//! ## What v0.1 will NOT do
//!
//! Variational inference (ADVI / SVI), reversible-jump MCMC, sequential
//! Monte Carlo, `model { ... }` DSL block in `.cjcl`, Bayes-factor
//! marginal-likelihood computation. See the handoff §4.7 for the full
//! deferral list and rationale.
//!
//! ## Composing with cjc-locke
//!
//! Caller supplies a [`cjc_locke::LockeReport`] to the sampler's `run()`
//! method. The sampler inspects the report for refusal-grade findings on
//! the input data (E9001 missingness, etc.) and refuses to start if any
//! match. Tempest-side findings emit in the `E9300..=E9399` range per the
//! handoff §5.3 reservation.

pub mod error;
pub mod posterior;
pub mod sampler;

pub use error::TempestError;
pub use posterior::{ConvergenceDiagnostics, PosteriorSamples};
pub use sampler::{AcceptResult, Sampler};

/// Re-export of `cjc_locke::id::FingerprintId` so callers don't need a
/// direct dep on cjc-locke just to spell content-addressed IDs.
pub use cjc_locke::id::FingerprintId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_compiles() {
        let a = AcceptResult::Reject;
        assert!(matches!(a, AcceptResult::Reject));
    }

    #[test]
    fn fingerprint_id_reexport_resolves() {
        let id = FingerprintId(0xDEAD_BEEF);
        assert_eq!(format!("{}", id), "00000000deadbeef");
    }
}
