//! # cjc-causal — formal causal inference for CJC-Lang
//!
//! **Status:** v0.1 SCAFFOLDING. The foundational types are defined; the
//! estimator implementations land in a dedicated implementation session.
//! See the handoff at
//! `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md`
//! and `ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML`.
//!
//! ## Headline value claim
//!
//! Treatment-effect estimates that are **byte-identically reproducible**
//! across runs, seeds (when explicitly threaded), and platforms. Two runs
//! over the same input data + same declared assumptions + same caller-supplied
//! seed produce the same [`EffectEstimate`] including its content-addressed
//! [`identifier`](EffectEstimate::identifier).
//!
//! Python causal-inference incumbents (DoWhy, EconML, CausalML) do not
//! provide structural determinism. cjc-causal's contract is exactly that.
//!
//! ## Determinism contract
//!
//! 1. All floating-point reductions go through `cjc_repro::KahanAccumulatorF64`.
//!    Raw `Vec<f64>::iter().sum()` is **banned** anywhere in the crate.
//! 2. All map iteration uses `BTreeMap` / `BTreeSet`. No `HashMap`.
//! 3. All randomness routes through `cjc_repro::Rng` (SplitMix64) with the
//!    seed threaded explicitly from the caller.
//! 4. Nearest-neighbor matching ties resolve by ascending row index, never
//!    by hash order.
//! 5. No FMA. `RUSTFLAGS` must not enable `target-feature=+fma`.
//!
//! See [[ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML]] for
//! the full determinism note + seed-flow diagram.
//!
//! ## What v0.1 will ship (after implementation session)
//!
//! - [`PropensityScoreMatcher`] — IRLS logistic-regression propensity +
//!   nearest-neighbor matching with Austin 2011 caliper default.
//! - [`IVRegression`] — 2SLS with Stock-Yogo weak-instrument F-stat
//!   (surfaced as Locke finding `E9100`) and HC1 sandwich standard errors.
//! - [`DoubleMLEstimator`] — Chernozhukov et al. 2018 orthogonal scoring
//!   with K-fold cross-fitting and `cjc_ad::GradGraph`-trained MLP
//!   nuisance functions.
//!
//! These types are intentionally **not stubbed in this scaffolding**. The
//! implementation session designs the per-estimator API as it writes the
//! math — stubbing them now would commit to an interface before the math
//! is in hand. (Per [`CLAUDE`]'s "no half-finished implementations" rule.)
//!
//! [`CLAUDE`]: ../CLAUDE.md
//!
//! ## What v0.1 will NOT do
//!
//! Out of scope for v0.1: do-calculus, structural equation modelling,
//! mediation analysis, regression discontinuity, difference-in-differences,
//! synthetic control, causal forests. See the handoff §2.7 for the full
//! deferral list and rationale.
//!
//! ## Composing with cjc-locke
//!
//! Caller passes a [`cjc_locke::LockeReport`] to each estimator. Estimators
//! refuse to run if the report shows known-fatal findings (E9001 ≥ 0.30 on
//! treatment or outcome, E9009 missing on continuous covariates, E9060
//! leakage on covariates), returning [`CausalError::DataQualityRefusal`]
//! with the offending findings attached.
//!
//! ## Quick start (planned, not yet wired)
//!
//! ```ignore
//! use cjc_causal::{PropensityScoreMatcher, IdentificationAssumption};
//! # let df: cjc_data::DataFrame = unimplemented!();
//! # let locke_report: cjc_locke::LockeReport = unimplemented!();
//!
//! let matcher = PropensityScoreMatcher::new()
//!     .with_caliper_sd(0.2)
//!     .with_seed(42);
//!
//! let estimate = matcher.estimate(
//!     &df,
//!     "treatment",
//!     "outcome",
//!     &["age", "income", "education"],
//!     &[IdentificationAssumption::Unconfoundedness,
//!       IdentificationAssumption::Positivity,
//!       IdentificationAssumption::NoInterference],
//!     &locke_report,
//! )?;
//!
//! println!("ATE: {} ± {} (id: {})",
//!     estimate.point, estimate.std_error, estimate.identifier);
//! # Ok::<(), cjc_causal::CausalError>(())
//! ```

pub mod assumption;
pub mod balance;
pub mod content_hash;
pub mod dml;
pub mod error;
pub mod estimate;
pub mod iv_regression;
pub mod linalg;
pub mod matching;
pub mod nuisance;
pub mod orthogonal_moment;
pub mod propensity_score;
pub mod refusal;

pub use assumption::IdentificationAssumption;
pub use dml::DoubleMLEstimator;
pub use error::CausalError;
pub use estimate::{BalanceReport, EffectEstimate};
pub use iv_regression::{weak_instrument_finding, IVRegression};
pub use propensity_score::PropensityScoreMatcher;

/// Re-export of `cjc_locke::id::FingerprintId` so callers don't need a
/// direct dep on cjc-locke just to spell the estimate's content-addressed ID.
pub use cjc_locke::id::FingerprintId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_compiles() {
        // Foundational types construct and compare cleanly.
        let a = IdentificationAssumption::Unconfoundedness;
        let b = IdentificationAssumption::Positivity;
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_id_reexport_resolves() {
        let id = FingerprintId(0xDEAD_BEEF);
        assert_eq!(format!("{}", id), "00000000deadbeef");
    }
}
