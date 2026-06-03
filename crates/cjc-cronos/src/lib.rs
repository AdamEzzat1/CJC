//! # cjc-cronos — deterministic time-series and forecasting for CJC-Lang
//!
//! **Status:** v0.1 SCAFFOLDING. The foundational types are defined; the
//! estimator + forecaster implementations land across subsequent sessions.
//! See the handoff at
//! `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md`
//! and `ADR-0044 cjc-cronos v0.1`.
//!
//! ## Headline value claim
//!
//! Backtests, forecasts, and decompositions that are **byte-identically
//! reproducible** across runs, library upgrades, and platforms. Two runs
//! over the same time series + same configuration produce the same
//! [`Forecast`] (including its content-addressed [`fitted_model_id`])
//! and the same [`BacktestReport`] (including its content-addressed
//! identifier).
//!
//! Python time-series incumbents (statsmodels, Prophet, darts) do not
//! provide structural determinism. cjc-cronos's contract is exactly that.
//!
//! ## Determinism contract
//!
//! 1. All floating-point reductions go through `cjc_repro::KahanAccumulatorF64`.
//! 2. All map iteration uses `BTreeMap` / `BTreeSet`.
//! 3. All randomness (bootstrap CIs, if any) routes through `cjc_repro::Rng`
//!    (SplitMix64) with explicit seed threading.
//! 4. Kalman recursions use the Joseph form for numerical PSD preservation.
//! 5. No FMA. `RUSTFLAGS` must not enable `target-feature=+fma`.
//! 6. STL convergence test uses `|new - old| < eps` (absolute), not signed
//!    difference.
//!
//! See ADR-0044 for the full determinism note + per-method audit checklist.
//!
//! ## What v0.1 will ship (across implementation sessions)
//!
//! - **ETS** — Simple / Holt / Holt-Winters with additive + multiplicative
//!   seasonality. Grid-search smoothing-parameter optimisation in v0.1
//!   (deterministic by construction); gradient-based in v0.2.
//! - **ARIMA / SARIMA** — Yule-Walker initialisation, CSS maximum-likelihood
//!   refinement, MA via Hannan-Rissanen. Hand-specified `(p, d, q)` in v0.1;
//!   `auto_arima` in v0.2.
//! - **State-space / Kalman filter and smoother** — Forward filter (Joseph
//!   form) + RTS smoother. Local-level + local-linear-trend + Basic
//!   Structural Model (BSM) variants.
//! - **STL decomposition** — Cleveland 1990 algorithm. Per-iteration loess
//!   smoother (Kahan-summed weighted sum).
//!
//! ## What v0.1 will NOT do
//!
//! Out of scope for v0.1: `auto_arima`, gradient-based ETS fitting,
//! multivariate VAR/VECM, Prophet-style changepoint detection, neural
//! forecasting (DeepAR / N-BEATS / TFT). See the handoff §3.7 for the
//! full deferral list and rationale.
//!
//! ## Composing with cjc-locke
//!
//! cjc-locke ships `drift::compare(train, test)` for distributional drift.
//! cjc-cronos is the natural producer of `test`: fit on train → forecast →
//! cjc-locke drift-compares the forecast distribution to the held-out
//! ground truth. Cronos-side findings emit in the `E9200..=E9299` range
//! per the handoff §5.3 reservation.

pub mod backtest;
pub mod error;
pub mod arima;
pub mod ets;
pub mod forecast;
pub mod frequency;
pub mod time_series;

pub use arima::Arima;
pub use backtest::backtest_ets;
pub use error::CronosError;
pub use ets::{Ets, EtsKind};
pub use forecast::{BacktestReport, Forecast};
pub use frequency::Frequency;
pub use time_series::TimeSeries;

/// Re-export of `cjc_locke::id::FingerprintId` so callers don't need a
/// direct dep on cjc-locke just to spell content-addressed IDs.
pub use cjc_locke::id::FingerprintId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_compiles() {
        let f = Frequency::Daily;
        assert_eq!(f, Frequency::Daily);
        assert_ne!(f, Frequency::Hourly);
    }

    #[test]
    fn fingerprint_id_reexport_resolves() {
        let id = FingerprintId(0xDEAD_BEEF);
        assert_eq!(format!("{}", id), "00000000deadbeef");
    }
}
