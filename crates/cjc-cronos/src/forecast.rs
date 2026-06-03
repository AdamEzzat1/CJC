//! Output types for cjc-cronos forecasters and backtest harnesses.
//!
//! All identifiers in this module are content-addressed via
//! [`cjc_locke::id::fingerprint_compose`] under [`IdDomain::CausalClaim`].
//! We deliberately reuse Locke's existing `CausalClaim` domain rather than
//! introducing a `TimeSeriesForecast` variant so the dependency direction
//! stays clean (cjc-cronos → cjc-locke only, no upward dep).

use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use std::collections::BTreeMap;

/// A point-and-interval forecast over `horizon` steps.
///
/// `point_estimates`, `lower_bound`, and `upper_bound` all have length
/// `horizon`. The content-addressed [`fitted_model_id`] is a function of
/// the model class, hyperparameters, training data fingerprint, and seed —
/// two runs that fit the same model on the same data produce the same
/// `fitted_model_id`.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    /// Number of steps ahead.
    pub horizon: usize,
    /// Per-step point estimates (length `horizon`).
    pub point_estimates: Vec<f64>,
    /// Per-step lower confidence bound (length `horizon`).
    pub lower_bound: Vec<f64>,
    /// Per-step upper confidence bound (length `horizon`).
    pub upper_bound: Vec<f64>,
    /// Confidence level used for the bounds; default `0.95`.
    pub confidence_level: f64,
    /// Content-addressed identifier of the fitted model. Stable across runs
    /// for identical inputs.
    pub fitted_model_id: FingerprintId,
}

/// Backtest output: per-horizon error metrics + content-addressed identifiers.
///
/// `per_horizon_*` is keyed by horizon (1-indexed) and stored in `BTreeMap`
/// so iteration order is deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct BacktestReport {
    /// Mean Absolute Error per horizon step.
    pub per_horizon_mae: BTreeMap<usize, f64>,
    /// Mean Absolute Percentage Error per horizon step.
    pub per_horizon_mape: BTreeMap<usize, f64>,
    /// Root Mean Squared Error per horizon step.
    pub per_horizon_rmse: BTreeMap<usize, f64>,
    /// Number of rolling-origin folds completed.
    pub n_folds: u64,
    /// Content-addressed identifier of the model used in the backtest
    /// (same value across all folds).
    pub fitted_model_id: FingerprintId,
    /// Content-addressed identifier of the backtest configuration:
    /// `(fitted_model_id, initial_window, step, n_folds, max_horizon)`.
    pub backtest_id: FingerprintId,
}

/// Compute the content-addressed `fitted_model_id` for an ETS model.
///
/// Inputs hashed (in canonical order):
/// 1. Model-class label (`"ets_simple"` or `"ets_holt"`)
/// 2. `alpha` bits (`f64::to_bits`)
/// 3. `beta` bits (or 0u64 for Simple ETS, which has no `β`)
/// 4. Training series length
/// 5. SHA-style fingerprint of the training values (`f64::to_bits` LE bytes)
/// 6. Confidence level bits
pub fn compute_ets_model_id(
    model_class: &str,
    alpha: f64,
    beta: Option<f64>,
    training_values: &[f64],
    confidence_level: f64,
) -> FingerprintId {
    let mut parts: Vec<FingerprintId> = Vec::with_capacity(6);
    parts.push(fingerprint_str(IdDomain::CausalClaim, model_class));
    parts.push(fingerprint(IdDomain::CausalClaim, &alpha.to_bits().to_le_bytes()));
    let beta_bits = beta.unwrap_or(0.0).to_bits();
    parts.push(fingerprint(IdDomain::CausalClaim, &beta_bits.to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &(training_values.len() as u64).to_le_bytes()));
    // Hash the training values to a single fingerprint by feeding all bytes.
    let mut value_bytes: Vec<u8> = Vec::with_capacity(training_values.len() * 8);
    for v in training_values {
        value_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    parts.push(fingerprint(IdDomain::CausalClaim, &value_bytes));
    parts.push(fingerprint(IdDomain::CausalClaim, &confidence_level.to_bits().to_le_bytes()));
    fingerprint_compose(IdDomain::CausalClaim, "ets_model", &parts)
}

/// Compute the content-addressed `backtest_id` for a backtest configuration.
pub fn compute_backtest_id(
    fitted_model_id: FingerprintId,
    initial_window: usize,
    step: usize,
    n_folds: u64,
    max_horizon: usize,
) -> FingerprintId {
    let parts: Vec<FingerprintId> = vec![
        fitted_model_id,
        fingerprint(IdDomain::CausalClaim, &(initial_window as u64).to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &(step as u64).to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &n_folds.to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &(max_horizon as u64).to_le_bytes()),
    ];
    fingerprint_compose(IdDomain::CausalClaim, "ets_backtest", &parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ets_model_id_is_deterministic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let a = compute_ets_model_id("ets_simple", 0.3, None, &v, 0.95);
        let b = compute_ets_model_id("ets_simple", 0.3, None, &v, 0.95);
        assert_eq!(a, b);
    }

    #[test]
    fn ets_model_id_changes_with_alpha() {
        let v = vec![1.0, 2.0, 3.0];
        let a = compute_ets_model_id("ets_simple", 0.3, None, &v, 0.95);
        let b = compute_ets_model_id("ets_simple", 0.4, None, &v, 0.95);
        assert_ne!(a, b);
    }

    #[test]
    fn ets_model_id_changes_with_class() {
        let v = vec![1.0, 2.0, 3.0];
        let a = compute_ets_model_id("ets_simple", 0.3, None, &v, 0.95);
        let b = compute_ets_model_id("ets_holt", 0.3, Some(0.2), &v, 0.95);
        assert_ne!(a, b);
    }

    #[test]
    fn ets_model_id_changes_with_training_data() {
        let a = compute_ets_model_id("ets_simple", 0.3, None, &[1.0, 2.0, 3.0], 0.95);
        let b = compute_ets_model_id("ets_simple", 0.3, None, &[1.0, 2.0, 4.0], 0.95);
        assert_ne!(a, b);
    }

    #[test]
    fn backtest_id_is_deterministic() {
        let model_id = FingerprintId(0xCAFE);
        let a = compute_backtest_id(model_id, 50, 1, 30, 5);
        let b = compute_backtest_id(model_id, 50, 1, 30, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn backtest_id_changes_with_window_or_horizon() {
        let model_id = FingerprintId(0xCAFE);
        let a = compute_backtest_id(model_id, 50, 1, 30, 5);
        let b = compute_backtest_id(model_id, 60, 1, 30, 5); // different window
        assert_ne!(a, b);
        let c = compute_backtest_id(model_id, 50, 1, 30, 10); // different horizon
        assert_ne!(a, c);
    }
}
