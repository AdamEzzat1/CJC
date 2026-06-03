//! Rolling-origin backtest harness for cjc-cronos forecasters.
//!
//! At each fold, the harness fits on `values[0..end]` and produces an
//! `max_horizon`-step forecast. Error metrics are accumulated per horizon
//! step across all folds. Final output is a [`BacktestReport`] keyed by
//! horizon (in `BTreeMap` for deterministic iteration).
//!
//! ## Determinism contract
//!
//! - All sums (per-horizon AE / SE / APE) go through [`KahanAccumulatorF64`].
//! - Per-horizon aggregates live in `BTreeMap<usize, f64>`.
//! - The fold loop advances by a fixed `step` and stops when the last fold
//!   has fewer than `max_horizon` future observations to evaluate against.
//! - The same `(model, training_data, initial_window, step, max_horizon)`
//!   produces the same `backtest_id`.
//!
//! ## API
//!
//! v0.1 exposes [`backtest_ets`] — a method-specific backtest for ETS. v0.2
//! will generalise to a `Forecaster` trait once ARIMA + Kalman ship.

use crate::error::CronosError;
use crate::ets::Ets;
use crate::forecast::{compute_backtest_id, BacktestReport};
use crate::time_series::TimeSeries;
use cjc_repro::KahanAccumulatorF64;
use std::collections::BTreeMap;

/// Run a rolling-origin backtest of an [`Ets`] forecaster on the supplied
/// time series.
///
/// # Arguments
///
/// - `ets` — configured [`Ets`] forecaster.
/// - `ts` — full time series (will be split into train + test at each fold).
/// - `initial_window` — number of observations in the smallest training set.
/// - `step` — number of observations to advance between folds. Must be `> 0`.
/// - `max_horizon` — forecast horizon evaluated at each fold. Must be `> 0`.
///
/// # Errors
///
/// - [`CronosError::Unsupported`] for invalid `initial_window` / `step` /
///   `max_horizon`.
/// - [`CronosError::Numerical`] if the series is too short to support even
///   one fold (`initial_window + max_horizon > ts.len()`).
/// - Any error returned by the underlying `Ets::fit_and_forecast` call.
pub fn backtest_ets(
    ets: &Ets,
    ts: &TimeSeries,
    initial_window: usize,
    step: usize,
    max_horizon: usize,
) -> Result<BacktestReport, CronosError> {
    if step == 0 {
        return Err(CronosError::Unsupported {
            detail: "step must be > 0".to_string(),
        });
    }
    if max_horizon == 0 {
        return Err(CronosError::Unsupported {
            detail: "max_horizon must be > 0".to_string(),
        });
    }
    if initial_window < 2 {
        return Err(CronosError::Unsupported {
            detail: format!("initial_window must be >= 2, got {}", initial_window),
        });
    }
    let n = ts.len();
    if initial_window + max_horizon > n {
        return Err(CronosError::Numerical {
            detail: format!(
                "backtest: initial_window ({}) + max_horizon ({}) > n ({})",
                initial_window, max_horizon, n
            ),
        });
    }

    // Per-horizon accumulators. Each map starts with zero entries and gets
    // populated lazily by `commit_step_metrics`.
    let mut abs_err: BTreeMap<usize, KahanAccumulatorF64> = BTreeMap::new();
    let mut sq_err: BTreeMap<usize, KahanAccumulatorF64> = BTreeMap::new();
    let mut abs_pct_err: BTreeMap<usize, KahanAccumulatorF64> = BTreeMap::new();
    // Per-horizon count of folds that contributed an evaluation (used as
    // the divisor for the metrics).
    let mut counts: BTreeMap<usize, u64> = BTreeMap::new();

    let mut n_folds: u64 = 0;
    let mut fitted_model_id_first = None;

    let values = ts.values();
    let mut train_end = initial_window;
    while train_end + max_horizon <= n {
        // Build the training-fold TimeSeries. Time index is preserved.
        let time_subset: Vec<i64> = ts.time()[..train_end].to_vec();
        let value_subset: Vec<f64> = values[..train_end].to_vec();
        let train_ts = TimeSeries::new(time_subset, value_subset, ts.frequency())?;

        let forecast = ets.fit_and_forecast(&train_ts, max_horizon)?;
        if fitted_model_id_first.is_none() {
            fitted_model_id_first = Some(forecast.fitted_model_id);
        }

        for h in 1..=max_horizon {
            let predicted = forecast.point_estimates[h - 1];
            let actual = values[train_end + h - 1];
            let err = actual - predicted;
            let ae = err.abs();
            abs_err.entry(h).or_insert_with(KahanAccumulatorF64::new).add(ae);
            sq_err.entry(h).or_insert_with(KahanAccumulatorF64::new).add(err * err);
            // MAPE skips rows where the actual is exactly zero (would
            // divide by zero); we increment the count anyway so the
            // denominator is consistent with MAE/RMSE.
            if actual != 0.0 {
                let ape = (err / actual).abs();
                abs_pct_err
                    .entry(h)
                    .or_insert_with(KahanAccumulatorF64::new)
                    .add(ape);
            }
            *counts.entry(h).or_insert(0) += 1;
        }

        n_folds += 1;
        train_end += step;
    }

    if n_folds == 0 {
        return Err(CronosError::Numerical {
            detail: "backtest produced 0 folds".to_string(),
        });
    }

    // Aggregate.
    let mut per_horizon_mae: BTreeMap<usize, f64> = BTreeMap::new();
    let mut per_horizon_mape: BTreeMap<usize, f64> = BTreeMap::new();
    let mut per_horizon_rmse: BTreeMap<usize, f64> = BTreeMap::new();
    for (h, count) in &counts {
        let c = *count as f64;
        let mae = abs_err.get(h).map(|a| a.finalize() / c).unwrap_or(f64::NAN);
        let mape = abs_pct_err
            .get(h)
            .map(|a| a.finalize() / c)
            .unwrap_or(f64::NAN);
        let rmse = sq_err
            .get(h)
            .map(|a| (a.finalize() / c).sqrt())
            .unwrap_or(f64::NAN);
        per_horizon_mae.insert(*h, mae);
        per_horizon_mape.insert(*h, mape);
        per_horizon_rmse.insert(*h, rmse);
    }

    let fitted_model_id = fitted_model_id_first.expect("at least one fold completed");
    let backtest_id = compute_backtest_id(
        fitted_model_id,
        initial_window,
        step,
        n_folds,
        max_horizon,
    );

    Ok(BacktestReport {
        per_horizon_mae,
        per_horizon_mape,
        per_horizon_rmse,
        n_folds,
        fitted_model_id,
        backtest_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ets::EtsKind;
    use crate::frequency::Frequency;

    fn make_ts(values: Vec<f64>) -> TimeSeries {
        let time: Vec<i64> = (0..values.len() as i64).collect();
        TimeSeries::new(time, values, Frequency::Daily).unwrap()
    }

    #[test]
    fn backtest_on_constant_series_has_zero_error() {
        // ETS Simple on a constant series → forecast is the constant →
        // per-horizon MAE / RMSE / MAPE all 0.
        let ts = make_ts(vec![5.0; 40]);
        let ets = Ets::new(EtsKind::Simple);
        let report = backtest_ets(&ets, &ts, 20, 1, 3).unwrap();
        for h in 1..=3 {
            assert!(report.per_horizon_mae[&h] < 1e-9);
            assert!(report.per_horizon_rmse[&h] < 1e-9);
            assert!(report.per_horizon_mape[&h] < 1e-9);
        }
        assert!(report.n_folds > 0);
    }

    #[test]
    fn backtest_id_is_deterministic_across_runs() {
        let ts = make_ts((0..40).map(|i| 1.0 + 0.1 * i as f64).collect());
        let ets = Ets::new(EtsKind::Holt);
        let r1 = backtest_ets(&ets, &ts, 20, 2, 5).unwrap();
        let r2 = backtest_ets(&ets, &ts, 20, 2, 5).unwrap();
        assert_eq!(r1.backtest_id, r2.backtest_id);
        assert_eq!(r1.fitted_model_id, r2.fitted_model_id);
        assert_eq!(r1.n_folds, r2.n_folds);
    }

    #[test]
    fn step_zero_returns_unsupported() {
        let ts = make_ts(vec![1.0; 30]);
        let err = backtest_ets(&Ets::new(EtsKind::Simple), &ts, 10, 0, 3).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn max_horizon_zero_returns_unsupported() {
        let ts = make_ts(vec![1.0; 30]);
        let err = backtest_ets(&Ets::new(EtsKind::Simple), &ts, 10, 1, 0).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn initial_window_too_small_returns_unsupported() {
        let ts = make_ts(vec![1.0; 30]);
        let err = backtest_ets(&Ets::new(EtsKind::Simple), &ts, 1, 1, 3).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn series_too_short_for_any_fold_returns_numerical_error() {
        let ts = make_ts(vec![1.0; 10]);
        let err = backtest_ets(&Ets::new(EtsKind::Simple), &ts, 8, 1, 5).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn per_horizon_metrics_have_max_horizon_entries() {
        let ts = make_ts((0..40).map(|i| 1.0 + 0.1 * i as f64).collect());
        let ets = Ets::new(EtsKind::Holt);
        let report = backtest_ets(&ets, &ts, 20, 1, 5).unwrap();
        assert_eq!(report.per_horizon_mae.len(), 5);
        assert_eq!(report.per_horizon_mape.len(), 5);
        assert_eq!(report.per_horizon_rmse.len(), 5);
    }
}
