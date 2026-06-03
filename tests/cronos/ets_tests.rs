//! Integration tests for ETS Simple + Holt forecasters.

use cjc_cronos::{
    backtest_ets, CronosError, Ets, EtsKind, Frequency, TimeSeries,
};
use cjc_data::{Column, DataFrame};

fn linear_series(n: usize, slope: f64, intercept: f64) -> TimeSeries {
    let time: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| intercept + slope * i as f64).collect();
    TimeSeries::new(time, values, Frequency::Daily).unwrap()
}

fn noisy_linear_series(n: usize, slope: f64, intercept: f64, noise_seed: u64) -> TimeSeries {
    let mut rng = cjc_repro::Rng::seeded(noise_seed);
    let time: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n)
        .map(|i| intercept + slope * i as f64 + (rng.next_f64() - 0.5) * 0.5)
        .collect();
    TimeSeries::new(time, values, Frequency::Daily).unwrap()
}

#[test]
fn ets_simple_on_constant_series_predicts_constant() {
    let ts = TimeSeries::new(
        (0..30).collect(),
        vec![10.0; 30],
        Frequency::Daily,
    )
    .unwrap();
    let f = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 5).unwrap();
    for &p in &f.point_estimates {
        assert!((p - 10.0).abs() < 1e-6, "got {}", p);
    }
}

#[test]
fn ets_holt_on_linear_series_recovers_trend() {
    // y_t = 2 + 0.5·t. Holt should recover trend ≈ 0.5.
    let ts = linear_series(40, 0.5, 2.0);
    let f = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 10).unwrap();
    // Last point in training is 2 + 0.5·39 = 21.5.
    // Forecast h=1: 21.5 + 0.5 = 22.0 (approximately).
    // Forecast h=10: 21.5 + 5.0 = 26.5 (approximately).
    let last_train = 2.0 + 0.5 * 39.0;
    for h in 0..10 {
        let expected = last_train + 0.5 * (h as f64 + 1.0);
        assert!(
            (f.point_estimates[h] - expected).abs() < 0.5,
            "h={}: expected {}, got {}",
            h + 1, expected, f.point_estimates[h]
        );
    }
}

#[test]
fn ets_simple_bounds_widen_with_horizon() {
    let ts = noisy_linear_series(40, 0.0, 5.0, 42);
    let f = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 10).unwrap();
    let w1 = f.upper_bound[0] - f.lower_bound[0];
    let w5 = f.upper_bound[4] - f.lower_bound[4];
    let w10 = f.upper_bound[9] - f.lower_bound[9];
    assert!(w5 >= w1);
    assert!(w10 >= w5);
}

#[test]
fn ets_forecast_is_byte_identical_across_runs() {
    let ts = noisy_linear_series(40, 0.3, 2.0, 99);
    let ets = Ets::new(EtsKind::Holt);
    let f1 = ets.fit_and_forecast(&ts, 5).unwrap();
    let f2 = ets.fit_and_forecast(&ts, 5).unwrap();
    for h in 0..5 {
        assert_eq!(f1.point_estimates[h].to_bits(), f2.point_estimates[h].to_bits());
        assert_eq!(f1.lower_bound[h].to_bits(), f2.lower_bound[h].to_bits());
        assert_eq!(f1.upper_bound[h].to_bits(), f2.upper_bound[h].to_bits());
    }
    assert_eq!(f1.fitted_model_id, f2.fitted_model_id);
}

#[test]
fn ets_horizon_zero_returns_unsupported() {
    let ts = linear_series(20, 1.0, 0.0);
    let err = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 0).unwrap_err();
    assert!(matches!(err, CronosError::Unsupported { .. }));
}

#[test]
fn ets_holt_needs_at_least_three_observations() {
    let ts = TimeSeries::new(vec![1, 2], vec![1.0, 2.0], Frequency::Daily).unwrap();
    let err = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 5).unwrap_err();
    assert!(matches!(err, CronosError::Numerical { .. }));
}

#[test]
fn ets_simple_needs_at_least_two_observations() {
    let ts = TimeSeries::new(vec![1], vec![1.0], Frequency::Daily).unwrap();
    let err = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 5).unwrap_err();
    assert!(matches!(err, CronosError::Numerical { .. }));
}

#[test]
fn ets_with_invalid_alpha_grid_value_returns_unsupported() {
    let ts = linear_series(20, 1.0, 0.0);
    let err = Ets::new(EtsKind::Simple)
        .with_alpha_grid(vec![0.5, 1.5]) // 1.5 invalid
        .fit_and_forecast(&ts, 3)
        .unwrap_err();
    assert!(matches!(err, CronosError::Unsupported { .. }));
}

#[test]
fn ets_invalid_confidence_level_returns_unsupported() {
    let ts = linear_series(20, 1.0, 0.0);
    let err = Ets::new(EtsKind::Holt)
        .with_confidence_level(1.5)
        .fit_and_forecast(&ts, 3)
        .unwrap_err();
    assert!(matches!(err, CronosError::Unsupported { .. }));
}

#[test]
fn ets_simple_and_holt_have_different_fitted_model_ids() {
    let ts = noisy_linear_series(30, 0.2, 1.0, 7);
    let f_simple = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 5).unwrap();
    let f_holt = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 5).unwrap();
    assert_ne!(f_simple.fitted_model_id, f_holt.fitted_model_id);
}

#[test]
fn backtest_on_linear_series_produces_expected_fold_count() {
    // n=40, initial_window=20, step=1, max_horizon=5
    // Folds: train_end starts at 20, advances by 1 while train_end + 5 ≤ 40.
    //   First fold: train_end = 20, evaluates rows 20..25.
    //   Last fold: train_end = 35, evaluates rows 35..40.
    //   So n_folds = 35 - 20 + 1 = 16.
    let ts = linear_series(40, 0.5, 2.0);
    let report = backtest_ets(&Ets::new(EtsKind::Holt), &ts, 20, 1, 5).unwrap();
    assert_eq!(report.n_folds, 16);
    assert_eq!(report.per_horizon_mae.len(), 5);
}

#[test]
fn backtest_with_larger_step_produces_fewer_folds() {
    let ts = linear_series(60, 0.5, 2.0);
    let r_step1 = backtest_ets(&Ets::new(EtsKind::Holt), &ts, 20, 1, 3).unwrap();
    let r_step5 = backtest_ets(&Ets::new(EtsKind::Holt), &ts, 20, 5, 3).unwrap();
    assert!(r_step5.n_folds < r_step1.n_folds);
}

#[test]
fn backtest_id_is_byte_identical_across_runs() {
    let ts = noisy_linear_series(40, 0.3, 2.0, 99);
    let ets = Ets::new(EtsKind::Holt);
    let r1 = backtest_ets(&ets, &ts, 20, 1, 3).unwrap();
    let r2 = backtest_ets(&ets, &ts, 20, 1, 3).unwrap();
    assert_eq!(r1.backtest_id, r2.backtest_id);
    assert_eq!(r1.n_folds, r2.n_folds);
    for h in 1..=3 {
        assert_eq!(r1.per_horizon_mae[&h].to_bits(), r2.per_horizon_mae[&h].to_bits());
        assert_eq!(r1.per_horizon_rmse[&h].to_bits(), r2.per_horizon_rmse[&h].to_bits());
    }
}

#[test]
fn backtest_step_zero_returns_unsupported() {
    let ts = linear_series(30, 1.0, 0.0);
    let err = backtest_ets(&Ets::new(EtsKind::Simple), &ts, 10, 0, 3).unwrap_err();
    assert!(matches!(err, CronosError::Unsupported { .. }));
}

#[test]
fn time_series_from_dataframe_reads_correctly() {
    let df = DataFrame::from_columns(vec![
        ("time".into(), Column::Int(vec![1, 2, 3, 4, 5])),
        ("value".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
    ])
    .unwrap();
    let ts = TimeSeries::from_dataframe(&df, "time", "value", Frequency::Daily).unwrap();
    assert_eq!(ts.len(), 5);
    assert_eq!(ts.values()[2], 30.0);
    assert_eq!(ts.frequency(), Frequency::Daily);
}

#[test]
fn time_series_from_dataframe_rejects_string_value_column() {
    let df = DataFrame::from_columns(vec![
        ("time".into(), Column::Int(vec![1, 2, 3])),
        ("value".into(), Column::Str(vec!["a".into(), "b".into(), "c".into()])),
    ])
    .unwrap();
    let err = TimeSeries::from_dataframe(&df, "time", "value", Frequency::Daily).unwrap_err();
    assert!(matches!(err, CronosError::WrongColumnType { .. }));
}

#[test]
fn time_series_from_dataframe_rejects_unsorted_time_index() {
    let df = DataFrame::from_columns(vec![
        ("time".into(), Column::Int(vec![1, 3, 2, 4])),
        ("value".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0])),
    ])
    .unwrap();
    let err = TimeSeries::from_dataframe(&df, "time", "value", Frequency::Daily).unwrap_err();
    assert!(matches!(err, CronosError::UnsortedTimeIndex { .. }));
}
