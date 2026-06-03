//! Bolero fuzz targets for cjc-cronos.
//!
//! Invariant: every public entry point either returns a structured
//! `CronosError` or a well-formed output. **Never panics.**

use cjc_cronos::{backtest_ets, Ets, EtsKind, Frequency, TimeSeries};

#[test]
fn fuzz_arbitrary_values_never_panic() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|values: &Vec<f64>| {
            let n = values.len();
            if n < 3 {
                return;
            }
            let time: Vec<i64> = (0..n as i64).collect();
            // The TimeSeries constructor itself only fails on unsorted time,
            // which our generated `0..n` time index always satisfies.
            let ts = match TimeSeries::new(time, values.clone(), Frequency::Daily) {
                Ok(ts) => ts,
                Err(_) => return,
            };
            let _ = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 3);
            let _ = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 3);
        });
}

#[test]
fn fuzz_arbitrary_horizon_never_panics() {
    let n = 30;
    let time: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let ts = TimeSeries::new(time, values, Frequency::Daily).unwrap();

    bolero::check!()
        .with_type::<u8>()
        .for_each(|h: &u8| {
            let _ = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, *h as usize);
            let _ = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, *h as usize);
        });
}

#[test]
fn fuzz_arbitrary_confidence_level_never_panics() {
    let n = 30;
    let time: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let ts = TimeSeries::new(time, values, Frequency::Daily).unwrap();

    bolero::check!()
        .with_type::<f64>()
        .for_each(|cl: &f64| {
            let _ = Ets::new(EtsKind::Holt)
                .with_confidence_level(*cl)
                .fit_and_forecast(&ts, 3);
        });
}

#[test]
fn fuzz_arbitrary_backtest_step_never_panics() {
    let n = 60;
    let time: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let ts = TimeSeries::new(time, values, Frequency::Daily).unwrap();
    let ets = Ets::new(EtsKind::Simple);

    bolero::check!()
        .with_type::<u8>()
        .for_each(|step: &u8| {
            let _ = backtest_ets(&ets, &ts, 20, *step as usize, 3);
        });
}
