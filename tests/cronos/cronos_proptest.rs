//! Property tests for cjc-cronos ETS + backtest.

use cjc_cronos::{backtest_ets, Ets, EtsKind, Frequency, TimeSeries};
use proptest::prelude::*;

fn make_ts_from_vec(values: Vec<f64>) -> TimeSeries {
    let time: Vec<i64> = (0..values.len() as i64).collect();
    TimeSeries::new(time, values, Frequency::Daily).unwrap()
}

proptest! {
    /// p1 — same input ⇒ byte-identical Forecast.
    #[test]
    fn cronos_p1_same_input_byte_identical(
        seed in 0u64..1_000u64,
        n in 30usize..100usize,
        slope in -2.0f64..2.0f64,
        intercept in -10.0f64..10.0f64,
    ) {
        let mut rng = cjc_repro::Rng::seeded(seed);
        let values: Vec<f64> = (0..n).map(|i| {
            intercept + slope * i as f64 + (rng.next_f64() - 0.5) * 0.3
        }).collect();
        let ts = make_ts_from_vec(values);
        let ets = Ets::new(EtsKind::Holt);
        let f1 = ets.fit_and_forecast(&ts, 5);
        let f2 = ets.fit_and_forecast(&ts, 5);
        match (f1, f2) {
            (Ok(a), Ok(b)) => {
                for h in 0..5 {
                    prop_assert_eq!(a.point_estimates[h].to_bits(), b.point_estimates[h].to_bits());
                    prop_assert_eq!(a.lower_bound[h].to_bits(), b.lower_bound[h].to_bits());
                    prop_assert_eq!(a.upper_bound[h].to_bits(), b.upper_bound[h].to_bits());
                }
                prop_assert_eq!(a.fitted_model_id, b.fitted_model_id);
            }
            (Err(ea), Err(eb)) => {
                prop_assert_eq!(format!("{}", ea), format!("{}", eb));
            }
            _ => prop_assert!(false, "Ok/Err disagreement"),
        }
    }

    /// p2 — Simple ETS forecast of a constant series is a constant series.
    #[test]
    fn cronos_p2_simple_constant_forecast(
        c in -100.0f64..100.0f64,
        horizon in 1usize..20usize,
    ) {
        let ts = make_ts_from_vec(vec![c; 30]);
        if let Ok(f) = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, horizon) {
            for &p in &f.point_estimates {
                prop_assert!((p - c).abs() < 1e-6, "got {} for constant {}", p, c);
            }
        }
    }

    /// p3 — Holt ETS forecast bounds bracket the point estimate.
    #[test]
    fn cronos_p3_bounds_bracket_point(
        seed in 0u64..1_000u64,
        n in 30usize..80usize,
    ) {
        let mut rng = cjc_repro::Rng::seeded(seed);
        let values: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64 + (rng.next_f64() - 0.5) * 0.2).collect();
        let ts = make_ts_from_vec(values);
        if let Ok(f) = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 10) {
            for h in 0..10 {
                prop_assert!(f.lower_bound[h] <= f.point_estimates[h]);
                prop_assert!(f.point_estimates[h] <= f.upper_bound[h]);
            }
        }
    }

    /// p4 — Forecast bound width is non-decreasing in horizon.
    #[test]
    fn cronos_p4_bound_width_non_decreasing(
        seed in 0u64..1_000u64,
    ) {
        let mut rng = cjc_repro::Rng::seeded(seed);
        let values: Vec<f64> = (0..50).map(|i| 1.0 + 0.1 * i as f64 + (rng.next_f64() - 0.5) * 0.5).collect();
        let ts = make_ts_from_vec(values);
        if let Ok(f) = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 10) {
            for h in 1..10 {
                let w_prev = f.upper_bound[h - 1] - f.lower_bound[h - 1];
                let w_curr = f.upper_bound[h] - f.lower_bound[h];
                prop_assert!(
                    w_curr >= w_prev - 1e-12,
                    "h = {}: bound width decreased {} → {}",
                    h, w_prev, w_curr,
                );
            }
        }
    }

    /// p5 — Backtest with step = 1 produces at least as many folds as step = 2.
    #[test]
    fn cronos_p5_backtest_step_monotonicity(
        seed in 0u64..1_000u64,
    ) {
        let mut rng = cjc_repro::Rng::seeded(seed);
        let values: Vec<f64> = (0..60).map(|_| (rng.next_f64() - 0.5) * 10.0).collect();
        let ts = make_ts_from_vec(values);
        let ets = Ets::new(EtsKind::Simple);
        let r1 = backtest_ets(&ets, &ts, 20, 1, 3);
        let r2 = backtest_ets(&ets, &ts, 20, 2, 3);
        if let (Ok(a), Ok(b)) = (r1, r2) {
            prop_assert!(a.n_folds >= b.n_folds);
        }
    }
}
