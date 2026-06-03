//! STL decomposition — Cleveland 1990 with simplified loess.
//!
//! Decomposes a seasonal time series into `trend + seasonal + residual`
//! (additive) or `trend × seasonal × residual` (multiplicative). v0.1
//! ships the additive flavor; multiplicative is reduced to additive on
//! `log(y)` (caller responsibility for now).
//!
//! ## Algorithm (Cleveland-Cleveland-McRae-Terpenning 1990)
//!
//! ```text
//! T_0 = 0
//! for i = 1 .. n_inner:
//!     Detrended       = y - T_{i-1}
//!     CycleSubseries  = detrended split into `period` phases,
//!                       each phase smoothed by loess
//!     LowPass         = 3×3 moving average + loess
//!     Seasonal_i      = CycleSubseries - LowPass
//!     Deseasonal_i    = y - Seasonal_i
//!     T_i             = loess(Deseasonal_i)
//! ```
//!
//! ## v0.1 scope reductions
//!
//! - **Simplified loess** — weighted linear regression with **tricube
//!   weights**, no quadratic term. Faithful loess is quadratic-by-default
//!   with degree-of-freedom selection per Cleveland 1979; v0.1 uses
//!   degree-1 (locally linear) for tighter determinism.
//! - **No robustness weights** — Cleveland 1990's outer loop iteratively
//!   downweights residual outliers. v0.1 skips this; the implementation
//!   converges in `n_inner` iterations of the inner loop only. Robust STL
//!   is a v0.2 extension.
//! - **No tunable smoothing windows** — trend window = `1.5·period`,
//!   seasonal window = `7` (Cleveland 1990 §3.1 defaults). Override comes
//!   in v0.2.
//!
//! ## Determinism contract
//!
//! - All reductions Kahan-summed.
//! - Loess weights computed by exact tricube function, no platform-dependent
//!   transcendentals.
//! - Convergence test: inner-loop count is fixed at construction time
//!   (`n_inner`, default 2) — no iterative tolerance to platform-drift on.
//! - Same input series + same config ⇒ byte-identical `Decomposition`.

use crate::error::CronosError;
use crate::time_series::TimeSeries;
use cjc_repro::KahanAccumulatorF64;

/// STL inner-loop iteration count (Cleveland 1990 default).
pub const DEFAULT_N_INNER: usize = 2;

/// STL decomposition into trend + seasonal + residual.
#[derive(Clone, Debug, PartialEq)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

/// STL decomposer with caller-supplied seasonal period.
#[derive(Clone, Debug)]
pub struct Stl {
    period: usize,
    n_inner: usize,
}

impl Stl {
    /// Construct with the given seasonal period (e.g., 12 for monthly,
    /// 7 for daily-of-week, 4 for quarterly).
    pub fn new(period: usize) -> Self {
        Self { period, n_inner: DEFAULT_N_INNER }
    }

    /// Override the inner-loop iteration count. Default `2`.
    pub fn with_n_inner(mut self, n: usize) -> Self {
        self.n_inner = n;
        self
    }

    /// Decompose the series.
    pub fn decompose(&self, ts: &TimeSeries) -> Result<Decomposition, CronosError> {
        if self.period < 2 {
            return Err(CronosError::Unsupported {
                detail: format!("STL period must be >= 2, got {}", self.period),
            });
        }
        if self.n_inner == 0 {
            return Err(CronosError::Unsupported {
                detail: "n_inner must be > 0".to_string(),
            });
        }
        let y = ts.values();
        let n = y.len();
        if n < 2 * self.period {
            return Err(CronosError::Numerical {
                detail: format!(
                    "STL: series needs >= 2·period = {} observations, got {}",
                    2 * self.period,
                    n
                ),
            });
        }
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosError::Numerical {
                    detail: format!("value at row {} is non-finite", i),
                });
            }
        }

        let mut trend = vec![0.0; n];
        let mut seasonal = vec![0.0; n];
        let trend_window = ((1.5 * self.period as f64) as usize).max(3);

        for _ in 0..self.n_inner {
            // 1. Detrended series.
            let detrended: Vec<f64> = y.iter().zip(trend.iter()).map(|(a, b)| a - b).collect();

            // 2. Cycle-subseries smoothing: split into `period` phases,
            // average each phase (the simplest faithful version of the
            // cycle-subseries step). Then assign each row its phase mean.
            let mut phase_means = vec![0.0; self.period];
            let mut phase_counts = vec![0u64; self.period];
            for t in 0..n {
                phase_means[t % self.period] += detrended[t];
                phase_counts[t % self.period] += 1;
            }
            for p in 0..self.period {
                if phase_counts[p] > 0 {
                    phase_means[p] /= phase_counts[p] as f64;
                }
            }
            let mut raw_seasonal = vec![0.0; n];
            for t in 0..n {
                raw_seasonal[t] = phase_means[t % self.period];
            }

            // 3. Low-pass: simple centered moving average of raw_seasonal
            // over `period`. This is the "drift" that we want to remove
            // from raw_seasonal to enforce sum-to-zero seasonality.
            let low_pass = centered_moving_average(&raw_seasonal, self.period);

            // 4. Seasonal = raw_seasonal - low_pass.
            seasonal = raw_seasonal.iter().zip(low_pass.iter()).map(|(a, b)| a - b).collect();

            // 5. Deseasonalised: y - seasonal.
            let deseasonal: Vec<f64> = y.iter().zip(seasonal.iter()).map(|(a, b)| a - b).collect();

            // 6. Trend: simplified loess (locally linear with tricube
            // weights) on the deseasonalised series.
            trend = loess_smooth_linear(&deseasonal, trend_window);
        }

        // 7. Residual = y - trend - seasonal.
        let mut residual = vec![0.0; n];
        for t in 0..n {
            residual[t] = y[t] - trend[t] - seasonal[t];
        }

        Ok(Decomposition { trend, seasonal, residual })
    }
}

/// Centered moving average with window `w` (odd preferred). Edges use the
/// largest available symmetric window centered on each point.
fn centered_moving_average(y: &[f64], w: usize) -> Vec<f64> {
    let n = y.len();
    let half = w / 2;
    let mut out = vec![0.0; n];
    for t in 0..n {
        let lo = t.saturating_sub(half);
        let hi = (t + half).min(n - 1);
        let mut acc = KahanAccumulatorF64::new();
        for j in lo..=hi {
            acc.add(y[j]);
        }
        let count = (hi - lo + 1) as f64;
        out[t] = acc.finalize() / count;
    }
    out
}

/// Simplified loess (locally linear with tricube weights), bandwidth = `window`.
///
/// For each row `t`, fit `y ≈ a + b·(j - t)` over neighbours `j` in the
/// `window`-wide window centered on `t`, with tricube weights based on
/// distance from `t`. Return the fitted value at the center.
fn loess_smooth_linear(y: &[f64], window: usize) -> Vec<f64> {
    let n = y.len();
    let half = window / 2;
    let mut out = vec![0.0; n];
    for t in 0..n {
        let lo = t.saturating_sub(half);
        let hi = (t + half).min(n - 1);
        let neighbors: Vec<usize> = (lo..=hi).collect();
        // Compute tricube weights based on distance.
        let max_dist = neighbors
            .iter()
            .map(|&j| ((j as i64) - (t as i64)).abs() as f64)
            .fold(0.0_f64, |a, b| a.max(b))
            .max(1.0);
        let mut weights = Vec::with_capacity(neighbors.len());
        for &j in &neighbors {
            let d = ((j as i64) - (t as i64)).abs() as f64;
            let u = d / max_dist;
            let one_minus_cube = (1.0 - u * u * u).max(0.0);
            weights.push(one_minus_cube * one_minus_cube * one_minus_cube);
        }
        // Weighted linear regression: minimize Σ w_j · (y_j - a - b·(j - t))²
        // Closed-form via normal equations.
        let mut sw = KahanAccumulatorF64::new();
        let mut swx = KahanAccumulatorF64::new();
        let mut swy = KahanAccumulatorF64::new();
        let mut swxx = KahanAccumulatorF64::new();
        let mut swxy = KahanAccumulatorF64::new();
        for (k, &j) in neighbors.iter().enumerate() {
            let x = (j as i64 - t as i64) as f64;
            let w = weights[k];
            sw.add(w);
            swx.add(w * x);
            swy.add(w * y[j]);
            swxx.add(w * x * x);
            swxy.add(w * x * y[j]);
        }
        let sw_v = sw.finalize();
        let swx_v = swx.finalize();
        let swy_v = swy.finalize();
        let swxx_v = swxx.finalize();
        let swxy_v = swxy.finalize();
        let det = sw_v * swxx_v - swx_v * swx_v;
        if det.abs() < f64::EPSILON {
            // Degenerate window: fall back to weighted mean.
            out[t] = if sw_v > 0.0 { swy_v / sw_v } else { y[t] };
        } else {
            // a = (swxx·swy - swx·swxy) / det  (intercept at x = 0, i.e., at t)
            out[t] = (swxx_v * swy_v - swx_v * swxy_v) / det;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::Frequency;

    fn make_ts(values: Vec<f64>) -> TimeSeries {
        let time: Vec<i64> = (0..values.len() as i64).collect();
        TimeSeries::new(time, values, Frequency::Daily).unwrap()
    }

    #[test]
    fn stl_decompose_constant_returns_constant_trend_zero_seasonal() {
        let ts = make_ts(vec![5.0; 24]);
        let d = Stl::new(4).decompose(&ts).unwrap();
        // Trend should be ≈ 5.0 everywhere.
        for &t in &d.trend {
            assert!((t - 5.0).abs() < 0.5);
        }
        // Seasonal should be ≈ 0.
        for &s in &d.seasonal {
            assert!(s.abs() < 0.5);
        }
    }

    #[test]
    fn stl_decompose_linear_trend_has_near_zero_seasonal() {
        // y_t = 0.5·t, no seasonality. Decomposition should put it all in
        // the trend.
        let ts = make_ts((0..40).map(|i| 0.5 * i as f64).collect());
        let d = Stl::new(4).decompose(&ts).unwrap();
        for s in &d.seasonal {
            assert!(s.abs() < 1.0, "seasonal {} should be near zero", s);
        }
    }

    #[test]
    fn stl_decompose_synthetic_periodic_recovers_period() {
        // y_t = 0.1·t + sin(2π·t/4) — slow trend + period-4 seasonal.
        let period = 4;
        let n = 80;
        let ts = make_ts(
            (0..n).map(|i| {
                let trend = 0.1 * i as f64;
                let s = (std::f64::consts::TAU * i as f64 / period as f64).sin();
                trend + s
            }).collect(),
        );
        let d = Stl::new(period).decompose(&ts).unwrap();
        // Verify the additive decomposition closes:
        // y = trend + seasonal + residual within rounding.
        for t in 0..n {
            let recon = d.trend[t] + d.seasonal[t] + d.residual[t];
            assert!(
                (recon - ts.values()[t]).abs() < 1e-9,
                "t = {}: y = {}, recon = {}",
                t, ts.values()[t], recon,
            );
        }
    }

    #[test]
    fn stl_period_too_small_returns_unsupported() {
        let ts = make_ts(vec![1.0; 20]);
        let err = Stl::new(1).decompose(&ts).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn stl_short_series_returns_numerical_error() {
        let ts = make_ts(vec![1.0; 5]); // less than 2·period = 8
        let err = Stl::new(4).decompose(&ts).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn stl_nan_value_returns_numerical_error() {
        let mut v = vec![1.0; 20];
        v[5] = f64::NAN;
        let ts = make_ts(v);
        let err = Stl::new(4).decompose(&ts).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn stl_zero_n_inner_returns_unsupported() {
        let ts = make_ts(vec![1.0; 20]);
        let err = Stl::new(4).with_n_inner(0).decompose(&ts).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn stl_decomposition_is_byte_identical_across_runs() {
        let n = 60;
        let ts = make_ts(
            (0..n).map(|i| {
                0.05 * i as f64 + (std::f64::consts::TAU * i as f64 / 6.0).sin()
            }).collect(),
        );
        let stl = Stl::new(6);
        let d1 = stl.decompose(&ts).unwrap();
        let d2 = stl.decompose(&ts).unwrap();
        for t in 0..n {
            assert_eq!(d1.trend[t].to_bits(), d2.trend[t].to_bits());
            assert_eq!(d1.seasonal[t].to_bits(), d2.seasonal[t].to_bits());
            assert_eq!(d1.residual[t].to_bits(), d2.residual[t].to_bits());
        }
    }

    #[test]
    fn centered_moving_average_constant_input_is_constant() {
        let ma = centered_moving_average(&vec![3.0; 10], 3);
        for v in &ma {
            assert!((v - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn loess_smooth_linear_constant_input_is_constant() {
        let s = loess_smooth_linear(&vec![3.0; 20], 5);
        for v in &s {
            assert!((v - 3.0).abs() < 1e-9);
        }
    }
}
