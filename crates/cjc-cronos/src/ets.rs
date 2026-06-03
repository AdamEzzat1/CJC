//! Exponential smoothing (ETS) forecasters: Simple + Holt linear.
//!
//! v0.1 ships **Simple** (level only) and **Holt** (level + additive trend).
//! Holt-Winters (level + trend + seasonal) is a follow-up session inside the
//! v0.1 ship — multiplicative-vs-additive seasonality and period detection
//! deserve their own focused pass.
//!
//! ## Algorithm
//!
//! Hyndman-Athanasopoulos OTexts §7.1 (Simple), §7.2 (Holt) recursions:
//!
//! ### Simple ETS
//! ```text
//! init:  l_1 = y_1
//! step:  l_t = α·y_t + (1-α)·l_{t-1}
//! error: e_t = y_t - l_{t-1}    (one-step-ahead)
//! fcast: ŷ_{n+h} = l_n           (constant for all h)
//! ```
//!
//! ### Holt (additive trend)
//! ```text
//! init:  l_1 = y_1, b_1 = y_2 - y_1
//! step:  l_t = α·y_t + (1-α)·(l_{t-1} + b_{t-1})
//!        b_t = β·(l_t - l_{t-1}) + (1-β)·b_{t-1}
//! error: e_t = y_t - (l_{t-1} + b_{t-1})
//! fcast: ŷ_{n+h} = l_n + h·b_n
//! ```
//!
//! ## Hyperparameter selection
//!
//! v0.1 ships **grid search** over `α ∈ {0.05, 0.10, …, 0.95}` (19 values)
//! and `β ∈ {0.05, 0.10, …, 0.95}` — minimise sum of one-step-ahead squared
//! errors. Grid search is **deterministic by construction**: same series →
//! same `(α, β)` regardless of platform or run.
//!
//! Gradient-based smoothing-parameter fitting is a v0.2 extension (it needs
//! careful step-size handling to preserve byte identity across platforms).
//!
//! ## Forecast intervals
//!
//! Normal-approximation bounds: `ŷ_{n+h} ± z_{α/2} · σ̂ · sqrt(h)` for Simple
//! and `ŷ_{n+h} ± z_{α/2} · σ̂ · sqrt(h·(1 + (h-1)·α²·(1+...)))` for Holt.
//! v0.1 uses the simplified `sqrt(h)` scaling for both; the exact Holt
//! interval formula (Hyndman-Athanasopoulos §7.5) is a v0.2 refinement.
//!
//! ## Determinism
//!
//! - Every reduction (SSE, residual SD, forecast-bound sums) goes through
//!   [`KahanAccumulatorF64`].
//! - Grid search iterates over the explicit `Vec<f64>` in insertion order;
//!   ties on SSE keep the lexicographically-first candidate (the first one
//!   to achieve the minimum stays — strict-less-than comparison).

use crate::error::CronosError;
use crate::forecast::{compute_ets_model_id, Forecast};
use crate::time_series::TimeSeries;
use cjc_repro::KahanAccumulatorF64;

/// Which ETS model class to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EtsKind {
    /// Level-only smoother. Forecast is flat (`ŷ_{n+h} = l_n`).
    Simple,
    /// Level + additive trend. Forecast is linear (`ŷ_{n+h} = l_n + h·b_n`).
    Holt,
}

impl EtsKind {
    /// Stable string label used in [`Forecast::fitted_model_id`] hashing.
    pub const fn label(self) -> &'static str {
        match self {
            EtsKind::Simple => "ets_simple",
            EtsKind::Holt => "ets_holt",
        }
    }
}

/// Default smoothing-parameter grid: `0.05, 0.10, …, 0.95` (19 values).
pub fn default_grid() -> Vec<f64> {
    (1..20).map(|i| i as f64 * 0.05).collect()
}

/// Exponential smoothing forecaster.
#[derive(Clone, Debug)]
pub struct Ets {
    kind: EtsKind,
    confidence_level: f64,
    alpha_grid: Vec<f64>,
    beta_grid: Vec<f64>,
}

impl Ets {
    /// Construct with default `α ∈ {0.05, 0.10, …, 0.95}` (and `β` same for Holt).
    pub fn new(kind: EtsKind) -> Self {
        Self {
            kind,
            confidence_level: 0.95,
            alpha_grid: default_grid(),
            beta_grid: default_grid(),
        }
    }

    /// Confidence level in `(0, 1)` for the forecast bounds.
    pub fn with_confidence_level(mut self, l: f64) -> Self {
        self.confidence_level = l;
        self
    }

    /// Override the α grid. Values must each lie in `(0, 1)`; otherwise
    /// [`Self::fit_and_forecast`] returns [`CronosError::Unsupported`].
    pub fn with_alpha_grid(mut self, g: Vec<f64>) -> Self {
        self.alpha_grid = g;
        self
    }

    /// Override the β grid (only consulted for [`EtsKind::Holt`]).
    pub fn with_beta_grid(mut self, g: Vec<f64>) -> Self {
        self.beta_grid = g;
        self
    }

    /// Fit ETS on the supplied series and produce an `horizon`-step forecast.
    pub fn fit_and_forecast(
        &self,
        ts: &TimeSeries,
        horizon: usize,
    ) -> Result<Forecast, CronosError> {
        // Config validation.
        if horizon == 0 {
            return Err(CronosError::Unsupported {
                detail: "horizon must be > 0".to_string(),
            });
        }
        if !self.confidence_level.is_finite()
            || self.confidence_level <= 0.0
            || self.confidence_level >= 1.0
        {
            return Err(CronosError::Unsupported {
                detail: format!(
                    "confidence_level must be in (0, 1), got {}",
                    self.confidence_level
                ),
            });
        }
        if self.alpha_grid.is_empty() {
            return Err(CronosError::Unsupported {
                detail: "alpha_grid is empty".to_string(),
            });
        }
        for a in &self.alpha_grid {
            if !a.is_finite() || *a <= 0.0 || *a >= 1.0 {
                return Err(CronosError::Unsupported {
                    detail: format!("alpha {} is not in (0, 1)", a),
                });
            }
        }
        if matches!(self.kind, EtsKind::Holt) {
            if self.beta_grid.is_empty() {
                return Err(CronosError::Unsupported {
                    detail: "beta_grid is empty for Holt".to_string(),
                });
            }
            for b in &self.beta_grid {
                if !b.is_finite() || *b <= 0.0 || *b >= 1.0 {
                    return Err(CronosError::Unsupported {
                        detail: format!("beta {} is not in (0, 1)", b),
                    });
                }
            }
        }

        let values = ts.values();
        let n = values.len();
        let min_n = match self.kind {
            EtsKind::Simple => 2,
            EtsKind::Holt => 3,
        };
        if n < min_n {
            return Err(CronosError::Numerical {
                detail: format!(
                    "ETS {:?} needs at least {} observations, got {}",
                    self.kind, min_n, n
                ),
            });
        }
        // Disallow non-finite inputs (NaN / Inf would propagate everywhere).
        for (i, v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosError::Numerical {
                    detail: format!("value at row {} is non-finite ({})", i, v),
                });
            }
        }

        // Grid search + fit.
        let (point_estimates, lower_bound, upper_bound, alpha_best, beta_best) = match self.kind {
            EtsKind::Simple => {
                let (alpha, level_t, sigma) = grid_search_simple(values, &self.alpha_grid);
                let (p, lo, hi) =
                    forecast_simple(level_t, sigma, horizon, self.confidence_level);
                (p, lo, hi, alpha, None)
            }
            EtsKind::Holt => {
                let (alpha, beta, level_t, trend_t, sigma) =
                    grid_search_holt(values, &self.alpha_grid, &self.beta_grid);
                let (p, lo, hi) = forecast_holt(
                    level_t,
                    trend_t,
                    sigma,
                    horizon,
                    self.confidence_level,
                );
                (p, lo, hi, alpha, Some(beta))
            }
        };

        let fitted_model_id = compute_ets_model_id(
            self.kind.label(),
            alpha_best,
            beta_best,
            values,
            self.confidence_level,
        );

        Ok(Forecast {
            horizon,
            point_estimates,
            lower_bound,
            upper_bound,
            confidence_level: self.confidence_level,
            fitted_model_id,
        })
    }
}

/// Fit Simple ETS with the given α. Returns `(level_T, residual_sd)`.
///
/// Internal — the public entrypoint is [`Ets::fit_and_forecast`].
pub(crate) fn fit_simple_with_alpha(values: &[f64], alpha: f64) -> (f64, f64) {
    let n = values.len();
    let mut level = values[0];
    let mut sse = KahanAccumulatorF64::new();
    let mut k: u64 = 0; // count of one-step errors (n - 1)
    for t in 1..n {
        let err = values[t] - level;
        sse.add(err * err);
        k += 1;
        level = alpha * values[t] + (1.0 - alpha) * level;
    }
    let sigma = if k > 0 {
        (sse.finalize() / k as f64).sqrt()
    } else {
        0.0
    };
    (level, sigma)
}

/// Grid-search Simple ETS. Returns `(best_alpha, level_T, residual_sd)`.
pub(crate) fn grid_search_simple(values: &[f64], grid: &[f64]) -> (f64, f64, f64) {
    let mut best_alpha = grid[0];
    let mut best_level = values[0];
    let mut best_sigma = f64::INFINITY;
    let mut best_sse = f64::INFINITY;
    for &alpha in grid {
        let sse = compute_simple_sse(values, alpha);
        // Strict-less-than: equal-SSE ties keep the lower-index candidate.
        if sse < best_sse {
            best_sse = sse;
            best_alpha = alpha;
            let (l, s) = fit_simple_with_alpha(values, alpha);
            best_level = l;
            best_sigma = s;
        }
    }
    (best_alpha, best_level, best_sigma)
}

/// Compute the in-sample one-step-ahead SSE for Simple ETS with given α.
fn compute_simple_sse(values: &[f64], alpha: f64) -> f64 {
    let mut level = values[0];
    let mut sse = KahanAccumulatorF64::new();
    for t in 1..values.len() {
        let err = values[t] - level;
        sse.add(err * err);
        level = alpha * values[t] + (1.0 - alpha) * level;
    }
    sse.finalize()
}

/// Fit Holt with the given (α, β). Returns `(level_T, trend_T, residual_sd)`.
pub(crate) fn fit_holt_with_params(
    values: &[f64],
    alpha: f64,
    beta: f64,
) -> (f64, f64, f64) {
    let n = values.len();
    let mut level = values[0];
    let mut trend = values[1] - values[0];
    let mut sse = KahanAccumulatorF64::new();
    let mut k: u64 = 0;
    // Skip t = 0 (initialization). Start one-step-ahead errors from t = 1
    // using the projected level + trend from the initial state, which
    // matches the Hyndman-Athanasopoulos textbook recursion.
    for t in 1..n {
        let projected = level + trend;
        let err = values[t] - projected;
        sse.add(err * err);
        k += 1;
        let new_level = alpha * values[t] + (1.0 - alpha) * projected;
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        level = new_level;
        trend = new_trend;
    }
    let sigma = if k > 0 {
        (sse.finalize() / k as f64).sqrt()
    } else {
        0.0
    };
    (level, trend, sigma)
}

/// Grid-search Holt. Returns `(best_alpha, best_beta, level_T, trend_T, residual_sd)`.
pub(crate) fn grid_search_holt(
    values: &[f64],
    alpha_grid: &[f64],
    beta_grid: &[f64],
) -> (f64, f64, f64, f64, f64) {
    let mut best_alpha = alpha_grid[0];
    let mut best_beta = beta_grid[0];
    let mut best_level = values[0];
    let mut best_trend = values[1] - values[0];
    let mut best_sigma = f64::INFINITY;
    let mut best_sse = f64::INFINITY;
    for &alpha in alpha_grid {
        for &beta in beta_grid {
            let sse = compute_holt_sse(values, alpha, beta);
            if sse < best_sse {
                best_sse = sse;
                best_alpha = alpha;
                best_beta = beta;
                let (l, b, s) = fit_holt_with_params(values, alpha, beta);
                best_level = l;
                best_trend = b;
                best_sigma = s;
            }
        }
    }
    (best_alpha, best_beta, best_level, best_trend, best_sigma)
}

/// Compute the in-sample one-step-ahead SSE for Holt with given (α, β).
fn compute_holt_sse(values: &[f64], alpha: f64, beta: f64) -> f64 {
    let mut level = values[0];
    let mut trend = values[1] - values[0];
    let mut sse = KahanAccumulatorF64::new();
    for t in 1..values.len() {
        let projected = level + trend;
        let err = values[t] - projected;
        sse.add(err * err);
        let new_level = alpha * values[t] + (1.0 - alpha) * projected;
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        level = new_level;
        trend = new_trend;
    }
    sse.finalize()
}

/// Produce point + lower + upper bound for Simple ETS forecast.
///
/// `ŷ_{n+h} = l_T` (constant); bound width grows as `sqrt(h)` (simplified).
fn forecast_simple(
    level_t: f64,
    sigma: f64,
    horizon: usize,
    cl: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z = normal_quantile_two_sided(cl);
    let mut point = Vec::with_capacity(horizon);
    let mut lower = Vec::with_capacity(horizon);
    let mut upper = Vec::with_capacity(horizon);
    for h in 1..=horizon {
        let band = z * sigma * (h as f64).sqrt();
        point.push(level_t);
        lower.push(level_t - band);
        upper.push(level_t + band);
    }
    (point, lower, upper)
}

/// Produce point + lower + upper bound for Holt forecast.
///
/// `ŷ_{n+h} = l_T + h·b_T`; bound width uses simplified `sqrt(h)` scaling.
fn forecast_holt(
    level_t: f64,
    trend_t: f64,
    sigma: f64,
    horizon: usize,
    cl: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z = normal_quantile_two_sided(cl);
    let mut point = Vec::with_capacity(horizon);
    let mut lower = Vec::with_capacity(horizon);
    let mut upper = Vec::with_capacity(horizon);
    for h in 1..=horizon {
        let p = level_t + (h as f64) * trend_t;
        let band = z * sigma * (h as f64).sqrt();
        point.push(p);
        lower.push(p - band);
        upper.push(p + band);
    }
    (point, lower, upper)
}

// ---------------------------------------------------------------------------
// Acklam (2003) inverse normal quantile.
//
// Inlined here rather than depending on cjc-causal's `iv_regression::
// normal_quantile_two_sided` — that would create a cross-domain dep
// (cjc-cronos → cjc-causal) for one scalar function. Future work: extract
// to `cjc_runtime::distributions::normal_quantile_two_sided` so all three
// decision-layer crates share one source-of-truth.
// ---------------------------------------------------------------------------

/// Two-sided normal quantile for confidence level `cl ∈ (0, 1)`.
///
/// Returns `Φ⁻¹(1 - (1 - cl)/2)`. Accuracy ~1e-9 relative error
/// (Acklam 2003).
pub(crate) fn normal_quantile_two_sided(cl: f64) -> f64 {
    let p = 1.0 - (1.0 - cl) / 2.0;
    acklam_normal_quantile(p)
}

fn acklam_normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
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
    fn simple_ets_on_constant_series_predicts_constant() {
        let ts = make_ts(vec![5.0; 30]);
        let f = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 3).unwrap();
        for &p in &f.point_estimates {
            assert!((p - 5.0).abs() < 1e-9, "got {}", p);
        }
    }

    #[test]
    fn holt_ets_on_linear_series_predicts_linear() {
        // y_t = 2 + 0.5·t for t = 0..29 → trend = 0.5
        let values: Vec<f64> = (0..30).map(|i| 2.0 + 0.5 * i as f64).collect();
        let ts = make_ts(values);
        let f = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 5).unwrap();
        // ŷ_{n+h} = l_n + h·b_n ≈ y_{29} + h·0.5 = (2 + 14.5) + h·0.5 = 16.5 + 0.5·h
        for (h_minus_1, &p) in f.point_estimates.iter().enumerate() {
            let h = (h_minus_1 + 1) as f64;
            let expected = 16.5 + 0.5 * h;
            assert!(
                (p - expected).abs() < 0.5,
                "h={}: expected {}, got {}",
                h, expected, p
            );
        }
    }

    #[test]
    fn fit_simple_with_alpha_returns_finite_state() {
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (level, sigma) = fit_simple_with_alpha(&values, 0.5);
        assert!(level.is_finite());
        assert!(sigma >= 0.0);
    }

    #[test]
    fn fit_holt_with_params_returns_finite_state() {
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (level, trend, sigma) = fit_holt_with_params(&values, 0.5, 0.3);
        assert!(level.is_finite() && trend.is_finite());
        assert!(sigma >= 0.0);
    }

    #[test]
    fn forecast_horizon_zero_returns_unsupported() {
        let ts = make_ts(vec![1.0, 2.0, 3.0, 4.0]);
        let err = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 0).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn too_few_observations_returns_numerical_error() {
        let ts = make_ts(vec![1.0]);
        let err = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn nan_value_returns_numerical_error() {
        let ts = make_ts(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
        let err = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn invalid_confidence_level_returns_unsupported() {
        let ts = make_ts(vec![1.0; 10]);
        for bad in [-0.1, 0.0, 1.0, f64::NAN] {
            let err = Ets::new(EtsKind::Simple)
                .with_confidence_level(bad)
                .fit_and_forecast(&ts, 3)
                .unwrap_err();
            assert!(
                matches!(err, CronosError::Unsupported { .. }),
                "cl {} should be unsupported",
                bad
            );
        }
    }

    #[test]
    fn empty_alpha_grid_returns_unsupported() {
        let ts = make_ts(vec![1.0; 10]);
        let err = Ets::new(EtsKind::Simple)
            .with_alpha_grid(vec![])
            .fit_and_forecast(&ts, 3)
            .unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn alpha_out_of_range_returns_unsupported() {
        let ts = make_ts(vec![1.0; 10]);
        let err = Ets::new(EtsKind::Simple)
            .with_alpha_grid(vec![1.5])
            .fit_and_forecast(&ts, 3)
            .unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn bounds_bracket_point_estimate() {
        let ts = make_ts((0..30).map(|i| 1.0 + 0.1 * i as f64).collect());
        let f = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 5).unwrap();
        for h in 0..5 {
            assert!(f.lower_bound[h] <= f.point_estimates[h]);
            assert!(f.point_estimates[h] <= f.upper_bound[h]);
        }
    }

    #[test]
    fn bound_width_increases_with_horizon() {
        let ts = make_ts((0..30).map(|i| 1.0 + 0.1 * i as f64).collect());
        let f = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 10).unwrap();
        let w1 = f.upper_bound[0] - f.lower_bound[0];
        let w10 = f.upper_bound[9] - f.lower_bound[9];
        assert!(w10 >= w1, "h=10 bound {} should be >= h=1 bound {}", w10, w1);
    }

    #[test]
    fn fitted_model_id_is_byte_identical_across_runs() {
        let ts = make_ts((0..30).map(|i| 1.0 + 0.1 * i as f64 + (i % 3) as f64 * 0.05).collect());
        let f1 = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 3).unwrap();
        let f2 = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 3).unwrap();
        assert_eq!(f1.fitted_model_id, f2.fitted_model_id);
    }

    #[test]
    fn simple_and_holt_have_different_fitted_model_ids() {
        let ts = make_ts((0..30).map(|i| 1.0 + 0.1 * i as f64).collect());
        let f_simple = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 3).unwrap();
        let f_holt = Ets::new(EtsKind::Holt).fit_and_forecast(&ts, 3).unwrap();
        assert_ne!(f_simple.fitted_model_id, f_holt.fitted_model_id);
    }

    #[test]
    fn acklam_normal_quantile_at_95_pct_is_about_1_96() {
        let z = normal_quantile_two_sided(0.95);
        assert!((z - 1.9599639845400545).abs() < 1e-7, "got {}", z);
    }

    #[test]
    fn default_grid_has_19_values() {
        let g = default_grid();
        assert_eq!(g.len(), 19);
        assert!((g[0] - 0.05).abs() < 1e-12);
        assert!((g[18] - 0.95).abs() < 1e-12);
    }

    #[test]
    fn ets_kind_labels_are_stable() {
        assert_eq!(EtsKind::Simple.label(), "ets_simple");
        assert_eq!(EtsKind::Holt.label(), "ets_holt");
    }

    #[test]
    fn point_estimates_have_length_horizon() {
        let ts = make_ts(vec![1.0; 15]);
        let f = Ets::new(EtsKind::Simple).fit_and_forecast(&ts, 7).unwrap();
        assert_eq!(f.point_estimates.len(), 7);
        assert_eq!(f.lower_bound.len(), 7);
        assert_eq!(f.upper_bound.len(), 7);
        assert_eq!(f.horizon, 7);
    }

    #[test]
    fn simple_grid_search_on_constant_series_recovers_constant_level() {
        // On a "constant" series y = [3.0; 20], algebraically every α gives
        // SSE = 0, but the recursion l_t = α·3.0 + (1-α)·l_{t-1} drifts by
        // ~1 ULP per step because α·3.0 and (1-α)·3.0 aren't binary-exact
        // for most α values (0.05, 0.10, etc.). The tie-break may not pick
        // the FIRST grid α because there ARE no true ties — different α's
        // accumulate different drifts.
        //
        // What we can honestly assert:
        // 1. The chosen α is in the grid.
        // 2. The recovered level equals the constant within rounding tolerance.
        // 3. The residual SD is very small (noise from per-step rounding).
        let values: Vec<f64> = vec![3.0; 20];
        let grid = default_grid();
        let (alpha, level, sigma) = grid_search_simple(&values, &grid);
        assert!(grid.contains(&alpha), "α = {} should be in grid", alpha);
        assert!((level - 3.0).abs() < 1e-9, "level was {}, expected ≈ 3.0", level);
        assert!(sigma < 1e-6, "sigma was {}, expected very small", sigma);
    }

    #[test]
    fn simple_grid_search_on_zero_series_picks_first_alpha() {
        // On an EXACT-zero series, α·0 = 0 and (1-α)·0 = 0 with no drift,
        // so every α genuinely produces zero SSE. The tie-break then DOES
        // fire, and the strict-less-than rule keeps the first grid α.
        let values: Vec<f64> = vec![0.0; 20];
        let grid = default_grid();
        let (alpha, level, sigma) = grid_search_simple(&values, &grid);
        assert!((alpha - grid[0]).abs() < 1e-12, "expected first α, got {}", alpha);
        assert_eq!(level, 0.0);
        assert_eq!(sigma, 0.0);
    }
}
