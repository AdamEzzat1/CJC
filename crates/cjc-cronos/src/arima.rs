//! ARIMA(p, d, q) forecaster — hand-specified order, deterministic estimation.
//!
//! v0.1 ships:
//! - **AR(p)** via Yule-Walker (Levinson-Durbin recursion, O(p²)).
//! - **MA(q)** via Hannan-Rissanen two-step (long-AR residuals → joint OLS
//!   over y on AR + MA lags).
//! - **Differencing** for integration order `d ∈ {0, 1, 2}`.
//! - **Forecast** via iterated state-space rollout + Kahan-summed variance
//!   accumulation for bounds.
//!
//! ## Deferred to v0.2
//!
//! - **CSS MLE refinement** — current AR coefficients come straight from
//!   Levinson-Durbin (asymptotically efficient under Gaussian innovations).
//!   Conditional sum-of-squares maximum-likelihood refinement to tighten the
//!   small-sample estimates is documented but not implemented in v0.1.
//! - **auto_arima** — order selection via AIC grid is its own can of worms.
//! - **Seasonal AR / MA** (SARIMA) — separate `Sarima` struct, v0.2.
//!
//! ## Determinism contract
//!
//! - All reductions Kahan-summed (autocovariances, residuals, OLS, forecast SE).
//! - Yule-Walker uses Levinson-Durbin (closed-form, no iterative tolerance).
//! - Hannan-Rissanen reuses `cjc_runtime::hypothesis::lm` (already-audited
//!   Kahan-summed QR).
//! - No randomness anywhere in fit or forecast.
//! - Content-addressed [`Forecast::fitted_model_id`] hashes `(p, d, q,
//!   training values, confidence_level)`.

use crate::error::CronosError;
use crate::ets;
use crate::forecast::Forecast;
use crate::time_series::TimeSeries;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use cjc_repro::KahanAccumulatorF64;

/// ARIMA(p, d, q) forecaster.
#[derive(Clone, Debug)]
pub struct Arima {
    p: usize,
    d: usize,
    q: usize,
    confidence_level: f64,
}

impl Arima {
    /// Construct an ARIMA(p, d, q) forecaster.
    ///
    /// Defaults: `confidence_level = 0.95`.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q, confidence_level: 0.95 }
    }

    /// Confidence level in `(0, 1)` for forecast bounds.
    pub fn with_confidence_level(mut self, l: f64) -> Self {
        self.confidence_level = l;
        self
    }

    /// Fit the ARIMA model and produce an `horizon`-step forecast.
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
        if self.d > 2 {
            return Err(CronosError::Unsupported {
                detail: format!("d > 2 not supported in v0.1, got d = {}", self.d),
            });
        }
        if self.p == 0 && self.q == 0 && self.d == 0 {
            return Err(CronosError::Unsupported {
                detail: "ARIMA(0, 0, 0) is just the mean — use a constant series instead".to_string(),
            });
        }
        // Need enough observations: differencing eats d, fitting eats p + q.
        // Heuristic: n > 2 · (p + q) + 4 · d + 10.
        let min_n = 2 * (self.p + self.q) + 4 * self.d + 10;
        let raw_values = ts.values();
        let n_raw = raw_values.len();
        if n_raw < min_n {
            return Err(CronosError::Numerical {
                detail: format!(
                    "ARIMA({},{},{}) needs at least {} observations, got {}",
                    self.p, self.d, self.q, min_n, n_raw
                ),
            });
        }
        for (i, v) in raw_values.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosError::Numerical {
                    detail: format!("value at row {} is non-finite ({})", i, v),
                });
            }
        }

        // 1. Difference d times.
        let working = difference(raw_values, self.d);
        let n = working.len();

        // 2. Demean for stationarity.
        let mean = kahan_mean(&working);
        let centered: Vec<f64> = working.iter().map(|v| v - mean).collect();

        // Detect degenerate differenced series (variance = 0). This happens
        // when the original series has a deterministic polynomial trend of
        // order ≤ d; after differencing, the result is a constant. In that
        // case AR/MA fitting is meaningless — every coefficient produces
        // the same forecast (the mean). Return zero coefficients and let the
        // mean-only forecast path handle it downstream.
        let centered_variance: f64 = {
            let mut acc = KahanAccumulatorF64::new();
            for v in &centered {
                acc.add(v * v);
            }
            acc.finalize() / centered.len() as f64
        };
        let degenerate = centered_variance < f64::EPSILON;

        // 3. AR coefficients via Yule-Walker / Levinson-Durbin.
        let phi: Vec<f64> = if self.p > 0 && !degenerate {
            yule_walker(&centered, self.p)?
        } else {
            vec![0.0; self.p]
        };

        // 4. MA coefficients via Hannan-Rissanen (only if q > 0).
        let theta: Vec<f64> = if self.q > 0 && !degenerate {
            hannan_rissanen_ma(&centered, self.p, self.q)?
        } else {
            vec![0.0; self.q]
        };

        // 5. In-sample residuals for variance estimation.
        let residuals = compute_residuals(&centered, &phi, &theta);
        let sigma_sq = kahan_variance_unbiased(&residuals, phi.len() + theta.len() + 1)?;
        let sigma = sigma_sq.sqrt();

        // 6. Iterative forecast on the *differenced* scale.
        let diff_forecast = iterative_forecast(&centered, &phi, &theta, &residuals, horizon);

        // 7. Undo differencing to get forecast on the original scale.
        let undiff_forecast = undifference(raw_values, self.d, &diff_forecast, mean);

        // 8. Bounds via normal-approximation. Variance grows roughly with
        // accumulated MA(∞) representation. v0.1 ships the simplified
        // sqrt(h) scaling for AR-only; full MA(∞) expansion is a v0.2 refinement.
        let z = ets::normal_quantile_two_sided(self.confidence_level);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        // For differenced d > 0 series, the forecast SE on the undifferenced
        // scale grows roughly as sigma · h^{0.5 + d/2}.
        let var_exp = 0.5 + 0.5 * self.d as f64;
        for h in 1..=horizon {
            let band = z * sigma * (h as f64).powf(var_exp);
            lower.push(undiff_forecast[h - 1] - band);
            upper.push(undiff_forecast[h - 1] + band);
        }

        let fitted_model_id = compute_arima_model_id(
            self.p,
            self.d,
            self.q,
            raw_values,
            self.confidence_level,
        );

        Ok(Forecast {
            horizon,
            point_estimates: undiff_forecast,
            lower_bound: lower,
            upper_bound: upper,
            confidence_level: self.confidence_level,
            fitted_model_id,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn kahan_mean(values: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for v in values {
        acc.add(*v);
    }
    acc.finalize() / values.len() as f64
}

fn kahan_variance_unbiased(values: &[f64], df_consumed: usize) -> Result<f64, CronosError> {
    let n = values.len();
    if n <= df_consumed {
        return Err(CronosError::Numerical {
            detail: format!(
                "variance: n = {} <= df_consumed = {}",
                n, df_consumed
            ),
        });
    }
    let m = kahan_mean(values);
    let mut acc = KahanAccumulatorF64::new();
    for v in values {
        let d = v - m;
        acc.add(d * d);
    }
    Ok(acc.finalize() / (n - df_consumed) as f64)
}

/// Apply d successive first-differences to `values`. d=0 returns a clone.
pub(crate) fn difference(values: &[f64], d: usize) -> Vec<f64> {
    let mut current: Vec<f64> = values.to_vec();
    for _ in 0..d {
        let mut next = Vec::with_capacity(current.len().saturating_sub(1));
        for i in 1..current.len() {
            next.push(current[i] - current[i - 1]);
        }
        current = next;
    }
    current
}

/// Undo d successive differences. `last_levels` are the d last raw values
/// before forecasting (needed as initial conditions for cumulative sum).
fn undifference(
    raw_values: &[f64],
    d: usize,
    diff_forecast: &[f64],
    differenced_mean: f64,
) -> Vec<f64> {
    if d == 0 {
        // d=0: differenced series IS the original (centered around `differenced_mean`).
        // Re-add the mean.
        return diff_forecast.iter().map(|v| v + differenced_mean).collect();
    }
    // For d=1: forecast on first-difference scale; cumulative sum + last raw value.
    // For d=2: integrate twice.
    let mut current: Vec<f64> = diff_forecast.iter().map(|v| v + differenced_mean).collect();
    // current is forecasted differences (d-th). We need to undo (d-1, d-2, ..., 0).
    // Each step prepends the last "raw" value of the previous level.
    let mut intermediate_history: Vec<Vec<f64>> = Vec::with_capacity(d);
    let mut chain = raw_values.to_vec();
    for _ in 0..d {
        let next = (1..chain.len()).map(|i| chain[i] - chain[i - 1]).collect();
        intermediate_history.push(chain);
        chain = next;
    }
    // intermediate_history[i] is the i-th differenced series; the d-th
    // difference is the input to our forecast. To recover y_{n+1..} on the
    // original scale, we cumsum back through each level. At each level, the
    // last value of the previous (lower-order) series is the starting point.
    for level in (0..d).rev() {
        let starting_point = *intermediate_history[level].last().unwrap();
        let mut undone = Vec::with_capacity(current.len());
        let mut running = starting_point;
        for &dv in &current {
            running += dv;
            undone.push(running);
        }
        current = undone;
    }
    current
}

/// Solve the Yule-Walker equations via Levinson-Durbin (O(p²)).
///
/// Returns AR coefficients `[φ_1, …, φ_p]`.
fn yule_walker(centered: &[f64], p: usize) -> Result<Vec<f64>, CronosError> {
    let gamma = autocovariances(centered, p);
    if gamma[0] <= 0.0 {
        return Err(CronosError::Numerical {
            detail: format!("Yule-Walker: γ(0) = {} ≤ 0 — series has no variance", gamma[0]),
        });
    }
    let mut phi = vec![0.0; p];
    let mut sigma_sq = gamma[0];
    phi[0] = gamma[1] / gamma[0];
    sigma_sq *= 1.0 - phi[0] * phi[0];
    for k in 1..p {
        let mut num = KahanAccumulatorF64::new();
        num.add(gamma[k + 1]);
        for j in 0..k {
            num.add(-phi[j] * gamma[k - j]);
        }
        let phi_kk = num.finalize() / sigma_sq;
        let phi_prev: Vec<f64> = phi[..k].to_vec();
        for j in 0..k {
            phi[j] = phi_prev[j] - phi_kk * phi_prev[k - 1 - j];
        }
        phi[k] = phi_kk;
        sigma_sq *= 1.0 - phi_kk * phi_kk;
        if sigma_sq <= 0.0 {
            return Err(CronosError::Numerical {
                detail: format!(
                    "Yule-Walker: Levinson-Durbin σ² became non-positive at step {}",
                    k
                ),
            });
        }
    }
    Ok(phi)
}

/// Compute biased autocovariances γ(0..=max_lag).
fn autocovariances(centered: &[f64], max_lag: usize) -> Vec<f64> {
    let n = centered.len();
    let mut g = vec![0.0; max_lag + 1];
    for k in 0..=max_lag {
        let mut acc = KahanAccumulatorF64::new();
        for t in 0..(n - k) {
            acc.add(centered[t] * centered[t + k]);
        }
        g[k] = acc.finalize() / n as f64;
    }
    g
}

/// Hannan-Rissanen two-step MA estimation.
///
/// 1. Fit a long-AR(L) (L = max(2·(p+q)+1, 20)) to get residuals e_t.
/// 2. Build regressors `[y_{t-1}, …, y_{t-p}, e_{t-1}, …, e_{t-q}]`.
/// 3. Run OLS via `cjc_runtime::hypothesis::lm`, extract θ from the last q
///    coefficients.
fn hannan_rissanen_ma(centered: &[f64], p: usize, q: usize) -> Result<Vec<f64>, CronosError> {
    let l = (2 * (p + q) + 1).max(20);
    let n = centered.len();
    if n < l + p + q + 10 {
        return Err(CronosError::Numerical {
            detail: format!(
                "Hannan-Rissanen: n = {} too small for long-AR(L={}) + p={} + q={}",
                n, l, p, q
            ),
        });
    }
    // Step 1: long-AR via Yule-Walker.
    let long_phi = yule_walker(centered, l)?;
    // Compute residuals e_t = y_t - Σ φ_i · y_{t-i} for t ≥ l.
    let mut residuals = vec![0.0; n];
    for t in l..n {
        let mut acc = KahanAccumulatorF64::new();
        acc.add(centered[t]);
        for i in 0..l {
            acc.add(-long_phi[i] * centered[t - 1 - i]);
        }
        residuals[t] = acc.finalize();
    }
    // Step 2: build the augmented design matrix.
    // Rows start at t = max(l, max(p, q)).
    let start = l.max(p).max(q);
    let rows = n - start;
    let cols = p + q;
    if rows < cols + 5 {
        return Err(CronosError::Numerical {
            detail: format!(
                "Hannan-Rissanen: only {} usable rows for {} regressors",
                rows, cols
            ),
        });
    }
    let mut x_flat = Vec::with_capacity(rows * cols);
    let mut y = Vec::with_capacity(rows);
    for t in start..n {
        for i in 0..p {
            x_flat.push(centered[t - 1 - i]);
        }
        for j in 0..q {
            x_flat.push(residuals[t - 1 - j]);
        }
        y.push(centered[t]);
    }
    let fit = cjc_runtime::hypothesis::lm(&x_flat, &y, rows, cols)
        .map_err(|e| CronosError::Numerical {
            detail: format!("Hannan-Rissanen OLS failed: {}", e),
        })?;
    // fit.coefficients = [intercept, φ_1, …, φ_p, θ_1, …, θ_q]
    Ok(fit.coefficients[(p + 1)..(p + q + 1)].to_vec())
}

/// Compute in-sample one-step-ahead residuals e_t = y_t - ŷ_t.
fn compute_residuals(centered: &[f64], phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let n = centered.len();
    let p = phi.len();
    let q = theta.len();
    let mut residuals = vec![0.0; n];
    let start = p.max(q);
    for t in start..n {
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..p {
            acc.add(phi[i] * centered[t - 1 - i]);
        }
        for j in 0..q {
            acc.add(theta[j] * residuals[t - 1 - j]);
        }
        let predicted = acc.finalize();
        residuals[t] = centered[t] - predicted;
    }
    residuals
}

/// Iterated forecast on the centered/differenced scale.
fn iterative_forecast(
    centered: &[f64],
    phi: &[f64],
    theta: &[f64],
    residuals: &[f64],
    horizon: usize,
) -> Vec<f64> {
    let n = centered.len();
    let p = phi.len();
    let q = theta.len();
    // Build a window of recent observations + residuals for the rollout.
    let mut history: Vec<f64> = centered.to_vec();
    let mut res_history: Vec<f64> = residuals.to_vec();
    let mut forecast = Vec::with_capacity(horizon);
    for _h in 0..horizon {
        let t = history.len();
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..p {
            if t > i {
                acc.add(phi[i] * history[t - 1 - i]);
            }
        }
        for j in 0..q {
            // Forecasted future residuals are 0 (their expectation).
            // Only past residuals contribute.
            if t - 1 - j < n {
                acc.add(theta[j] * res_history[t - 1 - j]);
            }
        }
        let yhat = acc.finalize();
        history.push(yhat);
        // Future residual expectation is 0.
        res_history.push(0.0);
        forecast.push(yhat);
    }
    forecast
}

/// Content-addressed identifier for an ARIMA fit.
fn compute_arima_model_id(
    p: usize,
    d: usize,
    q: usize,
    training_values: &[f64],
    confidence_level: f64,
) -> FingerprintId {
    let parts: Vec<FingerprintId> = vec![
        fingerprint_str(IdDomain::CausalClaim, "arima"),
        fingerprint(IdDomain::CausalClaim, &(p as u64).to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &(d as u64).to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &(q as u64).to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &(training_values.len() as u64).to_le_bytes()),
        {
            let mut value_bytes: Vec<u8> = Vec::with_capacity(training_values.len() * 8);
            for v in training_values {
                value_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            fingerprint(IdDomain::CausalClaim, &value_bytes)
        },
        fingerprint(IdDomain::CausalClaim, &confidence_level.to_bits().to_le_bytes()),
    ];
    fingerprint_compose(IdDomain::CausalClaim, "arima_model", &parts)
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
    fn difference_d_zero_preserves_input() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(difference(&v, 0), v);
    }

    #[test]
    fn difference_d_one_first_diff() {
        let v = vec![1.0, 3.0, 6.0, 10.0];
        let d1 = difference(&v, 1);
        assert_eq!(d1, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn difference_d_two_second_diff() {
        let v = vec![1.0, 3.0, 6.0, 10.0];
        let d2 = difference(&v, 2);
        // d1 = [2, 3, 4]; d2 = [1, 1]
        assert_eq!(d2, vec![1.0, 1.0]);
    }

    #[test]
    fn autocovariances_lag_zero_is_variance_like() {
        let c: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]; // sum = 0
        let g = autocovariances(&c, 2);
        // γ(0) = Σ c_i² / n = (4+1+0+1+4)/5 = 2.0
        assert!((g[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn yule_walker_on_ar1_recovers_phi() {
        // Generate AR(1) y_t = 0.7·y_{t-1} + ε with known φ.
        let mut rng = cjc_repro::Rng::seeded(42);
        let n = 500;
        let phi_true = 0.7;
        let mut y = vec![0.0; n];
        for t in 1..n {
            let eps = (rng.next_f64() - 0.5) * 0.5;
            y[t] = phi_true * y[t - 1] + eps;
        }
        let m = kahan_mean(&y);
        let centered: Vec<f64> = y.iter().map(|v| v - m).collect();
        let phi = yule_walker(&centered, 1).unwrap();
        assert!(
            (phi[0] - phi_true).abs() < 0.1,
            "expected φ ≈ 0.7, got {}",
            phi[0]
        );
    }

    #[test]
    fn arima_on_constant_series_after_differencing_predicts_constant() {
        // Constant linear-trend series: y_t = 5 + 0.5·t. d=1 should give
        // a near-constant differenced series of 0.5. ARIMA(0, 1, 0) is the
        // random walk + drift; with d=1 the forecast continues the trend.
        let values: Vec<f64> = (0..40).map(|i| 5.0 + 0.5 * i as f64).collect();
        let ts = make_ts(values);
        let arima = Arima::new(1, 1, 0); // AR(1) + d=1
        let f = arima.fit_and_forecast(&ts, 5).unwrap();
        let last = 5.0 + 0.5 * 39.0;
        // First-difference forecast: roughly 0.5 each step; undifferenced
        // forecast should be approximately last + 0.5, last + 1.0, etc.
        for h in 0..5 {
            let expected = last + 0.5 * (h as f64 + 1.0);
            assert!(
                (f.point_estimates[h] - expected).abs() < 2.0,
                "h = {}: expected ≈ {}, got {}",
                h + 1, expected, f.point_estimates[h]
            );
        }
    }

    #[test]
    fn arima_horizon_zero_returns_unsupported() {
        let ts = make_ts((0..30).map(|i| i as f64).collect());
        let err = Arima::new(1, 0, 0).fit_and_forecast(&ts, 0).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn arima_d_greater_than_2_returns_unsupported() {
        let ts = make_ts((0..30).map(|i| i as f64).collect());
        let err = Arima::new(1, 3, 0).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn arima_p_q_d_all_zero_returns_unsupported() {
        let ts = make_ts((0..30).map(|i| i as f64).collect());
        let err = Arima::new(0, 0, 0).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn arima_short_series_returns_numerical_error() {
        let ts = make_ts(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let err = Arima::new(2, 0, 1).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn arima_nan_value_returns_numerical_error() {
        let mut v: Vec<f64> = (0..40).map(|i| i as f64).collect();
        v[10] = f64::NAN;
        let ts = make_ts(v);
        let err = Arima::new(1, 0, 0).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn arima_fitted_model_id_is_byte_identical_across_runs() {
        let mut rng = cjc_repro::Rng::seeded(7);
        let values: Vec<f64> = (0..80).map(|_| (rng.next_f64() - 0.5) * 2.0).collect();
        let ts = make_ts(values);
        let arima = Arima::new(2, 0, 1);
        let f1 = arima.fit_and_forecast(&ts, 5).unwrap();
        let f2 = arima.fit_and_forecast(&ts, 5).unwrap();
        assert_eq!(f1.fitted_model_id, f2.fitted_model_id);
        for h in 0..5 {
            assert_eq!(f1.point_estimates[h].to_bits(), f2.point_estimates[h].to_bits());
        }
    }

    #[test]
    fn arima_different_order_gives_different_model_id() {
        let mut rng = cjc_repro::Rng::seeded(11);
        let values: Vec<f64> = (0..80).map(|_| (rng.next_f64() - 0.5) * 2.0).collect();
        let ts = make_ts(values);
        let f_ar1 = Arima::new(1, 0, 0).fit_and_forecast(&ts, 3).unwrap();
        let f_ar2 = Arima::new(2, 0, 0).fit_and_forecast(&ts, 3).unwrap();
        assert_ne!(f_ar1.fitted_model_id, f_ar2.fitted_model_id);
    }

    #[test]
    fn arima_bounds_widen_with_horizon() {
        let mut rng = cjc_repro::Rng::seeded(13);
        let values: Vec<f64> = (0..80).map(|i| 0.1 * i as f64 + (rng.next_f64() - 0.5) * 0.5).collect();
        let ts = make_ts(values);
        let f = Arima::new(1, 0, 0).fit_and_forecast(&ts, 10).unwrap();
        let w1 = f.upper_bound[0] - f.lower_bound[0];
        let w10 = f.upper_bound[9] - f.lower_bound[9];
        assert!(w10 > w1);
    }

    #[test]
    fn arima_ar1_recovers_phi_in_fit_phase() {
        // Direct test of the yule_walker helper on AR(1) data.
        let mut rng = cjc_repro::Rng::seeded(17);
        let n = 300;
        let phi_true = 0.5;
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = phi_true * y[t - 1] + (rng.next_f64() - 0.5);
        }
        let m = kahan_mean(&y);
        let centered: Vec<f64> = y.iter().map(|v| v - m).collect();
        let phi = yule_walker(&centered, 1).unwrap();
        assert!((phi[0] - phi_true).abs() < 0.15);
    }
}
