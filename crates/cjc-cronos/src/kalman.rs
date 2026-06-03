//! Kalman filter + RTS smoother for the local-level state-space model.
//!
//! v0.1 ships the **local-level** model only:
//! ```text
//! observation:  y_t = α_t + ε_t,    ε_t ~ N(0, R)
//! state:        α_t = α_{t-1} + η_t, η_t ~ N(0, Q)
//! ```
//!
//! Local-linear-trend and Basic Structural Model (BSM with seasonal
//! components) are deferred to v0.2 — local-level proves the filter +
//! smoother + Joseph-form discipline; the multivariate state extensions
//! are mechanical once the scalar case is locked.
//!
//! ## Joseph form (the load-bearing determinism contract)
//!
//! The standard covariance update `P = (1 - K) · P` is *algebraically*
//! correct but loses positive-semidefiniteness under rounding for very
//! small `K` or `P`. The Joseph form `P = (1 - K)² · P + K² · R` is
//! symmetric-positive-semidefinite by construction. v0.1 uses Joseph form
//! exclusively. If `P` somehow goes non-positive, the filter returns
//! `CronosError::Numerical` (Locke `E9203`).
//!
//! Hyperparameter selection for `(R, Q)` is **caller-supplied** in v0.1;
//! MLE via the prediction-error decomposition is a v0.2 extension.

use crate::error::CronosError;
use crate::ets;
use crate::forecast::Forecast;
use crate::time_series::TimeSeries;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use cjc_repro::KahanAccumulatorF64;

/// Local-level Kalman filter + RTS smoother + forecaster.
///
/// User must supply `observation_variance` (`R`) and `state_variance` (`Q`).
#[derive(Clone, Debug)]
pub struct Kalman {
    observation_variance: f64,
    state_variance: f64,
    confidence_level: f64,
    initial_state_mean: Option<f64>,
    initial_state_var: f64,
}

impl Kalman {
    /// Construct with caller-supplied variances.
    pub fn new(observation_variance: f64, state_variance: f64) -> Self {
        Self {
            observation_variance,
            state_variance,
            confidence_level: 0.95,
            initial_state_mean: None,
            initial_state_var: 1e6, // diffuse-like prior
        }
    }

    /// Override the initial state mean. Default: first observation.
    pub fn with_initial_state_mean(mut self, m: f64) -> Self {
        self.initial_state_mean = Some(m);
        self
    }

    /// Override the initial state variance. Default: `1e6` (diffuse).
    pub fn with_initial_state_var(mut self, v: f64) -> Self {
        self.initial_state_var = v;
        self
    }

    /// Confidence level for forecast bounds. Default `0.95`.
    pub fn with_confidence_level(mut self, l: f64) -> Self {
        self.confidence_level = l;
        self
    }

    /// Run the forward Kalman filter. Returns per-step `(state_mean,
    /// state_var, innovation, innovation_var, log_likelihood_contribution)`.
    pub fn filter(&self, ts: &TimeSeries) -> Result<KalmanFilterOutput, CronosError> {
        self.validate_config()?;
        let y = ts.values();
        let n = y.len();
        if n == 0 {
            return Err(CronosError::Numerical {
                detail: "Kalman filter: empty series".to_string(),
            });
        }
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosError::Numerical {
                    detail: format!("value at row {} is non-finite ({})", i, v),
                });
            }
        }

        let mut state_means = Vec::with_capacity(n);
        let mut state_vars = Vec::with_capacity(n);
        let mut innovations = Vec::with_capacity(n);
        let mut innovation_vars = Vec::with_capacity(n);

        // Initialise.
        let m0 = self.initial_state_mean.unwrap_or(y[0]);
        let p0 = self.initial_state_var;

        let mut m_pred = m0;
        let mut p_pred = p0 + self.state_variance; // predict t=0
        let mut log_lik_acc = KahanAccumulatorF64::new();

        for t in 0..n {
            let innov = y[t] - m_pred;
            let s = p_pred + self.observation_variance; // innovation variance
            if !(s > 0.0) {
                return Err(CronosError::Numerical {
                    detail: format!(
                        "Kalman filter: innovation variance non-positive at t = {}",
                        t
                    ),
                });
            }
            let k = p_pred / s; // Kalman gain
            // Joseph form covariance update: P_new = (1 - K)^2 · P_pred + K^2 · R
            // This preserves symmetric-positive-(semi)definiteness.
            let one_minus_k = 1.0 - k;
            let p_filt = one_minus_k * one_minus_k * p_pred + k * k * self.observation_variance;
            let m_filt = m_pred + k * innov;
            state_means.push(m_filt);
            state_vars.push(p_filt);
            innovations.push(innov);
            innovation_vars.push(s);
            // Log-likelihood contribution (Gaussian).
            // ℓ = -0.5 · (log(2π) + log(s) + innov² / s)
            log_lik_acc.add(-0.5 * (std::f64::consts::TAU.ln() + s.ln() + innov * innov / s));
            // Predict next step.
            m_pred = m_filt;
            p_pred = p_filt + self.state_variance;
        }

        Ok(KalmanFilterOutput {
            state_means,
            state_vars,
            innovations,
            innovation_vars,
            log_likelihood: log_lik_acc.finalize(),
        })
    }

    /// Run the RTS (Rauch-Tung-Striebel) backward smoother.
    pub fn smooth(&self, ts: &TimeSeries) -> Result<KalmanSmootherOutput, CronosError> {
        let filtered = self.filter(ts)?;
        let n = filtered.state_means.len();
        let mut smoothed_means = filtered.state_means.clone();
        let mut smoothed_vars = filtered.state_vars.clone();
        // Backward recursion.
        for t in (0..n - 1).rev() {
            let p_filt = filtered.state_vars[t];
            // P_{t+1|t} = P_{t|t} + Q
            let p_pred_next = p_filt + self.state_variance;
            // Smoother gain.
            let g = p_filt / p_pred_next;
            // m_{t|n} = m_{t|t} + G · (m_{t+1|n} - m_{t|t})
            //   note: m_{t+1|t} = m_{t|t} for local-level (transition is identity).
            smoothed_means[t] = filtered.state_means[t] + g * (smoothed_means[t + 1] - filtered.state_means[t]);
            // P_{t|n} = P_{t|t} + G² · (P_{t+1|n} - P_{t+1|t})
            smoothed_vars[t] = p_filt + g * g * (smoothed_vars[t + 1] - p_pred_next);
        }
        Ok(KalmanSmootherOutput { smoothed_means, smoothed_vars })
    }

    /// Fit (caller-supplied variances) + smooth + forecast.
    pub fn fit_and_forecast(
        &self,
        ts: &TimeSeries,
        horizon: usize,
    ) -> Result<Forecast, CronosError> {
        if horizon == 0 {
            return Err(CronosError::Unsupported {
                detail: "horizon must be > 0".to_string(),
            });
        }
        self.validate_config()?;
        let smoothed = self.smooth(ts)?;
        let last_idx = smoothed.smoothed_means.len() - 1;
        let last_mean = smoothed.smoothed_means[last_idx];
        let last_var = smoothed.smoothed_vars[last_idx];

        let z = ets::normal_quantile_two_sided(self.confidence_level);
        let mut points = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        // For h steps ahead, state variance grows by h·Q and observation
        // variance adds another R. Variance of forecast observation:
        //   Var(ŷ_{T+h}) = P_T + h·Q + R
        for h in 1..=horizon {
            let var_obs = last_var + (h as f64) * self.state_variance + self.observation_variance;
            let band = z * var_obs.sqrt();
            points.push(last_mean);
            lower.push(last_mean - band);
            upper.push(last_mean + band);
        }

        let fitted_model_id = compute_kalman_model_id(
            self.observation_variance,
            self.state_variance,
            ts.values(),
            self.confidence_level,
        );
        Ok(Forecast {
            horizon,
            point_estimates: points,
            lower_bound: lower,
            upper_bound: upper,
            confidence_level: self.confidence_level,
            fitted_model_id,
        })
    }

    fn validate_config(&self) -> Result<(), CronosError> {
        if !self.observation_variance.is_finite() || self.observation_variance < 0.0 {
            return Err(CronosError::Unsupported {
                detail: format!(
                    "observation_variance must be >= 0 and finite, got {}",
                    self.observation_variance
                ),
            });
        }
        if !self.state_variance.is_finite() || self.state_variance < 0.0 {
            return Err(CronosError::Unsupported {
                detail: format!(
                    "state_variance must be >= 0 and finite, got {}",
                    self.state_variance
                ),
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
        if !self.initial_state_var.is_finite() || self.initial_state_var < 0.0 {
            return Err(CronosError::Unsupported {
                detail: format!("initial_state_var must be >= 0, got {}", self.initial_state_var),
            });
        }
        Ok(())
    }
}

/// Forward-filter output: per-step state means + vars + innovations.
#[derive(Clone, Debug, PartialEq)]
pub struct KalmanFilterOutput {
    pub state_means: Vec<f64>,
    pub state_vars: Vec<f64>,
    pub innovations: Vec<f64>,
    pub innovation_vars: Vec<f64>,
    pub log_likelihood: f64,
}

/// Backward-smoother output: per-step smoothed state means + vars.
#[derive(Clone, Debug, PartialEq)]
pub struct KalmanSmootherOutput {
    pub smoothed_means: Vec<f64>,
    pub smoothed_vars: Vec<f64>,
}

fn compute_kalman_model_id(
    obs_var: f64,
    state_var: f64,
    training_values: &[f64],
    confidence_level: f64,
) -> FingerprintId {
    let parts: Vec<FingerprintId> = vec![
        fingerprint_str(IdDomain::CausalClaim, "kalman_local_level"),
        fingerprint(IdDomain::CausalClaim, &obs_var.to_bits().to_le_bytes()),
        fingerprint(IdDomain::CausalClaim, &state_var.to_bits().to_le_bytes()),
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
    fingerprint_compose(IdDomain::CausalClaim, "kalman_model", &parts)
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
    fn filter_on_constant_series_recovers_constant() {
        let ts = make_ts(vec![5.0; 20]);
        let out = Kalman::new(1.0, 0.01).filter(&ts).unwrap();
        // After a few steps the filtered state should sit very close to 5.0.
        assert!((out.state_means[19] - 5.0).abs() < 0.5);
        // All state variances should be finite and non-negative.
        for v in &out.state_vars {
            assert!(*v >= 0.0 && v.is_finite());
        }
    }

    #[test]
    fn smoother_matches_filter_at_last_observation() {
        let ts = make_ts((0..20).map(|i| 1.0 + 0.1 * i as f64).collect());
        let filt = Kalman::new(0.5, 0.1).filter(&ts).unwrap();
        let smoothed = Kalman::new(0.5, 0.1).smooth(&ts).unwrap();
        // Smoothed value at t = N-1 equals filtered value at t = N-1.
        let last = filt.state_means.len() - 1;
        assert_eq!(
            filt.state_means[last].to_bits(),
            smoothed.smoothed_means[last].to_bits()
        );
    }

    #[test]
    fn forecast_horizon_zero_returns_unsupported() {
        let ts = make_ts(vec![1.0; 10]);
        let err = Kalman::new(0.5, 0.01).fit_and_forecast(&ts, 0).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
    }

    #[test]
    fn negative_variances_return_unsupported() {
        let ts = make_ts(vec![1.0; 10]);
        let err = Kalman::new(-0.5, 0.01).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err, CronosError::Unsupported { .. }));
        let err2 = Kalman::new(0.5, -0.01).fit_and_forecast(&ts, 3).unwrap_err();
        assert!(matches!(err2, CronosError::Unsupported { .. }));
    }

    #[test]
    fn nan_value_returns_numerical_error() {
        let ts = make_ts(vec![1.0, 2.0, f64::NAN, 4.0]);
        let err = Kalman::new(0.5, 0.01).filter(&ts).unwrap_err();
        assert!(matches!(err, CronosError::Numerical { .. }));
    }

    #[test]
    fn fitted_model_id_is_byte_identical_across_runs() {
        let ts = make_ts((0..20).map(|i| 1.0 + 0.1 * i as f64).collect());
        let k = Kalman::new(0.5, 0.01);
        let f1 = k.fit_and_forecast(&ts, 5).unwrap();
        let f2 = k.fit_and_forecast(&ts, 5).unwrap();
        assert_eq!(f1.fitted_model_id, f2.fitted_model_id);
        for h in 0..5 {
            assert_eq!(f1.point_estimates[h].to_bits(), f2.point_estimates[h].to_bits());
        }
    }

    #[test]
    fn different_state_variance_gives_different_model_id() {
        let ts = make_ts((0..20).map(|i| 1.0 + 0.1 * i as f64).collect());
        let f1 = Kalman::new(0.5, 0.01).fit_and_forecast(&ts, 3).unwrap();
        let f2 = Kalman::new(0.5, 0.02).fit_and_forecast(&ts, 3).unwrap();
        assert_ne!(f1.fitted_model_id, f2.fitted_model_id);
    }

    #[test]
    fn forecast_bounds_widen_with_horizon() {
        let ts = make_ts((0..20).map(|i| 1.0 + 0.1 * i as f64).collect());
        let f = Kalman::new(0.5, 0.01).fit_and_forecast(&ts, 10).unwrap();
        for h in 1..10 {
            let w_prev = f.upper_bound[h - 1] - f.lower_bound[h - 1];
            let w_curr = f.upper_bound[h] - f.lower_bound[h];
            assert!(w_curr >= w_prev);
        }
    }

    #[test]
    fn log_likelihood_is_finite() {
        let ts = make_ts((0..20).map(|i| 1.0 + 0.1 * i as f64).collect());
        let out = Kalman::new(0.5, 0.01).filter(&ts).unwrap();
        assert!(out.log_likelihood.is_finite());
    }

    #[test]
    fn joseph_form_preserves_psd_under_extreme_ratio() {
        // Very small observation variance relative to state — high Kalman
        // gain near 1. Standard form (1-K)·P would drift to non-PSD; Joseph
        // form should stay non-negative.
        let ts = make_ts((0..20).map(|i| 1.0 + 0.1 * i as f64).collect());
        let out = Kalman::new(1e-10, 1.0).filter(&ts).unwrap();
        for v in &out.state_vars {
            assert!(*v >= 0.0, "state var went negative: {}", v);
        }
    }
}
