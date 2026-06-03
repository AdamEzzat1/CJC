//! Double machine learning (DML) for the partially linear model.
//!
//! See ADR-0043 for the full design rationale. v0.1 ships **linear nuisances
//! via OLS** rather than MLP nuisances; the orthogonal-moment + K-fold
//! cross-fitting discipline is what makes this DML, and that part is fully
//! present.
//!
//! ## Pipeline
//!
//! 1. Validate config + columns + dimensions.
//! 2. Locke refusal check (treatment, outcome, covariates).
//! 3. K-fold split via [`cjc_runtime::ml::kfold_indices`] with the
//!    caller-supplied seed (deterministic Fisher-Yates).
//! 4. For each fold k = 0..K:
//!    a. Subset training rows: `(X_train, Y_train, T_train)`.
//!    b. Subset held-out rows: `X_test`.
//!    c. Fit OLS on `(X_train, Y_train)` → predict `ŷ_k` on `X_test`.
//!    d. Fit OLS on `(X_train, T_train)` → predict `t̂_k` on `X_test`.
//!    e. Stitch predictions into the global `ŷ` and `t̂` vectors at the
//!       row indices the held-out fold occupied.
//! 5. Compute the Chernozhukov 2018 orthogonal moment via
//!    [`super::orthogonal_moment::partial_linear_score`]:
//!    `β̂ = Σ (T - t̂)(Y - ŷ) / Σ (T - t̂)²`
//! 6. CI via the Acklam normal quantile from `iv_regression`.
//! 7. Identifier with `estimator_label = "double_ml_partial_linear"` and
//!    `instrument = None`.

use crate::assumption::IdentificationAssumption;
use crate::content_hash::compute_identifier;
use crate::error::CausalError;
use crate::estimate::EffectEstimate;
use crate::nuisance::fit_linear_predict;
use crate::orthogonal_moment::partial_linear_score;
use crate::propensity_score::extract_numeric_column;
use crate::refusal::check_locke_refusal;
use cjc_data::DataFrame;
use cjc_locke::LockeReport;

/// Estimator-label string used in the content-addressed [`FingerprintId`].
/// Changing this value would silently change every identifier — do not edit.
pub const ESTIMATOR_LABEL: &str = "double_ml_partial_linear";

/// Default number of cross-fitting folds.
pub const DEFAULT_K_FOLDS: usize = 5;

/// Double machine learning for the partially linear model
/// `y = β·T + g(X) + ε` with cross-fitted OLS nuisances.
///
/// # Defaults
///
/// | Knob | Default | Source |
/// |---|---|---|
/// | `k_folds` | `5` | DML convention |
/// | `seed` | `0` | — |
/// | `confidence_level` | `0.95` | convention |
#[derive(Clone, Debug)]
pub struct DoubleMLEstimator {
    k_folds: usize,
    seed: u64,
    confidence_level: f64,
}

impl Default for DoubleMLEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl DoubleMLEstimator {
    /// Construct with v0.1 defaults (K = 5, seed = 0, CL = 0.95).
    pub fn new() -> Self {
        Self {
            k_folds: DEFAULT_K_FOLDS,
            seed: 0,
            confidence_level: 0.95,
        }
    }

    /// Number of folds for cross-fitting. Must be `>= 2` (otherwise the
    /// "test fold" is empty and the cross-fit predictions can't be computed).
    pub fn with_k_folds(mut self, k: usize) -> Self {
        self.k_folds = k;
        self
    }

    /// Caller-supplied seed for the Fisher-Yates shuffle that drives the
    /// K-fold split. Part of the [`EffectEstimate::identifier`] content hash.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Confidence level for the CI, e.g. `0.95`. Must be in `(0, 1)`.
    pub fn with_confidence_level(mut self, l: f64) -> Self {
        self.confidence_level = l;
        self
    }

    /// Estimate the partially linear treatment effect via DML.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        df: &DataFrame,
        treatment: &str,
        outcome: &str,
        covariates: &[&str],
        assumptions: &[IdentificationAssumption],
        locke_report: &LockeReport,
    ) -> Result<EffectEstimate, CausalError> {
        // 1. Config validation.
        if self.k_folds < 2 {
            return Err(CausalError::Unsupported {
                detail: format!("k_folds must be >= 2, got {}", self.k_folds),
            });
        }
        if !self.confidence_level.is_finite() || self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(CausalError::Unsupported {
                detail: format!(
                    "confidence_level must be in (0, 1), got {}",
                    self.confidence_level
                ),
            });
        }

        // 2. Locke refusal check.
        check_locke_refusal(locke_report, treatment, outcome, covariates)?;

        // 3. Extract columns.
        let t = extract_numeric_column(df, treatment, "treatment")?;
        let y = extract_numeric_column(df, outcome, "outcome")?;
        if covariates.is_empty() {
            return Err(CausalError::Unsupported {
                detail: "DML requires at least one covariate".to_string(),
            });
        }
        let n = t.len();
        if y.len() != n {
            return Err(CausalError::Numerical {
                detail: format!("treatment and outcome length mismatch: t={}, y={}", n, y.len()),
            });
        }
        let p = covariates.len();
        let mut cov_flat: Vec<f64> = Vec::with_capacity(n * p);
        for i in 0..n {
            for &cv in covariates {
                let v = extract_numeric_column(df, cv, "covariate")?;
                if v.len() != n {
                    return Err(CausalError::Numerical {
                        detail: format!("covariate '{}' has {} rows, expected {}", cv, v.len(), n),
                    });
                }
                cov_flat.push(v[i]);
            }
        }

        // 4. Minimum-sample-size check.
        // Each fold's training set has roughly n * (k-1)/k rows. lm() needs
        // n_train > p + 2 (p covariates + intercept + 1 dof).
        let min_train_size = ((n as f64) * (self.k_folds as f64 - 1.0) / self.k_folds as f64) as usize;
        if min_train_size <= p + 2 {
            return Err(CausalError::Numerical {
                detail: format!(
                    "n = {} with k_folds = {} gives training subset of ~{} rows; need > {} (p + 2)",
                    n, self.k_folds, min_train_size, p + 2
                ),
            });
        }

        // 5. K-fold split.
        let folds = cjc_runtime::ml::kfold_indices(n, self.k_folds, self.seed);

        // 6. For each fold, fit nuisances on the training rows and predict
        // on the held-out rows. Stitch predictions back into n-vectors.
        let mut y_hat: Vec<f64> = vec![f64::NAN; n];
        let mut t_hat: Vec<f64> = vec![f64::NAN; n];
        for (train_idx, test_idx) in &folds {
            if train_idx.is_empty() || test_idx.is_empty() {
                return Err(CausalError::Numerical {
                    detail: format!(
                        "DML fold has empty train ({}) or test ({}) set",
                        train_idx.len(), test_idx.len()
                    ),
                });
            }
            // Build x_train, y_train, t_train from train_idx
            let mut x_train: Vec<f64> = Vec::with_capacity(train_idx.len() * p);
            let mut y_train: Vec<f64> = Vec::with_capacity(train_idx.len());
            let mut t_train: Vec<f64> = Vec::with_capacity(train_idx.len());
            for &i in train_idx {
                for j in 0..p {
                    x_train.push(cov_flat[i * p + j]);
                }
                y_train.push(y[i]);
                t_train.push(t[i]);
            }
            // Build x_test from test_idx
            let mut x_test: Vec<f64> = Vec::with_capacity(test_idx.len() * p);
            for &i in test_idx {
                for j in 0..p {
                    x_test.push(cov_flat[i * p + j]);
                }
            }
            // Predict
            let y_preds = fit_linear_predict(&x_train, &y_train, &x_test, p)?;
            let t_preds = fit_linear_predict(&x_train, &t_train, &x_test, p)?;
            // Stitch
            for (k_local, &i_global) in test_idx.iter().enumerate() {
                y_hat[i_global] = y_preds[k_local];
                t_hat[i_global] = t_preds[k_local];
            }
        }

        // Sanity check: every row got a prediction.
        for (i, (&yh, &th)) in y_hat.iter().zip(t_hat.iter()).enumerate() {
            if !yh.is_finite() || !th.is_finite() {
                return Err(CausalError::Numerical {
                    detail: format!(
                        "DML: row {} did not receive a cross-fit prediction (y_hat={}, t_hat={})",
                        i, yh, th
                    ),
                });
            }
        }

        // 7. Orthogonal moment.
        let score = partial_linear_score(&y, &t, &y_hat, &t_hat)?;

        // 8. CI via the Acklam normal quantile (defined in iv_regression).
        let z_crit = crate::iv_regression::normal_quantile_two_sided(self.confidence_level);
        let ci_lower = score.beta - z_crit * score.std_error;
        let ci_upper = score.beta + z_crit * score.std_error;

        // 9. Best-effort n_treated / n_control counts (T may be continuous).
        let mut n_treated_u: u64 = 0;
        let mut n_control_u: u64 = 0;
        for &val in &t {
            if val == 1.0 {
                n_treated_u += 1;
            } else if val == 0.0 {
                n_control_u += 1;
            }
        }

        // 10. Content-addressed identifier.
        let identifier = compute_identifier(
            ESTIMATOR_LABEL,
            treatment,
            outcome,
            None,
            covariates,
            assumptions,
            self.seed,
            score.beta,
            score.std_error,
        );

        Ok(EffectEstimate {
            point: score.beta,
            std_error: score.std_error,
            ci_lower,
            ci_upper,
            confidence_level: self.confidence_level,
            n_treated: n_treated_u,
            n_control: n_control_u,
            assumptions_declared: assumptions.to_vec(),
            balance_diagnostics: None,
            iv_first_stage_f: None,
            identifier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_are_sensible() {
        let est = DoubleMLEstimator::new();
        assert_eq!(est.k_folds, 5);
        assert_eq!(est.seed, 0);
        assert_eq!(est.confidence_level, 0.95);
    }

    #[test]
    fn builder_chain_sets_all_knobs() {
        let est = DoubleMLEstimator::new()
            .with_k_folds(10)
            .with_seed(42)
            .with_confidence_level(0.90);
        assert_eq!(est.k_folds, 10);
        assert_eq!(est.seed, 42);
        assert_eq!(est.confidence_level, 0.90);
    }
}
