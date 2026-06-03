//! Propensity-score matching estimator (`PropensityScoreMatcher`).
//!
//! See [`crate`] root docs and ADR-0043 for the full design rationale. This
//! file is the public orchestrator that composes [`super::refusal`],
//! [`super::matching`], [`super::balance`], and [`super::content_hash`] into
//! an `estimate()` call.
//!
//! ## Algorithm at a glance
//!
//! 1. Validate column names and types.
//! 2. Inspect the supplied [`LockeReport`] for refusal-grade findings; bail
//!    out with [`CausalError::DataQualityRefusal`] on match.
//! 3. Fit IRLS logistic regression `P(T=1 | X)` via
//!    [`cjc_runtime::hypothesis::logistic_regression`].
//! 4. Predict per-row propensity `p_i`; clamp to `[1e-10, 1 - 1e-10]`.
//! 5. Compute `logit(p_i)`; the caliper width is
//!    `caliper_sd * SD(logit(p))` (Austin 2011 default `0.2`).
//! 6. Greedy nearest-neighbor matching without replacement
//!    ([`super::matching::nearest_neighbor_match`]). Deterministic
//!    tie-break = ascending control row index.
//! 7. Compute matched-pair ATT via Kahan-summed outcome differences
//!    ([`super::balance::compute_att`]).
//! 8. Compute matched-pair bootstrap CI with the caller's seed
//!    ([`super::balance::bootstrap_ci`]).
//! 9. Compute per-covariate balance (SMD + variance ratio).
//! 10. Build [`EffectEstimate`] including content-addressed
//!     [`FingerprintId`].

use crate::assumption::IdentificationAssumption;
use crate::balance::{bootstrap_ci, compute_att, compute_balance};
use crate::content_hash::compute_identifier;
use crate::error::CausalError;
use crate::estimate::EffectEstimate;
use crate::matching::nearest_neighbor_match;
use crate::refusal::check_locke_refusal;
use cjc_data::{Column, DataFrame};
use cjc_locke::LockeReport;
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Estimator-label string used in the content-addressed [`FingerprintId`].
/// Changing this value would silently change every identifier — do not edit.
pub const ESTIMATOR_LABEL: &str = "propensity_score_matcher";

/// Greedy nearest-neighbor propensity-score matching with caliper.
///
/// Builder-style configuration; the only fallible operation is
/// [`Self::estimate`].
///
/// # Defaults
///
/// | Knob | Default | Source |
/// |---|---|---|
/// | `caliper_sd` | `0.2` | Austin 2011 |
/// | `seed` | `0` | — |
/// | `confidence_level` | `0.95` | convention |
/// | `bootstrap_reps` | `200` | balance speed vs. CI noise |
#[derive(Clone, Debug)]
pub struct PropensityScoreMatcher {
    caliper_sd: f64,
    seed: u64,
    confidence_level: f64,
    bootstrap_reps: usize,
}

impl Default for PropensityScoreMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityScoreMatcher {
    /// Construct with v0.1 defaults.
    pub fn new() -> Self {
        Self {
            caliper_sd: 0.2,
            seed: 0,
            confidence_level: 0.95,
            bootstrap_reps: 200,
        }
    }

    /// Caliper width in units of SD(logit(propensity)). Must be `> 0.0`.
    pub fn with_caliper_sd(mut self, c: f64) -> Self {
        self.caliper_sd = c;
        self
    }

    /// Bootstrap RNG seed. Part of the [`EffectEstimate::identifier`] hash.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Confidence level for the CI, e.g. `0.95`. Must be in `(0, 1)`.
    pub fn with_confidence_level(mut self, l: f64) -> Self {
        self.confidence_level = l;
        self
    }

    /// Number of bootstrap replications for the CI. Must be `> 0`.
    pub fn with_bootstrap_reps(mut self, n: usize) -> Self {
        self.bootstrap_reps = n;
        self
    }

    /// Estimate the average treatment effect on the treated (ATT) by
    /// propensity-score matching.
    ///
    /// See [`crate`] root docs for the full quick-start. The caller must
    /// supply a [`LockeReport`] computed over `df`; cjc-causal does not call
    /// Locke internally (ADR-0043 §5).
    pub fn estimate(
        &self,
        df: &DataFrame,
        treatment: &str,
        outcome: &str,
        covariates: &[&str],
        assumptions: &[IdentificationAssumption],
        locke_report: &LockeReport,
    ) -> Result<EffectEstimate, CausalError> {
        // 1. Config sanity checks.
        if !self.caliper_sd.is_finite() || self.caliper_sd <= 0.0 {
            return Err(CausalError::Unsupported {
                detail: format!("caliper_sd must be > 0 and finite, got {}", self.caliper_sd),
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
        if self.bootstrap_reps == 0 {
            return Err(CausalError::Unsupported {
                detail: "bootstrap_reps must be > 0".to_string(),
            });
        }

        // 2. Locke refusal check.
        check_locke_refusal(locke_report, treatment, outcome, covariates)?;

        // 3. Extract treatment + outcome + covariates as Vec<f64>.
        let treat_vec = extract_numeric_column(df, treatment, "treatment")?;
        let outcome_vec = extract_numeric_column(df, outcome, "outcome")?;
        if covariates.is_empty() {
            return Err(CausalError::Unsupported {
                detail: "at least one covariate is required".to_string(),
            });
        }
        let n_rows = treat_vec.len();
        if outcome_vec.len() != n_rows {
            return Err(CausalError::Numerical {
                detail: format!(
                    "treatment ({} rows) and outcome ({} rows) length mismatch",
                    n_rows,
                    outcome_vec.len()
                ),
            });
        }
        let mut cov_named: Vec<(String, Vec<f64>)> = Vec::with_capacity(covariates.len());
        for &cv in covariates {
            let v = extract_numeric_column(df, cv, "covariate")?;
            if v.len() != n_rows {
                return Err(CausalError::Numerical {
                    detail: format!("covariate '{}' has {} rows, expected {}", cv, v.len(), n_rows),
                });
            }
            cov_named.push((cv.to_string(), v));
        }

        // 4. Validate treatment is binary 0/1.
        let mut n_treated: u64 = 0;
        let mut n_control: u64 = 0;
        for &t in &treat_vec {
            if t == 1.0 {
                n_treated += 1;
            } else if t == 0.0 {
                n_control += 1;
            } else {
                return Err(CausalError::WrongColumnType {
                    name: treatment.to_string(),
                    expected: "binary 0/1".to_string(),
                    found: format!("non-binary value {}", t),
                });
            }
        }
        if n_treated < 2 || n_control < 2 {
            return Err(CausalError::Numerical {
                detail: format!(
                    "need at least 2 treated and 2 control units, got {} treated and {} control",
                    n_treated, n_control
                ),
            });
        }

        // 5. Fit IRLS logistic regression: P(T=1 | X).
        let p = covariates.len();
        let mut x_flat: Vec<f64> = Vec::with_capacity(n_rows * p);
        for i in 0..n_rows {
            for (_, v) in &cov_named {
                x_flat.push(v[i]);
            }
        }
        let logit_fit = cjc_runtime::hypothesis::logistic_regression(&x_flat, &treat_vec, n_rows, p)
            .map_err(|e| CausalError::Numerical { detail: format!("logistic regression failed: {}", e) })?;

        // 6. Predict per-row propensity p_i = sigmoid(intercept + x_i @ beta).
        //    Then clamp to [1e-10, 1 - 1e-10] (ADR-0043 §determinism rule 2).
        //    Then logit transform.
        let mut logits: Vec<f64> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut eta = logit_fit.coefficients[0]; // intercept
            for (j, (_, v)) in cov_named.iter().enumerate() {
                eta += logit_fit.coefficients[j + 1] * v[i];
            }
            let p_i = 1.0 / (1.0 + (-eta).exp());
            let p_i = p_i.clamp(1e-10, 1.0 - 1e-10);
            logits.push((p_i / (1.0 - p_i)).ln());
        }

        // 7. Compute caliper width from SD of all logits.
        let logit_sd = sd_kahan(&logits);
        if !logit_sd.is_finite() || logit_sd == 0.0 {
            return Err(CausalError::Numerical {
                detail: format!(
                    "propensity logit has degenerate SD = {} — covariates may be uninformative",
                    logit_sd
                ),
            });
        }
        let caliper = self.caliper_sd * logit_sd;

        // 8. Nearest-neighbor matching.
        let (pairs, n_treated_unmatched) = nearest_neighbor_match(&logits, &treat_vec, caliper);
        if pairs.is_empty() {
            return Err(CausalError::Numerical {
                detail: format!(
                    "no matched pairs within caliper {} — overlap may be insufficient",
                    caliper
                ),
            });
        }
        let treated_idx: Vec<usize> = pairs.iter().map(|p| p.0).collect();
        let control_idx: Vec<usize> = pairs.iter().map(|p| p.1).collect();

        // 9. ATT point estimate.
        let (point, _n_pairs) = compute_att(&outcome_vec, &treated_idx, &control_idx);

        // 10. Bootstrap CI.
        let mut rng = Rng::seeded(self.seed);
        let (ci_lower, ci_upper, std_error) = bootstrap_ci(
            &outcome_vec,
            &treated_idx,
            &control_idx,
            &mut rng,
            self.bootstrap_reps,
            self.confidence_level,
        );

        // 11. Balance report.
        let balance = compute_balance(&cov_named, &treated_idx, &control_idx, n_treated_unmatched);

        // 12. Content-addressed identifier.
        let identifier = compute_identifier(
            ESTIMATOR_LABEL,
            treatment,
            outcome,
            covariates,
            assumptions,
            self.seed,
            point,
            std_error,
        );

        Ok(EffectEstimate {
            point,
            std_error,
            ci_lower,
            ci_upper,
            confidence_level: self.confidence_level,
            n_treated: pairs.len() as u64,
            n_control: pairs.len() as u64,
            assumptions_declared: assumptions.to_vec(),
            balance_diagnostics: Some(balance),
            identifier,
        })
    }
}

/// Extract a numeric column as `Vec<f64>`, accepting `Column::Float` or
/// `Column::Int`. Returns `CausalError::UnknownColumn` if missing or
/// `CausalError::WrongColumnType` for non-numeric kinds.
fn extract_numeric_column(df: &DataFrame, name: &str, role: &str) -> Result<Vec<f64>, CausalError> {
    let col = df.get_column(name).ok_or_else(|| CausalError::UnknownColumn { name: name.to_string() })?;
    match col {
        Column::Float(v) => Ok(v.clone()),
        Column::Int(v) => Ok(v.iter().map(|&x| x as f64).collect()),
        other => Err(CausalError::WrongColumnType {
            name: name.to_string(),
            expected: format!("numeric ({})", role),
            found: other.type_name().to_string(),
        }),
    }
}

/// Sample standard deviation via Kahan-compensated two-pass.
fn sd_kahan(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut mean_acc = KahanAccumulatorF64::new();
    for &v in values {
        mean_acc.add(v);
    }
    let mean = mean_acc.finalize() / n as f64;
    let mut var_acc = KahanAccumulatorF64::new();
    for &v in values {
        let d = v - mean;
        var_acc.add(d * d);
    }
    (var_acc.finalize() / (n - 1) as f64).sqrt()
}
