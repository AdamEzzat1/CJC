//! Two-stage least squares (2SLS) instrumental-variables estimator.
//!
//! See ADR-0043 for the full design rationale. v0.1 ships the **just-identified
//! case only** — exactly one instrument `Z` for exactly one endogenous
//! treatment `T`. Over-identified 2SLS (multiple instruments) is deferred to
//! v0.2 because (a) it requires the projection-formula β̂ = (X' P_Z X)⁻¹ X' P_Z y
//! instead of naive two-stage OLS, and (b) the partial-F statistic stops being
//! equal to the squared t-statistic and needs a restricted-vs-unrestricted
//! regression pair.
//!
//! ## Pipeline
//!
//! 1. Validate config + columns + dimensions.
//! 2. Locke refusal check (treatment, outcome, instrument, covariates).
//! 3. First-stage OLS via [`cjc_runtime::hypothesis::lm`]:
//!    `T = α + γ·Z + δ'·X + u`
//! 4. First-stage F-statistic = `t_γ²` (just-identified shortcut).
//! 5. Compute fitted treatment `T̂_i = α̂ + γ̂·Z_i + δ̂'·X_i` (Kahan-summed).
//! 6. Second-stage OLS via [`cjc_runtime::hypothesis::lm`]:
//!    `y = β₀ + β·T̂ + θ'·X + ε`
//! 7. Compute **2SLS residuals** using **original T** (not fitted T̂):
//!    `e_i = y_i - β̂₀ - β̂·T_i - θ̂'·X_i`
//!    (Critical: this is the standard 2SLS variance estimator; using T̂ here
//!    would understate the standard error.)
//! 8. HC1 sandwich SE for `β̂`:
//!    `V = (n/(n-k)) · (X'X)⁻¹ · X'diag(e²)X · (X'X)⁻¹`
//!    where `X = [1 | T̂ | X_covariates]`.
//! 9. CI: `β̂ ± Φ⁻¹(1 - α/2) · SE_β` (normal approximation; t-quantile
//!    deferred to v0.2 when `cjc-runtime` exposes a t-distribution quantile).
//! 10. Surface `iv_first_stage_f` on the `EffectEstimate`. Caller may invoke
//!     [`weak_instrument_finding`] to convert to a Locke `E9100` finding.
//! 11. Content-addressed identifier with `instrument = Some(name)`.

use crate::assumption::IdentificationAssumption;
use crate::content_hash::compute_identifier;
use crate::error::CausalError;
use crate::estimate::EffectEstimate;
use crate::linalg::{cholesky_invert, gram, gram_weighted, matmul_square};
use crate::propensity_score::extract_numeric_column;
use crate::refusal::check_locke_refusal;
use cjc_data::DataFrame;
use cjc_locke::report::{FindingEvidence, FindingSeverity, ValidationFinding};
use cjc_locke::LockeReport;
use cjc_repro::KahanAccumulatorF64;

/// Estimator-label string used in the content-addressed [`FingerprintId`].
/// Changing this value would silently change every identifier — do not edit.
pub const ESTIMATOR_LABEL: &str = "iv_regression";

/// Locke error code emitted for weak instruments (handoff §5.3).
pub const WEAK_INSTRUMENT_CODE: &str = "E9100";

/// Stock-Yogo (2005) F-statistic critical value for the "5% bias" threshold
/// in the single-instrument case. F below this value indicates a weak
/// instrument; 2SLS causal interpretation becomes unreliable.
pub const WEAK_INSTRUMENT_THRESHOLD_DEFAULT: f64 = 10.0;

/// Just-identified 2SLS instrumental-variables estimator with HC1 sandwich SE.
///
/// # Defaults
///
/// | Knob | Default | Source |
/// |---|---|---|
/// | `confidence_level` | `0.95` | convention |
/// | `weak_instrument_threshold` | `10.0` | Stock-Yogo 2005 5% bias |
#[derive(Clone, Debug)]
pub struct IVRegression {
    confidence_level: f64,
    weak_instrument_threshold: f64,
}

impl Default for IVRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl IVRegression {
    /// Construct with v0.1 defaults.
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
            weak_instrument_threshold: WEAK_INSTRUMENT_THRESHOLD_DEFAULT,
        }
    }

    /// Confidence level for the CI, e.g. `0.95`. Must be in `(0, 1)`.
    pub fn with_confidence_level(mut self, l: f64) -> Self {
        self.confidence_level = l;
        self
    }

    /// First-stage F-statistic threshold below which the instrument is
    /// flagged as weak by [`weak_instrument_finding`]. Default `10.0`
    /// per Stock-Yogo 2005. Must be `≥ 0`.
    pub fn with_weak_instrument_threshold(mut self, f: f64) -> Self {
        self.weak_instrument_threshold = f;
        self
    }

    /// Estimate the average treatment effect of `treatment` on `outcome`
    /// using `instrument` as an exogenous source of treatment variation,
    /// adjusting for `covariates`.
    ///
    /// # Errors
    ///
    /// - [`CausalError::DataQualityRefusal`] if `locke_report` flags
    ///   refusal-grade findings on T, Y, the instrument, or any covariate.
    /// - [`CausalError::UnknownColumn`] / [`CausalError::WrongColumnType`] /
    ///   [`CausalError::Numerical`] / [`CausalError::Unsupported`] per the
    ///   shared cjc-causal error contract.
    ///
    /// # Notes
    ///
    /// The returned `EffectEstimate.iv_first_stage_f` is `Some(F)` where
    /// `F = t_γ²` is the squared t-statistic of the instrument coefficient
    /// in the first stage. Use [`weak_instrument_finding`] to convert this
    /// to a Locke `E9100` finding if it falls below the threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        df: &DataFrame,
        treatment: &str,
        outcome: &str,
        instrument: &str,
        covariates: &[&str],
        assumptions: &[IdentificationAssumption],
        locke_report: &LockeReport,
    ) -> Result<EffectEstimate, CausalError> {
        // 1. Config validation.
        if !self.confidence_level.is_finite() || self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(CausalError::Unsupported {
                detail: format!(
                    "confidence_level must be in (0, 1), got {}",
                    self.confidence_level
                ),
            });
        }
        if !self.weak_instrument_threshold.is_finite() || self.weak_instrument_threshold < 0.0 {
            return Err(CausalError::Unsupported {
                detail: format!(
                    "weak_instrument_threshold must be >= 0 and finite, got {}",
                    self.weak_instrument_threshold
                ),
            });
        }

        // 2. Locke refusal check — instrument is treated as an additional
        // covariate-like column for refusal purposes (E9009 if not promoted,
        // E9060 if it leaks the outcome too directly).
        let mut all_covlike: Vec<&str> = covariates.to_vec();
        all_covlike.push(instrument);
        check_locke_refusal(locke_report, treatment, outcome, &all_covlike)?;

        // 3. Extract columns.
        let t = extract_numeric_column(df, treatment, "treatment")?;
        let y = extract_numeric_column(df, outcome, "outcome")?;
        let z = extract_numeric_column(df, instrument, "instrument")?;
        let n_rows = t.len();
        if y.len() != n_rows || z.len() != n_rows {
            return Err(CausalError::Numerical {
                detail: format!(
                    "treatment/outcome/instrument length mismatch: t={}, y={}, z={}",
                    n_rows, y.len(), z.len()
                ),
            });
        }
        let mut cov_data: Vec<(String, Vec<f64>)> = Vec::with_capacity(covariates.len());
        for &cv in covariates {
            let v = extract_numeric_column(df, cv, "covariate")?;
            if v.len() != n_rows {
                return Err(CausalError::Numerical {
                    detail: format!("covariate '{}' has {} rows, expected {}", cv, v.len(), n_rows),
                });
            }
            cov_data.push((cv.to_string(), v));
        }

        // 4. Minimum-sample-size check.
        // k = number of regressors in the SECOND stage including intercept = covariates + 2.
        // We need n > k + 1 for lm() to even run without rank issues.
        let k = 2 + cov_data.len();
        if n_rows <= k + 1 {
            return Err(CausalError::Numerical {
                detail: format!(
                    "n = {} rows is too small for IV with {} covariates (need n > {})",
                    n_rows,
                    cov_data.len(),
                    k + 1
                ),
            });
        }

        // 5. First-stage regression: T = α + γ·Z + δ'·X + u.
        let p_first = 1 + cov_data.len(); // Z + covariates; lm() auto-adds intercept.
        let mut x_first: Vec<f64> = Vec::with_capacity(n_rows * p_first);
        for i in 0..n_rows {
            x_first.push(z[i]);
            for (_, v) in &cov_data {
                x_first.push(v[i]);
            }
        }
        let first = cjc_runtime::hypothesis::lm(&x_first, &t, n_rows, p_first)
            .map_err(|e| CausalError::Numerical { detail: format!("first-stage regression failed: {}", e) })?;

        // first.coefficients = [α, γ, δ_1, ..., δ_p_cov]
        // First-stage F-stat (just-identified): F = t_γ² where t_γ = first.t_values[1].
        let t_gamma = first.t_values[1];
        let first_stage_f = t_gamma * t_gamma;

        // 6. Compute fitted treatment T̂_i = α + γ·Z_i + δ'·X_i (Kahan-summed).
        let mut fitted_t: Vec<f64> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut acc = KahanAccumulatorF64::new();
            acc.add(first.coefficients[0]);
            acc.add(first.coefficients[1] * z[i]);
            for (j, (_, v)) in cov_data.iter().enumerate() {
                acc.add(first.coefficients[2 + j] * v[i]);
            }
            fitted_t.push(acc.finalize());
        }

        // 7. Second-stage regression: y = β₀ + β·T̂ + θ'·X + ε.
        let p_second = 1 + cov_data.len(); // T̂ + covariates; lm() auto-adds intercept.
        let mut x_second: Vec<f64> = Vec::with_capacity(n_rows * p_second);
        for i in 0..n_rows {
            x_second.push(fitted_t[i]);
            for (_, v) in &cov_data {
                x_second.push(v[i]);
            }
        }
        let second = cjc_runtime::hypothesis::lm(&x_second, &y, n_rows, p_second)
            .map_err(|e| CausalError::Numerical { detail: format!("second-stage regression failed: {}", e) })?;

        // second.coefficients = [β₀, β, θ_1, ..., θ_p_cov]
        let beta = second.coefficients[1];

        // 8. 2SLS residuals using ORIGINAL T (not T̂).
        // e_i = y_i - β₀ - β·T_i - θ'·X_i
        let mut residuals: Vec<f64> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut acc = KahanAccumulatorF64::new();
            acc.add(y[i]);
            acc.add(-second.coefficients[0]);
            acc.add(-beta * t[i]);
            for (j, (_, v)) in cov_data.iter().enumerate() {
                acc.add(-second.coefficients[2 + j] * v[i]);
            }
            residuals.push(acc.finalize());
        }

        // 9. HC1 sandwich SE.
        // X_aug = [1 | T̂ | X_covariates], shape (n × k).
        let mut x_aug: Vec<f64> = Vec::with_capacity(n_rows * k);
        for i in 0..n_rows {
            x_aug.push(1.0);
            x_aug.push(fitted_t[i]);
            for (_, v) in &cov_data {
                x_aug.push(v[i]);
            }
        }
        let xtx = gram(&x_aug, n_rows, k);
        let bread = cholesky_invert(&xtx, k)?;
        let e_sq: Vec<f64> = residuals.iter().map(|e| e * e).collect();
        let meat = gram_weighted(&x_aug, n_rows, k, &e_sq);
        let bm = matmul_square(&bread, &meat, k);
        let v_mat = matmul_square(&bm, &bread, k);
        // HC1 finite-sample correction.
        let factor = (n_rows as f64) / ((n_rows - k) as f64);
        let var_beta = v_mat[k + 1] * factor; // index [1, 1] in row-major k×k
        if !(var_beta >= 0.0) {
            return Err(CausalError::Numerical {
                detail: format!("HC1 variance for β is non-positive: {}", var_beta),
            });
        }
        let se_beta = var_beta.sqrt();

        // 10. CI with normal approximation.
        let z_crit = normal_quantile_two_sided(self.confidence_level);
        let ci_lower = beta - z_crit * se_beta;
        let ci_upper = beta + z_crit * se_beta;

        // 11. Best-effort n_treated / n_control counts (T may be continuous).
        let mut n_treated_u: u64 = 0;
        let mut n_control_u: u64 = 0;
        for &val in &t {
            if val == 1.0 {
                n_treated_u += 1;
            } else if val == 0.0 {
                n_control_u += 1;
            }
        }

        // 12. Content-addressed identifier. seed=0 because v0.1 IV has no
        // RNG-driven step (no bootstrap yet).
        let identifier = compute_identifier(
            ESTIMATOR_LABEL,
            treatment,
            outcome,
            Some(instrument),
            covariates,
            assumptions,
            0,
            beta,
            se_beta,
        );

        Ok(EffectEstimate {
            point: beta,
            std_error: se_beta,
            ci_lower,
            ci_upper,
            confidence_level: self.confidence_level,
            n_treated: n_treated_u,
            n_control: n_control_u,
            assumptions_declared: assumptions.to_vec(),
            balance_diagnostics: None,
            iv_first_stage_f: Some(first_stage_f),
            identifier,
        })
    }
}

/// Convert a weak first-stage F-statistic into a Locke `E9100` finding.
///
/// Returns `Some(finding)` when `estimate.iv_first_stage_f` is `Some(f)` and
/// `f < threshold`, otherwise `None`. The caller composes the returned
/// finding into their existing Locke report pipeline (e.g. via
/// `LockeReport.findings.push(...)` or a custom-detector adapter).
///
/// Severity is `Error` because a weak instrument invalidates the causal
/// interpretation of the IV estimate, not just its precision.
pub fn weak_instrument_finding(
    estimate: &EffectEstimate,
    instrument_col: &str,
    threshold: f64,
) -> Option<ValidationFinding> {
    let f = estimate.iv_first_stage_f?;
    if !(f < threshold) {
        return None;
    }
    Some(ValidationFinding::new(
        WEAK_INSTRUMENT_CODE,
        FindingSeverity::Error,
        format!(
            "first-stage F = {:.3} is below the weak-instrument threshold {:.3}",
            f, threshold
        ),
        Some(instrument_col.to_string()),
        None,
        vec![
            FindingEvidence::Metric { label: "first_stage_f".to_string(), value: f },
            FindingEvidence::Metric { label: "threshold".to_string(), value: threshold },
        ],
        estimate.n_treated + estimate.n_control,
        vec![
            "treatment is continuous OR truly binary 0/1 (the count is best-effort otherwise)".into(),
        ],
        vec![
            "consider a stronger instrument".into(),
            "see Stock-Yogo (2005) for over-identified case critical values".into(),
        ],
    ))
}

/// Two-sided normal quantile via Acklam's (2003) rational approximation.
///
/// Given confidence level `cl` in `(0, 1)`, returns `Φ⁻¹(1 - (1-cl)/2)` — i.e.
/// the critical value such that `P(|Z| <= q) = cl`. For `cl = 0.95` the result
/// is approximately `1.959963984540054`.
///
/// Accuracy: better than `1e-7` over the full input range (Acklam's published
/// relative error is `1.15e-9`; the absolute tolerance is set by the largest
/// expected `z` value, e.g. ~3e-9 at z=2.576). Deterministic across runs and
/// platforms (no transcendental library calls beyond `f64::ln` and
/// `f64::sqrt`, both of which are IEEE-754-correct).
///
/// `pub(crate)` so the DML orchestrator can reuse the same normal-quantile
/// implementation rather than duplicating Acklam coefficients.
pub(crate) fn normal_quantile_two_sided(cl: f64) -> f64 {
    let p = 1.0 - (1.0 - cl) / 2.0;
    acklam_normal_quantile(p)
}

/// Acklam (2003) approximation to `Φ⁻¹(p)` for `p ∈ (0, 1)`.
fn acklam_normal_quantile(p: f64) -> f64 {
    // Coefficients from the Acklam paper, Table 1.
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

    #[test]
    fn acklam_normal_quantile_at_0_5_is_zero() {
        let z = acklam_normal_quantile(0.5);
        assert!(z.abs() < 1e-12);
    }

    // Tolerance is 1e-7, comfortably above Acklam's published relative-error
    // ceiling of 1.15e-9 (absolute error ~3e-9 for z ≈ 2.576). Tighter
    // tolerances catch IEEE-754 ULP noise without flagging real bugs.
    const QUANTILE_TOL: f64 = 1e-7;

    #[test]
    fn acklam_normal_quantile_at_0_975_is_about_1_96() {
        let z = acklam_normal_quantile(0.975);
        assert!((z - 1.9599639845400545).abs() < QUANTILE_TOL, "got {}", z);
    }

    #[test]
    fn acklam_normal_quantile_is_antisymmetric() {
        // Φ⁻¹(p) = -Φ⁻¹(1-p)
        for p in [0.1, 0.25, 0.4, 0.49] {
            let a = acklam_normal_quantile(p);
            let b = acklam_normal_quantile(1.0 - p);
            assert!((a + b).abs() < QUANTILE_TOL, "symmetry broken at p={}", p);
        }
    }

    #[test]
    fn normal_quantile_two_sided_at_95_pct_is_about_1_96() {
        let z = normal_quantile_two_sided(0.95);
        assert!((z - 1.9599639845400545).abs() < QUANTILE_TOL, "got {}", z);
    }

    #[test]
    fn normal_quantile_two_sided_at_99_pct_is_about_2_576() {
        let z = normal_quantile_two_sided(0.99);
        assert!((z - 2.5758293035489008).abs() < QUANTILE_TOL, "got {}", z);
    }

    #[test]
    fn weak_instrument_finding_fires_below_threshold() {
        let est = EffectEstimate {
            point: 1.0,
            std_error: 0.1,
            ci_lower: 0.8,
            ci_upper: 1.2,
            confidence_level: 0.95,
            n_treated: 100,
            n_control: 100,
            assumptions_declared: vec![],
            balance_diagnostics: None,
            iv_first_stage_f: Some(5.0), // < 10.0 threshold
            identifier: cjc_locke::id::FingerprintId(0),
        };
        let finding = weak_instrument_finding(&est, "Z", 10.0).unwrap();
        assert_eq!(finding.code, WEAK_INSTRUMENT_CODE);
        assert_eq!(finding.severity, FindingSeverity::Error);
        assert_eq!(finding.column.as_deref(), Some("Z"));
    }

    #[test]
    fn weak_instrument_finding_does_not_fire_at_or_above_threshold() {
        let est = EffectEstimate {
            point: 1.0,
            std_error: 0.1,
            ci_lower: 0.8,
            ci_upper: 1.2,
            confidence_level: 0.95,
            n_treated: 100,
            n_control: 100,
            assumptions_declared: vec![],
            balance_diagnostics: None,
            iv_first_stage_f: Some(10.0), // exactly at threshold; strict-less-than
            identifier: cjc_locke::id::FingerprintId(0),
        };
        assert!(weak_instrument_finding(&est, "Z", 10.0).is_none());

        let est2 = EffectEstimate {
            iv_first_stage_f: Some(15.0),
            ..est
        };
        assert!(weak_instrument_finding(&est2, "Z", 10.0).is_none());
    }

    #[test]
    fn weak_instrument_finding_returns_none_when_f_is_none() {
        let est = EffectEstimate {
            point: 1.0,
            std_error: 0.1,
            ci_lower: 0.8,
            ci_upper: 1.2,
            confidence_level: 0.95,
            n_treated: 100,
            n_control: 100,
            assumptions_declared: vec![],
            balance_diagnostics: None,
            iv_first_stage_f: None,
            identifier: cjc_locke::id::FingerprintId(0),
        };
        assert!(weak_instrument_finding(&est, "Z", 10.0).is_none());
    }
}
