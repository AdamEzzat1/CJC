//! Error type returned from every estimator's `estimate()` method.
//!
//! cjc-causal estimators **never panic** on user-supplied data. Every failure
//! mode (data quality, model misspecification, numerical breakdown, declared
//! assumptions inconsistent with the requested estimator) returns a
//! structured [`CausalError`] variant the caller can match on.

use cjc_locke::report::ValidationFinding;
use std::fmt;

/// Failure modes for any cjc-causal estimator.
#[derive(Debug, Clone)]
pub enum CausalError {
    /// The Locke report passed to the estimator contained findings severe
    /// enough that the estimator refused to run. The attached findings are
    /// the offenders the estimator inspected.
    ///
    /// Default refusal thresholds (revisable in the implementation session):
    /// `E9001` missingness ≥ 0.30 on treatment or outcome; `E9009` continuous
    /// covariate not promoted; `E9060` strong leakage detected on a covariate.
    DataQualityRefusal { findings: Vec<ValidationFinding> },

    /// The caller's declared assumption set is inconsistent with the
    /// requested estimator. For example, requesting [`super::IVRegression`]
    /// without `IdentificationAssumption::ExcludabilityOfInstrument`.
    AssumptionMismatch { detail: String },

    /// A column referenced by the caller (treatment, outcome, covariate,
    /// instrument) was not present in the input DataFrame.
    UnknownColumn { name: String },

    /// A column referenced by the caller existed but had the wrong type for
    /// the estimator (e.g., a categorical column passed as the outcome to
    /// an estimator that requires `Column::Float`).
    WrongColumnType { name: String, expected: String, found: String },

    /// Numerical breakdown — typically a singular design matrix, a logistic
    /// regression that failed to converge, or a Kalman filter innovation
    /// variance that lost positive-semidefiniteness.
    Numerical { detail: String },

    /// The requested estimator does not support the input shape. Examples:
    /// 2SLS requested with an over-identified system but more instruments
    /// than supported in v0.1; DML requested with fewer rows than `5 × K`.
    Unsupported { detail: String },
}

impl fmt::Display for CausalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CausalError::DataQualityRefusal { findings } => write!(
                f,
                "causal estimator refused to run: {} data-quality findings exceed refusal threshold",
                findings.len()
            ),
            CausalError::AssumptionMismatch { detail } => {
                write!(f, "assumption mismatch: {}", detail)
            }
            CausalError::UnknownColumn { name } => write!(f, "unknown column: {}", name),
            CausalError::WrongColumnType { name, expected, found } => write!(
                f,
                "column '{}' has wrong type: expected {}, found {}",
                name, expected, found
            ),
            CausalError::Numerical { detail } => write!(f, "numerical breakdown: {}", detail),
            CausalError::Unsupported { detail } => write!(f, "unsupported configuration: {}", detail),
        }
    }
}

impl std::error::Error for CausalError {}
