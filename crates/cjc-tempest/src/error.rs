//! Error type returned from cjc-tempest sampler methods.
//!
//! cjc-tempest methods **never panic** on user-supplied data or log-posterior
//! closures. Every failure mode (data quality, invalid initial state,
//! numerical breakdown, convergence failure) returns a structured
//! [`TempestError`] variant the caller can match on.

use cjc_locke::report::ValidationFinding;
use std::fmt;

/// Failure modes for any cjc-tempest sampler.
#[derive(Debug, Clone)]
pub enum TempestError {
    /// The supplied Locke report contained findings severe enough that
    /// the sampler refused to start.
    DataQualityRefusal { findings: Vec<ValidationFinding> },

    /// The user-supplied log-posterior closure returned a non-finite value
    /// (NaN, +Inf, or -Inf is acceptable for hard constraints but the user
    /// must declare this is intentional). Returned eagerly during the
    /// initial-state evaluation.
    InvalidLogPosterior { detail: String },

    /// The supplied initial state is invalid (wrong dimensionality, NaN
    /// entries, or violates a declared constraint).
    InvalidInitialState { detail: String },

    /// Numerical breakdown during sampling — e.g., the leapfrog integrator
    /// produced infinite kinetic energy, or the mass-matrix adaptation
    /// lost positive-definiteness. Differs from `InvalidLogPosterior` in
    /// that the breakdown occurred *during* sampling, not at initial
    /// evaluation.
    Numerical { detail: String },

    /// Convergence diagnostics failed thresholds after sampling completed.
    /// `r_hat` is the worst R-hat across parameters.
    ConvergenceFailure { detail: String, r_hat: f64 },

    /// The requested configuration is not supported. Examples: zero chains
    /// requested, mass-matrix adaptation enabled in v0.1 (deferred to v0.2),
    /// over-identified parameter dimensions.
    Unsupported { detail: String },
}

impl fmt::Display for TempestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TempestError::DataQualityRefusal { findings } => write!(
                f,
                "tempest refused to start: {} data-quality findings exceed refusal threshold",
                findings.len()
            ),
            TempestError::InvalidLogPosterior { detail } => {
                write!(f, "invalid log-posterior: {}", detail)
            }
            TempestError::InvalidInitialState { detail } => {
                write!(f, "invalid initial state: {}", detail)
            }
            TempestError::Numerical { detail } => write!(f, "numerical breakdown: {}", detail),
            TempestError::ConvergenceFailure { detail, r_hat } => write!(
                f,
                "convergence failure (R-hat = {:.4}): {}",
                r_hat, detail
            ),
            TempestError::Unsupported { detail } => write!(f, "unsupported configuration: {}", detail),
        }
    }
}

impl std::error::Error for TempestError {}
