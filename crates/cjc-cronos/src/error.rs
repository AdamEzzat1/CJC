//! Error type returned from cjc-cronos methods.
//!
//! cjc-cronos methods **never panic** on user-supplied data. Every failure
//! mode (data quality, model misspecification, numerical breakdown, frequency
//! mismatch) returns a structured [`CronosError`] variant the caller can
//! match on.

use cjc_locke::report::ValidationFinding;
use std::fmt;

/// Failure modes for any cjc-cronos method.
#[derive(Debug, Clone)]
pub enum CronosError {
    /// The supplied Locke report contained findings severe enough that the
    /// forecaster/decomposer refused to run. The attached findings are the
    /// offenders the method inspected.
    DataQualityRefusal { findings: Vec<ValidationFinding> },

    /// A column or value referenced by the caller was not present.
    UnknownColumn { name: String },

    /// A column existed but had the wrong type for the requested operation.
    WrongColumnType { name: String, expected: String, found: String },

    /// The time index is not monotonically increasing — many cronos methods
    /// require sorted timestamps. Surfaces with the first offending row index.
    UnsortedTimeIndex { first_offending_row: usize },

    /// The requested model class is incompatible with the supplied
    /// [`super::Frequency`]. Example: a seasonal model with `Frequency::Irregular`.
    UnsupportedFrequency { detail: String },

    /// Numerical breakdown — singular design matrix, Kalman innovation
    /// variance lost positive-definiteness, ARIMA likelihood divergence,
    /// STL did not converge within the iteration budget.
    Numerical { detail: String },

    /// The requested operation does not support the input shape. Examples:
    /// forecast horizon of zero, requesting a seasonal model on a series
    /// shorter than one full cycle.
    Unsupported { detail: String },
}

impl fmt::Display for CronosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CronosError::DataQualityRefusal { findings } => write!(
                f,
                "cronos refused to run: {} data-quality findings exceed refusal threshold",
                findings.len()
            ),
            CronosError::UnknownColumn { name } => write!(f, "unknown column: {}", name),
            CronosError::WrongColumnType { name, expected, found } => write!(
                f,
                "column '{}' has wrong type: expected {}, found {}",
                name, expected, found
            ),
            CronosError::UnsortedTimeIndex { first_offending_row } => write!(
                f,
                "time index is not monotonically increasing (first offending row: {})",
                first_offending_row
            ),
            CronosError::UnsupportedFrequency { detail } => {
                write!(f, "unsupported frequency: {}", detail)
            }
            CronosError::Numerical { detail } => write!(f, "numerical breakdown: {}", detail),
            CronosError::Unsupported { detail } => write!(f, "unsupported configuration: {}", detail),
        }
    }
}

impl std::error::Error for CronosError {}
