//! Error type returned from cjc-cronos-gan operations.
//!
//! Like sibling decision-layer crates (cjc-causal, cjc-cronos, cjc-tempest),
//! cjc-cronos-gan **never panics** on user-supplied data, configuration, or
//! seeds. Every failure mode returns a structured [`CronosGanError`] variant
//! the caller can match on.

use std::fmt;

/// Failure modes for any cjc-cronos-gan operation.
#[derive(Debug, Clone)]
pub enum CronosGanError {
    /// Caller-supplied dimensions do not agree (e.g. `TimeSeries` value
    /// vector length is not a multiple of `n_dim`).
    DimensionMismatch { detail: String },

    /// A user-supplied configuration field is outside its valid range
    /// (e.g. `state_dim == 0`, `dt <= 0`, `tau_min >= tau_max`).
    InvalidConfig { detail: String },

    /// A user-supplied tensor / vector contains non-finite values (NaN or
    /// ±∞). Returned eagerly so non-finite values cannot propagate through
    /// state updates and silently produce more non-finite values.
    NonFiniteInput { detail: String },

    /// A `SequenceMask` length disagrees with the `TimeSeries` it is
    /// applied to.
    MaskLengthMismatch { mask_len: usize, series_len: usize },

    /// A forecast horizon does not fit within the source `TimeSeries`.
    ForecastHorizonOutOfRange {
        start_step: usize,
        horizon: usize,
        n_steps: usize,
    },
}

impl fmt::Display for CronosGanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CronosGanError::DimensionMismatch { detail } => {
                write!(f, "dimension mismatch: {}", detail)
            }
            CronosGanError::InvalidConfig { detail } => {
                write!(f, "invalid configuration: {}", detail)
            }
            CronosGanError::NonFiniteInput { detail } => {
                write!(f, "non-finite input: {}", detail)
            }
            CronosGanError::MaskLengthMismatch { mask_len, series_len } => write!(
                f,
                "mask length {} does not match series length {}",
                mask_len, series_len
            ),
            CronosGanError::ForecastHorizonOutOfRange {
                start_step,
                horizon,
                n_steps,
            } => write!(
                f,
                "forecast window [start={}, horizon={}] exceeds series length {}",
                start_step, horizon, n_steps
            ),
        }
    }
}

impl std::error::Error for CronosGanError {}
