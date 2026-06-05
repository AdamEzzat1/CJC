//! Error type for the NSS crate.
//!
//! Every fallible operation in NSS returns `Result<T, NssError>`. The
//! variants are designed to be informative (carry the failing field /
//! dimension / value) without being a panic in disguise — invalid
//! configurations and impossible states surface as errors, never as
//! `unwrap()` on user input.

use std::fmt;

/// All failure modes for the NSS crate.
#[derive(Debug, Clone, PartialEq)]
pub enum NssError {
    /// A configuration field violated its invariant (range, finiteness,
    /// shape compatibility, etc.).
    InvalidConfig {
        /// Human-readable explanation. Should be specific enough that a
        /// failing test names the offending field.
        detail: String,
    },
    /// Pressure-graph topology error: missing node, dangling edge, or
    /// duplicate edge insertion.
    PressureGraph {
        /// What went wrong.
        detail: String,
    },
    /// A `SystemState` rejected an update (e.g. inserting a pressure for
    /// a field that violates `0 ≤ magnitude` or `magnitude.is_finite()`).
    InvalidState {
        /// What went wrong.
        detail: String,
    },
    /// A pressure magnitude became non-finite during propagation. This
    /// should never happen with the structural-stability construction,
    /// so it indicates a bug, not a numerical edge case.
    NonFinitePressure {
        /// Name of the field that overflowed.
        field: String,
    },
    /// Trajectory mis-aligned: state count and event count differ, or
    /// the trajectory is empty when a non-empty trajectory was required.
    InvalidTrajectory {
        /// What went wrong.
        detail: String,
    },
    /// Replay verification failed: the recomputed run produced a
    /// different `NssRunId` than the original trace.
    ReplayMismatch {
        /// Original run id.
        expected: String,
        /// Recomputed run id.
        actual: String,
    },
    /// Simulator configuration produced an unreachable state (e.g. zero
    /// workers + non-zero arrival rate would block forever).
    SimulatorBlocked {
        /// What went wrong.
        detail: String,
    },
}

impl fmt::Display for NssError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NssError::InvalidConfig { detail } => write!(f, "invalid config: {}", detail),
            NssError::PressureGraph { detail } => write!(f, "pressure graph: {}", detail),
            NssError::InvalidState { detail } => write!(f, "invalid state: {}", detail),
            NssError::NonFinitePressure { field } => {
                write!(f, "non-finite pressure in field `{}`", field)
            }
            NssError::InvalidTrajectory { detail } => write!(f, "invalid trajectory: {}", detail),
            NssError::ReplayMismatch { expected, actual } => write!(
                f,
                "replay mismatch: expected run-id {}, got {}",
                expected, actual
            ),
            NssError::SimulatorBlocked { detail } => write!(f, "simulator blocked: {}", detail),
        }
    }
}

impl std::error::Error for NssError {}
