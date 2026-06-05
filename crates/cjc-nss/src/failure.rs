//! Failure states & predictions.
//!
//! NSS Phase 1 predicts a single failure axis — *collapse probability*,
//! which combines instability across all fields into a single bounded
//! quantity. Phase 3 splits this into separate heads for collapse,
//! throughput-degradation, latency-spike, and partial-outage.

use crate::pressure::PressureKind;

/// Discrete categories of operational failure that NSS reasons about.
///
/// Phase 1 uses [`FailureKind::Collapse`] as the single positive label;
/// later phases add finer-grained categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureKind {
    /// System is operating within nominal pressure envelope.
    Nominal,
    /// Throughput is degraded but the system has not collapsed.
    Degraded,
    /// Collapse: cascading instability past the recoverable envelope.
    Collapse,
}

impl FailureKind {
    /// Canonical short label for serialisation.
    pub fn label(self) -> &'static str {
        match self {
            FailureKind::Nominal => "nominal",
            FailureKind::Degraded => "degraded",
            FailureKind::Collapse => "collapse",
        }
    }
}

/// Failure state recorded at a tick (used by the simulator to label
/// training data).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FailureState {
    /// Categorical state.
    pub kind: FailureKind,
    /// Optional dominant pressure source. `None` for [`FailureKind::Nominal`].
    pub dominant_source: Option<PressureKind>,
}

impl FailureState {
    /// Build a nominal state.
    pub fn nominal() -> Self {
        Self {
            kind: FailureKind::Nominal,
            dominant_source: None,
        }
    }

    /// Build a degraded state with a named dominant source.
    pub fn degraded(source: PressureKind) -> Self {
        Self {
            kind: FailureKind::Degraded,
            dominant_source: Some(source),
        }
    }

    /// Build a collapse state with a named dominant source.
    pub fn collapse(source: PressureKind) -> Self {
        Self {
            kind: FailureKind::Collapse,
            dominant_source: Some(source),
        }
    }

    /// Canonical bytes for hashing. Stable across runs.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(self.kind.label().as_bytes());
        bytes.push(b'|');
        match self.dominant_source {
            Some(k) => bytes.extend_from_slice(k.label().as_bytes()),
            None => bytes.extend_from_slice(b"none"),
        }
        bytes
    }
}

/// NSS's prediction of the next-step failure state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FailurePrediction {
    /// Probability in `[0, 1]` that the next tick is a collapse.
    pub collapse_probability: f64,
    /// Probability in `[0, 1]` that the next tick is degraded
    /// (non-collapse but non-nominal).
    pub degraded_probability: f64,
    /// Calibrated confidence in `[0, 1]` — the maximum of the per-class
    /// probabilities (cheap proxy in Phase 1; Phase 3 replaces this
    /// with a calibrated head).
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_state_has_no_source() {
        let s = FailureState::nominal();
        assert_eq!(s.kind, FailureKind::Nominal);
        assert!(s.dominant_source.is_none());
    }

    #[test]
    fn collapse_carries_source() {
        let s = FailureState::collapse(PressureKind::Queue);
        assert_eq!(s.kind, FailureKind::Collapse);
        assert_eq!(s.dominant_source, Some(PressureKind::Queue));
    }

    #[test]
    fn canonical_bytes_differ_by_kind() {
        let a = FailureState::nominal().canonical_bytes();
        let b = FailureState::collapse(PressureKind::Queue).canonical_bytes();
        assert_ne!(a, b);
    }
}
