//! Scheduler actions. In Phase 1 the scheduler is **observed** (the
//! simulator emits scheduler actions into the trajectory); Phase 3 adds
//! the [`SchedulerAdvisoryHead`](crate) that *predicts* the next action.
//!
//! Schedulers are not passive metadata — scheduler actions themselves
//! create and redistribute pressure. The NSS architecture exposes them as
//! first-class events in the [`crate::SystemTrajectory`].

/// Discrete categories of scheduler decision NSS models in Phase 1.
/// Closed enum; new policies require an explicit code change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SchedulerKind {
    /// No-op tick — scheduler observed but didn't act.
    Idle,
    /// Routed work to a different worker.
    Reroute,
    /// Created a new worker (autoscale up).
    ScaleUp,
    /// Retired a worker (autoscale down).
    ScaleDown,
    /// Pre-empted a running task.
    Preempt,
    /// Applied admission control / shed load.
    ShedLoad,
}

impl SchedulerKind {
    /// Canonical short label for serialisation.
    pub fn label(self) -> &'static str {
        match self {
            SchedulerKind::Idle => "idle",
            SchedulerKind::Reroute => "reroute",
            SchedulerKind::ScaleUp => "scale_up",
            SchedulerKind::ScaleDown => "scale_down",
            SchedulerKind::Preempt => "preempt",
            SchedulerKind::ShedLoad => "shed_load",
        }
    }
}

/// One scheduler action observed (or predicted) at a tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedulerAction {
    /// Discrete category.
    pub kind: SchedulerKind,
    /// Magnitude of the action in `[0, 1]` — e.g. "rerouted 30% of work"
    /// or "scaled up by 50% of current pool". 0 for `Idle`.
    pub intensity: f64,
}

impl SchedulerAction {
    /// Build an idle action — sentinel "scheduler did not act this tick".
    pub fn idle() -> Self {
        Self {
            kind: SchedulerKind::Idle,
            intensity: 0.0,
        }
    }

    /// Build a non-idle action. Clips `intensity` into `[0, 1]` so a
    /// noisy simulator can't produce out-of-range values.
    pub fn new(kind: SchedulerKind, intensity: f64) -> Self {
        let i = if !intensity.is_finite() {
            0.0
        } else {
            intensity.max(0.0).min(1.0)
        };
        Self { kind, intensity: i }
    }

    /// Canonical bytes for hashing. Stable across runs.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(self.kind.label().as_bytes());
        bytes.push(b'|');
        bytes.extend_from_slice(&self.intensity.to_bits().to_le_bytes());
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_action_has_zero_intensity() {
        let a = SchedulerAction::idle();
        assert_eq!(a.kind, SchedulerKind::Idle);
        assert_eq!(a.intensity, 0.0);
    }

    #[test]
    fn intensity_is_clipped() {
        let a = SchedulerAction::new(SchedulerKind::Reroute, 1.5);
        assert_eq!(a.intensity, 1.0);
        let a = SchedulerAction::new(SchedulerKind::Reroute, -0.1);
        assert_eq!(a.intensity, 0.0);
        let a = SchedulerAction::new(SchedulerKind::Reroute, f64::NAN);
        assert_eq!(a.intensity, 0.0);
    }
}
