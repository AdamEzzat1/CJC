//! `SystemState`, `SystemEvent` (Systems IR record), and `SystemTrajectory`.
//!
//! The Systems IR (SIR) is, by design, an *immutable event log* rather
//! than a mutable graph. Each `SystemEvent` is the canonical record of
//! one simulator (or runtime) tick:
//!
//! 1. the [`SystemState`] before the tick,
//! 2. the [`SchedulerAction`](crate::SchedulerAction) that fired,
//! 3. the [`FailureState`](crate::FailureState) observed.
//!
//! The next tick's `SystemState` is the previous tick's *after-state*.
//! Phase 1 trajectories chain these together; replay then becomes a
//! pure function `(initial_state, events, seed) → trajectory`. This
//! mirrors the cjc-mir-exec philosophy: the IR is the contract, the
//! executor is replaceable.

use crate::error::NssError;
use crate::failure::FailureState;
use crate::pressure::PressureField;
use crate::scheduler::SchedulerAction;
use cjc_repro::KahanAccumulatorF64;

/// Snapshot of the system at one tick.
///
/// `tick` is monotonic across a trajectory and starts at 0. Trajectories
/// reject out-of-order or duplicate `tick` values.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemState {
    /// Monotonic tick index (0-based).
    pub tick: u64,
    /// Full pressure field at this tick.
    pub pressures: PressureField,
    /// Number of in-flight tasks the system is currently servicing.
    pub in_flight: u64,
    /// Cumulative tasks completed up to this tick (monotonic).
    pub completed: u64,
    /// Cumulative tasks rejected (admission-controlled or shed) up to
    /// this tick (monotonic).
    pub rejected: u64,
    /// Observed mean service time over the last window (in arbitrary
    /// time units — the simulator defines the unit).
    pub mean_service_time: f64,
}

impl SystemState {
    /// Build an empty initial state at tick 0 with the default pressure
    /// field.
    pub fn initial() -> Self {
        Self {
            tick: 0,
            pressures: PressureField::with_default_thresholds(),
            in_flight: 0,
            completed: 0,
            rejected: 0,
            mean_service_time: 1.0,
        }
    }

    /// Validate every numerical field is finite + bounded. Called by the
    /// simulator after every tick.
    pub fn validate(&self) -> Result<(), NssError> {
        if !self.mean_service_time.is_finite() || self.mean_service_time < 0.0 {
            return Err(NssError::InvalidState {
                detail: format!(
                    "mean_service_time must be finite and >= 0, got {}",
                    self.mean_service_time
                ),
            });
        }
        if !self.pressures.all_finite() {
            return Err(NssError::InvalidState {
                detail: "pressure field contains non-finite values".to_string(),
            });
        }
        Ok(())
    }

    /// Canonical byte representation. Two states with identical field
    /// values produce identical bytes — used by [`crate::InputHash`].
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&self.tick.to_le_bytes());
        bytes.extend_from_slice(&self.in_flight.to_le_bytes());
        bytes.extend_from_slice(&self.completed.to_le_bytes());
        bytes.extend_from_slice(&self.rejected.to_le_bytes());
        bytes.extend_from_slice(&self.mean_service_time.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.pressures.canonical_bytes());
        bytes
    }
}

/// One Systems-IR record: the (state, scheduler-action, failure-state)
/// tuple at a single tick.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemEvent {
    /// State *before* the scheduler action fires.
    pub state: SystemState,
    /// Scheduler action observed at this tick.
    pub action: SchedulerAction,
    /// Failure state observed at this tick.
    pub failure: FailureState,
}

impl SystemEvent {
    /// Build an event. Validates the underlying state.
    pub fn new(
        state: SystemState,
        action: SchedulerAction,
        failure: FailureState,
    ) -> Result<Self, NssError> {
        state.validate()?;
        Ok(Self { state, action, failure })
    }

    /// Canonical bytes covering state + action + failure. Order is
    /// fixed; two events with identical fields produce identical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.state.canonical_bytes();
        bytes.push(b'#');
        bytes.extend_from_slice(&self.action.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.failure.canonical_bytes());
        bytes
    }
}

/// An immutable sequence of [`SystemEvent`]s. The trajectory is the
/// canonical input/output type for both the simulator and NSS — the
/// simulator emits one, NSS trains on one, and the replay validator
/// consumes one.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SystemTrajectory {
    events: Vec<SystemEvent>,
}

impl SystemTrajectory {
    /// Empty trajectory.
    pub fn empty() -> Self {
        Self { events: Vec::new() }
    }

    /// Build a trajectory from a `Vec<SystemEvent>`, validating tick
    /// monotonicity. Empty input is allowed.
    pub fn from_events(events: Vec<SystemEvent>) -> Result<Self, NssError> {
        for (i, w) in events.windows(2).enumerate() {
            if w[1].state.tick != w[0].state.tick + 1 {
                return Err(NssError::InvalidTrajectory {
                    detail: format!(
                        "tick must increase by 1: events[{}].tick={}, events[{}].tick={}",
                        i,
                        w[0].state.tick,
                        i + 1,
                        w[1].state.tick
                    ),
                });
            }
        }
        Ok(Self { events })
    }

    /// Push one event, enforcing tick = previous + 1. An empty
    /// trajectory accepts any starting tick (for symmetry with
    /// `from_events` and to support fork/resume scenarios).
    pub fn push(&mut self, ev: SystemEvent) -> Result<(), NssError> {
        if let Some(prev) = self.events.last() {
            let expected = prev.state.tick + 1;
            if ev.state.tick != expected {
                return Err(NssError::InvalidTrajectory {
                    detail: format!("expected tick {}, got {}", expected, ev.state.tick),
                });
            }
        }
        self.events.push(ev);
        Ok(())
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate events in tick order.
    pub fn iter(&self) -> impl Iterator<Item = &SystemEvent> {
        self.events.iter()
    }

    /// Borrow events as a slice.
    pub fn as_slice(&self) -> &[SystemEvent] {
        &self.events
    }

    /// Return the last state (the simulator's after-state). Useful as
    /// the input to `NeuralSystemsSimulator::predict_next`.
    pub fn last_state(&self) -> Option<&SystemState> {
        self.events.last().map(|e| &e.state)
    }

    /// Mean pressure magnitude across the trajectory for one
    /// [`crate::PressureKind`] — Kahan-compensated. Useful for tests
    /// that assert trajectory-level invariants ("queue pressure rises
    /// monotonically under retry-storm load").
    pub fn mean_pressure(&self, kind: crate::PressureKind) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        let mut acc = KahanAccumulatorF64::new();
        for ev in &self.events {
            if let Some(p) = ev.state.pressures.get(kind) {
                acc.add(p.magnitude);
            }
        }
        acc.finalize() / self.events.len() as f64
    }

    /// Canonical bytes covering every event in tick order. Two
    /// trajectories with identical contents produce identical bytes;
    /// the input-hash of an NSS run is computed over this.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.events.len() * 96);
        bytes.extend_from_slice(&(self.events.len() as u64).to_le_bytes());
        for ev in &self.events {
            bytes.extend_from_slice(&ev.canonical_bytes());
            bytes.push(b'\n');
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::{Pressure, PressureKind};

    fn ev(tick: u64) -> SystemEvent {
        let mut s = SystemState::initial();
        s.tick = tick;
        SystemEvent::new(s, SchedulerAction::idle(), FailureState::nominal()).unwrap()
    }

    #[test]
    fn trajectory_accepts_monotonic_ticks() {
        let mut t = SystemTrajectory::empty();
        for i in 0..5 {
            t.push(ev(i)).unwrap();
        }
        assert_eq!(t.len(), 5);
    }

    #[test]
    fn trajectory_rejects_out_of_order_ticks() {
        let mut t = SystemTrajectory::empty();
        t.push(ev(0)).unwrap();
        assert!(t.push(ev(2)).is_err());
    }

    #[test]
    fn last_state_returns_final_event() {
        let mut t = SystemTrajectory::empty();
        for i in 0..3 {
            t.push(ev(i)).unwrap();
        }
        assert_eq!(t.last_state().unwrap().tick, 2);
    }

    #[test]
    fn canonical_bytes_stable_across_constructions() {
        let mut a = SystemTrajectory::empty();
        let mut b = SystemTrajectory::empty();
        for i in 0..4 {
            a.push(ev(i)).unwrap();
            b.push(ev(i)).unwrap();
        }
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn mean_pressure_kahan_sums_constant_load() {
        let mut t = SystemTrajectory::empty();
        for i in 0..10 {
            let mut s = SystemState::initial();
            s.tick = i;
            s.pressures.set(
                PressureKind::Queue,
                Pressure::new(0.1, 1.0, 0.0).unwrap(),
            );
            t.push(SystemEvent::new(s, SchedulerAction::idle(), FailureState::nominal()).unwrap())
                .unwrap();
        }
        let m = t.mean_pressure(PressureKind::Queue);
        assert!((m - 0.1).abs() < 1e-15);
    }
}
