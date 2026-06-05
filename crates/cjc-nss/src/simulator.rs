//! Deterministic Queue Simulator — the Phase 1 training-data source.
//!
//! Models a single-tier worker-pool / queue system with the canonical
//! pathologies the NSS architecture is designed to learn:
//!
//! - queue buildup under sustained over-arrival,
//! - retry storms (positive feedback on rejection),
//! - overload collapse (throughput drops as queue grows past a knee),
//! - scheduler shed-load decisions.
//!
//! ## Determinism
//!
//! Three RNG sub-streams (per-tick arrivals / per-task service time /
//! retry jitter) are seeded from the master [`crate::NssSeed`] with
//! distinct salts. Two runs with the same `(config, seed)` produce
//! byte-identical [`crate::SystemTrajectory`]s.
//!
//! ## What's modelled
//!
//! At each tick:
//! 1. RNG draws an arrival count from a clamped Poisson(λ).
//! 2. Arrivals are appended to the queue (subject to admission control).
//! 3. Workers each serve up to one task, with a per-task service time
//!    drawn from `Uniform(service_min, service_max)`.
//! 4. Pressure fields are updated:
//!    - `Queue` ← saturation of `queue_len` against `queue_capacity`.
//!    - `Cpu` ← workers-busy fraction.
//!    - `Sync` ← contention proxy: `min(1, queue_len * workers / capacity²)`.
//!    - `Throughput` ← `1 - tasks_served / arrivals` over a sliding window.
//!    - Retry storm: rejected tasks become "retries" that increase
//!      effective arrival rate next tick.
//! 5. The propagator runs one tick.
//! 6. The scheduler emits an action (shed-load if queue ≥ knee, idle
//!    otherwise — Phase 1 has no autoscaling).
//! 7. Failure-state label: `Collapse` if queue is full *and* throughput
//!    < 0.3 for `collapse_window` consecutive ticks; `Degraded` if
//!    queue ≥ knee; `Nominal` otherwise.
//!
//! Phase 2 will replace this with a richer distributed-cluster
//! simulator. The interface (configure → run → trajectory) stays.

use crate::error::NssError;
use crate::failure::FailureState;
use crate::pressure::{Pressure, PressureField, PressureKind};
use crate::propagation::{PressurePropagator, PropagationConfig};
use crate::pressure::PressureGraph;
use crate::scheduler::{SchedulerAction, SchedulerKind};
use crate::seed::NssSeed;
use crate::system::{SystemEvent, SystemState, SystemTrajectory};
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Knobs for the Phase 1 queue simulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueueConfig {
    /// Number of parallel workers. Must be ≥ 1.
    pub workers: u32,
    /// Maximum queue depth (admission-control cap). Must be ≥ workers.
    pub queue_capacity: u32,
    /// Mean arrivals per tick (Poisson λ). Must be ≥ 0 and finite.
    pub arrival_rate: f64,
    /// Per-task service time lower bound (uniform). Must be ≥ 0.
    pub service_min: f64,
    /// Per-task service time upper bound (uniform). Must be ≥
    /// `service_min`.
    pub service_max: f64,
    /// Queue-occupancy fraction at or above which the simulator labels
    /// the state `Degraded` and the scheduler starts shedding load.
    /// Must be in `[0, 1]`. Default 0.75.
    pub degraded_knee: f64,
    /// Ticks of "queue full + low throughput" required for the failure
    /// label to flip from `Degraded` to `Collapse`. Must be ≥ 1.
    pub collapse_window: u32,
    /// Multiplier on the next-tick effective arrival rate per rejected
    /// task (retry storm). 0 = no retries, 1 = each rejected task
    /// retries once next tick. Must be in `[0, 4]`.
    pub retry_amplifier: f64,
    /// Propagation knobs (default = Phase 1 default).
    pub propagation: PropagationConfig,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            queue_capacity: 32,
            arrival_rate: 4.0,
            service_min: 0.5,
            service_max: 1.5,
            degraded_knee: 0.75,
            collapse_window: 4,
            retry_amplifier: 0.5,
            propagation: PropagationConfig::default(),
        }
    }
}

impl QueueConfig {
    /// Validate. Called by [`QueueSimulator::new`].
    pub fn validate(&self) -> Result<(), NssError> {
        if self.workers == 0 {
            return Err(NssError::InvalidConfig {
                detail: "workers must be >= 1".into(),
            });
        }
        if self.queue_capacity < self.workers {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "queue_capacity ({}) must be >= workers ({})",
                    self.queue_capacity, self.workers
                ),
            });
        }
        if !self.arrival_rate.is_finite() || self.arrival_rate < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("arrival_rate must be finite and >= 0, got {}", self.arrival_rate),
            });
        }
        if !self.service_min.is_finite() || self.service_min < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("service_min must be finite and >= 0, got {}", self.service_min),
            });
        }
        if !self.service_max.is_finite() || self.service_max < self.service_min {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "service_max must be finite and >= service_min, got {}",
                    self.service_max
                ),
            });
        }
        if !self.degraded_knee.is_finite() || !(0.0..=1.0).contains(&self.degraded_knee) {
            return Err(NssError::InvalidConfig {
                detail: format!("degraded_knee must be in [0,1], got {}", self.degraded_knee),
            });
        }
        if self.collapse_window == 0 {
            return Err(NssError::InvalidConfig {
                detail: "collapse_window must be >= 1".into(),
            });
        }
        if !self.retry_amplifier.is_finite() || !(0.0..=4.0).contains(&self.retry_amplifier) {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "retry_amplifier must be in [0,4], got {}",
                    self.retry_amplifier
                ),
            });
        }
        self.propagation.validate()?;
        Ok(())
    }

    /// Canonical bytes (used by `NssRunId`).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&(self.workers as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.queue_capacity as u64).to_le_bytes());
        bytes.extend_from_slice(&self.arrival_rate.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.service_min.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.service_max.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.degraded_knee.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(self.collapse_window as u64).to_le_bytes());
        bytes.extend_from_slice(&self.retry_amplifier.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.propagation.canonical_bytes());
        bytes
    }
}

/// One snapshot of internal simulator counters. Exposed for tests and
/// for the `QueueSimulator::run_with_snapshots` helper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueueSnapshot {
    /// Tick index.
    pub tick: u64,
    /// Current queue length (tasks waiting).
    pub queue_len: u32,
    /// Tasks that arrived this tick.
    pub arrivals: u32,
    /// Tasks served this tick.
    pub served: u32,
    /// Tasks rejected this tick (admission control + shed-load).
    pub rejected: u32,
    /// Retries pending for the *next* tick.
    pub pending_retries: u32,
    /// Throughput proxy at this tick: served / max(1, arrivals).
    pub throughput: f64,
}

/// The simulator itself.
#[derive(Debug)]
pub struct QueueSimulator {
    cfg: QueueConfig,
    seed: NssSeed,
    propagator: PressurePropagator,
    rng_arrivals: Rng,
    rng_service: Rng,
    rng_retries: Rng,
    queue_len: u32,
    pending_retries: u32,
    completed: u64,
    rejected: u64,
    tick: u64,
    /// Rolling count of consecutive "queue-full + low-throughput" ticks
    /// for the collapse-window threshold.
    collapse_streak: u32,
    /// Last observed throughput proxy.
    last_throughput: f64,
    field: PressureField,
}

impl QueueSimulator {
    /// Build a simulator. Validates the config.
    pub fn new(cfg: QueueConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let propagator =
            PressurePropagator::new(PressureGraph::default_phase1(), cfg.propagation)?;
        Ok(Self {
            cfg,
            seed,
            propagator,
            rng_arrivals: seed.substream("queue_arrivals"),
            rng_service: seed.substream("queue_service"),
            rng_retries: seed.substream("queue_retries"),
            queue_len: 0,
            pending_retries: 0,
            completed: 0,
            rejected: 0,
            tick: 0,
            collapse_streak: 0,
            last_throughput: 1.0,
            field: PressureField::with_default_thresholds(),
        })
    }

    /// Run the simulator for `n_ticks` and return the trajectory.
    pub fn run(&mut self, n_ticks: u64) -> Result<SystemTrajectory, NssError> {
        let mut traj = SystemTrajectory::empty();
        for _ in 0..n_ticks {
            let ev = self.step_once()?;
            traj.push(ev)?;
        }
        Ok(traj)
    }

    /// Run the simulator for `n_ticks` and also return per-tick
    /// internal snapshots — useful for tests that assert on counters
    /// the trajectory doesn't directly expose.
    pub fn run_with_snapshots(
        &mut self,
        n_ticks: u64,
    ) -> Result<(SystemTrajectory, Vec<QueueSnapshot>), NssError> {
        let mut traj = SystemTrajectory::empty();
        let mut snaps = Vec::with_capacity(n_ticks as usize);
        for _ in 0..n_ticks {
            let (ev, snap) = self.step_once_with_snapshot()?;
            traj.push(ev)?;
            snaps.push(snap);
        }
        Ok((traj, snaps))
    }

    fn step_once(&mut self) -> Result<SystemEvent, NssError> {
        let (ev, _snap) = self.step_once_with_snapshot()?;
        Ok(ev)
    }

    fn step_once_with_snapshot(&mut self) -> Result<(SystemEvent, QueueSnapshot), NssError> {
        // 1. Arrivals: clamped Poisson(λ + retry_amplifier * retries).
        let effective_rate =
            self.cfg.arrival_rate + self.cfg.retry_amplifier * self.pending_retries as f64;
        let arrivals = poisson_clamped(&mut self.rng_arrivals, effective_rate, 4 * self.cfg.queue_capacity);
        self.pending_retries = 0;

        // 2. Admission: queue_capacity - queue_len slots free.
        let free = self.cfg.queue_capacity.saturating_sub(self.queue_len);
        // Shed-load if queue ≥ knee: only accept `1 - shed_fraction` of
        // arrivals. Shed-fraction grows linearly from 0 at the knee to
        // 1 when the queue is full.
        let occupancy = self.queue_len as f64 / self.cfg.queue_capacity as f64;
        let shed_fraction = if occupancy >= self.cfg.degraded_knee {
            ((occupancy - self.cfg.degraded_knee) / (1.0 - self.cfg.degraded_knee + f64::EPSILON))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let admitted_after_shed = ((arrivals as f64) * (1.0 - shed_fraction)).floor() as u32;
        let admitted = admitted_after_shed.min(free);
        let rejected = arrivals.saturating_sub(admitted);
        self.queue_len += admitted;
        // Peak queue length *before* the workers drain — this is the
        // right quantity for the collapse-detection check below, since
        // post-serve the queue is always at most `capacity - workers`
        // even under fully-saturated conditions.
        let peak_queue_len = self.queue_len;

        // 3. Service: each worker pops at most one task per tick. Mean
        // service time is recorded for the SystemState; the rate is
        // implicit (one task per worker per tick) in Phase 1.
        let to_serve = self.queue_len.min(self.cfg.workers);
        let mut service_acc = KahanAccumulatorF64::new();
        for _ in 0..to_serve {
            let t = self.cfg.service_min
                + (self.cfg.service_max - self.cfg.service_min) * self.rng_service.next_f64();
            service_acc.add(t);
        }
        self.queue_len -= to_serve;
        self.completed = self.completed.saturating_add(to_serve as u64);
        self.rejected = self.rejected.saturating_add(rejected as u64);
        let mean_service_time = if to_serve == 0 {
            0.0
        } else {
            service_acc.finalize() / to_serve as f64
        };

        // 4. Throughput proxy: served / max(1, arrivals + queue_len_at_start).
        // We use admitted as the work-eligible count instead of raw arrivals
        // to keep this bounded in `[0, 1]` and meaningful when arrivals=0.
        let denom = (admitted + self.queue_len + to_serve).max(1) as f64;
        let throughput = to_serve as f64 / denom;
        self.last_throughput = throughput;

        // 5. Translate the per-tick numbers into pressure-field updates.
        let queue_p = (self.queue_len as f64) / (self.cfg.queue_capacity as f64);
        let cpu_p = (to_serve as f64) / (self.cfg.workers as f64);
        // Sync contention: grows with sqrt(queue_len * workers).
        let sync_p =
            ((self.queue_len as f64 * self.cfg.workers as f64).sqrt() / self.cfg.queue_capacity as f64)
                .min(1.5);
        // Throughput "pressure" rises *as* throughput falls.
        let thr_p = (1.0 - throughput).max(0.0);

        self.field.set(
            PressureKind::Queue,
            Pressure::new(queue_p, 1.0, 0.05)?,
        );
        self.field.set(
            PressureKind::Cpu,
            Pressure::new(cpu_p, 1.0, 0.1)?,
        );
        self.field.set(
            PressureKind::Sync,
            Pressure::new(sync_p, 1.0, 0.08)?,
        );
        self.field.set(
            PressureKind::Throughput,
            Pressure::new(thr_p, 1.0, 0.05)?,
        );

        // 6. Propagate.
        let _flows = self.propagator.step(&mut self.field)?;

        // 7. Scheduler action.
        let action = if occupancy >= self.cfg.degraded_knee {
            SchedulerAction::new(SchedulerKind::ShedLoad, shed_fraction)
        } else {
            SchedulerAction::idle()
        };

        // 8. Failure label. Use the pre-serve peak so a fully-saturated
        // tick (where workers drain the queue back below capacity)
        // still counts as "queue was full this tick".
        let queue_full = peak_queue_len >= self.cfg.queue_capacity;
        let low_throughput = throughput < 0.3;
        if queue_full && low_throughput {
            self.collapse_streak += 1;
        } else {
            self.collapse_streak = 0;
        }
        let failure = if self.collapse_streak >= self.cfg.collapse_window {
            FailureState::collapse(PressureKind::Queue)
        } else if occupancy >= self.cfg.degraded_knee {
            FailureState::degraded(PressureKind::Queue)
        } else {
            FailureState::nominal()
        };

        // 9. Retries pending for next tick: deterministic fraction of
        // this tick's rejections, jittered by the retry RNG so two
        // simulators with the same seed but different retry rates
        // still diverge on retry-tick-by-tick decisions.
        let retry_jitter = self.rng_retries.next_f64();
        let retry_count =
            ((rejected as f64) * self.cfg.retry_amplifier.min(1.0) * (0.5 + retry_jitter)).floor()
                as u32;
        self.pending_retries = retry_count;

        // 10. Materialise SystemState + SystemEvent.
        let state = SystemState {
            tick: self.tick,
            pressures: self.field.clone(),
            in_flight: self.queue_len as u64 + to_serve as u64,
            completed: self.completed,
            rejected: self.rejected,
            mean_service_time,
        };
        let ev = SystemEvent::new(state, action, failure)?;
        let snap = QueueSnapshot {
            tick: self.tick,
            queue_len: self.queue_len,
            arrivals,
            served: to_serve,
            rejected,
            pending_retries: self.pending_retries,
            throughput,
        };
        self.tick += 1;
        Ok((ev, snap))
    }

    /// Canonical bytes for the simulator config (used by `NssRunId`).
    pub fn config_canonical_bytes(&self) -> Vec<u8> {
        self.cfg.canonical_bytes()
    }

    /// Seed accessor (used by audit traces).
    pub fn seed(&self) -> NssSeed {
        self.seed
    }
}

/// Inverse-CDF Poisson(λ) draw, clamped to `[0, max]`. Deterministic
/// given the RNG state. For λ ≤ 0 returns 0.
fn poisson_clamped(rng: &mut Rng, lambda: f64, max: u32) -> u32 {
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0;
    }
    // Knuth's algorithm: count uniforms until their product is below
    // exp(-λ). Stable for λ up to about 30; for larger λ we'd switch
    // to a normal approximation, but Phase 1 caps `arrival_rate` at a
    // sensible single-tier load.
    let l = (-lambda).exp();
    let mut k: u32 = 0;
    let mut p = 1.0f64;
    loop {
        if k >= max {
            return max;
        }
        let u = rng.next_f64();
        p *= u;
        if p <= l {
            return k;
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_validates() {
        assert!(QueueConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_invalid_fields() {
        let mut c = QueueConfig::default();
        c.workers = 0;
        assert!(c.validate().is_err());
        let mut c = QueueConfig::default();
        c.queue_capacity = 1; // < workers (4)
        assert!(c.validate().is_err());
        let mut c = QueueConfig::default();
        c.service_min = 1.0;
        c.service_max = 0.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn determinism_same_seed_same_trajectory() {
        let cfg = QueueConfig::default();
        let mut sim_a = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let mut sim_b = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let t_a = sim_a.run(64).unwrap();
        let t_b = sim_b.run(64).unwrap();
        assert_eq!(t_a.canonical_bytes(), t_b.canonical_bytes());
    }

    #[test]
    fn determinism_different_seed_different_trajectory() {
        let cfg = QueueConfig::default();
        let mut sim_a = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let mut sim_b = QueueSimulator::new(cfg, NssSeed(43)).unwrap();
        let t_a = sim_a.run(64).unwrap();
        let t_b = sim_b.run(64).unwrap();
        assert_ne!(t_a.canonical_bytes(), t_b.canonical_bytes());
    }

    #[test]
    fn trajectory_ticks_are_monotonic() {
        let mut sim = QueueSimulator::new(QueueConfig::default(), NssSeed(42)).unwrap();
        let t = sim.run(20).unwrap();
        for (i, ev) in t.iter().enumerate() {
            assert_eq!(ev.state.tick, i as u64);
        }
    }

    #[test]
    fn overload_drives_collapse_label() {
        // Configure a system that's guaranteed to collapse: arrivals
        // far exceed service capacity, small queue, retries amplify.
        let cfg = QueueConfig {
            workers: 2,
            queue_capacity: 8,
            arrival_rate: 20.0,
            service_min: 1.0,
            service_max: 1.0,
            degraded_knee: 0.5,
            collapse_window: 2,
            retry_amplifier: 1.0,
            propagation: PropagationConfig::default(),
        };
        let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let t = sim.run(64).unwrap();
        let collapses = t
            .iter()
            .filter(|ev| ev.failure.kind == crate::FailureKind::Collapse)
            .count();
        assert!(collapses > 0, "expected at least one Collapse label");
    }

    #[test]
    fn nominal_load_produces_no_collapse() {
        // arrival_rate < workers * 1 (service mean = 1.0), so queue
        // empties faster than it fills.
        let cfg = QueueConfig {
            workers: 4,
            queue_capacity: 32,
            arrival_rate: 1.0,
            service_min: 1.0,
            service_max: 1.0,
            degraded_knee: 0.75,
            collapse_window: 4,
            retry_amplifier: 0.0,
            propagation: PropagationConfig::default(),
        };
        let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let t = sim.run(128).unwrap();
        let collapses = t
            .iter()
            .filter(|ev| ev.failure.kind == crate::FailureKind::Collapse)
            .count();
        assert_eq!(collapses, 0, "nominal load must never collapse");
    }

    #[test]
    fn snapshots_align_with_trajectory_length() {
        let mut sim = QueueSimulator::new(QueueConfig::default(), NssSeed(42)).unwrap();
        let (t, snaps) = sim.run_with_snapshots(32).unwrap();
        assert_eq!(t.len(), snaps.len());
        for (i, (ev, snap)) in t.iter().zip(snaps.iter()).enumerate() {
            assert_eq!(ev.state.tick, snap.tick, "tick mismatch at {}", i);
        }
    }

    #[test]
    fn shed_load_action_fires_above_knee() {
        // Trigger high occupancy.
        let cfg = QueueConfig {
            workers: 1,
            queue_capacity: 4,
            arrival_rate: 10.0,
            service_min: 1.0,
            service_max: 1.0,
            degraded_knee: 0.25,
            collapse_window: 2,
            retry_amplifier: 0.0,
            propagation: PropagationConfig::default(),
        };
        let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let t = sim.run(32).unwrap();
        let any_shed = t
            .iter()
            .any(|ev| ev.action.kind == SchedulerKind::ShedLoad);
        assert!(any_shed, "expected shed-load actions above the knee");
    }

    #[test]
    fn finiteness_holds_for_long_runs() {
        let cfg = QueueConfig {
            workers: 2,
            queue_capacity: 16,
            arrival_rate: 6.0,
            service_min: 0.5,
            service_max: 1.5,
            degraded_knee: 0.5,
            collapse_window: 3,
            retry_amplifier: 0.8,
            propagation: PropagationConfig::default(),
        };
        let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let t = sim.run(512).unwrap();
        for ev in t.iter() {
            assert!(ev.state.pressures.all_finite());
            for p in PressureKind::all() {
                let v = ev.state.pressures.get(p).unwrap().magnitude;
                assert!(v.is_finite() && v >= 0.0, "non-finite or negative {:?}={}", p, v);
            }
        }
    }
}
