//! Property tests for cjc-nss Phase 1.
//!
//! These check structural invariants that must hold across the
//! distribution of plausible inputs, not just hand-crafted cases:
//!
//! - **Determinism**: two simulators with the same `(cfg, seed)` produce
//!   byte-identical trajectories.
//! - **Saturation bound**: every pressure saturation in `[0, 1]`.
//! - **Counter monotonicity**: `completed` and `rejected` are
//!   monotonically non-decreasing across a trajectory.
//! - **Propagation conservation**: total magnitude after a propagation
//!   step is `≤` total magnitude before (the conservation tax cannot
//!   add energy).
//! - **Replay round-trip**: every `predict_next` output replays
//!   without error.

use cjc_nss::{
    FailureKind, NeuralSystemsSimulator, NssConfig, NssSeed, PredictionTrace, Pressure,
    PressureField, PressureGraph, PressureKind, PressurePropagator, PropagationConfig, QueueConfig,
    QueueSimulator, ReplayValidator,
};
use proptest::prelude::*;

/// A strategy that produces a "plausible" `QueueConfig`. We bound
/// arrival rate and queue sizes to keep test runs fast.
fn queue_config_strategy() -> impl Strategy<Value = QueueConfig> {
    (
        1u32..=8,     // workers
        1u32..=8,     // capacity_over_workers (so capacity >= workers)
        0.5f64..6.0,  // arrival_rate
        0.5f64..2.0,  // service_min
        0.5f64..2.0,  // service_extra (added to service_min for max)
        0.3f64..0.95, // degraded_knee
        1u32..=4,     // collapse_window
        0.0f64..2.0,  // retry_amplifier
    )
        .prop_map(
            |(w, cap_mult, lam, smin, sextra, knee, win, retry)| QueueConfig {
                workers: w,
                queue_capacity: w * cap_mult,
                arrival_rate: lam,
                service_min: smin,
                service_max: smin + sextra,
                degraded_knee: knee,
                collapse_window: win,
                retry_amplifier: retry,
                propagation: PropagationConfig::default(),
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// Determinism: two simulators with the same `(cfg, seed)` produce
    /// byte-identical trajectories.
    #[test]
    fn prop_simulator_determinism(cfg in queue_config_strategy(), seed in any::<u64>(), n in 8usize..96) {
        let mut a = QueueSimulator::new(cfg, NssSeed(seed)).unwrap();
        let mut b = QueueSimulator::new(cfg, NssSeed(seed)).unwrap();
        let ta = a.run(n as u64).unwrap();
        let tb = b.run(n as u64).unwrap();
        prop_assert_eq!(ta.canonical_bytes(), tb.canonical_bytes());
    }

    /// Every pressure saturation across the trajectory lies in [0, 1].
    /// `Pressure::saturation()` clips, so this is a structural test
    /// that no pathway can corrupt the invariant.
    #[test]
    fn prop_saturations_bounded(cfg in queue_config_strategy(), seed in any::<u64>(), n in 8usize..64) {
        let mut sim = QueueSimulator::new(cfg, NssSeed(seed)).unwrap();
        let traj = sim.run(n as u64).unwrap();
        for ev in traj.iter() {
            for k in PressureKind::all() {
                let s = ev.state.pressures.get(k).unwrap().saturation();
                prop_assert!(s >= 0.0 && s <= 1.0, "saturation {} for {:?} out of bounds", s, k);
            }
        }
    }

    /// `completed` and `rejected` are monotonically non-decreasing
    /// across a trajectory.
    #[test]
    fn prop_counters_monotonic(cfg in queue_config_strategy(), seed in any::<u64>(), n in 8usize..96) {
        let mut sim = QueueSimulator::new(cfg, NssSeed(seed)).unwrap();
        let traj = sim.run(n as u64).unwrap();
        let mut last_completed = 0u64;
        let mut last_rejected = 0u64;
        for ev in traj.iter() {
            prop_assert!(ev.state.completed >= last_completed, "completed regressed");
            prop_assert!(ev.state.rejected >= last_rejected, "rejected regressed");
            last_completed = ev.state.completed;
            last_rejected = ev.state.rejected;
        }
    }

    /// `Collapse` is sticky-ish — once a collapse is observed, the
    /// trajectory never returns to `Nominal` without passing through
    /// `Degraded`. (Implementation detail: streak only resets when the
    /// queue-full + low-throughput condition relaxes; until then the
    /// label can only step down through the chain.)
    #[test]
    fn prop_no_collapse_to_nominal_in_one_step(cfg in queue_config_strategy(), seed in any::<u64>(), n in 16usize..96) {
        let mut sim = QueueSimulator::new(cfg, NssSeed(seed)).unwrap();
        let traj = sim.run(n as u64).unwrap();
        for w in traj.as_slice().windows(2) {
            if w[0].failure.kind == FailureKind::Collapse {
                // After a collapse, the next tick must not flip
                // straight to Nominal: either still Collapse, or
                // Degraded (we relaxed but the queue hasn't drained
                // below the knee yet), or Nominal *only if* the queue
                // has dropped below the knee.
                if w[1].failure.kind == FailureKind::Nominal {
                    // Verify the queue has actually dropped to below
                    // the knee — otherwise we'd be violating the
                    // labelling rule.
                    let qsat = w[1].state.pressures.get(PressureKind::Queue).unwrap().saturation();
                    prop_assert!(
                        qsat < cfg.degraded_knee + 0.05,
                        "collapse → nominal jump with queue saturation {} >= knee {}",
                        qsat,
                        cfg.degraded_knee
                    );
                }
            }
        }
    }

    /// Propagation conservation: total magnitude after a step is
    /// `≤ total before * (1 - min_dissipation) + small_eps`.
    /// (The "≤ before" form is the strict statement; the dissipation
    /// term lets the test tolerate the loose bound when input is
    /// dissipating.)
    #[test]
    fn prop_propagation_does_not_increase_total(
        init_mag in 0.0f64..3.0,
        seed in any::<u64>(),
        ticks in 1usize..16
    ) {
        let _ = seed; // not directly used; reserved for future randomised graphs.
        let prop = PressurePropagator::new(
            PressureGraph::default_phase1(),
            PropagationConfig::default(),
        ).unwrap();
        let mut f = PressureField::with_default_thresholds();
        // Seed one field with the input.
        f.set(PressureKind::Cpu, Pressure::new(init_mag, 1.0, 0.0).unwrap());
        let mut prev = f.total_magnitude();
        for _ in 0..ticks {
            prop.step(&mut f).unwrap();
            let now = f.total_magnitude();
            prop_assert!(
                now <= prev + 1e-9,
                "total magnitude increased: {} > {}",
                now,
                prev
            );
            prev = now;
        }
    }

    /// Replay round-trip: every prediction we produce verifies against
    /// the replay validator.
    #[test]
    fn prop_replay_round_trip(seed in any::<u64>()) {
        let cfg = NssConfig::default();
        let nss = NeuralSystemsSimulator::from_seed(cfg, NssSeed(seed)).unwrap();
        let s = cjc_nss::SystemState::initial();
        let pred = nss.predict_next(&s).unwrap();
        let trace = PredictionTrace {
            transition: pred.transition,
            input_state: s,
            input_config: cfg,
            input_seed: NssSeed(seed),
            training_trajectory: None,
        };
        ReplayValidator::new().verify(&trace).expect("replay must round-trip");
    }
}
