//! Top-level integration tests for cjc-nss.
//!
//! These cover the end-to-end Phase 1 contract:
//! 1. Simulator → Trajectory → NSS.fit → predict → replay round-trip.
//! 2. Determinism across two independent NSS instances built from the
//!    same `(NssConfig, NssSeed)`.
//! 3. Failure-label semantics under three load regimes (nominal,
//!    transient overload, sustained collapse).
//! 4. Pressure-graph propagation invariants on long runs.

use cjc_nss::{
    FailureKind, NeuralSystemsSimulator, NssConfig, NssError, NssSeed,
    PredictionTrace, PressureField, PressureKind, PressureGraph, PressurePropagator,
    PropagationConfig, QueueConfig, QueueSimulator, ReplayValidator,
};

#[test]
fn end_to_end_simulator_to_replay() {
    let seed = NssSeed(2026);
    let mut sim = QueueSimulator::new(QueueConfig::default(), seed).unwrap();
    let traj = sim.run(96).unwrap();

    let cfg = NssConfig::default();
    let mut nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
    nss.fit(&traj).unwrap();
    let s = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&s).unwrap();

    let trace = PredictionTrace {
        transition: pred.transition,
        input_state: s,
        input_config: cfg,
        input_seed: seed,
        training_trajectory: Some(traj),
    };
    ReplayValidator::new().verify(&trace).expect("replay should succeed");
}

#[test]
fn determinism_two_independent_runs() {
    let seed = NssSeed(7);
    let cfg = NssConfig::default();
    let qcfg = QueueConfig::default();

    let mut sim_a = QueueSimulator::new(qcfg, seed).unwrap();
    let mut sim_b = QueueSimulator::new(qcfg, seed).unwrap();
    let traj_a = sim_a.run(64).unwrap();
    let traj_b = sim_b.run(64).unwrap();
    assert_eq!(traj_a.canonical_bytes(), traj_b.canonical_bytes());

    let mut nss_a = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
    let mut nss_b = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
    nss_a.fit(&traj_a).unwrap();
    nss_b.fit(&traj_b).unwrap();

    let s = traj_a.last_state().unwrap().clone();
    let pa = nss_a.predict_next(&s).unwrap();
    let pb = nss_b.predict_next(&s).unwrap();
    assert_eq!(pa.run_id, pb.run_id);
    assert_eq!(
        pa.failure.collapse_probability.to_bits(),
        pb.failure.collapse_probability.to_bits()
    );
    assert_eq!(
        pa.failure.degraded_probability.to_bits(),
        pb.failure.degraded_probability.to_bits()
    );
}

#[test]
fn nominal_load_keeps_failure_label_nominal() {
    let cfg = QueueConfig {
        workers: 4,
        queue_capacity: 64,
        arrival_rate: 0.5,
        service_min: 1.0,
        service_max: 1.0,
        degraded_knee: 0.75,
        collapse_window: 4,
        retry_amplifier: 0.0,
        propagation: PropagationConfig::default(),
    };
    let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
    let t = sim.run(256).unwrap();
    let collapses = t
        .iter()
        .filter(|ev| ev.failure.kind == FailureKind::Collapse)
        .count();
    assert_eq!(collapses, 0, "nominal load should never collapse");
}

#[test]
fn sustained_overload_eventually_collapses() {
    let cfg = QueueConfig {
        workers: 1,
        queue_capacity: 4,
        arrival_rate: 6.0,
        service_min: 1.0,
        service_max: 1.0,
        degraded_knee: 0.5,
        collapse_window: 2,
        retry_amplifier: 0.5,
        propagation: PropagationConfig::default(),
    };
    let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
    let t = sim.run(64).unwrap();
    let collapses = t
        .iter()
        .filter(|ev| ev.failure.kind == FailureKind::Collapse)
        .count();
    assert!(collapses > 4, "sustained overload should collapse repeatedly");
}

#[test]
fn long_run_pressures_stay_finite_and_bounded() {
    // The structural-stability claim (Phase-1 default graph) must hold
    // under a long, varied load.
    let cfg = QueueConfig {
        workers: 2,
        queue_capacity: 32,
        arrival_rate: 4.0,
        service_min: 0.5,
        service_max: 1.5,
        degraded_knee: 0.7,
        collapse_window: 4,
        retry_amplifier: 0.5,
        propagation: PropagationConfig::default(),
    };
    let mut sim = QueueSimulator::new(cfg, NssSeed(2026)).unwrap();
    let t = sim.run(2048).unwrap();
    for ev in t.iter() {
        for k in PressureKind::all() {
            let p = ev.state.pressures.get(k).expect("kind set");
            assert!(p.magnitude.is_finite(), "{:?} non-finite at tick {}", k, ev.state.tick);
            assert!(p.magnitude >= 0.0, "{:?} negative at tick {}", k, ev.state.tick);
            // magnitude_cap defaults to 1e6.
            assert!(p.magnitude <= 1e6 + 1e-9);
        }
    }
}

#[test]
fn replay_rejects_tampered_input_state() {
    let seed = NssSeed(42);
    let nss = NeuralSystemsSimulator::from_seed(NssConfig::default(), seed).unwrap();
    let s = cjc_nss::SystemState::initial();
    let pred = nss.predict_next(&s).unwrap();
    let mut tampered = s.clone();
    // Bump in_flight — changes canonical bytes → run_id mismatch.
    tampered.in_flight = 999;
    let trace = PredictionTrace {
        transition: pred.transition,
        input_state: tampered,
        input_config: NssConfig::default(),
        input_seed: seed,
        training_trajectory: None,
    };
    let r = ReplayValidator::new().verify(&trace);
    assert!(matches!(r, Err(NssError::ReplayMismatch { .. })));
}

#[test]
fn propagator_steady_state_under_constant_input_is_finite() {
    let graph = PressureGraph::default_phase1();
    let prop = PressurePropagator::new(graph, PropagationConfig::default()).unwrap();
    let mut f = PressureField::with_default_thresholds();
    // Constant high pressure on Cpu; observe propagation to Queue.
    f.set(
        PressureKind::Cpu,
        cjc_nss::Pressure::new(0.8, 1.0, 0.05).unwrap(),
    );
    for _ in 0..256 {
        // Refill Cpu each tick so we have a steady "source".
        let p = f.get(PressureKind::Cpu).unwrap();
        let mut p2 = *p;
        p2.magnitude = 0.8;
        f.set(PressureKind::Cpu, p2);
        prop.step(&mut f).unwrap();
    }
    // Queue should have grown above zero; throughput should also rise.
    let queue = f.get(PressureKind::Queue).unwrap().magnitude;
    assert!(queue > 0.0, "expected propagation to lift queue pressure, got {}", queue);
    assert!(queue.is_finite());
}
