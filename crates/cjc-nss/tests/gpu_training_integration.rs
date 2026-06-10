//! Phase 2b — end-to-end GPU training tests.
//!
//! Verifies the GPU training simulator produces a `ClusterTrajectory`
//! that the Phase 2 `ClusterNeuralSystemsSimulator` can fit and predict
//! on with no changes — the architectural payoff of having a single
//! cluster trajectory type.

use cjc_nss::{
    ClusterNeuralSystemsSimulator, ClusterNssConfig, ClusterReplayValidator, ClusterTopology,
    ClusterTrace, FailureKind, GpuTrainingConfig, GpuTrainingSimulator, Intervention, NodeHealth,
    NodeId, NssSeed, PressureKind, NSS_MODEL_VERSION,
};

#[test]
fn gpu_training_emits_cluster_trajectory_consumable_by_nss() {
    let cfg = GpuTrainingConfig::default();
    let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
    let mut sim = GpuTrainingSimulator::new(cfg, top.clone(), NssSeed(42), vec![]).unwrap();
    let traj = sim.run(64).unwrap();

    let mut nss =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42)).unwrap();
    nss.fit(&traj).unwrap();
    let last = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&last).unwrap();
    assert!(pred.failure.collapse_probability.is_finite());
    assert!(pred.failure.collapse_probability >= 0.0 && pred.failure.collapse_probability <= 1.0);
    // The dominant node must be a real GPU in the topology.
    assert!(last.nodes.contains_key(&pred.attribution.dominant_node));
}

#[test]
fn gpu_training_replay_round_trip() {
    let cfg = GpuTrainingConfig::default();
    let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
    let seed = NssSeed(7);
    let mut sim = GpuTrainingSimulator::new(cfg, top.clone(), seed, vec![]).unwrap();
    let traj = sim.run(48).unwrap();
    let mut nss =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();
    nss.fit(&traj).unwrap();
    let last = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&last).unwrap();
    let trace = ClusterTrace {
        run_id: pred.run_id,
        input_hash: pred.input_hash,
        input_state: last,
        topology: top,
        // We reuse `ClusterConfig::default()` for the trace's
        // `simulator_config` field because the trace bundle's intent is
        // to identify the input bytes; the *GPU* config bytes aren't
        // part of the cluster NSS run-id — only the cluster NSS config
        // is. (The bytes still bind into the trace via `topology`
        // canonical bytes, which include node/edge counts.)
        simulator_config: cjc_nss::ClusterConfig::default(),
        intervention_script: vec![],
        nss_config: ClusterNssConfig::default(),
        seed,
        training_trajectory: Some(traj),
        collapse_probability: pred.failure.collapse_probability,
        degraded_probability: pred.failure.degraded_probability,
        model_version: NSS_MODEL_VERSION.to_string(),
    };
    ClusterReplayValidator::new()
        .verify(&trace)
        .expect("GPU-training replay must round-trip");
}

#[test]
fn oom_intervention_produces_cluster_collapse_label() {
    let cfg = GpuTrainingConfig {
        n_gpus: 4,
        ..GpuTrainingConfig::default()
    };
    let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let ivs = vec![Intervention::FailNode {
        tick: 6,
        node: NodeId(2),
    }];
    let mut sim = GpuTrainingSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
    let traj = sim.run(24).unwrap();
    // After tick 6, GPU 2 is failed and the cluster rollup is Collapse
    // (because any per-node Collapse forces the rollup).
    for ev in traj.iter().skip(6) {
        assert_eq!(ev.cluster_failure.kind, FailureKind::Collapse);
        assert_eq!(
            ev.state.node_health.get(&NodeId(2)).copied(),
            Some(NodeHealth::Failed)
        );
    }
}

#[test]
fn memory_pressure_eventually_dominates_long_runs_without_gc() {
    // No GC + heavy memory_per_microbatch + zero fragmentation_growth
    // → memory grows toward capacity, Memory pressure saturation rises.
    let cfg = GpuTrainingConfig {
        n_gpus: 2,
        gc_interval: 1_000_000, // effectively disabled
        memory_per_microbatch: 0.05,
        fragmentation_growth: 0.0,
        service_jitter: 0.01,
        ..GpuTrainingConfig::default()
    };
    let top = ClusterTopology::complete(2, 8, 0.5).unwrap();
    let mut sim = GpuTrainingSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
    let traj = sim.run(20).unwrap();
    // Memory pressure should be much higher at the end than the start.
    let mem_at = |i: usize| -> f64 {
        traj.as_slice()[i]
            .state
            .nodes
            .get(&NodeId(0))
            .unwrap()
            .pressures
            .get(PressureKind::Memory)
            .unwrap()
            .saturation()
    };
    let start = mem_at(2);
    let end = mem_at(19);
    assert!(
        end > start + 0.5,
        "memory pressure must climb (start={}, end={})",
        start,
        end
    );
}

#[test]
fn cluster_rollup_consistent_with_per_gpu_labels() {
    let cfg = GpuTrainingConfig {
        n_gpus: 4,
        memory_per_microbatch: 0.08,
        gc_interval: 1_000_000,
        ..GpuTrainingConfig::default()
    };
    let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let mut sim = GpuTrainingSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
    let traj = sim.run(24).unwrap();
    for ev in traj.iter() {
        let any_collapse = ev
            .failures
            .values()
            .any(|f| f.kind == FailureKind::Collapse);
        let any_degraded = ev
            .failures
            .values()
            .any(|f| f.kind == FailureKind::Degraded);
        if any_collapse {
            assert_eq!(ev.cluster_failure.kind, FailureKind::Collapse);
        } else if any_degraded {
            assert_eq!(ev.cluster_failure.kind, FailureKind::Degraded);
        } else {
            assert_eq!(ev.cluster_failure.kind, FailureKind::Nominal);
        }
    }
}

#[test]
fn intervention_changes_trajectory_bytes() {
    let cfg = GpuTrainingConfig::default();
    let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
    let mut clean = GpuTrainingSimulator::new(cfg, top.clone(), NssSeed(42), vec![]).unwrap();
    let mut perturbed = GpuTrainingSimulator::new(
        cfg,
        top,
        NssSeed(42),
        vec![Intervention::FailNode {
            tick: 4,
            node: NodeId(1),
        }],
    )
    .unwrap();
    let a = clean.run(20).unwrap();
    let b = perturbed.run(20).unwrap();
    assert_ne!(a.canonical_bytes(), b.canonical_bytes());
}
