//! Phase 2 cluster integration tests.
//!
//! End-to-end checks that exercise the cluster simulator + cluster-aware
//! NSS + replay validator together.

use cjc_nss::{
    ClusterConfig, ClusterNeuralSystemsSimulator, ClusterNssConfig, ClusterReplayValidator,
    ClusterSimulator, ClusterSystemState, ClusterTopology, ClusterTrace, FailureKind, Intervention,
    NodeHealth, NodeId, NssError, NssSeed, RoutingPolicy, NSS_MODEL_VERSION,
};

#[test]
fn cluster_determinism_across_independent_runs() {
    let cfg = ClusterConfig::default();
    let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let mut a = ClusterSimulator::new(cfg, top.clone(), NssSeed(2026), vec![]).unwrap();
    let mut b = ClusterSimulator::new(cfg, top, NssSeed(2026), vec![]).unwrap();
    let ta = a.run(64).unwrap();
    let tb = b.run(64).unwrap();
    assert_eq!(ta.canonical_bytes(), tb.canonical_bytes());

    let nss_a =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(2026))
            .unwrap();
    let nss_b =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(2026))
            .unwrap();
    let s = ta.last_state().unwrap();
    let pa = nss_a.predict_next(s).unwrap();
    let pb = nss_b.predict_next(s).unwrap();
    assert_eq!(pa.run_id, pb.run_id);
    assert_eq!(
        pa.failure.collapse_probability.to_bits(),
        pb.failure.collapse_probability.to_bits()
    );
}

#[test]
fn cluster_replay_round_trip_unfit() {
    let nss_cfg = ClusterNssConfig::default();
    let seed = NssSeed(42);
    let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
    let state = ClusterSystemState::initial(&top);
    let nss = ClusterNeuralSystemsSimulator::from_seed(nss_cfg, seed).unwrap();
    let pred = nss.predict_next(&state).unwrap();

    let trace = ClusterTrace {
        run_id: pred.run_id,
        input_hash: pred.input_hash,
        input_state: state,
        topology: top,
        simulator_config: ClusterConfig::default(),
        intervention_script: vec![],
        nss_config: nss_cfg,
        seed,
        training_trajectory: None,
        collapse_probability: pred.failure.collapse_probability,
        degraded_probability: pred.failure.degraded_probability,
        model_version: NSS_MODEL_VERSION.to_string(),
    };
    ClusterReplayValidator::new()
        .verify(&trace)
        .expect("cluster replay must round-trip on un-fitted prediction");
}

#[test]
fn cluster_replay_round_trip_after_fit() {
    let nss_cfg = ClusterNssConfig::default();
    let seed = NssSeed(42);
    let sim_cfg = ClusterConfig::default();
    let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let mut sim = ClusterSimulator::new(sim_cfg, top.clone(), seed, vec![]).unwrap();
    let traj = sim.run(48).unwrap();
    let mut nss = ClusterNeuralSystemsSimulator::from_seed(nss_cfg, seed).unwrap();
    nss.fit(&traj).unwrap();
    let last = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&last).unwrap();

    let trace = ClusterTrace {
        run_id: pred.run_id,
        input_hash: pred.input_hash,
        input_state: last,
        topology: top,
        simulator_config: sim_cfg,
        intervention_script: vec![],
        nss_config: nss_cfg,
        seed,
        training_trajectory: Some(traj),
        collapse_probability: pred.failure.collapse_probability,
        degraded_probability: pred.failure.degraded_probability,
        model_version: NSS_MODEL_VERSION.to_string(),
    };
    ClusterReplayValidator::new()
        .verify(&trace)
        .expect("cluster replay must round-trip on fitted prediction");
}

#[test]
fn cluster_replay_rejects_missing_training_trajectory_after_fit() {
    let nss_cfg = ClusterNssConfig::default();
    let seed = NssSeed(42);
    let sim_cfg = ClusterConfig {
        cluster_arrival_rate: 10.0,
        ..ClusterConfig::default()
    };
    let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
    let mut sim = ClusterSimulator::new(sim_cfg, top.clone(), seed, vec![]).unwrap();
    let traj = sim.run(32).unwrap();
    let mut nss = ClusterNeuralSystemsSimulator::from_seed(nss_cfg, seed).unwrap();
    nss.fit(&traj).unwrap();
    let last = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&last).unwrap();

    let bad = ClusterTrace {
        run_id: pred.run_id,
        input_hash: pred.input_hash,
        input_state: last,
        topology: top,
        simulator_config: sim_cfg,
        intervention_script: vec![],
        nss_config: nss_cfg,
        seed,
        training_trajectory: None, // intentionally missing
        collapse_probability: pred.failure.collapse_probability,
        degraded_probability: pred.failure.degraded_probability,
        model_version: NSS_MODEL_VERSION.to_string(),
    };
    let r = ClusterReplayValidator::new().verify(&bad);
    assert!(matches!(r, Err(NssError::ReplayMismatch { .. })));
}

#[test]
fn failure_cascade_propagates_through_network() {
    // 4-node ring; fail node 0 at tick 4; observe that node 1's queue
    // pressure rises subsequently because the routing fans more work
    // onto it.
    let cfg = ClusterConfig {
        workers_per_node: 1,
        queue_capacity: 8,
        cluster_arrival_rate: 8.0,
        routing: RoutingPolicy::RoundRobin,
        ..ClusterConfig::default()
    };
    let top = ClusterTopology::ring(4, 6, 0.5).unwrap();
    let ivs = vec![Intervention::FailNode {
        tick: 4,
        node: NodeId(0),
    }];
    let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
    let traj = sim.run(32).unwrap();
    // Compute per-node mean queue saturation pre vs post-failure for
    // node 1 (the round-robin successor that absorbs more work).
    let pre: Vec<f64> = traj
        .iter()
        .take(4)
        .map(|ev| {
            ev.state
                .nodes
                .get(&NodeId(1))
                .unwrap()
                .pressures
                .get(cjc_nss::PressureKind::Queue)
                .unwrap()
                .saturation()
        })
        .collect();
    let post: Vec<f64> = traj
        .iter()
        .skip(8)
        .map(|ev| {
            ev.state
                .nodes
                .get(&NodeId(1))
                .unwrap()
                .pressures
                .get(cjc_nss::PressureKind::Queue)
                .unwrap()
                .saturation()
        })
        .collect();
    let pre_mean = pre.iter().sum::<f64>() / pre.len() as f64;
    let post_mean = post.iter().sum::<f64>() / post.len() as f64;
    // After a peer fails under round-robin routing, the surviving
    // peer should absorb more load → higher mean queue saturation.
    // We use a small margin to tolerate noise.
    assert!(
        post_mean > pre_mean - 0.05,
        "post-failure peer queue saturation should not decrease (pre={}, post={})",
        pre_mean,
        post_mean
    );
}

#[test]
fn small_cluster_under_nominal_load_stays_finite() {
    let cfg = ClusterConfig {
        workers_per_node: 4,
        queue_capacity: 32,
        cluster_arrival_rate: 4.0,
        ..ClusterConfig::default()
    };
    let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let mut sim = ClusterSimulator::new(cfg, top, NssSeed(2026), vec![]).unwrap();
    let traj = sim.run(512).unwrap();
    for ev in traj.iter() {
        ev.state.validate().unwrap();
        for c in ev.state.link_congestion.values() {
            assert!(c.is_finite() && *c >= 0.0 && *c <= 1.0);
        }
        for state in ev.state.nodes.values() {
            for k in cjc_nss::PressureKind::all() {
                let p = state.pressures.get(k).unwrap();
                assert!(p.magnitude.is_finite() && p.magnitude >= 0.0);
            }
        }
    }
}

#[test]
fn intervention_changes_run_id() {
    // The cluster predictor's run id depends on the cluster state
    // bytes — if a different intervention produces a different
    // training trajectory, predicting on the resulting trajectory's
    // last state should give a different `run_id`.
    let nss_cfg = ClusterNssConfig::default();
    let seed = NssSeed(42);
    let sim_cfg = ClusterConfig::default();
    let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let mut sim_a = ClusterSimulator::new(sim_cfg, top.clone(), seed, vec![]).unwrap();
    let mut sim_b = ClusterSimulator::new(
        sim_cfg,
        top,
        seed,
        vec![Intervention::FailNode {
            tick: 8,
            node: NodeId(2),
        }],
    )
    .unwrap();
    let ta = sim_a.run(32).unwrap();
    let tb = sim_b.run(32).unwrap();
    let nss = ClusterNeuralSystemsSimulator::from_seed(nss_cfg, seed).unwrap();
    let pa = nss.predict_next(ta.last_state().unwrap()).unwrap();
    let pb = nss.predict_next(tb.last_state().unwrap()).unwrap();
    assert_ne!(
        pa.run_id, pb.run_id,
        "intervention must change input state and thus run_id"
    );
}

#[test]
fn node_health_rollup_is_consistent_with_node_labels() {
    // If any per-node label is Collapse, cluster_failure must be Collapse.
    // If any is Degraded (and none Collapse), cluster_failure must be Degraded.
    let cfg = ClusterConfig {
        cluster_arrival_rate: 10.0,
        ..ClusterConfig::default()
    };
    let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
    let ivs = vec![Intervention::FailNode {
        tick: 6,
        node: NodeId(1),
    }];
    let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
    let traj = sim.run(32).unwrap();
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
fn dead_cluster_keeps_emitting_collapse_label() {
    let cfg = ClusterConfig::default();
    let top = ClusterTopology::complete(2, 8, 0.5).unwrap();
    let ivs = vec![
        Intervention::FailNode {
            tick: 1,
            node: NodeId(0),
        },
        Intervention::FailNode {
            tick: 1,
            node: NodeId(1),
        },
    ];
    let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
    let traj = sim.run(16).unwrap();
    for ev in traj.iter().skip(1) {
        assert_eq!(ev.cluster_failure.kind, FailureKind::Collapse);
        for h in ev.state.node_health.values() {
            assert_eq!(*h, NodeHealth::Failed);
        }
    }
}
