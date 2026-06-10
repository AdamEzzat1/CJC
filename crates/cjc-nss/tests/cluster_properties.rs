//! Phase 2 property tests for the cluster simulator + cluster NSS.

use cjc_nss::{
    ClusterConfig, ClusterNeuralSystemsSimulator, ClusterNssConfig, ClusterSimulator,
    ClusterTopology, FailureKind, Intervention, NodeId, NssSeed, RoutingPolicy,
};
use proptest::prelude::*;

fn cluster_config_strategy() -> impl Strategy<Value = ClusterConfig> {
    (
        1u32..=4,     // workers_per_node
        1u32..=8,     // capacity_multiplier (so capacity >= workers)
        0.5f64..8.0,  // cluster_arrival_rate
        0.5f64..2.0,  // service_min
        0.5f64..2.0,  // service_extra
        0.3f64..0.95, // degraded_knee
        1u32..=4,     // collapse_window
        prop_oneof![
            Just(RoutingPolicy::RoundRobin),
            Just(RoutingPolicy::LeastLoaded),
            Just(RoutingPolicy::HashPartition),
        ],
        0.05f64..0.5, // link_dissipation
    )
        .prop_map(
            |(w, mul, lam, smin, sextra, knee, win, routing, link_diss)| ClusterConfig {
                workers_per_node: w,
                queue_capacity: w * mul,
                cluster_arrival_rate: lam,
                service_min: smin,
                service_max: smin + sextra,
                degraded_knee: knee,
                collapse_window: win,
                routing,
                link_dissipation: link_diss,
                propagation: Default::default(),
            },
        )
}

fn small_topology_strategy() -> impl Strategy<Value = ClusterTopology> {
    (2u32..=6, 4u32..=12, 0.1f64..0.9).prop_map(|(n, cap, w)| {
        // Mix of complete and ring topologies.
        if n % 2 == 0 {
            ClusterTopology::complete(n, cap, w).unwrap()
        } else {
            ClusterTopology::ring(n, cap, w).unwrap()
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    /// Two simulators with the same (config, topology, seed,
    /// intervention script) produce byte-identical trajectories.
    #[test]
    fn prop_cluster_determinism(
        cfg in cluster_config_strategy(),
        top in small_topology_strategy(),
        seed in any::<u64>(),
        n in 8usize..32,
    ) {
        let mut a = ClusterSimulator::new(cfg, top.clone(), NssSeed(seed), vec![]).unwrap();
        let mut b = ClusterSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        let ta = a.run(n as u64).unwrap();
        let tb = b.run(n as u64).unwrap();
        prop_assert_eq!(ta.canonical_bytes(), tb.canonical_bytes());
    }

    /// Intervention script ordering is irrelevant — sorting it on
    /// entry means any permutation produces the same trajectory.
    #[test]
    fn prop_intervention_order_independence(
        seed in any::<u64>(),
        fail_tick in 0u64..16,
        recover_tick in 0u64..16,
    ) {
        let cfg = ClusterConfig::default();
        let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
        let mut ivs_a = vec![
            Intervention::FailNode { tick: fail_tick, node: NodeId(1) },
            Intervention::RecoverNode { tick: fail_tick.saturating_add(recover_tick.max(1)), node: NodeId(1) },
        ];
        let mut ivs_b = ivs_a.clone();
        ivs_b.reverse();
        // Add another intervention to make the permutation non-trivial.
        ivs_a.push(Intervention::FailNode { tick: 0, node: NodeId(2) });
        ivs_b.insert(0, Intervention::FailNode { tick: 0, node: NodeId(2) });
        let mut a = ClusterSimulator::new(cfg, top.clone(), NssSeed(seed), ivs_a).unwrap();
        let mut b = ClusterSimulator::new(cfg, top, NssSeed(seed), ivs_b).unwrap();
        prop_assert_eq!(a.run(20).unwrap().canonical_bytes(), b.run(20).unwrap().canonical_bytes());
    }

    /// Link congestion stays in [0, 1] across every tick of every run.
    #[test]
    fn prop_link_congestion_bounded(
        cfg in cluster_config_strategy(),
        top in small_topology_strategy(),
        seed in any::<u64>(),
        n in 8usize..32,
    ) {
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        let traj = sim.run(n as u64).unwrap();
        for ev in traj.iter() {
            for c in ev.state.link_congestion.values() {
                prop_assert!(c.is_finite() && *c >= 0.0 && *c <= 1.0);
            }
        }
    }

    /// Cluster-level failure label is consistent with the per-node
    /// labels: Collapse iff any node is Collapse; Degraded iff any
    /// node is Degraded (and none Collapse).
    #[test]
    fn prop_cluster_failure_rollup_consistency(
        cfg in cluster_config_strategy(),
        top in small_topology_strategy(),
        seed in any::<u64>(),
        n in 4usize..24,
    ) {
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        let traj = sim.run(n as u64).unwrap();
        for ev in traj.iter() {
            let any_collapse = ev.failures.values().any(|f| f.kind == FailureKind::Collapse);
            let any_degraded = ev.failures.values().any(|f| f.kind == FailureKind::Degraded);
            if any_collapse {
                prop_assert_eq!(ev.cluster_failure.kind, FailureKind::Collapse);
            } else if any_degraded {
                prop_assert_eq!(ev.cluster_failure.kind, FailureKind::Degraded);
            } else {
                prop_assert_eq!(ev.cluster_failure.kind, FailureKind::Nominal);
            }
        }
    }

    /// Cluster NSS predictions are always finite and bounded.
    #[test]
    fn prop_cluster_predictions_bounded(
        cfg in cluster_config_strategy(),
        top in small_topology_strategy(),
        seed in any::<u64>(),
        n in 8usize..24,
    ) {
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        let traj = sim.run(n as u64).unwrap();
        let nss = ClusterNeuralSystemsSimulator::from_seed(
            ClusterNssConfig::default(),
            NssSeed(seed),
        ).unwrap();
        let last = traj.last_state().unwrap();
        let pred = nss.predict_next(last).unwrap();
        prop_assert!(pred.failure.collapse_probability.is_finite());
        prop_assert!(pred.failure.collapse_probability >= 0.0 && pred.failure.collapse_probability <= 1.0);
        prop_assert!(pred.failure.degraded_probability >= 0.0 && pred.failure.degraded_probability <= 1.0);
    }

    /// Predictions are deterministic given (cfg, seed, input state).
    #[test]
    fn prop_cluster_prediction_determinism(
        seed in any::<u64>(),
        n in 8usize..24,
    ) {
        let sim_cfg = ClusterConfig::default();
        let top = ClusterTopology::complete(3, 6, 0.5).unwrap();
        let mut sim = ClusterSimulator::new(sim_cfg, top, NssSeed(seed), vec![]).unwrap();
        let traj = sim.run(n as u64).unwrap();
        let nss_a = ClusterNeuralSystemsSimulator::from_seed(
            ClusterNssConfig::default(),
            NssSeed(seed),
        ).unwrap();
        let nss_b = ClusterNeuralSystemsSimulator::from_seed(
            ClusterNssConfig::default(),
            NssSeed(seed),
        ).unwrap();
        let last = traj.last_state().unwrap();
        let pa = nss_a.predict_next(last).unwrap();
        let pb = nss_b.predict_next(last).unwrap();
        prop_assert_eq!(pa.run_id, pb.run_id);
        prop_assert_eq!(pa.failure.collapse_probability.to_bits(), pb.failure.collapse_probability.to_bits());
    }
}
