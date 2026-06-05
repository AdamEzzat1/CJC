//! Phase 2b — property tests for the GPU training simulator.

use cjc_nss::{
    ClusterTopology, FailureKind, GpuTrainingConfig, GpuTrainingSimulator, Intervention, NodeId,
    NssSeed, PressureKind,
};
use proptest::prelude::*;

fn gpu_config_strategy() -> impl Strategy<Value = GpuTrainingConfig> {
    (
        2u32..=8,        // n_gpus
        0.1f64..2.0,     // service_mean
        0.0f64..0.4,     // service_jitter
        0.0f64..0.5,     // allreduce_base
        1.0e7f64..5.0e9, // allreduce_bytes
        1.0e9f64..1.0e11,// nccl_bandwidth
        0.0f64..0.1,     // memory_per_microbatch
        2u32..=32,       // gc_interval
        0.0f64..0.05,    // fragmentation_growth
    )
        .prop_map(|(n, sm, sj, ab, abytes, bw, mpm, gci, fg)| GpuTrainingConfig {
            n_gpus: n,
            service_mean: sm,
            service_jitter: sj,
            allreduce_base: ab,
            allreduce_bytes: abytes,
            nccl_bandwidth: bw,
            memory_per_microbatch: mpm,
            gc_interval: gci,
            gc_recovery: 0.7,
            fragmentation_growth: fg,
            memory_capacity: 1.0,
            pipeline_stages: 1,
            microbatches_per_iteration: 1,
            ..GpuTrainingConfig::default()
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    /// Two simulators with the same (config, topology, seed) produce
    /// byte-identical trajectories.
    #[test]
    fn prop_determinism(
        cfg in gpu_config_strategy(),
        seed in any::<u64>(),
        n in 4usize..32,
    ) {
        let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
        let mut a = GpuTrainingSimulator::new(cfg, top.clone(), NssSeed(seed), vec![]).unwrap();
        let mut b = GpuTrainingSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        prop_assert_eq!(
            a.run(n as u64).unwrap().canonical_bytes(),
            b.run(n as u64).unwrap().canonical_bytes()
        );
    }

    /// Intervention-script ordering is irrelevant.
    #[test]
    fn prop_intervention_order_independence(
        seed in any::<u64>(),
        n_gpus in 2u32..=6,
        fail_tick in 0u64..16,
    ) {
        let cfg = GpuTrainingConfig { n_gpus, ..GpuTrainingConfig::default() };
        let top = ClusterTopology::complete(n_gpus, 8, 0.5).unwrap();
        let ivs_a = vec![
            Intervention::FailNode { tick: fail_tick, node: NodeId(0) },
            Intervention::FailNode { tick: fail_tick + 2, node: NodeId(1) },
        ];
        let ivs_b: Vec<_> = ivs_a.iter().rev().copied().collect();
        let mut a = GpuTrainingSimulator::new(cfg, top.clone(), NssSeed(seed), ivs_a).unwrap();
        let mut b = GpuTrainingSimulator::new(cfg, top, NssSeed(seed), ivs_b).unwrap();
        prop_assert_eq!(
            a.run(20).unwrap().canonical_bytes(),
            b.run(20).unwrap().canonical_bytes(),
        );
    }

    /// Pressure-field saturations stay in [0, 1.5] (the cap built
    /// into the simulator). No NaN, no infinities, no negatives.
    #[test]
    fn prop_pressure_saturations_bounded(
        cfg in gpu_config_strategy(),
        seed in any::<u64>(),
        n in 8usize..32,
    ) {
        let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
        let mut sim = GpuTrainingSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        let traj = sim.run(n as u64).unwrap();
        for ev in traj.iter() {
            for s in ev.state.nodes.values() {
                for k in PressureKind::all() {
                    let p = s.pressures.get(k).unwrap();
                    prop_assert!(p.magnitude.is_finite());
                    prop_assert!(p.magnitude >= 0.0);
                    prop_assert!(p.magnitude <= 1.5 + 1e-9);
                }
            }
        }
    }

    /// Iteration counts on healthy GPUs grow monotonically.
    #[test]
    fn prop_iteration_monotonicity(
        cfg in gpu_config_strategy(),
        seed in any::<u64>(),
        n in 4usize..16,
    ) {
        let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
        let mut sim = GpuTrainingSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
        let traj = sim.run(n as u64).unwrap();
        let mut prev: std::collections::BTreeMap<NodeId, u64> = std::collections::BTreeMap::new();
        for ev in traj.iter() {
            for (id, s) in ev.state.nodes.iter() {
                let last = *prev.get(id).unwrap_or(&0);
                prop_assert!(s.completed >= last, "iteration count went backwards");
                prev.insert(*id, s.completed);
            }
        }
    }

    /// Failure rollup is consistent with per-GPU labels.
    #[test]
    fn prop_failure_rollup_consistency(
        cfg in gpu_config_strategy(),
        seed in any::<u64>(),
        n in 4usize..16,
    ) {
        let top = ClusterTopology::complete(cfg.n_gpus, 8, 0.5).unwrap();
        let mut sim = GpuTrainingSimulator::new(cfg, top, NssSeed(seed), vec![]).unwrap();
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
}
