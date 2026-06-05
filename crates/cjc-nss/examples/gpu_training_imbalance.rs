//! Phase 2b end-to-end demo: 8-GPU data-parallel training with a
//! mid-run OOM event. We run the GPU training simulator, observe
//! batch-imbalance + memory-pressure dynamics, fit the cluster-aware
//! NSS, predict imminent training collapse with per-GPU attribution,
//! and verify the prediction replays byte-identically.
//!
//! Run with:
//! ```text
//! cargo run --example gpu_training_imbalance -p cjc-nss
//! ```

use cjc_nss::{
    cluster_summary_label, ClusterNeuralSystemsSimulator, ClusterNssConfig,
    ClusterReplayValidator, ClusterTopology, ClusterTrace, FailureKind, GpuTrainingConfig,
    GpuTrainingSimulator, Intervention, NodeId, NssSeed, PressureKind, CLUSTER_SUMMARY_FEATURES,
    NSS_MODEL_VERSION,
};

fn main() {
    let seed = NssSeed(2026);

    // 1. 8-GPU training with moderate jitter (5% std-dev) and a
    //    realistic memory-per-microbatch (1.5% of capacity → without
    //    GC, we'd OOM around iteration 66).
    let cfg = GpuTrainingConfig {
        n_gpus: 8,
        service_mean: 1.0,
        service_jitter: 0.05,
        allreduce_base: 0.02,
        allreduce_bytes: 2.0e9,   // 2 GB per iter
        nccl_bandwidth: 5.0e10,   // 50 GB/s
        memory_per_microbatch: 0.015,
        gc_interval: 20,
        gc_recovery: 0.7,
        fragmentation_growth: 0.002,
        memory_capacity: 1.0,
        pipeline_stages: 1,         // pure data-parallel
        microbatches_per_iteration: 1,
        ..GpuTrainingConfig::default()
    };

    // 2. Fully-connected topology (NCCL all-to-all).
    let topology = ClusterTopology::complete(cfg.n_gpus, 16, 0.5).unwrap();

    // 3. Scripted intervention: GPU 5 OOM-crashes at iteration 48.
    let script = vec![Intervention::FailNode {
        tick: 48,
        node: NodeId(5),
    }];

    let mut sim = GpuTrainingSimulator::new(cfg, topology.clone(), seed, script.clone()).unwrap();
    let traj = sim.run(96).unwrap();

    // 4. Print iteration counts and idle accumulation per GPU.
    println!(
        "[sim] gpus={} iterations={} jitter={:.2}",
        cfg.n_gpus,
        traj.len(),
        cfg.service_jitter
    );
    println!("[sim] iterations completed per GPU:");
    let counts = sim.iteration_counts();
    let idle = sim.idle_accumulated();
    for id in topology.nodes() {
        println!(
            "       gpu {:>2}: {:>3} iters, idle accumulated = {:.4}",
            id,
            counts.get(&id).copied().unwrap_or(0),
            idle.get(&id).copied().unwrap_or(0.0),
        );
    }

    // 5. Cluster-failure timeline.
    let mut last_kind = None;
    println!("\n[sim] cluster_failure transitions:");
    for (i, ev) in traj.iter().enumerate() {
        if last_kind != Some(ev.cluster_failure.kind) {
            println!(
                "       tick {:>3}: {:<8}  (failed_gpus={})",
                i,
                ev.cluster_failure.kind.label(),
                ev.state.failed_count()
            );
            last_kind = Some(ev.cluster_failure.kind);
        }
    }

    // 6. Fit cluster NSS on the trajectory.
    let mut nss =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();
    nss.fit(&traj).unwrap();
    let last = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&last).unwrap();

    println!(
        "\n[nss] run_id={} P(collapse)={:.4} P(degraded)={:.4} conf={:.4}",
        pred.run_id,
        pred.failure.collapse_probability,
        pred.failure.degraded_probability,
        pred.failure.confidence
    );
    println!(
        "[nss] dominant_gpu={} dominant_kind={} contribution={:+.4}",
        pred.attribution.dominant_node,
        pred.attribution.dominant_contribution.kind.label(),
        pred.attribution.dominant_contribution.magnitude,
    );

    // 7. Cluster-summary attribution.
    println!("\n[nss] cluster-summary contributions:");
    for i in 0..CLUSTER_SUMMARY_FEATURES {
        println!(
            "       {:>26}  contribution = {:+.4}",
            cluster_summary_label(i),
            pred.attribution.summary_contributions[i],
        );
    }

    // 8. Per-GPU memory & sync saturation in the final state.
    println!("\n[nss] per-GPU final saturations (Memory / Sync / Network):");
    for (id, state) in last.nodes.iter() {
        let mem = state
            .pressures
            .get(PressureKind::Memory)
            .map(|p| p.saturation())
            .unwrap_or(0.0);
        let sync = state
            .pressures
            .get(PressureKind::Sync)
            .map(|p| p.saturation())
            .unwrap_or(0.0);
        let net = state
            .pressures
            .get(PressureKind::Network)
            .map(|p| p.saturation())
            .unwrap_or(0.0);
        println!(
            "       gpu {:>2}: mem={:.3}  sync={:.3}  net={:.3}",
            id, mem, sync, net
        );
    }

    // 9. Replay validation.
    let trace = ClusterTrace {
        run_id: pred.run_id,
        input_hash: pred.input_hash,
        input_state: last,
        topology,
        simulator_config: cjc_nss::ClusterConfig::default(),
        intervention_script: script,
        nss_config: ClusterNssConfig::default(),
        seed,
        training_trajectory: Some(traj),
        collapse_probability: pred.failure.collapse_probability,
        degraded_probability: pred.failure.degraded_probability,
        model_version: NSS_MODEL_VERSION.to_string(),
    };
    ClusterReplayValidator::new()
        .verify(&trace)
        .expect("GPU-training replay must succeed");
    println!(
        "\n[replay] verified: trajectory={} iterations, topology={} GPUs",
        trace.training_trajectory.as_ref().map(|t| t.len()).unwrap_or(0),
        trace.topology.node_count(),
    );

    // 10. Sanity check on what the model "saw".
    let n_collapse = trace
        .training_trajectory
        .as_ref()
        .unwrap()
        .iter()
        .filter(|ev| ev.cluster_failure.kind == FailureKind::Collapse)
        .count();
    println!(
        "[summary] training_trajectory had {} collapse ticks (out of {})",
        n_collapse,
        trace.training_trajectory.as_ref().unwrap().len(),
    );
}
