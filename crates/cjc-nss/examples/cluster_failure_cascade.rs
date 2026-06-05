//! Phase 2 end-to-end demo: a 4-node cluster with a scripted
//! mid-run node failure. We observe the failure-cascade dynamics
//! (link congestion rises, peer queue saturation grows, cluster
//! rollup transitions from Nominal → Degraded → Collapse → recovery),
//! predict cluster-level instability with per-node attribution, and
//! verify the prediction replays byte-identically.
//!
//! Run with:
//! ```text
//! cargo run --example cluster_failure_cascade -p cjc-nss
//! ```

use cjc_nss::{
    cluster_summary_label, ClusterConfig, ClusterNeuralSystemsSimulator, ClusterNssConfig,
    ClusterReplayValidator, ClusterSimulator, ClusterTopology, ClusterTrace, FailureKind,
    Intervention, NodeId, NssSeed, PressureKind, RoutingPolicy, NSS_MODEL_VERSION,
    CLUSTER_SUMMARY_FEATURES,
};

fn main() {
    let seed = NssSeed(2026);

    // 1. 4-node ring topology + mild-overload load.
    let sim_cfg = ClusterConfig {
        workers_per_node: 2,
        queue_capacity: 16,
        cluster_arrival_rate: 8.0,
        service_min: 0.6,
        service_max: 1.4,
        degraded_knee: 0.65,
        collapse_window: 3,
        routing: RoutingPolicy::LeastLoaded,
        link_dissipation: 0.2,
        propagation: Default::default(),
    };
    let topology = ClusterTopology::ring(4, 8, 0.5).expect("valid ring");

    // 2. Scripted intervention: fail node 1 at tick 16, recover at tick 32.
    let script = vec![
        Intervention::FailNode {
            tick: 16,
            node: NodeId(1),
        },
        Intervention::RecoverNode {
            tick: 32,
            node: NodeId(1),
        },
    ];

    let mut sim =
        ClusterSimulator::new(sim_cfg, topology.clone(), seed, script.clone()).unwrap();
    let traj = sim.run(48).unwrap();

    // 3. Summarise the trajectory.
    let n_nominal = traj
        .iter()
        .filter(|ev| ev.cluster_failure.kind == FailureKind::Nominal)
        .count();
    let n_degraded = traj
        .iter()
        .filter(|ev| ev.cluster_failure.kind == FailureKind::Degraded)
        .count();
    let n_collapse = traj
        .iter()
        .filter(|ev| ev.cluster_failure.kind == FailureKind::Collapse)
        .count();
    println!(
        "[sim] cluster=4-node ring, ticks={}, nominal={} degraded={} collapse={}",
        traj.len(),
        n_nominal,
        n_degraded,
        n_collapse
    );

    // Print the cluster_failure transition timeline so the cascade
    // is visible.
    let mut last_kind = None;
    for (i, ev) in traj.iter().enumerate() {
        if last_kind != Some(ev.cluster_failure.kind) {
            println!(
                "[sim]   tick {:>3}: cluster_failure -> {:<8} (failed_nodes={})",
                i,
                ev.cluster_failure.kind.label(),
                ev.state.failed_count(),
            );
            last_kind = Some(ev.cluster_failure.kind);
        }
    }

    // 4. Fit a cluster NSS on the trajectory.
    let mut nss =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();
    nss.fit(&traj).unwrap();

    // 5. Predict on the final state.
    let last = traj.last_state().unwrap().clone();
    let pred = nss.predict_next(&last).unwrap();
    println!(
        "\n[nss] run_id={} P(collapse)={:.4} P(degraded)={:.4} conf={:.4}",
        pred.run_id,
        pred.failure.collapse_probability,
        pred.failure.degraded_probability,
        pred.failure.confidence,
    );
    println!(
        "[nss] dominant_node={} dominant_kind={} contribution={:+.4}",
        pred.attribution.dominant_node,
        pred.attribution.dominant_contribution.kind.label(),
        pred.attribution.dominant_contribution.magnitude,
    );

    // 6. Cluster-summary attribution (which cluster-level signal
    //    moved the cluster head's collapse logit).
    println!("\n[nss] cluster-summary contributions:");
    for i in 0..CLUSTER_SUMMARY_FEATURES {
        println!(
            "       {:>26}  contribution = {:+.4}",
            cluster_summary_label(i),
            pred.attribution.summary_contributions[i],
        );
    }

    // 7. Per-node contributions on the dominant pressure kinds.
    println!("\n[nss] per-node top-1 contribution:");
    for (id, attr) in pred.attribution.per_node.iter() {
        let top = attr.contributions[0];
        let q = last
            .nodes
            .get(id)
            .unwrap()
            .pressures
            .get(PressureKind::Queue)
            .unwrap()
            .saturation();
        println!(
            "       node {:>2} (Q_sat={:.2}): {:<11} = {:+.4}",
            id,
            q,
            top.kind.label(),
            top.magnitude
        );
    }

    // 8. Replay validation.
    let trace = ClusterTrace {
        run_id: pred.run_id,
        input_hash: pred.input_hash,
        input_state: last,
        topology,
        simulator_config: sim_cfg,
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
        .expect("cluster replay must succeed");
    println!(
        "\n[replay] verified: trajectory={} ticks, topology={} nodes / {} edges",
        trace
            .training_trajectory
            .as_ref()
            .map(|t| t.len())
            .unwrap_or(0),
        trace.topology.node_count(),
        trace.topology.edge_count(),
    );
}
