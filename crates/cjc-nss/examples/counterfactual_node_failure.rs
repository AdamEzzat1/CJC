//! Phase 3a demo: counterfactual node-failure analysis.
//!
//! Workflow:
//! 1. Run a 4-node cluster for 16 ticks (the "lead-in" phase).
//! 2. Snapshot the simulator.
//! 3. Fork into two futures:
//!    - **Baseline**: no intervention, cluster continues normally.
//!    - **Failure**: node 2 fails at tick 18.
//! 4. Each fork runs 24 more ticks.
//! 5. NSS predicts on each fork's final state.
//! 6. Print the comparison: P(collapse) delta, dominant-node flip,
//!    cluster-label change, per-node health disagreements.
//!
//! This is the canonical "what would happen if node X failed?" question
//! a cluster operator asks before pre-emptively rerouting traffic.
//!
//! Run with:
//! ```text
//! cargo run --example counterfactual_node_failure -p cjc-nss
//! ```

use cjc_nss::{
    run_cluster_counterfactual, ClusterConfig, ClusterNeuralSystemsSimulator, ClusterNssConfig,
    ClusterSimulator, ClusterTopology, FailureKind, Intervention, NodeId, NssSeed,
};

fn main() {
    let seed = NssSeed(2026);
    let sim_cfg = ClusterConfig {
        cluster_arrival_rate: 8.0,
        ..ClusterConfig::default()
    };
    let topology = ClusterTopology::complete(4, 8, 0.5).unwrap();

    // 1. Lead-in: run the cluster for 16 ticks before the decision
    //    point. The snapshot tick is when an operator might be
    //    asking "should we proactively reroute?"
    let mut sim = ClusterSimulator::new(sim_cfg, topology.clone(), seed, vec![]).unwrap();
    let _lead_in = sim.run(16).unwrap();
    let snapshot = sim.snapshot();
    println!(
        "[lead-in] cluster ran {} ticks, snapshot taken at tick {}",
        snapshot.tick(),
        snapshot.tick()
    );

    // 2. Build the cluster NSS.
    let nss =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();

    // 3. Counterfactual experiment.
    let comparison = run_cluster_counterfactual(
        &snapshot,
        "baseline",
        vec![],
        "node_2_fails",
        vec![Intervention::FailNode {
            tick: 18,
            node: NodeId(2),
        }],
        24, // horizon
        &nss,
    )
    .unwrap();

    // 4. Headline metrics.
    println!("\n[counterfactual] horizon=24 ticks");
    println!(
        "  P(collapse) delta = {:+.4}   (baseline -> intervention)",
        comparison.collapse_probability_delta
    );
    println!(
        "  P(degraded) delta = {:+.4}",
        comparison.degraded_probability_delta
    );
    println!(
        "  final-tick label flipped? {}",
        comparison.final_label_flipped
    );
    println!(
        "  dominant-node flipped?    {}",
        comparison.dominant_node_flipped
    );
    println!(
        "  dominant-kind flipped?    {}",
        comparison.dominant_kind_flipped
    );

    // 5. Trajectory-level summaries.
    let (a_nom, a_deg, a_col) = comparison.a.label_distribution();
    let (b_nom, b_deg, b_col) = comparison.b.label_distribution();
    println!(
        "\n[trajectories] (nominal / degraded / collapse) counts over {} post-snapshot ticks",
        comparison.a.trajectory.len()
    );
    println!(
        "  baseline:        ({:>2}, {:>2}, {:>2})  mean_queue_sat={:.3}",
        a_nom,
        a_deg,
        a_col,
        comparison.a.mean_queue_saturation()
    );
    println!(
        "  node_2_fails:    ({:>2}, {:>2}, {:>2})  mean_queue_sat={:.3}",
        b_nom,
        b_deg,
        b_col,
        comparison.b.mean_queue_saturation()
    );

    // 6. Per-node health disagreements (the structural diff between
    //    the two futures).
    if comparison.node_health_disagreements.is_empty() {
        println!("\n[diff] no per-node health disagreements (intervention had no structural effect)");
    } else {
        println!("\n[diff] node-health disagreements at final tick:");
        for (id, (ha, hb)) in &comparison.node_health_disagreements {
            println!(
                "  node {:>2}: baseline={}  intervention={}",
                id,
                ha.label(),
                hb.label()
            );
        }
    }

    // 7. NSS predictions side by side.
    let pa = &comparison.a.prediction;
    let pb = &comparison.b.prediction;
    println!("\n[nss predictions]");
    println!(
        "  baseline:        P(collapse)={:.4}  P(degraded)={:.4}  dom_node={}  dom_kind={}",
        pa.failure.collapse_probability,
        pa.failure.degraded_probability,
        pa.attribution.dominant_node,
        pa.attribution.dominant_contribution.kind.label(),
    );
    println!(
        "  node_2_fails:    P(collapse)={:.4}  P(degraded)={:.4}  dom_node={}  dom_kind={}",
        pb.failure.collapse_probability,
        pb.failure.degraded_probability,
        pb.attribution.dominant_node,
        pb.attribution.dominant_contribution.kind.label(),
    );

    // 8. Decision signal.
    println!("\n[decision signal]");
    let harmful = comparison.intervention_is_harmful(0.05);
    let beneficial = comparison.intervention_is_beneficial(0.05);
    match (harmful, beneficial) {
        (true, _) => println!("  → intervention makes things WORSE by ≥ 0.05"),
        (_, true) => println!("  → intervention makes things BETTER by ≥ 0.05"),
        _ => println!("  → intervention has negligible effect on predicted outcome (|Δ| < 0.05)"),
    }

    let _final_a = comparison.a.trajectory.last_state().map(|s| s.tick).unwrap_or(0);
    let _final_b = comparison.b.trajectory.last_state().map(|s| s.tick).unwrap_or(0);
    println!(
        "\n[summary] snapshot_tick={}, baseline_final_tick={}, intervention_final_tick={}",
        snapshot.tick(),
        comparison.a.trajectory.last_state().map(|s| s.tick).unwrap_or(0),
        comparison.b.trajectory.last_state().map(|s| s.tick).unwrap_or(0),
    );
    // Just to use the variable — the `cluster_failure.kind` matters
    // semantically as the demo's "did we predict differently?" signal.
    let _ = FailureKind::Nominal;
}
