//! Phase 3e demo: per-node advisor with the full extended action set
//! (DoNothing / Recover / ShedLoad@intensity / Add / Remove).
//!
//! Setup:
//! 1. 5-node cluster topology — 4 nodes start `Healthy`, 1 starts
//!    `Absent` (pre-allocated capacity slot, not yet in service).
//! 2. Drive moderate-to-heavy load.
//! 3. Inject a failure on one node mid-run.
//! 4. Snapshot.
//! 5. Ask the advisor for **per-node** rankings with all action kinds
//!    enabled.
//! 6. Print per-node recommendations.
//!
//! Run with:
//! ```text
//! cargo run --example advisor_per_node -p cjc-nss
//! ```

use cjc_nss::{
    AdvisorConfig, AdvisoryAction, ClusterConfig, ClusterNeuralSystemsSimulator,
    ClusterNssConfig, ClusterSimulator, ClusterTopology, Intervention, NodeHealth, NodeId,
    NssSeed, SchedulerAdvisor,
};

fn main() {
    let seed = NssSeed(2026);

    // 1. Build a 5-node topology. The 5th node is pre-allocated as
    //    Absent (autoscaling capacity slot) via a RemoveNode at tick 0.
    let topology = ClusterTopology::complete(5, 8, 0.5).unwrap();
    let sim_cfg = ClusterConfig {
        cluster_arrival_rate: 12.0,
        ..ClusterConfig::default()
    };
    let initial_script = vec![
        // Make node 4 Absent from the start (pre-allocated capacity).
        Intervention::RemoveNode {
            tick: 0,
            node: NodeId(4),
        },
        // Fail node 2 at tick 6.
        Intervention::FailNode {
            tick: 6,
            node: NodeId(2),
        },
    ];
    let mut sim = ClusterSimulator::new(sim_cfg, topology, seed, initial_script).unwrap();

    let lead_in = sim.run(12).unwrap();
    println!("[lead-in] 5-slot cluster, 12 ticks; final-tick node health:");
    for (id, h) in &lead_in.last_state().unwrap().node_health {
        let badge = match h {
            NodeHealth::Healthy => "✓",
            NodeHealth::Failed => "✗",
            NodeHealth::Absent => "○",
        };
        println!("       node {:>2}: {} {}", id, badge, h.label());
    }
    let snapshot = sim.snapshot();
    println!("\n[decision] snapshot at tick {}", snapshot.tick());

    let nss =
        ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();

    // 2. Build the advisor with the full Phase 3e action set enabled.
    let advisor = SchedulerAdvisor::new(AdvisorConfig {
        horizon: 12,
        consider_failure_actions: true,
        consider_recovery_actions: true,
        consider_shed_load: true,
        shed_intensities: vec![0.25, 0.50, 0.75],
        consider_autoscaling: true,
    })
    .unwrap();

    // 3. Per-node recommendations.
    let per_node = advisor.recommend_per_node(&snapshot, &nss).unwrap();
    println!(
        "\n[advisor — per-node] (horizon={} ticks, all action kinds enabled)",
        advisor.config().horizon
    );
    for (id, ranking) in &per_node {
        let badge = match lead_in.last_state().unwrap().node_health.get(id).copied() {
            Some(NodeHealth::Healthy) => "✓",
            Some(NodeHealth::Failed) => "✗",
            Some(NodeHealth::Absent) => "○",
            None => "?",
        };
        println!(
            "\n  node {} {} — recommended: {:?}  confidence_margin={:.4}",
            id, badge, ranking.recommended, ranking.confidence_margin
        );
        // Show the top 3 candidates for this node.
        for (i, cand) in ranking.candidates.iter().take(3).enumerate() {
            let marker = if i == 0 { "★" } else { " " };
            println!(
                "       {} #{} {:<25} P(coll)={:.4}  collapse#={}/{}",
                marker,
                i,
                cand.action.label(),
                cand.predicted_collapse,
                cand.collapse_tick_count,
                ranking.horizon,
            );
        }
    }

    // 4. Aggregate: count how many nodes have each kind of recommendation.
    let mut counts = std::collections::BTreeMap::<&str, u32>::new();
    for (_id, ranking) in &per_node {
        let kind = match ranking.recommended {
            AdvisoryAction::DoNothing => "do_nothing",
            AdvisoryAction::FailNode { .. } => "fail_node",
            AdvisoryAction::RecoverNode { .. } => "recover_node",
            AdvisoryAction::ShedLoad { .. } => "shed_load",
            AdvisoryAction::AddNode { .. } => "add_node",
            AdvisoryAction::RemoveNode { .. } => "remove_node",
        };
        *counts.entry(kind).or_insert(0) += 1;
    }
    println!("\n[aggregate] recommendation distribution across 5 nodes:");
    for (kind, n) in &counts {
        println!("       {:>15} : {}", kind, n);
    }

    // 5. Cluster-wide recommendation (existing recommend() — for
    //    comparison).
    let cluster = advisor.recommend(&snapshot, &nss).unwrap();
    println!(
        "\n[advisor — cluster-wide] recommended: {:?}  (margin={:.4})",
        cluster.recommended, cluster.confidence_margin
    );

    // 6. Determinism receipt.
    let per_node_2 = advisor.recommend_per_node(&snapshot, &nss).unwrap();
    let identical = per_node.iter().zip(per_node_2.iter()).all(|((a_id, ra), (b_id, rb))| {
        a_id == b_id
            && ra.recommended == rb.recommended
            && ra
                .candidates
                .iter()
                .zip(rb.candidates.iter())
                .all(|(x, y)| x.predicted_collapse.to_bits() == y.predicted_collapse.to_bits())
    });
    println!("\n[determinism] second per-node recommendation: bit-identical = {}", identical);
}
