//! Phase 3c demo: scheduler advisory head.
//!
//! Workflow:
//! 1. Run a 4-node cluster for 10 ticks of nominal load.
//! 2. Inject a node failure at tick 6 (so the cluster is operating
//!    with one node down by the snapshot tick).
//! 3. Snapshot the simulator.
//! 4. Ask the advisor: "given this state, what action should I take?"
//! 5. Print the full ranking: do-nothing, recover each node, etc.
//! 6. Highlight the recommended action and its rationale (collapse-tick
//!    count + predicted post-action collapse probability).
//!
//! Run with:
//! ```text
//! cargo run --example scheduler_advisor -p cjc-nss
//! ```

use cjc_nss::{
    AdvisorConfig, AdvisoryAction, ClusterConfig, ClusterNeuralSystemsSimulator, ClusterNssConfig,
    ClusterSimulator, ClusterTopology, Intervention, NodeHealth, NodeId, NssSeed, SchedulerAdvisor,
};

fn main() {
    let seed = NssSeed(2026);

    // 1. Build a 4-node cluster with a node failure scheduled at tick 6.
    let sim_cfg = ClusterConfig {
        cluster_arrival_rate: 8.0,
        ..ClusterConfig::default()
    };
    let topology = ClusterTopology::complete(4, 8, 0.5).unwrap();
    let initial_script = vec![Intervention::FailNode {
        tick: 6,
        node: NodeId(1),
    }];
    let mut sim = ClusterSimulator::new(sim_cfg, topology, seed, initial_script).unwrap();

    // 2. Run 10 ticks of lead-in. After tick 6, node 1 is failed.
    let lead_in = sim.run(10).unwrap();
    println!("[lead-in] cluster ran 10 ticks; final-tick health:");
    for (id, h) in &lead_in.last_state().unwrap().node_health {
        let badge = match h {
            NodeHealth::Healthy => "✓",
            NodeHealth::Failed => "✗",
            NodeHealth::Absent => "○",
        };
        println!("       node {:>2}: {} {}", id, badge, h.label());
    }

    // 3. Snapshot at tick 10 — the "decision point".
    let snapshot = sim.snapshot();
    println!("\n[decision] snapshot taken at tick {}", snapshot.tick());

    // 4. Build the cluster NSS predictor.
    let nss = ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();

    // 5. Build the advisor. Consider BOTH failure and recovery
    //    actions — the operator wants to see the full menu.
    let advisor = SchedulerAdvisor::new(AdvisorConfig {
        horizon: 12,
        consider_failure_actions: true,
        consider_recovery_actions: true,
        ..AdvisorConfig::default()
    })
    .unwrap();

    let ranking = advisor.recommend(&snapshot, &nss).unwrap();
    println!(
        "\n[advisor] evaluated {} candidate actions over horizon={}",
        ranking.candidates.len(),
        ranking.horizon
    );
    println!(
        "[advisor] recommendation: {:?}  (confidence margin = {:.4})",
        ranking.recommended, ranking.confidence_margin
    );

    // 6. Print the full ranking.
    println!("\n[ranking]   (sorted ascending by P(collapse) — best first)");
    println!(
        "  {:<3} {:<22} {:>10} {:>10} {:>11} {:<10}",
        "#", "action", "P(coll.)", "P(deg.)", "collapse#", "dom_kind"
    );
    for (i, cand) in ranking.candidates.iter().enumerate() {
        let marker = if cand.action == ranking.recommended {
            "★"
        } else {
            " "
        };
        println!(
            "  {}{:<3} {:<22} {:>10.4} {:>10.4} {:>11} {:<10}",
            marker,
            i,
            cand.action.label(),
            cand.predicted_collapse,
            cand.predicted_degraded,
            cand.collapse_tick_count,
            cand.dominant_kind.label(),
        );
    }

    // 7. Compare best vs worst.
    let best = &ranking.candidates[0];
    let worst = &ranking.candidates[ranking.candidates.len() - 1];
    println!(
        "\n[delta] best ({}) vs worst ({}) — P(collapse) gap = {:+.4}",
        best.action.label(),
        worst.action.label(),
        worst.predicted_collapse - best.predicted_collapse,
    );

    // 8. Rationale: what the supporting trajectory showed.
    let best_traj_collapse_ticks = best.collapse_tick_count;
    let do_nothing = ranking
        .candidates
        .iter()
        .find(|c| c.action == AdvisoryAction::DoNothing)
        .unwrap();
    println!(
        "[rationale] best action had {}/{} collapse ticks in the counterfactual horizon;",
        best_traj_collapse_ticks, ranking.horizon
    );
    println!(
        "            do_nothing had     {}/{} collapse ticks for comparison.",
        do_nothing.collapse_tick_count, ranking.horizon
    );

    // 9. Determinism receipt.
    let ranking2 = advisor.recommend(&snapshot, &nss).unwrap();
    let identical = ranking
        .candidates
        .iter()
        .zip(ranking2.candidates.iter())
        .all(|(a, b)| a.predicted_collapse.to_bits() == b.predicted_collapse.to_bits());
    println!(
        "\n[determinism] second recommend() on same snapshot: bit-identical = {}",
        identical
    );
}
