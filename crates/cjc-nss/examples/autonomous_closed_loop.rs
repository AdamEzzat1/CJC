//! Phase 4 demo: closed-loop autonomous controller on a degrading
//! cluster.
//!
//! Setup:
//! 1. 4-node cluster, moderate load.
//! 2. Inject a node failure at tick 6 (cluster is now degraded).
//! 3. Run a closed loop with the autonomous controller — it should
//!    notice the degraded cluster and apply a recovery action.
//! 4. Print the audit log so we can see *what* it decided + *why*.
//!
//! Run with:
//! ```text
//! cargo run --example autonomous_closed_loop -p cjc-nss
//! ```

use cjc_nss::{
    AdvisorConfig, AutonomousOptimizer, ClusterConfig, ClusterNeuralSystemsSimulator,
    ClusterNssConfig, ClusterSimulator, ClusterTopology, DecisionOutcome, FailureKind,
    Intervention, NodeId, NssSeed, OptimizerConfig, SafetyMode,
};

fn main() {
    let seed = NssSeed(2026);

    // 1. Build a 4-node cluster with a scripted failure at tick 6.
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

    // 2. Build NSS predictor.
    let nss = ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();

    // 3. Build the autonomous controller. Conservative mode = only
    //    Recovery actions are allowed; min_improvement=0 lets the
    //    controller act on any net-positive recommendation.
    let mut optimizer = AutonomousOptimizer::new(OptimizerConfig {
        control_period: 4,
        min_improvement: 0.0,
        min_confidence: 0.0,
        action_cooldown_ticks: 6,
        max_actions: Some(5),
        safety_mode: SafetyMode::Conservative,
        advisor: AdvisorConfig {
            horizon: 8,
            consider_failure_actions: false,
            consider_recovery_actions: true,
            ..AdvisorConfig::default()
        },
    })
    .unwrap();

    // 4. Run the closed loop for 32 ticks.
    let report = optimizer.run_closed_loop(&mut sim, &nss, 32).unwrap();

    // 5. Report.
    println!("Phase 4 — autonomous closed-loop demo");
    println!("  trajectory length:    {} ticks", report.trajectory.len());
    println!(
        "  decisions made:       {} ({} applied, {} skipped, {} no-ops)",
        report.decisions.len(),
        report.actions_applied,
        report.actions_skipped,
        report.no_ops
    );
    println!(
        "  collapse tick count:  {} / {}",
        report.collapse_tick_count(),
        report.trajectory.len()
    );
    println!(
        "  nominal tick count:   {} / {}",
        report.nominal_tick_count(),
        report.trajectory.len()
    );

    // Distribution of failure-label across the trajectory.
    let mut nom = 0;
    let mut deg = 0;
    let mut col = 0;
    for ev in report.trajectory.iter() {
        match ev.cluster_failure.kind {
            FailureKind::Nominal => nom += 1,
            FailureKind::Degraded => deg += 1,
            FailureKind::Collapse => col += 1,
        }
    }
    println!(
        "\n[label distribution]  ({} nominal / {} degraded / {} collapse)",
        nom, deg, col
    );

    // 6. Audit log.
    println!("\n[audit log]");
    println!(
        "  {:<5} {:<10} {:<30} {:>10} {:>10} {:>10}  {}",
        "tick", "outcome", "recommended", "P(coll)", "baseline", "margin", "reason"
    );
    for d in &report.decisions {
        let outcome = match d.outcome {
            DecisionOutcome::Applied => "APPLIED",
            DecisionOutcome::Skipped => "skipped",
            DecisionOutcome::NoOp => "no-op",
        };
        println!(
            "  {:<5} {:<10} {:<30} {:>10.4} {:>10.4} {:>10.4}  {}",
            d.snapshot_tick,
            outcome,
            format!("{:?}", d.recommended),
            d.recommended_collapse,
            d.baseline_collapse,
            d.confidence_margin,
            d.skip_reason
        );
    }

    // 7. Determinism receipt.
    let mut sim2 = ClusterSimulator::new(
        sim_cfg,
        ClusterTopology::complete(4, 8, 0.5).unwrap(),
        seed,
        vec![Intervention::FailNode {
            tick: 6,
            node: NodeId(1),
        }],
    )
    .unwrap();
    let nss2 = ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), seed).unwrap();
    let mut opt2 = AutonomousOptimizer::new(OptimizerConfig {
        control_period: 4,
        min_improvement: 0.0,
        min_confidence: 0.0,
        action_cooldown_ticks: 6,
        max_actions: Some(5),
        safety_mode: SafetyMode::Conservative,
        advisor: AdvisorConfig {
            horizon: 8,
            consider_failure_actions: false,
            consider_recovery_actions: true,
            ..AdvisorConfig::default()
        },
    })
    .unwrap();
    let report2 = opt2.run_closed_loop(&mut sim2, &nss2, 32).unwrap();
    let identical = report.trajectory.canonical_bytes() == report2.trajectory.canonical_bytes();
    println!(
        "\n[determinism] second closed-loop run produces identical trajectory: {}",
        identical
    );
}
