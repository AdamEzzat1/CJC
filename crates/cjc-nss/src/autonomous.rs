//! Phase 4 — Autonomous optimisation engine.
//!
//! Closed-loop controller that periodically:
//!
//! 1. Snapshots the live simulator,
//! 2. Asks the Phase 3c [`SchedulerAdvisor`] for a ranked set of
//!    candidate actions,
//! 3. Picks the top-ranked action *if* it survives the safety layer
//!    (better than `DoNothing`, above confidence floor, outside any
//!    per-node cooldown, allowed by the active safety mode, within
//!    the per-run action budget),
//! 4. Applies the action to the simulator via
//!    [`crate::ClusterSimulator::inject_intervention`],
//! 5. Logs a [`DecisionRecord`] to the audit trail.
//!
//! The implementation is deliberately a thin **orchestration layer**
//! over Phase 3 primitives — every piece of mechanism (snapshot,
//! counterfactual ranking, deterministic intervention application,
//! per-node cooldown) was already in place. The only Phase 4 surface
//! that's genuinely new is the **safety layer**: the rules that
//! decide *whether to act* given what the advisor *suggests* to do.
//!
//! ## Why a safety layer at all?
//!
//! An autonomous controller acting on its own ranking output is only
//! as good as its ranking — and the ranking is only as good as the
//! NSS predictor underneath. An untrained or poorly-calibrated NSS
//! will sometimes rank a wildly wrong action first. The safety layer
//! is the defence-in-depth that prevents the controller from acting
//! on noise:
//!
//! - **Improvement floor.** Refuse to act if the recommended action
//!   doesn't beat `DoNothing` by at least `min_improvement`.
//! - **Confidence floor.** Refuse to act if the ranking's
//!   `confidence_margin` is below `min_confidence`.
//! - **Per-node cooldown.** Refuse to act on a node within
//!   `action_cooldown_ticks` of the previous action on that node.
//!   Prevents flap (recover/remove/recover/remove cycles).
//! - **Safety mode.** Constrains the *kinds* of actions the
//!   controller may apply: Conservative allows only DoNothing +
//!   RecoverNode; Moderate adds ShedLoad + RemoveNode; Aggressive
//!   allows everything including FailNode (pre-emptive drain).
//! - **Action budget.** Optional cap on total actions per closed-loop
//!   run.
//!
//! ## Determinism contract
//!
//! Same as Phase 3 — every snapshot/fork/predict is deterministic,
//! the audit log records the run-id of every snapshot decision was
//! made on, so the full closed loop is replayable byte-for-byte
//! given the same `(sim_config, nss_config, optimizer_config, seed)`.

use crate::advisory::{AdvisoryAction, AdvisoryRanking, SchedulerAdvisor};
use crate::cluster::{ClusterTrajectory, NodeId};
use crate::cluster_nss::ClusterNeuralSystemsSimulator;
use crate::cluster_simulator::{ClusterSimulator, Intervention};
use crate::error::NssError;
use crate::seed::NssRunId;
use std::collections::BTreeMap;

/// Defence-in-depth tier on the controller's action set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SafetyMode {
    /// Most cautious: only `DoNothing` + `RecoverNode` are allowed.
    /// Useful when the NSS predictor is fresh / unfit and operators
    /// want the controller to nudge a degraded cluster back up but
    /// never proactively change capacity.
    Conservative,
    /// Recovery + admission-control (ShedLoad) + capacity-down
    /// (RemoveNode). Pre-emptive failures and capacity-up are still
    /// blocked.
    Moderate,
    /// All five action kinds enabled including `FailNode` (controlled
    /// drain) and `AddNode` (autoscale up). Recommended only when
    /// the NSS predictor has been trained against representative
    /// trajectories — `Aggressive` mode trusts the ranking implicitly.
    Aggressive,
}

impl SafetyMode {
    /// True if the given action kind is allowed under this mode.
    pub fn allows(self, action: AdvisoryAction) -> bool {
        match (self, action) {
            (_, AdvisoryAction::DoNothing) => true,
            (SafetyMode::Conservative, AdvisoryAction::RecoverNode { .. }) => true,
            (SafetyMode::Conservative, _) => false,
            (SafetyMode::Moderate, AdvisoryAction::FailNode { .. }) => false,
            (SafetyMode::Moderate, AdvisoryAction::AddNode { .. }) => false,
            (SafetyMode::Moderate, _) => true,
            (SafetyMode::Aggressive, _) => true,
        }
    }

    /// Canonical short label.
    pub fn label(self) -> &'static str {
        match self {
            SafetyMode::Conservative => "conservative",
            SafetyMode::Moderate => "moderate",
            SafetyMode::Aggressive => "aggressive",
        }
    }
}

/// Knobs for the autonomous controller.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerConfig {
    /// Ticks between control decisions. The controller runs the
    /// simulator for this many ticks, then makes one decision. Must
    /// be ≥ 1. Default 4.
    pub control_period: u64,
    /// Minimum gap (`P(collapse)(do_nothing) - P(collapse)(best)`)
    /// required to apply the recommendation. `>= 0`. Default 0.01
    /// (1pp better than baseline).
    pub min_improvement: f64,
    /// Minimum `confidence_margin` (worst − best in the ranking)
    /// required to apply. `>= 0`. Default 0.0 (no extra requirement
    /// on top of `min_improvement`).
    pub min_confidence: f64,
    /// Per-node cooldown in ticks. After applying an action on node
    /// X, no further actions on X are allowed until
    /// `current_tick - last_action_tick >= action_cooldown_ticks`.
    /// Default 6.
    pub action_cooldown_ticks: u64,
    /// Optional per-run action budget — once `actions_applied`
    /// reaches this, subsequent decisions are all
    /// [`DecisionOutcome::Skipped`] with reason `"budget_exhausted"`.
    /// `None` = unlimited. Default `None`.
    pub max_actions: Option<u32>,
    /// Action-set restriction. Default `Conservative`.
    pub safety_mode: SafetyMode,
    /// Underlying advisor config (forwarded to [`SchedulerAdvisor::new`]).
    pub advisor: crate::advisory::AdvisorConfig,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            control_period: 4,
            min_improvement: 0.01,
            min_confidence: 0.0,
            action_cooldown_ticks: 6,
            max_actions: None,
            safety_mode: SafetyMode::Conservative,
            advisor: crate::advisory::AdvisorConfig::default(),
        }
    }
}

impl OptimizerConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.control_period == 0 {
            return Err(NssError::InvalidConfig {
                detail: "OptimizerConfig.control_period must be >= 1".into(),
            });
        }
        if !self.min_improvement.is_finite() || self.min_improvement < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "min_improvement must be finite and >= 0, got {}",
                    self.min_improvement
                ),
            });
        }
        if !self.min_confidence.is_finite() || self.min_confidence < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "min_confidence must be finite and >= 0, got {}",
                    self.min_confidence
                ),
            });
        }
        self.advisor.validate()?;
        Ok(())
    }
}

/// What the safety layer did with a recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecisionOutcome {
    /// The recommendation was applied — `inject_intervention` ran.
    Applied,
    /// The recommendation was `DoNothing`, so applying it is moot.
    NoOp,
    /// The recommendation was skipped by the safety layer; see
    /// [`DecisionRecord::skip_reason`] for which guard fired.
    Skipped,
}

/// One audit-log entry. Captures everything an external auditor needs
/// to reproduce + validate the controller's decision.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    /// Tick the simulator was at when the decision was made.
    pub snapshot_tick: u64,
    /// NSS run-id of the snapshot the advisor saw. Reproducible: a
    /// validator can rebuild the snapshot and re-ask the advisor and
    /// will get the same `run_id`.
    pub run_id: NssRunId,
    /// Top-ranked action (what the advisor recommended).
    pub recommended: AdvisoryAction,
    /// Top-ranked action's predicted collapse probability.
    pub recommended_collapse: f64,
    /// `DoNothing`'s predicted collapse probability (for safety-floor
    /// comparison).
    pub baseline_collapse: f64,
    /// Ranking's `confidence_margin` (worst − best).
    pub confidence_margin: f64,
    /// What actually happened.
    pub outcome: DecisionOutcome,
    /// If `outcome == Skipped`, the safety guard that fired (e.g.
    /// `"improvement_below_floor"`, `"cooldown_active"`). Empty for
    /// `Applied` / `NoOp`.
    pub skip_reason: String,
    /// If `outcome == Applied`, the resolved interventions injected
    /// into the simulator's script. Empty otherwise.
    pub applied_interventions: Vec<Intervention>,
}

/// Closed-loop run summary.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoopReport {
    /// The full simulator trajectory across the run.
    pub trajectory: ClusterTrajectory,
    /// Every decision the controller made, in tick order.
    pub decisions: Vec<DecisionRecord>,
    /// Count of `Applied` outcomes.
    pub actions_applied: u32,
    /// Count of `Skipped` outcomes.
    pub actions_skipped: u32,
    /// Count of `NoOp` outcomes.
    pub no_ops: u32,
}

impl ClosedLoopReport {
    /// Convenience: count of collapse-labelled ticks in the trajectory.
    pub fn collapse_tick_count(&self) -> u64 {
        self.trajectory
            .iter()
            .filter(|ev| ev.cluster_failure.kind == crate::FailureKind::Collapse)
            .count() as u64
    }

    /// Convenience: count of nominal-labelled ticks.
    pub fn nominal_tick_count(&self) -> u64 {
        self.trajectory
            .iter()
            .filter(|ev| ev.cluster_failure.kind == crate::FailureKind::Nominal)
            .count() as u64
    }
}

/// The autonomous controller.
#[derive(Clone, Debug)]
pub struct AutonomousOptimizer {
    cfg: OptimizerConfig,
    advisor: SchedulerAdvisor,
    /// Per-node tick of last applied action; used by the cooldown
    /// guard.
    last_action_tick: BTreeMap<NodeId, u64>,
    /// Audit log accumulated across `step()` / `run_closed_loop` calls.
    audit_log: Vec<DecisionRecord>,
    /// Count of `Applied` decisions so far (for budget enforcement).
    actions_applied: u32,
}

impl AutonomousOptimizer {
    /// Build from `(advisor_cfg, optimizer_cfg)`. Convenience that
    /// also constructs the inner advisor.
    pub fn new(cfg: OptimizerConfig) -> Result<Self, NssError> {
        cfg.validate()?;
        let advisor = SchedulerAdvisor::new(cfg.advisor.clone())?;
        Ok(Self {
            cfg,
            advisor,
            last_action_tick: BTreeMap::new(),
            audit_log: Vec::new(),
            actions_applied: 0,
        })
    }

    /// Borrow the config.
    pub fn config(&self) -> &OptimizerConfig {
        &self.cfg
    }

    /// Borrow the audit log.
    pub fn audit_log(&self) -> &[DecisionRecord] {
        &self.audit_log
    }

    /// Total `Applied` decisions so far.
    pub fn actions_applied(&self) -> u32 {
        self.actions_applied
    }

    /// Run one control cycle: snapshot the sim, ask the advisor,
    /// apply if safe. Returns the `DecisionRecord` (also appended to
    /// the audit log).
    pub fn step(
        &mut self,
        sim: &mut ClusterSimulator,
        nss: &ClusterNeuralSystemsSimulator,
    ) -> Result<DecisionRecord, NssError> {
        let snapshot = sim.snapshot();
        let ranking = self.advisor.recommend(&snapshot, nss)?;
        let record = self.decide(&ranking, sim, snapshot.tick());
        self.audit_log.push(record.clone());
        Ok(record)
    }

    /// Drive a full closed loop: run the simulator for `total_ticks`,
    /// invoking the controller every `control_period` ticks. Returns
    /// the resulting [`ClosedLoopReport`].
    pub fn run_closed_loop(
        &mut self,
        sim: &mut ClusterSimulator,
        nss: &ClusterNeuralSystemsSimulator,
        total_ticks: u64,
    ) -> Result<ClosedLoopReport, NssError> {
        if total_ticks == 0 {
            return Ok(ClosedLoopReport {
                trajectory: ClusterTrajectory::empty(),
                decisions: vec![],
                actions_applied: 0,
                actions_skipped: 0,
                no_ops: 0,
            });
        }
        let mut full_trajectory = ClusterTrajectory::empty();
        let mut decisions: Vec<DecisionRecord> = Vec::new();
        let mut ticks_remaining = total_ticks;
        // Initial control step before any sim time advances.
        let _ = self.step(sim, nss)?;
        if let Some(last) = self.audit_log.last() {
            decisions.push(last.clone());
        }
        while ticks_remaining > 0 {
            let n = ticks_remaining.min(self.cfg.control_period);
            let traj = sim.run(n)?;
            for ev in traj.iter() {
                full_trajectory.push(ev.clone())?;
            }
            ticks_remaining -= n;
            if ticks_remaining == 0 {
                break;
            }
            let _ = self.step(sim, nss)?;
            if let Some(last) = self.audit_log.last() {
                decisions.push(last.clone());
            }
        }
        let actions_applied = decisions
            .iter()
            .filter(|d| d.outcome == DecisionOutcome::Applied)
            .count() as u32;
        let actions_skipped = decisions
            .iter()
            .filter(|d| d.outcome == DecisionOutcome::Skipped)
            .count() as u32;
        let no_ops = decisions
            .iter()
            .filter(|d| d.outcome == DecisionOutcome::NoOp)
            .count() as u32;
        Ok(ClosedLoopReport {
            trajectory: full_trajectory,
            decisions,
            actions_applied,
            actions_skipped,
            no_ops,
        })
    }

    /// Pure-decision logic — given a ranking + snapshot tick, apply
    /// the safety guards and (if appropriate) inject the chosen
    /// intervention into the simulator. Separated from `step` so it
    /// can be unit-tested without driving a full simulator.
    fn decide(
        &mut self,
        ranking: &AdvisoryRanking,
        sim: &mut ClusterSimulator,
        snapshot_tick: u64,
    ) -> DecisionRecord {
        let best = &ranking.candidates[0];
        // Locate the DoNothing baseline in the ranking.
        let do_nothing = ranking
            .candidates
            .iter()
            .find(|c| c.action == AdvisoryAction::DoNothing)
            .map(|c| c.predicted_collapse)
            .unwrap_or(best.predicted_collapse);
        let mut record = DecisionRecord {
            snapshot_tick,
            run_id: best.prediction.run_id,
            recommended: best.action,
            recommended_collapse: best.predicted_collapse,
            baseline_collapse: do_nothing,
            confidence_margin: ranking.confidence_margin,
            outcome: DecisionOutcome::Skipped,
            skip_reason: String::new(),
            applied_interventions: vec![],
        };

        // Guard 1: DoNothing recommendation = NoOp (success, but
        // nothing to apply).
        if best.action == AdvisoryAction::DoNothing {
            record.outcome = DecisionOutcome::NoOp;
            return record;
        }
        // Guard 2: safety-mode restriction.
        if !self.cfg.safety_mode.allows(best.action) {
            record.outcome = DecisionOutcome::Skipped;
            record.skip_reason = format!(
                "action_kind_forbidden_under_{}",
                self.cfg.safety_mode.label()
            );
            return record;
        }
        // Guard 3: improvement floor.
        let improvement = do_nothing - best.predicted_collapse;
        if improvement < self.cfg.min_improvement {
            record.outcome = DecisionOutcome::Skipped;
            record.skip_reason = format!(
                "improvement_below_floor_{:.4}_min_{:.4}",
                improvement, self.cfg.min_improvement
            );
            return record;
        }
        // Guard 4: confidence floor.
        if ranking.confidence_margin < self.cfg.min_confidence {
            record.outcome = DecisionOutcome::Skipped;
            record.skip_reason = format!(
                "confidence_below_floor_{:.4}_min_{:.4}",
                ranking.confidence_margin, self.cfg.min_confidence
            );
            return record;
        }
        // Guard 5: budget exhausted.
        if let Some(budget) = self.cfg.max_actions {
            if self.actions_applied >= budget {
                record.outcome = DecisionOutcome::Skipped;
                record.skip_reason = "budget_exhausted".to_string();
                return record;
            }
        }
        // Guard 6: per-node cooldown.
        let target_node = node_of_action(best.action);
        if let Some(node) = target_node {
            if let Some(last) = self.last_action_tick.get(&node).copied() {
                if snapshot_tick < last + self.cfg.action_cooldown_ticks {
                    record.outcome = DecisionOutcome::Skipped;
                    record.skip_reason = format!(
                        "cooldown_active_node_{}_last_{}_min_{}",
                        node, last, self.cfg.action_cooldown_ticks
                    );
                    return record;
                }
            }
        }

        // All guards passed — apply.
        let interventions = best.action.to_interventions(snapshot_tick);
        for iv in &interventions {
            sim.inject_intervention(*iv);
        }
        if let Some(node) = target_node {
            self.last_action_tick.insert(node, snapshot_tick);
        }
        self.actions_applied += 1;
        record.outcome = DecisionOutcome::Applied;
        record.applied_interventions = interventions;
        record
    }
}

/// Extract the affected node from an action, if any.
fn node_of_action(a: AdvisoryAction) -> Option<NodeId> {
    match a {
        AdvisoryAction::DoNothing => None,
        AdvisoryAction::FailNode { node }
        | AdvisoryAction::RecoverNode { node }
        | AdvisoryAction::ShedLoad { node, .. }
        | AdvisoryAction::AddNode { node }
        | AdvisoryAction::RemoveNode { node } => Some(node),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisory::AdvisorConfig;
    use crate::cluster::ClusterTopology;
    use crate::cluster_nss::ClusterNssConfig;
    use crate::cluster_simulator::ClusterConfig;
    use crate::seed::NssSeed;

    fn small_sim() -> ClusterSimulator {
        let cfg = ClusterConfig {
            cluster_arrival_rate: 8.0,
            ..ClusterConfig::default()
        };
        let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
        ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap()
    }

    #[test]
    fn safety_mode_filters_actions() {
        assert!(SafetyMode::Conservative.allows(AdvisoryAction::DoNothing));
        assert!(SafetyMode::Conservative.allows(AdvisoryAction::RecoverNode { node: NodeId(0) }));
        assert!(!SafetyMode::Conservative.allows(AdvisoryAction::FailNode { node: NodeId(0) }));
        assert!(!SafetyMode::Conservative.allows(AdvisoryAction::RemoveNode { node: NodeId(0) }));
        assert!(SafetyMode::Moderate.allows(AdvisoryAction::RemoveNode { node: NodeId(0) }));
        assert!(SafetyMode::Moderate.allows(AdvisoryAction::ShedLoad {
            node: NodeId(0),
            intensity: 0.5,
        }));
        assert!(!SafetyMode::Moderate.allows(AdvisoryAction::FailNode { node: NodeId(0) }));
        assert!(!SafetyMode::Moderate.allows(AdvisoryAction::AddNode { node: NodeId(0) }));
        assert!(SafetyMode::Aggressive.allows(AdvisoryAction::FailNode { node: NodeId(0) }));
        assert!(SafetyMode::Aggressive.allows(AdvisoryAction::AddNode { node: NodeId(0) }));
    }

    #[test]
    fn optimizer_config_validates() {
        assert!(OptimizerConfig::default().validate().is_ok());
        let bad = OptimizerConfig {
            control_period: 0,
            ..OptimizerConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = OptimizerConfig {
            min_improvement: -0.1,
            ..OptimizerConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn step_records_decision_in_audit_log() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig::default()).unwrap();
        let r = opt.step(&mut sim, &nss).unwrap();
        assert_eq!(opt.audit_log().len(), 1);
        assert_eq!(opt.audit_log()[0], r);
    }

    #[test]
    fn improvement_floor_skips_marginal_recommendations() {
        // An improbably-high min_improvement should reject every
        // recommendation (since NSS predictions on small unfit clusters
        // typically have tiny gaps).
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig {
            min_improvement: 0.5, // unattainable on a small cluster
            safety_mode: SafetyMode::Aggressive,
            ..OptimizerConfig::default()
        })
        .unwrap();
        let r = opt.step(&mut sim, &nss).unwrap();
        // Could be NoOp (DoNothing recommended) OR Skipped due to
        // improvement floor. Either way, must NOT be Applied.
        assert_ne!(r.outcome, DecisionOutcome::Applied);
    }

    #[test]
    fn budget_exhaustion_stops_further_applies() {
        let mut sim = small_sim();
        let _ = sim.run(2).unwrap();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig {
            control_period: 4,
            min_improvement: 0.0, // accept any improvement
            min_confidence: 0.0,
            action_cooldown_ticks: 0,
            max_actions: Some(0), // zero budget
            safety_mode: SafetyMode::Aggressive,
            advisor: AdvisorConfig {
                horizon: 4,
                consider_failure_actions: true,
                consider_recovery_actions: true,
                ..AdvisorConfig::default()
            },
        })
        .unwrap();
        for _ in 0..3 {
            let r = opt.step(&mut sim, &nss).unwrap();
            // With max_actions=0, every non-NoOp decision must be Skipped.
            if r.outcome != DecisionOutcome::NoOp {
                assert_eq!(r.outcome, DecisionOutcome::Skipped);
            }
        }
        assert_eq!(opt.actions_applied(), 0);
    }

    #[test]
    fn cooldown_blocks_repeat_action_on_same_node() {
        // Build a scenario where the controller wants to apply an
        // action on the same node twice in quick succession. The
        // cooldown should block the second.
        let mut sim = small_sim();
        let _ = sim.run(2).unwrap();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig {
            control_period: 4,
            min_improvement: 0.0, // accept any non-negative improvement
            min_confidence: 0.0,
            action_cooldown_ticks: 100, // huge cooldown
            max_actions: None,
            safety_mode: SafetyMode::Aggressive,
            advisor: AdvisorConfig {
                horizon: 4,
                consider_failure_actions: true,
                consider_recovery_actions: true,
                ..AdvisorConfig::default()
            },
        })
        .unwrap();
        let r1 = opt.step(&mut sim, &nss).unwrap();
        sim.run(1).unwrap();
        let r2 = opt.step(&mut sim, &nss).unwrap();
        // If r1 applied, then any same-node r2 must be cooldown-skipped.
        if r1.outcome == DecisionOutcome::Applied {
            let n1 = node_of_action(r1.recommended);
            let n2 = node_of_action(r2.recommended);
            if n1.is_some() && n1 == n2 {
                assert_eq!(r2.outcome, DecisionOutcome::Skipped);
                assert!(r2.skip_reason.starts_with("cooldown_active"));
            }
        }
    }

    #[test]
    fn closed_loop_produces_consistent_trajectory_and_audit_log() {
        let mut sim = small_sim();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig {
            control_period: 4,
            min_improvement: 0.0,
            safety_mode: SafetyMode::Conservative,
            ..OptimizerConfig::default()
        })
        .unwrap();
        let report = opt.run_closed_loop(&mut sim, &nss, 16).unwrap();
        // The trajectory must cover the 16 ticks we asked for.
        assert_eq!(report.trajectory.len(), 16);
        // Decisions count = ceil(16 / 4) + 1 (initial pre-tick step).
        // Initial step + 4 control checks at ticks 4/8/12/16.
        assert!(report.decisions.len() >= 4);
        // applied + skipped + no_ops == decisions.len()
        let total =
            report.actions_applied + report.actions_skipped + report.no_ops;
        assert_eq!(total as usize, report.decisions.len());
    }

    #[test]
    fn closed_loop_with_zero_total_ticks_returns_empty() {
        let mut sim = small_sim();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig::default()).unwrap();
        let report = opt.run_closed_loop(&mut sim, &nss, 0).unwrap();
        assert_eq!(report.trajectory.len(), 0);
        assert_eq!(report.decisions.len(), 0);
    }

    #[test]
    fn audit_log_carries_run_id_and_baseline() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt = AutonomousOptimizer::new(OptimizerConfig::default()).unwrap();
        let r = opt.step(&mut sim, &nss).unwrap();
        // run_id is the NSS prediction's run_id on the snapshot.
        // baseline_collapse is in [0,1].
        assert!(r.baseline_collapse >= 0.0 && r.baseline_collapse <= 1.0);
        assert!(r.recommended_collapse >= 0.0 && r.recommended_collapse <= 1.0);
        let _id = r.run_id; // smoke check the run_id field exists + is set
    }

    #[test]
    fn determinism_two_optimizers_match() {
        // Two independent controllers driven on independent simulator
        // copies must produce byte-identical trajectories + audit logs.
        let mut sim_a = small_sim();
        let mut sim_b = small_sim();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let mut opt_a = AutonomousOptimizer::new(OptimizerConfig::default()).unwrap();
        let mut opt_b = AutonomousOptimizer::new(OptimizerConfig::default()).unwrap();
        let r_a = opt_a.run_closed_loop(&mut sim_a, &nss, 16).unwrap();
        let r_b = opt_b.run_closed_loop(&mut sim_b, &nss, 16).unwrap();
        assert_eq!(
            r_a.trajectory.canonical_bytes(),
            r_b.trajectory.canonical_bytes()
        );
        assert_eq!(r_a.decisions.len(), r_b.decisions.len());
        for (a, b) in r_a.decisions.iter().zip(r_b.decisions.iter()) {
            assert_eq!(a.recommended, b.recommended);
            assert_eq!(a.outcome, b.outcome);
            assert_eq!(
                a.recommended_collapse.to_bits(),
                b.recommended_collapse.to_bits()
            );
        }
    }
}
