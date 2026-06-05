//! Phase 3c — scheduler advisory head.
//!
//! Given a cluster's current state (via [`crate::ClusterSnapshot`]),
//! enumerate the set of candidate actions a scheduler could take and
//! rank them by **predicted post-action collapse probability**. The
//! best-ranked action becomes the advisor's *recommendation*; the full
//! ranking and per-candidate counterfactual trajectory are returned for
//! audit purposes.
//!
//! ## Why counterfactual ranking instead of a learned policy head?
//!
//! Two reasons:
//!
//! 1. **Interpretability.** Every recommendation comes with the
//!    supporting trajectory: "fork B (apply action X) ran for H ticks
//!    and the NSS predicted P(collapse)=0.31, beating fork A
//!    (do-nothing) at P(collapse)=0.48". An operator can audit *why*
//!    by reading the trajectory. A learned policy head would emit a
//!    single number with no rationale.
//! 2. **Substrate reuse.** Phase 3a already shipped the snapshot/fork
//!    primitives. The advisor is mostly orchestration over them. A
//!    learned policy would need a separate reward signal + training
//!    loop; we'd add that complexity *for free* in Phase 4 when the
//!    autonomous-optimisation engine actually needs to act in
//!    real-time.
//!
//! ## Candidate action space (Phase 3c initial)
//!
//! - [`AdvisoryAction::DoNothing`] — the baseline. No intervention.
//! - [`AdvisoryAction::FailNode { node }`] — pre-emptive drain. Useful
//!   when an operator suspects a node is about to fail anyway and
//!   wants to *control* the failure instead of *enduring* it.
//! - [`AdvisoryAction::RecoverNode { node }`] — bring back a previously
//!   failed node (only valid if there are failed nodes in the
//!   snapshot).
//!
//! Phase 4 will extend this with autoscaling actions (`AddNode`,
//! `RemoveNode`) and load-shedding-policy adjustments.
//!
//! ## Determinism contract
//!
//! Same as Phase 3a — every fork is constructed deterministically from
//! the snapshot, so the entire advisory ranking is reproducible. Two
//! `SchedulerAdvisor::recommend` calls on the same snapshot + same NSS
//! produce identical rankings.

use crate::cluster::{NodeHealth, NodeId};
use crate::cluster_nss::{ClusterNeuralSystemsSimulator, ClusterPrediction};
use crate::cluster_simulator::Intervention;
use crate::counterfactual::ClusterSnapshot;
use crate::error::NssError;
use crate::pressure::PressureKind;

/// A candidate scheduler action the advisor can recommend.
///
/// **Phase 3e** — added autoscaling + intensity actions:
/// [`AdvisoryAction::ShedLoad`], [`AdvisoryAction::AddNode`], and
/// [`AdvisoryAction::RemoveNode`].
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum AdvisoryAction {
    /// No intervention. Always a valid candidate.
    DoNothing,
    /// Force-fail a node 1 tick after the snapshot — pre-emptive
    /// drain. Useful when an operator suspects the node is about
    /// to fail anyway and wants a controlled vs uncontrolled failure.
    FailNode {
        /// Which node to fail.
        node: NodeId,
    },
    /// Recover a previously failed node 1 tick after the snapshot.
    /// Only meaningful when the snapshot has at least one failed node.
    RecoverNode {
        /// Which node to recover.
        node: NodeId,
    },
    /// **Phase 3e** — apply admission control on a specific node at
    /// the given intensity. `intensity` is the shed fraction in
    /// `[0, 1]`: 0 = accept everything, 1 = reject all incoming work.
    ShedLoad {
        /// Which node.
        node: NodeId,
        /// Shed fraction; clipped on application.
        intensity: f64,
    },
    /// **Phase 3e** — autoscale up: bring an `Absent` node into the
    /// active cluster. The advisor enumerates this candidate for
    /// every `NodeHealth::Absent` slot.
    AddNode {
        /// Which (currently absent) node to add.
        node: NodeId,
    },
    /// **Phase 3e** — autoscale down: remove a `Healthy` node from
    /// the active cluster. The advisor enumerates this for every
    /// healthy node (when `consider_autoscaling` is enabled).
    RemoveNode {
        /// Which node to remove.
        node: NodeId,
    },
}

// Custom Eq + Ord + Hash because AdvisoryAction now contains f64.
impl Eq for AdvisoryAction {}

impl Ord for AdvisoryAction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by label so the ordering is total and deterministic
        // even across f64 values.
        self.label().cmp(&other.label())
    }
}

impl std::hash::Hash for AdvisoryAction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label().hash(state);
    }
}

impl AdvisoryAction {
    /// Canonical short label.
    pub fn label(self) -> String {
        match self {
            AdvisoryAction::DoNothing => "do_nothing".to_string(),
            AdvisoryAction::FailNode { node } => format!("fail_node_{}", node.0),
            AdvisoryAction::RecoverNode { node } => format!("recover_node_{}", node.0),
            AdvisoryAction::ShedLoad { node, intensity } => {
                // Embed intensity in label with 2-decimal precision so
                // two ShedLoad actions with different intensities have
                // distinct labels (used as sort key + map key).
                format!("shed_load_{}_{:0>4}", node.0, (intensity * 100.0) as u32)
            }
            AdvisoryAction::AddNode { node } => format!("add_node_{}", node.0),
            AdvisoryAction::RemoveNode { node } => format!("remove_node_{}", node.0),
        }
    }

    /// Convert to a list of [`Intervention`]s for forking. `DoNothing`
    /// returns an empty list; all other variants return a single
    /// intervention scheduled for `snapshot_tick + 1`.
    pub fn to_interventions(self, snapshot_tick: u64) -> Vec<Intervention> {
        let t = snapshot_tick + 1;
        match self {
            AdvisoryAction::DoNothing => vec![],
            AdvisoryAction::FailNode { node } => vec![Intervention::FailNode { tick: t, node }],
            AdvisoryAction::RecoverNode { node } => {
                vec![Intervention::RecoverNode { tick: t, node }]
            }
            AdvisoryAction::ShedLoad { node, intensity } => vec![Intervention::ShedLoadOverride {
                tick: t,
                node,
                intensity,
            }],
            AdvisoryAction::AddNode { node } => vec![Intervention::AddNode { tick: t, node }],
            AdvisoryAction::RemoveNode { node } => vec![Intervention::RemoveNode { tick: t, node }],
        }
    }
}

/// One candidate's evaluation: the action, the NSS prediction after
/// running the fork for the configured horizon, and a few quick-look
/// metrics for the audit log.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvisoryCandidate {
    /// The action that was simulated.
    pub action: AdvisoryAction,
    /// NSS prediction on the fork's final state.
    pub prediction: ClusterPrediction,
    /// Predicted post-action collapse probability. Pulled out of
    /// `prediction.failure.collapse_probability` for sort convenience.
    pub predicted_collapse: f64,
    /// Predicted degraded probability.
    pub predicted_degraded: f64,
    /// Dominant pressure kind at the fork's final state.
    pub dominant_kind: PressureKind,
    /// Number of collapse-labelled ticks in the fork's post-snapshot
    /// trajectory. Cross-checks the NSS prediction against the
    /// simulator's actual labels.
    pub collapse_tick_count: u64,
}

/// Full advisor output. `candidates` is sorted ascending by
/// `predicted_collapse`, so `candidates[0]` is the best action.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvisoryRanking {
    /// All evaluated candidates, sorted ascending by
    /// `predicted_collapse` (best first). `f64::total_cmp` is used for
    /// the sort so NaN handling is deterministic.
    pub candidates: Vec<AdvisoryCandidate>,
    /// The recommended action — equivalent to `candidates[0].action`.
    pub recommended: AdvisoryAction,
    /// `predicted_collapse` of the worst candidate minus that of the
    /// best. Larger gap = more confident recommendation.
    pub confidence_margin: f64,
    /// Snapshot tick the recommendation was made at.
    pub snapshot_tick: u64,
    /// Horizon used for the counterfactual rollouts.
    pub horizon: u64,
}

impl AdvisoryRanking {
    /// True if there's only one candidate (trivially recommended).
    pub fn is_trivial(self) -> bool {
        self.candidates.len() <= 1
    }
}

/// Knobs for the advisor.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvisorConfig {
    /// Counterfactual horizon — ticks each fork runs forward before
    /// the NSS predicts on the final state. Must be ≥ 1. Default 12.
    pub horizon: u64,
    /// If `true`, the advisor enumerates a `FailNode` candidate for
    /// every healthy node in the snapshot. Default `false` because
    /// "pre-emptive drain" is rarely the right answer; turn on when
    /// you actively want to consider controlled failures.
    pub consider_failure_actions: bool,
    /// If `true`, enumerates a `RecoverNode` candidate for every
    /// failed node. Default `true` — recovery is almost always a
    /// sensible candidate when nodes are down.
    pub consider_recovery_actions: bool,
    /// **Phase 3e** — if `true`, enumerate `ShedLoad` candidates
    /// (one per healthy node) at each intensity in `shed_intensities`.
    /// Default `false` to keep the default candidate set small;
    /// operators who want explicit shed-load advisories enable this.
    pub consider_shed_load: bool,
    /// **Phase 3e** — intensities to enumerate when `consider_shed_load`
    /// is on. Default `[0.25, 0.50, 0.75]`. Each entry must be in
    /// `[0, 1]`.
    pub shed_intensities: Vec<f64>,
    /// **Phase 3e** — if `true`, enumerate `AddNode` (for every
    /// `Absent` node) and `RemoveNode` (for every `Healthy` node)
    /// candidates. Default `false` because autoscaling is the
    /// most operationally consequential category.
    pub consider_autoscaling: bool,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            horizon: 12,
            consider_failure_actions: false,
            consider_recovery_actions: true,
            consider_shed_load: false,
            shed_intensities: vec![0.25, 0.50, 0.75],
            consider_autoscaling: false,
        }
    }
}

impl AdvisorConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.horizon == 0 {
            return Err(NssError::InvalidConfig {
                detail: "AdvisorConfig.horizon must be >= 1".into(),
            });
        }
        for i in &self.shed_intensities {
            if !i.is_finite() || !(0.0..=1.0).contains(i) {
                return Err(NssError::InvalidConfig {
                    detail: format!("shed_intensity {} must be in [0, 1] and finite", i),
                });
            }
        }
        Ok(())
    }
}

/// The scheduler advisory head. Holds an [`AdvisorConfig`]; the
/// recommendation method takes a snapshot + an NSS predictor.
///
/// Phase 3e — `Copy` was dropped because `AdvisorConfig` now contains a
/// `Vec<f64>` (`shed_intensities`). The advisor is `Clone`, which is
/// the right semantic anyway — the advisor's state is just its config.
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerAdvisor {
    cfg: AdvisorConfig,
}

impl SchedulerAdvisor {
    /// Build with the given config.
    pub fn new(cfg: AdvisorConfig) -> Result<Self, NssError> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Build with default config.
    pub fn default() -> Self {
        Self::new(AdvisorConfig::default()).expect("default valid")
    }

    /// Borrow the config.
    pub fn config(&self) -> &AdvisorConfig {
        &self.cfg
    }

    /// Enumerate the candidate-action set for a given snapshot, based
    /// on what's in the snapshot (which nodes are healthy / failed)
    /// and the advisor's config flags.
    ///
    /// **Phase 3e** — extended to include `ShedLoad` (per healthy node
    /// × `shed_intensities`), `AddNode` (per absent node), and
    /// `RemoveNode` (per healthy node) when their respective config
    /// flags are enabled.
    pub fn enumerate_candidates(&self, snapshot: &ClusterSnapshot) -> Vec<AdvisoryAction> {
        let mut candidates: Vec<AdvisoryAction> = vec![AdvisoryAction::DoNothing];
        let sim = snapshot.peek();
        // Phase 3c: enumerate against the topology. The
        // `consider_*` flags + the eventual fork validation prune
        // invalid candidates. A FailNode against an already-failed
        // node is a no-op in the simulator (documented in
        // `cluster_simulator.rs`); same for RecoverNode against an
        // already-healthy node. So the candidate set is *safe* to
        // be wider than strictly necessary.
        for node in sim.topology().nodes() {
            if self.cfg.consider_failure_actions {
                candidates.push(AdvisoryAction::FailNode { node });
            }
            if self.cfg.consider_recovery_actions {
                candidates.push(AdvisoryAction::RecoverNode { node });
            }
            // Phase 3e — autoscaling + shed-load actions.
            if self.cfg.consider_autoscaling {
                candidates.push(AdvisoryAction::AddNode { node });
                candidates.push(AdvisoryAction::RemoveNode { node });
            }
            if self.cfg.consider_shed_load {
                for intensity in &self.cfg.shed_intensities {
                    candidates.push(AdvisoryAction::ShedLoad {
                        node,
                        intensity: *intensity,
                    });
                }
            }
        }
        candidates
    }

    /// **Phase 3e** — per-node ranking. For each node in the
    /// snapshot's topology, build a *node-scoped* candidate set
    /// (`DoNothing` + node-specific actions) and run `recommend` over
    /// that set. Returns `BTreeMap<NodeId, AdvisoryRanking>` —
    /// deterministic iteration in `NodeId` order.
    ///
    /// This surfaces operator-actionable per-node recommendations:
    /// "node 2 should be recovered (high confidence), node 3 should
    /// be left alone (low confidence)." Much more useful for triaging
    /// a partially-degraded cluster than a single cluster-wide
    /// recommendation.
    pub fn recommend_per_node(
        &self,
        snapshot: &ClusterSnapshot,
        nss: &ClusterNeuralSystemsSimulator,
    ) -> Result<std::collections::BTreeMap<NodeId, AdvisoryRanking>, NssError> {
        let sim = snapshot.peek();
        let mut out: std::collections::BTreeMap<NodeId, AdvisoryRanking> =
            std::collections::BTreeMap::new();
        for node in sim.topology().nodes() {
            // Per-node candidate set: DoNothing + the node-scoped
            // actions enabled by config.
            let mut node_candidates: Vec<AdvisoryAction> = vec![AdvisoryAction::DoNothing];
            if self.cfg.consider_failure_actions {
                node_candidates.push(AdvisoryAction::FailNode { node });
            }
            if self.cfg.consider_recovery_actions {
                node_candidates.push(AdvisoryAction::RecoverNode { node });
            }
            if self.cfg.consider_autoscaling {
                node_candidates.push(AdvisoryAction::AddNode { node });
                node_candidates.push(AdvisoryAction::RemoveNode { node });
            }
            if self.cfg.consider_shed_load {
                for intensity in &self.cfg.shed_intensities {
                    node_candidates.push(AdvisoryAction::ShedLoad {
                        node,
                        intensity: *intensity,
                    });
                }
            }
            // Score each candidate.
            let mut scored: Vec<AdvisoryCandidate> = Vec::with_capacity(node_candidates.len());
            for action in node_candidates {
                scored.push(self.score_candidate(snapshot, action, nss)?);
            }
            scored.sort_by(|a, b| {
                a.predicted_collapse
                    .total_cmp(&b.predicted_collapse)
                    .then_with(|| a.predicted_degraded.total_cmp(&b.predicted_degraded))
                    .then_with(|| a.action.label().cmp(&b.action.label()))
            });
            let recommended = scored[0].action;
            let confidence_margin = if scored.len() >= 2 {
                scored[scored.len() - 1].predicted_collapse - scored[0].predicted_collapse
            } else {
                0.0
            };
            out.insert(
                node,
                AdvisoryRanking {
                    candidates: scored,
                    recommended,
                    confidence_margin,
                    snapshot_tick: snapshot.tick(),
                    horizon: self.cfg.horizon,
                },
            );
        }
        Ok(out)
    }

    /// Score one candidate by forking the snapshot, applying the
    /// action's interventions at `snapshot_tick + 1`, running the
    /// fork forward `horizon` ticks, and asking the NSS to predict on
    /// the final state. Returns the candidate's evaluation.
    pub fn score_candidate(
        &self,
        snapshot: &ClusterSnapshot,
        action: AdvisoryAction,
        nss: &ClusterNeuralSystemsSimulator,
    ) -> Result<AdvisoryCandidate, NssError> {
        let interventions = action.to_interventions(snapshot.tick());
        let mut fork = snapshot.fork(interventions)?;
        let traj = fork.run(self.cfg.horizon)?;
        let last = traj.last_state().ok_or_else(|| NssError::InvalidTrajectory {
            detail: "candidate fork produced empty trajectory".into(),
        })?;
        let prediction = nss.predict_next(last)?;
        let collapse_tick_count = traj
            .iter()
            .filter(|ev| ev.cluster_failure.kind == crate::FailureKind::Collapse)
            .count() as u64;
        Ok(AdvisoryCandidate {
            action,
            predicted_collapse: prediction.failure.collapse_probability,
            predicted_degraded: prediction.failure.degraded_probability,
            dominant_kind: prediction.attribution.dominant_contribution.kind,
            collapse_tick_count,
            prediction,
        })
    }

    /// Score every enumerated candidate and return the full ranking.
    /// `candidates` is sorted by predicted post-action collapse
    /// probability (best first).
    pub fn recommend(
        &self,
        snapshot: &ClusterSnapshot,
        nss: &ClusterNeuralSystemsSimulator,
    ) -> Result<AdvisoryRanking, NssError> {
        let candidates_in = self.enumerate_candidates(snapshot);
        let mut candidates: Vec<AdvisoryCandidate> = Vec::with_capacity(candidates_in.len());
        for action in candidates_in {
            candidates.push(self.score_candidate(snapshot, action, nss)?);
        }
        // Sort by predicted_collapse ascending (best first).
        // Ties broken by predicted_degraded, then by action label so
        // the result is fully deterministic.
        candidates.sort_by(|a, b| {
            a.predicted_collapse
                .total_cmp(&b.predicted_collapse)
                .then_with(|| a.predicted_degraded.total_cmp(&b.predicted_degraded))
                .then_with(|| a.action.label().cmp(&b.action.label()))
        });
        let recommended = candidates[0].action;
        let confidence_margin = if candidates.len() >= 2 {
            candidates[candidates.len() - 1].predicted_collapse - candidates[0].predicted_collapse
        } else {
            0.0
        };
        Ok(AdvisoryRanking {
            candidates,
            recommended,
            confidence_margin,
            snapshot_tick: snapshot.tick(),
            horizon: self.cfg.horizon,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterTopology;
    use crate::cluster_nss::ClusterNssConfig;
    use crate::cluster_simulator::{ClusterConfig, ClusterSimulator};
    use crate::seed::NssSeed;

    fn small_sim() -> ClusterSimulator {
        let cfg = ClusterConfig {
            cluster_arrival_rate: 6.0,
            ..ClusterConfig::default()
        };
        let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
        ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap()
    }

    #[test]
    fn advisor_config_validates() {
        assert!(AdvisorConfig::default().validate().is_ok());
        let bad = AdvisorConfig {
            horizon: 0,
            ..AdvisorConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn enumerate_candidates_always_includes_do_nothing() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let advisor = SchedulerAdvisor::default();
        let candidates = advisor.enumerate_candidates(&snap);
        assert!(candidates.contains(&AdvisoryAction::DoNothing));
    }

    #[test]
    fn enumerate_candidates_respects_config_flags() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        // Only do-nothing + recoveries.
        let advisor = SchedulerAdvisor::new(AdvisorConfig {
            horizon: 8,
            consider_failure_actions: false,
            consider_recovery_actions: true,
            ..AdvisorConfig::default()
        })
        .unwrap();
        let c = advisor.enumerate_candidates(&snap);
        // 1 do-nothing + 3 recovery candidates (one per node) = 4
        assert_eq!(c.len(), 4);
        for cand in &c {
            assert!(matches!(
                cand,
                AdvisoryAction::DoNothing | AdvisoryAction::RecoverNode { .. }
            ));
        }
    }

    #[test]
    fn recommend_returns_sorted_candidates() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::new(AdvisorConfig {
            horizon: 6,
            consider_failure_actions: true,
            consider_recovery_actions: true,
            ..AdvisorConfig::default()
        })
        .unwrap();
        let ranking = advisor.recommend(&snap, &nss).unwrap();
        for w in ranking.candidates.windows(2) {
            assert!(
                w[0].predicted_collapse <= w[1].predicted_collapse,
                "candidates must be sorted ascending by predicted_collapse"
            );
        }
        assert_eq!(ranking.recommended, ranking.candidates[0].action);
    }

    #[test]
    fn recommend_is_deterministic() {
        // Two recommendations on the same snapshot + NSS must agree
        // bit-for-bit on every candidate's predicted_collapse.
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::default();
        let r1 = advisor.recommend(&snap, &nss).unwrap();
        let r2 = advisor.recommend(&snap, &nss).unwrap();
        assert_eq!(r1.candidates.len(), r2.candidates.len());
        for (a, b) in r1.candidates.iter().zip(r2.candidates.iter()) {
            assert_eq!(a.action, b.action);
            assert_eq!(
                a.predicted_collapse.to_bits(),
                b.predicted_collapse.to_bits(),
                "predicted_collapse must be bit-identical across recommend() calls"
            );
        }
        assert_eq!(r1.recommended, r2.recommended);
    }

    #[test]
    fn recommend_for_failed_node_cluster_includes_recovery_in_top_candidates() {
        // Inject a node failure into the simulator, snapshot, then
        // ask the advisor. The recovery action for the failed node
        // should not be dominated by DoNothing.
        let cfg = ClusterConfig {
            cluster_arrival_rate: 8.0,
            ..ClusterConfig::default()
        };
        let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
        let mut sim = ClusterSimulator::new(
            cfg,
            top,
            NssSeed(42),
            vec![Intervention::FailNode {
                tick: 3,
                node: NodeId(1),
            }],
        )
        .unwrap();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::default();
        let ranking = advisor.recommend(&snap, &nss).unwrap();
        // The candidates must include a RecoverNode for the failed node.
        let recovers_failed = ranking
            .candidates
            .iter()
            .any(|c| c.action == AdvisoryAction::RecoverNode { node: NodeId(1) });
        assert!(recovers_failed, "advisor must enumerate recovery for failed node");
    }

    #[test]
    fn score_candidate_produces_finite_prediction() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::default();
        let cand = advisor
            .score_candidate(&snap, AdvisoryAction::DoNothing, &nss)
            .unwrap();
        assert!(cand.predicted_collapse.is_finite());
        assert!(cand.predicted_collapse >= 0.0 && cand.predicted_collapse <= 1.0);
        assert!(cand.predicted_degraded.is_finite());
    }

    #[test]
    fn action_label_is_stable() {
        assert_eq!(AdvisoryAction::DoNothing.label(), "do_nothing");
        assert_eq!(
            AdvisoryAction::FailNode { node: NodeId(2) }.label(),
            "fail_node_2"
        );
        assert_eq!(
            AdvisoryAction::RecoverNode { node: NodeId(5) }.label(),
            "recover_node_5"
        );
    }

    // ---- Phase 3e advisor extension tests ----

    #[test]
    fn shed_load_action_label_distinguishes_intensities() {
        let a = AdvisoryAction::ShedLoad {
            node: NodeId(2),
            intensity: 0.25,
        };
        let b = AdvisoryAction::ShedLoad {
            node: NodeId(2),
            intensity: 0.75,
        };
        assert_ne!(a.label(), b.label());
        // Sanity: same intensity = same label.
        let c = AdvisoryAction::ShedLoad {
            node: NodeId(2),
            intensity: 0.25,
        };
        assert_eq!(a.label(), c.label());
    }

    #[test]
    fn shed_load_to_interventions_carries_intensity() {
        let action = AdvisoryAction::ShedLoad {
            node: NodeId(1),
            intensity: 0.5,
        };
        let ivs = action.to_interventions(10);
        assert_eq!(ivs.len(), 1);
        match ivs[0] {
            Intervention::ShedLoadOverride {
                tick: 11,
                node,
                intensity,
            } => {
                assert_eq!(node, NodeId(1));
                assert!((intensity - 0.5).abs() < 1e-12);
            }
            _ => panic!("expected ShedLoadOverride"),
        }
    }

    #[test]
    fn add_remove_node_to_interventions_map_correctly() {
        let add = AdvisoryAction::AddNode { node: NodeId(3) };
        let rem = AdvisoryAction::RemoveNode { node: NodeId(3) };
        assert!(matches!(
            add.to_interventions(5)[0],
            Intervention::AddNode { tick: 6, .. }
        ));
        assert!(matches!(
            rem.to_interventions(5)[0],
            Intervention::RemoveNode { tick: 6, .. }
        ));
    }

    #[test]
    fn config_rejects_out_of_range_intensity() {
        let bad = AdvisorConfig {
            shed_intensities: vec![1.5],
            ..AdvisorConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = AdvisorConfig {
            shed_intensities: vec![-0.1],
            ..AdvisorConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn enumerate_candidates_emits_shed_load_when_enabled() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let snap = sim.snapshot();
        let advisor = SchedulerAdvisor::new(AdvisorConfig {
            horizon: 4,
            consider_failure_actions: false,
            consider_recovery_actions: false,
            consider_shed_load: true,
            shed_intensities: vec![0.25, 0.50, 0.75],
            consider_autoscaling: false,
        })
        .unwrap();
        let c = advisor.enumerate_candidates(&snap);
        // 1 do-nothing + 3 nodes * 3 intensities = 10 candidates.
        assert_eq!(c.len(), 10);
        let shed_count = c
            .iter()
            .filter(|x| matches!(x, AdvisoryAction::ShedLoad { .. }))
            .count();
        assert_eq!(shed_count, 9);
    }

    #[test]
    fn enumerate_candidates_emits_autoscaling_when_enabled() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let snap = sim.snapshot();
        let advisor = SchedulerAdvisor::new(AdvisorConfig {
            horizon: 4,
            consider_failure_actions: false,
            consider_recovery_actions: false,
            consider_shed_load: false,
            consider_autoscaling: true,
            ..AdvisorConfig::default()
        })
        .unwrap();
        let c = advisor.enumerate_candidates(&snap);
        // 1 do-nothing + 3 nodes * 2 actions (Add/Remove) = 7
        assert_eq!(c.len(), 7);
        let add_count = c
            .iter()
            .filter(|x| matches!(x, AdvisoryAction::AddNode { .. }))
            .count();
        assert_eq!(add_count, 3);
    }

    #[test]
    fn recommend_per_node_returns_one_ranking_per_node() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::new(AdvisorConfig {
            horizon: 4,
            consider_failure_actions: true,
            consider_recovery_actions: true,
            consider_shed_load: true,
            shed_intensities: vec![0.5],
            consider_autoscaling: false,
            ..AdvisorConfig::default()
        })
        .unwrap();
        let per_node = advisor.recommend_per_node(&snap, &nss).unwrap();
        // 3 nodes in small_sim's topology → 3 rankings.
        assert_eq!(per_node.len(), 3);
        for (id, ranking) in &per_node {
            // Each ranking should contain only actions targeting THIS node.
            for cand in &ranking.candidates {
                let n = match cand.action {
                    AdvisoryAction::DoNothing => None,
                    AdvisoryAction::FailNode { node } => Some(node),
                    AdvisoryAction::RecoverNode { node } => Some(node),
                    AdvisoryAction::ShedLoad { node, .. } => Some(node),
                    AdvisoryAction::AddNode { node } => Some(node),
                    AdvisoryAction::RemoveNode { node } => Some(node),
                };
                if let Some(n) = n {
                    assert_eq!(n, *id, "per-node ranking must only contain actions for node {}", id);
                }
            }
        }
    }

    #[test]
    fn recommend_per_node_is_deterministic() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::default();
        let a = advisor.recommend_per_node(&snap, &nss).unwrap();
        let b = advisor.recommend_per_node(&snap, &nss).unwrap();
        assert_eq!(a.len(), b.len());
        for (id, ra) in &a {
            let rb = &b[id];
            assert_eq!(ra.recommended, rb.recommended);
            assert_eq!(ra.candidates.len(), rb.candidates.len());
            for (ca, cb) in ra.candidates.iter().zip(rb.candidates.iter()) {
                assert_eq!(
                    ca.predicted_collapse.to_bits(),
                    cb.predicted_collapse.to_bits(),
                );
            }
        }
    }

    #[test]
    fn ranking_confidence_margin_is_nonneg() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let advisor = SchedulerAdvisor::default();
        let ranking = advisor.recommend(&snap, &nss).unwrap();
        assert!(ranking.confidence_margin >= 0.0);
    }
}

#[allow(dead_code)]
fn _suppress_unused_node_health() {
    // Silence the unused-import warning for `NodeHealth` — kept in
    // scope because the advisor's docstring references it and so a
    // future variant (e.g. `MarkDraining { node }`) can use it without
    // re-importing.
    let _ = NodeHealth::Healthy;
}
