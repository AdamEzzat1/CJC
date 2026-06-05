//! Phase 3a — counterfactual forking.
//!
//! "What would have happened if node 7 had failed at tick 16?"
//!
//! Counterfactual reasoning runs two (or more) simulator instances
//! from a **shared mid-run state**, applies different interventions,
//! and compares the resulting NSS predictions. The cost is structural:
//!
//! 1. `simulator.snapshot()` returns a [`ClusterSnapshot`] /
//!    [`GpuTrainingSnapshot`] — literally a `Clone` of the simulator's
//!    full runtime state, including every RNG (SplitMix64 = `u64`),
//!    every per-node queue, the in-flight intervention script, and
//!    the tick counter.
//! 2. `snapshot.fork(extra_interventions)` returns a fresh simulator
//!    instance that *continues from the snapshot tick* with the
//!    additional interventions merged into the script in canonical
//!    `(tick, node, kind)` order.
//! 3. After running each fork forward, the NSS predicts on the
//!    final state of each. [`CounterfactualComparison`] surfaces the
//!    cross-fork divergence — collapse-probability delta, dominant
//!    node flip, cluster-label flip, per-node disagreement count.
//!
//! ## Why this is essentially free
//!
//! The Phase 1 + Phase 2 determinism contract pays off here: every
//! piece of simulator state is `Copy` or `Clone`, every RNG is a `u64`,
//! every map is a `BTreeMap`. A snapshot is literally a `Clone`. No
//! separate serialization layer is needed — the simulator's runtime
//! state already *is* the serializable form.
//!
//! ## Order-independence of fork interventions
//!
//! `snapshot.fork(extras)` appends `extras` to a clone of the base
//! script and re-sorts. The script is sorted by `(tick, node, kind)`
//! at simulator construction time, so the resulting fork's
//! intervention order is independent of the user's input order — same
//! property the cluster simulator already guarantees.

use crate::cluster::{ClusterTrajectory, NodeHealth, NodeId};
use crate::cluster_nss::{ClusterNeuralSystemsSimulator, ClusterPrediction};
use crate::cluster_simulator::{ClusterSimulator, Intervention};
use crate::error::NssError;
use crate::failure::FailureKind;
use crate::gpu_training::GpuTrainingSimulator;
use crate::pressure::PressureKind;
use std::collections::BTreeMap;

/// A frozen snapshot of a [`ClusterSimulator`] at some tick. Created
/// via [`ClusterSimulator::snapshot`]. Forking is via [`ClusterSnapshot::fork`].
///
/// Snapshots are independent of the originating simulator — taking a
/// snapshot does **not** consume the simulator, and running the
/// original after snapshotting does not affect the snapshot's state.
/// This is the structural payoff of `Clone`: copy semantics throughout.
#[derive(Clone, Debug)]
pub struct ClusterSnapshot {
    inner: ClusterSimulator,
}

impl ClusterSnapshot {
    /// Tick the snapshot was taken at.
    pub fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Fork this snapshot into an independent [`ClusterSimulator`]
    /// instance with `extra_interventions` merged into the
    /// intervention script. The base interventions are preserved;
    /// only ones with `tick >= snapshot.tick()` will actually fire
    /// on the fork. Returns an error if any extra intervention
    /// targets an unknown node or fires at a tick already past the
    /// snapshot tick (the latter would be a no-op but signals
    /// caller confusion).
    pub fn fork(&self, extra_interventions: Vec<Intervention>) -> Result<ClusterSimulator, NssError> {
        let mut sim = self.inner.clone();
        let snap_tick = sim.tick();
        for iv in &extra_interventions {
            if iv.tick() < snap_tick {
                return Err(NssError::InvalidConfig {
                    detail: format!(
                        "fork intervention tick {} < snapshot tick {}; would never fire",
                        iv.tick(),
                        snap_tick,
                    ),
                });
            }
            if !sim.topology().nodes().any(|n| n == iv.node()) {
                return Err(NssError::InvalidConfig {
                    detail: format!("fork intervention targets unknown node {}", iv.node()),
                });
            }
        }
        sim.append_and_sort_interventions(extra_interventions);
        Ok(sim)
    }

    /// Borrow the underlying simulator (read-only). Use [`fork`] to
    /// get a mutable copy you can run.
    pub fn peek(&self) -> &ClusterSimulator {
        &self.inner
    }
}

/// A frozen snapshot of a [`GpuTrainingSimulator`]. Same semantics as
/// [`ClusterSnapshot`].
#[derive(Clone, Debug)]
pub struct GpuTrainingSnapshot {
    inner: GpuTrainingSimulator,
}

impl GpuTrainingSnapshot {
    /// Tick the snapshot was taken at.
    pub fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Fork into an independent GPU training simulator with extra
    /// interventions merged.
    pub fn fork(
        &self,
        extra_interventions: Vec<Intervention>,
    ) -> Result<GpuTrainingSimulator, NssError> {
        let mut sim = self.inner.clone();
        let snap_tick = sim.tick();
        for iv in &extra_interventions {
            if iv.tick() < snap_tick {
                return Err(NssError::InvalidConfig {
                    detail: format!(
                        "fork intervention tick {} < snapshot tick {}; would never fire",
                        iv.tick(),
                        snap_tick,
                    ),
                });
            }
            if !sim.topology().nodes().any(|n| n == iv.node()) {
                return Err(NssError::InvalidConfig {
                    detail: format!("fork intervention targets unknown gpu {}", iv.node()),
                });
            }
        }
        sim.append_and_sort_interventions(extra_interventions);
        Ok(sim)
    }

    /// Borrow the underlying simulator (read-only).
    pub fn peek(&self) -> &GpuTrainingSimulator {
        &self.inner
    }
}

/// One fork's outcome: post-fork trajectory + NSS prediction on its
/// last state.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualOutcome {
    /// Human-readable label for this fork (e.g. "no-op", "fail-node-3").
    pub label: String,
    /// Post-snapshot trajectory of this fork.
    pub trajectory: ClusterTrajectory,
    /// NSS prediction on the fork's final state.
    pub prediction: ClusterPrediction,
}

impl CounterfactualOutcome {
    /// Cluster-failure label distribution across the post-snapshot
    /// trajectory: `(nominal, degraded, collapse)`.
    pub fn label_distribution(&self) -> (u64, u64, u64) {
        let mut nom = 0u64;
        let mut deg = 0u64;
        let mut col = 0u64;
        for ev in self.trajectory.iter() {
            match ev.cluster_failure.kind {
                FailureKind::Nominal => nom += 1,
                FailureKind::Degraded => deg += 1,
                FailureKind::Collapse => col += 1,
            }
        }
        (nom, deg, col)
    }

    /// Mean per-tick queue saturation across all nodes — useful as a
    /// coarse "how heavy is the workload?" summary across the fork.
    pub fn mean_queue_saturation(&self) -> f64 {
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        let mut n = 0u64;
        for ev in self.trajectory.iter() {
            for s in ev.state.nodes.values() {
                acc.add(
                    s.pressures
                        .get(PressureKind::Queue)
                        .map(|p| p.saturation())
                        .unwrap_or(0.0),
                );
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            acc.finalize() / n as f64
        }
    }
}

/// Comparison between two forks. The prediction divergence is the
/// analytically-interesting payoff.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualComparison {
    /// Outcome of fork A (typically "baseline" / no-op).
    pub a: CounterfactualOutcome,
    /// Outcome of fork B (typically "intervention applied").
    pub b: CounterfactualOutcome,
    /// `P(collapse_b) - P(collapse_a)`. Positive = intervention
    /// increased collapse risk; negative = intervention decreased it.
    pub collapse_probability_delta: f64,
    /// `P(degraded_b) - P(degraded_a)`.
    pub degraded_probability_delta: f64,
    /// `true` if the rolled-up cluster failure label at the final
    /// state differs between the two forks (e.g. nominal vs collapse).
    pub final_label_flipped: bool,
    /// `true` if the NSS-attributed dominant node changed across the
    /// two forks.
    pub dominant_node_flipped: bool,
    /// `true` if the NSS-attributed dominant pressure kind changed.
    pub dominant_kind_flipped: bool,
    /// Per-node "did this node end up in different health states at
    /// the final tick?" — captures whether the intervention rerouted
    /// failure topology.
    pub node_health_disagreements: BTreeMap<NodeId, (NodeHealth, NodeHealth)>,
}

impl CounterfactualComparison {
    /// Build a comparison from two paired outcomes.
    pub fn between(a: CounterfactualOutcome, b: CounterfactualOutcome) -> Self {
        let collapse_delta = b.prediction.failure.collapse_probability
            - a.prediction.failure.collapse_probability;
        let degraded_delta = b.prediction.failure.degraded_probability
            - a.prediction.failure.degraded_probability;
        let a_label = a
            .trajectory
            .iter()
            .last()
            .map(|ev| ev.cluster_failure.kind)
            .unwrap_or(FailureKind::Nominal);
        let b_label = b
            .trajectory
            .iter()
            .last()
            .map(|ev| ev.cluster_failure.kind)
            .unwrap_or(FailureKind::Nominal);
        let final_label_flipped = a_label != b_label;
        let dominant_node_flipped =
            a.prediction.attribution.dominant_node != b.prediction.attribution.dominant_node;
        let dominant_kind_flipped = a.prediction.attribution.dominant_contribution.kind
            != b.prediction.attribution.dominant_contribution.kind;
        // Node-health disagreements at the final state.
        let mut node_health_disagreements = BTreeMap::new();
        if let (Some(sa), Some(sb)) = (a.trajectory.last_state(), b.trajectory.last_state()) {
            for id in sa.nodes.keys() {
                let ha = sa.node_health.get(id).copied().unwrap_or(NodeHealth::Healthy);
                let hb = sb.node_health.get(id).copied().unwrap_or(NodeHealth::Healthy);
                if ha != hb {
                    node_health_disagreements.insert(*id, (ha, hb));
                }
            }
        }
        Self {
            a,
            b,
            collapse_probability_delta: collapse_delta,
            degraded_probability_delta: degraded_delta,
            final_label_flipped,
            dominant_node_flipped,
            dominant_kind_flipped,
            node_health_disagreements,
        }
    }

    /// True if the comparison shows the intervention made things
    /// *materially* worse: P(collapse) increased by ≥ `threshold` AND
    /// at least one of the structural-flip flags fired.
    pub fn intervention_is_harmful(&self, threshold: f64) -> bool {
        self.collapse_probability_delta >= threshold
            && (self.final_label_flipped
                || self.dominant_node_flipped
                || !self.node_health_disagreements.is_empty())
    }

    /// True if the comparison shows the intervention strictly
    /// improved the outlook by `threshold`.
    pub fn intervention_is_beneficial(&self, threshold: f64) -> bool {
        self.collapse_probability_delta <= -threshold
    }
}

/// Run a counterfactual experiment: snapshot, fork into two named
/// branches, run each `horizon` ticks, predict with `nss` on the
/// final state, compare. Convenience for the common "A vs B" case.
pub fn run_cluster_counterfactual(
    snapshot: &ClusterSnapshot,
    label_a: &str,
    interventions_a: Vec<Intervention>,
    label_b: &str,
    interventions_b: Vec<Intervention>,
    horizon: u64,
    nss: &ClusterNeuralSystemsSimulator,
) -> Result<CounterfactualComparison, NssError> {
    let mut fork_a = snapshot.fork(interventions_a)?;
    let mut fork_b = snapshot.fork(interventions_b)?;
    let traj_a = fork_a.run(horizon)?;
    let traj_b = fork_b.run(horizon)?;
    let pred_a = nss.predict_next(traj_a.last_state().ok_or_else(|| {
        NssError::InvalidTrajectory {
            detail: "fork A produced empty trajectory".into(),
        }
    })?)?;
    let pred_b = nss.predict_next(traj_b.last_state().ok_or_else(|| {
        NssError::InvalidTrajectory {
            detail: "fork B produced empty trajectory".into(),
        }
    })?)?;
    Ok(CounterfactualComparison::between(
        CounterfactualOutcome {
            label: label_a.to_string(),
            trajectory: traj_a,
            prediction: pred_a,
        },
        CounterfactualOutcome {
            label: label_b.to_string(),
            trajectory: traj_b,
            prediction: pred_b,
        },
    ))
}

// ---- Bridge methods on the simulators ----

impl ClusterSimulator {
    /// Take a snapshot. Cheap — internally a `Clone`.
    pub fn snapshot(&self) -> ClusterSnapshot {
        ClusterSnapshot {
            inner: self.clone(),
        }
    }
}

impl GpuTrainingSimulator {
    /// Take a snapshot.
    pub fn snapshot(&self) -> GpuTrainingSnapshot {
        GpuTrainingSnapshot {
            inner: self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterTopology;
    use crate::cluster_simulator::ClusterConfig;
    use crate::seed::NssSeed;

    fn small_sim() -> ClusterSimulator {
        let cfg = ClusterConfig {
            cluster_arrival_rate: 8.0,
            ..ClusterConfig::default()
        };
        let top = ClusterTopology::complete(4, 8, 0.5).unwrap();
        ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap()
    }

    #[test]
    fn snapshot_is_independent_of_original() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let snap_tick = snap.tick();
        // Continue the original; the snapshot stays at its tick.
        let _ = sim.run(8).unwrap();
        assert_eq!(snap.tick(), snap_tick);
        assert_ne!(sim.tick(), snap.tick());
    }

    #[test]
    fn fork_with_no_extras_is_identical_to_original_continuation() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let mut clone = snap.fork(vec![]).unwrap();
        let original_continued = sim.run(8).unwrap();
        let cloned_continued = clone.run(8).unwrap();
        assert_eq!(
            original_continued.canonical_bytes(),
            cloned_continued.canonical_bytes()
        );
    }

    #[test]
    fn fork_with_intervention_diverges_from_baseline() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let mut baseline = snap.fork(vec![]).unwrap();
        let mut intervention = snap
            .fork(vec![Intervention::FailNode {
                tick: 10,
                node: NodeId(1),
            }])
            .unwrap();
        let t_base = baseline.run(16).unwrap();
        let t_int = intervention.run(16).unwrap();
        assert_ne!(t_base.canonical_bytes(), t_int.canonical_bytes());
    }

    #[test]
    fn fork_rejects_past_tick_intervention() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        // Snapshot tick is 8; intervention at tick 4 would never fire.
        let r = snap.fork(vec![Intervention::FailNode {
            tick: 4,
            node: NodeId(1),
        }]);
        assert!(matches!(r, Err(NssError::InvalidConfig { .. })));
    }

    #[test]
    fn fork_rejects_unknown_node() {
        let mut sim = small_sim();
        let _ = sim.run(4).unwrap();
        let snap = sim.snapshot();
        let r = snap.fork(vec![Intervention::FailNode {
            tick: 10,
            node: NodeId(99),
        }]);
        assert!(matches!(r, Err(NssError::InvalidConfig { .. })));
    }

    #[test]
    fn counterfactual_comparison_surfaces_label_flip() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss = ClusterNeuralSystemsSimulator::from_seed(
            crate::cluster_nss::ClusterNssConfig::default(),
            NssSeed(42),
        )
        .unwrap();
        let cmp = run_cluster_counterfactual(
            &snap,
            "baseline",
            vec![],
            "fail_node_1",
            vec![Intervention::FailNode {
                tick: 10,
                node: NodeId(1),
            }],
            16,
            &nss,
        )
        .unwrap();
        // The intervention should at minimum produce node-health
        // disagreements (node 1 healthy in baseline, failed in fork B).
        assert!(
            !cmp.node_health_disagreements.is_empty(),
            "intervention must produce at least one node-health disagreement"
        );
        assert_eq!(
            cmp.node_health_disagreements.get(&NodeId(1)),
            Some(&(NodeHealth::Healthy, NodeHealth::Failed))
        );
    }

    #[test]
    fn label_distribution_counts_match_trajectory_length() {
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let mut fork = snap.fork(vec![]).unwrap();
        let traj = fork.run(16).unwrap();
        let outcome = CounterfactualOutcome {
            label: "test".into(),
            trajectory: traj.clone(),
            prediction: ClusterNeuralSystemsSimulator::from_seed(
                crate::cluster_nss::ClusterNssConfig::default(),
                NssSeed(42),
            )
            .unwrap()
            .predict_next(traj.last_state().unwrap())
            .unwrap(),
        };
        let (n, d, c) = outcome.label_distribution();
        assert_eq!(n + d + c, traj.len() as u64);
    }

    #[test]
    fn intervention_is_harmful_thresholds_are_one_sided() {
        // Build a hand-crafted comparison and probe the threshold.
        let mut sim = small_sim();
        let _ = sim.run(8).unwrap();
        let snap = sim.snapshot();
        let nss = ClusterNeuralSystemsSimulator::from_seed(
            crate::cluster_nss::ClusterNssConfig::default(),
            NssSeed(42),
        )
        .unwrap();
        let cmp = run_cluster_counterfactual(
            &snap,
            "baseline",
            vec![],
            "fail_node_1",
            vec![Intervention::FailNode {
                tick: 10,
                node: NodeId(1),
            }],
            16,
            &nss,
        )
        .unwrap();
        // With a threshold of 1.0, no intervention can be "harmful"
        // (delta is in [-1, 1]). With threshold 0.0, harmful iff
        // delta is positive and any structural flag flipped.
        assert!(!cmp.intervention_is_harmful(1.0));
    }
}
