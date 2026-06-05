//! Phase 5b — pressure-legality verifier.
//!
//! Static analysis on intervention scripts. Answers the question:
//! "is this set of scheduler / autoscaling actions safe to apply?"
//! before any simulator (or compiler-side optimiser) acts on it.
//!
//! ## Violations the verifier catches
//!
//! 1. **Oscillation** — alternating FailNode/RecoverNode (or
//!    AddNode/RemoveNode) on the same node faster than the configured
//!    `min_action_cooldown`. This prevents the controller from
//!    thrashing a node.
//! 2. **Empty-cluster** — a script that would, at some tick, leave
//!    the cluster with fewer than `min_active_nodes` Healthy nodes.
//!    Critical for capacity planning.
//! 3. **Unknown-node** — script targets a node that isn't in the
//!    topology.
//! 4. **Invalid-intensity** — `ShedLoadOverride` with intensity outside
//!    `[0, 1]` or non-finite.
//! 5. **Insufficient-cooldown** — two actions on the same node within
//!    `min_action_cooldown` ticks of each other (a weaker version of
//!    Oscillation that catches "same kind twice in a row" too).
//! 6. **Aggressive-when-forbidden** — `FailNode` actions when
//!    `allow_aggressive_actions = false`. Useful as a global circuit
//!    breaker.
//!
//! ## Why static analysis instead of runtime guards?
//!
//! The Phase 4 controller already has runtime safety guards
//! (cooldowns, improvement floors, etc.). The legality verifier is
//! complementary: it catches structural bugs in **pre-built scripts**
//! before any execution happens. Two key use cases:
//!
//! - **Compiler-side decisions.** When the eventual CJC-Lang compiler
//!   fork uses NSS to drive optimisation, every optimisation pass
//!   produces a script of "compile-time interventions" (inline this,
//!   unroll that, vectorise this loop). The legality verifier catches
//!   structurally illegal scripts *before* the compiler commits to
//!   them — e.g., "you inlined function F at site A but also
//!   uninlined it at site B in the same pass" is an oscillation.
//! - **Audit / compliance.** A regulator can verify that a published
//!   intervention script meets safety policy *without* running the
//!   simulator. The verifier is pure, deterministic, and produces a
//!   structured report.
//!
//! ## Determinism
//!
//! Verification is pure: no RNG, no time, no IO. Same `(script,
//! topology, config)` always produces the same [`LegalityReport`].

use crate::cluster::{ClusterTopology, NodeId};
use crate::cluster_simulator::Intervention;
use std::collections::BTreeMap;

/// Knobs for the legality verifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegalityConfig {
    /// Minimum ticks that must elapse between consecutive actions on
    /// the same node. Two actions closer together than this trigger
    /// `InsufficientCooldown`. Default 4.
    pub min_action_cooldown: u64,
    /// Minimum number of `Healthy` nodes the cluster must have at
    /// every tick. A script that would push the active-node count
    /// below this triggers `EmptyCluster`. Default 1.
    pub min_active_nodes: u32,
    /// If `false`, `FailNode` actions trigger
    /// `AggressiveActionForbidden`. Default `true` (allow them).
    pub allow_aggressive_actions: bool,
    /// Maximum number of *adjacent-tick* FailNode↔RecoverNode (or
    /// AddNode↔RemoveNode) pairs on the same node before flagging
    /// Oscillation. `1` = catch any reversal within `min_action_cooldown`;
    /// `2` = require two reversals before flagging; etc. Default 1.
    pub max_reversals_per_node: u32,
    /// Initial number of `Healthy` nodes the cluster starts with
    /// (i.e., the count of slots that are NOT initially Absent).
    /// Used by the empty-cluster check. Set this from the topology +
    /// initial-state knowledge of the caller. Default `u32::MAX` (no
    /// initial-count assumption — only checks deltas).
    pub initial_active_nodes: u32,
}

impl Default for LegalityConfig {
    fn default() -> Self {
        Self {
            min_action_cooldown: 4,
            min_active_nodes: 1,
            allow_aggressive_actions: true,
            max_reversals_per_node: 1,
            initial_active_nodes: u32::MAX,
        }
    }
}

impl LegalityConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), crate::error::NssError> {
        if self.max_reversals_per_node == 0 {
            return Err(crate::error::NssError::InvalidConfig {
                detail: "max_reversals_per_node must be >= 1".into(),
            });
        }
        Ok(())
    }
}

/// One specific safety violation found in a script.
#[derive(Clone, Debug, PartialEq)]
pub enum LegalityViolation {
    /// Two or more reversals (Fail↔Recover or Add↔Remove) on the
    /// same node, all within the cooldown window.
    Oscillation {
        /// Which node.
        node: NodeId,
        /// Tick indices of the actions that constitute the
        /// oscillation, in script order.
        ticks: Vec<u64>,
    },
    /// At some tick, the cluster's projected active-node count drops
    /// below `min_active_nodes`.
    EmptyCluster {
        /// Tick at which the projected count became too low.
        tick: u64,
        /// Projected count of Healthy nodes at that tick.
        projected_active: u32,
    },
    /// An intervention targets a node not in the topology.
    UnknownNode {
        /// When the intervention fires.
        tick: u64,
        /// The unknown node id.
        node: NodeId,
    },
    /// A `ShedLoadOverride` carries a non-finite intensity or one
    /// outside `[0, 1]`.
    InvalidIntensity {
        /// When the action fires.
        tick: u64,
        /// Which node.
        node: NodeId,
        /// The offending intensity value.
        intensity: f64,
    },
    /// Two actions on the same node within the cooldown window
    /// (weaker than Oscillation — doesn't require reversal).
    InsufficientCooldown {
        /// Which node.
        node: NodeId,
        /// First action's tick.
        tick_a: u64,
        /// Second action's tick.
        tick_b: u64,
        /// Minimum cooldown required (from config).
        min_required: u64,
    },
    /// A `FailNode` was scheduled while
    /// `allow_aggressive_actions = false`.
    AggressiveActionForbidden {
        /// When the action fires.
        tick: u64,
        /// Which node.
        node: NodeId,
    },
}

impl LegalityViolation {
    /// Short categorical label (one of "oscillation", "empty_cluster",
    /// "unknown_node", "invalid_intensity", "insufficient_cooldown",
    /// "aggressive_forbidden"). Stable — safe to use as a key.
    pub fn kind_label(&self) -> &'static str {
        match self {
            LegalityViolation::Oscillation { .. } => "oscillation",
            LegalityViolation::EmptyCluster { .. } => "empty_cluster",
            LegalityViolation::UnknownNode { .. } => "unknown_node",
            LegalityViolation::InvalidIntensity { .. } => "invalid_intensity",
            LegalityViolation::InsufficientCooldown { .. } => "insufficient_cooldown",
            LegalityViolation::AggressiveActionForbidden { .. } => "aggressive_forbidden",
        }
    }

    /// The tick the violation pertains to (first relevant tick for
    /// multi-tick violations like Oscillation).
    pub fn tick(&self) -> u64 {
        match self {
            LegalityViolation::Oscillation { ticks, .. } => *ticks.first().unwrap_or(&0),
            LegalityViolation::EmptyCluster { tick, .. } => *tick,
            LegalityViolation::UnknownNode { tick, .. } => *tick,
            LegalityViolation::InvalidIntensity { tick, .. } => *tick,
            LegalityViolation::InsufficientCooldown { tick_a, .. } => *tick_a,
            LegalityViolation::AggressiveActionForbidden { tick, .. } => *tick,
        }
    }

    /// The node the violation targets (or `None` for none).
    pub fn node(&self) -> Option<NodeId> {
        match self {
            LegalityViolation::Oscillation { node, .. } => Some(*node),
            LegalityViolation::EmptyCluster { .. } => None,
            LegalityViolation::UnknownNode { node, .. } => Some(*node),
            LegalityViolation::InvalidIntensity { node, .. } => Some(*node),
            LegalityViolation::InsufficientCooldown { node, .. } => Some(*node),
            LegalityViolation::AggressiveActionForbidden { node, .. } => Some(*node),
        }
    }
}

/// Result of a verification pass.
#[derive(Clone, Debug, PartialEq)]
pub struct LegalityReport {
    /// Every violation found, in script order (sorted by `tick`,
    /// then by `kind_label`).
    pub violations: Vec<LegalityViolation>,
    /// Number of interventions scanned.
    pub script_size: usize,
    /// Set of node ids referenced by the script. Useful for
    /// downstream tools that want to audit "what did this script
    /// touch?"
    pub touched_nodes: Vec<NodeId>,
}

impl LegalityReport {
    /// `true` if no violations were found.
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    /// Count of violations by kind label.
    pub fn violations_by_kind(&self) -> BTreeMap<&'static str, u32> {
        let mut out: BTreeMap<&'static str, u32> = BTreeMap::new();
        for v in &self.violations {
            *out.entry(v.kind_label()).or_insert(0) += 1;
        }
        out
    }

    /// Pretty-print summary line — useful for CLI tools.
    pub fn summary(&self) -> String {
        if self.passed() {
            format!(
                "OK ({} interventions, {} nodes touched, 0 violations)",
                self.script_size,
                self.touched_nodes.len(),
            )
        } else {
            let by_kind = self.violations_by_kind();
            let parts: Vec<String> = by_kind.iter().map(|(k, n)| format!("{}={}", k, n)).collect();
            format!(
                "FAIL ({} interventions, {} violations: {})",
                self.script_size,
                self.violations.len(),
                parts.join(", ")
            )
        }
    }
}

/// The legality verifier — stateless, just holds the config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegalityVerifier {
    cfg: LegalityConfig,
}

impl LegalityVerifier {
    /// Build with the given config.
    pub fn new(cfg: LegalityConfig) -> Result<Self, crate::error::NssError> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Build with default config.
    pub fn default() -> Self {
        Self {
            cfg: LegalityConfig::default(),
        }
    }

    /// Borrow the config.
    pub fn config(&self) -> &LegalityConfig {
        &self.cfg
    }

    /// Verify a script against the topology + config.
    pub fn verify(
        &self,
        script: &[Intervention],
        topology: &ClusterTopology,
    ) -> LegalityReport {
        let mut violations: Vec<LegalityViolation> = Vec::new();
        let mut touched: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        // Pre-collect topology node IDs for quick lookup.
        let known_nodes: std::collections::BTreeSet<NodeId> = topology.nodes().collect();

        // Pass 1: per-intervention validity checks (unknown node,
        // invalid intensity, aggressive-forbidden).
        for iv in script {
            let node = iv.node();
            touched.insert(node);
            if !known_nodes.contains(&node) {
                violations.push(LegalityViolation::UnknownNode {
                    tick: iv.tick(),
                    node,
                });
            }
            if let Intervention::ShedLoadOverride { intensity, .. } = iv {
                if !intensity.is_finite() || !(0.0..=1.0).contains(intensity) {
                    violations.push(LegalityViolation::InvalidIntensity {
                        tick: iv.tick(),
                        node,
                        intensity: *intensity,
                    });
                }
            }
            if !self.cfg.allow_aggressive_actions {
                if matches!(iv, Intervention::FailNode { .. }) {
                    violations.push(LegalityViolation::AggressiveActionForbidden {
                        tick: iv.tick(),
                        node,
                    });
                }
            }
        }

        // Pass 2: per-node cooldown + oscillation analysis. Walk the
        // script in script-order (caller may not have pre-sorted it,
        // so we copy + sort by (tick, node, kind)).
        let mut sorted = script.to_vec();
        sorted.sort_by(|a, b| a.cmp(b));
        let mut per_node_history: BTreeMap<NodeId, Vec<(u64, ActionPolarity)>> = BTreeMap::new();
        for iv in &sorted {
            let polarity = ActionPolarity::of(iv);
            let history = per_node_history.entry(iv.node()).or_default();
            // Cooldown check vs the immediately preceding action on
            // this node.
            if let Some((last_tick, _last_pol)) = history.last().copied() {
                let gap = iv.tick().saturating_sub(last_tick);
                if gap < self.cfg.min_action_cooldown {
                    violations.push(LegalityViolation::InsufficientCooldown {
                        node: iv.node(),
                        tick_a: last_tick,
                        tick_b: iv.tick(),
                        min_required: self.cfg.min_action_cooldown,
                    });
                }
            }
            history.push((iv.tick(), polarity));
        }
        // Oscillation: scan each node's history for windows of
        // alternating polarities packed inside `min_action_cooldown`.
        for (node, hist) in &per_node_history {
            let mut reversals: Vec<u64> = Vec::new();
            for w in hist.windows(2) {
                let (t0, p0) = w[0];
                let (t1, p1) = w[1];
                if p0 != ActionPolarity::Other
                    && p1 != ActionPolarity::Other
                    && p0 != p1
                    && t1.saturating_sub(t0) < self.cfg.min_action_cooldown
                {
                    reversals.push(t0);
                    reversals.push(t1);
                }
            }
            // Dedup + sort the reversal ticks.
            reversals.sort_unstable();
            reversals.dedup();
            // Flag if reversal count is at least the configured threshold.
            // We compute `reversals.len() / 2` pairs as the reversal count.
            let pair_count = reversals.len() as u32 / 2;
            if pair_count >= self.cfg.max_reversals_per_node {
                violations.push(LegalityViolation::Oscillation {
                    node: *node,
                    ticks: reversals,
                });
            }
        }

        // Pass 3: empty-cluster check. Simulate the active-node count
        // forward in script order, treating Healthy→Failed/Absent as
        // -1 and Failed/Absent→Healthy as +1. The initial count is
        // `cfg.initial_active_nodes` (or the topology size if that
        // sentinel is u32::MAX).
        let initial = if self.cfg.initial_active_nodes == u32::MAX {
            known_nodes.len() as u32
        } else {
            self.cfg.initial_active_nodes
        };
        // Track an *abstract* per-node health (Active/Inactive). We
        // start with all nodes Active up to `initial`; if
        // `initial < known_nodes.len()` the remainder are Inactive.
        // The verifier doesn't know *which* nodes are initially Absent,
        // so it makes a worst-case assumption: count Healthy nodes
        // from the start of `known_nodes` order.
        let mut abstract_health: BTreeMap<NodeId, bool> = BTreeMap::new();
        for (i, n) in known_nodes.iter().enumerate() {
            abstract_health.insert(*n, (i as u32) < initial);
        }
        // Walk sorted script and track active count.
        for iv in &sorted {
            let was_active = abstract_health.get(&iv.node()).copied().unwrap_or(false);
            let now_active = match iv {
                Intervention::FailNode { .. } => false,
                Intervention::RecoverNode { .. } => true,
                Intervention::AddNode { .. } => true,
                Intervention::RemoveNode { .. } => false,
                Intervention::ShedLoadOverride { .. } => was_active, // no health change
            };
            abstract_health.insert(iv.node(), now_active);
            // Recount.
            let count = abstract_health.values().filter(|v| **v).count() as u32;
            if count < self.cfg.min_active_nodes {
                violations.push(LegalityViolation::EmptyCluster {
                    tick: iv.tick(),
                    projected_active: count,
                });
                // Don't add multiple empty-cluster violations for
                // back-to-back undershoots; only flag the transition.
                // We do this with a simple guard — if the previous
                // violation was also EmptyCluster, skip.
                let mut dedup = false;
                if violations.len() >= 2 {
                    let prev = &violations[violations.len() - 2];
                    if let LegalityViolation::EmptyCluster {
                        projected_active: prev_n,
                        ..
                    } = prev
                    {
                        if *prev_n == count {
                            dedup = true;
                        }
                    }
                }
                if dedup {
                    violations.pop();
                }
            }
        }

        // Sort violations by (tick, kind_label) for stable ordering.
        violations.sort_by(|a, b| {
            a.tick()
                .cmp(&b.tick())
                .then_with(|| a.kind_label().cmp(b.kind_label()))
        });

        LegalityReport {
            violations,
            script_size: script.len(),
            touched_nodes: touched.into_iter().collect(),
        }
    }
}

/// Tracks whether an action is "positive" (brings node into the
/// active set), "negative" (takes it out), or "neutral" (no effect
/// on active set). Oscillation looks for adjacent +/- pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ActionPolarity {
    /// Brings the node Up (RecoverNode, AddNode).
    Positive,
    /// Brings the node Down (FailNode, RemoveNode).
    Negative,
    /// No effect on active-set membership (ShedLoadOverride).
    Other,
}

impl ActionPolarity {
    fn of(iv: &Intervention) -> Self {
        match iv {
            Intervention::FailNode { .. } | Intervention::RemoveNode { .. } => {
                ActionPolarity::Negative
            }
            Intervention::RecoverNode { .. } | Intervention::AddNode { .. } => {
                ActionPolarity::Positive
            }
            Intervention::ShedLoadOverride { .. } => ActionPolarity::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_topology() -> ClusterTopology {
        ClusterTopology::complete(4, 8, 0.5).unwrap()
    }

    #[test]
    fn empty_script_passes() {
        let v = LegalityVerifier::default();
        let report = v.verify(&[], &small_topology());
        assert!(report.passed());
        assert_eq!(report.violations.len(), 0);
        assert_eq!(report.script_size, 0);
        assert_eq!(report.touched_nodes.len(), 0);
    }

    #[test]
    fn unknown_node_flagged() {
        let v = LegalityVerifier::default();
        let script = vec![Intervention::FailNode {
            tick: 5,
            node: NodeId(99),
        }];
        let report = v.verify(&script, &small_topology());
        assert!(!report.passed());
        assert!(matches!(
            report.violations[0],
            LegalityViolation::UnknownNode { tick: 5, node: NodeId(99) }
        ));
    }

    #[test]
    fn invalid_intensity_flagged() {
        let v = LegalityVerifier::default();
        let script = vec![Intervention::ShedLoadOverride {
            tick: 3,
            node: NodeId(1),
            intensity: 1.5,
        }];
        let report = v.verify(&script, &small_topology());
        assert!(report.violations.iter().any(|x| matches!(
            x,
            LegalityViolation::InvalidIntensity { .. }
        )));
    }

    #[test]
    fn cooldown_violation_flagged() {
        let v = LegalityVerifier::new(LegalityConfig {
            min_action_cooldown: 10,
            ..LegalityConfig::default()
        })
        .unwrap();
        let script = vec![
            Intervention::FailNode {
                tick: 2,
                node: NodeId(1),
            },
            Intervention::RecoverNode {
                tick: 5,
                node: NodeId(1),
            },
        ];
        let report = v.verify(&script, &small_topology());
        // Both an Oscillation and an InsufficientCooldown should fire.
        let has_cooldown = report
            .violations
            .iter()
            .any(|x| matches!(x, LegalityViolation::InsufficientCooldown { .. }));
        assert!(has_cooldown);
    }

    #[test]
    fn oscillation_flagged() {
        let v = LegalityVerifier::new(LegalityConfig {
            min_action_cooldown: 10,
            max_reversals_per_node: 1,
            ..LegalityConfig::default()
        })
        .unwrap();
        let script = vec![
            Intervention::FailNode {
                tick: 2,
                node: NodeId(1),
            },
            Intervention::RecoverNode {
                tick: 5,
                node: NodeId(1),
            },
        ];
        let report = v.verify(&script, &small_topology());
        let has_osc = report
            .violations
            .iter()
            .any(|x| matches!(x, LegalityViolation::Oscillation { .. }));
        assert!(has_osc);
    }

    #[test]
    fn empty_cluster_flagged() {
        let v = LegalityVerifier::new(LegalityConfig {
            min_active_nodes: 2,
            min_action_cooldown: 0, // disable cooldown to focus on this check
            ..LegalityConfig::default()
        })
        .unwrap();
        // Remove 3 of 4 nodes — cluster has 1 left, below min=2.
        let script = vec![
            Intervention::RemoveNode {
                tick: 5,
                node: NodeId(0),
            },
            Intervention::RemoveNode {
                tick: 10,
                node: NodeId(1),
            },
            Intervention::RemoveNode {
                tick: 15,
                node: NodeId(2),
            },
        ];
        let report = v.verify(&script, &small_topology());
        let has_empty = report
            .violations
            .iter()
            .any(|x| matches!(x, LegalityViolation::EmptyCluster { .. }));
        assert!(
            has_empty,
            "expected empty-cluster violation, got: {:?}",
            report.violations
        );
    }

    #[test]
    fn aggressive_action_forbidden_when_disallowed() {
        let v = LegalityVerifier::new(LegalityConfig {
            allow_aggressive_actions: false,
            ..LegalityConfig::default()
        })
        .unwrap();
        let script = vec![Intervention::FailNode {
            tick: 5,
            node: NodeId(1),
        }];
        let report = v.verify(&script, &small_topology());
        assert!(report.violations.iter().any(|x| matches!(
            x,
            LegalityViolation::AggressiveActionForbidden { .. }
        )));
    }

    #[test]
    fn clean_script_passes() {
        let v = LegalityVerifier::default();
        let script = vec![
            Intervention::FailNode {
                tick: 5,
                node: NodeId(1),
            },
            Intervention::RecoverNode {
                tick: 20, // well after cooldown
                node: NodeId(1),
            },
            Intervention::ShedLoadOverride {
                tick: 30,
                node: NodeId(2),
                intensity: 0.5,
            },
        ];
        let report = v.verify(&script, &small_topology());
        assert!(
            report.passed(),
            "expected clean script to pass, got: {:?}",
            report.violations
        );
    }

    #[test]
    fn verifier_is_deterministic() {
        let v = LegalityVerifier::default();
        let script = vec![
            Intervention::FailNode {
                tick: 5,
                node: NodeId(99),
            },
            Intervention::ShedLoadOverride {
                tick: 3,
                node: NodeId(1),
                intensity: 2.0,
            },
        ];
        let r1 = v.verify(&script, &small_topology());
        let r2 = v.verify(&script, &small_topology());
        assert_eq!(r1, r2);
    }

    #[test]
    fn report_summary_distinguishes_pass_and_fail() {
        let v = LegalityVerifier::default();
        let ok_report = v.verify(&[], &small_topology());
        let fail_report = v.verify(
            &[Intervention::FailNode {
                tick: 0,
                node: NodeId(99),
            }],
            &small_topology(),
        );
        assert!(ok_report.summary().starts_with("OK"));
        assert!(fail_report.summary().starts_with("FAIL"));
    }

    #[test]
    fn report_violations_by_kind_count_correctly() {
        let v = LegalityVerifier::default();
        let script = vec![
            Intervention::FailNode {
                tick: 0,
                node: NodeId(99),
            },
            Intervention::FailNode {
                tick: 5,
                node: NodeId(88),
            },
            Intervention::ShedLoadOverride {
                tick: 10,
                node: NodeId(1),
                intensity: -1.0,
            },
        ];
        let report = v.verify(&script, &small_topology());
        let counts = report.violations_by_kind();
        assert_eq!(counts.get("unknown_node").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("invalid_intensity").copied().unwrap_or(0), 1);
    }
}
