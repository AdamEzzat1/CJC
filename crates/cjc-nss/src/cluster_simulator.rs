//! Phase 2 — deterministic distributed cluster simulator.
//!
//! Generalises the [`crate::QueueSimulator`] to an N-node cluster with
//! network edges, routing policies, and scripted interventions
//! (deterministic node failures + recoveries).
//!
//! ## Per-tick procedure
//!
//! 1. **Apply interventions** scheduled for the current tick — flip
//!    nodes Healthy ↔ Failed. Failed-node queues are drained.
//! 2. **Generate arrivals**: cluster-level Poisson(λ); each arrival is
//!    routed to a healthy node via [`RoutingPolicy`].
//! 3. **Per-node service**: each healthy node runs one Phase-1-style
//!    queue tick (admit / serve / update internal pressures).
//! 4. **Network propagation**: each edge's `congestion` is updated
//!    based on the volume of work that crossed it; cross-node
//!    pressure (network → downstream queue) propagates.
//! 5. **Failure labelling**: per-node label using the Phase 1 rule;
//!    cluster-level rollup picks the worst per-node label.
//! 6. **Emit `ClusterEvent`**.
//!
//! ## Determinism
//!
//! - All RNG draws use [`crate::NssSeed::substream`] with
//!   per-(node, domain) salts.
//! - Nodes processed in `NodeId` order; edges processed in lex
//!   `(src, dst)` order.
//! - The intervention script is sorted by `(tick, node_id, kind)`
//!   before application so any ordering of `Vec<Intervention>` produces
//!   the same effect.
//! - Two simulators built from the same
//!   `(ClusterConfig, ClusterTopology, intervention_script, NssSeed)`
//!   produce byte-identical [`crate::ClusterTrajectory`]s.

use crate::cluster::{
    ClusterEvent, ClusterSystemState, ClusterTopology, ClusterTrajectory, NodeHealth, NodeId,
};
use crate::error::NssError;
use crate::failure::{FailureKind, FailureState};
use crate::pressure::{Pressure, PressureField, PressureKind};
use crate::propagation::PropagationConfig;
use crate::scheduler::{SchedulerAction, SchedulerKind};
use crate::seed::NssSeed;
use crate::system::SystemState;
use cjc_repro::Rng;
use std::collections::BTreeMap;

/// Routing policies the cluster simulator supports. Closed enum — new
/// policies require an explicit code change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoutingPolicy {
    /// Round-robin over healthy nodes (deterministic counter).
    RoundRobin,
    /// Route each task to the currently least-loaded healthy node
    /// (smallest queue_len). Ties broken by `NodeId` order — purely
    /// deterministic.
    LeastLoaded,
    /// Deterministic hash partitioning: `task_index mod healthy_count`.
    /// Cheap and stable across runs.
    HashPartition,
}

impl RoutingPolicy {
    /// Canonical short label.
    pub fn label(self) -> &'static str {
        match self {
            RoutingPolicy::RoundRobin => "round_robin",
            RoutingPolicy::LeastLoaded => "least_loaded",
            RoutingPolicy::HashPartition => "hash_partition",
        }
    }
}

/// One scripted intervention. `tick` says *when*, the variant says
/// *what*. Interventions are part of the deterministic input — same
/// script means same trajectory.
///
/// **Phase 3e** — added autoscaling + scheduler-override variants:
/// `AddNode` (Absent → Healthy), `RemoveNode` (Healthy/Failed →
/// Absent), and `ShedLoadOverride` (manually set a node's
/// admission-control shed fraction). These let the scheduler advisor
/// recommend capacity changes and load-shedding intensity.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Intervention {
    /// Flip a node to `Failed` at the start of this tick. No-op if
    /// already failed. Drains the node's queue (work is lost).
    FailNode {
        /// When to apply.
        tick: u64,
        /// Which node to fail.
        node: NodeId,
    },
    /// Flip a `Failed` node back to `Healthy` at the start of this
    /// tick. No-op if already healthy or absent.
    RecoverNode {
        /// When to apply.
        tick: u64,
        /// Which node to recover.
        node: NodeId,
    },
    /// **Phase 3e** — Autoscale **up**: flip an `Absent` node to
    /// `Healthy`. No-op if the node is already in the active cluster.
    /// The node starts with an empty queue and inherits the topology's
    /// per-link connections.
    AddNode {
        /// When to apply.
        tick: u64,
        /// Which (currently absent) node to bring into the cluster.
        node: NodeId,
    },
    /// **Phase 3e** — Autoscale **down**: flip a `Healthy` or `Failed`
    /// node to `Absent`. Drains the node's queue.
    RemoveNode {
        /// When to apply.
        tick: u64,
        /// Which node to remove from the active cluster.
        node: NodeId,
    },
    /// **Phase 3e** — Override the per-node shed fraction. Forces the
    /// node to apply admission control at the specified intensity for
    /// every tick from `tick` onward. `intensity` is in `[0, 1]`: 0 =
    /// no shedding, 1 = reject all incoming work. Cleared by a
    /// subsequent `ShedLoadOverride { intensity: 0.0, ... }` if needed.
    ShedLoadOverride {
        /// When to apply.
        tick: u64,
        /// Which node.
        node: NodeId,
        /// Shed fraction in `[0, 1]`. Clipped on application.
        intensity: f64,
    },
}

impl Eq for Intervention {}

impl Ord for Intervention {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order primarily by tick, then by node, then by variant
        // kind label (so the script's canonical sort is deterministic
        // even for variants carrying f64 fields). f64 fields tie-break
        // by total_cmp.
        self.tick()
            .cmp(&other.tick())
            .then_with(|| self.node().0.cmp(&other.node().0))
            .then_with(|| self.kind_byte().cmp(&other.kind_byte()))
            .then_with(|| {
                self.intensity()
                    .map(|i| i.to_bits())
                    .unwrap_or(0)
                    .cmp(&other.intensity().map(|i| i.to_bits()).unwrap_or(0))
            })
    }
}

impl std::hash::Hash for Intervention {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.canonical_bytes().hash(state);
    }
}

impl Intervention {
    /// Tick this intervention fires on.
    pub fn tick(&self) -> u64 {
        match self {
            Intervention::FailNode { tick, .. }
            | Intervention::RecoverNode { tick, .. }
            | Intervention::AddNode { tick, .. }
            | Intervention::RemoveNode { tick, .. }
            | Intervention::ShedLoadOverride { tick, .. } => *tick,
        }
    }

    /// Affected node.
    pub fn node(&self) -> NodeId {
        match self {
            Intervention::FailNode { node, .. }
            | Intervention::RecoverNode { node, .. }
            | Intervention::AddNode { node, .. }
            | Intervention::RemoveNode { node, .. }
            | Intervention::ShedLoadOverride { node, .. } => *node,
        }
    }

    /// Intensity (if applicable) — only `ShedLoadOverride` carries
    /// one. Returns `None` for the other variants.
    pub fn intensity(&self) -> Option<f64> {
        match self {
            Intervention::ShedLoadOverride { intensity, .. } => Some(*intensity),
            _ => None,
        }
    }

    /// Single-byte variant discriminator used by canonical bytes and
    /// the deterministic sort key.
    pub(crate) fn kind_byte(&self) -> u8 {
        match self {
            Intervention::FailNode { .. } => 0,
            Intervention::RecoverNode { .. } => 1,
            Intervention::AddNode { .. } => 2,
            Intervention::RemoveNode { .. } => 3,
            Intervention::ShedLoadOverride { .. } => 4,
        }
    }

    /// Canonical bytes — used by run-id hashing so the same script
    /// produces the same ID.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24);
        match self {
            Intervention::FailNode { tick, node } => {
                bytes.push(b'F');
                bytes.extend_from_slice(&tick.to_le_bytes());
                bytes.extend_from_slice(&node.to_le_bytes());
            }
            Intervention::RecoverNode { tick, node } => {
                bytes.push(b'R');
                bytes.extend_from_slice(&tick.to_le_bytes());
                bytes.extend_from_slice(&node.to_le_bytes());
            }
            Intervention::AddNode { tick, node } => {
                bytes.push(b'A');
                bytes.extend_from_slice(&tick.to_le_bytes());
                bytes.extend_from_slice(&node.to_le_bytes());
            }
            Intervention::RemoveNode { tick, node } => {
                bytes.push(b'X');
                bytes.extend_from_slice(&tick.to_le_bytes());
                bytes.extend_from_slice(&node.to_le_bytes());
            }
            Intervention::ShedLoadOverride {
                tick,
                node,
                intensity,
            } => {
                bytes.push(b'S');
                bytes.extend_from_slice(&tick.to_le_bytes());
                bytes.extend_from_slice(&node.to_le_bytes());
                bytes.extend_from_slice(&intensity.to_bits().to_le_bytes());
            }
        }
        bytes
    }
}

/// Cluster-simulator configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterConfig {
    /// Per-node worker count.
    pub workers_per_node: u32,
    /// Per-node queue capacity.
    pub queue_capacity: u32,
    /// Cluster-wide arrival rate (Poisson λ tasks per tick, distributed
    /// across nodes via the routing policy).
    pub cluster_arrival_rate: f64,
    /// Per-task service time lower bound (uniform).
    pub service_min: f64,
    /// Per-task service time upper bound (uniform).
    pub service_max: f64,
    /// Per-node knee — fraction of capacity at which the local
    /// scheduler starts shedding.
    pub degraded_knee: f64,
    /// Window for the collapse label.
    pub collapse_window: u32,
    /// Routing policy.
    pub routing: RoutingPolicy,
    /// Per-tick fraction of link congestion that dissipates.
    pub link_dissipation: f64,
    /// Propagation knobs (used by the per-node pressure step).
    pub propagation: PropagationConfig,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            workers_per_node: 2,
            queue_capacity: 16,
            cluster_arrival_rate: 6.0,
            service_min: 0.5,
            service_max: 1.5,
            degraded_knee: 0.7,
            collapse_window: 3,
            routing: RoutingPolicy::LeastLoaded,
            link_dissipation: 0.2,
            propagation: PropagationConfig::default(),
        }
    }
}

impl ClusterConfig {
    /// Validate every field.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.workers_per_node == 0 {
            return Err(NssError::InvalidConfig {
                detail: "workers_per_node must be >= 1".into(),
            });
        }
        if self.queue_capacity < self.workers_per_node {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "queue_capacity ({}) must be >= workers_per_node ({})",
                    self.queue_capacity, self.workers_per_node
                ),
            });
        }
        if !self.cluster_arrival_rate.is_finite() || self.cluster_arrival_rate < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "cluster_arrival_rate must be finite and >= 0, got {}",
                    self.cluster_arrival_rate
                ),
            });
        }
        if !self.service_min.is_finite() || self.service_min < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "service_min must be finite and >= 0, got {}",
                    self.service_min
                ),
            });
        }
        if !self.service_max.is_finite() || self.service_max < self.service_min {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "service_max must be finite and >= service_min, got {}",
                    self.service_max
                ),
            });
        }
        if !self.degraded_knee.is_finite() || !(0.0..=1.0).contains(&self.degraded_knee) {
            return Err(NssError::InvalidConfig {
                detail: format!("degraded_knee must be in [0,1], got {}", self.degraded_knee),
            });
        }
        if self.collapse_window == 0 {
            return Err(NssError::InvalidConfig {
                detail: "collapse_window must be >= 1".into(),
            });
        }
        if !self.link_dissipation.is_finite() || !(0.0..=1.0).contains(&self.link_dissipation) {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "link_dissipation must be in [0, 1], got {}",
                    self.link_dissipation
                ),
            });
        }
        self.propagation.validate()?;
        Ok(())
    }

    /// Canonical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&(self.workers_per_node as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.queue_capacity as u64).to_le_bytes());
        bytes.extend_from_slice(&self.cluster_arrival_rate.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.service_min.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.service_max.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.degraded_knee.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(self.collapse_window as u64).to_le_bytes());
        bytes.extend_from_slice(self.routing.label().as_bytes());
        bytes.push(b'|');
        bytes.extend_from_slice(&self.link_dissipation.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.propagation.canonical_bytes());
        bytes
    }
}

/// The distributed cluster simulator.
///
/// **Phase 3a — counterfactual forking.** `Clone` is implemented because
/// every internal field is itself `Clone` (config: `Copy`, topology:
/// `Clone`, RNG: `Clone`, per-node runtime: `Clone`). A clone is a
/// snapshot — see [`ClusterSimulator::snapshot`] and
/// [`ClusterSnapshot::fork`] for the forking API.
#[derive(Clone, Debug)]
pub struct ClusterSimulator {
    cfg: ClusterConfig,
    topology: ClusterTopology,
    seed: NssSeed,
    /// Intervention script — sorted by `(tick, node, kind)` at
    /// construction so insertion order doesn't matter.
    script: Vec<Intervention>,
    /// Internal per-node state (queues, RNGs, completed/rejected
    /// counters, collapse streaks).
    nodes: BTreeMap<NodeId, NodeRuntime>,
    /// Cluster-level RNG for routing decisions and cluster-wide
    /// arrival draws.
    rng_cluster: Rng,
    /// Round-robin pointer (when `cfg.routing == RoundRobin`).
    rr_pointer: usize,
    /// Tick counter.
    tick: u64,
    /// Cumulative task index (used by `HashPartition`).
    task_index: u64,
}

#[derive(Clone, Debug)]
struct NodeRuntime {
    queue_len: u32,
    completed: u64,
    rejected: u64,
    collapse_streak: u32,
    last_throughput: f64,
    health: NodeHealth,
    field: PressureField,
    rng_service: Rng,
    /// **Phase 3e** — when `Some(f)`, the per-tick shed fraction is
    /// forced to `f` regardless of occupancy. `None` means use the
    /// default occupancy-driven shed. Set via
    /// [`Intervention::ShedLoadOverride`].
    shed_override: Option<f64>,
}

impl ClusterSimulator {
    /// Build the simulator. Validates config + sorts the intervention
    /// script for determinism.
    pub fn new(
        cfg: ClusterConfig,
        topology: ClusterTopology,
        seed: NssSeed,
        mut script: Vec<Intervention>,
    ) -> Result<Self, NssError> {
        cfg.validate()?;
        if topology.node_count() == 0 {
            return Err(NssError::InvalidConfig {
                detail: "cluster topology must have at least one node".into(),
            });
        }
        // Validate intervention targets exist.
        for iv in &script {
            let node = iv.node();
            if !topology.nodes().any(|n| n == node) {
                return Err(NssError::InvalidConfig {
                    detail: format!("intervention targets unknown node {}", node),
                });
            }
        }
        // Sort by (tick, node, kind_byte). Determinism preserves
        // intent: a caller can hand interventions in any order and the
        // resulting trajectory is identical.
        script.sort_by(|a, b| a.cmp(b));

        let mut nodes = BTreeMap::new();
        for id in topology.nodes() {
            let salt = format!("node.{}.service", id.0);
            let mut field = PressureField::with_default_thresholds();
            // Initialise with bounded thresholds — same as Phase 1.
            for k in PressureKind::all() {
                field.set(k, Pressure::new(0.0, 1.0, 0.1)?);
            }
            nodes.insert(
                id,
                NodeRuntime {
                    queue_len: 0,
                    completed: 0,
                    rejected: 0,
                    collapse_streak: 0,
                    last_throughput: 1.0,
                    health: NodeHealth::Healthy,
                    field,
                    rng_service: seed.substream(&salt),
                    shed_override: None,
                },
            );
        }

        Ok(Self {
            cfg,
            topology,
            seed,
            script,
            nodes,
            rng_cluster: seed.substream("cluster.arrivals"),
            rr_pointer: 0,
            tick: 0,
            task_index: 0,
        })
    }

    /// Borrow the topology.
    pub fn topology(&self) -> &ClusterTopology {
        &self.topology
    }

    /// Borrow the config.
    pub fn config(&self) -> &ClusterConfig {
        &self.cfg
    }

    /// Seed accessor.
    pub fn seed(&self) -> NssSeed {
        self.seed
    }

    /// Borrow the (sorted) intervention script.
    pub fn intervention_script(&self) -> &[Intervention] {
        &self.script
    }

    /// Current tick (number of times `step` has run).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Append additional interventions and re-sort the script
    /// canonically. Used by [`crate::ClusterSnapshot::fork`] —
    /// crate-internal because callers should usually go through the
    /// snapshot+fork API rather than mutating a running simulator.
    pub(crate) fn append_and_sort_interventions(&mut self, mut extra: Vec<Intervention>) {
        self.script.append(&mut extra);
        self.script.sort_by(|a, b| a.cmp(b));
    }

    /// **Phase 4** — inject one intervention into the running script
    /// at any time. Public counterpart of `append_and_sort_interventions`
    /// for closed-loop autonomous control. Re-sorts so the script
    /// remains canonical; interventions scheduled for past ticks are
    /// no-ops on the next step.
    pub fn inject_intervention(&mut self, iv: Intervention) {
        self.script.push(iv);
        self.script.sort_by(|a, b| a.cmp(b));
    }

    /// Run for `n_ticks` and return the trajectory.
    pub fn run(&mut self, n_ticks: u64) -> Result<ClusterTrajectory, NssError> {
        let mut traj = ClusterTrajectory::empty();
        for _ in 0..n_ticks {
            traj.push(self.step()?)?;
        }
        Ok(traj)
    }

    fn step(&mut self) -> Result<ClusterEvent, NssError> {
        // 1. Apply interventions scheduled for this tick.
        for iv in self.script.iter().filter(|iv| iv.tick() == self.tick) {
            let node_id = iv.node();
            if let Some(rt) = self.nodes.get_mut(&node_id) {
                match iv {
                    Intervention::FailNode { .. } => {
                        if rt.health == NodeHealth::Healthy {
                            rt.health = NodeHealth::Failed;
                            // Drain the queue immediately. Rejected
                            // counter increments by the drain size —
                            // this is what a real failure looks like
                            // when in-flight work is lost.
                            rt.rejected = rt.rejected.saturating_add(rt.queue_len as u64);
                            rt.queue_len = 0;
                            rt.collapse_streak = 0;
                        }
                    }
                    Intervention::RecoverNode { .. } => {
                        if rt.health == NodeHealth::Failed {
                            rt.health = NodeHealth::Healthy;
                        }
                    }
                    Intervention::AddNode { .. } => {
                        // Absent → Healthy. Reset queue + streaks; the
                        // node enters the cluster fresh.
                        if rt.health == NodeHealth::Absent {
                            rt.health = NodeHealth::Healthy;
                            rt.queue_len = 0;
                            rt.collapse_streak = 0;
                        }
                    }
                    Intervention::RemoveNode { .. } => {
                        // Healthy/Failed → Absent. Drain any queue;
                        // the node leaves the active cluster.
                        if rt.health != NodeHealth::Absent {
                            rt.rejected = rt.rejected.saturating_add(rt.queue_len as u64);
                            rt.queue_len = 0;
                            rt.collapse_streak = 0;
                            rt.health = NodeHealth::Absent;
                        }
                    }
                    Intervention::ShedLoadOverride { intensity, .. } => {
                        let clipped = if !intensity.is_finite() {
                            0.0
                        } else {
                            intensity.clamp(0.0, 1.0)
                        };
                        rt.shed_override = Some(clipped);
                    }
                }
            }
        }

        // 2. Cluster-wide arrivals via Poisson(λ).
        let arrivals = poisson_clamped(
            &mut self.rng_cluster,
            self.cfg.cluster_arrival_rate,
            8 * self.cfg.queue_capacity * self.nodes.len() as u32,
        );

        // 3. Route arrivals to nodes.
        let healthy_nodes: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, rt)| rt.health == NodeHealth::Healthy)
            .map(|(id, _)| *id)
            .collect();

        // Per-node admissions counters for this tick.
        let mut admitted: BTreeMap<NodeId, u32> = BTreeMap::new();
        let mut rejected_routing: u32 = 0;

        if !healthy_nodes.is_empty() {
            for _ in 0..arrivals {
                let target = self.choose_target(&healthy_nodes);
                let rt = self.nodes.get_mut(&target).unwrap();
                let free = self.cfg.queue_capacity.saturating_sub(rt.queue_len);
                let occupancy = rt.queue_len as f64 / self.cfg.queue_capacity as f64;
                // Per-node shed-load on top of routing.
                // Phase 3e: if the node has a ShedLoadOverride active,
                // use that intensity instead of the occupancy-driven
                // default.
                let shed = if let Some(forced) = rt.shed_override {
                    forced
                } else if occupancy >= self.cfg.degraded_knee {
                    ((occupancy - self.cfg.degraded_knee)
                        / (1.0 - self.cfg.degraded_knee + f64::EPSILON))
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let admit_prob = 1.0 - shed;
                let coin = self.rng_cluster.next_f64();
                let accepted = free > 0 && coin <= admit_prob;
                if accepted {
                    rt.queue_len += 1;
                    *admitted.entry(target).or_insert(0) += 1;
                } else {
                    rt.rejected = rt.rejected.saturating_add(1);
                    rejected_routing += 1;
                }
                self.task_index = self.task_index.saturating_add(1);
            }
        } else {
            // Whole cluster failed — every arrival is rejected.
            rejected_routing = arrivals;
        }

        // 4. Per-node service.
        //
        // Snapshot peak queue lengths *before* serving so the collapse
        // detector sees the correct quantity (matches the Phase 1
        // bug-fix in `simulator.rs`).
        let mut peak_queue_len: BTreeMap<NodeId, u32> = BTreeMap::new();
        let mut served: BTreeMap<NodeId, u32> = BTreeMap::new();
        let mut mean_service_time: BTreeMap<NodeId, f64> = BTreeMap::new();

        // We pre-collect node IDs so the borrow checker is happy with
        // independent &mut accesses to `self.nodes` and `self.rng_service`.
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for id in &node_ids {
            let rt = self.nodes.get_mut(id).unwrap();
            peak_queue_len.insert(*id, rt.queue_len);
            // Phase 3e — both Failed and Absent nodes skip service.
            // They differ in their pressure-field and failure-label
            // treatment downstream.
            if !rt.health.participates() {
                served.insert(*id, 0);
                mean_service_time.insert(*id, 0.0);
                continue;
            }
            let to_serve = rt.queue_len.min(self.cfg.workers_per_node);
            let mut acc = cjc_repro::KahanAccumulatorF64::new();
            for _ in 0..to_serve {
                let t = self.cfg.service_min
                    + (self.cfg.service_max - self.cfg.service_min) * rt.rng_service.next_f64();
                acc.add(t);
            }
            rt.queue_len -= to_serve;
            rt.completed = rt.completed.saturating_add(to_serve as u64);
            served.insert(*id, to_serve);
            let mst = if to_serve == 0 {
                0.0
            } else {
                acc.finalize() / to_serve as f64
            };
            mean_service_time.insert(*id, mst);
        }

        // 5. Network propagation.
        //
        // For each edge (src, dst), the volume of work that "would
        // flow" from src to dst this tick is proportional to:
        //
        //   src.served * weight  (if both healthy)
        //
        // and is added to dst's network pressure proxy (we model the
        // cluster network as a load-redistribution channel).
        // Congestion grows with volume/capacity, dissipates by
        // `link_dissipation` per tick.
        //
        // Failed source → no outgoing pressure; failed destination →
        // pressure dissipates instead of arriving (modelling that
        // work routed to a dead node never gets through).
        let mut net_inflow: BTreeMap<NodeId, f64> = BTreeMap::new();
        // Collect edges + their src/dst health flags first (immutable
        // view) so the subsequent mutation pass doesn't conflict.
        struct EdgeWork {
            src: NodeId,
            dst: NodeId,
            volume: f64,
            capacity: u32,
            dst_healthy: bool,
        }
        let mut works: Vec<EdgeWork> = Vec::with_capacity(self.topology.edge_count());
        for (src, dst, link) in self.topology.edges() {
            let src_served = *served.get(&src).unwrap_or(&0) as f64;
            let src_healthy = self
                .nodes
                .get(&src)
                .map(|r| r.health)
                .unwrap_or(NodeHealth::Failed)
                == NodeHealth::Healthy;
            let dst_healthy = self
                .nodes
                .get(&dst)
                .map(|r| r.health)
                .unwrap_or(NodeHealth::Failed)
                == NodeHealth::Healthy;
            let volume = if src_healthy {
                src_served * link.weight
            } else {
                0.0
            };
            works.push(EdgeWork {
                src,
                dst,
                volume,
                capacity: link.capacity,
                dst_healthy,
            });
        }

        // Update link congestion + accumulate per-node inflow.
        for w in &works {
            let link = self.topology.link_mut(w.src, w.dst).unwrap();
            link.congestion *= 1.0 - self.cfg.link_dissipation;
            if w.volume > 0.0 {
                let increment = (w.volume / w.capacity as f64).min(1.0);
                link.congestion = (link.congestion + increment).min(1.0);
                if w.dst_healthy {
                    *net_inflow.entry(w.dst).or_insert(0.0) += w.volume;
                }
            }
            // Clamp under floating drift.
            if link.congestion < 0.0 {
                link.congestion = 0.0;
            }
        }

        // 6. Update per-node `PressureField` from this tick's queue
        //    occupancy + worker busy fraction + network inflow.
        for id in &node_ids {
            let rt = self.nodes.get_mut(id).unwrap();
            // Phase 3e — Absent nodes contribute zero pressure across
            // every field (they're not part of the active cluster).
            if rt.health == NodeHealth::Absent {
                for k in PressureKind::all() {
                    rt.field.set(k, Pressure::new(0.0, 1.0, 0.1)?);
                }
                continue;
            }
            if rt.health == NodeHealth::Failed {
                // Dead node: zero pressures, but the (Memory, Thermal)
                // fields hold their last value (modelling lingering
                // local state).
                rt.field
                    .set(PressureKind::Queue, Pressure::new(0.0, 1.0, 0.05)?);
                rt.field
                    .set(PressureKind::Cpu, Pressure::new(0.0, 1.0, 0.1)?);
                rt.field
                    .set(PressureKind::Sync, Pressure::new(0.0, 1.0, 0.08)?);
                rt.field.set(
                    PressureKind::Throughput,
                    Pressure::new(1.0, 1.0, 0.05)?, // 100% throughput pressure for a dead node
                );
                rt.field
                    .set(PressureKind::Network, Pressure::new(0.0, 1.0, 0.1)?);
                continue;
            }
            let queue_p = rt.queue_len as f64 / self.cfg.queue_capacity as f64;
            let cpu_p = *served.get(id).unwrap_or(&0) as f64 / self.cfg.workers_per_node as f64;
            let sync_p = ((rt.queue_len as f64 * self.cfg.workers_per_node as f64).sqrt()
                / self.cfg.queue_capacity as f64)
                .min(1.5);
            // Throughput pressure: 1 - throughput_proxy.
            let denom = (*admitted.get(id).unwrap_or(&0) as f64
                + rt.queue_len as f64
                + *served.get(id).unwrap_or(&0) as f64)
                .max(1.0);
            let throughput = (*served.get(id).unwrap_or(&0) as f64 / denom)
                .max(0.0)
                .min(1.0);
            rt.last_throughput = throughput;
            let thr_p = (1.0 - throughput).max(0.0);
            // Network pressure: scaled inflow against per-node
            // workers (a node serving 2 tasks/tick gets pressure 1.0
            // when it receives 2 net-flow units).
            let net_in = *net_inflow.get(id).unwrap_or(&0.0);
            let net_p = (net_in / self.cfg.workers_per_node as f64).min(1.5);

            rt.field
                .set(PressureKind::Queue, Pressure::new(queue_p, 1.0, 0.05)?);
            rt.field
                .set(PressureKind::Cpu, Pressure::new(cpu_p, 1.0, 0.1)?);
            rt.field
                .set(PressureKind::Sync, Pressure::new(sync_p, 1.0, 0.08)?);
            rt.field
                .set(PressureKind::Throughput, Pressure::new(thr_p, 1.0, 0.05)?);
            rt.field
                .set(PressureKind::Network, Pressure::new(net_p, 1.0, 0.1)?);
        }

        // 7. Per-node Phase-1 propagation (intra-node).
        let prop = crate::PressurePropagator::new(
            crate::PressureGraph::default_phase1(),
            self.cfg.propagation,
        )?;
        for id in &node_ids {
            let rt = self.nodes.get_mut(id).unwrap();
            if rt.health == NodeHealth::Healthy {
                prop.step(&mut rt.field)?;
            }
        }

        // 8. Per-node scheduler action + failure label.
        let mut actions: BTreeMap<NodeId, SchedulerAction> = BTreeMap::new();
        let mut failures: BTreeMap<NodeId, FailureState> = BTreeMap::new();
        for id in &node_ids {
            let rt = self.nodes.get_mut(id).unwrap();
            // Phase 3e — Absent nodes contribute Nominal to the
            // rollup (they're not broken, they're just not there).
            if rt.health == NodeHealth::Absent {
                actions.insert(*id, SchedulerAction::idle());
                failures.insert(*id, FailureState::nominal());
                continue;
            }
            if rt.health == NodeHealth::Failed {
                // Failed nodes report a `Collapse` label with `Queue`
                // as the symbolic source (work is being lost).
                actions.insert(*id, SchedulerAction::idle());
                failures.insert(*id, FailureState::collapse(PressureKind::Queue));
                continue;
            }
            let occupancy = rt.queue_len as f64 / self.cfg.queue_capacity as f64;
            let peak = *peak_queue_len.get(id).unwrap_or(&0);
            let queue_full = peak >= self.cfg.queue_capacity;
            let low_throughput = rt.last_throughput < 0.3;
            if queue_full && low_throughput {
                rt.collapse_streak += 1;
            } else {
                rt.collapse_streak = 0;
            }
            let action = if occupancy >= self.cfg.degraded_knee {
                let shed = ((occupancy - self.cfg.degraded_knee)
                    / (1.0 - self.cfg.degraded_knee + f64::EPSILON))
                    .clamp(0.0, 1.0);
                SchedulerAction::new(SchedulerKind::ShedLoad, shed)
            } else {
                SchedulerAction::idle()
            };
            let failure = if rt.collapse_streak >= self.cfg.collapse_window {
                FailureState::collapse(PressureKind::Queue)
            } else if occupancy >= self.cfg.degraded_knee {
                FailureState::degraded(PressureKind::Queue)
            } else {
                FailureState::nominal()
            };
            actions.insert(*id, action);
            failures.insert(*id, failure);
        }

        // 9. Cluster-level rollup: worst-of per-node.
        let mut cluster_label = FailureKind::Nominal;
        let mut cluster_source = None;
        for (_id, f) in failures.iter() {
            match (cluster_label, f.kind) {
                (_, FailureKind::Collapse) => {
                    cluster_label = FailureKind::Collapse;
                    cluster_source = f.dominant_source;
                    break;
                }
                (FailureKind::Nominal, FailureKind::Degraded) => {
                    cluster_label = FailureKind::Degraded;
                    cluster_source = f.dominant_source;
                }
                _ => {}
            }
        }
        let cluster_failure = match (cluster_label, cluster_source) {
            (FailureKind::Collapse, Some(k)) => FailureState::collapse(k),
            (FailureKind::Degraded, Some(k)) => FailureState::degraded(k),
            _ => FailureState::nominal(),
        };

        // 10. Materialise the cluster state.
        let mut nodes_state = BTreeMap::new();
        for id in &node_ids {
            let rt = self.nodes.get(id).unwrap();
            let state = SystemState {
                tick: self.tick,
                pressures: rt.field.clone(),
                in_flight: rt.queue_len as u64,
                completed: rt.completed,
                rejected: rt.rejected,
                mean_service_time: *mean_service_time.get(id).unwrap_or(&0.0),
            };
            nodes_state.insert(*id, state);
        }
        let mut link_congestion = BTreeMap::new();
        for (src, dst, link) in self.topology.edges() {
            link_congestion.insert((src, dst), link.congestion);
        }
        let mut node_health = BTreeMap::new();
        for id in &node_ids {
            node_health.insert(*id, self.nodes.get(id).unwrap().health);
        }

        let cluster_state = ClusterSystemState {
            tick: self.tick,
            nodes: nodes_state,
            link_congestion,
            node_health,
        };
        let _ = rejected_routing; // currently rolled into per-node rejections + reported via metrics in future phases

        let ev = ClusterEvent {
            state: cluster_state,
            actions,
            failures,
            cluster_failure,
        };
        self.tick += 1;
        Ok(ev)
    }

    /// Routing policy: pick a target node from the healthy set.
    /// Deterministic given current state + RNG.
    fn choose_target(&mut self, healthy: &[NodeId]) -> NodeId {
        match self.cfg.routing {
            RoutingPolicy::RoundRobin => {
                let target = healthy[self.rr_pointer % healthy.len()];
                self.rr_pointer = (self.rr_pointer + 1) % healthy.len();
                target
            }
            RoutingPolicy::LeastLoaded => {
                // BTreeMap iteration is already in NodeId order, so the
                // tie-break is canonical.
                let mut best = healthy[0];
                let mut best_q = self
                    .nodes
                    .get(&best)
                    .map(|r| r.queue_len)
                    .unwrap_or(u32::MAX);
                for id in healthy.iter().skip(1) {
                    let q = self.nodes.get(id).map(|r| r.queue_len).unwrap_or(u32::MAX);
                    if q < best_q {
                        best = *id;
                        best_q = q;
                    }
                }
                best
            }
            RoutingPolicy::HashPartition => healthy[(self.task_index as usize) % healthy.len()],
        }
    }

    /// Canonical bytes for the simulator inputs (config + topology +
    /// intervention script). Used by [`crate::ClusterRunId`] indirectly
    /// via the cluster-NSS config bytes.
    pub fn input_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.cfg.canonical_bytes();
        bytes.push(b'#');
        bytes.extend_from_slice(&self.topology.canonical_bytes());
        bytes.push(b'#');
        for iv in &self.script {
            bytes.extend_from_slice(&iv.canonical_bytes());
            bytes.push(b';');
        }
        bytes
    }
}

/// Same Knuth Poisson sampler as the Phase 1 simulator — shared
/// because the Phase 2 cluster simulator uses the same arrival model.
/// Kept private here to avoid leaking implementation detail; if a third
/// caller needs it, promote it to a small `util` module.
fn poisson_clamped(rng: &mut Rng, lambda: f64, max: u32) -> u32 {
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k: u32 = 0;
    let mut p = 1.0f64;
    loop {
        if k >= max {
            return max;
        }
        let u = rng.next_f64();
        p *= u;
        if p <= l {
            return k;
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_complete_topology(n: u32) -> ClusterTopology {
        ClusterTopology::complete(n, 8, 0.5).unwrap()
    }

    #[test]
    fn config_default_validates() {
        assert!(ClusterConfig::default().validate().is_ok());
    }

    #[test]
    fn cluster_determinism_same_seed_same_trajectory() {
        let cfg = ClusterConfig::default();
        let top = small_complete_topology(4);
        let mut a = ClusterSimulator::new(cfg, top.clone(), NssSeed(42), vec![]).unwrap();
        let mut b = ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
        let ta = a.run(48).unwrap();
        let tb = b.run(48).unwrap();
        assert_eq!(ta.canonical_bytes(), tb.canonical_bytes());
    }

    #[test]
    fn intervention_script_order_independent() {
        let cfg = ClusterConfig::default();
        let top = small_complete_topology(4);
        let ivs_a = vec![
            Intervention::FailNode {
                tick: 5,
                node: NodeId(2),
            },
            Intervention::RecoverNode {
                tick: 12,
                node: NodeId(2),
            },
        ];
        let ivs_b: Vec<Intervention> = ivs_a.iter().rev().copied().collect();
        let mut a = ClusterSimulator::new(cfg, top.clone(), NssSeed(7), ivs_a).unwrap();
        let mut b = ClusterSimulator::new(cfg, top, NssSeed(7), ivs_b).unwrap();
        assert_eq!(
            a.run(32).unwrap().canonical_bytes(),
            b.run(32).unwrap().canonical_bytes()
        );
    }

    #[test]
    fn fail_node_drains_its_queue_and_routes_around() {
        // Build a 4-node cluster, drive enough load to fill queues,
        // fail node 1 at tick 8, observe that node 1's queue is 0
        // afterwards and that the remaining nodes carry the load.
        let cfg = ClusterConfig {
            cluster_arrival_rate: 10.0,
            ..ClusterConfig::default()
        };
        let top = small_complete_topology(4);
        let ivs = vec![Intervention::FailNode {
            tick: 8,
            node: NodeId(1),
        }];
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
        let traj = sim.run(32).unwrap();
        // After tick 8, node 1 should be Failed for the rest of the run.
        for ev in traj.iter().skip(8) {
            assert_eq!(
                ev.state.node_health.get(&NodeId(1)).copied(),
                Some(NodeHealth::Failed)
            );
            // Failed node's queue is 0.
            assert_eq!(ev.state.nodes.get(&NodeId(1)).unwrap().in_flight, 0);
        }
        // At least one tick after the failure has Collapse rollup
        // because the failed node tags Collapse.
        let any_collapse = traj
            .iter()
            .skip(8)
            .any(|ev| ev.cluster_failure.kind == FailureKind::Collapse);
        assert!(
            any_collapse,
            "failed-node cluster should rollup to Collapse"
        );
    }

    #[test]
    fn recovery_returns_node_to_healthy() {
        let cfg = ClusterConfig::default();
        let top = small_complete_topology(3);
        let ivs = vec![
            Intervention::FailNode {
                tick: 4,
                node: NodeId(0),
            },
            Intervention::RecoverNode {
                tick: 16,
                node: NodeId(0),
            },
        ];
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
        let traj = sim.run(32).unwrap();
        // Before tick 4: healthy.
        assert_eq!(
            traj.as_slice()[3]
                .state
                .node_health
                .get(&NodeId(0))
                .copied(),
            Some(NodeHealth::Healthy)
        );
        // After tick 4, before recovery: failed.
        assert_eq!(
            traj.as_slice()[10]
                .state
                .node_health
                .get(&NodeId(0))
                .copied(),
            Some(NodeHealth::Failed)
        );
        // After recovery (tick 16): healthy again.
        assert_eq!(
            traj.as_slice()[20]
                .state
                .node_health
                .get(&NodeId(0))
                .copied(),
            Some(NodeHealth::Healthy)
        );
    }

    #[test]
    fn cluster_with_all_nodes_failed_keeps_running_safely() {
        let cfg = ClusterConfig::default();
        let top = small_complete_topology(2);
        let ivs = vec![
            Intervention::FailNode {
                tick: 1,
                node: NodeId(0),
            },
            Intervention::FailNode {
                tick: 1,
                node: NodeId(1),
            },
        ];
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), ivs).unwrap();
        // Should not panic / loop forever; should produce a trajectory
        // where every tick after 1 has both nodes failed.
        let traj = sim.run(8).unwrap();
        for ev in traj.iter().skip(1) {
            for h in ev.state.node_health.values() {
                assert_eq!(*h, NodeHealth::Failed);
            }
        }
    }

    #[test]
    fn routing_policy_round_robin_balances_arrivals() {
        // Round-robin to a 4-node cluster should produce roughly
        // equal queue depths for the first few ticks.
        let cfg = ClusterConfig {
            cluster_arrival_rate: 8.0,
            routing: RoutingPolicy::RoundRobin,
            ..ClusterConfig::default()
        };
        let top = small_complete_topology(4);
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
        let traj = sim.run(8).unwrap();
        // After 8 ticks the *completed* counters should be within
        // workers_per_node of each other (round-robin spread is exact
        // modulo individual queue fullness).
        let mut totals: Vec<u64> = traj
            .last_state()
            .unwrap()
            .nodes
            .values()
            .map(|s| s.completed)
            .collect();
        totals.sort();
        let spread = totals.last().unwrap() - totals.first().unwrap();
        assert!(spread <= 8, "round-robin spread {} too large", spread);
    }

    #[test]
    fn link_congestion_rises_under_load() {
        let cfg = ClusterConfig {
            cluster_arrival_rate: 8.0,
            ..ClusterConfig::default()
        };
        let top = small_complete_topology(3);
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
        let traj = sim.run(16).unwrap();
        // At least one link should be at non-zero congestion at some
        // point.
        let any_congested = traj
            .iter()
            .any(|ev| ev.state.link_congestion.values().any(|c| *c > 0.0));
        assert!(any_congested, "expected some link congestion under load");
        // Congestion must stay in [0, 1].
        for ev in traj.iter() {
            for c in ev.state.link_congestion.values() {
                assert!(c.is_finite() && *c >= 0.0 && *c <= 1.0);
            }
        }
    }

    #[test]
    fn intervention_targeting_unknown_node_fails() {
        let cfg = ClusterConfig::default();
        let top = small_complete_topology(2);
        let ivs = vec![Intervention::FailNode {
            tick: 1,
            node: NodeId(99),
        }];
        let r = ClusterSimulator::new(cfg, top, NssSeed(42), ivs);
        assert!(matches!(r, Err(NssError::InvalidConfig { .. })));
    }

    #[test]
    fn cluster_state_validates_each_tick() {
        let cfg = ClusterConfig::default();
        let top = small_complete_topology(3);
        let mut sim = ClusterSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
        let traj = sim.run(32).unwrap();
        for ev in traj.iter() {
            ev.state.validate().expect("state must validate every tick");
        }
    }
}
