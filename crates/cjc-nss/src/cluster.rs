//! Phase 2 — cluster primitives.
//!
//! Phase 1 modelled a single-tier queue / worker pool: one `SystemState`,
//! one `PressureField`, one trajectory. Phase 2 generalises to an
//! **N-node distributed cluster** with:
//!
//! - per-node `SystemState` (each node has its own queue + pressures),
//! - **network edges** between nodes (deterministic, weighted, with
//!   bounded capacity and per-edge congestion state),
//! - **routing policies** that decide which node receives an arriving
//!   task,
//! - **node health** (Healthy / Failed) — failures drain a node's
//!   queue and force the router around it.
//!
//! Determinism contract carried forward:
//! - Nodes keyed by [`NodeId`] (a `u32`) in a [`BTreeMap`]; iteration
//!   is monotonic in `NodeId`.
//! - Edges keyed by `(src, dst)` tuples in a [`BTreeMap`]; same
//!   property.
//! - All RNG draws still flow through [`NssSeed::substream`] with
//!   distinct salts per (node, domain).

use crate::error::NssError;
use crate::failure::FailureState;
use crate::scheduler::SchedulerAction;
use crate::system::SystemState;
use std::collections::BTreeMap;

/// Identifier for a node in a cluster topology. We keep this a small
/// `u32` so canonical-byte encodings stay compact even at 10k-node
/// scale (an order of magnitude beyond Phase 2's plausible scope).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Canonical 4-byte representation (little-endian).
    pub fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Node-health status. Modelled as a closed enum so failure-cascade
/// reasoning is exhaustive (no `_Other` escape hatch).
///
/// **Phase 3e** — added [`NodeHealth::Absent`] for capacity-aware
/// autoscaling: a node in the topology that is *not currently part of
/// the active cluster*. Different from `Failed` because the failure
/// rollup treats `Absent` as Nominal (it's not broken, it's just not
/// there). Routing and service skip `Absent` nodes the same way they
/// skip `Failed` ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeHealth {
    /// Accepts work, propagates pressure normally.
    Healthy,
    /// Rejects all incoming work, drains internal queue, contributes
    /// no outgoing network pressure. A `Failed` node still exists in
    /// the topology and contributes a `Collapse` label to the rollup
    /// — it's broken, not absent.
    Failed,
    /// Not currently part of the active cluster (autoscaled down or
    /// not yet spawned). Skipped for routing, service, network
    /// propagation, and failure rollup. The topology slot persists so
    /// the node can be re-added later via [`crate::Intervention::AddNode`].
    Absent,
}

impl NodeHealth {
    /// Canonical short label.
    pub fn label(self) -> &'static str {
        match self {
            NodeHealth::Healthy => "healthy",
            NodeHealth::Failed => "failed",
            NodeHealth::Absent => "absent",
        }
    }

    /// True if the node participates in scheduling decisions
    /// (routing, service, network flow). Only `Healthy` participates.
    pub fn participates(self) -> bool {
        matches!(self, NodeHealth::Healthy)
    }
}

/// One network edge between two cluster nodes.
///
/// `capacity` is the maximum tasks-per-tick the link can carry without
/// becoming congested. `latency_ticks` is the propagation delay — task
/// pressure injected at the source surfaces at the destination this
/// many ticks later. Phase 2 starts with `latency_ticks = 0` (i.e.
/// same-tick propagation through the network); the field exists so
/// Phase 3 can introduce delay buffers without an ABI break.
///
/// `congestion` is the *running* congestion magnitude in `[0, 1]`:
/// 0 means the link is idle, 1 means it's saturated. The cluster
/// simulator updates this every tick based on how much work crossed
/// the link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkLink {
    /// Tasks-per-tick capacity. Must be ≥ 1.
    pub capacity: u32,
    /// Propagation delay in ticks. Must be ≥ 0. Phase 2: typically 0.
    pub latency_ticks: u32,
    /// Multiplicative weight on pressure that flows across this link.
    /// Must be in `[0, 1]`. Acts the same way as a [`crate::PressureEdge`]
    /// weight, just at the cluster scale.
    pub weight: f64,
    /// Running congestion in `[0, 1]`. The simulator updates this; the
    /// cluster's pressure propagator reads it to drive downstream
    /// node-local network pressure.
    pub congestion: f64,
}

impl NetworkLink {
    /// Build a link. Validates every field.
    pub fn new(capacity: u32, latency_ticks: u32, weight: f64) -> Result<Self, NssError> {
        if capacity == 0 {
            return Err(NssError::InvalidConfig {
                detail: "NetworkLink.capacity must be >= 1".into(),
            });
        }
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(NssError::InvalidConfig {
                detail: format!("NetworkLink.weight must be in [0, 1], got {weight}"),
            });
        }
        Ok(Self {
            capacity,
            latency_ticks,
            weight,
            congestion: 0.0,
        })
    }

    /// Canonical bytes — used by [`ClusterTopology::canonical_bytes`].
    /// `congestion` is intentionally *not* included: it's runtime state,
    /// not topology. Two topologies with identical (capacity, latency,
    /// weight) produce identical bytes regardless of how far through
    /// a run they are.
    pub fn topology_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&self.capacity.to_le_bytes());
        bytes.extend_from_slice(&self.latency_ticks.to_le_bytes());
        bytes.extend_from_slice(&self.weight.to_bits().to_le_bytes());
        bytes
    }
}

/// Directed cluster topology: which nodes exist + how they're wired.
///
/// Stored as two [`BTreeMap`]s (nodes + edges) so iteration order is
/// `NodeId`-monotonic / `(src, dst)`-lexicographic. The cluster
/// simulator and the cluster propagator both consume these
/// iterators directly; the ordering *is* the propagation order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ClusterTopology {
    nodes: BTreeMap<NodeId, ()>,
    edges: BTreeMap<(NodeId, NodeId), NetworkLink>,
}

impl ClusterTopology {
    /// Empty topology — no nodes, no edges.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a fully-connected `n`-node topology with uniform link
    /// parameters. Convenient default for small cluster experiments.
    /// Self-edges are not added.
    pub fn complete(n: u32, link_capacity: u32, link_weight: f64) -> Result<Self, NssError> {
        if n == 0 {
            return Err(NssError::InvalidConfig {
                detail: "ClusterTopology::complete requires n >= 1".into(),
            });
        }
        let mut t = Self::empty();
        for i in 0..n {
            t.add_node(NodeId(i))?;
        }
        let link = NetworkLink::new(link_capacity, 0, link_weight)?;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    t.add_edge(NodeId(i), NodeId(j), link)?;
                }
            }
        }
        Ok(t)
    }

    /// Build a ring topology — each node connects to its `(id + 1) mod n`
    /// successor. Useful for testing partition + cascade dynamics where
    /// connectivity is fragile.
    pub fn ring(n: u32, link_capacity: u32, link_weight: f64) -> Result<Self, NssError> {
        if n < 2 {
            return Err(NssError::InvalidConfig {
                detail: "ClusterTopology::ring requires n >= 2".into(),
            });
        }
        let mut t = Self::empty();
        for i in 0..n {
            t.add_node(NodeId(i))?;
        }
        let link = NetworkLink::new(link_capacity, 0, link_weight)?;
        for i in 0..n {
            let next = (i + 1) % n;
            t.add_edge(NodeId(i), NodeId(next), link)?;
        }
        Ok(t)
    }

    /// Add a node to the topology. Fails on duplicate insertion.
    pub fn add_node(&mut self, id: NodeId) -> Result<(), NssError> {
        if self.nodes.contains_key(&id) {
            return Err(NssError::InvalidConfig {
                detail: format!("duplicate node {}", id),
            });
        }
        self.nodes.insert(id, ());
        Ok(())
    }

    /// Add a directed edge. Fails on self-loop, on duplicate, or if
    /// either endpoint is unknown.
    pub fn add_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        link: NetworkLink,
    ) -> Result<(), NssError> {
        if src == dst {
            return Err(NssError::InvalidConfig {
                detail: format!("self-loop on node {}", src),
            });
        }
        if !self.nodes.contains_key(&src) {
            return Err(NssError::InvalidConfig {
                detail: format!("edge source {} not in topology", src),
            });
        }
        if !self.nodes.contains_key(&dst) {
            return Err(NssError::InvalidConfig {
                detail: format!("edge destination {} not in topology", dst),
            });
        }
        if self.edges.contains_key(&(src, dst)) {
            return Err(NssError::InvalidConfig {
                detail: format!("duplicate edge {} -> {}", src, dst),
            });
        }
        self.edges.insert((src, dst), link);
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate node IDs in `NodeId` order.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    /// Iterate every `(src, dst, link)` triple in lex `(src, dst)`
    /// order. The cluster simulator and propagator both consume this
    /// directly.
    pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &NetworkLink)> {
        self.edges.iter().map(|((s, d), l)| (*s, *d, l))
    }

    /// Get a specific link (by source and destination).
    pub fn link(&self, src: NodeId, dst: NodeId) -> Option<&NetworkLink> {
        self.edges.get(&(src, dst))
    }

    /// Get a specific link mutably — used by the simulator to update
    /// `congestion` per tick. The topology itself is otherwise treated
    /// as immutable.
    pub fn link_mut(&mut self, src: NodeId, dst: NodeId) -> Option<&mut NetworkLink> {
        self.edges.get_mut(&(src, dst))
    }

    /// Out-degree of a node (count of edges where `src == node`).
    pub fn out_degree(&self, node: NodeId) -> usize {
        self.edges.keys().filter(|(s, _)| *s == node).count()
    }

    /// Outgoing-edge destinations from a node, in deterministic order.
    pub fn neighbours(&self, node: NodeId) -> Vec<NodeId> {
        self.edges
            .keys()
            .filter(|(s, _)| *s == node)
            .map(|(_, d)| *d)
            .collect()
    }

    /// Canonical topology bytes (no runtime state). Two topologies
    /// with identical nodes + edges + link parameters produce
    /// identical bytes regardless of insertion order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64 + self.nodes.len() * 4 + self.edges.len() * 24);
        bytes.extend_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        for id in self.nodes.keys() {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        bytes.push(b'|');
        bytes.extend_from_slice(&(self.edges.len() as u64).to_le_bytes());
        for ((src, dst), link) in self.edges.iter() {
            bytes.extend_from_slice(&src.to_le_bytes());
            bytes.extend_from_slice(&dst.to_le_bytes());
            bytes.extend_from_slice(&link.topology_bytes());
        }
        bytes
    }
}

/// One snapshot of the cluster at one tick. Carries per-node
/// `SystemState` keyed by `NodeId`, the topology (with current per-link
/// `congestion`), and per-node health.
///
/// We embed the topology by reference in the simulator (`Rc<...>` would
/// add a runtime cost; instead we pass the topology alongside the state
/// where needed). The state itself only carries the *mutable* runtime
/// data — node states + link congestion + node health.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterSystemState {
    /// Tick index (monotonic).
    pub tick: u64,
    /// Per-node `SystemState`, keyed by `NodeId` for deterministic
    /// iteration.
    pub nodes: BTreeMap<NodeId, SystemState>,
    /// Per-link congestion in `[0, 1]`, keyed by `(src, dst)`. Tracked
    /// separately from the topology so the topology can stay
    /// conceptually immutable through a run.
    pub link_congestion: BTreeMap<(NodeId, NodeId), f64>,
    /// Per-node health.
    pub node_health: BTreeMap<NodeId, NodeHealth>,
}

impl ClusterSystemState {
    /// Build an initial state for a given topology — every node gets
    /// `SystemState::initial()`, every link starts at zero congestion,
    /// every node is healthy.
    pub fn initial(topology: &ClusterTopology) -> Self {
        let mut nodes = BTreeMap::new();
        let mut node_health = BTreeMap::new();
        for id in topology.nodes() {
            nodes.insert(id, SystemState::initial());
            node_health.insert(id, NodeHealth::Healthy);
        }
        let mut link_congestion = BTreeMap::new();
        for (src, dst, _) in topology.edges() {
            link_congestion.insert((src, dst), 0.0);
        }
        Self {
            tick: 0,
            nodes,
            link_congestion,
            node_health,
        }
    }

    /// Validate every numeric field. Called by the simulator after
    /// every tick.
    pub fn validate(&self) -> Result<(), NssError> {
        for (id, s) in self.nodes.iter() {
            s.validate().map_err(|e| match e {
                NssError::InvalidState { detail } => NssError::InvalidState {
                    detail: format!("node {}: {}", id, detail),
                },
                other => other,
            })?;
        }
        for ((src, dst), c) in self.link_congestion.iter() {
            if !c.is_finite() || !(0.0..=1.0).contains(c) {
                return Err(NssError::InvalidState {
                    detail: format!(
                        "link {}->{} congestion {} out of [0, 1]",
                        src, dst, c
                    ),
                });
            }
        }
        Ok(())
    }

    /// Canonical bytes for hashing. Nodes in `NodeId` order, links in
    /// lex `(src, dst)` order, health in node order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128 + self.nodes.len() * 96);
        bytes.extend_from_slice(&self.tick.to_le_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        for (id, s) in self.nodes.iter() {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&s.canonical_bytes());
            bytes.push(b';');
        }
        bytes.push(b'#');
        for ((src, dst), c) in self.link_congestion.iter() {
            bytes.extend_from_slice(&src.to_le_bytes());
            bytes.extend_from_slice(&dst.to_le_bytes());
            bytes.extend_from_slice(&c.to_bits().to_le_bytes());
        }
        bytes.push(b'#');
        for (id, h) in self.node_health.iter() {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(h.label().as_bytes());
            bytes.push(b';');
        }
        bytes
    }

    /// True if every node is healthy.
    pub fn all_healthy(&self) -> bool {
        self.node_health
            .values()
            .all(|h| *h == NodeHealth::Healthy)
    }

    /// Number of failed nodes.
    pub fn failed_count(&self) -> usize {
        self.node_health
            .values()
            .filter(|h| **h == NodeHealth::Failed)
            .count()
    }
}

/// One cluster-level SIR record — per-tick (state, per-node scheduler
/// actions, per-node failure labels). The cluster equivalent of a
/// Phase 1 [`crate::SystemEvent`].
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterEvent {
    /// Cluster state at this tick.
    pub state: ClusterSystemState,
    /// Per-node scheduler action.
    pub actions: BTreeMap<NodeId, SchedulerAction>,
    /// Per-node failure label (each node carries its own Phase 1
    /// `FailureState`).
    pub failures: BTreeMap<NodeId, FailureState>,
    /// Cluster-level rolled-up failure label — `Collapse` if any node
    /// is in `Collapse`, `Degraded` if any node is in `Degraded`
    /// (and none are in `Collapse`), else `Nominal`.
    pub cluster_failure: FailureState,
}

impl ClusterEvent {
    /// Canonical bytes — state + actions + failures + rollup.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.state.canonical_bytes();
        bytes.push(b'#');
        bytes.extend_from_slice(&(self.actions.len() as u64).to_le_bytes());
        for (id, a) in self.actions.iter() {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&a.canonical_bytes());
            bytes.push(b';');
        }
        bytes.push(b'#');
        for (id, f) in self.failures.iter() {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&f.canonical_bytes());
            bytes.push(b';');
        }
        bytes.push(b'#');
        bytes.extend_from_slice(&self.cluster_failure.canonical_bytes());
        bytes
    }
}

/// A cluster trajectory — sequence of `ClusterEvent`s in tick order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ClusterTrajectory {
    events: Vec<ClusterEvent>,
}

impl ClusterTrajectory {
    /// Empty trajectory.
    pub fn empty() -> Self {
        Self { events: Vec::new() }
    }

    /// Build from a vec of events; validates tick monotonicity.
    pub fn from_events(events: Vec<ClusterEvent>) -> Result<Self, NssError> {
        for w in events.windows(2) {
            if w[1].state.tick != w[0].state.tick + 1 {
                return Err(NssError::InvalidTrajectory {
                    detail: format!(
                        "tick must increase by 1: {} -> {}",
                        w[0].state.tick, w[1].state.tick
                    ),
                });
            }
        }
        Ok(Self { events })
    }

    /// Append an event, enforcing tick = previous + 1. An empty
    /// trajectory accepts any starting tick — this is required by the
    /// Phase 3a counterfactual-forking API where a fork resumes at the
    /// snapshot tick rather than at 0.
    pub fn push(&mut self, ev: ClusterEvent) -> Result<(), NssError> {
        if let Some(prev) = self.events.last() {
            let expected = prev.state.tick + 1;
            if ev.state.tick != expected {
                return Err(NssError::InvalidTrajectory {
                    detail: format!("expected tick {}, got {}", expected, ev.state.tick),
                });
            }
        }
        self.events.push(ev);
        Ok(())
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate in tick order.
    pub fn iter(&self) -> impl Iterator<Item = &ClusterEvent> {
        self.events.iter()
    }

    /// Borrow as slice.
    pub fn as_slice(&self) -> &[ClusterEvent] {
        &self.events
    }

    /// Last state, if any.
    pub fn last_state(&self) -> Option<&ClusterSystemState> {
        self.events.last().map(|e| &e.state)
    }

    /// Canonical bytes — every event in tick order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.events.len() * 128);
        bytes.extend_from_slice(&(self.events.len() as u64).to_le_bytes());
        for ev in &self.events {
            bytes.extend_from_slice(&ev.canonical_bytes());
            bytes.push(b'\n');
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_complete_constructs_correct_edge_count() {
        let t = ClusterTopology::complete(4, 8, 0.5).unwrap();
        assert_eq!(t.node_count(), 4);
        // Complete digraph on 4 nodes (no self-loops) has 4*3 = 12 edges.
        assert_eq!(t.edge_count(), 12);
    }

    #[test]
    fn topology_ring_has_n_edges() {
        let t = ClusterTopology::ring(5, 4, 0.5).unwrap();
        assert_eq!(t.node_count(), 5);
        assert_eq!(t.edge_count(), 5);
        // Successor check: 0 -> 1, 1 -> 2, ..., 4 -> 0.
        for i in 0..5u32 {
            let next = NodeId((i + 1) % 5);
            assert!(t.link(NodeId(i), next).is_some());
        }
    }

    #[test]
    fn topology_rejects_self_loop_duplicate_and_missing_endpoints() {
        let mut t = ClusterTopology::empty();
        t.add_node(NodeId(0)).unwrap();
        t.add_node(NodeId(1)).unwrap();
        let link = NetworkLink::new(4, 0, 0.5).unwrap();
        assert!(t.add_edge(NodeId(0), NodeId(0), link).is_err());
        assert!(t.add_edge(NodeId(0), NodeId(1), link).is_ok());
        assert!(t.add_edge(NodeId(0), NodeId(1), link).is_err());
        assert!(t.add_edge(NodeId(0), NodeId(7), link).is_err());
    }

    #[test]
    fn canonical_bytes_stable_under_insertion_order() {
        // Build same topology two different ways; canonical bytes must agree.
        let mut a = ClusterTopology::empty();
        a.add_node(NodeId(2)).unwrap();
        a.add_node(NodeId(0)).unwrap();
        a.add_node(NodeId(1)).unwrap();
        let link = NetworkLink::new(4, 0, 0.5).unwrap();
        a.add_edge(NodeId(2), NodeId(0), link).unwrap();
        a.add_edge(NodeId(0), NodeId(1), link).unwrap();

        let mut b = ClusterTopology::empty();
        b.add_node(NodeId(0)).unwrap();
        b.add_node(NodeId(1)).unwrap();
        b.add_node(NodeId(2)).unwrap();
        b.add_edge(NodeId(0), NodeId(1), link).unwrap();
        b.add_edge(NodeId(2), NodeId(0), link).unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn initial_cluster_state_is_all_healthy_zero_congestion() {
        let t = ClusterTopology::complete(3, 4, 0.5).unwrap();
        let s = ClusterSystemState::initial(&t);
        assert!(s.all_healthy());
        assert_eq!(s.failed_count(), 0);
        for c in s.link_congestion.values() {
            assert_eq!(*c, 0.0);
        }
        assert_eq!(s.nodes.len(), 3);
    }

    #[test]
    fn cluster_trajectory_enforces_monotonic_ticks() {
        let t = ClusterTopology::complete(2, 4, 0.5).unwrap();
        let init = ClusterSystemState::initial(&t);
        let mut ev0_state = init.clone();
        ev0_state.tick = 0;
        let ev0 = ClusterEvent {
            state: ev0_state,
            actions: BTreeMap::new(),
            failures: BTreeMap::new(),
            cluster_failure: FailureState::nominal(),
        };
        let mut tr = ClusterTrajectory::empty();
        tr.push(ev0).unwrap();
        let mut ev2_state = init;
        ev2_state.tick = 2;
        let ev2 = ClusterEvent {
            state: ev2_state,
            actions: BTreeMap::new(),
            failures: BTreeMap::new(),
            cluster_failure: FailureState::nominal(),
        };
        assert!(tr.push(ev2).is_err(), "non-consecutive tick must be rejected");
    }
}
