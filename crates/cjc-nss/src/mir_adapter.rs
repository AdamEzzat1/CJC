//! Phase 5a — MIR-trace → NSS-trajectory adapter.
//!
//! This is the **bridge between the CJC-Lang compiler and NSS**. The
//! adapter consumes a stream of [`MirTraceEvent`]s emitted by an
//! instrumented MIR executor (or a static MIR analyser, or any source
//! that can project compilation/execution into the event shape below)
//! and produces a [`ClusterTrajectory`] that the existing
//! [`crate::ClusterNeuralSystemsSimulator`] can predict on without
//! any modifications.
//!
//! **Why this works**: NSS's pressure model is intentionally abstract.
//! It doesn't care whether a `PressureKind::Memory` increase came
//! from a real GPU's OOM crash or from a MIR allocator's heap-growth
//! event. The same pressure-propagation graph, the same multi-timescale
//! memory, the same advisor — they all operate on the trajectory
//! shape, not the domain semantics. The adapter just *projects*
//! compiler-side concepts onto the existing pressure kinds:
//!
//! | MIR concept                  | NSS [`PressureKind`]    |
//! |------------------------------|--------------------------|
//! | Register pressure / spills   | `Cpu`                    |
//! | Heap bytes in use            | `Memory`                 |
//! | Heap-fragmentation events    | `Memory` (sustained)     |
//! | Call-stack depth             | `Sync`                   |
//! | Syscalls / IO events         | `Io`                     |
//! | Hot-path branch frequency    | `Throughput`             |
//! | Cross-function jumps         | `Network` (control flow) |
//! | GC pauses                    | `Sync` (synchronisation) |
//!
//! Each "basic block" in the MIR trace becomes a [`crate::NodeId`] in
//! the cluster topology. The adapter aggregates MIR events into NSS
//! ticks at a configurable granularity (`events_per_tick`), producing
//! one cluster event per NSS tick.
//!
//! ## Why basic blocks as nodes?
//!
//! - Basic blocks have **clear entry/exit semantics** (one entry, one
//!   exit), so per-block pressure is well-defined.
//! - The compiler's **decision points** (inline? unroll? vectorise?)
//!   naturally attach to basic blocks.
//! - The cluster topology's **network edges** between blocks become
//!   the control-flow graph — adjacent blocks share edges, so
//!   cross-block pressure (e.g. register-allocator state being
//!   carried across a basic-block boundary) propagates correctly.
//!
//! ## Per-block edge construction
//!
//! In the absence of an explicit control-flow graph from the caller,
//! the adapter builds a **complete** topology (every block connects
//! to every other block). This is the most pessimistic edge structure
//! — pressure can flow anywhere. Phase 5b's legality verifier can
//! prune unreachable edges if a real CFG is provided. For Phase 5a
//! the complete topology is the simplest correct choice.
//!
//! ## Determinism contract
//!
//! Adapter is a **pure projection** — no RNG, no parallelism. Same
//! `(events, config)` always produces the same `ClusterTrajectory`.
//! Compiler-side determinism is therefore inherited automatically:
//! the same MIR trace + same adapter config = the same NSS
//! prediction.

use crate::cluster::{
    ClusterEvent, ClusterSystemState, ClusterTopology, ClusterTrajectory, NodeHealth, NodeId,
};
use crate::error::NssError;
use crate::failure::FailureState;
use crate::pressure::{Pressure, PressureField, PressureKind};
use crate::scheduler::SchedulerAction;
use crate::system::SystemState;
use cjc_repro::KahanAccumulatorF64;
use std::collections::BTreeMap;

/// One trace event emitted by an instrumented CJC-Lang MIR executor.
///
/// Designed to be **small, deterministic, and serialisable**. Any
/// MIR-aware source (`cjc-mir-exec` runtime instrumentation, a
/// post-hoc trace replay, a static cost analyser) can produce these.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirTraceEvent {
    /// Monotonic event index (the executor's instruction or basic-block
    /// counter). The adapter uses this to bucket events into NSS ticks.
    pub tick: u64,
    /// Which basic block emitted this event. Maps directly to a
    /// [`NodeId`] in the cluster topology.
    pub block_id: u32,
    /// Register pressure in `[0, 1]` — fraction of physical registers
    /// currently spilled. Loads [`PressureKind::Cpu`].
    pub register_pressure: f64,
    /// Currently-allocated heap bytes. Loads [`PressureKind::Memory`]
    /// via `bytes / heap_capacity_bytes` saturation.
    pub heap_bytes_in_use: u64,
    /// Call-stack depth at the event. Loads [`PressureKind::Sync`]
    /// via `depth / call_depth_threshold` (proxy for synchronisation
    /// pressure from deep call chains).
    pub call_depth: u32,
    /// True if a branch was taken at this event. Aggregated as
    /// branch-density per tick → [`PressureKind::Throughput`].
    pub branch_taken: bool,
    /// True if a syscall / IO operation happened. Loads
    /// [`PressureKind::Io`] (counted per tick).
    pub io_event: bool,
    /// True if a GC / heap-compaction event happened. Loads
    /// [`PressureKind::Sync`] (GC pauses are synchronisation surface).
    pub gc_event: bool,
    /// Number of MIR instructions executed in this event window
    /// (typically the size of the basic block). Used by the adapter
    /// to weight per-block aggregates.
    pub instruction_count: u32,
    /// FP-op density in `[0, 1]` for this event window — fraction of
    /// executed instructions that were floating-point arithmetic.
    /// Loads [`PressureKind::Thermal`] (FPU utilisation is the
    /// dominant heat source on numeric workloads).
    ///
    /// Option B (real instrumentation) populates this from the
    /// executor's FP-op counter; Option A (synthetic traces) leaves it
    /// `0.0`, which preserves the documented pre-Option-B behaviour
    /// that synthetic predictions carry no thermal signal.
    pub thermal_intensity: f64,
}

impl MirTraceEvent {
    /// Build a minimal MIR event with sensible defaults for the
    /// remaining fields. Useful in tests + when a caller only knows
    /// some of the pressure dimensions.
    pub fn minimal(tick: u64, block_id: u32) -> Self {
        Self {
            tick,
            block_id,
            register_pressure: 0.0,
            heap_bytes_in_use: 0,
            call_depth: 1,
            branch_taken: false,
            io_event: false,
            gc_event: false,
            instruction_count: 1,
            thermal_intensity: 0.0,
        }
    }
}

/// Knobs for the adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirAdapterConfig {
    /// Number of MIR events that aggregate into one NSS tick. Higher
    /// = coarser-grained predictions, lower = finer-grained but more
    /// expensive. Must be ≥ 1. Default 16.
    pub events_per_tick: u32,
    /// Total number of distinct basic blocks (= NSS nodes). Must
    /// match the maximum `block_id` + 1 in the trace. Must be ≥ 1.
    pub n_blocks: u32,
    /// Heap-capacity reference value. Used to normalise
    /// `heap_bytes_in_use` into a `[0, 1]` saturation. Must be ≥ 1.
    /// Default 1 GiB.
    pub heap_capacity_bytes: u64,
    /// Call-depth threshold above which `Sync` pressure saturates.
    /// Default 32 (deep call chains).
    pub call_depth_threshold: u32,
    /// Per-edge network-link capacity in the synthesised topology.
    /// Default 8.
    pub link_capacity: u32,
    /// Per-edge weight in the synthesised topology. Default 0.25.
    pub link_weight: f64,
    /// Tick at which the synthesised trajectory starts. Defaults to
    /// 0, but a caller building multiple consecutive batches can pass
    /// a non-zero start.
    pub initial_tick: u64,
}

impl Default for MirAdapterConfig {
    fn default() -> Self {
        Self {
            events_per_tick: 16,
            n_blocks: 1,
            heap_capacity_bytes: 1024 * 1024 * 1024, // 1 GiB
            call_depth_threshold: 32,
            link_capacity: 8,
            link_weight: 0.25,
            initial_tick: 0,
        }
    }
}

impl MirAdapterConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.events_per_tick == 0 {
            return Err(NssError::InvalidConfig {
                detail: "events_per_tick must be >= 1".into(),
            });
        }
        if self.n_blocks == 0 {
            return Err(NssError::InvalidConfig {
                detail: "n_blocks must be >= 1".into(),
            });
        }
        if self.heap_capacity_bytes == 0 {
            return Err(NssError::InvalidConfig {
                detail: "heap_capacity_bytes must be >= 1".into(),
            });
        }
        if self.call_depth_threshold == 0 {
            return Err(NssError::InvalidConfig {
                detail: "call_depth_threshold must be >= 1".into(),
            });
        }
        if self.link_capacity == 0 {
            return Err(NssError::InvalidConfig {
                detail: "link_capacity must be >= 1".into(),
            });
        }
        if !self.link_weight.is_finite() || !(0.0..=1.0).contains(&self.link_weight) {
            return Err(NssError::InvalidConfig {
                detail: format!("link_weight must be in [0, 1], got {}", self.link_weight),
            });
        }
        Ok(())
    }
}

/// Build the synthesised cluster topology that the adapter uses. One
/// node per basic block; complete-graph edges. Exposed so callers can
/// inspect it / feed it into [`crate::ClusterNeuralSystemsSimulator`].
pub fn build_topology(cfg: &MirAdapterConfig) -> Result<ClusterTopology, NssError> {
    cfg.validate()?;
    if cfg.n_blocks == 1 {
        // Special case: complete-topology requires n >= 2 edges; for a
        // single-block trace we build an edgeless topology with just
        // the one node.
        let mut t = ClusterTopology::empty();
        t.add_node(NodeId(0))?;
        return Ok(t);
    }
    ClusterTopology::complete(cfg.n_blocks, cfg.link_capacity, cfg.link_weight)
}

/// Adapter result: the synthesised trajectory **plus** the topology
/// it was built against. Both are needed downstream — the topology
/// goes into [`crate::ClusterNssConfig`] / the cluster NSS's
/// run-id-binding, and the trajectory is fed to `fit` / `predict_next`.
#[derive(Clone, Debug)]
pub struct MirAdapterOutput {
    /// Synthesised cluster topology (n_blocks nodes, complete-graph
    /// edges with the configured link parameters).
    pub topology: ClusterTopology,
    /// Synthesised cluster trajectory — one event per NSS tick.
    pub trajectory: ClusterTrajectory,
    /// How many MIR events were ingested.
    pub events_ingested: u64,
    /// How many NSS ticks were produced.
    pub ticks_produced: u64,
}

/// Convert a batch of MIR trace events into a cluster trajectory.
///
/// Algorithm:
/// 1. Validate config + check the event stream's `block_id`s are all
///    `< cfg.n_blocks`.
/// 2. Group events by NSS-tick bucket
///    (`event.tick / cfg.events_per_tick`).
/// 3. For each bucket, group events further by `block_id`.
/// 4. For each (tick, block) bucket, aggregate the events into a
///    per-block [`PressureField`] using Kahan accumulation:
///    - `Cpu` ← mean register_pressure
///    - `Memory` ← (mean heap_bytes / capacity) clipped to [0, 1.5]
///    - `Sync` ← mean (call_depth / depth_threshold) + GC density
///    - `Io` ← fraction of events with `io_event = true`
///    - `Throughput` ← fraction of events with `branch_taken = true`
///    - `Thermal` ← mean thermal_intensity (FP-op density; 0.0 on
///      synthetic Option-A traces, real under Option-B instrumentation)
/// 5. Emit one [`ClusterEvent`] per tick bucket.
pub fn adapt_mir_trace_to_cluster_trajectory(
    events: &[MirTraceEvent],
    cfg: &MirAdapterConfig,
) -> Result<MirAdapterOutput, NssError> {
    cfg.validate()?;
    // Bounds-check block_ids.
    for ev in events {
        if ev.block_id >= cfg.n_blocks {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "MIR event block_id {} >= n_blocks {}",
                    ev.block_id, cfg.n_blocks
                ),
            });
        }
    }

    let topology = build_topology(cfg)?;

    // Group events by NSS tick bucket.
    let mut buckets: BTreeMap<u64, Vec<&MirTraceEvent>> = BTreeMap::new();
    for ev in events {
        let bucket = ev.tick / (cfg.events_per_tick as u64);
        buckets.entry(bucket).or_default().push(ev);
    }

    // Build the trajectory tick by tick. We iterate over buckets in
    // ascending key order (BTreeMap iteration) so the resulting tick
    // sequence is monotonic.
    let mut traj = ClusterTrajectory::empty();
    let mut produced: u64 = 0;
    for (bucket_idx, bucket_events) in buckets.iter() {
        let nss_tick = cfg.initial_tick + bucket_idx;
        let state = build_cluster_state(nss_tick, bucket_events, cfg, &topology)?;
        // Phase 5a: scheduler actions + per-node failure labels are
        // synthesised as Idle + Nominal. A future Phase 5c could
        // refine these (e.g. tag a basic block with `Collapse` if
        // its predicted pressure exceeds a threshold).
        let mut actions: BTreeMap<NodeId, SchedulerAction> = BTreeMap::new();
        let mut failures: BTreeMap<NodeId, FailureState> = BTreeMap::new();
        for id in topology.nodes() {
            actions.insert(id, SchedulerAction::idle());
            failures.insert(id, FailureState::nominal());
        }
        let event = ClusterEvent {
            state,
            actions,
            failures,
            cluster_failure: FailureState::nominal(),
        };
        traj.push(event)?;
        produced += 1;
    }

    Ok(MirAdapterOutput {
        topology,
        trajectory: traj,
        events_ingested: events.len() as u64,
        ticks_produced: produced,
    })
}

/// Build one tick's `ClusterSystemState` from the MIR events
/// belonging to that tick bucket. Aggregates per-block.
fn build_cluster_state(
    tick: u64,
    bucket_events: &[&MirTraceEvent],
    cfg: &MirAdapterConfig,
    topology: &ClusterTopology,
) -> Result<ClusterSystemState, NssError> {
    // Per-block aggregates.
    struct BlockAgg {
        reg_sum: KahanAccumulatorF64,
        heap_sum: KahanAccumulatorF64, // bytes
        call_sum: KahanAccumulatorF64,
        thermal_sum: KahanAccumulatorF64,
        branch_count: u64,
        io_count: u64,
        gc_count: u64,
        instr_count: u64,
        event_count: u64,
    }
    impl BlockAgg {
        fn new() -> Self {
            Self {
                reg_sum: KahanAccumulatorF64::new(),
                heap_sum: KahanAccumulatorF64::new(),
                call_sum: KahanAccumulatorF64::new(),
                thermal_sum: KahanAccumulatorF64::new(),
                branch_count: 0,
                io_count: 0,
                gc_count: 0,
                instr_count: 0,
                event_count: 0,
            }
        }
    }
    let mut per_block: BTreeMap<u32, BlockAgg> = BTreeMap::new();
    for ev in bucket_events {
        let agg = per_block.entry(ev.block_id).or_insert_with(BlockAgg::new);
        agg.reg_sum.add(ev.register_pressure);
        agg.heap_sum.add(ev.heap_bytes_in_use as f64);
        agg.call_sum.add(ev.call_depth as f64);
        agg.thermal_sum.add(ev.thermal_intensity);
        if ev.branch_taken {
            agg.branch_count += 1;
        }
        if ev.io_event {
            agg.io_count += 1;
        }
        if ev.gc_event {
            agg.gc_count += 1;
        }
        agg.instr_count = agg.instr_count.saturating_add(ev.instruction_count as u64);
        agg.event_count += 1;
    }

    let mut nodes_state: BTreeMap<NodeId, SystemState> = BTreeMap::new();
    let mut node_health: BTreeMap<NodeId, NodeHealth> = BTreeMap::new();
    for id in topology.nodes() {
        node_health.insert(id, NodeHealth::Healthy);
        let block = id.0;
        let mut field = PressureField::with_default_thresholds();
        // Initialise every pressure to zero with the standard
        // dissipation profile.
        for k in PressureKind::all() {
            field.set(k, Pressure::new(0.0, 1.0, 0.1)?);
        }
        let agg = per_block.remove(&block);
        if let Some(agg) = agg {
            let n = agg.event_count.max(1) as f64;
            let cpu_p = (agg.reg_sum.finalize() / n).clamp(0.0, 1.5);
            let heap_avg = agg.heap_sum.finalize() / n;
            let mem_p = (heap_avg / cfg.heap_capacity_bytes as f64).min(1.5);
            let call_avg = agg.call_sum.finalize() / n;
            let call_sat = (call_avg / cfg.call_depth_threshold as f64).min(1.0);
            let gc_density = agg.gc_count as f64 / n;
            // Sync pressure combines call-depth and GC.
            let sync_p = (call_sat + gc_density).min(1.5);
            let branch_p = agg.branch_count as f64 / n;
            let io_p = agg.io_count as f64 / n;
            let thermal_p = (agg.thermal_sum.finalize() / n).clamp(0.0, 1.5);
            field.set(PressureKind::Cpu, Pressure::new(cpu_p, 1.0, 0.1)?);
            field.set(PressureKind::Memory, Pressure::new(mem_p, 1.0, 0.05)?);
            field.set(PressureKind::Sync, Pressure::new(sync_p, 1.0, 0.1)?);
            field.set(PressureKind::Io, Pressure::new(io_p, 1.0, 0.1)?);
            field.set(PressureKind::Throughput, Pressure::new(branch_p, 1.0, 0.1)?);
            field.set(PressureKind::Thermal, Pressure::new(thermal_p, 1.0, 0.1)?);
        }
        // in_flight: events seen in this tick on this block; completed:
        // running total of instructions executed.
        let in_flight = per_block.get(&block).map(|a| a.event_count).unwrap_or(0);
        let completed = bucket_events
            .iter()
            .filter(|e| e.block_id == block)
            .map(|e| e.instruction_count as u64)
            .sum();
        let state = SystemState {
            tick,
            pressures: field,
            in_flight,
            completed,
            rejected: 0,
            mean_service_time: 1.0,
        };
        nodes_state.insert(id, state);
    }

    // Initial link_congestion: zero everywhere. The Phase 1
    // intra-node propagator on a downstream `predict_next` will
    // figure out per-tick congestion.
    let mut link_congestion = BTreeMap::new();
    for (src, dst, _) in topology.edges() {
        link_congestion.insert((src, dst), 0.0);
    }

    Ok(ClusterSystemState {
        tick,
        nodes: nodes_state,
        link_congestion,
        node_health,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_block_trace() -> Vec<MirTraceEvent> {
        // 48 events across 3 blocks, 16 events per tick → 3 ticks.
        let mut v = Vec::new();
        for i in 0..48u64 {
            let block = (i % 3) as u32;
            v.push(MirTraceEvent {
                tick: i,
                block_id: block,
                register_pressure: 0.3 + (block as f64) * 0.1,
                heap_bytes_in_use: 1024 * 1024 * (block as u64 + 1), // 1/2/3 MiB
                call_depth: 4 + block,
                branch_taken: i % 5 == 0,
                io_event: i % 11 == 0,
                gc_event: i % 17 == 0,
                instruction_count: 4,
                thermal_intensity: 0.2 + (block as f64) * 0.2, // 0.2/0.4/0.6
            });
        }
        v
    }

    #[test]
    fn config_default_validates() {
        let mut cfg = MirAdapterConfig::default();
        cfg.n_blocks = 4;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_rejects_invalid_fields() {
        let mut bad = MirAdapterConfig::default();
        bad.events_per_tick = 0;
        assert!(bad.validate().is_err());
        let mut bad = MirAdapterConfig::default();
        bad.n_blocks = 0;
        assert!(bad.validate().is_err());
        let mut bad = MirAdapterConfig {
            n_blocks: 2,
            link_weight: 1.5,
            ..MirAdapterConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn topology_has_correct_node_count() {
        let cfg = MirAdapterConfig {
            n_blocks: 4,
            ..MirAdapterConfig::default()
        };
        let top = build_topology(&cfg).unwrap();
        assert_eq!(top.node_count(), 4);
        // 4 nodes complete graph → 12 edges.
        assert_eq!(top.edge_count(), 12);
    }

    #[test]
    fn topology_single_block_has_no_edges() {
        let cfg = MirAdapterConfig {
            n_blocks: 1,
            ..MirAdapterConfig::default()
        };
        let top = build_topology(&cfg).unwrap();
        assert_eq!(top.node_count(), 1);
        assert_eq!(top.edge_count(), 0);
    }

    #[test]
    fn adapter_produces_one_tick_per_bucket() {
        let cfg = MirAdapterConfig {
            n_blocks: 3,
            events_per_tick: 16,
            ..MirAdapterConfig::default()
        };
        let events = three_block_trace();
        let out = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        // 48 events / 16 per tick = 3 NSS ticks.
        assert_eq!(out.ticks_produced, 3);
        assert_eq!(out.trajectory.len(), 3);
        assert_eq!(out.events_ingested, 48);
    }

    #[test]
    fn adapter_assigns_pressure_to_correct_blocks() {
        let cfg = MirAdapterConfig {
            n_blocks: 3,
            events_per_tick: 16,
            ..MirAdapterConfig::default()
        };
        let events = three_block_trace();
        let out = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        // Block 2 was given the highest register_pressure (0.5) and
        // largest heap usage (3 MiB) — its per-node CPU and Memory
        // saturation should exceed block 0's.
        let last = out.trajectory.iter().last().unwrap();
        let b0 = last.state.nodes.get(&NodeId(0)).unwrap();
        let b2 = last.state.nodes.get(&NodeId(2)).unwrap();
        let cpu0 = b0.pressures.get(PressureKind::Cpu).unwrap().saturation();
        let cpu2 = b2.pressures.get(PressureKind::Cpu).unwrap().saturation();
        assert!(
            cpu2 > cpu0,
            "block 2 cpu pressure ({}) must exceed block 0 ({})",
            cpu2,
            cpu0
        );
        // Memory: 3 MiB / 1 GiB ≈ 0.003 — well below 1.0 saturation,
        // but b2 should still be > b0.
        let mem0 = b0.pressures.get(PressureKind::Memory).unwrap().saturation();
        let mem2 = b2.pressures.get(PressureKind::Memory).unwrap().saturation();
        assert!(mem2 > mem0);
    }

    #[test]
    fn adapter_loads_thermal_from_thermal_intensity() {
        // three_block_trace assigns thermal_intensity 0.2/0.4/0.6 to
        // blocks 0/1/2 — the per-block Thermal magnitudes must follow,
        // and (the Option-B point) Thermal must NOT be degenerate with
        // Cpu, which derives from register_pressure (0.3/0.4/0.5).
        let cfg = MirAdapterConfig {
            n_blocks: 3,
            events_per_tick: 16,
            ..MirAdapterConfig::default()
        };
        let events = three_block_trace();
        let out = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        let last = out.trajectory.iter().last().unwrap();
        let thermal_of = |b: u32| {
            last.state
                .nodes
                .get(&NodeId(b))
                .unwrap()
                .pressures
                .get(PressureKind::Thermal)
                .unwrap()
                .magnitude
        };
        assert!((thermal_of(0) - 0.2).abs() < 1e-12, "block 0 thermal");
        assert!((thermal_of(1) - 0.4).abs() < 1e-12, "block 1 thermal");
        assert!((thermal_of(2) - 0.6).abs() < 1e-12, "block 2 thermal");
        // Not degenerate with Cpu (0.3/0.4/0.5 from register_pressure):
        let b0 = last.state.nodes.get(&NodeId(0)).unwrap();
        let cpu0 = b0.pressures.get(PressureKind::Cpu).unwrap().magnitude;
        assert!(
            (thermal_of(0) - cpu0).abs() > 1e-6,
            "thermal must carry independent signal from cpu"
        );
    }

    #[test]
    fn zero_thermal_intensity_events_produce_zero_thermal() {
        // Option-A synthetic traces leave thermal_intensity at 0.0 —
        // the documented pre-Option-B behaviour (no thermal signal)
        // must be preserved exactly.
        let cfg = MirAdapterConfig {
            n_blocks: 1,
            events_per_tick: 4,
            ..MirAdapterConfig::default()
        };
        let events: Vec<MirTraceEvent> = (0..8).map(|i| MirTraceEvent::minimal(i, 0)).collect();
        let out = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        let last = out.trajectory.iter().last().unwrap();
        let thermal = last
            .state
            .nodes
            .get(&NodeId(0))
            .unwrap()
            .pressures
            .get(PressureKind::Thermal)
            .unwrap()
            .magnitude;
        assert_eq!(thermal, 0.0);
    }

    #[test]
    fn adapter_rejects_event_with_unknown_block_id() {
        let cfg = MirAdapterConfig {
            n_blocks: 3,
            ..MirAdapterConfig::default()
        };
        let events = vec![MirTraceEvent {
            block_id: 99,
            ..MirTraceEvent::minimal(0, 99)
        }];
        let r = adapt_mir_trace_to_cluster_trajectory(&events, &cfg);
        assert!(r.is_err());
    }

    #[test]
    fn adapter_is_deterministic() {
        let cfg = MirAdapterConfig {
            n_blocks: 3,
            events_per_tick: 16,
            ..MirAdapterConfig::default()
        };
        let events = three_block_trace();
        let a = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        let b = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        assert_eq!(
            a.trajectory.canonical_bytes(),
            b.trajectory.canonical_bytes()
        );
    }

    #[test]
    fn output_can_drive_cluster_nss() {
        // Full end-to-end: adapter → cluster NSS prediction. The
        // architectural payoff of Phase 5a — no NSS changes needed.
        use crate::cluster_nss::{ClusterNeuralSystemsSimulator, ClusterNssConfig};
        use crate::seed::NssSeed;
        let cfg = MirAdapterConfig {
            n_blocks: 3,
            events_per_tick: 16,
            ..MirAdapterConfig::default()
        };
        let events = three_block_trace();
        let out = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let last = out.trajectory.last_state().unwrap();
        let pred = nss.predict_next(last).unwrap();
        assert!(pred.failure.collapse_probability.is_finite());
        assert!(
            pred.failure.collapse_probability >= 0.0 && pred.failure.collapse_probability <= 1.0
        );
    }

    #[test]
    fn initial_tick_offset_applied_correctly() {
        let cfg = MirAdapterConfig {
            n_blocks: 2,
            events_per_tick: 4,
            initial_tick: 100,
            ..MirAdapterConfig::default()
        };
        let events: Vec<_> = (0..8u64)
            .map(|i| MirTraceEvent::minimal(i, (i % 2) as u32))
            .collect();
        let out = adapt_mir_trace_to_cluster_trajectory(&events, &cfg).unwrap();
        // First produced tick should be initial_tick=100.
        assert_eq!(out.trajectory.iter().next().unwrap().state.tick, 100);
    }

    #[test]
    fn empty_event_list_produces_empty_trajectory() {
        let cfg = MirAdapterConfig {
            n_blocks: 2,
            ..MirAdapterConfig::default()
        };
        let out = adapt_mir_trace_to_cluster_trajectory(&[], &cfg).unwrap();
        assert_eq!(out.trajectory.len(), 0);
        assert_eq!(out.events_ingested, 0);
    }
}
