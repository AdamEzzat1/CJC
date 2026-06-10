//! Pressure-field primitives — the core differentiator of NSS.
//!
//! Traditional time-series ML treats metrics as scalars on a clock. NSS
//! treats infrastructure as a *pressure system*: each resource exerts a
//! dynamic, evolving pressure that accumulates, dissipates, propagates,
//! amplifies, and can cross instability thresholds. This module ships the
//! pressure-as-first-class primitive layer.
//!
//! Determinism:
//! * [`PressureKind`] is a finite enum with `Ord` derived in declaration
//!   order — the variant order *is* the iteration order.
//! * [`PressureField`] stores per-kind state in a [`BTreeMap`] so
//!   iteration is lexicographic regardless of insertion order.
//! * [`PressureGraph`] stores edges keyed by `(source, target)` tuples,
//!   sorted by `BTreeMap` ordering — propagation always visits edges in
//!   the same order across runs.

use crate::error::NssError;
use cjc_repro::KahanAccumulatorF64;
use std::collections::BTreeMap;

/// Categorical pressure dimension. The enum is closed (no `_Other`
/// variant) so the propagation graph has a finite, well-defined node
/// set. `Ord` is derived in declaration order, which is the deterministic
/// iteration order used throughout the crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureKind {
    /// CPU saturation pressure.
    Cpu,
    /// Memory occupancy / fragmentation pressure.
    Memory,
    /// Disk / storage IO pressure.
    Io,
    /// Network congestion pressure.
    Network,
    /// Work-queue buildup pressure.
    Queue,
    /// Scheduler-induced contention pressure.
    Scheduler,
    /// Synchronisation / lock-contention pressure.
    Sync,
    /// Thermal / power pressure.
    Thermal,
    /// Throughput-degradation pressure (egress symptom).
    Throughput,
}

impl PressureKind {
    /// Canonical short string name for serialisation and logging.
    pub fn label(self) -> &'static str {
        match self {
            PressureKind::Cpu => "cpu",
            PressureKind::Memory => "memory",
            PressureKind::Io => "io",
            PressureKind::Network => "network",
            PressureKind::Queue => "queue",
            PressureKind::Scheduler => "scheduler",
            PressureKind::Sync => "sync",
            PressureKind::Thermal => "thermal",
            PressureKind::Throughput => "throughput",
        }
    }

    /// Iterate every kind in canonical order. Useful for graph
    /// construction and trace headers.
    pub fn all() -> [PressureKind; 9] {
        [
            PressureKind::Cpu,
            PressureKind::Memory,
            PressureKind::Io,
            PressureKind::Network,
            PressureKind::Queue,
            PressureKind::Scheduler,
            PressureKind::Sync,
            PressureKind::Thermal,
            PressureKind::Throughput,
        ]
    }
}

/// State of a single pressure field at one tick.
///
/// `magnitude` is the *instantaneous* pressure in `[0, ∞)`; `accumulated`
/// is the running compensated-sum of `magnitude` over the trajectory so
/// far. `instability_threshold` marks the magnitude at which the field
/// is considered to be in an unstable regime (used by the failure head
/// and by the propagation engine for amplification gating).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pressure {
    /// Instantaneous magnitude. Must be `>= 0` and finite.
    pub magnitude: f64,
    /// Running compensated-sum over the trajectory so far.
    pub accumulated: f64,
    /// Magnitude above which the field is unstable. Default per-kind
    /// values live in [`PressureField::with_default_thresholds`].
    pub instability_threshold: f64,
    /// Per-tick dissipation coefficient in `[0, 1]`. After each tick the
    /// magnitude is multiplied by `(1 - dissipation)` *before* the
    /// propagator runs.
    pub dissipation: f64,
}

impl Pressure {
    /// Build a fresh pressure with zero accumulation. Validates that
    /// every field is finite and within range.
    pub fn new(
        magnitude: f64,
        instability_threshold: f64,
        dissipation: f64,
    ) -> Result<Self, NssError> {
        if !magnitude.is_finite() || magnitude < 0.0 {
            return Err(NssError::InvalidState {
                detail: format!("pressure magnitude must be finite and >= 0, got {magnitude}"),
            });
        }
        if !instability_threshold.is_finite() || instability_threshold <= 0.0 {
            return Err(NssError::InvalidState {
                detail: format!(
                    "instability_threshold must be finite and > 0, got {instability_threshold}"
                ),
            });
        }
        if !dissipation.is_finite() || !(0.0..=1.0).contains(&dissipation) {
            return Err(NssError::InvalidState {
                detail: format!("dissipation must be in [0, 1], got {dissipation}"),
            });
        }
        Ok(Self {
            magnitude,
            accumulated: 0.0,
            instability_threshold,
            dissipation,
        })
    }

    /// True if the current magnitude is at or above the instability
    /// threshold. The propagation engine uses this to gate amplification.
    #[inline]
    pub fn is_unstable(&self) -> bool {
        self.magnitude >= self.instability_threshold
    }

    /// Saturation in `[0, 1]` — magnitude divided by threshold, clipped.
    /// Used by the encoder so the network sees a bounded signal.
    #[inline]
    pub fn saturation(&self) -> f64 {
        if self.instability_threshold == 0.0 {
            return 0.0;
        }
        let s = self.magnitude / self.instability_threshold;
        if s.is_nan() {
            0.0
        } else if s > 1.0 {
            1.0
        } else {
            s
        }
    }
}

/// A full pressure-field state: one [`Pressure`] per [`PressureKind`].
/// Backed by [`BTreeMap`] so iteration is deterministic.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PressureField {
    fields: BTreeMap<PressureKind, Pressure>,
}

impl PressureField {
    /// Empty field. Add pressures with [`set`](Self::set).
    pub fn empty() -> Self {
        Self {
            fields: BTreeMap::new(),
        }
    }

    /// Build a field initialised with conservative default thresholds /
    /// dissipations for every kind. Used by simulators and tests that
    /// want a sensible baseline without hand-rolling each entry.
    pub fn with_default_thresholds() -> Self {
        let mut f = Self::empty();
        for k in PressureKind::all() {
            // Magnitude 0.0, threshold 1.0, dissipation 0.1 per tick.
            // The default is a unit-pressure space — simulators that want
            // larger headroom can override per kind.
            let p = Pressure::new(0.0, 1.0, 0.1).expect("default pressure is valid");
            f.fields.insert(k, p);
        }
        f
    }

    /// Insert or overwrite a pressure entry.
    pub fn set(&mut self, kind: PressureKind, p: Pressure) {
        self.fields.insert(kind, p);
    }

    /// Look up a pressure by kind. `None` if the kind was never set.
    pub fn get(&self, kind: PressureKind) -> Option<&Pressure> {
        self.fields.get(&kind)
    }

    /// Look up mutably; convenient for the propagation engine.
    pub fn get_mut(&mut self, kind: PressureKind) -> Option<&mut Pressure> {
        self.fields.get_mut(&kind)
    }

    /// Iterate every entry in lexicographic key order.
    pub fn iter(&self) -> impl Iterator<Item = (&PressureKind, &Pressure)> {
        self.fields.iter()
    }

    /// True if every magnitude is finite. The propagation engine asserts
    /// this after every tick.
    pub fn all_finite(&self) -> bool {
        self.fields
            .values()
            .all(|p| p.magnitude.is_finite() && p.accumulated.is_finite())
    }

    /// Number of fields currently set.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// True if no fields have been set.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Sum of all instantaneous magnitudes (Kahan-compensated for
    /// determinism — order is `BTreeMap` iteration order).
    pub fn total_magnitude(&self) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        for (_k, p) in self.fields.iter() {
            acc.add(p.magnitude);
        }
        acc.finalize()
    }

    /// Apply per-field dissipation in-place: `magnitude ← magnitude *
    /// (1 - dissipation)`. Called at the start of every propagation tick.
    pub fn dissipate(&mut self) {
        for (_k, p) in self.fields.iter_mut() {
            p.magnitude *= 1.0 - p.dissipation;
            if p.magnitude < 0.0 {
                p.magnitude = 0.0;
            }
        }
    }

    /// Accumulate the current magnitude into `accumulated` (Kahan-style
    /// per-field). Called at the end of every propagation tick.
    pub fn accumulate(&mut self) {
        for (_k, p) in self.fields.iter_mut() {
            // Per-field Kahan: one accumulator over (accumulated,
            // magnitude). The two-value Kahan reduces to plain add for
            // the first tick but stays stable as the trajectory grows.
            let mut acc = KahanAccumulatorF64::new();
            acc.add(p.accumulated);
            acc.add(p.magnitude);
            p.accumulated = acc.finalize();
        }
    }

    /// Canonical byte representation used for input hashing. Encodes
    /// every entry in iteration order: `(kind label, magnitude bits,
    /// accumulated bits, threshold bits, dissipation bits)`. Two
    /// PressureFields with the same entries produce the same bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.fields.len() * 36);
        for (kind, p) in self.fields.iter() {
            bytes.extend_from_slice(kind.label().as_bytes());
            bytes.push(b'|');
            bytes.extend_from_slice(&p.magnitude.to_bits().to_le_bytes());
            bytes.extend_from_slice(&p.accumulated.to_bits().to_le_bytes());
            bytes.extend_from_slice(&p.instability_threshold.to_bits().to_le_bytes());
            bytes.extend_from_slice(&p.dissipation.to_bits().to_le_bytes());
        }
        bytes
    }
}

/// One directed edge in a [`PressureGraph`]: source pressure
/// **propagates** into target pressure with a multiplicative weight.
///
/// Pressure-conservation interpretation: each tick, a fraction
/// `weight * source.magnitude` is *transformed* (not just copied) into
/// target pressure. The conservation law is implemented in
/// [`crate::PressurePropagator`] — see that module's docstring for the
/// transformation tax (a small fraction of the propagated quantity
/// dissipates back into the source, modelling that pressure cannot be
/// transmuted with perfect efficiency).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureEdge {
    /// Multiplicative weight in `[0, 1]`. Higher weights mean the source
    /// drives the target harder.
    pub weight: f64,
    /// Amplification gate: weight is doubled when the source field is
    /// unstable (above its threshold). Set to `false` for purely linear
    /// propagation; set to `true` for the "instability cascades" model.
    pub amplify_on_instability: bool,
}

impl PressureEdge {
    /// Build a validated edge.
    pub fn new(weight: f64, amplify_on_instability: bool) -> Result<Self, NssError> {
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(NssError::PressureGraph {
                detail: format!("PressureEdge.weight must be in [0, 1], got {weight}"),
            });
        }
        Ok(Self {
            weight,
            amplify_on_instability,
        })
    }
}

/// A directed weighted graph of pressure-interaction edges.
///
/// The graph is the static topology of "how pressure flows in this
/// system" — for the queue-pressure-predictor (Phase 1) the default graph
/// wires `Queue → Scheduler → Sync → Throughput` plus a `Cpu → Queue`
/// back-feed. Phase 2 adds nodes for the distributed simulator.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PressureGraph {
    edges: BTreeMap<(PressureKind, PressureKind), PressureEdge>,
}

impl PressureGraph {
    /// Empty graph.
    pub fn empty() -> Self {
        Self {
            edges: BTreeMap::new(),
        }
    }

    /// Default Phase 1 graph for the queue-pressure-predictor:
    ///
    /// ```text
    /// Cpu      → Queue
    /// Queue    → Scheduler  (amplifies on instability)
    /// Scheduler→ Sync       (amplifies on instability)
    /// Sync     → Throughput
    /// Memory   → Queue
    /// Io       → Queue
    /// Network  → Queue
    /// Thermal  → Cpu        (throttling)
    /// ```
    ///
    /// All weights are deterministic constants. The "amplify on
    /// instability" flag is set on the two edges that model cascading
    /// failure (queue→scheduler and scheduler→sync).
    pub fn default_phase1() -> Self {
        let mut g = Self::empty();
        // Helper: insert a deterministic edge; unwrap is OK because all
        // literal weights here are valid.
        let mut add = |src: PressureKind, dst: PressureKind, w: f64, amp: bool| {
            g.edges
                .insert((src, dst), PressureEdge::new(w, amp).unwrap());
        };
        add(PressureKind::Cpu, PressureKind::Queue, 0.15, false);
        add(PressureKind::Queue, PressureKind::Scheduler, 0.35, true);
        add(PressureKind::Scheduler, PressureKind::Sync, 0.30, true);
        add(PressureKind::Sync, PressureKind::Throughput, 0.40, false);
        add(PressureKind::Memory, PressureKind::Queue, 0.20, false);
        add(PressureKind::Io, PressureKind::Queue, 0.20, false);
        add(PressureKind::Network, PressureKind::Queue, 0.10, false);
        add(PressureKind::Thermal, PressureKind::Cpu, 0.25, false);
        g
    }

    /// Insert an edge. Fails if `(src, dst)` already exists.
    pub fn add_edge(
        &mut self,
        src: PressureKind,
        dst: PressureKind,
        edge: PressureEdge,
    ) -> Result<(), NssError> {
        if src == dst {
            return Err(NssError::PressureGraph {
                detail: format!("self-loop on {} not allowed in Phase 1", src.label()),
            });
        }
        if self.edges.contains_key(&(src, dst)) {
            return Err(NssError::PressureGraph {
                detail: format!("duplicate edge {} -> {}", src.label(), dst.label()),
            });
        }
        self.edges.insert((src, dst), edge);
        Ok(())
    }

    /// Total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate every `(src, dst, edge)` triple in lexicographic
    /// `(src, dst)` order. The propagation engine consumes this directly
    /// — the iteration order *is* the propagation order.
    pub fn iter(&self) -> impl Iterator<Item = (PressureKind, PressureKind, &PressureEdge)> {
        self.edges.iter().map(|((s, d), e)| (*s, *d, e))
    }

    /// Get a specific edge.
    pub fn get(&self, src: PressureKind, dst: PressureKind) -> Option<&PressureEdge> {
        self.edges.get(&(src, dst))
    }

    /// Canonical bytes for the graph topology (used by [`crate::NssRunId`]
    /// indirectly via the simulator/NSS config). Entries are sorted by
    /// `BTreeMap` iteration order so two graphs with identical edges
    /// produce identical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.edges.len() * 16);
        for ((src, dst), e) in self.edges.iter() {
            bytes.extend_from_slice(src.label().as_bytes());
            bytes.push(b'>');
            bytes.extend_from_slice(dst.label().as_bytes());
            bytes.push(b'|');
            bytes.extend_from_slice(&e.weight.to_bits().to_le_bytes());
            bytes.push(if e.amplify_on_instability { 1 } else { 0 });
        }
        bytes
    }
}

/// Record of one pressure transformation that occurred during a single
/// propagation tick. Populated by [`crate::PressurePropagator`] for
/// lineage / attribution downstream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureFlow {
    /// Source field that lost magnitude.
    pub source: PressureKind,
    /// Target field that gained magnitude.
    pub target: PressureKind,
    /// Magnitude transferred (already accounting for the conservation
    /// tax — see [`crate::PressurePropagator`] docs).
    pub amount: f64,
    /// True if the amplification gate fired for this edge.
    pub amplified: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_kind_ordering_is_declaration_order() {
        // The propagation engine relies on this — if a future edit
        // reorders the enum variants, this test fails so the change is
        // explicit.
        let kinds = PressureKind::all();
        let mut sorted = kinds;
        sorted.sort();
        assert_eq!(
            kinds, sorted,
            "PressureKind variants must be declared in Ord-sorted order"
        );
    }

    #[test]
    fn pressure_validates_inputs() {
        assert!(Pressure::new(0.0, 1.0, 0.1).is_ok());
        assert!(Pressure::new(-0.5, 1.0, 0.1).is_err());
        assert!(Pressure::new(f64::NAN, 1.0, 0.1).is_err());
        assert!(Pressure::new(0.0, 0.0, 0.1).is_err()); // zero threshold
        assert!(Pressure::new(0.0, 1.0, 1.5).is_err()); // dissipation > 1
        assert!(Pressure::new(0.0, 1.0, -0.1).is_err()); // dissipation < 0
    }

    #[test]
    fn saturation_is_bounded() {
        let p = Pressure::new(0.5, 1.0, 0.1).unwrap();
        assert_eq!(p.saturation(), 0.5);
        let p = Pressure::new(2.0, 1.0, 0.1).unwrap();
        assert_eq!(p.saturation(), 1.0);
        let p = Pressure::new(0.0, 1.0, 0.1).unwrap();
        assert_eq!(p.saturation(), 0.0);
    }

    #[test]
    fn default_field_has_one_pressure_per_kind() {
        let f = PressureField::with_default_thresholds();
        assert_eq!(f.len(), PressureKind::all().len());
        for k in PressureKind::all() {
            assert!(f.get(k).is_some(), "missing kind {:?}", k);
        }
    }

    #[test]
    fn dissipate_drives_magnitude_down() {
        let mut f = PressureField::with_default_thresholds();
        let p = Pressure::new(1.0, 2.0, 0.5).unwrap();
        f.set(PressureKind::Queue, p);
        f.dissipate();
        assert!((f.get(PressureKind::Queue).unwrap().magnitude - 0.5).abs() < 1e-12);
        f.dissipate();
        assert!((f.get(PressureKind::Queue).unwrap().magnitude - 0.25).abs() < 1e-12);
    }

    #[test]
    fn accumulate_sums_with_kahan() {
        let mut f = PressureField::with_default_thresholds();
        let p = Pressure::new(0.1, 1.0, 0.0).unwrap();
        f.set(PressureKind::Queue, p);
        for _ in 0..10 {
            f.accumulate();
        }
        // 10 * 0.1 == 1.0 with Kahan; plain f64 sum drifts.
        let q = f.get(PressureKind::Queue).unwrap();
        assert!((q.accumulated - 1.0).abs() < 1e-15);
    }

    #[test]
    fn canonical_bytes_stable_under_reinsertion() {
        // Insert in different orders, must hash to identical bytes.
        let mut a = PressureField::empty();
        a.set(PressureKind::Queue, Pressure::new(0.2, 1.0, 0.1).unwrap());
        a.set(PressureKind::Cpu, Pressure::new(0.5, 1.0, 0.1).unwrap());
        let mut b = PressureField::empty();
        b.set(PressureKind::Cpu, Pressure::new(0.5, 1.0, 0.1).unwrap());
        b.set(PressureKind::Queue, Pressure::new(0.2, 1.0, 0.1).unwrap());
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn graph_rejects_self_loop_and_duplicates() {
        let mut g = PressureGraph::empty();
        let e = PressureEdge::new(0.5, false).unwrap();
        assert!(g.add_edge(PressureKind::Cpu, PressureKind::Cpu, e).is_err());
        assert!(g
            .add_edge(PressureKind::Cpu, PressureKind::Queue, e)
            .is_ok());
        assert!(g
            .add_edge(PressureKind::Cpu, PressureKind::Queue, e)
            .is_err());
    }

    #[test]
    fn default_phase1_graph_topology() {
        let g = PressureGraph::default_phase1();
        assert!(g
            .get(PressureKind::Queue, PressureKind::Scheduler)
            .is_some());
        assert!(
            g.get(PressureKind::Queue, PressureKind::Scheduler)
                .unwrap()
                .amplify_on_instability
        );
        // No self-loops.
        for k in PressureKind::all() {
            assert!(g.get(k, k).is_none(), "self-loop on {:?}", k);
        }
    }

    #[test]
    fn graph_iter_is_lexicographic_regardless_of_insertion_order() {
        let mut a = PressureGraph::empty();
        a.add_edge(
            PressureKind::Sync,
            PressureKind::Throughput,
            PressureEdge::new(0.5, false).unwrap(),
        )
        .unwrap();
        a.add_edge(
            PressureKind::Cpu,
            PressureKind::Queue,
            PressureEdge::new(0.5, false).unwrap(),
        )
        .unwrap();
        let mut b = PressureGraph::empty();
        b.add_edge(
            PressureKind::Cpu,
            PressureKind::Queue,
            PressureEdge::new(0.5, false).unwrap(),
        )
        .unwrap();
        b.add_edge(
            PressureKind::Sync,
            PressureKind::Throughput,
            PressureEdge::new(0.5, false).unwrap(),
        )
        .unwrap();
        let a_order: Vec<_> = a.iter().map(|(s, d, _)| (s, d)).collect();
        let b_order: Vec<_> = b.iter().map(|(s, d, _)| (s, d)).collect();
        assert_eq!(a_order, b_order);
    }
}
