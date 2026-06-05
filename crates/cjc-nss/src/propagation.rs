//! Pressure propagation engine — applies the conservation law to a
//! [`PressureField`] one tick at a time.
//!
//! ## The conservation law
//!
//! Each tick, every edge `(src → dst)` in the [`PressureGraph`] transfers
//! a quantity:
//!
//! ```text
//! gross   = weight · src.magnitude · (amplification_factor if src is unstable else 1)
//! net     = gross · transfer_efficiency
//! tax     = gross · (1 - transfer_efficiency)
//! ```
//!
//! - `net` is added to `dst.magnitude` (pressure flows into the target).
//! - `gross` is subtracted from `src.magnitude` (the source loses what it
//!   gave up).
//! - `tax` is *lost* — pressure does not perfectly transmute. This is
//!   what stops amplified-edge feedback loops from blowing up.
//!
//! ## Determinism contract
//!
//! - The propagator iterates edges in `BTreeMap` order (lexicographic on
//!   `(source, target)`), so two runs with the same input field and
//!   graph execute the same sequence of transfers in the same order.
//! - All accumulators are [`cjc_repro::KahanAccumulatorF64`].
//! - No allocations dependent on iteration order — the returned
//!   `Vec<PressureFlow>` is appended in iteration order.
//! - Per-tick state (`dissipate` → `propagate` → `accumulate`) is a
//!   fixed three-step routine; no implicit ordering.
//!
//! ## Stability
//!
//! Stability is **structural, not empirical**. Given
//! `transfer_efficiency ≤ 1` and `weight ∈ [0, 1]` and
//! `amplification_factor ≤ 2`, the worst-case per-tick magnitude growth
//! at any node is bounded by `Σ_in (weight * amplification * efficiency)`
//! plus the node's own retained fraction `(1 - Σ_out weight * amp)`.
//! As long as the row-sum invariant holds (each source's total outgoing
//! `weight * amp` ≤ 1), the field cannot blow up to infinity. The
//! default Phase-1 graph satisfies this by inspection.

use crate::error::NssError;
use crate::pressure::{PressureField, PressureFlow, PressureGraph, PressureKind};
use cjc_repro::KahanAccumulatorF64;

/// Knobs for the propagation engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropagationConfig {
    /// Fraction of gross flow that actually crosses the edge. Must be in
    /// `(0, 1]`. Default `0.85` — 15% "tax" per transfer.
    pub transfer_efficiency: f64,
    /// Multiplier on the gross flow when the source field is at or above
    /// its instability threshold *and* the edge has
    /// `amplify_on_instability=true`. Must be in `[1, 2]`. Default `1.5`.
    pub amplification_factor: f64,
    /// Cap on per-tick magnitude growth — after the tick, every
    /// field's magnitude is clipped to `magnitude_cap` if it exceeds.
    /// Provides a defence-in-depth ceiling above the structural bound.
    /// Must be finite and > 0. Default `1e6`.
    pub magnitude_cap: f64,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            transfer_efficiency: 0.85,
            amplification_factor: 1.5,
            magnitude_cap: 1e6,
        }
    }
}

impl PropagationConfig {
    /// Validate. Used by the simulator and NSS configs.
    pub fn validate(&self) -> Result<(), NssError> {
        if !self.transfer_efficiency.is_finite()
            || self.transfer_efficiency <= 0.0
            || self.transfer_efficiency > 1.0
        {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "transfer_efficiency must be in (0, 1], got {}",
                    self.transfer_efficiency
                ),
            });
        }
        if !self.amplification_factor.is_finite()
            || self.amplification_factor < 1.0
            || self.amplification_factor > 2.0
        {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "amplification_factor must be in [1, 2], got {}",
                    self.amplification_factor
                ),
            });
        }
        if !self.magnitude_cap.is_finite() || self.magnitude_cap <= 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("magnitude_cap must be > 0 and finite, got {}", self.magnitude_cap),
            });
        }
        Ok(())
    }

    /// Canonical bytes — included in [`crate::NssRunId`] indirectly.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&self.transfer_efficiency.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.amplification_factor.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.magnitude_cap.to_bits().to_le_bytes());
        bytes
    }
}

/// Stateless propagation engine. Holds the graph + config; the field is
/// passed in per tick.
#[derive(Clone, Debug, PartialEq)]
pub struct PressurePropagator {
    graph: PressureGraph,
    cfg: PropagationConfig,
}

impl PressurePropagator {
    /// Build a propagator. Validates the config.
    pub fn new(graph: PressureGraph, cfg: PropagationConfig) -> Result<Self, NssError> {
        cfg.validate()?;
        Ok(Self { graph, cfg })
    }

    /// Borrow the graph (read-only).
    pub fn graph(&self) -> &PressureGraph {
        &self.graph
    }

    /// Borrow the config.
    pub fn config(&self) -> &PropagationConfig {
        &self.cfg
    }

    /// One propagation tick:
    ///
    /// 1. dissipate every field by its per-kind dissipation,
    /// 2. compute gross / net / tax for every edge in deterministic
    ///    order, recording flows,
    /// 3. apply net deltas to source and target,
    /// 4. clip every magnitude to `[0, magnitude_cap]`,
    /// 5. call `field.accumulate()` for the post-propagation magnitudes.
    ///
    /// Returns the list of [`PressureFlow`] records in iteration order —
    /// used by the causal attribution head.
    pub fn step(&self, field: &mut PressureField) -> Result<Vec<PressureFlow>, NssError> {
        field.dissipate();

        // Compute all flows in deterministic edge order *before*
        // applying them. This decouples the order in which we
        // *compute* magnitudes from the order in which we *write* them,
        // so an amplified edge doesn't see partially-updated state from
        // earlier edges in the same tick.
        let mut flows: Vec<PressureFlow> = Vec::with_capacity(self.graph.edge_count());

        // Snapshot per-source magnitudes once at tick-start; gross flow
        // is always computed against the snapshot, never against a
        // mid-tick magnitude.
        let mut snapshot_magnitudes: std::collections::BTreeMap<PressureKind, f64> =
            std::collections::BTreeMap::new();
        let mut snapshot_unstable: std::collections::BTreeMap<PressureKind, bool> =
            std::collections::BTreeMap::new();
        for (kind, p) in field.iter() {
            snapshot_magnitudes.insert(*kind, p.magnitude);
            snapshot_unstable.insert(*kind, p.is_unstable());
        }

        for (src, dst, edge) in self.graph.iter() {
            let src_mag = *snapshot_magnitudes.get(&src).unwrap_or(&0.0);
            if src_mag <= 0.0 {
                continue;
            }
            let amplified = edge.amplify_on_instability
                && *snapshot_unstable.get(&src).unwrap_or(&false);
            let amp = if amplified { self.cfg.amplification_factor } else { 1.0 };
            let gross = edge.weight * src_mag * amp;
            let net = gross * self.cfg.transfer_efficiency;
            flows.push(PressureFlow { source: src, target: dst, amount: net, amplified });
        }

        // Per-source: sum gross outflow; per-target: sum net inflow.
        // Two passes so each is Kahan-compensated in deterministic
        // BTreeMap key order.
        let mut gross_out: std::collections::BTreeMap<PressureKind, KahanAccumulatorF64> =
            std::collections::BTreeMap::new();
        let mut net_in: std::collections::BTreeMap<PressureKind, KahanAccumulatorF64> =
            std::collections::BTreeMap::new();

        for (src, dst, edge) in self.graph.iter() {
            let src_mag = *snapshot_magnitudes.get(&src).unwrap_or(&0.0);
            if src_mag <= 0.0 {
                continue;
            }
            let amplified = edge.amplify_on_instability
                && *snapshot_unstable.get(&src).unwrap_or(&false);
            let amp = if amplified { self.cfg.amplification_factor } else { 1.0 };
            let gross = edge.weight * src_mag * amp;
            let net = gross * self.cfg.transfer_efficiency;
            gross_out
                .entry(src)
                .or_insert_with(KahanAccumulatorF64::new)
                .add(gross);
            net_in
                .entry(dst)
                .or_insert_with(KahanAccumulatorF64::new)
                .add(net);
        }

        // Apply: subtract gross from sources, add net to targets, clip.
        // BTreeMap iteration is deterministic.
        for (kind, acc) in gross_out.iter() {
            if let Some(p) = field.get_mut(*kind) {
                p.magnitude -= acc.finalize();
                // A source can never go below 0; we cap it explicitly
                // because Kahan + per-tick dissipation may produce a
                // tiny negative drift on a fully-quiescent field.
                if p.magnitude < 0.0 {
                    p.magnitude = 0.0;
                }
            }
        }
        for (kind, acc) in net_in.iter() {
            if let Some(p) = field.get_mut(*kind) {
                p.magnitude += acc.finalize();
            }
        }

        // Defence-in-depth: non-finite guard (errors with a named
        // field) followed by a magnitude clip pass. We collect the
        // kinds first so we can take `&mut Pressure` per entry without
        // aliasing the iterator.
        let kinds: Vec<PressureKind> = field.iter().map(|(k, _)| *k).collect();
        for kind in &kinds {
            if let Some(p) = field.get(*kind) {
                if !p.magnitude.is_finite() {
                    return Err(NssError::NonFinitePressure {
                        field: kind.label().to_string(),
                    });
                }
            }
        }
        for kind in &kinds {
            if let Some(p) = field.get_mut(*kind) {
                if p.magnitude > self.cfg.magnitude_cap {
                    p.magnitude = self.cfg.magnitude_cap;
                }
            }
        }

        field.accumulate();
        Ok(flows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::{Pressure, PressureEdge};

    fn small_graph() -> PressureGraph {
        let mut g = PressureGraph::empty();
        g.add_edge(
            PressureKind::Queue,
            PressureKind::Scheduler,
            PressureEdge::new(0.5, true).unwrap(),
        )
        .unwrap();
        g.add_edge(
            PressureKind::Scheduler,
            PressureKind::Sync,
            PressureEdge::new(0.5, false).unwrap(),
        )
        .unwrap();
        g
    }

    #[test]
    fn propagation_config_validates() {
        assert!(PropagationConfig::default().validate().is_ok());
        let mut bad = PropagationConfig::default();
        bad.transfer_efficiency = 1.5;
        assert!(bad.validate().is_err());
        let mut bad = PropagationConfig::default();
        bad.amplification_factor = 0.5;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn empty_field_propagates_to_zero_flows() {
        let prop = PressurePropagator::new(small_graph(), PropagationConfig::default()).unwrap();
        let mut f = PressureField::with_default_thresholds();
        let flows = prop.step(&mut f).unwrap();
        assert!(flows.is_empty(), "expected no flows for empty field, got {:?}", flows);
    }

    #[test]
    fn single_source_flows_to_target() {
        let prop = PressurePropagator::new(small_graph(), PropagationConfig::default()).unwrap();
        let mut f = PressureField::with_default_thresholds();
        // Override Queue with non-dissipating, sub-threshold pressure
        // so the test isolates the propagation effect.
        f.set(
            PressureKind::Queue,
            Pressure::new(0.5, 10.0, 0.0).unwrap(),
        );
        let flows = prop.step(&mut f).unwrap();
        // Queue → Scheduler should fire; Scheduler had no pressure so
        // Scheduler → Sync starts from zero and does not fire.
        assert!(!flows.is_empty(), "expected at least the Queue->Scheduler flow");
        let queue_to_scheduler = flows
            .iter()
            .find(|fl| fl.source == PressureKind::Queue && fl.target == PressureKind::Scheduler);
        assert!(queue_to_scheduler.is_some());
        let f1 = queue_to_scheduler.unwrap();
        // gross = 0.5 * 0.5 * 1.0 = 0.25; net = 0.25 * 0.85 = 0.2125
        assert!((f1.amount - 0.2125).abs() < 1e-12, "unexpected net flow {}", f1.amount);
        assert!(!f1.amplified, "should not amplify below threshold");
    }

    #[test]
    fn amplification_fires_above_threshold() {
        let prop = PressurePropagator::new(small_graph(), PropagationConfig::default()).unwrap();
        let mut f = PressureField::with_default_thresholds();
        f.set(
            PressureKind::Queue,
            Pressure::new(2.0, 1.0, 0.0).unwrap(),
        );
        let flows = prop.step(&mut f).unwrap();
        let amp_flow = flows
            .iter()
            .find(|fl| fl.source == PressureKind::Queue && fl.target == PressureKind::Scheduler)
            .expect("missing Queue->Scheduler flow");
        assert!(amp_flow.amplified, "should amplify above threshold");
        // gross = 0.5 * 2.0 * 1.5 = 1.5; net = 1.5 * 0.85 = 1.275
        assert!((amp_flow.amount - 1.275).abs() < 1e-12, "got {}", amp_flow.amount);
    }

    #[test]
    fn conservation_tax_reduces_total_magnitude() {
        let prop = PressurePropagator::new(small_graph(), PropagationConfig::default()).unwrap();
        let mut f = PressureField::with_default_thresholds();
        // Zero dissipation everywhere so the only loss is the tax.
        for k in PressureKind::all() {
            f.set(k, Pressure::new(0.0, 10.0, 0.0).unwrap());
        }
        f.set(PressureKind::Queue, Pressure::new(1.0, 10.0, 0.0).unwrap());
        let before = f.total_magnitude();
        prop.step(&mut f).unwrap();
        let after = f.total_magnitude();
        assert!(after < before, "tax must strictly reduce total magnitude (before={}, after={})", before, after);
    }

    #[test]
    fn propagation_preserves_finiteness_under_runaway_input() {
        let prop = PressurePropagator::new(
            PressureGraph::default_phase1(),
            PropagationConfig::default(),
        )
        .unwrap();
        let mut f = PressureField::with_default_thresholds();
        // Hammer every field with above-threshold pressure for 256 ticks.
        for k in PressureKind::all() {
            f.set(k, Pressure::new(5.0, 1.0, 0.05).unwrap());
        }
        for _ in 0..256 {
            prop.step(&mut f).unwrap();
            assert!(f.all_finite(), "field went non-finite under runaway input");
        }
    }

    #[test]
    fn determinism_two_runs_match() {
        let prop = PressurePropagator::new(
            PressureGraph::default_phase1(),
            PropagationConfig::default(),
        )
        .unwrap();
        let mut f1 = PressureField::with_default_thresholds();
        let mut f2 = PressureField::with_default_thresholds();
        f1.set(PressureKind::Queue, Pressure::new(0.3, 1.0, 0.05).unwrap());
        f2.set(PressureKind::Queue, Pressure::new(0.3, 1.0, 0.05).unwrap());
        for _ in 0..16 {
            let a = prop.step(&mut f1).unwrap();
            let b = prop.step(&mut f2).unwrap();
            assert_eq!(a, b);
        }
        assert_eq!(f1.canonical_bytes(), f2.canonical_bytes());
    }
}
