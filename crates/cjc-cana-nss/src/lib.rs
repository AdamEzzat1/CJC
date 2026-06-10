//! # `cjc-cana-nss` — CANA ↔ NSS bridge crate
//!
//! Implements [`cjc_cana::pressure::PressurePredictor`] backed by the
//! Neural Systems Simulator (`cjc-nss`). This is the §4B.2 deliverable
//! from `docs/cana/HANDOFF_NEXT_SESSION.md`.
//!
//! ## Mode: Option A (synthetic-trace projection)
//!
//! The design-options document
//! ([`CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md`](../../../docs/cana/CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md))
//! enumerates four implementations (A: synthetic trace, B: real MIR-exec
//! instrumentation, C: structural-only, D: hybrid). This crate now
//! ships **Option A** (was Option C until §19 + §20).
//!
//! Option A turns CANA's static program features into per-function
//! pressure predictions by:
//!
//! 1. Mapping each [`MirFunction`](cjc_mir::MirFunction) to a NSS
//!    [`NodeId`](cjc_nss::NodeId) (one per function, deterministic by
//!    function-name iteration order via the BTreeMap-keyed
//!    [`CanaFeatures::per_fn`](cjc_cana::features::CanaFeatures::per_fn)).
//! 2. Synthesizing a [`Vec<MirTraceEvent>`](cjc_nss::MirTraceEvent) from
//!    [`FnFeatures`](cjc_cana::features::FnFeatures) — one event per
//!    estimated basic block per function. Each event encodes the
//!    function's CFG + memory profile into the trace fields
//!    (`register_pressure`, `heap_bytes_in_use`, `call_depth`,
//!    `branch_taken`).
//! 3. Feeding the synthetic events through
//!    [`adapt_mir_trace_to_cluster_trajectory`](cjc_nss::adapt_mir_trace_to_cluster_trajectory)
//!    which aggregates them into a [`ClusterTrajectory`](cjc_nss::ClusterTrajectory).
//! 4. Reading the last tick's per-node [`PressureField`](cjc_nss::PressureField)
//!    and extracting the requested [`PressureKind`](cjc_nss::PressureKind)
//!    magnitude per function.
//!
//! The result is a deterministic [`BTreeMap<String, f64>`] keyed by
//! function name with values in `[0, 1]`. Same `(program, features,
//! seed)` always produces the same map.
//!
//! ### Why this is "Option A" and not "Option B"
//!
//! Option B would replace step 2's *synthesized* events with *real*
//! `MirTraceEvent` streams collected from an instrumented MIR-exec run.
//! That's ~200-400 LOC of instrumentation in `cjc-mir-exec` (per
//! `docs/nss/HANDOFF_PHASE_5_COMPILER_INTEGRATION.md` §6.1) that hasn't
//! landed. Option A's heuristic synthesis captures the static-feature
//! signal at compile-time cost; Option B will eventually give NSS real
//! runtime signal at the cost of executor instrumentation.
//!
//! ### What "activates" with Option A vs what's still queued
//!
//! After Option A, `predict_thermal/memory/cpu` return non-empty maps.
//! [`ThermalAwareCostModel`](cjc_cana::thermal_cost_model::ThermalAwareCostModel)
//! reads them via `.get(name).copied().unwrap_or(0.0)` and applies
//! [`THERMAL_PENALTY_FACTOR`](cjc_cana::thermal_cost_model::THERMAL_PENALTY_FACTOR)
//! when both:
//!
//! - The pass is in
//!   [`THERMALLY_AGGRESSIVE_PASSES`](cjc_cana::thermal_cost_model::THERMALLY_AGGRESSIVE_PASSES)
//!   = `["loop_unroll", "vectorize", "specialize", "monomorphize"]`.
//! - The function's predicted thermal pressure exceeds the threshold
//!   (default `0.80`).
//!
//! **None of the 5 currently-trainable passes** (CF, SR, DCE, CSE,
//! LICM) are in `THERMALLY_AGGRESSIVE_PASSES`. So Option A activates
//! the *infrastructure* (predictions flow, audit captures them, the
//! 5-layer chain is end-to-end live) without changing PassPlans for the
//! current pass set. When future passes like `loop_unroll` land in
//! `CANONICAL_PASSES`, they automatically benefit from thermal
//! awareness — the wiring is in place.
//!
//! ## Determinism contract
//!
//! - Synthesis is a pure function of `(program, features, seed)`.
//! - All maps and iteration use `BTreeMap`, never `HashMap`.
//! - The seed is stored for future randomized synthesis variants (not
//!   used in current synthesis).
//! - Output sort order is lexicographic (BTreeMap key order).

#![warn(missing_docs)]

pub mod pinn_bridge;

pub use pinn_bridge::physical_estimate_to_pressure_deltas;

use std::collections::BTreeMap;

use cjc_cana::features::{CanaFeatures, FnFeatures};
use cjc_cana::pressure::PressurePredictor;
use cjc_mir::MirProgram;
use cjc_nss::{
    adapt_mir_trace_to_cluster_trajectory, MirAdapterConfig, MirTraceEvent, NodeId, NssSeed,
    PressureKind,
};

/// Threshold for `identify_structural_hot_kernels`: a function is
/// surfaced if `max_loop_depth ≥ HOT_LOOP_DEPTH_MIN` AND
/// `branch_count ≥ HOT_BRANCH_COUNT_MIN`.
///
/// Calibrated against the §3A.4 feature distribution audit (corpus
/// q75 of `max_loop_depth` is 1, max is 4). Threshold of 2 surfaces
/// the top quartile of structurally-deep functions while excluding the
/// straight-line-arithmetic majority.
pub const HOT_LOOP_DEPTH_MIN: u32 = 2;

/// Threshold for `identify_structural_hot_kernels`: see
/// [`HOT_LOOP_DEPTH_MIN`]. q75 of `branch_count` in the §3A.4 audit was
/// also 1, max 4. Threshold of 2 produces the AND-conjunction with
/// loop depth that screens out simple loops with no internal branching.
pub const HOT_BRANCH_COUNT_MIN: u32 = 2;

/// An NSS-backed `PressurePredictor` for CANA's compile-time
/// recommendation engine.
///
/// See crate-level docs for the Option-C design choice. Use
/// [`NssPressurePredictor::from_seed`] to construct.
#[derive(Debug, Clone)]
pub struct NssPressurePredictor {
    /// Deterministic seed for future trace-synthesis or sampling needs.
    /// Currently unused; reserved per the forward-compatibility contract
    /// documented at the crate level.
    seed: NssSeed,
}

impl NssPressurePredictor {
    /// Construct an Option-C predictor with a given deterministic seed.
    ///
    /// The seed is currently unused (Option C does no randomness; all
    /// identification is structural). It exists so the constructor
    /// signature is stable across the future Option-A and Option-B
    /// migrations — those will need an RNG substream salt.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            seed: NssSeed(seed),
        }
    }

    /// Return the seed this predictor was constructed with. Exposed for
    /// audit + test purposes.
    pub fn seed(&self) -> NssSeed {
        self.seed
    }
}

impl Default for NssPressurePredictor {
    /// Default seed = 42. Matches the cjc-mir-exec convention for
    /// reproducible runs.
    fn default() -> Self {
        Self::from_seed(42)
    }
}

impl NssPressurePredictor {
    /// Synthesize a `Vec<MirTraceEvent>` from CANA's static per-function
    /// features.
    ///
    /// One event per estimated basic block per function. Block count is
    /// `max(expr_count / 8, 1)`. The `block_id` is the function's NodeId
    /// (1:1 mapping). Field heuristics:
    ///
    /// - `register_pressure`: `expr_count / 256`, capped at 1.0. Large
    ///   functions tend to have more live SSA values.
    /// - `heap_bytes_in_use`: `alloc_sites × 4 KiB` — a single
    ///   allocation-site count carries no size info, so we estimate a
    ///   page-size allocation per site.
    /// - `call_depth`: `max_loop_depth + 1`. Loop nesting is the closest
    ///   static proxy for call-stack depth (real depth needs a trace).
    /// - `branch_taken`: alternates true/false on even events, but only
    ///   if the function has any branches. Functions with `branch_count
    ///   = 0` are flat — no branches taken.
    /// - `io_event` / `gc_event`: always false (static analysis can't
    ///   predict these without a trace).
    /// - `instruction_count`: 8 per event (matches the events_per_tick
    ///   default of 16, so each tick aggregates 16 events × 8 = 128
    ///   instructions).
    fn synthesize_events(
        &self,
        node_assignments: &BTreeMap<String, u32>,
        features: &CanaFeatures,
    ) -> Vec<MirTraceEvent> {
        let mut events = Vec::new();
        let mut tick: u64 = 0;
        for (fname, node_idx) in node_assignments {
            let Some(ff) = features.per_fn.get(fname) else {
                continue;
            };
            let n_events = block_count_estimate(ff);
            for i in 0..n_events {
                events.push(MirTraceEvent {
                    tick,
                    block_id: *node_idx,
                    register_pressure: (ff.memory.expr_count as f64 / 256.0).min(1.0),
                    heap_bytes_in_use: (ff.memory.alloc_sites as u64).saturating_mul(4096),
                    call_depth: ff.cfg.max_loop_depth.saturating_add(1),
                    branch_taken: (i % 2 == 0) && ff.cfg.branch_count > 0,
                    io_event: false,
                    gc_event: false,
                    instruction_count: 8,
                    // Option A carries no thermal signal — static
                    // features can't estimate FP-op density. This
                    // preserves the documented pre-Option-B behaviour
                    // (predict_thermal → 0.0 under synthesis); real
                    // thermal comes from Option-B recorded traces.
                    thermal_intensity: 0.0,
                });
                tick = tick.saturating_add(1);
            }
        }
        events
    }

    /// Project CanaFeatures onto a per-function map of a single
    /// `PressureKind` magnitude. Shared core of
    /// `predict_thermal/memory_peak/cpu_saturation`.
    ///
    /// Returns an empty map when the program has no functions or the
    /// synthesis produces no events.
    fn predict_kind(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
        kind: PressureKind,
    ) -> BTreeMap<String, f64> {
        // 1. Deterministic NodeId assignment: enumerate functions in
        //    program order. We mirror cjc_cana::features::CanaFeatures'
        //    BTreeMap-by-name iteration so the assignment is stable
        //    across invocations.
        let mut node_assignments: BTreeMap<String, u32> = BTreeMap::new();
        for (idx, func) in program.functions.iter().enumerate() {
            // Skip functions not in the features map (e.g. synthetic
            // wrappers that didn't get featurized). This guards against
            // out-of-bounds NodeId allocations later.
            if features.per_fn.contains_key(&func.name) {
                node_assignments.insert(func.name.clone(), idx as u32);
            }
        }
        if node_assignments.is_empty() {
            return BTreeMap::new();
        }
        let n_blocks =
            (node_assignments.values().copied().max().unwrap_or(0) as u32).saturating_add(1);

        // 2. Synthesize events.
        let events = self.synthesize_events(&node_assignments, features);
        if events.is_empty() {
            return BTreeMap::new();
        }

        // 3. Adapter config: one NSS node per function, default
        //    events-per-tick aggregation.
        let cfg = MirAdapterConfig {
            n_blocks,
            ..Default::default()
        };

        // 4. Run through NSS's adapter. On error (only possible if our
        //    synthesis violates a contract), return empty rather than
        //    panic — the call site interprets empty as "no adjustment."
        let Ok(adapter_out) = adapt_mir_trace_to_cluster_trajectory(&events, &cfg) else {
            return BTreeMap::new();
        };

        // 5. The trajectory's last state holds the aggregated pressures.
        //    Use the LAST tick because that's the most "settled"
        //    aggregation across all per-function events.
        //    `ClusterTrajectory::last_state` returns Option<&ClusterSystemState>.
        let Some(last_state) = adapter_out.trajectory.last_state() else {
            return BTreeMap::new();
        };

        // 6. For each function, find its NodeId's pressure of the
        //    requested kind. Clamp to [0, 1] (the adapter can emit
        //    above 1 for over-saturated dimensions; we normalize for
        //    the [0, 1] PressurePredictor trait contract).
        let mut result = BTreeMap::new();
        for (fname, node_idx) in &node_assignments {
            let node_id = NodeId(*node_idx);
            if let Some(node_state) = last_state.nodes.get(&node_id) {
                let magnitude = node_state
                    .pressures
                    .get(kind)
                    .map(|p| p.magnitude)
                    .unwrap_or(0.0);
                let normalized = magnitude.clamp(0.0, 1.0);
                result.insert(fname.clone(), normalized);
            }
        }
        result
    }
}

/// Estimate the number of basic blocks for a function as a function of
/// its `expr_count`. We use `max(expr_count / 8, 1)` to ensure every
/// featurized function contributes at least one event to the trace.
fn block_count_estimate(ff: &FnFeatures) -> u64 {
    let raw = (ff.memory.expr_count as u64) / 8;
    raw.max(1)
}

impl PressurePredictor for NssPressurePredictor {
    fn predict_thermal(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        // Option A: synthesize trace → NSS adapter → extract per-node
        // Thermal-pressure magnitudes. See predict_kind for details.
        self.predict_kind(program, features, PressureKind::Thermal)
    }

    fn predict_memory_peak(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        self.predict_kind(program, features, PressureKind::Memory)
    }

    fn predict_cpu_saturation(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        self.predict_kind(program, features, PressureKind::Cpu)
    }

    fn identify_structural_hot_kernels(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> Vec<String> {
        // Pure structural analysis: no NSS predictor needed.
        //
        // Strategy: a function is structurally "hot" when its CFG shape
        // suggests a non-trivial inner loop that gets executed many
        // times AND has internal control flow worth specializing.
        //
        // Concretely (per the §3A.4 feature distribution audit):
        //
        //   * `max_loop_depth >= HOT_LOOP_DEPTH_MIN` — at least one
        //     deeply-nested loop. Q75 of the corpus was 1, max 4, so
        //     2 surfaces the upper quartile.
        //   * `branch_count >= HOT_BRANCH_COUNT_MIN` — at least two
        //     conditional branches inside the function. Single-loop
        //     accumulators don't qualify; if/else-in-loop does.
        //
        // The AND-conjunction prevents false positives like
        // `init_weights` (single loop, no branches → not hot) and
        // surfaces real workhorses like a PINN `forward` pass (nested
        // loops + per-layer activation branches → hot).
        //
        // Output is sorted lexicographically because `per_fn` is a
        // BTreeMap (sorted iteration); we preserve that determinism in
        // the returned Vec.
        let mut hot: Vec<String> = Vec::new();
        for (fn_name, fn_features) in &features.per_fn {
            // Only consider functions actually in the program — guards
            // against stale features. (Defensive: in practice
            // `features` is computed from `program`.)
            if !program.functions.iter().any(|f| &f.name == fn_name) {
                continue;
            }
            if fn_features.cfg.max_loop_depth >= HOT_LOOP_DEPTH_MIN
                && fn_features.cfg.branch_count >= HOT_BRANCH_COUNT_MIN
            {
                hot.push(fn_name.clone());
            }
        }
        hot
    }

    fn name(&self) -> &'static str {
        "nss_synthetic_trace_v1"
    }

    fn version(&self) -> u32 {
        2
    }
}

// ---------------------------------------------------------------------------
// Option B (PR 5) — RecordedPressurePredictor
// ---------------------------------------------------------------------------

/// A [`PressurePredictor`] backed by REAL recorded executor traces
/// (Option B), instead of Option A's synthetic projection.
///
/// Construct from the outputs of
/// `cjc_mir_exec::run_program_instrumented`: the event vec plus the
/// executor's `trace_node_assignments()` (the fn-name → node-index map
/// captured at execution time — using the recorded assignment instead
/// of re-deriving it at predict time makes the pairing immune to any
/// re-lowering reordering).
///
/// ## What changes vs Option A
///
/// Every event field reflects actual runtime observation: real
/// per-iteration loop events, real FP-op density (→ Thermal — a kind
/// Option A structurally cannot populate), real frame sizes, real
/// branch outcomes. Per-function pressures genuinely diverge, which is
/// what makes `ThermalAwareCostModel` / `PinnPhysicalCostModel`
/// penalties non-degenerate.
///
/// ## Determinism
///
/// The stored events came from a deterministic instrumented run; the
/// adapter pipeline is deterministic; `predict_*` is a pure function
/// of `(stored events, stored assignments, queried features)`.
#[derive(Debug, Clone)]
pub struct RecordedPressurePredictor {
    events: Vec<MirTraceEvent>,
    node_assignments: BTreeMap<String, u32>,
}

impl RecordedPressurePredictor {
    /// Build from a recorded instrumented run. `node_assignments` is
    /// `MirExecutor::trace_node_assignments()` cloned.
    pub fn from_recorded_events(
        events: Vec<MirTraceEvent>,
        node_assignments: BTreeMap<String, u32>,
    ) -> Self {
        Self {
            events,
            node_assignments,
        }
    }

    /// Number of recorded events backing this predictor.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Shared core: adapter over the RECORDED events, then per-function
    /// extraction of one pressure kind. Mirrors
    /// `NssPressurePredictor::predict_kind` except the event source.
    fn predict_kind(
        &self,
        _program: &MirProgram,
        features: &CanaFeatures,
        kind: PressureKind,
    ) -> BTreeMap<String, f64> {
        if self.events.is_empty() || self.node_assignments.is_empty() {
            return BTreeMap::new();
        }
        let n_blocks = self
            .node_assignments
            .values()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let cfg = MirAdapterConfig {
            n_blocks,
            ..Default::default()
        };
        let Ok(adapter_out) = adapt_mir_trace_to_cluster_trajectory(&self.events, &cfg) else {
            return BTreeMap::new();
        };
        let Some(last_state) = adapter_out.trajectory.last_state() else {
            return BTreeMap::new();
        };
        let mut result = BTreeMap::new();
        for (fname, node_idx) in &self.node_assignments {
            // Only report functions the caller actually featurized —
            // keeps the output surface identical to Option A's.
            if !features.per_fn.contains_key(fname) {
                continue;
            }
            let node_id = NodeId(*node_idx);
            if let Some(node_state) = last_state.nodes.get(&node_id) {
                let magnitude = node_state
                    .pressures
                    .get(kind)
                    .map(|p| p.magnitude)
                    .unwrap_or(0.0);
                result.insert(fname.clone(), magnitude.clamp(0.0, 1.0));
            }
        }
        result
    }
}

impl PressurePredictor for RecordedPressurePredictor {
    fn predict_thermal(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        self.predict_kind(program, features, PressureKind::Thermal)
    }

    fn predict_memory_peak(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        self.predict_kind(program, features, PressureKind::Memory)
    }

    fn predict_cpu_saturation(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        self.predict_kind(program, features, PressureKind::Cpu)
    }

    fn identify_structural_hot_kernels(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> Vec<String> {
        // Structural identification is static — delegate to the same
        // logic Option A uses (it consults features, not events).
        NssPressurePredictor::default().identify_structural_hot_kernels(program, features)
    }

    fn name(&self) -> &'static str {
        "nss_recorded_trace_v1"
    }

    fn version(&self) -> u32 {
        1
    }
}

// ---------------------------------------------------------------------------
// In-crate unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_records_seed() {
        let p = NssPressurePredictor::from_seed(7);
        assert_eq!(p.seed().0, 7);
    }

    #[test]
    fn default_seed_is_42() {
        let p = NssPressurePredictor::default();
        assert_eq!(p.seed().0, 42);
    }

    #[test]
    fn name_and_version_are_stable() {
        let p = NssPressurePredictor::default();
        assert_eq!(p.name(), "nss_synthetic_trace_v1");
        assert_eq!(p.version(), 2);
    }

    // Tests that need to construct a `MirProgram` / `CanaFeatures` live in
    // `tests/integration.rs` because the helper imports (`cjc_parser`,
    // `cjc_ast`, `cjc_hir`) are dev-dependencies and unavailable to
    // src-level unit tests.
}
