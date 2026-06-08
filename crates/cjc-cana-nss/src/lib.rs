//! # `cjc-cana-nss` — CANA ↔ NSS bridge crate
//!
//! Implements [`cjc_cana::pressure::PressurePredictor`] backed by the
//! Neural Systems Simulator (`cjc-nss`). This is the §4B.2 deliverable
//! from `docs/cana/HANDOFF_NEXT_SESSION.md`.
//!
//! ## Mode: Option C (structural-only)
//!
//! The design-options document
//! ([`CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md`](../../../docs/cana/CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md))
//! enumerates four implementations (A: synthetic trace, B: real MIR-exec
//! instrumentation, C: structural-only, D: hybrid). This crate ships
//! **Option C**:
//!
//! - [`NssPressurePredictor::identify_structural_hot_kernels`] uses pure
//!   CFG analysis from [`cjc_cana::features::CfgMetrics`] +
//!   [`cjc_cana::features::MemoryProxy`] to surface
//!   structurally-likely-to-be-hot functions. Deterministic. No trace
//!   required.
//! - [`NssPressurePredictor::predict_thermal`],
//!   [`NssPressurePredictor::predict_memory_peak`], and
//!   [`NssPressurePredictor::predict_cpu_saturation`] return **empty
//!   maps**. NSS's predictor needs a `Vec<MirTraceEvent>` from a real
//!   MIR execution, which would require ~200-400 LOC of instrumentation
//!   in `cjc-mir-exec` (deferred per `§3A.2` AB-test finding that the
//!   base ranker is currently inactive on real workloads — adding a
//!   thermal layer on top of an inactive base is solving a problem we
//!   don't yet have).
//!
//! Empty thermal maps compose cleanly with [`cjc_cana::thermal_cost_model::ThermalAwareCostModel`]:
//! it queries the map with `.get(name).copied().unwrap_or(0.0)`, so a
//! missing entry is interpreted as zero thermal pressure → no
//! adjustment.
//!
//! ## Determinism contract
//!
//! - All identification is purely structural: same MIR → byte-identical
//!   hot-kernel list.
//! - The `seed` field is stored for future-version use (Option A
//!   synthetic-trace generation would need an RNG). Currently unused
//!   in Option C; existence is a forward-compatibility contract.
//! - All iteration is over `BTreeMap`, never `HashMap`. Output `Vec` is
//!   sorted by function name.
//!
//! ## Future evolution
//!
//! When the base CANA ranker starts producing differential decisions on
//! real workloads (see `§3A.2` follow-up: more diverse corpus, lower
//! skip thresholds, chess RL / LendingClub workloads), this crate should
//! be upgraded:
//!
//! - **Option A path**: implement `synthesize_trace(program, features)
//!   -> Vec<MirTraceEvent>` and route the `predict_*` methods through
//!   NSS's `adapt_mir_trace_to_cluster_trajectory` +
//!   `ClusterNeuralSystemsSimulator::predict_next`.
//! - **Option B path**: wait for `cjc-mir-exec` instrumentation, then
//!   route `predict_*` through real `MirTraceEvent` streams collected
//!   from a profiling run.
//!
//! The current trait surface accommodates both — only the internals of
//! the four `predict_*`/`identify_*` methods need to change.

#![warn(missing_docs)]

use std::collections::BTreeMap;

use cjc_cana::features::CanaFeatures;
use cjc_cana::pressure::PressurePredictor;
use cjc_mir::MirProgram;
use cjc_nss::NssSeed;

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
        Self { seed: NssSeed(seed) }
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

impl PressurePredictor for NssPressurePredictor {
    fn predict_thermal(
        &self,
        _program: &MirProgram,
        _features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        // Option C: empty map. ThermalAwareCostModel reads this as "no
        // adjustment for any function" via `.get(...).unwrap_or(0.0)`.
        // When Option A/B lands, this returns NSS's real per-function
        // thermal predictions sourced from a (synthesized or real) trace.
        BTreeMap::new()
    }

    fn predict_memory_peak(
        &self,
        _program: &MirProgram,
        _features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        BTreeMap::new()
    }

    fn predict_cpu_saturation(
        &self,
        _program: &MirProgram,
        _features: &CanaFeatures,
    ) -> BTreeMap<String, f64> {
        BTreeMap::new()
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
        "nss_structural_v1"
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
        assert_eq!(p.name(), "nss_structural_v1");
        assert_eq!(p.version(), 1);
    }

    // Tests that need to construct a `MirProgram` / `CanaFeatures` live in
    // `tests/integration.rs` because the helper imports (`cjc_parser`,
    // `cjc_ast`, `cjc_hir`) are dev-dependencies and unavailable to
    // src-level unit tests.
}
