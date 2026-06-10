//! # CANA — Compiler-Aware Neural Architecture (Phase 1: Passive Observer)
//!
//! CANA is a deterministic compiler/runtime optimization advisor for CJC-Lang.
//! Phase 1 is intentionally **passive** — it observes MIR and emits structured
//! reports but never influences which optimization passes the compiler runs.
//!
//! ## Design contract
//!
//! 1. **Additive.** `cjc-cana` reads `&MirProgram` (and derived analyses) and
//!    returns owned report values. It never mutates the IR.
//! 2. **Deterministic.** Same MIR program → byte-identical `CanaReport`,
//!    regardless of platform, thread count, or run order. All maps are
//!    `BTreeMap`, all reductions are integer counts (no floats in Phase 1),
//!    all hashes use a stable FNV-1a implementation.
//! 3. **Legality-first.** A `LegalityGate` trait surface exists so Phase 2
//!    advisory recommendations have somewhere to be vetoed. Phase 1 ships a
//!    default gate that refuses any pass-reordering touching a function with a
//!    strict reduction (`StrictFold` / `KahanFold` / `Unknown`).
//! 4. **Cost model is a seam, not an implementation.** `CostModel` is a trait;
//!    Phase 1 ships only `NullCostModel` returning [`CostEstimate::Unknown`].
//!    Phase 2 will plug in a tiny deterministic regression head.
//!
//! ## Pipeline
//!
//! ```text
//!   MirProgram
//!      │
//!      ├─► cfg::CfgBuilder::build      ─► MirCfg
//!      ├─► DominatorTree::compute      ─► DominatorTree
//!      ├─► compute_loop_tree           ─► LoopTree
//!      ├─► detect_reductions           ─► ReductionReport
//!      │
//!      ▼
//!   cjc_cana::analyze_program(&program) ─► CanaReport
//!      │   ├─ features::CanaFeatures
//!      │   │   ├─ per_fn: BTreeMap<FnName, FnFeatures>
//!      │   │   │   ├─ cfg_metrics
//!      │   │   │   ├─ memory_proxy
//!      │   │   │   └─ reduction_axes
//!      │   │   └─ program_hash, feature_hash
//!      │   └─ legality::LegalityVerdict (over the no-op pass sequence)
//!      ▼
//!   JSON sidecar (deterministic ordering)
//! ```
//!
//! ## Future phases (not in this crate yet)
//!
//! - **Phase 2** — Advisory: pass-ordering recommendations + linear cost head.
//! - **Phase 3** — Active legality-gated: compiler reads recommendations and
//!   applies the ones the gate approves.
//! - **Phase 4** — NSS integration: project `FnFeatures` onto NSS's
//!   `PressureKind × NodeId` substrate for thermal-aware compilation.
//! - **Phase 5** — Profile-guided: persist `PassHistory` across runs and feed
//!   measured runtime cost back into the cost model.
//!
//! ## Cross-references
//!
//! - NSS handoff (separate fork): `crates/cjc-nss/` operates over the same
//!   `PressureKind × NodeId` substrate but for production-system dynamics.
//!   See the NSS architecture doc, §3 ("Why this architecture is a good fit
//!   for compilers") for the projection mapping.

pub mod caching_ranker;
pub mod cfg_metrics;
pub mod cost_model;
pub mod features;
pub mod fusion;
pub mod hash;
pub mod kernel_variant;
pub mod legality;
pub mod linear_cost_model;
pub mod memory_proxy;
pub mod pass_history;
pub mod pass_ranker;
pub mod physical_cost;
pub mod pinn_cost_model;
pub mod pressure;
pub mod reduction_axes;
pub mod report;
pub mod thermal_cost_model;

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

use cjc_mir::MirProgram;

pub use crate::caching_ranker::{
    default_caching_ranker, CacheStats, CachingPassRanker, DEFAULT_CACHE_CAPACITY,
};
pub use crate::cfg_metrics::CfgMetrics;
pub use crate::cost_model::{CostEstimate, CostModel, NullCostModel};
pub use crate::features::{CanaFeatures, FnFeatures};
pub use crate::fusion::{
    identify_fusion_candidates, is_native_primitive, ChainEntry, FusionCandidate,
    FusionPlan, NATIVE_PRIMITIVES,
};
pub use crate::hash::{CanaHasher, CfgHash, FeatureHash, ProgramHash};
pub use crate::legality::{
    DefaultLegalityGate, LegalityGate, LegalityVerdict, LegalityViolation, PassSequence,
    ProposedPass,
};
pub use crate::linear_cost_model::LinearCostModel;
pub use crate::memory_proxy::MemoryProxy;
pub use crate::pass_history::{PassHistory, PassRecord};
pub use crate::pass_ranker::{
    default_ranker, pass_plan_from, recommend_pass_plan, FunctionRanking, PassRanker,
    PassRecommendation, RankingRationale, RankingReport, CANONICAL_PASSES,
    DEFAULT_SKIP_THRESHOLD,
};
pub use crate::physical_cost::{
    build_physical_query, predict_physical, PhysicalCoefficients, PhysicalConstraints,
    PhysicalCostEstimate, PhysicalCostQuery,
};
pub use crate::pinn_cost_model::{
    PinnPhysicalCostModel, PINN_V1_MODEL_ID, PINN_V1_MODEL_VERSION,
};
pub use crate::reduction_axes::ReductionAxes;
pub use crate::report::CanaReport;

/// Run the CANA passive observer over a MIR program.
///
/// Returns a fully populated [`CanaReport`] with:
/// - Per-function features (CFG metrics, memory proxy, reduction axes)
/// - Program-level content-addressed hashes
/// - The default legality verdict over an empty proposed pass sequence
///   (always [`LegalityVerdict::Approved`] when nothing is proposed — useful
///   as a sanity baseline for downstream gating)
///
/// # Determinism
///
/// For a given `MirProgram`, this function produces a byte-identical
/// `CanaReport.canonical_bytes()` across runs, OS, CPU architecture, and
/// thread count. This is the foundational invariant Phase 1 establishes;
/// all later phases inherit it.
///
/// # Cost
///
/// O(N) over MIR statements + the cost of building one CFG / dominator tree /
/// loop tree per function. None of these allocate per-element heap — the
/// underlying analyses use `Vec`-indexed dense IDs.
pub fn analyze_program(program: &MirProgram) -> CanaReport {
    let features = features::extract(program);
    let gate = DefaultLegalityGate::new();
    let baseline = gate.verify(program, &PassSequence::empty(), &features);
    CanaReport::new(features, baseline)
}
