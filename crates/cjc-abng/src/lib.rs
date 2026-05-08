//! ABNG — Adaptive Belief Radix Graph.
//!
//! See:
//! - [`docs/abng/PHASE_0_1_DESIGN.md`](../../../docs/abng/PHASE_0_1_DESIGN.md) — substrate design.
//! - [`docs/abng/PHASE_0_2_DESIGN.md`](../../../docs/abng/PHASE_0_2_DESIGN.md) — multi-node topology, children, prefix encoder, descend routing, per-node stats chain.
//!
//! Phase 0.2 ships:
//!
//! 1. Multi-node arena with parent/child edges.
//! 2. `AdaptiveChildren` enum (None / Node4 / Node16 / Node48 / Node256)
//!    with insert-time auto-promotion.
//! 3. Frozen-quantile prefix encoder ([`codebook::QuantileCodebook`]).
//! 4. `descend(prefix) -> RouteEvidence` for radix-style routing.
//! 5. Per-node stats hash chain decoupled from the global event chain.
//! 6. Snapshot format v2 — bumped magic; Phase 0.1 snapshots no longer
//!    load.
//! 7. ~11 new `abng_*` builtins reachable from `.cjcl` source.
//!
//! Phase 0.2 deliberately does **not** ship structural decision triggers
//! (Grow/Split/Merge/Prune/Compress/Freeze) — those depend on the
//! per-node neural head that arrives in Phase 0.3.

pub mod audit;
pub mod blr;
pub mod calibration;
pub mod children;
pub mod codebook;
pub mod density;
pub mod dispatch;
pub mod drift;
pub mod graph;
pub mod leaf_head;
pub mod maturity;
pub mod node;
pub mod policy;
pub mod route;
pub mod serialize;
pub mod signature;
pub mod stats;

pub use audit::{AuditEvent, AuditKind};
pub use blr::{BlrError, BlrPrior, BlrState};
pub use calibration::{CalibrationBins, CalibrationError};
pub use children::{AdaptiveChildren, ChildrenKind};
pub use codebook::{CodebookError, QuantileCodebook};
pub use density::{DensityError, DensityTracker};
pub use dispatch::dispatch_abng;
pub use drift::{DriftBaseline, DriftError};
pub use graph::{ActionKind, AdaptiveBeliefGraph, GraphError};
pub use leaf_head::{LeafHead, LeafHeadError};
pub use maturity::Maturity;
pub use node::{AdaptiveBeliefNode, NodeId};
pub use policy::{DecisionPolicy, PolicyError};
pub use route::RouteEvidence;
pub use signature::NodeSignature;
pub use stats::NodeStats;

/// Returns the canonical genesis hash used as `previous_hash` for the first
/// audit event in any graph.
pub fn genesis_hash() -> [u8; 32] {
    cjc_snap::hash::sha256(b"ABNG-GENESIS-v1")
}
