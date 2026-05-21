//! Adaptive belief node — Phase 0.2 (multi-node arena).
//!
//! Phase 0.2 adds two structural fields on top of Phase 0.1's stats-only node:
//!
//! * `parent: Option<NodeId>` — `None` for the root only; every other node
//!   carries its parent's `NodeId` so reverse traversal is O(1).
//! * `children: AdaptiveChildren` — the radix-style child container
//!   (None / Node4 / Node16 / Node48 / Node256).
//!
//! The Phase 0.1 audit-chain contract is preserved: a node still owns a
//! per-node `stats_chain_head` that advances *only* on observations to that
//! node. In 0.2 this chain decouples from the global event chain (multiple
//! nodes ⇒ each chain advances at its own pace).
//!
//! Future phases will attach:
//!
//! * `BeliefState` — μ/σ²/epistemic/aleatoric tensor head (Phase 0.3)
//! * `Maturity`, `DriftState`, `CalibrationState`, `OodBoundary`,
//!   `NodeSignature` (Phase 0.3+)
//! * `Dense` children variant for compressed terminal nodes (Phase 0.4)

use cjc_runtime::tensor::Tensor;

use crate::blr::BlrState;
use crate::calibration::CalibrationBins;
use crate::children::AdaptiveChildren;
use crate::density::DensityTracker;
use crate::drift::DriftBaseline;
use crate::signature::SignatureWelford;
use crate::stats::NodeStats;

/// Stable identifier for a node within its parent graph. Equal to the arena
/// index in `AdaptiveBeliefGraph::nodes`. Aliased as `u32` so the snapshot
/// format and audit-event payload sizes remain fixed regardless of
/// platform `usize` width.
pub type NodeId = u32;

/// One node of the [`AdaptiveBeliefGraph`](crate::graph::AdaptiveBeliefGraph).
#[derive(Debug, Clone)]
pub struct AdaptiveBeliefNode {
    /// Unique node id within its parent graph; equals the arena index.
    pub node_id: NodeId,
    /// `None` for the root, `Some(parent_id)` for every other node.
    pub parent: Option<NodeId>,
    /// Adaptive child container (Phase 0.2: starts as `None`; promotes
    /// through Node4/16/48/256 as children are added).
    pub children: AdaptiveChildren,
    /// Sufficient statistics for this node's belief state.
    pub stats: NodeStats,
    /// Per-node version counter, bumped on every observation.
    pub stats_version: u64,
    /// Per-node statistical hash chain head. Advances *only* on observations
    /// to this node:
    ///
    /// `stats_chain_head_new = sha256(stats_chain_head_prev || stats.canonical_bytes())`
    ///
    /// Independent of the graph's global event chain.
    pub stats_chain_head: [u8; 32],
    /// Phase 0.3a — per-node MLP parameter tensors. Empty until the graph
    /// installs a [`LeafHead`](crate::leaf_head::LeafHead); thereafter the
    /// `Vec` contains alternating `W_i, b_i` tensors for each layer.
    ///
    /// The leaf-head installation hook initializes these via
    /// [`crate::leaf_head::init_params`] so values are deterministic from
    /// `(graph.seed, node_id, layer_idx, kind_bit)`. Optimizer-driven
    /// updates flow back into this vec via
    /// [`AdaptiveBeliefGraph::leaf_set_param`](crate::AdaptiveBeliefGraph::leaf_set_param).
    pub params: Vec<Tensor>,
    /// Phase 0.3b — per-node Bayesian linear regression posterior state.
    /// `None` until `set_blr_prior` is installed; thereafter every node
    /// (including the root) carries a fresh `BlrState` initialized from
    /// the prior. NIG updates flow through
    /// [`AdaptiveBeliefGraph::blr_update`](crate::AdaptiveBeliefGraph::blr_update).
    pub blr: Option<BlrState>,
    /// Phase 0.3c — per-node density tracker over the BLR feature space.
    /// `None` until `set_density_tracker` is installed.
    pub density: Option<DensityTracker>,
    /// Phase 0.3c — per-node calibration bins (15-bin reliability).
    /// `None` until `set_calibration` is installed.
    pub calibration: Option<CalibrationBins>,
    /// Phase 0.3c — frozen drift baseline. `None` until
    /// `freeze_drift_baseline(node)` snapshots the density tracker.
    pub drift_baseline: Option<DriftBaseline>,
    /// Phase 0.3d-2 — per-node training-time epistemic-σ reference,
    /// captured once when the BLR posterior reaches stability. Used by
    /// [`AdaptiveBeliefGraph::ood_score`](crate::AdaptiveBeliefGraph::ood_score)
    /// to compute the calibrated `epistemic_z` ratio
    /// `(epi / expected_epistemic).clamp(0, 1)`. `None` until captured;
    /// one-shot per node.
    pub expected_epistemic: Option<f64>,
    /// Phase 0.3d-3 — set by `force_freeze` (or 0.3d-4's policy-driven
    /// Freeze). Blocks structural mutations on this node until Unfreeze
    /// (deferred to 0.3d-4 with audit tag `0x16`). Stats / observation
    /// updates are still permitted on a frozen node.
    pub is_frozen: bool,
    /// Phase 0.3d-3 — set to `false` by `force_prune` or absorbed into a
    /// Merge. The node persists in the arena (per the never-reorder-pushes
    /// invariant), but `descend` should treat it as logically removed.
    /// Default: `true` for fresh nodes.
    pub is_active: bool,
    /// Phase 0.3d-4 — the
    /// [`NodeSignature::canonical_bytes`](crate::signature::NodeSignature::canonical_bytes)
    /// captured at the previous `decide_step` call. `None` until the
    /// first call observes the node. Compared bit-for-bit against the
    /// fresh signature on each pass; equality bumps
    /// [`signature_stable_calls`], inequality resets it to zero.
    pub last_signature: Option<[u8; 32]>,
    /// Phase 0.3d-4 — count of consecutive `decide_step` calls that
    /// observed an unchanged [`last_signature`]. Drives the
    /// stability-based triggers (Prune's `prune_grace_epochs`,
    /// Freeze's `freeze_after`). `0` until two consecutive matching
    /// observations have happened.
    pub signature_stable_calls: u64,
    /// Phase 0.4 Track B-2.2.2 — 3-window ECE history buffer. Each
    /// `decide_step` call shifts the buffer left and records the
    /// current calibration ECE in slot `[2]` (the freshest). When
    /// [`ece_fill_count`] reaches 3, [`Maturity::calibration_stable`]
    /// inspects the buffer for `|Δ| < ECE_STABILITY_MAX_DELTA` across
    /// every consecutive pair. Newly-created nodes start with all
    /// zeros and `ece_fill_count = 0`.
    pub ece_history: [f64; 3],
    /// Number of `decide_step` calls that have populated [`ece_history`].
    /// Saturates at 3.
    pub ece_fill_count: u8,
    /// Phase 0.4 Track B-2.2.2 — 3-window epistemic σ history buffer.
    /// Mirror of `ece_history` for `Maturity::uncertainty_stable`. The
    /// observed value is the BLR's `epistemic_leverage` at posterior
    /// mean (see [`crate::graph::epistemic_leverage_at_posterior_mean`]).
    pub sigma_history: [f64; 3],
    /// Number of `decide_step` calls that have populated [`sigma_history`].
    /// Saturates at 3.
    pub sigma_fill_count: u8,
    /// Phase 0.4 Track B-2.2.1 — Welford-folded prediction profile.
    /// Each `decide_step` call observes the running NodeStats mean
    /// into this accumulator; the 8-byte prediction signature byte
    /// string is `sha256(welford_prediction.canonical_bytes)[..8]`.
    pub welford_prediction: SignatureWelford,
    /// Phase 0.4 Track B-2.2.1 — Welford-folded uncertainty profile.
    /// Observed metric: `epistemic_leverage` at posterior mean (BLR).
    /// Stays at default (`n_seen=0`) when BLR is not installed.
    pub welford_uncertainty: SignatureWelford,
    /// Phase 0.4 Track B-2.2.1 — Welford-folded calibration profile.
    /// Observed metric: `CalibrationBins::ece()`. Stays at default
    /// when calibration is not installed.
    pub welford_calibration: SignatureWelford,
    /// Phase 0.4 Track B-2.2.1 — Welford-folded routing profile.
    /// Observed metric: hash-derived f64 of the children layout (see
    /// `crate::signature::routing_observation_value`).
    pub welford_routing: SignatureWelford,
    /// Phase 0.5 Item 1 — opt-in provenance fingerprint binding this
    /// node's training history to the dataset / feature transform that
    /// produced its observations. `[0u8; 32]` (the all-zero default)
    /// means "unstamped"; any other value is a SHA-256 chosen by the
    /// caller (typically `sha256(dataset_bytes ‖ feature_version)`)
    /// and recorded via [`AdaptiveBeliefGraph::stamp_provenance`].
    /// Stamping fires an
    /// [`AuditKind::ProvenanceStamped`](crate::audit::AuditKind::ProvenanceStamped)
    /// event so the lineage is in the audit chain. Idempotent for
    /// repeated stamps with the same hash — only a *change* emits a
    /// new event.
    pub provenance_stamp_hash: [u8; 32],
    /// Phase 0.9.5 R1-2 — lazily-computed cache of
    /// `params_hash(&self.params)`. `validate_blr_inputs` checks the
    /// per-node MLP-params hash against the BLR's `feature_version_hash`
    /// on **every** `blr_update` / `train_step`; caching it skips
    /// re-hashing the (unchanged) params on every training row.
    ///
    /// Invalidated to `None` by `leaf_set_param` / `leaf_set_params_batch`
    /// (the only ops that mutate an existing node's `params`); `None`
    /// after construction and after any (de)serialization round-trip,
    /// where the first `validate_blr_inputs` re-populates it. **NOT
    /// serialized** — a pure performance field, fully derived from
    /// `params`, so the wire format is unchanged.
    pub params_hash_cache: Option<[u8; 32]>,
}

impl AdaptiveBeliefNode {
    /// Construct a fresh node with zero statistics, empty params, and no
    /// BLR / density / calibration / drift state. The graph layer fills
    /// these separately when their respective configurations install.
    pub fn new(node_id: NodeId, parent: Option<NodeId>, initial_chain_head: [u8; 32]) -> Self {
        Self {
            node_id,
            parent,
            children: AdaptiveChildren::new(),
            stats: NodeStats::new(),
            stats_version: 0,
            stats_chain_head: initial_chain_head,
            params: Vec::new(),
            blr: None,
            density: None,
            calibration: None,
            drift_baseline: None,
            expected_epistemic: None,
            is_frozen: false,
            is_active: true,
            last_signature: None,
            signature_stable_calls: 0,
            ece_history: [0.0; 3],
            ece_fill_count: 0,
            sigma_history: [0.0; 3],
            sigma_fill_count: 0,
            welford_prediction: SignatureWelford::new(),
            welford_uncertainty: SignatureWelford::new(),
            welford_calibration: SignatureWelford::new(),
            welford_routing: SignatureWelford::new(),
            // Phase 0.5 Item 1 — unstamped by default.
            provenance_stamp_hash: [0u8; 32],
            // Phase 0.9.5 R1-2 — lazily populated on first validate.
            params_hash_cache: None,
        }
    }

    /// Apply an observation and advance the per-node stats chain in place.
    /// Returns the new `stats_version`. The graph layer is responsible for
    /// adding the corresponding `BeliefUpdate` audit event to the global
    /// chain afterwards.
    pub fn observe(&mut self, value: f64) -> u64 {
        self.stats.observe(value);
        self.stats_version += 1;
        self.advance_stats_chain();
        self.stats_version
    }

    /// Phase 0.6 Item 4 — apply a batch of observations as a single
    /// audit-eligible mutation. Folds N values through Welford in row
    /// order (preserves Kahan determinism), bumps `stats_version` by
    /// 1 (NOT N — the batch is one logical event), and advances the
    /// per-node stats chain head ONCE at the end.
    ///
    /// Determinism contract: post-batch `stats.canonical_bytes()` is
    /// bit-identical to applying the same `values` slice through N
    /// sequential `observe()` calls. The stats_chain_head WILL differ
    /// because per-row updates emit N chain advances vs one for a
    /// batch — they're different audit histories of the same final
    /// state.
    pub fn observe_batch_apply(&mut self, values: &[f64]) -> u64 {
        for v in values {
            self.stats.observe(*v);
        }
        self.stats_version += 1;
        self.advance_stats_chain();
        self.stats_version
    }

    /// Advance the per-node stats chain to the post-update stats hash.
    /// `pub(crate)` so the replay path can drive batch / install
    /// updates from outside this module without going through
    /// `observe`.
    pub(crate) fn advance_stats_chain(&mut self) {
        // 32 (prev chain head) + 32 (canonical_bytes, Phase 0.5 v12).
        // Phase 0.7 (F) — stack-allocated `[u8; 64]` instead of a fresh
        // `Vec::with_capacity(64)` per call. Both halves are fixed-size
        // (`stats_chain_head: [u8; 32]`, `stats.canonical_bytes() ->
        // [u8; 32]`) so the buffer's full size is known at compile time
        // and there's no reason to pay an allocator round-trip on every
        // observation. SHA-256 input bytes are byte-identical to the
        // pre-0.7 Vec form.
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&self.stats_chain_head);
        buf[32..].copy_from_slice(&self.stats.canonical_bytes());
        self.stats_chain_head = cjc_snap::hash::sha256(&buf);
    }

    /// Whether this node is a leaf (no children attached).
    pub fn is_leaf(&self) -> bool {
        matches!(self.children, AdaptiveChildren::None)
    }
}
