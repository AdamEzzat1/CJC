//! Tamper-evident audit log.
//!
//! Every mutation to the graph appends one [`AuditEvent`] whose `new_hash`
//! is `sha256(previous_hash ‖ canonical_payload)`. This produces a hash
//! chain that detects any post-hoc tampering: corrupting one event breaks
//! every subsequent `new_hash`.
//!
//! ## Phase 0.1 kinds (preserved verbatim)
//!
//! * `Created` — emitted once when a node is added (root, at graph
//!   construction time).
//! * `BeliefUpdate { value }` — one per observation.
//!
//! ## Phase 0.2 kinds (additions)
//!
//! * `NodeAdded { parent, key_byte }` — manual or future automatic
//!   `add_node`. The new node lands at the next free arena index; the
//!   payload records its parent + the prefix byte that bound it.
//! * `ChildrenPromoted { from, to }` — when an insert into a child
//!   container at capacity triggers a promotion. `from`/`to` are
//!   `ChildrenKind` codes (0 = None, 1 = Node4, 2 = Node16, 3 = Node48,
//!   4 = Node256). The payload deliberately does **not** record which
//!   key triggered the promotion — the immediately-following `NodeAdded`
//!   event carries that.
//! * `CodebookFrozen { codebook_hash }` — emitted once when
//!   `set_codebook` installs the quantile codebook. Subsequent
//!   `set_codebook` calls error.
//!
//! Tag bytes 0x00–0x04 are *frozen* — adding a new kind in Phase 0.3+ must
//! allocate a new tag rather than reusing one.

use crate::node::NodeId;

/// What happened, plus any payload needed to deterministically replay it.
#[derive(Debug, Clone, PartialEq)]
pub enum AuditKind {
    /// A node was added to the graph (Phase 0.1: only the root, at
    /// construction time. Phase 0.2: now used for child nodes too — the
    /// `NodeAdded` kind below carries the parent edge for non-root nodes).
    Created,
    /// One observation was applied to a node's [`NodeStats`](crate::stats::NodeStats).
    BeliefUpdate {
        /// The observed value, recorded so replay can reproduce the post-update
        /// statistics deterministically.
        value: f64,
    },
    /// A child node was added (Phase 0.2).
    NodeAdded {
        /// Parent node id.
        parent: NodeId,
        /// Prefix byte that binds this child to its parent's children.
        key_byte: u8,
    },
    /// A node's [`AdaptiveChildren`](crate::children::AdaptiveChildren)
    /// variant was promoted (Phase 0.2).
    ChildrenPromoted {
        /// The variant before promotion (numeric code from
        /// [`ChildrenKind`]).
        from: u8,
        /// The variant after promotion.
        to: u8,
    },
    /// The graph's quantile codebook was installed and frozen (Phase 0.2).
    CodebookFrozen {
        /// SHA-256 of the codebook's canonical-byte encoding.
        codebook_hash: [u8; 32],
    },
    /// The per-node MLP head architecture was installed and frozen
    /// (Phase 0.3a). Subsequent `set_leaf_head` calls error.
    LeafHeadConfigured {
        /// SHA-256 of the head's canonical-byte encoding.
        config_hash: [u8; 32],
    },
    /// A node's MLP params were Xavier-initialized for the first time
    /// (Phase 0.3a). Fires once per node — for the root at
    /// `set_leaf_head` time, for children at `add_node` time.
    LeafParamsInitialized {
        /// SHA-256 of the freshly-initialized params (witness for replay).
        params_hash: [u8; 32],
    },
    /// A node's MLP params were updated (Phase 0.3a). Fires once per
    /// `leaf_set_param` call. The new params bytes live in the per-node
    /// section of the snapshot, not in this event payload — this event
    /// carries only the 32-byte hash witness so the event log stays
    /// compact under heavy training workloads.
    LeafParamsUpdated {
        /// SHA-256 of the post-update params.
        params_hash: [u8; 32],
    },
    /// The graph-wide Bayesian linear regression prior was installed
    /// and frozen (Phase 0.3b). Subsequent `set_blr_prior` calls error.
    BlrPriorConfigured {
        /// SHA-256 of the prior's canonical bytes.
        config_hash: [u8; 32],
    },
    /// A node's BLR posterior state was initialized to the prior
    /// (Phase 0.3b). Fires once per node — for the root at
    /// `set_blr_prior` time, for children at `add_node` time.
    BlrInitialized {
        /// SHA-256 of the freshly-initialized BLR state (witness for replay).
        state_hash: [u8; 32],
    },
    /// A node's BLR posterior was updated via NIG conjugate update
    /// (Phase 0.3b). Fires once per `blr_update` call. The new state
    /// bytes live in the per-node section of the snapshot, not in this
    /// event payload — this event carries only the 32-byte hash
    /// witness.
    BlrUpdated {
        /// SHA-256 of the post-update BLR state.
        state_hash: [u8; 32],
    },
    /// A node's density tracker was installed (Phase 0.3c). Fires once
    /// per node — for the root at `set_density_tracker` time, for
    /// children at `add_node` time after the tracker is configured.
    DensityTrackerInstalled {
        /// SHA-256 of the freshly-initialized density tracker.
        state_hash: [u8; 32],
    },
    /// A node's density tracker was updated by a batch of observations
    /// (Phase 0.3c).
    DensityUpdated {
        /// SHA-256 of the post-update density tracker.
        state_hash: [u8; 32],
    },
    /// A node's calibration bins were installed (Phase 0.3c).
    CalibrationInstalled {
        /// SHA-256 of the freshly-initialized bins.
        state_hash: [u8; 32],
    },
    /// A node's calibration bins were updated by one observation
    /// (Phase 0.3c).
    CalibrationUpdated {
        /// SHA-256 of the post-update bins.
        state_hash: [u8; 32],
    },
    /// A node's drift baseline was frozen from its current density
    /// tracker (Phase 0.3c).
    DriftBaselineFrozen {
        /// SHA-256 of the frozen baseline.
        state_hash: [u8; 32],
    },
    /// A node's per-node training-time epistemic-σ reference was
    /// captured (Phase 0.3d-2). One-shot per node — once a node has
    /// captured this value, subsequent attempts error. The actual
    /// captured `f64` lives in the per-node section of the snapshot;
    /// this event carries only the 32-byte witness so the audit log
    /// stays compact under heavy training workloads.
    ///
    /// Tag `0x17` deliberately skips `0x10..0x16` which are reserved
    /// for the structural-action events that arrive in Phase 0.3d-3/4
    /// (Grow/Split/Merge/Prune/Compress/Freeze/Unfreeze).
    ExpectedEpistemicCaptured {
        /// SHA-256 of the captured-value canonical bytes
        /// (`f64::to_bits().to_be_bytes()`).
        state_hash: [u8; 32],
    },
    /// Phase 0.3d-3 — policy-driven (or `force_grow`) addition of a
    /// new child. Distinct from [`AuditKind::NodeAdded`] so replay can
    /// distinguish user-driven topology from structural-decision
    /// topology. Carries the full `(parent, key_byte, child)` tuple
    /// because the event itself drives graph reconstruction.
    Grow {
        /// Parent node that gained the new child.
        parent: NodeId,
        /// Prefix byte that binds the new child to the parent.
        key_byte: u8,
        /// Newly created node id (= arena index after the push).
        child: NodeId,
    },
    /// Phase 0.3d-3 — policy-driven (or `force_split`) split of a leaf
    /// into two new children. Carries both new node ids so replay can
    /// reconstruct the exact arena order.
    Split {
        /// Parent node whose children container was extended.
        parent: NodeId,
        /// First new child (lower arena index).
        child_a: NodeId,
        /// Second new child (higher arena index).
        child_b: NodeId,
    },
    /// Phase 0.3d-3 — policy-driven (or `force_merge`) absorption of
    /// one node's responsibilities into another. Phase 0.3d-3 ships
    /// the *event channel* only; semantic merging (NIG combination,
    /// stats fusion) lands with `decide_step` in 0.3d-4.
    Merge {
        /// Node being absorbed and marked inactive.
        absorbed: NodeId,
        /// Node receiving the absorbed node's responsibility.
        into: NodeId,
    },
    /// Phase 0.3d-3 — policy-driven (or `force_prune`) deactivation of
    /// a node. The node persists in the arena (per the
    /// never-reorder-pushes invariant) but `is_active` becomes false.
    Prune {
        /// Node marked inactive.
        node_id: NodeId,
    },
    /// Phase 0.3d-3 — policy-driven (or `force_compress`) replacement
    /// of a node's children container with the [`Dense`](crate::children::AdaptiveChildren::Dense)
    /// variant. Descendants persist in the arena but become unreachable
    /// from this node through `descend`.
    Compress {
        /// 32-byte signature representing the compressed sub-tree.
        signature: [u8; 32],
    },
    /// Phase 0.3d-3 — policy-driven (or `force_freeze`) freezing of a
    /// node. Structural mutations on the frozen node are blocked until
    /// the un-freeze path lands at tag `0x16` in Phase 0.3d-4.
    Freeze {
        /// Node marked frozen.
        node_id: NodeId,
    },
    /// Phase 0.3d-4 — un-freeze a previously frozen node. Re-enables
    /// structural mutations on it. Used by 0.3d-4's `decide_step` when
    /// drift signals exceed a threshold (drift-trip un-freeze) and
    /// directly via `abng_unfreeze` for testing / manual override.
    Unfreeze {
        /// Node un-frozen.
        node_id: NodeId,
    },
    /// Phase 0.4 Track C-2.3.4 — diagnostic event fired when a numerical
    /// rescue branch inside a BLR update silently clamped a value to
    /// keep the InverseGamma posterior well-defined. The state-changing
    /// event for the same update is the immediately-preceding
    /// `BlrUpdated`; this event is metadata-only (no further state
    /// mutation). Consumers filter the audit log for this kind to
    /// identify unstable training regimes.
    BlrNumericalRescue {
        /// Which numerical branch fired. Phase 0.4 ships only `0x00 =
        /// b_below_epsilon` (post-update `b` would have fallen below
        /// `f64::EPSILON`); future numerical-rescue branches allocate
        /// from `0x01..` and are documented alongside the audit-kind
        /// table in `ABNG_CURRENT_ARCHITECTURE.md` §3.6.
        reason: u8,
        /// `f64::to_bits()` of the pre-clamp value. Big-endian on the
        /// wire. Recovers the original problematic computation for
        /// post-hoc analysis.
        b_pre_clamp_bits: u64,
    },
    /// Phase 0.4 Track C-2.3.6 — emitted by the batch param-writeback
    /// builtin `abng_leaf_set_params_batch`. Replaces the per-tensor
    /// `LeafParamsUpdated` events that fire under the
    /// individual-`leaf_set_param`-per-tensor pattern: one optimizer
    /// step on a 2-layer head writes 6 events under the old API
    /// (`2(L+1)` per step) but exactly 1 under the batch API. Same
    /// witness shape as `LeafParamsUpdated` — the per-node section
    /// holds the actual params bytes, hashed against this witness on
    /// replay.
    LeafParamsUpdatedBatch {
        /// SHA-256 of the post-update params blob (whole vector).
        params_hash: [u8; 32],
    },
    /// Phase 0.4 Track A — emitted by `descend_traced()` when route
    /// tracing is enabled. Records the leaf node a descend resolved
    /// to and how many prefix bytes matched. Tag `0x1B`. Opt-in via
    /// the explicit traced-descend method to avoid log explosion under
    /// heavy inference; the untraced `descend` remains silent.
    /// Replay reapplies as a no-op (descend doesn't change graph
    /// state) — the chain witness is enough to prove the route
    /// happened.
    Routed {
        /// Resolved leaf id (terminal node of the walk).
        leaf: NodeId,
        /// How many prefix bytes were successfully matched
        /// (0 ≤ matched ≤ prefix.len(); prefix.len() ≤ 255 in practice).
        matched_prefix: u8,
    },
    /// Phase 0.4 Track A — log-compaction checkpoint marker. Emitted
    /// by `compact_log` for each touched node, captures a SHA-256 of
    /// the node's full Welford-stats state at the compaction point.
    /// Tag `0x1A`. Phase 0.4 ships the marker only; smart-replay that
    /// uses StatsSnapshot to fast-forward past `*Updated` runs is
    /// deferred to Phase 0.5 (Phase 0.4's `apply_event` for this kind
    /// is a no-op, so the chain advances but no graph state changes).
    StatsSnapshot {
        /// Node whose stats are checkpointed.
        node_id: NodeId,
        /// SHA-256 of `NodeStats::canonical_bytes()` at emission time.
        stats_hash: [u8; 32],
    },
    /// Phase 0.5 Item 1 — opt-in provenance stamp for one node. Binds
    /// the node's training history to a caller-chosen 32-byte SHA-256
    /// (typically `sha256(dataset_bytes ‖ feature_version)`) so that
    /// `cjcl abng explain` can verify dataset / feature-transform
    /// lineage in addition to the existing model + codebook + leaf head
    /// + BLR coverage. Tag `0x1C`. Idempotent: repeated stamps with
    /// the same hash are no-op (no event); only a *change* fires a new
    /// event. Replay reapplies by writing the stored hash into the
    /// target node's `provenance_stamp_hash` field.
    ProvenanceStamped {
        /// Node being stamped.
        node_id: NodeId,
        /// 32-byte caller-chosen provenance fingerprint.
        hash: [u8; 32],
    },
    /// Phase 0.6 Item 4 — batched `BeliefUpdate`. Collapses N
    /// per-row observations on the same node into one audit event,
    /// saving (N-1) chain-hash recomputations and (N-1) per-node
    /// `stats_chain_head` SHA-256 advances. Tag `0x1D`.
    ///
    /// Determinism contract: the post-batch `NodeStats::canonical_bytes`
    /// must be bit-identical to applying the same `values` slice via
    /// N sequential `observe()` calls in row order. The stats chain
    /// head WILL differ from the per-row path (one advance per batch
    /// vs N advances), but per-batch and per-row are not interchangeable
    /// from a chain-witness perspective — they are different audit
    /// histories.
    ///
    /// Payload format: `count u32 BE ‖ values f64×count (each as
    /// `.to_bits().to_be_bytes()`) ‖ batch_hash [u8; 32]`. The
    /// `batch_hash` is `sha256(count_be ‖ values_be)`, redundant with
    /// the audit chain hash but explicit so consumers (e.g.
    /// `cjcl abng inspect --json`) can present a per-batch tamper
    /// signal without rehashing the full payload.
    BeliefUpdateBatch {
        /// Number of observations in this batch (>= 1, > u32::MAX is
        /// rejected at the boundary).
        count: u32,
        /// SHA-256 of `count_be ‖ values_be`. Redundant w.r.t. the
        /// audit chain hash but kept for explicit per-batch tamper
        /// inspection.
        batch_hash: [u8; 32],
        /// The actual observation values, in arrival order. Stored in
        /// the kind so replay can reproduce the Welford state by
        /// applying each value sequentially.
        values: Vec<f64>,
    },
    /// Phase 0.8c v14 Item A2 — fused per-row training step. Collapses
    /// `BlrUpdated + BeliefUpdate` (the audit events that
    /// `Graph::train_step` previously emitted as a 3-call sequence)
    /// into ONE chain event. Halves the audit-log size and chain-hash
    /// compute on training-heavy workloads. Tag `0x1E`.
    ///
    /// Determinism contract: after applying `TrainStep { value,
    /// state_hash }` at a node, the node's `NodeStats::canonical_bytes`
    /// and `BlrState` are bit-identical to what the pre-A2 3-event
    /// sequence (`BlrUpdated + BeliefUpdate`) would have produced on
    /// the same pre-state with the same `(phi, value)` inputs. The
    /// chain head WILL differ from the pre-A2 sequence (one chain
    /// step vs two) — that is the entire point of the fusion. Forward
    /// and replay paths under v14 produce the same chain head as each
    /// other; the pre-A2 chain head is not recoverable post-fusion.
    ///
    /// Payload format: `value f64 (.to_bits().to_be_bytes()) ‖
    /// state_hash [u8; 32]`. The BLR design row `phi` is NOT in the
    /// payload (matches the pre-A2 `BlrUpdated` convention of
    /// witness-only); the post-train-step BLR state lives in the
    /// per-node section of the snapshot and is verified against
    /// `state_hash` by the end-of-replay verifier.
    ///
    /// Wire-format gating: this tag is only valid under `WireVersion::V14`.
    /// Decoders reading v13 archives reject `0x1E` as
    /// `DecodeError::UnknownKindTag(0x1E)`.
    TrainStep {
        /// The observation value (the BLR target `y[0]` and the
        /// Welford observation). Encoded so replay can reapply
        /// `nodes[node_id].observe(value)` — mirrors `BeliefUpdate.value`.
        value: f64,
        /// SHA-256 of the post-update BLR state (= what `BlrUpdated`
        /// would have carried). End-of-replay verifier matches this
        /// against the snapshot's reconstructed BLR state for the
        /// node.
        state_hash: [u8; 32],
    },
}

/// `AuditKind::BlrNumericalRescue::reason` value: post-update `b` would
/// have fallen below `f64::EPSILON`. The graph layer uses this when
/// emitting the rescue event from `blr_update`.
pub const BLR_RESCUE_B_BELOW_EPSILON: u8 = 0x00;

impl AuditKind {
    /// Tag byte for canonical encoding. Frozen — new kinds must allocate
    /// new tags rather than reusing existing ones.
    pub fn tag(&self) -> u8 {
        match self {
            AuditKind::Created => 0x00,
            AuditKind::BeliefUpdate { .. } => 0x01,
            AuditKind::NodeAdded { .. } => 0x02,
            AuditKind::ChildrenPromoted { .. } => 0x03,
            AuditKind::CodebookFrozen { .. } => 0x04,
            AuditKind::LeafHeadConfigured { .. } => 0x05,
            AuditKind::LeafParamsInitialized { .. } => 0x06,
            AuditKind::LeafParamsUpdated { .. } => 0x07,
            AuditKind::BlrPriorConfigured { .. } => 0x08,
            AuditKind::BlrInitialized { .. } => 0x09,
            AuditKind::BlrUpdated { .. } => 0x0A,
            AuditKind::DensityTrackerInstalled { .. } => 0x0B,
            AuditKind::DensityUpdated { .. } => 0x0C,
            AuditKind::CalibrationInstalled { .. } => 0x0D,
            AuditKind::CalibrationUpdated { .. } => 0x0E,
            AuditKind::DriftBaselineFrozen { .. } => 0x0F,
            // Phase 0.3d-3 structural-action events (`0x16` Unfreeze
            // arrives in 0.3d-4).
            AuditKind::Grow { .. } => 0x10,
            AuditKind::Split { .. } => 0x11,
            AuditKind::Merge { .. } => 0x12,
            AuditKind::Prune { .. } => 0x13,
            AuditKind::Compress { .. } => 0x14,
            AuditKind::Freeze { .. } => 0x15,
            AuditKind::Unfreeze { .. } => 0x16,
            AuditKind::ExpectedEpistemicCaptured { .. } => 0x17,
            AuditKind::BlrNumericalRescue { .. } => 0x18,
            AuditKind::LeafParamsUpdatedBatch { .. } => 0x19,
            // Phase 0.4 Track A — Routed (opt-in trace event) and
            // StatsSnapshot (log-compaction marker).
            AuditKind::Routed { .. } => 0x1B,
            AuditKind::StatsSnapshot { .. } => 0x1A,
            // Phase 0.5 Item 1 — opt-in provenance stamping. 36-byte
            // canonical body (node_id u32 BE + hash [u8; 32]).
            AuditKind::ProvenanceStamped { .. } => 0x1C,
            // Phase 0.6 Item 4 — batched BeliefUpdate. Variable-size
            // body (4 + 8*count + 32). Forces wire-format v13.
            AuditKind::BeliefUpdateBatch { .. } => 0x1D,
            // Phase 0.8c v14 Item A2 — fused per-row training step.
            // 40-byte body (value u64 BE + state_hash [u8; 32]).
            // Forces wire-format v14.
            AuditKind::TrainStep { .. } => 0x1E,
        }
    }
}

/// One entry in the audit log. The `new_hash` field is the running chain
/// head *after* this event has been applied.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Monotonic global sequence number. `seq == k` is the (k+1)-th event.
    pub seq: u64,
    /// Logical epoch in which the event was recorded.
    pub epoch: u64,
    /// Which node this event affects. For `NodeAdded` this is the
    /// *newly created* node id (the parent is in the payload).
    pub node_id: NodeId,
    /// What happened, including any payload required to replay it.
    pub kind: AuditKind,
    /// Per-node stats version *after* the event has been applied.
    /// For non-stats events (NodeAdded / ChildrenPromoted /
    /// CodebookFrozen) this is the affected node's *current* stats
    /// version (no change).
    pub stats_version: u64,
    /// SHA-256 of the post-update [`NodeStats`](crate::stats::NodeStats)
    /// of the affected node. For `CodebookFrozen` this is the root's
    /// stats hash (since codebook installation doesn't bind to a single
    /// node, but every event carries a node-context hash for shape
    /// uniformity).
    pub stats_hash: [u8; 32],
    /// Chain head *before* this event.
    pub previous_hash: [u8; 32],
    /// `sha256(previous_hash ‖ canonical_payload(self))`. The replay path
    /// recomputes this from the running chain and compares for equality.
    pub new_hash: [u8; 32],
}

impl AuditEvent {
    /// Canonical byte encoding of the event payload (everything except
    /// `previous_hash` and `new_hash`).
    ///
    /// Layout (all big-endian) for the *common header*:
    /// ```text
    ///   [0..8]    seq            u64
    ///   [8..16]   epoch          u64
    ///   [16..20]  node_id        u32
    ///   [20..21]  kind tag       u8
    /// ```
    /// then a kind-specific body:
    /// ```text
    ///   Created           : (no body)
    ///   BeliefUpdate      : value.to_bits  u64
    ///   NodeAdded         : parent u32 + key_byte u8
    ///   ChildrenPromoted  : from u8 + to u8
    ///   CodebookFrozen    : codebook_hash [u8; 32]
    /// ```
    /// then unconditionally:
    /// ```text
    ///   stats_version          u64
    ///   stats_hash             [u8; 32]
    /// ```
    /// The hash chain step is `new_hash = sha256(previous_hash ‖ payload)`.
    ///
    /// Allocates a fresh `Vec<u8>`. For hot loops (chain verification,
    /// snapshot serialization) prefer [`write_payload`](Self::write_payload),
    /// which writes into a caller-provided buffer reused across the
    /// whole loop.
    pub fn payload_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96);
        self.write_payload(&mut out);
        out
    }

    /// Write the payload bytes into a caller-provided `Vec<u8>` buffer.
    ///
    /// Phase 0.7 (A) — buffer-reuse variant of [`payload_bytes`](Self::payload_bytes).
    /// Clears `out` at entry, then appends the same bytes that
    /// `payload_bytes` would have returned. Used by `verify_chain` and
    /// `serialize` to amortize the allocation across N audit events.
    ///
    /// Determinism: byte output is identical to `payload_bytes` —
    /// they share this implementation. Verified by the in-crate parity
    /// test `write_payload_matches_payload_bytes`.
    pub fn write_payload(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.seq.to_be_bytes());
        out.extend_from_slice(&self.epoch.to_be_bytes());
        out.extend_from_slice(&self.node_id.to_be_bytes());
        out.push(self.kind.tag());
        match &self.kind {
            AuditKind::Created => {}
            AuditKind::BeliefUpdate { value } => {
                out.extend_from_slice(&value.to_bits().to_be_bytes());
            }
            AuditKind::NodeAdded { parent, key_byte } => {
                out.extend_from_slice(&parent.to_be_bytes());
                out.push(*key_byte);
            }
            AuditKind::ChildrenPromoted { from, to } => {
                out.push(*from);
                out.push(*to);
            }
            AuditKind::CodebookFrozen { codebook_hash } => {
                out.extend_from_slice(codebook_hash);
            }
            AuditKind::LeafHeadConfigured { config_hash } => {
                out.extend_from_slice(config_hash);
            }
            AuditKind::LeafParamsInitialized { params_hash } => {
                out.extend_from_slice(params_hash);
            }
            AuditKind::LeafParamsUpdated { params_hash } => {
                out.extend_from_slice(params_hash);
            }
            AuditKind::BlrPriorConfigured { config_hash } => {
                out.extend_from_slice(config_hash);
            }
            AuditKind::BlrInitialized { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::BlrUpdated { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::DensityTrackerInstalled { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::DensityUpdated { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::CalibrationInstalled { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::CalibrationUpdated { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::DriftBaselineFrozen { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::ExpectedEpistemicCaptured { state_hash } => {
                out.extend_from_slice(state_hash);
            }
            AuditKind::Grow {
                parent,
                key_byte,
                child,
            } => {
                out.extend_from_slice(&parent.to_be_bytes());
                out.push(*key_byte);
                out.extend_from_slice(&child.to_be_bytes());
            }
            AuditKind::Split {
                parent,
                child_a,
                child_b,
            } => {
                out.extend_from_slice(&parent.to_be_bytes());
                out.extend_from_slice(&child_a.to_be_bytes());
                out.extend_from_slice(&child_b.to_be_bytes());
            }
            AuditKind::Merge { absorbed, into } => {
                out.extend_from_slice(&absorbed.to_be_bytes());
                out.extend_from_slice(&into.to_be_bytes());
            }
            AuditKind::Prune { node_id } => {
                out.extend_from_slice(&node_id.to_be_bytes());
            }
            AuditKind::Compress { signature } => {
                out.extend_from_slice(signature);
            }
            AuditKind::Freeze { node_id } => {
                out.extend_from_slice(&node_id.to_be_bytes());
            }
            AuditKind::Unfreeze { node_id } => {
                out.extend_from_slice(&node_id.to_be_bytes());
            }
            AuditKind::BlrNumericalRescue {
                reason,
                b_pre_clamp_bits,
            } => {
                out.push(*reason);
                out.extend_from_slice(&b_pre_clamp_bits.to_be_bytes());
            }
            AuditKind::LeafParamsUpdatedBatch { params_hash } => {
                out.extend_from_slice(params_hash);
            }
            AuditKind::Routed {
                leaf,
                matched_prefix,
            } => {
                // 5-byte body: leaf u32 BE + matched_prefix u8.
                out.extend_from_slice(&leaf.to_be_bytes());
                out.push(*matched_prefix);
            }
            AuditKind::StatsSnapshot {
                node_id,
                stats_hash,
            } => {
                // 36-byte body: node_id u32 BE + stats_hash [u8; 32].
                out.extend_from_slice(&node_id.to_be_bytes());
                out.extend_from_slice(stats_hash);
            }
            AuditKind::ProvenanceStamped { node_id, hash } => {
                // Phase 0.5 Item 1 — 36-byte body: node_id u32 BE +
                // hash [u8; 32].
                out.extend_from_slice(&node_id.to_be_bytes());
                out.extend_from_slice(hash);
            }
            AuditKind::BeliefUpdateBatch {
                count,
                batch_hash,
                values,
            } => {
                // Phase 0.6 Item 4 — variable body:
                //   count u32 BE (4)
                //   values f64×count (8 * count, each .to_bits().to_be_bytes())
                //   batch_hash [u8; 32] (32)
                out.extend_from_slice(&count.to_be_bytes());
                for v in values {
                    out.extend_from_slice(&v.to_bits().to_be_bytes());
                }
                out.extend_from_slice(batch_hash);
            }
            AuditKind::TrainStep { value, state_hash } => {
                // Phase 0.8c v14 Item A2 — fused per-row training step.
                // 40-byte body: value f64 BE (.to_bits()) + state_hash [u8; 32].
                out.extend_from_slice(&value.to_bits().to_be_bytes());
                out.extend_from_slice(state_hash);
            }
        }
        out.extend_from_slice(&self.stats_version.to_be_bytes());
        out.extend_from_slice(&self.stats_hash);
    }

    /// Compute the chain step `new_hash = sha256(previous_hash ‖ payload)`.
    ///
    /// Phase 0.7 (C) — implemented via the streaming
    /// [`cjc_snap::hash::Sha256`] hasher, which feeds `previous_hash`
    /// and `payload` into the SHA-256 state directly without first
    /// concatenating them into an intermediate `Vec<u8>`. The output
    /// digest is byte-identical to the pre-0.7 form (verified in
    /// `cjc-snap` by `streaming_matches_concat_one_shot_for_audit_pattern`),
    /// so the 28 locked audit chain canaries are preserved.
    pub fn compute_new_hash(previous_hash: &[u8; 32], payload: &[u8]) -> [u8; 32] {
        let mut hasher = cjc_snap::hash::Sha256::new();
        hasher.update(previous_hash);
        hasher.update(payload);
        hasher.finalize()
    }

    /// Recompute `new_hash` from `previous_hash` and the current payload.
    /// Used by [`crate::graph::AdaptiveBeliefGraph::verify_chain`].
    pub fn recompute_new_hash(&self) -> [u8; 32] {
        Self::compute_new_hash(&self.previous_hash, &self.payload_bytes())
    }
}

// ─── Phase 0.8 Item B4 — columnar audit-log storage ────────────────────────
//
// `AuditLog` replaces `Vec<AuditEvent>` as the in-memory audit-log
// representation on `AdaptiveBeliefGraph`. The scalar fields live in
// per-column `Vec`s ("struct-of-arrays" / SoA), which gives:
//
// * **Contiguous slices for batch ops.** `previous_hashes()` /
//   `new_hashes()` / `seqs()` return `&[..]` directly, no
//   re-materialization. This is the prerequisite for Item A3
//   (Merkle-indexed audit chain) and Item C2 (parallel verify_chain).
// * **Better cache utilization on full-log scans** for callers that
//   only need a subset of fields.
//
// Wire format unchanged: all 28 SHA-256 canaries remain valid because
// `serialize_into` writes the same bytes regardless of in-memory
// layout. The compatibility shape — `push(event)`, `iter()`, `len()`,
// `last()` — matches `Vec<AuditEvent>` closely enough that most
// existing call sites need no change. The one mechanical exception
// is `audit[i]`, which has no clean SoA equivalent (Rust's `Index`
// requires `&Output`); use `audit.get(i)` instead.

/// Columnar audit-log storage (Phase 0.8 Item B4).
///
/// Scalar fields are stored in per-column `Vec`s so batch operations
/// (Merkle indexing, parallel verify) can borrow contiguous slices
/// without materializing intermediate `AuditEvent` values. The variant
/// payload (`AuditKind`) stays AoS because it is heterogeneous.
///
/// Public API mirrors `Vec<AuditEvent>` where possible. `push` accepts
/// a fully-constructed `AuditEvent` and decomposes it; `iter` /
/// `last` / `get` materialize an owned `AuditEvent` per access (one
/// `AuditKind::clone`). Hot-path callers that don't need the full
/// `AuditEvent` should use the columnar accessors directly.
#[derive(Debug, Clone, Default)]
pub struct AuditLog {
    seq: Vec<u64>,
    epoch: Vec<u64>,
    node_id: Vec<NodeId>,
    stats_version: Vec<u64>,
    stats_hash: Vec<[u8; 32]>,
    previous_hash: Vec<[u8; 32]>,
    new_hash: Vec<[u8; 32]>,
    kind: Vec<AuditKind>,
}

impl AuditLog {
    /// An empty audit log with no allocation.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of events currently in the log.
    #[inline]
    pub fn len(&self) -> usize {
        self.seq.len()
    }

    /// `true` if no events have been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Append an event by decomposing into the column vectors.
    ///
    /// Each column gets one push; the eight `Vec`s stay length-aligned
    /// by construction. `kind` is moved (no clone).
    pub fn push(&mut self, event: AuditEvent) {
        self.seq.push(event.seq);
        self.epoch.push(event.epoch);
        self.node_id.push(event.node_id);
        self.stats_version.push(event.stats_version);
        self.stats_hash.push(event.stats_hash);
        self.previous_hash.push(event.previous_hash);
        self.new_hash.push(event.new_hash);
        self.kind.push(event.kind);
    }

    /// Truncate the log to `len` events. If `len` exceeds the current
    /// length, this is a no-op.
    pub fn truncate(&mut self, len: usize) {
        self.seq.truncate(len);
        self.epoch.truncate(len);
        self.node_id.truncate(len);
        self.stats_version.truncate(len);
        self.stats_hash.truncate(len);
        self.previous_hash.truncate(len);
        self.new_hash.truncate(len);
        self.kind.truncate(len);
    }

    /// Materialize the event at index `i` as an owned `AuditEvent`.
    /// Returns `None` if `i >= len()`.
    pub fn get(&self, i: usize) -> Option<AuditEvent> {
        if i >= self.len() {
            return None;
        }
        Some(AuditEvent {
            seq: self.seq[i],
            epoch: self.epoch[i],
            node_id: self.node_id[i],
            kind: self.kind[i].clone(),
            stats_version: self.stats_version[i],
            stats_hash: self.stats_hash[i],
            previous_hash: self.previous_hash[i],
            new_hash: self.new_hash[i],
        })
    }

    /// The last event, materialized as an owned `AuditEvent`. `None`
    /// if the log is empty.
    pub fn last(&self) -> Option<AuditEvent> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Iterate the log, yielding owned `AuditEvent`s. Each step clones
    /// the variant payload. The returned iterator implements
    /// `DoubleEndedIterator` and `ExactSizeIterator`, so `.rev()`
    /// works as a drop-in replacement for `Vec::iter().rev()`.
    pub fn iter(&self) -> AuditLogIter<'_> {
        AuditLogIter {
            log: self,
            range: 0..self.len(),
        }
    }

    // ── Columnar accessors ────────────────────────────────────────────
    //
    // Zero-copy `&[..]` views into individual columns. Enables future
    // Merkle layer construction (Item A3) and parallel verify
    // (Item C2) to borrow contiguous slices instead of materializing
    // intermediate `Vec`s from an AoS iter.

    /// Slice of every event's pre-event chain head.
    #[inline]
    pub fn previous_hashes(&self) -> &[[u8; 32]] {
        &self.previous_hash
    }

    /// Slice of every event's post-event chain head (the "new_hash"
    /// field). This is the column Merkle indexing layers over.
    #[inline]
    pub fn new_hashes(&self) -> &[[u8; 32]] {
        &self.new_hash
    }

    /// Slice of every event's monotonic global sequence number.
    #[inline]
    pub fn seqs(&self) -> &[u64] {
        &self.seq
    }

    /// Slice of every event's logical epoch.
    #[inline]
    pub fn epochs(&self) -> &[u64] {
        &self.epoch
    }

    /// Slice of every event's affected `NodeId`.
    #[inline]
    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_id
    }

    /// Slice of every event's recorded post-update `stats_version`.
    #[inline]
    pub fn stats_versions(&self) -> &[u64] {
        &self.stats_version
    }

    /// Slice of every event's recorded post-update `stats_hash`.
    #[inline]
    pub fn stats_hashes(&self) -> &[[u8; 32]] {
        &self.stats_hash
    }

    /// Slice of every event's variant. Heterogeneous; stays AoS.
    #[inline]
    pub fn kinds(&self) -> &[AuditKind] {
        &self.kind
    }

    // ── Mutable columnar accessors ────────────────────────────────────
    //
    // Used by:
    //   * Replay-invariant tests that forge tampered audit logs and
    //     verify the new-hash chain still validates after the
    //     `rebuild_chain` helper recomputes it.
    //   * Future Phase 0.8c work that splices in `StatsSnapshot` events
    //     and re-derives the chain head.
    //
    // The eight column `Vec`s stay length-aligned only by careful
    // discipline at the call site. These accessors deliberately do NOT
    // resize the column they expose; callers that need to add/remove
    // events should go through `push` / `truncate` instead.

    /// Mutable slice of every event's pre-event chain head.
    #[inline]
    pub fn previous_hashes_mut(&mut self) -> &mut [[u8; 32]] {
        &mut self.previous_hash
    }

    /// Mutable slice of every event's post-event chain head.
    #[inline]
    pub fn new_hashes_mut(&mut self) -> &mut [[u8; 32]] {
        &mut self.new_hash
    }

    /// Mutable slice of every event's monotonic global sequence number.
    #[inline]
    pub fn seqs_mut(&mut self) -> &mut [u64] {
        &mut self.seq
    }

    /// Mutable slice of every event's logical epoch.
    #[inline]
    pub fn epochs_mut(&mut self) -> &mut [u64] {
        &mut self.epoch
    }

    /// Mutable slice of every event's affected `NodeId`.
    #[inline]
    pub fn node_ids_mut(&mut self) -> &mut [NodeId] {
        &mut self.node_id
    }

    /// Mutable slice of every event's `stats_version`.
    #[inline]
    pub fn stats_versions_mut(&mut self) -> &mut [u64] {
        &mut self.stats_version
    }

    /// Mutable slice of every event's `stats_hash`.
    #[inline]
    pub fn stats_hashes_mut(&mut self) -> &mut [[u8; 32]] {
        &mut self.stats_hash
    }

    /// Mutable slice of every event's variant.
    #[inline]
    pub fn kinds_mut(&mut self) -> &mut [AuditKind] {
        &mut self.kind
    }

    /// Swap the events at indices `i` and `j` across every column.
    /// Panics if either index is out of bounds.
    pub fn swap(&mut self, i: usize, j: usize) {
        self.seq.swap(i, j);
        self.epoch.swap(i, j);
        self.node_id.swap(i, j);
        self.stats_version.swap(i, j);
        self.stats_hash.swap(i, j);
        self.previous_hash.swap(i, j);
        self.new_hash.swap(i, j);
        self.kind.swap(i, j);
    }

    /// Replace the event at index `i` with `event`, decomposing into
    /// the column vectors. Panics if `i >= len()`.
    pub fn set(&mut self, i: usize, event: AuditEvent) {
        assert!(i < self.len(), "AuditLog::set: index {i} out of bounds (len={})", self.len());
        self.seq[i] = event.seq;
        self.epoch[i] = event.epoch;
        self.node_id[i] = event.node_id;
        self.stats_version[i] = event.stats_version;
        self.stats_hash[i] = event.stats_hash;
        self.previous_hash[i] = event.previous_hash;
        self.new_hash[i] = event.new_hash;
        self.kind[i] = event.kind;
    }
}

/// Owned-yielding iterator over an [`AuditLog`]. Each step clones the
/// variant payload to materialize an owned [`AuditEvent`]. Implements
/// [`DoubleEndedIterator`] and [`ExactSizeIterator`] so `.rev()`,
/// `.len()`, etc. work as drop-in replacements for the previous
/// `Vec<AuditEvent>::iter()` shape.
pub struct AuditLogIter<'a> {
    log: &'a AuditLog,
    range: std::ops::Range<usize>,
}

impl<'a> Iterator for AuditLogIter<'a> {
    type Item = AuditEvent;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().and_then(|i| self.log.get(i))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a> DoubleEndedIterator for AuditLogIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().and_then(|i| self.log.get(i))
    }
}

impl<'a> ExactSizeIterator for AuditLogIter<'a> {}

impl<'a> IntoIterator for &'a AuditLog {
    type Item = AuditEvent;
    type IntoIter = AuditLogIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::children::ChildrenKind;
    use crate::genesis_hash;

    fn dummy_event(seq: u64, kind: AuditKind, prev: [u8; 32]) -> AuditEvent {
        let stats_hash = cjc_snap::hash::sha256(b"stats");
        let mut e = AuditEvent {
            seq,
            epoch: 0,
            node_id: 0,
            kind,
            stats_version: seq + 1,
            stats_hash,
            previous_hash: prev,
            new_hash: [0u8; 32],
        };
        e.new_hash = AuditEvent::compute_new_hash(&prev, &e.payload_bytes());
        e
    }

    #[test]
    fn payload_size_belief_update() {
        // 8+8+4+1 (header) + 8 (value) + 8 (stats_version) + 32 (stats_hash) = 69
        let e = dummy_event(0, AuditKind::BeliefUpdate { value: 1.5 }, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 69);
    }

    // ── Phase 0.7 (A) — write_payload parity + buffer-reuse semantics ─

    #[test]
    fn write_payload_matches_payload_bytes_for_every_kind() {
        // The buffer-reuse variant must produce byte-identical output
        // to the allocating variant; otherwise verify_chain or
        // serialize would diverge from the chain witness, breaking the
        // 28 SHA-256 canaries.
        //
        // We exercise one event of each AuditKind tag (0x00..0x1D) so
        // any future variant that's added without mirroring its
        // payload to write_payload is caught.
        let prev = genesis_hash();
        let kinds = [
            AuditKind::Created,
            AuditKind::BeliefUpdate { value: 1.5 },
            AuditKind::NodeAdded { parent: 1, key_byte: 7 },
            AuditKind::ChildrenPromoted {
                from: ChildrenKind::None as u8,
                to: ChildrenKind::Node4 as u8,
            },
            AuditKind::CodebookFrozen { codebook_hash: [9u8; 32] },
            AuditKind::LeafHeadConfigured { config_hash: [1u8; 32] },
            AuditKind::LeafParamsInitialized { params_hash: [2u8; 32] },
            AuditKind::LeafParamsUpdated { params_hash: [3u8; 32] },
            AuditKind::BlrPriorConfigured { config_hash: [4u8; 32] },
            AuditKind::BlrInitialized { state_hash: [5u8; 32] },
            AuditKind::BlrUpdated { state_hash: [6u8; 32] },
            AuditKind::DensityTrackerInstalled { state_hash: [7u8; 32] },
            AuditKind::DensityUpdated { state_hash: [8u8; 32] },
            AuditKind::CalibrationInstalled { state_hash: [10u8; 32] },
            AuditKind::CalibrationUpdated { state_hash: [11u8; 32] },
            AuditKind::DriftBaselineFrozen { state_hash: [12u8; 32] },
            AuditKind::ExpectedEpistemicCaptured { state_hash: [13u8; 32] },
            AuditKind::Grow {
                parent: 1,
                key_byte: 2,
                child: 3,
            },
            AuditKind::Split {
                parent: 1,
                child_a: 2,
                child_b: 3,
            },
            AuditKind::Merge { absorbed: 1, into: 2 },
            AuditKind::Prune { node_id: 5 },
            AuditKind::Compress { signature: [14u8; 32] },
            AuditKind::Freeze { node_id: 6 },
            AuditKind::Unfreeze { node_id: 7 },
            AuditKind::BlrNumericalRescue {
                reason: 0,
                b_pre_clamp_bits: 0xdeadbeefu64,
            },
            AuditKind::LeafParamsUpdatedBatch {
                params_hash: [15u8; 32],
            },
            AuditKind::Routed {
                leaf: 9,
                matched_prefix: 4,
            },
            AuditKind::StatsSnapshot {
                node_id: 10,
                stats_hash: [16u8; 32],
            },
            AuditKind::ProvenanceStamped {
                node_id: 11,
                hash: [17u8; 32],
            },
            AuditKind::BeliefUpdateBatch {
                count: 3,
                batch_hash: [18u8; 32],
                values: vec![1.5, -2.5, 3.5],
            },
            AuditKind::TrainStep {
                value: 1.5,
                state_hash: [19u8; 32],
            },
        ];
        let mut buf = Vec::new();
        for (i, kind) in kinds.into_iter().enumerate() {
            let e = dummy_event(i as u64, kind, prev);
            let direct = e.payload_bytes();
            e.write_payload(&mut buf);
            assert_eq!(direct, buf, "mismatch at kind index {i}");
        }
    }

    #[test]
    fn write_payload_clears_existing_contents() {
        // Buffer reuse contract: clear-on-entry, so callers can pass a
        // dirty buffer and still get the canonical bytes out.
        let e = dummy_event(0, AuditKind::Created, genesis_hash());
        let direct = e.payload_bytes();
        let mut buf = vec![99u8; 200];
        e.write_payload(&mut buf);
        assert_eq!(buf, direct);
    }

    #[test]
    fn write_payload_reuse_is_byte_stable() {
        // Two `write_payload` calls on the same buffer with the same
        // event must produce identical bytes — proves the clear is
        // correct (no stale bytes leak across calls).
        let e = dummy_event(0, AuditKind::BeliefUpdate { value: 1.5 }, genesis_hash());
        let mut buf = Vec::with_capacity(96);
        e.write_payload(&mut buf);
        let first = buf.clone();
        e.write_payload(&mut buf);
        assert_eq!(first, buf);
    }

    #[test]
    fn payload_size_created() {
        // 21 + 0 + 8 + 32 = 61
        let e = dummy_event(0, AuditKind::Created, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 61);
    }

    #[test]
    fn payload_size_node_added() {
        // 21 + 5 (parent + key_byte) + 8 + 32 = 66
        let e = dummy_event(
            0,
            AuditKind::NodeAdded { parent: 0, key_byte: 7 },
            genesis_hash(),
        );
        assert_eq!(e.payload_bytes().len(), 66);
    }

    #[test]
    fn payload_size_children_promoted() {
        // 21 + 2 (from + to) + 8 + 32 = 63
        let e = dummy_event(
            0,
            AuditKind::ChildrenPromoted {
                from: ChildrenKind::Node4 as u8,
                to: ChildrenKind::Node16 as u8,
            },
            genesis_hash(),
        );
        assert_eq!(e.payload_bytes().len(), 63);
    }

    #[test]
    fn payload_size_codebook_frozen() {
        // 21 + 32 (codebook_hash) + 8 + 32 = 93
        let e = dummy_event(
            0,
            AuditKind::CodebookFrozen {
                codebook_hash: [42u8; 32],
            },
            genesis_hash(),
        );
        assert_eq!(e.payload_bytes().len(), 93);
    }

    #[test]
    fn payload_size_provenance_stamped() {
        // Phase 0.5 Item 1 — 21 (header) + 36 (node_id u32 + hash [u8; 32])
        // + 8 (stats_version) + 32 (stats_hash) = 97
        let e = dummy_event(
            0,
            AuditKind::ProvenanceStamped {
                node_id: 7,
                hash: [0xABu8; 32],
            },
            genesis_hash(),
        );
        assert_eq!(e.payload_bytes().len(), 97);
    }

    #[test]
    fn provenance_stamped_tag_is_0x1c() {
        let k = AuditKind::ProvenanceStamped {
            node_id: 0,
            hash: [0u8; 32],
        };
        assert_eq!(k.tag(), 0x1C);
    }

    #[test]
    fn tamper_detection_provenance_hash() {
        let mut e = dummy_event(
            7,
            AuditKind::ProvenanceStamped {
                node_id: 3,
                hash: [0x01u8; 32],
            },
            genesis_hash(),
        );
        e.kind = AuditKind::ProvenanceStamped {
            node_id: 3,
            hash: [0x02u8; 32],
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    #[test]
    fn recompute_round_trip() {
        let e = dummy_event(7, AuditKind::BeliefUpdate { value: 1.5 }, genesis_hash());
        assert_eq!(e.recompute_new_hash(), e.new_hash);
    }

    #[test]
    fn tamper_detection_value() {
        let mut e = dummy_event(7, AuditKind::BeliefUpdate { value: 1.5 }, genesis_hash());
        e.kind = AuditKind::BeliefUpdate { value: 2.5 };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    #[test]
    fn tamper_detection_node_added_parent() {
        let mut e = dummy_event(
            7,
            AuditKind::NodeAdded { parent: 0, key_byte: 7 },
            genesis_hash(),
        );
        e.kind = AuditKind::NodeAdded { parent: 1, key_byte: 7 };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    #[test]
    fn chain_links_two_events() {
        let e0 = dummy_event(0, AuditKind::Created, genesis_hash());
        let e1 = dummy_event(
            1,
            AuditKind::NodeAdded { parent: 0, key_byte: 3 },
            e0.new_hash,
        );
        assert_eq!(e1.previous_hash, e0.new_hash);
        assert_eq!(e1.recompute_new_hash(), e1.new_hash);
    }

    #[test]
    fn expected_epistemic_captured_tag_and_payload_size() {
        // Tag is 0x17 — deliberately past the 0x10..0x16 block reserved
        // for Phase 0.3d-3/4 structural actions.
        let kind = AuditKind::ExpectedEpistemicCaptured {
            state_hash: [9u8; 32],
        };
        assert_eq!(kind.tag(), 0x17);
        // 21 (header) + 32 (state_hash) + 8 (stats_version) + 32 (stats_hash) = 93
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 93);
    }

    #[test]
    fn expected_epistemic_tamper_detection() {
        let mut e = dummy_event(
            7,
            AuditKind::ExpectedEpistemicCaptured {
                state_hash: [1u8; 32],
            },
            genesis_hash(),
        );
        e.kind = AuditKind::ExpectedEpistemicCaptured {
            state_hash: [2u8; 32],
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    // ── Phase 0.3d-3: structural-action audit kinds ──────────────

    #[test]
    fn grow_tag_and_payload_size() {
        let kind = AuditKind::Grow {
            parent: 1,
            key_byte: 7,
            child: 2,
        };
        assert_eq!(kind.tag(), 0x10);
        // 21 (header) + 9 (parent u32 + key u8 + child u32) + 8 (stats_version) + 32 (stats_hash) = 70
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 70);
    }

    #[test]
    fn split_tag_and_payload_size() {
        let kind = AuditKind::Split {
            parent: 1,
            child_a: 2,
            child_b: 3,
        };
        assert_eq!(kind.tag(), 0x11);
        // 21 + 12 (3 × u32) + 8 + 32 = 73
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 73);
    }

    #[test]
    fn merge_tag_and_payload_size() {
        let kind = AuditKind::Merge {
            absorbed: 5,
            into: 1,
        };
        assert_eq!(kind.tag(), 0x12);
        // 21 + 8 (2 × u32) + 8 + 32 = 69
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 69);
    }

    #[test]
    fn prune_tag_and_payload_size() {
        let kind = AuditKind::Prune { node_id: 7 };
        assert_eq!(kind.tag(), 0x13);
        // 21 + 4 (u32) + 8 + 32 = 65
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 65);
    }

    #[test]
    fn compress_tag_and_payload_size() {
        let kind = AuditKind::Compress {
            signature: [9u8; 32],
        };
        assert_eq!(kind.tag(), 0x14);
        // 21 + 32 + 8 + 32 = 93
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 93);
    }

    #[test]
    fn freeze_tag_and_payload_size() {
        let kind = AuditKind::Freeze { node_id: 4 };
        assert_eq!(kind.tag(), 0x15);
        // 21 + 4 + 8 + 32 = 65
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 65);
    }

    #[test]
    fn structural_tags_distinct() {
        // Pin the 0x10..0x17 mapping so a future change must update this test.
        assert_eq!(AuditKind::Grow { parent: 0, key_byte: 0, child: 0 }.tag(), 0x10);
        assert_eq!(AuditKind::Split { parent: 0, child_a: 0, child_b: 0 }.tag(), 0x11);
        assert_eq!(AuditKind::Merge { absorbed: 0, into: 0 }.tag(), 0x12);
        assert_eq!(AuditKind::Prune { node_id: 0 }.tag(), 0x13);
        assert_eq!(AuditKind::Compress { signature: [0u8; 32] }.tag(), 0x14);
        assert_eq!(AuditKind::Freeze { node_id: 0 }.tag(), 0x15);
        assert_eq!(AuditKind::Unfreeze { node_id: 0 }.tag(), 0x16);
        assert_eq!(
            AuditKind::ExpectedEpistemicCaptured { state_hash: [0u8; 32] }.tag(),
            0x17
        );
    }

    #[test]
    fn unfreeze_tag_and_payload_size() {
        let kind = AuditKind::Unfreeze { node_id: 4 };
        assert_eq!(kind.tag(), 0x16);
        // 21 (header) + 4 (u32) + 8 (stats_version) + 32 (stats_hash) = 65
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 65);
    }

    #[test]
    fn unfreeze_tamper_detection() {
        let mut e = dummy_event(
            7,
            AuditKind::Unfreeze { node_id: 1 },
            genesis_hash(),
        );
        e.kind = AuditKind::Unfreeze { node_id: 2 };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    #[test]
    fn grow_tamper_detection() {
        let mut e = dummy_event(
            7,
            AuditKind::Grow {
                parent: 0,
                key_byte: 1,
                child: 1,
            },
            genesis_hash(),
        );
        e.kind = AuditKind::Grow {
            parent: 0,
            key_byte: 2,
            child: 1,
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    // ── Phase 0.4 Track C-2.3.4: BlrNumericalRescue ───────────────────

    #[test]
    fn blr_numerical_rescue_tag_and_payload_size() {
        let kind = AuditKind::BlrNumericalRescue {
            reason: BLR_RESCUE_B_BELOW_EPSILON,
            b_pre_clamp_bits: 0u64,
        };
        assert_eq!(kind.tag(), 0x18);
        // 21 (header) + 1 (reason) + 8 (b_pre_clamp_bits)
        //   + 8 (stats_version) + 32 (stats_hash) = 70
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 70);
    }

    #[test]
    fn blr_numerical_rescue_payload_changes_with_pre_clamp() {
        // Pin the wire ordering of the body fields (reason u8 then
        // b_pre_clamp_bits u64 BE).
        let a = AuditKind::BlrNumericalRescue {
            reason: 0,
            b_pre_clamp_bits: 0u64,
        };
        let b = AuditKind::BlrNumericalRescue {
            reason: 0,
            b_pre_clamp_bits: 0x0102030405060708u64,
        };
        let ea = dummy_event(0, a, genesis_hash());
        let eb = dummy_event(0, b, genesis_hash());
        assert_ne!(ea.payload_bytes(), eb.payload_bytes());

        // The 8 reason+pre_clamp_bits bytes start at offset 21 (after
        // seq u64 + epoch u64 + node_id u32 + tag u8); reason is at
        // 21, body at 22..30 in big-endian.
        let bytes_b = eb.payload_bytes();
        assert_eq!(bytes_b[21], 0); // reason
        assert_eq!(
            &bytes_b[22..30],
            &0x0102030405060708u64.to_be_bytes()
        );
    }

    #[test]
    fn blr_numerical_rescue_tamper_detection() {
        let mut e = dummy_event(
            5,
            AuditKind::BlrNumericalRescue {
                reason: 0,
                b_pre_clamp_bits: 0x1234u64,
            },
            genesis_hash(),
        );
        // Tamper the pre-clamp bits — chain hash should diverge.
        e.kind = AuditKind::BlrNumericalRescue {
            reason: 0,
            b_pre_clamp_bits: 0x5678u64,
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    // ── Phase 0.4 Track C-2.3.6: LeafParamsUpdatedBatch ───────────────

    #[test]
    fn leaf_params_updated_batch_tag_and_payload_size() {
        let kind = AuditKind::LeafParamsUpdatedBatch {
            params_hash: [0u8; 32],
        };
        assert_eq!(kind.tag(), 0x19);
        // 21 (header) + 32 (params_hash) + 8 (stats_version)
        //   + 32 (stats_hash) = 93
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 93);
    }

    #[test]
    fn leaf_params_updated_batch_tamper_detection() {
        let mut e = dummy_event(
            5,
            AuditKind::LeafParamsUpdatedBatch {
                params_hash: [0xAA; 32],
            },
            genesis_hash(),
        );
        e.kind = AuditKind::LeafParamsUpdatedBatch {
            params_hash: [0xBB; 32],
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    // ── Phase 0.8c v14 Item A2: TrainStep ─────────────────────────────

    #[test]
    fn train_step_tag_is_0x1e() {
        let k = AuditKind::TrainStep {
            value: 0.0,
            state_hash: [0u8; 32],
        };
        assert_eq!(k.tag(), 0x1E);
    }

    #[test]
    fn train_step_payload_size() {
        // 21 (header) + 40 (value u64 + state_hash [u8; 32])
        //   + 8 (stats_version) + 32 (stats_hash) = 101.
        let kind = AuditKind::TrainStep {
            value: 1.5,
            state_hash: [9u8; 32],
        };
        let e = dummy_event(0, kind, genesis_hash());
        assert_eq!(e.payload_bytes().len(), 101);
    }

    #[test]
    fn train_step_payload_encodes_value_then_state_hash() {
        // Pin the wire ordering of the body fields: value bits at
        // [21..29], state_hash at [29..61]. This is what every replay
        // path depends on.
        let value = 1.5f64;
        let state_hash = [0xABu8; 32];
        let e = dummy_event(
            0,
            AuditKind::TrainStep { value, state_hash },
            genesis_hash(),
        );
        let bytes = e.payload_bytes();
        assert_eq!(&bytes[21..29], &value.to_bits().to_be_bytes());
        assert_eq!(&bytes[29..61], &state_hash);
    }

    #[test]
    fn train_step_tamper_detection_value() {
        let mut e = dummy_event(
            7,
            AuditKind::TrainStep {
                value: 1.5,
                state_hash: [1u8; 32],
            },
            genesis_hash(),
        );
        e.kind = AuditKind::TrainStep {
            value: 2.5,
            state_hash: [1u8; 32],
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }

    #[test]
    fn train_step_tamper_detection_state_hash() {
        let mut e = dummy_event(
            7,
            AuditKind::TrainStep {
                value: 1.5,
                state_hash: [1u8; 32],
            },
            genesis_hash(),
        );
        e.kind = AuditKind::TrainStep {
            value: 1.5,
            state_hash: [2u8; 32],
        };
        assert_ne!(e.recompute_new_hash(), e.new_hash);
    }
}
