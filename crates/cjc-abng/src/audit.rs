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
            // Phase 0.4 Track A — Routed (opt-in trace event). Tag
            // 0x1A is reserved for `StatsSnapshot` (G3.7); 0x1C is
            // reserved for `ProvenanceStamped` (Phase 0.5).
            AuditKind::Routed { .. } => 0x1B,
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
    pub fn payload_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96);
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
        }
        out.extend_from_slice(&self.stats_version.to_be_bytes());
        out.extend_from_slice(&self.stats_hash);
        out
    }

    /// Compute the chain step `new_hash = sha256(previous_hash ‖ payload)`.
    pub fn compute_new_hash(previous_hash: &[u8; 32], payload: &[u8]) -> [u8; 32] {
        let mut buf = Vec::with_capacity(32 + payload.len());
        buf.extend_from_slice(previous_hash);
        buf.extend_from_slice(payload);
        cjc_snap::hash::sha256(&buf)
    }

    /// Recompute `new_hash` from `previous_hash` and the current payload.
    /// Used by [`crate::graph::AdaptiveBeliefGraph::verify_chain`].
    pub fn recompute_new_hash(&self) -> [u8; 32] {
        Self::compute_new_hash(&self.previous_hash, &self.payload_bytes())
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
}
