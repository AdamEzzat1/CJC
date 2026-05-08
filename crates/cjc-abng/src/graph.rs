//! [`AdaptiveBeliefGraph`] — the central arena type.
//!
//! Phase 0.2 surface:
//!
//! * `observe(node_id, value)` — Welford-update node stats; appends a
//!   `BeliefUpdate` event to the audit log; advances the per-node stats
//!   chain.
//! * `add_node(parent, key_byte) -> NodeId` — bind a new child to its
//!   parent's `AdaptiveChildren`; appends `NodeAdded` (and
//!   `ChildrenPromoted` when promotion fires).
//! * `set_codebook(...)` — install the prefix-encoder quantile codebook;
//!   one-shot, frozen on first install.
//! * `encode_prefix(x)` — encode an input vector to a prefix using the
//!   installed codebook.
//! * `descend(prefix) -> RouteEvidence` — walk root-to-leaf matching
//!   prefix bytes.
//! * `verify_chain()` — recompute every event's `new_hash` and confirm
//!   the chain is consistent.

use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;

use crate::audit::{AuditEvent, AuditKind, BLR_RESCUE_B_BELOW_EPSILON};
use crate::blr::{BlrError, BlrPrior, BlrState};
use crate::calibration::{CalibrationBins, CalibrationError};
use crate::children::{AdaptiveChildren, ChildrenKind};
use crate::codebook::{CodebookError, QuantileCodebook};
use crate::density::{DensityError, DensityTracker};
use crate::drift::{DriftBaseline, DriftError};
use crate::genesis_hash;
use crate::leaf_head::{expected_param_shape, init_params, params_hash, LeafHead, LeafHeadError};
use crate::node::{AdaptiveBeliefNode, NodeId};
use crate::policy::{DecisionPolicy, PolicyError};
use crate::route::RouteEvidence;
use crate::signature::NodeSignature;

/// Number of structural-action kinds tracked by [`AdaptiveBeliefGraph::action_counts`].
/// Indexed in tag order: 0=Grow, 1=Split, 2=Merge, 3=Prune, 4=Compress, 5=Freeze.
pub const N_ACTION_KINDS: usize = 6;

/// Numeric index for each structural action — matches its
/// position in [`AdaptiveBeliefGraph::action_counts`] and the offset
/// from the audit-tag baseline `0x10`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionKind {
    Grow = 0,
    Split = 1,
    Merge = 2,
    Prune = 3,
    Compress = 4,
    Freeze = 5,
}

impl ActionKind {
    /// Decode a `0..=5` index into an `ActionKind`. Used by
    /// `abng_action_count`'s i64 argument.
    pub fn from_index(i: u8) -> Option<Self> {
        Some(match i {
            0 => ActionKind::Grow,
            1 => ActionKind::Split,
            2 => ActionKind::Merge,
            3 => ActionKind::Prune,
            4 => ActionKind::Compress,
            5 => ActionKind::Freeze,
            _ => return None,
        })
    }
}

/// Errors returned by graph operations.
#[derive(Debug, PartialEq)]
pub enum GraphError {
    /// `node_id` did not refer to a node in this graph.
    NodeOutOfRange { node_id: NodeId, n_nodes: u32 },
    /// `add_node` was called with a `key_byte` already bound on the parent.
    KeyAlreadyBound { parent: NodeId, key_byte: u8 },
    /// `set_codebook` was called on a graph that already has a codebook.
    CodebookAlreadyFrozen,
    /// Codebook construction or encoding error.
    Codebook(CodebookError),
    /// `encode_prefix`/`descend` was called before a codebook was installed.
    NoCodebook,
    /// Leaf-head subsystem error.
    LeafHead(LeafHeadError),
    /// BLR subsystem error.
    Blr(BlrError),
    /// Density-tracker subsystem error.
    Density(DensityError),
    /// Calibration subsystem error.
    Calibration(CalibrationError),
    /// Drift-detector subsystem error.
    Drift(DriftError),
    /// The audit chain failed to verify; carries the index of the first
    /// event whose `new_hash` did not match the recomputed value.
    ChainBroken { at_seq: u64 },
    /// `set_expected_epistemic` was called on a node that does not have
    /// a BLR posterior installed.
    ExpectedEpistemicNoBlr,
    /// `set_expected_epistemic` was called on a node that already has a
    /// captured value. One-shot per node.
    ExpectedEpistemicAlreadyCaptured { node_id: NodeId },
    /// `set_expected_epistemic` was given a value that is not strictly
    /// positive or not finite.
    ExpectedEpistemicInvalidValue(f64),
    /// `set_decision_policy` was called twice on the same graph.
    DecisionPolicyAlreadyFrozen,
    /// `set_decision_policy` rejected the supplied threshold tensor.
    DecisionPolicy(PolicyError),
    /// A structural mutation was attempted on a frozen node.
    NodeFrozen { node_id: NodeId },
    /// A structural mutation was attempted on a pruned (inactive) node.
    NodeNotActive { node_id: NodeId },
    /// A structural mutation requires a non-Dense children container
    /// (`force_grow` / `force_split` / `add_node` on a compressed sub-tree).
    NodeIsDense { node_id: NodeId },
    /// `force_merge(into, into)` is not permitted — a node cannot
    /// absorb itself.
    ForceMergeSelf,
    /// `force_merge` was given an absorbed-node id that is already
    /// inactive.
    ForceMergeAlreadyAbsorbed { node_id: NodeId },
    /// `force_split` requires the parent to be a non-frozen leaf
    /// (Dense-or-non-empty containers can't be split in 0.3d-3).
    ForceSplitNotLeaf { node_id: NodeId },
    /// `abng_action_count` was given an index outside `0..=5`.
    UnknownActionKind(u8),
    /// `observe` was called with a non-finite value (NaN, +Inf, or -Inf).
    /// Rejected at the boundary before any state mutation or audit append,
    /// so a rejected call leaves the chain head and stats version
    /// unchanged.
    ObserveNonFinite { value: f64 },
}

impl From<PolicyError> for GraphError {
    fn from(err: PolicyError) -> Self {
        GraphError::DecisionPolicy(err)
    }
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeOutOfRange { node_id, n_nodes } => write!(
                f,
                "abng: node id {node_id} out of range (graph has {n_nodes} nodes)"
            ),
            GraphError::KeyAlreadyBound { parent, key_byte } => write!(
                f,
                "abng: parent {parent} already has a child bound to key byte {key_byte}"
            ),
            GraphError::CodebookAlreadyFrozen => {
                write!(f, "abng: codebook is already frozen")
            }
            GraphError::Codebook(err) => write!(f, "{err}"),
            GraphError::NoCodebook => write!(f, "abng: no codebook installed"),
            GraphError::LeafHead(err) => write!(f, "{err}"),
            GraphError::Blr(err) => write!(f, "{err}"),
            GraphError::Density(err) => write!(f, "{err}"),
            GraphError::Calibration(err) => write!(f, "{err}"),
            GraphError::Drift(err) => write!(f, "{err}"),
            GraphError::ChainBroken { at_seq } => {
                write!(f, "abng: audit chain broken at seq {at_seq}")
            }
            GraphError::ExpectedEpistemicNoBlr => write!(
                f,
                "abng expected_epistemic: BLR posterior must be installed before capture"
            ),
            GraphError::ExpectedEpistemicAlreadyCaptured { node_id } => write!(
                f,
                "abng expected_epistemic: node {node_id} already captured (one-shot)"
            ),
            GraphError::ExpectedEpistemicInvalidValue(v) => write!(
                f,
                "abng expected_epistemic: value {v} must be strictly positive and finite"
            ),
            GraphError::DecisionPolicyAlreadyFrozen => {
                write!(f, "abng decision policy: already frozen")
            }
            GraphError::DecisionPolicy(err) => write!(f, "{err}"),
            GraphError::NodeFrozen { node_id } => write!(
                f,
                "abng: node {node_id} is frozen; structural mutations blocked"
            ),
            GraphError::NodeNotActive { node_id } => write!(
                f,
                "abng: node {node_id} is not active (already pruned or absorbed)"
            ),
            GraphError::NodeIsDense { node_id } => write!(
                f,
                "abng: node {node_id} has a Dense (compressed) children container; \
                 structural mutations require a routable container"
            ),
            GraphError::ForceMergeSelf => write!(
                f,
                "abng force_merge: a node cannot absorb itself"
            ),
            GraphError::ForceMergeAlreadyAbsorbed { node_id } => write!(
                f,
                "abng force_merge: node {node_id} is already inactive"
            ),
            GraphError::ForceSplitNotLeaf { node_id } => write!(
                f,
                "abng force_split: node {node_id} must be a non-frozen leaf"
            ),
            GraphError::UnknownActionKind(i) => write!(
                f,
                "abng action_count: unknown action kind index {i} (must be 0..=5)"
            ),
            GraphError::ObserveNonFinite { value } => write!(
                f,
                "abng observe: value {value} must be finite (rejected NaN/+Inf/-Inf)"
            ),
        }
    }
}

impl From<BlrError> for GraphError {
    fn from(err: BlrError) -> Self {
        GraphError::Blr(err)
    }
}

impl From<DensityError> for GraphError {
    fn from(err: DensityError) -> Self {
        GraphError::Density(err)
    }
}

impl From<CalibrationError> for GraphError {
    fn from(err: CalibrationError) -> Self {
        GraphError::Calibration(err)
    }
}

impl From<DriftError> for GraphError {
    fn from(err: DriftError) -> Self {
        GraphError::Drift(err)
    }
}

impl From<CodebookError> for GraphError {
    fn from(err: CodebookError) -> Self {
        GraphError::Codebook(err)
    }
}

impl From<LeafHeadError> for GraphError {
    fn from(err: LeafHeadError) -> Self {
        GraphError::LeafHead(err)
    }
}

/// Adaptive belief radix graph. Phase 0.2: multi-node arena + frozen
/// codebook + radix-style children + per-node stats chain.
#[derive(Debug, Clone)]
pub struct AdaptiveBeliefGraph {
    /// SplitMix64 seed, threaded for future structural-decision RNG.
    pub seed: u64,
    /// Logical training epoch.
    pub epoch: u64,
    /// All nodes in the graph, indexed by `node_id`.
    pub nodes: Vec<AdaptiveBeliefNode>,
    /// Append-only audit log.
    pub audit: Vec<AuditEvent>,
    /// Current global chain head. Equals `audit.last().new_hash` if the log
    /// is non-empty, else [`crate::genesis_hash`].
    pub chain_head: [u8; 32],
    /// Optional frozen quantile codebook for the prefix encoder. Installed
    /// at most once via [`set_codebook`](Self::set_codebook); subsequent
    /// installs error.
    pub codebook: Option<QuantileCodebook>,
    /// Optional frozen per-node MLP architecture (Phase 0.3a). Installed
    /// at most once via [`set_leaf_head`](Self::set_leaf_head); subsequent
    /// installs error. Must be installed before any
    /// [`add_node`](Self::add_node).
    pub head: Option<LeafHead>,
    /// Optional frozen BLR prior (Phase 0.3b). Installed at most once via
    /// [`set_blr_prior`](Self::set_blr_prior). Must be installed *after*
    /// the leaf head (which determines penultimate-feature dim `d`) and
    /// *before* any [`add_node`](Self::add_node).
    pub blr_prior: Option<BlrPrior>,
    /// Phase 0.3c — `true` once `set_density_tracker` is installed.
    /// Determines whether each node carries a `DensityTracker`.
    pub density_enabled: bool,
    /// Phase 0.3c — number of calibration bins (15 typical). `None`
    /// when calibration is not installed.
    pub calibration_n_bins: Option<u8>,
    /// Phase 0.3d-3 — frozen structural-decision thresholds. `None`
    /// until `set_decision_policy` installs them. Phase 0.3d-3 only
    /// stores the policy; the engine that reads thresholds and fires
    /// actions is `decide_step` in 0.3d-4.
    pub decision_policy: Option<DecisionPolicy>,
    /// Phase 0.3d-3 — counter per structural-action kind, indexed by
    /// [`ActionKind`]. Increments every time the corresponding event
    /// is appended to the audit chain.
    pub action_counts: [u64; N_ACTION_KINDS],
}

impl AdaptiveBeliefGraph {
    /// Construct a fresh graph with one root node and a `Created` audit
    /// event.
    pub fn new(seed: u64) -> Self {
        let mut g = Self {
            seed,
            epoch: 0,
            nodes: Vec::new(),
            audit: Vec::new(),
            chain_head: genesis_hash(),
            codebook: None,
            head: None,
            blr_prior: None,
            density_enabled: false,
            calibration_n_bins: None,
            decision_policy: None,
            action_counts: [0u64; N_ACTION_KINDS],
        };
        let root = AdaptiveBeliefNode::new(0, None, g.chain_head);
        g.nodes.push(root);
        // Append the Created event for the root.
        g.append_event(0, AuditKind::Created);
        g
    }

    /// Number of nodes currently in the graph.
    pub fn node_count(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// Number of audit events recorded so far.
    pub fn audit_len(&self) -> u64 {
        self.audit.len() as u64
    }

    /// Apply one observation to the named node. Bumps the per-node
    /// `stats_version`, advances the per-node stats chain, and appends a
    /// `BeliefUpdate` event to the global chain.
    ///
    /// Non-finite values (NaN, +Inf, -Inf) are rejected at the boundary
    /// with [`GraphError::ObserveNonFinite`]. The check runs before any
    /// state mutation or audit append, so a rejected call leaves the
    /// chain head, stats version, and audit log identical to pre-call.
    pub fn observe(&mut self, node_id: NodeId, value: f64) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        if !value.is_finite() {
            return Err(GraphError::ObserveNonFinite { value });
        }
        // Apply update first; the audit event hashes the *post-update* stats.
        self.nodes[node_id as usize].observe(value);
        self.append_event(node_id, AuditKind::BeliefUpdate { value });
        Ok(())
    }

    /// Apply a slice of observations in slice order.
    pub fn observe_slice(&mut self, node_id: NodeId, values: &[f64]) -> Result<(), GraphError> {
        for &v in values {
            self.observe(node_id, v)?;
        }
        Ok(())
    }

    /// Add a child node to `parent` bound at `key_byte`.
    ///
    /// The new node is appended to the arena and given the next free
    /// `node_id`. Promotion of the parent's `AdaptiveChildren` happens
    /// automatically at insert time and emits a `ChildrenPromoted` event
    /// *before* the `NodeAdded` event so the audit log replays in
    /// structural-then-mutational order.
    pub fn add_node(&mut self, parent: NodeId, key_byte: u8) -> Result<NodeId, GraphError> {
        let n_nodes = self.node_count();
        if parent >= n_nodes {
            return Err(GraphError::NodeOutOfRange {
                node_id: parent,
                n_nodes,
            });
        }
        if self.nodes[parent as usize].children.get(key_byte).is_some() {
            return Err(GraphError::KeyAlreadyBound { parent, key_byte });
        }

        let new_id: NodeId = n_nodes;
        // Promotion happens inside add_child; capture the kind transition
        // so we can emit ChildrenPromoted *before* NodeAdded.
        let (from_kind, to_kind) = self.nodes[parent as usize]
            .children
            .add_child(key_byte, new_id);
        if from_kind != to_kind {
            self.append_event(
                parent,
                AuditKind::ChildrenPromoted {
                    from: from_kind as u8,
                    to: to_kind as u8,
                },
            );
        }

        let mut child = AdaptiveBeliefNode::new(new_id, Some(parent), self.chain_head);
        // Initialize per-node params if a head is configured. Done before
        // pushing so the audit event sequence stays consistent: NodeAdded
        // first (the structural mutation), LeafParamsInitialized after.
        if let Some(head) = &self.head {
            child.params = init_params(head, self.seed, new_id);
        }
        // Initialize per-node BLR state if a prior is configured. d is
        // determined by the leaf head's penultimate dim. Phase 0.4 Track
        // C-2.3.5 — stamp `feature_version_hash` from the child's freshly
        // Xavier-init'd params so the BLR posterior is locked to that
        // feature space until the next reset.
        if let (Some(head), Some(prior)) = (&self.head, &self.blr_prior) {
            let d = blr_feature_dim(head);
            let mut blr = BlrState::from_prior(prior, d);
            blr.feature_version_hash = params_hash(&child.params);
            child.blr = Some(blr);
        }
        // Phase 0.3c — density tracker (uses BLR feature dim).
        if let (true, Some(head)) = (self.density_enabled, &self.head) {
            let d = blr_feature_dim(head);
            child.density = Some(DensityTracker::new(d));
        }
        // Phase 0.3c — calibration bins.
        if let Some(n_bins) = self.calibration_n_bins {
            child.calibration = Some(
                CalibrationBins::new(n_bins)
                    .expect("n_bins was validated at install time"),
            );
        }
        self.nodes.push(child);

        self.append_event(
            new_id,
            AuditKind::NodeAdded {
                parent,
                key_byte,
            },
        );
        if self.head.is_some() {
            let phash = params_hash(&self.nodes[new_id as usize].params);
            self.append_event(
                new_id,
                AuditKind::LeafParamsInitialized {
                    params_hash: phash,
                },
            );
        }
        if self.blr_prior.is_some() {
            let shash = self.nodes[new_id as usize]
                .blr
                .as_ref()
                .expect("blr state should be set when prior is")
                .state_hash();
            self.append_event(
                new_id,
                AuditKind::BlrInitialized { state_hash: shash },
            );
        }
        // Phase 0.3c — density / calibration init events.
        if self.density_enabled {
            let shash = self.nodes[new_id as usize]
                .density
                .as_ref()
                .expect("density tracker set when enabled")
                .state_hash();
            self.append_event(
                new_id,
                AuditKind::DensityTrackerInstalled { state_hash: shash },
            );
        }
        if self.calibration_n_bins.is_some() {
            let shash = self.nodes[new_id as usize]
                .calibration
                .as_ref()
                .expect("calibration set when n_bins is")
                .state_hash();
            self.append_event(
                new_id,
                AuditKind::CalibrationInstalled { state_hash: shash },
            );
        }
        Ok(new_id)
    }

    /// Install the quantile codebook. Errors if a codebook is already
    /// frozen (`CodebookAlreadyFrozen`) or the supplied parameters are
    /// invalid (`Codebook(...)`).
    pub fn set_codebook(
        &mut self,
        n_dims: usize,
        n_bins: u16,
        flat_boundaries: &[f64],
    ) -> Result<(), GraphError> {
        if self.codebook.is_some() {
            return Err(GraphError::CodebookAlreadyFrozen);
        }
        let cb = QuantileCodebook::from_flat(n_dims, n_bins, flat_boundaries)?;
        let codebook_hash = cb.frozen_hash;
        self.codebook = Some(cb);
        // The CodebookFrozen event is bound to the root for the
        // stats_hash field (codebook installation has no per-node effect,
        // but every event needs a node-context hash).
        self.append_event(0, AuditKind::CodebookFrozen { codebook_hash });
        Ok(())
    }

    /// Encode an input vector to a prefix using the installed codebook.
    pub fn encode_prefix(&self, x: &[f64]) -> Result<Vec<u8>, GraphError> {
        let cb = self.codebook.as_ref().ok_or(GraphError::NoCodebook)?;
        Ok(cb.encode(x)?)
    }

    /// Walk the radix tree from the root, matching one prefix byte per hop.
    /// Returns a [`RouteEvidence`] capturing the path taken and how many
    /// prefix bytes were successfully matched.
    pub fn descend(&self, prefix: &[u8]) -> RouteEvidence {
        let root_id: NodeId = 0;
        let mut path = Vec::with_capacity(prefix.len() + 1);
        path.push(root_id);
        let mut current = root_id;
        let mut matched: u8 = 0;
        for &byte in prefix {
            match self.nodes[current as usize].children.get(byte) {
                Some(next) => {
                    path.push(next);
                    current = next;
                    matched = matched.saturating_add(1);
                }
                None => break,
            }
        }
        RouteEvidence {
            matched_prefix: matched,
            leaf_id: current,
            path,
        }
    }

    /// Build, hash-link, and append the next audit event for `node_id`.
    /// Internal — every kind funnels through here so the chain logic lives
    /// in exactly one place.
    fn append_event(&mut self, node_id: NodeId, kind: AuditKind) {
        let stats_hash = self.nodes[node_id as usize].stats.stats_hash();
        let stats_version = self.nodes[node_id as usize].stats_version;
        let previous_hash = self.chain_head;
        let seq = self.audit.len() as u64;

        let mut event = AuditEvent {
            seq,
            epoch: self.epoch,
            node_id,
            kind,
            stats_version,
            stats_hash,
            previous_hash,
            new_hash: [0u8; 32],
        };
        event.new_hash = AuditEvent::compute_new_hash(&previous_hash, &event.payload_bytes());
        self.chain_head = event.new_hash;
        self.audit.push(event);
    }

    /// Recompute every event's `new_hash` from scratch and check that the
    /// stored chain is consistent.
    pub fn verify_chain(&self) -> Result<(), GraphError> {
        let mut prev = genesis_hash();
        for event in &self.audit {
            if event.previous_hash != prev {
                return Err(GraphError::ChainBroken { at_seq: event.seq });
            }
            let recomputed = AuditEvent::compute_new_hash(&prev, &event.payload_bytes());
            if recomputed != event.new_hash {
                return Err(GraphError::ChainBroken { at_seq: event.seq });
            }
            prev = recomputed;
        }
        if prev != self.chain_head {
            return Err(GraphError::ChainBroken {
                at_seq: self.audit.len() as u64,
            });
        }
        Ok(())
    }

    /// Install the per-node MLP head architecture (Phase 0.3a).
    ///
    /// One-shot — subsequent calls error with [`LeafHeadError::AlreadyFrozen`].
    /// **Must be called before any `add_node`** since existing children would
    /// otherwise carry empty `params` mismatching the architecture.
    /// Initializes the root's params via deterministic Xavier from
    /// `(seed, 0)` and emits `LeafHeadConfigured` + `LeafParamsInitialized`
    /// audit events.
    pub fn set_leaf_head(
        &mut self,
        input_dim: u32,
        hidden_dims: Vec<u32>,
        output_dim: u32,
        activation: Activation,
    ) -> Result<(), GraphError> {
        if self.head.is_some() {
            return Err(LeafHeadError::AlreadyFrozen.into());
        }
        if self.nodes.len() > 1 {
            return Err(LeafHeadError::NotEmptyGraph {
                n_nodes: self.nodes.len() as u32,
            }
            .into());
        }
        if input_dim == 0 || output_dim == 0 || hidden_dims.iter().any(|&h| h == 0) {
            return Err(LeafHeadError::ZeroDim.into());
        }
        let head = LeafHead::new(input_dim, hidden_dims, output_dim, activation);
        let config_hash = head.config_hash;
        // Initialize the root's params before installing so post-init
        // hashes are computable in one pass.
        let root_params = init_params(&head, self.seed, 0);
        self.nodes[0].params = root_params;
        self.head = Some(head);
        self.append_event(0, AuditKind::LeafHeadConfigured { config_hash });
        let phash = params_hash(&self.nodes[0].params);
        self.append_event(
            0,
            AuditKind::LeafParamsInitialized {
                params_hash: phash,
            },
        );
        Ok(())
    }

    /// Number of MLP param tensors stored on a node (`2 × num_layers`),
    /// or `0` if no head is configured.
    pub fn leaf_param_count(&self, node_id: NodeId) -> Result<usize, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize].params.len())
    }

    /// Read a single MLP param tensor by index. Errors if no head is
    /// configured or the index is out of range.
    pub fn leaf_param(&self, node_id: NodeId, k: u32) -> Result<Tensor, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let params = &self.nodes[node_id as usize].params;
        if params.is_empty() {
            return Err(LeafHeadError::NoLeafHead.into());
        }
        if (k as usize) >= params.len() {
            return Err(LeafHeadError::ParamIndexOutOfRange {
                param_index: k,
                n_params: params.len() as u32,
            }
            .into());
        }
        Ok(params[k as usize].clone())
    }

    /// Write back a single MLP param tensor. Validates that the supplied
    /// tensor's shape matches the architecture's expected shape for index
    /// `k`. Emits a `LeafParamsUpdated` audit event with the new
    /// `params_hash` so the chain witnesses the update.
    pub fn leaf_set_param(
        &mut self,
        node_id: NodeId,
        k: u32,
        t: Tensor,
    ) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let head = self.head.as_ref().ok_or(LeafHeadError::NoLeafHead)?;
        let n_params = head.param_count() as u32;
        if k >= n_params {
            return Err(LeafHeadError::ParamIndexOutOfRange {
                param_index: k,
                n_params,
            }
            .into());
        }
        let expected = expected_param_shape(head, k);
        if t.shape() != expected.as_slice() {
            return Err(LeafHeadError::ShapeMismatch {
                node_id,
                param_index: k,
                expected,
                got: t.shape().to_vec(),
            }
            .into());
        }
        self.nodes[node_id as usize].params[k as usize] = t;
        let phash = params_hash(&self.nodes[node_id as usize].params);
        self.append_event(
            node_id,
            AuditKind::LeafParamsUpdated {
                params_hash: phash,
            },
        );
        Ok(())
    }

    /// Write back the entire MLP param vector for a node in one call.
    /// Phase 0.4 Track C-2.3.6 — collapses the `2(L+1)` per-tensor
    /// `LeafParamsUpdated` events that an optimizer step under
    /// `leaf_set_param` would emit into a single
    /// `LeafParamsUpdatedBatch` audit event with one hash witness for
    /// the whole post-update vector.
    ///
    /// `params` must have length `head.param_count()` and each tensor's
    /// shape must match `expected_param_shape(head, k)` for its index.
    /// Validation is all-or-nothing — if any tensor mismatches, the
    /// node's params are left unchanged and the corresponding
    /// `LeafHeadError` is returned without an audit append.
    pub fn leaf_set_params_batch(
        &mut self,
        node_id: NodeId,
        params: Vec<Tensor>,
    ) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let head = self.head.as_ref().ok_or(LeafHeadError::NoLeafHead)?;
        let n_params = head.param_count();
        if params.len() != n_params {
            return Err(LeafHeadError::ParamIndexOutOfRange {
                param_index: params.len() as u32,
                n_params: n_params as u32,
            }
            .into());
        }
        // Validate every tensor's shape *before* mutating any node
        // state — partial writes would diverge from the all-or-nothing
        // contract that callers expect from a "batch" operation.
        for (k, t) in params.iter().enumerate() {
            let expected = expected_param_shape(head, k as u32);
            if t.shape() != expected.as_slice() {
                return Err(LeafHeadError::ShapeMismatch {
                    node_id,
                    param_index: k as u32,
                    expected,
                    got: t.shape().to_vec(),
                }
                .into());
            }
        }
        // All params validated — write atomically.
        self.nodes[node_id as usize].params = params;
        let phash = params_hash(&self.nodes[node_id as usize].params);
        self.append_event(
            node_id,
            AuditKind::LeafParamsUpdatedBatch {
                params_hash: phash,
            },
        );
        Ok(())
    }

    /// SHA-256 of a node's full params blob (canonical bytes).
    pub fn leaf_params_hash(&self, node_id: NodeId) -> Result<[u8; 32], GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(params_hash(&self.nodes[node_id as usize].params))
    }

    /// Wire a leaf's MLP into the *ambient* `cjc_ad::GradGraph`. Returns
    /// `(y_idx, param_indices)` where `y_idx` is the GradGraph node index
    /// of the MLP output and `param_indices` is the list of GradGraph
    /// node indices of the registered parameters (for
    /// `grad_graph_backward_collect`).
    ///
    /// Errors if no leaf head is configured or `node_id` is out of range.
    /// The leaf's stored `Vec<Tensor>` is *cloned* into the GradGraph as
    /// parameters; updates flow back via `leaf_set_param` after the user
    /// runs their optimizer.
    pub fn leaf_forward(
        &self,
        node_id: NodeId,
        x_idx: usize,
    ) -> Result<(usize, Vec<usize>), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let head = self.head.as_ref().ok_or(LeafHeadError::NoLeafHead)?;
        let params = &self.nodes[node_id as usize].params;
        if params.is_empty() {
            return Err(LeafHeadError::NoLeafHead.into());
        }
        let num_layers = head.num_layers();
        let mut param_indices: Vec<usize> = Vec::with_capacity(params.len());
        let y_idx = cjc_ad::dispatch::with_ambient(|g| {
            // Register every param tensor as a parameter on the graph.
            for t in params {
                param_indices.push(g.parameter(t.clone()));
            }
            // Walk layers. Final layer uses Activation::None; prior
            // layers use the head's configured activation.
            let mut current = x_idx;
            for layer_idx in 0..num_layers {
                let w = param_indices[2 * layer_idx];
                let b = param_indices[2 * layer_idx + 1];
                let act = if layer_idx + 1 == num_layers {
                    Activation::None
                } else {
                    head.activation
                };
                current = g.mlp_layer(current, w, b, act);
            }
            current
        });
        Ok((y_idx, param_indices))
    }

    /// Install the graph-wide BLR prior (Phase 0.3b).
    ///
    /// One-shot. Must be called *after* [`set_leaf_head`](Self::set_leaf_head)
    /// (so the penultimate-feature dim `d` is determined) and *before* any
    /// [`add_node`](Self::add_node) (so all nodes carry consistent BLR
    /// state). Initializes the root's `BlrState` and emits
    /// `BlrPriorConfigured` + `BlrInitialized` audit events.
    pub fn set_blr_prior(
        &mut self,
        precision: f64,
        a: f64,
        b: f64,
    ) -> Result<(), GraphError> {
        if self.blr_prior.is_some() {
            return Err(BlrError::AlreadyFrozen.into());
        }
        let head = self.head.as_ref().ok_or(BlrError::NoLeafHead)?;
        if self.nodes.len() > 1 {
            return Err(BlrError::NotEmptyGraph {
                n_nodes: self.nodes.len() as u32,
            }
            .into());
        }
        let prior = BlrPrior::new(precision, a, b)?;
        let config_hash = prior.config_hash;
        let d = blr_feature_dim(head);
        // Initialize root's BLR state from prior before installing so
        // post-init hashes are computable in one pass. Phase 0.4 Track
        // C-2.3.5 — also stamp `feature_version_hash` from the root's
        // current MLP params hash so subsequent `blr_update` calls can
        // detect drift.
        let mut root_blr = BlrState::from_prior(&prior, d);
        root_blr.feature_version_hash = params_hash(&self.nodes[0].params);
        self.nodes[0].blr = Some(root_blr);
        self.blr_prior = Some(prior);
        self.append_event(0, AuditKind::BlrPriorConfigured { config_hash });
        let shash = self.nodes[0]
            .blr
            .as_ref()
            .expect("blr state was just set")
            .state_hash();
        self.append_event(0, AuditKind::BlrInitialized { state_hash: shash });
        Ok(())
    }

    /// Read the penultimate-feature dim `d` (or `0` if no leaf head).
    /// Defined here so the dispatch layer doesn't have to reach into
    /// `LeafHead`.
    pub fn blr_feature_dim(&self) -> u32 {
        self.head
            .as_ref()
            .map(blr_feature_dim)
            .unwrap_or(0)
    }

    /// Wire the leaf's MLP up to the *penultimate* layer into the ambient
    /// `cjc_ad::GradGraph`. Returns the GradGraph node index of the
    /// penultimate-features tensor.
    ///
    /// For a head with `hidden_dims = [h_1, …, h_L]`, this runs all
    /// hidden layers (each with the head's activation) but **does not**
    /// run the final linear layer. For an empty `hidden_dims`, returns
    /// the input index unchanged (degenerate but supported).
    pub fn blr_features(
        &self,
        node_id: NodeId,
        x_idx: usize,
    ) -> Result<usize, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let head = self.head.as_ref().ok_or(LeafHeadError::NoLeafHead)?;
        let params = &self.nodes[node_id as usize].params;
        if params.is_empty() {
            return Err(LeafHeadError::NoLeafHead.into());
        }
        let n_hidden = head.hidden_dims.len();
        if n_hidden == 0 {
            // No hidden layers — penultimate features = raw input.
            return Ok(x_idx);
        }
        let act = head.activation;
        let mut current = x_idx;
        let result = cjc_ad::dispatch::with_ambient(|g| {
            for layer_idx in 0..n_hidden {
                let w = g.parameter(params[2 * layer_idx].clone());
                let b = g.parameter(params[2 * layer_idx + 1].clone());
                current = g.mlp_layer(current, w, b, act);
            }
            current
        });
        Ok(result)
    }

    /// Apply a Normal-Inverse-Gamma update to a node's BLR posterior.
    ///
    /// `features` is row-major `[n, d]`; `y` is `[n]`. Errors if the
    /// node has no BLR state, the dimensions don't match, or Cholesky
    /// of the updated precision matrix fails (corrupt state).
    pub fn blr_update(
        &mut self,
        node_id: NodeId,
        features: &[f64],
        y: &[f64],
    ) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        // Phase 0.4 Track C-2.3.5 — feature-version stale check. Snapshot
        // the current per-node MLP params hash *before* taking a mutable
        // borrow on the BLR state; if the BLR was trained against a
        // different feature space, refuse the update so the posterior
        // never trains on inconsistent features.
        let current_hash = params_hash(&self.nodes[node_id as usize].params);
        let blr = self.nodes[node_id as usize]
            .blr
            .as_mut()
            .ok_or(BlrError::NoBlrPrior)?;
        if blr.feature_version_hash != current_hash {
            return Err(BlrError::FeatureVersionStale {
                stored: blr.feature_version_hash,
                current: current_hash,
            }
            .into());
        }
        let n = y.len();
        let expected = blr.d as usize;
        if features.len() != n * expected.max(1) || (n > 0 && features.len() % expected.max(1) != 0)
        {
            return Err(BlrError::FeatureDimMismatch {
                expected: expected as u32,
                got: if n == 0 { 0 } else { (features.len() / n) as u32 },
            }
            .into());
        }
        let rescue = blr.update(features, y)?;
        let shash = self.nodes[node_id as usize]
            .blr
            .as_ref()
            .expect("blr present")
            .state_hash();
        self.append_event(node_id, AuditKind::BlrUpdated { state_hash: shash });
        // Phase 0.4 Track C-2.3.4 — diagnostic event when the b<ε
        // numerical rescue fired. Emitted *after* BlrUpdated so the
        // state-changing event is the canonical one and the rescue is
        // metadata. Filtering for BlrNumericalRescue identifies which
        // immediately-preceding update needed rescue.
        if let Some(b_pre_clamp) = rescue {
            self.append_event(
                node_id,
                AuditKind::BlrNumericalRescue {
                    reason: BLR_RESCUE_B_BELOW_EPSILON,
                    b_pre_clamp_bits: b_pre_clamp.to_bits(),
                },
            );
        }
        Ok(())
    }

    /// Phase 0.4 Track C-2.3.5 — reset the BLR posterior on a node back
    /// to the configured prior, refreshing `feature_version_hash` to
    /// the current per-node MLP params. Use this after intentionally
    /// modifying the MLP weights (`leaf_set_param` /
    /// `leaf_set_params_batch`) to clear the stale `BlrError::FeatureVersionStale`
    /// guard before continuing training.
    ///
    /// Emits a `BlrInitialized` audit event so the chain witnesses the
    /// reset; the apply_event path during replay handles the reset
    /// uniformly (whether it originated at install time or here).
    pub fn reset_blr(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let prior = self.blr_prior.as_ref().ok_or(BlrError::NoBlrPrior)?.clone();
        if self.nodes[node_id as usize].blr.is_none() {
            return Err(BlrError::NoBlrPrior.into());
        }
        let head = self.head.as_ref().ok_or(LeafHeadError::NoLeafHead)?;
        let d = blr_feature_dim(head);
        let new_hash = params_hash(&self.nodes[node_id as usize].params);
        let mut fresh = BlrState::from_prior(&prior, d);
        fresh.feature_version_hash = new_hash;
        self.nodes[node_id as usize].blr = Some(fresh);
        let shash = self.nodes[node_id as usize]
            .blr
            .as_ref()
            .expect("just set")
            .state_hash();
        self.append_event(node_id, AuditKind::BlrInitialized { state_hash: shash });
        Ok(())
    }

    /// Predict at a single feature vector `phi`. Returns
    /// `(mean, epistemic_leverage, aleatoric_var)` — see
    /// [`BlrState::predict`] for the leverage-vs-variance distinction.
    pub fn blr_predict(
        &self,
        node_id: NodeId,
        phi: &[f64],
    ) -> Result<(f64, f64, f64), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let blr = self.nodes[node_id as usize]
            .blr
            .as_ref()
            .ok_or(BlrError::NoBlrPrior)?;
        Ok(blr.predict(phi)?)
    }

    /// Phase 0.4 Track C-2.3.8 — predict at `phi`, walking up the
    /// parent chain if the target node has not yet seen any
    /// observations (`blr.n_seen == 0`). Returns the predict tuple
    /// of the **nearest ancestor (incl. self) with `n_seen >= 1`**
    /// plus the node id it landed on. Read-only; no audit event,
    /// no RNG.
    ///
    /// Walk semantics:
    /// 1. Start at `node_id`. If `n_seen >= 1`, predict and return.
    /// 2. Else if the node has a parent, recurse on the parent.
    /// 3. Else (root with `n_seen == 0`), error with
    ///    `BlrError::NoEvidence { walked }`.
    ///
    /// Errors propagated from the resolved ancestor's
    /// `BlrState::predict` (e.g. `FeatureDimMismatch`) are returned
    /// as-is — they signal corrupt state, not lack of evidence.
    pub fn blr_predict_with_fallback(
        &self,
        node_id: NodeId,
        phi: &[f64],
    ) -> Result<(f64, f64, f64, NodeId), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        // Walk up. `walked` counts visited ancestors so the error
        // message has a useful diagnostic.
        let mut current = node_id;
        let mut walked: u32 = 0;
        loop {
            walked = walked.saturating_add(1);
            let node = &self.nodes[current as usize];
            let blr = node.blr.as_ref().ok_or(BlrError::NoBlrPrior)?;
            if blr.n_seen >= 1 {
                let (mean, leverage, ale) = blr.predict(phi)?;
                return Ok((mean, leverage, ale, current));
            }
            match node.parent {
                Some(p) => current = p,
                None => return Err(BlrError::NoEvidence { walked }.into()),
            }
        }
    }

    /// SHA-256 of a node's BLR state canonical bytes. Errors if no BLR.
    pub fn blr_state_hash(&self, node_id: NodeId) -> Result<[u8; 32], GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let blr = self.nodes[node_id as usize]
            .blr
            .as_ref()
            .ok_or(BlrError::NoBlrPrior)?;
        Ok(blr.state_hash())
    }

    /// Number of observations applied to a node's BLR posterior, or `0`
    /// if no BLR state.
    pub fn blr_n_seen(&self, node_id: NodeId) -> Result<u64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize]
            .blr
            .as_ref()
            .map(|s| s.n_seen)
            .unwrap_or(0))
    }

    // ── Phase 0.3c: density tracker ───────────────────────────

    /// Install a per-node density tracker. One-shot, requires leaf
    /// head + empty (root-only) graph. Initializes the root's tracker
    /// at d=BLR-feature-dim.
    pub fn set_density_tracker(&mut self) -> Result<(), GraphError> {
        if self.density_enabled {
            return Err(DensityError::AlreadyFrozen.into());
        }
        let head = self.head.as_ref().ok_or(DensityError::NoLeafHead)?;
        if self.nodes.len() > 1 {
            return Err(DensityError::NotEmptyGraph {
                n_nodes: self.nodes.len() as u32,
            }
            .into());
        }
        let d = blr_feature_dim(head);
        self.nodes[0].density = Some(DensityTracker::new(d));
        self.density_enabled = true;
        let shash = self.nodes[0]
            .density
            .as_ref()
            .expect("just set")
            .state_hash();
        self.append_event(
            0,
            AuditKind::DensityTrackerInstalled { state_hash: shash },
        );
        Ok(())
    }

    /// Apply a batch of feature observations to a node's density tracker.
    /// `features` is row-major `[n, d]`.
    pub fn density_observe(
        &mut self,
        node_id: NodeId,
        features: &[f64],
    ) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let dens = self.nodes[node_id as usize]
            .density
            .as_mut()
            .ok_or(DensityError::NoDensityTracker)?;
        dens.observe_batch(features)?;
        let shash = self.nodes[node_id as usize]
            .density
            .as_ref()
            .expect("present")
            .state_hash();
        self.append_event(node_id, AuditKind::DensityUpdated { state_hash: shash });
        Ok(())
    }

    /// Diagonal-Mahalanobis density score at `phi`. Returns `0.0` when
    /// `n < 2` (no signal yet).
    pub fn density_score(&self, node_id: NodeId, phi: &[f64]) -> Result<f64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let dens = self.nodes[node_id as usize]
            .density
            .as_ref()
            .ok_or(DensityError::NoDensityTracker)?;
        Ok(dens.density_score(phi)?)
    }

    /// Number of feature observations applied to a node's tracker.
    pub fn density_n_seen(&self, node_id: NodeId) -> Result<u64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize]
            .density
            .as_ref()
            .map(|d| d.n)
            .unwrap_or(0))
    }

    // ── Phase 0.3c: calibration bins ──────────────────────────

    /// Install a per-node calibration-bins set with the given number of
    /// bins. One-shot, requires empty graph.
    pub fn set_calibration(&mut self, n_bins: u8) -> Result<(), GraphError> {
        if self.calibration_n_bins.is_some() {
            return Err(CalibrationError::AlreadyFrozen.into());
        }
        if self.nodes.len() > 1 {
            return Err(CalibrationError::NotEmptyGraph {
                n_nodes: self.nodes.len() as u32,
            }
            .into());
        }
        let bins = CalibrationBins::new(n_bins)?;
        self.nodes[0].calibration = Some(bins);
        self.calibration_n_bins = Some(n_bins);
        let shash = self.nodes[0]
            .calibration
            .as_ref()
            .expect("just set")
            .state_hash();
        self.append_event(
            0,
            AuditKind::CalibrationInstalled { state_hash: shash },
        );
        Ok(())
    }

    /// Record one observation: a predicted probability + ground-truth
    /// outcome. Updates the bin counts and fires `CalibrationUpdated`.
    pub fn calibration_observe(
        &mut self,
        node_id: NodeId,
        predicted_prob: f64,
        was_correct: bool,
    ) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let bins = self.nodes[node_id as usize]
            .calibration
            .as_mut()
            .ok_or(CalibrationError::NoCalibration)?;
        bins.observe(predicted_prob, was_correct)?;
        let shash = self.nodes[node_id as usize]
            .calibration
            .as_ref()
            .expect("present")
            .state_hash();
        self.append_event(node_id, AuditKind::CalibrationUpdated { state_hash: shash });
        Ok(())
    }

    /// Expected Calibration Error for a node. `0.0` when no observations
    /// have been recorded.
    pub fn calibration_ece(&self, node_id: NodeId) -> Result<f64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let bins = self.nodes[node_id as usize]
            .calibration
            .as_ref()
            .ok_or(CalibrationError::NoCalibration)?;
        Ok(bins.ece())
    }

    /// Number of observations recorded into a node's calibration bins.
    pub fn calibration_n_seen(&self, node_id: NodeId) -> Result<u64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize]
            .calibration
            .as_ref()
            .map(|c| c.n_seen())
            .unwrap_or(0))
    }

    // ── Phase 0.3c: drift detector ────────────────────────────

    /// Snapshot the node's current density tracker as a drift baseline.
    /// Per-node, can be called any time the density tracker has at
    /// least 2 observations. Subsequent `drift_score` calls compare
    /// the live tracker to this frozen baseline.
    pub fn freeze_drift_baseline(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let dens = self.nodes[node_id as usize]
            .density
            .as_ref()
            .ok_or(DriftError::NoDensityTracker)?;
        let baseline = DriftBaseline::from_density(dens)?;
        let shash = baseline.state_hash();
        self.nodes[node_id as usize].drift_baseline = Some(baseline);
        self.append_event(
            node_id,
            AuditKind::DriftBaselineFrozen { state_hash: shash },
        );
        Ok(())
    }

    /// Drift score for a node: per-dim z-shift L2-normalised between
    /// current density tracker and frozen baseline. Errors if either is
    /// missing.
    pub fn drift_score(&self, node_id: NodeId) -> Result<f64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let dens = self.nodes[node_id as usize]
            .density
            .as_ref()
            .ok_or(DriftError::NoDensityTracker)?;
        let baseline = self.nodes[node_id as usize]
            .drift_baseline
            .as_ref()
            .ok_or(DriftError::NoBaseline)?;
        Ok(baseline.drift_score(dens)?)
    }

    // ── Phase 0.3c: composite OOD score ───────────────────────

    /// Composite OOD score: `max(density_score, prefix_distance,
    /// epistemic_z)`. Each missing-subsystem term contributes `0.0` so
    /// the composite is well-defined even with partial config.
    ///
    /// * `phi`            — penultimate-feature vector for `node_id`
    /// * `matched_prefix` — number of prefix bytes matched on descent
    /// * `prefix_max`     — total prefix length (D)
    pub fn ood_score(
        &self,
        node_id: NodeId,
        phi: &[f64],
        matched_prefix: u8,
        prefix_max: u8,
    ) -> Result<f64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let node = &self.nodes[node_id as usize];
        let density_score = match node.density.as_ref() {
            Some(d) => d.density_score(phi).unwrap_or(0.0),
            None => 0.0,
        };
        let prefix_distance = if prefix_max > 0 {
            let unmatched = prefix_max.saturating_sub(matched_prefix);
            unmatched as f64 / prefix_max as f64
        } else {
            0.0
        };
        // Epistemic z: BLR's epistemic *leverage* at `phi` (the second
        // tuple element of `predict`; named "epistemic_var" pre-0.4 but
        // it has always been dimensionless leverage — see Phase 0.4
        // Track C-2.3.1). When the node has captured a training-time
        // `expected_epistemic` reference (Phase 0.3d-2), use the
        // calibrated ratio `(lev / expected).clamp(0, 1)`. Otherwise
        // fall back to the raw `lev.clamp(0, 1)` Phase 0.3c behavior —
        // meaningful for nodes that haven't reached uncertainty
        // stability yet.
        let epistemic_z = match node.blr.as_ref() {
            Some(blr) => match blr.predict(phi) {
                Ok((_m, lev, _ale)) => match node.expected_epistemic {
                    Some(expected) if expected > 0.0 => {
                        (lev / expected).min(1.0).max(0.0)
                    }
                    _ => lev.min(1.0).max(0.0),
                },
                Err(_) => 0.0,
            },
            None => 0.0,
        };
        Ok(density_score.max(prefix_distance).max(epistemic_z))
    }

    /// `ChildrenKind` numeric code for a node's container. Convenience
    /// wrapper for the dispatch layer.
    pub fn node_kind(&self, node_id: NodeId) -> Result<ChildrenKind, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize].children.kind())
    }

    // ── Phase 0.3d-2: per-node expected_epistemic capture ─────────

    /// Capture a per-node training-time epistemic-σ reference. After
    /// capture, [`ood_score`](Self::ood_score) uses the calibrated
    /// ratio `(epi / expected).clamp(0, 1)` instead of the raw clamp.
    ///
    /// Validation:
    /// 1. Node must have a BLR posterior installed
    ///    (`set_blr_prior` was called).
    /// 2. `value` must be strictly positive and finite.
    /// 3. One-shot per node — re-capture errors with
    ///    [`GraphError::ExpectedEpistemicAlreadyCaptured`].
    ///
    /// Phase 0.3d-2 is manual-only; Phase 0.3d-4's `decide_step` will
    /// auto-capture when `Maturity.uncertainty_stable` first holds.
    pub fn set_expected_epistemic(
        &mut self,
        node_id: NodeId,
        value: f64,
    ) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        if !value.is_finite() || value <= 0.0 {
            return Err(GraphError::ExpectedEpistemicInvalidValue(value));
        }
        let node = &self.nodes[node_id as usize];
        if node.blr.is_none() {
            return Err(GraphError::ExpectedEpistemicNoBlr);
        }
        if node.expected_epistemic.is_some() {
            return Err(GraphError::ExpectedEpistemicAlreadyCaptured { node_id });
        }
        self.nodes[node_id as usize].expected_epistemic = Some(value);
        let state_hash =
            cjc_snap::hash::sha256(&value.to_bits().to_be_bytes());
        self.append_event(
            node_id,
            AuditKind::ExpectedEpistemicCaptured { state_hash },
        );
        Ok(())
    }

    /// Read a node's captured `expected_epistemic` reference, or
    /// `None` if uncaptured.
    pub fn expected_epistemic(
        &self,
        node_id: NodeId,
    ) -> Result<Option<f64>, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize].expected_epistemic)
    }

    /// Phase 0.4 Track C-2.3.12 — force a fresh capture of
    /// `expected_epistemic` from the current BLR posterior, overwriting
    /// any previously-captured value. Required after `reset_blr` (which
    /// reseats the posterior on a new feature space and invalidates
    /// the previously-captured leverage reference) so `ood_score`'s
    /// calibrated ratio doesn't drift against stale evidence.
    ///
    /// Unlike [`set_expected_epistemic`](Self::set_expected_epistemic),
    /// this is NOT one-shot. Each call emits a fresh
    /// `ExpectedEpistemicCaptured` audit event (tag `0x17`) so replay
    /// rebuilds the same sequence of capture states.
    ///
    /// The captured value is
    /// `epistemic_leverage(blr.predict(blr.mean))` — the same
    /// deterministic capture logic [`decide_step`](Self::decide_step)
    /// runs when `Maturity.uncertainty_stable` first holds.
    ///
    /// Validation:
    /// 1. Node must exist (`GraphError::NodeOutOfRange` otherwise).
    /// 2. Node must have a BLR posterior installed
    ///    (`GraphError::ExpectedEpistemicNoBlr` otherwise).
    /// 3. The posterior `predict(blr.mean)` must produce a finite,
    ///    positive leverage value — otherwise re-capture errors with
    ///    `GraphError::ExpectedEpistemicInvalidValue(lev)`. This
    ///    matches `set_expected_epistemic`'s rejection contract.
    pub fn force_recapture_expected_epistemic(
        &mut self,
        node_id: NodeId,
    ) -> Result<f64, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let blr = self.nodes[node_id as usize]
            .blr
            .as_ref()
            .ok_or(GraphError::ExpectedEpistemicNoBlr)?;
        let value = epistemic_leverage_at_posterior_mean(blr)
            .ok_or_else(|| {
                // Mirror the public-API rejection: a non-positive /
                // non-finite leverage has the same shape as a manually-
                // supplied bad value. Use NaN as the carried diagnostic
                // since the actual offending value isn't useful (it was
                // either Err from predict or non-positive).
                GraphError::ExpectedEpistemicInvalidValue(f64::NAN)
            })?;
        self.nodes[node_id as usize].expected_epistemic = Some(value);
        let state_hash =
            cjc_snap::hash::sha256(&value.to_bits().to_be_bytes());
        self.append_event(
            node_id,
            AuditKind::ExpectedEpistemicCaptured { state_hash },
        );
        Ok(value)
    }

    // ── Phase 0.3d-1: maturity + signature (lazy / read-only) ─────

    /// Compute a node's [`Maturity`](crate::maturity::Maturity) snapshot
    /// from its current evidence-layer state. Phase 0.3d-1 is purely
    /// derived — no persistent maturity field on the node, no audit
    /// event, no snapshot impact. Phase 0.3d-2/4 promotes maturity to
    /// persistent state.
    pub fn node_maturity(
        &self,
        node_id: NodeId,
    ) -> Result<crate::maturity::Maturity, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(crate::maturity::Maturity::from_node(
            &self.nodes[node_id as usize],
        ))
    }

    /// Compute a node's [`NodeSignature`](crate::signature::NodeSignature)
    /// fingerprint from its current per-node state. Phase 0.3d-1 is
    /// purely derived; Phase 0.3d-4 promotes the four profiles to
    /// Welford-smoothed persistent state without changing the wire
    /// shape.
    pub fn node_signature(
        &self,
        node_id: NodeId,
    ) -> Result<crate::signature::NodeSignature, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(crate::signature::NodeSignature::from_node(
            &self.nodes[node_id as usize],
        ))
    }

    // ── Phase 0.3d-3: DecisionPolicy install ─────────────────────

    /// Install the structural-decision policy thresholds. One-shot —
    /// re-installation errors with [`GraphError::DecisionPolicyAlreadyFrozen`].
    /// Unlike other one-shot install builtins, this can be called at
    /// any point in the graph's lifecycle (no `n_nodes ≤ 1` requirement).
    /// Phase 0.3d-3 only stores the policy; the engine that consumes
    /// it is `decide_step` in 0.3d-4.
    pub fn set_decision_policy(&mut self, thresholds: &[f64]) -> Result<(), GraphError> {
        if self.decision_policy.is_some() {
            return Err(GraphError::DecisionPolicyAlreadyFrozen);
        }
        let policy = DecisionPolicy::new(thresholds)?;
        self.decision_policy = Some(policy);
        Ok(())
    }

    /// SHA-256 of the installed [`DecisionPolicy`]'s canonical bytes,
    /// or `None` if no policy is installed.
    pub fn decision_policy_hash(&self) -> Option<[u8; 32]> {
        self.decision_policy.as_ref().map(|p| p.policy_hash)
    }

    // ── Phase 0.3d-3: structural mutations (force-*) ─────────────

    /// Bump the per-action counter. Internal — called by every
    /// successful `force_*` after the corresponding event lands on the
    /// chain.
    fn bump_action_count(&mut self, kind: ActionKind) {
        self.action_counts[kind as usize] = self.action_counts[kind as usize]
            .saturating_add(1);
    }

    /// Read the count of how many times a given structural action
    /// has fired. Used by `abng_action_count` for inspection.
    pub fn action_count(&self, kind: ActionKind) -> u64 {
        self.action_counts[kind as usize]
    }

    /// Return whether a node is frozen.
    pub fn is_frozen(&self, node_id: NodeId) -> Result<bool, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize].is_frozen)
    }

    /// Return whether a node is active (not pruned / not absorbed).
    pub fn is_active(&self, node_id: NodeId) -> Result<bool, GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        Ok(self.nodes[node_id as usize].is_active)
    }

    /// Internal — assert that a node is eligible for structural
    /// mutation (in range, active, not frozen, not Dense).
    fn check_structural_target(&self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let node = &self.nodes[node_id as usize];
        if !node.is_active {
            return Err(GraphError::NodeNotActive { node_id });
        }
        if node.is_frozen {
            return Err(GraphError::NodeFrozen { node_id });
        }
        if node.children.is_dense() {
            return Err(GraphError::NodeIsDense { node_id });
        }
        Ok(())
    }

    /// Policy-driven (or test-driven) addition of a child. Conceptually
    /// identical to `add_node`, but emits [`AuditKind::Grow`] instead
    /// of [`AuditKind::NodeAdded`] so replay can distinguish
    /// user-driven topology from policy-driven topology.
    ///
    /// Errors if the parent is out-of-range, inactive, frozen, Dense,
    /// or already has a child at `key_byte`.
    pub fn force_grow(
        &mut self,
        parent: NodeId,
        key_byte: u8,
    ) -> Result<NodeId, GraphError> {
        self.check_structural_target(parent)?;
        if self.nodes[parent as usize].children.get(key_byte).is_some() {
            return Err(GraphError::KeyAlreadyBound { parent, key_byte });
        }

        let new_id: NodeId = self.node_count();
        let (from_kind, to_kind) = self.nodes[parent as usize]
            .children
            .add_child(key_byte, new_id);
        if from_kind != to_kind {
            self.append_event(
                parent,
                AuditKind::ChildrenPromoted {
                    from: from_kind as u8,
                    to: to_kind as u8,
                },
            );
        }

        let mut child = AdaptiveBeliefNode::new(new_id, Some(parent), self.chain_head);
        if let Some(head) = &self.head {
            child.params = init_params(head, self.seed, new_id);
        }
        if let (Some(head), Some(prior)) = (&self.head, &self.blr_prior) {
            let d = blr_feature_dim(head);
            // Phase 0.4 Track C-2.3.5 — stamp feature_version_hash from
            // the freshly Xavier-init'd params (force_grow / force_split
            // paths).
            let mut blr = BlrState::from_prior(prior, d);
            blr.feature_version_hash = params_hash(&child.params);
            child.blr = Some(blr);
        }
        if let (true, Some(head)) = (self.density_enabled, &self.head) {
            let d = blr_feature_dim(head);
            child.density = Some(DensityTracker::new(d));
        }
        if let Some(n_bins) = self.calibration_n_bins {
            child.calibration = Some(
                CalibrationBins::new(n_bins)
                    .expect("n_bins was validated at install time"),
            );
        }
        self.nodes.push(child);

        // Single Grow event drives reconstruction; subsystem
        // *Initialized events follow the existing add_node pattern so
        // replay can verify each subsystem's per-node hash.
        self.append_event(
            new_id,
            AuditKind::Grow {
                parent,
                key_byte,
                child: new_id,
            },
        );
        if self.head.is_some() {
            let phash = params_hash(&self.nodes[new_id as usize].params);
            self.append_event(
                new_id,
                AuditKind::LeafParamsInitialized {
                    params_hash: phash,
                },
            );
        }
        if self.blr_prior.is_some() {
            let shash = self.nodes[new_id as usize]
                .blr
                .as_ref()
                .expect("BLR set when prior is")
                .state_hash();
            self.append_event(new_id, AuditKind::BlrInitialized { state_hash: shash });
        }
        if self.density_enabled {
            let shash = self.nodes[new_id as usize]
                .density
                .as_ref()
                .expect("density set when enabled")
                .state_hash();
            self.append_event(
                new_id,
                AuditKind::DensityTrackerInstalled { state_hash: shash },
            );
        }
        if self.calibration_n_bins.is_some() {
            let shash = self.nodes[new_id as usize]
                .calibration
                .as_ref()
                .expect("calibration set when n_bins is")
                .state_hash();
            self.append_event(
                new_id,
                AuditKind::CalibrationInstalled { state_hash: shash },
            );
        }
        self.bump_action_count(ActionKind::Grow);
        Ok(new_id)
    }

    /// Policy-driven (or test-driven) split of a leaf into two new
    /// children. The two children are placed at deterministic
    /// key bytes derived from `(seed, parent)` so replay can
    /// reconstruct them. Phase 0.3d-3 ships only the structural
    /// mutation; the impurity-based split criterion arrives in 0.3d-4.
    ///
    /// Requires the parent to be a non-frozen, non-Dense leaf
    /// (i.e. `children.kind() == None`).
    pub fn force_split(&mut self, parent: NodeId) -> Result<(NodeId, NodeId), GraphError> {
        self.check_structural_target(parent)?;
        if !self.nodes[parent as usize].is_leaf() {
            return Err(GraphError::ForceSplitNotLeaf { node_id: parent });
        }

        // Deterministic key bytes from (seed, parent_id). Use two distinct
        // bits of a SplitMix64 mix to avoid collisions.
        let mix = self.seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(parent as u64);
        let key_a = (mix as u8) & 0xFE;            // even
        let key_b = key_a.wrapping_add(1);          // odd, distinct from key_a

        // Add the two children using the same plumbing as force_grow,
        // but emit a single Split event covering both arena indices.
        let id_a: NodeId = self.node_count();
        let (fa, ta) = self.nodes[parent as usize].children.add_child(key_a, id_a);
        if fa != ta {
            self.append_event(
                parent,
                AuditKind::ChildrenPromoted {
                    from: fa as u8,
                    to: ta as u8,
                },
            );
        }
        self.nodes.push(self.fresh_child_for(parent, id_a));

        let id_b: NodeId = self.node_count();
        let (fb, tb) = self.nodes[parent as usize].children.add_child(key_b, id_b);
        if fb != tb {
            self.append_event(
                parent,
                AuditKind::ChildrenPromoted {
                    from: fb as u8,
                    to: tb as u8,
                },
            );
        }
        self.nodes.push(self.fresh_child_for(parent, id_b));

        self.append_event(
            parent,
            AuditKind::Split {
                parent,
                child_a: id_a,
                child_b: id_b,
            },
        );

        // Per-subsystem *Initialized events for both new nodes,
        // matching the add_node ordering contract.
        for &id in &[id_a, id_b] {
            if self.head.is_some() {
                let phash = params_hash(&self.nodes[id as usize].params);
                self.append_event(
                    id,
                    AuditKind::LeafParamsInitialized {
                        params_hash: phash,
                    },
                );
            }
            if self.blr_prior.is_some() {
                let shash = self.nodes[id as usize]
                    .blr
                    .as_ref()
                    .expect("BLR")
                    .state_hash();
                self.append_event(id, AuditKind::BlrInitialized { state_hash: shash });
            }
            if self.density_enabled {
                let shash = self.nodes[id as usize]
                    .density
                    .as_ref()
                    .expect("density")
                    .state_hash();
                self.append_event(
                    id,
                    AuditKind::DensityTrackerInstalled { state_hash: shash },
                );
            }
            if self.calibration_n_bins.is_some() {
                let shash = self.nodes[id as usize]
                    .calibration
                    .as_ref()
                    .expect("calibration")
                    .state_hash();
                self.append_event(
                    id,
                    AuditKind::CalibrationInstalled { state_hash: shash },
                );
            }
        }
        self.bump_action_count(ActionKind::Split);
        Ok((id_a, id_b))
    }

    /// Internal — build a fresh child node with all installed
    /// subsystems pre-initialized. Used by `force_split` to avoid
    /// duplicating the per-subsystem init dance.
    fn fresh_child_for(&self, parent: NodeId, new_id: NodeId) -> AdaptiveBeliefNode {
        let mut child = AdaptiveBeliefNode::new(new_id, Some(parent), self.chain_head);
        if let Some(head) = &self.head {
            child.params = init_params(head, self.seed, new_id);
        }
        if let (Some(head), Some(prior)) = (&self.head, &self.blr_prior) {
            let d = blr_feature_dim(head);
            // Phase 0.4 Track C-2.3.5 — stamp feature_version_hash from
            // the freshly Xavier-init'd params (force_grow / force_split
            // paths).
            let mut blr = BlrState::from_prior(prior, d);
            blr.feature_version_hash = params_hash(&child.params);
            child.blr = Some(blr);
        }
        if let (true, Some(head)) = (self.density_enabled, &self.head) {
            let d = blr_feature_dim(head);
            child.density = Some(DensityTracker::new(d));
        }
        if let Some(n_bins) = self.calibration_n_bins {
            child.calibration = Some(
                CalibrationBins::new(n_bins)
                    .expect("n_bins was validated at install time"),
            );
        }
        child
    }

    /// Policy-driven (or test-driven) absorption of one node into
    /// another. Phase 0.3d-3 ships the *event channel* only — the
    /// only graph-state change is `absorbed.is_active = false`. Real
    /// NIG-aware merging (BLR posterior combination, stats fusion)
    /// lands with `decide_step` in 0.3d-4.
    pub fn force_merge(
        &mut self,
        absorbed: NodeId,
        into: NodeId,
    ) -> Result<(), GraphError> {
        if absorbed == into {
            return Err(GraphError::ForceMergeSelf);
        }
        self.check_structural_target(into)?;
        let n_nodes = self.node_count();
        if absorbed >= n_nodes {
            return Err(GraphError::NodeOutOfRange {
                node_id: absorbed,
                n_nodes,
            });
        }
        if !self.nodes[absorbed as usize].is_active {
            return Err(GraphError::ForceMergeAlreadyAbsorbed { node_id: absorbed });
        }
        if self.nodes[absorbed as usize].is_frozen {
            return Err(GraphError::NodeFrozen { node_id: absorbed });
        }

        // Phase 0.4 Track B-2.2.6 — fold absorbed's evidence into `into`
        // before deactivation. Pre-0.4 the merge was a "lose absorbed's
        // training history" no-op; now the BLR posteriors and Welford
        // node stats are properly combined. Both combines are pure
        // functions of the two states — no external entropy — so
        // training and replay produce bit-identical results.

        // 1. Combine NodeStats (parallel Welford merge).
        let absorbed_stats = self.nodes[absorbed as usize].stats.clone();
        self.nodes[into as usize].stats.combine(&absorbed_stats);
        self.nodes[into as usize].stats_version = self.nodes[into as usize]
            .stats_version
            .saturating_add(1);

        // 2. Combine BLR posteriors (NIG-aware combine).
        if let (Some(prior), true, true) = (
            self.blr_prior.clone(),
            self.nodes[into as usize].blr.is_some(),
            self.nodes[absorbed as usize].blr.is_some(),
        ) {
            let absorbed_blr = self.nodes[absorbed as usize]
                .blr
                .as_ref()
                .expect("absorbed.blr present")
                .clone();
            let into_blr = self.nodes[into as usize]
                .blr
                .as_mut()
                .expect("into.blr present");
            into_blr.combine(&absorbed_blr, &prior)?;
        }

        // 3. Mark absorbed inactive and append the structural Merge
        //    event. The event's `stats_hash` records *absorbed's*
        //    stats (unchanged by the combine, since combine wrote into
        //    `into`).
        self.nodes[absorbed as usize].is_active = false;
        self.append_event(absorbed, AuditKind::Merge { absorbed, into });

        // 4. Phase 0.4 Track B-2.2.6 — append a `BlrUpdated` witness on
        //    `into` carrying the post-combine state_hash. Without this,
        //    the end-of-replay per-node BLR verifier (which walks the
        //    audit log for the latest `BlrInitialized`/`BlrUpdated`
        //    event on the node) would compare the snapshot's combined
        //    state against an obsolete pre-merge witness and reject the
        //    blob with `BlrStateHashMismatch`. The witness is emitted
        //    only when into has a BLR posterior installed.
        if let Some(blr) = self.nodes[into as usize].blr.as_ref() {
            let combined_hash = blr.state_hash();
            self.append_event(into, AuditKind::BlrUpdated { state_hash: combined_hash });
        }

        self.bump_action_count(ActionKind::Merge);
        Ok(())
    }

    /// Policy-driven (or test-driven) deactivation of a node. Sets
    /// `is_active = false`. The node persists in the arena per the
    /// never-reorder-pushes invariant.
    pub fn force_prune(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        // Pruning a frozen node is blocked — Freeze is a "structural
        // ops blocked" marker.
        if self.nodes[node_id as usize].is_frozen {
            return Err(GraphError::NodeFrozen { node_id });
        }
        if !self.nodes[node_id as usize].is_active {
            return Err(GraphError::NodeNotActive { node_id });
        }
        self.nodes[node_id as usize].is_active = false;
        self.append_event(node_id, AuditKind::Prune { node_id });
        self.bump_action_count(ActionKind::Prune);
        Ok(())
    }

    /// Policy-driven (or test-driven) compression of a node's
    /// children container into the [`AdaptiveChildren::Dense`]
    /// variant. Descendants persist in the arena but become
    /// unreachable from this node through `descend`.
    ///
    /// The signature is computed from the current
    /// [`NodeSignature::from_node`](crate::signature::NodeSignature::from_node)
    /// so a future `decide_step`-driven Compress produces the same
    /// bytes for the same evidence shape.
    pub fn force_compress(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        let node = &self.nodes[node_id as usize];
        if !node.is_active {
            return Err(GraphError::NodeNotActive { node_id });
        }
        if node.is_frozen {
            return Err(GraphError::NodeFrozen { node_id });
        }
        if node.children.is_dense() {
            return Err(GraphError::NodeIsDense { node_id });
        }
        // Compute the signature *before* mutating children (so the
        // routing-profile component reflects the pre-compress shape).
        let signature = NodeSignature::from_node(node).canonical_bytes();
        self.nodes[node_id as usize].children = AdaptiveChildren::Dense { signature };
        self.append_event(node_id, AuditKind::Compress { signature });
        self.bump_action_count(ActionKind::Compress);
        Ok(())
    }

    /// Policy-driven (or test-driven) freeze of a node. Subsequent
    /// structural mutations on the frozen node error with
    /// [`GraphError::NodeFrozen`]. Idempotent — re-freezing a frozen
    /// node is a no-op (no event, no action-count bump).
    pub fn force_freeze(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        if self.nodes[node_id as usize].is_frozen {
            return Ok(()); // idempotent
        }
        self.nodes[node_id as usize].is_frozen = true;
        self.append_event(node_id, AuditKind::Freeze { node_id });
        self.bump_action_count(ActionKind::Freeze);
        Ok(())
    }

    // ── Phase 0.3d-4: unfreeze + decide_step engine ──────────────

    /// Phase 0.3d-4 — un-freeze a previously frozen node. Subsequent
    /// structural mutations on it are permitted again. Idempotent —
    /// un-freezing an already-active node is a no-op (no event).
    /// Used directly by tests / manual override; `decide_step`'s
    /// drift-trip auto-unfreeze path is deferred to Phase 0.4.
    pub fn unfreeze(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let n_nodes = self.node_count();
        if node_id >= n_nodes {
            return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
        }
        if !self.nodes[node_id as usize].is_frozen {
            return Ok(()); // idempotent
        }
        self.nodes[node_id as usize].is_frozen = false;
        self.append_event(node_id, AuditKind::Unfreeze { node_id });
        Ok(())
    }

    /// Phase 0.3d-4 — run one structural-decision pass.
    ///
    /// Iterates nodes in `NodeId` ascending order over the arena
    /// snapshot taken at call entry (newly-grown / split nodes from
    /// this pass are NOT re-visited). For each node:
    ///
    /// 1. Update `last_signature` + `signature_stable_calls`.
    /// 2. If `is_frozen`, skip the trigger ladder.
    /// 3. Auto-capture `expected_epistemic` if Maturity says
    ///    `uncertainty_stable` and no value is captured yet.
    /// 4. Try, in fall-through order: Compress, Merge, Split, Prune,
    ///    Grow, Freeze. The first trigger that fires consumes the
    ///    node's quota for this pass; subsequent triggers are skipped.
    ///
    /// Returns a per-action count tensor `[grow, split, merge, prune,
    /// compress, freeze]` — same indexing as
    /// [`ActionKind`].
    ///
    /// **No-op when no [`DecisionPolicy`] is installed** — returns all
    /// zeros and does not advance signature-stability tracking. This
    /// avoids surprising behaviour for users who haven't configured
    /// the policy yet.
    pub fn decide_step(&mut self) -> [u64; N_ACTION_KINDS] {
        let mut counts = [0u64; N_ACTION_KINDS];
        let policy = match self.decision_policy.clone() {
            Some(p) => p,
            None => return counts,
        };
        let n = self.node_count();

        for node_id in 0..n {
            let nid_idx = node_id as usize;

            // Skip absent / inactive nodes outright.
            if !self.nodes[nid_idx].is_active {
                continue;
            }

            // Always advance signature stability — even on frozen
            // nodes, so the stability counter keeps climbing while
            // they're frozen (allows Phase 0.4 drift-trip un-freeze
            // to use the same counter).
            self.advance_signature_stability(node_id);

            // Phase 0.4 Track B-2.2.2 — advance the 3-window ECE / σ
            // stability buffers. Like signature stability, these keep
            // accumulating while a node is frozen so the Maturity
            // flags reflect the latest trajectory once the node
            // un-freezes.
            self.advance_stability_history(node_id);

            // Phase 0.4 Track B-2.2.7 — drift-trip auto-unfreeze. If a
            // frozen node's drift score has crossed the policy's
            // `drift_unfreeze` threshold, unfreeze it before any other
            // trigger considers it. This single ladder step lets a
            // frozen sub-tree return to active learning when the
            // distribution it was trained on shifts.
            if self.nodes[nid_idx].is_frozen
                && self.nodes[nid_idx].drift_baseline.is_some()
                && self.nodes[nid_idx].density.is_some()
            {
                let drift_score = self.nodes[nid_idx]
                    .drift_baseline
                    .as_ref()
                    .unwrap()
                    .drift_score(self.nodes[nid_idx].density.as_ref().unwrap())
                    .unwrap_or(0.0);
                if drift_score > policy.drift_unfreeze() {
                    let _ = self.unfreeze(node_id);
                }
            }

            if self.nodes[nid_idx].is_frozen {
                continue;
            }

            // Recompute Maturity AFTER signature-stability advance so
            // `uncertainty_stable` sees the freshest counter.
            let maturity = crate::maturity::Maturity::from_node(&self.nodes[nid_idx]);

            // Auto-capture expected_epistemic.
            if maturity.uncertainty_stable
                && self.nodes[nid_idx].expected_epistemic.is_none()
                && self.nodes[nid_idx].blr.is_some()
            {
                if let Some(value) = epistemic_leverage_at_posterior_mean(
                    self.nodes[nid_idx].blr.as_ref().unwrap(),
                ) {
                    // Use the public path so the audit witness fires
                    // exactly as a manual capture would.
                    let _ = self.set_expected_epistemic(node_id, value);
                }
            }

            // Snapshot the current signature (post-advance) for trigger
            // comparisons. Avoids re-computing inside each branch.
            let new_sig: [u8; 32] = self.nodes[nid_idx]
                .last_signature
                .expect("just set by advance_signature_stability");

            // Step 4: Try Compress.
            if try_compress(self, node_id, &new_sig, &policy) {
                counts[ActionKind::Compress as usize] += 1;
                continue;
            }

            // Step 5: Try Merge.
            if try_merge(self, node_id, &new_sig, &policy) {
                counts[ActionKind::Merge as usize] += 1;
                continue;
            }

            // Step 6: Try Split.
            if try_split(self, node_id, &maturity, &policy) {
                counts[ActionKind::Split as usize] += 1;
                continue;
            }

            // Step 7: Try Prune.
            if try_prune(self, node_id, &maturity, &policy) {
                counts[ActionKind::Prune as usize] += 1;
                continue;
            }

            // Step 8: Try Grow.
            if try_grow(self, node_id, &maturity, &policy) {
                counts[ActionKind::Grow as usize] += 1;
                continue;
            }

            // Step 9: Try Freeze.
            if try_freeze(self, node_id, &policy) {
                counts[ActionKind::Freeze as usize] += 1;
                continue;
            }
        }

        counts
    }

    /// Phase 0.4 Track B-2.2.2 — advance the per-node 3-window
    /// stability buffers (`ece_history`, `sigma_history`) by one
    /// window. Each call shifts the existing values left
    /// (`history[0] ← history[1]`, `history[1] ← history[2]`) and
    /// records the freshest reading in `history[2]`. Fill counters
    /// saturate at 3.
    ///
    /// Called from `decide_step` once per node per call. Public so
    /// users (and tests) without a full decision policy can drive the
    /// buffers manually.
    pub fn advance_stability_history(&mut self, node_id: NodeId) {
        let nid_idx = node_id as usize;
        if nid_idx >= self.nodes.len() {
            return;
        }
        // Snapshot scalars first to avoid mutable-borrow aliasing.
        let ece_opt = self.nodes[nid_idx]
            .calibration
            .as_ref()
            .map(|cal| cal.ece());
        let sigma_opt = self.nodes[nid_idx]
            .blr
            .as_ref()
            .and_then(epistemic_leverage_at_posterior_mean);

        let node = &mut self.nodes[nid_idx];
        // ECE window: only fills when calibration is installed.
        if let Some(ece) = ece_opt {
            let h = &mut node.ece_history;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = ece;
            if node.ece_fill_count < 3 {
                node.ece_fill_count += 1;
            }
        }
        // σ window: only fills when BLR posterior is installed AND
        // `epistemic_leverage_at_posterior_mean` produces a positive
        // finite value (zero / non-finite means "no signal yet" —
        // also covers singular Cholesky on a pre-Welford state).
        if let Some(sigma) = sigma_opt {
            let h = &mut node.sigma_history;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = sigma;
            if node.sigma_fill_count < 3 {
                node.sigma_fill_count += 1;
            }
        }
    }

    /// Phase 0.4 Track B-2.2.1 — fold one fresh observation into each
    /// of the four `SignatureWelford` accumulators on `node_id`.
    /// Called from `advance_signature_stability` so the per-call
    /// Welford-derived signature reflects the most-recent metric
    /// readings before the stability comparison.
    ///
    /// Observation rules:
    /// - prediction:  always (uses `NodeStats.mean`)
    /// - uncertainty: only if BLR installed AND
    ///   `epistemic_leverage_at_posterior_mean` returns Some
    /// - calibration: only if calibration bins installed
    /// - routing:     always (uses `routing_observation_value` —
    ///   even an empty children container produces a defined value)
    fn advance_signature_welfords(&mut self, node_id: NodeId) {
        let nid_idx = node_id as usize;
        // Snapshot every scalar reading via immutable borrow first,
        // then mutate the Welfords via independent mutable borrows —
        // sidesteps `&mut self.nodes[i]` aliasing issues.
        let mean = self.nodes[nid_idx].stats.mean;
        let lev_opt = self.nodes[nid_idx]
            .blr
            .as_ref()
            .and_then(epistemic_leverage_at_posterior_mean);
        let ece_opt = self
            .nodes[nid_idx]
            .calibration
            .as_ref()
            .map(|cal| cal.ece());
        let routing_value =
            crate::signature::routing_observation_value(&self.nodes[nid_idx]);

        let node = &mut self.nodes[nid_idx];
        node.welford_prediction.observe(mean);
        if let Some(lev) = lev_opt {
            node.welford_uncertainty.observe(lev);
        }
        if let Some(ece) = ece_opt {
            node.welford_calibration.observe(ece);
        }
        node.welford_routing.observe(routing_value);
    }

    /// Internal — advance `last_signature` + `signature_stable_calls`
    /// for one node based on its current state. Used by `decide_step`
    /// for both active and frozen nodes.
    ///
    /// Phase 0.4 Track B-2.2.1 — folds one observation into each
    /// `SignatureWelford` BEFORE computing the new signature, so the
    /// signature is always Welford-smoothed.
    fn advance_signature_stability(&mut self, node_id: NodeId) {
        self.advance_signature_welfords(node_id);
        let nid_idx = node_id as usize;
        let new_sig = NodeSignature::from_node(&self.nodes[nid_idx]).canonical_bytes();
        match self.nodes[nid_idx].last_signature {
            Some(prev) if prev == new_sig => {
                self.nodes[nid_idx].signature_stable_calls = self.nodes[nid_idx]
                    .signature_stable_calls
                    .saturating_add(1);
            }
            _ => {
                self.nodes[nid_idx].signature_stable_calls = 0;
            }
        }
        self.nodes[nid_idx].last_signature = Some(new_sig);
    }
}

// ── Phase 0.3d-4: trigger helpers ───────────────────────────────────

/// Byte-Hamming distance between two 32-byte signatures: count of
/// differing bytes, 0..=32. Used by Compress / Merge triggers, both
/// of which compare against [`DecisionPolicy::tau_compress`] /
/// [`DecisionPolicy::tau_merge`] thresholds in the `[0, 32]` range.
fn hamming_byte_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u32
}

/// Compute the BLR's epistemic *leverage* at the posterior mean. Used
/// as the canonical "training-time reference" when auto-capturing
/// `expected_epistemic`. Returns `None` when `predict` errors or
/// produces a non-positive / non-finite value.
///
/// Note: the value stored in `expected_epistemic` is leverage, not
/// variance — Phase 0.4 Track C-2.3.1 corrected the naming. The OOD
/// ratio `(epi / expected).clamp(0, 1)` works on its own terms because
/// units cancel; external callers who want output-unit predictive
/// variance should multiply by `aleatoric_var` themselves (see
/// [`BlrState::predict`]).
fn epistemic_leverage_at_posterior_mean(blr: &BlrState) -> Option<f64> {
    let phi = blr.mean.to_vec();
    match blr.predict(&phi) {
        Ok((_m, lev, _ale)) if lev.is_finite() && lev > 0.0 => Some(lev),
        _ => None,
    }
}

/// Try to fire a Compress action on `node_id`. Returns `true` if
/// fired. Criterion: node has at least one routable child AND every
/// immediate child's signature lies within τ_compress Hamming of
/// `new_sig`.
fn try_compress(
    g: &mut AdaptiveBeliefGraph,
    node_id: NodeId,
    new_sig: &[u8; 32],
    policy: &DecisionPolicy,
) -> bool {
    let nid_idx = node_id as usize;
    if matches!(g.nodes[nid_idx].children, AdaptiveChildren::None)
        || g.nodes[nid_idx].children.is_dense()
    {
        return false;
    }
    let pairs = g.nodes[nid_idx].children.iter();
    if pairs.is_empty() {
        return false;
    }
    let tau = policy.tau_compress() as u32;
    let all_close = pairs.iter().all(|&(_, child_id)| {
        let child_sig =
            NodeSignature::from_node(&g.nodes[child_id as usize]).canonical_bytes();
        hamming_byte_distance(new_sig, &child_sig) <= tau
    });
    if all_close {
        g.force_compress(node_id).is_ok()
    } else {
        false
    }
}

/// Try to fire a Merge action on `node_id`. Returns `true` if fired.
/// Criterion: node has a sibling with a smaller `NodeId` whose
/// signature lies within τ_merge Hamming. The smaller-NodeId
/// tiebreak makes the action deterministic in replay.
fn try_merge(
    g: &mut AdaptiveBeliefGraph,
    node_id: NodeId,
    new_sig: &[u8; 32],
    policy: &DecisionPolicy,
) -> bool {
    let parent = match g.nodes[node_id as usize].parent {
        Some(p) => p,
        None => return false,
    };
    let tau = policy.tau_merge() as u32;
    let kl_thresh = policy.kl_merge();
    let siblings: Vec<NodeId> = g.nodes[parent as usize]
        .children
        .iter()
        .into_iter()
        .map(|(_, id)| id)
        .filter(|&id| id < node_id)
        .filter(|&id| g.nodes[id as usize].is_active)
        .filter(|&id| !g.nodes[id as usize].is_frozen)
        .collect();
    for sib_id in siblings {
        let sib_sig =
            NodeSignature::from_node(&g.nodes[sib_id as usize]).canonical_bytes();
        if hamming_byte_distance(new_sig, &sib_sig) > tau {
            continue;
        }
        // Phase 0.4 Track B-2.2.3 — KL-divergence gate. If both nodes
        // have BLR posteriors installed, only merge when the
        // weight-distribution KL falls below `policy.kl_merge()`.
        // Without BLR (graph has no prior), the gate is skipped — the
        // pre-0.4 Hamming-only behavior is preserved for graphs that
        // never installed a BLR head.
        if let (Some(blr_self), Some(blr_sib)) = (
            g.nodes[node_id as usize].blr.as_ref(),
            g.nodes[sib_id as usize].blr.as_ref(),
        ) {
            match blr_self.kl_divergence(blr_sib) {
                Ok(kl) if kl <= kl_thresh => {}
                _ => continue, // KL too large or compute error → skip
            }
        }
        if g.force_merge(node_id, sib_id).is_ok() {
            return true;
        }
    }
    false
}

/// Try to fire a Split action on `node_id`. Returns `true` if fired.
/// Criterion (Phase 0.4 Track B-2.2.4):
/// 1. node is a leaf
/// 2. `samples_seen ≥ split_min`
/// 3. `variance ≥ impurity_min` (impurity gate)
/// 4. estimated held-out ΔNLL gain `≥ nll_split_gain` (deterministic
///    bootstrap on samples drawn from the node's Gaussian model)
fn try_split(
    g: &mut AdaptiveBeliefGraph,
    node_id: NodeId,
    maturity: &crate::maturity::Maturity,
    policy: &DecisionPolicy,
) -> bool {
    if !g.nodes[node_id as usize].is_leaf() {
        return false;
    }
    if maturity.samples_seen < policy.split_min() {
        return false;
    }
    // Phase 0.4 Track B-2.2.4 — impurity + ΔNLL gates.
    let stats = &g.nodes[node_id as usize].stats;
    let var = stats.variance();
    if var < policy.impurity_min() {
        return false;
    }
    let action_count_sum: u64 = g.action_counts.iter().sum();
    let dnll = estimate_split_nll_gain(stats, g.seed, node_id, action_count_sum);
    if dnll < policy.nll_split_gain() {
        return false;
    }
    g.force_split(node_id).is_ok()
}

/// Estimate the held-out ΔNLL gain of a hypothetical 50-50 median
/// split of the node's observed distribution. Phase 0.4 Track B-2.2.4
/// — uses a deterministic bootstrap on samples drawn from the node's
/// Gaussian sufficient statistics `(μ, σ²)`. Per-node observation
/// history would let us bootstrap on real data; that's deferred to
/// 0.5+ since adding history is a snapshot-format change beyond
/// Phase 0.4's scope. The synthetic bootstrap uses the node's
/// Gaussian model as the data-generating process, which is what the
/// BLR posterior would predict at the leaf anyway.
///
/// Determinism: seeded from `(graph.seed, node_id, action_count_sum)`
/// so two replays of the same audit log produce bit-identical ΔNLL
/// estimates (and therefore bit-identical Split decisions).
fn estimate_split_nll_gain(
    stats: &crate::stats::NodeStats,
    seed: u64,
    node_id: NodeId,
    action_count_sum: u64,
) -> f64 {
    const N_BOOT: usize = 32;
    if stats.n_seen < 4 {
        return 0.0;
    }
    let var = stats.variance();
    if var <= f64::EPSILON {
        return 0.0;
    }
    let mu = stats.mean;
    let sigma = var.sqrt();

    // Deterministic seed mixing all three inputs through SplitMix64
    // multiplications so adjacent (node_id, action_count_sum) values
    // produce well-spread streams.
    let bootstrap_seed = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((node_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(action_count_sum.wrapping_mul(0x94D0_49BB_1331_11EB));
    let mut rng = cjc_repro::Rng::seeded(bootstrap_seed);

    // Draw N_BOOT samples from N(μ, σ²).
    let mut samples = [0.0f64; N_BOOT];
    for s in samples.iter_mut() {
        *s = mu + sigma * rng.next_normal_f64();
    }

    // 50/50 train/test partition (deterministic order).
    let n_train = N_BOOT / 2;
    let train = &samples[..n_train];
    let test = &samples[n_train..];

    // Pre-split: fit single Gaussian on train, NLL on test.
    let (mean_pre, var_pre) = fit_gaussian(train);
    let nll_pre = nll_under_gaussian(test, mean_pre, var_pre);

    // Post-split: median-partition train into two groups; NLL on test
    // is the per-sample minimum of the two groups' NLLs (each test
    // sample is "routed" to its better-fitting group).
    let mut sorted = [0.0f64; N_BOOT / 2];
    sorted.copy_from_slice(train);
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let group_a: Vec<f64> = train.iter().copied().filter(|&x| x < median).collect();
    let group_b: Vec<f64> = train.iter().copied().filter(|&x| x >= median).collect();
    if group_a.is_empty() || group_b.is_empty() {
        return 0.0;
    }
    let (mean_a, var_a) = fit_gaussian(&group_a);
    let (mean_b, var_b) = fit_gaussian(&group_b);
    let nll_post: f64 = test
        .iter()
        .map(|&x| {
            let nll_a = single_sample_nll(x, mean_a, var_a);
            let nll_b = single_sample_nll(x, mean_b, var_b);
            nll_a.min(nll_b)
        })
        .sum();

    (nll_pre - nll_post).max(0.0)
}

fn fit_gaussian(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, f64::EPSILON);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = if data.len() > 1 {
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        f64::EPSILON
    };
    (mean, var.max(f64::EPSILON))
}

fn single_sample_nll(x: f64, mean: f64, var: f64) -> f64 {
    let z = (x - mean).powi(2) / var;
    0.5 * (z + var.ln() + (2.0 * std::f64::consts::PI).ln())
}

fn nll_under_gaussian(samples: &[f64], mean: f64, var: f64) -> f64 {
    samples.iter().map(|&x| single_sample_nll(x, mean, var)).sum()
}

/// Try to fire a Prune action on `node_id`. Returns `true` if fired.
/// Criterion: `samples_seen < prune_floor` AND signature stable for
/// at least `prune_grace_epochs` consecutive `decide_step` calls.
/// Never prunes the root (`node_id == 0`).
fn try_prune(
    g: &mut AdaptiveBeliefGraph,
    node_id: NodeId,
    maturity: &crate::maturity::Maturity,
    policy: &DecisionPolicy,
) -> bool {
    if node_id == 0 {
        return false; // never prune the root
    }
    if maturity.samples_seen >= policy.prune_floor() {
        return false;
    }
    if g.nodes[node_id as usize].signature_stable_calls < policy.prune_grace_epochs() {
        return false;
    }
    g.force_prune(node_id).is_ok()
}

/// Try to fire a Grow action on `node_id`. Returns `true` if fired.
/// Criterion (simplified): node is a leaf AND `samples_seen ≥ grow_min`
/// AND a deterministic-from-`(seed, node_id)` key byte isn't already
/// bound. Phase 0.4 will gate on route entropy.
fn try_grow(
    g: &mut AdaptiveBeliefGraph,
    node_id: NodeId,
    maturity: &crate::maturity::Maturity,
    policy: &DecisionPolicy,
) -> bool {
    if !g.nodes[node_id as usize].is_leaf() {
        return false;
    }
    if maturity.samples_seen < policy.grow_min() {
        return false;
    }
    // Phase 0.4 Track B-2.2.5 — route-entropy gate. When a codebook is
    // installed, the routing concept is well-defined; gate Grow on
    // Shannon entropy of the candidate's depth being above
    // `policy.h_grow()`. Codebook-less graphs fall back to pre-0.4
    // Hamming-only behavior for backward compatibility (existing
    // decide_step tests run without a codebook).
    if g.codebook.is_some() {
        let h = route_key_entropy_at_candidate_depth(g, node_id);
        if h <= policy.h_grow() {
            return false;
        }
    }
    let mix = g
        .seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(node_id as u64);
    let key = mix as u8;
    g.force_grow(node_id, key).is_ok()
}

/// Shannon entropy of the key-byte distribution among nodes at the
/// candidate's depth — proxied by the candidate's parent's children
/// key bytes. Returns `f64::INFINITY` for the root (no parent →
/// bootstrap allowed) or for parents with fewer than two children
/// (single-child parents still warrant their first growth).
///
/// Phase 0.4 Track B-2.2.5 — gates `try_grow` so that nodes in
/// low-diversity routes don't keep spawning useless children.
fn route_key_entropy_at_candidate_depth(g: &AdaptiveBeliefGraph, node_id: NodeId) -> f64 {
    let parent = match g.nodes[node_id as usize].parent {
        Some(p) => p,
        None => return f64::INFINITY,
    };
    let mut counts = [0u64; 256];
    let mut total = 0u64;
    for (key, _id) in g.nodes[parent as usize].children.iter() {
        counts[key as usize] = counts[key as usize].saturating_add(1);
        total = total.saturating_add(1);
    }
    if total < 2 {
        return f64::INFINITY;
    }
    let total_f = total as f64;
    let mut h_acc = cjc_repro::KahanAccumulatorF64::new();
    for &c in counts.iter() {
        if c > 0 {
            let p = (c as f64) / total_f;
            h_acc.add(-p * p.ln());
        }
    }
    h_acc.finalize()
}

/// Try to fire a Freeze action on `node_id`. Returns `true` if fired.
/// Criterion: signature stable for `freeze_after` consecutive
/// `decide_step` calls.
fn try_freeze(
    g: &mut AdaptiveBeliefGraph,
    node_id: NodeId,
    policy: &DecisionPolicy,
) -> bool {
    if g.nodes[node_id as usize].signature_stable_calls < policy.freeze_after() {
        return false;
    }
    g.force_freeze(node_id).is_ok()
}

/// Penultimate-feature dim for a leaf head: the last hidden-layer width,
/// or `input_dim` if `hidden_dims` is empty (the BLR-on-raw-input
/// degenerate case).
pub(crate) fn blr_feature_dim(head: &LeafHead) -> u32 {
    *head.hidden_dims.last().unwrap_or(&head.input_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_graph_has_one_node_and_one_event() {
        let g = AdaptiveBeliefGraph::new(42);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.audit_len(), 1);
        assert_eq!(g.audit[0].kind, AuditKind::Created);
        assert_eq!(g.audit[0].previous_hash, genesis_hash());
        assert_eq!(g.chain_head, g.audit[0].new_hash);
        assert_eq!(g.nodes[0].parent, None);
    }

    #[test]
    fn observe_advances_global_and_per_node_chain() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let global_before = g.chain_head;
        let per_node_before = g.nodes[0].stats_chain_head;
        g.observe(0, 1.5).unwrap();
        assert_ne!(g.chain_head, global_before);
        assert_ne!(g.nodes[0].stats_chain_head, per_node_before);
    }

    #[test]
    fn add_node_creates_child_and_records_audit() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let new_id = g.add_node(0, 7).unwrap();
        assert_eq!(new_id, 1);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.nodes[1].parent, Some(0));
        // children promoted None -> Node4 + NodeAdded event
        assert_eq!(g.audit_len(), 3); // Created + ChildrenPromoted + NodeAdded
        assert!(matches!(
            g.audit[1].kind,
            AuditKind::ChildrenPromoted { from: 0, to: 1 }
        ));
        assert!(matches!(
            g.audit[2].kind,
            AuditKind::NodeAdded { parent: 0, key_byte: 7 }
        ));
    }

    #[test]
    fn add_node_duplicate_key_errs() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.add_node(0, 7).unwrap();
        let err = g.add_node(0, 7).unwrap_err();
        assert_eq!(
            err,
            GraphError::KeyAlreadyBound {
                parent: 0,
                key_byte: 7
            }
        );
    }

    #[test]
    fn add_node_promotes_through_all_classes() {
        let mut g = AdaptiveBeliefGraph::new(0);
        // Add 100 children — drives root through Node4→16→48→256.
        for k in 0u8..100 {
            g.add_node(0, k).unwrap();
        }
        assert_eq!(g.nodes[0].children.kind(), ChildrenKind::Node256);
        // Three promotions: Node4→16, Node16→48, Node48→256.
        let promotions: Vec<_> = g
            .audit
            .iter()
            .filter_map(|e| match &e.kind {
                AuditKind::ChildrenPromoted { from, to } => Some((*from, *to)),
                _ => None,
            })
            .collect();
        // None→Node4 happens on the first add; then 4→16 at the 5th, 16→48 at
        // the 17th, 48→256 at the 49th.
        assert_eq!(promotions, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn descend_walks_root_then_first_match() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let n1 = g.add_node(0, 7).unwrap();
        let n2 = g.add_node(n1, 9).unwrap();
        let route = g.descend(&[7, 9]);
        assert_eq!(route.matched_prefix, 2);
        assert_eq!(route.leaf_id, n2);
        assert_eq!(route.path, vec![0, n1, n2]);
    }

    #[test]
    fn descend_bails_on_first_unmatched_byte() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let n1 = g.add_node(0, 7).unwrap();
        let route = g.descend(&[7, 99]);
        assert_eq!(route.matched_prefix, 1);
        assert_eq!(route.leaf_id, n1);
        assert_eq!(route.path, vec![0, n1]);
    }

    #[test]
    fn descend_root_only_when_first_byte_unbound() {
        let g = AdaptiveBeliefGraph::new(0);
        let route = g.descend(&[5, 1]);
        assert_eq!(route.matched_prefix, 0);
        assert_eq!(route.leaf_id, 0);
        assert_eq!(route.path, vec![0]);
    }

    #[test]
    fn set_codebook_freezes_and_emits_event() {
        let mut g = AdaptiveBeliefGraph::new(0);
        // 2 dims × 4 bins → 3 boundaries each
        let flat = vec![0.5, 1.5, 2.5, 0.5, 1.5, 2.5];
        g.set_codebook(2, 4, &flat).unwrap();
        assert!(g.codebook.is_some());
        assert!(matches!(
            g.audit.last().unwrap().kind,
            AuditKind::CodebookFrozen { .. }
        ));
        // Second install must error.
        let err = g.set_codebook(2, 4, &flat).unwrap_err();
        assert_eq!(err, GraphError::CodebookAlreadyFrozen);
    }

    #[test]
    fn encode_prefix_round_trip() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let flat = vec![0.5, 1.5, 2.5, 0.5, 1.5, 2.5];
        g.set_codebook(2, 4, &flat).unwrap();
        let p = g.encode_prefix(&[1.0, 2.0]).unwrap();
        assert_eq!(p, vec![1, 2]);
    }

    #[test]
    fn encode_prefix_without_codebook_errs() {
        let g = AdaptiveBeliefGraph::new(0);
        let err = g.encode_prefix(&[1.0]).unwrap_err();
        assert_eq!(err, GraphError::NoCodebook);
    }

    #[test]
    fn chain_verifies_after_complex_mutation() {
        let mut g = AdaptiveBeliefGraph::new(0);
        for k in 0u8..20 {
            g.add_node(0, k).unwrap();
        }
        for i in 0..50 {
            g.observe(0, i as f64).unwrap();
        }
        let flat = vec![0.5, 1.5, 2.5];
        g.set_codebook(1, 4, &flat).unwrap();
        assert!(g.verify_chain().is_ok());
    }

    #[test]
    fn determinism_double_run_chain_head() {
        let mk = || {
            let mut g = AdaptiveBeliefGraph::new(7);
            for k in 0u8..30 {
                g.add_node(0, k).unwrap();
            }
            for i in 0..100 {
                g.observe(0, (i as f64) * 0.01).unwrap();
            }
            g
        };
        let a = mk();
        let b = mk();
        assert_eq!(a.chain_head, b.chain_head);
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.stats_chain_head, nb.stats_chain_head);
        }
    }

    #[test]
    fn per_node_chain_independent_of_global() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let n1 = g.add_node(0, 1).unwrap();
        // Observe only node 0; node 1's per-node chain must NOT advance.
        let head_n1_before = g.nodes[n1 as usize].stats_chain_head;
        g.observe(0, 1.0).unwrap();
        let head_n1_after = g.nodes[n1 as usize].stats_chain_head;
        assert_eq!(head_n1_before, head_n1_after);
        // Global chain advanced (BeliefUpdate event was appended).
        assert!(g.audit.iter().any(|e| matches!(e.kind, AuditKind::BeliefUpdate { .. })));
    }

    // ── Phase 0.3d-2: expected_epistemic ─────────────────────────

    fn graph_with_blr() -> AdaptiveBeliefGraph {
        let mut g = AdaptiveBeliefGraph::new(7);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g
    }

    #[test]
    fn set_expected_epistemic_happy_path() {
        let mut g = graph_with_blr();
        assert_eq!(g.expected_epistemic(0).unwrap(), None);
        g.set_expected_epistemic(0, 0.5).unwrap();
        assert_eq!(g.expected_epistemic(0).unwrap(), Some(0.5));
        // Capture event landed on the chain.
        assert!(matches!(
            g.audit.last().unwrap().kind,
            AuditKind::ExpectedEpistemicCaptured { .. }
        ));
        // Chain still verifies.
        assert!(g.verify_chain().is_ok());
    }

    #[test]
    fn set_expected_epistemic_one_shot() {
        let mut g = graph_with_blr();
        g.set_expected_epistemic(0, 0.5).unwrap();
        let err = g.set_expected_epistemic(0, 0.7).unwrap_err();
        assert!(matches!(
            err,
            GraphError::ExpectedEpistemicAlreadyCaptured { node_id: 0 }
        ));
    }

    #[test]
    fn set_expected_epistemic_requires_blr() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let err = g.set_expected_epistemic(0, 0.5).unwrap_err();
        assert!(matches!(err, GraphError::ExpectedEpistemicNoBlr));
    }

    #[test]
    fn set_expected_epistemic_validates_value() {
        let mut g = graph_with_blr();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = g.set_expected_epistemic(0, bad).unwrap_err();
            assert!(
                matches!(err, GraphError::ExpectedEpistemicInvalidValue(_)),
                "expected InvalidValue for {bad}, got {err:?}"
            );
        }
    }

    #[test]
    fn set_expected_epistemic_out_of_range_node() {
        let mut g = graph_with_blr();
        let err = g.set_expected_epistemic(99, 0.5).unwrap_err();
        assert!(matches!(err, GraphError::NodeOutOfRange { .. }));
    }

    #[test]
    fn ood_score_uses_ratio_when_captured() {
        // Build a node with a BLR posterior and a known epistemic var.
        // Without capture: ood_score uses raw clamped epistemic.
        // With a small captured reference: ratio formula amplifies the
        // signal, pushing OOD up. Verifies the formula switch fires.
        let mut g = graph_with_blr();
        // Train BLR a bit so epistemic var has a stable value.
        let xs: Vec<f64> = (0..20)
            .flat_map(|i| [(i as f64) * 0.1, ((i % 5) as f64) * 0.1])
            .collect();
        let ys: Vec<f64> = (0..20).map(|i| (i as f64) * 0.05).collect();
        g.blr_update(0, &xs, &ys).unwrap();
        let phi = vec![3.0, 3.0];
        let raw = g.ood_score(0, &phi, 0, 0).unwrap();
        // Capture a *small* reference so (epi / expected) >= raw — the
        // captured-ratio branch should produce a value ≥ the unclamped
        // raw branch (or equal at saturation).
        let (_m, epi_var, _ale) = g.nodes[0]
            .blr
            .as_ref()
            .unwrap()
            .predict(&phi)
            .unwrap();
        let expected = (epi_var * 0.5).max(1e-9); // half of current epi var
        g.set_expected_epistemic(0, expected).unwrap();
        let calibrated = g.ood_score(0, &phi, 0, 0).unwrap();
        assert!(
            calibrated >= raw - 1e-12,
            "expected calibrated ({calibrated}) >= raw ({raw})"
        );
    }

    #[test]
    fn ood_score_falls_back_when_uncaptured() {
        // Without capture, ood_score must equal the Phase 0.3c behaviour:
        // raw clamped epistemic var.
        let mut g = graph_with_blr();
        g.blr_update(0, &[1.0, 1.0, 2.0, 2.0], &[1.0, 2.0]).unwrap();
        let phi = vec![0.5, 0.5];
        let s = g.ood_score(0, &phi, 0, 0).unwrap();
        assert!((0.0..=1.0).contains(&s));
        // Confirm uncaptured.
        assert_eq!(g.expected_epistemic(0).unwrap(), None);
    }

    #[test]
    fn determinism_double_run_with_capture() {
        let mk = || {
            let mut g = graph_with_blr();
            g.set_expected_epistemic(0, 0.42).unwrap();
            (g.chain_head, g.expected_epistemic(0).unwrap())
        };
        assert_eq!(mk(), mk());
    }

    // ── Phase 0.3d-3: structural decisions ───────────────────────

    fn ok_thresholds() -> [f64; 12] {
        [
            0.5, 64.0, 128.0, 0.05, 0.02,
            4.0, 0.1, 32.0, 10.0, 8.0,
            20.0, f64::MAX, // drift_unfreeze disabled
        ]
    }

    #[test]
    fn set_decision_policy_happy_path() {
        let mut g = AdaptiveBeliefGraph::new(0);
        assert!(g.decision_policy.is_none());
        g.set_decision_policy(&ok_thresholds()).unwrap();
        assert!(g.decision_policy.is_some());
        let h = g.decision_policy_hash();
        assert!(h.is_some());
    }

    #[test]
    fn set_decision_policy_one_shot() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let err = g.set_decision_policy(&ok_thresholds()).unwrap_err();
        assert_eq!(err, GraphError::DecisionPolicyAlreadyFrozen);
    }

    #[test]
    fn set_decision_policy_validates() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let err = g.set_decision_policy(&[0.0; 5]).unwrap_err();
        assert!(matches!(err, GraphError::DecisionPolicy(_)));
    }

    #[test]
    fn force_grow_emits_grow_event_and_bumps_counter() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let c = g.force_grow(0, 7).unwrap();
        assert_eq!(c, 1);
        assert!(matches!(
            g.audit.last().unwrap().kind,
            AuditKind::Grow { .. }
        ));
        assert_eq!(g.action_count(ActionKind::Grow), 1);
        // Chain still verifies.
        assert!(g.verify_chain().is_ok());
    }

    #[test]
    fn force_grow_blocked_on_frozen_node() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_freeze(0).unwrap();
        let err = g.force_grow(0, 7).unwrap_err();
        assert!(matches!(err, GraphError::NodeFrozen { node_id: 0 }));
    }

    #[test]
    fn force_grow_blocked_on_dense_node() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_compress(0).unwrap();
        let err = g.force_grow(0, 7).unwrap_err();
        assert!(matches!(err, GraphError::NodeIsDense { node_id: 0 }));
    }

    #[test]
    fn force_grow_blocked_on_pruned_node() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let c = g.add_node(0, 1).unwrap();
        g.force_prune(c).unwrap();
        let err = g.force_grow(c, 7).unwrap_err();
        assert!(matches!(err, GraphError::NodeNotActive { .. }));
    }

    #[test]
    fn force_split_emits_split_and_pushes_two_children() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let (a, b) = g.force_split(0).unwrap();
        assert!(a < b);
        assert_eq!(g.node_count(), 3);
        let kinds: Vec<_> = g.audit.iter().map(|e| e.kind.clone()).collect();
        assert!(kinds.iter().any(|k| matches!(k, AuditKind::Split { .. })));
        assert_eq!(g.action_count(ActionKind::Split), 1);
    }

    #[test]
    fn force_split_requires_leaf() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let _c = g.add_node(0, 1).unwrap();
        let err = g.force_split(0).unwrap_err();
        assert!(matches!(err, GraphError::ForceSplitNotLeaf { .. }));
    }

    #[test]
    fn force_merge_marks_absorbed_inactive() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let a = g.add_node(0, 1).unwrap();
        let b = g.add_node(0, 2).unwrap();
        g.force_merge(a, b).unwrap();
        assert!(!g.is_active(a).unwrap());
        assert!(g.is_active(b).unwrap());
        assert_eq!(g.action_count(ActionKind::Merge), 1);
    }

    #[test]
    fn force_merge_self_errs() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let err = g.force_merge(0, 0).unwrap_err();
        assert_eq!(err, GraphError::ForceMergeSelf);
    }

    #[test]
    fn force_merge_already_absorbed_errs() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let a = g.add_node(0, 1).unwrap();
        g.force_prune(a).unwrap();
        let err = g.force_merge(a, 0).unwrap_err();
        assert!(matches!(err, GraphError::ForceMergeAlreadyAbsorbed { .. }));
    }

    #[test]
    fn force_prune_marks_inactive_and_emits_event() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let c = g.add_node(0, 1).unwrap();
        g.force_prune(c).unwrap();
        assert!(!g.is_active(c).unwrap());
        assert!(matches!(
            g.audit.last().unwrap().kind,
            AuditKind::Prune { .. }
        ));
        assert_eq!(g.action_count(ActionKind::Prune), 1);
    }

    #[test]
    fn force_prune_already_pruned_errs() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_prune(0).unwrap();
        let err = g.force_prune(0).unwrap_err();
        assert!(matches!(err, GraphError::NodeNotActive { node_id: 0 }));
    }

    #[test]
    fn force_compress_replaces_children_with_dense() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.add_node(0, 7).unwrap();
        g.force_compress(0).unwrap();
        assert!(matches!(
            g.nodes[0].children,
            AdaptiveChildren::Dense { .. }
        ));
        assert_eq!(g.action_count(ActionKind::Compress), 1);
    }

    #[test]
    fn force_compress_idempotent_after_dense() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_compress(0).unwrap();
        let err = g.force_compress(0).unwrap_err();
        assert!(matches!(err, GraphError::NodeIsDense { node_id: 0 }));
    }

    #[test]
    fn force_freeze_idempotent() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_freeze(0).unwrap();
        let count_before = g.action_count(ActionKind::Freeze);
        // Re-freezing is a no-op (no event, no counter bump).
        g.force_freeze(0).unwrap();
        assert_eq!(g.action_count(ActionKind::Freeze), count_before);
    }

    #[test]
    fn action_count_index_round_trip() {
        for i in 0..N_ACTION_KINDS as u8 {
            let k = ActionKind::from_index(i).unwrap();
            assert_eq!(k as u8, i);
        }
        assert!(ActionKind::from_index(99).is_none());
    }

    #[test]
    fn force_actions_chain_verifies() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let c = g.force_grow(0, 1).unwrap();
        let _ = g.force_grow(0, 2).unwrap();
        g.force_freeze(c).unwrap();
        let pruned = g.add_node(0, 3).unwrap();
        g.force_prune(pruned).unwrap();
        assert!(g.verify_chain().is_ok());
    }

    #[test]
    fn determinism_double_run_full_p3d3() {
        // Cover Split + Grow + Freeze in one sequence. Split first
        // (root must be a leaf at that point), then grow on one of the
        // new leaves, then freeze the other.
        let mk = || {
            let mut g = AdaptiveBeliefGraph::new(7);
            g.set_decision_policy(&ok_thresholds()).unwrap();
            let (a, b) = g.force_split(0).unwrap();
            let _gc = g.force_grow(a, 33).unwrap();
            g.force_freeze(b).unwrap();
            (g.chain_head, g.action_counts)
        };
        assert_eq!(mk(), mk());
    }

    // ── Phase 0.3d-4: decide_step + unfreeze ─────────────────────

    #[test]
    fn unfreeze_emits_event_and_clears_flag() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_freeze(0).unwrap();
        assert!(g.is_frozen(0).unwrap());
        g.unfreeze(0).unwrap();
        assert!(!g.is_frozen(0).unwrap());
        assert!(matches!(
            g.audit.last().unwrap().kind,
            AuditKind::Unfreeze { node_id: 0 }
        ));
        assert!(g.verify_chain().is_ok());
    }

    #[test]
    fn unfreeze_idempotent_on_active_node() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let pre_len = g.audit.len();
        g.unfreeze(0).unwrap();
        assert_eq!(g.audit.len(), pre_len, "unfreeze on active node is no-op");
    }

    #[test]
    fn unfreeze_does_not_bump_action_counts() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_freeze(0).unwrap();
        let counts_before = g.action_counts;
        g.unfreeze(0).unwrap();
        assert_eq!(g.action_counts, counts_before);
    }

    #[test]
    fn decide_step_no_op_without_policy() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let counts = g.decide_step();
        assert_eq!(counts, [0u64; 6]);
        // Stability tracking also doesn't advance.
        assert_eq!(g.nodes[0].last_signature, None);
        assert_eq!(g.nodes[0].signature_stable_calls, 0);
    }

    #[test]
    fn decide_step_advances_signature_stability() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let counts1 = g.decide_step();
        // First call captures last_signature, doesn't bump counter.
        assert!(g.nodes[0].last_signature.is_some());
        assert_eq!(g.nodes[0].signature_stable_calls, 0);
        // Second call (no state change) bumps counter to 1.
        let _ = counts1;
        let _ = g.decide_step();
        assert_eq!(g.nodes[0].signature_stable_calls, 1);
        let _ = g.decide_step();
        assert_eq!(g.nodes[0].signature_stable_calls, 2);
    }

    #[test]
    fn decide_step_signature_change_resets_stability() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let _ = g.decide_step();
        let _ = g.decide_step();
        assert_eq!(g.nodes[0].signature_stable_calls, 1);
        // Add a child — routing profile changes → signature changes.
        g.add_node(0, 7).unwrap();
        let _ = g.decide_step();
        assert_eq!(
            g.nodes[0].signature_stable_calls, 0,
            "signature change should reset counter"
        );
    }

    #[test]
    fn decide_step_freeze_after_stability() {
        // freeze_after = 20 in ok_thresholds. After 21 stable calls
        // (first call captures, next 20 bump) Freeze should fire.
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        // First call captures. 20 more bump counter. The 21st call
        // sees signature_stable_calls == 20 → trigger fires.
        for _ in 0..21 {
            g.decide_step();
        }
        assert!(g.is_frozen(0).unwrap());
        assert_eq!(g.action_count(ActionKind::Freeze), 1);
    }

    #[test]
    fn decide_step_no_action_on_frozen_node() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        g.force_freeze(0).unwrap();
        let pre_action_counts = g.action_counts;
        let counts = g.decide_step();
        // No new action triggers.
        assert_eq!(counts, [0u64; 6]);
        // action_counts only has the prior force_freeze count.
        assert_eq!(g.action_counts, pre_action_counts);
    }

    #[test]
    fn decide_step_grow_fires_on_well_observed_leaf() {
        // grow_min = 64. Observe enough times, then decide_step.
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        for _ in 0..64 {
            g.observe(0, 1.0).unwrap();
        }
        // Two decide_step calls — first establishes stability (so
        // signature isn't changing), second fires Grow if eligible.
        // But Grow doesn't require stability — only samples_seen >=
        // grow_min and leaf-state. So one call should suffice.
        let counts = g.decide_step();
        assert_eq!(counts[ActionKind::Grow as usize], 1);
        assert_eq!(g.node_count(), 2); // root + new child
    }

    #[test]
    fn decide_step_returns_per_action_counts() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let counts = g.decide_step();
        assert_eq!(counts.len(), 6);
        for c in counts.iter() {
            assert_eq!(*c, 0);
        }
    }

    #[test]
    fn decide_step_preserves_chain_integrity() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        for _ in 0..70 {
            g.observe(0, 1.0).unwrap();
        }
        for _ in 0..5 {
            g.decide_step();
        }
        assert!(g.verify_chain().is_ok());
    }

    #[test]
    fn decide_step_determinism_double_run() {
        let mk = || {
            let mut g = AdaptiveBeliefGraph::new(42);
            g.set_decision_policy(&ok_thresholds()).unwrap();
            for _ in 0..70 {
                g.observe(0, 1.0).unwrap();
            }
            // Drive several decide_step calls.
            for _ in 0..3 {
                g.decide_step();
            }
            (g.chain_head, g.action_counts)
        };
        assert_eq!(mk(), mk());
    }

    #[test]
    fn hamming_byte_distance_known_values() {
        let a = [0u8; 32];
        let b = [0u8; 32];
        assert_eq!(super::hamming_byte_distance(&a, &b), 0);
        let mut c = [0u8; 32];
        c[5] = 1;
        c[10] = 2;
        assert_eq!(super::hamming_byte_distance(&a, &c), 2);
        let d = [0xFFu8; 32];
        assert_eq!(super::hamming_byte_distance(&a, &d), 32);
    }
}
