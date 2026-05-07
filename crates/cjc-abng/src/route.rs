//! Routing: descend the radix-tree from the root, matching prefix bytes.
//!
//! Phase 0.2 ships the *routing primitive* but not the structural decisions
//! that would react to its output. A model that wires `descend` into its
//! inference path can use [`RouteEvidence::matched_prefix`] vs the
//! configured `min_prefix_len` to decide whether to predict, abstain,
//! escalate, or fall back.

use crate::node::NodeId;

/// Evidence describing how a prefix routed through the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteEvidence {
    /// Number of prefix bytes successfully matched, `0..prefix.len()`.
    pub matched_prefix: u8,
    /// Final node reached. May be intermediate if descent bailed because
    /// the next byte wasn't bound to a child.
    pub leaf_id: NodeId,
    /// Path taken, root-first. Always non-empty (root is always included);
    /// `path.len() == matched_prefix as usize + 1`.
    pub path: Vec<NodeId>,
}

impl RouteEvidence {
    /// Construct an evidence for a descent that started and stopped at the
    /// root (zero matched bytes).
    pub fn root_only(root_id: NodeId) -> Self {
        Self {
            matched_prefix: 0,
            leaf_id: root_id,
            path: vec![root_id],
        }
    }
}
