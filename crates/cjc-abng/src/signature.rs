//! Per-node fingerprint — Phase 0.3d-1 (lazy / read-only).
//!
//! `NodeSignature` is the structural-decision policy's compressed view
//! of "who is this node, structurally?" — four 8-byte profile hashes
//! over the per-node prediction / uncertainty / calibration / routing
//! state. In Phase 0.3d-1 each profile is computed lazily by hashing
//! the relevant subsystem's existing canonical bytes and truncating
//! to 8 bytes; "subsystem not installed" is represented as eight zero
//! bytes so signature comparisons across nodes stay meaningful before
//! evidence has flowed.
//!
//! Phase 0.3d-4 will replace the per-call recompute with persistent
//! Welford-smoothed profiles updated by `decide_step`. The wire shape
//! is fixed now (32 bytes, four `[u8; 8]` profiles in declaration
//! order) so the upgrade is a behavioural change, not a format change.
//!
//! # Routing canonical bytes
//!
//! [`AdaptiveChildren`](crate::children::AdaptiveChildren) does not
//! expose its own `canonical_bytes`, so this module defines the bytes
//! used for the routing profile inline:
//!
//! ```text
//!   kind_tag       u8       (the ChildrenKind code)
//!   n_children     u32 BE
//!   for each (key, child_id) sorted by key ascending:
//!     key          u8
//!     child_id     u32 BE
//! ```
//!
//! Sorting by key keeps the bytes determined by the *set* of children,
//! not the order they were inserted — Node48 / Node256 stable
//! iteration is via the existing
//! [`AdaptiveChildren::iter`](crate::children::AdaptiveChildren::iter)
//! helper.

use crate::leaf_head::params_hash;
use crate::node::AdaptiveBeliefNode;

/// Length of each individual profile field, in bytes.
pub const PROFILE_LEN: usize = 8;

/// Total canonical-bytes length of a [`NodeSignature`]. Frozen — the
/// structural-decision policy treats this as fixed.
pub const SIGNATURE_LEN: usize = 4 * PROFILE_LEN;

/// Per-node fingerprint. Recomputed on every read in 0.3d-1; replaced
/// by Welford-smoothed persistent state in 0.3d-4 without changing the
/// wire shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSignature {
    /// First 8 bytes of `sha256(params canonical bytes)`. Zeros when
    /// the leaf head is not installed (params is empty).
    pub prediction: [u8; PROFILE_LEN],
    /// First 8 bytes of `BlrState::state_hash`. Zeros when the BLR
    /// posterior is not installed.
    pub uncertainty: [u8; PROFILE_LEN],
    /// First 8 bytes of `CalibrationBins::state_hash`. Zeros when
    /// calibration bins are not installed.
    pub calibration: [u8; PROFILE_LEN],
    /// First 8 bytes of `sha256(routing canonical bytes)`. Always
    /// present — every node has a children container, even if it's
    /// `AdaptiveChildren::None`.
    pub routing: [u8; PROFILE_LEN],
}

impl NodeSignature {
    /// Compute a fresh signature snapshot from a node's current state.
    pub fn from_node(node: &AdaptiveBeliefNode) -> Self {
        let prediction = if node.params.is_empty() {
            [0u8; PROFILE_LEN]
        } else {
            truncate8(&params_hash(&node.params))
        };
        let uncertainty = node
            .blr
            .as_ref()
            .map(|s| truncate8(&s.state_hash()))
            .unwrap_or([0u8; PROFILE_LEN]);
        let calibration = node
            .calibration
            .as_ref()
            .map(|c| truncate8(&c.state_hash()))
            .unwrap_or([0u8; PROFILE_LEN]);
        let routing = truncate8(&cjc_snap::hash::sha256(&routing_canonical_bytes(node)));
        Self {
            prediction,
            uncertainty,
            calibration,
            routing,
        }
    }

    /// Canonical big-endian byte encoding for hashing / future snapshot.
    /// Layout: prediction ‖ uncertainty ‖ calibration ‖ routing — 32 bytes.
    pub fn canonical_bytes(&self) -> [u8; SIGNATURE_LEN] {
        let mut out = [0u8; SIGNATURE_LEN];
        out[0..8].copy_from_slice(&self.prediction);
        out[8..16].copy_from_slice(&self.uncertainty);
        out[16..24].copy_from_slice(&self.calibration);
        out[24..32].copy_from_slice(&self.routing);
        out
    }
}

fn truncate8(hash: &[u8; 32]) -> [u8; PROFILE_LEN] {
    let mut out = [0u8; PROFILE_LEN];
    out.copy_from_slice(&hash[..PROFILE_LEN]);
    out
}

/// Build the canonical-bytes representation of a node's children for
/// the routing profile. See module docs for the layout.
fn routing_canonical_bytes(node: &AdaptiveBeliefNode) -> Vec<u8> {
    let mut pairs = node.children.iter();
    pairs.sort_by_key(|&(key, _)| key);
    let mut out = Vec::with_capacity(1 + 4 + pairs.len() * 5);
    out.push(node.children.kind() as u8);
    out.extend_from_slice(&(pairs.len() as u32).to_be_bytes());
    for (key, child_id) in pairs {
        out.push(key);
        out.extend_from_slice(&child_id.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_ad::pinn::Activation;

    use crate::graph::AdaptiveBeliefGraph;

    fn fresh_graph() -> AdaptiveBeliefGraph {
        AdaptiveBeliefGraph::new(7)
    }

    #[test]
    fn from_node_no_subsystems_zeros_three_profiles() {
        let g = fresh_graph();
        let s = NodeSignature::from_node(&g.nodes[0]);
        assert_eq!(s.prediction, [0u8; 8]);
        assert_eq!(s.uncertainty, [0u8; 8]);
        assert_eq!(s.calibration, [0u8; 8]);
        // Routing is always defined — empty children still hash to a
        // well-defined value.
        assert_ne!(s.routing, [0u8; 8]);
    }

    #[test]
    fn with_head_prediction_nonzero() {
        let mut g = fresh_graph();
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        let s = NodeSignature::from_node(&g.nodes[0]);
        assert_ne!(s.prediction, [0u8; 8]);
        assert_eq!(s.uncertainty, [0u8; 8]);
        assert_eq!(s.calibration, [0u8; 8]);
    }

    #[test]
    fn with_blr_uncertainty_nonzero() {
        let mut g = fresh_graph();
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        let s = NodeSignature::from_node(&g.nodes[0]);
        assert_ne!(s.uncertainty, [0u8; 8]);
    }

    #[test]
    fn with_calibration_calibration_nonzero() {
        let mut g = fresh_graph();
        g.set_calibration(15).unwrap();
        let s = NodeSignature::from_node(&g.nodes[0]);
        assert_ne!(s.calibration, [0u8; 8]);
    }

    #[test]
    fn routing_changes_with_children() {
        let mut g = fresh_graph();
        let s_before = NodeSignature::from_node(&g.nodes[0]).routing;
        let _child = g.add_node(0, 7).unwrap();
        let s_after = NodeSignature::from_node(&g.nodes[0]).routing;
        assert_ne!(s_before, s_after);
    }

    #[test]
    fn canonical_bytes_size() {
        let s = NodeSignature {
            prediction: [0u8; 8],
            uncertainty: [0u8; 8],
            calibration: [0u8; 8],
            routing: [0u8; 8],
        };
        assert_eq!(s.canonical_bytes().len(), SIGNATURE_LEN);
        assert_eq!(SIGNATURE_LEN, 32);
    }

    #[test]
    fn canonical_bytes_layout() {
        let s = NodeSignature {
            prediction: [1u8; 8],
            uncertainty: [2u8; 8],
            calibration: [3u8; 8],
            routing: [4u8; 8],
        };
        let bytes = s.canonical_bytes();
        assert_eq!(&bytes[0..8], &[1u8; 8]);
        assert_eq!(&bytes[8..16], &[2u8; 8]);
        assert_eq!(&bytes[16..24], &[3u8; 8]);
        assert_eq!(&bytes[24..32], &[4u8; 8]);
    }

    #[test]
    fn determinism_double_run() {
        let mk = || {
            let mut g = AdaptiveBeliefGraph::new(42);
            g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
            g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
            g.set_calibration(15).unwrap();
            g.observe(0, 1.5).unwrap();
            NodeSignature::from_node(&g.nodes[0]).canonical_bytes()
        };
        assert_eq!(mk(), mk());
    }

    #[test]
    fn signature_changes_after_observe_via_routing() {
        // Routing is independent of stats, so an observation alone
        // shouldn't change routing. But adding a child does.
        let mut g = fresh_graph();
        let s0 = NodeSignature::from_node(&g.nodes[0]);
        g.observe(0, 3.14).unwrap();
        let s1 = NodeSignature::from_node(&g.nodes[0]);
        // Routing unchanged by observe.
        assert_eq!(s0.routing, s1.routing);
        // Adding a child changes routing.
        let _c = g.add_node(0, 12).unwrap();
        let s2 = NodeSignature::from_node(&g.nodes[0]);
        assert_ne!(s1.routing, s2.routing);
    }

    #[test]
    fn routing_canonical_bytes_empty_layout() {
        let g = fresh_graph();
        let bytes = routing_canonical_bytes(&g.nodes[0]);
        // kind tag (None = 0) + n_children u32 BE = 0
        assert_eq!(bytes, vec![0x00, 0x00, 0x00, 0x00, 0x00]);
    }
}
