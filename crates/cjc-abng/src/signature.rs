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

use crate::node::AdaptiveBeliefNode;

/// Length of each individual profile field, in bytes.
pub const PROFILE_LEN: usize = 8;

/// Total canonical-bytes length of a [`NodeSignature`]. Frozen — the
/// structural-decision policy treats this as fixed.
pub const SIGNATURE_LEN: usize = 4 * PROFILE_LEN;

/// Phase 0.4 Track B-2.2.1 — Welford-folded summary for one
/// `NodeSignature` profile. Each `decide_step` call adds the current
/// metric reading; the 8-byte signature byte string is
/// `sha256(canonical_bytes)[..8]`. Pre-0.4 the signature was a hash
/// of *current* state — Welford-smoothing makes "stability" mean
/// "the running summary is no longer drifting" rather than "no
/// state has changed since last call".
///
/// Wire layout (24 bytes, BE — same shape as `NodeStats::canonical_bytes`
/// but standalone since profiles aren't fungible with NodeStats):
/// ```text
///   n_seen      u64 BE  (8)
///   mean        f64 BE  (8) — IEEE 754 bit pattern
///   m2          f64 BE  (8) — running sum of squared deviations
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignatureWelford {
    pub n_seen: u64,
    pub mean: f64,
    pub m2: f64,
}

impl SignatureWelford {
    pub const CANONICAL_LEN: usize = 24;

    pub fn new() -> Self {
        Self {
            n_seen: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Apply one observation in Welford's streaming form.
    pub fn observe(&mut self, value: f64) {
        self.n_seen += 1;
        let n = self.n_seen as f64;
        let delta = value - self.mean;
        self.mean += delta / n;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn canonical_bytes(&self) -> [u8; Self::CANONICAL_LEN] {
        let mut out = [0u8; Self::CANONICAL_LEN];
        out[0..8].copy_from_slice(&self.n_seen.to_be_bytes());
        out[8..16].copy_from_slice(&self.mean.to_bits().to_be_bytes());
        out[16..24].copy_from_slice(&self.m2.to_bits().to_be_bytes());
        out
    }

    /// 8-byte signature: `sha256(mean.to_bits() ‖ m2.to_bits())[..8]`.
    ///
    /// Phase 0.4 Track B-2.2.1 — hashes ONLY `(mean, m2)`, NOT
    /// `n_seen`. Rationale: signatures are meant to detect *trajectory
    /// drift*, not *count-of-observations*. Including `n_seen` would
    /// make every Welford observation drift the signature even when
    /// the underlying data hasn't changed (e.g., a stationary stream
    /// where the mean and M2 settle but n_seen keeps ticking),
    /// preventing `signature_stable_calls` from ever accumulating.
    /// `canonical_bytes` (used for serialization) still encodes the
    /// full state so replay reconstructs correctly.
    ///
    /// Returns `[0u8; 8]` when no observations have been folded —
    /// matches the pre-0.4 "subsystem not installed" sentinel.
    pub fn profile_bytes(&self) -> [u8; PROFILE_LEN] {
        if self.n_seen == 0 {
            return [0u8; PROFILE_LEN];
        }
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.mean.to_bits().to_be_bytes());
        buf[8..16].copy_from_slice(&self.m2.to_bits().to_be_bytes());
        let h = cjc_snap::hash::sha256(&buf);
        let mut out = [0u8; PROFILE_LEN];
        out.copy_from_slice(&h[..PROFILE_LEN]);
        out
    }
}

impl Default for SignatureWelford {
    fn default() -> Self {
        Self::new()
    }
}

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
    /// Phase 0.4 Track B-2.2.1 — each profile is hashed from the
    /// node's persistent Welford summary instead of the current
    /// subsystem state. This makes the signature *smooth*: it
    /// changes only when the running summary drifts, not on every
    /// observation. `signature_stable_calls` therefore counts
    /// "consecutive `decide_step` calls without significant
    /// trajectory change", which is the original prompt's intent
    /// (vs the 0.3d-1 lazy "any state change" semantics).
    pub fn from_node(node: &AdaptiveBeliefNode) -> Self {
        Self {
            prediction: node.welford_prediction.profile_bytes(),
            uncertainty: node.welford_uncertainty.profile_bytes(),
            calibration: node.welford_calibration.profile_bytes(),
            routing: node.welford_routing.profile_bytes(),
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

/// Build the canonical-bytes representation of a node's children for
/// the routing profile observation. Phase 0.4 Track B-2.2.1 — feeds
/// into the routing Welford accumulator each `decide_step` call. See
/// module docs for the layout.
pub(crate) fn routing_canonical_bytes(node: &AdaptiveBeliefNode) -> Vec<u8> {
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

/// Phase 0.4 Track B-2.2.1 — scalar metric observed into the routing
/// Welford each `decide_step` call. Hashes the canonical-bytes
/// representation of the children layout into a u64, then casts to
/// f64 (`u64 as f64`) — non-finiteness-safe (avoids reinterpreting
/// arbitrary bits as IEEE 754, which could produce NaN and poison
/// the Welford accumulator). Loses precision for u64 > 2^53 but the
/// Welford signature still tracks "routing layout has drifted" because
/// the cast is deterministic.
pub(crate) fn routing_observation_value(node: &AdaptiveBeliefNode) -> f64 {
    let bytes = routing_canonical_bytes(node);
    let h = cjc_snap::hash::sha256(&bytes);
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&h[..8]);
    u64::from_be_bytes(buf) as f64
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
    fn from_node_no_observations_zero_profiles() {
        // Phase 0.4 Track B-2.2.1 — signatures are now Welford-folded
        // summaries seeded from `decide_step` calls. A node that has
        // never been advanced has every Welford at n_seen=0, so all
        // four profiles are the all-zeros sentinel.
        let g = fresh_graph();
        let s = NodeSignature::from_node(&g.nodes[0]);
        assert_eq!(s.prediction, [0u8; 8]);
        assert_eq!(s.uncertainty, [0u8; 8]);
        assert_eq!(s.calibration, [0u8; 8]);
        assert_eq!(s.routing, [0u8; 8]);
    }

    #[test]
    fn from_node_after_observe_routing_nonzero() {
        // The prediction profile observes `NodeStats.mean` per
        // `decide_step` call. The routing profile observes a hash of
        // the children layout. After at least one
        // advance_signature_stability (or a manual Welford observation),
        // those profiles flip non-zero.
        let mut g = fresh_graph();
        g.nodes[0].welford_routing.observe(1.0);
        g.nodes[0].welford_prediction.observe(2.0);
        let s = NodeSignature::from_node(&g.nodes[0]);
        assert_ne!(s.prediction, [0u8; 8]);
        assert_ne!(s.routing, [0u8; 8]);
        // BLR + calibration still untouched.
        assert_eq!(s.uncertainty, [0u8; 8]);
        assert_eq!(s.calibration, [0u8; 8]);
    }

    #[test]
    fn welford_signature_stable_under_repeated_identical_observations() {
        // Pin the "smoothed" semantics: observing the same value twice
        // keeps mean and M2 unchanged (delta = 0), so the profile
        // signature stays bit-identical even though n_seen advanced.
        // This is what makes `signature_stable_calls` actually
        // accumulate on stationary streams.
        let mut g = fresh_graph();
        g.nodes[0].welford_prediction.observe(3.14);
        let s1 = NodeSignature::from_node(&g.nodes[0]).prediction;
        g.nodes[0].welford_prediction.observe(3.14);
        let s2 = NodeSignature::from_node(&g.nodes[0]).prediction;
        assert_eq!(s1, s2, "stationary stream → stable signature");
    }

    #[test]
    fn welford_signature_drifts_when_distribution_changes() {
        // Conversely, observing a DIFFERENT value drifts mean and M2
        // and therefore the profile signature.
        let mut g = fresh_graph();
        g.nodes[0].welford_prediction.observe(3.14);
        let s1 = NodeSignature::from_node(&g.nodes[0]).prediction;
        g.nodes[0].welford_prediction.observe(2.71);
        let s2 = NodeSignature::from_node(&g.nodes[0]).prediction;
        assert_ne!(s1, s2);
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
        // After identical observation history (and identical Welford
        // observations driven by identical decide_step calls), the
        // signature canonical bytes are bit-identical.
        let mk = || {
            let mut g = AdaptiveBeliefGraph::new(42);
            g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
            g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
            g.set_calibration(15).unwrap();
            g.observe(0, 1.5).unwrap();
            // Drive the Welfords once each so we test the non-trivial
            // case (n_seen ≥ 1 across all 4 profiles).
            g.nodes[0].welford_prediction.observe(0.5);
            g.nodes[0].welford_uncertainty.observe(0.7);
            g.nodes[0].welford_calibration.observe(0.0);
            g.nodes[0].welford_routing.observe(1.0);
            NodeSignature::from_node(&g.nodes[0]).canonical_bytes()
        };
        assert_eq!(mk(), mk());
    }

    #[test]
    fn signature_routing_drifts_with_children_via_welford() {
        // Routing observation hashes the children layout; observing
        // the layout of a 0-child root gives one Welford reading,
        // observing the 1-child layout (after add_node) gives a
        // different one — so subsequent advance_stability_history
        // calls should yield different signatures.
        let mut g = fresh_graph();
        // Manual Welford observation matching pre-add-child state.
        let v0 = routing_observation_value(&g.nodes[0]);
        g.nodes[0].welford_routing.observe(v0);
        let s_before = NodeSignature::from_node(&g.nodes[0]).routing;
        let _c = g.add_node(0, 12).unwrap();
        // Manual Welford observation matching post-add-child state.
        let v1 = routing_observation_value(&g.nodes[0]);
        g.nodes[0].welford_routing.observe(v1);
        let s_after = NodeSignature::from_node(&g.nodes[0]).routing;
        assert_ne!(s_before, s_after);
    }

    #[test]
    fn routing_canonical_bytes_empty_layout() {
        let g = fresh_graph();
        let bytes = routing_canonical_bytes(&g.nodes[0]);
        // kind tag (None = 0) + n_children u32 BE = 0
        assert_eq!(bytes, vec![0x00, 0x00, 0x00, 0x00, 0x00]);
    }
}
