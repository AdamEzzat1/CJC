//! Phase 0.8c v14 Item A3 — Merkle-indexed audit chain.
//!
//! Layers a deterministic binary Merkle tree over the audit log's
//! `new_hash` column (exposed by Phase 0.8's columnar [`AuditLog`]).
//! The root of the tree is a single 32-byte attestation that fixes
//! the entire chain; inclusion proofs let a third party verify any
//! prefix in O(log N) hashes without downloading the whole log.
//!
//! ## Design constraints (from `docs/abng/PHASE_0_8c_V14_HANDOFF.md`)
//!
//! 1. **Zero canary impact.** The Merkle root is computed FROM
//!    `audit.new_hashes()` but never feeds INTO any individual
//!    event's `payload_bytes`. The audit chain witness shape is
//!    unchanged. The Merkle root lives in a snapshot trailer
//!    (v14-only), alongside the audit log, not inside any event
//!    payload.
//!
//! 2. **No new `AuditKind` variants.** A3's storage cost is one
//!    33-byte trailer block per snapshot (tag `0x01` + 32-byte root).
//!
//! 3. **Determinism preserved.** All combiners use the same
//!    `sha256(left ‖ right)` primitive the audit chain itself uses
//!    (via [`cjc_snap::hash::Sha256`]). Output is byte-identical
//!    across platforms.
//!
//! ## Tree layout
//!
//! Bottom-up complete binary tree. Layer 0 is the leaf list (copied
//! from `audit.new_hashes()`). Layer `k` is produced from layer
//! `k-1` by pair-hashing adjacent elements; when a layer has odd
//! length, the last element pairs with **itself** (the standard
//! "duplicate-the-last" convention).
//!
//! For `N` leaves the tree has `ceil(log2(max(N, 1)))` levels above
//! the leaf layer, plus the leaf layer itself. The root is the
//! single element at the topmost layer.
//!
//! Edge cases:
//! * `N == 0` → root = [`crate::genesis_hash()`]. Matches the
//!   chain-head-of-empty-graph convention.
//! * `N == 1` → root = `leaves[0]`. A 1-leaf tree has no pair to
//!   hash; the root is the leaf itself.

use crate::genesis_hash;

/// A deterministic binary Merkle tree built from a slice of 32-byte
/// audit hashes. Layers are stored bottom-up: `layers[0]` is the
/// leaf list, `layers.last()` is a single-element vec holding the
/// root.
///
/// Build cost: `O(N)` hashes + `O(N)` memory. The tree is owned by
/// the caller and not cached on [`crate::graph::AdaptiveBeliefGraph`] —
/// callers that need the tree repeatedly should keep it.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    layers: Vec<Vec<[u8; 32]>>,
}

impl MerkleTree {
    /// Build the Merkle tree from a slice of leaf hashes (typically
    /// `audit.new_hashes()`).
    ///
    /// Determinism: the same leaf slice always produces the same
    /// tree on every platform — `cjc_snap::hash::Sha256` is the
    /// audit chain's own SHA-256 implementation, byte-identical
    /// across MSVC / GNU / macOS.
    pub fn build(leaves: &[[u8; 32]]) -> Self {
        if leaves.is_empty() {
            // Empty chain — the root is the genesis hash, matching
            // `AdaptiveBeliefGraph::chain_head` at construction time.
            return Self {
                layers: vec![vec![genesis_hash()]],
            };
        }
        let mut layers: Vec<Vec<[u8; 32]>> = Vec::with_capacity(8);
        layers.push(leaves.to_vec());
        while layers.last().expect("≥ 1 layer").len() > 1 {
            let prev = layers.last().expect("just pushed");
            let mut next: Vec<[u8; 32]> = Vec::with_capacity(prev.len().div_ceil(2));
            let mut i = 0;
            while i < prev.len() {
                let left = prev[i];
                // Standard "duplicate-the-last" convention: at an
                // odd-length layer, the trailing element pairs with
                // itself to produce the parent. The same rule fires
                // in `proof()` so the verifier reproduces this hash.
                let right = if i + 1 < prev.len() {
                    prev[i + 1]
                } else {
                    prev[i]
                };
                next.push(pair_hash(&left, &right));
                i += 2;
            }
            layers.push(next);
        }
        Self { layers }
    }

    /// The root hash. Always defined — an empty chain's root is
    /// [`crate::genesis_hash()`]; a 1-leaf chain's root is the leaf
    /// itself.
    #[inline]
    pub fn root(&self) -> [u8; 32] {
        *self
            .layers
            .last()
            .expect("MerkleTree always has ≥ 1 layer")
            .first()
            .expect("topmost layer always has 1 element")
    }

    /// Number of leaves the tree was built over. `0` is a valid
    /// value (the empty tree).
    #[inline]
    pub fn n_leaves(&self) -> usize {
        // The first layer holds the leaves directly. For the empty
        // case, layers[0] is `[genesis_hash()]` (1 element), but we
        // want to return 0. Distinguish by depth: an empty tree
        // has exactly 1 layer (the synthetic genesis); a 1-leaf
        // tree also has 1 layer but the leaf is "real."
        //
        // Rather than depend on that ambiguity, store the count
        // alongside the layers — but for now we only have one
        // synthetic case (`len == 0`), so disambiguate via the
        // synthetic root pattern.
        if self.layers.len() == 1 && self.layers[0].len() == 1 && self.layers[0][0] == genesis_hash() {
            // Could be either: a real 1-leaf graph whose first
            // event's `new_hash` happens to equal `genesis_hash()`,
            // OR the synthetic "empty chain" case. The former is
            // cryptographically impossible (would require a
            // SHA-256 collision with a known input), so we treat
            // the synthetic case as canonical for empty.
            0
        } else {
            self.layers[0].len()
        }
    }

    /// Number of internal layers above the leaves (`= log2_ceil(N)`).
    /// Returns 0 for `N <= 1`.
    #[inline]
    pub fn depth(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// Inclusion proof for leaf at index `i`: the `depth()`
    /// sibling hashes needed to recompute the root from the leaf.
    /// Empty for a 1-leaf tree (the leaf IS the root).
    ///
    /// # Panics
    /// Panics if `i >= n_leaves()`. The caller is expected to
    /// validate the index against the audit log they're proving
    /// over.
    pub fn proof(&self, i: usize) -> Vec<[u8; 32]> {
        assert!(
            i < self.n_leaves(),
            "MerkleTree::proof: leaf index {i} out of range (n_leaves = {})",
            self.n_leaves()
        );
        // Walk bottom-up. At each layer, the sibling at position
        // `idx ^ 1` is what the verifier needs. For odd-length
        // layers where `idx ^ 1 == layer.len()`, the sibling-of-self
        // duplicate (matching `build`) is the value the verifier
        // expects to see.
        let mut proof: Vec<[u8; 32]> = Vec::with_capacity(self.depth());
        let mut idx = i;
        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling_idx = idx ^ 1;
            let sibling = if sibling_idx < layer.len() {
                layer[sibling_idx]
            } else {
                // Odd-end duplicate-of-self case.
                layer[idx]
            };
            proof.push(sibling);
            idx >>= 1;
        }
        proof
    }

    /// Verify an inclusion proof without holding the tree. Pure
    /// function: a third party with `(leaf, i, n_leaves, proof,
    /// expected_root)` can call this to attest that `leaf` was at
    /// position `i` in a chain of `n_leaves` events whose Merkle
    /// root is `expected_root`.
    ///
    /// Returns `false` on any inconsistency: index out of range,
    /// wrong proof length, or root mismatch. Returns `true` iff
    /// the recomputed root equals `expected_root`.
    pub fn verify_proof(
        leaf: [u8; 32],
        i: usize,
        n_leaves: usize,
        proof: &[[u8; 32]],
        expected_root: [u8; 32],
    ) -> bool {
        if n_leaves == 0 {
            // Empty chain has no leaves to prove inclusion of.
            return false;
        }
        if i >= n_leaves {
            return false;
        }
        // Expected proof length is exactly the number of layers
        // above the leaves for a tree of `n_leaves` size.
        let expected_depth = depth_for_n_leaves(n_leaves);
        if proof.len() != expected_depth {
            return false;
        }
        let mut current = leaf;
        let mut idx = i;
        for &sibling in proof {
            let (left, right) = if idx & 1 == 0 {
                (current, sibling)
            } else {
                (sibling, current)
            };
            current = pair_hash(&left, &right);
            idx >>= 1;
        }
        current == expected_root
    }
}

/// SHA-256 combiner for an internal node. Matches the chain's own
/// hashing primitive byte-for-byte.
fn pair_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = cjc_snap::hash::Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize()
}

/// Number of internal layers needed for a tree with `n_leaves`
/// leaves under the duplicate-the-last padding rule. `0` for
/// `n_leaves <= 1`.
fn depth_for_n_leaves(n_leaves: usize) -> usize {
    if n_leaves <= 1 {
        return 0;
    }
    let mut depth = 0;
    let mut n = n_leaves;
    while n > 1 {
        n = n.div_ceil(2);
        depth += 1;
    }
    depth
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn empty_chain_root_is_genesis() {
        let tree = MerkleTree::build(&[]);
        assert_eq!(tree.root(), genesis_hash());
        assert_eq!(tree.n_leaves(), 0);
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn single_leaf_root_is_the_leaf() {
        let leaves = vec![h(1)];
        let tree = MerkleTree::build(&leaves);
        assert_eq!(tree.root(), h(1));
        assert_eq!(tree.n_leaves(), 1);
        assert_eq!(tree.depth(), 0);
        // Proof for the lone leaf is empty.
        assert!(tree.proof(0).is_empty());
    }

    #[test]
    fn two_leaves_root_is_pair_hash() {
        let leaves = vec![h(1), h(2)];
        let tree = MerkleTree::build(&leaves);
        let expected = pair_hash(&h(1), &h(2));
        assert_eq!(tree.root(), expected);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn three_leaves_duplicates_last_at_leaf_layer() {
        // N=3 → layer 0 = [h(1), h(2), h(3)], layer 1 pairs as
        // (h(1), h(2)) and (h(3), h(3)).
        let leaves = vec![h(1), h(2), h(3)];
        let tree = MerkleTree::build(&leaves);
        let l1_left = pair_hash(&h(1), &h(2));
        let l1_right = pair_hash(&h(3), &h(3));
        assert_eq!(tree.root(), pair_hash(&l1_left, &l1_right));
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn build_is_deterministic_across_runs() {
        // The build path is referentially transparent — the same
        // leaf list produces byte-identical trees.
        let leaves: Vec<[u8; 32]> = (0u8..16).map(h).collect();
        let a = MerkleTree::build(&leaves);
        let b = MerkleTree::build(&leaves);
        assert_eq!(a.root(), b.root());
        assert_eq!(a.depth(), b.depth());
        assert_eq!(a.layers.len(), b.layers.len());
        for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
            assert_eq!(la, lb);
        }
    }

    #[test]
    fn proof_roundtrip_for_every_leaf_in_power_of_two_tree() {
        let leaves: Vec<[u8; 32]> = (0u8..16).map(h).collect();
        let tree = MerkleTree::build(&leaves);
        let root = tree.root();
        for (i, leaf) in leaves.iter().enumerate() {
            let proof = tree.proof(i);
            assert!(
                MerkleTree::verify_proof(*leaf, i, leaves.len(), &proof, root),
                "proof for leaf {i} should verify against root"
            );
        }
    }

    #[test]
    fn proof_roundtrip_for_every_leaf_in_odd_size_tree() {
        // N=7 — exercises the duplicate-the-last path at multiple
        // levels (leaf layer is odd; the resulting layer-1 has 4
        // elements; layer-2 has 2 elements; layer-3 is the root).
        let leaves: Vec<[u8; 32]> = (0u8..7).map(h).collect();
        let tree = MerkleTree::build(&leaves);
        let root = tree.root();
        for (i, leaf) in leaves.iter().enumerate() {
            let proof = tree.proof(i);
            assert!(
                MerkleTree::verify_proof(*leaf, i, leaves.len(), &proof, root),
                "proof for leaf {i} of N=7 should verify"
            );
        }
    }

    #[test]
    fn proof_rejects_wrong_leaf() {
        let leaves: Vec<[u8; 32]> = (0u8..8).map(h).collect();
        let tree = MerkleTree::build(&leaves);
        let root = tree.root();
        // Tamper leaf 3 to leaf 99 (different value); proof against
        // the original index must fail.
        let proof_for_3 = tree.proof(3);
        assert!(!MerkleTree::verify_proof(h(99), 3, 8, &proof_for_3, root));
    }

    #[test]
    fn proof_rejects_wrong_index() {
        let leaves: Vec<[u8; 32]> = (0u8..8).map(h).collect();
        let tree = MerkleTree::build(&leaves);
        let root = tree.root();
        // Real proof for index 3 used with claimed index 5 must
        // fail (the verifier walks a different path through the
        // tree).
        let proof_for_3 = tree.proof(3);
        assert!(!MerkleTree::verify_proof(h(3), 5, 8, &proof_for_3, root));
    }

    #[test]
    fn proof_rejects_wrong_root() {
        let leaves: Vec<[u8; 32]> = (0u8..8).map(h).collect();
        let tree = MerkleTree::build(&leaves);
        let proof = tree.proof(0);
        // Genesis is definitely not the real root for a real tree
        // with 8 distinct leaves.
        assert!(!MerkleTree::verify_proof(h(0), 0, 8, &proof, genesis_hash()));
    }

    #[test]
    fn proof_rejects_wrong_proof_length() {
        let leaves: Vec<[u8; 32]> = (0u8..8).map(h).collect();
        let tree = MerkleTree::build(&leaves);
        let root = tree.root();
        let proof = tree.proof(0);
        // Truncated proof
        assert!(!MerkleTree::verify_proof(
            h(0),
            0,
            8,
            &proof[..proof.len() - 1],
            root
        ));
        // Extended proof (padded with a spurious sibling)
        let mut extended = proof.clone();
        extended.push(h(0xFF));
        assert!(!MerkleTree::verify_proof(h(0), 0, 8, &extended, root));
    }

    #[test]
    fn proof_rejects_n_leaves_zero() {
        let proof: Vec<[u8; 32]> = vec![];
        assert!(!MerkleTree::verify_proof(h(0), 0, 0, &proof, genesis_hash()));
    }

    #[test]
    fn proof_rejects_index_out_of_range() {
        let proof: Vec<[u8; 32]> = vec![];
        assert!(!MerkleTree::verify_proof(h(0), 5, 3, &proof, h(0)));
    }

    #[test]
    fn depth_for_n_leaves_matches_tree_depth() {
        for n in 0..=32usize {
            let leaves: Vec<[u8; 32]> = (0..n as u8).map(h).collect();
            let tree = MerkleTree::build(&leaves);
            assert_eq!(
                tree.depth(),
                depth_for_n_leaves(n),
                "depth mismatch at n={n}"
            );
        }
    }
}
