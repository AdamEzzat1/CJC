//! Phase 0.4 Track A — Prediction snapshot.
//!
//! A self-contained, deterministic byte format that captures
//! everything `cjcl abng explain` needs to reconstruct lineage from a
//! single prediction:
//!
//! * The model's `chain_head` at predict time (fingerprints the entire
//!   training history.)
//! * The codebook + leaf-head + BLR-state hashes (so a stale or
//!   tampered model produces a mismatch on explain.)
//! * The predicting node id + its BLR `n_seen` (evidence count).
//! * The feature vector `phi` that was predicted on.
//! * The prediction tuple `(mean, epistemic_leverage, aleatoric_var)`.
//! * The captured `expected_epistemic` (or NaN-sentinel if uncaptured).
//!
//! Format (all integers and f64-bits big-endian):
//!
//! ```text
//! magic              "ABNG-PRED\x01"   (10 bytes)
//! model_chain_head   [u8; 32]
//! node_id            u32
//! codebook_hash      [u8; 32]          (zero if no codebook)
//! leaf_head_hash     [u8; 32]          (zero if no head)
//! blr_state_hash     [u8; 32]
//! blr_n_seen         u64
//! phi_dim            u32
//! phi                f64 × phi_dim
//! mean               f64
//! epistemic_leverage f64
//! aleatoric_var      f64
//! expected_epistemic f64 bits          (NaN-sentinel if uncaptured)
//! ```
//!
//! The "PRED" magic distinguishes prediction snapshots from model
//! snapshots (which use `b"ABNG\x0A"`); a v1 magic byte allows future
//! format changes if the prediction record needs additional lineage
//! fields.
//!
//! All packing is deterministic; two calls with bit-identical inputs
//! produce bit-identical bytes across runs and platforms.

use crate::blr::BlrError;
use crate::graph::{AdaptiveBeliefGraph, GraphError};
use crate::leaf_head::params_hash;
use crate::node::NodeId;

/// Magic bytes for prediction snapshots. Distinct from the model
/// magic (`b"ABNG\x0A"`).
pub const PRED_MAGIC: &[u8; 10] = b"ABNG-PRED\x01";

/// In-memory representation of a prediction snapshot. Produced by
/// [`pack`] and reconstructed by [`unpack`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionSnap {
    pub model_chain_head: [u8; 32],
    pub node_id: NodeId,
    pub codebook_hash: [u8; 32],
    pub leaf_head_hash: [u8; 32],
    pub blr_state_hash: [u8; 32],
    pub blr_n_seen: u64,
    pub phi: Vec<f64>,
    pub mean: f64,
    pub epistemic_leverage: f64,
    pub aleatoric_var: f64,
    /// `f64::NAN` indicates the predicting node had no captured
    /// reference at predict time. Otherwise this is the captured
    /// leverage reference used by the calibrated OOD ratio.
    pub expected_epistemic: f64,
}

/// Errors returned by [`unpack`]. Mirrors the discipline of the model
/// snapshot's `DecodeError` — every malformed input maps to a
/// specific variant rather than a panic.
#[derive(Debug, PartialEq)]
pub enum PredictionSnapError {
    /// Input shorter than the fixed-size header.
    Truncated,
    /// First 10 bytes don't match `PRED_MAGIC`.
    BadMagic { got: [u8; 10] },
    /// `phi_dim * 8 + ...` would overflow remaining bytes.
    ShortBody {
        expected: usize,
        got: usize,
    },
    /// `phi_dim` would force a `Vec::with_capacity` larger than the
    /// remaining cursor — likely an adversarial blob.
    SuspiciousPhiDim { phi_dim: u32 },
}

/// Pack a predict-with-lineage call into a self-contained byte blob.
/// Runs `g.blr_predict(node_id, phi)` internally and bundles the
/// result with the model's lineage hashes.
pub fn pack(
    g: &AdaptiveBeliefGraph,
    node_id: NodeId,
    phi: &[f64],
) -> Result<Vec<u8>, GraphError> {
    let n_nodes = g.node_count();
    if node_id >= n_nodes {
        return Err(GraphError::NodeOutOfRange { node_id, n_nodes });
    }
    let blr = g.nodes[node_id as usize]
        .blr
        .as_ref()
        .ok_or(BlrError::NoBlrPrior)?;
    let (mean, lev, ale) = blr.predict(phi)?;

    let codebook_hash = match &g.codebook {
        Some(cb) => cb.frozen_hash,
        None => [0u8; 32],
    };
    let leaf_head_hash = match &g.head {
        Some(h) => h.config_hash,
        None => [0u8; 32],
    };
    let blr_state_hash = blr.state_hash();
    let blr_n_seen = blr.n_seen;
    let expected_epistemic = g.nodes[node_id as usize]
        .expected_epistemic
        .unwrap_or(f64::NAN);

    // Capacity: 10 + 32 + 4 + 32 + 32 + 32 + 8 + 4 + 8·d + 8·4
    let mut out = Vec::with_capacity(186 + 8 * phi.len());
    out.extend_from_slice(PRED_MAGIC);
    out.extend_from_slice(&g.chain_head);
    out.extend_from_slice(&node_id.to_be_bytes());
    out.extend_from_slice(&codebook_hash);
    out.extend_from_slice(&leaf_head_hash);
    out.extend_from_slice(&blr_state_hash);
    out.extend_from_slice(&blr_n_seen.to_be_bytes());
    out.extend_from_slice(&(phi.len() as u32).to_be_bytes());
    for &x in phi {
        out.extend_from_slice(&x.to_bits().to_be_bytes());
    }
    out.extend_from_slice(&mean.to_bits().to_be_bytes());
    out.extend_from_slice(&lev.to_bits().to_be_bytes());
    out.extend_from_slice(&ale.to_bits().to_be_bytes());
    out.extend_from_slice(&expected_epistemic.to_bits().to_be_bytes());
    Ok(out)
}

/// Unpack a prediction snapshot, validating magic + lengths.
pub fn unpack(bytes: &[u8]) -> Result<PredictionSnap, PredictionSnapError> {
    // Header (no phi yet): 10 + 32 + 4 + 32 + 32 + 32 + 8 + 4 = 154
    const HEADER_LEN: usize = 154;
    if bytes.len() < HEADER_LEN {
        return Err(PredictionSnapError::Truncated);
    }
    let mut got_magic = [0u8; 10];
    got_magic.copy_from_slice(&bytes[0..10]);
    if &got_magic != PRED_MAGIC {
        return Err(PredictionSnapError::BadMagic { got: got_magic });
    }
    let mut p = 10;
    let mut chain_head = [0u8; 32];
    chain_head.copy_from_slice(&bytes[p..p + 32]);
    p += 32;
    let node_id = u32::from_be_bytes(bytes[p..p + 4].try_into().unwrap());
    p += 4;
    let mut codebook_hash = [0u8; 32];
    codebook_hash.copy_from_slice(&bytes[p..p + 32]);
    p += 32;
    let mut leaf_head_hash = [0u8; 32];
    leaf_head_hash.copy_from_slice(&bytes[p..p + 32]);
    p += 32;
    let mut blr_state_hash = [0u8; 32];
    blr_state_hash.copy_from_slice(&bytes[p..p + 32]);
    p += 32;
    let blr_n_seen = u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap());
    p += 8;
    let phi_dim = u32::from_be_bytes(bytes[p..p + 4].try_into().unwrap());
    p += 4;

    // Defensive bounds checks (mirrors the 0.3d-5 model-snapshot
    // decoder hardening contract).
    //
    // SuspiciousPhiDim fires when phi_dim is *value*-adversarial — a
    // gigantic dim that would force a > 8 MB allocation. The bound
    // 1_000_000 is well above any reasonable BLR feature dim and
    // protects against `phi_dim = u32::MAX`-style attacks regardless
    // of how many bytes follow.
    if phi_dim as usize > 1_000_000 {
        return Err(PredictionSnapError::SuspiciousPhiDim { phi_dim });
    }
    // ShortBody fires when the total expected length (header + phi
    // body + trailing 4 × 8B f64s) exceeds bytes.len(). This is the
    // common-case truncation error.
    let phi_bytes = (phi_dim as usize).saturating_mul(8);
    let expected_total = HEADER_LEN.saturating_add(phi_bytes).saturating_add(32);
    if bytes.len() < expected_total {
        return Err(PredictionSnapError::ShortBody {
            expected: expected_total,
            got: bytes.len(),
        });
    }
    let mut phi = Vec::with_capacity(phi_dim as usize);
    for _ in 0..phi_dim {
        let bits = u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap());
        phi.push(f64::from_bits(bits));
        p += 8;
    }
    let mean =
        f64::from_bits(u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap()));
    p += 8;
    let epistemic_leverage =
        f64::from_bits(u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap()));
    p += 8;
    let aleatoric_var =
        f64::from_bits(u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap()));
    p += 8;
    let expected_epistemic =
        f64::from_bits(u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap()));

    Ok(PredictionSnap {
        model_chain_head: chain_head,
        node_id,
        codebook_hash,
        leaf_head_hash,
        blr_state_hash,
        blr_n_seen,
        phi,
        mean,
        epistemic_leverage,
        aleatoric_var,
        expected_epistemic,
    })
}

// `params_hash` is re-exported here so callers don't need an extra
// import path when verifying lineage — see `cjcl abng explain`.
pub use crate::leaf_head::params_hash as leaf_params_hash;
// (Above re-export silences the unused-import warning when params_hash
// isn't used directly in this module; it's intentional public surface.)
#[allow(dead_code)]
fn _ensure_params_hash_used() {
    // No-op: pulls `params_hash` into scope so the re-export survives
    // tree-shaking if a future refactor inlines the explain CLI.
    let _ = params_hash;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdaptiveBeliefGraph;
    use cjc_ad::pinn::Activation;

    fn build_graph() -> AdaptiveBeliefGraph {
        let mut g = AdaptiveBeliefGraph::new(7);
        g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
        g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        g
    }

    #[test]
    fn pack_unpack_round_trip() {
        let g = build_graph();
        let phi = vec![1.0, 0.0];
        let bytes = pack(&g, 0, &phi).unwrap();
        let snap = unpack(&bytes).unwrap();
        assert_eq!(snap.node_id, 0);
        assert_eq!(snap.phi, phi);
        assert_eq!(snap.model_chain_head, g.chain_head);
        assert_eq!(snap.blr_n_seen, g.nodes[0].blr.as_ref().unwrap().n_seen);
        // Predict tuple should match a direct call.
        let (m, l, a) = g.blr_predict(0, &phi).unwrap();
        assert_eq!(snap.mean.to_bits(), m.to_bits());
        assert_eq!(snap.epistemic_leverage.to_bits(), l.to_bits());
        assert_eq!(snap.aleatoric_var.to_bits(), a.to_bits());
    }

    #[test]
    fn pack_is_deterministic() {
        let g = build_graph();
        let b1 = pack(&g, 0, &[1.0, 0.0]).unwrap();
        let b2 = pack(&g, 0, &[1.0, 0.0]).unwrap();
        assert_eq!(b1, b2);
    }

    #[test]
    fn pack_node_out_of_range_errs() {
        let g = build_graph();
        assert!(matches!(
            pack(&g, 99, &[1.0, 0.0]),
            Err(GraphError::NodeOutOfRange { node_id: 99, .. })
        ));
    }

    #[test]
    fn pack_no_blr_errs() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_leaf_head(2, vec![1], 1, Activation::Tanh).unwrap();
        // No set_blr_prior.
        assert!(matches!(
            pack(&g, 0, &[1.0, 0.0]),
            Err(GraphError::Blr(BlrError::NoBlrPrior))
        ));
    }

    #[test]
    fn unpack_bad_magic_errs() {
        let bytes = vec![0u8; 200];
        match unpack(&bytes) {
            Err(PredictionSnapError::BadMagic { .. }) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn unpack_truncated_errs() {
        assert_eq!(
            unpack(&[0u8; 10]).unwrap_err(),
            PredictionSnapError::Truncated
        );
    }

    #[test]
    fn unpack_short_body_errs() {
        // Header complete + magic correct, but no phi/footer bytes.
        let g = build_graph();
        let mut bytes = pack(&g, 0, &[1.0, 0.0]).unwrap();
        // Truncate after the header.
        bytes.truncate(154);
        match unpack(&bytes).unwrap_err() {
            PredictionSnapError::ShortBody { .. } => {}
            other => panic!("expected ShortBody, got {other:?}"),
        }
    }

    #[test]
    fn unpack_suspicious_phi_dim_errs() {
        let g = build_graph();
        let mut bytes = pack(&g, 0, &[1.0, 0.0]).unwrap();
        // Overwrite phi_dim u32 BE at offset 150 (after the 8-byte
        // blr_n_seen at offsets 142..150) with a huge value.
        // HEADER_LEN = 154, phi_dim is at offset 150..154.
        bytes[150..154].copy_from_slice(&(u32::MAX).to_be_bytes());
        match unpack(&bytes).unwrap_err() {
            PredictionSnapError::SuspiciousPhiDim { .. } => {}
            other => panic!("expected SuspiciousPhiDim, got {other:?}"),
        }
    }

    #[test]
    fn pack_with_no_codebook_uses_zero_hash() {
        // Graph without set_codebook — codebook_hash field is zeros.
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_leaf_head(2, vec![2], 1, Activation::Tanh).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g.blr_update(0, &[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        let bytes = pack(&g, 0, &[1.0, 0.0]).unwrap();
        let snap = unpack(&bytes).unwrap();
        assert_eq!(snap.codebook_hash, [0u8; 32]);
    }
}
