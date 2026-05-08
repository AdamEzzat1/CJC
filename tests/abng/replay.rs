//! Replay-equality and snapshot determinism tests.

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize, DecodeError};

fn build(seed: u64, values: &[f64]) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    for &v in values {
        g.observe(0, v).unwrap();
    }
    g
}

#[test]
fn round_trip_preserves_chain_head() {
    let g = build(42, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.nodes[0].stats.canonical_bytes(), g2.nodes[0].stats.canonical_bytes());
}

#[test]
fn round_trip_byte_identical_blob() {
    let g = build(123, &[0.5, 1.5, 2.5, 3.5]);
    let blob1 = serialize(&g);
    let g2 = replay(&blob1).unwrap();
    let blob2 = serialize(&g2);
    assert_eq!(blob1, blob2);
}

#[test]
fn empty_graph_round_trips() {
    let g = AdaptiveBeliefGraph::new(7);
    let blob = serialize(&g);
    let g2 = replay(&blob).unwrap();
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.audit_len(), g2.audit_len()); // 1 — the Created event
}

#[test]
fn double_run_blob_byte_identical() {
    let g1 = build(7, &(0..100).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    let g2 = build(7, &(0..100).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    assert_eq!(serialize(&g1), serialize(&g2));
}

#[test]
fn bad_magic_rejected() {
    let g = build(0, &[1.0]);
    let mut blob = serialize(&g);
    blob[0] = b'X';
    assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
}

#[test]
fn truncated_blob_rejected() {
    let g = build(0, &[1.0, 2.0]);
    let blob = serialize(&g);
    let truncated = &blob[..blob.len() / 2];
    assert!(matches!(replay(truncated), Err(DecodeError::UnexpectedEof)));
}

#[test]
fn payload_tampering_caught() {
    let g = build(0, &[1.0, 2.0, 3.0]);
    let mut blob = serialize(&g);
    // v10 layout, no optional sections, single node, no decide_step yet:
    //   Header (v5 base): 5 + 8 + 8 + 32 + 1 + 1 + 1 + 1 + 1 = 58
    //   Phase 0.3d-3 additions: 1 (policy_present=0) + 48 (action_counts u64×6) = 49
    //   n_nodes u32: 4
    //   = 111 header bytes
    //
    //   Per node (v5 base): 4 (parent) + 1 (kind) + 24 (canonical) + 8 (version)
    //           + 32 (chain head) + 4 (n_params=0)
    //           + 1 (blr_present) + 1 (density_present)
    //           + 1 (calibration_present) + 1 (drift_present) = 77
    //   Phase 0.3d-2: + 1 (ee_present=0) = 78
    //   Phase 0.3d-3: + 2 (is_frozen + is_active) = 80
    //   Phase 0.3d-4: + 1 (last_signature_present=0)
    //                 + 8 (signature_stable_calls u64) = 89
    //   Phase 0.4 Track B-2.2.2: + 6 × 8 (ece + sigma history)
    //                           + 1 (ece_fill_count) + 1 (sigma_fill_count)
    //                           = 50 → 139 cumulative
    //   Phase 0.4 Track B-2.2.1: + 4 × 24 (Welford accumulators) = 96 → 235
    //
    //   n_events u64: 8
    //   First event: 4 (payload_len), then payload
    let event_start = 111 + 235 + 8 + 4;
    blob[event_start + 30] ^= 0xFF;
    let err = replay(&blob);
    assert!(matches!(
        err,
        Err(DecodeError::ChainMismatch { .. })
            | Err(DecodeError::StatsMismatch { .. })
            | Err(DecodeError::ChildrenMismatch { .. })
            | Err(DecodeError::UnknownKindTag(_))
    ));
}

#[test]
fn replay_then_continue_keeps_chain_consistent() {
    let g = build(0, &[1.0, 2.0]);
    let blob = serialize(&g);
    let mut g2 = replay(&blob).unwrap();
    g2.observe(0, 3.0).unwrap();
    assert!(g2.verify_chain().is_ok());
}

#[test]
fn empty_graph_rejected() {
    // v8 layout: n_nodes lives after the v5 base + 0.3d-3 additions.
    //   v5 base: 5 magic + 8 seed + 8 epoch + 32 final_hash
    //          + 1 codebook + 1 head + 1 blr_prior
    //          + 1 density_enabled + 1 calibration_present = 58
    //   0.3d-3: + 1 policy_present + 48 action_counts (u64 × 6) = 49
    //   → n_nodes at offset 107..111 (unchanged in v8 — header layout
    //     is the same; v8 only added per-node fields)
    let g = build(0, &[]);
    let mut blob = serialize(&g);
    blob[107..111].copy_from_slice(&0u32.to_be_bytes());
    let err = replay(&blob).unwrap_err();
    assert!(matches!(err, DecodeError::EmptyGraph));
}
