//! Phase 0.8 Item B3 — zstd-compressed snapshot encode tests.
//!
//! Gated behind the root crate's `compression` feature (which forwards
//! to `cjc-abng/compression`). Invoke with:
//!
//! ```text
//! cargo test --features compression --test abng -- serialize_compressed
//! ```
//!
//! Verifies that the compressed wrapper:
//!   1. Round-trips through `replay` to a graph whose re-serialization
//!      is byte-equal to the original uncompressed snapshot.
//!   2. Produces a meaningful size reduction at audit-event-heavy
//!      workloads (the inner v13 layout's repetitive event headers
//!      are zstd's bread and butter).
//!   3. Is detected by `replay` via the `ABNGZ\x01` magic; calling
//!      `replay` on a compressed blob with a `cjc-abng` build that
//!      lacks the `compression` feature returns
//!      `DecodeError::CompressionFeatureDisabled` (cannot test
//!      directly here since the feature is on — see the in-crate
//!      docs for the disabled path).
//!   4. Is deterministic: two compressions of the same graph at the
//!      same level produce byte-identical compressed bytes.

#![cfg(feature = "compression")]

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{
    replay, serialize, serialize_compressed, serialize_into_compressed,
};

/// Phase 0.8 Item B3 — compressed snapshot magic. Mirrors the private
/// `COMPRESSED_MAGIC` const in `cjc-abng::serialize`; the test
/// hardcodes the same 6 bytes so any future drift in either is
/// caught immediately by the `compressed_round_trips_through_replay`
/// magic-prefix check.
const COMPRESSED_MAGIC_LOCAL: &[u8; 6] = b"ABNGZ\x01";

fn build_workload(seed: u64, n_obs: usize) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    for byte in 1u8..5 {
        let _ = g.add_node(0, byte).unwrap();
    }
    for i in 0..n_obs {
        let leaf = ((i % 4) + 1) as u32;
        let v = (i as f64 * 0.0001) - 0.5;
        g.observe(leaf, v).unwrap();
    }
    g
}

#[test]
fn compressed_round_trips_through_replay() {
    let g = build_workload(42, 1_024);
    let uncompressed = serialize(&g);
    let compressed = serialize_compressed(&g, 3);

    // Magic check.
    assert_eq!(
        &compressed[..6],
        COMPRESSED_MAGIC_LOCAL,
        "compressed snapshot must start with ABNGZ\\x01"
    );

    // Replay through the compressed path.
    let g2 = replay(&compressed).expect("replay(compressed)");
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.seed, g2.seed);
    assert_eq!(g.epoch, g2.epoch);
    assert_eq!(g.audit_len(), g2.audit_len());

    // The replayed graph's re-serialization must match the original
    // uncompressed blob byte-for-byte (the inner v13 stream survives
    // the compression round-trip intact).
    let re_serialized = serialize(&g2);
    assert_eq!(
        uncompressed, re_serialized,
        "round-trip via compressed path produced a graph that re-serializes differently",
    );
}

#[test]
fn compressed_smaller_than_uncompressed() {
    // Audit-event-heavy workload — zstd should reduce these
    // significantly because every event has the same 21-byte header
    // shape plus repetitive `previous_hash` / `new_hash` patterns.
    let g = build_workload(7, 2_048);
    let uncompressed = serialize(&g);
    let compressed = serialize_compressed(&g, 3);

    let ratio = uncompressed.len() as f64 / compressed.len() as f64;
    assert!(
        ratio > 1.5,
        "expected zstd ratio > 1.5 on audit-heavy workload, \
         got uncompressed={} bytes, compressed={} bytes, ratio={:.2}",
        uncompressed.len(),
        compressed.len(),
        ratio,
    );
    eprintln!(
        "compressed_smaller_than_uncompressed: \
         uncompressed={} compressed={} ratio={:.2}x",
        uncompressed.len(),
        compressed.len(),
        ratio
    );
}

#[test]
fn compression_is_deterministic_at_fixed_level() {
    // zstd is deterministic in single-thread mode (the Rust binding's
    // default). Two compressions of the same graph at the same level
    // must produce byte-identical compressed bytes.
    let g = build_workload(123, 256);

    let a = serialize_compressed(&g, 3);
    let b = serialize_compressed(&g, 3);
    let c = serialize_compressed(&g, 3);

    assert_eq!(a, b, "compression must be deterministic (1 vs 2)");
    assert_eq!(b, c, "compression must be deterministic (2 vs 3)");
}

#[test]
fn serialize_into_compressed_matches_serialize_compressed() {
    let g = build_workload(0, 128);

    let buffered = serialize_compressed(&g, 3);

    let mut streamed: Vec<u8> = Vec::new();
    serialize_into_compressed(&g, &mut streamed, 3).expect("serialize_into_compressed");

    assert_eq!(
        buffered, streamed,
        "serialize_compressed and serialize_into_compressed must produce identical bytes",
    );
}
