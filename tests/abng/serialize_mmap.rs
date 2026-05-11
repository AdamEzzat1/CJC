//! Phase 0.8 Item B1 — memory-mapped snapshot replay tests.
//!
//! Verifies that `replay_mmap` / `replay_mmap_with_outcome` produce
//! byte-identical output to the slice-based `replay` / `replay_with_outcome`
//! entry points, and that the new `DecodeError::Io` variant fires on
//! filesystem failures.

use std::io::Write;

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{
    replay, replay_mmap, replay_mmap_with_outcome, replay_with_outcome, serialize, DecodeError,
    ReplayOptions,
};

/// Build a non-trivial graph: 4 leaves, codebook frozen, a bulk of
/// observations spread across leaves so the audit log has enough
/// `BeliefUpdate` events to be worth memory-mapping.
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
fn replay_mmap_matches_replay_bytewise() {
    let g = build_workload(42, 1_024);
    let blob = serialize(&g);

    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(&blob).expect("write snapshot");
    tmp.flush().expect("flush snapshot");

    let g_slice = replay(&blob).expect("replay(&[u8])");
    let g_mmap = replay_mmap(tmp.path()).expect("replay_mmap(path)");

    // The decoder consumes the same byte stream; the rebuilt graphs
    // must re-serialize to the same bytes.
    assert_eq!(
        serialize(&g_slice),
        serialize(&g_mmap),
        "mmap replay produced a graph that re-serializes differently",
    );
    // Spot-check user-visible state.
    assert_eq!(g_slice.chain_head, g_mmap.chain_head);
    assert_eq!(g_slice.seed, g_mmap.seed);
    assert_eq!(g_slice.epoch, g_mmap.epoch);
    assert_eq!(g_slice.audit_len(), g_mmap.audit_len());
    for (a, b) in g_slice.nodes.iter().zip(g_mmap.nodes.iter()) {
        assert_eq!(a.stats.canonical_bytes(), b.stats.canonical_bytes());
        assert_eq!(a.stats_chain_head, b.stats_chain_head);
    }
}

#[test]
fn replay_mmap_smart_outcome_parity() {
    // Compact the audit log so smart-replay has StatsSnapshot
    // checkpoints to fast-forward over.
    let mut g = build_workload(123, 2_048);
    let _ = g.compact_log(g.audit.len() as u64);
    let blob = serialize(&g);

    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(&blob).expect("write snapshot");
    tmp.flush().expect("flush snapshot");

    let opts = ReplayOptions {
        smart_replay: true,
    };
    let slice_outcome = replay_with_outcome(&blob, opts).expect("slice smart replay");
    let mmap_outcome =
        replay_mmap_with_outcome(tmp.path(), opts).expect("mmap smart replay");

    assert_eq!(
        slice_outcome.fast_forwarded_events,
        mmap_outcome.fast_forwarded_events,
        "smart-replay fast-forward count differs between slice and mmap paths",
    );
    assert_eq!(
        serialize(&slice_outcome.graph),
        serialize(&mmap_outcome.graph),
        "smart-replay graph differs between slice and mmap paths",
    );
}

#[test]
fn replay_mmap_io_error_on_missing_file() {
    // A path that cannot exist (NUL byte is invalid on every supported
    // platform; if some OS ever accepts it, append a UUID-like suffix).
    let mut bogus = std::env::temp_dir();
    bogus.push("cjc_abng_mmap_b1_does_not_exist_0a1b2c3d4e5f.snap");
    // Be paranoid: make sure it really doesn't exist before we assert
    // on the error kind.
    let _ = std::fs::remove_file(&bogus);

    let err = replay_mmap(&bogus).expect_err("missing file must error");
    match err {
        DecodeError::Io { kind, message } => {
            assert_eq!(kind, std::io::ErrorKind::NotFound, "wrong io::ErrorKind");
            assert!(
                message.contains("open ") && message.contains(bogus.display().to_string().as_str()),
                "Io message should mention the failed op + path; got {message:?}"
            );
        }
        other => panic!("expected DecodeError::Io, got {other:?}"),
    }
}
