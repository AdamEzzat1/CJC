//! Exhaustive tamper sweep: every single-bit flip (and every full-byte
//! inversion) of a serialized graph must make `replay` / `smart_replay`
//! return, never panic.
//!
//! The Bolero targets `fuzz_abng_tamper_no_panic` and
//! `fuzz_abng_smart_replay_tamper_no_panic` (tests/bolero_fuzz) sample this
//! space at random. They failed in CI's debug builds and passed in release:
//! a tampered tensor shape overflowed an unchecked product in
//! `Tensor::from_vec` (panic with overflow checks on, silent wrap without).
//! This sweep is deterministic and runs in whichever profile the tests use,
//! so run it in a debug build (plain `cargo test`) to catch overflow panics.

use std::panic;

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize, smart_replay};

/// Returns the (position, mask) pairs for which decoding panicked.
fn sweep(blob: &[u8], smart: bool) -> Vec<(usize, u8)> {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {})); // keep expected-failure noise out of the log
    let mut panics = Vec::new();
    for pos in 0..blob.len() {
        for mask in (0..8).map(|b| 1u8 << b).chain([0xFF]) {
            let mut tampered = blob.to_vec();
            tampered[pos] ^= mask;
            let r = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let _ = replay(&tampered);
                if smart {
                    let _ = smart_replay(&tampered);
                }
            }));
            if r.is_err() {
                panics.push((pos, mask));
            }
        }
    }
    panic::set_hook(prev);
    panics
}

#[test]
fn replay_never_panics_on_any_single_byte_tamper() {
    // Same graph as `fuzz_abng_tamper_no_panic`.
    let mut g = AdaptiveBeliefGraph::new(0);
    let _ = g.set_decision_policy(&[0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0]);
    let _ = g.observe(0, 1.0);
    let _ = g.force_grow(0, 7);
    let _ = g.decide_step();
    let blob = serialize(&g);
    assert!(!blob.is_empty());
    let panics = sweep(&blob, false);
    assert!(panics.is_empty(), "replay panicked on {} tampered blobs, first: {:?}", panics.len(), &panics[..panics.len().min(5)]);
}

#[test]
fn smart_replay_never_panics_on_any_single_byte_tamper() {
    // Same graph as `fuzz_abng_smart_replay_tamper_no_panic`.
    let mut g = AdaptiveBeliefGraph::new(0);
    let _ = g.observe(0, 1.0);
    let _ = g.observe(0, 2.0);
    let _ = g.compact_log(g.audit.len() as u64);
    let blob = serialize(&g);
    assert!(!blob.is_empty());
    let panics = sweep(&blob, true);
    assert!(panics.is_empty(), "replay / smart_replay panicked on {} tampered blobs, first: {:?}", panics.len(), &panics[..panics.len().min(5)]);
}

#[test]
fn untampered_blobs_still_replay() {
    let mut g = AdaptiveBeliefGraph::new(0);
    let _ = g.observe(0, 1.0);
    let _ = g.observe(0, 2.0);
    let blob = serialize(&g);
    assert!(replay(&blob).is_ok());
}
