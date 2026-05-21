//! Phase 0.8c v14 Item C2 — parallel `verify_chain` via Merkle
//! segments.
//!
//! `Graph::verify_chain_par(n_threads)` is a drop-in replacement for
//! `verify_chain` that splits the audit log into `n_threads` chunks
//! and verifies them concurrently via `std::thread::scope`. The
//! Phase 0.8 architectural lesson — threshold-gate sub-microsecond
//! work — pins the parallel path to chains of >= 10,000 events;
//! smaller chains transparently fall through to the sequential
//! `verify_chain`.
//!
//! These tests pin three things:
//! 1. The parallel result matches the sequential result for both
//!    pristine and tampered chains, across `n_threads` ∈ {2, 4, 8}.
//! 2. The threshold-gate fires below 10,000 events.
//! 3. The cross-chunk boundary check catches tamper that straddles
//!    a chunk boundary.

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};

/// Build a graph with at least `n` audit events. We mix
/// `observe` and `add_node` so the events are heterogeneous
/// (exercises the variant-payload encoding path in
/// `verify_chain_chunk` across multiple `AuditKind` tags).
fn build_chain(n: usize) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(11);
    let mut next_byte: u8 = 0;
    while g.audit.len() < n {
        if g.audit.len() % 25 == 0 && (g.node_count() as u32) < 200 {
            // Occasional `add_node` so the chain has different
            // event kinds (NodeAdded, ChildrenPromoted).
            let parent = (g.audit.len() / 30) as u32 % g.node_count();
            let _ = g.add_node(parent, next_byte);
            next_byte = next_byte.wrapping_add(1);
        } else {
            // Otherwise observe at the root.
            g.observe(0, (g.audit.len() as f64) * 0.25).unwrap();
        }
    }
    g
}

#[test]
fn matches_sequential_on_pristine_chain() {
    // Build a chain just over the threshold so the parallel path
    // actually fires (10,000 events) and parallel + sequential
    // must agree. Also exercise n_threads ∈ {2, 4, 8} to confirm
    // determinism is independent of thread count.
    let g = build_chain(10_010);
    g.verify_chain().expect("sequential pristine chain must verify");
    for &k in &[2usize, 4, 8] {
        g.verify_chain_par(k)
            .unwrap_or_else(|e| panic!("parallel verify failed at n_threads={k}: {e:?}"));
    }
}

#[test]
fn detects_per_event_tamper_in_parallel() {
    // Tamper a single event's `new_hash` after construction; both
    // sequential and parallel must return `ChainBroken` at the
    // tampered seq. Parallel must catch it regardless of which
    // chunk the tampered event lands in (we tamper near the
    // middle of the chain).
    let mut g = build_chain(10_010);
    let target = g.audit.len() / 2;
    {
        let new_hashes_mut = g.audit.new_hashes_mut();
        new_hashes_mut[target][0] ^= 0x01;
    }
    let seq_err = g.verify_chain().expect_err("seq must catch tamper");
    let par_err = g.verify_chain_par(4).expect_err("par must catch tamper");
    assert!(matches!(seq_err, GraphError::ChainBroken { .. }));
    assert!(matches!(par_err, GraphError::ChainBroken { .. }));
}

#[test]
fn detects_cross_chunk_boundary_tamper() {
    // Tamper at exactly a chunk boundary: pick a target event near
    // a likely chunk_size boundary. With 10,010 events / 4 threads
    // chunk_size = 2503. So tamper near event 2503: the boundary
    // check (event 2503's previous_hash must equal event 2502's
    // new_hash) is what's expected to fire.
    let mut g = build_chain(10_010);
    let n_threads = 4;
    let chunk_size = g.audit.len().div_ceil(n_threads);
    let target = chunk_size; // first event of chunk 1
    // Tamper its `previous_hash`. The chain-link check inside
    // `verify_chain_chunk` won't fire on this event (since it's
    // the first in its chunk), but the cross-chunk boundary check
    // in the caller will.
    {
        let prev_hashes_mut = g.audit.previous_hashes_mut();
        prev_hashes_mut[target][0] ^= 0x80;
    }
    let par_err = g.verify_chain_par(n_threads).expect_err("must detect boundary tamper");
    assert!(matches!(par_err, GraphError::ChainBroken { .. }));
}

#[test]
fn threshold_gates_below_10000_events() {
    // Below the threshold (10,000), `verify_chain_par` falls
    // through to the sequential `verify_chain`. The result must
    // be byte-equal — same `Ok(())` on a pristine chain, same
    // `ChainBroken` on a tampered one.
    let g_clean = build_chain(500);
    assert!(g_clean.audit.len() < 10_000);
    g_clean.verify_chain_par(8).expect("threshold falls back to sequential");

    let mut g_tampered = build_chain(500);
    {
        let new_hashes_mut = g_tampered.audit.new_hashes_mut();
        new_hashes_mut[100][0] ^= 0x01;
    }
    let err = g_tampered
        .verify_chain_par(8)
        .expect_err("threshold path still detects tamper");
    assert!(matches!(err, GraphError::ChainBroken { .. }));
}

#[test]
fn n_threads_one_falls_through_to_sequential() {
    // n_threads <= 1 also bypasses the thread::scope path; same
    // result as `verify_chain`.
    let g = build_chain(10_010);
    g.verify_chain_par(1).expect("n_threads=1 == sequential");
    g.verify_chain_par(0).expect("n_threads=0 == sequential");
}

#[test]
fn determinism_across_thread_counts() {
    // For a pristine chain, the outcome is `Ok(())` across all
    // valid thread counts. For a tampered chain, the outcome is
    // `Err(ChainBroken { at_seq })` and the at_seq value is
    // deterministic-ish: the worker that lands on the tampered
    // event reports it. Different chunkings might surface
    // slightly different `at_seq` if multiple corruptions exist;
    // we test a single-point tamper, where every chunking maps to
    // the same tampered event.
    let mut g = build_chain(10_010);
    let tampered_idx = 5_007;
    let tampered_seq = g.audit.get(tampered_idx).unwrap().seq;
    {
        let new_hashes_mut = g.audit.new_hashes_mut();
        new_hashes_mut[tampered_idx][0] ^= 0x01;
    }
    for &k in &[2usize, 4, 8, 16] {
        let err = g.verify_chain_par(k).expect_err("k={k} must catch tamper");
        match err {
            GraphError::ChainBroken { at_seq } => {
                // The tampered new_hash breaks two checks: the
                // per-event integrity of event `tampered_idx`
                // AND the intra-chunk linkage at `tampered_idx +
                // 1` (whose stored previous_hash no longer
                // matches the new_hash at `tampered_idx`).
                // Whichever check fires first wins; both surface
                // the same chain break.
                assert!(
                    at_seq == tampered_seq || at_seq == tampered_seq + 1,
                    "expected at_seq ∈ {{{tampered_seq}, {}}}, got {at_seq}",
                    tampered_seq + 1
                );
            }
            other => panic!("expected ChainBroken, got {other:?}"),
        }
    }
}

#[test]
fn matches_sequential_with_heterogeneous_event_kinds() {
    // Confirm the parallel path handles all the audit-kind tags
    // that show up in a normal chain (Created, BeliefUpdate,
    // NodeAdded, ChildrenPromoted, plus codebook/head/blr/
    // calibration if installed). The `build_chain` helper
    // emits a mix of BeliefUpdate + NodeAdded; this test
    // additionally adds a few TrainStep events at the end to
    // exercise the v14 0x1E tag.
    use cjc_ad::pinn::Activation;
    let mut g = AdaptiveBeliefGraph::new(42);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for byte in 0..4u8 {
        g.add_node(0, byte).unwrap();
    }
    // Drive ~10,500 events: alternate observes and train_steps.
    let mut k = 0u32;
    while g.audit.len() < 10_500 {
        if k % 2 == 0 {
            g.observe(0, (k as f64) * 0.5).unwrap();
        } else {
            let x = ((k as f64) * 0.001) % 1.0;
            g.train_step(&[x], &[1.0, x, x * x, x * x * x], (k as f64) * 0.1)
                .unwrap();
        }
        k += 1;
    }
    g.verify_chain().expect("sequential must pass");
    g.verify_chain_par(4).expect("parallel must match sequential");
    // Spot-check that TrainStep events are actually in the chain.
    let n_train_steps = g
        .audit
        .iter()
        .filter(|e| matches!(e.kind, AuditKind::TrainStep { .. }))
        .count();
    assert!(n_train_steps > 100);
}
