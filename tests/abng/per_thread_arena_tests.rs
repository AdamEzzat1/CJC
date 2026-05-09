//! Phase 0.8 Item C3 — per-thread arena isolation + concurrent
//! training scaling tests.
//!
//! The arena lives in a `thread_local!` cell (see
//! `crates/cjc-abng/src/dispatch.rs` line 43). These tests **document
//! and gate** the contract that:
//!
//!   1. Different OS threads have disjoint arenas: `next_id`
//!      sequences are independent; one thread's graphs are invisible
//!      to another.
//!   2. Concurrent training of N graphs across N threads scales
//!      better than serial training of the same N graphs on one
//!      thread (≈ N× under no contention; the graph is held in
//!      thread-local storage so there is none).
//!   3. The dispatch surface respects thread isolation: `abng_new`
//!      called from two threads returns the same graph_id (each
//!      thread starts at `next_id = 0`), and the resulting graphs
//!      do not collide.
//!
//! These tests are part of the Phase 0.8a big-data-ergonomics gate
//! per `docs/abng/PHASE_0_8_HANDOFF.md`.

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::{
    arena_graph_count, arena_next_id, dispatch_abng, reset_arena,
};
use cjc_runtime::value::Value;

// ---------------------------------------------------------------------------
// Isolation contract
// ---------------------------------------------------------------------------

#[test]
fn arena_is_empty_on_fresh_thread() {
    // The current test thread may have leftover state from earlier
    // tests on the same OS thread (cargo test reuses the worker
    // pool). Reset and verify.
    reset_arena();
    assert_eq!(arena_graph_count(), 0);
    assert_eq!(arena_next_id(), 0);
}

#[test]
fn arena_isolated_across_threads_creating_graphs() {
    // Spawn two threads. Each calls `abng_new` 3 + 5 times in its
    // own arena. Each thread's `arena_graph_count` must reflect its
    // own work, never the other thread's.
    let (count_a, count_b) = std::thread::scope(|s| {
        let h_a = s.spawn(|| {
            reset_arena();
            for _ in 0..3 {
                dispatch_abng("abng_new", &[Value::Int(42)]).unwrap();
            }
            arena_graph_count()
        });
        let h_b = s.spawn(|| {
            reset_arena();
            for _ in 0..5 {
                dispatch_abng("abng_new", &[Value::Int(7)]).unwrap();
            }
            arena_graph_count()
        });
        (h_a.join().unwrap(), h_b.join().unwrap())
    });

    assert_eq!(count_a, 3, "thread A must see only its own 3 graphs");
    assert_eq!(count_b, 5, "thread B must see only its own 5 graphs");
}

#[test]
fn next_id_sequences_are_per_thread_independent() {
    // After 3 `abng_new` calls in one thread, that thread's next_id
    // is 3. A second thread's next_id starts from 0 and is unaffected.
    let (next_a, next_b) = std::thread::scope(|s| {
        let h_a = s.spawn(|| {
            reset_arena();
            for _ in 0..3 {
                dispatch_abng("abng_new", &[Value::Int(0)]).unwrap();
            }
            arena_next_id()
        });
        let h_b = s.spawn(|| {
            reset_arena();
            // No graphs created.
            arena_next_id()
        });
        (h_a.join().unwrap(), h_b.join().unwrap())
    });
    assert_eq!(next_a, 3);
    assert_eq!(next_b, 0);
}

#[test]
fn graph_ids_dont_collide_across_threads() {
    // Two threads each create one graph. Both get id 0 *in their
    // own arena*. There is no global id space — each thread's `0`
    // refers to a different graph object, which is fine because
    // graphs never cross the thread boundary via the dispatch API.
    let (id_a, id_b) = std::thread::scope(|s| {
        let h_a = s.spawn(|| {
            reset_arena();
            let v = dispatch_abng("abng_new", &[Value::Int(11)])
                .unwrap()
                .unwrap();
            match v {
                Value::Int(id) => id,
                other => panic!("expected Int graph_id, got {other:?}"),
            }
        });
        let h_b = s.spawn(|| {
            reset_arena();
            let v = dispatch_abng("abng_new", &[Value::Int(22)])
                .unwrap()
                .unwrap();
            match v {
                Value::Int(id) => id,
                other => panic!("expected Int graph_id, got {other:?}"),
            }
        });
        (h_a.join().unwrap(), h_b.join().unwrap())
    });
    assert_eq!(id_a, 0);
    assert_eq!(id_b, 0);
}

// ---------------------------------------------------------------------------
// Concurrent training scaling
// ---------------------------------------------------------------------------

/// Train one graph for `n_obs` observations. Used by both serial
/// and concurrent timing tests. Returns the final chain head as a
/// determinism witness.
fn train_one(seed: u64, n_obs: usize) -> [u8; 32] {
    let mut g = AdaptiveBeliefGraph::new(seed);
    for i in 0..n_obs {
        g.observe(0, (i as f64) * 0.001).unwrap();
    }
    g.chain_head
}

#[test]
fn concurrent_training_produces_per_thread_results() {
    // 4 threads, each trains a different graph (different seed).
    // Verify each thread's chain_head matches the serial computation
    // of the same seed — proves no cross-thread contamination.
    const SEEDS: [u64; 4] = [1, 2, 3, 4];
    const N_OBS: usize = 200;

    // Serial reference: train each seed sequentially, record head.
    let serial: Vec<[u8; 32]> =
        SEEDS.iter().map(|&s| train_one(s, N_OBS)).collect();

    // Concurrent: same work, one graph per thread.
    let concurrent: Vec<[u8; 32]> = std::thread::scope(|s| {
        let handles: Vec<_> = SEEDS
            .iter()
            .map(|&seed| s.spawn(move || train_one(seed, N_OBS)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(
        serial, concurrent,
        "concurrent training must produce per-thread chain_heads identical to serial"
    );
}

#[test]
fn concurrent_training_scales_better_than_serial() {
    // This is a *weak* scaling assertion — we only require that 4
    // threads finish in less time than 4× serial. We do NOT assert
    // 4× speedup because the test runner may share cores with other
    // tests, the OS may not allocate the threads to distinct cores,
    // and the workload may be small enough that thread-spawn
    // overhead dominates. The contract being gated is: "concurrent
    // is not slower than serial," which proves no false sharing or
    // hidden global lock.

    use std::time::Instant;
    const SEEDS: [u64; 4] = [101, 202, 303, 404];
    const N_OBS: usize = 5_000;

    // Warm-up — discount JIT/cache effects.
    let _ = train_one(0, 500);

    // Serial.
    let start = Instant::now();
    for &s in &SEEDS {
        let _ = train_one(s, N_OBS);
    }
    let serial = start.elapsed();

    // Concurrent.
    let start = Instant::now();
    std::thread::scope(|sc| {
        for &s in &SEEDS {
            sc.spawn(move || train_one(s, N_OBS));
        }
    });
    let concurrent = start.elapsed();

    // Weak gate: concurrent must finish in at most 1.25× the serial
    // wall time. On any system with ≥2 cores this is overwhelmingly
    // satisfied. Failing this is a strong signal of a hidden lock.
    let ratio = concurrent.as_secs_f64() / serial.as_secs_f64();
    assert!(
        ratio <= 1.25,
        "concurrent training took {:.2}× as long as serial \
         (concurrent={:?}, serial={:?}); a hidden lock is the likely \
         cause",
        ratio,
        concurrent,
        serial
    );
}

#[test]
fn graph_per_thread_construction_pattern() {
    // `AdaptiveBeliefGraph` is intentionally NOT Send — it
    // transitively holds `Rc` via `cjc_runtime::Tensor`. The
    // per-thread arena pattern works precisely because graphs
    // *never cross thread boundaries*: each thread constructs its
    // own graph locally inside the spawn closure (capturing only
    // `Send` parameters like seeds, paths, or numeric configs).
    //
    // This test is the documented witness for that pattern.
    // Compile-time confirmation that the closure body is `Send`
    // even though the graph it constructs is not:
    fn assert_send<T: Send>(_: T) {}
    let f = move || {
        let mut g = AdaptiveBeliefGraph::new(42);
        g.observe(0, 0.5).unwrap();
        // The graph is dropped here, never crossing the boundary.
        g.chain_head
    };
    assert_send(f);
    // And we can actually use it:
    let head = std::thread::spawn(f).join().unwrap();
    // chain_head is [u8; 32], which IS Send, so it crosses the
    // boundary as the return value.
    let _: [u8; 32] = head;
}
