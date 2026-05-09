//! ABNG Micro-Benchmark Suite (Phase 0.6 Items 2 + 3)
//! ====================================================
//!
//! Per-op cost measurement for the ABNG hot path. Establishes baseline
//! numbers BEFORE Item 3 (smart-replay fast-forward) and Item 4 (native
//! batch_observe + bulk BLR update) ship, so subsequent perf wins are
//! verifiable with cited deltas.
//!
//! Operations measured:
//!   - `observe`            — single Welford observation + chain-hash append
//!   - `blr_update`         — NIG conjugate update (n=1, d=4)
//!   - `blr_predict`        — Cholesky triangular solve
//!   - `encode_prefix`      — quantile-codebook encode
//!   - `descend`            — radix-tree route from prefix
//!   - `replay_smart_vs_naive` — Phase 0.6 Item 3: smart-replay
//!                              fast-forward speedup at n=10^4 events
//!                              over a compacted log
//!
//! Outputs JSONL to stdout for CI ingestion. Prints a human scorecard
//! to stderr. Manual `Instant` timing matches the existing
//! `bench/ad_bench/` convention (no Criterion dependency added).
//!
//! Invocation:
//!     cargo run -p abng-micro --release > abng_micro.jsonl

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay_with_outcome, serialize, ReplayOptions};
use cjc_ad::pinn::Activation;
use std::time::Instant;

const N_WARMUP: usize = 100;
const N_ITERS: usize = 10_000;

fn build_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

fn emit(op: &str, n: usize, total_ns: u128, per_op_ns: f64) {
    println!(
        r#"{{"op":"{op}","n":{n},"total_ns":{total_ns},"per_op_ns":{per_op_ns:.2}}}"#
    );
    eprintln!("  {op:<18}: {per_op_ns:>10.2} ns/op  ({n} iters)");
}

fn main() {
    eprintln!("=== ABNG Micro-Benchmark Suite (Phase 0.6 Item 2 baseline) ===");
    let seed = 42u64;

    // ── observe ────────────────────────────────────────────────────────
    {
        let mut g = build_graph(seed);
        for i in 0..N_WARMUP {
            g.observe(1, (i as f64) * 0.001).unwrap();
        }
        let start = Instant::now();
        for i in 0..N_ITERS {
            g.observe(1, (i as f64) * 0.001).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / N_ITERS as f64;
        emit("observe", N_ITERS, elapsed.as_nanos(), per_op);
    }

    // ── blr_update ─────────────────────────────────────────────────────
    {
        let mut g = build_graph(seed);
        let phi = [1.0_f64, 0.5, 0.25, 0.125];
        let y = [0.7_f64];
        for _ in 0..N_WARMUP {
            g.blr_update(1, &phi, &y).unwrap();
        }
        let start = Instant::now();
        for i in 0..N_ITERS {
            let yi = [0.7 + (i as f64) * 0.0001];
            g.blr_update(1, &phi, &yi).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / N_ITERS as f64;
        emit("blr_update", N_ITERS, elapsed.as_nanos(), per_op);
    }

    // ── blr_predict ────────────────────────────────────────────────────
    {
        let mut g = build_graph(seed);
        let phi = [1.0_f64, 0.5, 0.25, 0.125];
        let y = [0.7_f64];
        for _ in 0..100 {
            g.blr_update(1, &phi, &y).unwrap();
        }
        for _ in 0..N_WARMUP {
            let _ = g.blr_predict(1, &phi).unwrap();
        }
        let start = Instant::now();
        for _ in 0..N_ITERS {
            let _ = g.blr_predict(1, &phi).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / N_ITERS as f64;
        emit("blr_predict", N_ITERS, elapsed.as_nanos(), per_op);
    }

    // ── encode_prefix ──────────────────────────────────────────────────
    {
        let g = build_graph(seed);
        let x = [0.5_f64];
        for _ in 0..N_WARMUP {
            let _ = g.encode_prefix(&x).unwrap();
        }
        let start = Instant::now();
        for _ in 0..N_ITERS {
            let _ = g.encode_prefix(&x).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / N_ITERS as f64;
        emit("encode_prefix", N_ITERS, elapsed.as_nanos(), per_op);
    }

    // ── descend ────────────────────────────────────────────────────────
    {
        let g = build_graph(seed);
        let prefix = g.encode_prefix(&[0.5]).unwrap();
        for _ in 0..N_WARMUP {
            let _ = g.descend(&prefix);
        }
        let start = Instant::now();
        for _ in 0..N_ITERS {
            let _ = g.descend(&prefix);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / N_ITERS as f64;
        emit("descend", N_ITERS, elapsed.as_nanos(), per_op);
    }

    // ── Phase 0.6 Item 3: smart-replay fast-forward speedup ────────────
    bench_smart_replay_vs_naive(seed);

    eprintln!("=== Done ===");
    eprintln!("Phase 0.6 Item 2 baseline + Item 3 smart-replay speedup. Item 4 perf wins compare against these.");
}

/// Phase 0.6 Item 3 — measure smart-replay fast-forward speedup over
/// naive replay on a compacted audit log. Target per the Phase 0.6
/// handoff: ≥ 5× speedup at n_events ≈ 10^4.
///
/// Setup:
///   1. Build a graph with a small fanout of 4 leaves.
///   2. Observe 10^4 values cycling through the leaves.
///   3. compact_log over the full audit length — emits one
///      StatsSnapshot per touched node.
///   4. Serialize.
///   5. Replay both ways multiple times; pick best timing per side.
fn bench_smart_replay_vs_naive(seed: u64) {
    const N_OBS: usize = 10_000;
    const N_TRIALS: usize = 5;

    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    for byte in 1u8..5 {
        let _ = g.add_node(0, byte).unwrap();
    }
    for i in 0..N_OBS {
        let leaf = ((i % 4) + 1) as u32;
        let v = (i as f64 * 0.0001) - 0.5;
        g.observe(leaf, v).unwrap();
    }
    let _ = g.compact_log(g.audit.len() as u64);
    let blob = serialize(&g);
    let blob_size = blob.len();
    let pre_audit_len = g.audit.len();

    // Time naive replay (smart_replay = false). Take min over trials
    // to filter out noise.
    let mut naive_min_ns = u128::MAX;
    let mut ff_count = 0u64;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let outcome =
            replay_with_outcome(&blob, ReplayOptions::default()).unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < naive_min_ns {
            naive_min_ns = elapsed;
        }
        // Sanity check: naive must skip 0 events.
        assert_eq!(outcome.fast_forwarded_events, 0);
    }

    // Time smart replay.
    let mut smart_min_ns = u128::MAX;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let outcome = replay_with_outcome(
            &blob,
            ReplayOptions { smart_replay: true },
        )
        .unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < smart_min_ns {
            smart_min_ns = elapsed;
        }
        ff_count = outcome.fast_forwarded_events;
    }

    let speedup = naive_min_ns as f64 / smart_min_ns as f64;
    println!(
        r#"{{"op":"replay_smart_vs_naive","n_events":{pre_audit_len},"blob_bytes":{blob_size},"naive_min_ns":{naive_min_ns},"smart_min_ns":{smart_min_ns},"fast_forwarded":{ff_count},"speedup":{speedup:.2}}}"#
    );
    eprintln!(
        "  replay_smart_vs_naive: naive={:.2}ms smart={:.2}ms speedup={:.2}x ff_count={} ({} events, {} B blob)",
        naive_min_ns as f64 / 1e6,
        smart_min_ns as f64 / 1e6,
        speedup,
        ff_count,
        pre_audit_len,
        blob_size,
    );
}
