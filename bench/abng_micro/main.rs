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

use cjc_abng::blr::{BlrPrior, BlrState};
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

    // ── blr_state_update_direct (Phase 0.7 isolation bench) ──────────
    //
    // Calls `BlrState::update` directly with no graph dispatch / audit /
    // chain-hashing. The full graph-level `blr_update` above includes
    // ~5–8 µs of route+encode+chain-hash overhead per call, swamping
    // any sub-µs change in the BLR math itself. This bench bypasses
    // that scaffolding so future Phase 0.7+ work on the BlrState math
    // (Cholesky factor caching, SIMD Kahan, packed precision matrix,
    // etc.) can be measured against a tight noise floor (~929 ns/op,
    // CV ~4% on Windows MSVC at d=4, n_iters=100k).
    {
        let n_iters = 100_000_usize;
        let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
        let mut s = BlrState::from_prior(&prior, 4);
        let phi = [1.0_f64, 0.5, 0.25, 0.125];
        let y = [0.7_f64];
        for _ in 0..N_WARMUP {
            s.update(&phi, &y).unwrap();
        }
        let start = Instant::now();
        for i in 0..n_iters {
            let yi = [0.7 + (i as f64) * 0.0001];
            s.update(&phi, &yi).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / n_iters as f64;
        emit("blr_state_update_direct", n_iters, elapsed.as_nanos(), per_op);
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

    // ── Phase 0.6 Item 4: observe_batch vs per-row observe speedup ────
    bench_observe_batch(seed);

    // ── Phase 0.6 Item 7: route_to_leaf vs encode_prefix+descend ──────
    bench_route_to_leaf(seed);

    // ── Phase 0.6 Item 8: route_to_leaf_batch vs N per-row calls ──────
    bench_route_to_leaf_batch(seed);

    eprintln!("=== Done ===");
    eprintln!("Phase 0.6 baseline + Items 3/4/7/8 perf wins.");
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

/// Phase 0.6 Item 4 — measure observe_batch speedup at varying batch
/// sizes. The handoff target was ≥10× at n=1024.
///
/// Compares three paths for the same N observations on the same node:
///   - per-row N × `g.observe(node, value)` (N audit events, N
///     stats_chain advances)
///   - `g.observe_slice(node, values)` (legacy loop semantics — same
///     as per-row, just a single dispatch call)
///   - `g.observe_batch(node, values)` (1 BeliefUpdateBatch event,
///     1 stats_chain advance — the v13 perf win)
///
/// All three produce bit-identical NodeStats canonical_bytes. Only the
/// audit-chain accounting differs.
fn bench_observe_batch(seed: u64) {
    const N_TRIALS: usize = 5;

    for &n in &[64usize, 1024usize] {
        // Generate the same value sequence for all three paths.
        let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 0.3).collect();

        // Per-row.
        let mut per_row_min_ns = u128::MAX;
        for _ in 0..N_TRIALS {
            let mut g = AdaptiveBeliefGraph::new(seed);
            let start = Instant::now();
            for &v in &values {
                g.observe(0, v).unwrap();
            }
            let elapsed = start.elapsed().as_nanos();
            if elapsed < per_row_min_ns {
                per_row_min_ns = elapsed;
            }
        }

        // Batch.
        let mut batch_min_ns = u128::MAX;
        for _ in 0..N_TRIALS {
            let mut g = AdaptiveBeliefGraph::new(seed);
            let start = Instant::now();
            g.observe_batch(0, &values).unwrap();
            let elapsed = start.elapsed().as_nanos();
            if elapsed < batch_min_ns {
                batch_min_ns = elapsed;
            }
        }

        let speedup = per_row_min_ns as f64 / batch_min_ns as f64;
        println!(
            r#"{{"op":"observe_batch","n":{n},"per_row_min_ns":{per_row_min_ns},"batch_min_ns":{batch_min_ns},"speedup":{speedup:.2}}}"#
        );
        eprintln!(
            "  observe_batch n={n:>4}: per_row={:>9.2}us  batch={:>7.2}us  speedup={:.2}x",
            per_row_min_ns as f64 / 1e3,
            batch_min_ns as f64 / 1e3,
            speedup,
        );
    }
}

/// Phase 0.6 Item 7 — measure the cost of the route-to-leaf operation
/// at the Rust API boundary.
///
/// At the **Rust API** level there is no perf difference between the
/// 3-step pattern (`encode_prefix → descend → extract leaf_id`) and a
/// hand-written equivalent — the work is identical. This bench just
/// pins the absolute cost of one route operation so future kernel
/// optimizations have a number to compare against.
///
/// The actual perf win of the new `abng_route_to_leaf` *builtin* is
/// at the **CJC-Lang interpreter dispatch boundary**: it collapses 3
/// builtin dispatches + 1 Tensor allocation + 1 get-by-index call
/// into a single dispatch returning `Value::Int`. Each saved
/// dispatch is ~few hundred ns of interpreter overhead through the
/// AST tree-walk and MIR register-machine. A bench-harness measurement
/// of this CJC-Lang-side gain belongs in a separate `.cjcl` source
/// run, not here.
fn bench_route_to_leaf(seed: u64) {
    const N_ITERS: usize = 10_000;

    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    for byte in 0u8..4 {
        let _ = g.add_node(0, byte).unwrap();
    }

    // Warmup so cache is hot and branch predictors are settled.
    for i in 0..1000 {
        let x = (i as f64) * 0.0001;
        let bytes = g.encode_prefix(&[x]).unwrap();
        let _ = g.descend(&bytes).leaf_id;
    }

    // Measure the absolute per-route cost.
    let start = Instant::now();
    let mut leaf_acc = 0u32;
    for i in 0..N_ITERS {
        let x = (i as f64) * 0.0001;
        let bytes = g.encode_prefix(&[x]).unwrap();
        let leaf = g.descend(&bytes).leaf_id;
        leaf_acc = leaf_acc.wrapping_add(leaf);
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let per_op = elapsed_ns as f64 / N_ITERS as f64;
    // Touch leaf_acc so the optimizer can't elide the loop body.
    eprintln!(
        "  route_to_leaf rust:  {:.2} ns/op  ({} iters; loop-acc=0x{:x})",
        per_op, N_ITERS, leaf_acc
    );
    println!(
        r#"{{"op":"route_to_leaf_rust","n":{N_ITERS},"total_ns":{elapsed_ns},"per_op_ns":{per_op:.2}}}"#
    );
    eprintln!(
        "    (Rust-side cost only; the abng_route_to_leaf builtin's CJC-Lang-side win comes from collapsing 3 interpreter dispatches + 1 Tensor allocation into 1 dispatch.)"
    );
}

/// Phase 0.6 Item 8 — measure the chunked-dispatch (batched
/// route_to_leaf) speedup vs N single-row calls. The math is
/// identical row-for-row; the win comes from amortizing per-call
/// overhead (allocation reuse for the prefix buffer + single output
/// allocation for N leaf ids).
fn bench_route_to_leaf_batch(seed: u64) {
    const N_BATCH: usize = 1024;
    const N_TRIALS: usize = 5;

    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    for byte in 0u8..4 {
        let _ = g.add_node(0, byte).unwrap();
    }
    let xs: Vec<f64> = (0..N_BATCH).map(|i| (i as f64) * 0.0009).collect();

    // Per-row: N calls of encode_prefix + descend.
    let mut per_row_min_ns = u128::MAX;
    let mut leaf_acc = 0u32;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        for &x in &xs {
            let bytes = g.encode_prefix(&[x]).unwrap();
            let leaf = g.descend(&bytes).leaf_id;
            leaf_acc = leaf_acc.wrapping_add(leaf);
        }
        let elapsed = start.elapsed().as_nanos();
        if elapsed < per_row_min_ns {
            per_row_min_ns = elapsed;
        }
    }

    // Batched: one route_to_leaf_batch call.
    let mut batch_min_ns = u128::MAX;
    let mut leaf_acc_b = 0u32;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let leaves = g.route_to_leaf_batch(&xs, N_BATCH).unwrap();
        for l in &leaves {
            leaf_acc_b = leaf_acc_b.wrapping_add(*l);
        }
        let elapsed = start.elapsed().as_nanos();
        if elapsed < batch_min_ns {
            batch_min_ns = elapsed;
        }
    }
    assert_eq!(leaf_acc, leaf_acc_b);

    let speedup = per_row_min_ns as f64 / batch_min_ns as f64;
    println!(
        r#"{{"op":"route_to_leaf_batch","n":{N_BATCH},"per_row_min_ns":{per_row_min_ns},"batch_min_ns":{batch_min_ns},"speedup":{speedup:.2}}}"#
    );
    eprintln!(
        "  route_to_leaf_batch n={N_BATCH}: per_row={:.2}us  batch={:.2}us  speedup={:.2}x  (Rust-side; CJC-Lang dispatch savings stack on top)",
        per_row_min_ns as f64 / 1e3,
        batch_min_ns as f64 / 1e3,
        speedup,
    );
}
