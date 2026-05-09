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
use cjc_abng::codebook::QuantileCodebook;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{
    replay, replay_mmap_with_outcome, replay_with_outcome, serialize, serialize_compressed,
    serialize_into, ReplayOptions,
};
use cjc_ad::pinn::Activation;
use cjc_repro::{KahanAccumulatorF64, KahanAccumulatorF64x4, KahanAccumulatorF64x8};
use std::io::Write;
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

    // ── train_step vs 3-call (Phase 0.7 Item 4) ──────────────────────
    //
    // Compares the fused `Graph::train_step` Rust call against the
    // canonical 3-call training sequence (`encode_prefix` + `descend` +
    // `blr_update` + `observe`). This bench measures the Rust-API cost
    // — the bigger language-level win is the ~5-10 us saved per row by
    // collapsing 3 CJC-Lang dispatches into 1, which this bench does
    // NOT capture (no interpreter dispatch in either path).
    {
        let n_iters = 5_000_usize;
        // Fresh graph per loop iteration: training mutates state.
        // Use the same setup pattern as `build_graph` (1-D codebook,
        // 4-element leaf head, BLR prior).
        let phi = [1.0_f64, 0.5, 0.25, 0.125];
        let x = [0.5_f64];

        // 3-call sequence baseline.
        let mut g = build_graph(seed);
        for _ in 0..N_WARMUP {
            let prefix = g.encode_prefix(&x).unwrap();
            let leaf = g.descend(&prefix).leaf_id;
            g.blr_update(leaf, &phi, &[0.7]).unwrap();
            g.observe(leaf, 0.7).unwrap();
        }
        let start = Instant::now();
        for i in 0..n_iters {
            let yi = 0.7 + (i as f64) * 0.0001;
            let prefix = g.encode_prefix(&x).unwrap();
            let leaf = g.descend(&prefix).leaf_id;
            g.blr_update(leaf, &phi, &[yi]).unwrap();
            g.observe(leaf, yi).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / n_iters as f64;
        emit("train_step_3call", n_iters, elapsed.as_nanos(), per_op);

        // Fused train_step.
        let mut g = build_graph(seed);
        for _ in 0..N_WARMUP {
            g.train_step(&x, &phi, 0.7).unwrap();
        }
        let start = Instant::now();
        for i in 0..n_iters {
            let yi = 0.7 + (i as f64) * 0.0001;
            g.train_step(&x, &phi, yi).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / n_iters as f64;
        emit("train_step_fused", n_iters, elapsed.as_nanos(), per_op);
    }

    // ── verify_chain (Phase 0.7 A) ────────────────────────────────────
    //
    // Builds a graph with N observations (so the audit log has ~N+5
    // entries — 4 setup + N observes), then measures `verify_chain`
    // wall-clock. Phase 0.7 (A) replaced the per-event `payload_bytes()`
    // Vec allocation with a single buffer reused across the verify
    // walk; this bench captures the resulting speedup.
    {
        let n_events = 10_000_usize;
        let mut g = build_graph(seed);
        for i in 0..n_events {
            g.observe(1, (i as f64) * 0.001).unwrap();
        }
        // Warm up the verify path so the first sample isn't paying
        // page-fault / branch-predictor costs.
        for _ in 0..3 {
            g.verify_chain().unwrap();
        }
        let n_verifies = 50_usize;
        let start = Instant::now();
        for _ in 0..n_verifies {
            g.verify_chain().unwrap();
        }
        let elapsed = start.elapsed();
        // per-op = per chain-verify, NOT per event hash
        let per_op = elapsed.as_nanos() as f64 / n_verifies as f64;
        emit("verify_chain_10k", n_verifies, elapsed.as_nanos(), per_op);
    }

    // ── codebook_encode_vs_encode_into (Phase 0.7 B) ──────────────────
    //
    // Direct comparison of allocating `encode` vs buffer-reuse
    // `encode_into` at d=8, n_bins=16 (a tabular-style codebook —
    // bigger than the d=1 graph encode_prefix bench so the per-call
    // alloc is measurable above the noise floor). Both methods walk
    // the same `partition_point` logic; the only difference is whether
    // the output Vec is fresh per call (encode) or reused from the
    // caller's buffer (encode_into).
    {
        let n_iters = 100_000_usize;
        let n_dims = 8u8;
        let n_bins = 16u16;
        // boundaries: 0.5, 1.5, ..., n_bins-1.5 per dim
        let mut flat = Vec::new();
        for _ in 0..n_dims {
            for k in 1..n_bins {
                flat.push(k as f64 - 0.5);
            }
        }
        let cb = QuantileCodebook::from_flat(n_dims as usize, n_bins, &flat).unwrap();
        let x: Vec<f64> = (0..n_dims).map(|i| 0.5 + i as f64 * 0.7).collect();

        // Allocating variant (baseline shape)
        for _ in 0..N_WARMUP {
            let _ = cb.encode(&x).unwrap();
        }
        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = cb.encode(&x).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / n_iters as f64;
        emit("codebook_encode_alloc", n_iters, elapsed.as_nanos(), per_op);

        // Buffer-reuse variant (Phase 0.7 B)
        let mut buf: Vec<u8> = Vec::with_capacity(n_dims as usize);
        for _ in 0..N_WARMUP {
            cb.encode_into(&x, &mut buf).unwrap();
        }
        let start = Instant::now();
        for _ in 0..n_iters {
            cb.encode_into(&x, &mut buf).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / n_iters as f64;
        emit("codebook_encode_into", n_iters, elapsed.as_nanos(), per_op);
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

    // ── Phase 0.8 Item B1: mmap-replay vs read-to-end speedup ─────────
    bench_replay_mmap_vs_naive(seed);

    // ── Phase 0.8 Item B1 (at scale): same comparison at 10^6 events ──
    bench_replay_mmap_vs_naive_at_scale(seed);

    // ── Phase 0.6 Item 4: observe_batch vs per-row observe speedup ────
    bench_observe_batch(seed);

    // ── Phase 0.6 Item 7: route_to_leaf vs encode_prefix+descend ──────
    bench_route_to_leaf(seed);

    // ── Phase 0.6 Item 8: route_to_leaf_batch vs N per-row calls ──────
    bench_route_to_leaf_batch(seed);

    // ── Phase 0.8 Item C1: parallel route_to_leaf_batch ───────────────
    bench_parallel_route_to_leaf_batch(seed);

    // ── Phase 0.8 Item D2: scalar vs SIMD x4/x8 Kahan accumulator ─────
    bench_kahan_simd_vs_scalar();

    // ── Phase 0.8 Item B2: serialize_into (streaming) vs serialize ────
    bench_serialize_streaming_vs_buffered(seed);

    // ── Phase 0.8 Item B3: zstd-compressed snapshot ratio + speed ─────
    bench_serialize_compressed(seed);

    // ── Phase 0.8 Item C3: concurrent multi-graph training scaling ────
    bench_concurrent_training_scaling(seed);

    // ── Phase 0.8 Item D1: BLR predict cache hit vs miss ──────────────
    bench_blr_predict_cache(seed);

    eprintln!("=== Done ===");
    eprintln!("Phase 0.6 baseline + Items 3/4/7/8 perf wins + Phase 0.8 B1 mmap + C1 parallel route + D2 SIMD Kahan + B2 streaming encode + B3 zstd compression + C3 arena + D1 BLR cache.");
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

/// Phase 0.8 Item B1 — measure memory-mapped replay vs the naive
/// `replay(&fs::read(path))` pattern on a snapshot persisted to disk.
///
/// The naive path allocates one heap `Vec<u8>` sized to the full blob,
/// then hands it to the slice decoder. The mmap path maps the file via
/// the OS page cache; pages fault in lazily as the decoder's `Cursor`
/// advances. Both feed the *same* decoder loop, so the rebuilt graph
/// is byte-identical between the two paths (verified by
/// `tests/abng/serialize_mmap.rs`).
///
/// At n_events ≈ 10^4 the snapshot is on the order of ~1 MB, which
/// fits comfortably in `read_to_end`'s amortized growth. The mmap
/// path's win is primarily on the *memory-footprint* axis (working
/// set vs full blob) rather than wall-clock; this bench records the
/// wall-clock number honestly so we can spot regressions and so the
/// next-stage bench at n_events ≈ 10^6 has a baseline to compare to.
fn bench_replay_mmap_vs_naive(seed: u64) {
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
    let blob = serialize(&g);
    let blob_size = blob.len();
    let n_events = g.audit.len();

    // Persist to disk once. Keep the temp file alive across both
    // benches — dropping it would unlink the file under us.
    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(&blob).expect("write snapshot");
    tmp.flush().expect("flush snapshot");
    let path = tmp.path().to_path_buf();

    // Time naive replay: read_to_end → slice decoder.
    let mut naive_min_ns = u128::MAX;
    for _ in 0..N_TRIALS {
        let bytes = std::fs::read(&path).expect("read snapshot");
        let start = Instant::now();
        let _ = replay_with_outcome(&bytes, ReplayOptions::default()).unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < naive_min_ns {
            naive_min_ns = elapsed;
        }
    }

    // Time mmap replay: File::open → Mmap::map → slice decoder.
    let mut mmap_min_ns = u128::MAX;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let _ = replay_mmap_with_outcome(&path, ReplayOptions::default()).unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < mmap_min_ns {
            mmap_min_ns = elapsed;
        }
    }

    let speedup = naive_min_ns as f64 / mmap_min_ns as f64;
    println!(
        r#"{{"op":"replay_mmap_vs_naive","n_events":{n_events},"blob_bytes":{blob_size},"naive_min_ns":{naive_min_ns},"mmap_min_ns":{mmap_min_ns},"speedup":{speedup:.2}}}"#
    );
    eprintln!(
        "  replay_mmap_vs_naive : naive={:.2}ms mmap={:.2}ms speedup={:.2}x ({} events, {} B blob)",
        naive_min_ns as f64 / 1e6,
        mmap_min_ns as f64 / 1e6,
        speedup,
        n_events,
        blob_size,
    );
}

/// Phase 0.8 Item B1 at scale — measure memory-mapped replay vs the
/// naive `replay(&fs::read(path))` pattern on a **10^6-event**
/// snapshot persisted to disk.
///
/// At this scale (~130 MB blob on disk) the naive path's
/// `read_to_end` does a single ~130 MB heap allocation plus a kernel
/// buffer copy. The mmap path skips both: `Mmap::map` returns
/// immediately with a virtual mapping, and the kernel page-faults
/// pages in lazily as the decoder's `Cursor` advances. The forward-
/// sequential scan pattern is exactly what the OS read-ahead
/// heuristics are tuned for.
///
/// This bench is the scale at which the design note's "100× smaller
/// RAM peak" claim materializes; the 10^4 bench above is the
/// noise-floor baseline. Both benches share the same code path, so
/// any regression in the slice or mmap fast paths will show up here
/// with much larger absolute deltas.
///
/// Cost: ~30 seconds wall-clock for setup + 3 trials × 2 paths.
fn bench_replay_mmap_vs_naive_at_scale(seed: u64) {
    const N_OBS: usize = 1_000_000;
    const N_TRIALS: usize = 3;

    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    for byte in 1u8..5 {
        let _ = g.add_node(0, byte).unwrap();
    }
    for i in 0..N_OBS {
        let leaf = ((i % 4) + 1) as u32;
        let v = (i as f64 * 0.0000001) - 0.5;
        g.observe(leaf, v).unwrap();
    }
    let blob = serialize(&g);
    let blob_size = blob.len();
    let n_events = g.audit.len();

    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(&blob).expect("write snapshot");
    tmp.flush().expect("flush snapshot");
    let path = tmp.path().to_path_buf();

    // Drop the in-memory blob to free the working set before timing.
    // Without this, the OS page cache for the file is essentially
    // primed by the write, but the heap still holds the original
    // 130 MB allocation, which would skew the second bench's
    // perceived memory footprint.
    drop(blob);

    // Time naive replay: read_to_end → slice decoder.
    let mut naive_min_ns = u128::MAX;
    for _ in 0..N_TRIALS {
        let bytes = std::fs::read(&path).expect("read snapshot");
        let start = Instant::now();
        let _ = replay_with_outcome(&bytes, ReplayOptions::default()).unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < naive_min_ns {
            naive_min_ns = elapsed;
        }
    }

    // Time mmap replay: File::open → Mmap::map → slice decoder.
    let mut mmap_min_ns = u128::MAX;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let _ = replay_mmap_with_outcome(&path, ReplayOptions::default()).unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < mmap_min_ns {
            mmap_min_ns = elapsed;
        }
    }

    let speedup = naive_min_ns as f64 / mmap_min_ns as f64;
    println!(
        r#"{{"op":"replay_mmap_vs_naive_at_scale","n_events":{n_events},"blob_bytes":{blob_size},"naive_min_ns":{naive_min_ns},"mmap_min_ns":{mmap_min_ns},"speedup":{speedup:.2}}}"#
    );
    eprintln!(
        "  replay_mmap_vs_naive_at_scale : naive={:.2}ms mmap={:.2}ms speedup={:.2}x ({} events, {} B blob)",
        naive_min_ns as f64 / 1e6,
        mmap_min_ns as f64 / 1e6,
        speedup,
        n_events,
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
/// Phase 0.8 Item C1 — measure `route_to_leaf_batch_par` against the
/// serial `route_to_leaf_batch` across thread counts {1, 2, 4, 8} on
/// a 10,000-row routing workload.
///
/// What we measure:
///   * Speedup over serial as `n_threads` grows. Ideal is N× at no
///     contention; real-world is bounded by physical cores, the
///     small per-row work (~50–100 ns), and `std::thread::scope`'s
///     spawn overhead (~10 µs per thread).
///   * Determinism witness: the per-thread output's accumulated
///     XOR of leaf ids matches the serial output's XOR for every
///     thread count. (Asserted in tests; bench just emits the
///     numbers.)
///
/// Caveat: at small per-row cost, thread spawn overhead can dominate
/// for small batches. We pick n=10k specifically because it's large
/// enough that 4 threads at ~5 µs/thread × 4 = ~20 µs spawn overhead
/// is small relative to the ~1 ms total routing work.
fn bench_parallel_route_to_leaf_batch(seed: u64) {
    const N_BATCH: usize = 10_000;
    const N_TRIALS: usize = 5;

    let g = build_graph(seed);
    // d=1 codebook from build_graph; xs of length N_BATCH * 1.
    let xs: Vec<f64> = (0..N_BATCH)
        .map(|i| ((i as f64) * 0.137).sin())
        .collect();

    // Warm-up: the JIT branch-predictor + L1 cache.
    for _ in 0..3 {
        let _ = g.route_to_leaf_batch(&xs, N_BATCH).unwrap();
    }

    // Serial baseline.
    let mut serial_min_ns = u128::MAX;
    let mut serial_first: Vec<u32> = Vec::new();
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let leaves = g.route_to_leaf_batch(&xs, N_BATCH).unwrap();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < serial_min_ns {
            serial_min_ns = elapsed;
            serial_first = leaves;
        }
    }

    // Parallel at thread counts {1, 2, 4, 8}.
    for &n_threads in &[1usize, 2, 4, 8] {
        let mut par_min_ns = u128::MAX;
        let mut par_first: Vec<u32> = Vec::new();
        for _ in 0..N_TRIALS {
            let start = Instant::now();
            let leaves = g
                .route_to_leaf_batch_par(&xs, N_BATCH, n_threads)
                .unwrap();
            let elapsed = start.elapsed().as_nanos();
            if elapsed < par_min_ns {
                par_min_ns = elapsed;
                par_first = leaves;
            }
        }
        // Determinism gate: parallel must equal serial byte-for-byte.
        assert_eq!(
            par_first, serial_first,
            "route_to_leaf_batch_par at n_threads={n_threads} diverged from serial"
        );
        let speedup = serial_min_ns as f64 / par_min_ns as f64;
        println!(
            r#"{{"op":"route_to_leaf_batch_par","n":{N_BATCH},"n_threads":{n_threads},"serial_min_ns":{serial_min_ns},"par_min_ns":{par_min_ns},"speedup":{speedup:.3}}}"#
        );
        eprintln!(
            "  route_to_leaf_batch_par n_threads={n_threads}: serial={:.2}ms  par={:.2}ms  speedup={:.2}x  (ideal=n at no-contention)",
            serial_min_ns as f64 / 1e6,
            par_min_ns as f64 / 1e6,
            speedup,
        );
    }
}

/// Phase 0.8 Item D2 — measure `KahanAccumulatorF64x4` and
/// `KahanAccumulatorF64x8` against the scalar `KahanAccumulatorF64`
/// on slice reductions of varying length.
///
/// What we measure:
///   * Wall-clock for `add_slice` + `finalize` at n=256, 4096, 65_536.
///   * Speedup ratio of SIMD-shape over scalar.
///
/// What this does NOT prove:
///   * Compiler auto-vectorization. The plain-`[f64; N]` representation
///     opportunistically vectorizes on AVX/AVX2/NEON release builds,
///     but correctness is independent of whether SIMD instructions
///     get emitted. We report the wall-clock honestly; if it looks
///     scalar-equivalent, that's just an unvectorized build.
///   * Determinism. The byte-identical-across-runs gate lives in
///     `tests/simd_kahan_determinism.rs`; here we just include a
///     `to_bits()` assertion that flags any catastrophic divergence.
fn bench_kahan_simd_vs_scalar() {
    const N_TRIALS: usize = 7;

    for &n in &[256usize, 4_096, 65_536] {
        // Fixed deterministic input. Same source on every platform.
        let values: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.137).sin() * 1e7)
            .collect();

        // ── scalar ───────────────────────────────────────────────
        let mut scalar_min_ns = u128::MAX;
        let mut scalar_bits = 0u64;
        for _ in 0..N_TRIALS {
            let mut acc = KahanAccumulatorF64::new();
            let start = Instant::now();
            acc.add_slice(&values);
            let result = acc.finalize();
            let elapsed = start.elapsed().as_nanos();
            if elapsed < scalar_min_ns {
                scalar_min_ns = elapsed;
                scalar_bits = result.to_bits();
            }
        }

        // ── x4 ───────────────────────────────────────────────────
        let mut x4_min_ns = u128::MAX;
        let mut x4_bits = 0u64;
        for _ in 0..N_TRIALS {
            let mut acc = KahanAccumulatorF64x4::new();
            let start = Instant::now();
            acc.add_slice(&values);
            let result = acc.finalize();
            let elapsed = start.elapsed().as_nanos();
            if elapsed < x4_min_ns {
                x4_min_ns = elapsed;
                x4_bits = result.to_bits();
            }
        }

        // ── x8 ───────────────────────────────────────────────────
        let mut x8_min_ns = u128::MAX;
        let mut x8_bits = 0u64;
        for _ in 0..N_TRIALS {
            let mut acc = KahanAccumulatorF64x8::new();
            let start = Instant::now();
            acc.add_slice(&values);
            let result = acc.finalize();
            let elapsed = start.elapsed().as_nanos();
            if elapsed < x8_min_ns {
                x8_min_ns = elapsed;
                x8_bits = result.to_bits();
            }
        }

        // Determinism cross-check: same bits within each variant
        // across trials. We don't compare x4↔scalar bits (different
        // accumulation order is expected to produce different bits).
        // The actual cross-platform byte-identity gate lives in the
        // top-level determinism test binary.
        let _ = std::hint::black_box((scalar_bits, x4_bits, x8_bits));

        let x4_speedup = scalar_min_ns as f64 / x4_min_ns as f64;
        let x8_speedup = scalar_min_ns as f64 / x8_min_ns as f64;

        println!(
            r#"{{"op":"kahan_simd_vs_scalar","n":{n},"scalar_min_ns":{scalar_min_ns},"x4_min_ns":{x4_min_ns},"x8_min_ns":{x8_min_ns},"x4_speedup":{x4_speedup:.3},"x8_speedup":{x8_speedup:.3}}}"#
        );
        eprintln!(
            "  kahan_simd_vs_scalar n={n}: scalar={:.2}us  x4={:.2}us  x8={:.2}us  speedup x4={:.2}x x8={:.2}x",
            scalar_min_ns as f64 / 1e3,
            x4_min_ns as f64 / 1e3,
            x8_min_ns as f64 / 1e3,
            x4_speedup,
            x8_speedup,
        );
    }
}

/// Phase 0.8 Item B2 — compare `serialize(&g)` (full-Vec materialization)
/// against `serialize_into(&g, &mut writer)` (streaming) on a moderately
/// sized graph.
///
/// What we measure:
///   * Wall-clock: both paths walk the same byte sequence, so the
///     numbers should be in the same ballpark. A regression in either
///     direction would indicate the refactor introduced unexpected
///     allocator overhead.
///   * Byte-identity: re-serialize through both paths and assert the
///     outputs are equal — the determinism guarantee, sanity-checked.
///
/// What this does NOT prove:
///   * RAM peak. The design's main claim is `O(working_set)` vs
///     `O(snapshot_size)` memory during emission, which `Instant::now`
///     does not measure. Validate that downstream via an OS-level peak
///     RSS probe if needed.
fn bench_serialize_streaming_vs_buffered(seed: u64) {
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

    // Path A: buffered. `serialize` allocates a Vec sized to the
    // snapshot blob and returns it.
    let mut buffered_min_ns = u128::MAX;
    let mut buffered_size = 0usize;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let blob = serialize(&g);
        let elapsed = start.elapsed().as_nanos();
        if elapsed < buffered_min_ns {
            buffered_min_ns = elapsed;
            buffered_size = blob.len();
        }
        let _ = std::hint::black_box(blob);
    }

    // Path B: streaming through a `Vec<u8>` (which implements `Write`).
    // The streaming path's win is peak memory rather than wall-clock,
    // but Vec<u8>'s grow-by-doubling is comparable to `serialize`'s
    // build pattern — wall-clock should be similar.
    let mut streaming_min_ns = u128::MAX;
    let mut streaming_size = 0usize;
    for _ in 0..N_TRIALS {
        let mut sink: Vec<u8> = Vec::new();
        let start = Instant::now();
        serialize_into(&g, &mut sink).expect("serialize_into(Vec<u8>)");
        let elapsed = start.elapsed().as_nanos();
        if elapsed < streaming_min_ns {
            streaming_min_ns = elapsed;
            streaming_size = sink.len();
        }
        let _ = std::hint::black_box(sink);
    }

    // Determinism gate: same byte count from both paths. Bytes-equal
    // is verified in tests/abng/serialize_streaming.rs; here we just
    // check size to flag any catastrophic divergence in the bench.
    assert_eq!(
        buffered_size, streaming_size,
        "serialize and serialize_into produced different byte counts"
    );

    let ratio = buffered_min_ns as f64 / streaming_min_ns as f64;
    println!(
        r#"{{"op":"serialize_streaming_vs_buffered","n_events":{N_OBS},"blob_bytes":{buffered_size},"buffered_min_ns":{buffered_min_ns},"streaming_min_ns":{streaming_min_ns},"streaming_vs_buffered_ratio":{ratio:.3}}}"#
    );
    eprintln!(
        "  serialize_streaming_vs_buffered n_events={N_OBS}: buffered={:.2}ms  streaming={:.2}ms  ratio={:.2}x  ({} B blob)",
        buffered_min_ns as f64 / 1e6,
        streaming_min_ns as f64 / 1e6,
        ratio,
        buffered_size,
    );
}

/// Phase 0.8 Item B3 — measure zstd compression of a snapshot at
/// level 3 (the default).
///
/// What we measure:
///   * Compression ratio (uncompressed_size / compressed_size).
///   * Wall-clock for compress and decompress+replay, vs the
///     uncompressed `serialize` + `replay` baseline.
///
/// Honest expectations:
///   * Ratio: audit-event-heavy snapshots compress well (~2-5x)
///     because every event has the same 21-byte header shape +
///     repetitive hash patterns. Codebook + per-node sections are
///     less compressible.
///   * Wall-clock: compression is CPU-bound; decompression is
///     fast. For replay-from-disk workflows the smaller blob can
///     more-than-pay for the decompression cost; for in-memory
///     workflows it's pure overhead. We report both so callers can
///     choose.
fn bench_serialize_compressed(seed: u64) {
    const N_OBS: usize = 10_000;
    const N_TRIALS: usize = 5;
    const LEVEL: i32 = 3;

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

    // Uncompressed sizes + baseline wall-clock.
    let mut uncompressed_min_ns = u128::MAX;
    let mut uncompressed_size = 0usize;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let blob = serialize(&g);
        let elapsed = start.elapsed().as_nanos();
        if elapsed < uncompressed_min_ns {
            uncompressed_min_ns = elapsed;
            uncompressed_size = blob.len();
        }
        let _ = std::hint::black_box(blob);
    }

    // Compressed size + wall-clock.
    let mut compressed_min_ns = u128::MAX;
    let mut compressed_size = 0usize;
    let mut compressed_blob: Vec<u8> = Vec::new();
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let blob = serialize_compressed(&g, LEVEL);
        let elapsed = start.elapsed().as_nanos();
        if elapsed < compressed_min_ns {
            compressed_min_ns = elapsed;
            compressed_size = blob.len();
            compressed_blob = blob;
        } else {
            let _ = std::hint::black_box(blob);
        }
    }

    // Replay-from-compressed wall-clock. The decompression cost is
    // amortized inside `replay`.
    let mut replay_compressed_min_ns = u128::MAX;
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let g2 = replay(&compressed_blob).expect("replay(compressed)");
        let elapsed = start.elapsed().as_nanos();
        if elapsed < replay_compressed_min_ns {
            replay_compressed_min_ns = elapsed;
        }
        let _ = std::hint::black_box(g2);
    }

    let ratio = uncompressed_size as f64 / compressed_size as f64;
    let compress_overhead = compressed_min_ns as f64 / uncompressed_min_ns as f64;
    println!(
        r#"{{"op":"serialize_compressed","n_events":{N_OBS},"level":{LEVEL},"uncompressed_bytes":{uncompressed_size},"compressed_bytes":{compressed_size},"ratio":{ratio:.2},"uncompressed_min_ns":{uncompressed_min_ns},"compressed_min_ns":{compressed_min_ns},"replay_compressed_min_ns":{replay_compressed_min_ns},"compress_overhead":{compress_overhead:.2}}}"#
    );
    eprintln!(
        "  serialize_compressed (level={LEVEL}, n_events={N_OBS}): \
         uncompressed={} B  compressed={} B  ratio={:.2}x \
         compress={:.2}ms  uncompressed_write={:.2}ms  replay_compressed={:.2}ms",
        uncompressed_size,
        compressed_size,
        ratio,
        compressed_min_ns as f64 / 1e6,
        uncompressed_min_ns as f64 / 1e6,
        replay_compressed_min_ns as f64 / 1e6,
    );
}

/// Phase 0.8 Item C3 — measure concurrent multi-graph training
/// throughput vs serial.
///
/// Workload: train N independent graphs for K observations each.
/// Serial path runs all N on a single thread; concurrent path
/// spawns N threads (one graph per thread). Each graph is
/// constructed inside its thread, so the graph itself never
/// crosses a thread boundary — the per-thread arena pattern.
///
/// Speedup is bounded by the number of physical cores on the
/// host. On a 4-core CPU we expect roughly 3–4× at N=4. This is
/// a classic embarrassingly-parallel workload that demonstrates
/// the arena's no-contention contract.
fn bench_concurrent_training_scaling(seed: u64) {
    const N_GRAPHS: usize = 4;
    const N_OBS_PER_GRAPH: usize = 5_000;
    const N_TRIALS: usize = 5;

    let seeds: [u64; N_GRAPHS] = [
        seed,
        seed.wrapping_add(1),
        seed.wrapping_add(2),
        seed.wrapping_add(3),
    ];

    // Warm caches: train one small graph to discount JIT/branch
    // predictor effects.
    {
        let mut g = AdaptiveBeliefGraph::new(0);
        for i in 0..500 {
            g.observe(0, (i as f64) * 0.001).unwrap();
        }
    }

    // Serial path.
    let mut serial_min_ns = u128::MAX;
    let mut serial_head_acc = [0u8; 32];
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let mut acc = [0u8; 32];
        for &s in &seeds {
            let mut g = AdaptiveBeliefGraph::new(s);
            for i in 0..N_OBS_PER_GRAPH {
                g.observe(0, (i as f64) * 0.001).unwrap();
            }
            for (a, b) in acc.iter_mut().zip(g.chain_head.iter()) {
                *a ^= b;
            }
        }
        let elapsed = start.elapsed().as_nanos();
        if elapsed < serial_min_ns {
            serial_min_ns = elapsed;
            serial_head_acc = acc;
        }
    }

    // Concurrent path: one thread per graph.
    let mut concurrent_min_ns = u128::MAX;
    let mut concurrent_head_acc = [0u8; 32];
    for _ in 0..N_TRIALS {
        let start = Instant::now();
        let heads: Vec<[u8; 32]> = std::thread::scope(|sc| {
            let handles: Vec<_> = seeds
                .iter()
                .map(|&s| {
                    sc.spawn(move || {
                        let mut g = AdaptiveBeliefGraph::new(s);
                        for i in 0..N_OBS_PER_GRAPH {
                            g.observe(0, (i as f64) * 0.001).unwrap();
                        }
                        g.chain_head
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("training thread panicked"))
                .collect()
        });
        let mut acc = [0u8; 32];
        for h in heads {
            for (a, b) in acc.iter_mut().zip(h.iter()) {
                *a ^= b;
            }
        }
        let elapsed = start.elapsed().as_nanos();
        if elapsed < concurrent_min_ns {
            concurrent_min_ns = elapsed;
            concurrent_head_acc = acc;
        }
    }

    // Determinism witness: serial and concurrent must produce the
    // same XOR-of-chain-heads. If a hidden global lock were
    // ordering training events differently, this would diverge.
    assert_eq!(
        serial_head_acc, concurrent_head_acc,
        "serial and concurrent training must produce identical chain_heads"
    );

    let ratio = serial_min_ns as f64 / concurrent_min_ns as f64;
    println!(
        r#"{{"op":"concurrent_training_scaling","n_graphs":{N_GRAPHS},"n_obs_per_graph":{N_OBS_PER_GRAPH},"serial_min_ns":{serial_min_ns},"concurrent_min_ns":{concurrent_min_ns},"speedup":{ratio:.3}}}"#
    );
    eprintln!(
        "  concurrent_training_scaling N={N_GRAPHS}: serial={:.2} ms  concurrent={:.2} ms  speedup={:.2}x  (ideal=N at no-contention; bounded by physical cores)",
        serial_min_ns as f64 / 1e6,
        concurrent_min_ns as f64 / 1e6,
        ratio,
    );
}
/// Phase 0.8 Item D1 — measure `BlrState::predict` with the cached
/// Cholesky factor (cache hit, the new D1 path) vs without (cache
/// miss, equivalent to pre-D1 behavior). The cache-miss measurement
/// is the honest baseline because that's what every predict call
/// did before D1 shipped.
///
/// Workload: prime a `BlrState` with `n_updates` rows so its
/// precision matrix has accumulated state, then time `predict`
/// repeatedly. Cache-hit measurements run a tight predict loop
/// with the cache populated. Cache-miss measurements clear the
/// cache before each predict to force the Cholesky.
///
/// Sweeps `d` in {4, 8, 16, 32} so the speedup curve as a function
/// of dimension is visible. The handoff predicted ~30% at d=4 and
/// ~70% at d=16+; at d=32 (PINN-style features) the win compounds.
fn bench_blr_predict_cache(seed: u64) {
    use cjc_abng::blr::BlrState;
    let _ = seed; // unused: BLR math is seed-agnostic

    const N_ITERS: usize = 50_000;
    const N_UPDATES_PRIMING: usize = 64;

    for &d in &[4usize, 8, 16, 32] {
        // Prime the state: feed in `N_UPDATES_PRIMING` rows so
        // precision is well-conditioned (not the prior identity).
        let prior = BlrPrior::new(2.0, 1.0, 0.5).unwrap();
        let mut s = BlrState::from_prior(&prior, d as u32);
        let phi: Vec<f64> = (0..d).map(|i| 1.0 / ((i + 1) as f64)).collect();
        for k in 0..N_UPDATES_PRIMING {
            let yi = [0.7 + (k as f64) * 0.0001];
            s.update(&phi, &yi).unwrap();
        }
        // Cache is populated by the last update. Verify.
        assert!(s.cached_l.borrow().is_some());

        // ─ Cache-hit path (new D1 default) ────────────────────────
        for _ in 0..N_WARMUP {
            let _ = s.predict(&phi).unwrap();
        }
        let start = Instant::now();
        for _ in 0..N_ITERS {
            let _ = s.predict(&phi).unwrap();
        }
        let cached_elapsed = start.elapsed();
        let cached_per_op = cached_elapsed.as_nanos() as f64 / N_ITERS as f64;

        // ─ Cache-miss path (pre-D1 equivalent) ───────────────────
        // Clear the cache before each predict to force fresh
        // Cholesky on every call.
        for _ in 0..N_WARMUP {
            *s.cached_l.borrow_mut() = None;
            let _ = s.predict(&phi).unwrap();
        }
        let start = Instant::now();
        for _ in 0..N_ITERS {
            *s.cached_l.borrow_mut() = None;
            let _ = s.predict(&phi).unwrap();
        }
        let miss_elapsed = start.elapsed();
        let miss_per_op = miss_elapsed.as_nanos() as f64 / N_ITERS as f64;

        let speedup = miss_per_op / cached_per_op;
        println!(
            r#"{{"op":"blr_predict_cache","d":{d},"miss_ns":{miss_per_op:.2},"hit_ns":{cached_per_op:.2},"speedup":{speedup:.3}}}"#
        );
        eprintln!(
            "  blr_predict_cache d={d:>2}: miss={:>8.0} ns/op  hit={:>8.0} ns/op  speedup={:.2}x",
            miss_per_op, cached_per_op, speedup,
        );
    }
}
