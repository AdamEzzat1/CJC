//! ABNG Micro-Benchmark Suite (Phase 0.6 Item 2)
//! ===============================================
//!
//! Per-op cost measurement for the ABNG hot path. Establishes baseline
//! numbers BEFORE Item 3 (smart-replay fast-forward) and Item 4 (native
//! batch_observe + bulk BLR update) ship, so subsequent perf wins are
//! verifiable with cited deltas.
//!
//! Operations measured:
//!   - `observe`         — single Welford observation + chain-hash append
//!   - `blr_update`      — NIG conjugate update (n=1, d=4)
//!   - `blr_predict`     — Cholesky triangular solve
//!   - `encode_prefix`   — quantile-codebook encode
//!   - `descend`         — radix-tree route from prefix
//!
//! Outputs JSONL to stdout for CI ingestion. Prints a human scorecard
//! to stderr. Manual `Instant` timing matches the existing
//! `bench/ad_bench/` convention (no Criterion dependency added).
//!
//! Invocation:
//!     cargo run -p abng-micro --release > abng_micro.jsonl

use cjc_abng::graph::AdaptiveBeliefGraph;
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

    eprintln!("=== Done ===");
    eprintln!("Phase 0.6 Item 2 baseline. Items 3+4 perf wins compare against these numbers.");
}
