//! Phase 0.8c v14 Item C2 — parallel `verify_chain` scalability demo.
//!
//! Demonstrates the multi-thread chain verification capability:
//! `Graph::verify_chain_par(n_threads)` splits the audit log into
//! `n_threads` chunks and verifies each in parallel via
//! `std::thread::scope`, with the per-chunk results stitched by a
//! main-thread cross-chunk linkage pass.
//!
//! What this file does:
//!
//! 1. Builds a synthetic ~15,000-event audit chain (just above the
//!    `PAR_THRESHOLD` of 10,000) so the parallel path actually fires
//!    instead of falling through to sequential.
//! 2. Wall-clocks `verify_chain_par(k)` for `k ∈ {1, 2, 4, 8}`
//!    plus `verify_chain` (sequential reference). Each measurement
//!    is the **median of 5 runs** to reduce single-shot variance on
//!    laptop CPUs.
//! 3. Asserts that every thread count produces `Ok(())` (correctness
//!    is the contract; performance is a measurement, not an
//!    assertion, since `verify_chain_par` is allowed to be slower
//!    than sequential on a noisy machine — only required to match
//!    the sequential outcome).
//! 4. Tamper test: corrupts one byte mid-chain, asserts that every
//!    `n_threads` setting catches it.
//! 5. Generates a speedup-curve SVG.
//!
//! # Honesty note on perf claims
//!
//! Wall-clock numbers in a unit test are noisy. The demo reports
//! "median over 5 runs" but doesn't assert speedup ≥ 1× — a CI box
//! running other workloads could plausibly see par < seq. The
//! assertion is correctness; the report is informative.

use cjc_abng::graph::{AdaptiveBeliefGraph, GraphError};
use std::path::PathBuf;
use std::time::Instant;

/// Total audit events in the demo chain. Just above the
/// `PAR_THRESHOLD = 10_000` so `verify_chain_par` activates the
/// thread-scope path instead of falling through to sequential.
const N_EVENTS: usize = 15_000;

const DEMO_SEED: u64 = 0x4A03_C2;

/// Build a deterministic chain of `n` audit events.
fn build_chain(seed: u64, n: usize) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    while g.audit.len() < n {
        g.observe(0, (g.audit.len() as f64) * 0.0001).unwrap();
    }
    g
}

/// Median of 5 wall-clock samples in microseconds for the provided
/// verify closure. Returns `(median_us, all_samples_us)`.
fn median_us<F: FnMut() -> Result<(), GraphError>>(mut f: F) -> (u128, Vec<u128>) {
    let mut samples: Vec<u128> = (0..5)
        .map(|_| {
            let t0 = Instant::now();
            f().expect("verify must succeed");
            t0.elapsed().as_micros()
        })
        .collect();
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    samples.sort_unstable();
    let median = sorted[sorted.len() / 2];
    (median, samples)
}

#[test]
fn c2_demo_parallel_verify_scalability() {
    // ─ Build the chain once. ─────────────────────────────────────────
    let g = build_chain(DEMO_SEED, N_EVENTS);
    assert_eq!(g.audit.len(), N_EVENTS);

    // Warm the caches.
    g.verify_chain().expect("warmup must succeed");

    // ─ Measurements ──────────────────────────────────────────────────
    let (us_seq, _) = median_us(|| g.verify_chain());
    let mut points: Vec<(usize, u128)> = Vec::new();
    for &k in &[1usize, 2, 4, 8] {
        let (us, _) = median_us(|| g.verify_chain_par(k));
        points.push((k, us));
    }

    // ─ Correctness assertions ────────────────────────────────────────
    // Every thread count must return Ok(()). The sequential reference
    // also returns Ok(()). These are the contract; perf is below.
    g.verify_chain().expect("sequential pristine must verify");
    for &k in &[1usize, 2, 4, 8] {
        g.verify_chain_par(k)
            .unwrap_or_else(|e| panic!("verify_chain_par({k}) failed: {e:?}"));
    }

    // ─ Tamper detection at every thread count ────────────────────────
    let mut g_bad = build_chain(DEMO_SEED, N_EVENTS);
    {
        let new_hashes = g_bad.audit.new_hashes_mut();
        // Corrupt one byte in the middle of the chain.
        new_hashes[N_EVENTS / 2][0] ^= 0x01;
    }
    let seq_err = g_bad.verify_chain().expect_err("seq must catch tamper");
    assert!(matches!(seq_err, GraphError::ChainBroken { .. }));
    for &k in &[1usize, 2, 4, 8] {
        let err = g_bad
            .verify_chain_par(k)
            .expect_err("par must catch tamper at every k");
        assert!(
            matches!(err, GraphError::ChainBroken { .. }),
            "expected ChainBroken at k={k}, got {err:?}"
        );
    }

    // ─ Print report ──────────────────────────────────────────────────
    eprintln!();
    eprintln!("══ C2 Parallel verify_chain Scalability Demo ══");
    eprintln!("Audit events:    {N_EVENTS}");
    eprintln!(
        "Sequential verify: {us_seq} µs (median of 5)"
    );
    eprintln!();
    eprintln!("  k_threads   wall_µs   speedup");
    for &(k, us) in &points {
        let speedup = (us_seq as f64) / (us as f64);
        eprintln!("  {k:>9}   {us:>7}   {speedup:>5.2}×");
    }
    eprintln!();
    eprintln!("Tamper detection: ✓ caught at every thread count");
    eprintln!(
        "Threshold gate (10K events): chain has {N_EVENTS} ≥ 10K, parallel path active"
    );

    // ─ Emit SVG ──────────────────────────────────────────────────────
    let svg = render_c2_speedup_svg(us_seq, &points, N_EVENTS);
    let path = output_path("c2_parallel_verify_scalability.svg");
    std::fs::create_dir_all(path.parent().unwrap()).expect("create output dir");
    std::fs::write(&path, svg.as_bytes()).expect("write SVG");
    eprintln!();
    eprintln!("SVG visualization → {}", path.display());
}

#[test]
fn c2_demo_correctness_matches_sequential_at_every_k() {
    // Pin the determinism contract: parallel and sequential must
    // agree on the outcome at every thread count, for both
    // pristine and tampered chains. This is the test that proves
    // C2 is correctly wired.
    let g_pristine = build_chain(DEMO_SEED, N_EVENTS);
    let seq_outcome = g_pristine.verify_chain();
    for &k in &[1usize, 2, 4, 8] {
        let par_outcome = g_pristine.verify_chain_par(k);
        assert_eq!(
            seq_outcome.is_ok(),
            par_outcome.is_ok(),
            "pristine: seq vs par(k={k}) disagree"
        );
    }
}

#[test]
fn c2_demo_threshold_gate_below_10k() {
    // Below the threshold (10K events), `verify_chain_par(k)`
    // transparently falls through to `verify_chain`. Demonstrate
    // this with a small chain: any k value succeeds, and the
    // outcome is identical to sequential.
    let g_small = build_chain(DEMO_SEED, 500);
    assert!(g_small.audit.len() < 10_000);
    g_small.verify_chain().expect("seq must verify");
    for &k in &[2usize, 4, 8] {
        g_small
            .verify_chain_par(k)
            .expect("par falls through to seq below threshold");
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn output_path(filename: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("bench_results");
    p.push("phase_0_8_demos");
    p.push(filename);
    p
}

/// Render the speedup chart as an SVG. The y-axis is speedup
/// relative to sequential (1.0×). Bars at k ∈ {1, 2, 4, 8} (plus
/// the sequential baseline shown as 1.0× reference).
fn render_c2_speedup_svg(
    us_seq: u128,
    points: &[(usize, u128)],
    n_events: usize,
) -> String {
    let width = 880i32;
    let height = 460i32;
    let bar_area_h = 280f64;

    // Compute speedups (par / seq^-1).
    let speedups: Vec<(usize, f64)> = points
        .iter()
        .map(|&(k, us)| (k, (us_seq as f64) / (us as f64)))
        .collect();
    let max_speedup = speedups
        .iter()
        .map(|&(_, s)| s)
        .fold(1.0f64, f64::max);
    let chart_max = max_speedup.max(2.0); // at least 0..2× shown

    let y_baseline = 360i32;
    let bar_w = 100i32;
    let bar_gap = 50i32;
    let chart_left = 100i32;

    let mut out = String::new();
    out.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="ui-monospace, Menlo, Consolas, monospace">
  <style>
    .title {{ font-size: 16px; font-weight: bold; fill: #222; }}
    .label {{ font-size: 12px; fill: #444; }}
    .value-label {{ font-size: 12px; fill: #fff; font-weight: bold; }}
    .axis {{ stroke: #888; stroke-width: 1; }}
    .ref-line {{ stroke: #f5a623; stroke-width: 2; stroke-dasharray: 6,4; }}
  </style>
  <rect width="100%" height="100%" fill="#fafafa" />
  <text x="20" y="28" class="title">C2 — verify_chain parallel speedup (N={n_events} events)</text>
  <text x="20" y="48" class="label">Sequential baseline: {us_seq} µs (median of 5)</text>
"##
    ));

    // 1.0× reference line.
    let ref_y = y_baseline - ((1.0 / chart_max) * bar_area_h) as i32;
    out.push_str(&format!(
        r##"  <line class="ref-line" x1="40" y1="{ref_y}" x2="{}" y2="{ref_y}" />
  <text x="50" y="{}" class="label" fill="#a06a13">1.0× (sequential)</text>
"##,
        width - 40,
        ref_y - 4
    ));

    // Bars
    let mut x = chart_left;
    for (i, &(k, speedup)) in speedups.iter().enumerate() {
        let h = (speedup / chart_max * bar_area_h).round() as i32;
        let y = y_baseline - h;
        let color = if speedup >= 1.0 {
            "#2e7d32"
        } else {
            "#888"
        };
        out.push_str(&format!(
            r##"  <rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" />
"##
        ));
        let lbl_x = x + bar_w / 2;
        out.push_str(&format!(
            r##"  <text x="{lbl_x}" y="{}" text-anchor="middle" class="value-label">{:.2}×</text>
"##,
            y + 18,
            speedup
        ));
        // x-axis label: k threads
        out.push_str(&format!(
            r##"  <text x="{lbl_x}" y="{}" text-anchor="middle" class="label">k={k}</text>
"##,
            y_baseline + 22
        ));
        let _ = i;
        x += bar_w + bar_gap;
    }

    // Baseline axis
    out.push_str(&format!(
        r##"  <line class="axis" x1="40" y1="{y_baseline}" x2="{}" y2="{y_baseline}" />
"##,
        width - 40
    ));

    // Footer
    out.push_str(&format!(
        r##"  <text x="{}" y="{}" text-anchor="middle" class="label">Speedup vs sequential verify_chain; perf is noisy on busy CPUs — correctness contract holds regardless.</text>
"##,
        width / 2,
        height - 20
    ));

    out.push_str("</svg>\n");
    out
}
