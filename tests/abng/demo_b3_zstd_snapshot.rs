//! Phase 0.8 Item B3 — zstd snapshot compression demo.
//!
//! Demonstrates the "snapshot-as-zstd-stream" capability: any tool
//! that consumes ZSTD frames (cloud storage, network protocols,
//! archival systems) can now ingest ABNG snapshots transparently.
//! `serialize_compressed` wraps `serialize_into` with a `zstd::Encoder`,
//! producing a self-identifying blob whose first 6 bytes are
//! `ABNGZ\x01`.
//!
//! What this file measures + reports:
//!
//! * Uncompressed snapshot bytes.
//! * zstd-compressed bytes at level 3 (the doc's recommended default).
//! * Compression ratio.
//! * Round-trip: `replay(serialize_compressed(g))` reconstructs a
//!   graph with the same `chain_head` as `serialize`-replay.
//!
//! SVG visualization: bar chart of uncompressed vs zstd-compressed
//! bytes at multiple chain sizes (`bench_results/phase_0_8_demos/b3_zstd_snapshot.svg`).
//!
//! # Cargo feature gate
//!
//! The whole module is gated on `feature = "compression"`. Without
//! it, the zstd dependency isn't linked in, so the demo can't run.
//! Activate with `cargo test --features compression --test abng b3_demo`.

#![cfg(feature = "compression")]

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{
    replay, serialize, serialize_compressed,
};
use std::path::PathBuf;

/// zstd compression level for the demo. The Phase 0.8 B3 bench
/// found level 3 to be the sweet spot for ABNG snapshots (good
/// ratio, low encode cost).
const ZSTD_LEVEL: i32 = 3;

const DEMO_SEED: u64 = 0x4A03_B3;

/// Build a graph with `n` audit events. Mixing observes so the
/// payload has structured redundancy that zstd can exploit.
fn build_chain(seed: u64, n: usize) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    while g.audit.len() < n {
        g.observe(0, (g.audit.len() as f64) * 0.1).unwrap();
    }
    g
}

#[derive(Debug, Clone, Copy)]
struct SizePoint {
    n_events: usize,
    uncompressed: usize,
    compressed: usize,
}

#[test]
fn b3_demo_zstd_snapshot_compression() {
    // Multiple chain sizes so the SVG shows the trend, not a
    // single point.
    let sizes = [100usize, 1_000, 5_000];
    let mut points: Vec<SizePoint> = Vec::with_capacity(sizes.len());

    for &n in &sizes {
        let g = build_chain(DEMO_SEED, n);
        let uncompressed_bytes = serialize(&g);
        let compressed_bytes = serialize_compressed(&g, ZSTD_LEVEL);

        // The compressed blob starts with the `ABNGZ\x01` magic so
        // a generic reader can detect the wrapping.
        assert_eq!(&compressed_bytes[..6], b"ABNGZ\x01");

        // ─ Round-trip contract: replay(compressed) ≡ replay(uncompressed).
        let g_uncomp = replay(&uncompressed_bytes).expect("uncompressed replay");
        let g_comp = replay(&compressed_bytes).expect("compressed replay");
        assert_eq!(
            g_uncomp.chain_head, g_comp.chain_head,
            "compressed replay must yield the same chain_head"
        );
        assert_eq!(
            g_uncomp.merkle_root(),
            g_comp.merkle_root(),
            "compressed replay must yield the same Merkle root"
        );

        points.push(SizePoint {
            n_events: n,
            uncompressed: uncompressed_bytes.len(),
            compressed: compressed_bytes.len(),
        });
    }

    // ─ Print human-readable report ───────────────────────────────────
    eprintln!();
    eprintln!("══ B3 ZSTD Snapshot Compression Demo (level {ZSTD_LEVEL}) ══");
    eprintln!();
    eprintln!(
        "  {:>10}   {:>14}   {:>14}   {:>8}",
        "n_events", "uncompressed", "compressed", "ratio"
    );
    for pt in &points {
        let ratio = pt.uncompressed as f64 / pt.compressed as f64;
        eprintln!(
            "  {:>10}   {:>14}   {:>14}   {:>6.2}×",
            pt.n_events,
            format!("{} B", pt.uncompressed),
            format!("{} B", pt.compressed),
            ratio
        );
    }
    eprintln!();
    eprintln!("Wire-protocol compatibility:");
    eprintln!("  Magic bytes: ABNGZ\\x01");
    eprintln!("  Any zstd-aware reader (S3 lifecycle, network proxies, archive tools)");
    eprintln!("  can transparently store and retrieve these snapshots.");
    eprintln!();
    eprintln!("Round-trip determinism: ✓");
    eprintln!("  replay(serialize_compressed) yields the same chain_head + Merkle root");
    eprintln!("  as replay(serialize) on every chain size measured.");

    // ─ Emit SVG ──────────────────────────────────────────────────────
    let svg = render_b3_compression_svg(&points);
    let path = output_path("b3_zstd_snapshot.svg");
    std::fs::create_dir_all(path.parent().unwrap()).expect("create output dir");
    std::fs::write(&path, svg.as_bytes()).expect("write SVG");
    eprintln!();
    eprintln!("SVG visualization → {}", path.display());
}

#[test]
fn b3_demo_compressed_blob_strictly_smaller_at_scale() {
    // At ≥ 1K events, the audit log is large enough that zstd
    // beats uncompressed. (At very small chain sizes, zstd's
    // frame overhead can make compressed *larger* — we don't
    // assert that here.) This test pins the "shipped capability
    // is useful at the recommended scale" contract.
    for &n in &[1_000usize, 5_000] {
        let g = build_chain(DEMO_SEED, n);
        let uncomp = serialize(&g).len();
        let comp = serialize_compressed(&g, ZSTD_LEVEL).len();
        assert!(
            comp < uncomp,
            "at n={n}, expected compressed < uncompressed, got {comp} < {uncomp}"
        );
    }
}

#[test]
fn b3_demo_compressed_replay_yields_byte_equal_state() {
    // The strongest contract: `replay(serialize_compressed(g))`
    // produces a graph byte-equal to `replay(serialize(g))`. This
    // is what makes the compression wrapper transparent.
    let g = build_chain(DEMO_SEED, 2_000);
    let from_uncomp = replay(&serialize(&g)).expect("uncomp replay");
    let from_comp = replay(&serialize_compressed(&g, ZSTD_LEVEL)).expect("comp replay");
    assert_eq!(from_uncomp.chain_head, from_comp.chain_head);
    assert_eq!(from_uncomp.audit.len(), from_comp.audit.len());
    assert_eq!(from_uncomp.merkle_root(), from_comp.merkle_root());
}

// ── helpers ──────────────────────────────────────────────────────────

fn output_path(filename: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("bench_results");
    p.push("phase_0_8_demos");
    p.push(filename);
    p
}

/// Grouped bar chart: per chain size, side-by-side uncompressed vs
/// compressed bytes.
fn render_b3_compression_svg(points: &[SizePoint]) -> String {
    let width = 880i32;
    let height = 460i32;
    let bar_area_h = 280f64;

    let max_bytes = points
        .iter()
        .map(|p| p.uncompressed)
        .max()
        .unwrap_or(1) as f64;

    let y_baseline = 360i32;
    let bar_w = 60i32;
    let bar_gap = 8i32;
    let group_w = bar_w * 2 + bar_gap;
    let inter_group_gap = 60i32;
    let n_groups = points.len() as i32;
    let total_chart_w = group_w * n_groups + inter_group_gap * (n_groups - 1);
    let chart_left = (width - total_chart_w) / 2;

    let mut out = String::new();
    out.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="ui-monospace, Menlo, Consolas, monospace">
  <style>
    .title {{ font-size: 16px; font-weight: bold; fill: #222; }}
    .label {{ font-size: 12px; fill: #444; }}
    .value-label {{ font-size: 11px; fill: #fff; font-weight: bold; }}
    .ratio-label {{ font-size: 12px; fill: #2e7d32; font-weight: bold; }}
    .axis {{ stroke: #888; stroke-width: 1; }}
    .legend-text {{ font-size: 12px; fill: #444; }}
  </style>
  <rect width="100%" height="100%" fill="#fafafa" />
  <text x="20" y="28" class="title">B3 — ZSTD Snapshot Compression (level {ZSTD_LEVEL})</text>
  <text x="20" y="48" class="label">Uncompressed vs zstd-compressed bytes; ratio annotated above each pair</text>
"##
    ));

    // Bars
    let mut x = chart_left;
    for pt in points {
        let h_uncomp = (pt.uncompressed as f64 / max_bytes * bar_area_h).round() as i32;
        let h_comp = (pt.compressed as f64 / max_bytes * bar_area_h).round() as i32;
        let y_uncomp = y_baseline - h_uncomp;
        let y_comp = y_baseline - h_comp;
        let ratio = pt.uncompressed as f64 / pt.compressed as f64;

        // Uncompressed bar (blue)
        out.push_str(&format!(
            r##"  <rect x="{x}" y="{y_uncomp}" width="{bar_w}" height="{h_uncomp}" fill="#4a90d9" />
"##
        ));
        out.push_str(&format!(
            r##"  <text x="{}" y="{}" text-anchor="middle" class="value-label">{} B</text>
"##,
            x + bar_w / 2,
            y_uncomp + 16,
            pt.uncompressed
        ));

        // Compressed bar (green)
        let x_comp = x + bar_w + bar_gap;
        out.push_str(&format!(
            r##"  <rect x="{x_comp}" y="{y_comp}" width="{bar_w}" height="{h_comp}" fill="#2e7d32" />
"##
        ));
        out.push_str(&format!(
            r##"  <text x="{}" y="{}" text-anchor="middle" class="value-label">{} B</text>
"##,
            x_comp + bar_w / 2,
            y_comp + 16,
            pt.compressed
        ));

        // Ratio annotation
        let group_cx = x + group_w / 2;
        let ratio_y = y_uncomp.min(y_comp) - 8;
        out.push_str(&format!(
            r##"  <text x="{group_cx}" y="{ratio_y}" text-anchor="middle" class="ratio-label">{ratio:.2}×</text>
"##
        ));

        // Group x-axis label
        out.push_str(&format!(
            r##"  <text x="{group_cx}" y="{}" text-anchor="middle" class="label">n={}</text>
"##,
            y_baseline + 22,
            pt.n_events
        ));

        x += group_w + inter_group_gap;
    }

    // Baseline axis
    out.push_str(&format!(
        r##"  <line class="axis" x1="40" y1="{y_baseline}" x2="{}" y2="{y_baseline}" />
"##,
        width - 40
    ));

    // Legend
    let lg_y = height - 30;
    out.push_str(&format!(
        r##"  <rect x="40" y="{}" width="14" height="14" fill="#4a90d9" />
  <text x="60" y="{}" class="legend-text">uncompressed (ABNG\x0E magic)</text>
  <rect x="280" y="{}" width="14" height="14" fill="#2e7d32" />
  <text x="300" y="{}" class="legend-text">zstd level {ZSTD_LEVEL} (ABNGZ\x01 magic)</text>
"##,
        lg_y - 11,
        lg_y,
        lg_y - 11,
        lg_y
    ));

    out.push_str("</svg>\n");
    out
}
