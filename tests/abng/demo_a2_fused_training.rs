//! Phase 0.8c v14 Item A2 — fused TrainStep audit-log compactness demo.
//!
//! Demonstrates the per-row training compactness win: replacing the
//! pre-A2 3-call sequence (`blr_update + observe`) with a single
//! `train_step` call collapses the per-row audit footprint from two
//! chain events (`BlrUpdated` + `BeliefUpdate`) into one (`TrainStep`,
//! tag `0x1E`).
//!
//! What this file measures + reports:
//!
//! * Audit-event count per row, before vs after.
//! * Audit-log payload bytes per row, before vs after.
//! * Total payload bytes for a 100-row training session.
//! * Cross-confirms that the post-call BLR state + Welford stats
//!   are **byte-identical** between the two paths (the v14 design
//!   contract: same numerical state, fewer chain steps).
//!
//! SVG visualization: side-by-side bar chart of audit-event count
//! and payload bytes (`bench_results/phase_0_8_demos/a2_fused_training_compactness.svg`).

use cjc_abng::audit::AuditKind;
use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_ad::pinn::Activation;
use std::path::PathBuf;

/// Number of training rows in the demo. 100 is enough to amortize
/// the fixed setup events (Codebook, LeafHead, BlrPrior, etc.) and
/// have the per-row delta dominate the totals.
const N_TRAIN_ROWS: usize = 100;

/// Demo seed pinned for byte-stable reports.
const DEMO_SEED: u64 = 0x4A02;

/// Build a graph with codebook + leaf head + BLR prior + 4 child
/// leaves, matching the layout used by the `train_step_v14_tests`
/// suite so the comparison is apples-to-apples.
fn build_setup_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    for byte in 0..4u8 {
        g.add_node(0, byte).unwrap();
    }
    g
}

/// Deterministic per-row training inputs. Same set used by the
/// fused and unfused runs so the comparison isolates the
/// pre-A2-vs-post-A2 wire shape, not the data.
fn training_row(i: usize) -> (f64, [f64; 4], f64) {
    let i = i as f64;
    let x = ((i * 0.0173).sin() + 1.0) * 0.5; // routing in [0, 1]
    let phi = [
        ((i * 0.07).sin() * 0.9),
        ((i * 0.11).cos() * 0.5),
        ((i * 0.13).sin() * 0.3),
        ((i * 0.17).cos() * 0.2),
    ];
    let y = ((i * 0.029).sin() * 1.2);
    (x, phi, y)
}

/// Payload-bytes-per-event sum: how big is each chain event's
/// canonical wire payload? Mirrors what serialize.rs writes per
/// event to the audit-log section.
fn total_payload_bytes(g: &AdaptiveBeliefGraph) -> usize {
    let mut buf: Vec<u8> = Vec::with_capacity(96);
    let mut total = 0usize;
    for ev in &g.audit {
        buf.clear();
        ev.write_payload(&mut buf);
        total += buf.len();
    }
    total
}

#[derive(Debug, Clone)]
struct RunSummary {
    label: &'static str,
    audit_events: usize,
    payload_bytes: usize,
    train_step_events: usize,
    blr_updated_events: usize,
    belief_update_events: usize,
}

fn summarize(g: &AdaptiveBeliefGraph, label: &'static str) -> RunSummary {
    let mut train_step = 0;
    let mut blr_updated = 0;
    let mut belief_update = 0;
    for ev in &g.audit {
        match ev.kind {
            AuditKind::TrainStep { .. } => train_step += 1,
            AuditKind::BlrUpdated { .. } => blr_updated += 1,
            AuditKind::BeliefUpdate { .. } => belief_update += 1,
            _ => {}
        }
    }
    RunSummary {
        label,
        audit_events: g.audit.len(),
        payload_bytes: total_payload_bytes(g),
        train_step_events: train_step,
        blr_updated_events: blr_updated,
        belief_update_events: belief_update,
    }
}

#[test]
fn a2_demo_fused_vs_unfused_compactness() {
    // ─ UNFUSED path: pre-A2 3-call sequence per row ──────────────────
    let mut g_unfused = build_setup_graph(DEMO_SEED);
    for i in 0..N_TRAIN_ROWS {
        let (x, phi, y) = training_row(i);
        let prefix = g_unfused.encode_prefix(&[x]).unwrap();
        let leaf = g_unfused.descend(&prefix).leaf_id;
        g_unfused.blr_update(leaf, &phi, &[y]).unwrap();
        g_unfused.observe(leaf, y).unwrap();
    }
    let unfused = summarize(&g_unfused, "pre-A2 (BlrUpdated + BeliefUpdate)");

    // ─ FUSED path: v14 train_step per row ────────────────────────────
    let mut g_fused = build_setup_graph(DEMO_SEED);
    for i in 0..N_TRAIN_ROWS {
        let (x, phi, y) = training_row(i);
        g_fused.train_step(&[x], &phi, y).unwrap();
    }
    let fused = summarize(&g_fused, "v14 A2 (TrainStep)");

    // ─ Audit-event count contract ────────────────────────────────────
    assert_eq!(
        fused.train_step_events, N_TRAIN_ROWS,
        "every fused row emits exactly one TrainStep event"
    );
    assert_eq!(
        unfused.blr_updated_events, N_TRAIN_ROWS,
        "every unfused row emits one BlrUpdated event"
    );
    assert_eq!(
        unfused.belief_update_events, N_TRAIN_ROWS,
        "every unfused row emits one BeliefUpdate event"
    );
    assert_eq!(
        unfused.audit_events - fused.audit_events,
        N_TRAIN_ROWS,
        "fused path saves exactly one chain event per row"
    );

    // ─ Audit-log payload bytes contract ──────────────────────────────
    assert!(
        fused.payload_bytes < unfused.payload_bytes,
        "fused payload bytes must be smaller: fused={}, unfused={}",
        fused.payload_bytes,
        unfused.payload_bytes
    );
    let saved_bytes = unfused.payload_bytes - fused.payload_bytes;
    let saved_pct = (saved_bytes as f64) / (unfused.payload_bytes as f64) * 100.0;

    // ─ State byte-equality contract (the A2 invariant) ───────────────
    // The whole point of A2: the per-leaf BLR + Welford state must
    // be byte-identical between the two paths. Only the chain
    // shape differs.
    for leaf in 1..=4u32 {
        let stats_unfused = g_unfused.nodes[leaf as usize].stats.canonical_bytes();
        let stats_fused = g_fused.nodes[leaf as usize].stats.canonical_bytes();
        assert_eq!(
            stats_unfused, stats_fused,
            "leaf {leaf}: Welford stats must be byte-identical between fused and unfused"
        );
        let blr_unfused = g_unfused.nodes[leaf as usize]
            .blr
            .as_ref()
            .map(|b| b.state_hash());
        let blr_fused = g_fused.nodes[leaf as usize]
            .blr
            .as_ref()
            .map(|b| b.state_hash());
        assert_eq!(
            blr_unfused, blr_fused,
            "leaf {leaf}: BLR state_hash must be byte-identical between fused and unfused"
        );
    }

    // ─ Chain heads differ (by design) ────────────────────────────────
    assert_ne!(
        g_unfused.chain_head, g_fused.chain_head,
        "chain heads MUST differ — A2's whole point is replacing two chain steps with one"
    );

    // ─ Print a human-readable report ─────────────────────────────────
    eprintln!();
    eprintln!("══ A2 Fused TrainStep Compactness Demo ══");
    eprintln!("Training rows: {N_TRAIN_ROWS}");
    eprintln!();
    eprintln!(
        "  {:>14}  {:>14}  {:>14}",
        "(metric)", "pre-A2", "v14 A2"
    );
    eprintln!(
        "  {:>14}  {:>14}  {:>14}",
        "audit events", unfused.audit_events, fused.audit_events
    );
    eprintln!(
        "  {:>14}  {:>14}  {:>14}",
        "payload bytes", unfused.payload_bytes, fused.payload_bytes
    );
    eprintln!();
    eprintln!(
        "Savings: {saved_bytes} bytes ({saved_pct:.1}%); {} chain events skipped (one per row)",
        N_TRAIN_ROWS
    );
    eprintln!();
    eprintln!("State byte-equality: ✓ Welford + BLR identical on all 4 leaves");
    eprintln!("Chain head divergence: ✓ (by design — different audit shapes)");

    // ─ Emit SVG visualization ────────────────────────────────────────
    let svg = render_a2_bars_svg(&unfused, &fused, N_TRAIN_ROWS, saved_pct);
    let path = output_path("a2_fused_training_compactness.svg");
    std::fs::create_dir_all(path.parent().unwrap()).expect("create output dir");
    std::fs::write(&path, svg.as_bytes()).expect("write SVG");
    eprintln!();
    eprintln!("SVG visualization → {}", path.display());

    // Determinism gate.
    let svg2 = render_a2_bars_svg(&unfused, &fused, N_TRAIN_ROWS, saved_pct);
    assert_eq!(svg, svg2, "SVG rendering must be byte-stable");
}

#[test]
fn a2_demo_savings_scale_linearly_in_n() {
    // The per-row delta is constant: every row saves exactly one
    // chain event and the same payload-byte delta. So the total
    // savings scale linearly with N. Pin this as a separate test
    // so a future regression that broke the per-row contract
    // wouldn't sneak through the headline demo.
    for &n in &[10usize, 50, 250] {
        let mut g_unfused = build_setup_graph(DEMO_SEED);
        let mut g_fused = build_setup_graph(DEMO_SEED);
        for i in 0..n {
            let (x, phi, y) = training_row(i);
            // Unfused
            let prefix = g_unfused.encode_prefix(&[x]).unwrap();
            let leaf = g_unfused.descend(&prefix).leaf_id;
            g_unfused.blr_update(leaf, &phi, &[y]).unwrap();
            g_unfused.observe(leaf, y).unwrap();
            // Fused
            g_fused.train_step(&[x], &phi, y).unwrap();
        }
        assert_eq!(
            g_unfused.audit.len() - g_fused.audit.len(),
            n,
            "savings at n={n} must be exactly n chain events"
        );
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

/// Render the audit-event + payload-bytes comparison as a
/// side-by-side bar chart SVG. Two metric pairs:
///   * Audit events (left)
///   * Payload bytes (right)
fn render_a2_bars_svg(
    unfused: &RunSummary,
    fused: &RunSummary,
    n_rows: usize,
    saved_pct: f64,
) -> String {
    let width = 880i32;
    let height = 460i32;

    // Compute bar heights from data. Bar area is 280 px tall.
    let bar_area_h = 280f64;
    let events_max = unfused.audit_events.max(fused.audit_events) as f64;
    let bytes_max = unfused.payload_bytes.max(fused.payload_bytes) as f64;
    let h_unfused_events = (unfused.audit_events as f64) / events_max * bar_area_h;
    let h_fused_events = (fused.audit_events as f64) / events_max * bar_area_h;
    let h_unfused_bytes = (unfused.payload_bytes as f64) / bytes_max * bar_area_h;
    let h_fused_bytes = (fused.payload_bytes as f64) / bytes_max * bar_area_h;

    let bar_w = 80i32;
    let bar_gap = 12i32;
    let group_gap = 80i32;
    let y_baseline = 360i32;

    let left_unfused_x = 110;
    let left_fused_x = left_unfused_x + bar_w + bar_gap;
    let right_unfused_x = left_fused_x + bar_w + group_gap;
    let right_fused_x = right_unfused_x + bar_w + bar_gap;

    let mut out = String::new();
    out.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="ui-monospace, Menlo, Consolas, monospace">
  <style>
    .title {{ font-size: 16px; font-weight: bold; fill: #222; }}
    .group-label {{ font-size: 13px; font-weight: bold; fill: #222; }}
    .bar-label {{ font-size: 12px; fill: #444; }}
    .value-label {{ font-size: 11px; fill: #fff; font-weight: bold; }}
    .axis {{ stroke: #888; stroke-width: 1; }}
    .savings {{ font-size: 14px; fill: #2e7d32; font-weight: bold; }}
  </style>
  <rect width="100%" height="100%" fill="#fafafa" />
  <text x="20" y="28" class="title">A2 — Fused TrainStep vs pre-A2 (BlrUpdated + BeliefUpdate)</text>
  <text x="20" y="48" class="bar-label">N = {n_rows} training rows · pre-A2 vs v14 A2</text>
"##
    ));

    // Baseline axis
    out.push_str(&format!(
        r##"  <line class="axis" x1="40" y1="{y_baseline}" x2="{}" y2="{y_baseline}" />
"##,
        width - 40
    ));

    // Bars + labels
    let bar = |x: i32, h: f64, color: &str, value_str: &str| -> String {
        let h_int = h.round() as i32;
        let y = y_baseline - h_int;
        let mut s = String::new();
        s.push_str(&format!(
            r##"  <rect x="{x}" y="{y}" width="{bar_w}" height="{h_int}" fill="{color}" />
"##
        ));
        // Value label inside top of bar
        let lbl_y = y + 16;
        let lbl_x = x + bar_w / 2;
        s.push_str(&format!(
            r##"  <text x="{lbl_x}" y="{lbl_y}" text-anchor="middle" class="value-label">{value_str}</text>
"##
        ));
        s
    };

    out.push_str(&bar(
        left_unfused_x,
        h_unfused_events,
        "#4a90d9",
        &format!("{}", unfused.audit_events),
    ));
    out.push_str(&bar(
        left_fused_x,
        h_fused_events,
        "#e94f37",
        &format!("{}", fused.audit_events),
    ));
    out.push_str(&bar(
        right_unfused_x,
        h_unfused_bytes,
        "#4a90d9",
        &format!("{}", unfused.payload_bytes),
    ));
    out.push_str(&bar(
        right_fused_x,
        h_fused_bytes,
        "#e94f37",
        &format!("{}", fused.payload_bytes),
    ));

    // Group labels
    let events_cx = (left_unfused_x + left_fused_x) / 2 + bar_w / 2;
    let bytes_cx = (right_unfused_x + right_fused_x) / 2 + bar_w / 2;
    out.push_str(&format!(
        r##"  <text x="{events_cx}" y="{}" text-anchor="middle" class="group-label">audit events</text>
"##,
        y_baseline + 24
    ));
    out.push_str(&format!(
        r##"  <text x="{bytes_cx}" y="{}" text-anchor="middle" class="group-label">payload bytes</text>
"##,
        y_baseline + 24
    ));

    // Per-bar labels
    let bar_label_y = y_baseline + 42;
    out.push_str(&format!(
        r##"  <text x="{}" y="{bar_label_y}" text-anchor="middle" class="bar-label">pre-A2</text>
"##,
        left_unfused_x + bar_w / 2
    ));
    out.push_str(&format!(
        r##"  <text x="{}" y="{bar_label_y}" text-anchor="middle" class="bar-label">v14 A2</text>
"##,
        left_fused_x + bar_w / 2
    ));
    out.push_str(&format!(
        r##"  <text x="{}" y="{bar_label_y}" text-anchor="middle" class="bar-label">pre-A2</text>
"##,
        right_unfused_x + bar_w / 2
    ));
    out.push_str(&format!(
        r##"  <text x="{}" y="{bar_label_y}" text-anchor="middle" class="bar-label">v14 A2</text>
"##,
        right_fused_x + bar_w / 2
    ));

    // Savings annotation
    out.push_str(&format!(
        r##"  <text x="{}" y="{}" text-anchor="middle" class="savings">{:.1}% audit-log payload reduction at N={n_rows}</text>
"##,
        width / 2,
        height - 30,
        saved_pct
    ));

    out.push_str("</svg>\n");
    out
}
