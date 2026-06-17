//! Deterministic renderers. All output is a pure function of the report, with
//! fixed key order and no wall-clock — so the JSON sidecar is itself
//! content-addressable and diffable in CI.

use crate::analyze::{pct_milli, FlameNode};
use crate::report::PolytraceReport;

// ─── tiny deterministic JSON helpers (no serde dependency) ──────────────────

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn flame_json(n: &FlameNode, out: &mut String) {
    out.push_str(&format!(
        "{{\"label\":\"{}\",\"self\":{},\"total\":{},\"children\":[",
        esc(&n.label),
        n.self_count,
        n.total_count
    ));
    for (i, c) in n.children.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        flame_json(c, out);
    }
    out.push_str("]}");
}

/// Render the report as a deterministic JSON string. Keys are in a fixed order;
/// `content_hash` is embedded so a reader can verify reproducibility.
pub fn json(r: &PolytraceReport) -> String {
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"content_hash\":\"{:016x}\",", r.content_hash()));

    // flamegraph
    s.push_str(&format!(
        "\"flamegraph\":{{\"total_samples\":{},\"root\":",
        r.flamegraph.total_samples
    ));
    flame_json(&r.flamegraph.root, &mut s);
    s.push_str("},");

    // boundary
    let bpct = pct_milli(r.boundary.boundary_samples, r.boundary.total_samples);
    s.push_str(&format!(
        "\"boundary\":{{\"samples\":{},\"self\":{},\"crossings\":{},\"share_milli\":{}}},",
        r.boundary.boundary_samples, r.boundary.boundary_self, r.boundary.crossings, bpct
    ));

    // copy
    s.push_str(&format!(
        "\"copy\":{{\"total_bytes\":{},\"avoidable_bytes\":{},\"flows\":[",
        r.copy.total_bytes, r.copy.avoidable_bytes
    ));
    for (i, f) in r.copy.flows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"from\":\"{}\",\"to\":\"{}\",\"bytes\":{},\"count\":{},\"avoidable\":{}}}",
            esc(&f.from), esc(&f.to), f.bytes, f.count, f.avoidable
        ));
    }
    s.push_str("]},");

    // contention
    s.push_str(&format!(
        "\"contention\":{{\"running\":{},\"gil_wait\":{},\"lock_wait\":{},\"channel_wait\":{},\"io_wait\":{},\"async_idle\":{},\"total\":{}}},",
        r.contention.running, r.contention.gil_wait, r.contention.lock_wait,
        r.contention.channel_wait, r.contention.io_wait, r.contention.async_idle, r.contention.total
    ));

    // async
    s.push_str(&format!(
        "\"async\":{{\"tasks\":{},\"resumes\":{},\"wakeups\":{},\"idle_ticks\":{},\"per_task\":[",
        r.async_stall.tasks.len(),
        r.async_stall.total_resumes,
        r.async_stall.total_wakeups,
        r.async_stall.total_stall_ticks
    ));
    for (i, (label, st)) in r.async_stall.tasks.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"task\":\"{}\",\"resumes\":{},\"max_await_ticks\":{},\"idle_ticks\":{}}}",
            esc(label),
            st.resumes,
            st.max_wait_ticks,
            st.stall_ticks
        ));
    }
    s.push_str("]},");

    // ownership
    s.push_str("\"ownership\":{\"total_allocated\":");
    s.push_str(&r.ownership.total_allocated.to_string());
    s.push_str(",\"per_domain_alloc\":{");
    for (i, (k, v)) in r.ownership.per_domain_alloc.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{}\":{}", esc(k), v));
    }
    s.push_str("}},");

    // peak
    s.push_str(&format!(
        "\"peak\":{{\"bytes\":{},\"seq\":{},\"narrative\":\"{}\"}},",
        r.peak.peak_bytes,
        r.peak.peak_seq,
        esc(&r.peak.narrative)
    ));

    // pipeline
    s.push_str("\"pipeline\":{");
    for (i, (k, v)) in r.pipeline.per_stage.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{}\":{}", esc(k), v));
    }
    s.push_str("},");

    // thermal
    s.push_str(&format!(
        "\"thermal\":{{\"counters\":{},\"throttle\":{},\"oversubscription\":{},\"baseline_mhz\":{},\"min_mhz\":{}}},",
        r.thermal.counters_available, r.thermal.throttle_detected, r.thermal.oversubscription,
        r.thermal.baseline_mhz, r.thermal.min_mhz
    ));

    // recommendations
    s.push_str("\"recommendations\":[");
    for (i, rec) in r.recommendations.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"code\":\"{}\",\"title\":\"{}\",\"evidence\":\"{}\"}}",
            esc(&rec.code),
            esc(&rec.title),
            esc(&rec.evidence)
        ));
    }
    s.push(']');

    s.push('}');
    s
}

/// Render a human-readable CLI text report.
pub fn text(r: &PolytraceReport) -> String {
    let mut s = String::new();
    s.push_str("═══ Polytrace report ═══════════════════════════════════════════\n");
    s.push_str(&format!(
        "  content_hash : {:016x}\n",
        r.content_hash()
    ));
    s.push_str(&format!(
        "  samples      : {} (advisory wall {} ns)\n",
        r.flamegraph.total_samples, r.wall_ns_total
    ));

    let bpct = pct_milli(r.boundary.boundary_samples, r.boundary.total_samples);
    let crossings = if r.boundary.crossings > 0 {
        format!(" ({} crossings)", r.boundary.crossings)
    } else {
        String::new()
    };
    s.push_str(&format!(
        "  Py↔Rust seam : {}.{:03}%{}\n",
        bpct / 1000,
        bpct % 1000,
        crossings
    ));
    s.push_str(&format!(
        "  copies       : {} bytes total, {} avoidable\n",
        r.copy.total_bytes, r.copy.avoidable_bytes
    ));
    let gpct = pct_milli(r.contention.gil_wait, r.contention.total);
    let lpct = pct_milli(r.contention.rust_blocked(), r.contention.total);
    s.push_str(&format!(
        "  contention   : GIL {}.{:03}%, Rust-lock {}.{:03}%\n",
        gpct / 1000,
        gpct % 1000,
        lpct / 1000,
        lpct % 1000
    ));

    if !r.async_stall.tasks.is_empty() {
        s.push_str(&format!(
            "  async        : {} tasks, {} resumes, {} wakeups, {} idle ticks\n",
            r.async_stall.tasks.len(),
            r.async_stall.total_resumes,
            r.async_stall.total_wakeups,
            r.async_stall.total_stall_ticks
        ));
        // top tasks by longest single await stall
        let mut tasks: Vec<(&String, &crate::analyze::AsyncTaskStat)> =
            r.async_stall.tasks.iter().collect();
        tasks.sort_by(|a, b| {
            b.1.max_wait_ticks
                .cmp(&a.1.max_wait_ticks)
                .then_with(|| a.0.cmp(b.0))
        });
        for (label, st) in tasks.iter().take(6) {
            s.push_str(&format!(
                "      {:<40} resumes={} max-await={} ticks\n",
                label, st.resumes, st.max_wait_ticks
            ));
        }
    }

    s.push_str(&format!("  memory peak  : {}\n", r.peak.narrative));

    if !r.pipeline.per_stage.is_empty() {
        s.push_str("  pipeline     :\n");
        for (stage, n) in &r.pipeline.per_stage {
            let p = pct_milli(*n, r.pipeline.total_samples);
            s.push_str(&format!("      {:<16} {}.{:03}%\n", stage, p / 1000, p % 1000));
        }
    }

    s.push_str("  recommendations:\n");
    if r.recommendations.is_empty() {
        s.push_str("      (none — no rule predicate fired)\n");
    } else {
        for rec in &r.recommendations {
            s.push_str(&format!("      [{}] {}\n", rec.code, rec.title));
            s.push_str(&format!("          → {}\n", rec.evidence));
        }
    }
    s.push_str("═════════════════════════════════════════════════════════════\n");
    s
}

/// Render the flamegraph as a minimal, dependency-free SVG (icicle layout).
/// Width is proportional to inclusive sample count; depth is stack depth.
pub fn flamegraph_svg(r: &PolytraceReport) -> String {
    const W: u64 = 1000;
    const ROW_H: u64 = 18;
    let total = r.flamegraph.total_samples.max(1);

    let mut rows: Vec<String> = Vec::new();
    fn emit(
        node: &FlameNode,
        x: u64,
        depth: u64,
        total: u64,
        rows: &mut Vec<String>,
    ) {
        // skip the synthetic root rectangle, but recurse its children
        if depth > 0 {
            let w = (node.total_count as u128 * W as u128 / total as u128) as u64;
            let color = match node.kind_tag {
                0 => "#4e79a7", // py
                1 => "#f28e2b", // rust
                2 => "#76b7b2", // native
                3 => "#e15759", // ffi boundary (red — the seam)
                4 => "#59a14f", // async
                _ => "#bab0ac",
            };
            rows.push(format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"#fff\" stroke-width=\"0.5\"><title>{} ({} samples)</title></rect>",
                x,
                depth * ROW_H,
                w.max(1),
                ROW_H - 1,
                color,
                xml_esc(&node.label),
                node.total_count
            ));
        }
        let mut cx = x;
        for c in &node.children {
            emit(c, cx, depth + 1, total, rows);
            let cw = (c.total_count as u128 * W as u128 / total as u128) as u64;
            cx += cw.max(1);
        }
    }
    emit(&r.flamegraph.root, 0, 0, total, &mut rows);

    let depth = max_depth(&r.flamegraph.root, 0);
    let height = (depth + 1) * ROW_H;
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {W} {height}\" width=\"{W}\" height=\"{height}\">{}</svg>",
        rows.join("")
    )
}

fn max_depth(n: &FlameNode, d: u64) -> u64 {
    n.children.iter().map(|c| max_depth(c, d + 1)).max().unwrap_or(d)
}

fn xml_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
