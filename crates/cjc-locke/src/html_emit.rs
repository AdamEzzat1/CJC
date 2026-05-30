//! HTML report emitter (v0.4 + v0.5 correlation matrix).
//!
//! Produces a single, self-contained HTML file: inline CSS, inline SVG
//! sparklines, no external assets, no JS. Opens offline in any browser
//! and is byte-identical to a single-shot rerun (because the underlying
//! `LockeReport` is).
//!
//! ## Design
//!
//! - **No template engine.** Hand-written `String` concatenation with
//!   explicit HTML escaping for all user-supplied strings (dataset
//!   labels, column names, messages). Matches the project's zero-dep
//!   discipline.
//! - **Severity colors** are CSS variables in `:root`, so a stylesheet
//!   override (a single line) can rebrand the report.
//! - **Inline SVG histograms** for numeric columns, built from
//!   `stats::equal_width_histogram`. No JS, no Plotly.

use crate::report::{FindingSeverity, LockeReport, ValidationFinding};
use cjc_data::{Column, DataFrame};

/// Emit a self-contained HTML report. The correlation-matrix section
/// is omitted (no DataFrame in scope); use [`emit_locke_report_html_with_df`]
/// to include it.
pub fn emit_locke_report_html(report: &LockeReport) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str("<!DOCTYPE html>\n");
    s.push_str("<html lang=\"en\">\n<head>\n");
    s.push_str("<meta charset=\"utf-8\">\n");
    s.push_str("<title>Locke Validation Report — ");
    push_escaped(&mut s, &report.input.dataset_label);
    s.push_str("</title>\n");
    push_inline_styles(&mut s);
    s.push_str("</head>\n<body>\n");

    push_header(&mut s, report);
    push_summary(&mut s, report);
    push_findings_table(&mut s, &report.findings);
    push_assumptions(&mut s, &report.assumptions);
    push_footer(&mut s, report);

    s.push_str("</body>\n</html>\n");
    s
}

/// v0.5: like `emit_locke_report_html` but additionally renders a
/// pairwise Pearson-correlation heatmap for the report's DataFrame.
pub fn emit_locke_report_html_with_df(report: &LockeReport, df: &DataFrame) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str("<!DOCTYPE html>\n");
    s.push_str("<html lang=\"en\">\n<head>\n");
    s.push_str("<meta charset=\"utf-8\">\n");
    s.push_str("<title>Locke Validation Report — ");
    push_escaped(&mut s, &report.input.dataset_label);
    s.push_str("</title>\n");
    push_inline_styles(&mut s);
    s.push_str("</head>\n<body>\n");

    push_header(&mut s, report);
    push_summary(&mut s, report);
    push_findings_table(&mut s, &report.findings);
    push_correlation_matrix(&mut s, df);
    push_assumptions(&mut s, &report.assumptions);
    push_footer(&mut s, report);

    s.push_str("</body>\n</html>\n");
    s
}

/// Build and render a pairwise Pearson correlation heatmap.
fn push_correlation_matrix(s: &mut String, df: &DataFrame) {
    // Collect numeric columns.
    let mut numeric: Vec<(&String, Vec<f64>)> = Vec::new();
    for (name, col) in &df.columns {
        let v: Option<Vec<f64>> = match col {
            Column::Float(v) => Some(v.clone()),
            Column::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
            _ => None,
        };
        if let Some(v) = v {
            numeric.push((name, v));
        }
    }
    if numeric.len() < 3 {
        return; // Pointless for 0/1/2 columns.
    }
    // Cap at 30 cols for legibility; if more, drop the lowest-variance ones.
    if numeric.len() > 30 {
        // Compute variance per column.
        let mut with_var: Vec<(&String, Vec<f64>, f64)> = numeric
            .into_iter()
            .map(|(n, v)| {
                let var = crate::stats::summarize_f64(&v).variance.unwrap_or(0.0);
                (n, v, var)
            })
            .collect();
        with_var.sort_by(|a, b| b.2.total_cmp(&a.2));
        with_var.truncate(30);
        numeric = with_var.into_iter().map(|(n, v, _)| (n, v)).collect();
    }
    let n = numeric.len();
    let cell = 22;
    let header = 120;
    let width = header + n * cell;
    let height = header + n * cell;

    s.push_str("<h2>Cross-column correlation</h2>\n");
    s.push_str("<p style=\"color: var(--muted); margin-top: -.5em;\">");
    s.push_str(&format!(
        "Pairwise Pearson correlation between numeric columns ({} shown). \
         Cells are colored by |r| — darker red = stronger correlation.",
        n
    ));
    s.push_str("</p>\n");
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" style=\"font-family: sans-serif; font-size: 11px;\">\n",
        width, height
    ));
    // Column labels (top, rotated -45deg).
    for (i, (name, _)) in numeric.iter().enumerate() {
        let x = header + i * cell + cell / 2;
        let y = header - 5;
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" transform=\"rotate(-45, {}, {})\" text-anchor=\"start\">",
            x, y, x, y
        ));
        push_escaped(s, name);
        s.push_str("</text>\n");
    }
    // Row labels (left).
    for (j, (name, _)) in numeric.iter().enumerate() {
        let x = header - 5;
        let y = header + j * cell + cell / 2 + 4;
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"end\">",
            x, y
        ));
        push_escaped(s, name);
        s.push_str("</text>\n");
    }
    // v0.7+ deep-dive perf-fix: Pearson is symmetric (r(i,j) = r(j,i))
    // but the previous code computed every (i,j) cell, including the
    // lower triangle — half the work was redundant. For a 30-column ×
    // 30K-row dataset (diabetes-130 magnitude), that's ~450 wasted
    // Pearson computations per HTML emit, each ~30K Kahan adds.
    //
    // Compute the upper triangle once into a flat Vec indexed by i*n+j,
    // then look up symmetrically when rendering. Pearson at (i,i) is 1.0
    // by definition (skip the computation).
    let mut r_cache: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        r_cache[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let r = crate::stats::pearson_correlation(&numeric[i].1, &numeric[j].1)
                .unwrap_or(0.0);
            r_cache[i * n + j] = r;
            r_cache[j * n + i] = r; // symmetric
        }
    }
    // Cells.
    for i in 0..n {
        for j in 0..n {
            let r = r_cache[i * n + j];
            let abs_r = r.abs();
            // Color: white at |r|=0, deep red at |r|=1. Use sign hue.
            let (hue, sat, lightness) = if r >= 0.0 {
                (0u32, (abs_r * 100.0) as u32, (100.0 - abs_r * 50.0) as u32)
            } else {
                (220u32, (abs_r * 100.0) as u32, (100.0 - abs_r * 50.0) as u32)
            };
            let x = header + i * cell;
            let y = header + j * cell;
            s.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"hsl({}, {}%, {}%)\" stroke=\"#eee\" stroke-width=\"0.5\">",
                x, y, cell, cell, hue, sat, lightness
            ));
            // v0.7+ deep-dive bug-fix: previously column names were
            // emitted into the <title> tooltip without HTML-escaping.
            // A DataFrame column named e.g. "<script>alert(1)</script>"
            // would land literal markup inside SVG (low actual XSS risk
            // because <title> in SVG is tooltip-only in browsers, but it
            // violates the push_escaped contract documented elsewhere
            // in this file).
            s.push_str("<title>");
            push_escaped(s, &numeric[i].0);
            s.push_str(" vs ");
            push_escaped(s, &numeric[j].0);
            s.push_str(&format!(": r = {:.3}</title>", r));
            s.push_str("</rect>\n");
            // r-value text for high |r|.
            if abs_r > 0.5 {
                let text_color = if abs_r > 0.7 { "#fff" } else { "#000" };
                s.push_str(&format!(
                    "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"{}\" font-size=\"9\">{:.2}</text>\n",
                    x + cell / 2,
                    y + cell / 2 + 3,
                    text_color,
                    r
                ));
            }
        }
    }
    s.push_str("</svg>\n");
}

fn push_inline_styles(s: &mut String) {
    s.push_str("<style>\n");
    s.push_str(":root {\n");
    s.push_str("  --sev-info: #6c757d;\n");
    s.push_str("  --sev-notice: #0d6efd;\n");
    s.push_str("  --sev-warning: #ffc107;\n");
    s.push_str("  --sev-error: #dc3545;\n");
    s.push_str("  --bg: #ffffff;\n");
    s.push_str("  --fg: #212529;\n");
    s.push_str("  --muted: #6c757d;\n");
    s.push_str("  --border: #dee2e6;\n");
    s.push_str("  --card-bg: #f8f9fa;\n");
    s.push_str("}\n");
    s.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; ");
    s.push_str("max-width: 1100px; margin: 2em auto; padding: 0 1em; color: var(--fg); background: var(--bg); }\n");
    s.push_str("h1, h2 { border-bottom: 1px solid var(--border); padding-bottom: .3em; }\n");
    s.push_str("h1 { margin-top: 0; }\n");
    s.push_str(".meta { color: var(--muted); font-size: .9em; }\n");
    s.push_str(".meta code { background: var(--card-bg); padding: 1px 4px; border-radius: 3px; }\n");
    s.push_str(".summary { display: flex; gap: 1em; margin: 1.5em 0; }\n");
    s.push_str(".summary .card { flex: 1; padding: 1em; background: var(--card-bg); border-radius: 6px; ");
    s.push_str("border-left: 4px solid var(--border); }\n");
    s.push_str(".summary .card.info { border-left-color: var(--sev-info); }\n");
    s.push_str(".summary .card.notice { border-left-color: var(--sev-notice); }\n");
    s.push_str(".summary .card.warning { border-left-color: var(--sev-warning); }\n");
    s.push_str(".summary .card.error { border-left-color: var(--sev-error); }\n");
    s.push_str(".summary .card .count { font-size: 2em; font-weight: 600; }\n");
    s.push_str(".summary .card .label { color: var(--muted); text-transform: uppercase; ");
    s.push_str("font-size: .8em; letter-spacing: .05em; }\n");
    s.push_str("table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: .92em; }\n");
    s.push_str("th, td { padding: .6em .8em; text-align: left; border-bottom: 1px solid var(--border); ");
    s.push_str("vertical-align: top; }\n");
    s.push_str("th { background: var(--card-bg); font-weight: 600; }\n");
    s.push_str(".sev { display: inline-block; padding: 2px 8px; border-radius: 12px; ");
    s.push_str("font-size: .8em; font-weight: 600; color: white; }\n");
    s.push_str(".sev.info { background: var(--sev-info); }\n");
    s.push_str(".sev.notice { background: var(--sev-notice); }\n");
    s.push_str(".sev.warning { background: var(--sev-warning); color: #212529; }\n");
    s.push_str(".sev.error { background: var(--sev-error); }\n");
    s.push_str(".code { font-family: 'SF Mono', Consolas, monospace; font-size: .85em; ");
    s.push_str("background: var(--card-bg); padding: 1px 6px; border-radius: 3px; }\n");
    s.push_str(".col { font-family: 'SF Mono', Consolas, monospace; color: var(--muted); }\n");
    s.push_str(".assumptions { background: var(--card-bg); padding: 1em 1.5em; ");
    s.push_str("border-radius: 6px; margin: 1em 0; }\n");
    s.push_str(".assumptions li { color: var(--muted); margin: .4em 0; }\n");
    s.push_str(".footer { color: var(--muted); font-size: .8em; margin-top: 3em; ");
    s.push_str("padding-top: 1em; border-top: 1px solid var(--border); }\n");
    s.push_str(".no-findings { text-align: center; padding: 2em; background: var(--card-bg); ");
    s.push_str("border-radius: 6px; color: var(--muted); }\n");
    s.push_str("</style>\n");
}

fn push_header(s: &mut String, report: &LockeReport) {
    s.push_str("<h1>Locke Validation Report</h1>\n");
    s.push_str("<div class=\"meta\">\n");
    s.push_str("<p>Dataset: <code>");
    push_escaped(s, &report.input.dataset_label);
    s.push_str("</code></p>\n");
    s.push_str(&format!(
        "<p>Schema version: <code>{}</code> &mdash; Run ID: <code>{}</code></p>\n",
        report.schema_version, report.run_id
    ));
    s.push_str(&format!(
        "<p>Shape: <code>{} rows &times; {} columns</code></p>\n",
        report.input.n_rows, report.input.n_cols
    ));
    s.push_str("</div>\n");
}

fn push_summary(s: &mut String, report: &LockeReport) {
    s.push_str("<h2>Summary</h2>\n");
    s.push_str("<div class=\"summary\">\n");
    let c = &report.severity_counts;
    push_card(s, "info", "Info", c.info);
    push_card(s, "notice", "Notice", c.notice);
    push_card(s, "warning", "Warning", c.warning);
    push_card(s, "error", "Error", c.error);
    s.push_str("</div>\n");
}

fn push_card(s: &mut String, kind: &str, label: &str, count: u64) {
    s.push_str(&format!(
        "<div class=\"card {}\"><div class=\"count\">{}</div><div class=\"label\">{}</div></div>\n",
        kind, count, label
    ));
}

fn push_findings_table(s: &mut String, findings: &[ValidationFinding]) {
    s.push_str("<h2>Findings</h2>\n");
    if findings.is_empty() {
        s.push_str("<div class=\"no-findings\">No findings — this dataset passed every check.</div>\n");
        return;
    }
    s.push_str("<table>\n<thead>\n<tr>\n");
    s.push_str("<th>Severity</th><th>Code</th><th>Column</th><th>Message</th><th>Evidence</th>\n");
    s.push_str("</tr>\n</thead>\n<tbody>\n");
    for f in findings {
        s.push_str("<tr>\n");
        let sev_class = match f.severity {
            FindingSeverity::Info => "info",
            FindingSeverity::Notice => "notice",
            FindingSeverity::Warning => "warning",
            FindingSeverity::Error => "error",
        };
        s.push_str(&format!(
            "<td><span class=\"sev {}\">{}</span></td>\n",
            sev_class, f.severity
        ));
        s.push_str(&format!("<td><span class=\"code\">{}</span></td>\n", f.code));
        s.push_str("<td><span class=\"col\">");
        push_escaped(s, f.column.as_deref().unwrap_or("&mdash;"));
        s.push_str("</span></td>\n");
        s.push_str("<td>");
        push_escaped(s, &f.message);
        s.push_str("</td>\n");
        s.push_str("<td>");
        push_evidence_list(s, &f.evidence);
        s.push_str("</td>\n");
        s.push_str("</tr>\n");
    }
    s.push_str("</tbody>\n</table>\n");
}

fn push_evidence_list(s: &mut String, evidence: &[crate::report::FindingEvidence]) {
    if evidence.is_empty() {
        s.push_str("&mdash;");
        return;
    }
    s.push_str("<ul style=\"margin: 0; padding-left: 1.2em;\">\n");
    for e in evidence {
        s.push_str("<li>");
        use crate::report::FindingEvidence::*;
        match e {
            Count { label, value } => {
                push_escaped(s, label);
                s.push_str(&format!(" = <code>{}</code>", value));
            }
            Ratio { label, value } => {
                push_escaped(s, label);
                s.push_str(&format!(" = <code>{:.4}</code>", value));
            }
            Range { label, min, max } => {
                push_escaped(s, label);
                s.push_str(&format!(" = <code>[{:.4}, {:.4}]</code>", min, max));
            }
            Metric { label, value } => {
                push_escaped(s, label);
                s.push_str(&format!(" = <code>{:.6}</code>", value));
            }
            Sample { label, value } => {
                push_escaped(s, label);
                s.push_str(" = <code>");
                push_escaped(s, value);
                s.push_str("</code>");
            }
        }
        s.push_str("</li>\n");
    }
    s.push_str("</ul>\n");
}

fn push_assumptions(s: &mut String, assumptions: &[String]) {
    if assumptions.is_empty() {
        return;
    }
    s.push_str("<h2>Assumptions</h2>\n");
    s.push_str("<div class=\"assumptions\">\n<ul>\n");
    for a in assumptions {
        s.push_str("<li>");
        push_escaped(s, a);
        s.push_str("</li>\n");
    }
    s.push_str("</ul>\n</div>\n");
}

fn push_footer(s: &mut String, report: &LockeReport) {
    s.push_str("<div class=\"footer\">\n");
    s.push_str(&format!(
        "Generated by Locke v{} &mdash; deterministic, content-addressed. ",
        env!("CARGO_PKG_VERSION")
    ));
    s.push_str(&format!(
        "Re-running over the same data will produce a report with the same run_id (<code>{}</code>).\n",
        report.run_id
    ));
    s.push_str("</div>\n");
}

/// HTML-escape `<`, `>`, `&`, `"`. Sufficient for text content + attribute
/// values; we use double-quoted attributes throughout.
fn push_escaped(s: &mut String, raw: &str) {
    for c in raw.chars() {
        match c {
            '<' => s.push_str("&lt;"),
            '>' => s.push_str("&gt;"),
            '&' => s.push_str("&amp;"),
            '"' => s.push_str("&quot;"),
            c => s.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{validate, ValidateOptions};
    use cjc_data::{Column, DataFrame};

    fn fixture_report() -> LockeReport {
        let df = DataFrame::from_columns(vec![
            ("age".into(), Column::Float(vec![25.0, 30.0, f64::NAN, 30.0, 45.0])),
            ("score".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ])
        .unwrap();
        let opts = ValidateOptions {
            dataset_label: "html-test".into(),
            ..Default::default()
        };
        validate(&df, &opts)
    }

    #[test]
    fn html_emit_is_deterministic_across_runs() {
        let r = fixture_report();
        let a = emit_locke_report_html(&r);
        let b = emit_locke_report_html(&r);
        assert_eq!(a, b);
    }

    #[test]
    fn html_emit_contains_dataset_label() {
        let r = fixture_report();
        let h = emit_locke_report_html(&r);
        assert!(h.contains("html-test"));
    }

    #[test]
    fn html_emit_contains_severity_classes() {
        let r = fixture_report();
        let h = emit_locke_report_html(&r);
        assert!(h.contains("class=\"sev"));
    }

    #[test]
    fn html_emit_escapes_special_characters() {
        use crate::report::{FindingEvidence, ValidationFinding};
        use std::collections::BTreeMap;
        let f = ValidationFinding::new(
            "E9001",
            FindingSeverity::Warning,
            "value < 0 or > max",
            Some("x".into()),
            None,
            vec![FindingEvidence::Sample {
                label: "sample".into(),
                value: "row1=\"foo\" & bar".into(),
            }],
            10,
            vec![],
            vec![],
        );
        let r = crate::report::LockeReport::new(
            crate::report::LockeInputSummary::default(),
            vec![f],
            BTreeMap::new(),
            vec![],
        );
        let h = emit_locke_report_html(&r);
        assert!(h.contains("&lt;"));
        assert!(h.contains("&gt;"));
        assert!(h.contains("&amp;"));
        assert!(h.contains("&quot;"));
    }

    #[test]
    fn correlation_matrix_only_rendered_with_df_overload() {
        let r = fixture_report();
        let h_plain = emit_locke_report_html(&r);
        assert!(!h_plain.contains("Cross-column correlation"));

        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Float((0..50).map(|i| i as f64).collect())),
            ("b".into(), Column::Float((0..50).map(|i| (i as f64) * 2.0).collect())),
            ("c".into(), Column::Float((0..50).map(|i| (i as f64).sin()).collect())),
        ])
        .unwrap();
        let h_with_df = emit_locke_report_html_with_df(&r, &df);
        assert!(h_with_df.contains("Cross-column correlation"));
        // SVG cell colors should appear.
        assert!(h_with_df.contains("hsl("));
    }

    #[test]
    fn correlation_matrix_skipped_for_fewer_than_three_numeric_cols() {
        let r = fixture_report();
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Float((0..50).map(|i| i as f64).collect())),
            ("b".into(), Column::Float((0..50).map(|i| (i as f64) * 2.0).collect())),
        ])
        .unwrap();
        let h = emit_locke_report_html_with_df(&r, &df);
        assert!(!h.contains("Cross-column correlation"));
    }

    #[test]
    fn correlation_matrix_is_deterministic() {
        let r = fixture_report();
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Float((0..50).map(|i| i as f64).collect())),
            ("b".into(), Column::Float((0..50).map(|i| (i as f64) * 2.0).collect())),
            ("c".into(), Column::Float((0..50).map(|i| (i as f64).sin()).collect())),
        ])
        .unwrap();
        let a = emit_locke_report_html_with_df(&r, &df);
        let b = emit_locke_report_html_with_df(&r, &df);
        assert_eq!(a, b);
    }

    #[test]
    fn html_emit_handles_empty_report() {
        let input = crate::report::LockeInputSummary::default();
        let r = crate::report::LockeReport::new(
            input,
            vec![],
            std::collections::BTreeMap::new(),
            vec![],
        );
        let h = emit_locke_report_html(&r);
        assert!(h.contains("No findings"));
    }
}
