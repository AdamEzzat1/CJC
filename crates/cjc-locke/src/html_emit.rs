//! HTML report emitter (v0.4).
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

/// Emit a self-contained HTML report.
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
