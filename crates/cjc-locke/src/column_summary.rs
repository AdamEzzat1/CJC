//! Per-column confidence summary (v0.6 batch 2).
//!
//! Locke historically emits individual `ValidationFinding`s — one per
//! detector firing — leaving the reader to mentally aggregate them per
//! column. This module synthesises a `ColumnConfidenceSummary` per
//! column from a `LockeReport`, combining:
//!
//! * the column's worst-severity finding,
//! * the list of fired codes (sorted, deduplicated),
//! * a coarse confidence band derived from severity counts,
//! * the count of findings per severity for that column.
//!
//! The output is a structured `Vec<ColumnConfidenceSummary>` plus a
//! deterministic text emit `emit_per_column_confidence_summary(...)` that
//! looks like:
//!
//! ```text
//! Column: country
//!   confidence: Moderate
//!   codes:      E9080, E9082, E9085
//!   findings:   info=0 notice=1 warning=2 error=0
//!   worst:      "column `country` has 1 case-fold collision group(s) ..."
//! ```
//!
//! Determinism: column order is the `BTreeMap` (sorted) iteration order;
//! within a column, codes are sorted ascending; severity counts are
//! integers. Two runs over the same report produce byte-identical text.
//!
//! ## Confidence band heuristic
//!
//! | Worst severity (per column) | Band |
//! |---|---|
//! | Error | `Low` |
//! | Warning | `Moderate` |
//! | Notice | `Moderate` (downgraded to `High` if exactly 1 finding total) |
//! | Info or none | `High` |
//!
//! The heuristic is deliberately coarse — it's a quick triage label, not
//! a calibrated metric. The [`BeliefScore`](crate::belief::BeliefScore)
//! is the right tool for the calibrated answer.

use std::collections::{BTreeMap, BTreeSet};

use crate::report::{FindingSeverity, LockeReport, ValidationFinding};

/// Coarse confidence band for a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfidenceBand {
    High,
    Moderate,
    Low,
}

impl ConfidenceBand {
    pub fn as_str(self) -> &'static str {
        match self {
            ConfidenceBand::High => "High",
            ConfidenceBand::Moderate => "Moderate",
            ConfidenceBand::Low => "Low",
        }
    }
}

/// One per-column synthesis of a `LockeReport`'s findings.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnConfidenceSummary {
    pub column: String,
    pub confidence: ConfidenceBand,
    pub codes: Vec<String>,
    pub n_info: u64,
    pub n_notice: u64,
    pub n_warning: u64,
    pub n_error: u64,
    pub worst_message: Option<String>,
}

impl ColumnConfidenceSummary {
    pub fn total_findings(&self) -> u64 {
        self.n_info + self.n_notice + self.n_warning + self.n_error
    }
}

/// Build one summary per column referenced in the report's findings.
/// Findings without a column (dataset-wide) are aggregated under a
/// synthetic `"<dataset>"` row so they aren't silently dropped.
pub fn build_per_column_summaries(report: &LockeReport) -> Vec<ColumnConfidenceSummary> {
    // BTreeMap<column_name, Vec<&ValidationFinding>>
    let mut by_column: BTreeMap<String, Vec<&ValidationFinding>> = BTreeMap::new();
    for f in &report.findings {
        let key = match &f.column {
            Some(c) => c.clone(),
            None => "<dataset>".to_string(),
        };
        by_column.entry(key).or_default().push(f);
    }
    let mut out = Vec::with_capacity(by_column.len());
    for (column, findings) in by_column {
        let mut codes: BTreeSet<String> = BTreeSet::new();
        let mut n_info = 0u64;
        let mut n_notice = 0u64;
        let mut n_warning = 0u64;
        let mut n_error = 0u64;
        let mut worst_severity = FindingSeverity::Info;
        let mut worst_message: Option<String> = None;
        for f in &findings {
            codes.insert(f.code.to_string());
            match f.severity {
                FindingSeverity::Info => n_info += 1,
                FindingSeverity::Notice => n_notice += 1,
                FindingSeverity::Warning => n_warning += 1,
                FindingSeverity::Error => n_error += 1,
            }
            if f.severity > worst_severity {
                worst_severity = f.severity;
                worst_message = Some(f.message.clone());
            } else if worst_message.is_none() && f.severity == worst_severity {
                worst_message = Some(f.message.clone());
            }
        }
        let total = n_info + n_notice + n_warning + n_error;
        let confidence = match worst_severity {
            FindingSeverity::Error => ConfidenceBand::Low,
            FindingSeverity::Warning => ConfidenceBand::Moderate,
            FindingSeverity::Notice => {
                if total == 1 {
                    ConfidenceBand::High
                } else {
                    ConfidenceBand::Moderate
                }
            }
            FindingSeverity::Info => ConfidenceBand::High,
        };
        let codes_sorted: Vec<String> = codes.into_iter().collect();
        out.push(ColumnConfidenceSummary {
            column,
            confidence,
            codes: codes_sorted,
            n_info,
            n_notice,
            n_warning,
            n_error,
            worst_message,
        });
    }
    out
}

/// Deterministic text emit of per-column summaries. Empty report
/// produces `"# Per-column confidence (empty)\n"`.
pub fn emit_per_column_confidence_summary(report: &LockeReport) -> String {
    let mut s = String::new();
    s.push_str("# Per-column confidence summary\n");
    let summaries = build_per_column_summaries(report);
    if summaries.is_empty() {
        s.push_str("(no per-column findings)\n");
        return s;
    }
    for c in &summaries {
        s.push_str(&format!("Column: {}\n", c.column));
        s.push_str(&format!("  confidence: {}\n", c.confidence.as_str()));
        s.push_str(&format!("  codes:      {}\n", c.codes.join(", ")));
        s.push_str(&format!(
            "  findings:   info={} notice={} warning={} error={}\n",
            c.n_info, c.n_notice, c.n_warning, c.n_error
        ));
        if let Some(m) = &c.worst_message {
            s.push_str(&format!("  worst:      {}\n", m));
        }
    }
    s
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{FindingEvidence, FindingSeverity, LockeInputSummary, LockeReport, ValidationFinding};

    fn finding(code: &'static str, sev: FindingSeverity, column: Option<&str>, msg: &str) -> ValidationFinding {
        ValidationFinding::new(
            code,
            sev,
            msg,
            column.map(|s| s.into()),
            None,
            vec![FindingEvidence::Count { label: "n".into(), value: 1 }],
            100,
            vec![],
            vec![],
        )
    }

    fn empty_report() -> LockeReport {
        LockeReport {
            schema_version: 1,
            run_id: crate::id::FingerprintId(0),
            input: LockeInputSummary {
                dataset_label: "test".into(),
                n_rows: 0,
                n_cols: 0,
                column_types: BTreeMap::new(),
            },
            severity_counts: Default::default(),
            findings: vec![],
            assumptions: vec![],
            column_reports: BTreeMap::new(),
            per_value_lineage: None,
        }
    }

    #[test]
    fn empty_report_yields_empty_summaries() {
        let r = empty_report();
        let s = build_per_column_summaries(&r);
        assert!(s.is_empty());
    }

    #[test]
    fn single_warning_yields_moderate_band() {
        let mut r = empty_report();
        r.findings.push(finding("E9080", FindingSeverity::Warning, Some("country"), "case-fold"));
        let s = build_per_column_summaries(&r);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].column, "country");
        assert_eq!(s[0].confidence, ConfidenceBand::Moderate);
        assert_eq!(s[0].n_warning, 1);
    }

    #[test]
    fn single_notice_yields_high_band() {
        let mut r = empty_report();
        r.findings.push(finding("E9081", FindingSeverity::Notice, Some("country"), "whitespace"));
        let s = build_per_column_summaries(&r);
        assert_eq!(s[0].confidence, ConfidenceBand::High);
    }

    #[test]
    fn two_notices_yield_moderate_band() {
        let mut r = empty_report();
        r.findings.push(finding("E9081", FindingSeverity::Notice, Some("country"), "whitespace"));
        r.findings.push(finding("E9016", FindingSeverity::Notice, Some("country"), "rare"));
        let s = build_per_column_summaries(&r);
        assert_eq!(s[0].confidence, ConfidenceBand::Moderate);
        assert_eq!(s[0].codes, vec!["E9016".to_string(), "E9081".to_string()]);
    }

    #[test]
    fn error_yields_low_band() {
        let mut r = empty_report();
        r.findings.push(finding("E9092", FindingSeverity::Error, Some("ssn"), "SSN"));
        let s = build_per_column_summaries(&r);
        assert_eq!(s[0].confidence, ConfidenceBand::Low);
    }

    #[test]
    fn dataset_wide_finding_attached_to_synthetic_row() {
        let mut r = empty_report();
        r.findings.push(finding("E9003", FindingSeverity::Notice, None, "23 duplicate rows"));
        let s = build_per_column_summaries(&r);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].column, "<dataset>");
    }

    #[test]
    fn multi_column_emit_is_sorted() {
        let mut r = empty_report();
        r.findings.push(finding("E9080", FindingSeverity::Warning, Some("zulu"), "case"));
        r.findings.push(finding("E9080", FindingSeverity::Warning, Some("alpha"), "case"));
        r.findings.push(finding("E9080", FindingSeverity::Warning, Some("mike"), "case"));
        let s = build_per_column_summaries(&r);
        assert_eq!(s[0].column, "alpha");
        assert_eq!(s[1].column, "mike");
        assert_eq!(s[2].column, "zulu");
    }

    #[test]
    fn text_emit_is_deterministic() {
        let mut r = empty_report();
        r.findings.push(finding("E9080", FindingSeverity::Warning, Some("country"), "case"));
        r.findings.push(finding("E9081", FindingSeverity::Notice, Some("country"), "ws"));
        let a = emit_per_column_confidence_summary(&r);
        let b = emit_per_column_confidence_summary(&r);
        assert_eq!(a, b);
        assert!(a.contains("Column: country"));
        assert!(a.contains("codes:      E9080, E9081"));
    }
}
