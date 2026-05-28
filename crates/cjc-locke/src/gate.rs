//! Report-diff and gate semantics (v0.4).
//!
//! `cjcl locke gate <reference.json> <current>` answers: "compared to
//! the snapshotted reference report, what changed?" Three relations:
//!
//! - **Appeared** — a finding in the current report whose `id` is NOT
//!   in the reference.
//! - **Disappeared** — a finding in the reference whose `id` is NOT in
//!   the current.
//! - **Unchanged** — present in both, same id.
//!
//! IDs are content-addressed, so "same id" means "byte-identical
//! evidence + severity + code + column + row_range." A finding whose
//! severity changed gets a *new* id, so it shows up as both
//! disappeared (old severity) and appeared (new severity) — clean.
//!
//! Exit semantics for the gate command:
//! - 0 if no findings appeared at or above the `--fail-on` severity.
//! - 1 if any "appeared" finding meets/exceeds the threshold.
//! - 2 on I/O / parse errors (handled by the CLI).

use std::collections::BTreeMap;

use crate::report::{FindingSeverity, LockeReport, ValidationFinding};

#[derive(Debug, Clone, PartialEq)]
pub struct ReportDiff {
    pub appeared: Vec<ValidationFinding>,
    pub disappeared: Vec<ValidationFinding>,
    pub unchanged: Vec<ValidationFinding>,
    pub ref_run_id: crate::id::FingerprintId,
    pub cur_run_id: crate::id::FingerprintId,
}

impl ReportDiff {
    pub fn is_clean(&self) -> bool {
        self.appeared.is_empty() && self.disappeared.is_empty()
    }

    /// Highest severity among findings that *appeared* in the current
    /// report (i.e. were not in the reference).
    pub fn appeared_worst_severity(&self) -> FindingSeverity {
        self.appeared
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(FindingSeverity::Info)
    }

    /// `true` if any appeared finding has severity >= threshold.
    pub fn gate_fails(&self, threshold: FindingSeverity) -> bool {
        self.appeared.iter().any(|f| f.severity >= threshold)
    }
}

/// Compute the structural diff between `reference` and `current`.
///
/// Findings are matched by content-addressed `id`. The output `Vec`s
/// are sorted by `sort_key()` for deterministic emission.
pub fn diff_reports(reference: &LockeReport, current: &LockeReport) -> ReportDiff {
    let ref_ids: BTreeMap<crate::id::FingerprintId, ValidationFinding> = reference
        .findings
        .iter()
        .map(|f| (f.id, f.clone()))
        .collect();
    let cur_ids: BTreeMap<crate::id::FingerprintId, ValidationFinding> = current
        .findings
        .iter()
        .map(|f| (f.id, f.clone()))
        .collect();

    let mut appeared: Vec<ValidationFinding> = cur_ids
        .iter()
        .filter(|(id, _)| !ref_ids.contains_key(id))
        .map(|(_, f)| f.clone())
        .collect();
    appeared.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    let mut disappeared: Vec<ValidationFinding> = ref_ids
        .iter()
        .filter(|(id, _)| !cur_ids.contains_key(id))
        .map(|(_, f)| f.clone())
        .collect();
    disappeared.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    let mut unchanged: Vec<ValidationFinding> = cur_ids
        .iter()
        .filter(|(id, _)| ref_ids.contains_key(id))
        .map(|(_, f)| f.clone())
        .collect();
    unchanged.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    ReportDiff {
        appeared,
        disappeared,
        unchanged,
        ref_run_id: reference.run_id,
        cur_run_id: current.run_id,
    }
}

/// Render a `ReportDiff` to canonical text (suitable for CLI emit and
/// snapshot testing).
pub fn emit_diff_text(diff: &ReportDiff) -> String {
    let mut s = String::new();
    s.push_str("# Locke Gate Diff\n");
    s.push_str(&format!("reference_run_id: {}\n", diff.ref_run_id));
    s.push_str(&format!("current_run_id:   {}\n", diff.cur_run_id));
    s.push_str(&format!(
        "summary: appeared={} disappeared={} unchanged={}\n",
        diff.appeared.len(),
        diff.disappeared.len(),
        diff.unchanged.len()
    ));
    if !diff.appeared.is_empty() {
        s.push_str("\nappeared:\n");
        for f in &diff.appeared {
            s.push_str(&format!(
                "  + code={} severity={} column={}\n    {}\n",
                f.code,
                f.severity,
                f.column.as_deref().unwrap_or("-"),
                f.message
            ));
        }
    }
    if !diff.disappeared.is_empty() {
        s.push_str("\ndisappeared:\n");
        for f in &diff.disappeared {
            s.push_str(&format!(
                "  - code={} severity={} column={}\n    {}\n",
                f.code,
                f.severity,
                f.column.as_deref().unwrap_or("-"),
                f.message
            ));
        }
    }
    if diff.appeared.is_empty() && diff.disappeared.is_empty() {
        s.push_str("\nno changes — current report is structurally identical to the reference.\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{validate, ValidateOptions};
    use cjc_data::{Column, DataFrame};

    fn df_clean() -> DataFrame {
        DataFrame::from_columns(vec![
            ("x".into(), Column::Float((0..100).map(|i| i as f64).collect())),
        ])
        .unwrap()
    }

    fn df_with_nans() -> DataFrame {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        for i in 0..50 {
            v[i] = f64::NAN;
        }
        DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap()
    }

    #[test]
    fn diff_of_identical_reports_is_clean() {
        let opts = ValidateOptions {
            dataset_label: "test".into(),
            ..Default::default()
        };
        let r = validate(&df_clean(), &opts);
        let diff = diff_reports(&r, &r);
        assert!(diff.is_clean());
        assert_eq!(diff.unchanged.len(), r.findings.len());
    }

    #[test]
    fn diff_detects_appeared_findings() {
        let opts = ValidateOptions {
            dataset_label: "test".into(),
            ..Default::default()
        };
        let clean = validate(&df_clean(), &opts);
        let nan = validate(&df_with_nans(), &opts);
        let diff = diff_reports(&clean, &nan);
        // 50% NaN should produce an E9001 Error that was not in the clean report.
        assert!(diff
            .appeared
            .iter()
            .any(|f| f.code == "E9001" && f.severity == FindingSeverity::Error));
    }

    #[test]
    fn gate_fails_when_appeared_severity_meets_threshold() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let diff = diff_reports(&validate(&df_clean(), &opts), &validate(&df_with_nans(), &opts));
        assert!(diff.gate_fails(FindingSeverity::Warning));
        assert!(diff.gate_fails(FindingSeverity::Error));
    }

    #[test]
    fn diff_is_deterministic() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let r1 = validate(&df_with_nans(), &opts);
        let r2 = validate(&df_clean(), &opts);
        let d1 = diff_reports(&r1, &r2);
        let d2 = diff_reports(&r1, &r2);
        assert_eq!(d1, d2);
    }

    #[test]
    fn diff_text_is_canonical() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let d = diff_reports(&validate(&df_clean(), &opts), &validate(&df_with_nans(), &opts));
        let s1 = emit_diff_text(&d);
        let s2 = emit_diff_text(&d);
        assert_eq!(s1, s2);
        assert!(s1.contains("appeared:"));
    }
}
