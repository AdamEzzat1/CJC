//! Core report types used by every Locke subsystem.
//!
//! A `LockeReport` is the top-level container. It bundles validation
//! findings, drift findings, causal warnings, lineage references, and a
//! belief-score summary. Inside, every `ValidationFinding` carries:
//!
//! * a deterministic `FindingId` derived from its content,
//! * a `FindingSeverity` (Info / Notice / Warning / Error),
//! * structured `FindingEvidence` (counts, ratios, ranges, samples),
//! * an explicit list of `assumptions` and `suggested_next_checks`.
//!
//! All collections use `BTreeMap` or sorted `Vec` so the JSON-style
//! emit is byte-identical across repeated runs.

use std::collections::BTreeMap;
use std::fmt;

use crate::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};

/// Stable ordinal severity. Ordering is meaningful — `Info < Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FindingSeverity {
    Info,
    Notice,
    Warning,
    Error,
}

impl FindingSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            FindingSeverity::Info => "info",
            FindingSeverity::Notice => "notice",
            FindingSeverity::Warning => "warning",
            FindingSeverity::Error => "error",
        }
    }
}

impl fmt::Display for FindingSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One piece of structured evidence supporting a finding. Strings are
/// fine for human-readable detail, but consumers can also inspect counts,
/// ratios, and ranges programmatically.
#[derive(Clone, Debug, PartialEq)]
pub enum FindingEvidence {
    /// An integer count (e.g. `Count{ label: "n_missing", value: 42 }`).
    Count { label: String, value: u64 },
    /// A ratio in `[0, 1]`. Clamped at construction time.
    Ratio { label: String, value: f64 },
    /// A min/max numeric range.
    Range { label: String, min: f64, max: f64 },
    /// A scalar metric not naturally a ratio (e.g. mean, PSI score).
    Metric { label: String, value: f64 },
    /// Short string sample (truncated; no PII heuristics — caller is
    /// responsible for redaction).
    Sample { label: String, value: String },
}

impl FindingEvidence {
    /// Canonical-bytes representation used inside fingerprint composition.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            FindingEvidence::Count { label, value } => {
                out.push(b'C');
                out.extend_from_slice(label.as_bytes());
                out.push(0);
                out.extend_from_slice(&value.to_le_bytes());
            }
            FindingEvidence::Ratio { label, value } => {
                out.push(b'R');
                out.extend_from_slice(label.as_bytes());
                out.push(0);
                out.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            FindingEvidence::Range { label, min, max } => {
                out.push(b'G');
                out.extend_from_slice(label.as_bytes());
                out.push(0);
                out.extend_from_slice(&min.to_bits().to_le_bytes());
                out.extend_from_slice(&max.to_bits().to_le_bytes());
            }
            FindingEvidence::Metric { label, value } => {
                out.push(b'M');
                out.extend_from_slice(label.as_bytes());
                out.push(0);
                out.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            FindingEvidence::Sample { label, value } => {
                out.push(b'S');
                out.extend_from_slice(label.as_bytes());
                out.push(0);
                out.extend_from_slice(value.as_bytes());
            }
        }
        out
    }
}

/// A single validation/drift/causal finding.
///
/// The `id` is a content fingerprint — it does **not** include a timestamp
/// or counter, so repeated runs produce identical IDs.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationFinding {
    pub id: FingerprintId,
    pub code: &'static str,
    pub severity: FindingSeverity,
    pub message: String,
    /// `None` for dataset-wide findings; `Some(name)` when scoped to a column.
    pub column: Option<String>,
    /// `None` for column-wide or dataset-wide findings; `Some((lo, hi_exclusive))`
    /// when the finding pinpoints a row range.
    pub row_range: Option<(usize, usize)>,
    pub evidence: Vec<FindingEvidence>,
    /// How many rows were inspected when this finding was produced.
    pub sample_size: u64,
    /// Assumptions the analyst should sanity-check (e.g. "treats NaN as missing").
    pub assumptions: Vec<String>,
    /// Concrete suggested follow-ups.
    pub suggested_next_checks: Vec<String>,
}

impl ValidationFinding {
    /// Build a finding, deriving its content-addressed `FingerprintId`.
    pub fn new(
        code: &'static str,
        severity: FindingSeverity,
        message: impl Into<String>,
        column: Option<String>,
        row_range: Option<(usize, usize)>,
        evidence: Vec<FindingEvidence>,
        sample_size: u64,
        assumptions: Vec<String>,
        suggested_next_checks: Vec<String>,
    ) -> Self {
        let message = message.into();
        let id = Self::derive_id(code, severity, &message, column.as_deref(), row_range, &evidence);
        Self {
            id,
            code,
            severity,
            message,
            column,
            row_range,
            evidence,
            sample_size,
            assumptions,
            suggested_next_checks,
        }
    }

    fn derive_id(
        code: &str,
        severity: FindingSeverity,
        message: &str,
        column: Option<&str>,
        row_range: Option<(usize, usize)>,
        evidence: &[FindingEvidence],
    ) -> FingerprintId {
        let mut parts: Vec<FingerprintId> = Vec::with_capacity(3 + evidence.len());
        parts.push(fingerprint_str(IdDomain::Finding, code));
        parts.push(fingerprint(IdDomain::Finding, &[severity as u8]));
        parts.push(fingerprint_str(IdDomain::Finding, message));
        if let Some(c) = column {
            parts.push(fingerprint_str(IdDomain::Finding, c));
        }
        if let Some((lo, hi)) = row_range {
            let mut buf = [0u8; 16];
            buf[..8].copy_from_slice(&(lo as u64).to_le_bytes());
            buf[8..].copy_from_slice(&(hi as u64).to_le_bytes());
            parts.push(fingerprint(IdDomain::Finding, &buf));
        }
        for e in evidence {
            parts.push(fingerprint(IdDomain::Finding, &e.canonical_bytes()));
        }
        fingerprint_compose(IdDomain::Finding, "finding", &parts)
    }

    /// Stable sort key: severity first (Error first when sorted descending),
    /// then code, then column, then ID.
    pub fn sort_key(&self) -> (FindingSeverity, &str, Option<&str>, FingerprintId) {
        (self.severity, self.code, self.column.as_deref(), self.id)
    }
}

/// Aggregate counts of findings by severity. Used inside `LockeReport`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeverityCounts {
    pub info: u64,
    pub notice: u64,
    pub warning: u64,
    pub error: u64,
}

impl SeverityCounts {
    pub fn from_findings(fs: &[ValidationFinding]) -> Self {
        let mut sc = Self::default();
        for f in fs {
            match f.severity {
                FindingSeverity::Info => sc.info += 1,
                FindingSeverity::Notice => sc.notice += 1,
                FindingSeverity::Warning => sc.warning += 1,
                FindingSeverity::Error => sc.error += 1,
            }
        }
        sc
    }
    pub fn total(&self) -> u64 {
        self.info + self.notice + self.warning + self.error
    }
}

/// The user-supplied input to a validation run that the report can echo back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LockeInputSummary {
    pub dataset_label: String,
    pub n_rows: u64,
    pub n_cols: u64,
    pub column_types: BTreeMap<String, String>,
}

/// Per-column belief sub-report. Constructed by the belief aggregator;
/// listed here because validation already emits enough signal to compute it.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnBeliefReport {
    pub column: String,
    pub findings: Vec<ValidationFinding>,
    pub missingness_rate: f64,
    pub distinct_count: u64,
    pub constant: bool,
    pub near_constant: bool,
}

/// Top-level container for a single validate-or-compare invocation.
///
/// Fields are ordered alphabetically when emitted to JSON so two equal
/// reports produce byte-identical bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct LockeReport {
    pub schema_version: u32,
    pub input: LockeInputSummary,
    pub findings: Vec<ValidationFinding>,
    pub severity_counts: SeverityCounts,
    pub column_reports: BTreeMap<String, ColumnBeliefReport>,
    pub assumptions: Vec<String>,
    /// Deterministic run identifier — fingerprint of (input summary + findings).
    /// **Does not include `per_value_lineage`** so attaching lineage to an
    /// existing report does not change its identity — the lineage is a
    /// derived observation, not a primary fact.
    pub run_id: FingerprintId,
    /// **v0.7+ (A2-by-default)** — optional per-value canonicalisation
    /// lineage, populated when [`crate::validation::ValidationConfig::collect_per_value_lineage`]
    /// is `true`. `None` for the historical byte-identical-to-v0.7 path.
    /// Not currently included in JSON emit / parse round-trips — the
    /// lineage map is in-memory only for this batch. See ADR-0038 for
    /// the data model.
    pub per_value_lineage: Option<crate::per_value_lineage::PerValueLineageMap>,
}

impl LockeReport {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn new(
        input: LockeInputSummary,
        mut findings: Vec<ValidationFinding>,
        column_reports: BTreeMap<String, ColumnBeliefReport>,
        assumptions: Vec<String>,
    ) -> Self {
        findings.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        let severity_counts = SeverityCounts::from_findings(&findings);
        let run_id = Self::derive_run_id(&input, &findings);
        Self {
            schema_version: Self::SCHEMA_VERSION,
            input,
            findings,
            severity_counts,
            column_reports,
            assumptions,
            run_id,
            per_value_lineage: None,
        }
    }

    /// Attach a per-value lineage map to this report, returning the
    /// modified report. Used by [`crate::api::validate`] when
    /// [`crate::validation::ValidationConfig::collect_per_value_lineage`]
    /// is `true`. Does not change `run_id` — lineage is derived data.
    pub fn with_per_value_lineage(
        mut self,
        lineage: crate::per_value_lineage::PerValueLineageMap,
    ) -> Self {
        self.per_value_lineage = Some(lineage);
        self
    }

    fn derive_run_id(input: &LockeInputSummary, findings: &[ValidationFinding]) -> FingerprintId {
        let mut parts = vec![
            fingerprint_str(IdDomain::Finding, &input.dataset_label),
            fingerprint(IdDomain::Finding, &input.n_rows.to_le_bytes()),
            fingerprint(IdDomain::Finding, &input.n_cols.to_le_bytes()),
        ];
        for (name, ty) in &input.column_types {
            parts.push(fingerprint_str(IdDomain::Finding, name));
            parts.push(fingerprint_str(IdDomain::Finding, ty));
        }
        for f in findings {
            parts.push(f.id);
        }
        fingerprint_compose(IdDomain::Finding, "locke-run", &parts)
    }

    /// Highest severity present in the report; `Info` if empty.
    pub fn worst_severity(&self) -> FindingSeverity {
        self.findings
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(FindingSeverity::Info)
    }
}

/// Public alias for compatibility with the brief's `DatasetSkepticismReport` name.
pub type DatasetSkepticismReport = LockeReport;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_finding(code: &'static str, sev: FindingSeverity, col: &str) -> ValidationFinding {
        ValidationFinding::new(
            code,
            sev,
            "test message",
            Some(col.into()),
            None,
            vec![FindingEvidence::Count { label: "n_missing".into(), value: 3 }],
            100,
            vec!["NaN treated as missing".into()],
            vec!["inspect rows 0..10".into()],
        )
    }

    #[test]
    fn finding_id_is_content_addressed() {
        let f1 = sample_finding("E9001", FindingSeverity::Warning, "age");
        let f2 = sample_finding("E9001", FindingSeverity::Warning, "age");
        assert_eq!(f1.id, f2.id);
    }

    #[test]
    fn changing_column_changes_id() {
        let f1 = sample_finding("E9001", FindingSeverity::Warning, "age");
        let f2 = sample_finding("E9001", FindingSeverity::Warning, "height");
        assert_ne!(f1.id, f2.id);
    }

    #[test]
    fn severity_ordering_is_total_and_ascending() {
        use FindingSeverity::*;
        let mut s = vec![Error, Info, Warning, Notice];
        s.sort();
        assert_eq!(s, vec![Info, Notice, Warning, Error]);
    }

    #[test]
    fn report_sorts_findings_canonically() {
        let f1 = sample_finding("E9002", FindingSeverity::Info, "z");
        let f2 = sample_finding("E9001", FindingSeverity::Error, "a");
        let input = LockeInputSummary {
            dataset_label: "ds".into(),
            n_rows: 100,
            n_cols: 2,
            column_types: BTreeMap::new(),
        };
        let report = LockeReport::new(input, vec![f1, f2], BTreeMap::new(), vec![]);
        // Sort is by (severity, code, column) — Info < Error so Info comes first.
        assert_eq!(report.findings[0].code, "E9002");
        assert_eq!(report.findings[1].code, "E9001");
    }

    #[test]
    fn worst_severity_is_max() {
        let f1 = sample_finding("E9001", FindingSeverity::Notice, "a");
        let f2 = sample_finding("E9002", FindingSeverity::Warning, "b");
        let input = LockeInputSummary::default();
        let r = LockeReport::new(input, vec![f1, f2], BTreeMap::new(), vec![]);
        assert_eq!(r.worst_severity(), FindingSeverity::Warning);
    }

    #[test]
    fn severity_counts_sum_to_total() {
        let fs = vec![
            sample_finding("E9001", FindingSeverity::Info, "a"),
            sample_finding("E9002", FindingSeverity::Warning, "b"),
            sample_finding("E9003", FindingSeverity::Warning, "c"),
        ];
        let sc = SeverityCounts::from_findings(&fs);
        assert_eq!(sc.info, 1);
        assert_eq!(sc.warning, 2);
        assert_eq!(sc.total(), 3);
    }

    #[test]
    fn run_id_is_deterministic() {
        let f = sample_finding("E9001", FindingSeverity::Warning, "x");
        let input = LockeInputSummary {
            dataset_label: "ds".into(),
            n_rows: 10,
            n_cols: 1,
            column_types: BTreeMap::new(),
        };
        let a = LockeReport::new(input.clone(), vec![f.clone()], BTreeMap::new(), vec![]);
        let b = LockeReport::new(input, vec![f], BTreeMap::new(), vec![]);
        assert_eq!(a.run_id, b.run_id);
    }
}
