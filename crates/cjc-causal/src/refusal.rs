//! Locke-report inspection for refusal-grade findings.
//!
//! Per ADR-0043 §5, cjc-causal estimators refuse to run when the analyst's
//! input data fails any of three checks against the supplied Locke report:
//!
//! - **E9001 missingness ≥ 0.30** on the treatment or outcome column
//!   (severity-grade `Error` is the secondary signal when the precise
//!   `Ratio` evidence isn't available)
//! - **E9009** *not yet promoted to `Column::Float`* on any declared covariate
//!   (the auto-promotion didn't take — see ADR-0042)
//! - **E9060** strong target leakage on any declared covariate
//!   (covariate is a near-perfect predictor of the outcome)
//!
//! On match, the offending [`ValidationFinding`]s are bundled into a
//! [`CausalError::DataQualityRefusal`] so the caller can route them straight
//! into their existing Locke reporting pipeline.

use crate::error::CausalError;
use cjc_locke::report::{FindingEvidence, FindingSeverity, LockeReport, ValidationFinding};

/// Default missingness fraction at which E9001 forces refusal.
///
/// 0.30 was chosen as the boundary at which propensity-score matching
/// becomes too sensitive to missing-at-random assumptions for the v0.1
/// surface. Implementation sessions for v0.2 may make this configurable
/// per-estimator.
pub const REFUSAL_MISSINGNESS_THRESHOLD: f64 = 0.30;

/// Inspect a Locke report for refusal-grade findings against the given
/// treatment / outcome / covariate column set.
///
/// Returns `Ok(())` if no refusal-grade findings match. Returns
/// `Err(CausalError::DataQualityRefusal)` carrying the offending findings
/// on the first match — the estimator never runs in that case.
pub fn check_locke_refusal(
    report: &LockeReport,
    treatment: &str,
    outcome: &str,
    covariates: &[&str],
) -> Result<(), CausalError> {
    let mut offenders: Vec<ValidationFinding> = Vec::new();

    for finding in &report.findings {
        let col = finding.column.as_deref();

        // E9001 — missingness on treatment or outcome
        if finding.code == "E9001" && (col == Some(treatment) || col == Some(outcome)) {
            if refusal_grade_missingness(finding) {
                offenders.push(finding.clone());
            }
            continue;
        }

        // E9009 — Str column not auto-promoted (continuous covariate stayed Str)
        if finding.code == "E9009" {
            if let Some(c) = col {
                if covariates.iter().any(|cv| *cv == c) {
                    offenders.push(finding.clone());
                }
            }
            continue;
        }

        // E9060 — strong target leakage on a declared covariate
        if finding.code == "E9060" {
            if let Some(c) = col {
                if covariates.iter().any(|cv| *cv == c) {
                    offenders.push(finding.clone());
                }
            }
            continue;
        }
    }

    if offenders.is_empty() {
        Ok(())
    } else {
        Err(CausalError::DataQualityRefusal { findings: offenders })
    }
}

/// E9001 refuses if any `Ratio` evidence on the finding meets or exceeds the
/// threshold, OR if the finding's severity is `Error` (Locke's signal that the
/// missingness is severe even when the exact ratio label varies across
/// versions).
fn refusal_grade_missingness(finding: &ValidationFinding) -> bool {
    if finding.severity == FindingSeverity::Error {
        return true;
    }
    finding.evidence.iter().any(|e| match e {
        FindingEvidence::Ratio { value, .. } => *value >= REFUSAL_MISSINGNESS_THRESHOLD,
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_locke::report::{LockeInputSummary, LockeReport};
    use std::collections::BTreeMap;

    fn finding(
        code: &'static str,
        severity: FindingSeverity,
        column: Option<&str>,
        evidence: Vec<FindingEvidence>,
    ) -> ValidationFinding {
        ValidationFinding::new(
            code,
            severity,
            "test",
            column.map(|s| s.to_string()),
            None,
            evidence,
            0,
            vec![],
            vec![],
        )
    }

    fn report_with(findings: Vec<ValidationFinding>) -> LockeReport {
        LockeReport::new(
            LockeInputSummary::default(),
            findings,
            BTreeMap::new(),
            vec![],
        )
    }

    #[test]
    fn empty_report_passes() {
        let report = report_with(vec![]);
        assert!(check_locke_refusal(&report, "T", "Y", &["x"]).is_ok());
    }

    #[test]
    fn e9001_on_treatment_with_high_ratio_refuses() {
        let f = finding(
            "E9001",
            FindingSeverity::Warning,
            Some("T"),
            vec![FindingEvidence::Ratio { label: "missing_fraction".to_string(), value: 0.45 }],
        );
        let report = report_with(vec![f]);
        let err = check_locke_refusal(&report, "T", "Y", &["x"]).unwrap_err();
        match err {
            CausalError::DataQualityRefusal { findings } => {
                assert_eq!(findings.len(), 1);
                assert_eq!(findings[0].code, "E9001");
            }
            other => panic!("expected DataQualityRefusal, got {:?}", other),
        }
    }

    #[test]
    fn e9001_on_outcome_with_error_severity_refuses() {
        let f = finding("E9001", FindingSeverity::Error, Some("Y"), vec![]);
        let report = report_with(vec![f]);
        assert!(matches!(
            check_locke_refusal(&report, "T", "Y", &["x"]),
            Err(CausalError::DataQualityRefusal { .. })
        ));
    }

    #[test]
    fn e9001_low_ratio_warning_severity_passes() {
        // 0.10 < 0.30 threshold and Warning < Error → not refusal-grade.
        let f = finding(
            "E9001",
            FindingSeverity::Warning,
            Some("T"),
            vec![FindingEvidence::Ratio { label: "missing_fraction".to_string(), value: 0.10 }],
        );
        let report = report_with(vec![f]);
        assert!(check_locke_refusal(&report, "T", "Y", &["x"]).is_ok());
    }

    #[test]
    fn e9001_on_unrelated_column_passes() {
        // E9001 fires on a column that is neither treatment, outcome, nor covariate.
        let f = finding(
            "E9001",
            FindingSeverity::Error,
            Some("unrelated"),
            vec![],
        );
        let report = report_with(vec![f]);
        assert!(check_locke_refusal(&report, "T", "Y", &["x"]).is_ok());
    }

    #[test]
    fn e9009_on_covariate_refuses() {
        let f = finding("E9009", FindingSeverity::Info, Some("age"), vec![]);
        let report = report_with(vec![f]);
        let err = check_locke_refusal(&report, "T", "Y", &["age", "income"]).unwrap_err();
        assert!(matches!(err, CausalError::DataQualityRefusal { .. }));
    }

    #[test]
    fn e9009_on_non_covariate_passes() {
        let f = finding("E9009", FindingSeverity::Info, Some("city"), vec![]);
        let report = report_with(vec![f]);
        assert!(check_locke_refusal(&report, "T", "Y", &["age", "income"]).is_ok());
    }

    #[test]
    fn e9060_on_covariate_refuses() {
        let f = finding("E9060", FindingSeverity::Error, Some("leaky_feature"), vec![]);
        let report = report_with(vec![f]);
        assert!(matches!(
            check_locke_refusal(&report, "T", "Y", &["leaky_feature"]),
            Err(CausalError::DataQualityRefusal { .. })
        ));
    }

    #[test]
    fn unrelated_code_passes() {
        // E9050 is a Locke temporal code — not on the refusal list.
        let f = finding("E9050", FindingSeverity::Error, Some("T"), vec![]);
        let report = report_with(vec![f]);
        assert!(check_locke_refusal(&report, "T", "Y", &["x"]).is_ok());
    }

    #[test]
    fn multiple_offenders_all_returned() {
        let f1 = finding("E9001", FindingSeverity::Error, Some("T"), vec![]);
        let f2 = finding("E9060", FindingSeverity::Error, Some("x"), vec![]);
        let report = report_with(vec![f1, f2]);
        match check_locke_refusal(&report, "T", "Y", &["x"]) {
            Err(CausalError::DataQualityRefusal { findings }) => {
                assert_eq!(findings.len(), 2, "both offenders must be reported");
            }
            other => panic!("expected DataQualityRefusal, got {:?}", other),
        }
    }
}
