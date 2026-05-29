//! Locke governance policy DSL (v0.7+ heavy, A3).
//!
//! At scale Locke produces many findings — 142+ on the diabetes-130
//! workload — most of which are valid concerns the team has *already
//! triaged*. Without a way to suppress acknowledged findings, owners
//! cannot run Locke in CI without their alerting volume drowning real
//! regressions. A3 adds the missing governance layer.
//!
//! Three rule kinds, all declarative:
//!
//! 1. **Suppression** — drop findings matching `(code, column?)` with
//!    a recorded `reason`. Every suppression decision is content-
//!    addressed via [`FingerprintId`] for the audit log.
//! 2. **Owner annotation** — tag findings with a responsible team /
//!    person. Useful for routing CI failures to the right channel
//!    without changing the finding itself.
//! 3. **Required-finding policy** — assert findings of code X satisfy
//!    a threshold. Used to enforce contractual guarantees ("no E9004
//!    duplicate keys, ever").
//!
//! ## Determinism
//!
//! - [`PolicyResult::suppressions`] is in the order findings appear in
//!   the input report (which is itself canonically sorted).
//! - [`PolicyResult::attributions`] is in the same order as the
//!   filtered findings.
//! - [`PolicyResult::requirements`] follows [`Policy::requirements`]
//!   declaration order.
//! - Every [`SuppressionDecision::decision_id`] is a content-addressed
//!   fingerprint over (finding_id, rule_fingerprint); two runs with the
//!   same inputs produce byte-identical IDs.
//!
//! ## Audit chain
//!
//! Each suppression decision can be replayed: given the original
//! `LockeReport` and the `Policy`, the same `PolicyResult` is produced
//! bit-for-bit. The `decision_id` is the content-address that lets
//! downstream consumers (the gate, CI alerts, dashboards) reason about
//! "is this the same suppression decision the team approved last
//! week?" purely by ID comparison.
//!
//! ## Out of scope (v0.7+ A3.1)
//!
//! - Column wildcards / patterns (`column = "diag_*"`).
//! - Multi-policy file inheritance / merging.
//! - Suppression expiration dates.
//! - Conditional policies ("require E9004=0 only if `target_column`
//!   is `readmitted`").
//!
//! These are A3.2+; the current rule semantics are exact-match.

use std::collections::BTreeMap;

use crate::id::{fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use crate::report::{FindingSeverity, LockeReport, ValidationFinding};

// ─── Operators ────────────────────────────────────────────────────────────

/// Comparison operator for required-finding policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RequirementOperator {
    /// `observed == 0` — strict prohibition. Threshold field ignored.
    EqZero,
    /// `observed <= threshold`.
    LessOrEqual,
    /// `observed >= threshold`.
    GreaterOrEqual,
    /// `observed < threshold`.
    Less,
    /// `observed > threshold`.
    Greater,
    /// `observed == threshold`.
    Equal,
}

impl RequirementOperator {
    /// Stable text representation for emit and audit IDs.
    pub fn label(self) -> &'static str {
        match self {
            RequirementOperator::EqZero => "==0",
            RequirementOperator::LessOrEqual => "<=",
            RequirementOperator::GreaterOrEqual => ">=",
            RequirementOperator::Less => "<",
            RequirementOperator::Greater => ">",
            RequirementOperator::Equal => "==",
        }
    }

    /// Parse from a stable string representation.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "==0" => Some(RequirementOperator::EqZero),
            "<=" | "lte" => Some(RequirementOperator::LessOrEqual),
            ">=" | "gte" => Some(RequirementOperator::GreaterOrEqual),
            "<" | "lt" => Some(RequirementOperator::Less),
            ">" | "gt" => Some(RequirementOperator::Greater),
            "==" | "eq" => Some(RequirementOperator::Equal),
            _ => None,
        }
    }

    /// Evaluate the operator against an observed count.
    pub fn evaluate(self, observed: u64, threshold: u64) -> bool {
        match self {
            RequirementOperator::EqZero => observed == 0,
            RequirementOperator::LessOrEqual => observed <= threshold,
            RequirementOperator::GreaterOrEqual => observed >= threshold,
            RequirementOperator::Less => observed < threshold,
            RequirementOperator::Greater => observed > threshold,
            RequirementOperator::Equal => observed == threshold,
        }
    }
}

// ─── Rule data model ──────────────────────────────────────────────────────

/// Drop findings matching `(code, column?)` with a recorded `reason`.
/// `column = None` matches findings on any column (including dataset-wide).
#[derive(Clone, Debug, PartialEq)]
pub struct SuppressionRule {
    pub code: String,
    pub column: Option<String>,
    pub reason: String,
}

impl SuppressionRule {
    /// Content-addressed fingerprint over (code, column, reason). Used
    /// inside [`SuppressionDecision::decision_id`] and stable across runs.
    pub fn fingerprint(&self) -> FingerprintId {
        let canon = format!(
            "{}\u{1f}{}\u{1f}{}",
            self.code,
            self.column.as_deref().unwrap_or(""),
            self.reason
        );
        fingerprint_str(IdDomain::AuditEvent, &canon)
    }
}

/// Tag findings with an owner. `column = None` means the rule applies
/// to *any* column; `code = None` means *any* code.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerRule {
    pub team: String,
    pub column: Option<String>,
    pub code: Option<String>,
}

impl OwnerRule {
    pub fn fingerprint(&self) -> FingerprintId {
        let canon = format!(
            "{}\u{1f}{}\u{1f}{}",
            self.team,
            self.column.as_deref().unwrap_or(""),
            self.code.as_deref().unwrap_or("")
        );
        fingerprint_str(IdDomain::AuditEvent, &canon)
    }
}

/// Assert findings of `code` satisfy `operator(observed, threshold)`
/// after suppression. Optionally attributed to an owner for routing.
#[derive(Clone, Debug, PartialEq)]
pub struct RequiredFindingRule {
    pub code: String,
    pub operator: RequirementOperator,
    pub threshold: u64,
    pub owner: Option<String>,
}

impl RequiredFindingRule {
    pub fn fingerprint(&self) -> FingerprintId {
        let canon = format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}",
            self.code,
            self.operator.label(),
            self.threshold,
            self.owner.as_deref().unwrap_or("")
        );
        fingerprint_str(IdDomain::AuditEvent, &canon)
    }
}

/// Composite policy: ordered lists of suppressions, owners, and
/// requirements. Order matters — rules apply first-match-wins (for
/// suppression and owner) and requirements are evaluated in
/// declaration order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Policy {
    pub suppressions: Vec<SuppressionRule>,
    pub owners: Vec<OwnerRule>,
    pub requirements: Vec<RequiredFindingRule>,
}

impl Policy {
    /// Content-addressed fingerprint over the entire policy. Two policies
    /// with byte-identical rules in the same order produce byte-identical
    /// fingerprints; reordering or adding rules produces a new fingerprint.
    pub fn fingerprint(&self) -> FingerprintId {
        let mut parts = Vec::with_capacity(
            self.suppressions.len() + self.owners.len() + self.requirements.len(),
        );
        for s in &self.suppressions {
            parts.push(s.fingerprint());
        }
        for o in &self.owners {
            parts.push(o.fingerprint());
        }
        for r in &self.requirements {
            parts.push(r.fingerprint());
        }
        fingerprint_compose(IdDomain::AuditEvent, "policy", &parts)
    }
}

// ─── Result data model ────────────────────────────────────────────────────

/// One suppression that occurred during [`apply_policy`].
#[derive(Clone, Debug, PartialEq)]
pub struct SuppressionDecision {
    /// ID of the finding that was suppressed.
    pub finding_id: FingerprintId,
    /// Position of the rule in [`Policy::suppressions`] that matched.
    pub rule_index: usize,
    /// Content-addressed fingerprint over (finding_id, rule_fingerprint).
    /// Stable across runs; usable as an audit log primary key.
    pub decision_id: FingerprintId,
    /// Copy of the matched rule's `code`, `column`, `reason` for emit.
    pub code: String,
    pub column: Option<String>,
    pub reason: String,
    /// Severity of the original suppressed finding — preserved so the
    /// audit log can answer "what severity got dropped?"
    pub severity: FindingSeverity,
}

/// One owner attribution that occurred during [`apply_policy`].
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerAttribution {
    pub finding_id: FingerprintId,
    pub team: String,
    /// Position of the rule in [`Policy::owners`] that matched.
    pub rule_index: usize,
}

/// One required-finding evaluation result.
#[derive(Clone, Debug, PartialEq)]
pub struct RequirementResult {
    pub code: String,
    pub operator: RequirementOperator,
    pub threshold: u64,
    pub observed: u64,
    pub satisfied: bool,
    pub owner: Option<String>,
}

/// Result of applying a [`Policy`] to a [`LockeReport`].
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyResult {
    pub suppressions: Vec<SuppressionDecision>,
    pub attributions: Vec<OwnerAttribution>,
    pub requirements: Vec<RequirementResult>,
    /// Findings that did NOT match any suppression rule, in input order.
    pub remaining_findings: Vec<ValidationFinding>,
    /// Stable fingerprint over the applied policy, included in the
    /// result for downstream consumers that want to verify "is this
    /// from the policy I expect?"
    pub policy_fingerprint: FingerprintId,
}

impl PolicyResult {
    /// True iff every requirement was satisfied.
    pub fn all_requirements_satisfied(&self) -> bool {
        self.requirements.iter().all(|r| r.satisfied)
    }

    /// True iff at least one requirement failed — the gate should
    /// surface a non-zero exit.
    pub fn gate_fails(&self) -> bool {
        !self.all_requirements_satisfied()
    }

    /// Group attributions by team for routing summaries. `BTreeMap`
    /// iteration order → deterministic team listing.
    pub fn attributions_by_team(&self) -> BTreeMap<String, Vec<&OwnerAttribution>> {
        let mut out: BTreeMap<String, Vec<&OwnerAttribution>> = BTreeMap::new();
        for a in &self.attributions {
            out.entry(a.team.clone()).or_default().push(a);
        }
        out
    }
}

// ─── Apply ────────────────────────────────────────────────────────────────

/// Apply a [`Policy`] to a [`LockeReport`]. Returns a [`PolicyResult`]
/// containing every suppression decision, owner attribution, and
/// requirement-evaluation outcome. Pure function — does not mutate the
/// input report.
pub fn apply_policy(report: &LockeReport, policy: &Policy) -> PolicyResult {
    let mut suppressions: Vec<SuppressionDecision> = Vec::new();
    let mut remaining: Vec<ValidationFinding> = Vec::new();

    for f in &report.findings {
        let matched = policy
            .suppressions
            .iter()
            .enumerate()
            .find(|(_, rule)| suppression_matches(rule, f));
        match matched {
            Some((rule_index, rule)) => {
                let decision = SuppressionDecision {
                    finding_id: f.id,
                    rule_index,
                    decision_id: compose_decision_id(f.id, rule),
                    code: rule.code.clone(),
                    column: rule.column.clone(),
                    reason: rule.reason.clone(),
                    severity: f.severity,
                };
                suppressions.push(decision);
            }
            None => {
                remaining.push(f.clone());
            }
        }
    }

    // Findings are already canonically sorted in the LockeReport. We
    // preserve that order here so the emit and the audit log line
    // up with the existing finding-emit conventions.

    let attributions: Vec<OwnerAttribution> = remaining
        .iter()
        .filter_map(|f| {
            policy
                .owners
                .iter()
                .enumerate()
                .find(|(_, rule)| owner_matches(rule, f))
                .map(|(idx, rule)| OwnerAttribution {
                    finding_id: f.id,
                    team: rule.team.clone(),
                    rule_index: idx,
                })
        })
        .collect();

    let requirements: Vec<RequirementResult> = policy
        .requirements
        .iter()
        .map(|req| {
            let observed = remaining.iter().filter(|f| f.code == req.code).count() as u64;
            let satisfied = req.operator.evaluate(observed, req.threshold);
            RequirementResult {
                code: req.code.clone(),
                operator: req.operator,
                threshold: req.threshold,
                observed,
                satisfied,
                owner: req.owner.clone(),
            }
        })
        .collect();

    PolicyResult {
        suppressions,
        attributions,
        requirements,
        remaining_findings: remaining,
        policy_fingerprint: policy.fingerprint(),
    }
}

fn suppression_matches(rule: &SuppressionRule, f: &ValidationFinding) -> bool {
    if rule.code != f.code {
        return false;
    }
    if let Some(rule_col) = &rule.column {
        if f.column.as_deref() != Some(rule_col.as_str()) {
            return false;
        }
    }
    true
}

fn owner_matches(rule: &OwnerRule, f: &ValidationFinding) -> bool {
    if let Some(c) = &rule.code {
        if c != f.code {
            return false;
        }
    }
    if let Some(col) = &rule.column {
        if f.column.as_deref() != Some(col.as_str()) {
            return false;
        }
    }
    true
}

fn compose_decision_id(finding_id: FingerprintId, rule: &SuppressionRule) -> FingerprintId {
    fingerprint_compose(
        IdDomain::AuditEvent,
        "policy_suppress",
        &[finding_id, rule.fingerprint()],
    )
}

// ─── Emit ─────────────────────────────────────────────────────────────────

/// Render a [`PolicyResult`] as canonical text — stable across runs,
/// suitable for CLI emit and snapshot testing.
pub fn emit_policy_result_text(result: &PolicyResult) -> String {
    let mut s = String::new();
    s.push_str("# Locke Policy Result\n");
    s.push_str(&format!("policy_fingerprint: {}\n", result.policy_fingerprint));
    s.push_str(&format!(
        "summary: suppressions={} attributions={} requirements={} remaining_findings={}\n",
        result.suppressions.len(),
        result.attributions.len(),
        result.requirements.len(),
        result.remaining_findings.len(),
    ));

    if !result.suppressions.is_empty() {
        s.push_str("\nsuppressions:\n");
        for d in &result.suppressions {
            s.push_str(&format!(
                "  - decision_id={} finding_id={} rule_index={}\n",
                d.decision_id, d.finding_id, d.rule_index
            ));
            s.push_str(&format!(
                "    code={} column={} severity={} reason={:?}\n",
                d.code,
                d.column.as_deref().unwrap_or("-"),
                d.severity,
                d.reason
            ));
        }
    }

    if !result.attributions.is_empty() {
        s.push_str("\nattributions:\n");
        let grouped = result.attributions_by_team();
        for (team, atts) in &grouped {
            s.push_str(&format!("  team {}: {} finding(s)\n", team, atts.len()));
            for a in atts {
                s.push_str(&format!(
                    "    - finding_id={} rule_index={}\n",
                    a.finding_id, a.rule_index
                ));
            }
        }
    }

    if !result.requirements.is_empty() {
        s.push_str("\nrequirements:\n");
        for r in &result.requirements {
            let status = if r.satisfied { "OK" } else { "FAIL" };
            s.push_str(&format!(
                "  - [{}] code={} {} {} (observed={}) owner={}\n",
                status,
                r.code,
                r.operator.label(),
                r.threshold,
                r.observed,
                r.owner.as_deref().unwrap_or("-")
            ));
        }
    }

    if result.gate_fails() {
        s.push_str("\nstatus: GATE FAIL — one or more requirements not satisfied.\n");
    }

    s
}

// ─── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{validate, ValidateOptions};
    use cjc_data::{Column, DataFrame};

    fn df_clean() -> DataFrame {
        DataFrame::from_columns(vec![(
            "x".into(),
            Column::Float((0..100).map(|i| i as f64).collect()),
        )])
        .unwrap()
    }

    fn df_with_nans() -> DataFrame {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        for i in 0..50 {
            v[i] = f64::NAN;
        }
        DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap()
    }

    fn report_with_e9001() -> LockeReport {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        validate(&df_with_nans(), &opts)
    }

    // ── Operator semantics ────────────────────────────────────────────

    #[test]
    fn requirement_operator_evaluate_covers_all_variants() {
        assert!(RequirementOperator::EqZero.evaluate(0, 999));
        assert!(!RequirementOperator::EqZero.evaluate(1, 999));
        assert!(RequirementOperator::LessOrEqual.evaluate(5, 5));
        assert!(RequirementOperator::LessOrEqual.evaluate(4, 5));
        assert!(!RequirementOperator::LessOrEqual.evaluate(6, 5));
        assert!(RequirementOperator::GreaterOrEqual.evaluate(5, 5));
        assert!(RequirementOperator::GreaterOrEqual.evaluate(6, 5));
        assert!(!RequirementOperator::GreaterOrEqual.evaluate(4, 5));
        assert!(RequirementOperator::Less.evaluate(4, 5));
        assert!(!RequirementOperator::Less.evaluate(5, 5));
        assert!(RequirementOperator::Greater.evaluate(6, 5));
        assert!(!RequirementOperator::Greater.evaluate(5, 5));
        assert!(RequirementOperator::Equal.evaluate(5, 5));
        assert!(!RequirementOperator::Equal.evaluate(4, 5));
    }

    #[test]
    fn requirement_operator_label_round_trips() {
        for op in [
            RequirementOperator::EqZero,
            RequirementOperator::LessOrEqual,
            RequirementOperator::GreaterOrEqual,
            RequirementOperator::Less,
            RequirementOperator::Greater,
            RequirementOperator::Equal,
        ] {
            assert_eq!(RequirementOperator::from_label(op.label()), Some(op));
        }
    }

    #[test]
    fn requirement_operator_from_label_accepts_aliases() {
        assert_eq!(
            RequirementOperator::from_label("lte"),
            Some(RequirementOperator::LessOrEqual)
        );
        assert_eq!(
            RequirementOperator::from_label("eq"),
            Some(RequirementOperator::Equal)
        );
        assert_eq!(RequirementOperator::from_label("garbage"), None);
    }

    // ── Suppression matching ──────────────────────────────────────────

    #[test]
    fn suppression_matches_on_code_only_when_column_omitted() {
        let r = SuppressionRule {
            code: "E9001".into(),
            column: None,
            reason: "ack".into(),
        };
        let report = report_with_e9001();
        let f = report.findings.iter().find(|f| f.code == "E9001").unwrap();
        assert!(suppression_matches(&r, f));
    }

    #[test]
    fn suppression_filters_by_column_when_supplied() {
        let r = SuppressionRule {
            code: "E9001".into(),
            column: Some("not_the_real_column".into()),
            reason: "wrong column on purpose".into(),
        };
        let report = report_with_e9001();
        let f = report.findings.iter().find(|f| f.code == "E9001").unwrap();
        assert!(!suppression_matches(&r, f));
    }

    #[test]
    fn apply_policy_drops_matching_findings_and_records_decisions() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "acknowledged missingness".into(),
            }],
            owners: vec![],
            requirements: vec![],
        };
        let result = apply_policy(&report, &policy);
        // At least one E9001 should have been suppressed.
        assert!(!result.suppressions.is_empty());
        // None of the remaining findings should be E9001.
        assert!(result.remaining_findings.iter().all(|f| f.code != "E9001"));
    }

    #[test]
    fn apply_policy_first_match_wins_for_suppressions() {
        let report = report_with_e9001();
        // Two rules that both match E9001 — only the first should fire.
        let policy = Policy {
            suppressions: vec![
                SuppressionRule {
                    code: "E9001".into(),
                    column: None,
                    reason: "first".into(),
                },
                SuppressionRule {
                    code: "E9001".into(),
                    column: None,
                    reason: "second".into(),
                },
            ],
            owners: vec![],
            requirements: vec![],
        };
        let result = apply_policy(&report, &policy);
        for d in &result.suppressions {
            assert_eq!(d.rule_index, 0);
            assert_eq!(d.reason, "first");
        }
    }

    // ── Owner attribution ─────────────────────────────────────────────

    #[test]
    fn owner_rule_matches_finding_when_code_omitted() {
        let r = OwnerRule {
            team: "data".into(),
            column: None,
            code: None,
        };
        let report = report_with_e9001();
        let f = report.findings.iter().find(|f| f.code == "E9001").unwrap();
        assert!(owner_matches(&r, f));
    }

    #[test]
    fn owner_attribution_only_applied_after_suppression() {
        let report = report_with_e9001();
        // Suppression drops E9001 → no remaining E9001 → no owner attribution
        // even though owner rule matches E9001.
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![OwnerRule {
                team: "data".into(),
                column: None,
                code: Some("E9001".into()),
            }],
            requirements: vec![],
        };
        let result = apply_policy(&report, &policy);
        assert!(result.attributions.iter().all(|a| {
            // The remaining findings drove attribution — verify by scanning.
            result
                .remaining_findings
                .iter()
                .any(|f| f.id == a.finding_id)
        }));
    }

    // ── Requirement evaluation ────────────────────────────────────────

    #[test]
    fn requirement_eqzero_passes_when_no_e9001_remains_post_suppression() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: None,
            }],
        };
        let result = apply_policy(&report, &policy);
        assert!(result.all_requirements_satisfied());
        assert!(!result.gate_fails());
    }

    #[test]
    fn requirement_eqzero_fails_when_e9001_present_without_suppression() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![],
            owners: vec![],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: Some("data".into()),
            }],
        };
        let result = apply_policy(&report, &policy);
        assert!(!result.all_requirements_satisfied());
        assert!(result.gate_fails());
        assert_eq!(result.requirements[0].owner.as_deref(), Some("data"));
    }

    // ── Determinism ───────────────────────────────────────────────────

    #[test]
    fn apply_policy_is_deterministic_across_runs() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![OwnerRule {
                team: "data".into(),
                column: None,
                code: None,
            }],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: Some("data".into()),
            }],
        };
        let r1 = apply_policy(&report, &policy);
        let r2 = apply_policy(&report, &policy);
        assert_eq!(r1, r2);
    }

    #[test]
    fn decision_id_is_content_addressed_and_stable() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![],
            requirements: vec![],
        };
        let r1 = apply_policy(&report, &policy);
        let r2 = apply_policy(&report, &policy);
        for (a, b) in r1.suppressions.iter().zip(r2.suppressions.iter()) {
            assert_eq!(a.decision_id, b.decision_id);
        }
    }

    #[test]
    fn changing_rule_reason_changes_decision_id() {
        let report = report_with_e9001();
        let policy_a = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "alpha".into(),
            }],
            owners: vec![],
            requirements: vec![],
        };
        let mut policy_b = policy_a.clone();
        policy_b.suppressions[0].reason = "beta".into();
        let r_a = apply_policy(&report, &policy_a);
        let r_b = apply_policy(&report, &policy_b);
        // Different reason → different decision IDs.
        for (a, b) in r_a.suppressions.iter().zip(r_b.suppressions.iter()) {
            assert_ne!(a.decision_id, b.decision_id);
        }
    }

    #[test]
    fn policy_fingerprint_is_stable_and_order_sensitive() {
        let r1 = SuppressionRule {
            code: "E9001".into(),
            column: None,
            reason: "alpha".into(),
        };
        let r2 = SuppressionRule {
            code: "E9080".into(),
            column: None,
            reason: "beta".into(),
        };
        let p_ab = Policy {
            suppressions: vec![r1.clone(), r2.clone()],
            owners: vec![],
            requirements: vec![],
        };
        let p_ba = Policy {
            suppressions: vec![r2, r1],
            owners: vec![],
            requirements: vec![],
        };
        // Stability.
        assert_eq!(p_ab.fingerprint(), p_ab.fingerprint());
        // Order matters.
        assert_ne!(p_ab.fingerprint(), p_ba.fingerprint());
    }

    // ── Empty / edge cases ────────────────────────────────────────────

    #[test]
    fn empty_policy_passes_through_every_finding() {
        let report = report_with_e9001();
        let result = apply_policy(&report, &Policy::default());
        assert_eq!(result.suppressions.len(), 0);
        assert_eq!(result.attributions.len(), 0);
        assert_eq!(result.requirements.len(), 0);
        assert_eq!(result.remaining_findings.len(), report.findings.len());
        assert!(result.all_requirements_satisfied());
    }

    #[test]
    fn empty_report_with_active_requirements_evaluates_correctly() {
        let opts = ValidateOptions {
            dataset_label: "clean".into(),
            ..Default::default()
        };
        let report = validate(&df_clean(), &opts);
        let policy = Policy {
            suppressions: vec![],
            owners: vec![],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: None,
            }],
        };
        let result = apply_policy(&report, &policy);
        assert_eq!(result.requirements[0].observed, 0);
        assert!(result.requirements[0].satisfied);
    }

    // ── Emit ──────────────────────────────────────────────────────────

    #[test]
    fn emit_policy_result_text_is_deterministic() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: None,
            }],
        };
        let result = apply_policy(&report, &policy);
        let s1 = emit_policy_result_text(&result);
        let s2 = emit_policy_result_text(&result);
        assert_eq!(s1, s2);
        assert!(s1.contains("policy_fingerprint:"));
        assert!(s1.contains("suppressions:"));
        assert!(s1.contains("requirements:"));
    }

    #[test]
    fn emit_text_marks_gate_fail_when_requirement_fails() {
        let report = report_with_e9001();
        let policy = Policy {
            suppressions: vec![],
            owners: vec![],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: None,
            }],
        };
        let result = apply_policy(&report, &policy);
        let s = emit_policy_result_text(&result);
        assert!(s.contains("GATE FAIL"));
    }

    #[test]
    fn emit_text_omits_gate_fail_when_all_requirements_satisfied() {
        let opts = ValidateOptions {
            dataset_label: "clean".into(),
            ..Default::default()
        };
        let report = validate(&df_clean(), &opts);
        let policy = Policy {
            suppressions: vec![],
            owners: vec![],
            requirements: vec![RequiredFindingRule {
                code: "E9001".into(),
                operator: RequirementOperator::EqZero,
                threshold: 0,
                owner: None,
            }],
        };
        let result = apply_policy(&report, &policy);
        let s = emit_policy_result_text(&result);
        assert!(!s.contains("GATE FAIL"));
    }
}
