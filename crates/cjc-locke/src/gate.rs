//! Report-diff and gate semantics (v0.4, extended in v0.7 part 2).
//!
//! `cjcl locke gate <reference.json> <current>` answers: "compared to
//! the snapshotted reference report, what changed?" Three relations on
//! findings:
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
//! **v0.7 part 2 (ADR-0036)** — alongside the finding-set diff, the
//! `ReportDiff` also carries a [`BeliefPartialOrder`] surfacing the
//! meet-semilattice relationship between the two reports' belief scores
//! via [`algebra::le_componentwise`](crate::algebra::le_componentwise).
//! That answers the user-facing question "did dataset belief
//! monotonically degrade between the snapshot and now?" The classification
//! is deterministic and adds no new external API surface to `diff_reports`.
//!
//! Exit semantics for the gate command:
//! - 0 if no findings appeared at or above the `--fail-on` severity.
//! - 1 if any "appeared" finding meets/exceeds the threshold.
//! - 2 on I/O / parse errors (handled by the CLI).

use std::collections::BTreeMap;

use crate::algebra::le_componentwise;
use crate::api::belief_report_from_locke;
use crate::belief::BeliefScore;
use crate::policy::{apply_policy, Policy, PolicyResult};
use crate::report::{FindingSeverity, LockeReport, ValidationFinding};

/// Tolerance used when comparing two belief scores axis-by-axis inside
/// [`diff_reports`]. Belief axes are computed from independent reports
/// so we tolerate ULP-scale derivation differences; `1e-12` is tight
/// enough to detect any meaningful divergence while ignoring the noise
/// floor of the underlying f64 arithmetic. The same default is used by
/// the algebra integration tests in `tests/locke/algebra_tests.rs`.
pub const DEFAULT_BELIEF_COMPARISON_EPS: f64 = 1e-12;

/// Componentwise partial order between a reference and a current
/// [`BeliefScore`], using the meet-semilattice algebra from v0.7 part 1.
///
/// `current_le_reference` is `true` iff every axis of `current_belief`
/// is `≤` the corresponding axis of `reference_belief` within
/// [`DEFAULT_BELIEF_COMPARISON_EPS`]. `reference_le_current` is the dual.
/// Both `true` means the two scores are componentwise equal within eps
/// (the diagonal of the partial order); both `false` means the relation
/// is incomparable (some axes moved up, others moved down).
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefPartialOrder {
    pub reference_belief: BeliefScore,
    pub current_belief: BeliefScore,
    pub current_le_reference: bool,
    pub reference_le_current: bool,
}

/// Coarse classification of a [`BeliefPartialOrder`]. Useful for CLI
/// emit, gate logic, and report summarisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeliefDirection {
    /// `current` and `reference` are componentwise equal within eps.
    Equal,
    /// `current ≤ reference` componentwise (within eps) and they are
    /// not equal — the dataset has degraded along at least one axis
    /// and improved along none. This is the operational concern: in
    /// the meet-semilattice, belief never gets *better* under further
    /// derivation, so this is the expected direction when something
    /// is going wrong.
    MonotonicDecrease,
    /// `reference ≤ current` componentwise (within eps) and they are
    /// not equal — every axis improved or stayed the same. Means a
    /// remediation pipeline ran cleanly between the two reports.
    MonotonicIncrease,
    /// Some axes increased, others decreased. The partial order does
    /// not classify the relationship; the two reports sit in
    /// incomparable positions in the lattice.
    Incomparable,
}

impl BeliefPartialOrder {
    /// Build a `BeliefPartialOrder` from two scores, using the supplied
    /// tolerance. Both directions of the componentwise ≤ relation are
    /// recorded so consumers can classify equality vs strict order vs
    /// incomparability.
    pub fn from_scores(reference: BeliefScore, current: BeliefScore, eps: f64) -> Self {
        let current_le_reference = le_componentwise(&current, &reference, eps);
        let reference_le_current = le_componentwise(&reference, &current, eps);
        Self {
            reference_belief: reference,
            current_belief: current,
            current_le_reference,
            reference_le_current,
        }
    }

    /// Classify the relationship into one of four `BeliefDirection` cases.
    pub fn direction(&self) -> BeliefDirection {
        match (self.current_le_reference, self.reference_le_current) {
            (true, true) => BeliefDirection::Equal,
            (true, false) => BeliefDirection::MonotonicDecrease,
            (false, true) => BeliefDirection::MonotonicIncrease,
            (false, false) => BeliefDirection::Incomparable,
        }
    }

    /// True iff `current_belief ≤ reference_belief` componentwise and
    /// they are not equal — belief monotonically degraded.
    pub fn is_monotonic_decrease(&self) -> bool {
        matches!(self.direction(), BeliefDirection::MonotonicDecrease)
    }

    /// True iff `reference_belief ≤ current_belief` componentwise and
    /// they are not equal — belief monotonically improved.
    pub fn is_monotonic_increase(&self) -> bool {
        matches!(self.direction(), BeliefDirection::MonotonicIncrease)
    }

    /// True iff the two scores are componentwise equal within eps.
    pub fn is_equal(&self) -> bool {
        matches!(self.direction(), BeliefDirection::Equal)
    }

    /// True iff the two scores are incomparable — some axes up, others down.
    pub fn is_incomparable(&self) -> bool {
        matches!(self.direction(), BeliefDirection::Incomparable)
    }
}

impl BeliefDirection {
    /// Stable short text label suitable for CLI emit.
    pub fn label(self) -> &'static str {
        match self {
            BeliefDirection::Equal => "equal",
            BeliefDirection::MonotonicDecrease => "monotonic_decrease",
            BeliefDirection::MonotonicIncrease => "monotonic_increase",
            BeliefDirection::Incomparable => "incomparable",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReportDiff {
    pub appeared: Vec<ValidationFinding>,
    pub disappeared: Vec<ValidationFinding>,
    pub unchanged: Vec<ValidationFinding>,
    pub ref_run_id: crate::id::FingerprintId,
    pub cur_run_id: crate::id::FingerprintId,
    /// v0.7 part 2: componentwise partial order between the two reports'
    /// belief scores under the meet-semilattice algebra. Computed via
    /// `belief_report_from_locke` (default `BeliefPenalty`). For
    /// custom penalty models, derive the scores separately and call
    /// [`BeliefPartialOrder::from_scores`] directly.
    pub belief_partial_order: BeliefPartialOrder,
    /// v0.7+ A3 (governance): `Some` when a `Policy` was attached via
    /// [`ReportDiff::with_policy`]; `None` for the default diff. The
    /// policy result is computed by applying the policy to the
    /// *current* report's findings (suppression, owner attribution,
    /// requirement evaluation). Independent of the finding-set diff
    /// fields above — both are useful signals; neither replaces the
    /// other.
    pub policy_result: Option<PolicyResult>,
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

    /// Convenience: classify the belief relationship between the two
    /// reports under the v0.7 part 1 meet-semilattice partial order.
    pub fn belief_direction(&self) -> BeliefDirection {
        self.belief_partial_order.direction()
    }

    /// v0.7+ A3: attach a [`Policy`] to this diff. The policy is
    /// applied to `current_report`'s findings (suppression + owner
    /// attribution + requirement evaluation) and stored as
    /// `policy_result`. Returns `self` for chaining:
    ///
    /// ```ignore
    /// let diff = diff_reports(&ref_report, &cur_report)
    ///     .with_policy(&policy, &cur_report);
    /// ```
    ///
    /// The diff's existing fields (`appeared`, `disappeared`,
    /// `unchanged`, belief partial order) are *not* affected — policy
    /// suppression is a separate channel.
    pub fn with_policy(mut self, policy: &Policy, current_report: &LockeReport) -> Self {
        self.policy_result = Some(apply_policy(current_report, policy));
        self
    }

    /// v0.7+ A3: true iff a policy is attached AND at least one
    /// requirement failed. Composes with [`Self::gate_fails`] — typical
    /// callers will check both:
    ///
    /// ```ignore
    /// if diff.gate_fails(FindingSeverity::Warning) || diff.policy_gate_fails() {
    ///     std::process::exit(1);
    /// }
    /// ```
    pub fn policy_gate_fails(&self) -> bool {
        self.policy_result
            .as_ref()
            .is_some_and(|p| p.gate_fails())
    }
}

/// Compute the structural diff between `reference` and `current`.
///
/// Findings are matched by content-addressed `id`. The output `Vec`s
/// are sorted by `sort_key()` for deterministic emission.
///
/// **v0.7 part 2 (ADR-0036)** — also computes a [`BeliefPartialOrder`]
/// classifying the meet-semilattice relationship between the two
/// reports' belief scores. Uses the default [`BeliefPenalty`](crate::belief::BeliefPenalty);
/// callers needing a custom penalty model should derive belief scores
/// via [`belief_report_from_locke_with_model`](crate::api::belief_report_from_locke_with_model)
/// and call [`BeliefPartialOrder::from_scores`] directly.
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

    // v0.7 part 2: classify the meet-semilattice partial order between
    // the two reports' belief scores. Default `BeliefPenalty` is used —
    // it's deterministic, byte-equivalent to the pre-v0.3 baked-in
    // values, and avoids broadening the `diff_reports` signature.
    let ref_belief = belief_report_from_locke(reference).score;
    let cur_belief = belief_report_from_locke(current).score;
    let belief_partial_order =
        BeliefPartialOrder::from_scores(ref_belief, cur_belief, DEFAULT_BELIEF_COMPARISON_EPS);

    ReportDiff {
        appeared,
        disappeared,
        unchanged,
        ref_run_id: reference.run_id,
        cur_run_id: current.run_id,
        belief_partial_order,
        // v0.7+ A3 (governance): policy is opt-in via `with_policy`.
        policy_result: None,
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
    // v0.7 part 2 — surface the belief partial-order classification.
    // Two-decimal precision keeps the line short and matches the
    // user-facing "overall" rendering elsewhere; the full f64 values
    // are available on `ReportDiff::belief_partial_order` for callers
    // that need bit-precise data.
    let bpo = &diff.belief_partial_order;
    s.push_str(&format!(
        "belief: direction={} ref_overall={:.3} cur_overall={:.3}\n",
        bpo.direction().label(),
        bpo.reference_belief.overall,
        bpo.current_belief.overall,
    ));
    // v0.7+ A3 — surface the policy result one-line summary when
    // attached. Full PolicyResult is available on
    // `ReportDiff::policy_result` for callers that need detail.
    if let Some(pr) = &diff.policy_result {
        let status = if pr.gate_fails() { "GATE FAIL" } else { "OK" };
        s.push_str(&format!(
            "policy: status={} suppressions={} attributions={} requirements={} fingerprint={}\n",
            status,
            pr.suppressions.len(),
            pr.attributions.len(),
            pr.requirements.len(),
            pr.policy_fingerprint,
        ));
    }
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

    // ─── v0.7 part 2 — belief partial-order tests ────────────────────

    #[test]
    fn diff_of_identical_reports_has_equal_belief_direction() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let r = validate(&df_clean(), &opts);
        let d = diff_reports(&r, &r);
        assert_eq!(d.belief_direction(), BeliefDirection::Equal);
        assert!(d.belief_partial_order.is_equal());
        assert!(d.belief_partial_order.current_le_reference);
        assert!(d.belief_partial_order.reference_le_current);
    }

    #[test]
    fn diff_with_more_missingness_shows_monotonic_decrease() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        // reference = clean dataset (high belief),
        // current = same dataset with 50% NaN (lower missingness_score, others unchanged
        // because the schema/duplication/etc. axes don't move under NaN injection).
        let d = diff_reports(&validate(&df_clean(), &opts), &validate(&df_with_nans(), &opts));
        assert_eq!(d.belief_direction(), BeliefDirection::MonotonicDecrease);
        assert!(d.belief_partial_order.is_monotonic_decrease());
        assert!(d.belief_partial_order.current_le_reference);
        assert!(!d.belief_partial_order.reference_le_current);
        // sanity: missingness genuinely degraded.
        assert!(
            d.belief_partial_order.current_belief.missingness_score
                < d.belief_partial_order.reference_belief.missingness_score
        );
    }

    #[test]
    fn diff_with_less_missingness_shows_monotonic_increase() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let d = diff_reports(&validate(&df_with_nans(), &opts), &validate(&df_clean(), &opts));
        assert_eq!(d.belief_direction(), BeliefDirection::MonotonicIncrease);
        assert!(d.belief_partial_order.is_monotonic_increase());
        assert!(!d.belief_partial_order.current_le_reference);
        assert!(d.belief_partial_order.reference_le_current);
    }

    #[test]
    fn belief_direction_is_deterministic_across_repeated_diffs() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let r1 = validate(&df_clean(), &opts);
        let r2 = validate(&df_with_nans(), &opts);
        let d1 = diff_reports(&r1, &r2);
        let d2 = diff_reports(&r1, &r2);
        assert_eq!(d1.belief_partial_order, d2.belief_partial_order);
        assert_eq!(d1.belief_direction(), d2.belief_direction());
    }

    #[test]
    fn diff_text_includes_belief_line() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let d = diff_reports(&validate(&df_clean(), &opts), &validate(&df_with_nans(), &opts));
        let s = emit_diff_text(&d);
        assert!(s.contains("belief: direction="));
        // Defensive: the direction label must be one of the stable enum strings.
        assert!(
            s.contains("direction=monotonic_decrease")
                || s.contains("direction=monotonic_increase")
                || s.contains("direction=equal")
                || s.contains("direction=incomparable")
        );
    }

    #[test]
    fn belief_partial_order_label_round_trips_via_enum() {
        assert_eq!(BeliefDirection::Equal.label(), "equal");
        assert_eq!(BeliefDirection::MonotonicDecrease.label(), "monotonic_decrease");
        assert_eq!(BeliefDirection::MonotonicIncrease.label(), "monotonic_increase");
        assert_eq!(BeliefDirection::Incomparable.label(), "incomparable");
    }

    #[test]
    fn from_scores_with_hand_built_inputs_classifies_incomparable() {
        // Hand-built scores: current is *better* on missingness, *worse*
        // on duplication. Neither dominates the other — incomparable.
        let reference = BeliefScore::from_dimensions(
            1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        let current = BeliefScore::from_dimensions(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0,
        );
        let bpo = BeliefPartialOrder::from_scores(reference, current, 1e-12);
        assert_eq!(bpo.direction(), BeliefDirection::Incomparable);
        assert!(bpo.is_incomparable());
        assert!(!bpo.current_le_reference);
        assert!(!bpo.reference_le_current);
    }

    // ─── v0.7+ A3 — policy integration ────────────────────────────────

    #[test]
    fn diff_default_has_no_policy_result() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let d = diff_reports(&validate(&df_clean(), &opts), &validate(&df_with_nans(), &opts));
        assert!(d.policy_result.is_none());
        assert!(!d.policy_gate_fails());
    }

    #[test]
    fn with_policy_populates_policy_result_and_drops_matched_findings() {
        use crate::policy::{Policy, SuppressionRule};
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let cur = validate(&df_with_nans(), &opts);
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![],
            requirements: vec![],
        };
        let d = diff_reports(&validate(&df_clean(), &opts), &cur).with_policy(&policy, &cur);
        let pr = d.policy_result.as_ref().expect("policy_result populated");
        assert!(!pr.suppressions.is_empty());
        assert!(pr.remaining_findings.iter().all(|f| f.code != "E9001"));
        assert!(!d.policy_gate_fails()); // no requirements → no gate failure
    }

    #[test]
    fn policy_gate_fails_when_required_finding_violation_present() {
        use crate::policy::{Policy, RequiredFindingRule, RequirementOperator};
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let cur = validate(&df_with_nans(), &opts);
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
        let d = diff_reports(&validate(&df_clean(), &opts), &cur).with_policy(&policy, &cur);
        assert!(d.policy_gate_fails());
        let pr = d.policy_result.unwrap();
        assert!(!pr.all_requirements_satisfied());
    }

    #[test]
    fn emit_diff_text_surfaces_policy_status_line_when_attached() {
        use crate::policy::{Policy, RequiredFindingRule, RequirementOperator};
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let cur = validate(&df_with_nans(), &opts);
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
        let d = diff_reports(&validate(&df_clean(), &opts), &cur).with_policy(&policy, &cur);
        let s = emit_diff_text(&d);
        assert!(s.contains("policy: status=GATE FAIL"));
        assert!(s.contains("suppressions=0"));
        assert!(s.contains("requirements=1"));
        assert!(s.contains("fingerprint="));
    }

    #[test]
    fn emit_diff_text_does_not_include_policy_line_when_unattached() {
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let d = diff_reports(&validate(&df_clean(), &opts), &validate(&df_with_nans(), &opts));
        let s = emit_diff_text(&d);
        assert!(!s.contains("policy:"));
    }

    #[test]
    fn with_policy_is_deterministic_across_calls() {
        use crate::policy::{Policy, SuppressionRule};
        let opts = ValidateOptions {
            dataset_label: "t".into(),
            ..Default::default()
        };
        let cur = validate(&df_with_nans(), &opts);
        let policy = Policy {
            suppressions: vec![SuppressionRule {
                code: "E9001".into(),
                column: None,
                reason: "ack".into(),
            }],
            owners: vec![],
            requirements: vec![],
        };
        let d1 = diff_reports(&validate(&df_clean(), &opts), &cur).with_policy(&policy, &cur);
        let d2 = diff_reports(&validate(&df_clean(), &opts), &cur).with_policy(&policy, &cur);
        assert_eq!(d1.policy_result, d2.policy_result);
    }
}
