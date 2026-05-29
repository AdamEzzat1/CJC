//! Integration tests for the v0.7+ A3 governance policy layer.
//!
//! These exercise the policy DSL end-to-end through the same DataFrame +
//! validate pipeline that real callers use: build a `LockeReport`, hand
//! it to `apply_policy(report, policy)`, and assert the suppression /
//! attribution / requirement signals match the diabetes-130-shaped
//! governance scenarios from the §A3 brief.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    apply_policy, belief_report_from_locke, diff_reports, emit_policy_result_text,
    BeliefDirection, OwnerRule, Policy, RequiredFindingRule, RequirementOperator,
    SuppressionRule,
};
use cjc_locke::api::{validate, ValidateOptions};

// ─── Helpers ──────────────────────────────────────────────────────────────

fn df_with_nans(n: usize, frac_nan: f64) -> DataFrame {
    let cutoff = (n as f64 * frac_nan) as usize;
    let xs: Vec<f64> = (0..n)
        .map(|i| if i < cutoff { f64::NAN } else { i as f64 })
        .collect();
    DataFrame::from_columns(vec![("x".into(), Column::Float(xs))]).unwrap()
}

fn df_clean(n: usize) -> DataFrame {
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    DataFrame::from_columns(vec![("x".into(), Column::Float(xs))]).unwrap()
}

fn validate_default(df: &DataFrame, label: &str) -> cjc_locke::LockeReport {
    let opts = ValidateOptions {
        dataset_label: label.into(),
        ..Default::default()
    };
    validate(df, &opts)
}

// ─── End-to-end suppression scenarios ─────────────────────────────────────

#[test]
fn e9001_suppression_drops_missingness_finding_end_to_end() {
    let df = df_with_nans(100, 0.5);
    let report = validate_default(&df, "missingness_scenario");

    let policy = Policy {
        suppressions: vec![SuppressionRule {
            code: "E9001".into(),
            column: None,
            reason: "50% missingness on this dev fixture is expected".into(),
        }],
        owners: vec![],
        requirements: vec![],
    };

    let result = apply_policy(&report, &policy);
    assert!(!result.suppressions.is_empty());
    assert!(result.remaining_findings.iter().all(|f| f.code != "E9001"));
}

#[test]
fn column_specific_suppression_does_not_drop_other_columns() {
    let mut v1: Vec<f64> = (0..100).map(|i| i as f64).collect();
    for x in v1.iter_mut().take(60) {
        *x = f64::NAN;
    }
    let mut v2: Vec<f64> = (0..100).map(|i| i as f64).collect();
    for x in v2.iter_mut().take(60) {
        *x = f64::NAN;
    }
    let df = DataFrame::from_columns(vec![
        ("acknowledged".into(), Column::Float(v1)),
        ("real_issue".into(), Column::Float(v2)),
    ])
    .unwrap();
    let report = validate_default(&df, "two_columns");

    // Suppress only the acknowledged column.
    let policy = Policy {
        suppressions: vec![SuppressionRule {
            code: "E9001".into(),
            column: Some("acknowledged".into()),
            reason: "known dev placeholder column".into(),
        }],
        owners: vec![],
        requirements: vec![],
    };
    let result = apply_policy(&report, &policy);
    // Findings on "real_issue" must remain.
    assert!(result
        .remaining_findings
        .iter()
        .any(|f| f.code == "E9001" && f.column.as_deref() == Some("real_issue")));
    // No findings on "acknowledged" should remain.
    assert!(result
        .remaining_findings
        .iter()
        .all(|f| f.column.as_deref() != Some("acknowledged") || f.code != "E9001"));
}

// ─── Owner attribution ────────────────────────────────────────────────────

#[test]
fn owner_attribution_groups_findings_by_team() {
    let df = df_with_nans(100, 0.3);
    let report = validate_default(&df, "owner_scenario");
    let policy = Policy {
        suppressions: vec![],
        owners: vec![OwnerRule {
            team: "data-platform".into(),
            column: Some("x".into()),
            code: None,
        }],
        requirements: vec![],
    };
    let result = apply_policy(&report, &policy);
    let by_team = result.attributions_by_team();
    let data_platform = by_team.get("data-platform").expect("data-platform team present");
    assert!(!data_platform.is_empty());
}

// ─── Required-finding policies + gate failure ────────────────────────────

#[test]
fn requirement_gate_fails_when_threshold_violated() {
    let df = df_with_nans(100, 0.5);
    let report = validate_default(&df, "gate_scenario");
    let policy = Policy {
        suppressions: vec![],
        owners: vec![],
        requirements: vec![RequiredFindingRule {
            code: "E9001".into(),
            operator: RequirementOperator::EqZero,
            threshold: 0,
            owner: Some("data-platform".into()),
        }],
    };
    let result = apply_policy(&report, &policy);
    assert!(result.gate_fails());
    assert!(!result.all_requirements_satisfied());
    assert_eq!(result.requirements[0].observed, 1); // one E9001 finding
}

#[test]
fn requirement_gate_passes_when_threshold_satisfied_after_suppression() {
    let df = df_with_nans(100, 0.5);
    let report = validate_default(&df, "gated_after_suppress");
    let policy = Policy {
        suppressions: vec![SuppressionRule {
            code: "E9001".into(),
            column: None,
            reason: "acknowledged".into(),
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
fn requirement_le_passes_when_observed_within_threshold() {
    let df = df_with_nans(100, 0.5);
    let report = validate_default(&df, "le_test");
    let policy = Policy {
        suppressions: vec![],
        owners: vec![],
        requirements: vec![RequiredFindingRule {
            code: "E9001".into(),
            operator: RequirementOperator::LessOrEqual,
            threshold: 5,
            owner: None,
        }],
    };
    let result = apply_policy(&report, &policy);
    assert!(result.all_requirements_satisfied());
}

// ─── Gate integration ────────────────────────────────────────────────────

#[test]
fn diff_with_policy_attaches_policy_result_and_gates_via_policy_gate_fails() {
    let clean = validate_default(&df_clean(100), "clean");
    let dirty = validate_default(&df_with_nans(100, 0.3), "dirty");
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
    let diff = diff_reports(&clean, &dirty).with_policy(&policy, &dirty);
    assert!(diff.policy_result.is_some());
    assert!(diff.policy_gate_fails());
    // The pre-existing belief direction signal is preserved alongside.
    assert_eq!(diff.belief_direction(), BeliefDirection::MonotonicDecrease);
}

// ─── Determinism ─────────────────────────────────────────────────────────

#[test]
fn apply_policy_is_byte_identical_across_runs() {
    let df = df_with_nans(50, 0.4);
    let report = validate_default(&df, "det");
    let policy = Policy {
        suppressions: vec![SuppressionRule {
            code: "E9001".into(),
            column: None,
            reason: "acknowledged".into(),
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
fn policy_fingerprint_changes_when_rule_order_changes() {
    let s1 = SuppressionRule {
        code: "E9001".into(),
        column: None,
        reason: "ack a".into(),
    };
    let s2 = SuppressionRule {
        code: "E9080".into(),
        column: None,
        reason: "ack b".into(),
    };
    let p_ab = Policy {
        suppressions: vec![s1.clone(), s2.clone()],
        owners: vec![],
        requirements: vec![],
    };
    let p_ba = Policy {
        suppressions: vec![s2, s1],
        owners: vec![],
        requirements: vec![],
    };
    assert_ne!(p_ab.fingerprint(), p_ba.fingerprint());
}

#[test]
fn decision_id_is_stable_for_same_rule_and_finding() {
    let df = df_with_nans(50, 0.4);
    let report = validate_default(&df, "dec");
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

// ─── Emit ─────────────────────────────────────────────────────────────────

#[test]
fn emit_policy_result_text_is_well_formed_and_deterministic() {
    let df = df_with_nans(50, 0.4);
    let report = validate_default(&df, "emit");
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
}

// ─── Belief composition: policy doesn't perturb belief score ──────────────

#[test]
fn applying_policy_does_not_change_belief_report_for_same_data() {
    let df = df_with_nans(100, 0.3);
    let report = validate_default(&df, "belief_invariant");
    let policy = Policy {
        suppressions: vec![SuppressionRule {
            code: "E9001".into(),
            column: None,
            reason: "ack".into(),
        }],
        owners: vec![],
        requirements: vec![],
    };
    // apply_policy filters findings, but the original LockeReport is
    // unchanged. The belief report computed from the original report
    // is unchanged regardless of policy.
    let belief_pre = belief_report_from_locke(&report);
    let _ = apply_policy(&report, &policy);
    let belief_post = belief_report_from_locke(&report);
    assert_eq!(belief_pre, belief_post);
}
