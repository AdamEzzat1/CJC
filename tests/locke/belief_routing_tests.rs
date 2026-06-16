//! v0.9 belief-arc tests: the leakage/drift axis wiring (#1), the
//! exhaustiveness guard over the routing registry (#2), and the noisy-OR
//! penalty curve (#3).
//!
//! These three changes share one theme — *the belief number must mean what
//! it says*. The wiring stops the leakage axis from reading 1.0 ("perfect")
//! while a leakage Error sits in the same report; the guard stops a future
//! detector from silently bypassing every belief axis; the curve stops the
//! "4 errors and 40 errors score identically" saturation.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::belief::{penalty_from_findings_with_model, BeliefPenalty};
use cjc_locke::belief_routing::{builtin_axis_for, ADVISORY_CODES, ALL_BUILTIN_CODES};
use cjc_locke::custom_detector::BeliefAxisSet;
use cjc_locke::leakage::{detect_target_leakage, LeakageConfig};
use cjc_locke::report::{FindingSeverity, LockeReport, ValidationFinding};

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// A clean single-column report — every axis starts at 1.0.
fn clean_report() -> LockeReport {
    let v: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
    validate(
        &df,
        &ValidateOptions {
            dataset_label: "clean".into(),
            ..Default::default()
        },
    )
}

/// Build a finding with no evidence/column scoping — enough to exercise
/// the belief routing.
fn finding(code: &'static str, sev: FindingSeverity) -> ValidationFinding {
    ValidationFinding::new(code, sev, "synthetic", None, None, vec![], 200, vec![], vec![])
}

// ----------------------------------------------------------------------
// #2 — exhaustiveness guard over the registry
// ----------------------------------------------------------------------

#[test]
fn every_builtin_code_is_routed_or_advisory() {
    // The whole point of the registry: no emitted code can be silently
    // unclassified. A new detector whose code lands in ALL_BUILTIN_CODES
    // but is forgotten by both `builtin_axis_for` and ADVISORY_CODES fails
    // here — forcing a deliberate "which belief axis?" decision.
    for &code in ALL_BUILTIN_CODES {
        let routed = !builtin_axis_for(code).is_empty();
        let advisory = ADVISORY_CODES.contains(&code);
        assert!(
            routed ^ advisory,
            "{code}: must be EITHER routed (axis={}) XOR advisory (advisory={}) — got routed={routed}, advisory={advisory}",
            builtin_axis_for(code),
            advisory,
        );
    }
}

#[test]
fn advisory_codes_are_all_known() {
    for &code in ADVISORY_CODES {
        assert!(
            ALL_BUILTIN_CODES.contains(&code),
            "{code} is in ADVISORY_CODES but not ALL_BUILTIN_CODES"
        );
        assert!(
            builtin_axis_for(code).is_empty(),
            "{code} is advisory but builtin_axis_for routes it to {}",
            builtin_axis_for(code)
        );
    }
}

#[test]
fn all_builtin_codes_unique_and_sorted() {
    let mut sorted = ALL_BUILTIN_CODES.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ALL_BUILTIN_CODES.len(),
        "ALL_BUILTIN_CODES has duplicates"
    );
    assert_eq!(
        sorted.as_slice(),
        ALL_BUILTIN_CODES,
        "ALL_BUILTIN_CODES must be sorted (it is the canonical registry list)"
    );
}

#[test]
fn leakage_and_drift_codes_are_actually_routed() {
    // Regression lock for the v0.9 fix: these specific codes must reach
    // their axes. If a refactor drops one, the headline bug returns.
    for code in ["E9060", "E9061", "E9063", "E9064"] {
        assert!(
            builtin_axis_for(code).contains(BeliefAxisSet::LEAKAGE),
            "{code} must route to the leakage axis"
        );
    }
    for code in ["E9018", "E9030", "E9039", "E9110"] {
        assert!(
            builtin_axis_for(code).contains(BeliefAxisSet::DRIFT),
            "{code} must route to the drift axis"
        );
    }
    // E9062 ("target not binary — skipped") is a diagnostic, NOT leakage
    // evidence — it must stay advisory or the axis would drop on a
    // non-binary target with no actual leakage.
    assert!(builtin_axis_for("E9062").is_empty());
}

// ----------------------------------------------------------------------
// #1 — the leakage/drift axis wiring (the headline fix)
// ----------------------------------------------------------------------

#[test]
fn clean_report_has_perfect_leakage_and_drift() {
    let b = belief_report_from_locke(&clean_report());
    assert_eq!(b.score.leakage_score, 1.0);
    assert_eq!(b.score.drift_score, 1.0);
}

#[test]
fn spliced_leakage_error_drags_leakage_score() {
    // THE bug, reproduced and fixed: pre-v0.9 this returned 1.0.
    let mut report = clean_report();
    let before = belief_report_from_locke(&report).score.leakage_score;
    assert_eq!(before, 1.0);

    report.findings.push(finding("E9060", FindingSeverity::Error));
    let after = belief_report_from_locke(&report).score.leakage_score;
    assert!(
        after < 1.0,
        "leakage_score must drop once an E9060 Error is present, got {after}"
    );
    // One Error under the default model: survival = 1 - 0.25 = 0.75.
    assert!((after - 0.75).abs() < 1e-12, "expected 0.75, got {after}");
}

#[test]
fn spliced_drift_finding_drags_drift_score() {
    let mut report = clean_report();
    report.findings.push(finding("E9039", FindingSeverity::Warning));
    let after = belief_report_from_locke(&report).score.drift_score;
    assert!((after - 0.90).abs() < 1e-12, "one Warning -> 0.90, got {after}");
}

#[test]
fn end_to_end_detector_leakage_drags_axis() {
    // Realistic path: a feature that perfectly separates a binary target
    // (AUC = 1.0 -> E9060), through the real detector, spliced into the
    // report exactly as the LendingClub demo's `run_locke_audit` does.
    let target: Vec<i64> = (0..100).map(|i| (i >= 50) as i64).collect();
    let feature: Vec<f64> = (0..100).map(|i| if i >= 50 { 1.0 } else { 0.0 }).collect();
    let df = DataFrame::from_columns(vec![
        ("y".into(), Column::Int(target)),
        ("leaky".into(), Column::Float(feature)),
    ])
    .unwrap();

    let leak = detect_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(
        leak.iter().any(|f| f.code == "E9060"),
        "expected E9060 on a perfectly-separating feature, got {:?}",
        leak.iter().map(|f| f.code).collect::<Vec<_>>()
    );

    let mut report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "leaky".into(),
            ..Default::default()
        },
    );
    report.findings.extend(leak);
    let b = belief_report_from_locke(&report);
    assert!(
        b.score.leakage_score < 1.0,
        "leakage axis must reflect the spliced E9060, got {}",
        b.score.leakage_score
    );
    // And the stale "Locke does not infer leakage" caveat must be gone.
    assert!(
        !b.assumptions.iter().any(|a| a.contains("leakage_score = 1.0")),
        "the leakage caveat must be suppressed once a leakage finding is present: {:?}",
        b.assumptions
    );
}

#[test]
fn leakage_caveat_present_only_when_no_leakage() {
    let clean = belief_report_from_locke(&clean_report());
    assert!(
        clean.assumptions.iter().any(|a| a.contains("leakage_score = 1.0")),
        "a clean report should still disclaim the unmeasured leakage axis"
    );
}

// ----------------------------------------------------------------------
// #3 — noisy-OR penalty curve
// ----------------------------------------------------------------------

#[test]
fn single_finding_is_byte_identical_to_linear() {
    // The curve must not move single-finding axes (1 - (1 - p) = p).
    let mut report = clean_report();
    report.findings.push(finding("E9060", FindingSeverity::Error));
    let s = belief_report_from_locke(&report).score.leakage_score;
    assert_eq!(s.to_bits(), 0.75_f64.to_bits());
}

#[test]
fn noisy_or_preserves_ordering_at_the_bad_end() {
    // The motivating defect: the old linear model saturated — 4 Errors and
    // 40 Errors both hit exactly 0.0. The product keeps them ordered.
    let mut r4 = clean_report();
    for _ in 0..4 {
        r4.findings.push(finding("E9060", FindingSeverity::Error));
    }
    let mut r40 = clean_report();
    for _ in 0..40 {
        r40.findings.push(finding("E9060", FindingSeverity::Error));
    }
    let s4 = belief_report_from_locke(&r4).score.leakage_score;
    let s40 = belief_report_from_locke(&r40).score.leakage_score;
    assert!(s4 > 0.0 && s40 > 0.0, "both must stay strictly positive: {s4}, {s40}");
    assert!(s40 < s4, "40 errors must score strictly below 4: {s40} !< {s4}");
    // 4 errors: 0.75^4 = 0.31640625
    assert!((s4 - 0.75_f64.powi(4)).abs() < 1e-12, "got {s4}");
}

#[test]
fn info_penalty_is_strictly_below_notice() {
    // Restored severity rank: an axis with an Info finding scores strictly
    // higher than the same axis with a Notice finding.
    let model = BeliefPenalty::default();
    let info = vec![finding("E9014", FindingSeverity::Info)];
    let notice = vec![finding("E9014", FindingSeverity::Notice)];
    let p_info = penalty_from_findings_with_model(&info, |c| c == "E9014", &model);
    let p_notice = penalty_from_findings_with_model(&notice, |c| c == "E9014", &model);
    assert!(p_info < p_notice, "info {p_info} must be < notice {p_notice}");
    assert_eq!(p_info.to_bits(), 0.01_f64.to_bits());
    assert_eq!(p_notice.to_bits(), 0.02_f64.to_bits());
}

#[test]
fn penalty_stays_in_unit_interval_under_many_findings() {
    let many: Vec<ValidationFinding> = (0..1000)
        .map(|_| finding("E9060", FindingSeverity::Error))
        .collect();
    let p = penalty_from_findings_with_model(&many, |c| c == "E9060", &BeliefPenalty::default());
    assert!((0.0..=1.0).contains(&p), "penalty out of [0,1]: {p}");
}

#[test]
fn belief_derivation_is_deterministic_with_spliced_findings() {
    let mut report = clean_report();
    report.findings.push(finding("E9060", FindingSeverity::Error));
    report.findings.push(finding("E9039", FindingSeverity::Warning));
    let a = belief_report_from_locke(&report).score;
    let b = belief_report_from_locke(&report).score;
    assert_eq!(a.leakage_score.to_bits(), b.leakage_score.to_bits());
    assert_eq!(a.drift_score.to_bits(), b.drift_score.to_bits());
    assert_eq!(a.overall.to_bits(), b.overall.to_bits());
}

// ----------------------------------------------------------------------
// Property tests (proptest) — the noisy-OR penalty laws
// ----------------------------------------------------------------------

use proptest::prelude::*;

/// One severity, drawn uniformly. Maps to a default-model penalty pᵢ.
fn arb_severity() -> impl Strategy<Value = FindingSeverity> {
    prop_oneof![
        Just(FindingSeverity::Info),
        Just(FindingSeverity::Notice),
        Just(FindingSeverity::Warning),
        Just(FindingSeverity::Error),
    ]
}

proptest! {
    /// Penalty always stays in [0, 1] for any multiset of findings — the
    /// product of factors in [0,1] can never escape the unit interval.
    #[test]
    fn noisy_or_penalty_in_unit_interval(sevs in prop::collection::vec(arb_severity(), 0..200)) {
        let findings: Vec<ValidationFinding> =
            sevs.iter().map(|&s| finding("E9060", s)).collect();
        let p = penalty_from_findings_with_model(&findings, |c| c == "E9060", &BeliefPenalty::default());
        prop_assert!((0.0..=1.0).contains(&p), "penalty out of range: {}", p);
    }

    /// Monotonicity: appending one more matching finding never DECREASES
    /// the penalty (an axis never improves by finding more defects). This
    /// is the property the old saturating model violated at the bad end
    /// (it was flat at 0, not strictly monotone — here we assert the
    /// weaker but always-true "non-decreasing").
    #[test]
    fn noisy_or_penalty_monotone_in_count(
        sevs in prop::collection::vec(arb_severity(), 0..100),
        extra in arb_severity(),
    ) {
        let base: Vec<ValidationFinding> = sevs.iter().map(|&s| finding("E9060", s)).collect();
        let mut more = base.clone();
        more.push(finding("E9060", extra));
        let f = |c: &str| c == "E9060";
        let p_base = penalty_from_findings_with_model(&base, f, &BeliefPenalty::default());
        let p_more = penalty_from_findings_with_model(&more, f, &BeliefPenalty::default());
        prop_assert!(p_more >= p_base - 1e-12, "{} < {}", p_more, p_base);
    }

    /// Backward-compat: for exactly one finding, noisy-OR equals the old
    /// linear value pᵢ bit-for-bit (1 - (1 - p) = p).
    #[test]
    fn noisy_or_single_finding_equals_linear(s in arb_severity()) {
        let one = vec![finding("E9060", s)];
        let p = penalty_from_findings_with_model(&one, |c| c == "E9060", &BeliefPenalty::default());
        prop_assert_eq!(p.to_bits(), BeliefPenalty::default().for_severity(s).to_bits());
    }

    /// Determinism: same findings → bit-identical penalty, every time.
    #[test]
    fn noisy_or_penalty_is_deterministic(sevs in prop::collection::vec(arb_severity(), 0..120)) {
        let findings: Vec<ValidationFinding> = sevs.iter().map(|&s| finding("E9060", s)).collect();
        let f = |c: &str| c == "E9060";
        let a = penalty_from_findings_with_model(&findings, f, &BeliefPenalty::default());
        let b = penalty_from_findings_with_model(&findings, f, &BeliefPenalty::default());
        prop_assert_eq!(a.to_bits(), b.to_bits());
    }

    /// The leakage axis never reads as "perfect" once any leakage Error is
    /// present, for any number of unrelated clean columns.
    #[test]
    fn leakage_axis_never_perfect_with_a_leakage_error(n_clean in 0usize..8) {
        let mut cols: Vec<(String, Column)> = Vec::new();
        for i in 0..=n_clean {
            cols.push((format!("c{i}"), Column::Float((0..50).map(|x| x as f64).collect())));
        }
        let df = DataFrame::from_columns(cols).unwrap();
        let mut report = validate(&df, &ValidateOptions { dataset_label: "p".into(), ..Default::default() });
        report.findings.push(finding("E9060", FindingSeverity::Error));
        let s = belief_report_from_locke(&report).score.leakage_score;
        prop_assert!(s < 1.0, "leakage axis stayed perfect with an E9060 present: {}", s);
    }
}

// ----------------------------------------------------------------------
// Structural fuzz (bolero) — no panic, always-in-[0,1], total routing
// ----------------------------------------------------------------------

fn sev_from_byte(b: u8) -> FindingSeverity {
    match b % 4 {
        0 => FindingSeverity::Info,
        1 => FindingSeverity::Notice,
        2 => FindingSeverity::Warning,
        _ => FindingSeverity::Error,
    }
}

#[test]
fn fuzz_penalty_aggregation_bounded_and_finite() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|bytes: &Vec<u8>| {
            let findings: Vec<ValidationFinding> = bytes
                .iter()
                .map(|&b| finding("E9060", sev_from_byte(b)))
                .collect();
            let p = penalty_from_findings_with_model(
                &findings,
                |c| c == "E9060",
                &BeliefPenalty::default(),
            );
            assert!(p.is_finite(), "penalty not finite: {p}");
            assert!((0.0..=1.0).contains(&p), "penalty out of [0,1]: {p}");
        });
}

#[test]
fn fuzz_builtin_axis_for_never_panics_on_arbitrary_code() {
    // The routing function must total over ALL &str — a malformed or
    // unknown code returns NONE (advisory), never panics.
    bolero::check!()
        .with_type::<String>()
        .for_each(|s: &String| {
            let _ = builtin_axis_for(s);
        });
}

#[test]
fn fuzz_spliced_findings_keep_all_axes_in_unit_interval() {
    // Arbitrary (code-selector, severity) pairs spliced into a real report
    // must leave every belief axis in [0, 1] and never panic — covers the
    // newly-wired drift/leakage axes under adversarial finding multisets.
    bolero::check!()
        .with_type::<Vec<(u8, u8)>>()
        .for_each(|ops: &Vec<(u8, u8)>| {
            let v: Vec<f64> = (0..30).map(|i| i as f64).collect();
            let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
            let mut report = validate(
                &df,
                &ValidateOptions { dataset_label: "fz".into(), ..Default::default() },
            );
            // Pick from a spread of routed codes across every axis.
            const CODES: &[&str] =
                &["E9003", "E9020", "E9014", "E9039", "E9060", "E9110", "E9064"];
            for &(c, s) in ops.iter().take(64) {
                let code = CODES[(c as usize) % CODES.len()];
                report.findings.push(finding(code, sev_from_byte(s)));
            }
            let b = belief_report_from_locke(&report).score;
            for axis in [
                b.schema_score, b.missingness_score, b.drift_score, b.leakage_score,
                b.lineage_score, b.sample_score, b.duplication_score, b.constraint_score,
                b.overall,
            ] {
                assert!(axis.is_finite() && (0.0..=1.0).contains(&axis), "axis out of range: {axis}");
            }
        });
}
