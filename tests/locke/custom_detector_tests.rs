//! Integration tests for the custom detector extension layer (ADR-0041).
//!
//! Covers:
//! - End-to-end: a registered custom detector contributes findings to a
//!   `LockeReport` and routes them to the right belief axis.
//! - Determinism: same dataframe + same detectors → byte-identical JSON.
//! - Sort merge: built-in + custom findings end up in canonical order.
//! - Namespace: invalid codes (E < 9500, E > 9999, malformed) are
//!   rejected at registration and do not affect the report.
//! - Proptest: arbitrary detector code + axes + emission count never
//!   produces non-deterministic output.
//! - Bolero: arbitrary detector code strings never panic and either
//!   produce a finding or surface a structured error.

use std::sync::Arc;

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::custom_detector::{
    run_custom_detectors, validate_custom_code, BeliefAxisSet, CustomDetector, CustomDetectorError,
    FindingSink,
};
use cjc_locke::emit_locke_report_json;
use cjc_locke::report::FindingSeverity;
use proptest::prelude::*;

// ─── Helpers ──────────────────────────────────────────────────────────

#[derive(Debug)]
struct NameMatchDetector {
    code: &'static str,
    axes: BeliefAxisSet,
    pattern: String,
}

impl NameMatchDetector {
    fn new(code: &'static str, axes: BeliefAxisSet, pattern: impl Into<String>) -> Self {
        Self {
            code,
            axes,
            pattern: pattern.into(),
        }
    }
}

impl CustomDetector for NameMatchDetector {
    fn code(&self) -> &'static str {
        self.code
    }
    fn belief_axes(&self) -> BeliefAxisSet {
        self.axes
    }
    fn run(&self, df: &DataFrame, sink: &mut FindingSink) {
        for (name, _col) in &df.columns {
            if name.starts_with(&self.pattern) {
                sink.emit(
                    FindingSeverity::Warning,
                    format!("`{}` matches pattern `{}*`", name, self.pattern),
                    Some(name.clone()),
                    None,
                    vec![],
                    df.nrows() as u64,
                );
            }
        }
    }
}

fn tiny_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("loan_amnt".into(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("total_pymnt".into(), Column::Float(vec![0.5, 1.0, 2.5])),
        ("total_rec_int".into(), Column::Float(vec![0.1, 0.2, 0.3])),
        ("dti".into(), Column::Float(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap()
}

// ─── Unit-style integration tests ─────────────────────────────────────

#[test]
fn end_to_end_custom_detector_contributes_to_report() {
    let df = tiny_df();
    let detector: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9500",
        BeliefAxisSet::LEAKAGE,
        "total_",
    ));
    let opts = ValidateOptions {
        dataset_label: "test".into(),
        custom_detectors: vec![detector],
        ..Default::default()
    };
    let report = validate(&df, &opts);

    // Detector should have flagged 2 columns (total_pymnt, total_rec_int).
    let e9500_findings: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.code == "E9500")
        .collect();
    assert_eq!(e9500_findings.len(), 2);
    let columns: Vec<_> = e9500_findings
        .iter()
        .filter_map(|f| f.column.as_deref())
        .collect();
    assert!(columns.contains(&"total_pymnt"));
    assert!(columns.contains(&"total_rec_int"));

    // Axis assignment landed in the report.
    assert_eq!(
        report.custom_axis_assignments.get("E9500").copied(),
        Some(BeliefAxisSet::LEAKAGE)
    );

    // Belief composition picked up the leakage axis.
    let belief = belief_report_from_locke(&report);
    assert!(belief.score.leakage_score < 1.0);
    // Sanity: other axes not affected by the leakage-flagged custom detector.
    assert_eq!(belief.score.drift_score, 1.0);
    assert_eq!(belief.score.lineage_score, 1.0);
}

#[test]
fn detector_with_no_axes_does_not_affect_belief() {
    let df = tiny_df();
    let detector: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9501",
        BeliefAxisSet::NONE,
        "total_",
    ));
    let opts = ValidateOptions {
        custom_detectors: vec![detector],
        ..Default::default()
    };
    let report = validate(&df, &opts);

    // The detector ran but its non-Info findings were dropped by the
    // sink. The advisory-only contract: Warning + NONE axes = rejected.
    let e9501_findings: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.code == "E9501")
        .collect();
    assert!(e9501_findings.is_empty(), "non-Info + NONE axes should drop");
}

#[test]
fn detector_with_invalid_code_is_skipped_silently() {
    let df = tiny_df();
    let detector: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9001", // built-in range
        BeliefAxisSet::LEAKAGE,
        "total_",
    ));
    let opts = ValidateOptions {
        custom_detectors: vec![detector],
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // No E9001 findings should be present because the detector was skipped.
    let e9001_findings: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.code == "E9001")
        .collect();
    assert!(e9001_findings.is_empty());
    // The report carries an assumption surfacing the rejection.
    assert!(report
        .assumptions
        .iter()
        .any(|a| a.contains("rejected at registration")));
}

#[test]
fn custom_findings_are_sorted_with_built_ins() {
    // Use a DataFrame with a Float column containing NaN so built-in
    // E9001 (missingness) fires alongside the custom detector.
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Float(vec![1.0, f64::NAN, 3.0, f64::NAN])),
        ("b".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0])),
        ("total_x".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();
    let detector: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9500",
        BeliefAxisSet::LEAKAGE,
        "total_",
    ));
    let opts = ValidateOptions {
        custom_detectors: vec![detector],
        ..Default::default()
    };
    let report = validate(&df, &opts);

    // Findings sorted ascending by (severity, code, column, id).
    // Verify no two consecutive findings are out of order.
    let pairs: Vec<_> = report.findings.iter().collect();
    for w in pairs.windows(2) {
        assert!(w[0].sort_key() <= w[1].sort_key());
    }
    // Both built-in E9001 and custom E9500 must be present.
    let codes: Vec<&str> = report.findings.iter().map(|f| f.code).collect();
    assert!(codes.contains(&"E9001"));
    assert!(codes.contains(&"E9500"));
}

#[test]
fn determinism_two_runs_byte_identical() {
    let df = tiny_df();
    let detectors: Vec<Arc<dyn CustomDetector>> = vec![
        Arc::new(NameMatchDetector::new(
            "E9500",
            BeliefAxisSet::LEAKAGE,
            "total_",
        )),
        Arc::new(NameMatchDetector::new(
            "E9501",
            BeliefAxisSet::SCHEMA,
            "loan_",
        )),
    ];
    let opts = ValidateOptions {
        custom_detectors: detectors,
        ..Default::default()
    };
    let r1 = validate(&df, &opts);
    let r2 = validate(&df, &opts);
    assert_eq!(emit_locke_report_json(&r1), emit_locke_report_json(&r2));
    assert_eq!(r1.run_id, r2.run_id);
}

#[test]
fn registration_order_does_not_affect_output() {
    let df = tiny_df();
    let d1: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9501",
        BeliefAxisSet::LEAKAGE,
        "total_",
    ));
    let d2: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9500",
        BeliefAxisSet::SCHEMA,
        "loan_",
    ));
    let opts_a = ValidateOptions {
        custom_detectors: vec![d1.clone(), d2.clone()],
        ..Default::default()
    };
    let opts_b = ValidateOptions {
        custom_detectors: vec![d2, d1],
        ..Default::default()
    };
    let r_a = validate(&df, &opts_a);
    let r_b = validate(&df, &opts_b);
    assert_eq!(
        emit_locke_report_json(&r_a),
        emit_locke_report_json(&r_b),
        "report bytes must be invariant under detector list order"
    );
}

#[test]
fn empty_message_emission_is_recorded_as_error() {
    let mut sink = FindingSink::new("E9500", BeliefAxisSet::LEAKAGE);
    sink.emit(FindingSeverity::Warning, "", None, None, vec![], 0);
    assert_eq!(sink.len(), 0);
    assert!(sink
        .errors()
        .iter()
        .any(|e| matches!(e, CustomDetectorError::EmptyMessage)));
}

// ─── Proptest: determinism + monotonicity ──────────────────────────────

#[derive(Debug, Clone)]
struct ProptestDetector {
    code: &'static str,
    axes: BeliefAxisSet,
    emit_count: u8,
}

impl CustomDetector for ProptestDetector {
    fn code(&self) -> &'static str {
        self.code
    }
    fn belief_axes(&self) -> BeliefAxisSet {
        self.axes
    }
    fn run(&self, df: &DataFrame, sink: &mut FindingSink) {
        let n = df.nrows() as u64;
        for i in 0..self.emit_count {
            sink.emit(
                FindingSeverity::Warning,
                format!("propt {} of {}", i, self.emit_count),
                Some(format!("col_{}", i)),
                None,
                vec![],
                n,
            );
        }
    }
}

fn arb_axes() -> impl Strategy<Value = BeliefAxisSet> {
    // Sample any nonzero 8-bit combination.
    (1u16..=0xFFu16).prop_map(BeliefAxisSet)
}

fn arb_code() -> impl Strategy<Value = &'static str> {
    // A small set of static strings — the detector code is &'static str
    // so we can't generate strings dynamically here.
    prop::sample::select(vec![
        "E9500", "E9501", "E9502", "E9700", "E9800", "E9999",
    ])
}

fn arb_df() -> impl Strategy<Value = DataFrame> {
    prop::collection::vec(any::<i32>(), 1..20).prop_map(|v| {
        let floats: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        DataFrame::from_columns(vec![
            ("x".into(), Column::Float(floats.clone())),
            ("total_x".into(), Column::Float(floats)),
        ])
        .unwrap()
    })
}

proptest! {
    #[test]
    fn proptest_two_runs_identical(
        code in arb_code(),
        axes in arb_axes(),
        emit_count in 0u8..5,
        df in arb_df(),
    ) {
        let det: Arc<dyn CustomDetector> = Arc::new(ProptestDetector { code, axes, emit_count });
        let opts = ValidateOptions {
            custom_detectors: vec![det],
            ..Default::default()
        };
        let a = validate(&df, &opts);
        let b = validate(&df, &opts);
        prop_assert_eq!(emit_locke_report_json(&a), emit_locke_report_json(&b));
    }

    #[test]
    fn proptest_belief_scores_stay_in_unit_interval(
        code in arb_code(),
        axes in arb_axes(),
        emit_count in 0u8..5,
        df in arb_df(),
    ) {
        let det: Arc<dyn CustomDetector> = Arc::new(ProptestDetector { code, axes, emit_count });
        let opts = ValidateOptions {
            custom_detectors: vec![det],
            ..Default::default()
        };
        let report = validate(&df, &opts);
        let belief = belief_report_from_locke(&report);
        prop_assert!(belief.score.overall >= 0.0 && belief.score.overall <= 1.0);
        prop_assert!(belief.score.leakage_score >= 0.0 && belief.score.leakage_score <= 1.0);
        prop_assert!(belief.score.schema_score >= 0.0 && belief.score.schema_score <= 1.0);
        prop_assert!(belief.score.drift_score >= 0.0 && belief.score.drift_score <= 1.0);
    }

    #[test]
    fn proptest_more_findings_does_not_improve_belief(
        code in arb_code(),
        axes in arb_axes(),
        n_low in 0u8..3,
        bump in 1u8..3,
        df in arb_df(),
    ) {
        let det_low: Arc<dyn CustomDetector> = Arc::new(ProptestDetector { code, axes, emit_count: n_low });
        let det_high: Arc<dyn CustomDetector> = Arc::new(ProptestDetector { code, axes, emit_count: n_low.saturating_add(bump) });
        let opts_low = ValidateOptions { custom_detectors: vec![det_low], ..Default::default() };
        let opts_high = ValidateOptions { custom_detectors: vec![det_high], ..Default::default() };
        let belief_low = belief_report_from_locke(&validate(&df, &opts_low));
        let belief_high = belief_report_from_locke(&validate(&df, &opts_high));
        // More findings cannot raise the overall score (more penalty
        // means lower or equal score). 1e-9 slack for fp noise.
        prop_assert!(belief_high.score.overall <= belief_low.score.overall + 1e-9);
    }
}

// ─── Namespace fuzz: arbitrary code strings never crash ─────────────────

#[test]
fn fuzz_arbitrary_code_strings_never_panic() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|bytes: &Vec<u8>| {
            // Build a candidate code from arbitrary bytes. The function
            // must not panic on any input — only return Ok / Err.
            let s = String::from_utf8_lossy(bytes).to_string();
            let _ = validate_custom_code(&s);
        });
}

#[test]
fn fuzz_arbitrary_emit_inputs_never_panic() {
    bolero::check!()
        .with_type::<(Vec<u8>, u8)>()
        .for_each(|(msg_bytes, sev_idx): &(Vec<u8>, u8)| {
            let msg = String::from_utf8_lossy(msg_bytes).to_string();
            let sev = match sev_idx % 4 {
                0 => FindingSeverity::Info,
                1 => FindingSeverity::Notice,
                2 => FindingSeverity::Warning,
                _ => FindingSeverity::Error,
            };
            let mut sink = FindingSink::new("E9500", BeliefAxisSet::LEAKAGE);
            sink.emit(sev, msg, None, None, vec![], 0);
            // Either the message was rejected (errors non-empty + 0 findings)
            // or accepted. Both are valid; the only invariant is no panic
            // and no half-state.
            assert!(sink.errors().is_empty() == (sink.len() > 0) || sink.is_empty());
        });
}

#[test]
fn fuzz_arbitrary_detector_lists_never_panic() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|seeds: &Vec<u8>| {
            // Build a Vec of detectors with arbitrary axis bitsets +
            // emission counts. Confirm `validate` always produces a
            // report and never panics.
            let df = tiny_df();
            let detectors: Vec<Arc<dyn CustomDetector>> = seeds
                .iter()
                .take(6)
                .enumerate()
                .map(|(i, &b)| {
                    let code: &'static str = match i {
                        0 => "E9500",
                        1 => "E9501",
                        2 => "E9502",
                        3 => "E9700",
                        4 => "E9800",
                        _ => "E9999",
                    };
                    let axes = BeliefAxisSet((b as u16) & 0xFF);
                    Arc::new(ProptestDetector {
                        code,
                        axes,
                        emit_count: b & 0x7,
                    }) as Arc<dyn CustomDetector>
                })
                .collect();
            let opts = ValidateOptions {
                custom_detectors: detectors,
                ..Default::default()
            };
            let _ = validate(&df, &opts);
        });
}

#[test]
fn registration_skips_duplicate_codes() {
    let df = tiny_df();
    let d1: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9500",
        BeliefAxisSet::LEAKAGE,
        "total_",
    ));
    let d2: Arc<dyn CustomDetector> = Arc::new(NameMatchDetector::new(
        "E9500", // same code
        BeliefAxisSet::LEAKAGE,
        "loan_",
    ));
    let outcome = run_custom_detectors(&df, &[d1, d2]);
    // Only the first registered E9500 detector ran; second was rejected.
    assert!(outcome.registration_errors.iter().any(|e| matches!(
        e,
        CustomDetectorError::CodeOutOfNamespace { reason, .. }
            if *reason == "duplicate code registration"
    )));
}
