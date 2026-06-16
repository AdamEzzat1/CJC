//! Integration tests for the v0.6 batch 2 PII detector (E9090-E9093).
//!
//! End-to-end through `validate()` so we also verify the new codes wire
//! into the full report and reach the BeliefScore aggregation.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{belief_report_from_locke, validate, ValidateOptions},
    detect_all_pii, looks_like_api_key, looks_like_email, looks_like_phone, looks_like_ssn,
    FindingSeverity, PiiConfig, ValidationConfig,
};

fn df_str(name: &str, values: &[&str]) -> DataFrame {
    DataFrame::from_columns(vec![(
        name.into(),
        Column::Str(values.iter().map(|s| (*s).into()).collect()),
    )])
    .unwrap()
}

#[test]
fn pii_runs_via_validate_end_to_end() {
    let mut values = vec!["111-22-3333", "444-55-6666", "777-88-9999"];
    values.extend(vec!["something"; 17]);
    let df = df_str("id", &values);
    let opts = ValidateOptions {
        dataset_label: "pii-e2e".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let ssn = report
        .findings
        .iter()
        .find(|f| f.code == "E9092")
        .expect("E9092 SSN expected via validate");
    assert_eq!(ssn.severity, FindingSeverity::Error);
}

#[test]
fn pii_samples_are_redacted_no_raw_secret_reaches_the_report() {
    // B6.12 — the raw secret must never reach the report. Real SSN shapes;
    // the sample evidence must be masked to `***-**-****`, and the full
    // serialized report must contain no raw SSN digits.
    let raws = ["123-45-6789", "987-65-4321", "555-66-7777"];
    let mut values: Vec<&str> = raws.to_vec();
    values.extend(vec!["filler"; 17]); // 3/20 = 15% ≥ default 10% share
    let df = df_str("ssn", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "redact".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let ssn = report
        .findings
        .iter()
        .find(|f| f.code == "E9092")
        .expect("E9092 expected");

    // Scope (B6.12): the PII DETECTOR's own sample evidence must be masked
    // — that is the leak this fix closes. (Whole-report no-leak is a larger
    // follow-up: non-PII categorical detectors like E9016 rare-category
    // still echo the raw value; see BELIEF_AXIS_ROUTING.md §follow-ups.)
    let mut saw_sample = false;
    for ev in &ssn.evidence {
        if let cjc_locke::FindingEvidence::Sample { label, value } = ev {
            if label == "sample" {
                saw_sample = true;
                assert!(value.contains("***-**-****"), "expected masked SSN shape, got {value}");
                for raw in raws {
                    assert!(!value.contains(raw), "raw SSN leaked into PII sample: {value}");
                }
            }
        }
    }
    assert!(saw_sample, "SSN finding must carry a (redacted) sample");

    // The redacted shape must reach serialization (proves the masked sample
    // is what's persisted, not the raw value).
    let json = cjc_locke::emit_locke_report_json(&report);
    assert!(json.contains("***-**-****"), "masked SSN sample must appear in report JSON");
}

#[test]
fn pii_weakens_constraint_axis() {
    let mut values = vec!["alice@a.com", "bob@b.com", "carol@c.com"];
    values.extend(vec!["filler"; 17]);
    let df = df_str("contact", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "pii-belief".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let belief = belief_report_from_locke(&report);
    assert!(
        belief.score.constraint_score < 1.0,
        "PII findings should weaken the constraint axis, got {}",
        belief.score.constraint_score
    );
}

#[test]
fn email_detection_canonical_shapes() {
    assert!(looks_like_email("alice@example.com"));
    assert!(looks_like_email("a.b+tag@sub.domain.co.uk"));
    assert!(!looks_like_email("no-at-sign"));
    assert!(!looks_like_email("two@@signs.com"));
}

#[test]
fn phone_detection_e164_and_na() {
    assert!(looks_like_phone("+14155552671"));
    assert!(looks_like_phone("(415) 555-2671"));
    assert!(!looks_like_phone("not-a-phone"));
}

#[test]
fn ssn_detection_exact_format_only() {
    assert!(looks_like_ssn("123-45-6789"));
    assert!(!looks_like_ssn("123456789"));
}

#[test]
fn api_key_detection_high_entropy_only() {
    assert!(looks_like_api_key("abcDEF123XYZ_456-PQR789mnop", 24, 3.5));
    assert!(!looks_like_api_key("aaaaaaaaaaaaaaaaaaaaaaaaaaaa", 24, 3.5));
}

#[test]
fn pii_share_threshold_respected() {
    // 1 email out of 30 = 3.3%, below default 10% threshold.
    let mut values = vec!["alice@example.com"];
    values.extend(vec!["plain"; 29]);
    let df = df_str("note", &values);
    let f = detect_all_pii(&df, &PiiConfig::default());
    assert!(f.is_empty(), "expected no findings, got {:?}", f);
}

#[test]
fn pii_is_deterministic_across_runs() {
    let mut values = vec!["111-22-3333", "alice@example.com", "+14155552671"];
    values.extend(vec!["filler"; 17]);
    let df = df_str("mixed", &values);
    let a = detect_all_pii(&df, &PiiConfig::default());
    let b = detect_all_pii(&df, &PiiConfig::default());
    assert_eq!(a, b);
}

#[test]
fn pii_skips_numeric_columns() {
    let df = DataFrame::from_columns(vec![("x".into(), Column::Float(vec![1.0; 30]))]).unwrap();
    let f = detect_all_pii(&df, &PiiConfig::default());
    assert!(f.is_empty());
}

#[test]
fn pii_handles_empty_dataframe() {
    let df = DataFrame::from_columns(vec![("c".into(), Column::Str(vec![]))]).unwrap();
    let f = detect_all_pii(&df, &PiiConfig::default());
    assert!(f.is_empty());
}
