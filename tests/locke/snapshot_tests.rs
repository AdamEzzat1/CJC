//! Insta snapshot tests for Locke's text emit (v0.2-T7).
//!
//! These tests pin the **rendered output** of the high-level Locke
//! reports — validate, drift, lineage, belief — by writing each
//! canonical text to a `.snap` file in `tests/locke/snapshots/`.
//!
//! Running tests with `INSTA_UPDATE=auto` (or `cargo insta review`)
//! lets the developer review changes deliberately; without it, any
//! drift in the rendered format fails the test.
//!
//! ## Why insta and not cjc-snap
//!
//! - `cjc-snap` is the project's *binary* serialization layer for
//!   `Value`. Different concept.
//! - `insta` records golden-text references and compares on later
//!   runs — exactly what we want for CLI emit stability.
//!
//! ## Determinism contract
//!
//! Run IDs and content-addressed fingerprints are *deterministic* but
//! depend on every byte of evidence. To keep snapshots stable under
//! unrelated changes (e.g. adding a new finding code that doesn't
//! affect the cases here), we use `insta`'s `redactions` to replace
//! hex fingerprints with `[FP]` placeholders before comparison.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::belief::sample_score_from_n;
use cjc_locke::drift::{compare, DriftConfig};
use cjc_locke::lineage::{
    emit_lineage_text, ImpressionKind, LineageBuilder, LockeImpression,
};
use cjc_locke::validation::ValidationConfig;

/// Replace any 16-hex-char fingerprint with `[FP]` so the snapshot
/// survives evidence-format tweaks. Also normalises Windows path
/// separators in the dataset_label.
fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Try to consume a 16-hex-char run.
        if i + 16 <= bytes.len()
            && bytes[i..i + 16]
                .iter()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            // Avoid eating numbers that just happen to be all digits;
            // require at least one a-f letter.
            let chunk = &bytes[i..i + 16];
            if chunk.iter().any(|b| (b'a'..=b'f').contains(b)) {
                out.push_str("[FP]");
                i += 16;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out.replace('\\', "/")
}

fn fixture_validation_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("age".into(), Column::Float(vec![25.0, 30.0, f64::NAN, 30.0, 45.0])),
        ("score".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        (
            "label".into(),
            Column::Str(vec!["a".into(), "b".into(), "a".into(), "c".into(), "b".into()]),
        ),
    ])
    .unwrap()
}

fn validate_text(report: &cjc_locke::LockeReport) -> String {
    let mut s = String::new();
    s.push_str("# Locke Validation Report\n");
    s.push_str(&format!("schema_version: {}\n", report.schema_version));
    s.push_str(&format!("dataset: {}\n", report.input.dataset_label));
    s.push_str(&format!("n_rows: {}\n", report.input.n_rows));
    s.push_str(&format!("n_cols: {}\n", report.input.n_cols));
    s.push_str(&format!("run_id: {}\n", report.run_id));
    s.push_str(&format!(
        "severity_counts: info={} notice={} warning={} error={}\n",
        report.severity_counts.info,
        report.severity_counts.notice,
        report.severity_counts.warning,
        report.severity_counts.error,
    ));
    s.push_str("findings:\n");
    for f in &report.findings {
        let column = f.column.as_deref().unwrap_or("-");
        s.push_str(&format!(
            "  - code={} severity={} column={}\n    {}\n",
            f.code, f.severity, column, f.message
        ));
    }
    s
}

#[test]
fn snapshot_validate_default_emit() {
    let df = fixture_validation_df();
    let opts = ValidateOptions {
        dataset_label: "fixtures/sample.csv".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let rendered = redact(&validate_text(&report));
    insta::assert_snapshot!("validate_default", rendered);
}

#[test]
fn snapshot_belief_explain() {
    let df = fixture_validation_df();
    let opts = ValidateOptions {
        dataset_label: "fixtures/sample.csv".into(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let belief = cjc_locke::api::belief_report_from_locke(&report);
    let rendered = format!(
        "n_rows: {}\nsample_score: {:.3}\n\n{}",
        report.input.n_rows,
        sample_score_from_n(report.input.n_rows),
        belief.score.explain()
    );
    insta::assert_snapshot!("belief_explain", rendered);
}

#[test]
fn snapshot_drift_emit() {
    fn mk(name: &str, v: Vec<f64>) -> (String, Column) {
        (name.into(), Column::Float(v))
    }
    let train = DataFrame::from_columns(vec![mk("x", (0..100).map(|i| i as f64).collect())]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", (50..150).map(|i| i as f64).collect())]).unwrap();
    let report = compare(&train, &test, &DriftConfig::default());

    let mut s = String::new();
    s.push_str("# Locke Induction-Risk Report\n");
    s.push_str(&format!("n_train: {}\n", report.n_train));
    s.push_str(&format!("n_test:  {}\n", report.n_test));
    s.push_str(&format!("shared_columns: {}\n", report.shared_columns.join(",")));
    s.push_str("findings:\n");
    for f in &report.findings {
        s.push_str(&format!(
            "  - code={} severity={} column={}\n    {}\n",
            f.code,
            f.severity,
            f.column.as_deref().unwrap_or("-"),
            f.message
        ));
    }
    insta::assert_snapshot!("drift_default", redact(&s));
}

#[test]
fn snapshot_lineage_emit() {
    let mut b = LineageBuilder::new("snapshot-run");
    let _ = b.add_impression(LockeImpression::new(
        "train.csv",
        ImpressionKind::Dataset,
        100,
        vec!["x".into(), "y".into()],
    ));
    let g = b.finish();
    let rendered = redact(&emit_lineage_text(&g));
    insta::assert_snapshot!("lineage_minimal", rendered);
}
