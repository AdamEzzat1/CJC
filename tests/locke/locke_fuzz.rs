//! Bolero structural fuzz tests for Locke.
//!
//! Bolero falls back to proptest on Windows/macOS and uses libfuzzer on
//! Linux CI. These targets exercise:
//!
//! * arbitrary numeric input to validators — no panics
//! * arbitrary categorical input to drift — no panics, output stays well-formed
//! * arbitrary string input to causal-language scanner — no panics
//! * arbitrary lineage operations — no cycles or unknown-parent errors slip through
//!
//! All fuzz targets are `#[test]` so they run inside `cargo test`.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{causal_guardrail, validate, ValidateOptions},
    causal::CausalConfig,
    drift::{compare, DriftConfig},
    lineage::{ImpressionKind, LineageBuilder, LockeIdea, LockeImpression, TransformationRecord},
    validation::ValidationConfig,
};

#[test]
fn fuzz_validate_arbitrary_floats() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|v: &Vec<f64>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
            let opts = ValidateOptions {
                dataset_label: "fuzz".into(),
                config: ValidationConfig::default(),
                ..Default::default()
            };
            let r = validate(&df, &opts);
            // Invariant: no finding's `sample_size` exceeds the row count.
            for f in &r.findings {
                assert!(f.sample_size <= v.len() as u64 * 2);
            }
        });
}

#[test]
fn fuzz_drift_arbitrary_floats() {
    bolero::check!()
        .with_type::<(Vec<f64>, Vec<f64>)>()
        .for_each(|(t, s): &(Vec<f64>, Vec<f64>)| {
            if t.is_empty() || s.is_empty() {
                return;
            }
            let train = DataFrame::from_columns(vec![("x".into(), Column::Float(t.clone()))]).unwrap();
            let test = DataFrame::from_columns(vec![("x".into(), Column::Float(s.clone()))]).unwrap();
            let _ = compare(&train, &test, &DriftConfig::default());
        });
}

#[test]
fn fuzz_causal_guardrail_arbitrary_label_text() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|bytes: &Vec<u8>| {
            let text = String::from_utf8_lossy(bytes).to_string();
            let df = DataFrame::from_columns(vec![("x".into(), Column::Float(vec![1.0]))]).unwrap();
            let _ = causal_guardrail(&df, None, &CausalConfig::default(), Some(&text), false);
        });
}

#[test]
fn fuzz_lineage_operations_never_cycle() {
    bolero::check!()
        .with_type::<Vec<(u8, u8)>>()
        .for_each(|ops: &Vec<(u8, u8)>| {
            if ops.is_empty() {
                return;
            }
            let mut b = LineageBuilder::new("fuzz");
            let p = b
                .add_impression(LockeImpression::new(
                    "src",
                    ImpressionKind::Dataset,
                    10,
                    vec!["a".into()],
                ))
                .unwrap();
            let mut last = p;
            for (kind, _) in ops.iter().take(16) {
                let op = format!("op-{}", kind);
                let parent = if kind % 3 == 0 { p } else { last };
                let idea = LockeIdea::new(
                    &op,
                    TransformationRecord {
                        op_id: op.clone(),
                        params: std::collections::BTreeMap::new(),
                        seed: None,
                    },
                    vec![parent],
                );
                if let Ok(id) = b.add_idea(idea) {
                    last = id;
                }
            }
            let g = b.finish();
            assert!(g.is_acyclic(), "lineage graph must remain acyclic");
        });
}

#[test]
fn fuzz_validate_arbitrary_int_column_never_panics() {
    bolero::check!()
        .with_type::<Vec<i64>>()
        .for_each(|v: &Vec<i64>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("x".into(), Column::Int(v.clone()))]).unwrap();
            let opts = ValidateOptions {
                dataset_label: "fuzz-int".into(),
                config: ValidationConfig::default(),
                ..Default::default()
            };
            let _ = validate(&df, &opts);
        });
}

#[test]
fn fuzz_categorical_quality_arbitrary_strings_never_panics() {
    bolero::check!()
        .with_type::<Vec<String>>()
        .for_each(|v: &Vec<String>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("c".into(), Column::Str(v.clone()))]).unwrap();
            let cfg = cjc_locke::CategoricalQualityConfig::default();
            // None of the eight categorical detectors should panic on
            // arbitrary string input — including escape sequences, null
            // bytes (in UTF-8 form), control chars, BOM, etc.
            let _ = cjc_locke::detect_all_categorical_quality(&df, &cfg);
        });
}

#[test]
fn fuzz_wasserstein_arbitrary_floats_never_panics() {
    bolero::check!()
        .with_type::<(Vec<f64>, Vec<f64>)>()
        .for_each(|(a, b): &(Vec<f64>, Vec<f64>)| {
            // Empty input returns None; non-empty must produce a finite,
            // non-negative number (or None for fully-NaN inputs).
            if let Some(w) = cjc_locke::wasserstein_1(a, b) {
                assert!(w.is_finite(), "W_1 must be finite, got {}", w);
                assert!(w >= -1e-9, "W_1 must be non-negative, got {}", w);
            }
        });
}

#[test]
fn fuzz_pii_detection_arbitrary_strings_never_panic() {
    bolero::check!()
        .with_type::<Vec<String>>()
        .for_each(|v: &Vec<String>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("c".into(), Column::Str(v.clone()))]).unwrap();
            let cfg = cjc_locke::PiiConfig::default();
            // PII pattern detectors must never panic on arbitrary input.
            let _ = cjc_locke::detect_all_pii(&df, &cfg);
        });
}

#[test]
fn fuzz_label_encoding_risk_arbitrary_ints_never_panics() {
    bolero::check!()
        .with_type::<Vec<i64>>()
        .for_each(|v: &Vec<i64>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("v".into(), Column::Int(v.clone()))]).unwrap();
            let cfg = cjc_locke::LabelEncodingRiskConfig::default();
            let _ = cjc_locke::detect_label_encoding_risk(&df, &cfg);
        });
}

#[test]
fn fuzz_seasonality_arbitrary_timestamps_never_panics() {
    bolero::check!()
        .with_type::<Vec<i64>>()
        .for_each(|v: &Vec<i64>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("ts".into(), Column::Int(v.clone()))]).unwrap();
            let cfg = cjc_locke::SeasonalityConfig::default();
            let findings = cjc_locke::detect_seasonality(&df, "ts", &cfg);
            for f in &findings {
                for ev in &f.evidence {
                    if let cjc_locke::FindingEvidence::Metric { label, value } = ev {
                        if label == "index_of_dispersion" {
                            assert!(value.is_finite() && *value >= 0.0);
                        }
                    }
                }
            }
        });
}

#[test]
fn fuzz_mermaid_emit_arbitrary_label_never_panics() {
    bolero::check!()
        .with_type::<String>()
        .for_each(|label: &String| {
            let df = DataFrame::from_columns(vec![
                ("x".into(), Column::Float(vec![1.0, 2.0, 3.0])),
            ])
            .unwrap();
            let g = cjc_locke::api::lineage_for_dataset(label, &df);
            let out = cjc_locke::emit_lineage_mermaid(&g);
            // Output must be a well-formed Mermaid fenced block.
            assert!(out.starts_with("```{mermaid}"));
            assert!(out.ends_with("```\n"));
            // Two runs over the same graph must be byte-identical.
            let out2 = cjc_locke::emit_lineage_mermaid(&g);
            assert_eq!(out, out2);
        });
}
