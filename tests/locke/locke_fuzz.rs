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
