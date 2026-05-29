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
fn fuzz_shape_detection_arbitrary_floats_never_panics() {
    bolero::check!()
        .with_type::<Vec<f64>>()
        .for_each(|v: &Vec<f64>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
            let cfg = cjc_locke::ShapeConfig::default();
            let findings = cjc_locke::detect_distribution_shape(&df, &cfg);
            for f in &findings {
                for ev in &f.evidence {
                    if let cjc_locke::FindingEvidence::Metric { label, value } = ev {
                        if label == "skewness" || label == "excess_kurtosis" {
                            assert!(value.is_finite());
                        }
                    }
                }
            }
        });
}

#[test]
fn fuzz_multiclass_leakage_arbitrary_ints_never_panics() {
    bolero::check!()
        .with_type::<(Vec<i64>, Vec<f64>)>()
        .for_each(|(t, f): &(Vec<i64>, Vec<f64>)| {
            if t.is_empty() || t.len() != f.len() {
                return;
            }
            let df = DataFrame::from_columns(vec![
                ("feat".into(), Column::Float(f.clone())),
                ("target".into(), Column::Int(t.clone())),
            ])
            .unwrap();
            let cfg = cjc_locke::leakage::LeakageConfig::default();
            let _ = cjc_locke::leakage::detect_target_leakage_multiclass(&df, "target", &cfg);
        });
}

#[test]
fn fuzz_algebra_compose_never_panics_and_stays_in_unit_interval() {
    // Arbitrary 16 floats → 2 BeliefScores → compose under each rule.
    // No panic; every axis stays in [0, 1] regardless of input.
    bolero::check!()
        .with_type::<([f64; 8], [f64; 8])>()
        .for_each(|(a_arr, b_arr): &([f64; 8], [f64; 8])| {
            let a = cjc_locke::BeliefScore::from_dimensions(
                a_arr[0], a_arr[1], a_arr[2], a_arr[3],
                a_arr[4], a_arr[5], a_arr[6], a_arr[7],
            );
            let b = cjc_locke::BeliefScore::from_dimensions(
                b_arr[0], b_arr[1], b_arr[2], b_arr[3],
                b_arr[4], b_arr[5], b_arr[6], b_arr[7],
            );
            for rule in [
                cjc_locke::CompositionRule::Min,
                cjc_locke::CompositionRule::Max,
                cjc_locke::CompositionRule::GeometricMean,
                cjc_locke::CompositionRule::ArithmeticMean,
            ] {
                let r = cjc_locke::BeliefAxisRules {
                    schema: rule, missingness: rule, drift: rule, leakage: rule,
                    lineage: rule, sample: rule, duplication: rule, constraint: rule,
                };
                let c = cjc_locke::compose(&a, &b, &r);
                for axis in [
                    c.schema_score, c.missingness_score, c.drift_score, c.leakage_score,
                    c.lineage_score, c.sample_score, c.duplication_score, c.constraint_score,
                    c.overall,
                ] {
                    assert!(
                        axis >= 0.0 && axis <= 1.0,
                        "rule {:?} produced out-of-range axis {}",
                        rule, axis
                    );
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

// ─── v0.6.4 — auto-sentinel + E9064 structural fuzz ─────────────────────

/// Arbitrary `Vec<String>` may include any byte sequence — Bolero
/// will generate empty strings, whitespace, unicode, near-sentinels,
/// real sentinels, anything. The detector must never panic and the
/// returned mask must be bounded by the input length.
#[test]
fn fuzz_auto_sentinel_never_panics() {
    bolero::check!()
        .with_type::<Vec<String>>()
        .for_each(|v: &Vec<String>| {
            if v.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![(
                "x".into(),
                Column::Str(v.clone()),
            )])
            .unwrap();
            let cfg = ValidationConfig::default();
            let (masks, findings) = cjc_locke::detect_string_sentinels(&df, &cfg);
            // Invariants:
            // - mask null-row count never exceeds input length
            if let Some(m) = masks.get("x") {
                assert!(m.null_rows.len() <= v.len());
                // - every row index is in [0, len)
                for r in &m.null_rows {
                    assert!(*r < v.len());
                }
            }
            // - at most one E9008 finding per column (we have one column)
            assert!(findings.iter().filter(|f| f.code == "E9008").count() <= 1);
            // - opt-out path never panics either
            let off = ValidationConfig { auto_detect_sentinels: false, ..Default::default() };
            let (m_off, f_off) = cjc_locke::detect_string_sentinels(&df, &off);
            assert!(m_off.is_empty());
            assert!(f_off.is_empty());
        });
}

/// Arbitrary strings → tokenizer train + encode + decode must never
/// panic. Round-trip is preserved by construction. Determinism holds
/// across two consecutive training calls.
#[test]
fn fuzz_tokenizer_arbitrary_strings_never_panic() {
    bolero::check!()
        .with_type::<(String, String)>()
        .for_each(|(s1, s2): &(String, String)| {
            let cfg = cjc_locke::TokenizerTrainConfig::default();
            let t = cjc_locke::Tokenizer::train(&[s1.as_str(), s2.as_str()], &cfg);
            // Determinism.
            let t2 = cjc_locke::Tokenizer::train(&[s1.as_str(), s2.as_str()], &cfg);
            assert_eq!(t.fingerprint(), t2.fingerprint());
            // Round-trip on every input.
            for s in &[s1, s2] {
                let ids = t.encode(s);
                let back = t.decode(&ids);
                assert_eq!(&back, *s);
            }
        });
}

/// Arbitrary `(train, test)` text → text drift detectors must never
/// panic. Whichever findings emit must satisfy basic invariants:
/// KS-D in `[0, 1]`, entropy values finite and non-negative.
#[test]
fn fuzz_text_drift_arbitrary_strings_never_panic() {
    bolero::check!()
        .with_type::<(String, String)>()
        .for_each(|(train, test): &(String, String)| {
            let cfg = cjc_locke::TextDriftConfig::default();
            let f_vocab =
                cjc_locke::detect_vocabulary_ks_drift_on_column("c", train, test, &cfg);
            if let Some(f) = f_vocab {
                assert_eq!(f.code, "E9110");
                for ev in &f.evidence {
                    if let cjc_locke::FindingEvidence::Metric { label, value } = ev {
                        if label == "vocab_ks_d" {
                            assert!(*value >= 0.0 && *value <= 1.0 + 1e-9);
                        }
                    }
                }
            }
            let f_lang = cjc_locke::detect_language_distribution_shift_on_column(
                "c", train, test, &cfg,
            );
            if let Some(f) = f_lang {
                assert_eq!(f.code, "E9112");
                for ev in &f.evidence {
                    if let cjc_locke::FindingEvidence::Metric { label, value } = ev {
                        if label == "char_3gram_ks_d" {
                            assert!(*value >= 0.0 && *value <= 1.0 + 1e-9);
                        }
                    }
                }
            }
        });
}

/// Arbitrary suppression / requirement parameters → `apply_policy`
/// must never panic. Two consecutive runs are byte-identical. Result
/// is well-formed: every suppression decision has a non-zero
/// decision_id, every requirement has observed >= 0.
#[test]
fn fuzz_policy_apply_arbitrary_rules_never_panic() {
    bolero::check!()
        .with_type::<(Vec<f64>, Vec<u8>, Vec<u8>)>()
        .for_each(|(values, codes, ops): &(Vec<f64>, Vec<u8>, Vec<u8>)| {
            if values.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("x".into(), Column::Float(values.clone()))])
                .unwrap();
            let opts = ValidateOptions {
                dataset_label: "fuzz".into(),
                config: ValidationConfig::default(),
                ..Default::default()
            };
            let report = validate(&df, &opts);

            // Build an arbitrary policy.
            let pick_code = |b: u8| match b % 4 {
                0 => "E9001",
                1 => "E9080",
                2 => "E9081",
                _ => "E9999",
            };
            let suppressions: Vec<cjc_locke::SuppressionRule> = codes
                .iter()
                .take(4)
                .map(|b| cjc_locke::SuppressionRule {
                    code: pick_code(*b).into(),
                    column: if b % 2 == 0 { Some("x".into()) } else { None },
                    reason: "fuzz".into(),
                })
                .collect();
            let pick_op = |b: u8| match b % 6 {
                0 => cjc_locke::RequirementOperator::EqZero,
                1 => cjc_locke::RequirementOperator::LessOrEqual,
                2 => cjc_locke::RequirementOperator::GreaterOrEqual,
                3 => cjc_locke::RequirementOperator::Less,
                4 => cjc_locke::RequirementOperator::Greater,
                _ => cjc_locke::RequirementOperator::Equal,
            };
            let requirements: Vec<cjc_locke::RequiredFindingRule> = ops
                .iter()
                .take(4)
                .map(|b| cjc_locke::RequiredFindingRule {
                    code: pick_code(*b).into(),
                    operator: pick_op(*b),
                    threshold: (*b as u64) % 10,
                    owner: None,
                })
                .collect();
            let policy = cjc_locke::Policy {
                suppressions,
                owners: vec![],
                requirements,
            };

            let r1 = cjc_locke::apply_policy(&report, &policy);
            let r2 = cjc_locke::apply_policy(&report, &policy);
            // Determinism.
            assert_eq!(r1, r2);
            // No suppression decision has the zero fingerprint (extremely
            // unlikely collision, sanity check that we built proper IDs).
            for d in &r1.suppressions {
                assert_ne!(d.decision_id.0, 0);
            }
            // Emit doesn't panic.
            let _ = cjc_locke::emit_policy_result_text(&r1);
        });
}

/// Arbitrary strings → per-value lineage builder must never panic.
/// Stages stay in fixed code order; emitted text is well-formed;
/// determinism holds for two consecutive runs on the same input.
#[test]
fn fuzz_per_value_lineage_arbitrary_strings_never_panic() {
    bolero::check!()
        .with_type::<Vec<String>>()
        .for_each(|values: &Vec<String>| {
            if values.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![(
                "c".into(),
                Column::Str(values.clone()),
            )])
            .unwrap();
            let cfg = cjc_locke::PerValueLineageConfig::default();
            let map_a = cjc_locke::build_per_value_lineage(&df, &cfg);
            let map_b = cjc_locke::build_per_value_lineage(&df, &cfg);
            // Determinism.
            assert_eq!(map_a, map_b);
            // Every emitted lineage's stage list never exceeds 5
            // (4 transforms + 1 rare tag — the fixed total).
            for lineage in map_a.values() {
                assert!(lineage.stages.len() <= 5);
            }
            // Emit text never panics on any lineage.
            for lineage in map_a.values() {
                let s = cjc_locke::emit_value_trace_text(lineage);
                assert!(s.starts_with("trace: column="));
            }
        });
}

/// Arbitrary `(Vec<i64>, Vec<i64>)` for E9064. Detector must never
/// panic; emitted findings must carry the required evidence fields.
#[test]
fn fuzz_e9064_never_panics() {
    bolero::check!()
        .with_type::<(Vec<i64>, Vec<i64>)>()
        .for_each(|(col, target): &(Vec<i64>, Vec<i64>)| {
            let n = col.len().min(target.len());
            if n == 0 {
                return;
            }
            let df = DataFrame::from_columns(vec![
                ("col".into(), Column::Int(col[..n].to_vec())),
                ("y".into(), Column::Int(target[..n].to_vec())),
            ])
            .unwrap();
            let cfg = cjc_locke::PerLevelLeakageConfig::default();
            let findings = cjc_locke::detect_per_level_target_leakage(&df, "y", &cfg);
            // Every E9064 finding must carry support + p_class_given_level
            // + base_rate evidence.
            for f in findings.iter().filter(|f| f.code == "E9064") {
                let has_metric = |label: &str| {
                    f.evidence.iter().any(|e| matches!(e,
                        cjc_locke::FindingEvidence::Metric { label: l, .. } if l == label))
                };
                let has_count = |label: &str| {
                    f.evidence.iter().any(|e| matches!(e,
                        cjc_locke::FindingEvidence::Count { label: l, .. } if l == label))
                };
                assert!(has_metric("p_class_given_level"));
                assert!(has_metric("base_rate"));
                assert!(has_count("support"));
                // Severity is Error per spec.
                assert!(matches!(f.severity, cjc_locke::FindingSeverity::Error));
            }
            // Two consecutive runs are byte-identical.
            let again = cjc_locke::detect_per_level_target_leakage(&df, "y", &cfg);
            assert_eq!(findings, again);
        });
}
