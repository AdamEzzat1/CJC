//! v0.9 #4 — categorical-feature target leakage (E9065, Cramér's V).
//!
//! Closes the gap between the numeric AUC path (which never sees a string
//! column) and E9072 (which only fires on ID-like cardinality): a
//! low-cardinality categorical column that nearly determines the target.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::leakage::{detect_categorical_target_leakage, LeakageConfig};
use cjc_locke::report::{FindingEvidence, FindingSeverity, ValidationFinding};

/// `feature` (Str) + binary Int `target`, n rows.
fn df_cat(feature: Vec<&str>, target: Vec<i64>) -> DataFrame {
    let f: Vec<String> = feature.into_iter().map(|s| s.to_string()).collect();
    DataFrame::from_columns(vec![
        ("feat".into(), Column::Str(f)),
        ("y".into(), Column::Int(target)),
    ])
    .unwrap()
}

fn e9065(findings: &[ValidationFinding]) -> Option<&ValidationFinding> {
    findings.iter().find(|f| f.code == "E9065")
}

#[test]
fn perfect_categorical_predictor_fires_error() {
    // feature "a" ⇒ y=0, "b" ⇒ y=1, deterministically. Cramér's V = 1.0.
    let feature: Vec<&str> = (0..40).map(|i| if i < 20 { "a" } else { "b" }).collect();
    let target: Vec<i64> = (0..40).map(|i| (i >= 20) as i64).collect();
    let df = df_cat(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    let hit = e9065(&f).expect("perfect predictor must fire E9065");
    assert_eq!(hit.severity, FindingSeverity::Error);
    assert_eq!(hit.column.as_deref(), Some("feat"));
    // Cramér's V evidence ≈ 1.0.
    let v = hit.evidence.iter().find_map(|e| match e {
        FindingEvidence::Metric { label, value } if label == "cramers_v" => Some(*value),
        _ => None,
    });
    assert!((v.unwrap() - 1.0).abs() < 1e-9, "V should be ~1.0, got {v:?}");
}

#[test]
fn independent_categorical_no_finding() {
    // Within each feature value the target is balanced ⇒ V ≈ 0.
    let feature: Vec<&str> = (0..40).map(|i| if i % 2 == 0 { "a" } else { "b" }).collect();
    let target: Vec<i64> = (0..40).map(|i| ((i / 2) % 2) as i64).collect();
    let df = df_cat(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "independent feature must NOT fire: {:?}", f);
}

#[test]
fn high_cardinality_feature_skipped() {
    // distinct == n > categorical_max_distinct ⇒ deferred to E9072.
    let feature: Vec<String> = (0..60).map(|i| format!("id-{i}")).collect();
    let target: Vec<i64> = (0..60).map(|i| (i % 2) as i64).collect();
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Str(feature)),
        ("y".into(), Column::Int(target)),
    ])
    .unwrap();
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "high-cardinality feature must be skipped");
}

#[test]
fn constant_feature_skipped() {
    let df = df_cat(vec!["a"; 40], (0..40).map(|i| (i % 2) as i64).collect());
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "constant feature has no association");
}

#[test]
fn float_target_skipped() {
    let feature: Vec<String> = (0..40).map(|i| if i < 20 { "a".into() } else { "b".into() }).collect();
    let yf: Vec<f64> = (0..40).map(|i| i as f64).collect();
    let df = DataFrame::from_columns(vec![
        ("feat".into(), Column::Str(feature)),
        ("y".into(), Column::Float(yf)),
    ])
    .unwrap();
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "continuous (Float) target is not classification");
}

#[test]
fn too_few_rows_skipped() {
    // n < min_class_count * 2 = 20.
    let df = df_cat(vec!["a", "a", "b", "b"], vec![0, 0, 1, 1]);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "sparse table must be skipped");
}

#[test]
fn e9065_feeds_the_leakage_belief_axis() {
    // End-to-end through the v0.9 registry: a spliced E9065 must drag the
    // leakage axis (it was routed to LEAKAGE in belief_routing).
    let feature: Vec<&str> = (0..40).map(|i| if i < 20 { "a" } else { "b" }).collect();
    let target: Vec<i64> = (0..40).map(|i| (i >= 20) as i64).collect();
    let df = df_cat(feature, target);
    let mut report = validate(&df, &ValidateOptions { dataset_label: "cat".into(), ..Default::default() });
    let before = belief_report_from_locke(&report).score.leakage_score;
    assert_eq!(before, 1.0);
    report.findings.extend(detect_categorical_target_leakage(&df, "y", &LeakageConfig::default()));
    let after = belief_report_from_locke(&report).score.leakage_score;
    assert!(after < 1.0, "E9065 must route to the leakage axis, got {after}");
}

#[test]
fn detector_is_deterministic() {
    let feature: Vec<&str> = (0..40).map(|i| if i < 20 { "a" } else { "b" }).collect();
    let target: Vec<i64> = (0..40).map(|i| (i >= 20) as i64).collect();
    let df = df_cat(feature, target);
    let a = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    let b = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.id, y.id, "finding ids must be identical across runs");
    }
}

#[test]
fn multiclass_categorical_leakage_fires() {
    // 3-class target, each feature value maps to one class. V = 1.0.
    let feature: Vec<&str> = (0..60).map(|i| ["a", "b", "c"][(i / 20) as usize]).collect();
    let target: Vec<i64> = (0..60).map(|i| (i / 20) as i64).collect();
    let df = df_cat(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    let hit = e9065(&f).expect("multi-class perfect predictor must fire");
    assert_eq!(hit.severity, FindingSeverity::Error);
}

// ----------------------------------------------------------------------
// Property + fuzz
// ----------------------------------------------------------------------

use proptest::prelude::*;

proptest! {
    /// Whatever the inputs, a fired E9065 carries a Cramér's V in [0,1] and
    /// the detector never panics; output is deterministic.
    #[test]
    fn e9065_evidence_in_unit_interval_and_deterministic(
        bits in prop::collection::vec(any::<bool>(), 20..120),
    ) {
        // feature = 3 levels by index%3; target = arbitrary binary bits.
        let feature: Vec<String> = (0..bits.len()).map(|i| format!("L{}", i % 3)).collect();
        let target: Vec<i64> = bits.iter().map(|&b| b as i64).collect();
        // Need ≥2 target classes for the check to engage.
        prop_assume!(target.iter().any(|&x| x == 0) && target.iter().any(|&x| x == 1));
        let df = DataFrame::from_columns(vec![
            ("feat".into(), Column::Str(feature)),
            ("y".into(), Column::Int(target)),
        ]).unwrap();
        let cfg = LeakageConfig::default();
        let a = detect_categorical_target_leakage(&df, "y", &cfg);
        let b = detect_categorical_target_leakage(&df, "y", &cfg);
        prop_assert_eq!(a.len(), b.len());
        for f in &a {
            for ev in &f.evidence {
                if let FindingEvidence::Metric { label, value } = ev {
                    if label == "cramers_v" {
                        prop_assert!(*value >= 0.0 && *value <= 1.0 + 1e-9, "V={}", value);
                    }
                }
            }
        }
    }
}

#[test]
fn fuzz_categorical_leakage_never_panics() {
    bolero::check!()
        .with_type::<(Vec<String>, Vec<i64>)>()
        .for_each(|(feat, tgt): &(Vec<String>, Vec<i64>)| {
            let n = feat.len().min(tgt.len());
            if n == 0 {
                return;
            }
            let df = DataFrame::from_columns(vec![
                ("feat".into(), Column::Str(feat[..n].to_vec())),
                ("y".into(), Column::Int(tgt[..n].to_vec())),
            ])
            .unwrap();
            let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
            for finding in &f {
                assert_eq!(finding.code, "E9065");
                for ev in &finding.evidence {
                    if let FindingEvidence::Metric { label, value } = ev {
                        if label == "cramers_v" {
                            assert!(value.is_finite() && *value >= 0.0 && *value <= 1.0 + 1e-9);
                        }
                    }
                }
            }
        });
}
