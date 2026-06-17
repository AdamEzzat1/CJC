//! v0.9 #4 — categorical-feature target leakage (E9065, Cramér's V).
//!
//! Closes the gap between the numeric AUC path (which never sees a string
//! column) and E9072 (which only fires on ID-like cardinality): a
//! low-cardinality categorical column that nearly determines the target.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::leakage::{binary_target_auc, detect_categorical_target_leakage, LeakageConfig};
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
// v0.9.1 — Int-coded categorical features (closing the AUC blind spot)
// ----------------------------------------------------------------------
//
// Many real categoricals arrive as integer codes (discharge_disposition_id,
// admission_type_id, ICD groupings). Before v0.9.1 E9065 was Str-only, so
// these slipped through the categorical-association net and were left to the
// rank-based AUC paths — which are blind whenever the codes are numerically
// interleaved with the target (the diabetes-130 death-code pattern).

/// Int-coded `feature` + Int `target`, both length-n.
fn df_cat_int(feature: Vec<i64>, target: Vec<i64>) -> DataFrame {
    DataFrame::from_columns(vec![
        ("feat".into(), Column::Int(feature)),
        ("y".into(), Column::Int(target)),
    ])
    .unwrap()
}

#[test]
fn int_coded_perfect_predictor_fires_error() {
    // Discharge-code shape: code 11 ⇒ y=0, code 12 ⇒ y=1, deterministically.
    let feature: Vec<i64> = (0..40).map(|i| if i < 20 { 11 } else { 12 }).collect();
    let target: Vec<i64> = (0..40).map(|i| (i >= 20) as i64).collect();
    let df = df_cat_int(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    let hit = e9065(&f).expect("Int-coded perfect predictor must fire E9065");
    assert_eq!(hit.severity, FindingSeverity::Error);
    assert_eq!(hit.column.as_deref(), Some("feat"));
    let v = hit.evidence.iter().find_map(|e| match e {
        FindingEvidence::Metric { label, value } if label == "cramers_v" => Some(*value),
        _ => None,
    });
    assert!((v.unwrap() - 1.0).abs() < 1e-9, "V should be ~1.0, got {v:?}");
}

#[test]
fn int_categorical_auc_blind_spot() {
    // THE motivating case for the Int reach. Codes are numerically
    // *interleaved* with the target so the rank-based AUC is exactly blind
    // (0.5), yet each code maps to exactly one class so the nominal
    // Cramér's V is exactly 1.0. Sorted by value the classes read 0,1,1,0
    // (palindromic) ⇒ positive rank-sum == negative rank-sum ⇒ AUC = 0.5.
    //   code 10 ⇒ y=0   code 20 ⇒ y=1   code 30 ⇒ y=1   code 40 ⇒ y=0
    let mut feature = Vec::new();
    let mut target = Vec::new();
    for (code, y) in [(10_i64, 0_i64), (20, 1), (30, 1), (40, 0)] {
        for _ in 0..10 {
            feature.push(code);
            target.push(y);
        }
    }
    let df = df_cat_int(feature.clone(), target.clone());

    // Cramér's V path fires Error (V = 1.0 — perfect nominal association).
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    let hit = e9065(&f).expect("interleaved Int categorical must fire E9065");
    assert_eq!(hit.severity, FindingSeverity::Error);
    let v = hit.evidence.iter().find_map(|e| match e {
        FindingEvidence::Metric { label, value } if label == "cramers_v" => Some(*value),
        _ => None,
    });
    assert!((v.unwrap() - 1.0).abs() < 1e-9, "V should be ~1.0, got {v:?}");

    // AUC path is blind: the same feature as f64 against the same target
    // scores exactly 0.5 — proving this leak is reachable *only* via E9065.
    let feat_f64: Vec<f64> = feature.iter().map(|&x| x as f64).collect();
    let tgt_u8: Vec<u8> = target.iter().map(|&x| x as u8).collect();
    let auc = binary_target_auc(&feat_f64, &tgt_u8, 10).expect("auc computable");
    let abs_auc = auc.max(1.0 - auc);
    assert!(
        (abs_auc - 0.5).abs() < 1e-9,
        "interleaved codes must collapse AUC to 0.5 (got |AUC|={abs_auc:.6})"
    );
}

#[test]
fn int_and_str_encodings_yield_identical_cramers_v() {
    // The same logical categorical, encoded once as Int codes 1..=4 and once
    // as the Str labels "1".."4" (digit labels sort identically under numeric
    // and lexical order), must route through the same exact-order math and
    // produce a *bit-identical* Cramér's V — the determinism guarantee that
    // the generic helper buys.
    let codes = [(1_i64, 0_i64), (2, 1), (3, 1), (4, 0)];
    let mut int_feat = Vec::new();
    let mut str_feat: Vec<String> = Vec::new();
    let mut target = Vec::new();
    for (code, y) in codes {
        for _ in 0..10 {
            int_feat.push(code);
            str_feat.push(code.to_string());
            target.push(y);
        }
    }
    let df_int = df_cat_int(int_feat, target.clone());
    let df_str = DataFrame::from_columns(vec![
        ("feat".into(), Column::Str(str_feat)),
        ("y".into(), Column::Int(target)),
    ])
    .unwrap();

    let cfg = LeakageConfig::default();
    let v_int = e9065(&detect_categorical_target_leakage(&df_int, "y", &cfg))
        .and_then(cramers_v_of)
        .expect("Int encoding must fire");
    let v_str = e9065(&detect_categorical_target_leakage(&df_str, "y", &cfg))
        .and_then(cramers_v_of)
        .expect("Str encoding must fire");
    assert_eq!(v_int, v_str, "Int/Str encodings must give bit-identical V");
}

#[test]
fn int_coded_high_cardinality_skipped() {
    // distinct == 60 > categorical_max_distinct (50) ⇒ deferred to E9072.
    let feature: Vec<i64> = (0..60).collect();
    let target: Vec<i64> = (0..60).map(|i| (i % 2) as i64).collect();
    let df = df_cat_int(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "high-cardinality Int feature must be skipped");
}

#[test]
fn int_coded_constant_skipped() {
    let df = df_cat_int(vec![7; 40], (0..40).map(|i| (i % 2) as i64).collect());
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "constant Int feature has no association");
}

#[test]
fn int_coded_independent_no_finding() {
    // Within each code the target is balanced ⇒ V ≈ 0.
    let feature: Vec<i64> = (0..40).map(|i| (i % 2) as i64).collect();
    let target: Vec<i64> = (0..40).map(|i| ((i / 2) % 2) as i64).collect();
    let df = df_cat_int(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    assert!(e9065(&f).is_none(), "independent Int feature must NOT fire: {f:?}");
}

#[test]
fn int_feature_vs_int_multiclass_target_fires() {
    // The diabetes shape: a 3-class Int target, an Int-coded feature whose
    // codes each map to one class. V = 1.0 ⇒ Error.
    let feature: Vec<i64> = (0..60).map(|i| [11_i64, 13, 14][(i / 20) as usize]).collect();
    let target: Vec<i64> = (0..60).map(|i| (i / 20) as i64).collect();
    let df = df_cat_int(feature, target);
    let f = detect_categorical_target_leakage(&df, "y", &LeakageConfig::default());
    let hit = e9065(&f).expect("Int feature vs multi-class Int target must fire");
    assert_eq!(hit.severity, FindingSeverity::Error);
}

#[test]
fn int_coded_detector_is_deterministic() {
    let feature: Vec<i64> = (0..40).map(|i| if i < 20 { 11 } else { 12 }).collect();
    let target: Vec<i64> = (0..40).map(|i| (i >= 20) as i64).collect();
    let df = df_cat_int(feature, target);
    let cfg = LeakageConfig::default();
    let a = detect_categorical_target_leakage(&df, "y", &cfg);
    let b = detect_categorical_target_leakage(&df, "y", &cfg);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.id, y.id, "finding ids must be identical across runs");
    }
}

/// Pull the `cramers_v` metric out of a finding (test helper).
fn cramers_v_of(f: &ValidationFinding) -> Option<f64> {
    f.evidence.iter().find_map(|e| match e {
        FindingEvidence::Metric { label, value } if label == "cramers_v" => Some(*value),
        _ => None,
    })
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

proptest! {
    /// Int-coded mirror of the Str property: a fired E9065 over an Int
    /// feature still carries a Cramér's V in [0,1], never panics, and is
    /// deterministic across runs.
    #[test]
    fn e9065_int_evidence_in_unit_interval_and_deterministic(
        bits in prop::collection::vec(any::<bool>(), 20..120),
    ) {
        // Int feature = 5 codes by index%5; target = arbitrary binary bits.
        let feature: Vec<i64> = (0..bits.len()).map(|i| (i % 5) as i64).collect();
        let target: Vec<i64> = bits.iter().map(|&b| b as i64).collect();
        prop_assume!(target.iter().any(|&x| x == 0) && target.iter().any(|&x| x == 1));
        let df = DataFrame::from_columns(vec![
            ("feat".into(), Column::Int(feature)),
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
fn fuzz_int_categorical_leakage_never_panics() {
    bolero::check!()
        .with_type::<(Vec<i64>, Vec<i64>)>()
        .for_each(|(feat, tgt): &(Vec<i64>, Vec<i64>)| {
            let n = feat.len().min(tgt.len());
            if n == 0 {
                return;
            }
            let df = DataFrame::from_columns(vec![
                ("feat".into(), Column::Int(feat[..n].to_vec())),
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
