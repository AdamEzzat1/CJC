//! Integration tests for the v0.6 categorical-quality detectors
//! (`detect_rare_categories`, `detect_encoding_risk`,
//! `detect_case_fold_collisions`, `detect_whitespace_punctuation_variants`,
//! `detect_near_duplicate_categories`).
//!
//! These are end-to-end through `validate()` + `belief_report_from_locke()`
//! so we also verify the new codes weaken the correct BeliefScore axes.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{belief_report_from_locke, validate, ValidateOptions},
    detect_all_categorical_quality, detect_case_fold_collisions, detect_confusable_scripts,
    detect_encoding_risk, detect_mojibake, detect_near_duplicate_categories,
    detect_rare_categories, detect_transitive_clusters,
    detect_unicode_normalization_variants, detect_whitespace_punctuation_variants,
    CategoricalQualityConfig, FindingSeverity, ValidationConfig,
};

// ─── Fixture helpers ──────────────────────────────────────────────────────

fn df_str(name: &str, values: &[&str]) -> DataFrame {
    DataFrame::from_columns(vec![(
        name.into(),
        Column::Str(values.iter().map(|s| (*s).into()).collect()),
    )])
    .unwrap()
}

fn df_multi_str(cols: &[(&str, &[&str])]) -> DataFrame {
    DataFrame::from_columns(
        cols.iter()
            .map(|(n, v)| {
                (
                    (*n).into(),
                    Column::Str(v.iter().map(|s| (*s).into()).collect()),
                )
            })
            .collect(),
    )
    .unwrap()
}

// ─── E9016 ────────────────────────────────────────────────────────────────

#[test]
fn e9016_fires_on_long_tail_via_validate() {
    let mut values: Vec<&str> = vec!["common"; 100];
    values.extend(["rare_a", "rare_b", "rare_c"]);
    let df = df_str("plan", &values);
    let opts = ValidateOptions {
        dataset_label: "e9010".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let any = report.findings.iter().any(|f| f.code == "E9016");
    assert!(any, "expected E9010 finding in {:?}", report.findings);
}

#[test]
fn e9016_weakens_constraint_axis() {
    let mut values: Vec<&str> = vec!["common"; 100];
    values.extend(["r1", "r2", "r3", "r4", "r5"]);
    let df = df_str("plan", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9010_belief".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let belief = belief_report_from_locke(&report);
    assert!(
        belief.score.constraint_score < 1.0,
        "rare categories should weaken constraint axis (got {})",
        belief.score.constraint_score
    );
}

// ─── E9017 ────────────────────────────────────────────────────────────────

#[test]
fn e9017_fires_on_wide_categorical() {
    // 100 rows, 60 distinct values — above default 50, below 0.95 ratio.
    let values: Vec<String> = (0..100).map(|i| format!("v{:02}", i % 60)).collect();
    let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
    let df = df_str("country", &v_refs);
    let f = detect_encoding_risk(&df, &CategoricalQualityConfig::default());
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].code, "E9017");
}

#[test]
fn e9017_weakens_schema_axis() {
    let values: Vec<String> = (0..100).map(|i| format!("v{:02}", i % 60)).collect();
    let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
    let df = df_str("country", &v_refs);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9011_belief".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let belief = belief_report_from_locke(&report);
    assert!(
        belief.score.schema_score < 1.0,
        "encoding-risk should weaken schema axis (got {})",
        belief.score.schema_score
    );
}

// ─── E9080 ────────────────────────────────────────────────────────────────

#[test]
fn e9080_fires_on_case_collision_via_validate() {
    let mut values = vec!["Premium", "premium", "PREMIUM"];
    values.extend(vec!["basic"; 20]);
    let df = df_str("tier", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9080".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let f = report
        .findings
        .iter()
        .find(|f| f.code == "E9080")
        .expect("E9080 expected");
    assert_eq!(f.severity, FindingSeverity::Warning);
    assert_eq!(f.column.as_deref(), Some("tier"));
}

#[test]
fn e9080_quiet_when_clean() {
    let mut values = vec!["Premium", "Basic", "Trial"];
    values.extend(vec!["Premium"; 15]);
    let df = df_str("tier", &values);
    let f = detect_case_fold_collisions(&df, &CategoricalQualityConfig::default());
    assert!(f.is_empty(), "no collisions expected, got {:?}", f);
}

// ─── E9081 ────────────────────────────────────────────────────────────────

#[test]
fn e9081_fires_on_whitespace_variants() {
    let values = vec![
        "California",
        "California ",
        "California.",
        "Texas",
        "Texas",
        "Texas",
        "Texas",
        "Texas",
        "Texas",
        "Texas",
    ];
    let df = df_str("state", &values);
    let f = detect_whitespace_punctuation_variants(&df, &CategoricalQualityConfig::default());
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].code, "E9081");
    assert_eq!(f[0].severity, FindingSeverity::Notice);
}

#[test]
fn e9081_defers_to_e9080_when_only_case_differs() {
    let values = vec![
        "Premium", "premium", "PREMIUM", "basic", "basic", "basic", "basic", "basic", "basic",
        "basic",
    ];
    let df = df_str("tier", &values);
    let f = detect_whitespace_punctuation_variants(&df, &CategoricalQualityConfig::default());
    assert!(
        f.is_empty(),
        "E9081 must not double-report what E9080 covers, got {:?}",
        f
    );
}

// ─── E9082 ────────────────────────────────────────────────────────────────

#[test]
fn e9082_detects_typo_pair_via_validate() {
    let mut values = vec!["enterprise", "enterprize"];
    values.extend(vec!["starter"; 30]);
    let df = df_str("plan", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9082".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let f = report
        .findings
        .iter()
        .find(|f| f.code == "E9082")
        .expect("E9082 expected for enterprise/enterprize");
    assert_eq!(f.severity, FindingSeverity::Warning);
}

#[test]
fn e9082_quiet_on_short_strings() {
    let mut values = vec!["M", "F"];
    values.extend(vec!["X"; 20]);
    let df = df_str("sex", &values);
    let f = detect_near_duplicate_categories(&df, &CategoricalQualityConfig::default());
    assert!(f.is_empty(), "got unexpected findings on short strings: {:?}", f);
}

#[test]
fn e9082_emits_info_when_above_cap() {
    let values: Vec<String> = (0..250).map(|i| format!("abcdef{:03}", i)).collect();
    let v_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
    let df = df_str("token", &v_refs);
    let f = detect_near_duplicate_categories(&df, &CategoricalQualityConfig::default());
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].code, "E9082");
    assert_eq!(f[0].severity, FindingSeverity::Info);
}

// ─── Multi-column + determinism ───────────────────────────────────────────

#[test]
fn multi_column_run_is_deterministic() {
    // Synthesise a realistic mess: one column with case-fold issues, one
    // with whitespace variants, one with rare categories, one with typo.
    let n_clean = 60;
    let plan: Vec<&str> = {
        let mut v = vec!["Premium", "premium", "PREMIUM"];
        v.extend(vec!["basic"; n_clean]);
        v
    };
    let state: Vec<&str> = {
        let mut v = vec!["California", "California ", "California."];
        v.extend(vec!["Texas"; n_clean]);
        v
    };
    let plan_typo: Vec<&str> = {
        let mut v = vec!["enterprise", "enterprize"];
        v.extend(vec!["starter"; n_clean + 1]);
        v
    };
    let mut tail = vec!["common"; n_clean];
    tail.extend(["rare_a", "rare_b", "rare_c"]);
    // Pad to equal length for DataFrame construction
    let len = tail.len();
    let plan = plan.into_iter().take(len).collect::<Vec<_>>();
    let state = state.into_iter().take(len).collect::<Vec<_>>();
    let plan_typo = plan_typo.into_iter().take(len).collect::<Vec<_>>();

    let df = df_multi_str(&[
        ("plan", &plan),
        ("state", &state),
        ("plan_typo", &plan_typo),
        ("tail", &tail),
    ]);
    let cfg = CategoricalQualityConfig::default();
    let a = detect_all_categorical_quality(&df, &cfg);
    let b = detect_all_categorical_quality(&df, &cfg);
    assert_eq!(a, b, "two runs over the same data must be identical");
    // And the run-id over those findings should match too.
    let report_a = validate(
        &df,
        &ValidateOptions {
            dataset_label: "multi".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let report_b = validate(
        &df,
        &ValidateOptions {
            dataset_label: "multi".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    assert_eq!(report_a.run_id, report_b.run_id);
}

#[test]
fn empty_or_tiny_df_is_quiet() {
    let df = df_str("plan", &["only"]);
    let cfg = CategoricalQualityConfig::default();
    assert!(detect_rare_categories(&df, &cfg).is_empty());
    assert!(detect_encoding_risk(&df, &cfg).is_empty());
    assert!(detect_case_fold_collisions(&df, &cfg).is_empty());
    assert!(detect_whitespace_punctuation_variants(&df, &cfg).is_empty());
    assert!(detect_near_duplicate_categories(&df, &cfg).is_empty());
}

#[test]
fn numeric_columns_are_skipped() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float(vec![1.0; 20])),
        ("y".into(), Column::Int(vec![1; 20])),
    ])
    .unwrap();
    let cfg = CategoricalQualityConfig::default();
    assert!(detect_all_categorical_quality(&df, &cfg).is_empty());
}

// ─── E9083 (confusable scripts) ─────────────────────────────────────────

#[test]
fn e9083_fires_via_validate_on_mixed_script() {
    // Cyrillic 'а' (U+0430) inside an otherwise-Latin string.
    let mut values = vec!["pаypal"];
    values.extend(vec!["paypal"; 20]);
    let df = df_str("brand", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9083".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    assert!(report.findings.iter().any(|f| f.code == "E9083"));
}

#[test]
fn e9083_quiet_on_pure_latin_data() {
    let values: Vec<&str> = vec!["paypal", "stripe", "amazon"]
        .into_iter()
        .cycle()
        .take(30)
        .collect();
    let df = df_str("brand", &values);
    let f = detect_confusable_scripts(&df, &CategoricalQualityConfig::default());
    assert!(f.is_empty());
}

// ─── E9084 (mojibake) ───────────────────────────────────────────────────

#[test]
fn e9084_fires_via_validate_on_mojibake() {
    let mut values = vec!["cafÃ©"];
    values.extend(vec!["bistro"; 20]);
    let df = df_str("venue", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9084".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    let f = report
        .findings
        .iter()
        .find(|f| f.code == "E9084")
        .expect("E9084 expected");
    assert_eq!(f.severity, FindingSeverity::Notice);
}

#[test]
fn e9084_quiet_on_clean_utf8() {
    let mut values = vec!["café", "thé", "naïve"];
    values.extend(vec!["bistro"; 20]);
    let df = df_str("venue", &values);
    assert!(detect_mojibake(&df, &CategoricalQualityConfig::default()).is_empty());
}

// ─── E9085 (transitive cluster) ─────────────────────────────────────────

#[test]
fn e9085_fires_when_e9080_and_e9082_both_fire() {
    // "Premium"/"premium" → E9080; "enterprise"/"enterprize" → E9082.
    // Both on same column "tier".
    let mut values = vec!["Premium", "premium", "enterprise", "enterprize"];
    values.extend(vec!["basic"; 25]);
    let df = df_str("tier", &values);
    let f = detect_all_categorical_quality(&df, &CategoricalQualityConfig::default());
    let codes: std::collections::BTreeSet<&str> = f.iter().map(|x| x.code).collect();
    assert!(codes.contains("E9080"));
    assert!(codes.contains("E9082"));
    assert!(codes.contains("E9085"));
    let e9085 = f
        .iter()
        .find(|x| x.code == "E9085")
        .expect("E9085 expected");
    assert_eq!(e9085.column.as_deref(), Some("tier"));
}

#[test]
fn e9085_quiet_when_only_one_channel_fires() {
    // Use spellings whose ONLY collision is case-fold. With many
    // case-differing positions, Levenshtein > 2 so E9082 doesn't fire,
    // and no whitespace/punctuation so E9081 doesn't fire either.
    let mut values = vec!["ABCDEFGHIJ", "abcdefghij"];
    values.extend(vec!["basic"; 25]);
    let df = df_str("tier", &values);
    let f = detect_all_categorical_quality(&df, &CategoricalQualityConfig::default());
    let codes: std::collections::BTreeSet<&str> = f.iter().map(|x| x.code).collect();
    assert!(codes.contains("E9080"), "E9080 should fire");
    assert!(!codes.contains("E9082"), "E9082 should not fire (edit dist > 2)");
    assert!(
        !codes.contains("E9085"),
        "E9085 should not fire with only one channel: {:?}",
        codes
    );
}

// ─── Determinism for the v0.6 codes ─────────────────────────────────────

#[test]
fn v06_categorical_codes_are_deterministic() {
    let mut values = vec!["Premium", "premium", "enterprise", "enterprize", "pаypal", "cafÃ©"];
    values.extend(vec!["basic"; 25]);
    let df = df_str("col", &values);
    let cfg = CategoricalQualityConfig::default();
    let a = detect_all_categorical_quality(&df, &cfg);
    let b = detect_all_categorical_quality(&df, &cfg);
    assert_eq!(a, b);
}

#[test]
fn categorical_storage_works_too() {
    // Same logical content as the case-fold test, but stored as Categorical.
    let levels: Vec<&str> = vec!["PREMIUM", "Premium", "basic", "premium"];
    let mut codes: Vec<u32> = vec![0, 1, 3]; // Premium spellings
    codes.extend((0..20).map(|_| 2u32)); // basic × 20
    let df = DataFrame::from_columns(vec![(
        "tier".into(),
        Column::Categorical {
            levels: levels.iter().map(|s| (*s).into()).collect(),
            codes,
        },
    )])
    .unwrap();
    let f = detect_case_fold_collisions(&df, &CategoricalQualityConfig::default());
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].code, "E9080");
}

// ─── E9086 Unicode NFC/NFD ─────────────────────────────────────────────────

#[test]
fn e9086_fires_on_nfc_vs_nfd_variants() {
    // NFC "café" U+00E9, NFD "café" = U+0065 U+0301.
    let nfc = "caf\u{00E9}"; // café in NFC
    let nfd = "caf\u{0065}\u{0301}"; // café in NFD (e + combining acute)
    let mut values = vec![nfc, nfd];
    values.extend(vec!["bistro"; 18]);
    let df = df_str("venue", &values);
    let f = detect_unicode_normalization_variants(&df, &CategoricalQualityConfig::default());
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].code, "E9086");
    assert_eq!(f[0].severity, FindingSeverity::Warning);
}

#[test]
fn e9086_quiet_on_only_one_form() {
    // All in NFC — no collision under combining-mark stripping.
    let mut values: Vec<&str> = vec!["caf\u{00E9}", "th\u{00E9}", "na\u{00EF}ve"];
    values.extend(vec!["bistro"; 18]);
    let df = df_str("venue", &values);
    let f = detect_unicode_normalization_variants(&df, &CategoricalQualityConfig::default());
    assert!(f.is_empty(), "expected quiet, got {:?}", f);
}

#[test]
fn e9086_fires_via_validate() {
    let nfc = "caf\u{00E9}";
    let nfd = "caf\u{0065}\u{0301}";
    let mut values = vec![nfc, nfd];
    values.extend(vec!["bistro"; 18]);
    let df = df_str("venue", &values);
    let report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "e9086".into(),
            config: ValidationConfig::default(),
            ..Default::default()
        },
    );
    assert!(report.findings.iter().any(|f| f.code == "E9086"));
}
