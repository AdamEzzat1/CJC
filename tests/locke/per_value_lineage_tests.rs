//! Integration tests for `per_value_lineage` (v0.7+ heavy, A2).
//!
//! These exercise the per-value lineage pipeline end-to-end through the
//! same DataFrame surface that real callers use: build a `Column::Str`
//! or `Column::Categorical`, run `build_per_value_lineage` or
//! `trace_value`, assert the stages match the diabetes-130 sentinel
//! story and the categorical-quality scenarios from the v0.6.4 work.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    build_per_value_lineage, emit_value_trace_text, trace_value, LineageStage,
    PerValueLineage, PerValueLineageConfig, PerValueStageSet, ValueTransform,
};
use std::collections::BTreeSet;

// ─── Helpers ──────────────────────────────────────────────────────────────

fn str_df(col: &str, values: &[&str]) -> DataFrame {
    DataFrame::from_columns(vec![(
        col.into(),
        Column::Str(values.iter().map(|s| s.to_string()).collect()),
    )])
    .unwrap()
}

fn diabetes130_weight_fixture() -> DataFrame {
    // Approximates the structure of UCI Diabetes-130's `weight` column.
    // 97 `?` rows mixed with 3 banded weight strings — the v0.6.4
    // motivating case where default Locke previously reported
    // `missingness_score = 1.0000`.
    let mut values: Vec<String> = (0..97).map(|_| "?".to_string()).collect();
    values.extend(vec!["[0-25)".into(), "[50-75)".into(), "[100-125)".into()]);
    DataFrame::from_columns(vec![("weight".into(), Column::Str(values))]).unwrap()
}

// ─── Diabetes-130 sentinel scenario ───────────────────────────────────────

#[test]
fn diabetes130_question_mark_trace_terminates_at_sentinel_mask() {
    // The flagship case from the A2 handoff:
    //   "?" → sentinel_mask (E9008) → null
    let df = diabetes130_weight_fixture();
    let lineage = trace_value(
        &df,
        &PerValueLineageConfig::default(),
        "weight",
        "?",
    )
    .expect("question mark is present in the weight column");
    assert_eq!(lineage.stages.len(), 1);
    match &lineage.stages[0].transform {
        ValueTransform::SentinelMask { sentinel } => assert_eq!(sentinel, "?"),
        other => panic!("expected SentinelMask, got {:?}", other),
    }
    // Result is null — canonical = None.
    assert_eq!(lineage.stages[0].canonical, None);
    // Emit text mentions both the E9008 code and `result: null`.
    let text = emit_value_trace_text(&lineage);
    assert!(text.contains("(E9008)"));
    assert!(text.contains("result: null"));
}

#[test]
fn diabetes130_real_weight_band_traces_without_canonicalisation() {
    // The non-? values like "[0-25)" produce no *canonicalisation*
    // stages (sentinel / case_fold / whitespace_punct /
    // unicode_normalize) — they're already canonical and have no
    // collision siblings. The rare-candidate stage may fire because
    // these bands appear once each while the `?` sentinel dominates
    // (97 rows >> threshold 5), which is itself a correct signal:
    // relative to the dataset's most common value, the band IS rare.
    let df = diabetes130_weight_fixture();
    let lineage = trace_value(
        &df,
        &PerValueLineageConfig::default(),
        "weight",
        "[0-25)",
    )
    .expect("real weight band is present");
    let n_canonicalisation_stages = lineage
        .stages
        .iter()
        .filter(|s| {
            !matches!(
                s.transform,
                ValueTransform::RareCandidate { .. }
                    | ValueTransform::TooManyDistinctValuesSkipped { .. }
            )
        })
        .count();
    assert_eq!(
        n_canonicalisation_stages, 0,
        "expected zero canonicalisation stages on a clean band, got {:?}",
        lineage.stages
    );
}

// ─── v0.6 categorical-quality scenarios ───────────────────────────────────

#[test]
fn case_fold_collisions_produce_lineage_with_siblings() {
    let df = str_df(
        "tier",
        &[
            "Premium", "premium", "PREMIUM", "Standard", "Basic",
        ],
    );
    let lineage = trace_value(
        &df,
        &PerValueLineageConfig::default(),
        "tier",
        "Premium",
    )
    .unwrap();
    let case_fold = lineage
        .stages
        .iter()
        .find(|s| matches!(s.transform, ValueTransform::CaseFold))
        .expect("case_fold stage");
    assert_eq!(case_fold.canonical.as_deref(), Some("premium"));
    let expected: BTreeSet<String> =
        ["premium", "PREMIUM"].iter().map(|s| s.to_string()).collect();
    assert_eq!(case_fold.siblings, expected);
}

#[test]
fn whitespace_strip_groups_usa_variants() {
    let df = str_df("country", &["USA", "USA.", " USA "]);
    let lineage = trace_value(
        &df,
        &PerValueLineageConfig::default(),
        "country",
        "USA.",
    )
    .unwrap();
    let ws = lineage
        .stages
        .iter()
        .find(|s| matches!(s.transform, ValueTransform::WhitespacePunctStrip { .. }))
        .expect("whitespace_punct stage");
    assert_eq!(ws.canonical.as_deref(), Some("usa"));
    assert!(ws.siblings.contains("USA"));
    assert!(ws.siblings.contains(" USA "));
}

#[test]
fn nfc_nfd_grouping_via_combining_mark_strip() {
    // "café" precomposed (U+00E9) vs decomposed (e + U+0301)
    let nfc = "caf\u{00E9}";
    let nfd = "cafe\u{0301}";
    let df = str_df("name", &[nfc, nfd, "Boston"]);
    let lineage = trace_value(
        &df,
        &PerValueLineageConfig::default(),
        "name",
        nfc,
    )
    .unwrap();
    let unorm = lineage
        .stages
        .iter()
        .find(|s| matches!(s.transform, ValueTransform::UnicodeNormalize))
        .expect("unicode_normalize stage");
    assert_eq!(unorm.canonical.as_deref(), Some("cafe"));
    assert!(unorm.siblings.contains(nfd));
}

// ─── Rare-candidate / column-skip / determinism ──────────────────────────

#[test]
fn rare_candidate_fires_only_when_a_non_rare_value_exists() {
    let mut values: Vec<String> = (0..8).map(|_| "common".to_string()).collect();
    values.push("rare_a".into());
    values.push("rare_b".into());
    let df = DataFrame::from_columns(vec![("tag".into(), Column::Str(values))]).unwrap();
    let cfg = PerValueLineageConfig::default();
    let lineage_rare = trace_value(&df, &cfg, "tag", "rare_a").unwrap();
    assert!(
        lineage_rare
            .stages
            .iter()
            .any(|s| matches!(s.transform, ValueTransform::RareCandidate { .. }))
    );
    let lineage_common = trace_value(&df, &cfg, "tag", "common").unwrap();
    assert!(
        !lineage_common
            .stages
            .iter()
            .any(|s| matches!(s.transform, ValueTransform::RareCandidate { .. }))
    );
}

#[test]
fn build_lineage_is_byte_identical_across_runs() {
    let df = str_df(
        "x",
        &["Apple", "apple", "APPLE", "Banana", "?", "banana", "BANANA"],
    );
    let cfg = PerValueLineageConfig::default();
    let a = build_per_value_lineage(&df, &cfg);
    let b = build_per_value_lineage(&df, &cfg);
    assert_eq!(a, b);
}

#[test]
fn emit_value_trace_text_is_deterministic() {
    let df = str_df("c", &["A", "a", "?"]);
    let cfg = PerValueLineageConfig::default();
    let lineage = trace_value(&df, &cfg, "c", "A").unwrap();
    let s1 = emit_value_trace_text(&lineage);
    let s2 = emit_value_trace_text(&lineage);
    assert_eq!(s1, s2);
}

#[test]
fn oversized_columns_emit_synthetic_skipped_entry() {
    let cfg = PerValueLineageConfig {
        max_distinct_per_column: 3,
        ..Default::default()
    };
    // 5 distinct values, exceeds limit of 3.
    let df = str_df("c", &["a", "b", "c", "d", "e"]);
    let map = build_per_value_lineage(&df, &cfg);
    assert_eq!(map.len(), 1);
    let lineage = map
        .get(&("c".to_string(), "__skipped__".to_string()))
        .expect("synthetic skipped entry missing");
    assert!(matches!(
        lineage.stages[0].transform,
        ValueTransform::TooManyDistinctValuesSkipped { .. }
    ));
}

// ─── Categorical-storage equivalence ──────────────────────────────────────

#[test]
fn categorical_storage_produces_equivalent_lineage_to_str() {
    // The categorical detectors normalise across Column::Str /
    // Column::Categorical / Column::CategoricalAdaptive via
    // `category_counts`. Per-value lineage should follow.
    let str_df = str_df("t", &["Premium", "premium", "PREMIUM"]);
    let cat_df = DataFrame::from_columns(vec![(
        "t".into(),
        Column::Categorical {
            levels: vec!["Premium".into(), "premium".into(), "PREMIUM".into()],
            codes: vec![0, 1, 2],
        },
    )])
    .unwrap();
    let cfg = PerValueLineageConfig::default();
    let s = trace_value(&str_df, &cfg, "t", "Premium").unwrap();
    let c = trace_value(&cat_df, &cfg, "t", "Premium").unwrap();
    assert_eq!(s, c, "Str and Categorical storage must produce the same lineage");
}

// ─── Sanity: PerValueLineage equality and Debug ───────────────────────────

#[test]
fn per_value_lineage_round_trips_through_emit_and_struct_eq() {
    let df = str_df("c", &["X", "x", "?"]);
    let cfg = PerValueLineageConfig::default();
    let a: PerValueLineage = trace_value(&df, &cfg, "c", "X").unwrap();
    let b: PerValueLineage = trace_value(&df, &cfg, "c", "X").unwrap();
    assert_eq!(a, b);
    // LineageStage equality covers the transform / canonical / siblings tuple.
    for (sa, sb) in a.stages.iter().zip(b.stages.iter()) {
        let sa: &LineageStage = sa;
        let sb: &LineageStage = sb;
        assert_eq!(sa, sb);
    }
}

#[test]
fn stage_set_can_isolate_a_single_transform_for_targeted_inspection() {
    // A user investigating *only* sentinel masking can disable everything
    // else to keep the trace focused.
    let df = diabetes130_weight_fixture();
    let cfg = PerValueLineageConfig {
        stages: PerValueStageSet {
            sentinel: true,
            case_fold: false,
            whitespace_punct: false,
            unicode_normalize: false,
            rare_candidate: false,
        },
        ..Default::default()
    };
    let lineage = trace_value(&df, &cfg, "weight", "?").unwrap();
    // Should still emit the sentinel stage.
    assert_eq!(lineage.stages.len(), 1);
    assert!(matches!(
        lineage.stages[0].transform,
        ValueTransform::SentinelMask { .. }
    ));
    // Disabling sentinel suppresses it.
    let cfg2 = PerValueLineageConfig {
        stages: PerValueStageSet {
            sentinel: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let lineage2 = trace_value(&df, &cfg2, "weight", "?").unwrap();
    assert!(
        !lineage2
            .stages
            .iter()
            .any(|s| matches!(s.transform, ValueTransform::SentinelMask { .. })),
        "disabling sentinel stage should remove it from the trace"
    );
}

// ─── A2-by-default: validate(...) opt-in lineage attachment ───────────────

#[test]
fn default_validate_does_not_attach_per_value_lineage() {
    use cjc_locke::api::{validate, ValidateOptions};
    use cjc_locke::validation::ValidationConfig;
    let df = diabetes130_weight_fixture();
    let opts = ValidateOptions {
        dataset_label: "default".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    assert!(
        report.per_value_lineage.is_none(),
        "default config must not attach lineage — byte-identical to v0.7"
    );
}

#[test]
fn opt_in_validate_attaches_per_value_lineage() {
    use cjc_locke::api::{validate, ValidateOptions};
    use cjc_locke::validation::ValidationConfig;
    let df = diabetes130_weight_fixture();
    let mut cfg = ValidationConfig::default();
    cfg.collect_per_value_lineage = true;
    let opts = ValidateOptions {
        dataset_label: "with_trace".into(),
        config: cfg,
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let lineage = report
        .per_value_lineage
        .expect("opt-in must attach lineage");
    // The diabetes-130 ? sentinel must appear in the lineage map.
    let weight_q_key = ("weight".to_string(), "?".to_string());
    let entry = lineage
        .get(&weight_q_key)
        .expect("? sentinel must appear in lineage");
    assert!(matches!(
        entry.stages[0].transform,
        ValueTransform::SentinelMask { .. }
    ));
}

#[test]
fn opt_in_validate_byte_identical_findings_to_default() {
    // Critical regression — turning on lineage must NOT change the
    // findings, run_id, severity_counts, etc. on the report. Only the
    // new field changes.
    use cjc_locke::api::{validate, ValidateOptions};
    use cjc_locke::validation::ValidationConfig;
    let df = diabetes130_weight_fixture();
    let opts_default = ValidateOptions {
        dataset_label: "regression_check".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let mut cfg_opt_in = ValidationConfig::default();
    cfg_opt_in.collect_per_value_lineage = true;
    let opts_opt_in = ValidateOptions {
        dataset_label: "regression_check".into(),
        config: cfg_opt_in,
        ..Default::default()
    };
    let r_default = validate(&df, &opts_default);
    let r_opt_in = validate(&df, &opts_opt_in);
    assert_eq!(r_default.findings, r_opt_in.findings);
    assert_eq!(r_default.run_id, r_opt_in.run_id);
    assert_eq!(r_default.severity_counts, r_opt_in.severity_counts);
    assert_eq!(r_default.column_reports, r_opt_in.column_reports);
    // The only difference is the new field.
    assert!(r_default.per_value_lineage.is_none());
    assert!(r_opt_in.per_value_lineage.is_some());
}

#[test]
fn opt_in_validate_is_deterministic_across_runs() {
    use cjc_locke::api::{validate, ValidateOptions};
    use cjc_locke::validation::ValidationConfig;
    let df = diabetes130_weight_fixture();
    let mut cfg = ValidationConfig::default();
    cfg.collect_per_value_lineage = true;
    let opts = ValidateOptions {
        dataset_label: "det".into(),
        config: cfg,
        ..Default::default()
    };
    let a = validate(&df, &opts);
    let b = validate(&df, &opts);
    assert_eq!(a.per_value_lineage, b.per_value_lineage);
    assert_eq!(a.findings, b.findings);
}
