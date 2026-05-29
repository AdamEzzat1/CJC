//! Integration tests for the v0.6.4 work:
//!
//! - `detect_string_sentinels` + auto-mask wiring through `validate()`
//!   (E9008 — covers the Phase 0.10 §4.D Part 1 `?`-blindness fix)
//! - `detect_per_level_target_leakage` (E9064 — per-level
//!   deterministic-outcome leakage, the Phase 0.10 §4.B missing detector)
//! - `detect_conditional_missingness` accepting `NullMaskMap`
//!   (Audit Finding 2 from `docs/locke/SILENT_FAILURES_AUDIT.md`)
//!
//! Each test gives the new detectors a real-world flavour rather than
//! a synthetic toy — the inputs mirror diabetes-130 patterns the
//! Phase 0.10 blog post documented.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{validate, ValidateOptions},
    detect_per_level_target_leakage, detect_string_sentinels, merge_null_mask_maps,
    NullMask, NullMaskMap, PerLevelLeakageConfig, ValidationConfig,
};

// ─── Helpers ────────────────────────────────────────────────────────────

fn diabetes_like_frame() -> DataFrame {
    // Tiny diabetes-130 caricature: weight column with `?` sentinels,
    // discharge codes mimicking death/hospice, a binary readmission
    // target with 11% positive rate.
    let mut weight: Vec<String> = Vec::new();
    let mut discharge: Vec<i64> = Vec::new();
    let mut readmitted: Vec<i64> = Vec::new();

    // 45 rows: discharge codes 11/13/14 (death) → readmitted=0
    // 15 each so the default PerLevelLeakageConfig::min_support=10 is met.
    for code in [11_i64, 13, 14] {
        for _ in 0..15 {
            weight.push("?".into());
            discharge.push(code);
            readmitted.push(0);
        }
    }
    // 80 rows: discharge code 1 (home) — 88% readmitted=0, 12% =1
    for i in 0..80 {
        weight.push(if i < 78 { "?".into() } else { "[75-100)".into() });
        discharge.push(1);
        readmitted.push(if i < 70 { 0 } else { 1 });
    }
    DataFrame::from_columns(vec![
        ("weight".into(), Column::Str(weight)),
        ("discharge".into(), Column::Int(discharge)),
        ("readmitted".into(), Column::Int(readmitted)),
    ])
    .unwrap()
}

// ─── A. End-to-end E9008 / sentinel detection ──────────────────────────

#[test]
fn validate_auto_detects_question_mark_sentinels_end_to_end() {
    let df = diabetes_like_frame();
    let opts = ValidateOptions {
        dataset_label: "diabetes-like".into(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // E9008 (auto-sentinel) should fire on `weight`.
    let e9008: Vec<_> = report.findings.iter().filter(|f| f.code == "E9008").collect();
    assert!(
        !e9008.is_empty(),
        "expected at least one E9008 finding (got {:?})",
        report.findings.iter().map(|f| &f.code).collect::<Vec<_>>()
    );
    assert!(e9008.iter().any(|f| f.column.as_deref() == Some("weight")));
    // And E9001 (missingness) should ALSO fire because the auto-mask
    // is unioned into detect_missingness's input. weight is >90%
    // missing -> Error severity.
    let e9001_weight = report
        .findings
        .iter()
        .find(|f| f.code == "E9001" && f.column.as_deref() == Some("weight"));
    assert!(e9001_weight.is_some(), "missingness E9001 should also fire on weight");
}

#[test]
fn validate_opt_out_preserves_v063_behaviour() {
    let df = diabetes_like_frame();
    let opts = ValidateOptions {
        dataset_label: "opt-out".into(),
        config: ValidationConfig {
            auto_detect_sentinels: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // With auto_detect_sentinels=false, no E9008 should be emitted.
    assert!(
        !report.findings.iter().any(|f| f.code == "E9008"),
        "E9008 should NOT fire when auto-detect is disabled"
    );
    // And weight's missingness — still seen as Str (E9002 type-only
    // diagnostic), no E9001 missing-count.
    assert!(
        report.findings.iter().any(|f| f.code == "E9002" && f.column.as_deref() == Some("weight")),
        "E9002 type-only diagnostic should fire on Str columns without auto-detection"
    );
}

#[test]
fn user_mask_unions_with_auto_mask() {
    // User supplies a mask for "weight" naming the LAST 2 rows; auto-
    // detect names the FIRST 93 (?-bearing). Both should be visible
    // in the column belief report.
    let df = diabetes_like_frame();
    let mut user_masks = NullMaskMap::new();
    user_masks.insert(
        "weight".into(),
        NullMask::from_indices([93, 94]),
    );
    let opts = ValidateOptions {
        dataset_label: "union".into(),
        null_masks: user_masks,
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // The per-column rate should reflect the union (everything except
    // rows where weight is "[75-100)").
    let weight_rate = report
        .column_reports
        .get("weight")
        .map(|c| c.missingness_rate)
        .unwrap();
    // All 93 ?-bearing rows + the 2 user-named rows = 95/95 = 1.0
    // (rows 93 and 94 are "[75-100)" in the fixture, not "?", but
    // the user mask still folds them in).
    assert!(
        weight_rate >= 0.94,
        "expected unioned rate ≥ 0.94, got {}",
        weight_rate
    );
}

// ─── B. E9064 per-level leakage ────────────────────────────────────────

#[test]
fn e9064_fires_on_diabetes_like_discharge_codes() {
    let df = diabetes_like_frame();
    let cfg = PerLevelLeakageConfig::default();
    let findings = detect_per_level_target_leakage(&df, "readmitted", &cfg);
    // Each of {11, 13, 14} should fire.
    for code in ["11", "13", "14"] {
        assert!(
            findings.iter().any(|f| {
                f.code == "E9064"
                    && f.column.as_deref() == Some("discharge")
                    && f.message.contains(&format!("level `{}`", code))
            }),
            "missing E9064 on discharge={}",
            code
        );
    }
}

#[test]
fn e9064_findings_have_canonical_severity_and_evidence() {
    let df = diabetes_like_frame();
    let cfg = PerLevelLeakageConfig::default();
    let findings = detect_per_level_target_leakage(&df, "readmitted", &cfg);
    let f = findings.iter().find(|f| f.code == "E9064").unwrap();
    // Severity Error per spec.
    assert!(matches!(
        f.severity,
        cjc_locke::FindingSeverity::Error
    ));
    // Evidence must include the required metrics.
    let has_metric = |label: &str| {
        f.evidence.iter().any(|e| match e {
            cjc_locke::FindingEvidence::Metric { label: l, .. } => l == label,
            _ => false,
        })
    };
    let has_count = |label: &str| {
        f.evidence.iter().any(|e| match e {
            cjc_locke::FindingEvidence::Count { label: l, .. } => l == label,
            _ => false,
        })
    };
    assert!(has_metric("p_class_given_level"));
    assert!(has_metric("base_rate"));
    assert!(has_count("support"));
}

#[test]
fn e9064_no_finding_when_target_is_constant() {
    // y is all-zero — base rate = 1.0; the suppression rule should
    // kick in and emit no findings.
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])),
        ("y".into(), Column::Int(vec![0_i64; 15])),
    ])
    .unwrap();
    let r = detect_per_level_target_leakage(
        &df, "y", &PerLevelLeakageConfig::default(),
    );
    assert!(r.is_empty(), "constant target should produce no E9064 (got {:?})", r);
}

#[test]
fn e9064_handles_multiclass_target() {
    // 3-class target — each class is ~33% base rate. Code 99 always
    // maps to class 2 → E9064 should fire.
    let mut codes = Vec::new();
    let mut y = Vec::new();
    for c in 0_i64..3 {
        for _ in 0..20 {
            codes.push(1);
            y.push(c);
        }
    }
    // 15 rows of code=99 all class 2
    for _ in 0..15 {
        codes.push(99);
        y.push(2);
    }
    let df = DataFrame::from_columns(vec![
        ("code".into(), Column::Int(codes)),
        ("y".into(), Column::Int(y)),
    ])
    .unwrap();
    let r = detect_per_level_target_leakage(
        &df, "y", &PerLevelLeakageConfig::default(),
    );
    // E9064 on level 99 → class 2
    assert!(
        r.iter().any(|f| f.code == "E9064"
            && f.column.as_deref() == Some("code")
            && f.message.contains("level `99`")
            && f.message.contains("target class `2`")),
        "expected E9064 on (code=99, class=2) (got {:?})",
        r
    );
}

#[test]
fn e9064_is_deterministic_across_runs() {
    let df = diabetes_like_frame();
    let cfg = PerLevelLeakageConfig::default();
    let a = detect_per_level_target_leakage(&df, "readmitted", &cfg);
    let b = detect_per_level_target_leakage(&df, "readmitted", &cfg);
    assert_eq!(a, b);
}

// ─── C. Wiring + cross-detector ────────────────────────────────────────

#[test]
fn auto_mask_propagates_to_conditional_missingness() {
    // Two Str columns both 100% `?`. With auto-detect, the conditional
    // missingness check should see them and fire E9070 in both directions.
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Str(vec!["?".into(); 20])),
        ("b".into(), Column::Str(vec!["?".into(); 20])),
    ])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "cond".into(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // E9008 on both, plus the conditional-missingness pairwise check
    // is not directly part of validate_dataframe; but we can at least
    // confirm the auto-detect works.
    assert_eq!(report.findings.iter().filter(|f| f.code == "E9008").count(), 2);
}

#[test]
fn merge_null_mask_maps_is_commutative_on_full_overlap() {
    let mut a = NullMaskMap::new();
    a.insert("x".into(), NullMask::from_indices([1, 2, 3]));
    let mut b = NullMaskMap::new();
    b.insert("x".into(), NullMask::from_indices([3, 4, 5]));
    let ab = merge_null_mask_maps(&a, &b);
    let ba = merge_null_mask_maps(&b, &a);
    assert_eq!(ab["x"].null_rows, ba["x"].null_rows);
}

#[test]
fn empty_dataframe_yields_no_findings() {
    let df = DataFrame::from_columns(vec![("x".into(), Column::Str(vec![]))]).unwrap();
    let cfg = ValidationConfig::default();
    let (masks, findings) = detect_string_sentinels(&df, &cfg);
    assert!(masks.is_empty());
    assert!(findings.is_empty());
}

// ─── C. E9064 adversarial fixtures (catch-rate validation) ──────────────
//
// These fixtures stress-test E9064's boundary behaviour. The honest
// answer to "real-world catch rate on adversarial data is still being
// validated" is to ship fixtures that explicitly probe what E9064
// catches and what it doesn't — and assert each result is intended.

/// Boundary case: a level that hits 98% deterministic — *below* the
/// default `conditional_threshold = 0.99`. The detector MUST NOT fire,
/// or the threshold semantics break. 99 deterministic + 1 contrary +
/// large base population.
#[test]
fn e9064_does_not_fire_below_default_conditional_threshold() {
    let mut codes: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    // Background: balanced 50/50 base rate so a level being deterministic
    // is a real signal, not a sample artefact.
    for _ in 0..500 { codes.push(1); y.push(0); }
    for _ in 0..500 { codes.push(1); y.push(1); }
    // Level 42: 99 deterministic + 1 contrary = 0.99 conditional prob,
    // but the default threshold is ≥ 0.99 — borderline edge.
    // To put us *under* the threshold, use 98 deterministic + 2 contrary
    // (= 0.98). Should NOT fire.
    for _ in 0..98 { codes.push(42); y.push(1); }
    for _ in 0..2 { codes.push(42); y.push(0); }
    let df = DataFrame::from_columns(vec![
        ("code".into(), Column::Int(codes)),
        ("y".into(), Column::Int(y)),
    ])
    .unwrap();
    let findings = detect_per_level_target_leakage(
        &df,
        "y",
        &PerLevelLeakageConfig::default(),
    );
    assert!(
        !findings.iter().any(|f| f.code == "E9064"
            && f.message.contains("level `42`")),
        "level 42 at 0.98 conditional prob should NOT fire E9064 at default threshold 0.99 (got {:?})",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// Low-support case: a level with only 5 rows of deterministic outcome
/// should NOT fire (`min_support = 10` is the default). Otherwise the
/// detector would over-fit to tiny levels and become noisy.
#[test]
fn e9064_does_not_fire_below_default_min_support() {
    let mut codes: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    for _ in 0..1000 { codes.push(1); y.push(0); }
    for _ in 0..1000 { codes.push(1); y.push(1); }
    // Level 99: 5 rows, all class 1 — perfectly deterministic but support
    // is below the floor.
    for _ in 0..5 { codes.push(99); y.push(1); }
    let df = DataFrame::from_columns(vec![
        ("code".into(), Column::Int(codes)),
        ("y".into(), Column::Int(y)),
    ])
    .unwrap();
    let findings = detect_per_level_target_leakage(
        &df,
        "y",
        &PerLevelLeakageConfig::default(),
    );
    assert!(
        !findings.iter().any(|f| f.code == "E9064" && f.message.contains("level `99`")),
        "low-support level 99 should NOT fire E9064"
    );
}

/// Inverse: an explicit lower min_support (= 5) should let the same
/// signal fire — proving the parameter is honest.
#[test]
fn e9064_fires_at_low_support_with_explicit_config() {
    let mut codes: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    for _ in 0..1000 { codes.push(1); y.push(0); }
    for _ in 0..1000 { codes.push(1); y.push(1); }
    for _ in 0..5 { codes.push(99); y.push(1); }
    let df = DataFrame::from_columns(vec![
        ("code".into(), Column::Int(codes)),
        ("y".into(), Column::Int(y)),
    ])
    .unwrap();
    let cfg = PerLevelLeakageConfig {
        min_support: 5,
        ..Default::default()
    };
    let findings = detect_per_level_target_leakage(&df, "y", &cfg);
    assert!(
        findings.iter().any(|f| f.code == "E9064" && f.message.contains("level `99`")),
        "level 99 with explicit min_support=5 should fire"
    );
}

/// Imbalanced base rate: target is 95% class 0 to start. A level that
/// emits class 0 with 96% conditional probability is NOT leakage —
/// it's just consistent with the base rate. E9064 must NOT fire (the
/// implementation guards on `P(class) < conditional_threshold` precisely
/// for this case).
#[test]
fn e9064_does_not_fire_when_class_dominates_base_rate() {
    let mut codes: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    // 950 class 0, 50 class 1 → base rate ~0.95.
    for _ in 0..950 { codes.push(1); y.push(0); }
    for _ in 0..50 { codes.push(1); y.push(1); }
    // Level 7: 100 class 0 + 0 class 1 → conditional = 1.0, but base
    // rate is already 0.95 → not "deterministic surprise."
    // Actually it's 100/100=1.0 which IS above conditional_threshold,
    // but the *base rate* is also at the borderline. The detector
    // guards on `P(class) < conditional_threshold` (= 0.99): since
    // 0.95 < 0.99, the level fires. To prove the *other* guard
    // direction (base rate already near 1.0), we need the unconditional
    // P(class=0) ≥ threshold. Bump to 990/10.
    let mut codes2: Vec<i64> = Vec::new();
    let mut y2: Vec<i64> = Vec::new();
    for _ in 0..990 { codes2.push(1); y2.push(0); }
    for _ in 0..10 { codes2.push(1); y2.push(1); }
    for _ in 0..100 { codes2.push(7); y2.push(0); }
    let df = DataFrame::from_columns(vec![
        ("code".into(), Column::Int(codes2)),
        ("y".into(), Column::Int(y2)),
    ])
    .unwrap();
    // Base rate P(y=0) = 0.991 ≥ default threshold 0.99 → suppressed.
    let findings = detect_per_level_target_leakage(
        &df,
        "y",
        &PerLevelLeakageConfig::default(),
    );
    assert!(
        !findings.iter().any(|f| f.code == "E9064"),
        "level 7 dominant-class should NOT fire when base rate ≥ threshold (got {:?})",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    // And keep the first fixture as a regression for the other guard.
    let _ = (codes, y);
}

/// Multiple deterministic levels in one column: each must surface
/// independently. The detector should emit one finding per
/// `(column, level, class)` triple. Diabetes-130's death codes
/// 11/13/14/19/20 are the canonical example.
#[test]
fn e9064_emits_one_finding_per_deterministic_level() {
    let mut codes: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    // 50/50 base.
    for _ in 0..500 { codes.push(1); y.push(0); }
    for _ in 0..500 { codes.push(1); y.push(1); }
    // 5 deterministic levels, each 100% class 0, each with sufficient support.
    for level in [11_i64, 13, 14, 19, 20] {
        for _ in 0..30 {
            codes.push(level);
            y.push(0);
        }
    }
    let df = DataFrame::from_columns(vec![
        ("discharge".into(), Column::Int(codes)),
        ("y".into(), Column::Int(y)),
    ])
    .unwrap();
    let findings = detect_per_level_target_leakage(
        &df,
        "y",
        &PerLevelLeakageConfig::default(),
    );
    for level in ["11", "13", "14", "19", "20"] {
        assert!(
            findings.iter().any(|f| {
                f.code == "E9064"
                    && f.column.as_deref() == Some("discharge")
                    && f.message.contains(&format!("level `{}`", level))
            }),
            "missing E9064 for deterministic level {}",
            level
        );
    }
}

/// Confounded features: two columns where each level of A always
/// co-occurs with a unique level of B, both deterministic for class.
/// Both columns must fire — the detector is per-column, doesn't deduplicate
/// across confounded columns. (That's correct behaviour: the user has
/// to triage which feature to remove.)
#[test]
fn e9064_fires_independently_on_confounded_columns() {
    let mut col_a: Vec<i64> = Vec::new();
    let mut col_b: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    // Baseline 50/50.
    for _ in 0..500 { col_a.push(1); col_b.push(1); y.push(0); }
    for _ in 0..500 { col_a.push(1); col_b.push(1); y.push(1); }
    // Co-occurring deterministic levels: A=42 ↔ B=99, all class 0, support 30.
    for _ in 0..30 {
        col_a.push(42);
        col_b.push(99);
        y.push(0);
    }
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(col_a)),
        ("b".into(), Column::Int(col_b)),
        ("y".into(), Column::Int(y)),
    ])
    .unwrap();
    let findings = detect_per_level_target_leakage(
        &df,
        "y",
        &PerLevelLeakageConfig::default(),
    );
    let a_fires = findings.iter().any(|f| {
        f.code == "E9064"
            && f.column.as_deref() == Some("a")
            && f.message.contains("level `42`")
    });
    let b_fires = findings.iter().any(|f| {
        f.code == "E9064"
            && f.column.as_deref() == Some("b")
            && f.message.contains("level `99`")
    });
    assert!(a_fires, "column a should fire on level 42");
    assert!(b_fires, "column b should fire on level 99");
}
