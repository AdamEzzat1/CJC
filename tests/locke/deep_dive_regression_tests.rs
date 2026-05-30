//! Regression tests for the v0.7+ deep-dive CRITICAL bug fixes.
//!
//! Each test in this module locks in the *post-fix* behaviour for a bug
//! the audit surfaced — if any of these fail, a future change has
//! re-introduced one of the originally-latent silent-correctness bugs:
//!
//! 1. `CategoricalAdaptive` duplicate detection (validation.rs::cell_bytes
//!    used to write the row index, so every row was unique → E9003/E9004
//!    never fired on this storage variant).
//! 2. `CategoricalAdaptive` distinct count (validation.rs::distinct_count
//!    used to return the row count → E9072 ID-like leakage falsely fired
//!    on every adaptive-categorical column).
//! 3. `CategoricalAdaptive` top-value frequency (validation.rs::top_value_freq
//!    used to return 0 → near-constant detection never fired).
//! 4. KS-D over tied-value clusters (stats.rs::ks_d_statistic used to
//!    over-report D by emitting gaps at intermediate single-side advances,
//!    breaking the textbook step-function CDF).

use cjc_data::{byte_dict::CategoricalColumn, Column, DataFrame};
use cjc_locke::{
    api::{validate, ValidateOptions},
    stats::ks_d_statistic,
    validation::detect_duplicates_full_row,
    ValidationConfig,
};

// ─── Helper: build an adaptive-categorical column from string values ──

fn cat_adaptive_from_strs(values: &[&str]) -> Column {
    let mut cc = CategoricalColumn::new();
    for v in values {
        cc.push(v.as_bytes())
            .expect("byte_dict::push within capacity");
    }
    Column::CategoricalAdaptive(Box::new(cc))
}

// ─── BUG 1: CategoricalAdaptive duplicate detection ──────────────────

#[test]
fn fix_categorical_adaptive_duplicate_detection_now_fires() {
    // Build a DataFrame whose CategoricalAdaptive column has obvious
    // duplicates. Pre-fix: zero duplicates ever reported.
    let df = DataFrame::from_columns(vec![(
        "tier".into(),
        cat_adaptive_from_strs(&[
            "premium", "basic", "premium", "premium", "basic", "premium",
        ]),
    )])
    .unwrap();
    let cfg = ValidationConfig::default();
    let findings = detect_duplicates_full_row(&df, &cfg);
    // Now we expect at least one E9003 (full-row duplicates) finding.
    // Pre-fix: findings was empty.
    assert!(
        findings.iter().any(|f| f.code == "E9003"),
        "expected E9003 on duplicate-rich CategoricalAdaptive column, got {:?}",
        findings.iter().map(|f| f.code).collect::<Vec<_>>()
    );
}

#[test]
fn fix_categorical_adaptive_cross_storage_consistency() {
    // Same logical content via Column::Categorical and Column::CategoricalAdaptive
    // should produce the SAME finding set (modulo the tag the byte stream
    // uses, which is now `b'c'` for both). The fix uses a shared tag so
    // the two storage variants emit identical cell_bytes for equal values.
    let values = vec!["premium", "premium", "basic", "premium"];
    let cat_df = DataFrame::from_columns(vec![(
        "x".into(),
        Column::Categorical {
            levels: vec!["premium".into(), "basic".into()],
            codes: vec![0, 0, 1, 0],
        },
    )])
    .unwrap();
    let adapt_df = DataFrame::from_columns(vec![(
        "x".into(),
        cat_adaptive_from_strs(&values),
    )])
    .unwrap();
    let cfg = ValidationConfig::default();
    let cat_findings = detect_duplicates_full_row(&cat_df, &cfg);
    let adapt_findings = detect_duplicates_full_row(&adapt_df, &cfg);
    // Same number of findings; same codes; same evidence shape.
    assert_eq!(cat_findings.len(), adapt_findings.len());
    let cat_codes: Vec<_> = cat_findings.iter().map(|f| f.code).collect();
    let adapt_codes: Vec<_> = adapt_findings.iter().map(|f| f.code).collect();
    assert_eq!(cat_codes, adapt_codes);
}

// ─── BUG 2: CategoricalAdaptive distinct_count ───────────────────────

#[test]
fn fix_categorical_adaptive_does_not_misfire_e9072_id_like() {
    // Column with very low cardinality (2 distinct values, 1000 rows) —
    // pre-fix, distinct_count returned 1000 → distinct/n_rows = 1.0 →
    // E9072 (ID-like leakage candidate, NOT what we want for a low-card
    // column). Post-fix, distinct_count returns 2 → ratio 0.002 → no E9072.
    let values: Vec<&str> = (0..1000)
        .map(|i| if i % 2 == 0 { "A" } else { "B" })
        .collect();
    let df = DataFrame::from_columns(vec![(
        "low_card".into(),
        cat_adaptive_from_strs(&values),
    )])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "test".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // The fix means E9072 should NOT fire on this low-cardinality column.
    let e9072_fires = report.findings.iter().any(|f| f.code == "E9072");
    assert!(
        !e9072_fires,
        "E9072 should not fire on a 2-distinct-value adaptive-categorical column, got findings: {:?}",
        report.findings.iter().map(|f| (f.code, &f.message)).collect::<Vec<_>>()
    );
}

// ─── BUG 3: CategoricalAdaptive top_value_freq ───────────────────────

#[test]
fn fix_categorical_adaptive_near_constant_now_fires() {
    // 99% "common" + 1% "rare" — passes the default 0.99 near-constant
    // threshold. Pre-fix, `top_value_freq` for CategoricalAdaptive
    // returned 0 unconditionally → ratio 0.0 → E9011 never fired even
    // on truly near-constant adaptive-categorical columns. Post-fix,
    // ratio = 99/100 = 0.99 → E9011 fires as expected.
    let mut values: Vec<&str> = Vec::new();
    for _ in 0..99 {
        values.push("common");
    }
    values.push("rare");
    let df = DataFrame::from_columns(vec![(
        "x".into(),
        cat_adaptive_from_strs(&values),
    )])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "test".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    assert!(
        report.findings.iter().any(|f| f.code == "E9011"),
        "expected E9011 (near-constant) on 99%/1% adaptive-categorical column, got {:?}",
        report.findings.iter().map(|f| f.code).collect::<Vec<_>>()
    );
}

// ─── BUG 4: KS-D over tied-value clusters ────────────────────────────

#[test]
fn fix_ks_d_returns_zero_on_textbook_tied_identical_samples() {
    // The canonical failing case from the audit: a=[1,1,2,2], b=[1,2].
    // Both samples represent the same step-function CDF (50% at 1, 100% at 2),
    // so D should be exactly 0. Pre-fix: returned 0.25.
    let a = vec![1.0, 1.0, 2.0, 2.0];
    let b = vec![1.0, 2.0];
    let d = ks_d_statistic(&a, &b).expect("non-empty input");
    assert_eq!(d, 0.0, "tied-cluster samples representing the same CDF must produce D=0");
}

#[test]
fn fix_ks_d_handles_heavily_tied_integer_columns() {
    // A 100-row column of values in {0, 1, 2, 3} compared to itself
    // (effectively) — D must be 0.
    let a: Vec<f64> = (0..100).map(|i| (i % 4) as f64).collect();
    let b: Vec<f64> = (0..100).map(|i| (i % 4) as f64).collect();
    let d = ks_d_statistic(&a, &b).expect("non-empty");
    assert_eq!(d, 0.0, "identical heavily-tied samples must produce D=0");
}

#[test]
fn fix_ks_d_disjoint_supports_still_max_d() {
    // Validates we didn't regress the non-tied case: disjoint supports
    // must still produce D=1.0 (the maximum possible).
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![10.0, 20.0, 30.0];
    let d = ks_d_statistic(&a, &b).expect("non-empty");
    assert_eq!(d, 1.0, "disjoint supports must produce D=1");
}

#[test]
fn fix_ks_d_is_deterministic_across_runs() {
    // Determinism check: same input → same output bit-for-bit.
    let a: Vec<f64> = (0..50).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..50).map(|i| (i % 5) as f64).collect();
    let d1 = ks_d_statistic(&a, &b);
    let d2 = ks_d_statistic(&a, &b);
    assert_eq!(d1, d2);
    // Plus bit-equal for the Some case
    if let (Some(x), Some(y)) = (d1, d2) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn fix_ks_d_textbook_one_sided_shift() {
    // a shifted entirely below b → D = 1.0 even with ties on each side.
    let a = vec![1.0, 1.0, 1.0, 2.0];
    let b = vec![10.0, 10.0, 10.0, 20.0];
    let d = ks_d_statistic(&a, &b).expect("non-empty");
    assert_eq!(d, 1.0);
}
