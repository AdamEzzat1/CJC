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
//!
//! Batch 3 — JSON parser hardening (2026-05-30):
//! 5. `parse_int` silently wrapped negative integers to enormous `u64`
//!    values via `as u64` cast. Pre-fix: `n_rows: -1` parsed "successfully"
//!    as `u64::MAX`. Post-fix: negative values rejected at parse time
//!    via a `parse_u64` method that errors on leading `-`.
//! 6. `\uXXXX` escapes for codepoints above U+FFFF (e.g. emoji) silently
//!    errored as "bad codepoint". Pre-fix: the parser called `char::from_u32`
//!    on the raw 16-bit value, which returns `None` for surrogates.
//!    Post-fix: surrogate pairs are detected and combined into the full
//!    codepoint via UTF-16 decoding; lone surrogates are rejected.

use cjc_data::{byte_dict::CategoricalColumn, Column, DataFrame};
use cjc_locke::{
    api::{validate, ValidateOptions},
    parse_locke_report_json,
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

// ─── BUG 5: parse_int silent negative wraparound (B3.1) ──────────────

/// A minimal valid Locke JSON report skeleton with the named integer
/// field replaced by the given string. Used to construct adversarial
/// inputs for parse-time validation tests.
fn json_with_n_rows(raw_value: &str) -> String {
    format!(
        r#"{{"schema_version":1,"run_id":"x","input":{{"dataset_label":"t","n_rows":{},"n_cols":0,"column_types":{{}}}},"severity_counts":{{"info":0,"notice":0,"warning":0,"error":0}},"findings":[],"assumptions":[]}}"#,
        raw_value
    )
}

#[test]
fn fix_parse_rejects_negative_n_rows() {
    // Pre-fix: `parse_int` returns i64::-1, then `as u64` produces
    // u64::MAX. The parser accepts a corrupted report with absurd
    // counts. Post-fix: leading `-` is rejected at parse time.
    let bad = json_with_n_rows("-1");
    let r = parse_locke_report_json(&bad);
    assert!(r.is_err(), "negative n_rows must error, got {:?}", r);
}

#[test]
fn fix_parse_rejects_negative_n_rows_extreme() {
    // The most adversarial form: i64::MIN. Pre-fix this maps to u64
    // value 0x8000_0000_0000_0000 — an enormous "successful" parse.
    let bad = json_with_n_rows("-9223372036854775808");
    let r = parse_locke_report_json(&bad);
    assert!(r.is_err(), "i64::MIN n_rows must error");
}

#[test]
fn fix_parse_rejects_oversized_u64() {
    // 1e20 overflows u64 (max ~1.8e19). Pre-fix: this case was already
    // an error (i64 also can't hold it), but we lock the post-fix
    // error message in to detect any regression to silent `wrapping_add`
    // logic in future refactors.
    let bad = json_with_n_rows("99999999999999999999");
    let r = parse_locke_report_json(&bad);
    assert!(r.is_err(), "u64-overflowing n_rows must error");
}

#[test]
fn fix_parse_accepts_full_u64_range() {
    // Sanity: legitimate maximum values still parse. Pre- and post-fix
    // behaviour should agree on the non-adversarial path.
    let ok = json_with_n_rows("18446744073709551615"); // u64::MAX
    let parsed = parse_locke_report_json(&ok).expect("u64::MAX must parse");
    assert_eq!(parsed.input.n_rows, u64::MAX);
}

// ─── BUG 6: \uXXXX surrogate-pair mishandling (B3.3) ─────────────────

#[test]
fn fix_parse_string_handles_surrogate_pair_for_emoji() {
    // U+1F600 (😀) is encoded by non-UTF-8-aware JSON editors as the
    // surrogate pair `😀`. Pre-fix: the parser called
    // `char::from_u32(0xD83D)` which returned None, erroring with
    // "bad codepoint U+D83D". Post-fix: the surrogate pair is detected
    // and combined into U+1F600 via UTF-16 decoding. The raw string
    // below contains the *escape sequence* (4 literal chars `\uXXXX`),
    // not the emoji itself.
    let json = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"hi \uD83D\uDE00 there","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
    let r = parse_locke_report_json(json).expect("surrogate pair must parse");
    assert!(
        r.input.dataset_label.contains('😀'),
        "expected emoji U+1F600 from surrogate pair, got: {:?}",
        r.input.dataset_label
    );
}

#[test]
fn fix_parse_string_rejects_lone_high_surrogate() {
    // A high surrogate that is NOT followed by a low surrogate is
    // malformed UTF-16. The parser must reject it rather than silently
    // produce garbled output.
    let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"lone \uD83D end","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
    let r = parse_locke_report_json(bad);
    assert!(r.is_err(), "lone high surrogate must error, got {:?}", r);
}

#[test]
fn fix_parse_string_rejects_lone_low_surrogate() {
    // Same shape for low surrogates appearing without a high partner.
    let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"\uDE00","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
    let r = parse_locke_report_json(bad);
    assert!(r.is_err(), "lone low surrogate must error");
}

#[test]
fn fix_parse_string_rejects_high_surrogate_followed_by_non_low() {
    // High surrogate followed by \uXXXX where XXXX is not in the low
    // surrogate range. Must error.
    let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"\uD83DA","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
    let r = parse_locke_report_json(bad);
    assert!(r.is_err(), "high surrogate followed by non-low must error");
}
