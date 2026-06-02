//! Integration tests for the Str→Float auto-promotion layer (ADR-0042).
//!
//! Covers the gap fix's headline claims:
//!
//! - Mostly-numeric `Str` columns with sentinel-rich first rows are
//!   rebuilt as `Float` with NaN replacing sentinels.
//! - After promotion, downstream Float-only detectors (notably E9070
//!   conditional missingness) actually fire on the promoted columns.
//! - Text columns are not promoted (no false positives on `addr_state`,
//!   `term`, etc.).
//! - The behaviour can be disabled via `ValidationConfig::auto_promote_str_to_float`
//!   for users who need byte-identical reports to a pre-v0.8 baseline.
//! - Determinism survives — same input + same config → same output.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::auto_promote::{auto_promote_str_columns, PROMOTION_FINDING_CODE};
use cjc_locke::emit_locke_report_json;
use cjc_locke::validation::ValidationConfig;
use proptest::prelude::*;

/// Synthesise an LC-style frame: two `Str` columns that mimic
/// `annual_inc_joint` / `dti_joint` (mostly sentinel-rich first rows,
/// real values only when `application_type == "Joint App"`), plus a
/// plain Float column that should not be touched.
fn lc_like_frame(n_rows: usize, joint_fraction: f64) -> DataFrame {
    let n_joint = (n_rows as f64 * joint_fraction).round() as usize;
    let mut annual_inc_joint: Vec<String> = Vec::with_capacity(n_rows);
    let mut dti_joint: Vec<String> = Vec::with_capacity(n_rows);
    let mut application_type: Vec<String> = Vec::with_capacity(n_rows);
    let mut loan_amnt: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        if i < n_joint {
            annual_inc_joint.push(format!("{}", 50_000.0 + (i as f64) * 100.0));
            dti_joint.push(format!("{}", 10.0 + (i as f64) * 0.01));
            application_type.push("Joint App".into());
        } else {
            // non-joint row: sentinel everywhere on the joint columns
            annual_inc_joint.push("".into());
            dti_joint.push("".into());
            application_type.push("Individual".into());
        }
        loan_amnt.push(5000.0 + (i as f64) * 10.0);
    }
    DataFrame::from_columns(vec![
        ("annual_inc_joint".into(), Column::Str(annual_inc_joint)),
        ("dti_joint".into(), Column::Str(dti_joint)),
        ("application_type".into(), Column::Str(application_type)),
        ("loan_amnt".into(), Column::Float(loan_amnt)),
    ])
    .unwrap()
}

// ─── Unit-style integration ───────────────────────────────────────────

#[test]
fn promotion_converts_joint_columns_to_float() {
    let df = lc_like_frame(50, 0.20); // 10 joint rows, 40 non-joint
    let cfg = ValidationConfig::default();
    let (Some(promoted), findings) = auto_promote_str_columns(&df, &cfg) else {
        panic!("expected promotion on LC-like frame");
    };

    // annual_inc_joint and dti_joint should now be Float; application_type
    // (text) should still be Str.
    assert_eq!(
        promoted.get_column("annual_inc_joint").unwrap().type_name(),
        "Float"
    );
    assert_eq!(
        promoted.get_column("dti_joint").unwrap().type_name(),
        "Float"
    );
    assert_eq!(
        promoted.get_column("application_type").unwrap().type_name(),
        "Str"
    );
    // The Float column was already Float; should be unchanged.
    assert_eq!(promoted.get_column("loan_amnt").unwrap().type_name(), "Float");

    // Two E9009 findings.
    let promotion_codes: Vec<&str> = findings
        .iter()
        .filter(|f| f.code == PROMOTION_FINDING_CODE)
        .map(|f| f.column.as_deref().unwrap_or(""))
        .collect();
    assert!(promotion_codes.contains(&"annual_inc_joint"));
    assert!(promotion_codes.contains(&"dti_joint"));
    assert!(!promotion_codes.contains(&"application_type"));
}

#[test]
fn promotion_emits_e9009_through_validate() {
    let df = lc_like_frame(50, 0.20);
    let opts = ValidateOptions {
        dataset_label: "promotion".into(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // E9009 findings should be in the report.
    let e9009: Vec<_> = report.findings.iter().filter(|f| f.code == "E9009").collect();
    assert!(e9009.len() >= 2, "expected ≥2 E9009 findings, got {}", e9009.len());
    let columns: std::collections::BTreeSet<&str> = e9009
        .iter()
        .filter_map(|f| f.column.as_deref())
        .collect();
    assert!(columns.contains("annual_inc_joint"));
    assert!(columns.contains("dti_joint"));
}

#[test]
fn report_column_types_reflect_promotion() {
    let df = lc_like_frame(50, 0.20);
    let opts = ValidateOptions {
        dataset_label: "promotion".into(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // The input summary's column_types should show Float for the
    // promoted columns (so consumers reading the report see the
    // post-promotion view).
    assert_eq!(report.input.column_types.get("annual_inc_joint").map(|s| s.as_str()), Some("Float"));
    assert_eq!(report.input.column_types.get("dti_joint").map(|s| s.as_str()), Some("Float"));
    assert_eq!(report.input.column_types.get("application_type").map(|s| s.as_str()), Some("Str"));
}

#[test]
fn e9070_fires_on_joint_columns_after_v08_fixes() {
    // ADR-0042 ships TWO fixes that together unblock E9070 on LC-style
    // joint columns:
    //
    //   1. `detect_conditional_missingness` is now invoked from
    //      `validate_dataframe` (it was defined but never called pre-v0.8).
    //   2. Auto-promotion converts mostly-numeric Str columns to Float,
    //      so the NaN-implication check sees real NaN positions rather
    //      than mask-only positions.
    //
    // Either fix alone partially closes the gap; together they fully
    // close it. This test confirms the headline outcome: E9070 fires
    // on the (annual_inc_joint, dti_joint) pair with default config.
    let df = lc_like_frame(200, 0.20); // 160 non-joint rows, 40 joint rows
    let opts = ValidateOptions {
        dataset_label: "promotion-fires".into(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let e9070: Vec<_> = report.findings.iter().filter(|f| f.code == "E9070").collect();
    assert!(
        !e9070.is_empty(),
        "ADR-0042 was supposed to enable E9070 firing on LC-like joint columns; got 0 findings"
    );
    // Both directions of the implication should fire.
    let columns: std::collections::BTreeSet<&str> = e9070
        .iter()
        .filter_map(|f| f.column.as_deref())
        .collect();
    assert!(columns.contains("annual_inc_joint"));
    assert!(columns.contains("dti_joint"));
}

#[test]
fn e9070_fires_via_sentinel_mask_path_when_promotion_disabled() {
    // When auto-promotion is disabled, the columns stay Str. Locke's
    // pre-existing sentinel-detection (E9008) still folds "" into the
    // effective null mask, and `detect_conditional_missingness` (now
    // wired into the pipeline) honours masks for non-Float columns.
    // The result: E9070 fires through the mask path, not the
    // Float-NaN path. This proves both paths work; the user can opt
    // out of promotion without losing the detector.
    let df = lc_like_frame(200, 0.20);
    let cfg = ValidationConfig {
        auto_promote_str_to_float: false,
        ..ValidationConfig::default()
    };
    let opts = ValidateOptions {
        dataset_label: "mask-path".into(),
        config: cfg,
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let e9070: Vec<_> = report.findings.iter().filter(|f| f.code == "E9070").collect();
    assert!(
        !e9070.is_empty(),
        "the mask-only path also fires E9070; the wiring fix alone closes the gap for Str columns"
    );
    // But there should be NO E9009 findings (no promotion happened).
    let e9009 = report.findings.iter().filter(|f| f.code == "E9009").count();
    assert_eq!(e9009, 0);
}

#[test]
fn disabled_by_config_preserves_str_typing() {
    let df = lc_like_frame(50, 0.20);
    let cfg = ValidationConfig {
        auto_promote_str_to_float: false,
        ..ValidationConfig::default()
    };
    let opts = ValidateOptions {
        dataset_label: "off".into(),
        config: cfg,
        ..Default::default()
    };
    let report = validate(&df, &opts);
    // Joint cols stay Str.
    assert_eq!(
        report.input.column_types.get("annual_inc_joint").map(|s| s.as_str()),
        Some("Str")
    );
    // No E9009 emitted.
    let e9009 = report.findings.iter().filter(|f| f.code == "E9009").count();
    assert_eq!(e9009, 0);
}

#[test]
fn determinism_two_runs_byte_identical() {
    let df = lc_like_frame(100, 0.25);
    let opts = ValidateOptions {
        dataset_label: "determinism".into(),
        ..Default::default()
    };
    let r1 = validate(&df, &opts);
    let r2 = validate(&df, &opts);
    assert_eq!(emit_locke_report_json(&r1), emit_locke_report_json(&r2));
}

#[test]
fn promotion_does_not_touch_text_only_columns() {
    let state: Vec<String> = (0..30)
        .map(|i| match i % 3 {
            0 => "NY",
            1 => "CA",
            _ => "TX",
        })
        .map(String::from)
        .collect();
    let term: Vec<String> = (0..30)
        .map(|i| if i % 2 == 0 { " 36 months" } else { " 60 months" })
        .map(String::from)
        .collect();
    let df = DataFrame::from_columns(vec![
        ("state".into(), Column::Str(state)),
        ("term".into(), Column::Str(term)),
    ])
    .unwrap();
    let cfg = ValidationConfig::default();
    let (maybe, findings) = auto_promote_str_columns(&df, &cfg);
    assert!(maybe.is_none(), "text columns must not be promoted");
    assert!(findings.is_empty());
}

// ─── Proptest: invariants ──────────────────────────────────────────────

fn arb_str_column() -> impl Strategy<Value = Vec<String>> {
    // Mix of integer-like, float-like, sentinel, and text values.
    prop::collection::vec(
        prop_oneof![
            // Sentinels
            Just("".to_string()),
            Just("?".to_string()),
            Just("NA".to_string()),
            // Integer literals
            any::<i32>().prop_map(|i| format!("{}", i)),
            // Float literals
            any::<i32>().prop_map(|i| format!("{}.0", i)),
            // Text
            "[a-z]{3,8}".prop_map(|s: String| s),
        ],
        1..40,
    )
}

proptest! {
    #[test]
    fn proptest_promotion_is_deterministic(values in arb_str_column()) {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
        let cfg = ValidationConfig::default();
        let (a, _) = auto_promote_str_columns(&df, &cfg);
        let (b, _) = auto_promote_str_columns(&df, &cfg);
        prop_assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            // Compare promoted columns bit-identically (including NaN bits).
            for ((_, c1), (_, c2)) in a.columns.iter().zip(b.columns.iter()) {
                if let (Column::Float(v1), Column::Float(v2)) = (c1, c2) {
                    prop_assert_eq!(v1.len(), v2.len());
                    for (x, y) in v1.iter().zip(v2.iter()) {
                        prop_assert_eq!(x.to_bits(), y.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn proptest_promotion_preserves_row_count(values in arb_str_column()) {
        let n = values.len();
        let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
        let cfg = ValidationConfig::default();
        let (maybe, _) = auto_promote_str_columns(&df, &cfg);
        if let Some(promoted) = maybe {
            prop_assert_eq!(promoted.nrows(), n);
        }
    }

    #[test]
    fn proptest_already_float_never_promotes(v in prop::collection::vec(any::<i32>().prop_map(|i| i as f64), 1..40)) {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let cfg = ValidationConfig::default();
        let (maybe, findings) = auto_promote_str_columns(&df, &cfg);
        prop_assert!(maybe.is_none(), "Float columns must never be promoted");
        prop_assert!(findings.is_empty());
    }

    #[test]
    fn proptest_disabled_never_promotes(values in arb_str_column()) {
        let cfg = ValidationConfig {
            auto_promote_str_to_float: false,
            ..ValidationConfig::default()
        };
        let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
        let (maybe, findings) = auto_promote_str_columns(&df, &cfg);
        prop_assert!(maybe.is_none());
        prop_assert!(findings.is_empty());
    }
}

// ─── Bolero: never panic on arbitrary inputs ───────────────────────────

#[test]
fn fuzz_arbitrary_str_columns_never_panic() {
    bolero::check!()
        .with_type::<Vec<Vec<u8>>>()
        .for_each(|byte_columns: &Vec<Vec<u8>>| {
            if byte_columns.is_empty() {
                return;
            }
            // Build a single Str column from arbitrary byte sequences
            // (lossily UTF-8-decoded so we always have valid strings).
            let values: Vec<String> = byte_columns
                .iter()
                .take(40)
                .map(|bytes| String::from_utf8_lossy(bytes).to_string())
                .collect();
            if values.is_empty() {
                return;
            }
            let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
            let cfg = ValidationConfig::default();
            let _ = auto_promote_str_columns(&df, &cfg);
        });
}

#[test]
fn fuzz_arbitrary_config_thresholds_never_panic() {
    bolero::check!()
        .with_type::<(u8, u16)>()
        .for_each(|(frac_byte, min_count): &(u8, u16)| {
            let cfg = ValidationConfig {
                auto_promote_str_to_float: true,
                // Map u8 → [0.0, 1.0] threshold
                min_parseable_fraction_for_promotion: (*frac_byte as f64) / 255.0,
                min_non_sentinel_rows_for_promotion: *min_count as usize,
                ..ValidationConfig::default()
            };
            let values: Vec<String> = (0..20).map(|i| format!("{}", i)).collect();
            let df = DataFrame::from_columns(vec![("x".into(), Column::Str(values))]).unwrap();
            let _ = auto_promote_str_columns(&df, &cfg);
        });
}
