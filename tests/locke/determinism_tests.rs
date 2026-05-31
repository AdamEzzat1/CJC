//! Determinism gates: a Locke report over the same inputs must be
//! byte-identical across repeated runs (within the same process and
//! across separate processes). These tests cover the same-process case;
//! the CLI smoke test in `commands::locke` covers the second.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, causal_guardrail, validate, ValidateOptions};
use cjc_locke::causal::CausalConfig;
use cjc_locke::drift::{compare, DriftConfig};
use cjc_locke::validation::ValidationConfig;

fn fixture_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("age".into(), Column::Float(vec![25.0, 30.0, f64::NAN, 30.0, 45.0])),
        ("score".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        (
            "label".into(),
            Column::Str(vec!["a".into(), "b".into(), "a".into(), "c".into(), "b".into()]),
        ),
    ])
    .unwrap()
}

#[test]
fn validate_is_bit_identical_across_runs() {
    let df = fixture_df();
    let opts = ValidateOptions {
        dataset_label: "determ".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let a = validate(&df, &opts);
    let b = validate(&df, &opts);
    assert_eq!(a, b);
    assert_eq!(a.run_id, b.run_id);
}

#[test]
fn belief_score_is_bit_identical_across_runs() {
    let df = fixture_df();
    let opts = ValidateOptions {
        dataset_label: "determ-b".into(),
        ..Default::default()
    };
    let a = belief_report_from_locke(&validate(&df, &opts));
    let b = belief_report_from_locke(&validate(&df, &opts));
    assert_eq!(a, b);
    assert_eq!(a.score, b.score);
}

#[test]
fn drift_is_bit_identical_across_runs() {
    let train = fixture_df();
    let mut test_cols = train.columns.clone();
    if let Some((_, Column::Float(v))) = test_cols.iter_mut().find(|(n, _)| n == "age") {
        for x in v.iter_mut() {
            *x = (*x) + 1.0;
        }
    }
    let test = DataFrame::from_columns(test_cols).unwrap();
    let cfg = DriftConfig::default();
    let a = compare(&train, &test, &cfg);
    let b = compare(&train, &test, &cfg);
    assert_eq!(a, b);
}

#[test]
fn causal_guardrail_is_bit_identical_across_runs() {
    let df = fixture_df();
    let cfg = CausalConfig::default();
    let a = causal_guardrail(&df, Some("score"), &cfg, None, false);
    let b = causal_guardrail(&df, Some("score"), &cfg, None, false);
    assert_eq!(a, b);
}

#[test]
fn finding_ids_match_across_independent_constructions() {
    use cjc_locke::report::{FindingEvidence, FindingSeverity, ValidationFinding};
    let f1 = ValidationFinding::new(
        "E9001",
        FindingSeverity::Warning,
        "x is missing 5/100",
        Some("x".into()),
        None,
        vec![
            FindingEvidence::Count {
                label: "n_missing".into(),
                value: 5,
            },
            FindingEvidence::Ratio {
                label: "rate".into(),
                value: 0.05,
            },
        ],
        100,
        vec![],
        vec![],
    );
    let f2 = ValidationFinding::new(
        "E9001",
        FindingSeverity::Warning,
        "x is missing 5/100",
        Some("x".into()),
        None,
        vec![
            FindingEvidence::Count {
                label: "n_missing".into(),
                value: 5,
            },
            FindingEvidence::Ratio {
                label: "rate".into(),
                value: 0.05,
            },
        ],
        100,
        vec![],
        vec![],
    );
    assert_eq!(f1.id, f2.id);
}
