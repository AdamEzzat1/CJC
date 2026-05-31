//! Ground-truth corpus tests (v0.3-T5).
//!
//! Each fixture is synthetic and has **exactly known** statistical
//! properties. Locke must detect those properties — no more, no less —
//! at the configured thresholds.
//!
//! Why not real datasets? Privacy / license / provenance hazards.
//! Synthetic data with seeded properties gives the same coverage
//! without those headaches.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::drift::{compare, DriftConfig};
use cjc_locke::stats::ks_d_statistic;
use cjc_locke::validation::ValidationConfig;
use cjc_locke::FindingEvidence;
use cjc_repro::Rng;

/// Synthetic dataset with exactly K NaN values in column `x`.
fn missingness_corpus(n: usize, k: usize, seed: u64) -> DataFrame {
    assert!(k <= n);
    let mut rng = Rng::seeded(seed);
    let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // Sample k distinct positions deterministically.
    let mut positions: Vec<usize> = (0..n).collect();
    // Deterministic shuffle via SplitMix64 keying.
    for i in (1..positions.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        positions.swap(i, j);
    }
    for &p in &positions[..k] {
        v[p] = f64::NAN;
    }
    DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap()
}

/// Exactly K duplicate row *pairs* (so total dup rows = K, found groups = K).
fn duplicates_corpus(n_unique: usize, n_dup_pairs: usize) -> DataFrame {
    assert!(n_dup_pairs <= n_unique);
    let mut ids: Vec<i64> = (0..n_unique as i64).collect();
    let mut values: Vec<f64> = (0..n_unique).map(|i| i as f64 * 1.5).collect();
    for i in 0..n_dup_pairs {
        ids.push(ids[i]);
        values.push(values[i]);
    }
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(ids)),
        ("value".into(), Column::Float(values)),
    ])
    .unwrap()
}

/// A drift corpus: two columns with a known KS D between them.
fn drift_corpus_known_d() -> (DataFrame, DataFrame, f64) {
    // Train: uniform on [0, 1]. Test: uniform on [0.5, 1.5].
    // Empirical KS D for these large samples ≈ 0.5 (the horizontal shift).
    let train_vals: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
    let test_vals: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0 + 0.5).collect();
    let expected_d = ks_d_statistic(&train_vals, &test_vals).unwrap();
    let train = DataFrame::from_columns(vec![("x".into(), Column::Float(train_vals))]).unwrap();
    let test = DataFrame::from_columns(vec![("x".into(), Column::Float(test_vals))]).unwrap();
    (train, test, expected_d)
}

fn n_missing_in_finding(f: &cjc_locke::ValidationFinding) -> Option<u64> {
    f.evidence.iter().find_map(|e| match e {
        FindingEvidence::Count { label, value } if label == "n_missing" => Some(*value),
        _ => None,
    })
}

fn n_duplicates_in_finding(f: &cjc_locke::ValidationFinding) -> Option<u64> {
    f.evidence.iter().find_map(|e| match e {
        FindingEvidence::Count { label, value } if label == "duplicate_rows" => Some(*value),
        _ => None,
    })
}

#[test]
fn ground_truth_exactly_17_nans_detected() {
    let df = missingness_corpus(100, 17, 0xCAFE);
    let opts = ValidateOptions {
        dataset_label: "gt-miss".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    let miss = r
        .findings
        .iter()
        .find(|f| f.code == "E9001")
        .expect("missingness finding");
    let n = n_missing_in_finding(miss).unwrap();
    assert_eq!(n, 17, "expected exactly 17 NaNs, got {}", n);
}

#[test]
fn ground_truth_no_nans_no_e9001() {
    let df = missingness_corpus(100, 0, 0xCAFE);
    let opts = ValidateOptions {
        dataset_label: "gt-no-miss".into(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    assert!(
        !r.findings.iter().any(|f| f.code == "E9001"),
        "no E9001 expected when there are zero NaNs"
    );
}

#[test]
fn ground_truth_exactly_5_duplicate_rows() {
    // 100 unique base rows + 5 duplicate-pair injections → 5 dup rows in 5 groups.
    let df = duplicates_corpus(100, 5);
    let opts = ValidateOptions {
        dataset_label: "gt-dup".into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    let dup = r
        .findings
        .iter()
        .find(|f| f.code == "E9003")
        .expect("duplicate finding");
    let n = n_duplicates_in_finding(dup).unwrap();
    assert_eq!(n, 5, "expected exactly 5 duplicate rows, got {}", n);
}

#[test]
fn ground_truth_known_ks_d_drift_value() {
    let (train, test, expected_d) = drift_corpus_known_d();
    let r = compare(&train, &test, &DriftConfig::default());
    let ks = r
        .findings
        .iter()
        .find(|f| f.code == "E9039")
        .expect("KS finding");
    let d = ks
        .evidence
        .iter()
        .find_map(|e| match e {
            FindingEvidence::Metric { label, value } if label == "ks_d" => Some(*value),
            _ => None,
        })
        .expect("ks_d metric on finding");
    assert!(
        (d - expected_d).abs() < 1e-12,
        "finding ks_d ({}) must equal stats::ks_d_statistic ({})",
        d,
        expected_d
    );
    assert!(
        (expected_d - 0.5).abs() < 0.01,
        "the corpus's expected D should be ~0.5, got {}",
        expected_d
    );
}

#[test]
fn ground_truth_drift_severity_at_known_threshold() {
    let (train, test, _) = drift_corpus_known_d();
    let r = compare(&train, &test, &DriftConfig::default());
    let ks = r.findings.iter().find(|f| f.code == "E9039").unwrap();
    // D ≈ 0.5 > 0.20 threshold → Error.
    assert_eq!(ks.severity, cjc_locke::FindingSeverity::Error);
}

#[test]
fn ground_truth_clean_dataset_yields_no_severe_findings() {
    // A fully clean dataset: 1000 rows, no NaN, no duplicates, no
    // constants. The only acceptable findings are Info-level.
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Float((0..1000).map(|i| i as f64).collect())),
        ("y".into(), Column::Float((0..1000).map(|i| i as f64 * 2.0).collect())),
    ])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "gt-clean".into(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    for f in &r.findings {
        assert!(
            matches!(
                f.severity,
                cjc_locke::FindingSeverity::Info | cjc_locke::FindingSeverity::Notice
            ),
            "clean dataset must not yield Warning+ findings, but saw {:?} ({}): {}",
            f.severity,
            f.code,
            f.message
        );
    }
}

#[test]
fn ground_truth_constant_column_detected() {
    let df = DataFrame::from_columns(vec![
        ("flag".into(), Column::Int(vec![1; 100])),
        ("x".into(), Column::Float((0..100).map(|i| i as f64).collect())),
    ])
    .unwrap();
    let opts = ValidateOptions {
        dataset_label: "gt-const".into(),
        ..Default::default()
    };
    let r = validate(&df, &opts);
    assert!(
        r.findings
            .iter()
            .any(|f| f.code == "E9010" && f.column.as_deref() == Some("flag")),
        "E9010 should fire on the constant `flag` column"
    );
    assert!(
        !r.findings
            .iter()
            .any(|f| f.code == "E9010" && f.column.as_deref() == Some("x")),
        "E9010 should NOT fire on the varying `x` column"
    );
}

#[test]
fn ground_truth_missingness_severity_thresholds() {
    // 5% missingness → Notice (rate < 0.10).
    let df = missingness_corpus(100, 5, 0x111);
    let r = validate(&df, &Default::default());
    let f = r.findings.iter().find(|f| f.code == "E9001").unwrap();
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Notice);

    // 30% missingness → Warning (0.10 <= rate < 0.50).
    let df = missingness_corpus(100, 30, 0x222);
    let r = validate(&df, &Default::default());
    let f = r.findings.iter().find(|f| f.code == "E9001").unwrap();
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Warning);

    // 60% missingness → Error (rate >= 0.50).
    let df = missingness_corpus(100, 60, 0x333);
    let r = validate(&df, &Default::default());
    let f = r.findings.iter().find(|f| f.code == "E9001").unwrap();
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Error);
}
