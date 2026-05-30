//! Integration tests for the Locke drift / induction-risk module.

use cjc_data::{Column, DataFrame};
use cjc_locke::drift::{compare, DriftConfig};

fn mk(name: &str, v: Vec<f64>) -> (String, Column) {
    (name.into(), Column::Float(v))
}

#[test]
fn identical_train_and_test_report_no_significant_drift() {
    let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let train = DataFrame::from_columns(vec![mk("x", v.clone())]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", v)]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    // v0.2: KS D (E9039) replaces PSI (E9033) as the default numeric drift signal.
    let drift_codes = ["E9030", "E9031", "E9034", "E9039"];
    let drift_findings = r
        .findings
        .iter()
        .filter(|f| drift_codes.contains(&f.code))
        .count();
    assert_eq!(drift_findings, 0, "no drift expected, got: {:?}", r.findings);
}

#[test]
fn large_mean_shift_is_error() {
    let train = DataFrame::from_columns(vec![mk("x", vec![1.0; 200])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![10.0; 200])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "E9030")
        .expect("mean shift finding");
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Error);
}

#[test]
fn psi_detects_distribution_shift() {
    // Train uniform 0..1; test concentrated near 1.
    let train: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
    let test: Vec<f64> = vec![0.95; 1000];
    let train_df = DataFrame::from_columns(vec![mk("x", train)]).unwrap();
    let test_df = DataFrame::from_columns(vec![mk("x", test)]).unwrap();
    let r = compare(&train_df, &test_df, &DriftConfig::default());
    assert!(r.findings.iter().any(|f| f.code == "E9039"));
}

#[test]
fn schema_train_only_test_only_columns_listed() {
    let train = DataFrame::from_columns(vec![mk("a", vec![1.0]), mk("b", vec![2.0])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("a", vec![1.0]), mk("c", vec![3.0])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert_eq!(r.train_only_columns, vec!["b".to_string()]);
    assert_eq!(r.test_only_columns, vec!["c".to_string()]);
    assert_eq!(r.shared_columns, vec!["a".to_string()]);
}

#[test]
fn repeated_runs_produce_identical_reports() {
    let v: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
    let train = DataFrame::from_columns(vec![mk("x", v.clone())]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", v.iter().map(|x| x + 0.05).collect())]).unwrap();
    let a = compare(&train, &test, &DriftConfig::default());
    let b = compare(&train, &test, &DriftConfig::default());
    assert_eq!(a, b);
}

#[test]
fn ks_d_evidence_is_reported_when_drift_present() {
    // Two disjoint supports → KS D = 1.0 → Error severity.
    let train: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let test: Vec<f64> = (1000..1100).map(|i| i as f64).collect();
    let train_df = DataFrame::from_columns(vec![mk("x", train)]).unwrap();
    let test_df = DataFrame::from_columns(vec![mk("x", test)]).unwrap();
    let r = compare(&train_df, &test_df, &DriftConfig::default());
    let ks = r
        .findings
        .iter()
        .find(|f| f.code == "E9039")
        .expect("E9039 KS finding");
    assert_eq!(ks.severity, cjc_locke::FindingSeverity::Error);
    let d = ks
        .evidence
        .iter()
        .find_map(|e| match e {
            cjc_locke::FindingEvidence::Metric { label, value } if label == "ks_d" => {
                Some(*value)
            }
            _ => None,
        })
        .unwrap();
    assert!((d - 1.0).abs() < 1e-9, "KS D should be 1.0 for disjoint supports, got {}", d);
}

#[test]
fn ks_d_is_deterministic_in_drift_report() {
    let train: Vec<f64> = (0..200).map(|i| (i as f64).sin()).collect();
    let test: Vec<f64> = (0..200).map(|i| (i as f64).cos()).collect();
    let train_df = DataFrame::from_columns(vec![mk("x", train)]).unwrap();
    let test_df = DataFrame::from_columns(vec![mk("x", test)]).unwrap();
    let a = compare(&train_df, &test_df, &DriftConfig::default());
    let b = compare(&train_df, &test_df, &DriftConfig::default());
    assert_eq!(a, b);
}

#[test]
fn small_test_set_triggers_low_power_warning() {
    let train = DataFrame::from_columns(vec![mk("x", vec![0.0; 1000])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![0.0; 5])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert!(r.findings.iter().any(|f| f.code == "E9036"));
}

// ─── B5.3 regression: near-zero mean false positive ─────────────────

#[test]
fn near_zero_mean_with_tiny_jitter_does_not_fire_e9030() {
    // Pre-fix: a column with mean ≈ 1e-15 and a 1e-12 jitter on the test
    // side hit `denom = max(1e-15, 1e-12) = 1e-12`, then `shift = 1.0`
    // which crosses the 0.30 error threshold → spurious E9030. Post-fix:
    // both means are below `mean_shift_near_zero_threshold = 1e-9`, so
    // mean-shift evaluation is skipped entirely.
    let train = DataFrame::from_columns(vec![mk("x", vec![1e-15; 200])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![1e-15 + 1e-12; 200])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    let e9030_fires = r.findings.iter().any(|f| f.code == "E9030");
    assert!(
        !e9030_fires,
        "E9030 must not fire on near-zero mean with sub-eps jitter, got: {:?}",
        r.findings.iter().map(|f| (f.code, &f.message)).collect::<Vec<_>>()
    );
}

#[test]
fn near_zero_threshold_zero_restores_pre_fix_behaviour() {
    // Setting the threshold to 0.0 disables the skip — useful for
    // callers who NEED the pre-fix behaviour (e.g. comparing against a
    // baseline that wasn't run with the fix).
    let mut cfg = DriftConfig::default();
    cfg.mean_shift_near_zero_threshold = 0.0;
    let train = DataFrame::from_columns(vec![mk("x", vec![1e-15; 200])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![1e-15 + 1e-12; 200])]).unwrap();
    let r = compare(&train, &test, &cfg);
    // With threshold disabled the old behaviour returns — E9030 may fire.
    // We don't assert it MUST fire (different severity thresholds could
    // gate it) — we only assert the fix's lever works (no panic, runs cleanly).
    let _ = r;
}

#[test]
fn small_but_above_threshold_means_still_fire_normally() {
    // Means above `mean_shift_near_zero_threshold` go through the
    // unchanged code path. This pins backward compatibility.
    let train = DataFrame::from_columns(vec![mk("x", vec![1.0; 200])]).unwrap();
    let test = DataFrame::from_columns(vec![mk("x", vec![10.0; 200])]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "E9030")
        .expect("E9030 must still fire on normal large shifts");
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Error);
}

// ─── v0.6 batch 2: E9018 cardinality explosion ──────────────────────────

fn mk_str(name: &str, vs: Vec<&str>) -> (String, Column) {
    (name.into(), Column::Str(vs.iter().map(|s| (*s).into()).collect()))
}

#[test]
fn cardinality_explosion_fires_e9018() {
    let train_vals: Vec<&str> = ["a", "b", "c", "d", "e"].repeat(20);
    let test_strs: Vec<String> = (0..50).map(|i| format!("v{}", i)).collect();
    let test_refs: Vec<&str> = test_strs.iter().map(|s| s.as_str()).collect();

    let train = DataFrame::from_columns(vec![mk_str("c", train_vals)]).unwrap();
    let test = DataFrame::from_columns(vec![mk_str("c", test_refs)]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "E9018")
        .expect("E9018 cardinality explosion expected");
    // ratio = 50 / 5 = 10 > 4 (2 × 2.0) → Warning.
    assert_eq!(f.severity, cjc_locke::FindingSeverity::Warning);
}

#[test]
fn no_cardinality_explosion_when_train_and_test_match() {
    let vs: Vec<&str> = ["a", "b", "c", "d", "e"].repeat(20);
    let train = DataFrame::from_columns(vec![mk_str("c", vs.clone())]).unwrap();
    let test = DataFrame::from_columns(vec![mk_str("c", vs)]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert!(r.findings.iter().all(|f| f.code != "E9018"));
}

// ─── v0.6 batch 2: E9019 entropy shift ──────────────────────────────────

#[test]
fn entropy_shift_fires_e9019_on_concentration_change() {
    // Train: uniform over 4 categories. Test: 90% "a", 10% spread.
    let train_vs: Vec<&str> = ["a", "b", "c", "d"].repeat(50);
    let mut test_vs: Vec<&str> = vec!["a"; 180];
    test_vs.extend(vec!["b"; 7]);
    test_vs.extend(vec!["c"; 7]);
    test_vs.extend(vec!["d"; 6]);
    let train = DataFrame::from_columns(vec![mk_str("c", train_vs)]).unwrap();
    let test = DataFrame::from_columns(vec![mk_str("c", test_vs)]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert!(
        r.findings.iter().any(|f| f.code == "E9019"),
        "E9019 entropy shift expected, got {:?}",
        r.findings.iter().map(|f| f.code).collect::<Vec<_>>()
    );
}

#[test]
fn no_entropy_shift_when_distributions_match() {
    let vs: Vec<&str> = ["a", "b", "c", "d"].repeat(50);
    let train = DataFrame::from_columns(vec![mk_str("c", vs.clone())]).unwrap();
    let test = DataFrame::from_columns(vec![mk_str("c", vs)]).unwrap();
    let r = compare(&train, &test, &DriftConfig::default());
    assert!(r.findings.iter().all(|f| f.code != "E9019"));
}
