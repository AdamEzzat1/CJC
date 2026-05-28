//! Train/test drift and induction-risk analysis.
//!
//! Locke compares two `cjc_data::DataFrame`s — usually `train` and `test`
//! or `train` and `prod` — and emits a deterministic
//! `InductionRiskReport` containing per-column drift findings.
//!
//! ## Implemented in v0
//!
//! * **Numeric mean/std/range shift** — relative shifts in mean and std,
//!   absolute shift in min/max.
//! * **Missingness rate shift** — `|p_train_missing − p_test_missing|`.
//! * **Category-frequency shift** — total variation distance between
//!   normalised level frequencies; categories present in only one side
//!   are surfaced explicitly.
//! * **PSI-like score** — Population Stability Index over equal-width
//!   bins on numeric columns and over level frequencies on categorical.
//! * **Sample-size warnings** — small `n_test` triggers a conservative
//!   "low-power" finding.
//!
//! ## Not implemented in v0 (explicit deferral)
//!
//! * Kolmogorov–Smirnov D statistic with correct CDF accumulation —
//!   bin-grid approximation is fine for v0; exact KS is deferred to v0.2.
//! * Label-drift conditional on features (requires modelling).
//! * Time-split detection (requires user-supplied time column).

use std::collections::BTreeMap;

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};
use crate::stats::{ks_d_statistic, summarize_f64, summarize_i64, NumericSummary};

#[derive(Clone, Debug)]
pub struct DriftConfig {
    pub mean_shift_warn: f64,
    pub mean_shift_error: f64,
    pub std_shift_warn: f64,
    /// PSI thresholds remain available for the legacy PSI helper; the
    /// default numeric drift signal in v0.2+ is the exact KS D-statistic.
    pub psi_warn: f64,
    pub psi_error: f64,
    /// Kolmogorov–Smirnov D thresholds (v0.2). D ≥ ks_d_warn → Warning,
    /// D ≥ ks_d_error → Error.
    pub ks_d_warn: f64,
    pub ks_d_error: f64,
    pub category_tvd_warn: f64,
    pub category_tvd_error: f64,
    pub missingness_shift_warn: f64,
    pub small_sample_threshold: u64,
    pub n_bins: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            mean_shift_warn: 0.10,
            mean_shift_error: 0.30,
            std_shift_warn: 0.25,
            psi_warn: 0.10,
            psi_error: 0.25,
            ks_d_warn: 0.10,
            ks_d_error: 0.20,
            category_tvd_warn: 0.10,
            category_tvd_error: 0.25,
            missingness_shift_warn: 0.05,
            small_sample_threshold: 30,
            n_bins: 10,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InductionRiskReport {
    pub findings: Vec<ValidationFinding>,
    pub n_train: u64,
    pub n_test: u64,
    pub shared_columns: Vec<String>,
    pub train_only_columns: Vec<String>,
    pub test_only_columns: Vec<String>,
}

impl InductionRiskReport {
    pub fn worst_severity(&self) -> FindingSeverity {
        self.findings
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(FindingSeverity::Info)
    }
}

fn numeric_summary(col: &Column) -> Option<NumericSummary> {
    match col {
        Column::Float(v) => Some(summarize_f64(v)),
        Column::Int(v) => Some(summarize_i64(v)),
        _ => None,
    }
}

/// PSI = Σᵢ (qᵢ - pᵢ) · ln(qᵢ / pᵢ), with both ratios floored at `eps`.
/// Inputs must be non-empty and the same length. Determinism: routes
/// through the same Kahan summation rule we use elsewhere.
fn psi(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let eps = 1e-6;
    let mut acc = cjc_repro::KahanAccumulatorF64::new();
    for i in 0..p.len() {
        let pi = p[i].max(eps);
        let qi = q[i].max(eps);
        acc.add((qi - pi) * (qi / pi).ln());
    }
    acc.finalize()
}

fn normalize_counts(counts: &[u64]) -> Vec<f64> {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return vec![0.0; counts.len()];
    }
    let t = total as f64;
    counts.iter().map(|c| *c as f64 / t).collect()
}

fn psi_severity(score: f64, cfg: &DriftConfig) -> Option<FindingSeverity> {
    if !score.is_finite() {
        return Some(FindingSeverity::Warning);
    }
    if score >= cfg.psi_error {
        Some(FindingSeverity::Error)
    } else if score >= cfg.psi_warn {
        Some(FindingSeverity::Warning)
    } else {
        None
    }
}

fn relative_shift(a: f64, b: f64) -> f64 {
    let denom = a.abs().max(1e-12);
    ((a - b).abs() / denom).min(1e6)
}

fn numeric_drift_findings(
    column: &str,
    train: &NumericSummary,
    test: &NumericSummary,
    cfg: &DriftConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let (Some(t_mean), Some(s_mean)) = (train.mean, test.mean) else {
        return out;
    };
    let mean_shift = relative_shift(t_mean, s_mean);
    let mean_sev = if mean_shift >= cfg.mean_shift_error {
        FindingSeverity::Error
    } else if mean_shift >= cfg.mean_shift_warn {
        FindingSeverity::Warning
    } else {
        FindingSeverity::Info
    };
    if mean_sev != FindingSeverity::Info {
        out.push(ValidationFinding::new(
            "E9030",
            mean_sev,
            format!(
                "mean of `{}` shifted: train={:.6}, test={:.6} (relative shift {:.4})",
                column, t_mean, s_mean, mean_shift
            ),
            Some(column.into()),
            None,
            vec![
                FindingEvidence::Metric {
                    label: "train_mean".into(),
                    value: t_mean,
                },
                FindingEvidence::Metric {
                    label: "test_mean".into(),
                    value: s_mean,
                },
                FindingEvidence::Metric {
                    label: "relative_shift".into(),
                    value: mean_shift,
                },
            ],
            train.n_total + test.n_total,
            vec!["population-variance summary; not an inferential test".into()],
            vec![
                "inspect distribution shape with a histogram or kde".into(),
                "if shift is intended (calibration), document it".into(),
            ],
        ));
    }

    if let (Some(t_std), Some(s_std)) = (train.std_dev, test.std_dev) {
        let std_shift = relative_shift(t_std, s_std);
        if std_shift >= cfg.std_shift_warn {
            out.push(ValidationFinding::new(
                "E9031",
                FindingSeverity::Warning,
                format!(
                    "std of `{}` shifted: train={:.6}, test={:.6} (relative shift {:.4})",
                    column, t_std, s_std, std_shift
                ),
                Some(column.into()),
                None,
                vec![
                    FindingEvidence::Metric {
                        label: "train_std".into(),
                        value: t_std,
                    },
                    FindingEvidence::Metric {
                        label: "test_std".into(),
                        value: s_std,
                    },
                    FindingEvidence::Metric {
                        label: "relative_shift".into(),
                        value: std_shift,
                    },
                ],
                train.n_total + test.n_total,
                vec!["population variance; not an F-test".into()],
                vec!["check whether new sources or units were introduced".into()],
            ));
        }
    }

    if let (Some(t_min), Some(s_min), Some(t_max), Some(s_max)) =
        (train.min, test.min, train.max, test.max)
    {
        if s_min < t_min || s_max > t_max {
            out.push(ValidationFinding::new(
                "E9032",
                FindingSeverity::Notice,
                format!(
                    "range of `{}` widened: train=[{:.3}, {:.3}], test=[{:.3}, {:.3}]",
                    column, t_min, t_max, s_min, s_max
                ),
                Some(column.into()),
                None,
                vec![
                    FindingEvidence::Range {
                        label: "train_range".into(),
                        min: t_min,
                        max: t_max,
                    },
                    FindingEvidence::Range {
                        label: "test_range".into(),
                        min: s_min,
                        max: s_max,
                    },
                ],
                train.n_total + test.n_total,
                vec!["test data has values outside the train support".into()],
                vec![
                    "extrapolation risk: model may behave unpredictably".into(),
                    "consider clipping or re-training with wider data".into(),
                ],
            ));
        }
    }

    out
}

/// Exact Kolmogorov–Smirnov D as the default numeric drift signal (v0.2).
///
/// Emits **E9039** when the configured threshold is crossed. Replaces the
/// equal-width-binned PSI proxy (`numeric_psi_finding`) which is retained
/// as a private helper for future opt-in.
fn numeric_ks_finding(
    column: &str,
    train_values: &[f64],
    test_values: &[f64],
    cfg: &DriftConfig,
) -> Option<ValidationFinding> {
    let d = ks_d_statistic(train_values, test_values)?;
    let sev = if d >= cfg.ks_d_error {
        FindingSeverity::Error
    } else if d >= cfg.ks_d_warn {
        FindingSeverity::Warning
    } else {
        return None;
    };
    let n_train_valid = train_values.iter().filter(|x| !x.is_nan()).count() as u64;
    let n_test_valid = test_values.iter().filter(|x| !x.is_nan()).count() as u64;
    Some(ValidationFinding::new(
        "E9039",
        sev,
        format!(
            "numeric distribution shift in `{}` (KS D = {:.4})",
            column, d
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Metric {
                label: "ks_d".into(),
                value: d,
            },
            FindingEvidence::Count {
                label: "n_train_valid".into(),
                value: n_train_valid,
            },
            FindingEvidence::Count {
                label: "n_test_valid".into(),
                value: n_test_valid,
            },
        ],
        n_train_valid + n_test_valid,
        vec![
            "exact two-sample KS D-statistic; not a significance test".into(),
            "NaN excluded before sorting".into(),
            format!(
                "thresholds: warn ≥ {:.2}, error ≥ {:.2}",
                cfg.ks_d_warn, cfg.ks_d_error
            ),
        ],
        vec![
            "inspect side-by-side ECDFs to localize the shift".into(),
            "if drift is real, retrain or re-calibrate".into(),
        ],
    ))
}

#[allow(dead_code)]
fn numeric_psi_finding(
    column: &str,
    train_values: &[f64],
    test_values: &[f64],
    cfg: &DriftConfig,
) -> Option<ValidationFinding> {
    let train_summary = summarize_f64(train_values);
    let test_summary = summarize_f64(test_values);
    let (t_min, t_max) = match (train_summary.min, train_summary.max) {
        (Some(a), Some(b)) if a < b => (a, b),
        _ => return None,
    };
    let s_min = test_summary.min.unwrap_or(t_min);
    let s_max = test_summary.max.unwrap_or(t_max);
    let lo = t_min.min(s_min);
    let hi = t_max.max(s_max);
    if !(hi > lo) {
        return None;
    }
    // Bin both into a shared [lo, hi] grid.
    let bins = cfg.n_bins.max(2);
    let width = (hi - lo) / bins as f64;
    let bin = |v: &[f64]| -> Vec<u64> {
        let mut c = vec![0u64; bins];
        for &x in v {
            if x.is_nan() {
                continue;
            }
            let mut idx = ((x - lo) / width) as i64;
            if idx < 0 {
                idx = 0;
            }
            if idx >= bins as i64 {
                idx = bins as i64 - 1;
            }
            c[idx as usize] += 1;
        }
        c
    };
    let p = normalize_counts(&bin(train_values));
    let q = normalize_counts(&bin(test_values));
    let score = psi(&p, &q);
    let sev = psi_severity(score, cfg)?;
    Some(ValidationFinding::new(
        "E9033",
        sev,
        format!(
            "numeric distribution shift in `{}` (PSI = {:.4})",
            column, score
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Metric {
                label: "psi".into(),
                value: score,
            },
            FindingEvidence::Count {
                label: "n_bins".into(),
                value: bins as u64,
            },
            FindingEvidence::Range {
                label: "shared_range".into(),
                min: lo,
                max: hi,
            },
        ],
        (train_values.len() + test_values.len()) as u64,
        vec![
            "PSI is computed on equal-width bins over the union of train+test ranges".into(),
            "PSI categorisation here uses Locke's default thresholds (warn ≥0.10, error ≥0.25)".into(),
        ],
        vec![
            "compare side-by-side histograms".into(),
            "if drift is real, retrain or re-calibrate".into(),
        ],
    ))
}

fn category_tvd_finding(
    column: &str,
    train: &[String],
    test: &[String],
    cfg: &DriftConfig,
) -> Option<ValidationFinding> {
    let mut train_freq: BTreeMap<&str, u64> = BTreeMap::new();
    let mut test_freq: BTreeMap<&str, u64> = BTreeMap::new();
    for s in train {
        *train_freq.entry(s.as_str()).or_insert(0) += 1;
    }
    for s in test {
        *test_freq.entry(s.as_str()).or_insert(0) += 1;
    }
    let t_total = train.len() as f64;
    let s_total = test.len() as f64;
    if t_total == 0.0 || s_total == 0.0 {
        return None;
    }
    let mut all_levels: Vec<&str> = train_freq.keys().copied().chain(test_freq.keys().copied()).collect();
    all_levels.sort();
    all_levels.dedup();
    let mut tvd_acc = cjc_repro::KahanAccumulatorF64::new();
    let mut new_in_test = 0u64;
    let mut new_in_train = 0u64;
    for lv in &all_levels {
        let p = train_freq.get(lv).copied().unwrap_or(0) as f64 / t_total;
        let q = test_freq.get(lv).copied().unwrap_or(0) as f64 / s_total;
        tvd_acc.add((p - q).abs());
        if !train_freq.contains_key(lv) {
            new_in_test += 1;
        }
        if !test_freq.contains_key(lv) {
            new_in_train += 1;
        }
    }
    let tvd = 0.5 * tvd_acc.finalize();
    let sev = if tvd >= cfg.category_tvd_error {
        FindingSeverity::Error
    } else if tvd >= cfg.category_tvd_warn {
        FindingSeverity::Warning
    } else if new_in_test > 0 {
        FindingSeverity::Notice
    } else {
        return None;
    };
    Some(ValidationFinding::new(
        "E9034",
        sev,
        format!(
            "categorical distribution shift in `{}` (TVD = {:.4}, {} new levels in test)",
            column, tvd, new_in_test
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Metric {
                label: "tvd".into(),
                value: tvd,
            },
            FindingEvidence::Count {
                label: "new_levels_in_test".into(),
                value: new_in_test,
            },
            FindingEvidence::Count {
                label: "new_levels_in_train".into(),
                value: new_in_train,
            },
        ],
        (train.len() + test.len()) as u64,
        vec!["TVD is computed over the union of categorical levels".into()],
        vec![
            "verify whether unseen levels are valid".into(),
            "consider whether to one-hot encode unseen levels or retrain".into(),
        ],
    ))
}

fn missingness_shift_finding(
    column: &str,
    train: &NumericSummary,
    test: &NumericSummary,
    cfg: &DriftConfig,
) -> Option<ValidationFinding> {
    if train.n_total == 0 || test.n_total == 0 {
        return None;
    }
    let p_train = train.n_missing as f64 / train.n_total as f64;
    let p_test = test.n_missing as f64 / test.n_total as f64;
    let delta = (p_train - p_test).abs();
    if delta < cfg.missingness_shift_warn {
        return None;
    }
    Some(ValidationFinding::new(
        "E9035",
        FindingSeverity::Warning,
        format!(
            "missingness rate of `{}` shifted: train={:.3}, test={:.3}",
            column, p_train, p_test
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Ratio {
                label: "train_missing_rate".into(),
                value: p_train,
            },
            FindingEvidence::Ratio {
                label: "test_missing_rate".into(),
                value: p_test,
            },
        ],
        train.n_total + test.n_total,
        vec!["NaN treated as missing in float columns".into()],
        vec![
            "differential missingness is a strong drift signal".into(),
            "check upstream ingestion changes".into(),
        ],
    ))
}

fn small_sample_finding(
    n_train: u64,
    n_test: u64,
    cfg: &DriftConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    if n_test < cfg.small_sample_threshold {
        out.push(ValidationFinding::new(
            "E9036",
            FindingSeverity::Warning,
            format!(
                "test set has only {} rows (threshold {}) — drift estimates have low power",
                n_test, cfg.small_sample_threshold
            ),
            None,
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_test".into(),
                    value: n_test,
                },
                FindingEvidence::Count {
                    label: "n_train".into(),
                    value: n_train,
                },
            ],
            n_train + n_test,
            vec!["small samples make distribution comparisons unreliable".into()],
            vec!["collect more test data before relying on these drift scores".into()],
        ));
    }
    out
}

/// Top-level drift comparison.
pub fn compare(
    train: &DataFrame,
    test: &DataFrame,
    cfg: &DriftConfig,
) -> InductionRiskReport {
    let n_train = train.nrows() as u64;
    let n_test = test.nrows() as u64;
    let train_cols: BTreeMap<String, &Column> =
        train.columns.iter().map(|(n, c)| (n.clone(), c)).collect();
    let test_cols: BTreeMap<String, &Column> =
        test.columns.iter().map(|(n, c)| (n.clone(), c)).collect();
    let mut shared: Vec<String> = train_cols
        .keys()
        .filter(|k| test_cols.contains_key(*k))
        .cloned()
        .collect();
    shared.sort();
    let mut train_only: Vec<String> = train_cols
        .keys()
        .filter(|k| !test_cols.contains_key(*k))
        .cloned()
        .collect();
    train_only.sort();
    let mut test_only: Vec<String> = test_cols
        .keys()
        .filter(|k| !train_cols.contains_key(*k))
        .cloned()
        .collect();
    test_only.sort();

    let mut findings: Vec<ValidationFinding> = Vec::new();

    // Sample-size warning.
    findings.extend(small_sample_finding(n_train, n_test, cfg));

    // Schema-shape findings — extra/missing columns.
    for c in &train_only {
        findings.push(ValidationFinding::new(
            "E9037",
            FindingSeverity::Error,
            format!("column `{}` is in train but missing from test", c),
            Some(c.clone()),
            None,
            vec![],
            n_train + n_test,
            vec![],
            vec!["train and test schemas must match for drift comparison".into()],
        ));
    }
    for c in &test_only {
        findings.push(ValidationFinding::new(
            "E9038",
            FindingSeverity::Error,
            format!("column `{}` is in test but missing from train", c),
            Some(c.clone()),
            None,
            vec![],
            n_train + n_test,
            vec![],
            vec!["train and test schemas must match for drift comparison".into()],
        ));
    }

    // Per-column drift.
    for col in &shared {
        let train_col = train_cols[col];
        let test_col = test_cols[col];
        // Numeric path.
        if let (Some(t_sum), Some(s_sum)) = (numeric_summary(train_col), numeric_summary(test_col)) {
            findings.extend(numeric_drift_findings(col, &t_sum, &s_sum, cfg));
            findings.extend(missingness_shift_finding(col, &t_sum, &s_sum, cfg));
            let (train_vals, test_vals) = match (train_col, test_col) {
                (Column::Float(t), Column::Float(s)) => (t.clone(), s.clone()),
                (Column::Float(t), Column::Int(s)) => {
                    (t.clone(), s.iter().map(|x| *x as f64).collect())
                }
                (Column::Int(t), Column::Float(s)) => {
                    (t.iter().map(|x| *x as f64).collect(), s.clone())
                }
                (Column::Int(t), Column::Int(s)) => (
                    t.iter().map(|x| *x as f64).collect(),
                    s.iter().map(|x| *x as f64).collect(),
                ),
                _ => continue,
            };
            if let Some(ks_f) = numeric_ks_finding(col, &train_vals, &test_vals, cfg) {
                findings.push(ks_f);
            }
            continue;
        }
        // Categorical path.
        let (train_levels, test_levels): (Vec<String>, Vec<String>) = match (train_col, test_col) {
            (Column::Str(t), Column::Str(s)) => (t.clone(), s.clone()),
            (Column::Categorical { levels: tl, codes: tc }, Column::Categorical { levels: sl, codes: sc }) => (
                tc.iter().map(|c| tl.get(*c as usize).cloned().unwrap_or_default()).collect(),
                sc.iter().map(|c| sl.get(*c as usize).cloned().unwrap_or_default()).collect(),
            ),
            (Column::Bool(t), Column::Bool(s)) => (
                t.iter().map(|b| b.to_string()).collect(),
                s.iter().map(|b| b.to_string()).collect(),
            ),
            _ => continue,
        };
        if let Some(f) = category_tvd_finding(col, &train_levels, &test_levels, cfg) {
            findings.push(f);
        }
    }

    findings.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    InductionRiskReport {
        findings,
        n_train,
        n_test,
        shared_columns: shared,
        train_only_columns: train_only,
        test_only_columns: test_only,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn mk_floats(name: &str, values: Vec<f64>) -> (String, Column) {
        (name.into(), Column::Float(values))
    }

    #[test]
    fn psi_zero_for_identical_distributions() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!(psi(&p, &q).abs() < 1e-9);
    }

    #[test]
    fn psi_positive_for_shifted_distributions() {
        let p = vec![0.5, 0.5, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.5, 0.5];
        assert!(psi(&p, &q) > 0.5);
    }

    #[test]
    fn mean_shift_flagged_when_relative_change_exceeds_threshold() {
        let train = DataFrame::from_columns(vec![mk_floats("x", vec![1.0; 100])]).unwrap();
        let test = DataFrame::from_columns(vec![mk_floats("x", vec![2.0; 100])]).unwrap();
        let r = compare(&train, &test, &DriftConfig::default());
        assert!(r.findings.iter().any(|f| f.code == "E9030"));
    }

    #[test]
    fn range_widening_flagged() {
        let train = DataFrame::from_columns(vec![mk_floats("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();
        let test =
            DataFrame::from_columns(vec![mk_floats("x", vec![1.0, 5.0, 100.0, 1000.0, 10000.0])])
                .unwrap();
        let r = compare(&train, &test, &DriftConfig::default());
        assert!(r.findings.iter().any(|f| f.code == "E9032"));
    }

    #[test]
    fn missingness_shift_flagged() {
        let train_vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut test_vals = train_vals.clone();
        for v in &mut test_vals[..50] {
            *v = f64::NAN;
        }
        let train = DataFrame::from_columns(vec![mk_floats("x", train_vals)]).unwrap();
        let test = DataFrame::from_columns(vec![mk_floats("x", test_vals)]).unwrap();
        let r = compare(&train, &test, &DriftConfig::default());
        assert!(r.findings.iter().any(|f| f.code == "E9035"));
    }

    #[test]
    fn small_test_set_warns() {
        let train = DataFrame::from_columns(vec![mk_floats("x", vec![0.0; 100])]).unwrap();
        let test = DataFrame::from_columns(vec![mk_floats("x", vec![0.0; 5])]).unwrap();
        let r = compare(&train, &test, &DriftConfig::default());
        assert!(r.findings.iter().any(|f| f.code == "E9036"));
    }

    #[test]
    fn schema_mismatch_columns_flagged() {
        let train = DataFrame::from_columns(vec![
            mk_floats("a", vec![1.0]),
            mk_floats("b", vec![1.0]),
        ])
        .unwrap();
        let test = DataFrame::from_columns(vec![mk_floats("a", vec![1.0])]).unwrap();
        let r = compare(&train, &test, &DriftConfig::default());
        assert!(r.train_only_columns.contains(&"b".to_string()));
        assert!(r.findings.iter().any(|f| f.code == "E9037"));
    }

    #[test]
    fn categorical_shift_detected() {
        let train = DataFrame::from_columns(vec![(
            "label".into(),
            Column::Str(vec!["a".into(); 100]),
        )])
        .unwrap();
        let test = DataFrame::from_columns(vec![(
            "label".into(),
            Column::Str(vec!["b".into(); 100]),
        )])
        .unwrap();
        let r = compare(&train, &test, &DriftConfig::default());
        assert!(r.findings.iter().any(|f| f.code == "E9034"));
    }

    #[test]
    fn drift_is_deterministic_across_runs() {
        let train_vals: Vec<f64> = (0..200).map(|i| (i as f64) * 0.5).collect();
        let test_vals: Vec<f64> = (0..200).map(|i| (i as f64) * 0.5 + 0.1).collect();
        let train = DataFrame::from_columns(vec![mk_floats("x", train_vals)]).unwrap();
        let test = DataFrame::from_columns(vec![mk_floats("x", test_vals)]).unwrap();
        let a = compare(&train, &test, &DriftConfig::default());
        let b = compare(&train, &test, &DriftConfig::default());
        assert_eq!(a, b);
    }
}
