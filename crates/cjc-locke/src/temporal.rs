//! Time-aware validation (v0.5).
//!
//! Most churn / time-series pipelines contain temporal bugs: a feature
//! computed at score-time leaks into training data; a "train cutoff"
//! is silently violated; test rows have timestamps before the train
//! window. Locke v0.5 catches all three classes when the caller
//! declares which column is the time column.
//!
//! ## Codes
//!
//! | Code  | Severity | What it flags |
//! |-------|----------|---------------|
//! | E9050 | Warning  | time column not sorted (non-decreasing) |
//! | E9051 | Error    | train/test temporal overlap or reversed |
//! | E9052 | Error    | row has timestamp > caller-supplied cutoff |
//! | E9053 | Notice   | unusual gap between successive timestamps |
//!
//! ## Determinism
//!
//! All scans are O(n) single-pass with `i64` arithmetic — bit-stable
//! across runs.

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

#[derive(Clone, Debug)]
pub struct TimeColumnConfig {
    /// Caller-supplied "training cutoff" timestamp. Rows with `time >
    /// max_timestamp` are flagged as future-leakage candidates (E9052).
    /// `None` disables the check.
    pub max_timestamp: Option<i64>,
    /// Successive timestamps separated by more than `gap_threshold`
    /// trigger an E9053 Notice. `None` disables the check.
    pub gap_threshold: Option<i64>,
    /// Treat the time column as epoch milliseconds (default) or seconds.
    pub unit_is_millis: bool,
}

impl Default for TimeColumnConfig {
    fn default() -> Self {
        Self {
            max_timestamp: None,
            gap_threshold: None,
            unit_is_millis: true,
        }
    }
}

/// Extract a column as a `Vec<i64>` for temporal checks. Returns `None`
/// if the column isn't suitable (wrong type or missing).
fn extract_time_column(df: &DataFrame, name: &str) -> Option<Vec<i64>> {
    let col = df.get_column(name)?;
    match col {
        Column::DateTime(v) => Some(v.clone()),
        Column::Int(v) => Some(v.clone()),
        Column::Float(v) => Some(v.iter().map(|x| *x as i64).collect()),
        _ => None,
    }
}

/// Run all configured temporal checks against `df[time_col]`.
pub fn detect_temporal_issues(
    df: &DataFrame,
    time_col: &str,
    cfg: &TimeColumnConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    let Some(times) = extract_time_column(df, time_col) else {
        out.push(ValidationFinding::new(
            "E9054",
            FindingSeverity::Error,
            format!(
                "time column `{}` not found or not a temporal type (Int / Float / DateTime)",
                time_col
            ),
            Some(time_col.into()),
            None,
            vec![],
            n_rows,
            vec![],
            vec!["verify the column name and that it's a temporal type".into()],
        ));
        return out;
    };

    // 1. Sortedness check.
    let mut bad_pair: Option<(usize, i64, i64)> = None;
    let mut n_unsorted = 0u64;
    for i in 1..times.len() {
        if times[i] < times[i - 1] {
            n_unsorted += 1;
            if bad_pair.is_none() {
                bad_pair = Some((i, times[i - 1], times[i]));
            }
        }
    }
    if n_unsorted > 0 {
        let (idx, prev, curr) = bad_pair.unwrap();
        out.push(ValidationFinding::new(
            "E9050",
            FindingSeverity::Warning,
            format!(
                "time column `{}` is not sorted: {} non-decreasing pair(s) found",
                time_col, n_unsorted
            ),
            Some(time_col.into()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_unsorted_pairs".into(),
                    value: n_unsorted,
                },
                FindingEvidence::Sample {
                    label: "first_violation".into(),
                    value: format!("row {} ({} -> {})", idx, prev, curr),
                },
            ],
            n_rows,
            vec![
                "time-series checks assume non-decreasing order".into(),
                "if this is intentional (e.g. multi-source merge), declare a key and sort".into(),
            ],
            vec![
                "df.sort_by(time_col) before validation".into(),
            ],
        ));
    }

    // 2. Future-leakage cutoff.
    if let Some(cutoff) = cfg.max_timestamp {
        let mut count = 0u64;
        let mut sample = String::new();
        let mut samples_added = 0;
        for (i, &t) in times.iter().enumerate() {
            if t > cutoff {
                count += 1;
                if samples_added < 3 {
                    if samples_added > 0 {
                        sample.push(',');
                    }
                    sample.push_str(&format!("row{}={}", i, t));
                    samples_added += 1;
                }
            }
        }
        if count > 0 {
            out.push(ValidationFinding::new(
                "E9052",
                FindingSeverity::Error,
                format!(
                    "{} rows have timestamp > cutoff {} in `{}` (future-leakage candidate)",
                    count, cutoff, time_col
                ),
                Some(time_col.into()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_after_cutoff".into(),
                        value: count,
                    },
                    FindingEvidence::Metric {
                        label: "cutoff_timestamp".into(),
                        value: cutoff as f64,
                    },
                    FindingEvidence::Sample {
                        label: "sample_violations".into(),
                        value: sample,
                    },
                ],
                n_rows,
                vec![
                    "future-leakage typically means features were computed at score time, not event time".into(),
                ],
                vec![
                    "verify feature pipeline computes features as-of the event timestamp".into(),
                    "filter rows with timestamp > cutoff before training".into(),
                ],
            ));
        }
    }

    // 3. Gap detection.
    if let Some(gap_thresh) = cfg.gap_threshold {
        let mut n_gaps = 0u64;
        let mut max_gap: i64 = 0;
        let mut max_gap_idx: usize = 0;
        for i in 1..times.len() {
            let g = times[i].saturating_sub(times[i - 1]);
            if g > gap_thresh {
                n_gaps += 1;
                if g > max_gap {
                    max_gap = g;
                    max_gap_idx = i;
                }
            }
        }
        if n_gaps > 0 {
            out.push(ValidationFinding::new(
                "E9053",
                FindingSeverity::Notice,
                format!(
                    "{} gap(s) larger than {} units in `{}`; largest = {} at row {}",
                    n_gaps, gap_thresh, time_col, max_gap, max_gap_idx
                ),
                Some(time_col.into()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "n_gaps".into(),
                        value: n_gaps,
                    },
                    FindingEvidence::Metric {
                        label: "max_gap".into(),
                        value: max_gap as f64,
                    },
                    FindingEvidence::Metric {
                        label: "gap_threshold".into(),
                        value: gap_thresh as f64,
                    },
                ],
                n_rows,
                vec![
                    "gaps may indicate ingestion outages or intentional batch boundaries".into(),
                ],
                vec![
                    "verify whether the gap corresponds to a known event".into(),
                    "if expected, raise gap_threshold accordingly".into(),
                ],
            ));
        }
    }

    out
}

/// Detect temporal overlap between a train and test dataframe on the
/// same time column. Emits E9051 if any test row has timestamp <= the
/// maximum train timestamp (i.e. test events occur during or before
/// the training window).
pub fn detect_train_test_temporal_overlap(
    train: &DataFrame,
    test: &DataFrame,
    time_col: &str,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let Some(t_train) = extract_time_column(train, time_col) else {
        return out;
    };
    let Some(t_test) = extract_time_column(test, time_col) else {
        return out;
    };
    if t_train.is_empty() || t_test.is_empty() {
        return out;
    }
    let max_train = *t_train.iter().max().unwrap();
    let min_test = *t_test.iter().min().unwrap();
    if min_test <= max_train {
        let n_overlap = t_test.iter().filter(|t| **t <= max_train).count() as u64;
        out.push(ValidationFinding::new(
            "E9051",
            FindingSeverity::Error,
            format!(
                "{} test rows have `{}` <= max train timestamp ({}); test set overlaps train window",
                n_overlap, time_col, max_train
            ),
            Some(time_col.into()),
            None,
            vec![
                FindingEvidence::Count {
                    label: "n_test_in_train_window".into(),
                    value: n_overlap,
                },
                FindingEvidence::Metric {
                    label: "max_train_timestamp".into(),
                    value: max_train as f64,
                },
                FindingEvidence::Metric {
                    label: "min_test_timestamp".into(),
                    value: min_test as f64,
                },
            ],
            (t_train.len() + t_test.len()) as u64,
            vec![
                "for time-series ML, test timestamps must all be strictly later than train timestamps".into(),
            ],
            vec![
                "use a temporal split: test = rows where time > train_cutoff".into(),
                "if reusing the same rows in both is intentional, this is not an ML train/test split".into(),
            ],
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn df_with_time(values: Vec<i64>) -> DataFrame {
        DataFrame::from_columns(vec![("ts".into(), Column::Int(values))]).unwrap()
    }

    #[test]
    fn sorted_time_column_yields_no_finding() {
        let df = df_with_time(vec![1, 2, 3, 4, 5]);
        let r = detect_temporal_issues(&df, "ts", &TimeColumnConfig::default());
        assert!(r.iter().all(|f| f.code != "E9050"));
    }

    #[test]
    fn unsorted_time_column_emits_e9050() {
        let df = df_with_time(vec![1, 3, 2, 5, 4]);
        let r = detect_temporal_issues(&df, "ts", &TimeColumnConfig::default());
        let f = r.iter().find(|f| f.code == "E9050").expect("E9050 expected");
        let n = f
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Count { label, value } if label == "n_unsorted_pairs" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn missing_time_column_emits_e9054() {
        let df = df_with_time(vec![1, 2, 3]);
        let r = detect_temporal_issues(&df, "does_not_exist", &TimeColumnConfig::default());
        assert!(r.iter().any(|f| f.code == "E9054"));
    }

    #[test]
    fn future_leakage_cutoff_fires_e9052() {
        let df = df_with_time(vec![10, 20, 30, 40, 50]);
        let mut cfg = TimeColumnConfig::default();
        cfg.max_timestamp = Some(35);
        let r = detect_temporal_issues(&df, "ts", &cfg);
        let f = r.iter().find(|f| f.code == "E9052").expect("E9052 expected");
        let n = f
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Count { label, value } if label == "n_after_cutoff" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 2); // rows with timestamp 40 and 50
    }

    #[test]
    fn gap_detection_fires_e9053() {
        // Successive diffs: 10, 10, 10, 1000, 10 — one gap > 100
        let df = df_with_time(vec![0, 10, 20, 30, 1030, 1040]);
        let mut cfg = TimeColumnConfig::default();
        cfg.gap_threshold = Some(100);
        let r = detect_temporal_issues(&df, "ts", &cfg);
        let f = r.iter().find(|f| f.code == "E9053").expect("E9053 expected");
        let g = f
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Metric { label, value } if label == "max_gap" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(g, 1000.0);
    }

    #[test]
    fn train_test_overlap_fires_e9051() {
        // Train timestamps end at 100; test starts at 50 (overlap!)
        let train = df_with_time(vec![10, 50, 100]);
        let test = df_with_time(vec![50, 150, 200]);
        let r = detect_train_test_temporal_overlap(&train, &test, "ts");
        let f = r.iter().find(|f| f.code == "E9051").expect("E9051 expected");
        let n = f
            .evidence
            .iter()
            .find_map(|e| match e {
                FindingEvidence::Count { label, value } if label == "n_test_in_train_window" => Some(*value),
                _ => None,
            })
            .unwrap();
        assert_eq!(n, 1); // only the 50 in test is <= max_train (100)
    }

    #[test]
    fn temporal_disjoint_train_test_yields_no_finding() {
        let train = df_with_time(vec![10, 50, 100]);
        let test = df_with_time(vec![150, 200, 250]);
        let r = detect_train_test_temporal_overlap(&train, &test, "ts");
        assert!(r.is_empty());
    }

    #[test]
    fn temporal_checks_are_deterministic() {
        let df = df_with_time(vec![1, 3, 2, 1000, 999]);
        let mut cfg = TimeColumnConfig::default();
        cfg.max_timestamp = Some(500);
        cfg.gap_threshold = Some(100);
        let a = detect_temporal_issues(&df, "ts", &cfg);
        let b = detect_temporal_issues(&df, "ts", &cfg);
        assert_eq!(a, b);
    }
}
