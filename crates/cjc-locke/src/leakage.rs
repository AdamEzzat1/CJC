//! Target leakage detection (v0.5).
//!
//! The classic ML pipeline bug: a feature is *almost* perfectly
//! predictive of the target on the training set, because it was
//! computed *after* the event (e.g., "days since last login" measured
//! at score time leaks "the customer churned recently").
//!
//! Locke flags this via per-feature ROC AUC against a caller-supplied
//! target column. Codes:
//!
//! | Code  | Severity | When |
//! |-------|----------|------|
//! | E9060 | Error    | AUC ≥ 0.95 (almost certainly leakage) |
//! | E9061 | Warning  | 0.85 ≤ AUC < 0.95 (worth investigating) |
//!
//! Plus E9072 for ID-like cardinality (a hint that may compound with
//! the AUC signal): a column whose distinct/n_rows ratio is ≥ 0.95 is
//! probably a key column leaking as a feature.
//!
//! ## Limitations
//!
//! - Only binary targets are supported in v0.5. Multi-class would
//!   need one-vs-rest AUC; deferred.
//! - AUC is computed from the full dataset, not a held-out split.
//!   For a true "is this leakage" test, the user should run the same
//!   check on a separate validation set.

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

#[derive(Clone, Debug)]
pub struct LeakageConfig {
    pub auc_error_threshold: f64,
    pub auc_warn_threshold: f64,
    /// Minimum count of each class — below this AUC is unreliable.
    pub min_class_count: u64,
    /// Cardinality / n_rows above which a column is flagged as ID-like.
    pub id_like_cardinality_ratio: f64,
    /// Minimum row count before ID-like checks fire.
    pub id_like_min_rows: u64,
}

impl Default for LeakageConfig {
    fn default() -> Self {
        Self {
            auc_error_threshold: 0.95,
            auc_warn_threshold: 0.85,
            min_class_count: 10,
            id_like_cardinality_ratio: 0.95,
            id_like_min_rows: 50,
        }
    }
}

/// Extract a binary target column as `Vec<u8>` (0/1). Returns `None`
/// if the column doesn't look binary (Bool always works; Int with
/// exactly 2 distinct values also works).
fn extract_binary_target(df: &DataFrame, name: &str) -> Option<Vec<u8>> {
    let col = df.get_column(name)?;
    match col {
        Column::Bool(v) => Some(v.iter().map(|b| if *b { 1u8 } else { 0u8 }).collect()),
        Column::Int(v) => {
            // Accept any column whose values are exactly {0, 1} or a {a, b}
            // pair where the smaller is encoded 0 and the larger 1.
            let mut distinct: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for x in v {
                distinct.insert(*x);
                if distinct.len() > 2 {
                    return None;
                }
            }
            if distinct.len() != 2 {
                return None;
            }
            let mut iter = distinct.iter();
            let lo = *iter.next().unwrap();
            let hi = *iter.next().unwrap();
            Some(
                v.iter()
                    .map(|x| if *x == lo { 0u8 } else if *x == hi { 1u8 } else { 0 })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Extract feature values as f64 for AUC computation. Returns `None`
/// for non-numeric columns. NaN is excluded along with its target.
fn extract_feature(df: &DataFrame, name: &str) -> Option<Vec<f64>> {
    let col = df.get_column(name)?;
    match col {
        Column::Float(v) => Some(v.clone()),
        Column::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
        Column::Bool(v) => Some(v.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect()),
        _ => None,
    }
}

/// Compute ROC AUC of `feature` against a binary `target` (0/1).
/// Returns `None` if either class has < `min_class_count` samples or
/// if no valid pairs remain after NaN filtering.
///
/// Algorithm: pair (feature, target), drop NaN rows, sort by feature,
/// compute rank-sum AUC. O(n log n) time, O(n) memory, fully
/// deterministic via `f64::total_cmp`.
pub fn binary_target_auc(
    feature: &[f64],
    target: &[u8],
    min_class_count: u64,
) -> Option<f64> {
    if feature.len() != target.len() {
        return None;
    }
    let mut pairs: Vec<(f64, u8)> = feature
        .iter()
        .zip(target.iter())
        .filter_map(|(f, t)| {
            if f.is_nan() {
                None
            } else {
                Some((*f, *t))
            }
        })
        .collect();
    let n_pos = pairs.iter().filter(|(_, t)| *t == 1).count() as u64;
    let n_neg = pairs.iter().filter(|(_, t)| *t == 0).count() as u64;
    if n_pos < min_class_count || n_neg < min_class_count {
        return None;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Rank-sum AUC. We need average ranks for ties.
    let n = pairs.len();
    let mut sum_ranks_positive = 0.0f64;
    let mut i = 0usize;
    while i < n {
        // Find tie group [i, j)
        let mut j = i + 1;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Average rank for this tie group (1-indexed).
        let avg_rank = (i as f64 + 1.0 + j as f64) / 2.0;
        for k in i..j {
            if pairs[k].1 == 1 {
                sum_ranks_positive += avg_rank;
            }
        }
        i = j;
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;
    let auc = (sum_ranks_positive - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg_f);
    Some(auc)
}

/// For each feature column (excluding the target), compute AUC and
/// emit E9060 (Error, AUC ≥ 0.95) or E9061 (Warning, AUC ≥ 0.85).
pub fn detect_target_leakage(
    df: &DataFrame,
    target_col: &str,
    cfg: &LeakageConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let Some(target) = extract_binary_target(df, target_col) else {
        out.push(ValidationFinding::new(
            "E9062",
            FindingSeverity::Notice,
            format!(
                "target column `{}` is not a binary type (Bool or Int-with-exactly-2-distinct); target leakage check skipped",
                target_col
            ),
            Some(target_col.into()),
            None,
            vec![],
            df.nrows() as u64,
            vec![
                "v0.5 supports only binary targets for the per-feature AUC check".into(),
            ],
            vec![
                "encode the target as Bool or 0/1 Int".into(),
                "multi-class AUC support is on the v0.6 roadmap".into(),
            ],
        ));
        return out;
    };

    let n_rows = df.nrows() as u64;
    for (name, _col) in &df.columns {
        if name == target_col {
            continue;
        }
        let Some(feat) = extract_feature(df, name) else {
            continue;
        };
        let Some(auc) = binary_target_auc(&feat, &target, cfg.min_class_count) else {
            continue;
        };
        // Use max(AUC, 1-AUC) so we don't miss negatively-perfect features.
        let abs_auc = auc.max(1.0 - auc);
        let (severity, code) = if abs_auc >= cfg.auc_error_threshold {
            (FindingSeverity::Error, "E9060")
        } else if abs_auc >= cfg.auc_warn_threshold {
            (FindingSeverity::Warning, "E9061")
        } else {
            continue;
        };
        out.push(ValidationFinding::new(
            code,
            severity,
            format!(
                "feature `{}` has |AUC| = {:.4} against target `{}` — suspicious target leakage",
                name, abs_auc, target_col
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Metric {
                    label: "abs_auc".into(),
                    value: abs_auc,
                },
                FindingEvidence::Metric {
                    label: "raw_auc".into(),
                    value: auc,
                },
                FindingEvidence::Sample {
                    label: "target".into(),
                    value: target_col.into(),
                },
            ],
            n_rows,
            vec![
                "AUC near 0.5 means no signal; near 0 or 1 means the feature is nearly deterministic of the target".into(),
                "high AUC may be legitimate (a strong feature) OR leakage — Locke surfaces both".into(),
                "this AUC is computed on the full dataset; verify on a held-out split".into(),
            ],
            vec![
                "check whether this feature is computed BEFORE or AFTER the target event".into(),
                "if computed after, exclude from training".into(),
            ],
        ));
    }
    out
}

/// ID-like cardinality hint — flags columns whose `distinct / n_rows`
/// ratio crosses the configured threshold. Often catches join keys or
/// row IDs accidentally left in the feature matrix.
pub fn detect_id_like_columns(df: &DataFrame, cfg: &LeakageConfig) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    if n_rows < cfg.id_like_min_rows {
        return out;
    }
    for (name, col) in &df.columns {
        let distinct = crate::validation::distinct_count(col);
        let ratio = distinct as f64 / n_rows as f64;
        if ratio >= cfg.id_like_cardinality_ratio {
            out.push(ValidationFinding::new(
                "E9072",
                FindingSeverity::Notice,
                format!(
                    "column `{}` has cardinality {} / {} rows = {:.3}; may be an ID leaking as a feature",
                    name, distinct, n_rows, ratio
                ),
                Some(name.clone()),
                None,
                vec![
                    FindingEvidence::Count {
                        label: "distinct".into(),
                        value: distinct,
                    },
                    FindingEvidence::Ratio {
                        label: "distinct_per_row".into(),
                        value: ratio,
                    },
                ],
                n_rows,
                vec![
                    "ID-like columns memorise the training set and don't generalise".into(),
                    "high cardinality may also be legitimate (a free-text field) — Locke surfaces a hint, not a verdict".into(),
                ],
                vec![
                    "drop the column if it's a customer/row/transaction ID".into(),
                    "hash to a low-cardinality bucket if you must retain the value".into(),
                ],
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn df_with_perfect_leak() -> DataFrame {
        // Feature is literally the target — AUC will be exactly 1.0.
        let n = 100;
        let target: Vec<i64> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let feature: Vec<f64> = target.iter().map(|t| *t as f64).collect();
        DataFrame::from_columns(vec![
            ("y".into(), Column::Int(target)),
            ("leaky".into(), Column::Float(feature)),
        ])
        .unwrap()
    }

    fn df_with_random_feature() -> DataFrame {
        use cjc_repro::Rng;
        let n = 200;
        let mut rng = Rng::seeded(42);
        let target: Vec<i64> = (0..n)
            .map(|i| if i < n / 2 { 0 } else { 1 })
            .collect();
        let feature: Vec<f64> = (0..n).map(|_| (rng.next_u64() % 1000) as f64).collect();
        DataFrame::from_columns(vec![
            ("y".into(), Column::Int(target)),
            ("noise".into(), Column::Float(feature)),
        ])
        .unwrap()
    }

    #[test]
    fn perfect_leak_yields_e9060_error() {
        let df = df_with_perfect_leak();
        let r = detect_target_leakage(&df, "y", &LeakageConfig::default());
        let f = r.iter().find(|f| f.code == "E9060").expect("E9060 for perfect leak");
        assert_eq!(f.severity, FindingSeverity::Error);
    }

    #[test]
    fn random_feature_yields_no_leakage_finding() {
        let df = df_with_random_feature();
        let r = detect_target_leakage(&df, "y", &LeakageConfig::default());
        assert!(r.iter().all(|f| f.code != "E9060" && f.code != "E9061"));
    }

    #[test]
    fn non_binary_target_emits_e9062_notice() {
        let df = DataFrame::from_columns(vec![
            ("y".into(), Column::Int(vec![1, 2, 3, 4, 5, 1, 2, 3])),
            ("x".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
        ])
        .unwrap();
        let r = detect_target_leakage(&df, "y", &LeakageConfig::default());
        assert!(r.iter().any(|f| f.code == "E9062"));
    }

    #[test]
    fn id_like_column_emits_e9072() {
        // 100 rows, 100 distinct user_ids → ratio = 1.0
        let user_ids: Vec<i64> = (0..100).collect();
        let df = DataFrame::from_columns(vec![("user_id".into(), Column::Int(user_ids))]).unwrap();
        let r = detect_id_like_columns(&df, &LeakageConfig::default());
        assert!(r.iter().any(|f| f.code == "E9072"));
    }

    #[test]
    fn low_cardinality_column_does_not_fire_id_like() {
        let df = DataFrame::from_columns(vec![
            ("status".into(), Column::Int((0..100).map(|i| i % 3).collect())),
        ])
        .unwrap();
        let r = detect_id_like_columns(&df, &LeakageConfig::default());
        assert!(r.is_empty());
    }

    #[test]
    fn small_dataset_skips_id_like_check() {
        // Below the min_rows threshold.
        let df = DataFrame::from_columns(vec![
            ("id".into(), Column::Int((0..10).collect())),
        ])
        .unwrap();
        let r = detect_id_like_columns(&df, &LeakageConfig::default());
        assert!(r.is_empty());
    }

    #[test]
    fn auc_of_identical_data_is_one() {
        let feat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tgt: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let auc = binary_target_auc(&feat, &tgt, 5).unwrap();
        assert!((auc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn auc_of_reversed_is_zero() {
        let feat: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let tgt: Vec<u8> = vec![1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let auc = binary_target_auc(&feat, &tgt, 5).unwrap();
        assert!(auc < 1e-12);
    }

    #[test]
    fn detection_is_deterministic() {
        let df = df_with_perfect_leak();
        let cfg = LeakageConfig::default();
        let a = detect_target_leakage(&df, "y", &cfg);
        let b = detect_target_leakage(&df, "y", &cfg);
        assert_eq!(a, b);
    }
}
