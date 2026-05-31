//! Distribution-shape diagnostics (v0.6.3).
//!
//! Computes skewness and excess kurtosis of numeric columns (Kahan-summed
//! central moments), surfaces the top-K modes, and emits E9024 when
//! shape is outside a configurable normal range. Top-K modes are always
//! attached as evidence — even for shape-clean columns they're useful
//! triage information when a downstream report flags an issue.
//!
//! ## Codes
//!
//! | Code  | Severity | What it flags |
//! |-------|----------|---------------|
//! | E9024 | Notice   | Distribution shape outside normal range: `|skewness| > skew_threshold` (default 2.0) OR `|excess_kurtosis| > kurt_threshold` (default 7.0) |
//!
//! ## Conventions
//!
//! * Skewness `g1 = m3 / m2^(3/2)` — population (biased) estimator. 0 for
//!   a symmetric distribution.
//! * Excess kurtosis `g2 = m4 / m2^2 - 3` — population (biased) estimator.
//!   0 for a Gaussian; 3 means 3 "extra" standard-deviations of tail mass.
//! * Central moments use Kahan summation.
//! * NaN excluded throughout (matching the rest of Locke's numeric path).

use std::collections::BTreeMap;

use cjc_data::{Column, DataFrame};

use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};

// ─── Config ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ShapeConfig {
    pub skew_threshold: f64,
    pub kurt_threshold: f64,
    /// Skip columns with fewer valid (non-NaN) values than this.
    pub min_n_valid: u64,
    /// Number of top modes to attach as evidence.
    pub top_k_modes: usize,
}

impl Default for ShapeConfig {
    fn default() -> Self {
        Self {
            skew_threshold: 2.0,
            kurt_threshold: 7.0,
            min_n_valid: 20,
            top_k_modes: 3,
        }
    }
}

// ─── Moment computation ───────────────────────────────────────────────────

/// Population-moment shape statistics (skewness, excess kurtosis) over a
/// non-empty `&[f64]` slice with NaN excluded. Returns `None` if
/// variance is zero (constant column — shape is undefined).
pub fn skew_and_kurtosis(values: &[f64]) -> Option<(f64, f64)> {
    let mut mean_acc = cjc_repro::KahanAccumulatorF64::new();
    let mut n: u64 = 0;
    for &v in values {
        if v.is_nan() {
            continue;
        }
        mean_acc.add(v);
        n += 1;
    }
    if n < 2 {
        return None;
    }
    let n_f = n as f64;
    let mean = mean_acc.finalize() / n_f;
    let mut m2_acc = cjc_repro::KahanAccumulatorF64::new();
    let mut m3_acc = cjc_repro::KahanAccumulatorF64::new();
    let mut m4_acc = cjc_repro::KahanAccumulatorF64::new();
    for &v in values {
        if v.is_nan() {
            continue;
        }
        let d = v - mean;
        let d2 = d * d;
        m2_acc.add(d2);
        m3_acc.add(d2 * d);
        m4_acc.add(d2 * d2);
    }
    let m2 = m2_acc.finalize() / n_f;
    let m3 = m3_acc.finalize() / n_f;
    let m4 = m4_acc.finalize() / n_f;
    if m2 <= 0.0 || !m2.is_finite() || !m3.is_finite() || !m4.is_finite() {
        return None;
    }
    let skew = m3 / m2.powf(1.5);
    let excess_kurt = m4 / (m2 * m2) - 3.0;
    // Final guard: m2.powf(1.5) or m2*m2 may underflow / overflow even
    // when m2, m3, m4 are all finite (e.g. m2 = 1e-300, m3 = 1e-200
    // → ratio = inf). Return None so the detector skips.
    if !skew.is_finite() || !excess_kurt.is_finite() {
        return None;
    }
    Some((skew, excess_kurt))
}

/// Top-K modes for numeric and string columns. For floats, NaN excluded;
/// ties broken by smaller-value-first for floats and BTreeMap-sorted-key
/// order for strings (i.e. lexicographic). Returns a Vec sorted by
/// frequency descending, then by key ascending.
pub fn top_k_modes(col: &Column, k: usize) -> Vec<(String, u64)> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    match col {
        Column::Float(v) => {
            for x in v {
                if x.is_nan() {
                    continue;
                }
                *counts.entry(format!("{}", x)).or_insert(0) += 1;
            }
        }
        Column::Int(v) => {
            for x in v {
                *counts.entry(x.to_string()).or_insert(0) += 1;
            }
        }
        Column::Str(v) => {
            for s in v {
                *counts.entry(s.clone()).or_insert(0) += 1;
            }
        }
        Column::Bool(v) => {
            for b in v {
                *counts.entry(b.to_string()).or_insert(0) += 1;
            }
        }
        Column::Categorical { levels, codes } => {
            for &c in codes {
                if let Some(lbl) = levels.get(c as usize) {
                    *counts.entry(lbl.clone()).or_insert(0) += 1;
                }
            }
        }
        _ => return vec![],
    }
    let mut pairs: Vec<(String, u64)> = counts.into_iter().collect();
    // Sort by count desc, then key asc for determinism.
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(k);
    pairs
}

// ─── Detector ─────────────────────────────────────────────────────────────

fn extract_floats(col: &Column) -> Option<Vec<f64>> {
    match col {
        Column::Float(v) => Some(v.clone()),
        Column::Int(v) => Some(v.iter().map(|x| *x as f64).collect()),
        _ => None,
    }
}

/// Fire E9024 (Notice) on numeric columns whose shape sits outside the
/// `skew_threshold` / `kurt_threshold` envelope. Constant or near-constant
/// columns (zero variance) are skipped — shape is undefined.
pub fn detect_distribution_shape(
    df: &DataFrame,
    cfg: &ShapeConfig,
) -> Vec<ValidationFinding> {
    let mut out = Vec::new();
    let n_rows = df.nrows() as u64;
    for (name, col) in &df.columns {
        let Some(values) = extract_floats(col) else { continue };
        let n_valid = values.iter().filter(|x| !x.is_nan()).count() as u64;
        if n_valid < cfg.min_n_valid {
            continue;
        }
        let Some((skew, ex_kurt)) = skew_and_kurtosis(&values) else {
            continue;
        };
        let skew_extreme = skew.abs() > cfg.skew_threshold;
        let kurt_extreme = ex_kurt.abs() > cfg.kurt_threshold;
        if !skew_extreme && !kurt_extreme {
            continue;
        }
        let modes = top_k_modes(col, cfg.top_k_modes);
        let modes_str = modes
            .iter()
            .map(|(k, c)| format!("{:?}:{}", k, c))
            .collect::<Vec<_>>()
            .join(", ");
        let reason = match (skew_extreme, kurt_extreme) {
            (true, true) => "skewness and kurtosis both outside normal range",
            (true, false) => "skewness outside normal range",
            (false, true) => "kurtosis outside normal range",
            (false, false) => unreachable!(),
        };
        out.push(ValidationFinding::new(
            "E9024",
            FindingSeverity::Notice,
            format!(
                "column `{}` distribution shape: {} (skew = {:.3}, excess_kurt = {:.3})",
                name, reason, skew, ex_kurt
            ),
            Some(name.clone()),
            None,
            vec![
                FindingEvidence::Metric {
                    label: "skewness".into(),
                    value: skew,
                },
                FindingEvidence::Metric {
                    label: "excess_kurtosis".into(),
                    value: ex_kurt,
                },
                FindingEvidence::Count {
                    label: "n_valid".into(),
                    value: n_valid,
                },
                FindingEvidence::Sample {
                    label: "top_modes".into(),
                    value: modes_str,
                },
            ],
            n_rows,
            vec![
                "skewness > 0 indicates right tail; < 0 indicates left tail".into(),
                "positive excess kurtosis means heavier tails than Gaussian".into(),
                "thresholds are heuristic (default |skew|>2, |excess_kurt|>7); tune via ShapeConfig".into(),
            ],
            vec![
                "consider log / Box-Cox transformation if downstream assumes normality".into(),
                "review the top modes — concentration around a few values is often a real signal (sentinels, defaults)".into(),
            ],
        ));
    }
    out
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn mk_float(name: &str, v: Vec<f64>) -> DataFrame {
        DataFrame::from_columns(vec![(name.into(), Column::Float(v))]).unwrap()
    }

    #[test]
    fn skew_and_kurt_zero_for_symmetric_uniform() {
        // 100 evenly-spaced points around 0 → near-symmetric, low kurtosis.
        let v: Vec<f64> = (-50..50).map(|i| i as f64).collect();
        let (s, k) = skew_and_kurtosis(&v).unwrap();
        assert!(s.abs() < 0.1, "skew = {}", s);
        // Uniform distribution has excess kurtosis -6/5 = -1.2.
        assert!((k + 1.2).abs() < 0.2, "ex_kurt = {}", k);
    }

    #[test]
    fn skew_positive_for_right_tail() {
        // Most values near 0, a few large positives → right-skewed.
        let mut v: Vec<f64> = vec![0.0; 100];
        v.extend(vec![100.0; 5]);
        let (s, _) = skew_and_kurtosis(&v).unwrap();
        assert!(s > 1.0, "expected right skew, got {}", s);
    }

    #[test]
    fn skew_negative_for_left_tail() {
        let mut v: Vec<f64> = vec![0.0; 100];
        v.extend(vec![-100.0; 5]);
        let (s, _) = skew_and_kurtosis(&v).unwrap();
        assert!(s < -1.0, "expected left skew, got {}", s);
    }

    #[test]
    fn high_kurtosis_for_heavy_tail() {
        // Mostly 0s, a few extreme outliers → very heavy tails.
        let mut v: Vec<f64> = vec![0.0; 100];
        v.extend(vec![1000.0, -1000.0]);
        let (_, k) = skew_and_kurtosis(&v).unwrap();
        assert!(k > 10.0, "expected heavy-tail kurtosis, got {}", k);
    }

    #[test]
    fn constant_column_yields_none() {
        let v: Vec<f64> = vec![5.0; 100];
        assert!(skew_and_kurtosis(&v).is_none());
    }

    #[test]
    fn nan_excluded_from_moments() {
        let v: Vec<f64> = vec![1.0, 2.0, 3.0, f64::NAN, 4.0, 5.0];
        let (_, _) = skew_and_kurtosis(&v).unwrap();
    }

    #[test]
    fn e9024_fires_on_skewed_column() {
        let mut v: Vec<f64> = vec![0.0; 100];
        v.extend(vec![100.0; 10]);
        let df = mk_float("x", v);
        let f = detect_distribution_shape(&df, &ShapeConfig::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "E9024");
        assert_eq!(f[0].severity, FindingSeverity::Notice);
    }

    #[test]
    fn e9024_quiet_on_normal_shape() {
        let v: Vec<f64> = (-50..50).map(|i| i as f64).collect();
        let df = mk_float("x", v);
        let f = detect_distribution_shape(&df, &ShapeConfig::default());
        assert!(f.is_empty(), "expected quiet, got {:?}", f);
    }

    #[test]
    fn top_k_modes_returns_counts() {
        let df = mk_float("x", vec![1.0, 1.0, 2.0, 2.0, 2.0, 3.0]);
        let col = df.get_column("x").unwrap();
        let modes = top_k_modes(col, 2);
        assert_eq!(modes[0].0, "2");
        assert_eq!(modes[0].1, 3);
        assert_eq!(modes[1].0, "1");
        assert_eq!(modes[1].1, 2);
    }

    #[test]
    fn shape_is_deterministic() {
        let mut v: Vec<f64> = vec![0.0; 100];
        v.extend(vec![100.0; 10]);
        let df = mk_float("x", v);
        let cfg = ShapeConfig::default();
        let a = detect_distribution_shape(&df, &cfg);
        let b = detect_distribution_shape(&df, &cfg);
        assert_eq!(a, b);
    }
}
