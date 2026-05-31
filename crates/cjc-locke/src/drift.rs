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
    /// v0.6 batch 2: fire E9018 (cardinality explosion) when
    /// `test_cardinality >= train_cardinality * cardinality_explosion_ratio`.
    /// Default 2.0 — test has at least twice as many distinct levels as train.
    pub cardinality_explosion_ratio: f64,
    /// v0.6 batch 2: fire E9019 (entropy shift) when the absolute Shannon
    /// entropy of category frequencies shifts by at least this many nats
    /// (natural log). Default 0.20 ≈ 0.29 bits of distributional
    /// concentration change.
    pub entropy_shift_warn: f64,
    /// Entropy shift skipped on columns with fewer than this many distinct
    /// categories in either side (too few levels to compute meaningful H).
    pub entropy_min_distinct: u64,
    /// v0.7+ B5.3 fix: skip mean-drift emission when both `|train_mean|`
    /// and `|test_mean|` are below this absolute threshold. Pre-fix the
    /// `relative_shift` denominator was floored at `1e-12`, so a column
    /// with mean ≈ 1e-15 and a tiny `1e-12` jitter produced a "1.0"
    /// relative shift that fired E9030. Default `1e-9` covers ordinary
    /// IEEE-754 noise on near-zero columns without blocking legitimate
    /// drift on small-but-meaningful values. Set to `0.0` to restore
    /// pre-fix behaviour (no skip).
    pub mean_shift_near_zero_threshold: f64,
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
            cardinality_explosion_ratio: 2.0,
            entropy_shift_warn: 0.20,
            entropy_min_distinct: 3,
            mean_shift_near_zero_threshold: 1e-9,
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

/// Magnitude of the shift from `a` to `b` relative to `|a|`. The
/// denominator is floored at `1e-12` to avoid division by literal zero;
/// callers that need protection against larger near-zero amplification
/// should additionally guard via `DriftConfig::mean_shift_near_zero_threshold`
/// (see [`numeric_drift_findings`]). Result is clamped at `1e6` to keep
/// formatted finding messages readable.
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
    // v0.7+ B5.3: when both means are below the near-zero threshold,
    // skip the mean-drift check entirely. The relative-shift formula's
    // `1e-12` denominator floor amplifies sub-femto-scale jitter into
    // a spurious "100% shift" that fires E9030 on noise. Cases above
    // the threshold are evaluated identically to the pre-fix code.
    let near_zero = cfg.mean_shift_near_zero_threshold;
    if near_zero > 0.0 && t_mean.abs() < near_zero && s_mean.abs() < near_zero {
        return out;
    }
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

/// 1-Wasserstein distance between two 1-D empirical distributions.
///
/// W₁(P, Q) = ∫ |F_P(x) − F_Q(x)| dx — the L¹ area between the two
/// empirical CDFs. Computed exactly in O((n+m) log(n+m)) by sorting both
/// samples and integrating along the union of support. NaN values are
/// dropped (matching `ks_d_statistic` semantics).
///
/// Returns `None` when either sample is empty. Returns 0 when the two
/// samples are bytewise-equal in sorted form. Unit: same units as the
/// input values (i.e. not unit-free like KS D).
///
/// This is the auxiliary metric attached to E9039 evidence in v0.6 —
/// surfaced because two distributions can be KS-close and Wasserstein-far
/// (mass moved a small distance) or vice versa (mass moved a large
/// distance but ECDFs cross).
pub fn wasserstein_1(a: &[f64], b: &[f64]) -> Option<f64> {
    let aa = crate::stats::sort_filter_nan(a);
    let bb = crate::stats::sort_filter_nan(b);
    wasserstein_1_sorted(&aa, &bb)
}

/// W₁ computed from pre-sorted, NaN-free inputs. Same contract as
/// [`wasserstein_1`] but skips the filter+sort step — paired with
/// [`crate::stats::ks_d_statistic_sorted`] this lets callers share
/// one sort across both metrics.
pub fn wasserstein_1_sorted(aa: &[f64], bb: &[f64]) -> Option<f64> {
    if aa.is_empty() || bb.is_empty() {
        return None;
    }
    let n = aa.len() as f64;
    let m = bb.len() as f64;

    // Two-pointer integration over union of support. At each step, track
    // (i, j) = number of a-points and b-points already <= current x.
    // ECDF_a = i/n, ECDF_b = j/m. Move to the next breakpoint and add
    // |ECDF_a - ECDF_b| * (x_next - x_current) to the accumulator.
    let mut acc = cjc_repro::KahanAccumulatorF64::new();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut prev_x = aa[0].min(bb[0]);
    while i < aa.len() || j < bb.len() {
        let next_x = match (aa.get(i), bb.get(j)) {
            (Some(&xa), Some(&xb)) => xa.min(xb),
            (Some(&xa), None) => xa,
            (None, Some(&xb)) => xb,
            (None, None) => break,
        };
        let dx = next_x - prev_x;
        if dx > 0.0 {
            let ecdf_a = i as f64 / n;
            let ecdf_b = j as f64 / m;
            acc.add((ecdf_a - ecdf_b).abs() * dx);
        }
        // Advance both pointers past `next_x` (handle ties).
        while i < aa.len() && aa[i] <= next_x {
            i += 1;
        }
        while j < bb.len() && bb[j] <= next_x {
            j += 1;
        }
        prev_x = next_x;
    }
    Some(acc.finalize())
}

/// Exact Kolmogorov–Smirnov D as the default numeric drift signal (v0.2).
///
/// Emits **E9039** when the configured threshold is crossed. Replaces the
/// equal-width-binned PSI proxy (`numeric_psi_finding`) which is retained
/// as a private helper for future opt-in.
///
/// As of v0.6, the evidence list also carries the 1-Wasserstein distance
/// between the two samples, computed via [`wasserstein_1`]. Reviewers can
/// use it alongside KS D — they're complementary signals.
fn numeric_ks_finding(
    column: &str,
    train_values: &[f64],
    test_values: &[f64],
    cfg: &DriftConfig,
) -> Option<ValidationFinding> {
    // v0.7+ B4.4 perf-fix: previously this function sorted both inputs
    // twice — once inside `ks_d_statistic`, once inside `wasserstein_1`.
    // Filter and sort once here, then pass the sorted slices to the
    // `_sorted` variants of both metrics.
    let train_sorted = crate::stats::sort_filter_nan(train_values);
    let test_sorted = crate::stats::sort_filter_nan(test_values);
    let d = crate::stats::ks_d_statistic_sorted(&train_sorted, &test_sorted)?;
    let sev = if d >= cfg.ks_d_error {
        FindingSeverity::Error
    } else if d >= cfg.ks_d_warn {
        FindingSeverity::Warning
    } else {
        return None;
    };
    let n_train_valid = train_sorted.len() as u64;
    let n_test_valid = test_sorted.len() as u64;
    let w1 = wasserstein_1_sorted(&train_sorted, &test_sorted).unwrap_or(0.0);
    Some(ValidationFinding::new(
        "E9039",
        sev,
        format!(
            "numeric distribution shift in `{}` (KS D = {:.4}, W1 = {:.4})",
            column, d, w1
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Metric {
                label: "ks_d".into(),
                value: d,
            },
            FindingEvidence::Metric {
                label: "wasserstein_1".into(),
                value: w1,
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
            "Wasserstein₁ has the same units as the values (not unit-free like KS)".into(),
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

/// E9018 — cardinality jumped between train and test. Fires when the
/// test side has at least `cfg.cardinality_explosion_ratio` × as many
/// distinct levels as train. Important because category TVD can be
/// arbitrarily low when each new category has tiny mass, so explosion
/// is a *separate* signal: "many new things appeared, each rare."
fn category_cardinality_explosion_finding(
    column: &str,
    train: &[String],
    test: &[String],
    cfg: &DriftConfig,
) -> Option<ValidationFinding> {
    let mut train_set: BTreeMap<&str, ()> = BTreeMap::new();
    let mut test_set: BTreeMap<&str, ()> = BTreeMap::new();
    for s in train {
        train_set.insert(s.as_str(), ());
    }
    for s in test {
        test_set.insert(s.as_str(), ());
    }
    let n_train = train_set.len() as u64;
    let n_test = test_set.len() as u64;
    if n_train == 0 || n_test == 0 {
        return None;
    }
    let ratio = n_test as f64 / n_train as f64;
    if ratio < cfg.cardinality_explosion_ratio {
        return None;
    }
    // Count newly-appeared categories explicitly.
    let new_in_test: u64 = test_set.keys().filter(|k| !train_set.contains_key(*k)).count() as u64;
    let sev = if ratio >= cfg.cardinality_explosion_ratio * 2.0 {
        FindingSeverity::Warning
    } else {
        FindingSeverity::Notice
    };
    Some(ValidationFinding::new(
        "E9018",
        sev,
        format!(
            "category cardinality of `{}` jumped: train={}, test={} (×{:.2}); {} new categories in test",
            column, n_train, n_test, ratio, new_in_test
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Count {
                label: "n_distinct_train".into(),
                value: n_train,
            },
            FindingEvidence::Count {
                label: "n_distinct_test".into(),
                value: n_test,
            },
            FindingEvidence::Metric {
                label: "cardinality_ratio".into(),
                value: ratio,
            },
            FindingEvidence::Count {
                label: "n_new_categories".into(),
                value: new_in_test,
            },
        ],
        (train.len() + test.len()) as u64,
        vec![
            "cardinality explosion can be a real signal (new product SKUs) or an ingestion bug".into(),
            "complement to E9034 TVD — TVD can be near 0 if each new category has tiny mass".into(),
        ],
        vec![
            "review the new categories; if they're unexpected, freeze your category mapping at train time".into(),
            "consider mapping unknown categories to `__other__` instead of one-hot expansion".into(),
        ],
    ))
}

/// E9019 — Shannon entropy of category frequencies shifted between
/// train and test. Captures *distributional concentration* changes that
/// neither TVD nor cardinality alone reveals (e.g. one tier going from
/// uniform-mass to dominating 90% of rows).
///
/// H computed in nats via `-Σ p ln p` with Kahan summation; convention
/// `0 ln 0 = 0` enforced explicitly.
fn category_entropy_shift_finding(
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
    if (train_freq.len() as u64) < cfg.entropy_min_distinct
        || (test_freq.len() as u64) < cfg.entropy_min_distinct
    {
        return None;
    }
    let entropy_nats = |freq: &BTreeMap<&str, u64>| -> f64 {
        let total: f64 = freq.values().sum::<u64>() as f64;
        if total == 0.0 {
            return 0.0;
        }
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for &c in freq.values() {
            if c == 0 {
                continue;
            }
            let p = c as f64 / total;
            acc.add(-p * p.ln());
        }
        acc.finalize()
    };
    let h_train = entropy_nats(&train_freq);
    let h_test = entropy_nats(&test_freq);
    let delta = (h_train - h_test).abs();
    if delta < cfg.entropy_shift_warn {
        return None;
    }
    Some(ValidationFinding::new(
        "E9019",
        FindingSeverity::Notice,
        format!(
            "category entropy of `{}` shifted by {:.3} nats: train_H={:.3}, test_H={:.3}",
            column, delta, h_train, h_test
        ),
        Some(column.into()),
        None,
        vec![
            FindingEvidence::Metric {
                label: "train_entropy_nats".into(),
                value: h_train,
            },
            FindingEvidence::Metric {
                label: "test_entropy_nats".into(),
                value: h_test,
            },
            FindingEvidence::Metric {
                label: "entropy_delta_nats".into(),
                value: delta,
            },
        ],
        (train.len() + test.len()) as u64,
        vec![
            "Shannon entropy in nats; convert to bits by dividing by ln(2)".into(),
            "captures concentration shifts (uniform → peaked) that TVD/cardinality miss".into(),
        ],
        vec![
            "inspect the top-K modes of each side to localise the concentration change".into(),
            "if the column drives a downstream feature, retrain or re-calibrate".into(),
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
        // v0.6 batch 2: cardinality explosion + entropy shift.
        if let Some(f) = category_cardinality_explosion_finding(col, &train_levels, &test_levels, cfg) {
            findings.push(f);
        }
        if let Some(f) = category_entropy_shift_finding(col, &train_levels, &test_levels, cfg) {
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

    // ── Wasserstein-1 ───────────────────────────────────────────────────

    #[test]
    fn wasserstein_zero_for_identical_samples() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = a.clone();
        let w = wasserstein_1(&a, &b).unwrap();
        assert!(w.abs() < 1e-12, "got {}", w);
    }

    #[test]
    fn wasserstein_translation_equals_shift() {
        // Translating every point by +c should give W_1 = c.
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b: Vec<f64> = a.iter().map(|x| x + 3.0).collect();
        let w = wasserstein_1(&a, &b).unwrap();
        assert!((w - 3.0).abs() < 1e-9, "got {}", w);
    }

    #[test]
    fn wasserstein_handles_nan() {
        let a = vec![1.0, f64::NAN, 2.0];
        let b = vec![1.0, 2.0];
        let w = wasserstein_1(&a, &b).unwrap();
        assert!(w.abs() < 1e-12);
    }

    #[test]
    fn wasserstein_none_on_empty() {
        assert!(wasserstein_1(&[], &[1.0]).is_none());
        assert!(wasserstein_1(&[1.0], &[]).is_none());
    }

    #[test]
    fn wasserstein_appears_as_evidence_in_e9039() {
        // Use a clear shift so KS fires and we can inspect evidence.
        let train: Vec<f64> = (0..200).map(|i| i as f64 * 0.01).collect();
        let test: Vec<f64> = (0..200).map(|i| i as f64 * 0.01 + 1.0).collect();
        let train_df = DataFrame::from_columns(vec![mk_floats("x", train)]).unwrap();
        let test_df = DataFrame::from_columns(vec![mk_floats("x", test)]).unwrap();
        let r = compare(&train_df, &test_df, &DriftConfig::default());
        let ks = r
            .findings
            .iter()
            .find(|f| f.code == "E9039")
            .expect("E9039 KS finding");
        let has_w1 = ks.evidence.iter().any(|e| matches!(
            e,
            FindingEvidence::Metric { label, .. } if label == "wasserstein_1"
        ));
        assert!(has_w1, "wasserstein_1 evidence missing from E9039: {:?}", ks.evidence);
    }
}
