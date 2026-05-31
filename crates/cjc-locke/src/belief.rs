//! Confidence-aware belief reports.
//!
//! A `BeliefScore` is **not** an opaque magic number. It's the average
//! of explicit per-dimension sub-scores, each in `[0, 1]`, each with its
//! own derivation rule. Every score carries a `breakdown` so a user can
//! see exactly why their dataset got, say, 0.65 confidence and which
//! dimension dragged the average down.
//!
//! ## Dimensions (v0)
//!
//! * `schema_score` — 1.0 if schema is fully typed and self-consistent;
//!   reduced by typed-vs-untyped column ratio and by schema-mismatch findings.
//! * `missingness_score` — `1.0 − overall_missingness_rate`, clamped.
//! * `drift_score` — `1.0` if no drift finding; reduced by drift magnitude.
//! * `leakage_score` — `1.0` if no leakage finding; otherwise stepwise.
//! * `lineage_score` — `1.0` if every column traces back to an Impression;
//!   reduced by dangling ideas.
//! * `sample_score` — saturating curve over `n_rows` (50% at n=30, 95% at n=500).
//! * `duplication_score` — `1.0 − duplicate_rate`.
//! * `constraint_score` — `1.0 − (n_constraint_violations / n_rows)`.
//!
//! A dimension whose evidence is absent is set to `1.0` (no signal ≠
//! bad signal) **and** added to the `assumptions` list, so an empty
//! report cannot artificially boost the score without acknowledgement.

use std::collections::BTreeMap;

use crate::report::{FindingSeverity, ValidationFinding};

#[derive(Clone, Debug, PartialEq)]
pub struct BeliefScore {
    pub overall: f64,
    pub schema_score: f64,
    pub missingness_score: f64,
    pub drift_score: f64,
    pub leakage_score: f64,
    pub lineage_score: f64,
    pub sample_score: f64,
    pub duplication_score: f64,
    pub constraint_score: f64,
}

/// User-tunable per-dimension weights for the belief overall score (v0.2).
///
/// Default is equal weights (every field = 1.0), which produces an
/// unweighted mean and is bit-identical to v0.1 behavior. Negative
/// weights are clamped to 0 at the point of use; if every weight is 0
/// the overall score is set to 0.0.
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefWeights {
    pub schema: f64,
    pub missingness: f64,
    pub drift: f64,
    pub leakage: f64,
    pub lineage: f64,
    pub sample: f64,
    pub duplication: f64,
    pub constraint: f64,
}

impl Default for BeliefWeights {
    fn default() -> Self {
        Self {
            schema: 1.0,
            missingness: 1.0,
            drift: 1.0,
            leakage: 1.0,
            lineage: 1.0,
            sample: 1.0,
            duplication: 1.0,
            constraint: 1.0,
        }
    }
}

impl BeliefWeights {
    /// Build from raw values. NaN or negative values are clamped to 0.
    pub fn from_values(
        schema: f64,
        missingness: f64,
        drift: f64,
        leakage: f64,
        lineage: f64,
        sample: f64,
        duplication: f64,
        constraint: f64,
    ) -> Self {
        let clean = |w: f64| {
            if !w.is_finite() || w < 0.0 {
                0.0
            } else {
                w
            }
        };
        Self {
            schema: clean(schema),
            missingness: clean(missingness),
            drift: clean(drift),
            leakage: clean(leakage),
            lineage: clean(lineage),
            sample: clean(sample),
            duplication: clean(duplication),
            constraint: clean(constraint),
        }
    }

    pub fn as_array(&self) -> [f64; 8] {
        [
            self.schema,
            self.missingness,
            self.drift,
            self.leakage,
            self.lineage,
            self.sample,
            self.duplication,
            self.constraint,
        ]
    }

    pub fn sum(&self) -> f64 {
        self.as_array().iter().sum()
    }
}

impl BeliefScore {
    /// Build a belief score from the named sub-dimensions. All inputs
    /// are clamped into `[0, 1]` and `overall` is their unweighted mean
    /// (weighting can be added in v0.2 if needed).
    pub fn from_dimensions(
        schema_score: f64,
        missingness_score: f64,
        drift_score: f64,
        leakage_score: f64,
        lineage_score: f64,
        sample_score: f64,
        duplication_score: f64,
        constraint_score: f64,
    ) -> Self {
        let clamp = |v: f64| {
            if !v.is_finite() {
                0.0
            } else {
                v.clamp(0.0, 1.0)
            }
        };
        let s = [
            clamp(schema_score),
            clamp(missingness_score),
            clamp(drift_score),
            clamp(leakage_score),
            clamp(lineage_score),
            clamp(sample_score),
            clamp(duplication_score),
            clamp(constraint_score),
        ];
        let overall = s.iter().sum::<f64>() / s.len() as f64;
        Self {
            overall,
            schema_score: s[0],
            missingness_score: s[1],
            drift_score: s[2],
            leakage_score: s[3],
            lineage_score: s[4],
            sample_score: s[5],
            duplication_score: s[6],
            constraint_score: s[7],
        }
    }

    /// Weighted variant of [`from_dimensions`]. Each sub-dimension contributes
    /// `weight_i * score_i` and the overall is `sum / sum_of_weights`. If every
    /// weight is zero or NaN, `overall` is `0.0`. Inputs are still clamped to
    /// `[0, 1]` as in the unweighted constructor.
    pub fn from_dimensions_weighted(
        schema_score: f64,
        missingness_score: f64,
        drift_score: f64,
        leakage_score: f64,
        lineage_score: f64,
        sample_score: f64,
        duplication_score: f64,
        constraint_score: f64,
        weights: &BeliefWeights,
    ) -> Self {
        let clamp = |v: f64| {
            if !v.is_finite() {
                0.0
            } else {
                v.clamp(0.0, 1.0)
            }
        };
        let s = [
            clamp(schema_score),
            clamp(missingness_score),
            clamp(drift_score),
            clamp(leakage_score),
            clamp(lineage_score),
            clamp(sample_score),
            clamp(duplication_score),
            clamp(constraint_score),
        ];
        let w = weights.as_array();
        let w_sum: f64 = w.iter().sum();
        let overall = if w_sum > 0.0 {
            let mut acc = cjc_repro::KahanAccumulatorF64::new();
            for i in 0..8 {
                acc.add(w[i] * s[i]);
            }
            acc.finalize() / w_sum
        } else {
            0.0
        };
        Self {
            overall,
            schema_score: s[0],
            missingness_score: s[1],
            drift_score: s[2],
            leakage_score: s[3],
            lineage_score: s[4],
            sample_score: s[5],
            duplication_score: s[6],
            constraint_score: s[7],
        }
    }

    /// Render the breakdown as a stable, human-readable string.
    pub fn explain(&self) -> String {
        format!(
            "overall={:.3}\n  schema      = {:.3}\n  missingness = {:.3}\n  drift       = {:.3}\n  leakage     = {:.3}\n  lineage     = {:.3}\n  sample      = {:.3}\n  duplication = {:.3}\n  constraint  = {:.3}\n",
            self.overall,
            self.schema_score,
            self.missingness_score,
            self.drift_score,
            self.leakage_score,
            self.lineage_score,
            self.sample_score,
            self.duplication_score,
            self.constraint_score,
        )
    }
}

/// User-tunable per-severity penalty model (v0.3).
///
/// Each finding in a relevant dimension subtracts `penalty(severity)` from
/// that dimension's score. `Default` is v0.2's baked-in opinion (Info=0.02,
/// Notice=0.02, Warning=0.10, Error=0.25).
///
/// Tunable ≠ calibrated. Users with calibration data (e.g. "in our last 100
/// production datasets, a Warning was a real defect 60% of the time") can
/// set their own penalties; users without it stay on the defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefPenalty {
    pub info: f64,
    pub notice: f64,
    pub warning: f64,
    pub error: f64,
}

impl Default for BeliefPenalty {
    fn default() -> Self {
        Self {
            info: 0.02,
            notice: 0.02,
            warning: 0.10,
            error: 0.25,
        }
    }
}

impl BeliefPenalty {
    /// Build a penalty model. NaN or negative values are clamped to 0.
    pub fn from_values(info: f64, notice: f64, warning: f64, error: f64) -> Self {
        let clean = |v: f64| {
            if !v.is_finite() || v < 0.0 {
                0.0
            } else {
                v
            }
        };
        Self {
            info: clean(info),
            notice: clean(notice),
            warning: clean(warning),
            error: clean(error),
        }
    }

    /// Penalty value for the given severity.
    pub fn for_severity(&self, s: crate::report::FindingSeverity) -> f64 {
        use crate::report::FindingSeverity::*;
        match s {
            Info => self.info,
            Notice => self.notice,
            Warning => self.warning,
            Error => self.error,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeliefReport {
    pub score: BeliefScore,
    pub assumptions: Vec<String>,
    pub evidence_summary: BTreeMap<String, String>,
    pub recommended_next_steps: Vec<String>,
}

impl BeliefReport {
    pub fn new(
        score: BeliefScore,
        assumptions: Vec<String>,
        evidence_summary: BTreeMap<String, String>,
        recommended_next_steps: Vec<String>,
    ) -> Self {
        Self {
            score,
            assumptions,
            evidence_summary,
            recommended_next_steps,
        }
    }
}

/// Saturating sample-size score: 0.5 at n=30, 0.95 at n=500, 1.0 asymptote.
pub fn sample_score_from_n(n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    // 1 - exp(-n / k) tuned so n=30 ≈ 0.5 (k ≈ 43.3).
    let k = 43.281; // = -30 / ln(0.5)
    1.0 - (-(n as f64) / k).exp()
}

/// Aggregate a flat list of validation findings into a coarse penalty
/// for the named dimension. Each `Warning` finding contributes `−0.10`
/// and each `Error` contributes `−0.25`. The result is clamped at zero.
///
/// The penalty model is intentionally simple — v0 favours explainability
/// over calibration. Future versions can use evidence-weighted penalties.
pub fn penalty_from_findings(findings: &[ValidationFinding], dim_filter: impl Fn(&str) -> bool) -> f64 {
    penalty_from_findings_with_model(findings, dim_filter, &BeliefPenalty::default())
}

/// v0.3: user-tunable variant. Same shape as `penalty_from_findings`
/// but takes an explicit `BeliefPenalty` model.
///
/// The return is **clamped to `[0, 1]`**: a sufficiently noisy report
/// (many high-severity findings) can produce an uncapped sum > 1.0,
/// but the caller's contract is `axis_score = 1.0 - penalty` which
/// only makes sense in `[0, 1]`. Every existing caller already routed
/// the result through `BeliefScore::from_dimensions` (which clamps
/// per-axis), so the clamp here is byte-identical at the caller's
/// boundary — it just makes the contract explicit at the source.
pub fn penalty_from_findings_with_model(
    findings: &[ValidationFinding],
    dim_filter: impl Fn(&str) -> bool,
    model: &BeliefPenalty,
) -> f64 {
    let mut penalty = 0.0;
    for f in findings {
        if !dim_filter(f.code) {
            continue;
        }
        penalty += model.for_severity(f.severity);
    }
    penalty.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_score_is_one() {
        let s = BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!((s.overall - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_score_is_zero() {
        let s = BeliefScore::from_dimensions(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((s.overall - 0.0).abs() < 1e-12);
    }

    #[test]
    fn out_of_range_inputs_get_clamped() {
        let s = BeliefScore::from_dimensions(-1.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        assert_eq!(s.schema_score, 0.0);
        assert_eq!(s.missingness_score, 1.0);
        assert!(s.overall >= 0.0 && s.overall <= 1.0);
    }

    #[test]
    fn nan_input_does_not_corrupt_score() {
        let s = BeliefScore::from_dimensions(f64::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(s.overall.is_finite());
        assert_eq!(s.schema_score, 0.0);
    }

    #[test]
    fn sample_score_is_monotonic_in_n() {
        for w in (0..2000).step_by(31).collect::<Vec<u64>>().windows(2) {
            assert!(sample_score_from_n(w[1]) >= sample_score_from_n(w[0]));
        }
    }

    #[test]
    fn sample_score_at_thirty_is_about_half() {
        let s = sample_score_from_n(30);
        assert!((s - 0.5).abs() < 0.005);
    }

    #[test]
    fn sample_score_at_five_hundred_is_high() {
        assert!(sample_score_from_n(500) > 0.9);
    }

    // ─── v0.2: BeliefWeights tests ─────────────────────────────────────

    #[test]
    fn default_weights_match_unweighted_overall() {
        let s = BeliefScore::from_dimensions(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.3);
        let w = BeliefScore::from_dimensions_weighted(
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            0.4,
            0.3,
            &BeliefWeights::default(),
        );
        assert!(
            (s.overall - w.overall).abs() < 1e-12,
            "default weights must reproduce unweighted overall ({} vs {})",
            s.overall,
            w.overall
        );
    }

    #[test]
    fn weight_emphasis_changes_overall() {
        // Boost missingness × 10 — overall should track missingness more.
        let mut w = BeliefWeights::default();
        w.missingness = 10.0;
        let s = BeliefScore::from_dimensions_weighted(1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &w);
        // unweighted overall = 7/8 = 0.875; weighted: (7*1 + 0*10)/(7+10) = 7/17 ≈ 0.412
        assert!((s.overall - 7.0 / 17.0).abs() < 1e-12);
    }

    #[test]
    fn negative_weights_are_clamped_to_zero() {
        let w = BeliefWeights::from_values(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(w.schema, 0.0);
        let s = BeliefScore::from_dimensions_weighted(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &w);
        // schema (0.5) is ignored due to weight 0 — overall = 7/7 = 1.0
        assert!((s.overall - 1.0).abs() < 1e-12);
    }

    #[test]
    fn all_zero_weights_gives_zero_overall() {
        let w = BeliefWeights::from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let s = BeliefScore::from_dimensions_weighted(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &w);
        assert_eq!(s.overall, 0.0);
    }

    #[test]
    fn default_penalty_matches_v02_baked_in_values() {
        let p = BeliefPenalty::default();
        assert_eq!(p.info, 0.02);
        assert_eq!(p.notice, 0.02);
        assert_eq!(p.warning, 0.10);
        assert_eq!(p.error, 0.25);
    }

    #[test]
    fn penalty_negative_or_nan_is_clamped() {
        let p = BeliefPenalty::from_values(-0.5, f64::NAN, 0.5, 1.0);
        assert_eq!(p.info, 0.0);
        assert_eq!(p.notice, 0.0);
        assert_eq!(p.warning, 0.5);
        assert_eq!(p.error, 1.0);
    }

    #[test]
    fn for_severity_dispatches_correctly() {
        use crate::report::FindingSeverity::*;
        let p = BeliefPenalty::from_values(0.01, 0.05, 0.2, 0.4);
        assert_eq!(p.for_severity(Info), 0.01);
        assert_eq!(p.for_severity(Notice), 0.05);
        assert_eq!(p.for_severity(Warning), 0.2);
        assert_eq!(p.for_severity(Error), 0.4);
    }

    #[test]
    fn weights_with_nan_are_clamped() {
        let w = BeliefWeights::from_values(f64::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(w.schema, 0.0);
    }
}
