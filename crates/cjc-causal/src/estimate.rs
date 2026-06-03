//! [`EffectEstimate`] — the output type every estimator returns.
//!
//! `EffectEstimate` is content-addressed: two runs over the same inputs and
//! the same declared assumptions produce the same
//! [`identifier`](EffectEstimate::identifier). Published causal analyses
//! cite this identifier; downstream code can rely on it as a primary key.

use crate::assumption::IdentificationAssumption;
use crate::FingerprintId;
use std::collections::BTreeMap;

/// A treatment-effect estimate plus everything needed to reproduce or audit it.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectEstimate {
    /// Point estimate of the treatment effect on the response scale of the
    /// outcome variable. For binary outcomes this is a risk difference; for
    /// continuous outcomes it is a mean difference. Estimators document
    /// their specific scale in their per-method docstrings.
    pub point: f64,

    /// Estimator standard error. Computed by the method appropriate to the
    /// estimator (HC1 sandwich for IV, bootstrap for matching, orthogonal-
    /// moment variance for DML).
    pub std_error: f64,

    /// Lower bound of the confidence interval at [`confidence_level`].
    pub ci_lower: f64,

    /// Upper bound of the confidence interval at [`confidence_level`].
    pub ci_upper: f64,

    /// The confidence level used for [`ci_lower`] and [`ci_upper`]. Default
    /// in all v0.1 estimators is `0.95`.
    pub confidence_level: f64,

    /// Number of treated units in the analytic sample (post-refusal,
    /// post-overlap-restriction).
    pub n_treated: u64,

    /// Number of control units in the analytic sample.
    pub n_control: u64,

    /// The assumptions the caller declared when requesting the estimate.
    /// Part of the [`identifier`](Self::identifier) content hash.
    pub assumptions_declared: Vec<IdentificationAssumption>,

    /// Per-estimator balance diagnostics. Populated for matching estimators
    /// (a per-covariate SMD + variance ratio breakdown); `None` for IV and
    /// DML which surface diagnostics through other slots.
    pub balance_diagnostics: Option<BalanceReport>,

    /// First-stage F-statistic for instrument strength, populated by
    /// [`super::IVRegression`] and any future IV-style estimator. `None`
    /// for non-IV estimators.
    ///
    /// Per the Stock-Yogo 2005 critical-value convention, a value of `F < 10`
    /// indicates a weak instrument (causal interpretation of the 2SLS
    /// coefficient is unreliable). cjc-causal also surfaces this case as
    /// Locke finding `E9100` when the threshold is breached.
    pub iv_first_stage_f: Option<f64>,

    /// Content-addressed identifier. Computed via SplitMix64 (per
    /// `cjc_locke::id::fingerprint`) over the canonical byte representation
    /// of (estimator type, treatment column name, outcome column name,
    /// optional instrument column name, covariate column names sorted
    /// ascending, assumptions sorted ascending, seed, point estimate bits,
    /// std_error bits).
    pub identifier: FingerprintId,
}

/// Per-covariate balance diagnostics, returned alongside matching estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct BalanceReport {
    /// Standardised mean difference (SMD) per covariate, post-match. Common
    /// threshold for "good balance" is `|SMD| < 0.10`. Values above this
    /// surface as Locke finding `E9102`.
    ///
    /// `BTreeMap` for deterministic iteration order — never `HashMap`.
    pub smd_post_match: BTreeMap<String, f64>,

    /// Variance ratio per covariate, post-match. Common threshold is
    /// `0.5 < ratio < 2.0`.
    pub variance_ratio_post_match: BTreeMap<String, f64>,

    /// Number of treated units left unmatched after the caliper restriction.
    pub n_treated_unmatched: u64,
}
