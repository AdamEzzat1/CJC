//! Phase 6.1 — cjc-locke `CustomDetector` adapters for Cronos GAN reports.
//!
//! Bridges [`crate::ExperimentSweepReport`] into Locke's tabular detector
//! framework via a "lift" step that converts each [`crate::SweepCell`]
//! into one row of a [`cjc_data::DataFrame`]. The three shipped detectors
//! consume that DataFrame and emit findings in the `E9500..=E9999`
//! custom-detector code range (ADR-0041) on the `Drift` belief axis.
//!
//! # Integration pattern
//!
//! ```ignore
//! use cjc_cronos_gan::{run_experiment_sweep, sweep_report_to_dataframe,
//!     cronos_default_detectors, CronosSeed, SweepBaseConfig};
//!
//! let base = SweepBaseConfig::new(8, 1, 1, 50, 200)
//!     .with_eval_steps(20)
//!     .with_n_seeds(3);
//! let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
//!
//! let df = sweep_report_to_dataframe(&report);
//! // Run the detectors directly via the FindingSink API, or wire them
//! // into cjc-locke's `validate(df, opts)` pipeline alongside the
//! // built-in E9001-E9112 detectors.
//! let detectors = cronos_default_detectors();
//! ```
//!
//! # Why Option C (lift to DataFrame) over the handoff's Option A
//!
//! The handoff sketched serializing `ExperimentReport` into a single
//! DataFrame column (`Option A`). The current shape — one row per
//! `(dataset, mode)` cell, columns per metric — matches Locke's
//! tabular mental model: row_range evidence points at a specific cell,
//! column evidence points at the specific metric (e.g.
//! `"regime_shift_score"`), and the same DataFrame is directly
//! inspectable in the REPL / debugger without a serialize step.
//!
//! # Determinism
//!
//! Both the lift function and the detectors are pure functions of the
//! input report — no `Instant::now`, no `HashMap` iteration, no random
//! draws. Two runs against the same report produce byte-identical
//! findings (same code, message, evidence, row_range).

use std::sync::Arc;

use cjc_data::{Column, DataFrame};
use cjc_locke::custom_detector::{BeliefAxisSet, CustomDetector, FindingSink};
use cjc_locke::report::{FindingEvidence, FindingSeverity};

use crate::experiment::ExperimentSweepReport;

// ─── Default thresholds ──────────────────────────────────────────────────

/// Default for [`CronosRegimeShiftDetector`]: a cell's
/// `max_regime_shift_score` above this fires E9500. Chosen against
/// Phase 4c's `examples/sweep` output, where the typical max regime
/// shift scores cluster in `[0.5, 1.2]` — `1.0` separates the
/// quiet-but-rising regime from the loud-and-localised one.
pub const DEFAULT_REGIME_SHIFT_THRESHOLD: f64 = 1.0;

/// Default for [`CronosPersistentDisagreementDetector`]: a cell's
/// held-out `eval_absolute_gap` above this fires E9501.
pub const DEFAULT_ABSOLUTE_GAP_THRESHOLD: f64 = 0.5;

/// Default for [`CronosSsmEvalDegradationDetector`]: a cell's
/// `eval_ssm_loss / final_loss_ssm` above this fires E9502 (SSM is
/// ≥2× worse on held-out data than on training data).
pub const DEFAULT_SSM_LOSS_DEGRADATION_RATIO: f64 = 2.0;

// ─── DataFrame schema ────────────────────────────────────────────────────

/// The 17 column names produced by [`sweep_report_to_dataframe`], in
/// canonical order. Exposed so detectors and external callers can use
/// them without hard-coding strings.
pub const SWEEP_DATAFRAME_COLUMNS: [&str; 17] = [
    "dataset",
    "mode",
    "n_seeds",
    "final_loss_ssm",
    "final_loss_ssm_std",
    "final_loss_liquid",
    "final_loss_liquid_std",
    "mean_absolute_gap",
    "mean_absolute_gap_std",
    "max_regime_shift_score",
    "max_regime_shift_score_std",
    "eval_ssm_loss",
    "eval_ssm_loss_std",
    "eval_liquid_loss",
    "eval_liquid_loss_std",
    "eval_absolute_gap",
    "eval_absolute_gap_std",
];

// ─── Lift function ───────────────────────────────────────────────────────

/// Convert an [`ExperimentSweepReport`] into a tabular
/// [`cjc_data::DataFrame`] suitable for cjc-locke custom-detector
/// consumption.
///
/// Each row corresponds to one [`SweepCell`] in the report's canonical
/// order (`SWEEP_DATASETS` outer, `SWEEP_MODES` inner). Per-seed
/// aggregation: each numeric column carries the cell's mean across
/// seeds, plus a paired `_std` column carrying the sample standard
/// deviation (0.0 when `n_seeds == 1`). Eval columns are NaN for cells
/// that ran without an eval window (`eval_steps == 0`).
///
/// See [`SWEEP_DATAFRAME_COLUMNS`] for the canonical column order.
pub fn sweep_report_to_dataframe(report: &ExperimentSweepReport) -> DataFrame {
    let n_rows = report.cells.len();

    let mut datasets: Vec<String> = Vec::with_capacity(n_rows);
    let mut modes: Vec<String> = Vec::with_capacity(n_rows);
    let mut n_seeds: Vec<f64> = Vec::with_capacity(n_rows);
    let mut ssm_loss: Vec<f64> = Vec::with_capacity(n_rows);
    let mut ssm_loss_std: Vec<f64> = Vec::with_capacity(n_rows);
    let mut liq_loss: Vec<f64> = Vec::with_capacity(n_rows);
    let mut liq_loss_std: Vec<f64> = Vec::with_capacity(n_rows);
    let mut gap: Vec<f64> = Vec::with_capacity(n_rows);
    let mut gap_std: Vec<f64> = Vec::with_capacity(n_rows);
    let mut regime: Vec<f64> = Vec::with_capacity(n_rows);
    let mut regime_std: Vec<f64> = Vec::with_capacity(n_rows);
    let mut eval_ssm: Vec<f64> = Vec::with_capacity(n_rows);
    let mut eval_ssm_std: Vec<f64> = Vec::with_capacity(n_rows);
    let mut eval_liq: Vec<f64> = Vec::with_capacity(n_rows);
    let mut eval_liq_std: Vec<f64> = Vec::with_capacity(n_rows);
    let mut eval_gap: Vec<f64> = Vec::with_capacity(n_rows);
    let mut eval_gap_std: Vec<f64> = Vec::with_capacity(n_rows);

    for cell in &report.cells {
        datasets.push(cell.dataset.label().to_string());
        modes.push(cell.mode.label().to_string());
        n_seeds.push(cell.n_seeds() as f64);
        ssm_loss.push(cell.mean.final_loss_ssm);
        ssm_loss_std.push(cell.variance.final_loss_ssm.sqrt());
        liq_loss.push(cell.mean.final_loss_liquid);
        liq_loss_std.push(cell.variance.final_loss_liquid.sqrt());
        gap.push(cell.mean.mean_absolute_gap);
        gap_std.push(cell.variance.mean_absolute_gap.sqrt());
        regime.push(cell.mean.max_regime_shift_score);
        regime_std.push(cell.variance.max_regime_shift_score.sqrt());
        eval_ssm.push(cell.mean.eval_ssm_loss.unwrap_or(f64::NAN));
        eval_ssm_std.push(
            cell.variance
                .eval_ssm_loss
                .map(|v| v.sqrt())
                .unwrap_or(f64::NAN),
        );
        eval_liq.push(cell.mean.eval_liquid_loss.unwrap_or(f64::NAN));
        eval_liq_std.push(
            cell.variance
                .eval_liquid_loss
                .map(|v| v.sqrt())
                .unwrap_or(f64::NAN),
        );
        eval_gap.push(cell.mean.eval_absolute_gap.unwrap_or(f64::NAN));
        eval_gap_std.push(
            cell.variance
                .eval_absolute_gap
                .map(|v| v.sqrt())
                .unwrap_or(f64::NAN),
        );
    }

    let cols: Vec<(String, Column)> = vec![
        ("dataset".to_string(), Column::Str(datasets)),
        ("mode".to_string(), Column::Str(modes)),
        ("n_seeds".to_string(), Column::Float(n_seeds)),
        ("final_loss_ssm".to_string(), Column::Float(ssm_loss)),
        ("final_loss_ssm_std".to_string(), Column::Float(ssm_loss_std)),
        ("final_loss_liquid".to_string(), Column::Float(liq_loss)),
        (
            "final_loss_liquid_std".to_string(),
            Column::Float(liq_loss_std),
        ),
        ("mean_absolute_gap".to_string(), Column::Float(gap)),
        ("mean_absolute_gap_std".to_string(), Column::Float(gap_std)),
        ("max_regime_shift_score".to_string(), Column::Float(regime)),
        (
            "max_regime_shift_score_std".to_string(),
            Column::Float(regime_std),
        ),
        ("eval_ssm_loss".to_string(), Column::Float(eval_ssm)),
        ("eval_ssm_loss_std".to_string(), Column::Float(eval_ssm_std)),
        ("eval_liquid_loss".to_string(), Column::Float(eval_liq)),
        (
            "eval_liquid_loss_std".to_string(),
            Column::Float(eval_liq_std),
        ),
        ("eval_absolute_gap".to_string(), Column::Float(eval_gap)),
        (
            "eval_absolute_gap_std".to_string(),
            Column::Float(eval_gap_std),
        ),
    ];

    // All column lengths are n_rows by construction, so from_columns
    // cannot fail; expect() to surface the assumption.
    DataFrame::from_columns(cols).expect("sweep_report_to_dataframe column lengths agree")
}

/// Convenience: also lift a single [`crate::ExperimentReport`] into a
/// one-row DataFrame using the same schema. Useful when callers run
/// `run_experiment` directly rather than `run_experiment_sweep`. The
/// per-column standard-deviation values are `0.0` since this is a
/// single observation (matches the `n_seeds == 1` convention used by
/// `sweep_report_to_dataframe`).
pub fn experiment_report_to_dataframe(
    report: &crate::experiment::ExperimentReport,
) -> DataFrame {
    let (eval_ssm, eval_liq, eval_gap) = match &report.eval {
        Some(e) => (e.ssm_loss, e.liquid_loss, e.disagreement.absolute_gap),
        None => (f64::NAN, f64::NAN, f64::NAN),
    };

    let cols: Vec<(String, Column)> = vec![
        (
            "dataset".to_string(),
            Column::Str(vec![report.dataset_label.to_string()]),
        ),
        (
            "mode".to_string(),
            Column::Str(vec![report.mode.label().to_string()]),
        ),
        ("n_seeds".to_string(), Column::Float(vec![1.0])),
        (
            "final_loss_ssm".to_string(),
            Column::Float(vec![report.final_loss_ssm]),
        ),
        ("final_loss_ssm_std".to_string(), Column::Float(vec![0.0])),
        (
            "final_loss_liquid".to_string(),
            Column::Float(vec![report.final_loss_liquid]),
        ),
        (
            "final_loss_liquid_std".to_string(),
            Column::Float(vec![0.0]),
        ),
        (
            "mean_absolute_gap".to_string(),
            Column::Float(vec![report.mean_absolute_gap]),
        ),
        (
            "mean_absolute_gap_std".to_string(),
            Column::Float(vec![0.0]),
        ),
        (
            "max_regime_shift_score".to_string(),
            Column::Float(vec![report.max_regime_shift_score]),
        ),
        (
            "max_regime_shift_score_std".to_string(),
            Column::Float(vec![0.0]),
        ),
        ("eval_ssm_loss".to_string(), Column::Float(vec![eval_ssm])),
        (
            "eval_ssm_loss_std".to_string(),
            Column::Float(vec![if eval_ssm.is_finite() { 0.0 } else { f64::NAN }]),
        ),
        ("eval_liquid_loss".to_string(), Column::Float(vec![eval_liq])),
        (
            "eval_liquid_loss_std".to_string(),
            Column::Float(vec![if eval_liq.is_finite() { 0.0 } else { f64::NAN }]),
        ),
        ("eval_absolute_gap".to_string(), Column::Float(vec![eval_gap])),
        (
            "eval_absolute_gap_std".to_string(),
            Column::Float(vec![if eval_gap.is_finite() { 0.0 } else { f64::NAN }]),
        ),
    ];

    DataFrame::from_columns(cols).expect("experiment_report_to_dataframe column lengths agree")
}

// ─── Detector: E9500 regime-shift ────────────────────────────────────────

/// E9500: high in-training regime-shift score.
///
/// Fires when a cell's `max_regime_shift_score` exceeds
/// [`Self::threshold`]. The regime-shift score is the localised
/// disagreement gap normalised by the magnitude of the predictor's
/// own response — a high value means the SSM and Liquid networks
/// diverged sharply on a specific subset of timesteps, which is the
/// brief's "regime shift" signal.
#[derive(Clone, Debug)]
pub struct CronosRegimeShiftDetector {
    pub threshold: f64,
}

impl Default for CronosRegimeShiftDetector {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_REGIME_SHIFT_THRESHOLD,
        }
    }
}

impl CronosRegimeShiftDetector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl CustomDetector for CronosRegimeShiftDetector {
    fn code(&self) -> &'static str {
        "E9500"
    }

    fn belief_axes(&self) -> BeliefAxisSet {
        BeliefAxisSet::DRIFT
    }

    fn name(&self) -> &str {
        "cronos_regime_shift"
    }

    fn run(&self, df: &DataFrame, sink: &mut FindingSink) {
        let Some((dataset_col, mode_col, regime_col, n_seeds_col)) =
            sweep_df_handles(df, "max_regime_shift_score")
        else {
            return;
        };
        let n_rows = regime_col.len();
        for row in 0..n_rows {
            let value = regime_col[row];
            if !value.is_finite() || value <= self.threshold {
                continue;
            }
            let dataset = &dataset_col[row];
            let mode = &mode_col[row];
            let n_seeds = n_seeds_col[row];
            let message = format!(
                "Cronos GAN regime shift on ({}, {}): max_regime_shift_score = {:.4} > threshold {:.4}",
                dataset, mode, value, self.threshold
            );
            sink.emit(
                FindingSeverity::Warning,
                message,
                Some("max_regime_shift_score".to_string()),
                Some((row, row + 1)),
                vec![
                    FindingEvidence::Metric {
                        label: "max_regime_shift_score".to_string(),
                        value,
                    },
                    FindingEvidence::Metric {
                        label: "threshold".to_string(),
                        value: self.threshold,
                    },
                ],
                n_seeds.max(1.0) as u64,
            );
        }
    }
}

// ─── Detector: E9501 persistent disagreement on held-out window ─────────

/// E9501: persistent calibrated disagreement on the held-out window.
///
/// Fires when a cell's `eval_absolute_gap` exceeds
/// [`Self::threshold`]. Non-fires for cells that ran without an eval
/// window (`eval_steps == 0` ⇒ NaN gap value).
#[derive(Clone, Debug)]
pub struct CronosPersistentDisagreementDetector {
    pub threshold: f64,
}

impl Default for CronosPersistentDisagreementDetector {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_ABSOLUTE_GAP_THRESHOLD,
        }
    }
}

impl CronosPersistentDisagreementDetector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl CustomDetector for CronosPersistentDisagreementDetector {
    fn code(&self) -> &'static str {
        "E9501"
    }

    fn belief_axes(&self) -> BeliefAxisSet {
        BeliefAxisSet::DRIFT
    }

    fn name(&self) -> &str {
        "cronos_persistent_disagreement"
    }

    fn run(&self, df: &DataFrame, sink: &mut FindingSink) {
        let Some((dataset_col, mode_col, gap_col, n_seeds_col)) =
            sweep_df_handles(df, "eval_absolute_gap")
        else {
            return;
        };
        let n_rows = gap_col.len();
        for row in 0..n_rows {
            let value = gap_col[row];
            // NaN ⇒ eval was not run on this cell ⇒ no signal to emit.
            if !value.is_finite() || value <= self.threshold {
                continue;
            }
            let dataset = &dataset_col[row];
            let mode = &mode_col[row];
            let n_seeds = n_seeds_col[row];
            let message = format!(
                "Persistent disagreement on held-out window for ({}, {}): eval_absolute_gap = {:.4} > threshold {:.4}",
                dataset, mode, value, self.threshold
            );
            sink.emit(
                FindingSeverity::Notice,
                message,
                Some("eval_absolute_gap".to_string()),
                Some((row, row + 1)),
                vec![
                    FindingEvidence::Metric {
                        label: "eval_absolute_gap".to_string(),
                        value,
                    },
                    FindingEvidence::Metric {
                        label: "threshold".to_string(),
                        value: self.threshold,
                    },
                ],
                n_seeds.max(1.0) as u64,
            );
        }
    }
}

// ─── Detector: E9502 SSM eval degradation ────────────────────────────────

/// E9502: SSM held-out loss substantially worse than training loss.
///
/// Fires when `eval_ssm_loss / final_loss_ssm` exceeds
/// [`Self::ratio_threshold`] — a sign the SSM overfit or the data
/// distribution shifted between the training and eval windows. Skips
/// cells without eval data and cells where `final_loss_ssm` is zero
/// or non-finite (ratio undefined).
#[derive(Clone, Debug)]
pub struct CronosSsmEvalDegradationDetector {
    pub ratio_threshold: f64,
}

impl Default for CronosSsmEvalDegradationDetector {
    fn default() -> Self {
        Self {
            ratio_threshold: DEFAULT_SSM_LOSS_DEGRADATION_RATIO,
        }
    }
}

impl CronosSsmEvalDegradationDetector {
    pub fn new(ratio_threshold: f64) -> Self {
        Self { ratio_threshold }
    }
}

impl CustomDetector for CronosSsmEvalDegradationDetector {
    fn code(&self) -> &'static str {
        "E9502"
    }

    fn belief_axes(&self) -> BeliefAxisSet {
        BeliefAxisSet::DRIFT
    }

    fn name(&self) -> &str {
        "cronos_ssm_eval_degradation"
    }

    fn run(&self, df: &DataFrame, sink: &mut FindingSink) {
        let Some((dataset_col, mode_col, eval_ssm_col, n_seeds_col)) =
            sweep_df_handles(df, "eval_ssm_loss")
        else {
            return;
        };
        let train_ssm_col = match df.columns.iter().find(|(n, _)| n == "final_loss_ssm") {
            Some((_, Column::Float(v))) => v,
            _ => return,
        };
        let n_rows = eval_ssm_col.len();
        for row in 0..n_rows {
            let eval_v = eval_ssm_col[row];
            let train_v = train_ssm_col[row];
            // Skip cells with no eval, undefined ratio, or
            // pathological denominators.
            if !eval_v.is_finite() || !train_v.is_finite() || train_v <= 0.0 {
                continue;
            }
            let ratio = eval_v / train_v;
            if ratio <= self.ratio_threshold {
                continue;
            }
            let dataset = &dataset_col[row];
            let mode = &mode_col[row];
            let n_seeds = n_seeds_col[row];
            let message = format!(
                "SSM eval degradation on ({}, {}): eval_ssm_loss / final_loss_ssm = {:.4} > ratio threshold {:.4}",
                dataset, mode, ratio, self.ratio_threshold
            );
            sink.emit(
                FindingSeverity::Notice,
                message,
                Some("eval_ssm_loss".to_string()),
                Some((row, row + 1)),
                vec![
                    FindingEvidence::Metric {
                        label: "eval_ssm_loss".to_string(),
                        value: eval_v,
                    },
                    FindingEvidence::Metric {
                        label: "final_loss_ssm".to_string(),
                        value: train_v,
                    },
                    FindingEvidence::Ratio {
                        label: "eval_over_train_ratio".to_string(),
                        value: (ratio / self.ratio_threshold).min(1.0),
                    },
                ],
                n_seeds.max(1.0) as u64,
            );
        }
    }
}

// ─── Convenience: bundle all three detectors with defaults ──────────────

/// Convenience: returns all three default-configured Cronos detectors
/// wrapped as `Arc<dyn CustomDetector>`, suitable for direct insertion
/// into [`cjc_locke::ValidateOptions::custom_detectors`].
pub fn cronos_default_detectors() -> Vec<Arc<dyn CustomDetector>> {
    vec![
        Arc::new(CronosRegimeShiftDetector::default()),
        Arc::new(CronosPersistentDisagreementDetector::default()),
        Arc::new(CronosSsmEvalDegradationDetector::default()),
    ]
}

// ─── Internal helpers ───────────────────────────────────────────────────

/// Borrow `(dataset_col, mode_col, metric_col, n_seeds_col)` from a
/// lifted sweep DataFrame. Returns `None` if any required column is
/// missing or has the wrong type — the detector silently no-ops on
/// non-Cronos DataFrames rather than panicking.
fn sweep_df_handles<'a>(
    df: &'a DataFrame,
    metric: &str,
) -> Option<(&'a [String], &'a [String], &'a [f64], &'a [f64])> {
    let dataset_col = match df.columns.iter().find(|(n, _)| n == "dataset") {
        Some((_, Column::Str(v))) => v.as_slice(),
        _ => return None,
    };
    let mode_col = match df.columns.iter().find(|(n, _)| n == "mode") {
        Some((_, Column::Str(v))) => v.as_slice(),
        _ => return None,
    };
    let metric_col = match df.columns.iter().find(|(n, _)| n == metric) {
        Some((_, Column::Float(v))) => v.as_slice(),
        _ => return None,
    };
    let n_seeds_col = match df.columns.iter().find(|(n, _)| n == "n_seeds") {
        Some((_, Column::Float(v))) => v.as_slice(),
        _ => return None,
    };
    Some((dataset_col, mode_col, metric_col, n_seeds_col))
}
