//! Phase 6.1 integration tests for the cjc-locke detector adapters.
//!
//! Coverage:
//! - § A `sweep_report_to_dataframe` lift shape + content (5)
//! - § B `experiment_report_to_dataframe` single-row lift (2)
//! - § C Per-detector fire / no-fire behaviour (6)
//! - § D Detector metadata: code, axes, name (3)
//! - § E Determinism: same input → same findings (1)
//! - § F Bundle helper + malformed-DataFrame safety (2)

use std::sync::Arc;

use cjc_cronos_gan::{
    cronos_default_detectors, disagreement_trajectory_to_dataframe,
    experiment_report_to_dataframe, run_experiment, run_experiment_sweep,
    sweep_disagreement_trajectory_to_dataframe, sweep_report_to_dataframe, CronosDataset,
    CronosPersistentDisagreementDetector, CronosRegimeShiftDetector, CronosSeed,
    CronosSsmEvalDegradationDetector, ExperimentConfig, SweepBaseConfig, TemporalGanConfig,
    SWEEP_DATAFRAME_COLUMNS, SWEEP_DATASETS, SWEEP_MODES, TRAJECTORY_DATAFRAME_COLUMNS,
};
use cjc_data::{Column, DataFrame};
use cjc_locke::custom_detector::{BeliefAxisSet, CustomDetector, FindingSink};

fn tiny_sweep() -> SweepBaseConfig {
    SweepBaseConfig::new(4, 1, 1, 10, 5).with_lambda_disagreement(0.1)
}

fn tiny_sweep_with_eval() -> SweepBaseConfig {
    tiny_sweep().with_eval_steps(4)
}

// ─── § A sweep_report_to_dataframe lift shape + content (5) ─────────────

#[test]
fn lift_has_one_row_per_cell_and_canonical_column_order() {
    let base = tiny_sweep();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_report_to_dataframe(&report);
    assert_eq!(df.columns.len(), SWEEP_DATAFRAME_COLUMNS.len());
    for (i, expected_name) in SWEEP_DATAFRAME_COLUMNS.iter().enumerate() {
        assert_eq!(
            df.columns[i].0, *expected_name,
            "column {} should be `{}`, got `{}`",
            i, expected_name, df.columns[i].0
        );
    }
    // 15 cells (5 datasets × 3 modes).
    for (_, col) in &df.columns {
        let len = match col {
            Column::Str(v) => v.len(),
            Column::Float(v) => v.len(),
            _ => 0,
        };
        assert_eq!(len, 15);
    }
}

#[test]
fn lift_dataset_and_mode_columns_have_string_labels() {
    let base = tiny_sweep();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_report_to_dataframe(&report);
    let datasets = match &df.columns[0].1 {
        Column::Str(v) => v.clone(),
        _ => panic!("dataset col not Str"),
    };
    let modes = match &df.columns[1].1 {
        Column::Str(v) => v.clone(),
        _ => panic!("mode col not Str"),
    };
    // Row order: datasets outer, modes inner.
    let mut expected: Vec<(String, String)> = Vec::new();
    for &ds in &SWEEP_DATASETS {
        for &mode in &SWEEP_MODES {
            expected.push((ds.label().to_string(), mode.label().to_string()));
        }
    }
    let actual: Vec<(String, String)> = datasets.into_iter().zip(modes.into_iter()).collect();
    assert_eq!(actual, expected);
}

#[test]
fn lift_eval_columns_are_nan_when_eval_absent() {
    let base = tiny_sweep();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_report_to_dataframe(&report);
    let eval_ssm = match df.columns.iter().find(|(n, _)| n == "eval_ssm_loss") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("eval_ssm_loss missing"),
    };
    for v in eval_ssm {
        assert!(v.is_nan(), "expected NaN for eval_ssm_loss without eval, got {}", v);
    }
}

#[test]
fn lift_eval_columns_are_finite_when_eval_present() {
    let base = tiny_sweep_with_eval();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_report_to_dataframe(&report);
    for col_name in ["eval_ssm_loss", "eval_liquid_loss", "eval_absolute_gap"] {
        let values = match df.columns.iter().find(|(n, _)| n == col_name) {
            Some((_, Column::Float(v))) => v.clone(),
            _ => panic!("{} missing", col_name),
        };
        for v in values {
            assert!(
                v.is_finite() && v >= 0.0,
                "{} value should be finite + non-negative, got {}",
                col_name,
                v
            );
        }
    }
}

#[test]
fn lift_std_columns_are_zero_when_n_seeds_is_one() {
    let base = tiny_sweep().with_n_seeds(1);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_report_to_dataframe(&report);
    for col_name in [
        "final_loss_ssm_std",
        "final_loss_liquid_std",
        "mean_absolute_gap_std",
        "max_regime_shift_score_std",
    ] {
        let values = match df.columns.iter().find(|(n, _)| n == col_name) {
            Some((_, Column::Float(v))) => v.clone(),
            _ => panic!("{} missing", col_name),
        };
        for v in values {
            assert_eq!(
                v.to_bits(),
                0.0_f64.to_bits(),
                "{} should be 0.0 for single-seed cell, got {}",
                col_name,
                v
            );
        }
    }
}

// ─── § B experiment_report_to_dataframe single-row lift (2) ─────────────

#[test]
fn single_experiment_lift_has_one_row_and_canonical_columns() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 12)
        .with_n_train_steps(8)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = experiment_report_to_dataframe(&report);
    assert_eq!(df.columns.len(), SWEEP_DATAFRAME_COLUMNS.len());
    for (i, expected_name) in SWEEP_DATAFRAME_COLUMNS.iter().enumerate() {
        assert_eq!(df.columns[i].0, *expected_name);
    }
    for (_, col) in &df.columns {
        let len = match col {
            Column::Str(v) => v.len(),
            Column::Float(v) => v.len(),
            _ => 0,
        };
        assert_eq!(len, 1, "single-experiment lift should have exactly one row");
    }
    let datasets = match &df.columns[0].1 {
        Column::Str(v) => v.clone(),
        _ => panic!("dataset col not Str"),
    };
    assert_eq!(datasets[0], "regime_shift");
}

#[test]
fn single_experiment_lift_carries_eval_when_present() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 12)
        .with_n_train_steps(10)
        .with_eval_steps(5)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = experiment_report_to_dataframe(&report);
    let eval_ssm = match df.columns.iter().find(|(n, _)| n == "eval_ssm_loss") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("eval_ssm_loss missing"),
    };
    assert!(
        eval_ssm[0].is_finite() && eval_ssm[0] >= 0.0,
        "single-experiment lift should carry eval_ssm_loss when eval ran, got {}",
        eval_ssm[0]
    );
}

// ─── § C Per-detector fire / no-fire behaviour (6) ──────────────────────

/// Construct a hand-built DataFrame matching the lift schema with
/// caller-supplied values for the metric columns. Used to drive the
/// detectors deterministically without paying for a full sweep run.
fn synthetic_df(
    regime_shift: Vec<f64>,
    eval_gap: Vec<f64>,
    final_loss_ssm: Vec<f64>,
    eval_ssm_loss: Vec<f64>,
) -> DataFrame {
    let n_rows = regime_shift.len();
    let datasets: Vec<String> = (0..n_rows).map(|i| format!("synth_{}", i)).collect();
    let modes: Vec<String> = (0..n_rows).map(|_| "symmetric".to_string()).collect();
    let cols: Vec<(String, Column)> = vec![
        ("dataset".to_string(), Column::Str(datasets)),
        ("mode".to_string(), Column::Str(modes)),
        ("n_seeds".to_string(), Column::Float(vec![1.0; n_rows])),
        ("final_loss_ssm".to_string(), Column::Float(final_loss_ssm)),
        ("final_loss_ssm_std".to_string(), Column::Float(vec![0.0; n_rows])),
        ("final_loss_liquid".to_string(), Column::Float(vec![0.05; n_rows])),
        ("final_loss_liquid_std".to_string(), Column::Float(vec![0.0; n_rows])),
        ("mean_absolute_gap".to_string(), Column::Float(vec![0.1; n_rows])),
        ("mean_absolute_gap_std".to_string(), Column::Float(vec![0.0; n_rows])),
        ("max_regime_shift_score".to_string(), Column::Float(regime_shift)),
        ("max_regime_shift_score_std".to_string(), Column::Float(vec![0.0; n_rows])),
        ("eval_ssm_loss".to_string(), Column::Float(eval_ssm_loss)),
        ("eval_ssm_loss_std".to_string(), Column::Float(vec![0.0; n_rows])),
        ("eval_liquid_loss".to_string(), Column::Float(vec![0.1; n_rows])),
        ("eval_liquid_loss_std".to_string(), Column::Float(vec![0.0; n_rows])),
        ("eval_absolute_gap".to_string(), Column::Float(eval_gap)),
        ("eval_absolute_gap_std".to_string(), Column::Float(vec![0.0; n_rows])),
    ];
    DataFrame::from_columns(cols).unwrap()
}

#[test]
fn regime_shift_detector_fires_above_threshold_and_not_below() {
    let df = synthetic_df(
        /* regime_shift */ vec![0.5, 1.5, 0.9, 2.0],
        /* eval_gap */ vec![0.1, 0.1, 0.1, 0.1],
        /* train_ssm */ vec![0.01, 0.01, 0.01, 0.01],
        /* eval_ssm */ vec![0.01, 0.01, 0.01, 0.01],
    );
    let detector = CronosRegimeShiftDetector::new(1.0);
    let mut sink = FindingSink::new("E9500", BeliefAxisSet::DRIFT);
    detector.run(&df, &mut sink);
    let (findings, _errors) = sink.drain();
    // Rows 1 (1.5) and 3 (2.0) exceed threshold 1.0; rows 0 (0.5) and
    // 2 (0.9) do not.
    assert_eq!(
        findings.len(),
        2,
        "expected 2 findings, got {}: {:?}",
        findings.len(),
        findings.iter().map(|f| f.message.as_str()).collect::<Vec<_>>()
    );
    for f in &findings {
        assert_eq!(f.code, "E9500");
    }
}

#[test]
fn regime_shift_detector_silent_when_all_under_threshold() {
    let df = synthetic_df(
        vec![0.0, 0.5, 0.9],
        vec![0.1, 0.1, 0.1],
        vec![0.01, 0.01, 0.01],
        vec![0.01, 0.01, 0.01],
    );
    let detector = CronosRegimeShiftDetector::new(1.0);
    let mut sink = FindingSink::new("E9500", BeliefAxisSet::DRIFT);
    detector.run(&df, &mut sink);
    let (findings, _) = sink.drain();
    assert_eq!(findings.len(), 0);
}

#[test]
fn persistent_disagreement_detector_fires_on_high_eval_gap() {
    let df = synthetic_df(
        vec![0.1, 0.1, 0.1, 0.1],
        /* eval_gap */ vec![0.2, 0.8, 0.4, 1.2],
        vec![0.01, 0.01, 0.01, 0.01],
        vec![0.01, 0.01, 0.01, 0.01],
    );
    let detector = CronosPersistentDisagreementDetector::new(0.5);
    let mut sink = FindingSink::new("E9501", BeliefAxisSet::DRIFT);
    detector.run(&df, &mut sink);
    let (findings, _errors) = sink.drain();
    // Rows 1 (0.8) and 3 (1.2) exceed threshold 0.5.
    assert_eq!(findings.len(), 2);
    for f in &findings {
        assert_eq!(f.code, "E9501");
    }
}

#[test]
fn persistent_disagreement_detector_silent_on_nan_eval_gap() {
    // eval_gap NaN ⇒ eval was not run on this cell ⇒ no signal.
    let df = synthetic_df(
        vec![0.1, 0.1],
        vec![f64::NAN, f64::NAN],
        vec![0.01, 0.01],
        vec![0.01, 0.01],
    );
    let detector = CronosPersistentDisagreementDetector::new(0.5);
    let mut sink = FindingSink::new("E9501", BeliefAxisSet::DRIFT);
    detector.run(&df, &mut sink);
    let (findings, _) = sink.drain();
    assert_eq!(findings.len(), 0);
}

#[test]
fn ssm_eval_degradation_detector_fires_above_ratio_threshold() {
    // ratio = eval_ssm / train_ssm
    // row 0: 0.01/0.01 = 1.0  → below default 2.0
    // row 1: 0.05/0.01 = 5.0  → above
    // row 2: 0.02/0.01 = 2.0  → at threshold (strict > used so no fire)
    // row 3: 0.10/0.02 = 5.0  → above
    let df = synthetic_df(
        vec![0.1; 4],
        vec![0.1; 4],
        /* train_ssm */ vec![0.01, 0.01, 0.01, 0.02],
        /* eval_ssm */ vec![0.01, 0.05, 0.02, 0.10],
    );
    let detector = CronosSsmEvalDegradationDetector::new(2.0);
    let mut sink = FindingSink::new("E9502", BeliefAxisSet::DRIFT);
    detector.run(&df, &mut sink);
    let (findings, _errors) = sink.drain();
    assert_eq!(findings.len(), 2);
    for f in &findings {
        assert_eq!(f.code, "E9502");
    }
}

#[test]
fn ssm_eval_degradation_silent_when_train_loss_zero_or_nan_eval() {
    let df = synthetic_df(
        vec![0.1; 3],
        vec![0.1; 3],
        /* train_ssm */ vec![0.0, 1.0, 1.0],
        /* eval_ssm  */ vec![0.5, f64::NAN, 1.5],
    );
    let detector = CronosSsmEvalDegradationDetector::new(2.0);
    let mut sink = FindingSink::new("E9502", BeliefAxisSet::DRIFT);
    detector.run(&df, &mut sink);
    let (findings, _errors) = sink.drain();
    // Row 0: train=0 → undefined ratio → skip.
    // Row 1: eval NaN → skip.
    // Row 2: ratio 1.5 < 2.0 → no fire.
    assert_eq!(findings.len(), 0);
}

// ─── § D Detector metadata (3) ──────────────────────────────────────────

#[test]
fn detector_codes_are_in_custom_range() {
    let detectors = cronos_default_detectors();
    let codes: Vec<&str> = detectors.iter().map(|d| d.code()).collect();
    assert_eq!(codes, vec!["E9500", "E9501", "E9502"]);
    for code in &codes {
        let n: u32 = code[1..].parse().unwrap();
        assert!(
            (9500..=9999).contains(&n),
            "code {} outside E9500..=E9999 custom range",
            code
        );
    }
}

#[test]
fn detector_belief_axes_all_drift() {
    let detectors = cronos_default_detectors();
    for d in &detectors {
        assert_eq!(
            d.belief_axes(),
            BeliefAxisSet::DRIFT,
            "detector {} should declare DRIFT axis only, got {:?}",
            d.code(),
            d.belief_axes()
        );
    }
}

#[test]
fn detector_names_are_descriptive() {
    let detectors = cronos_default_detectors();
    let names: Vec<&str> = detectors.iter().map(|d| d.name()).collect();
    assert_eq!(
        names,
        vec![
            "cronos_regime_shift",
            "cronos_persistent_disagreement",
            "cronos_ssm_eval_degradation"
        ]
    );
}

// ─── § E Determinism (1) ────────────────────────────────────────────────

#[test]
fn detector_findings_are_deterministic_across_runs() {
    // Same DataFrame, two detector runs must produce the same finding
    // messages and metrics. (FindingSink stores messages + evidence
    // verbatim; the framework's outer sort is keyed off the same
    // sort_key both times.)
    let df = synthetic_df(
        vec![1.5, 0.5, 2.0, 0.9],
        vec![0.8, 0.2, 1.2, 0.4],
        vec![0.01, 0.01, 0.02, 0.01],
        vec![0.05, 0.01, 0.10, 0.01],
    );
    let detectors = cronos_default_detectors();
    let run_all = || -> Vec<(String, String)> {
        let mut out = Vec::new();
        for d in &detectors {
            // The detector's static code is already &'static str.
            let mut sink = FindingSink::new(d.code(), d.belief_axes());
            d.run(&df, &mut sink);
            let (findings, _) = sink.drain();
            for f in findings {
                out.push((f.code.to_string(), f.message.clone()));
            }
        }
        out
    };
    let a = run_all();
    let b = run_all();
    assert_eq!(a, b, "two runs of the same detector set should be byte-equal");
    assert!(!a.is_empty(), "expected at least one finding from the synthetic input");
}

// ─── § F Bundle helper + malformed-DataFrame safety (2) ─────────────────

#[test]
fn cronos_default_detectors_returns_three_distinct_detectors() {
    let detectors: Vec<Arc<dyn CustomDetector>> = cronos_default_detectors();
    assert_eq!(detectors.len(), 3);
    let codes: Vec<&str> = detectors.iter().map(|d| d.code()).collect();
    assert_eq!(codes, vec!["E9500", "E9501", "E9502"]);
}

// ─── § G Phase 7a.2 per-timestep trajectory lifts (8) ──────────────────

#[test]
fn trajectory_lift_has_one_row_per_disagreement_step() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 15)
        .with_n_train_steps(8)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = disagreement_trajectory_to_dataframe(&report);
    // n_train_steps + 1 = initial step + 8 update steps = 9 rows.
    let expected_rows = report.disagreement_trajectory.len();
    assert_eq!(expected_rows, cfg.n_train_steps + 1);
    for (name, col) in &df.columns {
        let len = match col {
            Column::Str(v) => v.len(),
            Column::Float(v) => v.len(),
            _ => 0,
        };
        assert_eq!(
            len, expected_rows,
            "column `{}` has {} rows, expected {}",
            name, len, expected_rows
        );
    }
}

#[test]
fn trajectory_lift_canonical_column_order() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 10)
        .with_n_train_steps(5)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = disagreement_trajectory_to_dataframe(&report);
    assert_eq!(df.columns.len(), TRAJECTORY_DATAFRAME_COLUMNS.len());
    for (i, expected_name) in TRAJECTORY_DATAFRAME_COLUMNS.iter().enumerate() {
        assert_eq!(df.columns[i].0, *expected_name);
    }
}

#[test]
fn trajectory_lift_step_column_is_zero_to_n_inclusive() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::NoisySine, 10)
        .with_n_train_steps(6)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = disagreement_trajectory_to_dataframe(&report);
    let steps = match df.columns.iter().find(|(n, _)| n == "step") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("step column missing"),
    };
    let expected: Vec<f64> = (0..=cfg.n_train_steps).map(|i| i as f64).collect();
    assert_eq!(steps, expected);
}

#[test]
fn trajectory_lift_numeric_values_match_source_report() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::ChaoticSpike, 12)
        .with_n_train_steps(4)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = disagreement_trajectory_to_dataframe(&report);
    let gap_col = match df.columns.iter().find(|(n, _)| n == "absolute_gap") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("absolute_gap column missing"),
    };
    let ssm_loss_col = match df.columns.iter().find(|(n, _)| n == "ssm_loss") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("ssm_loss column missing"),
    };
    for (i, dis) in report.disagreement_trajectory.iter().enumerate() {
        assert_eq!(
            gap_col[i].to_bits(),
            dis.absolute_gap.to_bits(),
            "absolute_gap mismatch at step {}",
            i
        );
    }
    for (i, loss) in report.training_trajectory.ssm_losses.iter().enumerate() {
        assert_eq!(ssm_loss_col[i].to_bits(), loss.to_bits());
    }
}

#[test]
fn trajectory_lift_seed_idx_zero_for_single_experiment() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 10)
        .with_n_train_steps(3)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df = disagreement_trajectory_to_dataframe(&report);
    let seed_idx = match df.columns.iter().find(|(n, _)| n == "seed_idx") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("seed_idx column missing"),
    };
    for v in seed_idx {
        assert_eq!(v.to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn sweep_trajectory_lift_row_count_is_cells_times_seeds_times_steps() {
    let base = SweepBaseConfig::new(4, 1, 1, 10, 4)
        .with_lambda_disagreement(0.1)
        .with_n_seeds(2);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_disagreement_trajectory_to_dataframe(&report);
    // 15 cells × 2 seeds × (4 + 1) steps = 150 rows.
    let expected = 15 * 2 * (base.n_train_steps + 1);
    for (_, col) in &df.columns {
        let len = match col {
            Column::Str(v) => v.len(),
            Column::Float(v) => v.len(),
            _ => 0,
        };
        assert_eq!(len, expected, "expected {} rows", expected);
    }
}

#[test]
fn sweep_trajectory_lift_canonical_order_outer_inner() {
    let base = SweepBaseConfig::new(4, 1, 1, 10, 3)
        .with_lambda_disagreement(0.1)
        .with_n_seeds(2);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let df = sweep_disagreement_trajectory_to_dataframe(&report);
    let datasets = match df.columns.iter().find(|(n, _)| n == "dataset") {
        Some((_, Column::Str(v))) => v.clone(),
        _ => panic!("dataset col missing"),
    };
    let modes = match df.columns.iter().find(|(n, _)| n == "mode") {
        Some((_, Column::Str(v))) => v.clone(),
        _ => panic!("mode col missing"),
    };
    let seed_idxs = match df.columns.iter().find(|(n, _)| n == "seed_idx") {
        Some((_, Column::Float(v))) => v.clone(),
        _ => panic!("seed_idx col missing"),
    };
    let steps_per_seed = base.n_train_steps + 1;
    // First chunk of `steps_per_seed` rows: (smooth_sine, symmetric, seed=0)
    for i in 0..steps_per_seed {
        assert_eq!(datasets[i], "smooth_sine");
        assert_eq!(modes[i], "symmetric");
        assert_eq!(seed_idxs[i].to_bits(), 0.0_f64.to_bits());
    }
    // Second chunk: (smooth_sine, symmetric, seed=1)
    for i in steps_per_seed..2 * steps_per_seed {
        assert_eq!(datasets[i], "smooth_sine");
        assert_eq!(modes[i], "symmetric");
        assert_eq!(seed_idxs[i].to_bits(), 1.0_f64.to_bits());
    }
    // Third chunk: (smooth_sine, ssm_as_generator, seed=0) — mode inner.
    for i in 2 * steps_per_seed..3 * steps_per_seed {
        assert_eq!(datasets[i], "smooth_sine");
        assert_eq!(modes[i], "ssm_as_generator");
        assert_eq!(seed_idxs[i].to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn trajectory_lifts_are_deterministic_across_runs() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 12)
        .with_n_train_steps(5)
        .with_lr(1e-2);
    let r1 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let r2 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let df1 = disagreement_trajectory_to_dataframe(&r1);
    let df2 = disagreement_trajectory_to_dataframe(&r2);
    // Compare every numeric column bit-by-bit.
    for ((n1, c1), (n2, c2)) in df1.columns.iter().zip(df2.columns.iter()) {
        assert_eq!(n1, n2);
        if let (Column::Float(v1), Column::Float(v2)) = (c1, c2) {
            for (a, b) in v1.iter().zip(v2.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "trajectory column `{}` drifted across runs",
                    n1
                );
            }
        }
    }
}

// ─── § F Bundle helper + malformed-DataFrame safety (carried from Phase 6.1) ─

#[test]
fn detectors_silently_no_op_on_malformed_dataframe() {
    // A DataFrame missing the required columns must not panic the
    // detectors — they should silently emit nothing. This protects
    // wiring through `validate(df, opts)` where the caller may have
    // attached the Cronos detectors to a non-Cronos DataFrame.
    let df = DataFrame::from_columns(vec![(
        "irrelevant".to_string(),
        Column::Float(vec![1.0, 2.0, 3.0]),
    )])
    .unwrap();
    let detectors = cronos_default_detectors();
    for d in &detectors {
        let mut sink = FindingSink::new(d.code(), d.belief_axes());
        d.run(&df, &mut sink);
        let (findings, _) = sink.drain();
        // No findings, no panic.
        assert_eq!(
            findings.len(),
            0,
            "detector {} should no-op on malformed DataFrame",
            d.code()
        );
    }
}
