//! Phase 4b integration tests for the experiment sweep:
//!
//! - 15-cell coverage (5 datasets × 3 modes)
//! - Canonical iteration order
//! - Per-dataset learning-rate override
//! - Byte-identical sweep_hash across runs
//! - `format_table` output well-formed and contains every cell
//! - Phase 4b refactored single-cell run_experiment delivers an
//!   ExperimentReport with `mode` and `disagreement_trajectory` populated
//! - Cells differ across modes (asymmetric mode actually changes something)

use cjc_cronos_gan::{
    run_experiment, run_experiment_sweep, CronosDataset, CronosSeed, ExperimentConfig,
    SweepBaseConfig, TemporalGanConfig, TemporalGanMode, SWEEP_DATASETS, SWEEP_MODES,
};

fn small_sweep_config() -> SweepBaseConfig {
    SweepBaseConfig::new(
        /* state_dim */ 4,
        /* input_dim */ 1,
        /* output_dim */ 1,
        /* n_steps */ 15,
        /* n_train_steps */ 15,
    )
    .with_lambda_disagreement(0.1)
    .with_default_lr(1e-2)
}

// ─── § 15-cell coverage ──────────────────────────────────────────────────

#[test]
fn sweep_yields_exactly_fifteen_cells_in_canonical_order() {
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    assert_eq!(report.cells.len(), 5 * 3, "expected 15 cells");

    // Verify canonical iteration order: datasets outer, modes inner.
    let mut idx = 0;
    for &ds in &SWEEP_DATASETS {
        for &mode in &SWEEP_MODES {
            let c = &report.cells[idx];
            assert_eq!(c.dataset, ds, "cell {} dataset mismatch", idx);
            assert_eq!(c.mode, mode, "cell {} mode mismatch", idx);
            assert_eq!(c.report.dataset_label, ds.label());
            assert_eq!(c.report.mode, mode);
            idx += 1;
        }
    }
}

#[test]
fn sweep_cell_accessor_finds_each_combination() {
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for &ds in &SWEEP_DATASETS {
        for &mode in &SWEEP_MODES {
            let r = report
                .cell(ds, mode)
                .unwrap_or_else(|| panic!("missing cell ({:?}, {:?})", ds, mode));
            assert_eq!(r.dataset_label, ds.label());
            assert_eq!(r.mode, mode);
        }
    }
}

#[test]
fn sweep_each_cell_has_finite_losses_and_finite_replay_hash_field() {
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for c in &report.cells {
        assert!(
            c.report.final_loss_ssm.is_finite(),
            "non-finite ssm_loss in cell ({}, {})",
            c.dataset.label(),
            c.mode.label()
        );
        assert!(c.report.final_loss_liquid.is_finite());
        assert!(c.report.mean_absolute_gap.is_finite());
        assert!(c.report.max_regime_shift_score.is_finite());
        assert!(c.report.mean_absolute_gap >= 0.0);
        assert!(c.report.max_regime_shift_score >= 0.0);
    }
}

// ─── § Determinism: byte-identical sweep_hash ────────────────────────────

#[test]
fn sweep_byte_identical_across_runs() {
    let base = small_sweep_config();
    let r1 = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    assert_eq!(r1.sweep_hash, r2.sweep_hash, "sweep_hash must replay");
    assert_eq!(r1.cells.len(), r2.cells.len());
    for (a, b) in r1.cells.iter().zip(r2.cells.iter()) {
        assert_eq!(a.report.replay_hash, b.report.replay_hash);
        assert_eq!(
            a.report.final_loss_ssm.to_bits(),
            b.report.final_loss_ssm.to_bits()
        );
        assert_eq!(
            a.report.final_loss_liquid.to_bits(),
            b.report.final_loss_liquid.to_bits()
        );
        assert_eq!(
            a.report.mean_absolute_gap.to_bits(),
            b.report.mean_absolute_gap.to_bits()
        );
        assert_eq!(
            a.report.max_regime_shift_score.to_bits(),
            b.report.max_regime_shift_score.to_bits()
        );
    }
}

#[test]
fn sweep_seed_change_diverges_sweep_hash() {
    let base = small_sweep_config();
    let r1 = run_experiment_sweep(&base, CronosSeed(1)).unwrap();
    let r2 = run_experiment_sweep(&base, CronosSeed(2)).unwrap();
    assert_ne!(r1.sweep_hash, r2.sweep_hash);
}

#[test]
fn sweep_config_change_diverges_sweep_hash() {
    let base_a = small_sweep_config().with_lambda_disagreement(0.1);
    let base_b = small_sweep_config().with_lambda_disagreement(0.2);
    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();
    assert_ne!(r1.sweep_hash, r2.sweep_hash);
}

// ─── § Per-dataset learning-rate override ───────────────────────────────

#[test]
fn per_dataset_lr_override_changes_replay_hash() {
    let base_a = small_sweep_config();
    let base_b = small_sweep_config().with_lr_for(CronosDataset::SmoothSine, 5e-3);
    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();
    // Only the smooth_sine cells change; the others stay the same.
    let ss_a = r1.cell(CronosDataset::SmoothSine, TemporalGanMode::Symmetric).unwrap();
    let ss_b = r2.cell(CronosDataset::SmoothSine, TemporalGanMode::Symmetric).unwrap();
    assert_ne!(ss_a.replay_hash, ss_b.replay_hash, "lr override must change replay");

    let other_a = r1.cell(CronosDataset::NoisySine, TemporalGanMode::Symmetric).unwrap();
    let other_b = r2.cell(CronosDataset::NoisySine, TemporalGanMode::Symmetric).unwrap();
    assert_eq!(
        other_a.replay_hash, other_b.replay_hash,
        "non-overridden dataset must replay unchanged"
    );

    // And the overall sweep_hash differs (because base_config bytes
    // include the per-dataset lr override).
    assert_ne!(r1.sweep_hash, r2.sweep_hash);
}

#[test]
fn experiment_config_for_uses_default_lr_when_no_override() {
    let base = small_sweep_config().with_default_lr(7.5e-3);
    let cfg = base.experiment_config_for(CronosDataset::NoisySine, TemporalGanMode::Symmetric);
    assert_eq!(cfg.lr.to_bits(), 7.5e-3_f64.to_bits());
}

#[test]
fn experiment_config_for_uses_override_when_set() {
    let base = small_sweep_config().with_lr_for(CronosDataset::NoisySine, 1.25e-3);
    let cfg = base.experiment_config_for(CronosDataset::NoisySine, TemporalGanMode::Symmetric);
    assert_eq!(cfg.lr.to_bits(), 1.25e-3_f64.to_bits());
    // Non-overridden datasets fall back to default_lr.
    let cfg_other = base.experiment_config_for(CronosDataset::SmoothSine, TemporalGanMode::Symmetric);
    assert_eq!(cfg_other.lr.to_bits(), base.default_lr.to_bits());
}

// ─── § Format table ──────────────────────────────────────────────────────

#[test]
fn format_table_contains_every_cell_label() {
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let table = report.format_table();
    for ds in SWEEP_DATASETS {
        assert!(
            table.contains(ds.label()),
            "table must mention dataset {}",
            ds.label()
        );
    }
    for mode in SWEEP_MODES {
        assert!(
            table.contains(mode.label()),
            "table must mention mode {}",
            mode.label()
        );
    }
    assert!(
        table.contains(&format!("sweep_hash = {}", report.sweep_hash)),
        "table must include sweep_hash footer"
    );
}

#[test]
fn format_table_has_15_data_rows() {
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let table = report.format_table();
    // Count cell rows: each starts with "│ " followed by a dataset label.
    let mut count = 0;
    for line in table.lines() {
        for ds in SWEEP_DATASETS {
            if line.contains(ds.label())
                && line.contains("│")
                // Not the header row (which contains "dataset")
                && !line.contains("dataset                ")
            {
                count += 1;
                break;
            }
        }
    }
    assert_eq!(count, 15, "expected 15 data rows in formatted table, got {}", count);
}

// ─── § Phase 4b ExperimentReport extensions ─────────────────────────────

#[test]
fn experiment_report_carries_mode_and_disagreement_trajectory() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 15)
        .with_n_train_steps(10)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    assert_eq!(report.mode, TemporalGanMode::SsmAsGenerator);
    // Trajectory length: 1 initial + n_train_steps = 11
    assert_eq!(report.disagreement_trajectory.len(), 11);
    assert_eq!(report.training_trajectory.ssm_losses.len(), 11);
    assert_eq!(report.training_trajectory.liquid_losses.len(), 11);
    // mean_absolute_gap finite + non-negative
    assert!(report.mean_absolute_gap.is_finite() && report.mean_absolute_gap >= 0.0);
    assert!(
        report.max_regime_shift_score.is_finite()
            && report.max_regime_shift_score >= 0.0
    );
}

#[test]
fn experiment_report_disagreement_trajectory_byte_identical_across_runs() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 12)
        .with_n_train_steps(8)
        .with_lr(1e-2);
    let r1 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let r2 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    for (a, b) in r1
        .disagreement_trajectory
        .iter()
        .zip(r2.disagreement_trajectory.iter())
    {
        assert_eq!(a.ssm_score.to_bits(), b.ssm_score.to_bits());
        assert_eq!(a.liquid_score.to_bits(), b.liquid_score.to_bits());
        assert_eq!(a.absolute_gap.to_bits(), b.absolute_gap.to_bits());
        assert_eq!(
            a.regime_shift_score.to_bits(),
            b.regime_shift_score.to_bits()
        );
    }
}

// ─── § Mode separation in the sweep ─────────────────────────────────────

#[test]
fn modes_produce_distinct_replay_hashes_per_dataset() {
    // For each dataset, the three modes should produce three distinct
    // replay hashes (one of the headline claims Phase 4b makes — the
    // modes actually find different solutions).
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for ds in SWEEP_DATASETS {
        let sym = report.cell(ds, TemporalGanMode::Symmetric).unwrap().replay_hash;
        let ssm_g = report
            .cell(ds, TemporalGanMode::SsmAsGenerator)
            .unwrap()
            .replay_hash;
        let liq_g = report
            .cell(ds, TemporalGanMode::LiquidAsGenerator)
            .unwrap()
            .replay_hash;
        assert_ne!(sym, ssm_g, "sym vs ssm_as_gen replay match on {:?}", ds);
        assert_ne!(sym, liq_g, "sym vs liq_as_gen replay match on {:?}", ds);
        assert_ne!(ssm_g, liq_g, "ssm vs liq generators replay match on {:?}", ds);
    }
}

#[test]
fn ssm_loss_in_ssm_as_generator_equals_ssm_loss_in_symmetric() {
    // In SsmAsGenerator mode the SSM is the predictor (vanilla supervised)
    // — so it sees IDENTICAL gradient updates to symmetric mode for the
    // SSM only. The Liquid loss may differ. (Phase 3b's predictor/
    // challenger framing in action at the experiment level.)
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for ds in SWEEP_DATASETS {
        let sym = report.cell(ds, TemporalGanMode::Symmetric).unwrap();
        let ssm_g = report.cell(ds, TemporalGanMode::SsmAsGenerator).unwrap();
        assert_eq!(
            sym.final_loss_ssm.to_bits(),
            ssm_g.final_loss_ssm.to_bits(),
            "SSM final loss must match between Symmetric and SsmAsGenerator for {:?} (SSM is the predictor in both)",
            ds
        );
    }
}

#[test]
fn liquid_loss_in_liquid_as_generator_equals_liquid_loss_in_symmetric() {
    // Mirror image: Liquid is the predictor in LiquidAsGenerator.
    let base = small_sweep_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for ds in SWEEP_DATASETS {
        let sym = report.cell(ds, TemporalGanMode::Symmetric).unwrap();
        let liq_g = report.cell(ds, TemporalGanMode::LiquidAsGenerator).unwrap();
        assert_eq!(
            sym.final_loss_liquid.to_bits(),
            liq_g.final_loss_liquid.to_bits(),
            "Liquid final loss must match between Symmetric and LiquidAsGenerator for {:?}",
            ds
        );
    }
}
