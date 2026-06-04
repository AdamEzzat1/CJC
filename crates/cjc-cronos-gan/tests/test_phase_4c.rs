//! Phase 4c integration tests:
//!
//! 1. Eval-portion determinism — same `(config, seed)` → byte-identical
//!    `ExperimentReport.eval` (Option-equality on the bit patterns).
//! 2. `eval_steps = 0` → `report.eval == None` and shape matches Phase
//!    4b's report exactly (backward compat).
//! 3. `eval_steps > 0` → `report.eval == Some(EvalReport { … })` with
//!    finite losses + non-negative disagreement.
//! 4. Held-out eval is *not* training data — the SSM seen during eval
//!    is the post-training SSM, and its predictions differ from what
//!    they would be with random init.
//! 5. Per-mode λ in `SweepBaseConfig` is honored — different λ for
//!    `SsmAsGenerator` vs `LiquidAsGenerator` produces different
//!    cells.
//! 6. `with_lambda_for(mode, λ)` shows up in the per-cell
//!    `ExperimentConfig.gan.lambda_schedule` (Phase 4d wraps scalar
//!    λ into `LambdaSchedule::Constant`).
//! 7. `lambda_for(mode)` falls back to `lambda_disagreement` when no
//!    override.
//! 8. Sweep with eval shifts `sweep_hash` away from sweep without eval
//!    (eval bytes enter the per-cell `replay_hash`).
//! 9. `format_table` adapts: shows eval columns iff at least one cell
//!    has `Some(eval)`.

use cjc_cronos_gan::{
    run_experiment, run_experiment_sweep, CronosDataset, CronosSeed, ExperimentConfig,
    LambdaSchedule, SweepBaseConfig, TemporalGanConfig, TemporalGanMode,
};

fn sweep_small_no_eval() -> SweepBaseConfig {
    SweepBaseConfig::new(4, 1, 1, 15, 15).with_lambda_disagreement(0.1)
}

fn sweep_small_with_eval() -> SweepBaseConfig {
    sweep_small_no_eval().with_eval_steps(8)
}

// ─── § 1: Eval determinism ───────────────────────────────────────────────

#[test]
fn eval_byte_identical_across_runs() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 20)
        .with_n_train_steps(20)
        .with_eval_steps(10)
        .with_lr(1e-2);
    let r1 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let r2 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let e1 = r1.eval.unwrap();
    let e2 = r2.eval.unwrap();
    assert_eq!(e1.ssm_loss.to_bits(), e2.ssm_loss.to_bits());
    assert_eq!(e1.liquid_loss.to_bits(), e2.liquid_loss.to_bits());
    assert_eq!(
        e1.disagreement.absolute_gap.to_bits(),
        e2.disagreement.absolute_gap.to_bits()
    );
    assert_eq!(
        e1.disagreement.regime_shift_score.to_bits(),
        e2.disagreement.regime_shift_score.to_bits()
    );
}

// ─── § 2: eval_steps = 0 ⇒ eval = None (Phase 4b backward compat) ────────

#[test]
fn eval_none_when_eval_steps_zero() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 15)
        .with_n_train_steps(10)
        .with_lr(1e-2);
    // eval_steps defaults to 0
    assert_eq!(cfg.eval_steps, 0);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    assert!(report.eval.is_none());
}

// ─── § 3: eval_steps > 0 ⇒ populated EvalReport ─────────────────────────

#[test]
fn eval_populated_when_eval_steps_positive() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 15)
        .with_n_train_steps(20)
        .with_eval_steps(5)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let e = report.eval.expect("eval should be populated");
    assert!(e.ssm_loss.is_finite() && e.ssm_loss >= 0.0);
    assert!(e.liquid_loss.is_finite() && e.liquid_loss >= 0.0);
    assert!(e.disagreement.absolute_gap >= 0.0);
}

// ─── § 4: held-out eval reflects training, not random init ──────────────

#[test]
fn eval_ssm_loss_changes_with_n_train_steps() {
    // 0 train steps → SSM is at init. 50 train steps → SSM trained.
    // Held-out eval SSM MSE should be DIFFERENT (typically lower for
    // trained, but the test just asserts it's not bit-identical, which
    // would require the SSM had not been updated).
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg_untrained = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 15)
        .with_n_train_steps(0)
        .with_eval_steps(5)
        .with_lr(1e-2);
    let cfg_trained = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 15)
        .with_n_train_steps(50)
        .with_eval_steps(5)
        .with_lr(1e-2);
    let r_u = run_experiment(&cfg_untrained, CronosSeed(42)).unwrap();
    let r_t = run_experiment(&cfg_trained, CronosSeed(42)).unwrap();
    let ssm_u = r_u.eval.unwrap().ssm_loss;
    let ssm_t = r_t.eval.unwrap().ssm_loss;
    assert_ne!(
        ssm_u.to_bits(),
        ssm_t.to_bits(),
        "eval SSM MSE should differ between 0-step and 50-step training (got {} vs {})",
        ssm_u, ssm_t
    );
}

// ─── § 5: Per-mode λ ─────────────────────────────────────────────────────

#[test]
fn per_mode_lambda_override_changes_challenger_cell() {
    let base_a = sweep_small_no_eval()
        .with_lambda_for(TemporalGanMode::SsmAsGenerator, 0.1)
        .with_lambda_for(TemporalGanMode::LiquidAsGenerator, 0.1);
    let base_b = sweep_small_no_eval()
        .with_lambda_for(TemporalGanMode::SsmAsGenerator, 0.1)
        .with_lambda_for(TemporalGanMode::LiquidAsGenerator, 0.3); // different
    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();
    // SsmAsGenerator cells unchanged (same λ)
    for ds in cjc_cronos_gan::SWEEP_DATASETS {
        let a = r1.cell(ds, TemporalGanMode::SsmAsGenerator).unwrap();
        let b = r2.cell(ds, TemporalGanMode::SsmAsGenerator).unwrap();
        assert_eq!(a.replay_hash, b.replay_hash, "SsmAsGen unchanged for {:?}", ds);
    }
    // LiquidAsGenerator cells differ (different λ)
    for ds in cjc_cronos_gan::SWEEP_DATASETS {
        let a = r1.cell(ds, TemporalGanMode::LiquidAsGenerator).unwrap();
        let b = r2.cell(ds, TemporalGanMode::LiquidAsGenerator).unwrap();
        assert_ne!(a.replay_hash, b.replay_hash, "LiquidAsGen changed for {:?}", ds);
    }
}

#[test]
fn per_mode_lambda_propagates_into_experiment_config() {
    let base = sweep_small_no_eval()
        .with_lambda_for(TemporalGanMode::SsmAsGenerator, 0.25)
        .with_lambda_for(TemporalGanMode::LiquidAsGenerator, 0.75);
    let cfg_ssm = base.experiment_config_for(
        CronosDataset::SmoothSine,
        TemporalGanMode::SsmAsGenerator,
    );
    let cfg_liq = base.experiment_config_for(
        CronosDataset::SmoothSine,
        TemporalGanMode::LiquidAsGenerator,
    );
    // Phase 4d: `lambda_disagreement: f64` was promoted to
    // `lambda_schedule: LambdaSchedule`. The back-compat shim
    // `with_lambda_for(_, f64)` wraps into Constant.
    assert_eq!(cfg_ssm.gan.lambda_schedule, LambdaSchedule::Constant(0.25));
    assert_eq!(cfg_liq.gan.lambda_schedule, LambdaSchedule::Constant(0.75));
}

// ─── § 6: lambda_for fallback ─────────────────────────────────────────────

#[test]
fn lambda_for_falls_back_to_lambda_disagreement_when_no_override() {
    let base = sweep_small_no_eval().with_lambda_disagreement(0.42);
    // Phase 4d: `lambda_for` now returns `LambdaSchedule`. The
    // back-compat shim `with_lambda_disagreement(0.42)` wraps in
    // `Constant`, so both modes should see `Constant(0.42)`.
    assert_eq!(
        base.lambda_for(TemporalGanMode::SsmAsGenerator),
        LambdaSchedule::Constant(0.42)
    );
    assert_eq!(
        base.lambda_for(TemporalGanMode::LiquidAsGenerator),
        LambdaSchedule::Constant(0.42)
    );
}

#[test]
fn lambda_for_uses_override_when_present() {
    let base = sweep_small_no_eval()
        .with_lambda_disagreement(0.42)
        .with_lambda_for(TemporalGanMode::SsmAsGenerator, 0.99);
    assert_eq!(
        base.lambda_for(TemporalGanMode::SsmAsGenerator),
        LambdaSchedule::Constant(0.99)
    );
    // Non-overridden mode falls back to fallback.
    assert_eq!(
        base.lambda_for(TemporalGanMode::LiquidAsGenerator),
        LambdaSchedule::Constant(0.42)
    );
}

// ─── § 7: eval-bit-inclusion in replay_hash + sweep_hash ────────────────

#[test]
fn sweep_with_eval_diverges_from_sweep_without_eval() {
    let base_no_eval = sweep_small_no_eval();
    let base_with_eval = sweep_small_with_eval();
    let r_no = run_experiment_sweep(&base_no_eval, CronosSeed(42)).unwrap();
    let r_with = run_experiment_sweep(&base_with_eval, CronosSeed(42)).unwrap();
    assert_ne!(r_no.sweep_hash, r_with.sweep_hash);
}

#[test]
fn report_replay_hash_changes_with_eval_steps() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg_no = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 15)
        .with_n_train_steps(10)
        .with_lr(1e-2);
    let cfg_with = cfg_no.clone().with_eval_steps(5);
    let r_no = run_experiment(&cfg_no, CronosSeed(42)).unwrap();
    let r_with = run_experiment(&cfg_with, CronosSeed(42)).unwrap();
    assert_ne!(r_no.replay_hash, r_with.replay_hash);
}

// ─── § 8: format_table adapts ────────────────────────────────────────────

#[test]
fn format_table_omits_eval_columns_when_no_cell_has_eval() {
    let base = sweep_small_no_eval();
    let r = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let table = r.format_table();
    assert!(
        !table.contains("eval ssm"),
        "table should NOT contain eval columns when eval_steps=0"
    );
    assert!(
        !table.contains("eval |gap|"),
        "table should NOT contain eval gap column when eval_steps=0"
    );
}

#[test]
fn format_table_includes_eval_columns_when_eval_steps_positive() {
    let base = sweep_small_with_eval();
    let r = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let table = r.format_table();
    assert!(
        table.contains("eval ssm"),
        "table should contain 'eval ssm' header column"
    );
    assert!(
        table.contains("eval liq"),
        "table should contain 'eval liq' header column"
    );
    assert!(
        table.contains("eval |gap|"),
        "table should contain 'eval |gap|' header column"
    );
}

// ─── § 9: Phase 3b invariants survive Phase 4c (defense in depth) ───────

#[test]
fn ssm_loss_in_ssm_as_generator_still_equals_symmetric_with_eval() {
    // Phase 3b invariant: the SSM trains identically as predictor in both
    // Symmetric and SsmAsGenerator modes. Phase 4c's eval-mode shouldn't
    // disturb this — the eval rollout uses the SSM's POST-training state,
    // and the SSM's training trajectory is independent of mode here.
    let base = sweep_small_with_eval();
    let r = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for ds in cjc_cronos_gan::SWEEP_DATASETS {
        let sym = r.cell(ds, TemporalGanMode::Symmetric).unwrap();
        let ssm_g = r.cell(ds, TemporalGanMode::SsmAsGenerator).unwrap();
        assert_eq!(
            sym.final_loss_ssm.to_bits(),
            ssm_g.final_loss_ssm.to_bits(),
            "Phase 3b SSM-predictor invariant broken at {:?}",
            ds
        );
        // And the eval SSM MSE is identical too — same trained SSM, same
        // eval data.
        let sym_eval = sym.eval.unwrap();
        let ssm_g_eval = ssm_g.eval.unwrap();
        assert_eq!(
            sym_eval.ssm_loss.to_bits(),
            ssm_g_eval.ssm_loss.to_bits(),
            "Phase 4c eval invariant broken at {:?}: eval SSM MSE differs ({} vs {})",
            ds, sym_eval.ssm_loss, ssm_g_eval.ssm_loss
        );
    }
}

#[test]
fn liquid_loss_in_liquid_as_generator_still_equals_symmetric_with_eval() {
    let base = sweep_small_with_eval();
    let r = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for ds in cjc_cronos_gan::SWEEP_DATASETS {
        let sym = r.cell(ds, TemporalGanMode::Symmetric).unwrap();
        let liq_g = r.cell(ds, TemporalGanMode::LiquidAsGenerator).unwrap();
        assert_eq!(
            sym.final_loss_liquid.to_bits(),
            liq_g.final_loss_liquid.to_bits(),
            "Phase 3b Liquid-predictor invariant broken at {:?}",
            ds
        );
        let sym_eval = sym.eval.unwrap();
        let liq_g_eval = liq_g.eval.unwrap();
        assert_eq!(
            sym_eval.liquid_loss.to_bits(),
            liq_g_eval.liquid_loss.to_bits(),
            "Phase 4c eval invariant broken at {:?}: eval Liquid MSE differs",
            ds
        );
    }
}
