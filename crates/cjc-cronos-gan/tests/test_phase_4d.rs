//! Phase 4d integration tests:
//!
//! - § A LambdaSchedule public API math (5)
//! - § B Schedule integration with TemporalGanConfig + Trainer (3)
//! - § C Per-mode `lr` override (2)
//! - § D Multi-seed sweep shape + aggregation (5)
//! - § E Multi-seed determinism (3)
//! - § F format_table adaptation for multi-seed (3)
//! - § G Empirical-flip replication (4, `#[ignore]`'d — slow + load-bearing)
//!
//! Phase 4d is the upgrade documented in `docs/cronos-gan/PHASE_4D_HANDOFF.md`
//! (the handoff note this work was scoped from):
//!
//! - `TemporalGanConfig.lambda_disagreement: f64` → `lambda_schedule:
//!   LambdaSchedule`
//! - `SweepBaseConfig` gains `per_mode_lr`, `n_seeds`, `seed_stride`
//! - `SweepCell` carries `per_seed_reports: Vec<ExperimentReport>` +
//!   `mean` + `variance` aggregates
//! - `format_table` adapts to single-seed vs multi-seed display
//!
//! All `#[ignore]`'d tests are documented inline with the reason and
//! the command to run them.

use cjc_cronos_gan::{
    run_experiment_sweep, CellAggregate, CronosSeed, LambdaSchedule, SweepBaseConfig, TemporalGan,
    TemporalGanConfig, TemporalGanMode, TemporalGanTrainer, Trainable, DEFAULT_SEED_STRIDE,
    SWEEP_DATASETS,
};

fn sine_io(n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    let inputs: Vec<f64> = (0..n_steps).map(|t| (t as f64 * 0.4).sin()).collect();
    let targets: Vec<f64> = (0..n_steps).map(|t| ((t + 1) as f64 * 0.4).sin()).collect();
    (inputs, targets)
}

/// Tiny sweep config used by the multi-seed mechanics tests. The
/// numbers chosen here are *just* enough for `n_seeds > 1` × 15 cells
/// to finish in well under a second.
fn tiny_sweep() -> SweepBaseConfig {
    SweepBaseConfig::new(4, 1, 1, 10, 5).with_lambda_disagreement(0.1)
}

// ─── § A LambdaSchedule public API math (5) ─────────────────────────────

#[test]
fn schedule_constant_emits_same_value_at_every_step() {
    let s = LambdaSchedule::Constant(0.25);
    let samples = [0_u64, 1, 5, 100, 10_000];
    for step in samples {
        assert_eq!(s.lambda_at(step).to_bits(), 0.25_f64.to_bits());
    }
}

#[test]
fn schedule_linear_endpoints_match_start_and_end() {
    let s = LambdaSchedule::Linear {
        start: 0.4,
        end: 0.0,
        n_train_steps: 50,
    };
    assert_eq!(s.lambda_at(0).to_bits(), 0.4_f64.to_bits());
    assert_eq!(s.lambda_at(50).to_bits(), 0.0_f64.to_bits());
    // Midpoint
    assert!((s.lambda_at(25) - 0.2).abs() < 1e-12);
}

#[test]
fn schedule_exponential_decay_matches_closed_form_at_terminal() {
    let s = LambdaSchedule::ExponentialDecay {
        start: 1.0,
        decay_rate: 1.0, // exp(-1) at t = n_train_steps
        n_train_steps: 100,
    };
    let expected = (-1.0_f64).exp();
    assert!((s.lambda_at(100) - expected).abs() < 1e-12);
}

#[test]
fn schedule_warmup_then_linear_stays_at_start_during_warmup() {
    let s = LambdaSchedule::WarmupThenLinear {
        start: 0.1,
        end: 0.0,
        warmup_steps: 20,
        n_train_steps: 100,
    };
    for step in [0_u64, 5, 10, 15, 19] {
        assert_eq!(s.lambda_at(step).to_bits(), 0.1_f64.to_bits());
    }
    // At the warmup boundary, λ is still the warmup value (the linear
    // ramp *starts* here, the first ramp output uses (step - warmup) /
    // (n_train_steps - warmup) = 0/80 = 0.0 weight on `end`).
    assert_eq!(s.lambda_at(20).to_bits(), 0.1_f64.to_bits());
}

#[test]
fn schedule_lambda_at_is_finite_and_non_negative_for_all_well_formed_variants() {
    let schedules = [
        LambdaSchedule::Constant(0.0),
        LambdaSchedule::Constant(0.5),
        LambdaSchedule::Linear {
            start: 0.3,
            end: 0.0,
            n_train_steps: 200,
        },
        LambdaSchedule::ExponentialDecay {
            start: 0.5,
            decay_rate: 2.0,
            n_train_steps: 200,
        },
        LambdaSchedule::WarmupThenLinear {
            start: 0.1,
            end: 0.0,
            warmup_steps: 50,
            n_train_steps: 200,
        },
    ];
    for s in schedules {
        for step in 0_u64..=1000 {
            let v = s.lambda_at(step);
            assert!(v.is_finite(), "non-finite λ at step {} for {:?}", step, s);
            assert!(v >= 0.0, "negative λ ({}) at step {} for {:?}", v, step, s);
        }
    }
}

// ─── § B Schedule integration with TemporalGanConfig + Trainer (3) ──────

#[test]
fn linear_schedule_produces_different_final_params_than_constant() {
    // Same `(state_dim, input_dim, output_dim, mode, seed, inputs,
    // targets, n_train_steps, lr)` everywhere; only λ schedule
    // differs. After enough training, the two configs must reach
    // *different* final parameter states (their gradients differ on
    // the challenger update).
    let n_train_steps = 30;
    let lr = 1e-2;
    let (inputs, targets) = sine_io(15);

    let constant_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let linear_cfg = constant_cfg.with_lambda_schedule(LambdaSchedule::Linear {
        start: 0.5,
        end: 0.0,
        n_train_steps: n_train_steps as u64,
    });

    let train_run = |cfg: TemporalGanConfig| -> Vec<f64> {
        let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, lr);
        for _ in 0..n_train_steps {
            trainer.step(&mut gan, &inputs, &targets).unwrap();
        }
        gan.liquid().params_flat()
    };

    let params_const = train_run(constant_cfg);
    let params_linear = train_run(linear_cfg);

    // At least one parameter must differ. (Equality would mean the
    // schedule had no effect on training dynamics at all.)
    let any_diff = params_const
        .iter()
        .zip(params_linear.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        any_diff,
        "Liquid challenger params identical across Constant vs Linear λ — schedule had no effect"
    );
}

#[test]
fn same_schedule_same_seed_byte_identical_trajectory() {
    // Identical (schedule, seed, inputs, targets, n_train_steps) MUST
    // produce byte-identical training losses across runs.
    let n_train_steps = 20;
    let (inputs, targets) = sine_io(12);
    let schedule = LambdaSchedule::Linear {
        start: 0.2,
        end: 0.0,
        n_train_steps: n_train_steps as u64,
    };
    let cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.0)
        .with_lambda_schedule(schedule);

    let collect = |seed: CronosSeed| -> Vec<u64> {
        let mut gan = TemporalGan::from_seed(cfg, seed).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let mut losses = Vec::new();
        for _ in 0..n_train_steps {
            let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
            losses.push(step.ssm_loss.to_bits());
            losses.push(step.liquid_loss.to_bits());
        }
        losses
    };

    let a = collect(CronosSeed(42));
    let b = collect(CronosSeed(42));
    assert_eq!(a, b, "byte-identical replay broken for Linear schedule");
}

#[test]
fn constant_schedule_with_matching_scalar_matches_back_compat_setter() {
    // Two ways to set λ=0.15:
    //   (a) cfg.with_lambda_disagreement(0.15)            // back-compat shim
    //   (b) cfg.with_lambda_schedule(Constant(0.15))      // explicit Phase 4d
    // Should produce byte-identical training trajectories.
    let n_train_steps = 15;
    let (inputs, targets) = sine_io(10);

    let train = |cfg: TemporalGanConfig| -> Vec<u64> {
        let mut gan = TemporalGan::from_seed(cfg, CronosSeed(42)).unwrap();
        let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
        let mut losses = Vec::new();
        for _ in 0..n_train_steps {
            let step = trainer.step(&mut gan, &inputs, &targets).unwrap();
            losses.push(step.liquid_loss.to_bits());
        }
        losses
    };

    let via_shim = TemporalGanConfig::symmetric(4, 1, 1)
        .with_mode(TemporalGanMode::SsmAsGenerator)
        .with_lambda_disagreement(0.15);
    let via_explicit = TemporalGanConfig::symmetric(4, 1, 1)
        .with_mode(TemporalGanMode::SsmAsGenerator)
        .with_lambda_schedule(LambdaSchedule::Constant(0.15));

    assert_eq!(
        train(via_shim),
        train(via_explicit),
        "back-compat shim should produce byte-identical trajectory to explicit Constant"
    );
}

// ─── § C Per-mode `lr` override (2) ─────────────────────────────────────

#[test]
fn per_mode_lr_override_changes_only_overridden_modes_cells() {
    let base_a = tiny_sweep();
    let base_b = tiny_sweep().with_lr_for_mode(TemporalGanMode::LiquidAsGenerator, 5e-3);

    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();

    // LiquidAsGenerator cells differ across the runs.
    for ds in SWEEP_DATASETS {
        let liq_a = r1
            .cell(ds, TemporalGanMode::LiquidAsGenerator)
            .unwrap();
        let liq_b = r2
            .cell(ds, TemporalGanMode::LiquidAsGenerator)
            .unwrap();
        assert_ne!(
            liq_a.replay_hash, liq_b.replay_hash,
            "lr override should shift LiquidAsGenerator hash for {:?}",
            ds
        );
    }

    // SsmAsGenerator and Symmetric cells unchanged (no per-mode lr
    // override for them; they use default_lr).
    for ds in SWEEP_DATASETS {
        let ssm_a = r1.cell(ds, TemporalGanMode::SsmAsGenerator).unwrap();
        let ssm_b = r2.cell(ds, TemporalGanMode::SsmAsGenerator).unwrap();
        assert_eq!(
            ssm_a.replay_hash, ssm_b.replay_hash,
            "SsmAsGenerator should be unaffected by Liquid-only lr override for {:?}",
            ds
        );
        let sym_a = r1.cell(ds, TemporalGanMode::Symmetric).unwrap();
        let sym_b = r2.cell(ds, TemporalGanMode::Symmetric).unwrap();
        assert_eq!(
            sym_a.replay_hash, sym_b.replay_hash,
            "Symmetric should be unaffected for {:?}",
            ds
        );
    }
}

#[test]
fn lr_for_mode_falls_back_to_default_lr() {
    let base = tiny_sweep()
        .with_default_lr(2e-3)
        .with_lr_for_mode(TemporalGanMode::SsmAsGenerator, 7e-3);
    assert_eq!(base.lr_for(TemporalGanMode::SsmAsGenerator).to_bits(), 7e-3_f64.to_bits());
    // No override for these → fall back to default_lr
    assert_eq!(base.lr_for(TemporalGanMode::LiquidAsGenerator).to_bits(), 2e-3_f64.to_bits());
    assert_eq!(base.lr_for(TemporalGanMode::Symmetric).to_bits(), 2e-3_f64.to_bits());
}

// ─── § D Multi-seed sweep shape + aggregation (5) ───────────────────────

#[test]
fn n_seeds_three_produces_three_reports_per_cell() {
    let base = tiny_sweep().with_n_seeds(3);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    assert_eq!(report.cells.len(), 15);
    for c in &report.cells {
        assert_eq!(
            c.n_seeds(),
            3,
            "expected 3 per-seed reports in cell ({}, {})",
            c.dataset.label(),
            c.mode.label()
        );
        assert_eq!(c.per_seed_reports.len(), 3);
    }
}

#[test]
fn per_seed_replay_hashes_are_distinct_within_a_cell() {
    let base = tiny_sweep().with_n_seeds(4);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for c in &report.cells {
        let mut hashes: Vec<_> = c.per_seed_reports.iter().map(|r| r.replay_hash).collect();
        hashes.sort_by_key(|h| h.0 .0);
        hashes.dedup();
        assert_eq!(
            hashes.len(),
            4,
            "expected 4 distinct replay_hashes in cell ({}, {}); got duplicates",
            c.dataset.label(),
            c.mode.label()
        );
    }
}

#[test]
fn mean_matches_manual_kahan_sum_across_seeds() {
    let base = tiny_sweep().with_n_seeds(3);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for c in &report.cells {
        // Manual Kahan-sum of final_loss_ssm across seeds.
        let mut acc = 0.0_f64;
        let mut comp = 0.0_f64;
        for r in &c.per_seed_reports {
            let y = r.final_loss_ssm - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
        }
        let expected_mean = acc / c.per_seed_reports.len() as f64;
        assert!(
            (c.mean.final_loss_ssm - expected_mean).abs() < 1e-14,
            "mean.final_loss_ssm = {} vs manual Kahan mean = {} for cell ({}, {})",
            c.mean.final_loss_ssm,
            expected_mean,
            c.dataset.label(),
            c.mode.label()
        );
    }
}

#[test]
fn variance_is_non_negative_across_all_aggregated_fields() {
    let base = tiny_sweep().with_n_seeds(3).with_eval_steps(5);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for c in &report.cells {
        let v = &c.variance;
        assert!(v.final_loss_ssm >= 0.0 && v.final_loss_ssm.is_finite());
        assert!(v.final_loss_liquid >= 0.0 && v.final_loss_liquid.is_finite());
        assert!(v.mean_absolute_gap >= 0.0 && v.mean_absolute_gap.is_finite());
        assert!(v.max_regime_shift_score >= 0.0 && v.max_regime_shift_score.is_finite());
        if let Some(x) = v.eval_ssm_loss {
            assert!(x >= 0.0 && x.is_finite(), "eval_ssm_loss variance {}", x);
        }
        if let Some(x) = v.eval_liquid_loss {
            assert!(x >= 0.0 && x.is_finite());
        }
        if let Some(x) = v.eval_absolute_gap {
            assert!(x >= 0.0 && x.is_finite());
        }
    }
}

#[test]
fn aggregate_sweep_hash_byte_identical_across_runs_n_seeds_three() {
    let base = tiny_sweep().with_n_seeds(3);
    let r1 = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    assert_eq!(r1.sweep_hash, r2.sweep_hash);
    for (c1, c2) in r1.cells.iter().zip(r2.cells.iter()) {
        assert_eq!(c1.per_seed_reports.len(), c2.per_seed_reports.len());
        for (a, b) in c1.per_seed_reports.iter().zip(c2.per_seed_reports.iter()) {
            assert_eq!(a.replay_hash, b.replay_hash);
        }
        // And the aggregates themselves are byte-identical (Kahan +
        // Welford are deterministic functions of the inputs).
        assert_eq!(
            c1.mean.final_loss_ssm.to_bits(),
            c2.mean.final_loss_ssm.to_bits(),
            "mean drift for cell ({}, {})",
            c1.dataset.label(),
            c1.mode.label(),
        );
        assert_eq!(
            c1.variance.final_loss_ssm.to_bits(),
            c2.variance.final_loss_ssm.to_bits(),
        );
    }
}

// ─── § E Multi-seed determinism (3) ─────────────────────────────────────

#[test]
fn same_config_same_seed_same_stride_byte_identical_sweep_hash() {
    let base_a = tiny_sweep()
        .with_n_seeds(3)
        .with_seed_stride(DEFAULT_SEED_STRIDE);
    let base_b = tiny_sweep()
        .with_n_seeds(3)
        .with_seed_stride(DEFAULT_SEED_STRIDE);
    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();
    assert_eq!(r1.sweep_hash, r2.sweep_hash);
}

#[test]
fn changing_seed_stride_shifts_sweep_hash() {
    let base_a = tiny_sweep().with_n_seeds(3).with_seed_stride(DEFAULT_SEED_STRIDE);
    let base_b = tiny_sweep().with_n_seeds(3).with_seed_stride(1);
    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();
    assert_ne!(
        r1.sweep_hash, r2.sweep_hash,
        "different seed_stride must shift sweep_hash"
    );
}

#[test]
fn changing_n_seeds_shifts_sweep_hash() {
    let base_a = tiny_sweep().with_n_seeds(1);
    let base_b = tiny_sweep().with_n_seeds(2);
    let r1 = run_experiment_sweep(&base_a, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base_b, CronosSeed(42)).unwrap();
    assert_ne!(
        r1.sweep_hash, r2.sweep_hash,
        "different n_seeds must shift sweep_hash"
    );
}

// ─── § F format_table adaptation for multi-seed (3) ─────────────────────

#[test]
fn format_table_single_seed_uses_per_cell_rows() {
    let base = tiny_sweep().with_n_seeds(1);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let table = report.format_table();
    // The single-seed format does NOT include the "mean±std" header
    // marker.
    assert!(
        !table.contains("(mean±std)"),
        "single-seed format must not show '(mean±std)' header"
    );
    assert!(table.contains("ssm_loss"));
    assert!(report.is_multi_seed() == false);
}

#[test]
fn format_table_multi_seed_uses_mean_std_rows() {
    let base = tiny_sweep().with_n_seeds(3);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    assert!(report.is_multi_seed());
    let table = report.format_table();
    assert!(
        table.contains("(mean±std)"),
        "multi-seed format must show '(mean±std)' header"
    );
    // The ± symbol shows up in the rendered numeric cells.
    assert!(
        table.contains(" ± "),
        "multi-seed format must render mean ± std cells"
    );
}

#[test]
fn format_table_multi_seed_footer_reports_n_seeds() {
    let base = tiny_sweep().with_n_seeds(3);
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let table = report.format_table();
    assert!(
        table.contains("n_seeds    = 3"),
        "multi-seed footer must report n_seeds = 3; got:\n{}",
        table
    );
}

// ─── § G Phase 4c empirical-flip replication (4, #[ignore]'d) ───────────
//
// These are the LOAD-BEARING Phase 4d tests: they turn Phase 4c's
// single-seed finding (that `SsmAsGenerator` produces the only
// calibrated disagreement that transfers to held-out data) into a
// statistical claim across `n_seeds = 10`. They're marked `#[ignore]`
// because each run takes 10–30 seconds even in release mode, and
// failure here is informational (Phase 6 should not be built on top
// of an unstable empirical flip) rather than a hard correctness gate.
//
// Run with: `cargo test -p cjc-cronos-gan --release --test
// test_phase_4d -- --ignored`
//
// The configs used here are deliberately smaller than `examples/sweep.rs`
// to fit a CI-bounded budget while keeping enough training depth to
// reproduce the flip's qualitative direction. "Generous tolerance"
// (per the handoff) means we assert `mean(SsmAsGen) <=
// mean(comparator) * 1.10` — a 10% slack factor that accepts small
// statistical drift but flags substantial reversals.

fn empirical_flip_config() -> SweepBaseConfig {
    SweepBaseConfig::new(
        /* state_dim */ 6,
        /* input_dim */ 1,
        /* output_dim */ 1,
        /* n_steps */ 25,
        /* n_train_steps */ 60,
    )
    .with_eval_steps(12)
    .with_lambda_for(TemporalGanMode::SsmAsGenerator, 0.10)
    .with_lambda_for(TemporalGanMode::LiquidAsGenerator, 0.15)
    .with_default_lr(1e-2)
    .with_n_seeds(10)
}

/// Mean of `f(cell)` across all 5 datasets, fixing `mode`.
fn mean_across_datasets<F>(
    report: &cjc_cronos_gan::ExperimentSweepReport,
    mode: TemporalGanMode,
    f: F,
) -> f64
where
    F: Fn(&CellAggregate) -> Option<f64>,
{
    let mut sum = 0.0_f64;
    let mut n = 0_usize;
    for ds in SWEEP_DATASETS {
        if let Some(cell) = report.cell_full(ds, mode) {
            if let Some(v) = f(&cell.mean) {
                sum += v;
                n += 1;
            }
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

#[test]
#[ignore = "slow (10×15 experiments); load-bearing for Phase 6 decision"]
fn ssm_as_generator_eval_ssm_loss_lower_than_liquid_as_generator() {
    let base = empirical_flip_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let m_ssm = mean_across_datasets(
        &report,
        TemporalGanMode::SsmAsGenerator,
        |a| a.eval_ssm_loss,
    );
    let m_liq = mean_across_datasets(
        &report,
        TemporalGanMode::LiquidAsGenerator,
        |a| a.eval_ssm_loss,
    );
    eprintln!(
        "eval_ssm_loss mean (across 5 datasets, n_seeds=10):\n  SsmAsGenerator     = {:.4e}\n  LiquidAsGenerator  = {:.4e}\n  ratio (ssm/liq)    = {:.4}",
        m_ssm,
        m_liq,
        m_ssm / m_liq
    );
    assert!(
        m_ssm <= m_liq * 1.10,
        "Phase 4c flip reversed: eval_ssm_loss(SsmAsGen)={} > 1.10 × eval_ssm_loss(LiqAsGen)={}",
        m_ssm,
        m_liq
    );
}

#[test]
#[ignore = "slow (10×15 experiments); load-bearing for Phase 6 decision"]
fn ssm_as_generator_eval_gap_lower_than_symmetric() {
    let base = empirical_flip_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let m_ssm = mean_across_datasets(
        &report,
        TemporalGanMode::SsmAsGenerator,
        |a| a.eval_absolute_gap,
    );
    let m_sym = mean_across_datasets(
        &report,
        TemporalGanMode::Symmetric,
        |a| a.eval_absolute_gap,
    );
    eprintln!(
        "eval_absolute_gap mean (across 5 datasets, n_seeds=10):\n  SsmAsGenerator     = {:.4e}\n  Symmetric          = {:.4e}\n  ratio (ssm/sym)    = {:.4}",
        m_ssm, m_sym, m_ssm / m_sym
    );
    assert!(
        m_ssm <= m_sym * 1.10,
        "Phase 4c gap claim reversed: eval_gap(SsmAsGen)={} > 1.10 × eval_gap(Sym)={}",
        m_ssm,
        m_sym
    );
}

#[test]
#[ignore = "slow (10×15 experiments); load-bearing for Phase 6 decision"]
fn ssm_as_generator_eval_gap_lower_than_liquid_as_generator() {
    let base = empirical_flip_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let m_ssm = mean_across_datasets(
        &report,
        TemporalGanMode::SsmAsGenerator,
        |a| a.eval_absolute_gap,
    );
    let m_liq = mean_across_datasets(
        &report,
        TemporalGanMode::LiquidAsGenerator,
        |a| a.eval_absolute_gap,
    );
    eprintln!(
        "eval_absolute_gap mean (across 5 datasets, n_seeds=10):\n  SsmAsGenerator     = {:.4e}\n  LiquidAsGenerator  = {:.4e}\n  ratio (ssm/liq)    = {:.4}",
        m_ssm, m_liq, m_ssm / m_liq
    );
    assert!(
        m_ssm <= m_liq * 1.10,
        "Phase 4c gap claim reversed: eval_gap(SsmAsGen)={} > 1.10 × eval_gap(LiqAsGen)={}",
        m_ssm,
        m_liq
    );
}

#[test]
#[ignore = "slow (10×15 experiments); load-bearing for Phase 6 decision"]
fn multi_seed_aggregate_finite_and_well_formed_at_full_config() {
    // Sanity check on the same config the headline-flip tests above
    // use: every cell has the expected n_seeds reports, every mean
    // and variance is finite, and the per-seed replay_hashes within
    // each cell are distinct (no accidental seed-stride collisions
    // at this scale).
    let base = empirical_flip_config();
    let report = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    for c in &report.cells {
        assert_eq!(c.n_seeds(), 10);
        assert!(c.mean.final_loss_ssm.is_finite());
        assert!(c.mean.final_loss_liquid.is_finite());
        assert!(c.variance.final_loss_ssm >= 0.0);
        assert!(c.variance.final_loss_liquid >= 0.0);
        let mut hashes: Vec<_> = c.per_seed_reports.iter().map(|r| r.replay_hash).collect();
        hashes.sort_by_key(|h| h.0 .0);
        hashes.dedup();
        assert_eq!(
            hashes.len(),
            10,
            "seed-stride collision in cell ({}, {})",
            c.dataset.label(),
            c.mode.label()
        );
    }
}
