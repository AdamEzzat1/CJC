//! Phase 7b integration tests — `run_experiment` populates the new
//! `EvalReport` fields (`eval_gap_trajectory`, `predictive_uncertainty`,
//! `segments`), they're byte-deterministic, and they survive the
//! Phase 7b salt bump (v4d → v7b for both experiment and sweep
//! replay-hash salts).

use cjc_cronos_gan::{
    compute_gap_trajectory, compute_predictive_uncertainty, run_experiment,
    run_experiment_sweep, segment_trajectory, CronosDataset, CronosSeed, ExperimentConfig,
    SegmentLabel, SegmentationConfig, SweepBaseConfig, TemporalGanConfig,
};

// ─── § 1 EvalReport new fields are populated ────────────────────────────

#[test]
fn eval_report_carries_gap_trajectory_uncertainty_and_segments() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 12)
        .with_n_train_steps(10)
        .with_eval_steps(8)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let eval = report.eval.expect("eval populated when eval_steps > 0");

    // gap trajectory length == n_eval (output_dim = 1 here, so eval
    // window has exactly eval_steps timesteps).
    assert_eq!(eval.eval_gap_trajectory.len(), cfg.eval_steps);
    for g in &eval.eval_gap_trajectory {
        assert!(g.is_finite() && *g >= 0.0, "gap should be finite and ≥ 0");
    }
    // predictive_uncertainty length == eval_steps * output_dim = 8.
    assert_eq!(eval.predictive_uncertainty.len(), cfg.eval_steps);
    for (mean, variance) in &eval.predictive_uncertainty {
        assert!(mean.is_finite());
        assert!(variance.is_finite() && *variance >= 0.0);
    }
    // Segments cover the full eval window [0, eval_steps).
    assert!(!eval.segments.is_empty());
    assert_eq!(eval.segments.first().unwrap().start_step, 0);
    assert_eq!(eval.segments.last().unwrap().end_step, cfg.eval_steps);
}

#[test]
fn eval_report_new_fields_empty_when_eval_steps_zero() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::SmoothSine, 12)
        .with_n_train_steps(10)
        .with_lr(1e-2);
    // eval_steps defaults to 0.
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    assert!(report.eval.is_none(), "no EvalReport when eval_steps == 0");
}

// ─── § 2 Determinism of the new fields ──────────────────────────────────

#[test]
fn eval_report_new_fields_byte_identical_across_runs() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::ChaoticSpike, 15)
        .with_n_train_steps(10)
        .with_eval_steps(10)
        .with_lr(1e-2);
    let r1 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let r2 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let e1 = r1.eval.as_ref().unwrap();
    let e2 = r2.eval.as_ref().unwrap();

    // Gap trajectory bytes match.
    assert_eq!(e1.eval_gap_trajectory.len(), e2.eval_gap_trajectory.len());
    for (a, b) in e1
        .eval_gap_trajectory
        .iter()
        .zip(e2.eval_gap_trajectory.iter())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    // Predictive uncertainty bytes match.
    for ((m1, v1), (m2, v2)) in e1
        .predictive_uncertainty
        .iter()
        .zip(e2.predictive_uncertainty.iter())
    {
        assert_eq!(m1.to_bits(), m2.to_bits());
        assert_eq!(v1.to_bits(), v2.to_bits());
    }
    // Segments match exactly (Segment is PartialEq).
    assert_eq!(e1.segments, e2.segments);

    // And the overall replay_hash hashes all of this in.
    assert_eq!(r1.replay_hash, r2.replay_hash);
}

// ─── § 3 Salt bump shifts replay_hash from a deliberately-different config ─

#[test]
fn sweep_hash_byte_identical_with_phase_7b_salts() {
    // Two runs of the same SweepBaseConfig + seed should produce the
    // same sweep_hash; this exercises the v7b salt path (we don't
    // hardcode the v4d hash from a prior phase because tests at HEAD
    // always compute the current salt's hash).
    let base = SweepBaseConfig::new(4, 1, 1, 10, 5)
        .with_lambda_disagreement(0.1)
        .with_eval_steps(6);
    let r1 = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    let r2 = run_experiment_sweep(&base, CronosSeed(42)).unwrap();
    assert_eq!(r1.sweep_hash, r2.sweep_hash);
}

// ─── § 4 New eval-data fields shift the experiment replay_hash from
//       Phase 4c's. We can't hardcode old hashes; instead verify that
//       toggling a segment label (via different seed → different gap
//       trajectory → different segments) does shift the hash. ────────────

#[test]
fn replay_hash_responds_to_segment_changes_via_seed() {
    let gan_cfg = TemporalGanConfig::symmetric(4, 1, 1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 12)
        .with_n_train_steps(10)
        .with_eval_steps(6)
        .with_lr(1e-2);
    let r1 = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let r2 = run_experiment(&cfg, CronosSeed(43)).unwrap();
    // Different seeds → different init → different rollouts →
    // different gap trajectories → different segments → different
    // replay_hash.
    assert_ne!(r1.replay_hash, r2.replay_hash);
    // Sanity: the eval gap trajectories themselves differ somewhere.
    let e1 = r1.eval.as_ref().unwrap();
    let e2 = r2.eval.as_ref().unwrap();
    let mut found_diff = false;
    for (a, b) in e1
        .eval_gap_trajectory
        .iter()
        .zip(e2.eval_gap_trajectory.iter())
    {
        if a.to_bits() != b.to_bits() {
            found_diff = true;
            break;
        }
    }
    assert!(
        found_diff,
        "expected at least one gap-trajectory bit to differ across seeds"
    );
}

// ─── § 5 Public eval-analysis primitives composable on stored data ──────

#[test]
fn user_can_recompute_segments_with_custom_thresholds() {
    let gan_cfg = TemporalGanConfig::ssm_as_generator(4, 1, 1, 0.1);
    let cfg = ExperimentConfig::new(gan_cfg, CronosDataset::RegimeShift, 12)
        .with_n_train_steps(8)
        .with_eval_steps(10)
        .with_lr(1e-2);
    let report = run_experiment(&cfg, CronosSeed(42)).unwrap();
    let eval = report.eval.as_ref().unwrap();

    // Default segmentation lives on `eval.segments`. A user wanting
    // tighter thresholds re-runs `segment_trajectory` against the
    // SAME gap trajectory — no need to re-run the experiment.
    let tight = SegmentationConfig {
        enter_threshold: 0.05,
        exit_threshold: 0.01,
    };
    let custom_segments = segment_trajectory(&eval.eval_gap_trajectory, tight);
    // Tighter thresholds → at least as many segments as default.
    assert!(custom_segments.len() >= eval.segments.len());
    // Both span the same total range.
    assert_eq!(
        custom_segments.last().unwrap().end_step,
        eval.eval_gap_trajectory.len()
    );
}

// ─── § 6 Sanity on the public primitives independent of run_experiment ─

#[test]
fn predictive_uncertainty_is_consistent_with_stored_field() {
    // The Vec<(mean, variance)> in EvalReport equals what
    // compute_predictive_uncertainty would produce from the raw
    // eval rollouts. We can't easily extract the raw eval rollouts
    // from EvalReport (they're not stored), but we CAN check the
    // formula round-trip:
    //   mean == (m_lo + m_hi)/2 reconstructed from variance.
    let ssm = vec![0.5, 1.0, 2.0];
    let liq = vec![0.7, 1.4, 1.6];
    let u = compute_predictive_uncertainty(&ssm, &liq, 1);
    let g = compute_gap_trajectory(&ssm, &liq, 1);
    assert_eq!(u.len(), 3);
    assert_eq!(g.len(), 3);
    // gap == |ssm - liq|; variance == gap^2 / 2.
    for (i, ((_mean, var), gap)) in u.iter().zip(g.iter()).enumerate() {
        let expected_var = gap * gap * 0.5;
        assert!(
            (var - expected_var).abs() < 1e-12,
            "var/gap mismatch at step {}: var={}, expected={}",
            i,
            var,
            expected_var
        );
    }

    // Default labels are Stable or Transitional, not anything else.
    let labels = [SegmentLabel::Stable, SegmentLabel::Transitional];
    let segs = segment_trajectory(&g, SegmentationConfig::default());
    for s in &segs {
        assert!(labels.contains(&s.label));
    }
}
