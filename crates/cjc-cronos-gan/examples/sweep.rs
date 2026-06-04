//! Phase 4b + Phase 4c sweep deliverable.
//!
//! Single `cargo run --example sweep --release` produces the 15-cell
//! `(dataset, mode)` table with BOTH the in-sample disagreement metrics
//! (Phase 4b) AND the held-out forecastability metrics (Phase 4c).
//!
//! Defaults:
//! - `state_dim = 8`, `input_dim = 1`, `output_dim = 1`
//! - `n_steps = 50` per dataset (training window)
//! - `eval_steps = 20` per dataset (held-out forecasting horizon)
//! - `n_train_steps = 200`
//! - Per-mode λ: `SsmAsGenerator = 0.1`, `LiquidAsGenerator = 0.15`
//!   (a deliberate asymmetry to probe Phase 4c's open question about
//!   whether the optimal λ differs between the asymmetric modes).
//! - `default_lr = 1e-2`, with per-dataset overrides for the noisier
//!   datasets that benefit from a slower learning rate.
//! - `seed = 42`
//!
//! Customise by editing this file or copying it.

use cjc_cronos_gan::{
    run_experiment_sweep, CronosDataset, CronosGanError, CronosSeed, SweepBaseConfig,
    TemporalGanMode,
};

fn main() -> Result<(), CronosGanError> {
    eprintln!("Running 5×3 Cronos GAN sweep (release profile, ~5-10s)...");

    let base = SweepBaseConfig::new(
        /* state_dim */ 8,
        /* input_dim */ 1,
        /* output_dim */ 1,
        /* n_steps */ 50,
        /* n_train_steps */ 200,
    )
    // Phase 4c: held-out forecasting horizon.
    .with_eval_steps(20)
    // Phase 4c: per-mode λ. The SsmAsGenerator default is the canonical
    // Phase 4b value; LiquidAsGenerator gets a slightly higher λ to
    // probe whether the SSM-challenger benefits more from extra
    // divergence pressure (the Phase 4b empirical finding suggests yes).
    .with_lambda_for(TemporalGanMode::SsmAsGenerator, 0.10)
    .with_lambda_for(TemporalGanMode::LiquidAsGenerator, 0.15)
    .with_default_lr(1e-2)
    // Noisier datasets benefit from a slower lr — convergence stays
    // monotonic instead of bouncing.
    .with_lr_for(CronosDataset::NoisySine, 5e-3)
    .with_lr_for(CronosDataset::RegimeShift, 5e-3)
    // Step-change and chaotic-spike have sharp gradient discontinuities
    // around the anomaly steps; smaller lr keeps Adam moments from
    // saturating.
    .with_lr_for(CronosDataset::StepChangeAnomaly, 3e-3)
    .with_lr_for(CronosDataset::ChaoticSpike, 3e-3);

    let report = run_experiment_sweep(&base, CronosSeed(42))?;

    println!("{}", report.format_table());

    // Per-mode summaries side-by-side: in-sample vs held-out gap.
    println!();
    println!("Per-mode means across 5 datasets:");
    println!(
        "  {:<20}  {:<12}  {:<12}  {:<12}",
        "mode", "train |gap|", "eval ssm MSE", "eval |gap|"
    );
    for &mode in cjc_cronos_gan::SWEEP_MODES.iter() {
        let mut sum_train_gap = 0.0_f64;
        let mut sum_eval_ssm = 0.0_f64;
        let mut sum_eval_gap = 0.0_f64;
        let mut count = 0;
        let mut eval_count = 0;
        for cell in report.cells.iter().filter(|c| c.mode == mode) {
            let r = cell.first_report();
            sum_train_gap += r.mean_absolute_gap;
            count += 1;
            if let Some(e) = &r.eval {
                sum_eval_ssm += e.ssm_loss;
                sum_eval_gap += e.disagreement.absolute_gap;
                eval_count += 1;
            }
        }
        if count > 0 {
            let train_gap = sum_train_gap / count as f64;
            let eval_ssm_str = if eval_count > 0 {
                format!("{:.4e}", sum_eval_ssm / eval_count as f64)
            } else {
                "—".to_string()
            };
            let eval_gap_str = if eval_count > 0 {
                format!("{:.4e}", sum_eval_gap / eval_count as f64)
            } else {
                "—".to_string()
            };
            println!(
                "  {:<20}  {:.4e}    {:<12}  {:<12}",
                mode.label(),
                train_gap,
                eval_ssm_str,
                eval_gap_str
            );
        }
    }

    Ok(())
}
