//! Phase 4b sweep deliverable.
//!
//! Single `cargo run --example sweep --release` produces the 15-cell
//! `(dataset, mode)` → `(final_ssm_loss, final_liquid_loss,
//! mean_disagreement, max_regime_shift_score, replay_hash)` table that
//! makes the question "does asymmetric mode actually find different
//! solutions than symmetric mode?" empirically answerable.
//!
//! Defaults:
//! - `state_dim = 8`, `input_dim = 1`, `output_dim = 1`
//! - `n_steps = 50` per dataset
//! - `n_train_steps = 200`
//! - `lambda_disagreement = 0.1`
//! - `default_lr = 1e-2`, with per-dataset overrides for the noisier
//!   datasets that benefit from a slower learning rate.
//! - `seed = 42`
//!
//! All defaults are overridable on the command line via the standard
//! `cargo run` envvars are *not* wired — this is a "single cargo run"
//! deliverable, deliberately frictionless. Customise by editing this
//! file or copying it.

use cjc_cronos_gan::{
    run_experiment_sweep, CronosDataset, CronosGanError, CronosSeed, SweepBaseConfig,
};

fn main() -> Result<(), CronosGanError> {
    eprintln!("Running 5×3 Cronos GAN sweep (release profile recommended)...");

    let base = SweepBaseConfig::new(
        /* state_dim */ 8,
        /* input_dim */ 1,
        /* output_dim */ 1,
        /* n_steps */ 50,
        /* n_train_steps */ 200,
    )
    .with_lambda_disagreement(0.1)
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

    // A quick per-mode summary: mean of mean_absolute_gap across datasets,
    // by mode. Surfaces the headline question without forcing the user to
    // eyeball 15 rows.
    println!();
    println!("Per-mode mean of (mean |gap|) across 5 datasets:");
    for &mode in cjc_cronos_gan::SWEEP_MODES.iter() {
        let mut sum = 0.0_f64;
        let mut count = 0;
        for cell in report.cells.iter().filter(|c| c.mode == mode) {
            sum += cell.report.mean_absolute_gap;
            count += 1;
        }
        if count > 0 {
            println!(
                "  {:<20}  {:.4e}",
                mode.label(),
                sum / count as f64
            );
        }
    }

    Ok(())
}
