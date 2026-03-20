//! PINN Demo: 1D Harmonic Oscillator u_xx + u = 0
//!
//! Run with: cargo run -p cjc-ad --example pinn_harmonic
//!
//! Demonstrates a Physics-Informed Neural Network solving
//! the harmonic oscillator PDE using an MLP with tanh activations
//! and reverse-mode autodiff for gradient computation.

use cjc_ad::pinn::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CJC SciML Demo — PINN: Harmonic Oscillator (u_xx + u = 0) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("PDE:    u_xx + u = 0  on [0, π]");
    println!("BCs:    u(0) = 0, u(π) = 0");
    println!("Exact:  u(x) = sin(x)");
    println!("Model:  MLP [1 → 32 → 32 → 1] with tanh activations");
    println!();

    // --- Configuration (hardened defaults) ---
    let config = PinnConfig {
        layer_sizes: vec![1, 32, 32, 1],
        epochs: 2000,
        lr: 1e-3,
        physics_weight: 1.0,
        boundary_weight: 50.0,
        seed: 42,
        n_collocation: 50,
        n_data: 30,
        fd_eps: 1e-3,
    };

    println!("Configuration:");
    println!("  Architecture:       {:?}", config.layer_sizes);
    println!("  Activation:         tanh (hidden), linear (output)");
    println!("  Epochs:             {}", config.epochs);
    println!("  Learning rate:      {}", config.lr);
    println!("  Optimizer:          Adam (β1=0.9, β2=0.999)");
    println!("  Physics weight:     {}", config.physics_weight);
    println!("  Boundary weight:    {}", config.boundary_weight);
    println!("  Collocation points: {}", config.n_collocation);
    println!("  Data points:        {}", config.n_data);
    println!("  FD epsilon:         {}", config.fd_eps);
    println!("  Seed:               {}", config.seed);
    println!();

    // --- Count parameters ---
    let n_params: usize = config.layer_sizes.windows(2)
        .map(|w| w[0] * w[1] + w[1])
        .sum();
    println!("  Total parameters:   {}", n_params);
    println!();

    // --- Train ---
    println!("Training PINN (this may take 30-60 seconds)...");
    let result = pinn_harmonic_train(&config);
    println!("Training complete.");
    println!();

    // --- Results ---
    let first = &result.history[0];
    let last = result.history.last().unwrap();
    println!("Training Results:");
    println!("  Initial loss:      {:.6}", first.total_loss);
    println!("  Final loss:        {:.6}", last.total_loss);
    if last.total_loss > 0.0 {
        println!("  Loss reduction:    {:.1}x", first.total_loss / last.total_loss);
    }
    println!("  Final data loss:   {:.6}", last.data_loss);
    println!("  Final physics loss:{:.6}", last.physics_loss);
    println!("  Final bnd loss:    {:.6}", last.boundary_loss);
    println!("  Final grad norm:   {:.6}", last.grad_norm);
    println!();

    // --- Accuracy ---
    println!("Accuracy vs Analytical Solution:");
    println!("  L2 error:          {:.6}", result.l2_error.unwrap_or(f64::NAN));
    println!("  Max error:         {:.6}", result.max_error.unwrap_or(f64::NAN));
    println!("  Mean PDE residual: {:.6}", result.mean_residual);
    println!();

    // --- Loss history (sampled) ---
    println!("Loss History (sampled):");
    println!("  {:>6} {:>12} {:>12} {:>12} {:>12}", "Epoch", "Total", "Data", "Physics", "Boundary");
    let step = (config.epochs / 10).max(1);
    for entry in result.history.iter().step_by(step) {
        println!(
            "  {:>6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            entry.epoch, entry.total_loss, entry.data_loss, entry.physics_loss, entry.boundary_loss,
        );
    }
    if let Some(last) = result.history.last() {
        println!(
            "  {:>6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            last.epoch, last.total_loss, last.data_loss, last.physics_loss, last.boundary_loss,
        );
    }
    println!();

    // --- Determinism check ---
    println!("Determinism Check (re-running with same seed)...");
    let result2 = pinn_harmonic_train(&config);
    let params_match = result.final_params.iter().zip(result2.final_params.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let loss_match = result.history.iter().zip(result2.history.iter())
        .all(|(a, b)| a.total_loss.to_bits() == b.total_loss.to_bits());
    println!("  Parameters bit-identical: {}", if params_match { "PASS ✓" } else { "FAIL ✗" });
    println!("  Loss trajectory identical: {}", if loss_match { "PASS ✓" } else { "FAIL ✗" });
    println!();

    // --- Loss curve ---
    println!("{}", plot_loss_history(&result.history, 60, 12));

    // --- Solution comparison ---
    let n_plot = 40;
    let x_plot: Vec<f64> = (0..n_plot)
        .map(|i| (i as f64 + 0.5) / n_plot as f64 * std::f64::consts::PI)
        .collect();
    let u_exact: Vec<f64> = x_plot.iter().map(|&x| x.sin()).collect();
    println!("{}", ascii_plot(&x_plot, &u_exact, 60, 12, "u_exact = sin(x)"));

    println!("Done.");
    println!();
    println!("Verification commands:");
    println!("  cargo test --test pinn_correctness");
    println!("  cargo test --test sciml_determinism");
    println!("  cargo test -p cjc-ad -- pinn");
}
