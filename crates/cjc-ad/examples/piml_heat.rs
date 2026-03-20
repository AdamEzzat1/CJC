//! PIML Demo: 1D Steady-State Heat Equation
//!
//! Run with: cargo run -p cjc-ad --example piml_heat
//!
//! Demonstrates Physics-Informed ML using polynomial regression
//! with physics residual penalty for u_xx = f(x) on [0, 1].

use cjc_ad::pinn::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CJC SciML Demo — PIML: 1D Steady-State Heat Equation      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("PDE:    u_xx = f(x) = -π² sin(πx)  on [0, 1]");
    println!("BCs:    u(0) = 0, u(1) = 0");
    println!("Exact:  u(x) = sin(πx)");
    println!("Model:  Polynomial degree 8 (9 parameters)");
    println!();

    // --- Configuration ---
    // Hardened: degree 6 prevents overfitting to noise, stronger physics penalty
    let degree = 6;
    let n_data = 40;
    let n_colloc = 60;
    let noise_std = 0.01;
    let epochs = 5000;
    let lr = 1e-3;
    let physics_weight = 5.0;
    let boundary_weight = 50.0;
    let seed = 42u64;

    println!("Configuration:");
    println!("  Polynomial degree:  {}", degree);
    println!("  Training points:    {}", n_data);
    println!("  Collocation points: {}", n_colloc);
    println!("  Noise std:          {}", noise_std);
    println!("  Epochs:             {}", epochs);
    println!("  Learning rate:      {}", lr);
    println!("  Physics weight:     {}", physics_weight);
    println!("  Boundary weight:    {}", boundary_weight);
    println!("  Seed:               {}", seed);
    println!();

    // --- Train ---
    println!("Training...");
    let result = piml_heat_1d_train(
        degree, n_data, n_colloc, noise_std, epochs, lr, physics_weight, boundary_weight, seed,
    );

    // --- Results ---
    let first = &result.history[0];
    let last = result.history.last().unwrap();
    println!();
    println!("Training Results:");
    println!("  Initial loss:      {:.6}", first.total_loss);
    println!("  Final loss:        {:.6}", last.total_loss);
    println!("  Loss reduction:    {:.1}x", first.total_loss / last.total_loss);
    println!("  Final data loss:   {:.6}", last.data_loss);
    println!("  Final physics loss:{:.6}", last.physics_loss);
    println!("  Final bnd loss:    {:.6}", last.boundary_loss);
    println!("  Final grad norm:   {:.6}", last.grad_norm);
    println!();

    // --- Accuracy ---
    println!("Accuracy vs Analytical Solution:");
    println!("  L2 error:          {:.6}", result.l2_error.unwrap());
    println!("  Max error:         {:.6}", result.max_error.unwrap());
    println!("  Mean residual:     {:.6}", result.mean_residual);
    println!();

    // --- Polynomial coefficients ---
    println!("Learned Polynomial Coefficients:");
    for (i, c) in result.final_params.iter().enumerate() {
        println!("  a_{} = {:+.8}", i, c);
    }
    println!();

    // --- Determinism check ---
    println!("Determinism Check:");
    let result2 = piml_heat_1d_train(
        degree, n_data, n_colloc, noise_std, epochs, lr, physics_weight, boundary_weight, seed,
    );
    let params_match = result.final_params.iter().zip(result2.final_params.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let loss_match = result.history.iter().zip(result2.history.iter())
        .all(|(a, b)| a.total_loss.to_bits() == b.total_loss.to_bits());
    println!("  Parameters bit-identical: {}", if params_match { "PASS ✓" } else { "FAIL ✗" });
    println!("  Loss trajectory identical: {}", if loss_match { "PASS ✓" } else { "FAIL ✗" });
    println!();

    // --- Solution plot ---
    let n_plot = 50;
    let x_plot: Vec<f64> = (0..n_plot).map(|i| (i as f64 + 0.5) / n_plot as f64).collect();
    let u_exact: Vec<f64> = x_plot.iter().map(|&x| (std::f64::consts::PI * x).sin()).collect();
    let u_pred: Vec<f64> = x_plot.iter().map(|&x| {
        let mut val = 0.0;
        for i in (0..result.final_params.len()).rev() {
            val = val * x + result.final_params[i];
        }
        val
    }).collect();

    println!("Solution Comparison (exact vs predicted):");
    println!("{}", ascii_plot(&x_plot, &u_exact, 60, 12, "u_exact = sin(πx)"));
    println!("{}", ascii_plot(&x_plot, &u_pred, 60, 12, "u_predicted (polynomial)"));

    // --- Loss curve ---
    println!("{}", plot_loss_history(&result.history, 60, 12));

    println!("Done. Run verification: cargo test --test pinn_correctness --test sciml_determinism");
}
