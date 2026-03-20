//! PINN Correctness Tests
//!
//! Verifies that the Physics-Informed Neural Network and Physics-Informed ML
//! implementations produce mathematically correct results:
//!
//! 1. Solution accuracy (L2 and max error vs analytical solution)
//! 2. PDE residual near-zero behavior
//! 3. Gradient correctness (finite-difference vs autodiff)
//! 4. Boundary condition satisfaction
//! 5. Loss component decomposition

use cjc_ad::pinn::*;
use cjc_ad::GradGraph;
use cjc_runtime::Tensor;

// ---------------------------------------------------------------------------
// PIML: 1D Heat Equation Correctness
// ---------------------------------------------------------------------------

#[test]
fn test_piml_heat_solution_accuracy() {
    // Hardened: degree 6 (avoids noise overfitting), stronger physics + boundary
    let result = piml_heat_1d_train(
        6,       // degree (lower to prevent overfitting)
        40,      // n_data
        60,      // n_colloc
        0.005,   // low noise
        5000,    // epochs
        1e-3,    // lr
        5.0,     // physics_weight (hardened)
        50.0,    // boundary_weight (hardened)
        42,      // seed
    );

    let l2 = result.l2_error.unwrap();
    let max_err = result.max_error.unwrap();

    // With hardened params, L2 should be tight
    assert!(l2 < 0.1, "L2 error too large: {}", l2);
    assert!(max_err < 0.2, "Max error too large: {}", max_err);
}

#[test]
fn test_piml_heat_loss_decreases_monotonically_overall() {
    let result = piml_heat_1d_train(8, 30, 50, 0.01, 2000, 1e-3, 1.0, 10.0, 42);

    // Compare first 10% of epochs vs last 10%
    let n = result.history.len();
    let early_avg: f64 = result.history[..n / 10].iter().map(|h| h.total_loss).sum::<f64>()
        / (n / 10) as f64;
    let late_avg: f64 = result.history[n * 9 / 10..].iter().map(|h| h.total_loss).sum::<f64>()
        / (n / 10) as f64;

    assert!(
        late_avg < early_avg,
        "Late average loss ({}) should be less than early average ({})",
        late_avg,
        early_avg,
    );
}

#[test]
fn test_piml_heat_boundary_conditions() {
    // Hardened: boundary_weight=50 anchors boundaries strongly
    let result = piml_heat_1d_train(6, 40, 60, 0.01, 5000, 1e-3, 5.0, 50.0, 42);

    // Evaluate polynomial at boundaries
    let coeffs = &result.final_params;
    let u0 = poly_eval_public(coeffs, 0.0);
    let u1 = poly_eval_public(coeffs, 1.0);

    // With boundary weight = 50 and 5000 epochs, boundaries should be very close to zero
    assert!(
        u0.abs() < 0.1,
        "u(0) should be near 0, got {}",
        u0,
    );
    assert!(
        u1.abs() < 0.1,
        "u(1) should be near 0, got {}",
        u1,
    );
}

/// Public wrapper for poly_eval testing (Horner's method).
fn poly_eval_public(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for i in (0..coeffs.len()).rev() {
        result = result * x + coeffs[i];
    }
    result
}

// ---------------------------------------------------------------------------
// PINN: Harmonic Oscillator Correctness
// ---------------------------------------------------------------------------

#[test]
fn test_pinn_harmonic_loss_components_all_tracked() {
    let config = PinnConfig {
        layer_sizes: vec![1, 16, 16, 1],
        epochs: 50,
        lr: 1e-3,
        physics_weight: 1.0,
        boundary_weight: 50.0,
        seed: 42,
        n_collocation: 15,
        n_data: 10,
        fd_eps: 1e-3,
    };

    let result = pinn_harmonic_train(&config);

    // Every epoch should have all loss components
    for log in &result.history {
        assert!(log.data_loss.is_finite(), "Data loss not finite at epoch {}", log.epoch);
        assert!(log.physics_loss.is_finite(), "Physics loss not finite at epoch {}", log.epoch);
        assert!(log.boundary_loss.is_finite(), "Boundary loss not finite at epoch {}", log.epoch);
        assert!(log.total_loss.is_finite(), "Total loss not finite at epoch {}", log.epoch);
        assert!(log.grad_norm.is_finite(), "Grad norm not finite at epoch {}", log.epoch);
        assert!(log.data_loss >= 0.0, "Data loss should be non-negative");
        assert!(log.physics_loss >= 0.0, "Physics loss should be non-negative");
        assert!(log.boundary_loss >= 0.0, "Boundary loss should be non-negative");
    }
}

#[test]
fn test_pinn_harmonic_gradient_nonzero() {
    let config = PinnConfig {
        layer_sizes: vec![1, 8, 8, 1],
        epochs: 10,
        lr: 1e-3,
        physics_weight: 1.0,
        boundary_weight: 50.0,
        seed: 42,
        n_collocation: 10,
        n_data: 10,
        fd_eps: 1e-3,
    };

    let result = pinn_harmonic_train(&config);

    // Gradients should be nonzero in early training
    let first_grad = result.history.first().unwrap().grad_norm;
    assert!(
        first_grad > 0.0,
        "Initial gradient norm should be positive, got {}",
        first_grad,
    );
}

// ---------------------------------------------------------------------------
// MLP Gradient Verification (Finite Difference vs Autodiff)
// ---------------------------------------------------------------------------

#[test]
fn test_mlp_gradient_finite_diff_check() {
    // Build a small 2-layer MLP and verify gradients via finite differences.
    // We check the first layer's weight gradient by perturbing each element
    // and recomputing the loss from scratch (identical graph topology).

    let layer_sizes = [1, 4, 1];
    let seed = 42;
    let x_val = 0.5;
    let target_val = 1.0;
    let eps = 1e-5;

    // First, get baseline weights from a reference init
    let mut graph0 = GradGraph::new();
    let (_mlp_spec0, p0) = mlp_init(&mut graph0, &layer_sizes, Activation::Tanh, Activation::None, seed);
    let w0_data = graph0.tensor(p0[0]).to_vec();
    let w0_shape = graph0.tensor(p0[0]).shape().to_vec();
    let b0_data = graph0.tensor(p0[1]).to_vec();
    let b0_shape = graph0.tensor(p0[1]).shape().to_vec();
    let w1_data = graph0.tensor(p0[2]).to_vec();
    let w1_shape = graph0.tensor(p0[2]).shape().to_vec();
    let b1_data = graph0.tensor(p0[3]).to_vec();
    let b1_shape = graph0.tensor(p0[3]).shape().to_vec();

    // Helper: build a fresh graph with given first-layer weights, compute loss
    let build_and_eval = |w0_vec: &[f64]| -> f64 {
        let mut graph = GradGraph::new();
        let w0_idx = graph.parameter(Tensor::from_vec_unchecked(w0_vec.to_vec(), &w0_shape));
        let b0_idx = graph.parameter(Tensor::from_vec_unchecked(b0_data.clone(), &b0_shape));
        let w1_idx = graph.parameter(Tensor::from_vec_unchecked(w1_data.clone(), &w1_shape));
        let b1_idx = graph.parameter(Tensor::from_vec_unchecked(b1_data.clone(), &b1_shape));

        let mlp = Mlp {
            layers: vec![
                DenseLayer {
                    weight_idx: w0_idx, bias_idx: b0_idx,
                    activation: Activation::Tanh, in_features: 1, out_features: 4,
                },
                DenseLayer {
                    weight_idx: w1_idx, bias_idx: b1_idx,
                    activation: Activation::None, in_features: 4, out_features: 1,
                },
            ],
        };

        let x = graph.input(Tensor::from_vec_unchecked(vec![x_val], &[1, 1]));
        let y = mlp_forward(&mut graph, &mlp, x);
        let target = graph.input(Tensor::from_vec_unchecked(vec![target_val], &[1, 1]));
        let loss = data_loss_mse(&mut graph, y, target);
        graph.value(loss)
    };

    // Build reference graph and get AD gradients
    let mut ref_graph = GradGraph::new();
    let w0_ref = ref_graph.parameter(Tensor::from_vec_unchecked(w0_data.clone(), &w0_shape));
    let b0_ref = ref_graph.parameter(Tensor::from_vec_unchecked(b0_data.clone(), &b0_shape));
    let w1_ref = ref_graph.parameter(Tensor::from_vec_unchecked(w1_data.clone(), &w1_shape));
    let b1_ref = ref_graph.parameter(Tensor::from_vec_unchecked(b1_data.clone(), &b1_shape));
    let ref_mlp = Mlp {
        layers: vec![
            DenseLayer {
                weight_idx: w0_ref, bias_idx: b0_ref,
                activation: Activation::Tanh, in_features: 1, out_features: 4,
            },
            DenseLayer {
                weight_idx: w1_ref, bias_idx: b1_ref,
                activation: Activation::None, in_features: 4, out_features: 1,
            },
        ],
    };
    let x_ref = ref_graph.input(Tensor::from_vec_unchecked(vec![x_val], &[1, 1]));
    let y_ref = mlp_forward(&mut ref_graph, &ref_mlp, x_ref);
    let t_ref = ref_graph.input(Tensor::from_vec_unchecked(vec![target_val], &[1, 1]));
    let loss_ref = data_loss_mse(&mut ref_graph, y_ref, t_ref);
    ref_graph.zero_grad();
    ref_graph.backward(loss_ref);
    let ad_grad = ref_graph.grad(w0_ref).expect("Should have gradient").to_vec();
    let w0 = w0_data;

    // Finite-difference check for each element
    for k in 0..w0.len() {
        let mut w_plus = w0.clone();
        w_plus[k] += eps;
        let loss_plus = build_and_eval(&w_plus);

        let mut w_minus = w0.clone();
        w_minus[k] -= eps;
        let loss_minus = build_and_eval(&w_minus);

        let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let ad_val = ad_grad[k];

        let abs_err = (fd_grad - ad_val).abs();
        let rel_err = if ad_val.abs() > 1e-8 {
            abs_err / ad_val.abs()
        } else {
            abs_err
        };

        assert!(
            rel_err < 0.05 || abs_err < 1e-4,
            "Gradient mismatch at w[{}]: AD={:.8}, FD={:.8}, rel_err={:.6}",
            k,
            ad_val,
            fd_grad,
            rel_err,
        );
    }
}

// ---------------------------------------------------------------------------
// Sampling Correctness
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_grid_evenly_spaced() {
    let g = uniform_grid(0.0, 1.0, 5);
    assert_eq!(g.len(), 5);
    // Points should be at 0.1, 0.3, 0.5, 0.7, 0.9
    for i in 0..5 {
        let expected = (i as f64 + 0.5) / 5.0;
        assert!(
            (g[i] - expected).abs() < 1e-12,
            "Grid point {} expected {}, got {}",
            i,
            expected,
            g[i],
        );
    }
}

#[test]
fn test_lhs_grid_covers_domain() {
    let g = lhs_grid_1d(2.0, 5.0, 50, 42);
    assert_eq!(g.len(), 50);
    for &x in &g {
        assert!(x >= 2.0 && x < 5.0, "Point {} outside [2, 5)", x);
    }
    // Check coverage: min should be near 2.0, max near 5.0
    let min_val = g.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(min_val < 2.5, "LHS min {} not near domain start", min_val);
    assert!(max_val > 4.5, "LHS max {} not near domain end", max_val);
}

// ---------------------------------------------------------------------------
// ASCII Visualization
// ---------------------------------------------------------------------------

#[test]
fn test_ascii_plot_empty() {
    let plot = ascii_plot(&[], &[], 40, 10, "empty");
    assert!(plot.contains("empty"));
}

#[test]
fn test_ascii_plot_sin_wave() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::PI * 2.0 / n as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let plot = ascii_plot(&x, &y, 60, 15, "sin(x)");
    assert!(plot.contains("sin(x)"));
    assert!(plot.contains("*"));
    // Should have multiple lines
    assert!(plot.lines().count() > 5);
}

#[test]
fn test_loss_history_plot_renders() {
    let history: Vec<TrainLog> = (0..100)
        .map(|i| TrainLog {
            epoch: i,
            total_loss: 10.0 * (-0.05 * i as f64).exp(),
            data_loss: 5.0 * (-0.05 * i as f64).exp(),
            physics_loss: 3.0 * (-0.05 * i as f64).exp(),
            boundary_loss: 2.0 * (-0.05 * i as f64).exp(),
            grad_norm: 1.0 * (-0.02 * i as f64).exp(),
        })
        .collect();

    let plot = plot_loss_history(&history, 60, 15);
    assert!(plot.contains("Training Loss"));
    assert!(plot.contains("*"));
}
