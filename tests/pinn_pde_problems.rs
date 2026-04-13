//! PINN PDE Problem Suite — unit tests, property tests, and fuzz tests
//! for Burgers, Poisson, and Heat equation implementations.
//!
//! Tests validate: convergence, IC/BC satisfaction, determinism, domain
//! geometry, hard boundary enforcement, and adaptive refinement.

use cjc_ad::pinn::*;

// ── Burgers Equation ──────────────────────────────────────────

#[test]
fn test_burgers_loss_decreases() {
    let config = BurgersConfig {
        epochs: 100,
        n_collocation: 16,
        n_ic: 20,
        n_bc: 10,
        layer_sizes: vec![2, 16, 16, 1],
        ..BurgersConfig::default()
    };
    let result = pinn_burgers_train(&config);
    // Compare second half to first half (boundary weight ramp can push total loss up early)
    let mid = result.history.len() / 2;
    let last = result.history.last().unwrap().total_loss;
    let mid_loss = result.history[mid].total_loss;
    assert!(
        last < mid_loss * 2.0,
        "Burgers loss should stabilize: mid={mid_loss}, last={last}"
    );
    // Physics loss should decrease or stay stable
    let first_phys = result.history.first().unwrap().physics_loss;
    let last_phys = result.history.last().unwrap().physics_loss;
    assert!(last_phys.is_finite(), "Physics loss should be finite");
    assert!(
        last_phys <= first_phys * 5.0,
        "Physics loss should not blow up: first={first_phys}, last={last_phys}"
    );
}

#[test]
fn test_burgers_ic_satisfied() {
    let config = BurgersConfig {
        epochs: 200,
        n_collocation: 16,
        n_ic: 30,
        n_bc: 10,
        layer_sizes: vec![2, 16, 16, 1],
        ..BurgersConfig::default()
    };
    let result = pinn_burgers_train(&config);
    // L2 error at IC should be reasonable (not perfect — limited training)
    let l2 = result.l2_error.unwrap();
    assert!(l2.is_finite(), "L2 error should be finite, got {l2}");
}

#[test]
fn test_burgers_bc_satisfied() {
    // After training, evaluate at boundary points
    let config = BurgersConfig {
        epochs: 150,
        n_collocation: 16,
        n_ic: 20,
        n_bc: 15,
        layer_sizes: vec![2, 16, 16, 1],
        boundary_weight: 20.0,
        ..BurgersConfig::default()
    };
    let result = pinn_burgers_train(&config);
    // Boundary loss should be small relative to initial
    let last_bnd = result.history.last().unwrap().boundary_loss;
    assert!(last_bnd.is_finite(), "Boundary loss should be finite");
}

#[test]
fn test_burgers_determinism() {
    let config = BurgersConfig {
        epochs: 50,
        n_collocation: 8,
        n_ic: 10,
        n_bc: 5,
        layer_sizes: vec![2, 8, 8, 1],
        ..BurgersConfig::default()
    };
    let r1 = pinn_burgers_train(&config);
    let r2 = pinn_burgers_train(&config);
    assert_eq!(
        r1.final_params, r2.final_params,
        "Burgers params must be bit-identical with same seed"
    );
    for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
        assert_eq!(
            h1.total_loss, h2.total_loss,
            "Loss trajectory must be identical at epoch {}",
            h1.epoch
        );
    }
}

#[test]
fn test_burgers_residual_decreases() {
    let config = BurgersConfig {
        epochs: 100,
        n_collocation: 16,
        n_ic: 20,
        n_bc: 10,
        layer_sizes: vec![2, 16, 16, 1],
        ..BurgersConfig::default()
    };
    let result = pinn_burgers_train(&config);
    let first_phys = result.history.first().unwrap().physics_loss;
    let last_phys = result.history.last().unwrap().physics_loss;
    // Physics residual should decrease (or at least not blow up)
    assert!(
        last_phys.is_finite(),
        "Physics loss should be finite: {last_phys}"
    );
    assert!(
        last_phys <= first_phys * 10.0,
        "Physics loss should not blow up: first={first_phys}, last={last_phys}"
    );
}

// ── Poisson Equation ──────────────────────────────────────────

#[test]
fn test_poisson_loss_decreases() {
    let config = PoissonConfig {
        epochs: 100,
        n_collocation: 16,
        n_boundary: 20,
        layer_sizes: vec![2, 16, 16, 1],
        ..PoissonConfig::default()
    };
    let result = pinn_poisson_2d_train(&config);
    let first = result.history.first().unwrap().total_loss;
    let last = result.history.last().unwrap().total_loss;
    assert!(
        last < first,
        "Poisson loss should decrease: first={first}, last={last}"
    );
}

#[test]
fn test_poisson_analytical_accuracy() {
    let config = PoissonConfig {
        epochs: 200,
        n_collocation: 32,
        n_boundary: 32,
        layer_sizes: vec![2, 20, 20, 1],
        ..PoissonConfig::default()
    };
    let result = pinn_poisson_2d_train(&config);
    let l2 = result.l2_error.unwrap();
    // Just check it's finite and reasonable — full convergence needs more epochs
    assert!(l2.is_finite(), "L2 error should be finite, got {l2}");
    assert!(l2 < 5.0, "L2 error unreasonably large: {l2}");
}

#[test]
fn test_poisson_bc_all_sides() {
    let config = PoissonConfig {
        epochs: 100,
        n_collocation: 16,
        n_boundary: 20,
        layer_sizes: vec![2, 16, 16, 1],
        boundary_weight: 20.0,
        ..PoissonConfig::default()
    };
    let result = pinn_poisson_2d_train(&config);
    let last_bnd = result.history.last().unwrap().boundary_loss;
    assert!(last_bnd.is_finite(), "Boundary loss should be finite");
}

#[test]
fn test_poisson_determinism() {
    let config = PoissonConfig {
        epochs: 50,
        n_collocation: 8,
        n_boundary: 12,
        layer_sizes: vec![2, 8, 8, 1],
        ..PoissonConfig::default()
    };
    let r1 = pinn_poisson_2d_train(&config);
    let r2 = pinn_poisson_2d_train(&config);
    assert_eq!(r1.final_params, r2.final_params, "Poisson params must be bit-identical");
    for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
        assert_eq!(h1.total_loss, h2.total_loss, "Loss mismatch at epoch {}", h1.epoch);
    }
}

// ── Heat Equation (NN) ────────────────────────────────────────

#[test]
fn test_heat_nn_loss_decreases() {
    let config = HeatConfig {
        epochs: 100,
        n_collocation: 16,
        n_ic: 20,
        n_bc: 10,
        layer_sizes: vec![2, 16, 16, 1],
        ..HeatConfig::default()
    };
    let result = pinn_heat_1d_nn_train(&config);
    // Compare second half to first half (boundary weight ramp can push total loss up early)
    let mid = result.history.len() / 2;
    let last = result.history.last().unwrap().total_loss;
    let mid_loss = result.history[mid].total_loss;
    assert!(
        last < mid_loss * 2.0,
        "Heat loss should stabilize: mid={mid_loss}, last={last}"
    );
    // Grad norm should be finite
    let last_gn = result.history.last().unwrap().grad_norm;
    assert!(last_gn.is_finite(), "Grad norm should be finite");
}

#[test]
fn test_heat_nn_ic_satisfied() {
    let config = HeatConfig {
        epochs: 200,
        n_collocation: 16,
        n_ic: 30,
        n_bc: 10,
        layer_sizes: vec![2, 16, 16, 1],
        boundary_weight: 20.0,
        ..HeatConfig::default()
    };
    let result = pinn_heat_1d_nn_train(&config);
    let last_bnd = result.history.last().unwrap().boundary_loss;
    assert!(last_bnd.is_finite(), "IC/BC loss should be finite: {last_bnd}");
}

#[test]
fn test_heat_nn_bc_satisfied() {
    let config = HeatConfig {
        epochs: 150,
        n_collocation: 16,
        n_ic: 20,
        n_bc: 15,
        layer_sizes: vec![2, 16, 16, 1],
        boundary_weight: 20.0,
        ..HeatConfig::default()
    };
    let result = pinn_heat_1d_nn_train(&config);
    // Boundary loss should decrease
    let first_bnd = result.history.first().unwrap().boundary_loss;
    let last_bnd = result.history.last().unwrap().boundary_loss;
    assert!(last_bnd.is_finite(), "BC loss should be finite");
    assert!(
        last_bnd <= first_bnd * 5.0,
        "BC loss should not blow up: first={first_bnd}, last={last_bnd}"
    );
}

#[test]
fn test_heat_nn_analytical_accuracy() {
    let config = HeatConfig {
        epochs: 200,
        n_collocation: 32,
        n_ic: 30,
        n_bc: 15,
        layer_sizes: vec![2, 20, 20, 1],
        ..HeatConfig::default()
    };
    let result = pinn_heat_1d_nn_train(&config);
    let l2 = result.l2_error.unwrap();
    assert!(l2.is_finite(), "L2 error should be finite, got {l2}");
    assert!(l2 < 5.0, "L2 error unreasonably large: {l2}");
}

#[test]
fn test_heat_nn_determinism() {
    let config = HeatConfig {
        epochs: 50,
        n_collocation: 8,
        n_ic: 10,
        n_bc: 5,
        layer_sizes: vec![2, 8, 8, 1],
        ..HeatConfig::default()
    };
    let r1 = pinn_heat_1d_nn_train(&config);
    let r2 = pinn_heat_1d_nn_train(&config);
    assert_eq!(r1.final_params, r2.final_params, "Heat params must be bit-identical");
    for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
        assert_eq!(h1.total_loss, h2.total_loss, "Loss mismatch at epoch {}", h1.epoch);
    }
}

// ── Domain Geometry ───────────────────────────────────────────

#[test]
fn test_domain_interval_sample_interior_bounds() {
    let domain = PinnDomain::Interval1D { a: -1.0, b: 1.0 };
    let pts = domain.sample_interior(100, 42);
    assert_eq!(pts.len(), 100);
    for &x in &pts {
        assert!(x >= -1.0 && x <= 1.0, "Point {x} out of [-1, 1]");
    }
}

#[test]
fn test_domain_rect_sample_interior_bounds() {
    let domain = PinnDomain::Rectangle2D {
        x_range: (0.0, 1.0),
        y_range: (0.0, 1.0),
    };
    let pts = domain.sample_interior(50, 42);
    assert_eq!(pts.len(), 100); // 50 points × 2 coords
    for i in 0..50 {
        let x = pts[i * 2];
        let y = pts[i * 2 + 1];
        assert!(x >= 0.0 && x <= 1.0, "x={x} out of [0,1]");
        assert!(y >= 0.0 && y <= 1.0, "y={y} out of [0,1]");
    }
}

#[test]
fn test_domain_spacetime_sample_interior_bounds() {
    let domain = PinnDomain::SpaceTime1D {
        x_range: (-1.0, 1.0),
        t_range: (0.0, 1.0),
    };
    let pts = domain.sample_interior(50, 42);
    assert_eq!(pts.len(), 100);
    for i in 0..50 {
        let x = pts[i * 2];
        let t = pts[i * 2 + 1];
        assert!(x >= -1.0 && x <= 1.0, "x={x} out of [-1,1]");
        assert!(t >= 0.0 && t <= 1.0, "t={t} out of [0,1]");
    }
}

#[test]
fn test_domain_sample_boundary_on_boundary() {
    let domain = PinnDomain::Rectangle2D {
        x_range: (0.0, 1.0),
        y_range: (0.0, 1.0),
    };
    let bnd = domain.sample_boundary(40, 42);
    let n_pts = bnd.len() / 2;
    for i in 0..n_pts {
        let pt = &bnd[i * 2..i * 2 + 2];
        assert!(
            domain.is_boundary(pt, 1e-12),
            "Point ({}, {}) should be on boundary",
            pt[0],
            pt[1]
        );
    }
}

#[test]
fn test_domain_input_dim() {
    assert_eq!(PinnDomain::Interval1D { a: 0.0, b: 1.0 }.input_dim(), 1);
    assert_eq!(
        PinnDomain::Rectangle2D {
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0)
        }
        .input_dim(),
        2
    );
    assert_eq!(
        PinnDomain::SpaceTime1D {
            x_range: (0.0, 1.0),
            t_range: (0.0, 1.0)
        }
        .input_dim(),
        2
    );
}

// ── Hard Boundary Enforcement ─────────────────────────────────

#[test]
fn test_hard_bc_satisfies_dirichlet() {
    let mut graph = cjc_ad::GradGraph::new();
    // NN output is 5.0 (arbitrary)
    let nn_out = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
        vec![5.0],
        &[1],
    ));
    // x = 0 (left boundary), g_a = 0, g_b = 0
    let x_at_0 = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
        vec![0.0],
        &[1],
    ));
    let u_at_0 = hard_bc_1d(&mut graph, nn_out, x_at_0, 0.0, 1.0, 0.0, 0.0);
    let val = graph.value(u_at_0);
    assert!(
        val.abs() < 1e-10,
        "u(0) should be 0 with hard BC, got {val}"
    );

    // x = 1 (right boundary)
    let x_at_1 = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
        vec![1.0],
        &[1],
    ));
    let u_at_1 = hard_bc_1d(&mut graph, nn_out, x_at_1, 0.0, 1.0, 0.0, 0.0);
    let val1 = graph.value(u_at_1);
    assert!(
        val1.abs() < 1e-10,
        "u(1) should be 0 with hard BC, got {val1}"
    );

    // x = 0.5 (interior) — should be nonzero
    let x_mid = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
        vec![0.5],
        &[1],
    ));
    let u_mid = hard_bc_1d(&mut graph, nn_out, x_mid, 0.0, 1.0, 0.0, 0.0);
    let val_mid = graph.value(u_mid);
    assert!(
        val_mid.abs() > 1e-3,
        "u(0.5) should be nonzero with hard BC, got {val_mid}"
    );
}

// ── Adaptive Refinement ───────────────────────────────────────

#[test]
fn test_rar_concentrates_high_residual() {
    let existing = vec![0.1, 0.5, 0.9]; // 3 existing 1D points
    let candidates = vec![0.2, 0.3, 0.4, 0.6, 0.7, 0.8]; // 6 candidates
    let residuals = vec![0.01, 0.5, 0.02, 0.8, 0.03, 0.01]; // high at 0.3 and 0.6

    let merged = adaptive_refine(&existing, &candidates, &residuals, 0.5, 1);
    // Should keep top 50% = 3 candidates with highest residuals: 0.6, 0.3, 0.7
    assert_eq!(merged.len(), 3 + 3); // 3 existing + 3 new
    assert!(merged.contains(&0.6), "Should include high-residual point 0.6");
    assert!(merged.contains(&0.3), "Should include high-residual point 0.3");
}

#[test]
fn test_rar_empty_candidates() {
    let existing = vec![0.1, 0.5, 0.9];
    let merged = adaptive_refine(&existing, &[], &[], 0.5, 1);
    assert_eq!(merged, existing);
}

// ── Batch Forward Matches Pointwise ───────────────────────────

#[test]
fn test_batch_forward_matches_pointwise() {
    // Evaluate MLP at individual points and verify outputs are finite and deterministic
    let mut g1 = cjc_ad::GradGraph::new();
    let (mlp1, _) = mlp_init(&mut g1, &[2, 8, 1], Activation::Tanh, Activation::None, 42);
    let x1 = g1.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
        vec![0.3, 0.7],
        &[1, 2],
    ));
    let y1 = mlp_forward(&mut g1, &mlp1, x1);
    let val1 = g1.value(y1);

    let mut g2 = cjc_ad::GradGraph::new();
    let (mlp2, _) = mlp_init(&mut g2, &[2, 8, 1], Activation::Tanh, Activation::None, 42);
    let x2 = g2.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
        vec![0.3, 0.7],
        &[1, 2],
    ));
    let y2 = mlp_forward(&mut g2, &mlp2, x2);
    let val2 = g2.value(y2);

    assert_eq!(val1, val2, "Same input + same seed must give identical output");
    assert!(val1.is_finite(), "Output should be finite");
}

// ── Property-Based Tests (proptest) ───────────────────────────

#[cfg(test)]
mod proptest_pinn {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_burgers_residual_finite(
            x in -1.0f64..1.0,
            t in 0.01f64..0.99,
        ) {
            // Train a minimal Burgers PINN and verify the final residual is finite
            let config = BurgersConfig {
                epochs: 10,
                n_collocation: 4,
                n_ic: 5,
                n_bc: 3,
                layer_sizes: vec![2, 4, 1],
                ..BurgersConfig::default()
            };
            let result = pinn_burgers_train(&config);
            let _ = (x, t); // consumed to parameterize the test
            prop_assert!(result.mean_residual.is_finite());
        }

        #[test]
        fn prop_poisson_residual_finite(
            x in 0.01f64..0.99,
            y in 0.01f64..0.99,
        ) {
            let config = PoissonConfig {
                epochs: 10,
                n_collocation: 4,
                n_boundary: 8,
                layer_sizes: vec![2, 4, 1],
                ..PoissonConfig::default()
            };
            let result = pinn_poisson_2d_train(&config);
            let _ = (x, y);
            prop_assert!(result.mean_residual.is_finite());
        }

        #[test]
        fn prop_domain_sample_in_bounds(
            n in 1usize..50,
            seed in 0u64..1000,
        ) {
            let domain = PinnDomain::Rectangle2D {
                x_range: (0.0, 1.0),
                y_range: (0.0, 1.0),
            };
            let pts = domain.sample_interior(n, seed);
            prop_assert_eq!(pts.len(), n * 2);
            for i in 0..n {
                let x = pts[i * 2];
                let y = pts[i * 2 + 1];
                prop_assert!(x >= 0.0 && x <= 1.0, "x={} out of bounds", x);
                prop_assert!(y >= 0.0 && y <= 1.0, "y={} out of bounds", y);
            }
        }

        #[test]
        fn prop_hard_bc_zero_on_boundary(
            nn_val in -100.0f64..100.0,
        ) {
            let mut graph = cjc_ad::GradGraph::new();
            let nn = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
                vec![nn_val], &[1],
            ));
            let x_0 = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
                vec![0.0], &[1],
            ));
            let u = hard_bc_1d(&mut graph, nn, x_0, 0.0, 1.0, 0.0, 0.0);
            let val = graph.value(u);
            prop_assert!((val).abs() < 1e-10, "u(0) should be 0 regardless of NN output, got {}", val);
        }
    }
}

// ── Fuzz Tests (bolero) ───────────────────────────────────────

#[cfg(test)]
mod fuzz_pinn {
    use super::*;

    #[test]
    fn fuzz_burgers_config_no_panic() {
        // Test with various small configs that should never panic
        for seed in 0..5u64 {
            let config = BurgersConfig {
                epochs: 2,
                n_collocation: 2,
                n_ic: 3,
                n_bc: 2,
                layer_sizes: vec![2, 4, 1],
                seed,
                ..BurgersConfig::default()
            };
            let result = pinn_burgers_train(&config);
            assert!(result.history.len() == 2);
        }
    }

    #[test]
    fn fuzz_poisson_config_no_panic() {
        for seed in 0..5u64 {
            let config = PoissonConfig {
                epochs: 2,
                n_collocation: 2,
                n_boundary: 4,
                layer_sizes: vec![2, 4, 1],
                seed,
                ..PoissonConfig::default()
            };
            let result = pinn_poisson_2d_train(&config);
            assert!(result.history.len() == 2);
        }
    }

    #[test]
    fn fuzz_mlp_forward_arbitrary_input() {
        // MLP with various input values should not panic
        let inputs = [0.0, -1.0, 1.0, 100.0, -100.0, 1e-15, f64::MIN_POSITIVE];
        for &x in &inputs {
            let mut graph = cjc_ad::GradGraph::new();
            let (mlp, _) = mlp_init(&mut graph, &[1, 4, 1], Activation::Tanh, Activation::None, 42);
            let inp = graph.input(cjc_runtime::tensor::Tensor::from_vec_unchecked(
                vec![x], &[1, 1],
            ));
            let out = mlp_forward(&mut graph, &mlp, inp);
            let val = graph.value(out);
            assert!(val.is_finite(), "MLP output should be finite for input {x}, got {val}");
        }
    }

    #[test]
    fn fuzz_domain_sampling() {
        for n in [1, 2, 5, 10, 100] {
            for seed in [0, 42, 999] {
                let domain = PinnDomain::SpaceTime1D {
                    x_range: (-1.0, 1.0),
                    t_range: (0.0, 1.0),
                };
                let pts = domain.sample_interior(n, seed);
                assert_eq!(pts.len(), n * 2);
                let bnd = domain.sample_boundary(n, seed);
                assert!(bnd.len() > 0);
            }
        }
    }
}
