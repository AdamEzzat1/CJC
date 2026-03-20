//! SciML Determinism Tests
//!
//! Verifies bit-identical execution across multiple runs for:
//! 1. PIML training (same seed → identical params, identical loss trajectory)
//! 2. PINN training (same seed → identical params, identical loss trajectory)
//! 3. MLP initialization and forward pass
//! 4. Data generation and sampling
//! 5. Different seeds produce different results

use cjc_ad::pinn::*;
use cjc_ad::GradGraph;
use cjc_runtime::Tensor;

// ---------------------------------------------------------------------------
// PIML Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_piml_determinism_params_bit_identical() {
    let r1 = piml_heat_1d_train(6, 20, 30, 0.01, 1000, 1e-3, 1.0, 10.0, 42);
    let r2 = piml_heat_1d_train(6, 20, 30, 0.01, 1000, 1e-3, 1.0, 10.0, 42);

    assert_eq!(
        r1.final_params.len(),
        r2.final_params.len(),
        "Param count mismatch"
    );
    for (i, (&a, &b)) in r1.final_params.iter().zip(r2.final_params.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Param {} not bit-identical: {} vs {}",
            i,
            a,
            b,
        );
    }
}

#[test]
fn test_piml_determinism_loss_trajectory_identical() {
    let r1 = piml_heat_1d_train(6, 20, 30, 0.01, 500, 1e-3, 1.0, 10.0, 42);
    let r2 = piml_heat_1d_train(6, 20, 30, 0.01, 500, 1e-3, 1.0, 10.0, 42);

    assert_eq!(r1.history.len(), r2.history.len());
    for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
        assert_eq!(
            h1.total_loss.to_bits(),
            h2.total_loss.to_bits(),
            "Loss mismatch at epoch {}: {} vs {}",
            h1.epoch,
            h1.total_loss,
            h2.total_loss,
        );
        assert_eq!(
            h1.data_loss.to_bits(),
            h2.data_loss.to_bits(),
            "Data loss mismatch at epoch {}",
            h1.epoch,
        );
        assert_eq!(
            h1.physics_loss.to_bits(),
            h2.physics_loss.to_bits(),
            "Physics loss mismatch at epoch {}",
            h1.epoch,
        );
        assert_eq!(
            h1.grad_norm.to_bits(),
            h2.grad_norm.to_bits(),
            "Grad norm mismatch at epoch {}",
            h1.epoch,
        );
    }
}

#[test]
fn test_piml_different_seeds_different_results() {
    let r1 = piml_heat_1d_train(6, 20, 30, 0.01, 200, 1e-3, 1.0, 10.0, 42);
    let r2 = piml_heat_1d_train(6, 20, 30, 0.01, 200, 1e-3, 1.0, 10.0, 99);

    // With different seeds, data is different, so training should diverge
    assert_ne!(
        r1.final_params, r2.final_params,
        "Different seeds should produce different params"
    );
}

// ---------------------------------------------------------------------------
// PINN Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_pinn_determinism_params_bit_identical() {
    let config = PinnConfig {
        layer_sizes: vec![1, 8, 8, 1],
        epochs: 30,
        lr: 1e-3,
        physics_weight: 1.0,
        boundary_weight: 10.0,
        seed: 42,
        n_collocation: 10,
        n_data: 8,
        fd_eps: 1e-4,
    };

    let r1 = pinn_harmonic_train(&config);
    let r2 = pinn_harmonic_train(&config);

    assert_eq!(r1.final_params.len(), r2.final_params.len());
    for (i, (&a, &b)) in r1.final_params.iter().zip(r2.final_params.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "PINN param {} not bit-identical: {} vs {}",
            i,
            a,
            b,
        );
    }
}

#[test]
fn test_pinn_determinism_loss_trajectory() {
    let config = PinnConfig {
        layer_sizes: vec![1, 8, 8, 1],
        epochs: 30,
        lr: 1e-3,
        physics_weight: 1.0,
        boundary_weight: 10.0,
        seed: 42,
        n_collocation: 10,
        n_data: 8,
        fd_eps: 1e-4,
    };

    let r1 = pinn_harmonic_train(&config);
    let r2 = pinn_harmonic_train(&config);

    for (h1, h2) in r1.history.iter().zip(r2.history.iter()) {
        assert_eq!(
            h1.total_loss.to_bits(),
            h2.total_loss.to_bits(),
            "PINN loss mismatch at epoch {}",
            h1.epoch,
        );
    }
}

#[test]
fn test_pinn_different_seeds_different_results() {
    let config1 = PinnConfig {
        seed: 42,
        epochs: 20,
        n_collocation: 8,
        n_data: 8,
        ..PinnConfig::default()
    };
    let config2 = PinnConfig {
        seed: 99,
        epochs: 20,
        n_collocation: 8,
        n_data: 8,
        ..PinnConfig::default()
    };

    let r1 = pinn_harmonic_train(&config1);
    let r2 = pinn_harmonic_train(&config2);

    assert_ne!(
        r1.final_params, r2.final_params,
        "Different seeds should produce different PINN params"
    );
}

// ---------------------------------------------------------------------------
// MLP Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_mlp_init_determinism_bit_identical() {
    let mut g1 = GradGraph::new();
    let (_, p1) = mlp_init(&mut g1, &[1, 16, 16, 1], Activation::Tanh, Activation::None, 42);

    let mut g2 = GradGraph::new();
    let (_, p2) = mlp_init(&mut g2, &[1, 16, 16, 1], Activation::Tanh, Activation::None, 42);

    for (&i1, &i2) in p1.iter().zip(p2.iter()) {
        let v1 = g1.tensor(i1).to_vec();
        let v2 = g2.tensor(i2).to_vec();
        assert_eq!(v1.len(), v2.len());
        for (j, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "MLP weight {} elem {} not bit-identical",
                i1,
                j,
            );
        }
    }
}

#[test]
fn test_mlp_forward_determinism_bit_identical() {
    let run = |seed: u64, x_val: f64| -> Vec<u64> {
        let mut graph = GradGraph::new();
        let (mlp, _) =
            mlp_init(&mut graph, &[1, 16, 16, 1], Activation::Tanh, Activation::None, seed);
        let x = graph.input(Tensor::from_vec_unchecked(vec![x_val], &[1, 1]));
        let y = mlp_forward(&mut graph, &mlp, x);
        let out = graph.tensor(y).to_vec();
        out.iter().map(|&v| v.to_bits()).collect()
    };

    // Same seed, same input → identical bits
    let a = run(42, 0.5);
    let b = run(42, 0.5);
    assert_eq!(a, b, "MLP forward not bit-identical");

    // Different input → different output
    let c = run(42, 0.7);
    assert_ne!(a, c, "Different inputs should produce different outputs");
}

// ---------------------------------------------------------------------------
// Data Generation Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_heat_data_generation_determinism() {
    let (x1, u1) = heat_1d_generate_data(50, 0.01, 42);
    let (x2, u2) = heat_1d_generate_data(50, 0.01, 42);

    for (i, (&a, &b)) in x1.iter().zip(x2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "x[{}] not bit-identical", i);
    }
    for (i, (&a, &b)) in u1.iter().zip(u2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "u[{}] not bit-identical", i);
    }
}

#[test]
fn test_uniform_grid_determinism() {
    let g1 = uniform_grid(0.0, 1.0, 100);
    let g2 = uniform_grid(0.0, 1.0, 100);
    assert_eq!(g1, g2);
}

#[test]
fn test_lhs_grid_determinism_bit_identical() {
    let g1 = lhs_grid_1d(0.0, 1.0, 50, 42);
    let g2 = lhs_grid_1d(0.0, 1.0, 50, 42);

    for (i, (&a, &b)) in g1.iter().zip(g2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "LHS point {} not bit-identical", i);
    }
}

#[test]
fn test_lhs_grid_different_seeds() {
    let g1 = lhs_grid_1d(0.0, 1.0, 50, 42);
    let g2 = lhs_grid_1d(0.0, 1.0, 50, 99);
    assert_ne!(g1, g2, "Different seeds should produce different LHS grids");
}

// ---------------------------------------------------------------------------
// Visualization Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_ascii_plot_determinism() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    let p1 = ascii_plot(&x, &y, 40, 10, "test");
    let p2 = ascii_plot(&x, &y, 40, 10, "test");
    assert_eq!(p1, p2, "ASCII plot should be deterministic");
}
