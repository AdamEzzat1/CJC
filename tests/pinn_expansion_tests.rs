//! PINN Expansion Suite — tests for Phase C features:
//! - New activations (GELU, SiLU, ELU, SELU, SinAct)
//! - L-BFGS optimizer
//! - Boundary condition types (Neumann, Robin, Periodic)
//! - Domain extensions (Disk, LShape, Polygon, Cuboid3D, SpaceTime2D)
//! - New PDE solvers (Wave, Helmholtz, DiffReact, Allen-Cahn, KdV, Schrödinger, NS, Burgers2D)
//! - Inverse problem infrastructure

use cjc_ad::pinn::*;
use cjc_ad::GradGraph;
use cjc_runtime::tensor::Tensor;

// ═══════════════════════════════════════════════════════════════════
// Activation Function Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_gelu_forward_backward() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec_unchecked(vec![-1.0, 0.0, 1.0, 2.0], &[4]));
    let y = g.gelu(x);
    let loss = g.sum(y);
    g.backward(loss);
    let out = g.tensor(y).to_vec();
    // GELU(0) = 0, GELU(x) ≈ x for large x
    assert!((out[1]).abs() < 1e-10, "GELU(0) should be 0, got {}", out[1]);
    assert!(out[3] > 1.9, "GELU(2) should be close to 2, got {}", out[3]);
    assert!(g.grad(x).is_some(), "Should have gradient");
}

#[test]
fn test_silu_forward_backward() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec_unchecked(vec![-2.0, 0.0, 1.0, 3.0], &[4]));
    let y = g.silu(x);
    let loss = g.sum(y);
    g.backward(loss);
    let out = g.tensor(y).to_vec();
    // SiLU(0) = 0
    assert!((out[1]).abs() < 1e-10, "SiLU(0) should be 0");
    // SiLU(x) ≈ x for large x
    assert!(out[3] > 2.8, "SiLU(3) should be close to 3");
    assert!(g.grad(x).is_some());
}

#[test]
fn test_elu_forward_backward() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec_unchecked(vec![-2.0, 0.0, 1.0, 3.0], &[4]));
    let y = g.elu(x);
    let loss = g.sum(y);
    g.backward(loss);
    let out = g.tensor(y).to_vec();
    assert!((out[2] - 1.0).abs() < 1e-10, "ELU(1) = 1");
    assert!(out[0] < 0.0, "ELU(-2) < 0");
    assert!(out[0] > -1.0, "ELU(-2) > -1 (approaches -1 asymptotically)");
}

#[test]
fn test_selu_forward_backward() {
    let mut g = GradGraph::new();
    let x = g.parameter(Tensor::from_vec_unchecked(vec![-1.0, 0.0, 1.0], &[3]));
    let y = g.selu(x);
    let loss = g.sum(y);
    g.backward(loss);
    let out = g.tensor(y).to_vec();
    // SELU(1) = λ·1 ≈ 1.0507
    assert!((out[2] - 1.0507009873554804).abs() < 1e-6, "SELU(1) ≈ λ");
    assert!(out[0] < 0.0, "SELU(-1) < 0");
}

#[test]
fn test_sinact_via_mlp_layer() {
    // SinAct is used via MlpLayer, not as standalone GradOp
    let mut g = GradGraph::new();
    let input = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0], &[1, 2]));
    let weight = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let bias = g.parameter(Tensor::from_vec_unchecked(vec![0.0, 0.0], &[2]));
    let out = g.mlp_layer(input, weight, bias, Activation::SinAct);
    let loss = g.sum(out);
    g.backward(loss);
    let result = g.tensor(out).to_vec();
    assert!((result[0] - 1.0_f64.sin()).abs() < 1e-10);
    assert!((result[1] - 2.0_f64.sin()).abs() < 1e-10);
    assert!(g.grad(weight).is_some());
}

#[test]
fn test_activation_mlp_forward_all() {
    // Test that mlp_init + mlp_forward works with each activation
    for act in &[Activation::Gelu, Activation::Silu, Activation::Elu, Activation::Selu, Activation::SinAct] {
        let mut g = GradGraph::new();
        let (mlp, params) = mlp_init(&mut g, &[1, 4, 1], *act, Activation::None, 42);
        let inp = g.input(Tensor::from_vec_unchecked(vec![0.5], &[1, 1]));
        let out = mlp_forward(&mut g, &mlp, inp);
        let loss = g.sum(out);
        g.backward(loss);
        let val = g.tensor(out).to_vec()[0];
        assert!(val.is_finite(), "MLP with {:?} should produce finite output", act);
        for &p in &params {
            assert!(g.grad(p).is_some(), "Param {} should have gradient with {:?}", p, act);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// L-BFGS Optimizer Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_lbfgs_quadratic_convergence() {
    // Minimize f(x) = (x-3)² + (y-7)² — L-BFGS should converge quickly
    let mut state = LbfgsState::new(10, 0.1);
    let mut params = vec![0.0, 0.0];
    let target = vec![3.0, 7.0];
    for _ in 0..50 {
        let grad: Vec<f64> = params.iter().zip(&target).map(|(p, t)| 2.0 * (p - t)).collect();
        state.step(&mut params, &grad);
    }
    assert!((params[0] - 3.0).abs() < 0.1, "x should be near 3, got {}", params[0]);
    assert!((params[1] - 7.0).abs() < 0.1, "y should be near 7, got {}", params[1]);
}

#[test]
fn test_lbfgs_deterministic() {
    let mut s1 = LbfgsState::new(5, 0.01);
    let mut s2 = LbfgsState::new(5, 0.01);
    let mut p1 = vec![1.0, 2.0, 3.0];
    let mut p2 = vec![1.0, 2.0, 3.0];
    let grads = vec![0.5, -0.3, 0.8];
    for _ in 0..10 {
        s1.step(&mut p1, &grads);
        s2.step(&mut p2, &grads);
    }
    assert_eq!(p1, p2, "L-BFGS should be deterministic");
}

#[test]
fn test_two_stage_optimizer() {
    let mut opt = TwoStageOptimizer::new(2, 0.01, 100, 0.8);
    let mut params = vec![0.0, 0.0];
    for epoch in 0..100 {
        let grad = vec![2.0 * params[0], 2.0 * params[1]]; // f = x² + y²
        let is_lbfgs = opt.step(&mut params, &grad, epoch);
        if epoch < 80 { assert!(!is_lbfgs); }
        else { assert!(is_lbfgs); }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Boundary Condition Type Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_dirichlet_bc_loss() {
    let bc = BoundaryCondition::Dirichlet {
        points: vec![0.0, 1.0],
        values: vec![0.0, 0.0],
        dim: 1,
    };
    // Network that returns x → should have loss = 0² + 1² / 2 = 0.5
    let loss = bc_loss(&bc, &|pt: &[f64]| pt[0]);
    assert!((loss - 0.5).abs() < 1e-10, "Dirichlet loss should be 0.5, got {}", loss);
}

#[test]
fn test_neumann_bc_loss() {
    let bc = BoundaryCondition::Neumann {
        points: vec![0.0],
        values: vec![1.0], // Target: du/dn = 1
        normals: vec![1.0],
        dim: 1,
        fd_eps: 1e-4,
    };
    // Network u(x) = x → du/dx = 1 → loss should be ~0
    let loss = bc_loss(&bc, &|pt: &[f64]| pt[0]);
    assert!(loss < 1e-4, "Neumann loss for u=x with target du/dn=1 should be ~0, got {}", loss);
}

#[test]
fn test_robin_bc_loss() {
    let bc = BoundaryCondition::Robin {
        points: vec![0.0],
        values: vec![1.0], // α·u + β·du/dn = 1
        normals: vec![1.0],
        alpha: 1.0,
        beta: 1.0,
        dim: 1,
        fd_eps: 1e-4,
    };
    // u(x) = x → u(0)=0, du/dx(0)=1 → α·0 + β·1 = 1 → loss ≈ 0
    let loss = bc_loss(&bc, &|pt: &[f64]| pt[0]);
    assert!(loss < 1e-4, "Robin loss should be ~0, got {}", loss);
}

#[test]
fn test_periodic_bc_loss() {
    let bc = BoundaryCondition::Periodic {
        left_points: vec![0.0],
        right_points: vec![1.0],
        dim: 1,
    };
    // u(x) = sin(2πx) → u(0) = u(1) = 0 → loss = 0
    let loss = bc_loss(&bc, &|pt: &[f64]| (2.0 * std::f64::consts::PI * pt[0]).sin());
    assert!(loss < 1e-10, "Periodic loss for sin(2πx) should be ~0, got {}", loss);
}

// ═══════════════════════════════════════════════════════════════════
// Domain Extension Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_disk_domain_interior() {
    let d = PinnDomain::Disk { center: (0.0, 0.0), radius: 1.0 };
    let pts = d.sample_interior(100, 42);
    assert_eq!(pts.len(), 200);
    for i in 0..100 {
        let (x, y) = (pts[i * 2], pts[i * 2 + 1]);
        assert!(x * x + y * y < 1.0, "Point ({}, {}) should be inside disk", x, y);
    }
}

#[test]
fn test_disk_domain_boundary() {
    let d = PinnDomain::Disk { center: (0.0, 0.0), radius: 1.0 };
    let pts = d.sample_boundary(20, 42);
    for i in 0..20 {
        let (x, y) = (pts[i * 2], pts[i * 2 + 1]);
        let r = (x * x + y * y).sqrt();
        assert!((r - 1.0).abs() < 1e-10, "Boundary point should be on circle, r={}", r);
    }
}

#[test]
fn test_lshape_domain() {
    let d = PinnDomain::LShape;
    let pts = d.sample_interior(200, 42);
    for i in 0..pts.len() / 2 {
        let (x, y) = (pts[i * 2], pts[i * 2 + 1]);
        assert!(!(x > 0.5 && y > 0.5), "Point ({}, {}) should not be in upper-right quadrant", x, y);
    }
    assert_eq!(d.input_dim(), 2);
}

#[test]
fn test_polygon_domain() {
    // Unit square as polygon
    let d = PinnDomain::Polygon {
        vertices: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    };
    let pts = d.sample_interior(50, 42);
    for i in 0..pts.len() / 2 {
        let (x, y) = (pts[i * 2], pts[i * 2 + 1]);
        assert!(x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0, "Point ({}, {}) outside square", x, y);
    }
}

#[test]
fn test_cuboid_3d_domain() {
    let d = PinnDomain::Cuboid3D {
        x_range: (0.0, 1.0), y_range: (0.0, 1.0), z_range: (0.0, 1.0),
    };
    assert_eq!(d.input_dim(), 3);
    let pts = d.sample_interior(50, 42);
    assert_eq!(pts.len(), 150);
    for i in 0..50 {
        let (x, y, z) = (pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
        assert!(x >= 0.0 && x <= 1.0, "x out of range: {}", x);
        assert!(y >= 0.0 && y <= 1.0, "y out of range: {}", y);
        assert!(z >= 0.0 && z <= 1.0, "z out of range: {}", z);
    }
}

#[test]
fn test_spacetime_2d_domain() {
    let d = PinnDomain::SpaceTime2D {
        x_range: (0.0, 1.0), y_range: (0.0, 1.0), t_range: (0.0, 1.0),
    };
    assert_eq!(d.input_dim(), 3);
    let pts = d.sample_interior(30, 42);
    assert_eq!(pts.len(), 90);
}

#[test]
fn test_domain_deterministic() {
    let d = PinnDomain::Disk { center: (0.0, 0.0), radius: 1.0 };
    let pts1 = d.sample_interior(50, 42);
    let pts2 = d.sample_interior(50, 42);
    assert_eq!(pts1, pts2, "Domain sampling should be deterministic");
}

// ═══════════════════════════════════════════════════════════════════
// PDE Solver Tests (convergence + determinism)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_wave_trains() {
    let config = WaveConfig { epochs: 20, n_collocation: 8, n_ic: 10, n_bc: 5, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r = pinn_wave_train(&config);
    assert_eq!(r.history.len(), 20);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_wave_deterministic() {
    let config = WaveConfig { epochs: 10, n_collocation: 8, n_ic: 8, n_bc: 4, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r1 = pinn_wave_train(&config);
    let r2 = pinn_wave_train(&config);
    assert_eq!(r1.final_params, r2.final_params, "Wave should be deterministic");
}

#[test]
fn test_helmholtz_trains() {
    let config = HelmholtzConfig { epochs: 20, n_collocation: 8, n_bc: 5, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r = pinn_helmholtz_train(&config);
    assert_eq!(r.history.len(), 20);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_helmholtz_deterministic() {
    let config = HelmholtzConfig { epochs: 10, n_collocation: 8, n_bc: 4, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r1 = pinn_helmholtz_train(&config);
    let r2 = pinn_helmholtz_train(&config);
    assert_eq!(r1.final_params, r2.final_params);
}

#[test]
fn test_diffreact_trains() {
    let config = DiffReactConfig { epochs: 20, n_collocation: 8, n_ic: 8, n_bc: 4, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r = pinn_diffreact_train(&config);
    assert_eq!(r.history.len(), 20);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_allen_cahn_trains() {
    let config = AllenCahnConfig { epochs: 20, n_collocation: 8, n_ic: 8, n_bc: 4, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r = pinn_allen_cahn_train(&config);
    assert_eq!(r.history.len(), 20);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_kdv_trains() {
    let config = KdvConfig { epochs: 20, n_collocation: 8, n_ic: 8, n_bc: 4, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r = pinn_kdv_train(&config);
    assert_eq!(r.history.len(), 20);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_schrodinger_trains() {
    let config = SchrodingerConfig { epochs: 20, n_collocation: 8, n_ic: 8, n_bc: 4, layer_sizes: vec![2, 8, 8, 2], ..Default::default() };
    let r = pinn_schrodinger_train(&config);
    assert_eq!(r.history.len(), 20);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_navier_stokes_trains() {
    let config = NavierStokesConfig { epochs: 10, n_collocation: 4, n_bc: 4, layer_sizes: vec![2, 8, 1], ..Default::default() };
    let r = pinn_navier_stokes_train(&config);
    assert_eq!(r.history.len(), 10);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

#[test]
fn test_burgers_2d_trains() {
    let config = Burgers2DConfig { epochs: 10, n_collocation: 4, n_ic: 4, n_bc: 4, layer_sizes: vec![3, 8, 1], ..Default::default() };
    let r = pinn_burgers_2d_train(&config);
    assert_eq!(r.history.len(), 10);
    assert!(r.history.last().unwrap().total_loss.is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// Inverse Problem Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_inverse_diffusion_runs() {
    let config = InversePinnConfig { epochs: 20, n_collocation: 8, layer_sizes: vec![2, 8, 1], ..Default::default() };
    // Generate synthetic data with known λ = 0.05
    let true_lambda = 0.05;
    let n_obs = 10;
    let obs_x: Vec<f64> = (0..n_obs).map(|i| (i as f64 + 0.5) / n_obs as f64).collect();
    let obs_t: Vec<f64> = vec![0.5; n_obs];
    let obs_u: Vec<f64> = obs_x.iter().map(|&x| {
        (-true_lambda * std::f64::consts::PI * std::f64::consts::PI * 0.5).exp() * (std::f64::consts::PI * x).sin()
    }).collect();
    let r = inverse_diffusion_train(&config, &obs_x, &obs_t, &obs_u);
    assert_eq!(r.pinn_result.history.len(), 20);
    assert_eq!(r.discovered_params.len(), 1);
    assert!(r.discovered_params[0].is_finite(), "Lambda should be finite");
}

#[test]
fn test_inverse_diffusion_deterministic() {
    let config = InversePinnConfig { epochs: 10, n_collocation: 4, layer_sizes: vec![2, 4, 1], ..Default::default() };
    let obs_x = vec![0.25, 0.5, 0.75];
    let obs_t = vec![0.5, 0.5, 0.5];
    let obs_u = vec![0.1, 0.15, 0.1];
    let r1 = inverse_diffusion_train(&config, &obs_x, &obs_t, &obs_u);
    let r2 = inverse_diffusion_train(&config, &obs_x, &obs_t, &obs_u);
    assert_eq!(r1.discovered_params, r2.discovered_params, "Inverse should be deterministic");
}

// ═══════════════════════════════════════════════════════════════════
// Property Tests (proptest)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_gelu_bounded(x in -10.0f64..10.0) {
            let mut g = GradGraph::new();
            let inp = g.parameter(Tensor::from_vec_unchecked(vec![x], &[1]));
            let out = g.gelu(inp);
            let val = g.tensor(out).to_vec()[0];
            // GELU is bounded below by ≈ -0.17 and above by x for large x
            prop_assert!(val.is_finite());
            prop_assert!(val >= -1.0, "GELU({}) = {} should be >= -1", x, val);
        }

        #[test]
        fn prop_silu_bounded(x in -10.0f64..10.0) {
            let mut g = GradGraph::new();
            let inp = g.parameter(Tensor::from_vec_unchecked(vec![x], &[1]));
            let out = g.silu(inp);
            let val = g.tensor(out).to_vec()[0];
            prop_assert!(val.is_finite());
            prop_assert!(val >= -0.5, "SiLU({}) = {} should be >= -0.5", x, val);
        }

        #[test]
        fn prop_elu_continuous(x in -5.0f64..5.0) {
            let mut g = GradGraph::new();
            let inp = g.parameter(Tensor::from_vec_unchecked(vec![x], &[1]));
            let out = g.elu(inp);
            let val = g.tensor(out).to_vec()[0];
            prop_assert!(val.is_finite());
            if x > 0.0 { prop_assert!((val - x).abs() < 1e-10); }
            else { prop_assert!(val >= -1.0 && val <= 0.0); }
        }

        #[test]
        fn prop_lbfgs_finite(x0 in -10.0f64..10.0, x1 in -10.0f64..10.0) {
            let mut state = LbfgsState::new(5, 0.01);
            let mut params = vec![x0, x1];
            let grad = vec![2.0 * x0, 2.0 * x1];
            state.step(&mut params, &grad);
            prop_assert!(params[0].is_finite());
            prop_assert!(params[1].is_finite());
        }

        #[test]
        fn prop_disk_interior_valid(seed in 1u64..1000) {
            let d = PinnDomain::Disk { center: (0.5, 0.5), radius: 0.3 };
            let pts = d.sample_interior(10, seed);
            for i in 0..10 {
                let dx = pts[i * 2] - 0.5;
                let dy = pts[i * 2 + 1] - 0.5;
                prop_assert!((dx * dx + dy * dy).sqrt() < 0.3);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Fuzz Tests (bolero)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod fuzz_tests {
    use super::*;

    #[test]
    fn fuzz_gelu_no_panic() {
        bolero::check!().with_type::<f64>().for_each(|&x| {
            if x.is_finite() && x.abs() < 1e6 {
                let mut g = GradGraph::new();
                let inp = g.parameter(Tensor::from_vec_unchecked(vec![x], &[1]));
                let out = g.gelu(inp);
                let val = g.tensor(out).to_vec()[0];
                assert!(val.is_finite() || x.abs() > 100.0);
            }
        });
    }

    #[test]
    fn fuzz_silu_no_panic() {
        bolero::check!().with_type::<f64>().for_each(|&x| {
            if x.is_finite() && x.abs() < 1e6 {
                let mut g = GradGraph::new();
                let inp = g.parameter(Tensor::from_vec_unchecked(vec![x], &[1]));
                let out = g.silu(inp);
                let val = g.tensor(out).to_vec()[0];
                assert!(val.is_finite() || x.abs() > 100.0);
            }
        });
    }

    #[test]
    fn fuzz_lbfgs_no_panic() {
        bolero::check!().with_type::<(f64, f64, f64)>().for_each(|&(x, g1, g2)| {
            if x.is_finite() && g1.is_finite() && g2.is_finite()
                && x.abs() < 1e6 && g1.abs() < 1e6 && g2.abs() < 1e6
            {
                let mut state = LbfgsState::new(3, 0.01);
                let mut params = vec![x, 0.0];
                let grad = vec![g1, g2];
                state.step(&mut params, &grad);
                assert!(params[0].is_finite());
                assert!(params[1].is_finite());
            }
        });
    }

    #[test]
    fn fuzz_bc_loss_no_panic() {
        bolero::check!().with_type::<(f64, f64)>().for_each(|&(v1, v2)| {
            if v1.is_finite() && v2.is_finite() && v1.abs() < 100.0 && v2.abs() < 100.0 {
                let bc = BoundaryCondition::Dirichlet {
                    points: vec![0.0, 1.0],
                    values: vec![v1, v2],
                    dim: 1,
                };
                let loss = bc_loss(&bc, &|pt: &[f64]| pt[0]);
                assert!(loss.is_finite());
            }
        });
    }
}
