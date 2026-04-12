---
title: PINN Benchmark Results
tags: [advanced, ml, physics, benchmark]
status: Active
---

# PINN Benchmark Results

Benchmark results for eleven canonical PDE problems in `crates/cjc-ad/src/pinn.rs`. Reference implementations from maziarraissi/PINNs, PINNacle, and DeepXDE.

## Problems

### Burgers' Equation (Hyperbolic, 1D)

- **PDE**: `u_t + u·u_x = ν·u_xx`, ν = 0.01/π
- **Architecture**: [2, 20, 20, 20, 1] with tanh activations
- **Optimizer**: Adam with cosine LR annealing + boundary weight ramp
- **Derivative method**: Central finite differences (eps = 1e-4)
- **Determinism**: Bit-identical across runs (verified via `test_burgers_determinism`)

### 2D Poisson Equation (Elliptic)

- **PDE**: `u_xx + u_yy = -2π²sin(πx)sin(πy)`
- **Architecture**: [2, 20, 20, 1] with tanh activations
- **Collocation sampling**: Latin Hypercube Sampling on [0,1]²
- **Determinism**: Bit-identical (verified via `test_poisson_determinism`)

### 1D Heat Equation (Parabolic)

- **PDE**: `u_t = α·u_xx`, α = 0.01
- **Architecture**: [2, 20, 20, 1] with tanh activations
- **Analytical solution**: `u(x,t) = exp(-απ²t)·sin(πx)`
- **Determinism**: Bit-identical (verified via `test_heat_nn_determinism`)

### 1D Wave Equation (Hyperbolic)

- **PDE**: `u_tt = c²·u_xx` on `x ∈ [0,1], t ∈ [0,1]`
- **Exact**: `u(x,t) = sin(πx)·cos(cπt)`
- **Determinism**: Verified via `test_wave_determinism`

### 2D Helmholtz Equation (Elliptic)

- **PDE**: `u_xx + u_yy + k²·u = f(x,y)` on [0,1]²
- **Exact**: `sin(πx)sin(πy)` with manufactured source
- **Determinism**: Verified via `test_helmholtz_determinism`

### Diffusion-Reaction (Parabolic + Reaction)

- **PDE**: `u_t = D·u_xx + R·u·(1-u)` with Neumann BCs
- **IC**: Gaussian bump `exp(-50(x-0.5)²)`

### Allen-Cahn Equation (Phase-field)

- **PDE**: `u_t = ε²·u_xx + u - u³`, periodic BCs

### KdV Equation (Dispersive)

- **PDE**: `u_t + 6·u·u_x + u_xxx = 0` (soliton dynamics)
- **IC**: `0.5·sech²(x/2)` soliton

### Nonlinear Schrödinger Equation (Complex)

- **PDE**: `iψ_t + 0.5ψ_xx + |ψ|²ψ = 0` — split real/imag
- Network outputs 2 values: `[u(real), v(imag)]`

### 2D Navier-Stokes (Stream Function)

- **PDE**: Steady vorticity transport via stream function ψ
- Enforces divergence-free velocity by construction
- Lid-driven cavity BC

### 2D Burgers' Equation

- **PDE**: `u_t + u·u_x + u·u_y = ν·(u_xx + u_yy)`, 3D input (x,y,t)

## Test Coverage

| Category | Count | Files |
|---|---|---|
| Unit tests (PDE convergence, IC/BC, determinism) | 14 | `tests/pinn_pde_problems.rs` |
| Domain geometry tests | 5 | `tests/pinn_pde_problems.rs` |
| Hard BC enforcement | 1 | `tests/pinn_pde_problems.rs` |
| Adaptive refinement | 2 | `tests/pinn_pde_problems.rs` |
| Batch forward | 1 | `tests/pinn_pde_problems.rs` |
| Property tests (proptest) | 4+5 | `pinn_pde_problems.rs` + `pinn_expansion_tests.rs` |
| Fuzz tests (bolero) | 4+4 | `pinn_pde_problems.rs` + `pinn_expansion_tests.rs` |
| Expansion tests (activations, L-BFGS, BCs, domains, new solvers, inverse) | 41 | `tests/pinn_expansion_tests.rs` |
| Parity tests (eval vs MIR-exec) | 13 | `tests/pinn_parity.rs` |
| Original PINN tests | 8 | `tests/pinn_correctness.rs` |
| Unit tests in `pinn.rs` | 20 | `cjc-ad/src/pinn.rs` |
| **Total** | **113** | |

## CJC-Lang Builtins

All eleven builtins are wired in both executors (cjc-eval + cjc-mir-exec) with identical dispatch logic. Results are returned as `Value::Struct` with fields: `l2_error`, `max_error`, `mean_residual`, `n_epochs`, `loss_history`, `final_params`.

```
pinn_train_burgers(epochs, lr, n_colloc, nu, seed)        -> PinnResult
pinn_train_poisson(epochs, lr, n_colloc, seed)             -> PinnResult
pinn_train_heat(epochs, lr, n_colloc, alpha, seed)         -> PinnResult
pinn_train_wave(epochs, lr, n_colloc, c, seed)             -> PinnResult
pinn_train_helmholtz(epochs, lr, n_colloc, k, seed)        -> PinnResult
pinn_train_diffreact(epochs, lr, n_colloc, D, R, seed)     -> PinnResult
pinn_train_allen_cahn(epochs, lr, n_colloc, epsilon, seed) -> PinnResult
pinn_train_kdv(epochs, lr, n_colloc, seed)                 -> PinnResult
pinn_train_schrodinger(epochs, lr, n_colloc, seed)         -> PinnResult
pinn_train_navier_stokes(epochs, lr, n_colloc, nu, seed)   -> PinnResult
pinn_train_burgers_2d(epochs, lr, n_colloc, nu, seed)      -> PinnResult
```

### Inverse Problem

```
inverse_diffusion_train(config, obs_x, obs_t, obs_u) -> InversePinnResult
```

Discovers unknown diffusion coefficient λ from observational data. Returns `discovered_params` and `param_names`.

## Infrastructure

- **PinnDomain** enum: `Interval1D`, `Rectangle2D`, `SpaceTime1D`, `Disk`, `LShape`, `Polygon`, `Cuboid3D`, `SpaceTime2D`
- **Boundary conditions**: Dirichlet (value), Neumann (derivative), Robin (mixed), Periodic (matching)
- **Hard BC**: distance-function approach `u(x) = g(x) + d(x)*NN(x)`
- **RAR**: residual-based adaptive refinement (select top-K% by |residual|)
- **Activations**: Tanh, Sigmoid, ReLU, GELU, SiLU/Swish, ELU, SELU, SinAct, None
- **Optimizers**: Adam (default), L-BFGS (quasi-Newton), TwoStageOptimizer (Adam 80% + L-BFGS 20%)
- **Training features**: cosine LR annealing, boundary weight ramp (0-20% of epochs)

## Related

- [[PINN Support]]
- [[Autodiff]]
- [[Demonstrated Scientific Computing Capabilities]]
