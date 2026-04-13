---
title: PINN Support
tags: [advanced, ml, physics, pde]
status: Implemented
---

# PINN Support

**Source**: `crates/cjc-ad/src/pinn.rs` (~5,500 LOC).

## Summary

Physics-Informed Neural Networks: train neural networks whose loss function includes a **PDE residual term**. PINNs learn solutions to differential equations by minimizing loss over both boundary conditions and interior residuals.

## Why it lives in `cjc-ad`

PINNs are autodiff-native: you compute the network's output, take derivatives with respect to its inputs (the spatial/temporal coordinates), substitute those derivatives into the PDE's left-hand side, and minimize the squared residual. That requires nested differentiation — derivatives of the network, and derivatives of the loss with respect to the network's weights. Both are handled by [[Autodiff]].

## PDE Problem Suite

Eleven PDE benchmarks are implemented as Rust-level training functions and exposed as CJC-Lang builtins. Reference repos: maziarraissi/PINNs, PINNacle, DeepXDE.

### Burgers' Equation (Hyperbolic, 1D)

- **PDE**: `u_t + u·u_x = ν·u_xx` on `x ∈ [-1,1], t ∈ [0,1]`
- **IC**: `u(x,0) = -sin(πx)`, **BCs**: `u(±1,t) = 0`
- **Builtin**: `pinn_train_burgers(epochs, lr, n_colloc, nu, seed)`

### 2D Poisson Equation (Elliptic)

- **PDE**: `u_xx + u_yy = -2π²sin(πx)sin(πy)` on `[0,1]²`
- **Exact**: `u(x,y) = sin(πx)sin(πy)`
- **Builtin**: `pinn_train_poisson(epochs, lr, n_colloc, seed)`

### 1D Heat Equation (Parabolic)

- **PDE**: `u_t = α·u_xx` on `x ∈ [0,1], t ∈ [0,1]`
- **Exact**: `u(x,t) = exp(-απ²t)·sin(πx)`
- **Builtin**: `pinn_train_heat(epochs, lr, n_colloc, alpha, seed)`

### 1D Wave Equation (Hyperbolic)

- **PDE**: `u_tt = c²·u_xx` on `x ∈ [0,1], t ∈ [0,1]`
- **Exact**: `u(x,t) = sin(πx)·cos(cπt)`
- **Builtin**: `pinn_train_wave(epochs, lr, n_colloc, c, seed)`

### 2D Helmholtz Equation (Elliptic)

- **PDE**: `u_xx + u_yy + k²·u = f(x,y)` on `[0,1]²`
- **Exact**: `sin(πx)sin(πy)` with manufactured source
- **Builtin**: `pinn_train_helmholtz(epochs, lr, n_colloc, k, seed)`

### Diffusion-Reaction (Parabolic + Reaction)

- **PDE**: `u_t = D·u_xx + R·u·(1-u)` with Neumann BCs
- **IC**: Gaussian bump `exp(-50(x-0.5)²)`
- **Builtin**: `pinn_train_diffreact(epochs, lr, n_colloc, D, R, seed)`

### Allen-Cahn Equation (Phase-field)

- **PDE**: `u_t = ε²·u_xx + u - u³`, periodic BCs
- **Builtin**: `pinn_train_allen_cahn(epochs, lr, n_colloc, epsilon, seed)`

### KdV Equation (Dispersive)

- **PDE**: `u_t + 6·u·u_x + u_xxx = 0` (soliton dynamics)
- **IC**: `0.5·sech²(x/2)` soliton
- **Builtin**: `pinn_train_kdv(epochs, lr, n_colloc, seed)`

### Nonlinear Schrodinger Equation (Complex)

- **PDE**: `iψ_t + 0.5ψ_xx + |ψ|²ψ = 0` — split real/imag
- Network outputs 2 values: `[u(real), v(imag)]`
- **Builtin**: `pinn_train_schrodinger(epochs, lr, n_colloc, seed)`

### 2D Navier-Stokes (Stream Function)

- **PDE**: Steady vorticity transport via stream function ψ
- Enforces divergence-free velocity by construction
- Lid-driven cavity BC
- **Builtin**: `pinn_train_navier_stokes(epochs, lr, n_colloc, nu, seed)`

### 2D Burgers' Equation

- **PDE**: `u_t + u·u_x + u·u_y = ν·(u_xx + u_yy)`, 3D input (x,y,t)
- **Builtin**: `pinn_train_burgers_2d(epochs, lr, n_colloc, nu, seed)`

## Inverse Problem Infrastructure

- `inverse_diffusion_train(config, obs_x, obs_t, obs_u)` — discovers unknown diffusion coefficient λ from observational data
- Simultaneously learns solution field AND PDE parameters
- Returns `InversePinnResult` with `discovered_params` and `param_names`

## Activation Functions

Nine activations supported in both standalone GradOps and fused MlpLayer:

| Activation | Formula | Use case |
|---|---|---|
| Tanh | tanh(x) | Default PINN hidden |
| Sigmoid | 1/(1+e^{-x}) | Classification |
| ReLU | max(0,x) | Deep networks |
| GELU | x·Φ(x) | Transformers |
| SiLU/Swish | x·σ(x) | Modern CNNs |
| ELU | x if x>0, e^x-1 | Smooth ReLU |
| SELU | λ·ELU(α,x) | Self-normalizing |
| SinAct | sin(x) | PINNs with oscillatory solutions |
| None | identity | Output layer |

## Optimizers

- **Adam** — standard first-order optimizer (default)
- **L-BFGS** — quasi-Newton, second-order convergence. `LbfgsState` with two-loop recursion
- **TwoStageOptimizer** — Raissi pattern: Adam for 80% + L-BFGS for 20%

## Boundary Condition Types

- **Dirichlet** — prescribed values: `u(x_b) = g`
- **Neumann** — prescribed derivatives: `∂u/∂n = h` (via FD along normal)
- **Robin** — mixed: `α·u + β·∂u/∂n = g`
- **Periodic** — matching: `u(x_left) = u(x_right)`

## Domain Geometry (`PinnDomain`)

| Variant | Dim | Description |
|---|---|---|
| `Interval1D` | 1 | [a, b] |
| `Rectangle2D` | 2 | [x0,x1] × [y0,y1] |
| `SpaceTime1D` | 2 | x ∈ [x0,x1], t ∈ [t0,t1] |
| `Disk` | 2 | Circle with center + radius |
| `LShape` | 2 | [0,1]² minus [0.5,1]×[0.5,1] |
| `Polygon` | 2 | Convex polygon (ray-casting) |
| `Cuboid3D` | 3 | [x0,x1] × [y0,y1] × [z0,z1] |
| `SpaceTime2D` | 3 | (x,y) ∈ rect, t ∈ [t0,t1] |

Methods: `sample_interior(n, seed)`, `sample_boundary(n, seed)`, `is_boundary(point, tol)`, `input_dim()`

### Hard Boundary Enforcement

- `hard_bc_1d(graph, nn_output, x, a, b, g_a, g_b)` — distance-function approach
- Guarantees exact Dirichlet BCs: `u(x) = g(x) + d(x)·NN(x)` where `d(x)=0` on boundary

### Residual-Based Adaptive Refinement (RAR)

- `adaptive_refine(existing, candidates, residuals, fraction, dim)` — DeepXDE-style
- Deterministic sorting via `to_bits()` for NaN stability

### Derivative Computation

PDE residuals use **central finite differences** on the network's forward pass. This avoids requiring second-order autodiff (graph-of-graphs). Trade-off: O(ε²) approximation error but zero architectural changes to [[Autodiff|GradGraph]].

## Tests

- `tests/pinn_correctness.rs` — 8 original PINN correctness tests
- `tests/pinn_pde_problems.rs` — 31 tests: unit (convergence, IC/BC, determinism), property (proptest), fuzz
- `tests/pinn_expansion_tests.rs` — 41 tests: activations, L-BFGS, BCs, domains, new solvers, inverse, proptest, fuzz
- `tests/pinn_parity.rs` — 13 executor parity tests (eval vs MIR-exec)
- `cjc-ad/src/pinn.rs` mod tests — 20 unit tests (MLP, sampling, training)
- Total: **113 PINN-related tests**

## Related

- [[Autodiff]]
- [[ODE Integration]]
- [[ML Primitives]]
- [[Advanced Computing in CJC-Lang]]
- [[PINN Benchmark Results]]
