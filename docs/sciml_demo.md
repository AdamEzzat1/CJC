# CJC SciML Demonstration

## Deterministic Physics-Informed ML + PINN (Zero External Dependencies)

This document describes CJC's Physics-Informed Scientific ML infrastructure,
demonstrating deterministic neural PDE solving with full autodiff support.

---

## Problem A: PIML — 1D Steady-State Heat Equation

### Mathematical Formulation

**PDE:** `u_xx = f(x)` on `[0, 1]`

**Source:** `f(x) = -pi^2 * sin(pi*x)`

**Boundary Conditions:** `u(0) = 0`, `u(1) = 0`

**Analytical Solution:** `u(x) = sin(pi*x)`

### Model

Polynomial regression: `u_approx(x) = sum(a_i * x^i, i=0..degree)`

**Loss Function:**

```
L = MSE_data + lambda_phys * MSE_residual + lambda_bnd * (u(0)^2 + u(1)^2)
```

Where:
- `MSE_data = mean((u_approx(x_i) - u_data_i)^2)` over training points
- `MSE_residual = mean((u_xx_approx(x_j) - f(x_j))^2)` over collocation points
- Boundary loss enforces Dirichlet BCs

### Optimizer

Adam with lr=1e-3, beta1=0.9, beta2=0.999.

### Implementation

Module: `cjc_ad::pinn::piml_heat_1d_train`

Key features:
- Analytical gradient computation (polynomial derivatives computed exactly)
- Kahan summation for all loss reductions
- Deterministic data generation via SplitMix64 RNG

---

## Problem B: PINN — 1D Harmonic Oscillator

### Mathematical Formulation

**PDE:** `u_xx + u = 0` on `[0, pi]`

**Boundary Conditions:** `u(0) = 0`, `u(pi) = 0`

**Analytical Solution:** `u(x) = sin(x)`

### Model Architecture

```
MLP: Input(1) -> Dense(16, tanh) -> Dense(16, tanh) -> Dense(1, linear)
```

Total parameters: 321 (weights + biases across 3 layers).

Initialization: Xavier uniform with deterministic seed.

### Physics Loss (Finite-Difference u_xx)

For each collocation point `x`:

```
u_xx(x) ~ (u(x+eps) - 2*u(x) + u(x-eps)) / eps^2
residual = u_xx + u
```

Using `eps = 1e-4` for central differences.

### Composite Loss

```
L = MSE_data + lambda_phys * MSE(residual^2) + lambda_bnd * (u(0)^2 + u(pi)^2)
```

Gradients are computed via reverse-mode autodiff (`GradGraph.backward`),
flowing through the entire computation graph including finite-difference
stencils.

### Optimizer

Adam with lr=1e-3.

### Implementation

Module: `cjc_ad::pinn::pinn_harmonic_train`

Key features:
- Full GradGraph-based computation (all ops differentiable)
- Central finite differences for second-order PDE terms
- Graph rebuilt each epoch for clean gradient flow
- Kahan-deterministic Adam updates

---

## Verification Results

### 1. Mathematical Correctness

| Metric | PIML (Heat) | PINN (Harmonic) |
|--------|-------------|-----------------|
| Loss decreases | Yes | Yes |
| Boundary satisfaction | u(0), u(1) near 0 | u(0), u(pi) near 0 |
| Gradient check (FD vs AD) | Pass (rel_err < 5%) | Pass |
| All loss components finite | Yes | Yes |

### 2. PDE Residual

Physics residual is tracked each epoch and decreases during training.
Mean squared residual is reported in `PinnResult.mean_residual`.

### 3. Determinism Proof

Both PIML and PINN training are **bit-identical** across runs:
- Same seed produces identical parameter vectors (checked via `f64::to_bits()`)
- Loss trajectories match at every epoch (bit-level comparison)
- Gradient norms match at every epoch

This is verified by `tests/sciml_determinism.rs`.

### 4. Gradient Correctness

The `test_mlp_gradient_finite_diff_check` test verifies that autodiff
gradients match central finite differences with:
- Relative error < 5%
- Or absolute error < 1e-4

### 5. Test Summary

```
tests/pinn_correctness.rs:  11 tests (solution, boundary, gradient, visualization)
tests/sciml_determinism.rs: 13 tests (bit-identical params, loss, sampling)
cjc-ad pinn module:         21 unit tests (MLP, data, poly, loss, training)
                            ---
Total:                      45 SciML tests, 0 failures
```

---

## API Reference

### PIML Training

```rust
use cjc_ad::pinn::piml_heat_1d_train;

let result = piml_heat_1d_train(
    degree,          // polynomial degree
    n_data,          // number of training points
    n_colloc,        // number of collocation points
    noise_std,       // Gaussian noise on training data
    epochs,          // number of Adam steps
    lr,              // learning rate
    physics_weight,  // lambda_phys
    boundary_weight, // lambda_bnd
    seed,            // RNG seed (u64)
);
// result.final_params: Vec<f64>  -- polynomial coefficients
// result.history: Vec<TrainLog>  -- per-epoch loss breakdown
// result.l2_error: Option<f64>   -- L2 error vs analytical
// result.max_error: Option<f64>  -- max pointwise error
// result.mean_residual: f64      -- RMS PDE residual
```

### PINN Training

```rust
use cjc_ad::pinn::{PinnConfig, pinn_harmonic_train};

let config = PinnConfig {
    layer_sizes: vec![1, 32, 32, 1],
    epochs: 500,
    lr: 1e-3,
    physics_weight: 1.0,
    boundary_weight: 10.0,
    seed: 42,
    n_collocation: 50,
    n_data: 20,
    fd_eps: 1e-4,
};
let result = pinn_harmonic_train(&config);
```

### MLP Utilities

```rust
use cjc_ad::pinn::{mlp_init, mlp_forward, Activation, data_loss_mse};
use cjc_ad::GradGraph;

let mut graph = GradGraph::new();
let (mlp, param_indices) = mlp_init(
    &mut graph,
    &[1, 32, 32, 1],    // layer sizes
    Activation::Tanh,     // hidden activation
    Activation::None,     // output activation
    42,                   // seed
);

let x = graph.input(Tensor::from_vec_unchecked(vec![0.5], &[1, 1]));
let y = mlp_forward(&mut graph, &mlp, x);
let target = graph.input(Tensor::from_vec_unchecked(vec![1.0], &[1, 1]));
let loss = data_loss_mse(&mut graph, y, target);

graph.backward(loss);
// gradients now available via graph.grad(param_indices[i])
```

### Sampling

```rust
use cjc_ad::pinn::{uniform_grid, lhs_grid_1d};

let grid = uniform_grid(0.0, 1.0, 100);          // 100 evenly spaced points
let lhs = lhs_grid_1d(0.0, 1.0, 100, seed);      // Latin Hypercube on [0,1]
```

### Visualization

```rust
use cjc_ad::pinn::{ascii_plot, plot_loss_history};

let plot = ascii_plot(&x_vals, &y_vals, 60, 15, "u(x) vs x");
let loss_plot = plot_loss_history(&result.history, 60, 15);
println!("{}", plot);
println!("{}", loss_plot);
```

---

## Architecture Notes

### Memory Layout

- All tensors are contiguous `Vec<f64>` backed by COW `Buffer<f64>`
- No heap allocations during polynomial evaluation (stack arithmetic)
- GradGraph nodes use `Rc<RefCell<>>` for shared ownership during backprop
- Graph is rebuilt each PINN epoch (clean allocation, no stale state)

### Determinism Guarantees

1. **RNG:** SplitMix64 with explicit seed — no platform variance
2. **Summation:** Kahan accumulator for all loss reductions
3. **Ordering:** BTreeMap everywhere — no hash-based iteration
4. **No SIMD FMA:** Avoids fused multiply-add for bit-identical cross-platform results
5. **Single-threaded:** Training loop is sequential (deterministic by construction)

### Performance Considerations

- PIML (polynomial): ~5000 epochs in <200ms (pure scalar arithmetic)
- PINN (MLP [1,16,16,1]): ~100 epochs in ~30s (graph rebuild per epoch)
- For production PINN: consider caching graph topology and using `set_tensor` + `reforward`

---

## Files

| File | Purpose |
|------|---------|
| `crates/cjc-ad/src/pinn.rs` | PINN/PIML infrastructure module |
| `tests/pinn_correctness.rs` | Correctness tests (accuracy, gradients, boundary) |
| `tests/sciml_determinism.rs` | Determinism tests (bit-identical runs) |
| `examples/piml_demo.cjc` | PIML example script |
| `examples/pinn_demo.cjc` | PINN example script |
| `docs/sciml_demo.md` | This documentation |
