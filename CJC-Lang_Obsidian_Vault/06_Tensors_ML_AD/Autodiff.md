---
title: Autodiff
tags: [runtime, ad]
status: Implemented (forward + reverse at runtime) / MIR integration Planned
---

# Autodiff

**Crate**: `cjc-ad` — `crates/cjc-ad/src/lib.rs` + `pinn.rs` (~55K).

## Two modes

### Forward mode — Dual numbers

```rust
struct Dual { value: f64, deriv: f64 }
```

Arithmetic operators are overloaded so that `Dual::new(x, 1.0)` propagates a derivative through any computation. Used for:
- Jacobian-vector products
- Low-dimensional gradients (when input dim ≤ output dim)
- Sensitivity analysis

### Reverse mode — ComputeGraph tape

```rust
struct ComputeGraph { nodes: Vec<TapeNode> }
```

Every operation is recorded as a tape node during the forward pass. After the forward pass finishes, a reverse traversal walks the tape in reverse topological order, computing gradients via chain rule. Used for:
- Neural network training
- High-dimensional gradients (when input dim >> output dim)
- PINN loss minimization

## PINN support

`crates/cjc-ad/src/pinn.rs` (~5,500 LOC) is a substantial physics-informed neural network module. It uses the autodiff machinery to compute PDE residuals as loss terms. Eleven PDE solvers, nine activation functions, three optimizers (Adam, L-BFGS, two-stage), four BC types, and eight domain geometries. See [[PINN Support]] for the full problem suite.

### Activations (9)

Tanh, Sigmoid, ReLU, GELU, SiLU/Swish, ELU, SELU, SinAct, None. Each has standalone `GradOp` variants (or reuses existing ops) with forward, backward, and reforward support. The fused `MlpLayer` op dispatches all nine activations in a single graph node.

### FD-based second derivatives for PDE residuals

PDE residuals (e.g., `u_xx` in the heat equation) require second derivatives of the network with respect to spatial coordinates. Rather than implementing graph-of-graphs second-order AD, we use **central finite differences** on the network's forward pass: `u_xx ≈ (u(x+ε) − 2u(x) + u(x−ε)) / ε²`. This requires 3 forward passes per collocation point per spatial dimension, all flowing through the GradGraph for backpropagation of the residual loss.

### Batch forward for PINN training

Each PINN training epoch evaluates the MLP at O(n_colloc) collocation points × O(5) FD stencil points. The graph is rebuilt each epoch for 2D-input problems (simpler than reforward indexing for multi-point graphs). The harmonic oscillator (1D input) uses the more efficient `set_tensor` + `reforward` path.

## Determinism

- Tape traversal is deterministic (Vec order, not hash order).
- All floating-point ops on duals use the same rules as on f64 (no FMA, no reassociation).
- Reverse-mode reductions use Kahan accumulation.

## What's NOT integrated

The CLAUDE.md roadmap calls out:
> **MIR Integration for Autodiff**: Ensure Automatic Differentiation (AD) integrates with MIR execution.

Currently, reverse-mode AD runs as a runtime-level library using the tape. The target is to lift it into [[MIR]] so that gradient flow participates in the optimizer (CSE of shared subexpressions, DCE of unused branches, etc.). This is **Planned** — see [[Roadmap]].

## Related

- [[Tensor Runtime]]
- [[ML Primitives]]
- [[PINN Support]]
- [[MIR]]
- [[Roadmap]]
