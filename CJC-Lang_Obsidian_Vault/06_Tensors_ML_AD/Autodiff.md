---
title: Autodiff
tags: [runtime, ad]
status: Implemented (forward + reverse at runtime) / MIR integration Planned
---

# Autodiff

**Crate**: `cjc-ad` — `crates/cjc-ad/src/lib.rs` + `pinn.rs` (~44K).

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

`crates/cjc-ad/src/pinn.rs` is a substantial physics-informed neural network module. It uses the autodiff machinery to compute PDE residuals as loss terms. See [[PINN Support]] and the demo `08_pinn_heat_equation.cjcl`.

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
