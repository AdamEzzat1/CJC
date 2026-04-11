---
title: PINN Support
tags: [advanced, ml, physics]
status: Implemented
---

# PINN Support

**Source**: `crates/cjc-ad/src/pinn.rs` (~44K).

## Summary

Physics-Informed Neural Networks: train neural networks whose loss function includes a **PDE residual term**. PINNs learn solutions to differential equations by minimizing loss over both boundary conditions and interior residuals.

## Why it lives in `cjc-ad`

PINNs are autodiff-native: you compute the network's output, take derivatives with respect to its inputs (the spatial/temporal coordinates), substitute those derivatives into the PDE's left-hand side, and minimize the squared residual. That requires nested differentiation — derivatives of the network, and derivatives of the loss with respect to the network's weights. Both are handled by [[Autodiff]].

## Examples

- `examples/08_pinn_heat_equation.cjcl` — heat equation.
- `examples/pinn_demo.cjcl` and `piml_demo.cjcl` — more general physics-informed ML.

## Tests

`tests/pinn_correctness.rs` — correctness tests for PINN training.

## Related

- [[Autodiff]]
- [[ODE Integration]]
- [[ML Primitives]]
- [[Advanced Computing in CJC-Lang]]
