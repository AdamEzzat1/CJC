---
title: ODE Integration
tags: [advanced, numerics]
status: Implemented (primitive), more on Roadmap
---

# ODE Integration

**Source**: `crates/cjc-runtime/src/ode.rs`.

## Summary

Deterministic numerical integration of ordinary differential equations. Same seed, same step sequence, bit-identical output.

## Roadmap context

CLAUDE.md's feature scope lists:
> `ode_step()` — ODE solver primitive
> `pde_step()` — PDE solver primitive
> `symbolic_derivative()` — symbolic differentiation primitive
>
> **Goal**: Allow future libraries (like Bastion) to build on them.

This framing suggests the runtime has **primitive-level** solver support and the "library of solvers" is meant to be written on top. The extent to which higher-level solvers (RK4, adaptive step, implicit methods) are implemented should be checked in `ode.rs` — **Needs verification**.

## Determinism

Any ODE integration is deterministic *if* the vector field evaluator is deterministic and the step schedule is fixed. Both hold in CJC-Lang because the numerical kernels are deterministic ([[Determinism Contract]]) and CJC-Lang does not use adaptive parallel ODE schedulers.

## Related

- [[Autodiff]]
- [[PINN Support]]
- [[Advanced Computing in CJC-Lang]]
- [[Roadmap]]
