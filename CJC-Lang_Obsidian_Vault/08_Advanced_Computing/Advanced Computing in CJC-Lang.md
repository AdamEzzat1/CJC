---
title: Advanced Computing in CJC-Lang
tags: [advanced, hub]
status: Implemented (in parts)
---

# Advanced Computing in CJC-Lang

CJC-Lang has a surprisingly rich surface for advanced scientific computing: a full quantum circuit simulator, physics-informed neural networks, ODE integration, sparse eigensolvers, and a family of numerical solvers.

This is the hub for that cluster.

## Components

| Area | Crate / module | Note |
|---|---|---|
| Quantum simulation | `cjc-quantum` | [[Quantum Simulation]] |
| Physics-informed ML | `cjc-ad/src/pinn.rs` | [[PINN Support]] |
| ODE integration | `cjc-runtime/src/ode.rs` | [[ODE Integration]] |
| Optimization | `cjc-runtime/src/optimize.rs` | [[Optimization Solvers]] |
| Sparse linear algebra | `cjc-runtime/src/sparse*.rs` | [[Sparse Linear Algebra]] |
| Clustering | `cjc-runtime/src/clustering.rs` | k-means etc. |
| Interpolation | `cjc-runtime/src/interpolate.rs` | |
| Time series | `cjc-runtime/src/timeseries.rs` | |

## Which of these stand out

1. **Quantum simulation**. This is not a stub. The `cjc-quantum` crate has 20 source files with real implementations of statevector, MPS, density matrix, DMRG, stabilizer, VQE, QAOA, QEC, QML, Wirtinger calculus for complex AD, Trotter expansion, fermion operators, measurement with deterministic seeding, noise mitigation. **Research-grade**, not just a demo. Needs a real eval of its scaling and performance.

2. **PINN support**. `cjc-ad/src/pinn.rs` is a ~44K module that builds physics-informed neural network loss terms on top of [[Autodiff]]. The example `08_pinn_heat_equation.cjcl` demonstrates solving the heat equation.

3. **Deterministic ODE integration**. CJC-Lang ships an ODE solver in `cjc-runtime/src/ode.rs`. It is deterministic by construction — same step sequence produces identical output.

## Related

- [[Tensor and Scientific Computing]]
- [[Scientific Computing Concept Graph]]
- [[Advanced Computing Source Map]]
