---
title: Advanced Computing Source Map
tags: [source-map, advanced]
status: Grounded in crate layout
---

# Advanced Computing Source Map

Source-tree pointers for quantum simulation, PINN, ODE, optimization, and sparse methods.

## Quantum (`cjc-quantum`)

From `crates/cjc-quantum/src/lib.rs` (84 lines of pub module re-exports):

| Module | Responsibility |
|---|---|
| `core.rs` | Statevector core + gates |
| `gates.rs` | Single- and multi-qubit gate library |
| `circuits.rs` | Circuit construction DSL |
| `mps.rs` | Matrix product state simulator |
| `dmrg.rs` | DMRG ground-state solver |
| `density.rs` | Density matrix simulation |
| `noise.rs` | Noise channels |
| `stabilizer.rs` | Clifford / stabilizer simulation |
| `measurement.rs` | Measurement + sampling |
| `vqe.rs` | Variational Quantum Eigensolver |
| `qaoa.rs` | Quantum Approximate Optimization Algorithm |
| `qec.rs` | Quantum error correction (e.g. surface code) |
| `qml.rs` | Parameterized quantum machine learning |
| `tomography.rs` | State tomography |
| `entanglement.rs` | Entanglement measures |
| `hamiltonians.rs` | Hamiltonian construction |
| `trotter.rs` | Trotter-Suzuki decomposition |
| `builtins.rs` | CJC-Lang builtin bindings |
| `determinism.rs` | RNG + deterministic sampling |
| `export.rs` | Circuit / state export |

`lib.rs` itself defines the local `splitmix64` helper and re-exports all 20 submodules. See [[Quantum Simulation]].

Tests: `tests/bench_50q.rs` and `tests/quantum_*`.

## PINN and ODE

Physics-informed ML and ODE integration live closer to the ML primitives:

- `examples/07_physics_informed_ml.cjcl` — PINN basics.
- `examples/08_pinn_heat_equation.cjcl` — PINN solving the heat equation.
- ODE/solver infrastructure (`ode_step`, `pde_step`, `symbolic_derivative`) is **planned** as stubs in `cjc-runtime/src/builtins.rs` — see [[Roadmap]].

See [[PINN Support]] and [[ODE Integration]].

## Optimization

- Gradient descent, Adam, and related optimizers live in `cjc-runtime/src/builtins.rs` (search for `optimizer_`, `adam_`, `sgd_`).
- The autodiff engine in `cjc-ad` provides gradients consumed by optimizers.

See [[Optimization Solvers]] and [[Autodiff]].

## Sparse linear algebra

- `crates/cjc-runtime/src/sparse.rs` — `SparseCsr` / `SparseCoo` data structures.
- Sparse method dispatch (`matvec`, `to_dense`, `nnz`) is scheduled as S3-P2-04.
- Sparse eigensolvers (Lanczos / Arnoldi) are a separate roadmap item.

See [[Sparse Linear Algebra]].

## Chess RL integration test

- `tests/chess_rl_project/` — core suite (~49 tests per CLAUDE.md).
- `tests/chess_rl_advanced/` — advanced training (66 tests).
- `tests/chess_rl_hardening/` — robustness tests.
- `tests/chess_rl_playability/` — end-to-end gameplay parity.
- `examples/chess_rl_platform.html` — single-file JS browser frontend (~158 KB).

Related docs: `docs/chess_rl_project/` (README, DESIGN, DETERMINISM, RESULTS).

See [[Chess RL Demo]].

## Related

- [[Advanced Computing in CJC-Lang]]
- [[Scientific Computing Concept Graph]]
