---
title: Quantum Simulation
tags: [advanced, quantum]
status: Implemented (research-grade)
---

# Quantum Simulation

**Crate**: `cjc-quantum` — `crates/cjc-quantum/src/` (~20 source files, ~500K+ LOC total for this crate based on the survey).

**Docs**: `docs/QUANTUM_SIMULATION.md`.

## Summary

A full deterministic classical simulator for quantum circuits. This is not a stub — the crate contains real implementations of multiple simulation paradigms.

## Modules

| File | Purpose |
|---|---|
| `gates.rs` | Gate definitions — H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, Toffoli (CCX) |
| `statevector.rs` | Dense statevector simulator |
| `circuit.rs` | Quantum circuit container |
| `measure.rs` | Measurement with deterministic seeding |
| `mps.rs` | Matrix Product State simulator |
| `density.rs` | Density matrix simulator (mixed states, noise) |
| `dmrg.rs` | DMRG (Density Matrix Renormalization Group) ground-state solver |
| `vqe.rs` | Variational Quantum Eigensolver |
| `qaoa.rs` | Quantum Approximate Optimization Algorithm |
| `stabilizer.rs` | Stabilizer / Clifford simulator |
| `qml.rs` | Quantum machine learning primitives |
| `qec.rs` | Quantum error correction (repetition + surface codes) |
| `wirtinger.rs` | Wirtinger calculus for complex-valued AD |
| `adjoint.rs` | Adjoint differentiation for quantum circuits |
| `trotter.rs` | Trotter product expansion |
| `fermion.rs` | Fermionic operators |
| `mitigation.rs` | Noise mitigation |
| `simd_kernel.rs` | SIMD inner kernels |
| `pure.rs` | Unified pure-state entry point |
| `dispatch.rs` | `dispatch_quantum` for routing from [[cjc-runtime]] builtins |

## Determinism

From the crate's lib.rs docstring:
- Amplitude accumulations use [[Kahan Summation]]
- Complex multiplication uses fixed-sequence (no FMA)
- Measurement sampling via [[SplitMix64]] with explicit seed threading
- Gate application processes basis states in ascending index order
- All collections use deterministic ordering (`Vec`, not `HashMap`)

## Limitations (stated in lib.rs)

- Classical simulation: ~25-30 qubits max (2^N memory scaling)
- No noise model in `pure.rs` (use `density.rs` for noise)
- No hardware backend — simulation only

## Surface from user code

From `docs/QUANTUM_SIMULATION.md`:

```cjcl
let c = qubits(3);
c.h(0);
c.cnot(0, 1);
c.cnot(1, 2);
let probs = c.q_probs();
let outcome = c.q_measure();   // seeded
```

Plus `q_sample(shots)` for multi-shot sampling.

## Advanced features

- **VQE** — variational quantum eigensolver for ground-state energies.
- **QAOA** — quantum approximate optimization for combinatorial problems.
- **MPS** — matrix product states for larger-qubit simulation.
- **DMRG** — DMRG ground-state search.
- **Density matrix** — mixed states, noise channels.
- **Stabilizer** — efficient Clifford circuit simulation.
- **Wirtinger AD** — gradients for complex-valued parameters (hybrid classical-quantum training).
- **QEC** — repetition and surface codes.
- **QML** — parameterized circuits for quantum machine learning.

## Example

`examples/09_quantum_simulation.cjcl` — a runnable example covering the basic surface.

## Local SplitMix64

`crates/cjc-quantum/src/lib.rs` defines its own `splitmix64` rather than depending on `cjc-repro` internals, to avoid a cross-crate coupling. The constants match the standard SplitMix64, so results are the same as [[SplitMix64]] in `cjc-repro`.

## Related

- [[Tensor Runtime]]
- [[Linear Algebra]]
- [[Autodiff]]
- [[Determinism Contract]]
- [[SplitMix64]]
- [[Kahan Summation]]
