# CJC Quantum Simulation Module

**Crate**: `cjc-quantum`
**Date**: 2026-03-30
**Status**: v0.2 Beta Phase 2 (Quantum Extensions)

## Overview

CJC includes a deterministic quantum circuit simulator for classical simulation
of quantum algorithms. The simulator uses statevector representation (2^N complex
amplitudes for N qubits) with full determinism guarantees.

**Key constraint**: Same seed = bit-identical measurement outcomes across runs and platforms.

## Architecture

```
cjc-quantum/
  src/
    lib.rs          — Module root, SplitMix64 PRNG, re-exports
    statevector.rs  — Statevector representation (2^N amplitudes)
    gates.rs        — Gate definitions + application algorithms
    measure.rs      — Measurement with probabilistic collapse
    circuit.rs      — Circuit builder + execution
    dispatch.rs     — Builtin function dispatch (wired into eval + mir-exec)
    wirtinger.rs    — Wirtinger calculus (complex AD)
    adjoint.rs      — Adjoint differentiation + HybridCircuit
    simd_kernel.rs  — AVX2 SIMD kernels + cache blocking
    mps.rs          — Matrix Product States (tensor-train SVD)
    vqe.rs          — VQE optimizer (Ising + full Heisenberg)
    qaoa.rs         — QAOA for MaxCut optimization
    stabilizer.rs   — Clifford/Stabilizer tableau simulator
    density.rs      — Density matrix simulator (mixed states + noise)
    dmrg.rs         — DMRG ground-state solver (variational Lanczos)
    qec.rs          — Quantum error correction (repetition + surface code)
```

### Determinism Guarantees

- Complex arithmetic uses `ComplexF64::mul_fixed()` — fixed-sequence, no FMA
- Probability accumulation uses `KahanAccumulatorF64`
- Basis states processed in ascending index order
- Measurement sampling via SplitMix64 with explicit `&mut u64` seed threading
- No `HashMap` or non-deterministic data structures (Vec only)
- Post-measurement renormalization uses Kahan summation

## Supported Gates

### Single-Qubit Gates

| Gate | Function | Description |
|------|----------|-------------|
| H | `q_h(circuit, qubit)` | Hadamard — creates equal superposition |
| X | `q_x(circuit, qubit)` | Pauli-X (NOT) — bit flip |
| Y | `q_y(circuit, qubit)` | Pauli-Y |
| Z | `q_z(circuit, qubit)` | Pauli-Z — phase flip |
| S | `q_s(circuit, qubit)` | S gate (sqrt(Z)) — pi/2 phase |
| T | `q_t(circuit, qubit)` | T gate (sqrt(S)) — pi/4 phase |
| Rx | `q_rx(circuit, qubit, angle)` | Rotation around X axis |
| Ry | `q_ry(circuit, qubit, angle)` | Rotation around Y axis |
| Rz | `q_rz(circuit, qubit, angle)` | Rotation around Z axis |

### Two-Qubit Gates

| Gate | Function | Description |
|------|----------|-------------|
| CNOT | `q_cx(circuit, ctrl, target)` | Controlled-NOT |
| CZ | `q_cz(circuit, a, b)` | Controlled-Z (symmetric) |
| SWAP | `q_swap(circuit, a, b)` | Swap two qubits |

### Three-Qubit Gates

| Gate | Function | Description |
|------|----------|-------------|
| Toffoli | `q_toffoli(circuit, c1, c2, target)` | Doubly-controlled NOT |

## Builtin Functions

### Construction

```cjc
let q = qubits(3);       // Create 3-qubit circuit (initialized to |000>)
```

### Inspection

```cjc
q_n_qubits(q)            // Number of qubits (integer)
q_n_gates(q)             // Number of gates in circuit (integer)
```

### Execution

```cjc
q_probs(q)               // Execute circuit, return probability array
q_amplitudes(q)          // Execute circuit, return complex amplitude array
q_measure(q, seed)       // Execute + measure all qubits (returns array of 0/1)
q_sample(q, n_shots, seed) // Execute + sample distribution n_shots times
```

## Usage Examples

### Bell State

```cjc
fn main() -> Any {
    let q = qubits(2);
    let q = q_h(q, 0);        // Hadamard on qubit 0
    let q = q_cx(q, 0, 1);    // CNOT: qubit 0 controls qubit 1
    q_probs(q)                 // [0.5, 0.0, 0.0, 0.5]
}
```

### GHZ State

```cjc
fn main() -> Any {
    let q = qubits(3);
    let q = q_h(q, 0);
    let q = q_cx(q, 0, 1);
    let q = q_cx(q, 0, 2);
    q_measure(q, 42)           // Always returns [0,0,0] or [1,1,1]
}
```

### Deterministic Sampling

```cjc
fn main() -> Any {
    let q = qubits(1);
    let q = q_h(q, 0);
    q_sample(q, 1000, 42)     // 1000 samples, seed=42, always identical results
}
```

## Quantum Hardening (v0.2 Beta Phase 2)

### Wirtinger Calculus (`wirtinger.rs`)

Complex-valued automatic differentiation using Wirtinger derivatives for non-holomorphic
loss functions like |α|².

```rust
use cjc_quantum::wirtinger::*;

// Forward-mode complex AD
let z = WirtingerDual::variable(ComplexF64::new(0.3, 0.4));
let norm = z.norm_sq();  // |z|² with correct ∂/∂z and ∂/∂z*

// Parameter-shift gradient for quantum circuits
let grad = parameter_shift_gradient(
    |theta| { /* build circuit, return probs */ },
    theta, qubit_index, observable_weights
);
```

Key identities: ∂|z|²/∂z = z*, ∂|z|²/∂z* = z.

### Adjoint Differentiation (`adjoint.rs`)

O(1) memory gradient computation for variational quantum circuits:

```rust
use cjc_quantum::adjoint::*;
use cjc_quantum::Circuit;

let mut circ = Circuit::new(1);
circ.ry(0, theta);
let z_obs = vec![1.0, -1.0];  // Z observable eigenvalues
let grads = adjoint_differentiation(&circ, &z_obs).unwrap();
// grads.gradients[0] = ∂⟨Z⟩/∂θ
```

### Mid-Circuit Measurement (`adjoint.rs`)

Classical feed-forward with `HybridCircuit`:

```rust
use cjc_quantum::adjoint::{HybridCircuit, CircuitOp};
use cjc_quantum::gates::Gate;

let mut hc = HybridCircuit::new(3, 2);  // 3 qubits, 2 classical registers
hc.gate(Gate::H(0));
hc.measure(0, 0);                        // measure qubit 0 → creg 0
hc.if_then(0, 1, Gate::X(1));           // if creg[0]==1, apply X to qubit 1
let (sv, cregs) = hc.execute(&mut seed).unwrap();
```

### SIMD Kernels (`simd_kernel.rs`)

AVX2-accelerated gate application with cache-aware blocking:

- `apply_single_qubit_simd()` — 2 pairs per AVX2 iteration
- `apply_single_qubit_cached()` — L1-cache-friendly tiling for high qubit indices
- `complex_mul_batch_2()` — batched complex multiply, no FMA
- Runtime CPU detection with scalar fallback

### Matrix Product States (`mps.rs`)

Tensor-train decomposition for 50+ qubit simulation of low-entanglement states:

```rust
use cjc_quantum::mps::Mps;

let mut mps = Mps::new(50);  // 50-qubit product state, ~1.6KB memory
mps.apply_single_qubit(0, h_matrix());
mps.apply_cnot_adjacent(0, 1);
// ...build GHZ chain...
let sv = mps.to_statevector();  // only for small verification!
```

- Sign-stabilized SVD ensures bit-identical bond truncation
- Memory: O(N × χ²) where χ = max bond dimension
- GHZ states: χ=2 regardless of N

## v0.2 Quantum Extensions

### Extension 1: Full Heisenberg Hamiltonian (`vqe.rs`)

XX + YY + ZZ nearest-neighbor interactions for the isotropic Heisenberg model:

```rust
use cjc_quantum::vqe::*;

// Compute full Heisenberg energy: sum of XX + YY + ZZ per bond
let mps = build_mps_ansatz(4, &thetas, 64);
let energy = mps_full_heisenberg_energy(&mps);

// VQE optimization with full Heisenberg
let result = vqe_full_heisenberg_1d(4, 16, 0.15, 20, 42);

// Hamiltonian selector for generic dispatch
let e = mps_energy(&mps, Hamiltonian::Heisenberg);
```

### Extension 2: QAOA for MaxCut (`qaoa.rs`)

Quantum Approximate Optimization Algorithm for graph MaxCut problems:

```rust
use cjc_quantum::qaoa::*;

let graph = Graph::cycle(5);
let result = qaoa_maxcut(&graph, 2, 20, 0.1, 42);  // p=2, 20 iters
println!("Best cut: {}", result.best_cut);
println!("Cut value: {}", result.best_cut_value);
```

- MPS-based simulation (50+ qubit graphs for low-entanglement states)
- General ZZ expectation values for non-adjacent qubits via SWAP networks
- Deterministic parameter optimization with explicit seed threading

### Extension 3: Clifford/Stabilizer Simulator (`stabilizer.rs`)

Aaronson-Gottesman CHP algorithm for efficient simulation of Clifford circuits:

```rust
use cjc_quantum::stabilizer::StabilizerState;

let mut s = StabilizerState::new(1000);  // 1000 qubits!
s.h(0);
for q in 0..999 { s.cnot(q, q + 1); }   // GHZ state
let outcome = s.measure(0, &mut rng);     // O(n^2) per gate
let sv = s.to_statevector();              // only for n <= 12
```

- Bitpacked u64 rows for Pauli tableau (X, Z, phase)
- Gates: H, S, X, Y, Z, CNOT — O(n) per gate operation
- Measurement: O(n^2), handles both deterministic and random outcomes
- Scales to 1000+ qubits (O(n^2) memory vs 2^n for statevector)

### Extension 4: Density Matrix Simulator (`density.rs`)

Mixed quantum states and noise channels via Kraus operator formalism:

```rust
use cjc_quantum::density::*;
use cjc_quantum::gates::Gate;

let mut rho = DensityMatrix::new(2);
rho.apply_gate(&Gate::H(0));
rho.apply_gate(&Gate::Cx(0, 1));

// Apply noise channels
rho.apply_depolarizing(0, 0.01);    // 1% depolarizing noise
rho.apply_dephasing(1, 0.05);       // 5% dephasing
rho.apply_amplitude_damping(0, 0.02); // 2% amplitude damping

let purity = rho.purity();           // Tr(rho^2) < 1 for mixed states
let probs = rho.probabilities();     // measurement probabilities
let entropy = rho.von_neumann_entropy(); // -Tr(rho log rho)
let fidelity = rho.fidelity(&pure_rho); // state fidelity
```

- Max ~13-14 qubits (2^2N matrix entries)
- Partial trace for subsystem analysis
- Von Neumann entropy via sign-stabilized SVD eigenvalues

### Extension 5: DMRG Ground-State Solver (`dmrg.rs`)

Density Matrix Renormalization Group for 1D quantum lattice models:

```rust
use cjc_quantum::dmrg::*;

// Ising model: -sum Z_i Z_{i+1}
let result = dmrg_heisenberg_1d(8, 16, 20, 1e-8);
// result.energy converges to exact ground state

// Full Heisenberg: -sum (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
let result = dmrg_full_heisenberg_1d(8, 16, 20, 1e-8);
```

- Two-site variational DMRG with Lanczos eigensolver
- MPO-style operator-weighted environments (identity, Hamiltonian, dangling operators)
- Effective Hamiltonian includes all 5 term types per bond
- SVD bond truncation with configurable max bond dimension
- Supports Ising and full Heisenberg Hamiltonians

### Extension 6: Quantum Error Correction (`qec.rs`)

Repetition codes and surface codes with syndrome extraction:

```rust
use cjc_quantum::qec::*;
use cjc_quantum::stabilizer::StabilizerState;

// Repetition code (distance 3)
let code = build_repetition_code(3);
let mut state = StabilizerState::new(code.total_qubits);
encode_repetition(&mut state, &code);

// Surface code (distance 3)
let surface = build_surface_code(3);
let mut state = StabilizerState::new(surface.total_qubits);
let syndrome = extract_z_syndrome(&mut state, &surface, &mut rng);
let correction = decode_minimum_weight(&surface, &syndrome);
```

- Repetition code: encode, syndrome extraction, minimum-weight decoding
- Surface code: 2D lattice with X and Z stabilizers
- Decoder: minimum-weight matching (greedy approximation)
- Built on top of the Clifford/Stabilizer simulator for efficiency

## Limitations

- **Classical simulation**: ~25-26 qubits with statevector (50+ with MPS for low-entanglement)
- **Density matrix**: ~13-14 qubits (2^2N scaling)
- **Stabilizer**: Clifford gates only (no T gate or arbitrary rotations)
- **DMRG**: 1D nearest-neighbor Hamiltonians only
- **No hardware backend**: Simulation only
- **MPS**: Adjacent-qubit 2-qubit gates only (SWAP network needed for non-adjacent)

## Test Coverage

### Unit Tests (203 in cjc-quantum)

- Statevector: construction, normalization, probabilities (6 tests)
- Gates: all 13 gate types, involutions, unitarity, error handling (22 tests)
- Measurement: collapse, determinism, statistics, Bell correlations (9 tests)
- Circuit: builder, execution, sampling, display, error handling (10 tests)
- Dispatch: constructor, gate chain, probs, measure, sample, rotation (9 tests)
- Wirtinger: dual arithmetic, norm_sq, chain rule, numerical agreement, parameter shift (10 tests)
- Adjoint: expectation value, gradients (Ry/Rz), finite-diff parity, multi-param (10 tests)
- Adjoint/Hybrid: teleportation, determinism, feed-forward correlation (4 tests)
- SIMD: scalar/cached/SIMD parity, batch multiply determinism (6 tests)
- MPS: initial state, single-qubit, Bell, GHZ, SVD, memory, bond dim (11 tests)
- VQE: convergence, determinism, Heisenberg energy, XX/YY/ZZ expectations (16 tests)
- QAOA: graph construction, MaxCut energy, optimization, determinism (15 tests)
- Stabilizer: initial state, gates (H/S/X/Y/Z/CNOT), Bell/GHZ, 1000-qubit scaling (23 tests)
- Density Matrix: construction, gates, noise channels, purity, entropy, fidelity (23 tests)
- DMRG: 2/4/6-qubit Ising + Heisenberg, convergence, determinism (13 tests)
- QEC: repetition code, surface code, syndrome extraction, decoding (20 tests)

### Integration Tests (211 in beta_tests/)

- Quantum circuit parity (eval vs mir-exec): constructor, Bell, GHZ, sampling, rotation
- Full Heisenberg: XX/YY/ZZ expectations, MPS vs statevector cross-validation, VQE convergence
- QAOA: graph construction, MaxCut energy bounds, determinism, optimization quality
- Stabilizer: initial state, H/+state, Bell/GHZ, statevector extraction, 128-500 qubit scaling
- Density Matrix: construction, gates, noise channels, purity, entropy, partial trace
- DMRG: 2/4/6-qubit ground state energy, Ising vs Heisenberg, convergence, determinism
- QEC: repetition code structure, encoding, syndrome extraction, surface code decoding

### Property Tests (8 in beta_tests/quantum_prop/)

- 10-run determinism for circuit execution
- 50-seed measurement determinism
- 20-seed sampling determinism
- Unitarity preservation across 8 gate sequences
- Self-inverse gate verification (X, Y, Z, H, SWAP)
- Bell state correlation (1000 shots, 100% correlated)
- Probability normalization via Kahan summation
- Chi-squared test for H|0> measurement statistics
- GHZ all-qubits-agree for 2-5 qubit systems
- Different-seed non-degeneracy

## Files Modified/Created

| File | Change |
|------|--------|
| `crates/cjc-quantum/` | Quantum crate (15 source files) |
| `crates/cjc-quantum/src/wirtinger.rs` | Wirtinger calculus (complex AD) |
| `crates/cjc-quantum/src/adjoint.rs` | Adjoint differentiation + HybridCircuit |
| `crates/cjc-quantum/src/simd_kernel.rs` | AVX2 SIMD kernels + cache blocking |
| `crates/cjc-quantum/src/mps.rs` | MPS/Tensor-Train + sign-stabilized SVD |
| `crates/cjc-quantum/src/vqe.rs` | VQE optimizer (Ising + full Heisenberg) |
| `crates/cjc-quantum/src/qaoa.rs` | QAOA for MaxCut optimization |
| `crates/cjc-quantum/src/stabilizer.rs` | Clifford/Stabilizer CHP simulator |
| `crates/cjc-quantum/src/density.rs` | Density matrix + noise channels |
| `crates/cjc-quantum/src/dmrg.rs` | DMRG variational ground-state solver |
| `crates/cjc-quantum/src/qec.rs` | QEC repetition + surface code |
| `crates/cjc-runtime/src/value.rs` | Added `Value::QuantumState` variant |
| `crates/cjc-snap/src/encode.rs` | Added QuantumState to non-serializable list |
| `crates/cjc-eval/src/lib.rs` | Wired `dispatch_quantum` |
| `crates/cjc-mir-exec/src/lib.rs` | Wired `dispatch_quantum` |
| `Cargo.toml` | Added cjc-quantum to workspace |
| `tests/beta_tests/quantum/` | 211 integration tests (10 test files) |
| `tests/beta_tests/quantum_prop/` | 8 property tests |
