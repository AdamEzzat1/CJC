# CJC Quantum Simulation Module

**Crate**: `cjc-quantum`
**Date**: 2026-03-29
**Status**: v0.2 Beta Phase 1

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

## Limitations

- **Classical simulation**: ~25-26 qubits with statevector (50+ with MPS for low-entanglement)
- **No noise model**: Pure unitary evolution only
- **No hardware backend**: Simulation only
- **MPS**: Adjacent-qubit 2-qubit gates only (SWAP network needed for non-adjacent)

## Test Coverage

### Unit Tests (93 in cjc-quantum)

- Statevector: construction, normalization, probabilities (6 tests)
- Gates: all 13 gate types, involutions, unitarity, error handling (22 tests)
- Measurement: collapse, determinism, statistics, Bell correlations (9 tests)
- Circuit: builder, execution, sampling, display, error handling (10 tests)
- Dispatch: constructor, gate chain, probs, measure, sample, rotation (9 tests)
- Wirtinger: dual arithmetic, norm_sq, chain rule, numerical agreement, parameter shift (10 tests)
- Adjoint: expectation value, gradients (Ry/Rz), finite-diff parity, multi-param (10 tests)
- Adjoint/Hybrid: teleportation, determinism, feed-forward correlation (4 tests)
- SIMD: scalar/cached/SIMD parity, batch multiply determinism (6 tests)
- MPS: initial state, single-qubit, Bell, GHZ, SVD identity/reconstruction/stability, memory, bond dim (11 tests)

### Integration Tests (17 in beta_tests/quantum/)

- Eval + MIR-exec for: constructor, Bell state, measurement, sampling, GHZ, rotation
- Parity tests: eval vs mir-exec produce identical results for all operations
- Error handling: out-of-range qubit produces runtime error

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
| `crates/cjc-quantum/` | Quantum crate (9 source files) |
| `crates/cjc-quantum/src/wirtinger.rs` | Wirtinger calculus (complex AD) |
| `crates/cjc-quantum/src/adjoint.rs` | Adjoint differentiation + HybridCircuit |
| `crates/cjc-quantum/src/simd_kernel.rs` | AVX2 SIMD kernels + cache blocking |
| `crates/cjc-quantum/src/mps.rs` | MPS/Tensor-Train + sign-stabilized SVD |
| `crates/cjc-runtime/src/value.rs` | Added `Value::QuantumState` variant |
| `crates/cjc-snap/src/encode.rs` | Added QuantumState to non-serializable list |
| `crates/cjc-eval/src/lib.rs` | Wired `dispatch_quantum` |
| `crates/cjc-eval/Cargo.toml` | Added cjc-quantum dependency |
| `crates/cjc-mir-exec/src/lib.rs` | Wired `dispatch_quantum` |
| `crates/cjc-mir-exec/Cargo.toml` | Added cjc-quantum dependency |
| `Cargo.toml` | Added cjc-quantum to workspace |
| `tests/beta_tests/quantum/` | 17 integration tests |
| `tests/beta_tests/quantum_prop/` | 8 property tests |
