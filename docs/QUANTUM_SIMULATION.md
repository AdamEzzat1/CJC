# CJC Quantum Simulation Module

**Crate**: `cjc-quantum`
**Date**: 2026-03-30
**Status**: v0.2 Beta Phase 2 (Quantum Extensions)

> **Audit status (2026-09-25).** The surface audit in
> [`docs/quantum_simulation_research_stack/`](quantum_simulation_research_stack/README.md)
> found drift between this document and the code. The items in its §10 are now
> corrected here, including a builtin reference that covers all 84 builtins and
> is checked against `dispatch.rs` by a test. Still true and not yet addressed:
>
> - the older timing tables under "Performance Optimizations" predate the
>   benchmark harness and remain unverified. Measured numbers, with machine,
>   commit, and method, are in "Measured Performance (harness)";
> - `cjcl check` reports errors on some valid quantum programs, and
>   `cjcl run` does not type-check (see "Type System").

## Overview

CJC includes a deterministic quantum circuit simulator for classical simulation
of quantum algorithms. The simulator uses statevector representation (2^N complex
amplitudes for N qubits) with full determinism guarantees.

**Key constraint**: Same seed = bit-identical measurement outcomes across runs and across operating systems.

Rotation gates, Trotter phases, and every other transcendental in `cjc-quantum` use `cjc_repro::dmath` (ADR-0046), a
pure-Rust `sin`/`cos`/`exp`/`ln` built from IEEE `+ - * /` only. The platform libm, which Windows and Linux disagree
on for ~6% of gate angles ([`VERIFY_FOLLOWUPS.md`](quantum_simulation_research_stack/VERIFY_FOLLOWUPS.md) §2), is no
longer used. Verified: a 348k-evaluation dump is byte-identical on Windows (UCRT) and Linux (glibc), and a golden-hash
test runs on Linux, Windows, and macOS in CI. Accuracy is < 1 ulp against mpmath.

The general-purpose `sin`, `cos`, `exp`, and `log` builtins in `cjc-runtime` still call the platform libm. An angle
computed in `.cjcl` with them can therefore differ by one ulp between operating systems before it reaches a gate.

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
    qml.rs          — Data re-uploading classifier (MPS-based)
    fermion.rs      — Pauli algebra, Jordan–Wigner, H2 / LiH Hamiltonians
    trotter.rs      — Trotter–Suzuki time evolution (1st/2nd order)
    mitigation.rs   — Zero-noise extrapolation (Richardson, linear)
    pure.rs         — "pure" backend (second, simpler Rust implementation)
    kernels.rs      — statevector gate kernels (strided, optionally threaded)
    qasm.rs         — OpenQASM 2.0 import / export
```

### Determinism Guarantees

- Complex arithmetic uses `ComplexF64::mul_fixed()` — fixed-sequence, no FMA
- Probability accumulation uses `KahanAccumulatorF64`
- Basis states processed in ascending index order
- Measurement sampling via SplitMix64 with explicit `&mut u64` seed threading
- No `HashMap` or non-deterministic data structures (Vec only)
- Post-measurement renormalization uses Kahan summation
- `sin`, `cos`, `exp`, `ln`, and powers come from `cjc_repro::dmath`, never the platform libm (ADR-0046)
- Gate kernels split large statevectors across threads (thread count from
  `cjc_runtime::runtime_policy`, only above 2^15 amplitude pairs per thread).
  Each amplitude pair is written by exactly one thread with the same operations
  as the serial loop, and nothing is reduced across threads, so results are
  bit-identical at any thread count (tested at 1, 2, 3, and 8)
- Seeded sampling (`q_sample`, `Circuit::sample`) draws one `rand_f64` per shot
  and binary-searches a running maximum of the Kahan-finalised prefix sums. The
  shot stream is bit-identical to the earlier per-shot linear scan (tested,
  including unnormalised and near-zero distributions)
- An immutable circuit value caches its executed statevector (≤ 24 qubits), so
  `q_probs(c)` then `q_sample(c, …)` simulate once. A gate returns a new circuit
  with an empty cache, and `q_copy` starts empty too

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

Each of these also accepts a statevector in place of the circuit (see below).

### Value semantics (ADR-0044)

**Circuits are values.** A gate returns a new circuit and never changes its argument:

```cjc
let base = q_h(qubits(1), 0);
let a = q_ry(base, 0, 0.1);   // base still has 1 gate; a has 2
let b = q_ry(base, 0, 0.2);   // independent of a
q_x(base, 0);                 // result discarded: base is unchanged
```

Before ADR-0044, all names bound to one circuit shared it, so `a` and `b` above
would both have ended up with every gate. Adding a gate copies the gate list
(O(gates)), not the state.

**Simulator states are mutable handles.** `mps_*`, `stabilizer_*`, and
`density_*` operations update the state in place and return the same handle,
so `m = mps_h(m, 0)` and a bare `mps_h(m, 0)` do the same thing. Measurement
(`stabilizer_measure`) collapses the handle. To fork a state, copy it
explicitly:

```cjc
let s = stabilizer_new(2);
let fork = q_copy(s);         // independent deep copy
stabilizer_x(s, 0);           // changes s only
```

`q_copy(value)` works on every quantum value: circuits, statevectors, MPS,
stabilizer, density, graphs, codes, and Hamiltonians, on both backends.

### Passing states between builtins (ADR-0045)

A **state argument** is a circuit (executed on use) or a statevector (from
`q_run` or `q_trotter_evolve`). `q_probs`, `q_amplitudes`, `q_sample`,
`q_measure`, `q_n_qubits`, `q_fermion_expectation`, `q_trotter_evolve`,
`q_expect_pauli`, and `density_from_state` all take one. For any circuit `c`,
`q_sample(q_run(c), k, seed)` returns the same bits as `q_sample(c, k, seed)`.

```cjc
let psi = q_run(c);                          // execute once
let h = q_fermion_new(2);
h = q_fermion_add_term(h, "ZZ", 0.5);        // returns a new Hamiltonian
h = q_fermion_add_term(h, "XX", -0.25);
let e = q_fermion_expectation(h, psi);       // <psi|H|psi>
let zz = q_expect_pauli(psi, "ZZ");          // one Pauli observable
let later = q_trotter_evolve(h, psi, 0.4, 8, 2);
let counts = q_sample(later, 100, 7);        // measure the evolved state
let rho = density_from_state(later);         // |psi><psi|, up to 14 qubits
rho = density_depolarize(rho, 0, 0.05);      // then add noise
```

Pauli strings use one character from `I`, `X`, `Y`, `Z` per qubit; character
`k` acts on qubit `k`. Invalid characters, a wrong length, or a non-finite
coefficient return an error.

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

// Parameter-shift gradient: caller evaluates E(θ+π/2) and E(θ−π/2)
let grad = parameter_shift_gradient(e_plus, e_minus); // (e_plus - e_minus) / 2
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

### SIMD Kernels (`simd_kernel.rs`) — measured, deliberately not used

These kernels are unit-tested and bit-identical to the scalar path, but
**nothing outside `simd_kernel.rs` calls them**, by decision (2026-09-25).
Measured against the full-scan loop (`quantum_compare kernels`, one H gate,
n = 16–24, i7-11390H):

- `apply_single_qubit_simd` (AVX2) was **3–5× slower** at every size. It
  allocates a vector of index pairs on every call and loads lanes one at a
  time, which costs more than the vector arithmetic saves.
- `apply_single_qubit_cached` roughly tied the full-scan loop.

Gates are applied by `kernels.rs` instead. It walks only the index pairs a
gate touches, with contiguous inner loops, and threads large states. Over
whole circuits that is 1.2–1.5× faster than the old loop (see "Measured
Performance").

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
- Truncation keeps the largest χ Schmidt values. When a two-site update would
  drop more than round-off (`TRUNC_REL_TOL` = 1e-10 of the largest value), the
  chain is first brought into mixed-canonical form around that bond so the
  singular values are the true Schmidt coefficients, and the kept ones are
  rescaled to preserve the norm (Qiskit Aer's convention)
- Before 2026-09-25 the one-sided Jacobi SVD used the wrong rotation sign. It
  returned correct reconstructions but wrong singular values whenever two
  columns had unequal norms, so χ-truncation discarded the wrong directions.
  At depth-8 brickwork with χ = 16, ⟨Z_i⟩ was off by 0.53 against Qiskit Aer
  and a dense oracle. Fixed; see `mps.rs` `accuracy_tests`
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
// qaoa_maxcut(graph, p_layers, max_bond, learning_rate, max_iters, seed)
let result = qaoa_maxcut(&graph, 2, 16, 0.1, 20, 42);
println!("Energy: {}", result.energy);
println!("Cut value: {}", result.cut_value);
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
let outcome = s.measure(0, &mut rng);     // O(n^2) per measurement
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
rho.apply_gate(&Gate::CNOT(0, 1));

// Apply noise channels (Kraus sets built by helper functions)
rho.apply_single_qubit_channel(0, &depolarizing_channel(0.01));
rho.apply_single_qubit_channel(1, &dephasing_channel(0.05));
rho.apply_single_qubit_channel(0, &amplitude_damping_channel(0.02));

let purity = rho.purity();           // Tr(rho^2) < 1 for mixed states
let probs = rho.probabilities();     // measurement probabilities
let entropy = rho.von_neumann_entropy(); // -Tr(rho log rho)
let fidelity = DensityMatrix::fidelity(&rho, &pure_rho); // state fidelity
```

- At most 14 qubits (`MAX_QUBITS = 14`). From `.cjcl`, larger sizes are a runtime error on both backends; calling `DensityMatrix::new(15)` directly from Rust panics.
- Depolarizing convention: Kraus set √(1−p)·I, √(p/3)·{X,Y,Z}, i.e. ρ → (1−p)ρ + (p/3)(XρX+YρY+ZρZ), equivalent to replacement by I/2 with probability 4p/3.
- Partial trace for subsystem analysis
- Von Neumann entropy via sign-stabilized SVD eigenvalues

### Extension 5: DMRG Ground-State Solver (`dmrg.rs`)

Density Matrix Renormalization Group for 1D quantum lattice models:

```rust
use cjc_quantum::dmrg::*;

// Ising model: H = +sum Z_i Z_{i+1}
// (note: this Rust fn is named `dmrg_heisenberg_1d` but runs Ising;
//  the .cjcl builtin `dmrg_ising` calls it)
let result = dmrg_heisenberg_1d(8, 16, 20, 1e-8);

// Full Heisenberg: H = +sum (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
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
apply_noise_round(&mut state, &code, 0.05, &mut rng);
let syndrome = syndrome_extraction(&mut state, &code, &mut rng);
let corrections = decode_repetition_code(&syndrome, &code);

// Surface code (distance 3): layout + syndrome extraction only
let surface = build_surface_code(3);
let mut state = StabilizerState::new(surface.total_qubits);
let syndrome = syndrome_extraction(&mut state, &surface, &mut rng);
// No 2D decoder exists yet (qec.rs:14). The `.cjcl` builtin `qec_decode`
// currently applies the repetition decoder to any code, which gives
// meaningless corrections for surface codes.
```

- Repetition code: syndrome extraction and adjacent-defect pairing decoder
- Surface code: 2D lattice with X and Z stabilizers; **no decoder**
- Noise model: code-capacity (data-qubit errors, perfect syndrome measurement)
- Built on top of the Clifford/Stabilizer simulator for efficiency

### Extension 7: Quantum Machine Learning (`qml.rs`)

Data Re-Uploading (QC-REUP) quantum neural network for classification:

```rust
use cjc_quantum::qml::*;

let config = QmlConfig {
    n_qubits: 16,
    n_reupload_passes: 3,
    n_classes: 2,
    max_bond: 16,
    readout_qubits: vec![0, 1],
    learning_rate: 0.05,
    epochs: 20,
    batch_size: 32,
    loss: QmlLoss::CrossEntropy,
    seed: 42,
};

// Load and preprocess image data
let dataset = load_dataset(&image_bytes, &labels, 28, 28, 1000, 16, 2);

// Train
let result = qml_train(&config, &dataset);
println!("Final accuracy: {:.1}%", result.final_accuracy * 100.0);
```

- **QC-REUP architecture**: Data re-encoded at every layer via parameterized Rx/Ry/Rz rotations
- **MPS-friendly**: Adjacent CNOT entanglement only, scales to 50+ qubits
- **6 params per qubit per layer**: 3 data weights + 3 trainable biases
- **Softmax classification**: Z expectation values mapped to class probabilities
- **Finite-difference gradient**: Works correctly for all parameter types (weights and biases)
- **Data preprocessing**: Snake (boustrophedon) ordering + average pooling for images
- Memory: O(N * chi^2) per forward pass, well under 1MB for 50 qubits at chi=16

## Dual-Mode Architecture — Default Backend + "pure" Backend

CJC offers two quantum backends, selectable per-operation:

### Rust Backend (default)
Fast, optimized Rust implementations. Used when no `"pure"` flag is passed.
```cjc
let m = mps_new(50, 16);          // Rust backend (default)
let s = stabilizer_new(1000);     // Rust backend
let d = density_new(8);           // Rust backend
```

### Pure Backend
A second, simpler Rust implementation that keeps state in plain containers.
It is not written in CJC-Lang. Activated by passing `"pure"` as the last
constructor argument.
```cjc
let m = mps_new(50, 16, "pure");       // Pure CJC backend
let s = stabilizer_new(1000, "pure");  // Pure CJC backend
let d = density_new(8, "pure");        // Pure CJC backend

// Subsequent operations auto-detect the backend:
let m = mps_h(m, 0);                  // Uses pure backend (auto-detected)
let z = mps_z_expectation(m, 0);      // Uses pure backend (auto-detected)

// Inspect internal state (pure backend only):
let state_map = quantum_inspect(m);    // Returns CJC Map with all state data
```

### What the Pure Backend Gives You

| Feature | Description |
|---------|-------------|
| **Inspectability** | `quantum_inspect(state)` returns a read-only CJC Map snapshot of the internal data (no write-back) |
| **Readability** | Short, unoptimised algorithms: Jacobi SVD, tableau updates, dense density matrices |
| **Cross-checking** | A second implementation to compare the default backend against |
| **Determinism** | Each backend is bit-identical with itself across runs and operating systems (`crates/cjc-quantum/tests/cross_platform_golden.rs` covers both). The two backends are **not** bit-identical to each other |
| **Not provided** | Changing algorithms without recompiling; autodiff through the backend |

### State Representation (Pure Backend)

| Type | CJC Representation |
|------|-------------------|
| MPS | `Map { n_qubits, max_bond, tensors: [Map { bond_left, bond_right, data_re, data_im }] }` |
| Stabilizer | `Map { n, x: [[Int]], z: [[Int]], phase: [Int] }` (tableau as u64 word arrays) |
| Density | `Map { n_qubits, dim, data_re: [Float], data_im: [Float] }` (row-major flat) |

### Cross-Backend Compatibility

The backends use different algorithms, so they agree numerically, not bitwise:
- Z-expectations agree to 1e-10 in `tests/beta_tests/quantum/test_quantum_pure_backend.rs`
- Measurement outcomes agree where they are deterministic
- Both backends are covered by the dispatch fuzz sweep (panic freedom) and the
  cross-platform golden hash. The property tests in `quantum_prop/` exercise
  the default backend only.

## Type System

CJC's type system has dedicated quantum types, and 67 of the 84 quantum
builtins have registered signatures. Two limits apply:
- `cjcl run` does **not** run the type checker. Quantum type errors surface as
  runtime errors from the dispatch layer, identically in both executors.
- `cjcl check` does run it, but reports false errors on some valid quantum
  programs (seen on `examples/quantum_simulations` demo 01).

### Quantum Types

| Type                 | Description                                          | Qubit Scale |
|----------------------|------------------------------------------------------|-------------|
| `QuantumCircuit`     | Gate-level circuit description                       | 1-26        |
| `QuantumStatevector` | Full 2^N amplitude vector (from circuit execution)   | 1-26        |
| `QuantumMps`         | Matrix Product State (tensor-train SVD)              | 50+         |
| `QuantumStabilizer`  | Stabilizer/CHP state for Clifford circuits           | 1000+       |
| `QuantumDensity`     | Density matrix for mixed states + noise channels     | 1-14        |
| `QuantumGraph`       | Graph structure for QAOA optimization problems       | any         |
| `QuantumSurfaceCode` | Surface/repetition code for quantum error correction | any         |

All types are registered in `cjc-types` alongside the core primitives (`i64`, `f64`, `bool`, etc.)
and participate in type unification, pattern matching, and the NoGC verifier.

### Type-Checked Signatures

Signatures exist for 67 of 84 builtins. The 17 without one are the MPS
canonicalisation and SWAP builtins, the fermion, Trotter, and ZNE families,
`q_scale_noise`, `qml_train`, and `quantum_inspect` (the "Typed" column of the
reference below). Examples:

```
qubits(n: i64) -> QuantumCircuit
mps_new(n_qubits: i64, max_bond: i64) -> QuantumMps
stabilizer_new(n_qubits: i64) -> QuantumStabilizer
density_new(n_qubits: i64) -> QuantumDensity
mps_h(mps: QuantumMps, qubit: i64) -> QuantumMps
stabilizer_measure(state: QuantumStabilizer, qubit: i64, seed: i64) -> i64
density_purity(dm: QuantumDensity) -> f64
```

The type checker enforces that, e.g., `mps_h` receives a `QuantumMps` not a `QuantumCircuit`,
and that `stabilizer_measure` returns `i64`.

## Builtin Reference (all 84 quantum builtins)

Generated from the match arms, argument extractors, and `min_arity` table in
`crates/cjc-quantum/src/dispatch.rs`, and the signatures in `cjc-types`
(2026-09-25). The test `fuzz_dispatch_reference_documents_every_builtin`
fails if a builtin is added to `dispatch.rs` without a row here.

Both executors (`cjc-eval`, `cjc-mir-exec`) call the same `dispatch_quantum`,
so results are identical in both. Every builtin validates its arguments and
returns a runtime error (never a crash) on bad input.

Column key:
- **Pure:** ✓ = has a `"pure"`-backend implementation; *in* = accepts
  pure-backend input by converting it; — = default backend only.
- **Typed:** ✓ = has a `cjc-types` signature (67 of 84).
- `[x]` = optional argument. A *state* is a circuit or a statevector
  (ADR-0045). *Handle* = the state is changed in place (ADR-0044).

### Circuits (values: a gate returns a new circuit) and observables

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `qubits(n_qubits[, "pure"])` | circuit | ✓ | ✓ | 1 ≤ n ≤ 26, starts in \|0…0⟩ |
| `q_h(circuit, qubit)`, `q_x(…)`, `q_y(…)`, `q_z(…)`, `q_s(…)`, `q_t(…)` | circuit | ✓ | ✓ | |
| `q_rx(circuit, qubit, angle)`, `q_ry(…)`, `q_rz(…)` | circuit | ✓ | ✓ | angle must be finite |
| `q_cx(circuit, control, target)`, `q_cnot(…)` | circuit | ✓ | ✓ | aliases; operands must differ |
| `q_cz(circuit, a, b)`, `q_swap(circuit, a, b)` | circuit | ✓ | ✓ | |
| `q_toffoli(circuit, ctrl1, ctrl2, target)`, `q_ccx(…)` | circuit | — | ✓ | aliases |
| `q_run(circuit)` | statevector | ✓ | ✓ | |
| `q_probs(state)` | `[f64]` | ✓ | ✓ | |
| `q_amplitudes(state)` | `[Complex]` | *in* | ✓ | |
| `q_measure(state, seed)` | `[i64]` (one bit per qubit) | ✓ | ✓ | does not change a statevector argument |
| `q_sample(state, n_shots, seed)` | `[i64]` (basis indices) | *in* | ✓ | same draws for `c` and `q_run(c)` |
| `q_n_qubits(state)` | `i64` | ✓ | ✓ | any quantum state with qubits |
| `q_n_gates(circuit)` | `i64` | ✓ | ✓ | |
| `q_expect_pauli(state, pauli)` | `f64` | *in* | ✓ | e.g. `"XZIY"`; character k acts on qubit k |
| `q_copy(value)` | same kind | ✓ | ✓ | deep copy of any quantum value |
| `q_to_qasm(circuit)` | `String` | ✓ | ✓ | OpenQASM 2.0; angles round-trip exactly |
| `q_from_qasm(text[, "pure"])` | circuit | ✓ | ✓ | subset: `h x y z s t rx ry rz cx cz swap ccx id barrier`, trailing `measure`; anything else is an error |

### MPS (handles)

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `mps_new(n_qubits[, max_bond][, "pure"])` | MPS | ✓ | ✓ | 1 ≤ n ≤ 100,000; pure default χ = 32 |
| `mps_h(mps, qubit)`, `mps_x(mps, qubit)` | MPS | ✓ | ✓ | |
| `mps_ry(mps, qubit, theta)` | MPS | ✓ | ✓ | |
| `mps_cnot(mps, control, target)` | MPS | ✓ | ✓ | adjacent qubits only |
| `mps_swap(mps, qubit1, qubit2)` | MPS | — | — | |
| `mps_z_expectation(mps, qubit)` | `f64` | ✓ | ✓ | |
| `mps_energy(mps[, hamiltonian])` | `f64` | — | ✓ | `"ising"` (default) or `"heisenberg"` |
| `mps_memory(mps)` | `i64` bytes | ✓ | ✓ | |
| `mps_left_canonicalize(mps)`, `mps_right_canonicalize(mps)` | MPS | — | — | |
| `mps_mixed_canonicalize(mps, center)` | MPS | — | — | |

### Variational algorithms

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `vqe_heisenberg(n_qubits, max_bond, learning_rate, iterations, seed)` | `f64` energy | — | ✓ | ZZ-only (Ising) despite the name; n ≥ 2 |
| `vqe_full_heisenberg(n_qubits, max_bond, learning_rate, iterations, seed)` | `f64` energy | — | ✓ | XX + YY + ZZ |
| `qaoa_graph_cycle(n_vertices)` | graph | — | ✓ | n ≥ 3 |
| `qaoa_maxcut(graph, max_bond, p_layers, learning_rate, iterations, seed)` | `[f64 energy, i64 cut]` | — | ✓ | |
| `qml_train(n_qubits, layers, n_classes, max_bond, learning_rate, epochs, seed, samples, labels)` | `[f64 accuracy, [f64 loss]]` | — | — | finite-difference gradients |
| `qml_predict(n_qubits, layers, n_classes, max_bond, params, input)` | `i64` class | — | ✓ | |

### Stabilizer / Clifford (handles)

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `stabilizer_new(n_qubits[, "pure"])` | stabilizer | ✓ | ✓ | 1 ≤ n ≤ 32,768 |
| `stabilizer_h(state, qubit)`, `stabilizer_s(…)`, `stabilizer_x(…)`, `stabilizer_y(…)`, `stabilizer_z(…)` | stabilizer | ✓ | ✓ | O(n) per gate |
| `stabilizer_cnot(state, control, target)` | stabilizer | ✓ | ✓ | |
| `stabilizer_measure(state, qubit, seed)` | `i64` | ✓ | ✓ | collapses the state; O(n²) |
| `stabilizer_n_qubits(state)` | `i64` | ✓ | ✓ | |

### Density matrix and noise (handles)

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `density_new(n_qubits[, "pure"])` | density | ✓ | ✓ | 1 ≤ n ≤ 14 on both backends |
| `density_from_state(state)` | density | ✓ | ✓ | ρ = \|ψ⟩⟨ψ\| |
| `density_gate(dm, gate, qubit)` | density | ✓ | ✓ | gate ∈ `"H" "X" "Y" "Z" "S" "T"` |
| `density_cnot(dm, control, target)` | density | ✓ | ✓ | |
| `density_depolarize(dm, qubit, p)` | density | ✓ | ✓ | Kraus √(1−p)I, √(p/3){X,Y,Z}: replacement by I/2 with probability 4p/3 |
| `density_dephase(dm, qubit, p)` | density | ✓ | ✓ | |
| `density_amplitude_damp(dm, qubit, gamma)` | density | ✓ | ✓ | |
| `density_trace(dm)`, `density_purity(dm)`, `density_entropy(dm)` | `f64` | ✓ | ✓ | entropy in nats |
| `density_probs(dm)` | `[f64]` | ✓ | ✓ | |

### DMRG

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `dmrg_ising(n_qubits, max_bond, sweeps, tolerance)` | `f64` energy | — | ✓ | H = +Σ ZZ; n ≥ 2 |
| `dmrg_heisenberg(n_qubits, max_bond, sweeps, tolerance)` | `f64` energy | — | ✓ | H = +Σ (XX + YY + ZZ); n ≥ 2 |

### Quantum error correction

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `qec_repetition_code(distance)`, `qec_surface_code(distance)` | code | — | ✓ | distance ≤ 1,024 |
| `qec_syndrome(state, code, seed)` | `[i64]` | — | ✓ | state: stabilizer sized for the code |
| `qec_decode(syndrome, code)` | `[i64]` corrections | — | ✓ | repetition codes only; a surface code is an error |
| `qec_logical_error_rate(distance, error_rate, rounds, seed)` | `f64` | — | ✓ | code-capacity noise |

### Hamiltonians, time evolution, error mitigation

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `q_fermion_h2([ "pure"])` | Hamiltonian | ✓ | — | 2 qubits, electronic energy only (add 1/R = 0.713754 Ha) |
| `q_fermion_lih()` | Hamiltonian | — | — | 4 qubits, CAS(2,2) STO-3G, eigenvalues are total energies |
| `q_fermion_new(n_qubits)` | Hamiltonian | — | — | empty; 1 ≤ n ≤ 26 |
| `q_fermion_add_term(h, pauli, coeff)` | new Hamiltonian | ✓ | — | `h` unchanged |
| `q_fermion_n_terms(h)` | `i64` | — | — | |
| `q_fermion_expectation(h, state)` | `f64` | ✓ | — | ⟨ψ\|H\|ψ⟩ |
| `q_trotter_evolve(h, state, time, n_steps[, order])` | statevector | — | — | order 1 (default) or 2 |
| `q_trotter_error(h, time, n_steps[, order])` | `f64` bound | — | — | |
| `q_zne_mitigate(scale_factors, measured_values[, "pure"])` | `[f64, [f64 coefficients]]` | ✓ | — | Richardson; scales must be distinct |
| `q_zne_linear(lambda1, value1, lambda2, value2)` | `f64` | — | — | |
| `q_scale_noise(base_p, scale_factor[, noise_type])` | `f64` | — | — | `"depolarizing"` (default), `"dephasing"`, `"amplitude_damping"` |

### Inspection

| Builtin | Returns | Pure | Typed | Notes |
|---|---|---|---|---|
| `quantum_inspect(state)` | Map | ✓ | — | pure-backend states only; read-only snapshot |

## Measured Performance (harness)

Measured with `bench/quantum_compare` (BENCHMARK_PLAN §9 steps 1–2) on one
machine: i7-11390H (4 cores / 8 threads), Windows 11, rustc 1.97.1, release,
working tree on `671dfeb`. Raw records, the generated `REPORT.md`, and notes
are under `bench_results/quantum_compare/2026-09-25_*`. Every CJC output
replays bit-for-bit in a fresh process, and the Rust API, `cjc-eval`, and
`cjc-mir-exec` produce byte-identical output.

**Old vs new, interleaved in one process** (`quantum_compare kernels`, median
of 7). This is the controlled comparison: separate runs on this laptop varied
up to 2× on unchanged code.

| Change | Case | Before | After | Speedup | Output |
|---|---|---|---|---|---|
| Gate kernels (`kernels.rs`) | GHZ, 22 qubits | 161–217 ms | 122–180 ms | 1.2–1.3× | bit-identical |
| | random, 20 qubits, depth 10 | 472–629 ms | 309–408 ms | 1.5× | bit-identical |
| | random, 22 qubits, depth 10 | 2.09–2.80 s | 1.46–2.12 s | 1.3–1.4× | bit-identical |
| Batch sampler, 1,000 shots | GHZ, 22 qubits | 4.1–5.2 s | 14–18 ms | ~290× | same shots |
| | random, 22 qubits | 8.8–9.4 s | 23 ms | ~400× | same shots |
| Execution cache (`q_probs`, `q_sample`, `q_measure` on one circuit) | 20 qubits, eval and mir | 1.16–1.17 s | 0.50 s | 2.3× | byte-identical |

**Against external simulators** (one run each, so treat ratios under 2× as
noise). The *execute* phase is circuit in, statevector out.

| Workload | CJC | Qiskit Aer 0.17.2, 1 thread, no fusion | Aer default (8 threads, fusion) | Agreement |
|---|---|---|---|---|
| Dense GHZ, 26 qubits (execute) | 2.87 s | 7.92 s | 2.01 s | ≤ 1.1e-16 vs analytic |
| Dense random, 22 qubits, depth 20 (execute) | 3.39 s | 13.26 s | 1.86 s | L∞ 1.5e-17 vs Aer |
| MPS brickwork, 100 qubits, depth 8, χ = 16 (total) | 3.5 s | 0.047 s | 0.045 s | ⟨Z_i⟩ within 7.8e-11 |
| Clifford, 1,000 qubits, depth 1,000, vs Stim 1.16 tableau (total) | 28.6–57.9 s | — | Stim: 0.68 s | `peek_z` identical on all qubits |

What the numbers support:
- Dense statevector: CJC is faster than single-threaded Aer without fusion,
  and slower than default Aer by about 1.4–1.8× at 22–26 qubits.
- MPS: CJC is ~20–75× slower than Aer at depth 8 (n = 50–100). It uses a Jacobi SVD, and its
  bonds keep round-off singular values (an absolute `SVD_TOL`). That
  tolerance is a known follow-up; changing it changes output bits.
- Stabilizer: CJC is ~40–85× slower than Stim's tableau simulator.

## Performance Optimizations

The optimisations below are real code changes. The **timings** in this section
were not recorded with a harness, machine description, or commit, and have not
been reproduced; treat them as unverified until `BENCHMARK_PLAN.md` is
implemented. The allocation counts and complexity claims follow from the code.

### 1. MPS In-Place Single-Qubit Gates

**File**: `mps.rs` — `apply_single_qubit()`

Before: Allocated 2 new `DenseMatrix` objects per gate, then discarded the old ones.
After: Computes in-place using only stack temporaries. **Zero heap allocations per gate.**

| Metric | Before | After |
|--------|--------|-------|
| Heap allocations per gate | 2 matrices | 0 |
| 50q MPS create+5 gates | ~2.0ms | ~0.7ms |

### 2. MPS CNOT Fused Gate Permutation

**File**: `mps.rs` — `apply_cnot_adjacent()`

Before: Built 4 theta matrices, cloned all 4 for CNOT permutation, then copied into combined
matrix — 9 matrix allocations total (4 theta + 4 clones + 1 combined).
After: Fuses gate permutation into the contraction step. Writes directly to combined matrix
via a lookup table. **Eliminates 8 of 9 allocations.**

### 3. Stabilizer Word-Level Phase Accumulation

**File**: `stabilizer.rs` — `rowmult()`

Before: Per-bit `g_phase()` called once per qubit per row multiplication — N iterations
for N qubits, each extracting individual bits from packed words.
After: Word-level bitwise operations process 64 qubits simultaneously using `count_ones()`
(popcount) on u64 words. Decomposes the Pauli commutation relation into positive/negative
contribution bitmasks.

| System | Before (per-bit) | After (word-level) |
|--------|-------------------|---------------------|
| 64 qubits | 64 g_phase calls | 1 word iteration |
| 500 qubits | 500 g_phase calls | 8 word iterations |
| 1000 qubits | 1000 g_phase calls | 16 word iterations |

### 4. VQE Cached Transfer Matrices

**File**: `vqe.rs` — `mps_heisenberg_energy()`, `mps_full_heisenberg_energy()`

Before: Each ZZ/XX/YY expectation value did a full left-to-right sweep of N transfer matrix
contractions. For N-1 bond terms, total cost was O(N^2 chi^4).
After: Pre-computes and caches left and right identity environments. Each term only needs
the 2 operator contractions + 1 environment join. **Reduces total cost from O(N^2 chi^4) to O(N chi^4).**

### 5. Density Matrix In-Place Permutation

**File**: `density.rs` — `apply_permutation_gate()`

Before: Allocated a full dim x dim temporary matrix (O(4^N) complex numbers), copied with
permutation, then replaced original.
After: Uses cycle-following permutation algorithm to permute in-place. Only allocates O(dim^2)
bits for visited tracking instead of O(dim^2) complex numbers.

| N qubits | Before (temp alloc) | After (in-place) |
|----------|---------------------|-------------------|
| 8 | 1 MB | 64 KB visited bits |
| 10 | 16 MB | 1 MB visited bits |
| 12 | 256 MB | 16 MB visited bits |

### Memory Budget

| System | Memory |
|--------|--------|
| Statevector 50q | 16,384 TB (infeasible) |
| MPS 50q (chi=16) | < 1 MB |
| MPS 50q (chi=64) | < 10 MB |
| Stabilizer 500q | ~129 KB |
| Stabilizer 1000q | ~482 KB |
| Density 8q | ~1 MB |
| Density 13q | ~1 GB |

## Limitations

- **Dense statevector**: at most 26 qubits (2^N memory; MPS reaches 50+ for low-entanglement states)
- **Density matrix**: at most 14 qubits (4^N memory: 4 GiB at 14)
- **Noise**: density backend only (depolarizing, dephasing, amplitude damping); no trajectories on the statevector backend
- **Stabilizer**: Clifford gates only (no T gate or arbitrary rotations)
- **DMRG**: 1D nearest-neighbor Hamiltonians only
- **No hardware backend**: Simulation only
- **MPS**: Adjacent-qubit 2-qubit gates only (SWAP network needed for non-adjacent)

## Test Coverage

Counts from `cargo test --release -- --list` on 2026-09-25. They drift, so
re-run the command rather than trusting the numbers.

| Location | Tests | What they cover |
|---|---|---|
| `crates/cjc-quantum` (unit + `tests/`) | 282 | Every module: gates, measurement, MPS/SVD, VQE, QAOA, stabilizer, density, DMRG, QEC, QML, fermion, Trotter, ZNE, adjoint, Wirtinger, SIMD kernels, pure backend, plus the cross-platform golden hash |
| `tests/beta_tests/quantum/` (16 files) | 344 | `.cjcl` builtins through both executors, chemistry reference energies (PySCF), value semantics, state interop, pure backend, type registration |
| `tests/beta_tests/quantum_prop/` | 8 proptest properties | Normalisation, inverse circuits, bit-identical replay, MPS/density/stabilizer ≡ dense, channel trace and purity, eval ≡ MIR on generated programs |
| `tests/beta_tests/fuzz/test_fuzz_quantum*.rs` | 14 | Seeded dispatch sweep (250 random calls per builtin, panic freedom), builtin list ≡ `dispatch.rs`, reference ≡ `dispatch.rs`, size caps |
| `tests/bolero_fuzz/` (quantum targets) | 3 | Dispatch panic freedom, fermion and ZNE determinism |
| `tests/quantum_and_piml_demos_parity.rs` | 5 | Quantum demos 01–06 (and 3 PINN demos) byte-identical in both executors |
| `crates/cjc-repro` (`dmath`) | 7 | < 1 ulp spot checks, agreement with libm, platform-independent golden hash |

## Files Modified/Created

| File | Change |
|------|--------|
| `crates/cjc-quantum/` | Quantum crate (21 source files) |
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
| `crates/cjc-quantum/src/qml.rs` | QML data re-uploading neural network |
| `crates/cjc-runtime/src/value.rs` | Added `Value::QuantumState` variant |
| `crates/cjc-snap/src/encode.rs` | Added QuantumState to non-serializable list |
| `crates/cjc-eval/src/lib.rs` | Wired `dispatch_quantum` |
| `crates/cjc-mir-exec/src/lib.rs` | Wired `dispatch_quantum` |
| `Cargo.toml` | Added cjc-quantum to workspace |
| `crates/cjc-types/src/lib.rs` | 7 quantum type variants; signatures for 67 of 84 builtins |
| `crates/cjc-quantum/src/pure.rs` | "pure" backend (MPS, Stabilizer, Density, Circuit, H2, ZNE + Jacobi SVD) |
| `tests/beta_tests/quantum/` | 344 integration tests (16 test files) |
| `tests/beta_tests/quantum_prop/` | 8 proptest properties |
