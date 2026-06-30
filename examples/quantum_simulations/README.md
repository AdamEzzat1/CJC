# Quantum simulation demos

Small, deterministic, closed-form-verifiable quantum circuits expressed in
pure CJC-Lang against the existing `cjc-quantum` builtins. None of these
demos required any new compiler, runtime, or library work — they only
*compose* primitives that ship today.

| File | What it shows | Closed-form check |
| --- | --- | --- |
| `01_single_qubit_gates.cjcl` | `H`, `X`, `Z`, and the identity `H·Z·H = X` on \|0⟩. | Probability vectors equal the analytic values to ≤1e-12. |
| `02_bell_state_z_expectation.cjcl` | Bell state `\|Φ+⟩ = (\|00⟩+\|11⟩)/√2`, derived `⟨Z₀⟩`, `⟨Z₁⟩`, `⟨Z₀Z₁⟩` from `q_probs`. Deterministic 200-shot sampling with `q_sample(c, n, seed)`. | `P(00)=P(11)=½`, `P(01)=P(10)=0`, `⟨Z₀⟩=⟨Z₁⟩=0`, `⟨Z₀Z₁⟩=+1`; forbidden outcomes never sampled. |
| `03_ry_rotation_sweep.cjcl` | `Ry(θ)\|0⟩` swept at θ ∈ {0, π/4, π/2, 3π/4, π}; verifies `P(0)=cos²(θ/2)`, `P(1)=sin²(θ/2)`, `⟨Z⟩=cos(θ)`. | 15 numerical assertions to ≤1e-12. |
| `04_ghz_n_qubits.cjcl` | GHZ states for n ∈ {3, 4, 5}: `H` on q₀, then `CNOT(0,k)` for k=1..n-1. Verifies endpoints, normalisation, and `⟨Z⊗…⊗Z⟩ = (1+(-1)ⁿ)/2`. | 15 PASS lines, all to ≤1e-12. |
| `05_h2_vqe_sweep.cjcl` | H₂ Jordan-Wigner Hamiltonian (`q_fermion_h2`) — basis-state expectations against closed-form algebra; 51-point Ry-CNOT ansatz sweep; variational-principle check. | 4 basis-state energies to ≤1e-4; sweep recovers the matrix's true E₀ = −1.8512 Ha to within 1.1%. |
| `06_zne_richardson.cjcl` | Zero-noise extrapolation (`q_zne_linear`, `q_zne_mitigate`) on synthetic linear and quadratic noise models. | Richardson recovers `E_true` exactly; coefficients are the textbook [3, −3, 1] at scales [1, 2, 3]. |

## Run

```bash
cjcl run examples/quantum_simulations/01_single_qubit_gates.cjcl
cjcl run examples/quantum_simulations/02_bell_state_z_expectation.cjcl
cjcl run examples/quantum_simulations/03_ry_rotation_sweep.cjcl
```

## Parity gate

Each demo also passes the dual-executor parity gate:

```bash
cjcl parity examples/quantum_simulations/01_single_qubit_gates.cjcl
# Verdict: IDENTICAL
```

`cjcl parity` runs both the AST tree-walk interpreter and the MIR
register-machine executor and asserts byte-equal stdout (and identical
final value). All three demos return `IDENTICAL`.

> **Note:** `cjcl run --mir-opt` enables constant-folding optimisations on
> the MIR pipeline, which is *permitted* to disagree with the canonical
> executor on rounding (and does, for `01_harmonic_oscillator_residual`).
> The architectural parity gate is over **un-optimised** MIR vs AST — that
> gate is what `cjcl parity` checks, and it is bit-equal here.

## What this proves about CJC-Lang today

- The shipped statevector simulator gives closed-form correct probabilities
  on the textbook entangling and rotation circuits, including 5-qubit GHZ.
- Z-basis expectation values for arbitrary qubits, qubit pairs, and the
  full all-qubit Z-string are **derivable from `q_probs` alone** in pure
  CJC-Lang — no Rust.
- Seeded shot sampling is reproducible across runs and across the two
  executors.
- The fermionic Hamiltonian + circuit + expectation pipeline closes the
  loop end-to-end: `q_fermion_h2 → q_fermion_expectation(circuit)`, with
  basis-state energies matching closed-form algebra and a 51-point
  variational sweep that respects the variational principle (no point
  below the matrix eigenvalue minimum) and recovers the true minimum to
  within 1.1% on a coarse grid.
- Zero-noise extrapolation (`q_zne_mitigate`, `q_zne_linear`) is exposed
  at the language level and recovers the noise-free observable exactly
  on polynomial noise models, with deterministic Richardson coefficients.

## .cjcl-callable quantum surface (verified by grep against `cjc-quantum/src/dispatch.rs`)

| Category | Builtins | Status |
| --- | --- | --- |
| Statevector construction | `qubits(n)` (1 ≤ n ≤ 26) | ✓ |
| 1-qubit Pauli + Clifford | `q_h`, `q_x`, `q_y`, `q_z`, `q_s`, `q_t` | ✓ |
| 1-qubit rotations | `q_rx`, `q_ry`, `q_rz` | ✓ |
| 2-qubit | `q_cx`/`q_cnot`, `q_cz`, `q_swap` | ✓ |
| 3-qubit | `q_toffoli`/`q_ccx` | ✓ |
| Observables | `q_probs`, `q_amplitudes`, `q_n_qubits`, `q_n_gates` | ✓ |
| Sampling / measurement | `q_sample(c, n_shots, seed)`, `q_measure(c, seed)` | ✓ (terminal-only — see limits) |
| Fermionic Hamiltonians | `q_fermion_h2`, `q_fermion_lih`, `q_fermion_new`, `q_fermion_n_terms`, `q_fermion_expectation` | ✓ |
| Trotter time evolution | `q_trotter_evolve`, `q_trotter_error` | partial — see limits |
| Noise mitigation (ZNE) | `q_zne_mitigate`, `q_zne_linear`, `q_scale_noise` | ✓ |
| Pure (no-GC) circuits | same gate set on `qubits(n, "pure")` | ✓ (separate guarded dispatch arms) |

## What this does **not** prove

- That CJC-Lang is competitive with Qiskit/PennyLane in throughput,
  algorithmic breadth, or hardware connectivity. The demos exercise a
  *subset* of the surface above and at small qubit counts (≤5).
- They do not demonstrate **variational training** of a quantum circuit
  (no parameter-shift gradients, no gradient-based VQE optimiser at the
  language level). Demo 05 is a 51-point grid sweep, not a trained VQE.
  Trained VQE infrastructure exists in `cjc-quantum/src/vqe.rs` but is
  not yet exposed as a `q_vqe_*` builtin family.
- They do not demonstrate noise-channel simulation, density matrices,
  MPS/DMRG, stabilizer circuits, or QEC at the language level. Those
  modules exist in the `cjc-quantum` crate (`density.rs`, `mps.rs`,
  `dmrg.rs`, `stabilizer.rs`, `qec.rs`) and have unit tests, but their
  builtins are not part of the surface mapped above.

## Known limitations of the language-level surface

- **No mid-circuit measurement.** `q_measure(circuit, seed)` returns
  *terminal* outcomes after executing the entire circuit. A canonical
  teleportation circuit with a Bell measurement followed by conditional
  X/Z corrections cannot be expressed in `.cjcl` source today.
- **Trotter return doesn't compose with observables.**
  `q_trotter_evolve(...)` returns a `Statevector`, but `q_probs`,
  `q_amplitudes`, and `q_fermion_expectation` accept only a `Circuit`
  (verified by reading the `with_circuit` helper in
  `cjc-quantum/src/dispatch.rs:1399`). There is currently no language-
  level way to feed a Trotter-evolved state back into an observable.
- **`adjoint`, density-matrix, MPS, DMRG, stabilizer, QEC, QML, QAOA,
  VQE optimisers, Wirtinger AD** — all present as Rust modules under
  `cjc-quantum/src/`, none yet exposed as language-level builtins.
