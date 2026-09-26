# ADR-0045 Quantum State Interop Between Builtins

- **Status:** Accepted (2026-09-24)
- **Crate:** `cjc-quantum` (`dispatch.rs` only; no new types)
- **Companion docs:** `docs/quantum_simulation_research_stack/MISSING_FEATURES.md` (composability gaps), `docs/QUANTUM_SIMULATION.md`
- **Related:** [[ADR-0044 Quantum Value Semantics]], [[ADR-0016 Language-Level GradGraph Primitives]] (satellite dispatch precedent)

## Context

The quantum audit found that the builtins do not compose. Each family was
wired as a silo:

- Observables (`q_probs`, `q_amplitudes`, `q_sample`, `q_measure`,
  `q_fermion_expectation`, `q_trotter_evolve`) took only a **circuit** and
  re-executed it on every call. `q_run` returned a statevector that no
  observable accepted, so its result was a dead end. `q_trotter_evolve`
  returned a statevector, so a Trotter-evolved state could not be measured
  or have its energy taken.
- The only expectation value was "whole Hamiltonian". A single Pauli
  observable such as ⟨Z₀Z₁⟩ could only be built through `q_fermion_*`
  internals. `q_fermion_new(n)` created an empty Hamiltonian with no way to add
  a term, so it was unusable.
- Density matrices could start only from |0…0⟩. There was no way to take a
  pure state prepared by a circuit and then apply noise.
- The pure backend took a statevector in `q_fermion_expectation` where the
  Rust backend took a circuit. The two backends' signatures disagreed.

## Decision

**A "state argument" accepts a circuit or a statevector**, from either
backend where the data is interchangeable:

| Argument given | Treated as |
|---|---|
| `Circuit` | `circuit.execute()` (same as before) |
| `Statevector` (from `q_run`, `q_trotter_evolve`) | used directly |
| `PureCircuit` / `PureStatevector` | executed / amplitudes copied (pure backend arms keep priority where they exist) |

The observables that now accept a state argument:

| Builtin | New signature (first arg widened) |
|---|---|
| `q_probs`, `q_amplitudes`, `q_n_qubits` | `(state)` |
| `q_sample` | `(state, n_shots, seed)` |
| `q_measure` | `(state, seed)` |
| `q_fermion_expectation` | `(hamiltonian, state)`. The pure arm also accepts a `PureCircuit` |
| `q_trotter_evolve` | `(hamiltonian, state, time, n_steps, order)` |

Sampling and measurement on a statevector use the same functions that the
circuit path calls after executing (`measure::sample_basis_state`,
`measure::measure_all`). So for any circuit `c` and seed `s`,
`q_sample(c, k, s) == q_sample(q_run(c), k, s)` bit for bit. A test locks
this in.

**New builtins (3):**

| Builtin | Returns | Notes |
|---|---|---|
| `q_expect_pauli(state, pauli)` | `f64` | ⟨ψ\|P\|ψ⟩ for a Pauli string over `{I,X,Y,Z}`. Character k acts on qubit k (the convention `lih_hamiltonian` already uses). The length must equal n. Reuses the Kahan-summed `pauli_expectation` kernel. |
| `q_fermion_add_term(h, pauli, coeff)` | new Hamiltonian | Adds `coeff · P`. Follows the value semantics of [[ADR-0044 Quantum Value Semantics]]: a Hamiltonian is a description, so `h` is unchanged. Both backends. |
| `density_from_state(state)` | `DensityMatrix` | ρ = \|ψ⟩⟨ψ\| via the existing `DensityMatrix::from_statevector`. Limited to 14 qubits, the same cap as `density_new`. A pure-backend state gives a `PureDensity`. |

Validation follows the boundary rules from the audit fixes. Pauli characters
outside `IXYZ`, a length mismatch, a non-finite coefficient, or a state larger
than the density cap all return `Err`. None of them panic.

## Alternatives rejected

- **A separate `q_*_sv` builtin per observable** (`q_probs_sv`, …). This
  doubles the surface to 12+ names for one idea. Widening the argument type is
  backward compatible because every old call still type-checks.
- **Implicit caching of the executed statevector inside the circuit value.**
  That is the performance item (BENCHMARK_PLAN), not a semantics item. Being
  explicit (`let psi = q_run(c)`) gives users the same saving today without
  hidden state.
- **A general `q_pauli_sum([...])` constructor.** `q_fermion_new` plus repeated
  `q_fermion_add_term` covers it. Adding a pair-array constructor later is
  additive.

## Consequences

- A variational loop can execute once and take many observables:
  `let psi = q_run(c); let e = q_fermion_expectation(h, psi); let zz = q_expect_pauli(psi, "ZZII");`
- Trotter output can be measured and fed into a noise model through
  `density_from_state`.
- The existing circuit-taking call sites are unchanged, as are their output bits.
- The builtin count grows by 3 here, plus `q_copy` from ADR-0044: 78 → 82
  names. The fuzz sweep's BUILTINS list, the Bolero target, and the arity
  table grow with it, and a source-parsing test keeps them in sync.
- **Tests:** 16 in `tests/beta_tests/quantum/test_quantum_state_interop.rs`,
  covering bit-identity between circuit and `q_run` inputs, Pauli-string
  convention, H = Σ terms, pure-backend paths, and executor parity.
