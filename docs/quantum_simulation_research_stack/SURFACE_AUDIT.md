# CJC-Lang Quantum Simulation — Surface Audit

**Date:** 2026-09-24 · **Branch:** `claude/cjc-quantum-audit-5a7f9c` · **HEAD:** `671dfeb`
**Scope:** `crates/cjc-quantum` (21 source files, 15,423 LOC), its wiring into
`cjc-eval` / `cjc-mir-exec`, its tests, demos, and docs. Classical simulation on
conventional hardware only; QPU/provider execution is out of scope.
**Companion docs:** [EXTERNAL_COMPARISON.md](EXTERNAL_COMPARISON.md) ·
[MISSING_FEATURES.md](MISSING_FEATURES.md) · [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) ·
[VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) · reproducible probes in [`probes/`](probes/)

> **Scope-authority note.** The audit brief names
> `docs/quantum_simulation_research_stack/STACK_ROLE_GROUP.md` as the
> authority for scope and roles. **That file does not exist** in this branch
> or in git history. The only files with that name are
> `docs/language_hardening_phase/STACK_ROLE_GROUP.md` and
> `docs/mathematics_hardening_phase/STACK_ROLE_GROUP.md`. This audit
> therefore uses the brief itself as the scope authority. Follow-up: author
> the quantum role-group doc, or point to the intended one.

### Evidence vocabulary

| Label | Meaning |
|---|---|
| **implemented** | Code exists (file:line cited) |
| **exposed** | Reachable from `.cjcl` via `dispatch_quantum` |
| **language-tested** | A test drives it from `.cjcl` through both `cjc-eval` and `cjc-mir-exec` |
| **unit-tested** | Tested only through the Rust crate API |
| **probed** | Behaviour observed in this audit by running [`probes/`](probes/) with the branch's release `cjcl` build |
| **hypothesis** | Plausible, not verified |

---

## 1. Answers to the audit questions (short form)

**Q1 — What can CJC-Lang simulate today from `.cjcl`?**
Through 78 builtin names (76 distinct operations + 2 alias pairs:
`q_cx`/`q_cnot`, `q_toffoli`/`q_ccx`), `.cjcl` source can:

- **Build and run dense statevector circuits** (1–26 qubits) with the gate set
  H X Y Z S T Rx Ry Rz CX CZ SWAP CCX, then read back probabilities, amplitudes,
  terminal measurements, and seeded shot samples.
- **Run MPS circuits** using only H, X, Ry, adjacent CNOT, and SWAP-network SWAP.
  It can read ⟨Z_i⟩ and a 1D Ising/Heisenberg energy.
- **Run CHP stabilizer circuits** (H S X Y Z CNOT, single-qubit measurement).
- **Simulate density matrices** with 6 one-qubit gates, CNOT, and depolarizing,
  dephasing, and amplitude-damping channels, then read trace, purity, entropy,
  and probabilities.
- **Call closed-loop solvers** that return only final numbers: VQE, QAOA on
  cycle graphs, DMRG, QML, and QEC on the repetition code.
- **Use two pre-built fermionic Hamiltonians** (H₂ and LiH) with expectation
  values and Trotter evolution.
- **Use ZNE helpers** (Richardson and linear extrapolation, noise scaling).

**Q2 — What exists only in the Rust crate?**
The following have no builtin, so `.cjcl` can't reach them:

- Adjoint differentiation.
- Wirtinger calculus.
- `HybridCircuit` (mid-circuit measurement with classical feed-forward).
- The AVX2/cache-blocked kernels, which production code doesn't use at all.
- On `DensityMatrix`: general Kraus channels (`apply_kraus`), `partial_trace`,
  `fidelity`, and `from_statevector`.
- `Mps::to_statevector`.
- `Graph::new` and `Graph::complete` (only `cycle` is exposed).
- Parameter-shift / ansatz building blocks from `vqe.rs` and `qaoa.rs`.
- Custom Pauli-term construction (`q_fermion_new` creates an *empty*
  Hamiltonian, and no builtin can add terms to it).
- Pure-backend `pure_trotter_evolve` and `PureStatevector::sample`.

§3 has the full list.

> **Update (2026-09-24):** risks 1 and 2 below are fixed on this branch, and
> every probe now returns a runtime error or value (see README). Risks 3 and 4
> remain open.

**Headline risks** (all **probed**, §7):
1. **Nine** malformed-input probes crash the whole `cjcl` process with a Rust
   panic (exit 101) instead of returning a runtime error. A tenth aborts on a
   16 TiB allocation (exit 127).
2. **Silent defaults / silent wrong answers** in seven probed cases (p08, p15, p16, p18, p19, p20, p24), including
   `qec_decode` on a surface code (runs the repetition-code decoder and returns
   a "correction") and duplicate-operand gates (`q_cx(c,0,0)` is a silent no-op).
3. **In-place aliasing.** Gate builtins mutate the circuit they are passed:
   `let b = q_x(a, 0); q_probs(a)` → `[0, 1]`. The `let q = q_h(q, 0)` idiom
   used everywhere hides this.
4. **Composability gap.** `q_run`, `q_trotter_evolve` return a statevector that
   **no Rust-backend observable accepts**. Every observable re-executes a *circuit*.

---

## 2. Builtin inventory (all 78 names in `dispatch.rs`)

Columns:
- **Pure:** whether a `"pure"` backend arm exists (`dispatch.rs:923-1250`).
- **TySig:** whether a type signature is registered in `crates/cjc-types`.
- **LangT:** whether the name appears in an eval+MIR test (`test_quantum_integration.rs`, `test_quantum_native_types.rs`, `test_quantum_pure_backend.rs`, `bench_50q.rs`), found by text search.
- **Demo:** used by `examples/quantum_simulations/*` (01–03 run in CI parity; 04–06 do not).

### 2.1 Dense statevector circuits

| Builtin | Args → Return | Pure | TySig | LangT | Demo | Notes |
|---|---|---|---|---|---|---|
| `qubits` | (n[, "pure"]) → Circuit | ✓ | ✓ | ✓ | 01–06 | 1 ≤ n ≤ 26 enforced → `Err` (`dispatch.rs:54`) |
| `q_h q_x q_y q_z q_s q_t` | (c, q) → c | ✓ | ✓ | h,x only | h,x,z | Qubit range checked **at execute time**, not at add time (`circuit.rs:40`, `gates.rs:75`) → `Err` (probe p09). `q_y`,`q_z`,`q_s`,`q_t` not language-tested (`q_z` appears only in demos) |
| `q_rx q_ry q_rz` | (c, q, θ) → c | ✓ | ✓ | rx,ry | ry | `q_rz` not language-tested |
| `q_cx`/`q_cnot`, `q_cz`, `q_swap` | (c, a, b) → c | ✓ | ✓ | cx,cnot,cz | cx | **Duplicate operands accepted silently** (p08). `q_swap` not language-tested |
| `q_toffoli`/`q_ccx` | (c, c1, c2, t) → c | ✗ | ✓ | ✗ | — | Pure circuit → "expected a quantum circuit" |
| `q_run` | (c) → Statevector | ✓ (PureStatevector) | ✓ | ✗ | — | Rust Statevector has **no consumer** among Rust-backend builtins (p26) |
| `q_probs` | (c) → [f64] | ✓ | ✓ | ✓ | ✓ | Re-executes the circuit |
| `q_amplitudes` | (c) → [Complex] | ✗ | ✓ | ✗ | — | Re-executes |
| `q_measure` | (c, seed) → [0/1] | ✓ | ✓ | ✓ | — | Terminal only; sequential per-qubit collapse |
| `q_sample` | (c, shots, seed) → [idx] | ✗ | ✓ | ✓ | 02 | O(shots·2ⁿ) (`measure.rs:86`); pure circuit → error (p14) |
| `q_n_qubits`, `q_n_gates` | (c) → i64 | ✓ | ✓ | ✓ | ✓ | — |

### 2.2 MPS

| Builtin | Args → Return | Pure | TySig | LangT | Notes |
|---|---|---|---|---|---|
| `mps_new` | (n, χ[, "pure"]) → MPS | ✓ | ✓ | ✓ | n=0 → panic (p05); n<0 → capacity-overflow panic (p17). Rust default χ=64 (`mps.rs:26`), pure default χ=32 (`dispatch.rs:941`) |
| `mps_h`, `mps_x` | (m, q) → m | ✓ | ✓ | ✓ | Float qubit silently truncated (p18); out-of-range → `assert!` panic (`mps.rs:459`) |
| `mps_ry` | (m, q, θ) → m | ✓ | ✓ | ✓ | Missing θ → index-out-of-bounds panic (p12, `dispatch.rs:224`) |
| `mps_cnot` | (m, c, t) → m | ✓ | ✓ | ✓ | **Adjacent only**; non-adjacent → panic (p02, `mps.rs:490`) |
| `mps_swap` | (m, a, b) → m | ✗ | ✗ | ✗ | SWAP via swap network |
| `mps_z_expectation` | (m, q) → f64 | ✓ | ✓ | ✓ | — |
| `mps_energy` | (m, "heisenberg"\|other) → f64 | ✗ | ✓ | ✗ | Any string ≠ "heisenberg" silently means Ising (p16) |
| `mps_memory` | (m) → i64 | ✓ | ✓ | ✓ | — |
| `mps_left/right/mixed_canonicalize` | (m[, center]) → m | ✗ | ✗ | ✗ | — |

No `mps_rx`, `mps_rz`, `mps_z`, `mps_s`, `mps_t`, `mps_cz`, general 2-qubit gates, sampling, amplitude readout, or truncation-error readout from `.cjcl`.

### 2.3 Stabilizer (CHP)

| Builtin | Args → Return | Pure | TySig | LangT | Notes |
|---|---|---|---|---|---|
| `stabilizer_new` | (n[, "pure"]) → Stab | ✓ | ✓ | ✓ | n=0 → `assert!` panic (`stabilizer.rs:120`) |
| `stabilizer_h/s/x/y/z` | (s, q) → s | ✓ | ✓ | h,x,s,y,z | — |
| `stabilizer_cnot` | (s, c, t) → s | ✓ | ✓ | ✓ | — |
| `stabilizer_measure` | (s, q, seed) → 0/1 | ✓ | ✓ | ✓ | Out-of-range → panic (p03) |
| `stabilizer_n_qubits` | (s) → i64 | ✓ | ✓ | ✓ | — |

No CZ, S†, reset, measure-all, Pauli-string expectation, bulk sampling, noise, or detector annotations.

### 2.4 Density matrix + noise

| Builtin | Args → Return | Pure | TySig | LangT | Notes |
|---|---|---|---|---|---|
| `density_new` | (n[, "pure"]) → DM | ✓ | ✓ | ✓ | Rust n>14 → panic (p04); **pure has no cap** → 16 TiB alloc abort at n=20 (p22) |
| `density_gate` | (d, "H"\|"X"\|"Y"\|"Z"\|"S"\|"T", q) → d | ✓ | ✓ | ✓ | Unknown name → `Err`. No rotations |
| `density_cnot` | (d, c, t) → d | ✓ | ✓ | ✓ | No CZ/SWAP/Toffoli from `.cjcl` although Rust `apply_gate` supports them |
| `density_depolarize/dephase/amplitude_damp` | (d, q, p) → d | ✓ | ✓ | ✓ | p ∉ [0,1] → `assert!` panic (p01) |
| `density_trace/purity/entropy` | (d) → f64 | ✓ | ✓ | ✓ | — |
| `density_probs` | (d) → [f64] | ✓ | ✓ | 1 use | — |

### 2.5 Algorithms (closed-loop builtins)

| Builtin | Args → Return | Pure | TySig | LangT | Notes |
|---|---|---|---|---|---|
| `vqe_heisenberg` | (n, χ, lr, iters, seed) → f64 | ✗ | ✓ | ✓ | **ZZ-only (Ising)** despite the name (`vqe.rs:9-11`) |
| `vqe_full_heisenberg` | same → f64 | ✗ | ✓ | 1 use | XX+YY+ZZ |
| `qaoa_graph_cycle` | (n) → Graph | ✗ | ✓ | ✓ | Only graph family exposed |
| `qaoa_maxcut` | (g, χ, p, lr, iters, seed) → [E, cut] | ✗ | ✓ | ✓ | Non-adjacent edges are skipped in the ansatz (`qaoa.rs:8-11`) |
| `dmrg_ising` | (n, χ, sweeps, tol) → f64 | ✗ | ✓ | ✓ | Calls a Rust fn misleadingly named `dmrg_heisenberg_1d` that runs **Ising** (`dmrg.rs:866-874`) |
| `dmrg_heisenberg` | same → f64 | ✗ | ✓ | ✓ | Calls `dmrg_full_heisenberg_1d` |
| `qml_train` | (n, layers, classes, χ, lr, epochs, seed, samples, labels) → [acc, loss[]] | ✗ | ✗ | ✗ | Non-numeric samples silently → 0.0 (p19); batch_size hard-coded 4; params not returned, so `qml_predict` cannot use trained weights without re-deriving them |
| `qml_predict` | (n, layers, classes, χ, params, input) → i64 | ✗ | ✓ | ✓ | — |
| `qec_repetition_code`, `qec_surface_code` | (d) → Code | ✗ | ✓ | rep only | — |
| `qec_syndrome` | (stab, code, seed) → [0/1] | ✗ | ✓ | ✓ | — |
| `qec_decode` | (syndrome, code) → [q] | ✗ | ✓ | 1 use | **Always** uses the repetition decoder (`dispatch.rs:556`); on a surface code returns a meaningless correction (p15). Non-int syndrome entries silently → 0 |
| `qec_logical_error_rate` | (d, p, rounds, seed) → f64 | ✗ | ✓ | ✓ | Repetition code, code-capacity noise, perfect syndrome measurement (`qec.rs:306-345`) |

### 2.6 Fermion / Trotter / mitigation

| Builtin | Args → Return | Pure | TySig | LangT | Demo | Notes |
|---|---|---|---|---|---|---|
| `q_fermion_h2` | ([ "pure" ]) → H | ✓ | ✗ | ✗ | 05 | See §6.1 |
| `q_fermion_lih` | () → H | ✗ | ✗ | ✗ | — | Self-described "simplified … reduced" (`fermion.rs:436-437`) |
| `q_fermion_new` | (n) → H | ✗ | ✗ | ✗ | — | **Unusable**: no builtin adds terms |
| `q_fermion_n_terms` | (H) → i64 | ✗ | ✗ | ✗ | 05 | — |
| `q_fermion_expectation` | (H, circuit) → f64 | ✓ (H, PureStatevector) | ✗ | ✗ | 05 | Rust path takes a *circuit*, not a state; size mismatch → `assert_eq!` panic (`fermion.rs:148`) |
| `q_trotter_evolve` | (H, circuit, t, steps, order) → Statevector | ✗ | ✗ | ✗ | — | Result not consumable (p13); size mismatch → panic (p06); steps=0 → panic (p07); order ∉ {2} silently → 1st order (p20) |
| `q_trotter_error` | (H, t, steps, order) → f64 | ✗ | ✗ | ✗ | — | Same silent order default |
| `q_zne_mitigate` | (scales[], values[][, "pure"]) → [E0, coeffs[]] | ✓ | ✗ | ✗ | 06 | Duplicate scales → proper `Err` (p23) ✓ |
| `q_zne_linear` | (λ1, v1, λ2, v2) → f64 | ✗ | ✗ | ✗ | 06 | — |
| `q_scale_noise` | (p, scale[, type]) → f64 | ✗ | ✗ | ✗ | — | Unknown type silently → depolarizing (p24) |

### 2.7 Introspection

| Builtin | Notes |
|---|---|
| `quantum_inspect` | Pure MPS / stabilizer / density only → `Map`. Pure circuit/statevector and all Rust-backend states → `Err`. Returns a read-only snapshot; there is no builtin to write a modified map back into a state. |

**Type-signature coverage:** 62 of 78 names have a signature in `cjc-types`.
The 16 without one: `mps_left_canonicalize`, `mps_right_canonicalize`,
`mps_mixed_canonicalize`, `mps_swap`, all 5 `q_fermion_*`, `q_trotter_evolve`,
`q_trotter_error`, `q_zne_mitigate`, `q_zne_linear`, `q_scale_noise`, `qml_train`,
`quantum_inspect`. `docs/QUANTUM_SIMULATION.md:458` claims "Every quantum builtin
has a registered type signature".

**Static checking in practice (probed, p25):** `cjcl check` reports the
quantum type mismatch `mps_h(qubits(2), 0)`, but the expected/found types are
reversed ("expected `QuantumCircuit`, found `QuantumMps`"). `cjcl run` does not
run the checker, so the same program fails only at runtime ("expected MPS").
`cjcl check` also reports **28 errors on the valid, shipped demo
`01_single_qubit_gates.cjcl`**, so it can't be used as a gate for quantum
programs today.

---

## 3. Crate-only capabilities (implemented, not exposed)

| Capability | Location | Tests | Why it matters |
|---|---|---|---|
| Adjoint differentiation (diagonal observables) | `adjoint.rs:51` | 12 unit | Only gradient method with O(1) extra state memory; not exposed |
| `HybridCircuit` mid-circuit measure + `if_then` feed-forward | `adjoint.rs:218-290` | unit (teleportation) | Mid-circuit measurement/classical control absent from `.cjcl` |
| Wirtinger duals, parameter-shift helper | `wirtinger.rs` | 11 unit | — |
| AVX2 + cache-blocked 1q kernels | `simd_kernel.rs` | 6 unit | **Dead code**: not called by `gates.rs` or anywhere outside `simd_kernel.rs` |
| General Kraus channel, partial trace, fidelity, `from_statevector` | `density.rs:166,360,466,511` | unit | Noise modelling beyond 3 fixed channels |
| Density CZ / SWAP / Toffoli | `density.rs:205-220` | unit | — |
| MPS → statevector, general 4×4 gate via swap network | `mps.rs:584,802` | unit | Only SWAP matrix wired |
| `Graph::new(edges)`, `Graph::complete` | `qaoa.rs:42,65` | unit | QAOA limited to cycles from `.cjcl` |
| VQE/QAOA ansatz builders, parameter-shift gradients, `energy_history` | `vqe.rs:455-611`, `qaoa.rs:85-100,265` | unit | Only final energy returned to `.cjcl` |
| `PauliTerm`, `jw_one_body`, `jw_two_body` | `fermion.rs:44,225,318` | unit | No user-defined Hamiltonians from `.cjcl` |
| Richardson `run_zne` driver | `mitigation.rs:218` | unit | — |
| QML dataset loader / image preprocessing | `qml.rs:361-455` | unit | — |
| Pure `pure_trotter_evolve`, `PureStatevector::sample` | `pure.rs:1302,1536` | unit | Pure backend lacks these arms |
| Stabilizer → statevector (n ≤ 12) | `stabilizer.rs:487` | unit | Cross-check path only in Rust |

## 4. Capability classification

| Class | Definition | Members |
|---|---|---|
| **`.cjcl`-ready** | Exposed + language-tested + malformed input → `Err` | Circuit construction; gates H, X, CX, CZ, Rx, Ry; `q_probs`, `q_measure`, `q_sample` (Rust backend); `q_zne_mitigate` (demo-tested, error path probed OK) |
| **Partial** | Exposed, but panics on malformed input, applies silent defaults, or has a narrow gate set | MPS (panics; 4 gates), stabilizer (panics), density (panics; fixed channels), VQE/QAOA/DMRG (closed-loop, final value only), QML (silent coercion; weights not returned), QEC (surface decode wrong), Trotter (panics; output unusable), pure backend (subset of ops, uncapped density) |
| **Exposed, untested at language level** | Exposed but not language-tested | `q_y q_z q_s q_t q_rz q_swap q_toffoli q_amplitudes q_run`, `mps_energy`, `mps_*canonicalize`, `mps_swap`, `qec_surface_code`, `qml_train`, `q_fermion_lih/new`, `q_trotter_*`, `q_scale_noise` |
| **Tested-demo-only** | Only exercised by demos 04–06 (not in CI) | `q_fermion_h2`, `q_fermion_n_terms`, `q_fermion_expectation`, `q_zne_linear` (and `q_zne_mitigate` beyond its probe) |
| **Crate-only** | §3 | Adjoint, Wirtinger, HybridCircuit, SIMD, Kraus/partial trace/fidelity, general graphs, Pauli-term construction |
| **Missing** | Not implemented anywhere | Circuit interchange (OpenQASM 2/3, QIR); general/custom unitaries & controlled-U; U3/phase/iSWAP/√X gates; Pauli-string observables on statevector; reset; readout-error model; noise model object; trajectory (Monte-Carlo wavefunction) noise; multithreading; GPU; distributed; detector error model; MWPM/union-find decoder; surface-code decoder; circuit-level QEC noise; parameter binding / symbolic parameters; state serialization (QuantumState is non-serializable in `cjc-snap`) |

## 5. Composability matrix (Rust backend)

The input each consumer accepts (✓), rejects with `Err` (E), or panics on (P):

| Consumer ↓ / Value → | Circuit | Statevector (`q_run`, `q_trotter_evolve`) | MPS | Stabilizer | Density | FermionicHamiltonian |
|---|---|---|---|---|---|---|
| `q_probs` / `q_amplitudes` / `q_sample` / `q_measure` | ✓ (re-executes) | E ("expected a quantum circuit", p26) | E | E | E | E |
| `q_fermion_expectation(H, ·)` | ✓ (re-executes); P on size mismatch | E (p13) | E | E | E | — |
| `q_trotter_evolve(H, ·)` | ✓; P on size mismatch (p06) | E | E | E | E | — |
| `qec_syndrome(·, code)` | E | E | E | ✓ | E | E |

Consequences:
- No `.cjcl` program can read out, measure, or evaluate an observable on a
  Trotter-evolved state. `q_trotter_evolve` is therefore a dead end at the
  language level. The examples README acknowledges this.
- A statevector cannot be turned into a density matrix, MPS, or stabilizer
  state from `.cjcl`, and none of those can be turned back into one.
- The pure backend is the exception: its `q_run` returns a `PureStatevector`
  that pure `q_fermion_expectation` accepts.

## 6. Correctness and semantics findings (code-read, some probed)

### 6.1 H₂ Hamiltonian label (`fermion.rs:366-378`) — **verified in [VERIFY_FOLLOWUPS §1](VERIFY_FOLLOWUPS.md#1-h-hamiltonian-fermionrsh2_hamiltonian)**
Update: an independent from-scratch STO-3G FCI **and PySCF 2.14 + OpenFermion 1.8.1** confirm the hypothesis below. The ground state is within 1.75×10⁻⁴ Ha.
**New finding:** the matrix's excited N=2 and N=4 levels are off by 19–24 mHa, so only ground-state claims are safe.

Original analysis:
The docstring states "Exact ground state energy: −1.1373 Hartree" and labels
`g0 = −0.4804` as "Nuclear repulsion + constant". The minimum eigenvalue of the
shipped 2-qubit matrix is **−1.851199 Ha**; demo 05 verifies this with numpy.
**Hypothesis:** the coefficients are the *electronic* Hamiltonian. −1.8512 plus
the nuclear repulsion 1/R (R = 0.7414 Å = 1.401 bohr, 1/R ≈ 0.7138 Ha) gives
≈ −1.1374 Ha, which matches the docstring's total energy. If so, the `g0` comment
is wrong (nuclear repulsion is *not* included), and callers comparing to −1.1373
will see a 0.71 Ha discrepancy. Verify against OpenFermion/PySCF before relying
on either number.

### 6.2 LiH Hamiltonian — **refuted as LiH in [VERIFY_FOLLOWUPS §1](VERIFY_FOLLOWUPS.md#1-chemistry-hamiltonians-fermionrs)**
Update: PySCF 2.14 gives a LiH FCI total of −7.882762 Ha, so the docstring's number is right. But CJC's extracted matrix has its N=2 ground at
−10.009 Ha (true 2e/2o active space: −7.863 Ha). Its "nuclear repulsion" constant is −7.4983, whereas the true value is +1.0269, and its one-body
diagonal is essentially H₂'s. It is not a LiH Hamiltonian.

Original note:
`fermion.rs:436-437` says it has been reduced "to the most significant terms".
Its docstring claims a ground-state energy of about −7.8825 Ha, which nothing in
the repo validates. Treat LiH as a toy Hamiltonian.

### 6.3 Depolarizing-channel parameterisation (`density.rs:575-605`)
The Kraus set √(1−p)·I, √(p/3)·{X, Y, Z} implements ρ → (1−p)ρ + (p/3)(XρX+YρY+ZρZ).
That equals "replace with I/2 with probability **4p/3**". The docstring says
"with probability p". This is a documentation error, not an implementation bug,
but it matters for cross-simulator comparison: Qiskit's `depolarizing_error(λ)`
uses the replace-with-maximally-mixed convention.

### 6.4 Naming that misstates physics
- `vqe_heisenberg` simulates ZZ-only (Ising).
- The Rust function `dmrg_heisenberg_1d` runs **Ising**, while the `.cjcl`
  builtin `dmrg_ising` calls it, so behaviour is right but the code is misleading.
- `docs/QUANTUM_SIMULATION.md:306,310` writes the DMRG Hamiltonians with a
  leading minus sign, but the code builds +ZZ / +(XX+YY+ZZ) (`dmrg.rs:20-21,42-44`).

### 6.5 In-place mutation / aliasing (probed, p10)
`dispatch.rs` gate arms mutate the circuit through `RefCell::borrow_mut()` and
return a clone of the **same** `Rc` (`dispatch.rs:1554-1558`). MPS, stabilizer,
and density arms follow the same pattern. Observed:
`let a = qubits(1); let b = q_x(a, 0); q_probs(a)` returns `[0, 1]`. Every
test and demo shadows the variable (`let q = q_h(q, 0)`), which hides this. It
contradicts the copy-on-write value semantics of other CJC aggregates. Both
executors agree, so this is a semantics issue, not a parity issue.

### 6.6 Sampling edge case (`measure.rs:98-99`)
If rounding leaves the cumulative sum below `r`, `sample_basis_state` returns
index `2ⁿ−1` even when that state has probability 0. **Hypothesis:** this is
reachable when Σp < 1 by a few ulp and `r` is near 1. It needs a targeted test
(VERIFICATION_PLAN §3.2).

### 6.7 QML gradient method mislabelled
The module header says "Parameter-shift gradient training" (`qml.rs:8`). The
implementation uses central finite differences with ε = 1e-4 (`qml.rs` in
`qml_gradient`, comment "Uses central finite-difference"), and
`docs/QUANTUM_SIMULATION.md:377` says finite differences.

### 6.8 Determinism claims: scope
- **Supported by code:** a fixed iteration order; no `HashMap`; SplitMix64 with
  explicit seed threading; no FMA in `mul_fixed`; Kahan summation in
  reductions.
- **Now tested, and fails as stated:** the "across platforms" claim (`lib.rs:4`, `measure.rs:25`). See [VERIFY_FOLLOWUPS §2](VERIFY_FOLLOWUPS.md#2-cross-platform-bit-identity-of-rotation-gates).
  - Windows (MSVC) and Linux (glibc 2.36) `sin`/`cos` return different bits for 6.0% of gate angles (13,208 / 220,020), including θ = π/2.
  - `q_probs(Ry(π/2)|0⟩)` prints P(1)=0.5000000000000001 on Windows, where a correctly rounded libm gives 0.4999999999999999.
  - Earlier note (kept for context):
  Rx/Ry/Rz use `f64::sin/cos` from the platform math library, whose results
  are not guaranteed to be bit-identical across OSes.
- **Tested today:** determinism is checked only by same-process repeat tests
  in this audit's scope.

## 7. Probe results (both executors via `cjcl parity`, release build of this branch)

Programs are in [`probes/`](probes/). "Crash" = the `cjcl` process terminated
(Rust panic exit 101 or allocation abort exit 127). Because `parity` runs eval
first, a crash means neither executor produced a result.

| # | Program (abridged) | Result | Class |
|---|---|---|---|
| p01 | `density_depolarize(d, 0, 1.5)` | **Crash** — panic `density.rs:585` | Panic |
| p02 | `mps_cnot(m, 0, 2)` | **Crash** — panic `mps.rs:490` "adjacent qubits" | Panic |
| p03 | `stabilizer_measure(stabilizer_new(2), 5, 1)` | **Crash** — panic `stabilizer.rs:395` | Panic |
| p04 | `density_new(15)` | **Crash** — panic `density.rs:149` | Panic |
| p05 | `mps_new(0, 4)` | **Crash** — panic `mps.rs:433` | Panic |
| p06 | `q_trotter_evolve(q_fermion_h2(), qubits(3), …)` | **Crash** — `assert_eq!` `trotter.rs:126` | Panic |
| p07 | `q_trotter_evolve(…, n_steps = 0, …)` | **Crash** — panic `trotter.rs:127` | Panic |
| p08 | `q_cx(c, 0, 0)` after `q_h(c, 0)` | Returns `[0.5…, 0.5…, 0, 0]` — CX silently ignored | Silent |
| p09 | `q_h(qubits(2), 7)` | `Err` "qubit 7 out of range (n_qubits=2)" (both executors) | ✓ |
| p10 | `let b = q_x(a, 0); q_probs(a)` | `[0, 1]` — **`a` was mutated** | Semantics |
| p11 | `q_h(c)` | `Err` "gate requires (circuit, qubit), got 1 args" | ✓ |
| p12 | `mps_ry(m, 0)` | **Crash** — index out of bounds `dispatch.rs:224` | Panic |
| p13 | `q_fermion_expectation(h, q_trotter_evolve(…))` | `Err` "expected a quantum circuit" | Composability |
| p14 | `q_sample(qubits(2, "pure"), 5, 1)` | `Err` "expected a quantum circuit" — misleading message for a backend mismatch | Backend |
| p15 | `qec_decode(qec_syndrome(stabilizer_new(20), qec_surface_code(3), 1), code)` | `[3]` — repetition decoder applied to a surface code, "corrects" an error-free state | **Silent wrong** |
| p16 | `mps_energy(m, "heisenbrg")` | `3` (Ising energy) | Silent |
| p17 | `mps_new(-1, 4)` | **Crash** — "capacity overflow" | Panic |
| p18 | `mps_h(m, 1.9)` | Applies H to qubit 1 (float truncated by `extract_int`, `dispatch.rs:1477`); circuit path rejects floats | Silent / inconsistent |
| p19 | `qml_train(…, [[0.1, "x"], …], …)` | Trains; `"x"` treated as 0.0 | Silent |
| p20 | `q_trotter_error(h, 1.0, 4, 3)` | Same value as order 1 | Silent |
| p21 | `q_sample(c, 8, -1)` | Works (seed `-1 as u64`); deterministic | ✓ (document) |
| p22 | `density_new(20, "pure")` | **Crash** — "memory allocation of 17592186044416 bytes failed" (exit 127) | Unbounded alloc |
| p23 | `q_zne_mitigate([1.0, 1.0], …)` | `Err` "Vandermonde system is singular…" | ✓ |
| p24 | `q_scale_noise(0.1, 2.0, "amplitude-damping")` | `0.19` (depolarizing formula) | Silent |
| p25 | `mps_h(qubits(2), 0)` | Runtime `Err` "expected MPS"; `cjcl check` flags it with reversed expected/found | ✓ runtime / checker caveat |
| p26 | `q_probs(q_run(c))` | `Err` "expected a quantum circuit" | Composability |

Totals: 9 panics + 1 allocation abort; 7 silent defaults/wrong answers (p08,
p15, p16, p18, p19, p20, p24) plus the p10 aliasing semantics; 2 composability errors (p13, p26) and 1 backend-mismatch error (p14); and clean
`Err`s (p09, p11, p23, p25 runtime, and p21 as documented behaviour). Every
non-crashing probe gave identical eval and MIR results.

## 8. Test suites run during this audit

See "Test execution record" at the end of this file. Summary: 280/280 unit tests pass; the beta suite has **1 deterministic failure** (QAOA); demo parity, Bolero, and smoke benches pass.

## 9. Eval vs MIR parity coverage

- **Mechanism:** both executors call the same `cjc_quantum::dispatch_quantum`
  (`cjc-eval/src/lib.rs:2783`; `cjc-mir-exec/src/lib.rs:2326,2813`). MIR-exec
  adds a per-name inline cache (`CallDispatch::Quantum`). Because both share
  the implementation, parity failures could only come from argument
  marshalling or value handling in the executors. This explains why every
  probe agreed.
- **Coverage (text-search upper bound):** 51 of 78 names appear in a file that runs eval+MIR. 27 do not (listed in
  VERIFICATION_PLAN §2.2).
- **Comparison method:** parity tests compare `format!("{}", value)` strings,
  not `to_bits()`.
- **Demos:** 01–03 are gated in `tests/quantum_and_piml_demos_parity.rs`. 04–06
  are described there as "verified out-of-band via `cjcl parity`" and are not
  in CI.
- **`--mir-opt`:** no quantum-specific test.

## 10. Documentation drift

| # | Where | Claim | Reality (evidence) | Status (2026-09-25) |
|---|---|---|---|---|
| D1 | `crates/cjc-quantum/src/lib.rs:17` | "No noise model (pure unitary evolution only)" | `density.rs` implements 3 noise channels + Kraus | Fixed: `lib.rs` header rewritten |
| D2 | `docs/QUANTUM_SIMULATION.md` architecture list | 16 files | Omits `fermion.rs`, `trotter.rs`, `mitigation.rs`, `pure.rs` (21 files) | Fixed: architecture list has all 21 files |
| D3 | same, builtin reference | Lists MPS…QML builtins | Omits all 10 fermion/Trotter/ZNE builtins, `mps_*canonicalize`, `mps_swap`, `q_run` | Fixed: new 82-row Builtin Reference, checked by `fuzz_dispatch_reference_documents_every_builtin` |
| D4 | same, QAOA example | `qaoa_maxcut(&graph, 2, 20, 0.1, 42)`; `result.best_cut` | Signature is `(graph, p, max_bond, lr, iters, seed)`; fields are `energy`, `cut_value` (`qaoa.rs:85-100,359`) | Fixed (2026-09-24) |
| D5 | same, density example | `Gate::Cx`, `apply_depolarizing`, `rho.fidelity(&x)` | `Gate::CNOT`, `apply_single_qubit_channel(q, &depolarizing_channel(p))`, `DensityMatrix::fidelity(&a, &b)` | Fixed (2026-09-24) |
| D6 | same, QEC example | `encode_repetition`, `extract_z_syndrome`, `decode_minimum_weight`; "Decoder: minimum-weight matching" for surface code | None of those functions exist; `qec.rs:14` says the 2D decoder is **not implemented** | Fixed (2026-09-24) |
| D7 | same, Wirtinger example | `parameter_shift_gradient(closure, θ, q, weights)` | Signature is `(e_plus, e_minus)` (`wirtinger.rs:194`) | Fixed (2026-09-24) |
| D8 | same, SIMD section | Presented as an active optimisation | Kernels unused by production path | Fixed in docs (section says "not used"). Kernels still unwired |
| D9 | same, DMRG | Hamiltonians with a leading minus | Code uses +ZZ / +(XX+YY+ZZ) | Fixed (2026-09-24) |
| D10 | same, stabilizer | "O(n) per gate" and "O(n^2) per gate" in the same section | Gates are O(n) (column update over 2n rows); measurement O(n²) | Fixed |
| D11 | same, type system | "Every quantum builtin has a registered type signature"; "type checker validates quantum programs statically" | 16/78 lack signatures; `cjcl run` does not type-check; `cjcl check` errors on valid demo 01 | Fixed in docs (65/82 typed; `cjcl run` untyped; `cjcl check` false errors). Underlying gaps open |
| D12 | same, pure backend | "Both backends guarantee bit-identical output" **and** "Z-expectations match to 1e-10" | Self-contradictory; untested at bit level | Fixed: each backend self-bit-identical, cross-backend 1e-10 |
| D13 | same, pure backend | "All prop/fuzz/determinism tests pass for both backends" | Fuzz tests (`test_fuzz_quantum.rs`) use the Rust crate API only | Fixed: states which suites cover the pure backend |
| D14 | `pure.rs` header / doc table | "CJC all the way down"; "modifiable without recompiling"; "AD integration" | Pure backend is Rust code over `Vec<f64>`; state is inspectable read-only via `quantum_inspect`; no write-back; no AD path | Fixed: `pure.rs` header and doc table rewritten |
| D15 | same, tests section | "224 unit", "342 integration", "8 property tests in quantum_prop/" | 280 unit markers; 311 in `beta_tests/quantum/`; `quantum_prop` compiles **0** tests | Fixed: counts regenerated (282 / 344 / 8 / 14) |
| D16 | same, type table | Density "1-12" qubits; doc elsewhere "~13-14"; `density.rs` header "~12-13" | `MAX_QUBITS = 14` (Rust); pure uncapped | Fixed: 14 everywhere (pure capped via dispatch) |
| D17 | `examples/quantum_simulations/README.md` "What this does not prove" | Density, MPS, DMRG, stabilizer, QEC, QML, QAOA, VQE "none yet exposed as language-level builtins" | All are exposed (§2); only adjoint/Wirtinger/HybridCircuit are unexposed | Fixed: README lists every family |
| D18 | same README | `with_circuit` at `dispatch.rs:1399`; "Pure (no-GC) circuits" | `with_circuit` is at `dispatch.rs:1509`; the pure backend is unrelated to NoGC | Fixed: stale line ref removed; "no-GC" corrected |
| D19 | `qml.rs:8` | "Parameter-shift gradient training" | Finite differences | Fixed: "finite differences" |
| D20 | `fermion.rs:370,378` | H₂ ground state −1.1373; g0 includes nuclear repulsion | Matrix is electronic-only (min −1.8512); verified with PySCF (§6.1) | Fixed (2026-09-24, PySCF-verified) |
| D26 | `fermion.rs:428-440` | LiH, 4-qubit, ground ≈ −7.8825 Ha; "nuclear_repulsion = −7.4983" | Matrix N=2 ground −10.009 Ha; true E_nuc +1.027 Ha; not LiH (§6.2) | Fixed: regenerated from PySCF (2026-09-24) |
| D21 | `density.rs:575` | Depolarizing "with probability p … maximally mixed" | Effective replacement probability 4p/3 (§6.3) | Fixed: docstring states 4p/3 |
| D22 | `docs/QUANTUM_SIMULATION.md` perf tables | Before/after timings | No harness, machine, or commit recorded (BENCHMARK_PLAN §1) | Partly fixed (2026-09-25): a "Measured Performance (harness)" section with machine, commit, and method now sits beside the old tables, which stay labelled unverified |
| D23 | `tests/bench_50q.rs` | `bench_20q_dmrg` | Runs 8 qubits | Fixed: renamed `bench_8q_dmrg` |
| D24 | `cjc-cli/src/lib.rs:40` | `cjcl --version` → 0.1.4 | Workspace crates are 0.1.11; do not use `--version` as benchmark provenance | Fixed: `env!("CARGO_PKG_VERSION")` (0.1.11) |
| D25 | `CJC-Lang_Obsidian_Vault/08_Advanced_Computing/Quantum Simulation.md:53` | "~25-30 qubits max" | Hard cap 26 (`dispatch.rs:54`) | Fixed: vault note says 26 (and 14 for density) |

---

## Test execution record

Run on 2026-09-24, Windows 11 x86-64, `cargo test --release`, this branch at `671dfeb` (docs-only changes on top).

| Command | Result |
|---|---|
| `cargo test --release -p cjc-quantum` | **280 passed, 0 failed** (unit) |
| `cargo test --release --test test_beta_tests` (all beta tests incl. quantum, hardening, fuzz) | **429 passed, 1 FAILED**: `beta_tests::quantum::test_qaoa::qaoa_4_cycle_finds_good_cut` (`tests/beta_tests/quantum/test_qaoa.rs:165`, assertion `result.energy >= 1.5`) |
| `cargo test --release --test quantum_and_piml_demos_parity` | **5 passed, 0 failed** |
| `cargo test --release --test bolero_fuzz -- fuzz_fermion fuzz_zne` | 2 passed. **Not evidence**: both targets can't fail (VERIFICATION_PLAN T-1) |
| `cargo test --release --test bench_50q --test bench_dual_mode` | 14 + 17 passed (timed smoke tests only; not benchmarks, and no timings recorded here) |
| 26 `.cjcl` probes via `cjcl parity` (§7) | 9 panics, 1 alloc abort, 7 silent defaults, 1 aliasing finding, 5 clean errors/behaviours, 3 composability/backend errors |

**The failing QAOA test is deterministic** (seeded), so it is a regression or a
stale threshold, not flakiness. **Hypothesis on the cause:** commit `66b65bd`
("wide-matrix SVD routing") changed `svd_sign_stabilized`, which MPS-based QAOA
calls on every 2-qubit gate. Its commit message records verification of
`cargo test -p cjc-quantum --lib` only, not the integration suite. This audit did
not bisect or modify code. Next step: run the test at `66b65bd^` and `66b65bd`,
then fix either the SVD path or the threshold with a documented reason.
Meanwhile, the QAOA results in W7 of the benchmark plan should not be trusted.
