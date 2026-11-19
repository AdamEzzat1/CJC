# CJC-Lang Quantum Simulation — Missing Features (prioritised)

**Date:** 2026-09-24 · **Branch:** `claude/cjc-quantum-audit-5a7f9c` (`671dfeb`)
**Companion docs:** [SURFACE_AUDIT.md](SURFACE_AUDIT.md) · [EXTERNAL_COMPARISON.md](EXTERNAL_COMPARISON.md) ·
[BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) · [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md)

> **Resolution status (2026-09-24, same branch; see CHANGELOG `[Unreleased]`).**
> - **Fixed:** P0-1 (panics → runtime errors), P0-2 (silent defaults → errors),
>   P0-7 (Bolero targets, duplicates, `quantum_prop`, demos 04–06), P0-8 (LiH
>   regenerated from PySCF; H₂ documented), P0-9 (QAOA; the root cause was a
>   frozen optimizer, see below).
> - **Found while fixing and also fixed:** `StabilizerState::to_statevector`
>   returned a zero vector for states orthogonal to |+…+⟩.
> - **Decided and implemented (2026-09-24, ADRs in `CJC-Lang_Obsidian_Vault/13_ADRs/`):**
>   P0-3 → ADR-0044 (circuits are values, simulator states are handles,
>   `q_copy`), P0-4 → ADR-0045 (state arguments, `q_expect_pauli`,
>   `q_fermion_add_term`, `density_from_state`), P1-8 → ADR-0046
>   (`cjc_repro::dmath`; `cjc-quantum` is now bit-identical across OSes).
>   P0-3 did **not** follow recommendation (a): measurement showed Rc
>   copy-on-write copies on every call in this runtime (see ADR-0044).
> - **Implemented 2026-09-25:** P0-5 (benchmark harness, `bench/quantum_compare`),
>   P1-1 (OpenQASM 2.0 import/export), and P1-2 items 1, 2, and 4 (execution
>   cache, batch sampler, threaded kernels); P1-2 item 3 (SIMD) was measured
>   and rejected. The harness also found and fixed a wrong-sign Jacobi rotation
>   in the MPS SVD.
> - **Still open:** P0-6 (remaining doc drift), and everything else P1 and
>   below.

**Priority definitions**

| Priority | Meaning |
|---|---|
| **P0** | Blocks a *credible* correctness or performance claim. Fix before publishing any comparison. |
| **P1** | Major capability gap relative to mature classical simulators, or a missing enabler for cross-validation. |
| **P2** | Meaningful breadth or ergonomics gap. |
| **P3** | Nice-to-have, research, or long-horizon. |

**Out of scope, not a gap:** QPU/provider execution, pulse control, and cloud
job submission. CJC targets classical simulation on conventional hardware.

Architecture note: every recommendation must respect the CLAUDE.md prime
directives. That means determinism (fixed reduction order, no FMA, BTreeMap,
seeded SplitMix64), both executors agreeing, and minimal language primitives.
Anything larger belongs in libraries (e.g. Bastion). Items that would need a new
`Value` variant or a new external dependency are flagged as **architecture
decisions** and should get an ADR, not a quiet merge.

---

## Validation of the brief's suggested P0/P1 list

| Brief's item | Verdict | Evidence |
|---|---|---|
| P0: No published CJC-vs-external benchmark suite | **Confirmed P0** | No such harness exists; `bench_50q.rs` / `bench_dual_mode.rs` are timed smoke tests (BENCHMARK_PLAN §1) |
| P0: Documentation drift | **Confirmed P0** | 25 drift items (SURFACE_AUDIT §10), including non-existent APIs in examples and a README denying exposed features |
| P0: Uneven composability | **Confirmed P0** | Statevectors from `q_run`/`q_trotter_evolve` have no consumer (probes p13, p26; SURFACE_AUDIT §5) |
| P1: No OpenQASM/QIR | **Confirmed P1** | Nothing in the crate; also blocks benchmark cross-validation |
| P1: No GPU/distributed backend | **Confirmed, re-scoped P2** for GPU/distributed. **Promoted P1**: *CPU* multithreading + SIMD wiring, because it is the cheaper and larger lever and prerequisite for GPU determinism design |
| P1: Limited gate/operator/observable algebra | **Confirmed P1** | Fixed 13-gate enum; no custom unitaries; no Pauli-string observables on a statevector |
| P1: Noise less broad | **Confirmed P1** | 3 channels from `.cjcl`, no noise model/readout/trajectories |
| P1: Mid-circuit measurement not exposed | **Confirmed P1** | `HybridCircuit` crate-only (`adjoint.rs:218`) |
| P1: Stabilizer/QEC lacks Stim-class tooling | **Confirmed P1**, with a **P0 sub-item**: surface-code decode returns wrong answers today (p15) |
| P1: Differentiable workflows | **Confirmed P1** | Adjoint crate-only; no language-level gradient API for circuits |
| *Not in brief* | **New P0:** process-crashing panics on malformed input (9 probes); **new P0:** silent defaults; **new P0:** in-place aliasing semantics; **new P0:** test-suite integrity (vacuous Bolero, duplicates, empty prop module) | SURFACE_AUDIT §7, VERIFICATION_PLAN §2.4 |

---

## P0 — blocks credible correctness or performance claims

### P0-1 · Panics on malformed `.cjcl` input crash the process
- **Evidence:** probes p01–p07, p12, and p17 end in a Rust panic (exit 101), and p22 in an allocation abort (exit 127) (SURFACE_AUDIT §7).
- **Root causes:**
  - `assert!` in library code: `density.rs:149,585,…`, `mps.rs:433,459,490`, `stabilizer.rs:120,395`, `trotter.rs:126-127`, `fermion.rs:148`.
  - Unchecked `args[k]` indexing in `dispatch.rs`, e.g. line 224.
  - `extract_int(..) as usize` on negative values.
  - No cap on pure-backend density size (`pure.rs:818`).
- **Recommendation:**
  1. Add `validate_*` helpers in `dispatch.rs` that check arity, index range, adjacency, probability domain, size caps, and Hamiltonian/state size. Every arm should call them **before** entering library code.
  2. Replace `args[k]` with `args.get(k)`-based extractors. Reject negative ints before any `as usize` cast.
  3. Convert the library `assert!`s on user-controllable values into `Result`-returning APIs. Keep `debug_assert!` for internal invariants.
  4. Cap pure `density_new` at the same 14 as Rust.
- **Gate:** VERIFICATION_PLAN §8 table all `Err`; Bolero `bolero_dispatch_no_panic` green.

### P0-2 · Silent defaults and silent wrong answers
- **Evidence:** p08 (duplicate operands), p15 (surface-code decode), p16, p18, p19, p20, and p24.
- **Recommendation:**
  - Duplicate qubit operands → `Err` at gate-add time.
  - Enumerated string arguments (`mps_energy`, `q_scale_noise`) and the Trotter `order` → exact match or `Err`.
  - `extract_int` should reject non-integral floats (as `extract_qubit_index` already does).
  - Numeric array coercions in `qml_train`, `qml_predict`, `qec_decode`, and `q_zne_*` → `Err` on non-numeric elements.
  - `qec_decode` must return `Err` for surface codes until a 2D decoder exists (see P1-6).

### P0-3 · Value semantics: gate builtins mutate their argument in place — **DECIDED: ADR-0044**
- **Evidence:** p10. `let b = q_x(a, 0); q_probs(a)` returns `[0, 1]` (`dispatch.rs:1554-1558`: `borrow_mut` on the shared `Rc`, then clone the same `Rc`).
- **Why P0:** a program that keeps a reference to an intermediate circuit (for example to build variants of an ansatz) gets silently wrong results. Every existing test uses shadowing, so none catches it.
- **Recommendation (architecture decision; needs an ADR):** pick one rule, then document and test it (VERIFICATION_PLAN §4.4).
  - **(a) Copy-on-write:** `Rc::make_mut`-style clone when `Rc::strong_count > 1`. This matches `array_push` COW semantics in CJC. It needs `Clone` on the stored `dyn Any` (e.g. a `QuantumObject` trait with `clone_box`) and adds no `Value` variant.
  - **(b) Explicitly mutable handle types**, with the aliasing documented.
  - (a) is recommended for consistency with the rest of the language.
- **Outcome:** a split decision. Circuits became values (a gate copies the gate list), and MPS, stabilizer, and density became documented handles, with `q_copy` to fork them. Recommendation (a) was measured before adoption:
  - inside a builtin, the refcount is ≥ 2 even for unaliased `x = f(x)`, so COW copies on every call;
  - `array_push` turned out to be O(n²) for the same reason;
  - state copies cost 60–1,250× a gate for stabilizer and MPS.

### P0-4 · Composability of states, observables, and Hamiltonians — **IMPLEMENTED: ADR-0045**
- **Evidence:** SURFACE_AUDIT §5.
  - Rust `Statevector` values from `q_run`/`q_trotter_evolve` are accepted by **no** builtin.
  - Every observable re-executes the circuit, so a program calling `q_probs` and then `q_sample` simulates twice.
  - `q_fermion_new` produces a Hamiltonian that can't be populated.
- **Recommendation (minimal primitives, per CLAUDE.md directive #6):**
  1. Make `q_probs`, `q_amplitudes`, `q_sample`, `q_measure`, `q_fermion_expectation`, and `q_trotter_evolve` accept **either** a Circuit (execute) **or** a Statevector (use as-is). This is one helper, `with_state`, replacing `with_circuit` at 6 call sites.
  2. Add `q_pauli_expectation(state, "XZIY", coeff)` and `q_fermion_add_term(H, pauli_string, coeff)`, so observables and Hamiltonians are user-constructible.
  3. Add `density_from_state(sv)` (wraps the existing `DensityMatrix::from_statevector`) to bridge the dense and density families.
- **Gate:** VERIFICATION_PLAN §4.4 composability tests flip from expected-`Err` to expected-value.
- **Outcome:** all three items shipped. The Pauli builtin is named `q_expect_pauli(state, pauli)`, without a coefficient; scale the result, or use `q_fermion_add_term` for weighted sums. `q_copy` was added under ADR-0044. Probes p13 and p26 now return values. There are 16 tests in `test_quantum_state_interop.rs`.

### P0-5 · No benchmark harness or published comparison
- **Evidence:** BENCHMARK_PLAN §1. The perf tables in `docs/QUANTUM_SIMULATION.md` have no provenance.
- **Recommendation:** implement BENCHMARK_PLAN §8 layout steps 1–2 (generator, schema, CJC runner, Qiskit Aer + Stim drivers, replay checks).
- Until then, remove the unsourced before/after timing tables from the docs, or label them "unverified, no harness".
- **Outcome (2026-09-25):** implemented as `bench/quantum_compare`: a seeded circuit generator (W1 GHZ, W2 random, W3 Clifford, W4 MPS GHZ and brickwork, W6 repeated observation), the §5 schema, fresh-process replay, rust/eval/mir byte-identity, and Qiskit Aer 0.17.2 and Stim 1.16.0 drivers (pinned in `externals/requirements.lock`).
  - Baseline and after runs are in `bench_results/quantum_compare/`.
  - The first run found that the MPS SVD's Jacobi rotation had the wrong sign. Depth-8 brickwork ⟨Z_i⟩ was off by 0.53 against Aer; after the fix it agrees within 8.5e-11. The Clifford path agrees with Stim's `peek_z` on every qubit, and CJC's measurement record passes Stim's `postselect_z` replay.
  - The old timing tables stay labelled unverified. The measured numbers are in `docs/QUANTUM_SIMULATION.md` → "Measured Performance (harness)".

### P0-6 · Documentation drift
- **Evidence:** 25 items (SURFACE_AUDIT §10). The worst:
  - D4–D7: example code calls APIs that don't exist.
  - D17: the examples README says most families are not exposed, but they are.
  - D11: type-system claims don't hold.
  - D14: the pure backend is described as "CJC all the way down" and "modifiable".
  - D20: the H₂ energy label.
- **Recommendation:**
  - Regenerate the builtin reference from `dispatch.rs` (a table test fails if a dispatch arm lacks a doc row).
  - Fix the Rust examples to compile, as doc-tests where possible.
  - Correct the claims listed in §10.
  - Do this *before* any external-facing comparison.

### P0-7 · Test-suite integrity
- **Evidence:** VERIFICATION_PLAN §2.4.
  - The two Bolero quantum targets can't fail (T-1).
  - 46 tests are duplicated (T-2).
  - The `quantum_prop` module is empty (T-3).
  - Demos 04–06 are not in CI (T-7).
  - 27 of 78 builtins have no dual-executor test.
  - `proptest!` is used zero times.
- **Recommendation:** fix T-1…T-7 first (cheap), then add the per-builtin parity table test (VERIFICATION_PLAN §4.1).

### P0-8 · Chemistry reference values unvalidated — *partially resolved, see VERIFY_FOLLOWUPS §1*
- **Update (2026-09-24):**
  - A from-scratch STO-3G FCI confirms that the H₂ matrix is electronic-only, with the ground state within 1.75×10⁻⁴ Ha.
  - Its excited levels are off by 19–24 mHa, so only ground-state claims are safe.
  - PySCF + OpenFermion (Linux container) confirm the H₂ numbers exactly.
  - **LiH is refuted:** the extracted matrix's N=2 ground is −10.009 Ha against a true −7.863 Ha (active space) or −7.883 Ha (FCI), and its constant has the wrong sign. Remove `q_fermion_lih`, or regenerate it from PySCF with a committed script.
- **Evidence:** SURFACE_AUDIT §6.1–6.2.
  - The H₂ docstring's −1.1373 Ha doesn't match the matrix minimum (−1.8512 Ha). The likely cause is the omitted nuclear repulsion; that is a hypothesis.
  - The LiH Hamiltonian is self-described as simplified.
- **Why P0:** any VQE/Trotter accuracy claim in a benchmark depends on a correct reference.
- **Recommendation:**
  - Validate both Hamiltonians against OpenFermion + PySCF in a one-off script. Commit its output as a fixture, with the script and versions recorded.
  - Fix the docstrings, or add a `q_fermion_h2_total` that includes the nuclear repulsion.
  - Label LiH "toy" until validated.

### P0-9 · Existing QAOA integration test fails on this branch — **FIXED**
- **Resolution:** bisection confirmed that `66b65bd`'s SVD fix is correct. Without it, MPS QAOA energies disagree with a dense-statevector oracle (1.52 vs 1.71).
- **The real bug:** the optimizer's ±π/2 shift on *shared* parameters is identically zero, so the optimizer never moved.
- **Fix:** exact per-gate shifts (`qaoa_gradient`). The seeded result went from 1.439 to 2.609 (optimum 3).

- **Evidence:** `tests/beta_tests/quantum/test_qaoa.rs:165`, `qaoa_4_cycle_finds_good_cut`, fails deterministically. It was run in this audit (SURFACE_AUDIT, "Test execution record").
- **Hypothesis:** the SVD routing change in `66b65bd` altered MPS truncation behaviour. That commit was verified with `--lib` tests only.
- **Recommendation:**
  1. Bisect by running the test at `66b65bd^` and at `66b65bd`.
  2. If the SVD change is responsible, check the result against a dense-statevector QAOA oracle before deciding whether the code or the test threshold is wrong.
  3. Require `cargo test --test test_beta_tests` in the pre-merge gate for `cjc-quantum` changes.

---

## P1 — major capability gaps vs mature classical simulators

### P1-1 · Circuit interchange (OpenQASM 2.0 first, 3.0 subset later)
- **Why:** it enables cross-validation against every external simulator and removes hand transcription from benchmarks. Every external except QuEST and Stim has some QASM path (EXTERNAL_COMPARISON §1).
- **Recommendation:**
  - Add `q_from_qasm(text) -> Circuit` and `q_to_qasm(circuit) -> String` for the existing 13-gate set.
  - Unknown gates → `Err` naming the gate; no silent drop.
  - The parser should be a small hand-written recursive-descent in `cjc-quantum` (no dependency).
  - Round-trip property test: `from_qasm(to_qasm(c)) ≡ c`.
  - QIR: defer (P3). No external in the comparison set documents LLVM QIR support.
- **Outcome (2026-09-25):** `q_to_qasm` / `q_from_qasm` (`cjc-quantum/src/qasm.rs`), both backends and both executors.
  - Accepted: `qelib1` gates `h x y z s t rx ry rz cx cz swap ccx id`, `barrier`, `creg`, several `qreg`s, whole-register broadcast, and trailing `measure`.
  - Parameters take `pi`, arithmetic, and `sin cos tan exp ln sqrt` (via `dmath`).
  - Anything else (`u1`–`u3`, `sdg`, gate definitions, `reset`, `if`, OpenQASM 3) is an error naming the line.
  - Angles print in shortest round-trip form, so export → import is bit-identical (200 seeded circuits).
  - Four Qiskit-written fixtures reproduce Aer's amplitudes to < 1e-12.

### P1-2 · CPU parallelism and SIMD, deterministically
- **Evidence:**
  - No threads anywhere in the crate.
  - `simd_kernel.rs` is unused.
  - Sampling is O(shots·2ⁿ).
  - Circuits are re-executed per observable.
- **Recommendation, in order of expected payoff (all hypotheses until BENCHMARK_PLAN measures them):**
  1. Cache the executed statevector per circuit version (removes repeated simulation).
  2. Replace per-shot linear CDF walks with one prefix-sum array plus a binary search per shot: O(2ⁿ + shots·n). Store the same Kahan-finalised prefix values the linear walk compares against and keep one `rand_f64` draw per shot. Sample streams should then match today's bit for bit, **provided the stored prefix sequence is monotone**. That is a hypothesis: a Kahan-compensated partial sum could dip by an ulp. Gate the change on a replay test against the current implementation.
  3. Wire `apply_single_qubit_simd` behind a runtime check, gated by a bit-identity test against the scalar path.
  4. Chunked parallel gate application with a **fixed chunk partition independent of thread count**, and fixed-tree reductions for norms and probabilities. This follows the CLAUDE.md "identical regardless of thread count" rule.
- **Architecture note:** check how `cjc-runtime/parallel` handles threading before adding a threading dependency.
- **Outcome (2026-09-25), measured by `quantum_compare kernels` as an interleaved old-vs-new A/B:**
  1. Cache: an immutable circuit keeps its executed statevector (≤ 24 qubits). Observing one 20-qubit circuit three times is 2.3× faster in both executors, with byte-identical output.
  2. Batch sampler: the stored prefix is a running maximum of the Kahan-finalised sums, which makes it monotone and removes the ulp-dip risk. Shots are bit-identical to the linear walk, and 1,000 shots at 22 qubits are ~290–400× faster.
  3. SIMD: **not wired.** The AVX2 kernel was 3–5× slower than the scalar loop at n = 16–24, and the cache-blocked one tied it.
  4. Threads: `kernels.rs` uses `std::thread::scope` (no new dependency), with the thread count from `cjc_runtime::runtime_policy`. Each amplitude pair is owned by one thread and nothing is reduced across threads, so no fixed partition is needed for identical results. Kernels plus threads are 1.2–1.5× faster over whole circuits at 20–22 qubits, because gate application is memory-bound.

### P1-3 · Gate, operator, and observable algebra
- **Missing:**
  - Custom 1q/2q unitaries (`q_unitary(c, qubits[], matrix)`).
  - Controlled versions of arbitrary gates.
  - U3/phase/√X/iSWAP/CRz.
  - Adjoint/inverse of a circuit.
  - Pauli-string observables and sums on dense states (see P0-4.2).
  - Parameterised circuits with rebinding (today a new circuit must be built per parameter value).
- **Recommendation:** extend the `Gate` enum with `Unitary1(q, [[C;2];2])`, `Unitary2(a, b, [[C;4];4])`, and `Controlled(Box<Gate>, ctrl)`, plus a unitarity check on construction (`Err` if ‖U†U−I‖ > 1e-10). Then add the MPS equivalents: the general 4×4 swap-network path exists in `mps.rs:802` but only SWAP is wired.

### P1-4 · Noise modelling breadth
- **Missing from `.cjcl`:**
  - General Kraus channels (the crate has `apply_kraus`).
  - Two-qubit channels.
  - Readout error.
  - Thermal relaxation.
  - A noise-model object that attaches channels to gate types.
  - Trajectory (Monte Carlo wavefunction) simulation to go beyond 14 qubits with noise.
- **Recommendation:**
  - Expose `density_kraus(d, qubits[], kraus_list)` with a completeness check.
  - Add `density_readout_probs(d, p01, p10)`.
  - Then add a trajectory mode on the dense statevector, where channel selection draws from the seeded SplitMix64, so it stays deterministic per seed.
  - Document the depolarizing parameterisation (SURFACE_AUDIT §6.3) and add a Qiskit-convention helper.

### P1-5 · Mid-circuit measurement and classical control in `.cjcl`
- **Evidence:** `HybridCircuit` (`adjoint.rs:218-290`) implements measure-to-creg and `if_then`, but it is crate-only.
- **Recommendation:**
  - Expose `q_measure_mid(c, qubit, creg)` and `q_if(c, creg, value, gate…)`, plus `q_reset(c, q)`, as circuit ops executed with an explicit seed at `q_run(c, seed)`.
  - This is the minimum to express teleportation, repeat-until-success, and QEC rounds on dense states.

### P1-6 · Stabilizer/QEC tooling toward Stim-class workflows
- **Missing:**
  - Bulk multi-shot sampling (Pauli-frame).
  - Circuit-level noise (errors between gates, measurement errors).
  - Detectors and observables annotations.
  - A detector error model.
  - Any decoder beyond repetition (MWPM or union-find).
  - Surface-code decoding (currently wrong answers; P0-2).
  - Multi-round syndrome extraction.
- **Recommendation:**
  - Near term: implement multi-round repetition and surface-code memory experiments with measurement noise on the existing tableau, plus a deterministic union-find decoder (simpler than MWPM, no dependencies).
  - Longer term: Pauli-frame bulk sampling.
  - Stim is the reference oracle; import Stim circuit text (P2) for cross-validation.

### P1-7 · Differentiable quantum workflows at language level
- **Evidence:**
  - Adjoint differentiation is crate-only and diagonal-observable only (`adjoint.rs:51-75`).
  - VQE/QAOA return only final energy.
  - QML's module docs misstate the gradient method.
  - CJC's own AD (`cjc-ad`, `grad_graph_*`) doesn't connect to quantum circuits.
- **Recommendation:**
  1. Expose `q_expval_grad(circuit, observable) -> [E, grads[]]` using the existing adjoint path, first for diagonal observables and later for Pauli sums once P0-4.2 lands.
  2. Return parameters and `energy_history` from `vqe_*`/`qaoa_maxcut` (the data already exists in `VqeResult`/`QaoaResult`) so optimisation is observable and resumable.
  3. Return trained parameters from `qml_train`.
  4. Longer term: a `grad_graph` op wrapping expectation + adjoint, so quantum circuits join CJC's GradGraph. This needs its own ADR because of the AD ↔ quantum crate dependency direction.

### P1-8 · Cross-platform deterministic trigonometry (promoted from P3-7) — **IMPLEMENTED for `cjc-quantum`: ADR-0046**
- **Evidence:** VERIFY_FOLLOWUPS §2.
  - Windows (MSVC) and Linux (glibc 2.36) return different `sin`/`cos` bits for 6.0% of gate angles, including π/2. Neither is fully correctly rounded (3.1% vs 0.13% of evaluations are off).
  - `Ry(π/2)` probabilities therefore differ in the last bit between platforms.
  - The documented cross-platform bit-identity therefore does not hold.
- **Recommendation:**
  1. **Now:** qualify the claim in `lib.rs`, `measure.rs`, and the docs as "same platform and toolchain".
  2. Implement a pure-Rust, fixed-algorithm `sin`/`cos` (ideally correctly rounded) in `cjc-repro` or `cjc-runtime`, and use it in `gates.rs`, `mps`, `pure.rs`, and `trotter.rs`.
  3. Gate the change on `verification/libm_bits.rs` producing an identical hash on Windows, Linux, and macOS CI.
  4. Audit other libm uses (`exp`, `ln` in entropy/Trotter) the same way.
- **Outcome:**
  - `cjc_repro::dmath` provides `sin`, `cos`, `exp`, `ln`, `pow`, and `powi`, following fdlibm, with a `u128` Payne–Hanek reduction. Accuracy is < 1 ulp against mpmath, over 348k evaluations.
  - All 59 non-test call sites in `cjc-quantum` switched to it.
  - `verification/dmath_check/` produces byte-identical output on Windows and Linux (SHA-256 `e6a7c8ed…`), and a golden-hash unit test runs in the 3-OS CI matrix.
  - `libm_bits.rs` itself still measures the *platform* libm, as intended.
  - **Not done:** `cjc-runtime`'s `.cjcl` math builtins, which would change existing golden hashes.

---

## P2 — breadth and ergonomics

| ID | Gap | Recommendation |
|---|---|---|
| P2-1 | Commodity GPU backend | Optional feature-gated crate (keeps default zero-dependency). Requires the P1-2 determinism design first. The cuStateVec sampler takes caller-supplied randoms, which fits SplitMix64 threading. **Architecture decision.** |
| P2-2 | Distributed (MPI) statevector | After GPU; same determinism constraints (fixed partition, fixed reduction tree) |
| P2-3 | MPS breadth | Expose Rx, Rz, Z, S, T, CZ, general 2q gates, sampling, `mps_to_amplitudes` (n ≤ 20), and truncation error / discarded weight, which is needed to trust χ-limited results |
| P2-4 | QAOA graphs | `qaoa_graph(n, edges)` wrapping `Graph::new`; `qaoa_graph_complete` |
| P2-5 | Pure-backend completeness | Add `q_sample`, `q_amplitudes`, `q_toffoli`, `mps_energy`, canonicalisation, and Trotter arms, **or** document the pure backend as a teaching subset. Replace "expected a quantum circuit" with backend-aware messages (p14) |
| P2-6 | Type signatures for 16 builtins; `cjcl check` false positives on valid quantum demos; reversed expected/found message | Register signatures; fix the checker so shipped demos type-check clean; add a CI step `cjcl check examples/quantum_simulations/*.cjcl` |
| P2-7 | State serialization | Snapshot/restore of circuits (via QASM, P1-1) and states (`cjc-snap` currently lists QuantumState as non-serializable) for reproducible checkpoints |
| P2-8 | Stim circuit text import (Clifford subset) | Enables Stim cross-validation for W3 |
| P2-9 | Sampling statistics helpers | `q_counts(samples)` → sorted `(bitstring, count)` pairs (BTreeMap order); expectation-from-samples with standard error |

## P3 — research / long horizon

| ID | Item |
|---|---|
| P3-1 | QIR (LLVM) import/export. No external in the comparison set documents it. |
| P3-2 | Extended-stabilizer / Clifford+T (Aer `extended_stabilizer`-style) |
| P3-3 | General tensor-network contraction (beyond 1D MPS), e.g. PEPS/TTN, contraction-path search |
| P3-4 | Pauli propagation (cf. cuPauliProp) for large-scale expectation estimation |
| P3-5 | Lindblad master-equation / analog time evolution (cf. QuEST v4.2, cuDensityMat) |
| P3-6 | Bastion-level libraries (optimisers, chemistry integrals) built on the P0/P1 primitives, keeping language primitives minimal (CLAUDE.md #6) |
| P3-7 | *(promoted to P1-8 after VERIFY_FOLLOWUPS §2 showed libm rounding differences)* |

## Suggested sequencing

1. **Correctness hardening:** P0-1, P0-2, P0-7, the P0-6 doc corrections, and the P0-8 fixture. These are small, local changes that rebuild trust.
2. **Semantics:** P0-3 (ADR + implementation) and P0-4 (state-accepting observables, Pauli observables).
3. **Harness:** P0-5 with P1-1 (QASM import/export makes the harness honest).
4. **Performance:** P1-2, measured before and after with the harness. Hashes must stay bit-identical.
5. **Capability:** P1-3 … P1-7, then P2.
