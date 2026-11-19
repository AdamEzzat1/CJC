# Quantum Simulation Research Stack — Stack Role Group Prompt

**Phase:** Quantum Simulation Research Stack (competitive audit of `cjc-quantum`, then its follow-ups)
**Objective:** Establish what CJC-Lang can simulate today from `.cjcl`, how it compares with mature open-source classical simulators, what is missing, and what tests and benchmarks must exist before any correctness or performance claim is credible. Then close the gaps that block credible claims.

> **Provenance.** The audit brief named this file as "the authority for scope and
> role responsibilities", but it did not exist when the audit ran (2026-09-24).
> It was written on 2026-09-25 in the format of
> [`docs/language_hardening_phase/STACK_ROLE_GROUP.md`](../language_hardening_phase/STACK_ROLE_GROUP.md)
> and [`docs/mathematics_hardening_phase/STACK_ROLE_GROUP.md`](../mathematics_hardening_phase/STACK_ROLE_GROUP.md).
> It codifies the scope, roles, and rules the brief spelled out, which is what
> the audit actually applied. It does not add requirements after the fact. The
> "Status" section records how the work maps onto the roles.

---

## Scope

**In scope:** classical simulation of quantum circuits on traditional hardware,
CPU-first. Commodity GPU and distributed classical acceleration count as future
work.

**Out of scope:** QPU or provider execution. Missing hardware integration is not
a CJC gap, because CJC targets traditional hardware.

**Relevant even though not hardware:** circuit interchange (OpenQASM, QIR), because
the same circuit has to run on every simulator for comparison and validation.

**The differentiator to preserve:** deterministic replay and reproducible
scientific workflows. Same inputs and seed must give bit-identical results.

**Comparison set:** Qiskit Aer, Cirq + qsim, PennyLane Lightning, QuEST, Qulacs,
Stim, TensorCircuit, and NVIDIA cuQuantum (a commodity-GPU acceleration
reference, not a full framework).

---

## Roles

### 1. Surface Auditor
**Focus:** What is actually usable from `.cjcl` source, and on which backend.

**Responsibilities:**
- Inventory every quantum builtin visible from `.cjcl`, from the dispatch match arms, not the docs.
- Separate the default backend from the `"pure"` backend.
- Classify each capability as `.cjcl`-ready, crate-only, partial, tested-demo-only, or missing.
- Record eval vs MIR-exec parity coverage.
- Probe edge cases with runnable `.cjcl` programs, and record exit codes and both executors' output.
- Record documentation drift with file:line evidence.

**Key Files:**
- `crates/cjc-quantum/src/dispatch.rs`: the single entry point both executors call
- `crates/cjc-quantum/src/*.rs`: every simulator family
- `docs/QUANTUM_SIMULATION.md`, `examples/quantum_simulations/README.md`

**Deliverable:** `SURFACE_AUDIT.md`, `probes/`

### 2. External Comparison Researcher
**Focus:** How CJC compares with mature classical simulators, feature by feature.

**Responsibilities:**
- Build a feature matrix covering:
  - statevector;
  - density/noise;
  - MPS/tensor network;
  - stabilizer/QEC;
  - VQE/QAOA/QML;
  - gradients;
  - GPU/distributed classical acceleration;
  - circuit import/export;
  - determinism.
- Use official docs or primary project sources, with URLs and access dates.
- Say so when external sources disagree or are unclear.
- Keep the framing classical-simulation-only.

**Deliverable:** `EXTERNAL_COMPARISON.md`

### 3. Gap Analyst
**Focus:** What is missing, ranked by what blocks credible claims.

**Responsibilities:**
- Prioritise gaps as P0 through P3. P0 covers anything that blocks credible correctness or performance comparison.
- Validate, revise, or reject the brief's suggested P0/P1 items against the code.
- Give concrete engineering recommendations, and require an ADR for architecture decisions.
- Treat missing tests as first-class gaps, not footnotes.

**Deliverable:** `MISSING_FEATURES.md`

### 4. Benchmark Architect
**Focus:** The harness that must exist before any performance claim.

**Responsibilities:**
- Define workloads across the simulator families:
  - dense statevector and Bell/GHZ at 2–26 qubits;
  - random circuits at 16–28 qubits;
  - Clifford circuits at 100–10,000 qubits;
  - MPS at 50–1,000 qubits;
  - density/noise at 4–14 qubits;
  - VQE, QAOA p = 1–4, and fermion/Trotter/ZNE.
- Define the metrics: wall time, peak memory, output hash, numerical error against an analytic or trusted baseline, deterministic replay, backend configuration, machine metadata, and compiler/interpreter version.
- Define the result schema and file layout.
- Never report a performance result that was not measured.

**Deliverable:** `BENCHMARK_PLAN.md`

### 5. Verification Engineer
**Focus:** The tests that make correctness claims credible.

**Responsibilities:**
- Separate current coverage from proposed coverage.
- Specify tests in each category:
  - **Unit:** gates, channels, measurement, PRNG, SVD sign stabilisation, MPS, density, tableau, fermion, Trotter, ZNE, adjoint, Wirtinger.
  - **Integration/parity:** eval vs MIR-exec, default vs pure backend, and demos.
  - **Property:** normalisation, unitarity, probability sums, statistics, gate identities, MPS/density/stabilizer vs dense.
  - **Negative/error:** invalid qubits, duplicate operands, invalid probabilities, unsupported gates, malformed objects, seeds, and backend mismatch.
  - **Fuzz and Bolero:** panic freedom, determinism, normalisation, no unexpected NaN/Inf, and bounded memory.
- Where bit-identity is promised, compare bit patterns, not approximate floats.

**Deliverable:** `VERIFICATION_PLAN.md`

### 6. Determinism & Reproducibility Auditor
**Focus:** Whether "same seed ⇒ same bits" holds across runs, executors, backends, and operating systems.

**Responsibilities:**
- Test the replay claim at bit level, across executors, and across Windows, Linux, and macOS.
- Trace every source of cross-platform divergence (platform libm, FMA contraction, iteration order).
- Check scientific reference values against independent tools (for example PySCF/OpenFermion for chemistry).

**Deliverables:** `VERIFY_FOLLOWUPS.md`, `verification/`

### 7. Documentation Steward
**Focus:** No claim without evidence.

**Responsibilities:**
- Correct factual drift in `docs/QUANTUM_SIMULATION.md`, module headers, examples, and the vault.
- Replace promotional language ("full", "complete", "production-grade", "best") with "implemented", "tested", "partially exposed", "hypothesis", or "unknown", unless the narrow claim is proven.
- Keep generated references (the builtin list) tied to the code by a test.

---

## Constraints (audit rules)

1. Do not overclaim. Label unmeasured performance as a hypothesis.
2. Every claim about speed, scale, correctness, determinism, or feature parity cites source code, tests, benchmarks, docs, or an external primary source.
3. A claim that cannot be verified locally becomes a research question or benchmark hypothesis.
4. Distinguish "implemented in Rust" from "usable from `.cjcl`", and "tested directly" from "assumed from implementation".
5. Tests establish correctness claims; benchmarks establish performance claims. Neither substitutes for the other.
6. Code changes follow `CLAUDE.md`: both executors agree, determinism is preserved, and no `#[ignore]` hides a failure.

## Success Criteria

- The five deliverables exist and every factual claim in them carries evidence.
- Every audit probe either returns a value or a clean runtime error, identically in both executors.
- No performance number is published without a recorded harness run.

---

## Status (2026-09-25)

| Role | Deliverable | State |
|---|---|---|
| Surface Auditor | `SURFACE_AUDIT.md`, 26 probes | Done. All probes now exit cleanly and agree across executors |
| External Comparison Researcher | `EXTERNAL_COMPARISON.md` | Done |
| Gap Analyst | `MISSING_FEATURES.md` | Done. P0-1, P0-2, P0-3, P0-4, P0-7, P0-8, P0-9, and P1-8 are resolved |
| Benchmark Architect | `BENCHMARK_PLAN.md` | Harness built (`bench/quantum_compare`, 2026-09-25): W1–W4 and W6 measured against Qiskit Aer and Stim on one machine; W5 and other externals not run |
| Verification Engineer | `VERIFICATION_PLAN.md` | Plan done. The P0 test items are implemented: dispatch fuzz sweep, Bolero targets, `quantum_prop`, chemistry references, value semantics, state interop |
| Determinism Auditor | `VERIFY_FOLLOWUPS.md`, ADR-0046 | Cross-platform bit-identity of `cjc-quantum` verified on Windows and Linux; macOS runs in CI |
| Documentation Steward | `SURFACE_AUDIT.md` §10 | 25 of 26 drift items fixed; D22 partly fixed: measured tables added, the old tables stay labelled unverified |

Decisions recorded as ADRs in `CJC-Lang_Obsidian_Vault/13_ADRs/`:
- ADR-0044: quantum value semantics.
- ADR-0045: state interop.
- ADR-0046: deterministic elementary functions.
