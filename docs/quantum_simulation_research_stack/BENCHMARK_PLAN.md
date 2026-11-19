# CJC-Lang Quantum Simulation — Benchmark Plan

**Status (2026-09-25):** §9 steps 1–2 are implemented in `bench/quantum_compare`
(workloads W1–W4 plus W6, the §5 schema, fresh-process replay, cross-path checks,
Qiskit Aer and Stim drivers). Measured results are in
`bench_results/quantum_compare/2026-09-25_*` and summarised in
`docs/QUANTUM_SIMULATION.md` → "Measured Performance (harness)". W5 and the
other externals are not run. The text below is the original plan; its
expectations are still labelled **hypothesis** and were not edited to match
the measurements.

The first baseline run found a correctness bug: the MPS SVD's Jacobi rotation had
the wrong sign, and depth-8 brickwork ⟨Z_i⟩ disagreed with Aer by 0.53. That
agreement check was the harness's first real use.
**Date:** 2026-09-24 · **Branch audited:** `claude/cjc-quantum-audit-5a7f9c` (HEAD `671dfeb`)
**Companion docs:** [SURFACE_AUDIT.md](SURFACE_AUDIT.md) · [EXTERNAL_COMPARISON.md](EXTERNAL_COMPARISON.md) ·
[MISSING_FEATURES.md](MISSING_FEATURES.md) · [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md)

> Gate separation: **tests establish correctness claims; benchmarks establish
> performance claims.** A workload may be benchmarked only after its correctness
> gate in [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) is green. A benchmark
> result may be cited publicly only after its deterministic-replay check passes
> and its result record validates against the schema in §5.

---

## 1. Why a harness is needed before any claim

What exists today (verified in-tree):

| Artifact | What it is | Why it is not a benchmark |
|---|---|---|
| `tests/bench_50q.rs` (14 `#[test]`) | Timed `#[test]` functions that `eprintln!` elapsed time | Single run, no warm-up, no statistics, no machine metadata, run inside the test harness (parallel test threads contend), results not persisted. `bench_20q_dmrg` actually runs **8** qubits (`dmrg_heisenberg(8, 8, 3, 0.01)`). |
| `tests/bench_dual_mode.rs` (17 `#[test]`) | Rust vs `"pure"` backend timings via `cjc-eval` only | Same issues; eval only (no MIR-exec column). |
| `docs/QUANTUM_SIMULATION.md` §"Performance Optimizations" | Before/after tables (e.g. "50q MPS create+5 gates ~2.0ms → ~0.7ms") | No harness, commit, machine, or run count recorded; not reproducible from repo. Treat as **unverified**. |

No CJC-vs-external comparison exists anywhere in the repo.

## 2. Constraints discovered during the audit (they shape the workloads)

Each of these is cited to source; they must be reflected in the harness or the
comparison will be unfair in one direction or the other.

1. **Dense statevector is capped at 26 qubits** — `dispatch.rs:54` (`qubits()`
   rejects n > 26) and `statevector.rs:21` (`assert!(n_qubits <= 26)`).
   27–28-qubit rows can only be filled for external simulators; CJC cells are
   `N/A (cap)`, never "slower".
2. **Every observable call re-executes the whole circuit.** `q_probs`,
   `q_amplitudes`, `q_sample`, `q_measure`, `q_fermion_expectation`, and
   `q_trotter_evolve` each call `circ.execute()` (`dispatch.rs:106,131,156,166,731,762`).
   A program calling `q_probs` then `q_sample` pays for two full simulations.
   The harness must time *build*, *execute*, and *observe* phases separately and
   report the "naive .cjcl" cost as its own column.
3. **Shot sampling is O(shots × 2ⁿ).** `measure::sample_basis_state` walks the
   cumulative distribution linearly per shot, finalizing a Kahan accumulator per
   step (`measure.rs:86-100`). Hypothesis: sampling will dominate at high shot
   counts; external simulators typically use a single pass / sorted-uniform
   approach. Benchmark shots ∈ {1, 1e3, 1e5} separately from state evolution.
4. **Single-threaded, scalar gate kernels.** No `rayon`/`std::thread` in the
   crate. `simd_kernel.rs` (AVX2 + cache blocking) is **not called** by the
   production gate path (`gates.rs::apply_single_qubit` is scalar; `grep` finds
   no caller of `apply_single_qubit_simd`/`_cached` outside `simd_kernel.rs`).
   Every external comparison must therefore be reported **twice**: external at
   1 thread (algorithmic comparison) and external at default threads (user-
   experienced comparison).
5. **Interpreter overhead is real and separate.** Each `.cjcl` gate call goes
   through eval/MIR dispatch → `dispatch_quantum` string match → `RefCell`
   borrow. Benchmark both the `.cjcl` path and the direct Rust API path
   (`cjc_quantum::Circuit`) so dispatch overhead is visible, not hidden.
6. **MPS surface from `.cjcl` is narrow:** only `mps_h`, `mps_x`, `mps_ry`,
   adjacent `mps_cnot`, `mps_swap` (SWAP-network), Z expectation and energy.
   Low-entanglement workloads must be built from {H, X, Ry, adjacent CNOT}.
7. **Density matrix:** Rust backend capped at 14 qubits (`density.rs:20`,
   `MAX_QUBITS = 14`, enforced by `assert!` → panic); `"pure"` backend has **no
   cap** (`pure.rs:818`). Benchmarks must not run pure density above 12 qubits
   without a memory guard.
8. **Stabilizer measurement is O(n²) per measured qubit** (CHP algorithm,
   `stabilizer.rs:394`), and there is no bulk sampler. Stim-class sampling
   workloads (millions of shots) are not expressible; benchmark what exists and
   label the gap.
9. **VQE/QAOA are closed-loop builtins** (`vqe_heisenberg`, `qaoa_maxcut`) that
   return only final energy — no per-iteration hooks. Time-to-solution must be
   measured end-to-end with fixed iteration counts, not convergence thresholds.
10. **`vqe_heisenberg` is ZZ-only (Ising)**, `vqe_full_heisenberg` is XX+YY+ZZ
    (`vqe.rs:9-11`). Label workloads by the Hamiltonian actually simulated.

## 3. Workloads

All circuits are generated by one canonical, seeded generator (SplitMix64, same
constants as `cjc_quantum::splitmix64`) that emits **three** equivalent forms:
`.cjcl` source, OpenQASM 2.0 text (for Qiskit Aer / Cirq-qsim / Qulacs /
PennyLane / QuEST drivers), and Stim circuit text (Clifford workloads only).
Because CJC has no QASM import (see MISSING_FEATURES P1-1), equivalence of
forms is itself checked: the generator's `.cjcl` and QASM outputs must yield
statevectors agreeing to ≤1e-10 (L∞) for n ≤ 20 on a trusted reference before
any timing is recorded.

Gate set restricted to what CJC exposes: `H X Y Z S T Rx Ry Rz CX CZ SWAP CCX`.

### W1 — Bell/GHZ correctness + scaling (dense statevector)
- n ∈ {2, 4, 8, 12, 16, 20, 22, 24, 26}; circuit `H(0); CX(0,k)` for k=1..n-1.
- Observe: `q_probs` (full vector) and `q_sample` with shots ∈ {1, 1e3, 1e5}.
- Correctness oracle: analytic — P(0…0)=P(1…1)=½ exactly representable? No:
  `(1/√2)²` is not exactly ½ in binary64. Record max |p − ½| and require ≤ 4 ulp.
- Externals: all dense simulators; n=27,28 externals only.

### W2 — Dense random circuits
- n ∈ {16, 18, 20, 22, 24, 26} (+27, 28 external-only), depth ∈ {10, 20, 40}
  layers; each layer = random 1q gate on every qubit (from H/S/T/Rx/Ry/Rz with
  seeded angles) + CX on a seeded random perfect matching.
- 5 seeds per (n, depth). Observe: full probability vector hash + 1e3 shots.
- Oracle: cross-simulator agreement (Qiskit Aer statevector and qsim) ≤1e-10 L∞
  on amplitudes for n ≤ 24; fidelity |⟨ψ_cjc|ψ_ref⟩|² ≥ 1 − 1e-10.
- **Hypothesis to test:** CJC single-thread throughput is within a small
  constant of scalar reference implementations but well below SIMD/multithreaded
  externals. Unknown until measured.

### W3 — Random Clifford / stabilizer
- n ∈ {100, 500, 1000, 2000, 5000, 10000}; depth = n (layers of random
  H/S/X/Y/Z + CX matching); then measure all qubits once.
- CJC: `stabilizer_*` builtins and Rust `StabilizerState`. Stim: `stim.TableauSimulator`
  (single-shot, apples-to-apples) **and** `compile_sampler` (bulk) reported as
  separate columns — the bulk column has no CJC counterpart.
- Oracle: for n ≤ 12, `StabilizerState::to_statevector()` vs dense `Circuit`;
  for large n, deterministic-measurement outcomes compared to Stim's.
- Memory note: tableau is 2 arrays (x, z) × 2n rows × ⌈n/64⌉ u64 words → 2 × 20000 × 157 × 8 B ≈ 50 MB at n=10000 (arithmetic, not measured).

### W4 — MPS GHZ + low-entanglement
- GHZ chain via adjacent CNOTs: n ∈ {50, 100, 250, 500, 1000}, χ_max ∈ {2, 16, 64}.
- Brickwork Ry + adjacent-CX, depth ∈ {2, 4, 8}, n ∈ {50, 100, 500}, χ_max ∈ {8, 16, 32, 64}.
- Observe: ⟨Z_i⟩ for all i, `mps_memory`, and Heisenberg energy (`mps_energy`).
- Oracle: n ≤ 20 → compare with dense statevector (≤1e-10 when χ_max ≥ exact
  bond); n > 20 → GHZ analytic ⟨Z_i⟩ = 0, bond dim = 2.
- Externals: Qiskit Aer `matrix_product_state`, PennyLane `lightning.tensor`,
  TensorCircuit MPS, cuQuantum/cuTensorNet (GPU reference column only).
- Report truncation error / discarded weight where the external exposes it
  (CJC does not expose it from `.cjcl` — gap).

### W5 — Density matrix + noise
- n ∈ {4, 6, 8, 10, 12, 14}; circuit = W2 layer structure at depth 10, with a
  single-qubit channel after every gate: depolarizing p=1e-3, dephasing p=1e-3,
  amplitude damping γ=1e-3 (three separate runs).
- Observe: trace, purity, `density_probs`, von Neumann entropy.
- Oracle: trace = 1 within 1e-12; purity ∈ (0, 1]; Qiskit Aer `density_matrix`
  with *matching channel parameterisation* — note CJC's depolarizing Kraus set
  is `√(1−p)·I, √(p/3)·{X,Y,Z}` (`density.rs:584`), which equals Qiskit's
  `depolarizing_error(λ)` only under λ = 4p/3. The driver must convert.
- Pure backend: n ≤ 10 only (no cap in code — guard in harness).

### W6 — VQE (Heisenberg)
- `vqe_heisenberg` (ZZ-only) and `vqe_full_heisenberg` (XX+YY+ZZ) at
  n ∈ {4, 12, 50}, χ ∈ {8, 16}, lr = 0.05, iterations ∈ {10, 50}, seed 42.
- Oracle: exact diagonalisation for n ∈ {4, 12} (numpy/scipy, pinned version);
  `dmrg_heisenberg` + an external DMRG (e.g. ITensor/TeNPy — reference only)
  for n = 50. Report relative energy error, not just time.
- Externals: PennyLane Lightning adjoint VQE, Qiskit + Aer estimator. Compare
  time-per-gradient-evaluation (not time-to-convergence) because optimisers differ.

### W7 — QAOA MaxCut
- `qaoa_graph_cycle(n)` (only graph family exposed) with n ∈ {8, 16, 32, 64},
  p ∈ {1, 2, 3, 4}, χ = 16, iterations = 20, seed 42.
- Oracle: cycle MaxCut optimum is known analytically (n for even n, n−1 for odd).
  Report approximation ratio = cut_value / optimum.
- Externals: Qiskit Aer (n ≤ 26 statevector), PennyLane Lightning, TensorCircuit.

### W8 — Fermion / Trotter / ZNE
- H₂ (`q_fermion_h2`) and LiH (`q_fermion_lih`): expectation on basis states
  and on the demo-05 Ry–CX ansatz sweep (51 points).
- Trotter: H₂, t ∈ {0.5, 1, 2}, n_steps ∈ {1, 4, 16, 64}, order ∈ {1, 2};
  oracle = exact `expm(-iHt)` (numpy/scipy) fidelity; report error vs the
  `q_trotter_error` bound (bound must dominate the measured error).
- ZNE: `q_zne_mitigate` on synthetic polynomial data (exact oracle) and on
  density-simulated noisy expectations from W5 at scales {1, 2, 3}.
- Chemistry oracle caveat: the H₂ matrix is electronic-only (min −1.851199 Ha vs
  FCI −1.851024 Ha; add 1/R = 0.713754 Ha for the total). Its excited levels are off by
  19–24 mHa (VERIFY_FOLLOWUPS §1), so Trotter fidelity must be measured against
  `expm` of the **shipped** matrix, not against real H₂. LiH coefficients are self-described as "simplified …
  reduced to the most significant terms" (`fermion.rs:436-437`), and PySCF shows
  it is not LiH (N=2 ground −10.009 Ha vs −7.863 Ha; VERIFY_FOLLOWUPS §1).
  Use it as a **performance-only** workload, or drop it until it is regenerated.

## 4. Metrics (per run)

| Metric | Definition | How measured |
|---|---|---|
| `wall_time_s` | Median and IQR of ≥ 5 timed repetitions after ≥ 1 warm-up | `std::time::Instant` (Rust); `time.perf_counter` (Python drivers) |
| `phase_times_s` | build / execute / observe split (§2.2) | Instrumented runner; for `.cjcl` via separate programs per phase |
| `peak_rss_bytes` | Peak resident set of the benchmark process | Windows: `GetProcessMemoryInfo.PeakWorkingSetSize`; Linux: `getrusage` `ru_maxrss` |
| `output_hash` | SHA-256 over canonical bytes of the primary output (§6) | Runner |
| `numerical_error` | Max abs / L∞ / fidelity error vs oracle | Runner; oracle named in record |
| `replay_ok` | Two fresh-process runs with the same seed produce identical `output_hash` | Runner (mandatory) |
| `cross_executor_ok` | `.cjcl` output identical under `cjc-eval` and `cjc-mir-exec` | Runner (mandatory for CJC rows) |
| `backend_config` | simulator, method, precision, threads, SIMD flags, χ_max, fusion settings | Runner |
| `machine` | CPU model, cores/threads, RAM, OS + build, GPU (if any) | Runner |
| `toolchain` | `rustc -V`, cargo profile, CJC git SHA + dirty flag, `cjcl --version`; external package versions (`pip freeze` subset) | Runner |

## 5. Result schema (one JSON object per line, `results.jsonl`)

```json
{
  "schema": "cjc-quantum-bench/v1",
  "run_id": "2026-10-01T12:00:00Z-<short-sha>-<machine-slug>",
  "workload": "W2_dense_random",
  "params": {"n_qubits": 22, "depth": 20, "circuit_seed": 3, "shots": 1000},
  "simulator": {"name": "cjc-quantum", "path": "cjcl-mir", "backend": "rust",
                "method": "statevector", "precision": "f64", "threads": 1,
                "version": "0.1.11", "git_sha": "671dfeb", "dirty": false},
  "timing": {"reps": 5, "warmup": 1, "median_s": null, "iqr_s": null,
             "phases_median_s": {"build": null, "execute": null, "observe": null}},
  "memory": {"peak_rss_bytes": null},
  "output": {"kind": "probabilities_f64_le", "sha256": null, "len": 4194304},
  "accuracy": {"oracle": "qiskit-aer-statevector@<ver>", "linf": null, "fidelity": null},
  "determinism": {"replay_ok": null, "cross_executor_ok": null, "replay_runs": 2},
  "machine": {"cpu": null, "cores": null, "threads": null, "ram_bytes": null,
              "os": null, "gpu": null},
  "toolchain": {"rustc": null, "profile": "release", "python": null,
                "externals": {"qiskit-aer": null, "qsimcirq": null}},
  "status": "ok | skipped_cap | oom | timeout | error",
  "notes": ""
}
```

`null` values above are placeholders — the schema must be filled by a run.
`status = skipped_cap` is used for CJC cells beyond a documented cap (§2.1, §2.7)
and must never be rendered as a slow result.

## 6. Deterministic replay and output hashing

CJC's differentiator is reproducibility, so replay checks are first-class metrics:

1. **Canonical bytes.** Probabilities/amplitudes: little-endian IEEE-754 binary64,
   ascending basis index, amplitudes as (re, im) pairs. Samples: little-endian
   u64 basis indices in draw order. Scalars (energies): binary64 bits. Stabilizer
   measurement records: one byte per outcome in qubit order.
2. **Replay.** Each CJC row is run in two fresh processes with the same seed;
   `output_hash` must match bit-for-bit. Failure blocks publication of that row.
3. **Cross-executor.** Each `.cjcl` workload runs under `cjc-eval` and
   `cjc-mir-exec` (`cjcl parity` or the in-process API); hashes must match.
   `--mir-opt` is reported separately and is *allowed* to differ (per
   `examples/quantum_simulations/README.md` note on constant folding) — record,
   don't gate.
4. **Cross-platform (stretch).** Windows x86-64 vs Linux x86-64 vs Linux aarch64
   hash comparison. CJC claims "bit-identical across platforms" (`lib.rs:4`,
   `measure.rs:25`). VERIFY_FOLLOWUPS §2 measured Windows vs Linux: 6.0% of gate
   angles produce different `sin`/`cos` bits, so divergence is confirmed, not just expected. Use `verification/libm_bits.rs` hashes as the first cross-platform gate. `f64::sin/cos/sqrt` come from the platform libm, so trig-based
   gates (Rx/Ry/Rz) are the likeliest place for cross-platform divergence
   (hypothesis).
5. **External determinism** is recorded, not gated: externals are run with their
   documented seed options; whether they replay bit-identically is an observed
   property recorded in `determinism.replay_ok`.

## 7. Fairness rules

- Same circuit (from generator hash), same precision (complex128 everywhere;
  if an external defaults to complex64, set it to complex128 or report both).
- Disable features the CJC side lacks **only in the 1-thread algorithmic
  column**: gate fusion, multithreading, GPU. The "default" column uses each
  external as shipped.
- Exclude interpreter start-up from execute timings but report process wall
  time too (users pay it).
- No cherry-picking: every (workload, param) cell in §3 must appear in the
  report with a status, including `oom`/`timeout`/`skipped_cap`.
- External versions pinned in `bench/quantum_compare/externals/requirements.lock`.

## 8. Suggested file layout

Following the existing `bench/<name>/` crate + `bench_results/<name>/` convention:

```
bench/quantum_compare/
  Cargo.toml                  # publish = false; deps: cjc-quantum, cjc-parser, cjc-eval, cjc-mir-exec
  main.rs                     # CLI: --workload W2 --n 22 --depth 20 --seed 3 --path {rust,eval,mir}
  gen/
    circuit_gen.rs            # canonical SplitMix64 generator → .cjcl / .qasm / .stim
  workloads/                  # checked-in generated .cjcl for small canonical cases (n ≤ 12)
  externals/
    requirements.lock         # pinned: qiskit-aer, cirq, qsimcirq, pennylane-lightning, qulacs, stim, tensorcircuit
    run_qiskit_aer.py
    run_qsim.py
    run_lightning.py
    run_qulacs.py
    run_stim.py
    run_tensorcircuit.py
    run_quest.md              # QuEST is C: build + driver instructions
  schema/result.schema.json   # JSON Schema for §5
  REPORT_TEMPLATE.md
bench_results/quantum_compare/
  <YYYY-MM-DD>_<machine-slug>/
    results.jsonl
    machine.json
    REPORT.md                 # generated tables; every number links to a results.jsonl line
```

`tests/bench_50q.rs` and `tests/bench_dual_mode.rs` should stay as smoke tests
but be renamed or re-documented as such (they are not benchmarks — §1).

## 8a. External-reference caveats

Qualitative capability claims about the external simulators live in
[EXTERNAL_COMPARISON.md](EXTERNAL_COMPARISON.md). Any
published performance figures they cite (for example qsim's benchmark pages or
the Stim paper's sampling rates) are **their** measurements on **their**
hardware. Those figures must not be put in a table next to CJC measurements.
Every number in a CJC comparison has to come from the same machine and the same
harness run.

## 9. Execution order (proposed)

1. Land generator + schema + CJC runner for W1, W3, W4 (smallest surface) and
   the replay/cross-executor checks. **Gate:** VERIFICATION_PLAN P0 tests green.
2. Add Qiskit Aer + Stim drivers; run W1–W4 on one machine; publish internally.
3. Add W5–W8 and the remaining externals.
4. Only then consider optimisation work (wire `simd_kernel`, cache executed
   statevectors, O(2ⁿ + shots·log 2ⁿ) sampling, threading) — each optimisation
   must keep W1–W8 hashes bit-identical or document an intentional change.
