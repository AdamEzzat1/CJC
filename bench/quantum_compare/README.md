# quantum_compare — CJC quantum benchmark harness

Implements [`BENCHMARK_PLAN.md`](../../docs/quantum_simulation_research_stack/BENCHMARK_PLAN.md)
§9 steps 1–2: workloads W1–W4 and W6, CJC runner, Qiskit Aer and Stim drivers,
replay and cross-path checks, and an interleaved old-vs-new A/B for the
performance changes.

## Run

```bash
cargo build --release -p quantum-compare
D=bench_results/quantum_compare/<date>_<machine>
./target/release/quantum_compare run --suite baseline --out $D --reps 5
pip install -r bench/quantum_compare/externals/requirements.lock
python bench/quantum_compare/externals/run_qiskit_aer.py $D --reps 5
python bench/quantum_compare/externals/run_stim.py $D --reps 5
python bench/quantum_compare/report.py $D --title "..."
```

`--suite smoke` runs five tiny cases in seconds, and `--filter W2` restricts a
run to matching case ids.

```bash
./target/release/quantum_compare kernels --out $D --reps 7
```

`kernels` times, in one process and interleaved rep by rep:
- single H gates, applied by each kernel variant;
- whole circuits, with the full-scan reference loop against `kernels.rs`;
- 1,000 shots, with the per-shot sampler against the batch sampler;
- the W6 program with and without the execution cache.

Every pair is checked for bit-identical output before it is timed. Use these
numbers for old-vs-new claims: separate `run` invocations on the same laptop
varied up to 2× on unchanged code.

## What a run produces

| File | Contents |
|---|---|
| `results.jsonl` | One `cjc-quantum-bench/v1` record per (case, CJC path) |
| `results_aer.jsonl`, `results_stim.jsonl` | The same schema, for the external columns |
| `manifest.jsonl` | Case list read by the drivers |
| `work/<case>/circuit.{qasm,stim,cjcl}` | The exact circuit every simulator ran |
| `work/<case>/cjc_*.bin` | CJC outputs used as accuracy oracles by the drivers |
| `REPORT.md` | Tables generated from the JSONL files |

## Paths and columns

| Column | What is timed |
|---|---|
| CJC rust | Library API: QASM import (build), `Circuit::execute` (execute), then probabilities plus 1,000 seeded shots (observe) |
| CJC eval / mir | The generated `.cjcl` program in `cjc-eval` / `cjc-mir-exec`, parse included. Dense programs return `q_probs` only, with no shots, except W6, which also calls `q_sample` and `q_measure` on the same circuit |
| Aer 1-thread | `max_parallel_threads=1`, fusion off: algorithm against algorithm. Observe = numpy probabilities plus cumsum/`searchsorted` shots (Qiskit's `sample_memory` took ~54 s at 22 qubits and is no longer used) |
| Aer default | Aer as shipped, with all threads and fusion |
| Stim tableau | `TableauSimulator.do` plus `peek_z` on every qubit, matching CJC's `StabilizerState` + `peek_z` |
| Stim bulk | `compile_sampler().sample(1000)`. CJC has no counterpart, so this is context only |

## Checks (every record)

- **Replay:** a fresh process reproduces the SHA-256 of the output bytes.
- **Paths:** the Rust path, `cjc-eval`, and `cjc-mir-exec` produce byte-identical outputs.
- **Oracles:**
  - Analytic GHZ values, for W1 and W4 GHZ.
  - Amplitudes against Aer (L∞ and fidelity), for dense n ≤ 22.
  - ⟨Z_i⟩ against Aer's MPS.
  - For Clifford cases, exact agreement with Stim on `peek_z`, plus a replay of CJC's measurement record through Stim's `postselect_z`: every outcome Stim considers forced must match.
- **Fixed inputs:** circuits are seeded and outputs are hashed, so any change to the simulator that alters results changes a hash.

The report leads dense sections with the *execute* phase (circuit in,
statevector out). That is the simulator comparison; the *total* columns add
QASM parsing and observation, which differ between paths.

## Limits of a run

- Every number comes from one machine and one run of each case. Nothing here extrapolates to other hardware.
- External columns include Python call overhead, which matters for sub-millisecond cases.
- Generated `.cjcl` is skipped (`skipped_size`) above 22 dense qubits, because `q_probs` returns 2ⁿ runtime values. It is also skipped above 120k gates.
