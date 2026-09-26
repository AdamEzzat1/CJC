# Quantum Simulation Research Stack — Audit (2026-09-24)

A competitive audit of `cjc-quantum` as a deterministic, language-integrated
**classical** quantum simulator (CPU-first; QPU execution out of scope).

| Doc | Answers |
|---|---|
| [STACK_ROLE_GROUP.md](STACK_ROLE_GROUP.md) | Scope, roles, and audit rules (written after the audit to codify the brief), plus per-role status |
| [SURFACE_AUDIT.md](SURFACE_AUDIT.md) | What `.cjcl` can simulate today; what exists only in Rust; probe results; eval/MIR parity; doc drift; tests run |
| [EXTERNAL_COMPARISON.md](EXTERNAL_COMPARISON.md) | Feature matrix vs Qiskit Aer, Cirq+qsim, PennyLane Lightning, QuEST, Qulacs, Stim, TensorCircuit(-NG), cuQuantum (sources + access date) |
| [MISSING_FEATURES.md](MISSING_FEATURES.md) | P0–P3 gaps with engineering recommendations |
| [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) | Workloads, metrics, result schema, replay hashing, file layout (**no results — nothing has been benchmarked**) |
| [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) | Current vs proposed unit / parity / property / fuzz / Bolero / negative tests |
| [VERIFY_FOLLOWUPS.md](VERIFY_FOLLOWUPS.md) | Follow-up checks: independent H₂ FCI reference, cross-platform libm bit-identity test, resolution of external [verify] items |
| [`probes/`](probes/) | 26 small `.cjcl` programs behind every "probed" claim |
| [`verification/`](verification/) | Scripts and outputs for VERIFY_FOLLOWUPS and ADRs 0044–0046: PySCF/OpenFermion chemistry checks, libm and `dmath` bit dumps (mpmath accuracy check), gate-vs-copy cost benchmark |

`STACK_ROLE_GROUP.md` was named as the scope authority in the audit brief but
does not exist in the repo. See SURFACE_AUDIT's scope-authority note.

## Re-running the probes

```bash
cargo build --release --bin cjcl
```

```bash
for f in docs/quantum_simulation_research_stack/probes/p*.cjcl; do echo "== $f"; ./target/release/cjcl parity "$f" 2>&1 | tail -4; done
```

At audit time, probes p01–p07, p12 and p17 crashed the process (Rust panic,
exit 101), and p22 aborted on a 16 TiB allocation (exit 127). After the P0-1 and
P0-2 fixes, every probe exits 0 and both executors agree:

- Malformed inputs return runtime errors.
- p21 (negative seed) returns values.
- p10 (aliasing) returns `[1, 0]`: the gate no longer changes the original
  circuit (ADR-0044). At audit time it returned `[0, 1]`.
- p13, p14 and p26 now return values (ADR-0045). For example, p13 gives
  ⟨H⟩ = −1.8287352340711216 for a Trotter-evolved state. At audit time all
  three returned "expected a quantum circuit".
