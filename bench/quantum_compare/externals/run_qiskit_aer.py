"""Qiskit Aer driver for the CJC quantum benchmark (BENCHMARK_PLAN §3, §7).

    python run_qiskit_aer.py <run_dir> [--reps 5] [--filter W1]

Runs every dense (W1, W2) and MPS (W4) case from <run_dir>/manifest.jsonl in
two configurations (fairness rule §7):
  aer-1t-nofusion  1 thread, gate fusion off  (algorithmic comparison)
  aer-default      Aer's defaults             (user-experienced comparison)
and appends schema records to <run_dir>/results_aer.jsonl.

Accuracy is measured against CJC's own dumped output (amplitudes for dense
cases, <Z_i> for MPS cases): L-infinity and, for dense, fidelity |<cjc|aer>|^2.
Replay: each case is re-run once in a fresh process and its output hash
compared.

Phases (each record's timing.phases_median_s):
  build    qasm2.loads + save instructions
  execute  AerSimulator.run(...).result() and the statevector copy
  observe  dense only: probabilities + 1,000 seeded shots, drawn with a numpy
           cumulative sum and searchsorted (the same algorithm as CJC's batch
           sampler). qiskit.quantum_info.Statevector.sample_memory is not
           used: it took ~54 s at 22 qubits and swamped the Aer timings in
           the 2026-09-25 baseline. Shots are not expected to match CJC's
           (different RNG); only their hash is recorded.
"""
import json
import os
import sys
import time

import numpy as np
import qiskit
import qiskit_aer
from qiskit import qasm2
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CONFIGS = ["aer-1t-nofusion", "aer-default"]


def simulator(config, method, chi=None):
    opts = {"method": method, "precision": "double"}
    if config == "aer-1t-nofusion":
        opts.update(max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1,
                    fusion_enable=False)
    if chi is not None:
        opts["matrix_product_state_max_bond_dimension"] = chi
    return AerSimulator(**opts)


def dense_once(text, sim):
    t = time.perf_counter()
    qc = qasm2.loads(text)
    qc.save_statevector()
    build = time.perf_counter() - t
    t = time.perf_counter()
    sv = np.asarray(sim.run(qc).result().get_statevector(qc))
    execute = time.perf_counter() - t
    t = time.perf_counter()
    probs = np.abs(sv) ** 2
    cdf = np.cumsum(probs)
    r = np.random.default_rng(common.SAMPLE_SEED).random(common.SHOTS) * cdf[-1]
    shots = np.minimum(np.searchsorted(cdf, r, side="right"), len(cdf) - 1)
    observe = time.perf_counter() - t
    out = probs.astype("<f8").tobytes()
    samples = shots.astype("<u8").tobytes()
    return (build, execute, observe), out, sv, samples


def mps_once(text, sim, n):
    t = time.perf_counter()
    qc = qasm2.loads(text)
    for i in range(n):
        qc.save_expectation_value(Pauli("Z"), [i], label=f"z{i}")
    build = time.perf_counter() - t
    t = time.perf_counter()
    data = sim.run(qc).result().data()
    execute = time.perf_counter() - t
    z = np.array([float(np.real(data[f"z{i}"])) for i in range(n)])
    return (build, execute, 0.0), z.astype("<f8").tobytes(), z


def run_case(case, config):
    text = open(os.path.join(case["work_dir"], "circuit.qasm"), encoding="utf-8").read()
    n = case["params"]["n_qubits"]
    if case["family"] == "statevector":
        sim = simulator(config, "statevector")
        return lambda: dense_once(text, sim)
    if case["family"] == "matrix_product_state":
        sim = simulator(config, "matrix_product_state", case["params"]["chi_max"])
        return lambda: mps_once(text, sim, n)
    return None


def child(run_dir, case_id, config):
    case = next(c for c in common.load_manifest(run_dir) if c["case_id"] == case_id)
    r = run_case(case, config)()
    print(f"HASH {common.sha256(r[1])} RSS {common.peak_rss_bytes()}")


def main():
    run_dir = sys.argv[1]
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        return child(run_dir, sys.argv[i + 1], sys.argv[i + 2])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5
    filt = sys.argv[sys.argv.index("--filter") + 1] if "--filter" in sys.argv else None
    out_path = os.path.join(run_dir, "results_aer.jsonl")
    lines = []
    for case in common.load_manifest(run_dir):
        if filt and filt not in case["case_id"]:
            continue
        if case["family"] not in ("statevector", "matrix_product_state"):
            continue
        for config in CONFIGS:
            fn = run_case(case, config)
            sim_desc = {"name": "qiskit-aer", "path": config, "backend": "aer",
                        "method": case["family"], "precision": "f64",
                        "threads": 1 if config == "aer-1t-nofusion" else os.cpu_count(),
                        "version": qiskit_aer.__version__, "qiskit": qiskit.__version__}
            try:
                first, runs, consistent = common.timed_reps(fn, reps)
            except Exception as e:  # recorded, never hidden
                lines.append(json.dumps(common.record(case, sim_desc, [], None, None, None, None,
                                                      None, None, None, None, None, "error", repr(e))))
                continue
            sha = common.sha256(first[1])
            h, rss = common.child_replay(os.path.abspath(__file__), run_dir, case["case_id"], config)
            linf = fidelity = None
            oracle = None
            samples_sha = None
            if case["family"] == "statevector":
                sv, samples_sha = first[2], common.sha256(first[3])
                dump = os.path.join(case["work_dir"], "cjc_amplitudes_c128le.bin")
                if os.path.exists(dump):
                    cjc = np.fromfile(dump, dtype="<c16")
                    linf = float(np.max(np.abs(sv - cjc)))
                    fidelity = float(abs(np.vdot(cjc, sv)) ** 2)
                    oracle = "cjc-rust amplitudes"
                elif case["workload"] == "W1_ghz_dense":
                    p = np.abs(sv) ** 2
                    e = p.copy()
                    e[0] -= 0.5
                    e[-1] -= 0.5
                    linf, oracle = float(np.max(np.abs(e))), "analytic-ghz"
            else:
                dump = os.path.join(case["work_dir"], "cjc_z_expectations_f64le.bin")
                cjc = np.fromfile(dump, dtype="<f8")
                linf, oracle = float(np.max(np.abs(first[2] - cjc))), "cjc-rust <Z_i>"
            rec = common.record(case, sim_desc, runs, first[0], sha, len(first[1]), samples_sha,
                                oracle, linf, fidelity, consistent and h == sha, rss)
            lines.append(json.dumps(rec))
            print(f"{case['case_id']:<28} {config:<16} median {rec['timing']['median_s']:.4f}s "
                  f"linf {linf}", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
