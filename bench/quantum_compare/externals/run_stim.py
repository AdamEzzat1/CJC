"""Stim driver for the CJC quantum benchmark (W3, BENCHMARK_PLAN §3).

    python run_stim.py <run_dir> [--reps 5] [--filter W3]

Two columns per Clifford case (plan §3 W3):
  stim-tableau          TableauSimulator: apply the circuit, then peek_z on
                        every qubit. Apples-to-apples with CJC's
                        StabilizerState (apply, then peek_z).
  stim-compile_sampler  compile_sampler().sample(1000) of the circuit plus a
                        final measurement of every qubit. Bulk sampling; CJC
                        has no counterpart, so this column is context only.

Correctness oracles (seed-independent, exact):
  1. peek_z on every qubit after the circuit must equal CJC's.
  2. CJC's seeded measure-all record is replayed in Stim: before measuring
     qubit q, Stim's peek_z says whether the outcome is forced (and to what)
     given the earlier outcomes; then postselect_z forces CJC's outcome.
     Every forced outcome must match, so CJC's conditional collapse is
     checked at each step, not just on the first (often all-random) layer.
"""
import json
import os
import sys
import time

import numpy as np
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CONFIGS = ["stim-tableau", "stim-compile_sampler"]


def tableau_once(text, n):
    t = time.perf_counter()
    c = stim.Circuit(text)
    build = time.perf_counter() - t
    t = time.perf_counter()
    sim = stim.TableauSimulator()
    sim.do(c)
    execute = time.perf_counter() - t
    t = time.perf_counter()
    peek = np.array([sim.peek_z(q) for q in range(n)], dtype=np.int8)
    observe = time.perf_counter() - t
    return (build, execute, observe), peek.tobytes(), peek


def sampler_once(text, n):
    t = time.perf_counter()
    c = stim.Circuit(text)
    c.append("M", range(n))
    build = time.perf_counter() - t
    t = time.perf_counter()
    sampler = c.compile_sampler(seed=common.SAMPLE_SEED)
    execute = time.perf_counter() - t
    t = time.perf_counter()
    shots = sampler.sample(common.SHOTS)
    observe = time.perf_counter() - t
    return (build, execute, observe), np.packbits(shots, axis=1).tobytes(), None


def run_case(case, config):
    text = open(os.path.join(case["work_dir"], "circuit.stim"), encoding="utf-8").read()
    n = case["params"]["n_qubits"]
    return (lambda: tableau_once(text, n)) if config == "stim-tableau" else (lambda: sampler_once(text, n))


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
    out_path = os.path.join(run_dir, "results_stim.jsonl")
    lines = []
    for case in common.load_manifest(run_dir):
        if case["family"] != "stabilizer" or (filt and filt not in case["case_id"]):
            continue
        for config in CONFIGS:
            fn = run_case(case, config)
            sim_desc = {"name": "stim", "path": config, "backend": "stim", "method": "stabilizer",
                        "precision": None, "threads": 1, "version": stim.__version__}
            first, runs, consistent = common.timed_reps(fn, reps)
            sha = common.sha256(first[1])
            h, rss = common.child_replay(os.path.abspath(__file__), run_dir, case["case_id"], config)
            oracle = linf = None
            extra = {}
            notes = ""
            if config == "stim-tableau":
                cjc = np.fromfile(os.path.join(case["work_dir"], "cjc_peek_z_i8.bin"), dtype=np.int8)
                mism = int(np.sum(cjc != first[2]))
                oracle, linf = "cjc-rust peek_z (exact match required)", float(mism)
                rec_bytes = np.fromfile(os.path.join(case["work_dir"], "cjc_measure_record_u8.bin"),
                                        dtype=np.uint8)
                sim = stim.TableauSimulator()
                sim.do(stim.Circuit(open(os.path.join(case["work_dir"], "circuit.stim"),
                                         encoding="utf-8").read()))
                forced = forced_bad = 0
                for q, o in enumerate(rec_bytes):
                    p = sim.peek_z(q)
                    if p != 0:
                        forced += 1
                        forced_bad += int((p == 1) != (o == 0))
                    if (p == 1 and o == 1) or (p == -1 and o == 0):
                        break  # impossible branch; already counted
                    sim.postselect_z(q, desired_value=bool(o))
                mism += forced_bad
                linf = float(mism)
                extra = {"agreement": {"peek_z_mismatches": int(np.sum(cjc != first[2])),
                                       "deterministic_qubits": int(np.sum(first[2] != 0)),
                                       "record_forced_outcomes": forced,
                                       "record_forced_mismatches": forced_bad,
                                       "n": len(cjc)}}
            else:
                notes = "bulk sampler: no CJC counterpart (context only)"
            rec = common.record(case, sim_desc, runs, first[0], sha, len(first[1]), None, oracle,
                                linf, None, consistent and h == sha, rss, notes=notes, extra=extra)
            lines.append(json.dumps(rec))
            print(f"{case['case_id']:<20} {config:<22} median {rec['timing']['median_s']:.4f}s "
                  f"{extra.get('agreement', '')}", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
