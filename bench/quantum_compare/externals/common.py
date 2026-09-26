"""Shared helpers for the external-simulator drivers.

Each driver reads `manifest.jsonl` written by `quantum_compare run`, runs the
same circuits (from each case's `circuit.qasm` / `circuit.stim`), and appends
records with the same `cjc-quantum-bench/v1` schema to its own JSONL file.
"""
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

SHOTS = 1000
SAMPLE_SEED = 7


def load_manifest(run_dir):
    with open(os.path.join(run_dir, "manifest.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def peak_rss_bytes():
    """Peak working set (Windows) or max RSS (Linux) of this process."""
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                (n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # 64-bit HANDLE, not int
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def percentile(xs, p):
    s = sorted(xs)
    pos = p * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def timed_reps(fn, reps, warm_total=None):
    """Run fn() once as warm-up, then `reps` times. fn returns (phases, output_bytes).

    Returns (first_result, list_of_phase_tuples, consistent_hash_bool).
    Long cases (warm-up > 30 s) get at most 3 reps, as on the CJC side.
    """
    first = fn()
    total = sum(first[0])
    n = min(reps, 3) if total > 30 else reps
    runs, ok = [], True
    for _ in range(n):
        r = fn()
        runs.append(r[0])
        ok &= sha256(r[1]) == sha256(first[1])
    return first, runs, ok


def child_replay(script, run_dir, case_id, config):
    """Run one case in a fresh Python process; returns (sha256, peak_rss)."""
    out = subprocess.run(
        [sys.executable, script, run_dir, "--child", case_id, config],
        capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("HASH "):
            _, h, _, rss = line.split()
            return h, (int(rss) if rss.isdigit() else None)
    return None, None


def machine():
    return {"cpu": platform.processor() or None, "threads": str(os.cpu_count()),
            "os": f"{platform.system().lower()} {platform.machine().lower()}"}


def record(case, sim, runs, first_phases, sha, out_len, samples_sha, oracle, linf,
           fidelity, replay_ok, peak_rss, status="ok", notes="", extra=None):
    totals = [sum(r) for r in runs]
    med = statistics.median(totals) if totals else None
    iqr = (percentile(totals, 0.75) - percentile(totals, 0.25)) if totals else None
    ph = [statistics.median([r[i] for r in runs]) if runs else None for i in range(3)]
    rec = {
        "schema": "cjc-quantum-bench/v1",
        "run_id": None,
        "workload": case["workload"],
        "case_id": case["case_id"],
        "params": case["params"],
        "simulator": sim,
        "timing": {"reps": len(runs), "warmup": 1 if runs else 0, "median_s": med, "iqr_s": iqr,
                   "phases_median_s": {"build": ph[0], "execute": ph[1], "observe": ph[2]}},
        "memory": {"peak_rss_bytes": peak_rss},
        "output": {"kind": None, "sha256": sha, "len": out_len, "samples_sha256": samples_sha},
        "accuracy": {"oracle": oracle, "linf": linf, "fidelity": fidelity},
        "determinism": {"replay_ok": replay_ok, "cross_executor_ok": None, "replay_runs": 2},
        "machine": machine(),
        "toolchain": {"python": platform.python_version()},
        "status": status,
        "notes": notes,
    }
    if extra:
        rec.update(extra)
    return rec
