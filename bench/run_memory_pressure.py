"""
CJC vs Python — Memory Pressure Benchmark Runner
=================================================
Runs both benchmarks, parses output, generates clean results.

Usage: python bench/run_memory_pressure.py
Output: bench/MEMORY_PRESSURE_RESULTS.txt
"""

import subprocess
import sys
import os
import json
import time
import math
import re

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCH_DIR)
CJC_EXE = os.path.join(PROJECT_ROOT, "target", "release", "cjc.exe")
CJC_BENCH = os.path.join(BENCH_DIR, "memory_pressure.cjc")
PY_BENCH = os.path.join(BENCH_DIR, "memory_pressure.py")
OUTPUT_FILE = os.path.join(BENCH_DIR, "MEMORY_PRESSURE_RESULTS.txt")


def parse_cjc_output(stdout, stderr):
    """Parse CJC benchmark output into structured data."""
    data = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("CLEAN_MEAN:"):
            data["clean_mean"] = float(line.split(":")[1].strip())
        elif line.startswith("CLEAN_JITTER:"):
            data["clean_jitter"] = float(line.split(":")[1].strip())
        elif line.startswith("PRESSURE_MEAN:"):
            data["pressure_mean"] = float(line.split(":")[1].strip())
        elif line.startswith("PRESSURE_JITTER:"):
            data["pressure_jitter"] = float(line.split(":")[1].strip())
        elif line.startswith("INTERLEAVED_MEAN:"):
            data["interleaved_mean"] = float(line.split(":")[1].strip())
        elif line.startswith("INTERLEAVED_JITTER:"):
            data["interleaved_jitter"] = float(line.split(":")[1].strip())

    # Parse full phase data from output
    phases = {}
    current_phase = None
    for line in stdout.splitlines():
        line = line.strip()
        if "PHASE 1:" in line:
            current_phase = "clean"
            phases[current_phase] = {}
        elif "PHASE 2:" in line:
            current_phase = "pressure"
            phases[current_phase] = {}
        elif "PHASE 3:" in line:
            current_phase = "interleaved"
            phases[current_phase] = {}
        elif "PHASE 4:" in line:
            current_phase = None
        elif current_phase and line.startswith("Total time:"):
            phases[current_phase]["total"] = float(line.split(":")[1].strip())
        elif current_phase and line.startswith("Mean kernel:"):
            phases[current_phase]["mean"] = float(line.split(":")[1].strip())
        elif current_phase and line.startswith("Min:"):
            phases[current_phase]["min"] = float(line.split(":")[1].strip())
        elif current_phase and line.startswith("Max:"):
            phases[current_phase]["max"] = float(line.split(":")[1].strip())
        elif current_phase and line.startswith("Variance:"):
            phases[current_phase]["variance"] = float(line.split(":")[1].strip())
        elif current_phase and line.startswith("Jitter"):
            phases[current_phase]["jitter"] = float(line.split(":")[1].strip())

    data["phases"] = phases

    # Parse total execution time from stderr
    for line in stderr.splitlines():
        if "[cjc --time]" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "took":
                    data["total_time"] = float(parts[i + 1])
                    break
    return data


def run_cjc(runs=3):
    """Run CJC benchmark multiple times, return best run."""
    best = None
    for _ in range(runs):
        result = subprocess.run(
            [CJC_EXE, "run", CJC_BENCH, "--time"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.returncode != 0:
            print(f"CJC FAILED:\n{result.stderr}")
            sys.exit(1)
        data = parse_cjc_output(result.stdout, result.stderr)
        if best is None or data.get("total_time", 999) < best.get("total_time", 999):
            best = data
    return best


def run_python(runs=3):
    """Run Python benchmark multiple times, return best run."""
    best = None
    for _ in range(runs):
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, PY_BENCH],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            print(f"Python FAILED:\n{result.stderr}")
            sys.exit(1)
        data = json.loads(result.stdout)
        data["total_time"] = elapsed
        if best is None or elapsed < best.get("total_time", 999):
            best = data
    return best


def fmt_us(seconds):
    """Format seconds as microseconds string."""
    return f"{seconds * 1_000_000:.1f} us"


def fmt_ms(seconds):
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.3f} ms"


def fmt_sci(val):
    """Format small floats in scientific notation."""
    if val == 0:
        return "0"
    return f"{val:.2e}"


def main():
    print("=" * 60)
    print("  CJC vs Python - Memory Pressure Benchmark")
    print("=" * 60)
    print()

    print("Running CJC benchmark (3 runs, best of 3)...")
    cjc = run_cjc(runs=3)
    print(f"  CJC total: {cjc.get('total_time', 0):.3f}s")
    print()

    print("Running Python benchmark (3 runs, best of 3)...")
    py = run_python(runs=3)
    print(f"  Python total: {py.get('total_time', 0):.3f}s")
    print()

    # Extract CJC phases
    cjc_c = cjc["phases"]["clean"]
    cjc_p = cjc["phases"]["pressure"]
    cjc_i = cjc["phases"]["interleaved"]

    # Extract Python phases
    py_c = py["clean"]
    py_p = py["pressure"]
    py_i = py["interleaved"]

    # ── Generate report ──────────────────────────────────────

    lines = []
    w = lines.append

    w("=" * 78)
    w("  CJC vs PYTHON - MEMORY PRESSURE STRESS TEST RESULTS")
    w("=" * 78)
    w("")
    w(f"  Date:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  CJC Version:  v0.1.0 (tree-walk interpreter, release build)")
    w(f"  Python:       {sys.version.split()[0]}")
    w(f"  Platform:     Windows")
    w(f"  Matrix Size:  64x64 (4,096 elements per matrix)")
    w(f"  Trials:       20 per phase")
    w(f"  GC Burst:     5,000 objects per burst")
    w(f"  Runs:         3 (best of 3 reported)")
    w("")

    # ── What this test does ──────────────────────────────────
    w("=" * 78)
    w("  WHAT THIS TEST DOES")
    w("=" * 78)
    w("")
    w("  This benchmark measures whether garbage collection activity affects")
    w("  the timing consistency of math kernels (matrix multiplication).")
    w("")
    w("  Phase 1 (CLEAN):       Run 20 matmul kernels with zero GC activity.")
    w("                         This establishes the baseline timing.")
    w("")
    w("  Phase 2 (GC FLOOD):    Before EACH kernel, allocate 5,000 temporary")
    w("                         objects on the GC heap, then trigger a full")
    w("                         collection. Then run the matmul kernel.")
    w("")
    w("  Phase 3 (INTERLEAVED): Allocate 1,000 objects, run matmul, THEN")
    w("                         collect. This maximizes the chance of GC")
    w("                         pauses interfering with kernel timing.")
    w("")
    w("  KEY METRIC: Jitter (max_time - min_time across 20 trials).")
    w("  Low jitter  = GC does NOT affect math kernel performance.")
    w("  High jitter = GC pauses are bleeding into math kernel timing.")
    w("")

    # ── Speed comparison ─────────────────────────────────────
    w("=" * 78)
    w("  SECTION 1: KERNEL SPEED (64x64 matmul)")
    w("=" * 78)
    w("")
    w("  Mean time per 64x64 matmul (20 trials):")
    w("")
    w(f"                          CJC              Python           CJC Speedup")
    w(f"    -------------------------------------------------------------------------")
    cjc_clean_ms = cjc_c["mean"] * 1000
    py_clean_ms = py_c["mean"] * 1000
    speedup_clean = py_c["mean"] / cjc_c["mean"] if cjc_c["mean"] > 0 else 0
    w(f"    Clean baseline        {cjc_clean_ms:>8.3f} ms       {py_clean_ms:>8.3f} ms       {speedup_clean:>6.1f}x")

    cjc_pres_ms = cjc_p["mean"] * 1000
    py_pres_ms = py_p["mean"] * 1000
    speedup_pres = py_p["mean"] / cjc_p["mean"] if cjc_p["mean"] > 0 else 0
    w(f"    Under GC pressure     {cjc_pres_ms:>8.3f} ms       {py_pres_ms:>8.3f} ms       {speedup_pres:>6.1f}x")

    cjc_intl_ms = cjc_i["mean"] * 1000
    py_intl_ms = py_i["mean"] * 1000
    speedup_intl = py_i["mean"] / cjc_i["mean"] if cjc_i["mean"] > 0 else 0
    w(f"    Interleaved           {cjc_intl_ms:>8.3f} ms       {py_intl_ms:>8.3f} ms       {speedup_intl:>6.1f}x")
    w("")

    # ── Jitter comparison (the main event) ───────────────────
    w("=" * 78)
    w("  SECTION 2: JITTER (timing consistency under memory pressure)")
    w("=" * 78)
    w("")
    w("  Jitter = max_kernel_time - min_kernel_time across 20 trials.")
    w("  Lower is better. Measures how much GC pauses disrupt math timing.")
    w("")
    w(f"                          CJC              Python")
    w(f"    --------------------------------------------------------")

    cjc_cj = cjc_c["jitter"] * 1_000_000
    py_cj = py_c["jitter"] * 1_000_000
    w(f"    Clean jitter          {cjc_cj:>8.1f} us       {py_cj:>8.1f} us")

    cjc_pj = cjc_p["jitter"] * 1_000_000
    py_pj = py_p["jitter"] * 1_000_000
    w(f"    GC flood jitter       {cjc_pj:>8.1f} us       {py_pj:>8.1f} us")

    cjc_ij = cjc_i["jitter"] * 1_000_000
    py_ij = py_i["jitter"] * 1_000_000
    w(f"    Interleaved jitter    {cjc_ij:>8.1f} us       {py_ij:>8.1f} us")
    w("")

    # Compute CJC jitter change from clean to pressure
    cjc_jitter_ratio_p = cjc_p["jitter"] / cjc_c["jitter"] if cjc_c["jitter"] > 0 else 0
    cjc_jitter_ratio_i = cjc_i["jitter"] / cjc_c["jitter"] if cjc_c["jitter"] > 0 else 0
    py_jitter_ratio_p = py_p["jitter"] / py_c["jitter"] if py_c["jitter"] > 0 else 0
    py_jitter_ratio_i = py_i["jitter"] / py_c["jitter"] if py_c["jitter"] > 0 else 0

    w("  Jitter change from clean baseline:")
    w("")
    w(f"                          CJC              Python")
    w(f"    --------------------------------------------------------")
    w(f"    GC flood vs clean     {cjc_jitter_ratio_p:>7.2f}x          {py_jitter_ratio_p:>7.2f}x")
    w(f"    Interleaved vs clean  {cjc_jitter_ratio_i:>7.2f}x          {py_jitter_ratio_i:>7.2f}x")
    w("")

    # ── Variance comparison ──────────────────────────────────
    w("=" * 78)
    w("  SECTION 3: VARIANCE (statistical spread of kernel times)")
    w("=" * 78)
    w("")
    w("  Variance of kernel execution times (lower = more predictable):")
    w("")
    w(f"                          CJC              Python")
    w(f"    --------------------------------------------------------")
    w(f"    Clean                 {fmt_sci(cjc_c['variance']):>14}   {fmt_sci(py_c['variance']):>14}")
    w(f"    GC flood              {fmt_sci(cjc_p['variance']):>14}   {fmt_sci(py_p['variance']):>14}")
    w(f"    Interleaved           {fmt_sci(cjc_i['variance']):>14}   {fmt_sci(py_i['variance']):>14}")
    w("")

    # ── Detailed phase tables ────────────────────────────────
    w("=" * 78)
    w("  SECTION 4: DETAILED PHASE RESULTS")
    w("=" * 78)
    w("")

    for label, cjc_ph, py_ph in [
        ("Phase 1: CLEAN (no GC)", cjc_c, py_c),
        ("Phase 2: GC FLOOD (5000 allocs + collect before each kernel)", cjc_p, py_p),
        ("Phase 3: INTERLEAVED (1000 allocs, matmul, then collect)", cjc_i, py_i),
    ]:
        w(f"  {label}")
        w(f"  {'='*70}")
        w(f"    {'':30}  {'CJC':>14}  {'Python':>14}")
        w(f"    {'':30}  {'-'*14}  {'-'*14}")
        w(f"    {'Total time (20 trials)':30}  {fmt_ms(cjc_ph['total']):>14}  {fmt_ms(py_ph['total']):>14}")
        w(f"    {'Mean per kernel':30}  {fmt_ms(cjc_ph['mean']):>14}  {fmt_ms(py_ph['mean']):>14}")
        w(f"    {'Min kernel time':30}  {fmt_ms(cjc_ph['min']):>14}  {fmt_ms(py_ph['min']):>14}")
        w(f"    {'Max kernel time':30}  {fmt_ms(cjc_ph['max']):>14}  {fmt_ms(py_ph['max']):>14}")
        w(f"    {'Jitter (max - min)':30}  {fmt_us(cjc_ph['jitter']):>14}  {fmt_us(py_ph['jitter']):>14}")
        w(f"    {'Variance':30}  {fmt_sci(cjc_ph['variance']):>14}  {fmt_sci(py_ph['variance']):>14}")
        w("")

    # ── Explanation ───────────────────────────────────────────
    w("=" * 78)
    w("  SECTION 5: WHY CJC'S DESIGN MATTERS")
    w("=" * 78)
    w("")
    w("  CJC's 3-Layer Memory Architecture:")
    w("")
    w("    Layer 1 (nogc):  Tensors use Reference Counting + Copy-on-Write.")
    w("                     NO garbage collector touches tensor memory.")
    w("                     Matmul runs in a tight Rust loop — zero GC pauses.")
    w("")
    w("    Layer 2:         Multiple dispatch resolves at call time.")
    w("                     Function values are stack-allocated.")
    w("")
    w("    Layer 3 (gc):    Mark-sweep GC manages short-lived objects.")
    w("                     Collection runs BETWEEN kernels, never during.")
    w("                     The GC cannot \"stop the world\" during math.")
    w("")
    w("  Python's Memory Architecture:")
    w("")
    w("    Everything shares ONE memory system (reference counting + cyclic GC).")
    w("    Temporary dict/list allocations during a computation can trigger")
    w("    gc.collect() at ANY point — including inside your inner loop.")
    w("    This creates unpredictable jitter spikes in kernel timing.")
    w("")
    w("  Java / JVM languages have it worse: a \"stop-the-world\" GC pause can")
    w("  freeze ALL threads — including your GPU dispatch and math threads —")
    w("  for milliseconds or longer.")
    w("")
    w("  CJC's design guarantee: If your code is in a `nogc` block, the GC")
    w("  CANNOT pause it. Tensor operations are always in Layer 1 (RC + COW).")
    w("  You get the convenience of GC for high-level code AND the predictability")
    w("  of manual memory management for performance-critical math.")
    w("")

    # ── Total execution time ─────────────────────────────────
    w("=" * 78)
    w("  SECTION 6: TOTAL EXECUTION TIME")
    w("=" * 78)
    w("")
    total_speedup = py.get("total_time", 1) / cjc.get("total_time", 1)
    w(f"  CJC total:    {cjc.get('total_time', 0):.3f}s   (all 3 phases + GC + tensor ops)")
    w(f"  Python total: {py.get('total_time', 0):.3f}s   (all 3 phases + GC + pure-loop matmul)")
    w(f"  Speedup:      {total_speedup:.1f}x")
    w("")

    w("=" * 78)
    w("  END OF MEMORY PRESSURE BENCHMARK REPORT")
    w("=" * 78)

    report = "\n".join(lines) + "\n"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Results written to: {OUTPUT_FILE}")
    print()
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
