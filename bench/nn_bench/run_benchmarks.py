#!/usr/bin/env python3
"""
CJC vs NumPy Neural Network Benchmark Orchestrator
===================================================
Generates parameterized CJC benchmark files, runs both implementations,
collects JSONL results, validates determinism, and produces summary tables.

Usage:
    python run_benchmarks.py                   # Run all cases
    python run_benchmarks.py --case mini       # Run only mini case
    python run_benchmarks.py --impl cjc        # Run only CJC
    python run_benchmarks.py --trials 3        # Override trial count
    python run_benchmarks.py --timeout 300     # Timeout per trial (seconds)
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
import tempfile
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
CJC_ROOT = SCRIPT_DIR.parent.parent
CJC_EXE = CJC_ROOT / "target" / "release" / "cjc.exe"
CJC_TEMPLATE = SCRIPT_DIR / "bench_cjc_mlp.cjc"
NUMPY_BENCH = SCRIPT_DIR / "bench_numpy_mlp.py"
RESULTS_DIR = SCRIPT_DIR / "results"

CASES = {
    "mini": {
        "B": 4, "D": 8, "H": 16, "O": 4, "L": 2,
        "warmup": 10, "measure": 100, "lr": 0.01,
        "description": "Mini validation case (fast)",
    },
    "microbatch": {
        "B": 8, "D": 512, "H": 1024, "O": 64, "L": 8,
        "warmup": 2000, "measure": 50000, "lr": 0.01,
        "description": "Microbatch overhead killer",
    },
    "many_matmuls": {
        "B": 256, "D": 1024, "H": 1024, "O": 128, "L": 64,
        "warmup": 500, "measure": 10000, "lr": 0.01,
        "description": "Many small matmuls",
    },
    "stability": {
        "B": 64, "D": 2048, "H": 2048, "O": 256, "L": 16,
        "warmup": 1000, "measure": 50000, "lr": 0.01,
        "description": "Numerical stability and long-run reliability",
    },
}

# CJC interpreter is much slower than NumPy BLAS, so we use reduced
# step counts for CJC on the large cases to get meaningful results
# within a reasonable time.
CJC_CASE_OVERRIDES = {
    "microbatch": {"warmup": 20, "measure": 200},
    "many_matmuls": {"warmup": 5, "measure": 50},
    "stability": {"warmup": 10, "measure": 200},
}

SEED = 12345


# ── CJC Template Generator ────────────────────────────────────────────────

def generate_cjc_bench(case_name: str, case_cfg: dict, trial: int,
                       output_path: Path) -> None:
    """Generate a parameterized CJC benchmark file from the template."""
    template = CJC_TEMPLATE.read_text()

    # Apply CJC-specific overrides for large cases
    cfg = dict(case_cfg)
    if case_name in CJC_CASE_OVERRIDES:
        cfg.update(CJC_CASE_OVERRIDES[case_name])

    # Replace configuration variables at the top of the file
    replacements = {
        "let BATCH = 4;": f"let BATCH = {cfg['B']};",
        "let DIM_IN = 8;": f"let DIM_IN = {cfg['D']};",
        "let DIM_H = 16;": f"let DIM_H = {cfg['H']};",
        "let DIM_OUT = 4;": f"let DIM_OUT = {cfg['O']};",
        "let NUM_LAYERS = 2;": f"let NUM_LAYERS = {cfg['L']};",
        "let WARMUP_STEPS = 10;": f"let WARMUP_STEPS = {cfg['warmup']};",
        "let MEASURE_STEPS = 100;": f"let MEASURE_STEPS = {cfg['measure']};",
        "let LR = 0.01;": f"let LR = {cfg['lr']};",
        "let SEED_VAL = 12345;": f"let SEED_VAL = {SEED};",
        'let CASE_NAME = "mini";': f'let CASE_NAME = "{case_name}";',
        "let TRIAL_NUM = 1;": f"let TRIAL_NUM = {trial};",
    }

    result = template
    for old, new in replacements.items():
        result = result.replace(old, new)

    output_path.write_text(result)


# ── Runner Functions ───────────────────────────────────────────────────────

def run_numpy_trial(case_name: str, trial: int, timeout: int) -> dict:
    """Run a single NumPy benchmark trial."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        sys.executable, str(NUMPY_BENCH),
        "--case", case_name,
        "--trials", "1",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=env, cwd=str(SCRIPT_DIR)
        )
        if result.returncode != 0:
            print(f"  [WARN] NumPy {case_name} trial {trial} failed: {result.stderr[:200]}")
            return {"impl": "numpy", "case": case_name, "trial": trial, "error": result.stderr[:500]}

        # Parse JSONL output (first line)
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                data["trial"] = trial  # Override trial number
                return data

        return {"impl": "numpy", "case": case_name, "trial": trial, "error": "no JSONL output"}

    except subprocess.TimeoutExpired:
        return {"impl": "numpy", "case": case_name, "trial": trial, "error": f"timeout ({timeout}s)"}


def run_cjc_trial(case_name: str, case_cfg: dict, trial: int, timeout: int) -> dict:
    """Run a single CJC benchmark trial."""
    # Generate parameterized CJC file
    gen_dir = RESULTS_DIR / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    cjc_file = gen_dir / f"bench_{case_name}_t{trial}.cjc"
    generate_cjc_bench(case_name, case_cfg, trial, cjc_file)

    if not CJC_EXE.exists():
        # Try building
        print("  Building CJC...")
        subprocess.run(["cargo", "build", "--release"], cwd=str(CJC_ROOT),
                        capture_output=True, timeout=120)

    if not CJC_EXE.exists():
        return {"impl": "cjc", "case": case_name, "trial": trial,
                "error": "CJC binary not found"}

    cmd = [str(CJC_EXE), "run", str(cjc_file), "--seed", str(SEED)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(CJC_ROOT)
        )
        if result.returncode != 0:
            print(f"  [WARN] CJC {case_name} trial {trial} failed: {result.stderr[:200]}")
            return {"impl": "cjc", "case": case_name, "trial": trial,
                    "error": result.stderr[:500]}

        # Parse output — CJC prints JSON with spaces between args
        output = result.stdout.strip()
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("{") and "impl" in line:
                # Clean up CJC's space-separated print output
                # Remove spaces around colons and commas in the JSON
                cleaned = re.sub(r'\s+', '', line)
                # Re-add necessary structure
                # Actually, just try to parse it after removing extra spaces
                # CJC print joins with " ", so "{ \"impl\":\"cjc\" }" etc.
                try:
                    data = json.loads(cleaned)
                    data["trial"] = trial
                    return data
                except json.JSONDecodeError:
                    # Try a more aggressive cleanup
                    try:
                        # Remove ALL spaces then add back the ones JSON needs
                        cleaned2 = line.replace(" ", "")
                        data = json.loads(cleaned2)
                        data["trial"] = trial
                        return data
                    except json.JSONDecodeError:
                        pass

        # If JSON parsing failed, try to extract metrics from output
        return {"impl": "cjc", "case": case_name, "trial": trial,
                "error": f"could not parse output: {output[:300]}",
                "raw_output": output[:1000]}

    except subprocess.TimeoutExpired:
        return {"impl": "cjc", "case": case_name, "trial": trial,
                "error": f"timeout ({timeout}s)"}


# ── Summary Table Generation ──────────────────────────────────────────────

def format_table(results: list) -> str:
    """Generate a human-readable comparison table."""
    lines = []
    lines.append("=" * 100)
    lines.append("CJC vs NumPy Neural Network Benchmark Results")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 100)

    # Group by case
    cases_seen = {}
    for r in results:
        case = r.get("case", "unknown")
        impl = r.get("impl", "unknown")
        if case not in cases_seen:
            cases_seen[case] = {"numpy": [], "cjc": []}
        if "error" not in r:
            cases_seen[case][impl].append(r)

    for case_name, impls in cases_seen.items():
        lines.append("")
        lines.append(f"--- Case: {case_name} ---")
        cfg = CASES.get(case_name, {})
        lines.append(f"    {cfg.get('description', '')}")
        lines.append(f"    B={cfg.get('B')}, D={cfg.get('D')}, H={cfg.get('H')}, "
                     f"O={cfg.get('O')}, L={cfg.get('L')}")
        lines.append("")

        header = f"  {'Metric':<30} {'NumPy':>15} {'CJC':>15} {'CJC/NumPy':>12}"
        lines.append(header)
        lines.append("  " + "-" * 72)

        def avg(data_list, key):
            vals = [d.get(key, 0) for d in data_list if d.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0

        numpy_data = impls.get("numpy", [])
        cjc_data = impls.get("cjc", [])

        if not numpy_data and not cjc_data:
            lines.append("  No results available.")
            continue

        metrics = [
            ("Steps/sec (mean)", "steps_per_sec"),
            ("Examples/sec (mean)", "examples_per_sec"),
            ("p50 step time (us)", "p50_us"),
            ("p95 step time (us)", "p95_us"),
            ("p99 step time (us)", "p99_us"),
            ("Peak RSS (MB)", "peak_rss_mb"),
            ("Loss start", "loss_start"),
            ("Loss end", "loss_end"),
            ("Grad norm min", "grad_norm_min"),
            ("Grad norm max", "grad_norm_max"),
            ("NaN count", "nan_count"),
        ]

        for label, key in metrics:
            np_val = avg(numpy_data, key)
            cjc_val = avg(cjc_data, key)
            ratio = ""
            if np_val and cjc_val and np_val != 0:
                r = cjc_val / np_val
                ratio = f"{r:.2f}x"
            lines.append(f"  {label:<30} {np_val:>15.2f} {cjc_val:>15.2f} {ratio:>12}")

        # Determinism check
        lines.append("")
        lines.append("  Determinism:")
        for impl_name, data in [("NumPy", numpy_data), ("CJC", cjc_data)]:
            if len(data) >= 2:
                hashes = set(d.get("final_hash", "") for d in data)
                loss_hashes = set(d.get("loss_hash", "") for d in data)
                det_ok = len(hashes) == 1 and len(loss_hashes) == 1
                status = "PASS (all hashes match)" if det_ok else f"FAIL ({len(hashes)} unique hashes)"
                lines.append(f"    {impl_name}: {status}")
            elif len(data) == 1:
                lines.append(f"    {impl_name}: Only 1 trial (need >= 2 for check)")
            else:
                lines.append(f"    {impl_name}: No data")

        # Winner determination
        lines.append("")
        lines.append("  Winner by dimension:")
        np_sps = avg(numpy_data, "steps_per_sec")
        cjc_sps = avg(cjc_data, "steps_per_sec")
        if np_sps and cjc_sps:
            winner = "CJC" if cjc_sps > np_sps else "NumPy"
            ratio = max(np_sps, cjc_sps) / max(min(np_sps, cjc_sps), 1e-10)
            lines.append(f"    Throughput: {winner} ({ratio:.1f}x faster)")

        np_p99 = avg(numpy_data, "p99_us")
        cjc_p99 = avg(cjc_data, "p99_us")
        if np_p99 and cjc_p99:
            winner = "CJC" if cjc_p99 < np_p99 else "NumPy"
            lines.append(f"    Tail latency (p99): {winner}")

        np_rss = avg(numpy_data, "peak_rss_mb")
        cjc_rss = avg(cjc_data, "peak_rss_mb")
        if np_rss and cjc_rss:
            winner = "CJC" if cjc_rss < np_rss else "NumPy"
            lines.append(f"    Memory (peak RSS): {winner}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("NOTE: CJC uses a tree-walk interpreter (no LLVM/JIT).")
    lines.append("NumPy uses BLAS for matmul (single-threaded).")
    lines.append("CJC step counts may be reduced for large cases due to interpreter speed.")
    lines.append("=" * 100)

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CJC vs NumPy NN Benchmark Orchestrator")
    parser.add_argument("--case", choices=list(CASES.keys()), default=None,
                        help="Run only this case (default: all)")
    parser.add_argument("--impl", choices=["numpy", "cjc", "both"], default="both",
                        help="Which implementation to run")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per case (default: 5)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per trial in seconds (default: 600)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Determine which cases to run
    cases_to_run = [args.case] if args.case else list(CASES.keys())
    impls_to_run = [args.impl] if args.impl != "both" else ["numpy", "cjc"]

    # Setup output
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"results_{timestamp}.jsonl"
    summary_path = out_dir / f"summary_{timestamp}.txt"

    print(f"CJC vs NumPy NN Benchmark Suite")
    print(f"Cases: {cases_to_run}")
    print(f"Implementations: {impls_to_run}")
    print(f"Trials per case: {args.trials}")
    print(f"Timeout per trial: {args.timeout}s")
    print(f"Output: {jsonl_path}")
    print()

    all_results = []

    for case_name in cases_to_run:
        cfg = CASES[case_name]
        print(f"=== Case: {case_name} ({cfg['description']}) ===")

        for impl in impls_to_run:
            effective_steps = cfg["measure"]
            if impl == "cjc" and case_name in CJC_CASE_OVERRIDES:
                effective_steps = CJC_CASE_OVERRIDES[case_name]["measure"]

            print(f"  [{impl}] Running {args.trials} trials "
                  f"({effective_steps} measured steps each)...")

            for trial in range(1, args.trials + 1):
                print(f"    Trial {trial}/{args.trials}...", end=" ", flush=True)
                t0 = time.time()

                if impl == "numpy":
                    result = run_numpy_trial(case_name, trial, args.timeout)
                else:
                    result = run_cjc_trial(case_name, cfg, trial, args.timeout)

                elapsed = time.time() - t0

                if "error" in result:
                    print(f"ERROR ({elapsed:.1f}s): {result['error'][:80]}")
                else:
                    sps = result.get("steps_per_sec", 0)
                    p50 = result.get("p50_us", 0)
                    loss_e = result.get("loss_end", 0)
                    print(f"OK ({elapsed:.1f}s): {sps:.1f} steps/s, "
                          f"p50={p50:.0f}us, loss_end={loss_e:.6f}")

                all_results.append(result)

                # Write JSONL incrementally
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

        print()

    # Generate summary
    print("Generating summary...")
    summary = format_table(all_results)
    summary_path.write_text(summary)
    print(summary)
    print(f"\nResults saved to: {jsonl_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
