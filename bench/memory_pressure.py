"""
Python Equivalent: Memory Pressure Stress Test
===============================================
Same 3-phase design as memory_pressure.cjc:
  Phase 1: Clean baseline — matmul with no GC pressure
  Phase 2: GC flood before each kernel
  Phase 3: Interleaved GC churn + matmul

Python uses a tracing GC (reference counting + cyclic collector).
Unlike CJC, Python's GC can trigger DURING a matmul loop body if
the allocator decides it's time to collect — causing jitter spikes.

Usage: python bench/memory_pressure.py
"""

import time
import gc
import random
import sys
import math
import json

# ── Configuration ────────────────────────────────────────────

MATMUL_SIZE = 64
NUM_TRIALS = 20
GC_OBJECTS_PER_BURST = 5000

random.seed(42)


def matmul_pure(A, B, n):
    """Pure Python triple-loop matmul for n×n matrices (flat list)."""
    C = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i * n + k] * B[k * n + j]
            C[i * n + j] = s
    return C


def rand_matrix(n):
    """Generate a flat n×n matrix of random floats."""
    return [random.gauss(0, 1) for _ in range(n * n)]


def compute_stats(times):
    """Compute mean, min, max, variance, jitter from a list of times."""
    n = len(times)
    s = sum(times)
    mean = s / n
    sq_sum = sum(t * t for t in times)
    variance = sq_sum / n - mean * mean
    return {
        "total": s,
        "mean": mean,
        "min": min(times),
        "max": max(times),
        "variance": variance,
        "jitter": max(times) - min(times),
    }


# ── Phase 1: Clean baseline ─────────────────────────────────

# Disable automatic GC for fair comparison in Phase 1
gc.disable()

clean_times = []
for _ in range(NUM_TRIALS):
    A = rand_matrix(MATMUL_SIZE)
    B = rand_matrix(MATMUL_SIZE)

    t0 = time.perf_counter()
    C = matmul_pure(A, B, MATMUL_SIZE)
    t1 = time.perf_counter()

    clean_times.append(t1 - t0)

clean_stats = compute_stats(clean_times)

# ── Phase 2: GC Flood before each kernel ─────────────────────

gc.enable()

pressure_times = []
for _ in range(NUM_TRIALS):
    # Flood: create many temporary objects
    garbage = []
    for k in range(GC_OBJECTS_PER_BURST):
        garbage.append({"tag": "temp", "id": k, "data": [0.0] * 10})
    # Force collection
    del garbage
    gc.collect()

    A = rand_matrix(MATMUL_SIZE)
    B = rand_matrix(MATMUL_SIZE)

    t0 = time.perf_counter()
    C = matmul_pure(A, B, MATMUL_SIZE)
    t1 = time.perf_counter()

    pressure_times.append(t1 - t0)

pressure_stats = compute_stats(pressure_times)

# ── Phase 3: Interleaved GC + Matmul ────────────────────────

interleaved_times = []
for _ in range(NUM_TRIALS):
    # Allocate 1000 objects without collecting (heap grows)
    garbage = []
    for k in range(1000):
        garbage.append({"tag": "churn", "val": k})

    A = rand_matrix(MATMUL_SIZE)
    B = rand_matrix(MATMUL_SIZE)

    t0 = time.perf_counter()
    C = matmul_pure(A, B, MATMUL_SIZE)
    t1 = time.perf_counter()

    interleaved_times.append(t1 - t0)

    # Collect after matmul
    del garbage
    gc.collect()

interleaved_stats = compute_stats(interleaved_times)

# ── Output JSON for the runner to parse ──────────────────────

results = {
    "clean": {**clean_stats, "times": clean_times},
    "pressure": {**pressure_stats, "times": pressure_times},
    "interleaved": {**interleaved_stats, "times": interleaved_times},
    "gc_stats": {
        "collections": gc.get_count(),
        "thresholds": gc.get_threshold(),
    }
}

print(json.dumps(results))
