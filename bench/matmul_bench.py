"""
CJC vs Python Benchmark: Matrix Multiplication — Speed & Accuracy
=================================================================
This is the Python equivalent of bench/matmul_bench.cjc.

We implement THREE approaches:
  1. Pure Python loops (no libraries) — fair comparison to CJC's tree-walk interpreter
  2. Python math.fsum (compensated summation) — Python's best built-in accuracy
  3. Python naive sum (standard sum()) — the common approach most people use

This script outputs machine-readable lines that the runner script parses.
"""

import time
import math
import random

# ── Helpers ──────────────────────────────────────────────────

def matmul_pure(A, B, rows_a, cols_a, cols_b):
    """Pure Python triple-loop matmul. No numpy, no libraries."""
    C = [0.0] * (rows_a * cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += A[i * cols_a + k] * B[k * cols_b + j]
            C[i * cols_b + j] = s
    return C

def naive_sum(data):
    """Naive left-to-right summation — what most Python code does."""
    s = 0.0
    for x in data:
        s += x
    return s

def kahan_sum(data):
    """Kahan compensated summation — matches CJC's tensor.sum()."""
    s = 0.0
    c = 0.0
    for x in data:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

# ── Test 1: Small matmul correctness ─────────────────────────

A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
B = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

C = matmul_pure(A, B, 2, 3, 2)

assert C[0] == 58.0, f"Expected 58.0, got {C[0]}"
assert C[1] == 64.0, f"Expected 64.0, got {C[1]}"
assert C[2] == 139.0, f"Expected 139.0, got {C[2]}"
assert C[3] == 154.0, f"Expected 154.0, got {C[3]}"
print("PY_MATMUL_SMALL_PASS")

# ── Test 2: Identity matmul ──────────────────────────────────

I3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
X = [2.5, 3.7, 1.2, 4.8, 0.9, 6.1, 7.3, 8.4, 5.6]
XI = matmul_pure(X, I3, 3, 3, 3)
assert XI[0] == 2.5
assert XI[1] == 3.7
assert XI[5] == 6.1
assert XI[6] == 7.3
print("PY_IDENTITY_MATMUL_PASS")

# ── Test 3: Numerical accuracy ───────────────────────────────

# 4096 ones
ones_4k = [1.0] * 4096
naive_4k = naive_sum(ones_4k)
kahan_4k = kahan_sum(ones_4k)
fsum_4k = math.fsum(ones_4k)
print(f"PY_NAIVE_4096_ONES: {naive_4k}")
print(f"PY_KAHAN_4096_ONES: {kahan_4k}")
print(f"PY_FSUM_4096_ONES: {fsum_4k}")

# 10000 ones
ones_10k = [1.0] * 10000
naive_10k = naive_sum(ones_10k)
kahan_10k = kahan_sum(ones_10k)
fsum_10k = math.fsum(ones_10k)
print(f"PY_NAIVE_10000_ONES: {naive_10k}")
print(f"PY_KAHAN_10000_ONES: {kahan_10k}")
print(f"PY_FSUM_10000_ONES: {fsum_10k}")

# Sum 1+2+...+100 = 5050
seq100 = [float(i) for i in range(1, 101)]
naive_seq = naive_sum(seq100)
kahan_seq = kahan_sum(seq100)
fsum_seq = math.fsum(seq100)
print(f"PY_NAIVE_SEQ_100: {naive_seq}")
print(f"PY_KAHAN_SEQ_100: {kahan_seq}")
print(f"PY_FSUM_SEQ_100: {fsum_seq}")

# ── Test 4: Adversarial sum — the killer test ────────────────
# [1e15, 1.0 * 100, -1e15]  →  true answer = 100.0

adversarial = [1e15] + [1.0] * 100 + [-1e15]
naive_adv = naive_sum(adversarial)
kahan_adv = kahan_sum(adversarial)
fsum_adv = math.fsum(adversarial)
builtin_adv = sum(adversarial)  # Python's built-in sum()

print(f"PY_NAIVE_ADVERSARIAL: {naive_adv}")
print(f"PY_KAHAN_ADVERSARIAL: {kahan_adv}")
print(f"PY_FSUM_ADVERSARIAL: {fsum_adv}")
print(f"PY_BUILTIN_SUM_ADVERSARIAL: {builtin_adv}")
print(f"PY_ADVERSARIAL_EXPECTED: 100.0")

# ── Test 5: Speed — matmul at multiple sizes ─────────────────

sizes = [
    ("8x16 @ 16x8", 8, 16, 8),
    ("32x32 @ 32x32", 32, 32, 32),
    ("64x64 @ 64x64", 64, 64, 64),
    ("128x64 @ 64x128", 128, 64, 128),
]

random.seed(42)  # Deterministic like CJC

total_time = 0.0

for label, m, n, p in sizes:
    A = [random.gauss(0, 1) for _ in range(m * n)]
    B = [random.gauss(0, 1) for _ in range(n * p)]

    start = time.perf_counter()
    C = matmul_pure(A, B, m, n, p)
    elapsed = time.perf_counter() - start
    total_time += elapsed

    assert len(C) == m * p
    print(f"PY_MATMUL_{label}_TIME: {elapsed:.6f}s")
    print(f"PY_MATMUL_{label}_PASS")

# ── Total execution time ─────────────────────────────────────

# Time the full benchmark end-to-end for comparison
print(f"PY_MATMUL_TOTAL_TIME: {total_time:.6f}s")
print("=== ALL PYTHON BENCHMARK TESTS PASSED ===")
