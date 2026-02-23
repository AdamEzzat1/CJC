#!/usr/bin/env python3
"""
CJC Benchmark Suite: Python/NumPy MLP Implementation
=====================================================
Manual backprop MLP with ReLU activation, MSE loss, vanilla SGD.
Emits JSONL results. Single-threaded BLAS enforced via environment.

Usage:
    python bench_numpy_mlp.py --case microbatch --trials 5
    python bench_numpy_mlp.py --case many_matmuls --trials 5
    python bench_numpy_mlp.py --case stability --trials 5
    python bench_numpy_mlp.py --case mini --trials 1   # quick validation
"""

import os
import sys
import json
import time
import struct
import hashlib
import argparse

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Enforce single-threaded BLAS ───────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


# ── Deterministic RNG ──────────────────────────────────────────────────────
# We use NumPy's PCG64 with a fixed seed for reproducibility.
# This differs from CJC's SplitMix64, so data generation must be
# independently validated (same distribution, not bit-identical).
# For fairness, both use their native deterministic RNG.

SEED = 12345


# ── Benchmark Cases ────────────────────────────────────────────────────────

CASES = {
    "microbatch": {
        "B": 8, "D": 512, "H": 1024, "O": 64, "L": 8,
        "warmup": 2000, "measure": 50000, "lr": 0.01,
    },
    "many_matmuls": {
        "B": 256, "D": 1024, "H": 1024, "O": 128, "L": 64,
        "warmup": 500, "measure": 10000, "lr": 0.01,
    },
    "stability": {
        "B": 64, "D": 2048, "H": 2048, "O": 256, "L": 16,
        "warmup": 1000, "measure": 50000, "lr": 0.01,
    },
    "mini": {
        "B": 4, "D": 8, "H": 16, "O": 4, "L": 2,
        "warmup": 10, "measure": 100, "lr": 0.01,
    },
}


# ── MLP Implementation ────────────────────────────────────────────────────

class MLPLayer:
    """Single fully-connected layer with pre-allocated scratch buffers."""

    __slots__ = ("W", "b", "dW", "db", "Z", "A", "X_input", "relu_mask")

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator):
        # He initialization
        scale = np.sqrt(2.0 / fan_in)
        self.W = rng.standard_normal((fan_in, fan_out)).astype(np.float64) * scale
        self.b = np.zeros((1, fan_out), dtype=np.float64)
        # Scratch buffers (allocated once, reused)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.Z = None  # Set during forward
        self.A = None
        self.X_input = None
        self.relu_mask = None


def forward_layer(layer: MLPLayer, X: np.ndarray, is_output: bool = False) -> np.ndarray:
    """Forward pass: Z = X @ W + b, A = relu(Z) or Z for output layer."""
    layer.X_input = X
    layer.Z = X @ layer.W + layer.b
    if is_output:
        layer.A = layer.Z  # No activation on output
        layer.relu_mask = None
    else:
        layer.relu_mask = (layer.Z > 0).astype(np.float64)
        layer.A = layer.Z * layer.relu_mask
    return layer.A


def backward_layer(layer: MLPLayer, dA: np.ndarray, is_output: bool = False) -> np.ndarray:
    """Backward pass through one layer. Returns dX for the layer below."""
    B = dA.shape[0]
    if is_output:
        dZ = dA  # No activation derivative for output
    else:
        dZ = dA * layer.relu_mask  # Element-wise multiply with mask

    # dW = X^T @ dZ
    np.dot(layer.X_input.T, dZ, out=layer.dW)
    # db = sum_rows(dZ)
    np.sum(dZ, axis=0, keepdims=True, out=layer.db)
    # dX = dZ @ W^T
    dX = dZ @ layer.W.T
    return dX


def mse_loss(pred: np.ndarray, target: np.ndarray):
    """MSE loss and its gradient."""
    diff = pred - target
    loss = np.mean(diff * diff)
    # d(MSE)/d(pred) = 2 * (pred - target) / n_elements
    grad = 2.0 * diff / diff.size
    return loss, grad


def sgd_update(layer: MLPLayer, lr: float):
    """Vanilla SGD update."""
    layer.W -= lr * layer.dW
    layer.b -= lr * layer.db


def compute_grad_norm(layers):
    """Compute L2 norm of all gradients."""
    total = 0.0
    for layer in layers:
        total += np.sum(layer.dW * layer.dW)
        total += np.sum(layer.db * layer.db)
    return np.sqrt(total)


def hash_params(layers):
    """Compute a deterministic hash of all parameters."""
    h = hashlib.sha256()
    for layer in layers:
        h.update(layer.W.tobytes())
        h.update(layer.b.tobytes())
    return h.hexdigest()[:16]


def hash_floats(values):
    """Hash a list of float64 values."""
    h = hashlib.sha256()
    for v in values:
        h.update(struct.pack("<d", v))
    return h.hexdigest()[:16]


# ── Training Step ──────────────────────────────────────────────────────────

def train_step(layers, X, Y, lr):
    """One complete training step: forward + loss + backward + update."""
    # Forward
    A = X
    for i, layer in enumerate(layers):
        is_output = (i == len(layers) - 1)
        A = forward_layer(layer, A, is_output=is_output)

    # Loss
    loss, dA = mse_loss(A, Y)

    # Backward
    for i in range(len(layers) - 1, -1, -1):
        is_output = (i == len(layers) - 1)
        dA = backward_layer(layers[i], dA, is_output=is_output)

    # Update
    for layer in layers:
        sgd_update(layer, lr)

    return loss


# ── Benchmark Runner ───────────────────────────────────────────────────────

def run_trial(case_name: str, case_cfg: dict, trial: int):
    """Run a single benchmark trial and return JSONL-compatible dict."""
    B = case_cfg["B"]
    D = case_cfg["D"]
    H = case_cfg["H"]
    O = case_cfg["O"]
    L = case_cfg["L"]
    warmup = case_cfg["warmup"]
    measure = case_cfg["measure"]
    lr = case_cfg["lr"]

    rng = np.random.default_rng(SEED)

    # Initialize layers
    layers = []
    # Input -> first hidden
    layers.append(MLPLayer(D, H, rng))
    # Hidden -> hidden (L-1 more hidden layers, total L hidden)
    for _ in range(L - 1):
        layers.append(MLPLayer(H, H, rng))
    # Last hidden -> output
    layers.append(MLPLayer(H, O, rng))

    # Generate synthetic data: X random, Y = X_trunc @ teacher + noise
    X_data = rng.standard_normal((B, D)).astype(np.float64)
    teacher = rng.standard_normal((D, O)).astype(np.float64) * 0.1
    Y_data = (X_data @ teacher + rng.standard_normal((B, O)).astype(np.float64) * 0.01)

    # ── Warmup ──
    for _ in range(warmup):
        train_step(layers, X_data, Y_data, lr)

    # ── Measurement ──
    step_times = []
    loss_samples = []
    grad_norm_samples = []
    nan_count = 0
    diverged = False

    for step in range(measure):
        t0 = time.perf_counter()
        loss = train_step(layers, X_data, Y_data, lr)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        if step % 100 == 0:
            loss_samples.append(float(loss))
            gn = float(compute_grad_norm(layers))
            grad_norm_samples.append(gn)

            if np.isnan(loss) or np.isinf(loss):
                nan_count += 1
                diverged = True
            if loss > 1e10:
                diverged = True

    # ── Compute metrics ──
    step_times_us = sorted([t * 1e6 for t in step_times])
    n = len(step_times_us)
    total_seconds = sum(step_times)

    p50 = step_times_us[n * 50 // 100] if n > 0 else 0
    p95 = step_times_us[n * 95 // 100] if n > 0 else 0
    p99 = step_times_us[n * 99 // 100] if n > 0 else 0

    steps_per_sec = measure / total_seconds if total_seconds > 0 else 0
    examples_per_sec = steps_per_sec * B

    # Memory
    peak_rss_mb = 0.0
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        peak_rss_mb = proc.memory_info().rss / (1024 * 1024)
    elif HAS_RESOURCE:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss_mb = rusage.ru_maxrss / 1024  # KB on Linux
        if sys.platform == "darwin":
            peak_rss_mb = rusage.ru_maxrss / (1024 * 1024)  # bytes on macOS

    # Hashes
    final_h = hash_params(layers)
    loss_h = hash_floats(loss_samples)

    # Gradient norm stats
    gn_min = min(grad_norm_samples) if grad_norm_samples else 0
    gn_max = max(grad_norm_samples) if grad_norm_samples else 0

    result = {
        "impl": "numpy",
        "case": case_name,
        "trial": trial,
        "steps": measure,
        "batch_size": B,
        "seconds": round(total_seconds, 4),
        "steps_per_sec": round(steps_per_sec, 2),
        "examples_per_sec": round(examples_per_sec, 2),
        "p50_us": round(p50, 1),
        "p95_us": round(p95, 1),
        "p99_us": round(p99, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "final_hash": final_h,
        "loss_hash": loss_h,
        "loss_start": round(loss_samples[0], 8) if loss_samples else None,
        "loss_end": round(loss_samples[-1], 8) if loss_samples else None,
        "grad_norm_min": round(gn_min, 8),
        "grad_norm_max": round(gn_max, 8),
        "nan_count": nan_count,
        "diverged": diverged,
    }

    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NumPy MLP Benchmark")
    parser.add_argument("--case", required=True, choices=list(CASES.keys()),
                        help="Benchmark case to run")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (default: stdout)")
    args = parser.parse_args()

    cfg = CASES[args.case]
    out_file = open(args.output, "a") if args.output else sys.stdout

    for trial in range(1, args.trials + 1):
        result = run_trial(args.case, cfg, trial)
        line = json.dumps(result)
        print(line, file=out_file, flush=True)
        # Also print progress to stderr
        print(f"[numpy] {args.case} trial {trial}/{args.trials}: "
              f"{result['steps_per_sec']:.1f} steps/sec, "
              f"p50={result['p50_us']:.0f}us, "
              f"p95={result['p95_us']:.0f}us, "
              f"loss {result['loss_start']:.6f} -> {result['loss_end']:.6f}",
              file=sys.stderr)

    if args.output:
        out_file.close()


if __name__ == "__main__":
    main()
