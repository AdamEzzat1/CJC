# CJC vs Python/NumPy Neural Network Benchmark Specification

## Overview

This benchmark suite evaluates **base CJC** (no external libraries) against **Python + NumPy**
on training a fully-connected neural network (MLP) with manual backpropagation.

## Model Architecture

- **Type**: Multi-Layer Perceptron (MLP)
- **Layers**: `D -> H -> H -> ... -> H -> O` (L hidden layers, each of width H)
- **Activation**: ReLU
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Vanilla SGD with fixed learning rate 0.01

### Required Operations

**Forward pass per layer:**
- `Z = X @ W + b`  (matmul + bias add)
- `A = relu(Z)`     (element-wise ReLU)

**Backward pass per layer (manual backprop):**
- `dZ = dA * relu_mask(Z)`  (element-wise multiply with mask)
- `dW = X^T @ dZ`           (matmul with transposed input)
- `db = sum_rows(dZ)`       (reduce sum along batch dimension)
- `dX = dZ @ W^T`           (matmul with transposed weights)

**Parameter update:**
- `W -= lr * dW`
- `b -= lr * db`

## Data Generation

Synthetic deterministic data from a fixed seed:
- **Input X**: Generated via LCG-based RNG mapped to approximate normal distribution (Box-Muller)
- **Target Y**: `Y = X_trunc @ T + noise` where T is a fixed "teacher" matrix, noise is small

Both implementations must use identical seed logic producing identical initial data.

**Seed contract**: seed = `12345` for all runs.

## Benchmark Cases

### Case A: "Microbatch Overhead Killer"

| Parameter | Value |
|-----------|-------|
| Batch B   | 8     |
| Input D   | 512   |
| Hidden H  | 1024  |
| Output O  | 64    |
| Depth L   | 8     |
| Warmup    | 2000  |
| Measure   | 50000 |

**Purpose**: Measure interpreter/dispatch overhead and tight-loop efficiency with small batches.

### Case B: "Many Small Matmuls"

| Parameter | Value |
|-----------|-------|
| Batch B   | 256   |
| Input D   | 1024  |
| Hidden H  | 1024  |
| Output O  | 128   |
| Depth L   | 64    |
| Warmup    | 500   |
| Measure   | 10000 |

**Purpose**: Stress per-op overhead with many layers (many GEMM calls per step).

### Case C: "Numerical Stability and Long-Run Reliability"

| Parameter | Value |
|-----------|-------|
| Batch B   | 64    |
| Input D   | 2048  |
| Hidden H  | 2048  |
| Output O  | 256   |
| Depth L   | 16    |
| Warmup    | 1000  |
| Measure   | 50000 |

**Purpose**: Detect drift, memory growth, jitter, and numerical stability issues over long runs.

## Fairness Rules

1. **Threading**: NumPy single-threaded BLAS via environment variables:
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
2. **Dtype**: `float64` everywhere (CJC tensors are f64-only in v1)
3. **Same math**: Identical activation, initialization, optimizer, batch shapes
4. **Same seed**: Identical RNG seed (12345) and deterministic data generation
5. **No free wins**: NumPy must not use PyTorch/JAX; CJC must not call external BLAS
6. **Warmup**: Discard warmup steps before measurement
7. **Reporting**: p50/p95/p99 step time, not just mean

## Metrics (per trial)

### Performance
- `steps_per_sec`: Total measured steps / total measured time
- `examples_per_sec`: steps_per_sec * batch_size
- `p50_us`: Median step time in microseconds
- `p95_us`: 95th percentile step time
- `p99_us`: 99th percentile step time

### Memory
- `peak_rss_mb`: Peak resident set size in MB

### Determinism
- `final_hash`: Hash of all weights/biases at end of training
- `loss_hash`: Hash of loss values sampled every 100 steps

### Numerical Stability
- `loss_start`: Loss at step 0 (first measured step)
- `loss_end`: Loss at final measured step
- `grad_norm_min`: Minimum gradient norm observed (sampled every 100 steps)
- `grad_norm_max`: Maximum gradient norm observed
- `nan_count`: Number of NaN/Inf values encountered
- `diverged`: Boolean, true if loss exceeded 1e10 or NaN occurred

## JSONL Output Schema

Each trial produces one JSONL line:

```json
{
  "impl": "cjc",
  "case": "microbatch",
  "trial": 1,
  "steps": 50000,
  "seconds": 12.34,
  "steps_per_sec": 4050.1,
  "examples_per_sec": 32400.8,
  "p50_us": 220,
  "p95_us": 260,
  "p99_us": 310,
  "peak_rss_mb": 512,
  "final_hash": "a1b2c3d4e5f6",
  "loss_hash": "f6e5d4c3b2a1",
  "loss_start": 1.23,
  "loss_end": 0.45,
  "grad_norm_min": 0.001,
  "grad_norm_max": 5.2,
  "nan_count": 0,
  "diverged": false
}
```

## Run Protocol

For each case and implementation:

1. Set seed to 12345
2. Initialize model and data
3. Run warmup steps (discard timings)
4. Run measured steps:
   - Record step time every step
   - Record loss every 100 steps
   - Record gradient norm every 100 steps
5. Compute metrics and emit JSONL
6. Repeat for 5 trials
7. Determinism check: Run a 6th trial in a fresh process, compare hashes with trial 1

## Completion Criteria

- Both implementations run all 3 cases
- Each case has 5 trials + 1 determinism double-run
- JSONL outputs exist and can be diffed
- Summary table compares mean speed, p95/p99 jitter, peak RSS, determinism, stability

## Scaling Note (CJC Interpreter Limitation)

CJC v1 uses a tree-walk interpreter. The benchmark workload sizes specified above
(e.g., 512x1024 matmuls) will be extremely slow in the CJC interpreter compared to
NumPy's BLAS-backed matmul. The orchestrator script will:
- Run a "mini" validation pass first (small shapes, few steps) to verify correctness
- For full benchmarks, use configurable step counts and timeout logic
- Report whatever steps complete within the time budget
