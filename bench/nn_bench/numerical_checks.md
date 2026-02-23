# Numerical Checks: CJC vs NumPy MLP Correctness Validation

## Purpose

Before running full benchmarks, both implementations must produce identical results
on small test cases. This document defines the validation protocol.

## Check 1: Forward Pass Sanity (Tiny Model)

**Setup**: 1-layer MLP, no bias, no activation
- Input: `X = [[1.0, 2.0], [3.0, 4.0]]` (2x2)
- Weights: `W = [[0.5, -0.5, 1.0], [1.0, 0.5, -1.0]]` (2x3)
- Expected `Z = X @ W = [[2.5, 0.5, -1.0], [5.5, 0.5, -1.0]]`

Both implementations compute `Z = matmul(X, W)` and compare element-wise with tolerance 1e-10.

## Check 2: ReLU Correctness

- Input: `[-2.0, -1.0, 0.0, 1.0, 2.0]`
- Expected: `[0.0, 0.0, 0.0, 1.0, 2.0]`

## Check 3: MSE Loss

- Prediction: `[1.0, 2.0, 3.0]`
- Target: `[1.5, 2.5, 3.5]`
- Expected MSE: `mean((pred - target)^2) = mean([0.25, 0.25, 0.25]) = 0.25`

## Check 4: Gradient Check (Finite Differences)

For a 1-layer MLP (D=2, H=3, O=1):
1. Compute analytical gradient dW via manual backprop
2. Compute numerical gradient via finite differences: `(f(W+eps) - f(W-eps)) / (2*eps)` with eps=1e-5
3. Compare: relative error < 1e-4 for each parameter

**Parameters to check**: First weight W[0,0] and first bias b[0].

## Check 5: Cross-Implementation Agreement

Run both implementations for 10 steps on a tiny model (D=4, H=8, O=2, B=2, L=2)
with seed=12345, lr=0.01. Compare:

- Loss at each step (tolerance 1e-8)
- Final weights (tolerance 1e-8)

Any discrepancy indicates a bug in one implementation.

## Check 6: Numerical Stability Indicators

Both implementations must report:
- **Loss curve**: Loss at step 0 and every 100 steps
- **Gradient norms**: L2 norm of all gradients, sampled every 100 steps
- **NaN/Inf detection**: Flag any occurrence

## Stability Thresholds

- Loss should decrease monotonically for the first ~1000 steps on synthetic data
- Gradient norms should not exceed 100x the initial gradient norm
- No NaN/Inf values should appear with proper initialization (Xavier/He)
- After 50000 steps, loss should be < 50% of initial loss

## Weight Initialization

Both implementations use the same initialization:
- Weights: `W[i,j] = randn() * sqrt(2.0 / fan_in)` (He initialization)
- Biases: `b[j] = 0.0` (zero initialization)

Where `randn()` uses the deterministic SplitMix64-based RNG with seed 12345.
