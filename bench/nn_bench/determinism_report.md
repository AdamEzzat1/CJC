# Determinism Report: CJC vs NumPy MLP Benchmark

## Methodology

Each benchmark case is run multiple times with identical seeds in fresh processes.
Determinism is verified by comparing:

1. **Final weight hash**: SHA-256 hash of all weight/bias parameter bytes
2. **Loss trace hash**: SHA-256 hash of loss values sampled every 100 steps

## Expected Results

### NumPy (single-threaded BLAS)
- With `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`:
  NumPy should produce **bitwise identical** results across runs with the same seed.
- If hashes differ, the cause is likely:
  - Non-deterministic parallel reduction (should not occur with single-thread)
  - Different BLAS library versions
  - Platform-specific floating-point behavior

### CJC
- CJC uses SplitMix64 RNG with `--seed` flag for deterministic initialization
- All reductions use Kahan summation (deterministic order)
- Same seed + same program = **bitwise identical** output guaranteed
- The `--reproducible` flag enforces additional determinism constraints

## Results

*(Filled in by run_benchmarks.py)*

| Case | Impl | Trials | Final Hash Match | Loss Hash Match | Status |
|------|------|--------|-----------------|-----------------|--------|
| mini | numpy | 5 | | | |
| mini | cjc | 5 | | | |
| microbatch | numpy | 5 | | | |
| microbatch | cjc | 5 | | | |
| many_matmuls | numpy | 5 | | | |
| many_matmuls | cjc | 5 | | | |
| stability | numpy | 5 | | | |
| stability | cjc | 5 | | | |

## Discrepancy Analysis

If hashes do not match, investigate:

1. **RNG state**: Verify seed is set before each trial
2. **Reduction order**: Verify summation order is deterministic
3. **Threading**: Verify single-threaded execution
4. **Floating-point mode**: Check for non-default FP rounding modes
5. **Memory layout**: Verify tensor strides are consistent
