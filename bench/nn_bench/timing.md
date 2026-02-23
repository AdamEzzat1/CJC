# Timing Methodology

## Clock Source

### CJC
- Uses `clock()` builtin which calls `std::time::Instant::now()` (Rust)
- Monotonic, high-resolution (nanosecond on most platforms)
- Overhead: ~25ns per call on modern x86_64

### Python/NumPy
- Uses `time.perf_counter()` which uses the highest-resolution available clock
- Monotonic, typically nanosecond resolution
- Overhead: ~50-100ns per call

## Measurement Protocol

### Per-Step Timing

```
for step in measured_steps:
    t0 = clock()
    train_step(...)
    t1 = clock()
    step_times.append(t1 - t0)
```

- Each step includes: forward pass + loss computation + backward pass + parameter update
- Clock calls are outside the training step to minimize measurement overhead
- Overhead budget: 2 clock calls per step (~50-200ns) vs typical step time (microseconds to milliseconds)

### Warmup Separation

- Warmup steps are executed first to:
  - Warm CPU caches and branch predictors
  - Allow JIT compilation in Python (if applicable)
  - Stabilize memory allocation patterns
- Warmup step timings are **discarded**
- Measurement begins with a fresh timer after warmup completes

### Percentile Computation

Step times are collected into a sorted array. Percentiles are computed as:
- **p50**: `sorted_times[len * 50 / 100]` (median)
- **p95**: `sorted_times[len * 95 / 100]`
- **p99**: `sorted_times[len * 99 / 100]`

All times are reported in microseconds (us) for consistency.

## Overhead Controls

1. **No I/O in hot loop**: Loss/gradient logging uses a counter and samples every N steps
2. **Pre-allocated buffers**: Both implementations pre-allocate all scratch space before measurement
3. **No GC pressure in CJC**: Training step uses only Tensor/Buffer operations (nogc-compatible)
4. **Single-threaded NumPy**: BLAS thread count = 1 via environment variables
5. **Fresh process per trial**: Each trial runs in a separate process to avoid cross-trial interference

## Memory Measurement

### Peak RSS
- **Python**: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (in KB on Linux, bytes on macOS)
- **CJC**: Peak RSS measured by the orchestrator via `psutil` or `/proc/self/status`
- Measured at the end of training (after all steps complete)

### Allocation Tracking (CJC)
- CJC's nogc zone prevents GC allocations in the training loop
- Buffer allocations are tracked by observing `Buffer.refcount()` patterns
- If pre-allocation is correct, no new buffers should be created during measured steps

## Determinism Validation

Each trial records:
- **Step time array**: For percentile computation
- **Loss samples**: Loss value every 100 steps (for loss_hash)
- **Final weights**: All parameter values at end of training (for final_hash)

Hash computation:
- Concatenate all f64 values as little-endian bytes
- Compute a simple checksum (sum of all bytes modulo 2^64, reported as hex)
- Two runs with same seed must produce identical hashes
