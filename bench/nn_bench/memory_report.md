# Memory Report: CJC vs NumPy MLP Benchmark

## Measurement Method

### Peak RSS (Resident Set Size)
- **NumPy**: Measured via `psutil.Process().memory_info().rss` or `resource.getrusage()`
- **CJC**: Measured by orchestrator via external process monitoring

### Allocation Behavior

#### NumPy
- NumPy pre-allocates arrays where possible (`out=` parameter in `np.dot`)
- Python's garbage collector handles temporary arrays
- Expected: stable RSS after initialization for fixed-size workloads

#### CJC
- Tensors use `Buffer<T>` with reference counting and Copy-on-Write
- Training step uses only Tensor/Buffer operations (nogc-compatible)
- New tensors created per operation (functional style), old ones freed by refcount
- Expected: moderate allocation pressure but deterministic memory behavior
- No GC collections during training step (Buffer-only operations)

## Expected Memory Usage

| Case | Model Params | Activations | Data | Est. Total |
|------|-------------|-------------|------|------------|
| mini | ~2 KB | ~1 KB | ~1 KB | ~5 MB (+ runtime) |
| microbatch | ~70 MB | ~1 MB | ~1 MB | ~100 MB |
| many_matmuls | ~530 MB | ~33 MB | ~33 MB | ~700 MB |
| stability | ~550 MB | ~2 MB | ~2 MB | ~600 MB |

## Results

*(Filled in by run_benchmarks.py)*

| Case | NumPy Peak RSS (MB) | CJC Peak RSS (MB) | Ratio |
|------|--------------------|--------------------|-------|
| mini | | | |
| microbatch | | | |
| many_matmuls | | | |
| stability | | | |

## Memory Growth Analysis

For Case C (stability), track RSS at start and end of training:
- If RSS grows > 10% during measured steps, flag as potential memory leak
- CJC's refcount-based Buffer system should not leak
- NumPy's array temporaries should be collected by Python GC

## NoGC Zone Compliance (CJC)

The training step should operate entirely within Buffer/Tensor allocation:
- No class instances created during training
- No GC collections triggered
- Verify via `gc_live_count()` before and after training
