# Determinism@Scale Benchmark Suite

## Overview

The Determinism@Scale benchmark suite verifies that CJC produces **bit-identical
results** across runs, across execution engines (eval vs MIR), and across seeds.
This is the foundational guarantee for reproducible scientific computing.

## Running

```bash
# Run all benchmark tests
cargo test --test test_bench_v0_1

# Run a specific benchmark category
cargo test --test test_bench_v0_1 test_determinism
cargo test --test test_bench_v0_1 test_seed_sweep
cargo test --test test_bench_v0_1 test_gc_boundary
cargo test --test test_bench_v0_1 test_primitive_abi
```

## Benchmarks

### 1. Pipeline (`bench_pipeline.cjc`)

**1M-row data pipeline**: Generates a 1000x1000 tensor (1M elements) via
`Tensor.randn`, transforms with `broadcast2`, computes row sums via `matmul`,
and produces statistics. Six stage hashes verify determinism at each step.

### 2. Neural Network (`bench_nn_deep.cjc`)

**50-layer deep forward pass**: Width 64, batch size 4, ReLU activations.
Each layer: `matmul` -> `broadcast2("add", ...)` -> `broadcast("relu", ...)` ->
normalize. Stage hashes every 10 layers (5 checkpoints).

### 3. Seed Stress (`bench_seed_stress.cjc`)

**10-step matmul chain**: Accumulates `Tensor.randn` matrices through repeated
matrix multiplication with normalization. Tests that RNG state evolves
deterministically across many draws.

### 4. GC Boundary (`bench_gc_boundary.cjc`)

**GC stress test**: 120 rounds of 500 `gc_alloc` + `gc_collect` per round.
Verifies that a reference tensor's `snap_hash` is unchanged after GC pressure.
Tests that garbage collection never corrupts tensor data.

### 5. Primitive Coverage (`bench_primitive_coverage.cjc`)

**ABI lock for 40+ builtins**: Hashes the output of every tested primitive
(tensor constructors, tensor ops, broadcast unary/binary, math, stats) and
produces a single master hash. If ANY builtin changes behavior, the master
hash changes and the golden-hash test fails.

### 6. Reorder Determinism (`bench_reorder_det.cjc`)

**Operation reordering check**: Performs the same computation in different
orders and verifies that identical operations produce identical hashes
(determinism) while different operations produce different hashes (sanity).

## Test Categories

### Determinism (12 tests)

Each benchmark is run twice with seed 42 in the same engine. All BENCH and
STAGE hashes must match. This verifies run-to-run determinism.

Additionally, each benchmark is run in both eval and MIR engines. All hashes
must match. This verifies engine parity.

### Seed Sweep (4 tests)

Multiple seeds (0, 1, 42, 99, 12345, 999999) are tested. Each seed must be
internally deterministic (run twice = same hashes). Different seeds must
produce different hashes (sanity check).

### GC Boundary (3 tests)

Verifies tensor integrity under GC pressure: all 120 rounds produce matching
hashes, 4 stage checkpoints are present, and eval/MIR parity holds.

### Primitive ABI Lock (3 tests)

A golden master hash (pinned in source) locks down the behavior of 40+
builtins. If any builtin changes, the hash changes and the test fails.
Eval/MIR parity is also verified.

## Output Protocol

CJC programs emit structured lines:

```
BENCH:name:seed:time_ms:gc_live:out_hash
STAGE:name:stage_idx:stage_hash
```

- `time_ms` -- wall-clock milliseconds via `clock()`
- `gc_live` -- `gc_live_count()` at end
- `out_hash` / `stage_hash` -- `snap_hash(value)` (SHA-256)

Rust test helpers parse these lines and compare hashes for determinism and
parity assertions.

## Architecture

```
tests/
  test_bench_v0_1.rs            -- entry point
  bench_v0_1/
    mod.rs                      -- module index
    helpers.rs                  -- shared parse/run/assert helpers
    test_determinism.rs         -- 12 run-twice + parity tests
    test_seed_sweep.rs          -- 4 multi-seed tests
    test_gc_boundary.rs         -- 3 GC stress tests
    test_primitive_abi.rs       -- 3 golden hash + ABI lock tests
    cjc_programs/
      bench_pipeline.cjc        -- 1M-row pipeline
      bench_nn_deep.cjc         -- 50-layer NN
      bench_seed_stress.cjc     -- matmul chain
      bench_gc_boundary.cjc     -- GC pressure
      bench_primitive_coverage.cjc -- 40+ primitive ABI
      bench_reorder_det.cjc     -- operation reorder
```

## Determinism Guarantees

1. **SplitMix64 RNG**: Platform-independent PRNG, seeded via `--seed N`
2. **IEEE 754 f64**: All floating-point operations use deterministic IEEE semantics
3. **Row-major iteration**: Tensor operations iterate in fixed index order
4. **SHA-256 hashing**: Hand-rolled FIPS 180-4 implementation in `cjc-snap`
5. **No threading**: All computation is single-threaded
