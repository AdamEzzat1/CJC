# P1 Correctness & Types Change Log

## Summary

This document describes the changes made during the P1 correctness/types phase
of the CJC compiler project. All changes compile and pass the full test suite
(2080 tests, 0 failures, 0 regressions).

## What Was Added

### 1. End-to-End Fixture Runner (`tests/fixtures/`)

**Files:** `tests/fixtures/runner.rs`, 10 fixture directories with `.cjc` source + `.stdout`/`.stderr` golden files.

A production-quality fixture runner that:
- Discovers `.cjc` files recursively under `tests/fixtures/`
- Parses and executes each via the MIR interpreter (seed=42 for determinism)
- Compares stdout against `.stdout` golden files (with CRLF normalization)
- For error fixtures (`.stderr` exists), runs with type-checking and verifies errors
- Supports `CJC_FIXTURE_UPDATE=1` env var to auto-update/create goldens
- Prints clear diffs on mismatch

**Fixture categories:**
| Category       | Fixtures | Tests                                    |
|---------------|----------|------------------------------------------|
| `basic/`      | 2        | hello_world, arithmetic                  |
| `numeric/`    | 1        | float_ops (f64 arithmetic)               |
| `for_loops/`  | 1        | range_loop (0..N iteration)              |
| `closures/`   | 1        | basic_closure (lambda + higher-order fn) |
| `tco/`        | 1        | tail_call (tail-recursive sum)           |
| `fstring/`    | 1        | interpolation (f-string formatting)      |
| `enums/`      | 1        | option_basic (Option<T> + unwrap)        |
| `structs/`    | 1        | basic_struct (field access)              |
| `error_cases/`| 1        | type_error (type mismatch detection)     |

### 2. Proptest Infrastructure (`tests/prop_tests/`)

**Files:** `tests/prop_tests/mod.rs`, `parser_props.rs`, `type_checker_props.rs`, `round_trip_props.rs`

9 property-based tests using the `proptest` crate:

- **Parser properties (3 tests):**
  - Never panics on random ASCII input
  - Never panics on plausible-but-invalid CJC tokens
  - Valid minimal programs parse without errors

- **Type checker properties (3 tests):**
  - Never panics on type-mismatched programs
  - Handles function call programs without panicking
  - Full MIR pipeline never panics on well-formed programs

- **Round-trip / determinism properties (3 tests):**
  - Same program + same seed = identical output (determinism)
  - Let-binding programs are deterministic
  - Parse preserves function count across invocations

### 3. Runtime Module Split (`crates/cjc-runtime/src/`)

Split the monolithic `lib.rs` (3979 lines) into 12 focused submodules while
maintaining full backward API compatibility via re-exports.

**New modules:**
| Module          | Lines | Contents                              |
|----------------|-------|---------------------------------------|
| `buffer.rs`    | ~100  | `Buffer<T>` — COW memory allocation   |
| `tensor.rs`    | ~1270 | `Tensor` — N-D tensor with ops        |
| `scratchpad.rs`| ~140  | `Scratchpad` — KV-cache scratch buffer|
| `aligned_pool.rs`|~190 | `AlignedPool` — SIMD-aligned memory   |
| `kernel_bridge.rs`|~530| Raw-pointer kernel bridge functions    |
| `paged_kv.rs`  | ~260  | `PagedKvCache` — vLLM-style KV cache  |
| `gc.rs`        | ~140  | `GcHeap`/`GcRef` — mark-sweep GC      |
| `sparse.rs`    | ~140  | `SparseCsr`/`SparseCoo` — sparse tensors|
| `det_map.rs`   | ~290  | `DetMap` — deterministic hash map      |
| `linalg.rs`    | ~210  | LU decompose, SVD, determinant, inverse|
| `value.rs`     | ~280  | `Value` enum, `Bf16`, `FnValue`        |
| `error.rs`     | ~46   | `RuntimeError` enum                    |

**Backward compatibility:** All public types are re-exported from `lib.rs`.
Downstream code using `use cjc_runtime::Tensor` continues to work unchanged.

### 4. Pre-existing Features (from earlier sessions)

The following were implemented in prior sessions and verified in this run:

- **ADR documents** (`docs/adr/`): 10 ADRs including README index
- **Audit tests** (`tests/audit_tests/`): 16 test files covering traits, modules,
  match exhaustiveness, MIR form, type errors, float folding, matmul, parallelism,
  datatypes, const exprs, f-strings, impl traits, monomorphization, result/option,
  TCO, mutable bindings
- **Type system additions**: Option<T>, Result<T,E>, Range<T>, Set<T>, Queue<T>,
  numeric types (f16, i8, u8, etc.)
- **MIR infrastructure**: CFG formalization, phi nodes, use-def chains, dominator tree
- **TCO**: Tail call optimization with conditional branch support
- **Shape inference**: E0500/E0501/E0502 error codes
- **ML types**: DType, QuantizedTensor, MaskTensor, SparseMatrix methods

## New Folder Structure

```
tests/
  audit_tests/           — Deterministic audit tests (16 files)
  prop_tests/            — Property-based tests (3 suites, 9 tests)
    mod.rs
    parser_props.rs
    type_checker_props.rs
    round_trip_props.rs
  fixtures/              — End-to-end fixtures (10 fixtures)
    runner.rs
    README.md
    basic/
    numeric/
    for_loops/
    closures/
    tco/
    fstring/
    enums/
    structs/
    error_cases/

crates/cjc-runtime/src/  — Split into 12 submodules + 5 existing
  lib.rs                 — ~400 lines (re-exports + tests)
  buffer.rs
  tensor.rs
  scratchpad.rs
  aligned_pool.rs
  kernel_bridge.rs
  paged_kv.rs
  gc.rs
  sparse.rs
  det_map.rs
  linalg.rs
  value.rs
  error.rs
  accumulator.rs         (pre-existing)
  complex.rs             (pre-existing)
  dispatch.rs            (pre-existing)
  f16.rs                 (pre-existing)
  quantized.rs           (pre-existing)

docs/
  adr/                   — Architecture Decision Records
  changes/               — Change logs
    p1_correctness_types.md  (this file)
```

## How to Run

### Fixture tests
```bash
cargo test --test fixtures
# With output visible:
cargo test --test fixtures -- --nocapture
# Update/create golden files:
CJC_FIXTURE_UPDATE=1 cargo test --test fixtures
```

### Audit tests
```bash
cargo test --test test_audit
# Specific suite:
cargo test --test test_audit matmul_path
```

### Property-based tests
```bash
cargo test --test prop_tests
# With deterministic seed (CI-friendly):
PROPTEST_MAX_SHRINK_ITERS=0 cargo test --test prop_tests
```

### Full regression
```bash
cargo test --workspace
```

## Determinism Guarantees

- **Seed-based RNG**: All randomized operations use `cjc_repro::Rng` with explicit seeds
- **Kahan summation**: Float reductions use `KahanAccumulatorF64` for stable sums
- **Deterministic maps**: `DetMap` uses MurmurHash3 with insertion-order iteration
- **Parallel matmul**: (When rayon feature is enabled) Uses fixed tiling and stable reduction trees

## Test Results

```
Total: 2080 passed, 0 failed, ~9 ignored (doc-tests only)
```

## Known Limitations / TODO for P2/P3

- **i8/i16/i32/u16/u32/u64/u128/i128**: Numeric type variants exist in the type system but not all have full runtime arithmetic
- **Complex<T>**: `complex.rs` module exists, integration with interpreter is partial
- **Set<T>/Queue<T>**: Type variants added but no runtime methods
- **Arc/Mutex**: Not implemented (no multi-threading in interpreter)
- **Json type**: Not implemented
- **SparseMatrix full API**: Only basic construction + shape + dense multiply
- **Full shape inference**: E0500/E0501/E0502 diagnostics scaffolded
- **Parallel matmul with rayon**: ADR exists, implementation pending

## Risk Register (Top 5)

1. **Runtime split field visibility**: Some struct fields changed from private to `pub(crate)` to support cross-module `impl` blocks. This is intentional but expands internal API surface.
2. **Error fixture fragility**: The `error_cases/type_error.stderr` fixture checks for substring presence in error messages, which may break if diagnostic wording changes.
3. **Proptest reproducibility**: FileFailurePersistence warnings appear because the test binary doesn't have a sibling `lib.rs`. Shrink cases still work but aren't persisted to disk automatically.
4. **Float golden file precision**: Some golden files contain floating-point representations (e.g., `5.140000000000001`) that are platform-dependent. CRLF normalization handles Windows, but float formatting differences could cause CI failures on different platforms.
5. **Large test suite runtime**: The full `cargo test --workspace` takes ~90 seconds due to property tests and the 100K-iteration TCO stress test.
