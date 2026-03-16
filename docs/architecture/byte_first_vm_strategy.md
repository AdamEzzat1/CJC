# Byte-First VM Strategy

## Overview

CJC's byte-first VM strategy ensures that every computation produces bit-identical results across runs, platforms, and executor backends. This document describes the VM architecture and the determinism mechanisms at each layer.

## Dual Executor Architecture

CJC has two execution backends that must produce identical results:

### cjc-eval (AST Tree-Walk Interpreter)
- Directly walks the AST
- Simpler, used for rapid prototyping
- All builtins dispatched through `cjc-runtime/src/builtins.rs`

### cjc-mir-exec (MIR Register Machine)
- Compiles AST -> HIR -> MIR -> register-based bytecode
- Supports optimization (constant folding, dead code elimination)
- Same builtins dispatch through `cjc-runtime/src/builtins.rs`

### Shared Dispatch Layer

Both executors call the same stateless builtin functions in `builtins.rs`. This is the **single source of truth** for numerical operations, ensuring parity by construction.

## Determinism Mechanisms

### 1. Floating-Point Accumulation

**Problem:** IEEE 754 floating-point addition is not associative. Different summation orders produce different results.

**Solution:** Two accumulator types, used based on context:

| Accumulator | Use Case | Mechanism |
|-------------|----------|-----------|
| `BinnedAccumulatorF64` | Unordered reductions (tensor.sum, dot, norm, stats) | 2048 exponent bins, stack-allocated, order-invariant |
| `KahanAccumulatorF64` | Sequential operations (cumsum, scan) | Compensated summation, preserves sequential order |

**Where BinnedAccumulator is used:**
- `tensor.rs::sum()` - tensor reduction
- `builtins.rs::dot()` - dot product (hardened in this audit)
- `builtins.rs::norm()` - L1/L2/Lp norms (hardened in this audit)
- `tensor.rs::matmul_parallel_mode_a()` - per-element accumulation
- `accumulator.rs::binned_sum_f64()` - convenience function

### 2. Random Number Generation

- **Algorithm:** SplitMix64 (cjc-repro)
- **Properties:** Deterministic, portable, no platform-dependent state
- **Seeding:** Explicit seed parameter on `Interpreter::new(seed)` and `run_program_with_executor(&prog, seed)`
- **Forking:** `Rng::fork()` produces deterministic child RNG from parent state

### 3. Map/Dictionary Types

- **Struct fields:** `BTreeMap<String, Value>` - alphabetical order
- **User maps:** `DetMap` - MurmurHash3 with fixed seed `0x5f3759df`, Fibonacci hashing, insertion-order iteration via `order: Vec<usize>`
- **Compiler internals:** All scope maps, function registries, type environments use `BTreeMap`/`BTreeSet`

### 4. SIMD Operations

- **Constraint:** No FMA (fused multiply-add) instructions
- **Element-wise ops:** Each output element computed independently (no cross-element dependency)
- **Parallel mode:** `par_chunks_mut` assigns disjoint output regions to threads

### 5. Serialization

- **Format:** cjc-snap binary encoding
- **NaN:** Canonicalized to single bit pattern `0x7FF8_0000_0000_0000`
- **Struct fields:** Sorted by key before encoding
- **Content hash:** SHA-256 of canonical encoding

## VM Strategy Classification

### Immediate (Stack-like) Types
Values that fit in a machine word or two: Int, Float, Bool, U8, Bf16, F16, Void.
These are copied by value with zero overhead.

### Reference-Counted (COW) Types
Values behind `Rc`: String, Array, Tuple, Bytes, Tensor.
Shared until mutation triggers `make_unique()` (copy-on-write).

### Interior-Mutable Types
Values behind `Rc<RefCell<..>>`: Bytes, Map, Buffer.
Allow in-place mutation while maintaining reference counting.

### Computed Types
Types that carry computation graphs: GradGraph, Closure.
These capture their environment at creation time.

## Parity Enforcement

Every feature must pass parity tests:
```
eval_result = Interpreter::new(seed).exec(&prog)
mir_result  = run_program_with_executor(&prog, seed)
assert_eq!(eval_result, mir_result)  // bit-identical
```

The `tests/byte_first/test_vm_runtime_parity.rs` suite validates this for: integer arithmetic, float arithmetic, string ops, booleans, comparisons, arrays, structs, functions, loops, match, tensors, and recursion.
