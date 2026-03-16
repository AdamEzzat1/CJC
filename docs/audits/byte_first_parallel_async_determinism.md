# Byte-First Parallel/Async Determinism Audit

## Audit Date: 2026-03-15

## Scope

Review all parallelism, async, and concurrency in CJC to verify deterministic execution regardless of thread count, scheduling order, or platform.

## Parallel Code Paths

### 1. `tensor.rs::matmul_parallel_mode_a()` (rayon)

**Location:** `crates/cjc-runtime/src/tensor.rs`
**Feature gate:** `#[cfg(feature = "parallel")]` (default on)
**Pattern:** `par_chunks_mut` over output buffer

**Analysis:**
- Each output element `C[i][j]` is computed by exactly ONE thread
- No cross-thread reduction or shared mutable state
- Each element uses its own `BinnedAccumulatorF64` instance
- **Result:** Deterministic regardless of thread count

### 2. `tensor_simd.rs::simd_binop_parallel()` (rayon)

**Location:** `crates/cjc-runtime/src/tensor_simd.rs`
**Feature gate:** `#[cfg(feature = "parallel")]`
**Pattern:** `par_chunks_mut` for element-wise operations

**Analysis:**
- Element-wise ops (add, sub, mul, div) are independent per element
- Each output element depends only on corresponding input elements
- No reduction, no accumulation across elements
- **Result:** Deterministic by construction

### 3. No Other Parallel Code

**Verification:** `grep -r "par_iter\|par_chunks\|rayon\|thread::spawn\|tokio\|async fn" crates/` shows:
- Only `rayon` usage in `tensor.rs` and `tensor_simd.rs` (as described above)
- Zero `thread::spawn` calls
- Zero `tokio` or async runtime usage
- Zero lock-based synchronization (`Mutex`, `RwLock`, etc.)

## HashMap/HashSet Audit

**Verification:** `grep -r "use std::collections::HashMap" crates/` returns zero matches.

All map-like data structures use deterministic alternatives:
- `BTreeMap`/`BTreeSet` for compiler internals (scopes, type environments, function registries)
- `DetMap` for runtime CJC `Map` type (MurmurHash3, fixed seed, insertion-order iteration)

## GC/Allocation Determinism

- **Mark-sweep GC removed.** Replaced with `ObjectSlab` (RC-based, deterministic index allocation)
- **No finalizers** that could run in non-deterministic order
- **COW buffers** (`Rc<RefCell<Vec<T>>>`) use deterministic reference counting

## RNG Thread Safety

- `SplitMix64` is NOT thread-safe (single-threaded by design)
- RNG state is threaded explicitly through the interpreter seed parameter
- `Rng::fork()` produces deterministic child RNG from parent state
- No global mutable RNG state

## SIMD Constraints

- **No FMA:** Fused multiply-add is prohibited (would produce different results on different hardware)
- **Deterministic lane operations:** SIMD operations are element-wise with no cross-lane dependencies
- **Platform-independent:** SIMD is auto-vectorized by LLVM, not manual intrinsics

## Verification Tests

All tests in `tests/byte_first/test_parallel_determinism.rs`:

| Test | What It Verifies |
|------|-----------------|
| `matmul_repeated_runs_identical` | 5 consecutive matmul runs produce bit-identical results |
| `element_wise_ops_deterministic` | Complex element-wise chain (add, sub, mul) on 1000-element tensors |
| `simd_binop_large_tensor` | 200x200 tensor addition produces identical results |
| `eval_mir_parity_tensor_ops` | Eval and MIR produce identical matmul results |
| `tiled_matmul_deterministic` | Tiled matmul with tile_size=4 is deterministic |
| `no_hashmap_in_value_types` | Struct fields use BTreeMap (sorted order) |

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Rayon thread scheduling | None | Each output element computed by single thread |
| HashMap iteration order | None | Zero HashMap usage in codebase |
| GC non-determinism | None | GC replaced with RC-based ObjectSlab |
| SIMD platform differences | None | No FMA, no manual intrinsics |
| Global mutable state | None | All state threaded explicitly |

## Conclusion

CJC has exactly 2 parallel code paths, both provably deterministic by design (disjoint output elements, no cross-thread reduction). Zero HashMap usage. No async/await. No global mutable state. The architecture is determinism-safe.
