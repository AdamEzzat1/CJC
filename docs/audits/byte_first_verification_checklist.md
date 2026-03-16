# Byte-First Verification Checklist

## Audit Date: 2026-03-15

## Checklist

### Data Structure Determinism

- [x] Zero `HashMap` usage in production code (`grep -r "use std::collections::HashMap" crates/` = 0 matches)
- [x] Zero `HashSet` usage in production code
- [x] All struct fields stored in `BTreeMap` (alphabetical order)
- [x] All compiler scopes use `BTreeMap`/`BTreeSet`
- [x] `DetMap` uses fixed-seed MurmurHash3 (`0x5f3759df`)
- [x] `DetMap` iteration follows insertion order via `order: Vec<usize>`

### Floating-Point Determinism

- [x] `tensor.sum()` uses `binned_sum_f64` (order-invariant)
- [x] `dot()` builtin uses `binned_sum_f64` (fixed in this audit)
- [x] `norm()` builtin uses `binned_sum_f64` for all Lp variants (fixed in this audit)
- [x] `matmul_parallel_mode_a()` uses per-element `BinnedAccumulatorF64`
- [x] Tiled matmul uses fixed iteration order (single-thread per element)
- [x] GEMM inner products use fixed iteration order
- [x] Stats functions use KahanAccumulator (sequential, order-dependent by design)
- [x] No FMA instructions (would cause platform-dependent results)
- [x] NaN canonicalized in serialization (`0x7FF8_0000_0000_0000`)
- [x] Float equality uses `.to_bits()` comparison (NaN == NaN, -0 != +0)

### RNG Determinism

- [x] SplitMix64 implementation is portable (no platform-dependent state)
- [x] Seed explicitly threaded through `Interpreter::new(seed)`
- [x] `Rng::fork()` produces deterministic child RNG
- [x] No global mutable RNG state
- [x] `Tensor::randn()` uses explicit `Rng` parameter

### Parallel Determinism

- [x] Only 2 parallel code paths (tensor matmul, SIMD binop)
- [x] Both use `par_chunks_mut` with disjoint output regions
- [x] No cross-thread reduction or accumulation
- [x] No `Mutex`, `RwLock`, or lock-based synchronization
- [x] No `thread::spawn` or manual thread management
- [x] No async/await or tokio usage

### Serialization Determinism

- [x] cjc-snap uses tag-byte encoding
- [x] NaN canonicalized to single bit pattern
- [x] Struct fields sorted by key before encoding
- [x] Content-addressable SHA-256 hash
- [x] Little-endian throughout (platform-independent)

### GC/Memory Determinism

- [x] Mark-sweep GC removed, replaced with RC-based `ObjectSlab`
- [x] No finalizers
- [x] COW buffers use deterministic `Rc<RefCell<Vec<T>>>`
- [x] `make_unique()` for copy-on-write semantics

### Executor Parity

- [x] Both executors (eval, MIR) call same `builtins.rs` dispatch
- [x] Parity tests verify bit-identical results
- [x] Tested operations: int/float arithmetic, strings, booleans, comparisons, arrays, structs, functions, loops, match, tensors, recursion

## Test Coverage Summary

| Test Suite | Tests | Status |
|-----------|-------|--------|
| `test_determinism.rs` | 10 | All pass |
| `test_numerical_hardening.rs` | 12 | All pass |
| `test_parallel_determinism.rs` | 7 | All pass |
| `test_vm_runtime_parity.rs` | 14 | All pass |
| `test_type_representations.rs` | 16 | All pass |
| `test_serialization.rs` | 15 | All pass |
| **Total byte_first** | **74** | **All pass** |
| **Full workspace** | **4,898** | **All pass (0 failed, 48 ignored)** |

## Known Limitations

1. **No FMA:** Intentionally disabled for determinism. May impact performance on hardware with native FMA support.
2. **BTreeMap overhead:** O(log N) vs O(1) for scope lookups. Acceptable for CJC's scope sizes.
3. **BinnedAccumulator memory:** 2048 * 8 = 16KB per accumulator instance (stack-allocated). Acceptable for numerical workloads.
4. **Single-threaded RNG:** Cannot parallelize random number generation. Acceptable for deterministic execution.
