> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# CJC v0.2 — Performance & Scale Improvement Suite
## Stacked Role Group: Speed + Memory + Scale (Determinism-First)

> **Prime Directive**: Numerical accuracy and run-to-run determinism are NON-NEGOTIABLE.
> Every optimization MUST produce bit-identical `snap_hash` outputs for the same seed.
> The 22-test Determinism@Scale benchmark suite is the regression gate — zero failures allowed.
> If an optimization changes ANY hash, it is rejected.

---

## Current Baseline (v0.1 Release Build, seed 42)

| Benchmark | Data Size | Time (ms) | GC Live | Grade |
|-----------|-----------|-----------|---------|-------|
| Pipeline | 1M elements (1000×1000) | ~120 | 0 | B- |
| NN Deep | 50 layers, width 64, batch 4 | ~6 | 0 | B |
| Seed Stress | 10× 32×32 matmul chain | ~2 | 0 | B+ |
| GC Boundary | 60K allocs + 32×32 ref | ~310 | 60,000 | B |
| Primitives | 46 builtins | ~0.6 | 0 | B+ |

**Targets**: Pipeline < 30ms, NN Deep < 2ms, GC Boundary < 100ms.
**Comparison**: NumPy ~10-20ms for 1M pipeline, Julia ~15-30ms.

---

## Codebase Facts (From Deep-Dive Research)

### Tensor Internals
- **Storage**: `Buffer<f64>` = `Rc<RefCell<Vec<f64>>>` with COW semantics
- **Layout**: Row-major, contiguous, `shape: Vec<usize>`, `strides: Vec<usize>`
- **Matmul**: Three-path router in `tensor.rs:584-623`:
  - Sequential (default): naive i,j,k triple loop with per-element Kahan summation
  - Tiled (dims ≥ 64): 64×64 tiles in `tensor_tiled.rs:47-90`, naive accumulation (no Kahan)
  - Parallel (feature="parallel", dims ≥ 256): rayon row-parallel, Kahan per element
- **Element-wise binop** (`tensor.rs:307-361`): fast path (same shape, contiguous) zips slices; broadcast path creates stride-0 views then iterates with per-element stride arithmetic
- **Unary map** (`tensor.rs:434-441`): calls `to_vec()` (materializes), then `iter().map().collect()`
- **broadcast/broadcast2** (`builtins.rs:2307-2373`): dispatch through `Box<dyn Fn>` for unary, named method calls for binary

### Memory Architecture
- **GcHeap**: RC-backed `ObjectSlab` (Vec of `Option<Rc<RefCell<Box<dyn Any>>>>`) with LIFO free-list. `gc_collect()` is a **no-op** (just increments counter).
- **FrameArena**: 4KB page bump allocator for non-escaping values, bulk reset on return.
- **Escape analysis** (`cjc-mir/src/escape.rs`): routes allocations to Stack/Arena/Rc based on escape reason.
- **array_push**: O(n) unconditional clone — `(**a).clone()` + push + `Rc::new()`. No COW despite Rc wrapper.
- **Buffer::as_slice()**: Returns `.borrow().clone()` — O(n) clone of entire Vec on every call.
- **No peak RSS tracking** anywhere in codebase.

### MIR Optimizer
- **Passes** (`optimize.rs:27-55`): CF → SR → DCE → CSE → LICM → CF (second pass)
- **SSA optimizer** (`ssa_optimize.rs`): SCCP + DCE + strength reduction + CFG cleanup
- **Constraint**: Bit-identical results, no float reassociation, side effects preserved
- **`--mir-opt` flag**: Off by default, enables optimization before MIR execution

### Parity Gaps
- **String builtins in MIR**: 14 `str_*` functions registered in `is_known_builtin` but never dispatched. Eval routes to `tidy_dispatch::dispatch_tidy_builtin()` (line 1817); MIR executor has NO equivalent call.
- **Root cause**: MIR dispatch chain tries shared `dispatch_builtin()` → CSV → GradGraph → error. Missing: `dispatch_tidy_builtin()` call.

### Scale Limitations
- **matmul**: 2D only (`if self.ndim() != 2 { return Err(...) }`)
- **sum_axis**: 2D only, axis 0 or 1
- **transpose**: 2D only (separate `transpose_last_two()` exists for N-D)
- **Tensor infrastructure**: N-D capable (shape/strides support arbitrary rank), but ops are artificially restricted

### Dependencies
- **Zero external deps** (all 15 crates internal). `rayon` is optional behind `feature = "parallel"`.
- **No SIMD**, no `#[cfg(target_arch)]`, no `#[cfg(target_feature)]`, no `#[inline(always)]` on hot paths.

---

# ROLE 1 — SIMD Engineer
**Objective**: Add deterministic SIMD acceleration to tensor hot paths.
**Constraint**: Must produce bit-identical results to scalar code for every input.

## Scope

### 1a. Matmul SIMD Micro-Kernel

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 626-640, sequential matmul)

Current sequential matmul is a naive i,j,k loop with Kahan accumulation:
```rust
for i in 0..m {
    for j in 0..n {
        let mut acc = KahanAccumulatorF64::new();
        for p in 0..k {
            acc.add(a[i * k + p] * b[p * n + j]);
        }
        result[i * n + j] = acc.finalize();
    }
}
```

**Problems**:
1. Inner loop over `p` accesses `b[p * n + j]` — stride-n access, terrible cache locality
2. Kahan per-element adds ~3 extra FLOPs per multiply-accumulate
3. No SIMD: processes one f64 at a time

**Required Changes**:
- Transpose B before multiply (or use tiled approach) so inner loop is contiguous
- Add `#[cfg(target_arch = "x86_64")]` SIMD path using `std::arch::x86_64`
- Use `_mm256_fmadd_pd` (4-wide f64 FMA) for the inner dot product
- **Determinism rule**: SIMD reduction must use the same accumulation ORDER as scalar.
  Option A: Process 4 elements at a time, horizontal sum at end (different order — REJECTED unless proven bit-identical).
  Option B: Use Kahan accumulation on the 4-wide SIMD lanes independently, reduce at end with Kahan (preserves order within lanes, acceptable if cross-lane sum is deterministic).
  Option C: Use compensated pairwise summation (proven bit-identical to sequential for IEEE 754).
- **Fallback**: Keep scalar path as `#[cfg(not(...))]` default — MUST produce identical output.

**Verification**: Run `bench_primitive_coverage.cjc` — golden hash `e5e459f5...` must not change.

### 1b. Element-wise SIMD

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 312-322, fast path)

Current fast path:
```rust
let data: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();
```

**Required Changes**:
- For known ops (add, sub, mul, div): use `_mm256_add_pd` / `_mm256_mul_pd` etc.
- Process 4 elements per iteration, handle tail with scalar loop
- For closures (arbitrary `op`): keep scalar path (can't SIMD arbitrary closures)
- **Determinism**: Element-wise ops are embarrassingly parallel — SIMD produces bit-identical results since each element is independent.

### 1c. Unary Broadcast SIMD

**File**: `crates/cjc-runtime/src/builtins.rs` (lines 2307-2344)

Current path: `Box<dyn Fn>` → `tensor.map(f)` → `to_vec()` + `iter().map().collect()`

**Required Changes**:
- For known functions (exp, sqrt, abs, neg, relu): implement SIMD versions
- `relu`: `_mm256_max_pd(x, zero)` — trivially vectorized
- `abs`: `_mm256_and_pd(x, sign_mask)` — bit manipulation
- `neg`: `_mm256_xor_pd(x, sign_bit)` — bit flip
- `sqrt`: `_mm256_sqrt_pd(x)` — hardware instruction
- `exp`, `sin`, `cos`, `sigmoid`, `tanh`: Use polynomial approximations OR keep scalar (transcendentals are harder to vectorize deterministically).
- **Determinism**: Hardware `sqrt` is IEEE 754 mandated — bit-identical. `exp`/`sin` with polynomial approximations may differ — ONLY vectorize if proven bit-identical, otherwise keep scalar.

### 1d. Feature Flag Design

- New feature: `simd` (default-enabled on x86_64)
- `#[cfg(all(target_arch = "x86_64", feature = "simd"))]` gates all SIMD paths
- Runtime detection: `is_x86_feature_detected!("avx2")` + `is_x86_feature_detected!("fma")`
- Graceful fallback to scalar on older CPUs

### Deliverables
- [ ] SIMD matmul micro-kernel (AVX2 + FMA, 4-wide f64)
- [ ] SIMD element-wise add/sub/mul/div
- [ ] SIMD unary relu/abs/neg/sqrt
- [ ] Feature flag `simd` with runtime detection
- [ ] Benchmark: matmul 1000×1000 before/after (target: 3-5× speedup)
- [ ] Determinism gate: ALL 22 benchmark tests pass, golden hash unchanged

---

# ROLE 2 — Threading Engineer
**Objective**: Add deterministic parallelism to tensor operations.
**Constraint**: Parallel execution must produce bit-identical results to sequential.

## Scope

### 2a. Activate and Improve Parallel Matmul

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 666-688)

The parallel matmul path ALREADY EXISTS behind `feature = "parallel"`:
```rust
#[cfg(feature = "parallel")]
fn matmul_parallel_mode_a(a, b, m, n, k) {
    result.par_chunks_mut(n).enumerate().for_each(|(i, row)| { ... });
}
```

**Current issues**:
1. Feature is not default-enabled
2. Threshold is 256 — should be tunable
3. Row-parallel only — not optimal for tall-skinny or wide-short matrices
4. Each thread still uses scalar Kahan — should use SIMD within each thread

**Required Changes**:
- Enable `parallel` feature by default (or make it opt-out)
- Add 2D tiling: partition output into tiles, each tile assigned to one thread
- Combine with SIMD: each thread uses AVX2 micro-kernel on its tile
- **Determinism rule**: Each output element `C[i,j]` must be computed by EXACTLY one thread, with a FIXED reduction order. No atomics, no lock-free aggregation across threads.
- The existing row-parallel approach is already deterministic (each row computed independently).

### 2b. Parallel Element-wise Operations

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 312-322)

For large tensors (>100K elements), element-wise ops are embarrassingly parallel.

**Required Changes**:
- Add parallel fast path: `rayon::par_iter().map().collect()` for contiguous same-shape ops
- Threshold: only parallelize above 100K elements (below that, thread overhead dominates)
- **Determinism**: Element-wise ops are independent — parallel produces identical results.

### 2c. Parallel RNG Generation

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 68-77, `Tensor::randn`)

Current: sequential `(0..numel).map(|_| rng.next_normal_f64()).collect()`

**CANNOT simply parallelize** — RNG state is sequential. Options:
- **Option A (Recommended)**: Fork RNG per chunk using `rng.fork()` (SplitMix64 supports deterministic forking). Assign chunk i to fork i. Same fork sequence → same output.
- **Option B**: Pre-generate all uniform pairs sequentially, then Box-Muller transform in parallel.
- **Determinism**: Option A requires fork-determinism proof. Option B is trivially deterministic.

### 2d. Thread Pool Configuration

- Use rayon with fixed thread count (default: num_cpus, overridable via `CJC_THREADS` env var)
- Thread count must be deterministic (not auto-detected differently across runs)
- **Determinism rule**: Pin thread count at program start, never change during execution.

### Deliverables
- [ ] Enable `parallel` feature by default
- [ ] Parallel matmul with 2D tiling + SIMD inner kernel
- [ ] Parallel element-wise for large tensors (>100K threshold)
- [ ] Parallel RNG via fork-deterministic chunking
- [ ] Thread count pinning (env var `CJC_THREADS`)
- [ ] Benchmark: Pipeline 1M before/after (target: 2-4× speedup on 4+ cores)
- [ ] Determinism gate: ALL 22 benchmark tests pass, golden hash unchanged

---

# ROLE 3 — Memory Engineer
**Objective**: Reduce allocation overhead, add RSS tracking, fix array_push.
**Constraint**: No changes to observable program behavior or determinism.

## Scope

### 3a. Fix `Buffer::as_slice()` — Eliminate O(n) Clone

**File**: `crates/cjc-runtime/src/buffer.rs` (lines 67-69)

Current: `pub fn as_slice(&self) -> Vec<T> { self.inner.borrow().clone() }`

This clones the ENTIRE Vec on every call. Called from:
- `elementwise_binop()` fast path (tensor.rs:313-314) — 2 clones per binop
- `to_vec()` (tensor.rs:248-280) — 1 clone per materialization

**Fix**: Return a borrow guard instead of cloning:
```rust
pub fn as_ref(&self) -> Ref<Vec<T>> { self.inner.borrow() }
```
Then callers operate on `&[T]` slices without copying. This requires lifetime changes but eliminates the two largest allocation sources in element-wise ops.

**Impact**: Every `broadcast2("add", ...)` call currently allocates 3× the tensor size (2 input clones + 1 output). Fix reduces to 1× (output only).

### 3b. Fix `array_push` — Use Rc::make_mut for COW

**File**: `crates/cjc-runtime/src/builtins.rs` (lines 2125-2131)

Current: `let arr = match &args[0] { Value::Array(a) => (**a).clone(), ... };`

Always clones the entire Vec even when refcount is 1.

**Fix**:
```rust
"array_push" => {
    let mut arr_rc = match &args[0] { Value::Array(a) => Rc::clone(a), ... };
    Rc::make_mut(&mut arr_rc).push(args[1].clone());
    Ok(Some(Value::Array(arr_rc)))
}
```

`Rc::make_mut()` only clones if refcount > 1. For the common pattern `arr = array_push(arr, val)` where the old binding is immediately overwritten, refcount will be 1 → zero-copy push.

**Impact**: Turns O(n) per push into amortized O(1) for the common single-owner case.

### 3c. Broadcast Fusion — Eliminate Intermediate Tensors

**Current problem**: `broadcast2("mul", broadcast2("add", a, b), c)` creates:
1. Intermediate tensor for `add(a, b)` — O(n) alloc
2. Final tensor for `mul(result, c)` — O(n) alloc

**Approach**: Lazy tensor evaluation / fused kernels:
- Option A (Simple): Add fused builtins like `broadcast_fma(a, b, c)` = a*b+c in one pass
- Option B (Complex): Lazy tensor graph — record ops, fuse before materialization
- **Recommended**: Option A first (most bang for buck), Option B for v0.3

### 3d. Peak RSS Tracking

**No peak RSS tracking exists anywhere in the codebase.**

**Required**: Platform-specific memory queries:
```rust
#[cfg(target_os = "windows")]
fn peak_rss_kb() -> u64 {
    use std::mem::MaybeUninit;
    // GetProcessMemoryInfo → PeakWorkingSetSize
}
#[cfg(target_os = "linux")]
fn peak_rss_kb() -> u64 {
    // Read /proc/self/status → VmHWM
}
#[cfg(target_os = "macos")]
fn peak_rss_kb() -> u64 {
    // getrusage(RUSAGE_SELF) → ru_maxrss
}
```

Expose as CJC builtin: `peak_rss()` → returns i64 (kilobytes).
Add to benchmark output protocol: `BENCH:name:seed:time_ms:gc_live:rss_kb:out_hash`

### 3e. Tensor Pool / Scratch Buffer Reuse

For repeated operations (e.g., 50-layer NN forward pass), the same tensor sizes are allocated and freed every iteration.

**Approach**: Thread-local scratch buffer pool keyed by `(numel,)`:
- On allocation: check pool for matching-size buffer, reuse if available
- On drop: return to pool instead of deallocating
- **Determinism**: Pool reuse doesn't affect computed values (only memory addresses change)
- Cap pool size to prevent unbounded growth

### Deliverables
- [ ] Fix `Buffer::as_slice()` to return borrow guard
- [ ] Fix `array_push` with `Rc::make_mut()`
- [ ] Add fused `broadcast_fma(a, b, c)` builtin
- [ ] Add `peak_rss()` builtin (Windows + Linux + macOS)
- [ ] Update BENCH protocol to include RSS
- [ ] Benchmark: Pipeline 1M allocation count before/after (target: 3× reduction)
- [ ] Determinism gate: ALL 22 benchmark tests pass, golden hash unchanged

---

# ROLE 4 — Scale Engineer
**Objective**: Extend tensor operations to N-D, larger sizes, and wider NNs.
**Constraint**: All existing 2D behavior must be preserved exactly.

## Scope

### 4a. N-D Matmul (Batch Matmul)

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 584-625)

Current: `if self.ndim() != 2 || other.ndim() != 2 { return Err(...) }`

**Required**: Support batch dimensions. For tensors `A[..., M, K]` and `B[..., K, N]`:
- Batch dimensions broadcast following NumPy rules
- Inner two dimensions do standard matmul
- Result shape: `broadcast(A.shape[:-2], B.shape[:-2]) + [M, N]`

**Implementation**:
1. Extract batch dims, validate inner dims (M×K @ K×N)
2. Broadcast batch dims to common shape
3. Iterate over batch indices, call existing 2D matmul for each slice
4. **Determinism**: Batch iteration in fixed index order, each slice uses same matmul path

### 4b. N-D sum_axis

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 486-521)

Current: 2D only, axis 0 or 1.

**Required**: Reduce along arbitrary axis for N-D tensors.

**Implementation**:
1. Validate axis < ndim
2. Compute output shape (remove axis dimension)
3. Iterate over output indices, sum over the reduced axis
4. Use Kahan accumulation for the reduction (deterministic)
5. **Determinism**: Fixed iteration order over output indices, Kahan reduction

### 4c. N-D Transpose (Permutation)

**File**: `crates/cjc-runtime/src/tensor.rs` (lines 531-541)

Current: 2D transpose only.

**Required**: `transpose(axes)` where `axes` is a permutation of `[0, 1, ..., ndim-1]`.
Default (no args): reverse all axes (current 2D behavior preserved).

**Implementation**: Zero-copy — just permute shape and strides arrays.

### 4d. Larger Benchmark Targets

Add new benchmark CJC programs for scale testing:
- `bench_pipeline_10m.cjc`: 10M elements (10000×1000 or 3162×3162)
- `bench_nn_wide.cjc`: 10 layers, width 512, batch 16 (real-world-ish NN)
- `bench_matmul_large.cjc`: 2048×2048 matmul (standard BLAS benchmark size)

These are stretch benchmarks — they may be slow initially but establish targets.

### 4e. Tensor Size Guard

For very large tensors, add a soft limit with clear error:
```
if numel > 100_000_000 {
    // 100M elements = 800MB for f64
    // Warn but allow (don't hard-fail)
}
if numel > 1_000_000_000 {
    // 1B elements = 8GB
    return Err("Tensor too large: {numel} elements ({gb:.1} GB). Use --allow-large-tensors to override.")
}
```

### Deliverables
- [ ] Batch matmul for N-D tensors
- [ ] N-D sum_axis along arbitrary axis
- [ ] N-D transpose with permutation
- [ ] 3 new scale benchmarks (10M pipeline, wide NN, large matmul)
- [ ] Tensor size guard with override flag
- [ ] Determinism gate: ALL 22 existing benchmark tests pass unchanged

---

# ROLE 5 — Parity Engineer
**Objective**: Close eval/MIR gaps and strengthen the test suite.
**Constraint**: Fix gaps without breaking either engine.

## Scope

### 5a. Wire String Builtins into MIR (P0 — Critical)

**File**: `crates/cjc-mir-exec/src/lib.rs`

**Root Cause**: MIR executor never calls `dispatch_tidy_builtin()`. Eval does (line 1817 in cjc-eval/src/lib.rs).

**Fix**: In MIR's builtin dispatch chain, after the shared `dispatch_builtin()` returns `Ok(None)`, add:
```rust
// Try tidy builtins (string functions, tidy aggregators)
match cjc_data::tidy_dispatch::dispatch_tidy_builtin(name, &args) {
    Ok(Some(value)) => return Ok(value),
    Err(msg) => return Err(MirExecError::Runtime(msg)),
    Ok(None) => {} // not a tidy builtin, continue
}
```

**Verification**: Re-add string ops to `bench_primitive_coverage.cjc`, run golden hash test.
This will change the golden hash (more primitives covered) — pin new hash.

### 5b. Add String Ops Back to Primitive Coverage

After 5a, update `bench_primitive_coverage.cjc` to include:
```
hashes = hashes + snap_hash(str_len("hello world")) + "|";
hashes = hashes + snap_hash(str_to_upper("hello")) + "|";
hashes = hashes + snap_hash(str_replace("foo bar foo", "foo", "baz")) + "|";
hashes = hashes + snap_hash(str_split("a,b,c", ",")) + "|";
hashes = hashes + snap_hash(str_trim("  hello  ")) + "|";
```

Re-pin golden master hash after this change.

### 5c. Expand Benchmark Suite

Add new tests to the Determinism@Scale suite:
- `bench_string_ops.cjc`: String manipulation determinism (50+ string ops, hash verification)
- `bench_nd_tensor.cjc`: N-D tensor operations once Role 4 ships
- `bench_matmul_sizes.cjc`: Matmul at sizes 8, 32, 64, 128, 256, 512, 1024 — verify all produce deterministic hashes

### 5d. MIR Optimizer Parity Gate

Add tests that verify `--mir-opt` produces identical output:
```rust
#[test]
fn pipeline_mir_opt_parity() {
    let src = load_cjc("bench_pipeline.cjc");
    let mir_out = run_mir(&src, 42);
    let mir_opt_out = run_mir_optimized(&src, 42);
    assert_parity(&mir_out, &mir_opt_out);
}
```

This verifies that the MIR optimizer's CF/DCE/CSE/LICM passes don't change output.

### 5e. Performance Regression Tracking

Add timing assertions (soft — warn, don't fail) to catch performance regressions:
```rust
let bench = parse_bench_lines(&out);
let ms: f64 = bench[0].time_ms.parse().unwrap();
if ms > EXPECTED_MS * 2.0 {
    eprintln!("WARNING: {} took {:.1}ms (expected ~{:.1}ms)", bench[0].name, ms, EXPECTED_MS);
}
```

### Deliverables
- [ ] Wire `dispatch_tidy_builtin()` into MIR executor
- [ ] Re-add string ops to primitive coverage, re-pin golden hash
- [ ] Add 3 new benchmark CJC programs
- [ ] Add MIR optimizer parity tests
- [ ] Add soft timing regression warnings
- [ ] Determinism gate: ALL tests pass, golden hash updated for expanded coverage

---

# Implementation Priority Order

| Priority | Role | Task | Impact | Effort |
|----------|------|------|--------|--------|
| **P0** | R5 | Wire string builtins into MIR | Fixes 14-function parity gap | 1 hour |
| **P0** | R3 | Fix `Buffer::as_slice()` clone | 3× allocation reduction | 2 hours |
| **P0** | R3 | Fix `array_push` COW | O(n) → O(1) amortized | 30 min |
| **P1** | R1 | SIMD matmul micro-kernel | 3-5× matmul speedup | 2 days |
| **P1** | R1 | SIMD element-wise ops | 2-4× broadcast speedup | 1 day |
| **P1** | R2 | Enable parallel feature by default | Free speedup on multicore | 1 hour |
| **P1** | R3 | Peak RSS tracking | Enables memory profiling | 4 hours |
| **P2** | R2 | Parallel element-wise ops | 2-4× on large tensors | 1 day |
| **P2** | R3 | Broadcast fusion (FMA) | 2× for fused pipelines | 1 day |
| **P2** | R4 | N-D matmul, sum_axis, transpose | Enables real DL workloads | 3 days |
| **P2** | R4 | Larger benchmarks (10M, wide NN) | Establishes scale targets | 4 hours |
| **P3** | R2 | Parallel RNG generation | Faster Tensor.randn for large sizes | 1 day |
| **P3** | R3 | Tensor scratch pool | Reduces GC pressure in loops | 2 days |
| **P3** | R5 | MIR optimizer parity tests | Verifies --mir-opt safety | 4 hours |

## Expected Grade After Full Implementation

| Category | Current | After P0-P1 | After P0-P2 | Target |
|----------|---------|-------------|-------------|--------|
| **Speed** | B- | B+ | A- | A |
| **Memory** | B | B+ | A- | A |
| **Scale** | B | B | B+ | A- |
| **Determinism** | A+ | A+ | A+ | A+ |
| **Parity** | A | A+ | A+ | A+ |
| **Overall** | B+ | A- | A | A |
