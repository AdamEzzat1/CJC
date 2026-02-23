# CJC Language — Optimization Analysis & Roadmap

> **Audit revision:** Post-Hardening (Phase 5) — 2025 tests passing, 0 failures
> **Auditor roles:** Role 5 (Determinism), Role 6 (Performance), Role 7 (Safety), Role 8 (Architecture)
> **Status:** READ-ONLY audit — no source modifications

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Performance Profile](#2-current-performance-profile)
3. [Completed Hardening (Phase 5 Recap)](#3-completed-hardening-phase-5-recap)
4. [Known Limitations](#4-known-limitations)
5. [P0 Priority — Correctness and Safety](#5-p0-priority--correctness-and-safety)
6. [P1 Priority — Performance Optimization](#6-p1-priority--performance-optimization)
7. [P2 Priority — Language Completeness](#7-p2-priority--language-completeness)
8. [P3 Priority — Ecosystem and Tooling](#8-p3-priority--ecosystem-and-tooling)
9. [Architecture Decision Records](#9-architecture-decision-records)
10. [Missing Data Types](#10-missing-data-types)
11. [Best-Fit Use Cases](#11-best-fit-use-cases)
12. [Honest Technical Assessment](#12-honest-technical-assessment)
13. [Test Coverage Analysis](#13-test-coverage-analysis)
14. [Roadmap Summary Table](#14-roadmap-summary-table)

---

## 1. Executive Summary

CJC (Concurrent Julia-like Compiler) is a research-grade systems programming language targeting:
- **ML inference and training infrastructure** (tensor operations, auto-diff, stable reductions)
- **Data processing pipelines** (DataFrame, streaming CSV, regex)
- **Deterministic numerical computing** (reproducible floating-point via seeded RNG + stable accumulation)
- **Memory-safe systems programming** (three-layer memory model, nogc regions)

**Current state (Post-Hardening Phase 5):**
- 2025 tests passing, 0 failures
- Full interpreter pipeline from source → MIR → Value
- Production-hardened: span-aware diagnostics, exhaustiveness enforcement, trait resolution, CFG, allocation-free matmul
- Remaining gaps: backend codegen (LLVM/WASM), parallel execution, SSA, full generic monomorphization

**Honest grade:** B / 7.5 out of 10 (upgraded from B- / 7.2 after hardening work)

---

## 2. Current Performance Profile

### Execution Model: Tree-Walking Interpreter

CJC currently executes as a **tree-walking interpreter** over MIR. This means:

| Characteristic | Value |
|----------------|-------|
| Startup cost | Low (parse + lower, no compile step) |
| Numeric throughput | Moderate (host Rust speed for `Tensor` ops, interpreter overhead for control flow) |
| Allocation model | COW Buffer (Layer 2) — amortized, low allocation pressure |
| Matmul (post-hardening) | **Zero allocations per dot product** — O(k) accumulator |
| GC overhead | Triggered on `class` operations only — most programs never hit GC |
| Recursion depth | Stack-limited (Rust call stack) — no tail-call optimization |

### Bottlenecks

1. **HashMap scope lookup** — each variable lookup is `O(1)` average but with constant factor overhead from scope stack traversal and `HashMap` hashing. Deep call stacks cause multiple scope traversals.

2. **`Vec<Value>` cloning** — `Array` and `Tuple` values clone their entire element `Vec` on every assignment (no COW for `Vec<Value>`). Only `Buffer<T>` and `Tensor` have COW semantics.

3. **`MirBody` re-traversal** — each function call re-clones the `MirBody` from the interpreter's `HashMap<String, MirFunction>`. No function body interning.

4. **Pattern matching** — sequential arm-by-arm matching (no decision tree). For enums with many variants, O(n) in arm count.

5. **Lambda call overhead** — each closure call prepends captured values to args, creating a new `Vec<Value>` per invocation.

---

## 3. Completed Hardening (Phase 5 Recap)

Production Hardening brought the codebase from 1966 passing tests to **2025 passing tests** (+59 new hardening tests), with four P0 fixes:

### Phase 1: Span-Aware Unification
- **Added:** `unify_spanned(a, b, subst, span, diag) -> Type`
- **Effect:** Type errors now carry source location; IDE-quality error messages possible
- **Error code:** `E0100` with byte span
- **Backward compat:** Existing `unify()` unchanged; `unify_spanned` wraps it

### Phase 2: Match Exhaustiveness as Compile-Time Error
- **Fixed:** Bare enum variant identifiers (e.g., `Red`, `Green`) were parsed as `PatternKind::Binding`, causing the exhaustiveness checker to treat them as wildcards (suppressing E0130)
- **Fix:** Pre-compute all enum variant names; in `Binding` arm, check if name matches a variant name — if yes, treat as variant coverage, not wildcard
- **Added:** `type_check_program()` and `run_program_type_checked()` for gated execution
- **Error code:** `E0130` with span

### Phase 3: Trait Resolution Enforcement
- **Added:** `TypeChecker::check_impl()` method
- **Error codes:** E0200 (undefined trait), E0201 (duplicate impl), E0202 (missing method)
- **Effect:** `impl Type : Trait` now validated; trait contracts enforced

### Phase 4: MIR CFG
- **Added:** `cjc_mir::cfg` module — `MirCfg`, `BasicBlock`, `Terminator`, `CfgBuilder`
- **Effect:** Enables future SSA, live variable analysis, LICM, register allocation
- **Properties:** Deterministic, single-pass DFS builder

### Phase 5: Allocation-Free Matmul Inner Loop
- **Changed:** All 5 matmul/linear inner loops from `Vec<f64>` + `kahan_sum_f64` to `KahanAccumulatorF64`
- **Effect:** Zero heap allocations per dot product; bit-identical results preserved
- **Exported:** `KahanAccumulatorF32/F64` from `cjc-repro` root

---

## 4. Known Limitations

### Parser
| Limitation | Impact | Priority |
|------------|--------|----------|
| `impl Trait for Type` syntax not parsed | Cannot use standard Rust-style trait impls | P1 |
| No `async`/`await` | No async computation model | P2 |
| No `const` expressions | No compile-time evaluation | P2 |
| No macro system | No metaprogramming | P3 |
| No module path resolution | Imports are best-effort | P2 |

### Type System
| Limitation | Impact | Priority |
|------------|--------|----------|
| Generic monomorphization incomplete | Generic code paths not fully instantiated | P0 |
| Type inference partial | Requires explicit annotations in many places | P1 |
| No higher-kinded types | Cannot express `Functor`, `Monad`, etc. | P3 |
| No dependent types | Shape constraints in `Tensor<T, [M, N]>` not enforced at runtime | P1 |
| No type-level arithmetic | Cannot compute `M + N` in types | P2 |
| Trait bounds not checked at call sites | `<T: Numeric>` bounds declared but not enforced | P0 |

### Execution
| Limitation | Impact | Priority |
|------------|--------|----------|
| Tree-walking interpreter only | 10-100x slower than native code | P1 |
| No tail-call optimization | Recursive programs stack-overflow on deep recursion | P1 |
| No parallel execution | `BinnedAccumulator` parallel-ready but no executor parallelism | P1 |
| No JIT compilation | Can't optimize hot paths at runtime | P2 |
| No LLVM/WASM backend | Cannot produce standalone executables | P2 |

### Memory
| Limitation | Impact | Priority |
|------------|--------|----------|
| `Vec<Value>` Array/Tuple has no COW | Deep cloning on every assignment | P1 |
| GC not incremental | STW collection pause on large class graphs | P2 |
| No arena allocator | Many small `HashMap` allocations per scope | P2 |

---

## 5. P0 Priority — Correctness and Safety

These are blocking issues that affect correctness of existing programs.

### P0-1: Complete Generic Monomorphization

**Problem:** The `cjc-mir::monomorph` module exists but generic functions (e.g., `fn dot<T: Float>(...)`) are not fully instantiated. At the interpreter level, type variables are treated as wildcards, which works for most programs but fails for generic functions that branch on type kind (e.g., `f32` vs `f64` dispatch).

**Fix plan:**
1. Extend `HirToMir` to track pending monomorphizations
2. When a generic call site is encountered with concrete type arguments, add to a work queue
3. Instantiate the generic body with type substitution applied
4. Name-mangle the instance: `dot_f32`, `dot_f64`, etc.

**Test target:** Generic functions with type-specific behavior produce different results for different type instantiations.

### P0-2: Trait Bounds Enforcement at Call Sites

**Problem:** `fn sum<T: Numeric>(...)` declares a bound but `check_fn_call` doesn't verify that the actual argument type satisfies `T: Numeric`.

**Fix plan:**
1. When a generic call `sum(3.14)` is encountered, look up `T: Numeric` bound
2. Verify `f64` satisfies `Numeric` (via `env.trait_impls`)
3. Emit `E0300: trait bound not satisfied` if check fails

### P0-3: Mutable Binding Enforcement

**Problem:** `let x = 5; x = 6;` should be rejected (assigning to non-`mut` binding), but the interpreter currently allows it.

**Fix plan:**
1. Track `mutable: bool` in `TypeEnv` per binding
2. In `check_assign`, verify target binding is `mutable: true`
3. Emit `E0150: cannot assign to immutable binding`

---

## 6. P1 Priority — Performance Optimization

### P1-1: Native Code Generation (LLVM/Cranelift)

**Problem:** The tree-walking interpreter is 10-100x slower than compiled native code. For ML training loops that execute millions of iterations, this is a hard bottleneck.

**Approach:**
1. The CFG (`cjc_mir::cfg`) is now available — this is the critical prerequisite
2. Add SSA construction pass (dominators + φ-functions) on top of the CFG
3. Emit LLVM IR or Cranelift IR from SSA-form MIR
4. Tensor operations map to BLAS/LAPACK calls or SIMD intrinsics

**Effort estimate:** Large (6-12 weeks). The CFG work done in Phase 4 was specifically to unblock this.

### P1-2: Tail-Call Optimization

**Problem:** `fib(10000)` overflows the Rust call stack. CJC has no TCO.

**Fix plan (interpreter-level):**
1. Detect tail calls: `return f(args)` where `f` is the current function
2. Instead of recursive `call_function`, reuse current frame by resetting scope and jumping to function start
3. At CFG level, tail-call returns can be compiled to `jmp` instead of `call`

### P1-3: `Vec<Value>` COW for Array/Tuple

**Problem:** `Array(Vec<Value>)` deep-clones on every assignment. A 1000-element array used as a function argument causes 1000 `Value::clone()` calls.

**Fix plan:**
1. Replace `Vec<Value>` with `Buffer<Value>` (already COW-capable)
2. On array index assignment, `make_unique()` to trigger COW

**Impact:** Eliminates O(n) clone cost for array-heavy programs.

### P1-4: Scope Stack Optimization

**Problem:** Each variable lookup traverses the scope stack from top (innermost) to bottom (outermost). Deep nesting causes O(depth) lookups per variable access.

**Fix plan:**
1. Flat name-to-value map with generation counters (or shadowing stack per name)
2. Or: resolve variable locations statically at MIR lowering time (SlotId approach)

### P1-5: Interned Function Bodies

**Problem:** `self.functions.get(name).cloned()` clones the entire `MirFunction` (including body) on each call. For recursive functions, this happens on every recursive call.

**Fix plan:**
1. Store `MirFunction` in `Rc<MirFunction>` → `clone()` is O(1)
2. Or: pre-index by function name at program load time

### P1-6: Parallel Matmul

**Problem:** Matmul is currently single-threaded. The `BinnedAccumulatorF64::merge` operation provides the infrastructure for parallel reduction.

**Fix plan:**
1. Partition output matrix rows across threads
2. Each thread computes its rows independently using `KahanAccumulatorF64` per dot product
3. No merge needed (partition, not reduction)
4. Or: use Rayon for work-stealing parallel row loops

**Expected speedup:** 2-8x on multi-core hardware for large matrices.

---

## 7. P2 Priority — Language Completeness

### P2-1: `impl Trait for Type` Parser Fix

**Problem:** The standard Rust-style `impl Display for Point` syntax fails to parse. Users must use `impl Point : Display` or bare `impl Point`.

**Fix:** Update `parse_impl_decl` to support both syntaxes.

### P2-2: Dependent Shape Types

**Problem:** `Tensor<f64, [M, N]>` declares shape variables but the runtime doesn't enforce that `matmul(a: Tensor<f64, [M, K]>, b: Tensor<f64, [K, N]>)` produces `Tensor<f64, [M, N]>`. Shape errors are only caught at runtime.

**Fix plan:**
1. Track shape substitutions in `TypeChecker` (`ShapeSubst` already exists)
2. At matmul call, unify shape dimension `K` — emit compile-time error if mismatch
3. Propagate `[M, N]` result shape through type checker

### P2-3: Const Expressions

CJC has no `const` keyword. This means:
- Array sizes must be literal integers
- No compile-time constants
- No `const fn` for compile-time computation

**Fix plan:** Add `const NAME: Type = expr` decls. Evaluate at parse time for literal expressions.

### P2-4: Module System

Current `import` is stub-level. No real module resolution, visibility rules, or path-based namespacing.

**Fix plan:**
1. `import std.math` → loads built-in math module into scope
2. `import my_module` → looks for `my_module.cjc` relative to source
3. Pub/priv visibility via `pub fn`, `pub struct`

### P2-5: String Interpolation

CJC has no string interpolation (f-strings). Users must concatenate manually.

```cjc
// Desired:
print(f"x = {x}, y = {y}");

// Current workaround:
print("x = " + to_string(x) + ", y = " + to_string(y));
```

### P2-6: Error Handling (`Result` / `Option` as Language Types)

`Result<T, E>` and `Option<T>` are definable as user enums but:
- `?` try operator desugars to `match` but the type checker doesn't special-case `Result`/`Option`
- No `unwrap()`, `map()`, `and_then()` builtins
- No monadic chaining

**Fix plan:** Register `Result` and `Option` as builtin enums with special type-checker handling.

---

## 8. P3 Priority — Ecosystem and Tooling

### P3-1: Language Server Protocol (LSP)

**Status:** Not implemented. The `DiagnosticBag` and span infrastructure are ready; needs an LSP wrapper.

**Components needed:**
- Go-to-definition (requires symbol table with span tracking)
- Hover types (requires type-annotated AST)
- Completion (requires scope-aware name lookup)
- Inline diagnostics (already available via `DiagnosticBag`)

### P3-2: WASM Target

Emitting WebAssembly would enable:
- Browser-based CJC execution
- Plugin systems
- Sandboxed execution

**Prerequisite:** Needs LLVM or Cranelift backend (P1-1 first).

### P3-3: Standard Library

CJC's standard library is sparse:
- `std.math`: `sqrt`, `abs`, `sin`, `cos`, `exp`, `log`, `floor`, `ceil`, `min`, `max` — registered as builtins in TypeChecker
- No `std.io` (file I/O beyond CSV)
- No `std.net` (networking)
- No `std.collections` (beyond Map<K,V> type)
- No `std.concurrent` (channels, mutexes)
- No `std.fmt` (format/print beyond `print()` builtin)

### P3-4: Debugger Integration

**Status:** No debugger. The MIR executor has `output: Vec<String>` for observing `print()` calls, but no breakpoints, watchpoints, or step-through.

### P3-5: Package Manager

**Status:** No package/dependency management system. Projects are single-file or manually managed.

---

## 9. Architecture Decision Records

### ADR-1: Tree-Form MIR + Separate CFG (Chosen)

**Decision:** Keep MIR in tree-form (`MirBody`) and derive the CFG on-demand via `CfgBuilder`.

**Rationale:**
- The interpreter can directly tree-walk `MirBody` without CFG overhead
- Analyses that need explicit control flow build `MirCfg` from `MirBody`
- No flag day — existing code unaffected
- CFG is deterministic and O(n) to build

**Alternative considered:** Replace `MirBody` with `MirCfg` directly. Rejected because the interpreter would need rewriting and the tree-form is simpler for the reference implementation.

### ADR-2: KahanAccumulatorF64 over Vec<f64> (Chosen, Phase 5)

**Decision:** Replace `Vec<f64>; kahan_sum_f64(&products)` pattern with in-place `KahanAccumulatorF64`.

**Rationale:**
- Eliminates O(k) heap allocation per dot product (k = inner matrix dimension)
- Bit-identical results (same accumulation order)
- Zero API change — same numerical behavior, less memory pressure
- Stack overhead: 2 f64 registers per dot product computation

**Trade-off:** Slightly more verbose inner loop code (explicit `acc.add()` vs iterator `.map().collect()`).

### ADR-3: Backward-Compatible `run_program` (Chosen, Phase 2)

**Decision:** `run_program()` does NOT type-check. `run_program_type_checked()` does.

**Rationale:**
- The type checker doesn't know about builtins (`Tensor.zeros`, `print`, etc.)
- Integrating type-check into `run_program` would break 3 tests in the existing suite
- Opt-in type checking via separate function preserves all existing test coverage
- Users can choose: fast (unchecked) or safe (type-checked) execution

**Consequence:** Programs with type errors still run if called via `run_program`. This is a known trade-off.

### ADR-4: SplitMix64 for RNG (Existing)

**Decision:** Use SplitMix64 algorithm for `Rng`.

**Rationale:**
- Cross-platform determinism (no OS/SIMD dependency)
- Passes statistical tests (BigCrush)
- Extremely fast: 5 operations per 64-bit output
- `fork()` provides deterministic child streams (no entropy source)

### ADR-5: Exponent-Binned Accumulator (Existing)

**Decision:** 2048-bin superaccumulator for order-invariant summation.

**Rationale:**
- Stack-allocated (2048 f64s = 16KB) — no heap allocation ever
- Provably commutative and associative: enables parallel/distributed reduction
- IEEE-754 compliant NaN/Inf handling
- Two-sum (Knuth 2Sum) merge preserves all rounding errors

**Trade-off:** 16KB stack per accumulator instance. Acceptable for function scope; not suitable for per-element accumulation in tight loops.

---

## 10. Missing Data Types

These types are absent from CJC's current type system and would add significant capability:

### Numeric Types Missing

| Type | Priority | Use Case |
|------|----------|----------|
| `i8`, `i16`, `i128`, `u16`, `u32`, `u64`, `u128` | P2 | Low-level bit manipulation, protocol buffers |
| `f16` (true IEEE 754 half) | P1 | ML model weights (inference) |
| Complex numbers (`Complex<f64>`) | P2 | Signal processing, FFT |
| Fixed-point `Q<bits, frac>` | P3 | Embedded/FPGA targets |
| `Decimal` (exact decimal) | P3 | Financial calculations |

> **Note:** `bf16` is declared in the type system but `f16` (IEEE 754 half precision, different from bf16) is missing. The `cjc-runtime/src/f16.rs` module exists but is not fully integrated.

### Structural Types Missing

| Type | Priority | Use Case |
|------|----------|----------|
| `Set<T>` | P2 | Unique element collections |
| `Queue<T>`, `Deque<T>` | P2 | FIFO data structures |
| `Heap<T>` (priority queue) | P2 | Scheduling, Dijkstra |
| `Graph<V, E>` | P3 | Graph algorithms |
| `Tree<T>` (balanced BST) | P2 | Ordered maps |
| `Range<T>` | P1 | Range iteration (partially there via `0..n`) |
| `Slice<T>` | P1 | Non-owning view over array/buffer |
| `Option<T>` (builtin) | P1 | Nullable without boxing |
| `Result<T, E>` (builtin) | P1 | Error propagation |

### Protocol/Network Types Missing

| Type | Priority | Use Case |
|------|----------|----------|
| `Socket` | P3 | Networking |
| `File` | P2 | File I/O |
| `Channel<T>` | P2 | Message passing |
| `Mutex<T>` | P2 | Concurrent access |
| `Arc<T>` | P2 | Thread-safe shared ownership |
| `Json` | P2 | JSON parsing/serialization |
| `Protobuf` | P3 | Schema-based serialization |

### ML-Specific Types Missing

| Type | Priority | Use Case |
|------|----------|----------|
| `DType` (runtime-typed tensor) | P1 | Mixed-precision training |
| `DeviceTensor` (CUDA/Metal) | P2 | GPU tensor operations |
| `SymbolicTensor` | P2 | Lazy evaluation / graph compilation |
| `QuantizedTensor<bits>` | P2 | INT8/INT4 inference |
| `MaskTensor` | P2 | Attention mask (transformer) |
| `SparseMatrix` | P1 | Graph neural networks |

> **Note:** `SparseTensor<T>` is in the type system but not implemented in the runtime.

---

## 11. Best-Fit Use Cases

Based on the current implementation, CJC is best suited for:

### ✅ Strong Fit

**1. Numerical Research Prototyping**
- Seeded RNG + Kahan/Binned accumulators → reproducible experiments
- Tensor ops with stable reductions → numerical algorithms
- Auto-diff → gradient computation
- Pattern: write experiments in CJC, reproduce exact results later

**2. ML Inference Pipelines (CPU)**
- Matmul (allocation-free post-hardening) → efficient dot products
- Softmax, sigmoid, relu, linear_layer → feedforward networks
- DataFrame + CSV → data loading
- Pattern: inference pipeline without Python overhead

**3. Data Engineering with Type Safety**
- DataFrame operations + CSV streaming → ETL pipelines
- Regex + pipe operator → text transformation DSL
- Pattern: typed, reproducible data transformations

**4. Educational Language Design**
- Well-documented three-layer memory model
- Clean MIR with CFG available for teaching compiler concepts
- Complete test suite as specification
- Pattern: compiler courses, programming language research

**5. Determinism-Critical Applications**
- Financial simulations (Kahan + seeded → reproducible results)
- Scientific computing validation (bit-identical replication)
- Pattern: anywhere where "same seed → same output" is a hard requirement

### ⚠️ Marginal Fit (needs hardening)

**6. ML Training Loops**
- Requires: JIT or native codegen (currently interpreter-only)
- Requires: GPU tensor support (currently CPU only)
- Requires: Parallel matmul (infrastructure ready, not wired up)

**7. Production Systems**
- Requires: LSP, debugger, package manager
- Requires: Module system with visibility
- Requires: Stable ABI for FFI

### ❌ Poor Fit (missing fundamentals)

**8. Web/Network Services**
- No async/await, no networking primitives, no HTTP
**9. Systems Programming**
- No unsafe, no raw pointers, no FFI
**10. Mobile/Embedded**
- No static binary, no no_std mode, no interrupt handlers

---

## 12. Honest Technical Assessment

### Grade: **B / 7.5 out of 10** (upgraded from B- after Phase 5 hardening)

### What Works Well
- **Numerical precision:** Three independent stable accumulation strategies (Kahan, Pairwise, Binned). Better than most languages out of the box.
- **Determinism story:** Comprehensive — seeded RNG, allocation-free matmul, bit-identical guarantees backed by 10+ determinism tests.
- **Diagnostics infrastructure:** `Span`-aware `DiagnosticBag` with source-location rendering, error codes, hints. Production-quality foundation.
- **Test coverage:** 2025 passing tests across all crates. Comprehensive unit + integration coverage.
- **Memory model:** Three-layer design is principled and well-implemented. COW Buffer is elegant.
- **CFG infrastructure:** Complete, deterministic, well-documented. Ready for SSA construction.

### What Needs Work
1. **No native codegen.** The interpreter is educational but not production-performant. This is the single largest gap.
2. **Generics are shallow.** Type parameters are declared but not fully instantiated. Generic code often degrades to untyped execution.
3. **Type inference is partial.** Many programs require explicit type annotations that a full HM inference engine would infer automatically.
4. **Mutable binding enforcement missing.** `let x = 5; x = 6` incorrectly succeeds.
5. **Parser limitation on `impl Trait for Type`.** Standard Rust-style impls don't parse.
6. **No standard library.** Only builtins registered in TypeChecker. No file I/O, no collections beyond `Map`.

### Comparison to Peers

| Feature | CJC | Julia | Mojo | Zig |
|---------|-----|-------|------|-----|
| Stable floating-point | ✅ Excellent | ✅ Good | ✅ Good | Manual |
| Determinism | ✅ Built-in | ⚠️ Requires care | ⚠️ Requires care | Manual |
| Native codegen | ❌ Interpreter only | ✅ JIT+LLVM | ✅ LLVM | ✅ LLVM |
| ML tensor ops | ✅ Good | ✅ Excellent | ✅ Excellent | Manual |
| Memory safety | ✅ Three-layer | GC | ✅ Owned | Manual + safety checks |
| Type inference | ⚠️ Partial | ✅ Full | ✅ Full | ⚠️ Partial |
| Ecosystem | ❌ Minimal | ✅ Large | 🔶 Growing | ✅ Growing |

---

## 13. Test Coverage Analysis

### By Crate

| Crate | Tests | Key Coverage |
|-------|-------|-------------|
| `cjc-repro` | 8 | RNG determinism, Kahan sum, pairwise |
| `cjc-runtime` | ~120 | Buffer COW, Tensor ops, matmul, GC, BinnedAccumulator |
| `cjc-types` | ~80 | Unify, unify_spanned, exhaustiveness, trait resolution |
| `cjc-mir` | ~30 | HIR→MIR lowering, CFG construction |
| `cjc-mir-exec` | ~200 | Full end-to-end execution, builtins |
| `cjc-data` | ~60 | DataFrame, CSV |
| `cjc-ad` | ~40 | DualF64, gradients |
| `cjc-diag` | ~20 | Diagnostic rendering |
| Hardening tests | 51 | H1-H6: span, exhaustiveness, traits, CFG, matmul, determinism |
| Audit tests | ~60 | Reality-check claims |
| Integration tests | ~200+ | Full source programs |

### Coverage Gaps

1. **Generic monomorphization** — no tests for `fn foo<T>()` with multiple instantiations
2. **Mutable binding enforcement** — no negative test for `let x = 5; x = 6`
3. **Large-scale matmul performance** — no benchmark tests, only correctness
4. **GC stress tests** — no tests with millions of class allocations
5. **Parallel reduction** — `BinnedAccumulator::merge` tested in unit tests but no parallel execution tests
6. **Error recovery** — `Type::Error` propagation not systematically tested
7. **Module imports** — `import` statements not tested end-to-end
8. **Regex capture groups** — only match/no-match tested

---

## 14. Roadmap Summary Table

| Priority | Item | Effort | Unblocks |
|----------|------|--------|---------|
| **P0** | Complete generic monomorphization | Large | Correct generic code |
| **P0** | Trait bound enforcement at call sites | Medium | Type soundness |
| **P0** | Mutable binding enforcement | Small | Safety correctness |
| **P1** | Native codegen (LLVM/Cranelift) | X-Large | 10-100x speedup |
| **P1** | Tail-call optimization | Medium | Deep recursion |
| **P1** | `Vec<Value>` COW | Small | Array efficiency |
| **P1** | Scope stack optimization | Medium | Variable access speed |
| **P1** | Interned function bodies | Small | Recursion efficiency |
| **P1** | Parallel matmul | Medium | Multi-core ML |
| **P1** | `f16` IEEE 754 half | Small | ML inference weights |
| **P1** | `Result<T, E>` / `Option<T>` as builtins | Medium | Ergonomic error handling |
| **P1** | `Slice<T>`, `Range<T>` | Small | Collection ergonomics |
| **P2** | `impl Trait for Type` parser | Small | Standard trait syntax |
| **P2** | Dependent shape types | Large | Tensor shape safety |
| **P2** | Module system | Large | Code organization |
| **P2** | Const expressions | Medium | Compile-time values |
| **P2** | String interpolation | Small | Developer ergonomics |
| **P2** | Set, Queue, Heap, Tree | Medium | Data structures |
| **P2** | File I/O | Medium | Real-world programs |
| **P2** | `Complex<f64>`, `DType` | Medium | Signal processing, ML |
| **P3** | LSP integration | Large | IDE experience |
| **P3** | WASM target | X-Large | Browser/plugin |
| **P3** | Package manager | X-Large | Ecosystem |
| **P3** | Debugger | Large | Developer experience |
| **P3** | Standard library | X-Large | Completeness |

### Recommended Next Sprint (Post-Phase 5)

If forced to pick the highest-leverage next items:

1. **P0-3: Mutable binding enforcement** (2 days) — small but fixes a language correctness hole
2. **P0-2: Trait bound enforcement at call sites** (1 week) — makes the trait system actually useful
3. **P1-3: `Vec<Value>` COW** (3 days) — removes the biggest allocation regression in hot paths
4. **P2-1: `impl Trait for Type` parser** (1 day) — removes the most confusing parser limitation
5. **P1-1: Begin LLVM backend scoping** (2+ months) — the game-changer, but needs proper scoping

---

*Generated by the CJC Full Repo Audit (Read-Only, Post-Hardening Phase 5)*
*Total passing tests at time of audit: **2025***
*Files modified by this audit: **0** (read-only)*
