# CJC Stage 3 Roadmap — Infrastructure Hardening & Language Completeness

**Document Version:** 1.0
**Status:** Active
**Date:** 2025-01-01
**Supersedes:** `docs/spec/phase2_core_hardening.md` (Phase 2 deferred work)

---

## Executive Summary

Stage 3 is a consolidation phase following Stage 2's core hardening. The primary objective is to close the gap between CJC's current B+ grade and production-compiler quality (A/A+) by:

1. **Infrastructure hardening** — CI/CD, fixture runner, runtime module split, ADR materialization
2. **Performance** — Vec COW, parallel matmul, scope stack optimization
3. **Language completeness** — extended numeric types, collection types, string builtins
4. **Compiler depth** — true CFG/SSA, TCO completeness, shape inference at compile time
5. **Domain capability** — ML types (DType, QuantizedTensor, MaskTensor), sparse methods

---

## Task Registry

| ID | Priority | Status | Description | Crates | ADR |
|----|----------|--------|-------------|--------|-----|
| S3-P0-01 | P0 | ✅ Done | ADR materialization (ADR-0001 through ADR-0012) | docs/ | — |
| S3-P0-02 | P0 | ✅ Done | ADR index + cross-reference graph | docs/adr/ | — |
| S3-P0-03 | P0 | ✅ Done | Error code registry | docs/spec/ | — |
| S3-P0-04 | P0 | ✅ Done | Test coverage matrix | docs/progress/ | — |
| S3-P0-05 | P0 | 🔄 WIP | Split cjc-runtime/src/lib.rs into submodules | cjc-runtime | ADR-0008 |
| S3-P0-06 | P0 | 🔄 WIP | Proptest infrastructure (tests/prop_tests/) | root | — |
| S3-P0-07 | P0 | 🔄 WIP | End-to-end fixture runner (tests/fixtures/) | root | ADR-0003 |
| S3-P1-01 | P1 | 🔄 WIP | Vec COW for Value::Array and Value::Tuple | cjc-runtime, cjc-eval, cjc-mir-exec | ADR-0009 |
| S3-P1-02 | P1 | 🔄 WIP | Parallel matmul with rayon (optional feature) | cjc-runtime | ADR-0011 |
| S3-P1-03 | P1 | 🔄 WIP | Extended numeric types (i8..u128, f16, Complex) | cjc-types, cjc-runtime | — |
| S3-P1-04 | P1 | 🔄 WIP | Structural collection types (Set<T>, Queue<T>) | cjc-runtime, cjc-types | — |
| S3-P1-05 | P1 | 🔄 WIP | Option/Result/Range/Slice as first-class Type variants | cjc-types | — |
| S3-P1-06 | P1 | 🔄 WIP | CFG phi nodes + use-def chains + dominator tree | cjc-mir | ADR-0012 |
| S3-P1-07 | P1 | 🔄 WIP | TCO extension (conditional branches, mutual recursion) | cjc-mir-exec | — |
| S3-P1-08 | P1 | 🔄 WIP | Shape inference pipeline (E0500, E0501, E0502) | cjc-types | — |
| S3-P2-01 | P2 | 🔄 WIP | DType enum + Value::DTypeVal | cjc-runtime | — |
| S3-P2-02 | P2 | 🔄 WIP | QuantizedTensor (INT8/INT4) | cjc-runtime, cjc-mir-exec | — |
| S3-P2-03 | P2 | 🔄 WIP | MaskTensor (bit-packed attention mask) | cjc-runtime, cjc-mir-exec | — |
| S3-P2-04 | P2 | 🔄 WIP | SparseTensor method dispatch (matvec, to_dense, nnz) | cjc-runtime, cjc-mir-exec | — |
| S3-P3-01 | P3 | 📋 Planned | LLVM/Cranelift native backend (subset) | new crate: cjc-codegen | — |
| S3-P3-02 | P3 | 📋 Planned | Language Server Protocol (LSP) server | new crate: cjc-lsp | — |
| S3-P3-03 | P3 | 📋 Planned | Scope stack SmallVec optimization | cjc-mir-exec | ADR-0010 |
| S3-P3-04 | P3 | 📋 Planned | W0010 lint (recursive call not in tail position) | cjc-mir-exec | — |

---

## S3-P0: Infrastructure

### S3-P0-05: cjc-runtime Submodule Split

**Goal:** Reduce `cjc-runtime/src/lib.rs` from ~4000 lines to ≤80 lines (re-exports only).

**Target layout:**
```
crates/cjc-runtime/src/
  lib.rs          — pub mod + pub use, ≤80 lines
  buffer.rs       — Buffer<T> (lines 25-121)
  tensor.rs       — Tensor struct + matmul + linalg impls
  scratchpad.rs   — Scratchpad + AlignedPool + AlignedByteSlice
  kernel.rs       — Raw kernel functions (formerly pub mod kernel)
  paged_kv.rs     — PagedKvCache + KvBlock
  gc.rs           — GcRef + GcHeap
  sparse.rs       — SparseCsr + SparseCoo
  det_map.rs      — DetMap + murmurhash3 + value_hash
  value.rs        — Bf16 + FnValue + Value enum + Display
  error.rs        — RuntimeError enum
  collections.rs  — CjcSet + CjcQueue (new)
  ml_types.rs     — DType + QuantizedTensor + MaskTensor (new)
  # existing submodules (unchanged):
  accumulator.rs, complex.rs, dispatch.rs, f16.rs, quantized.rs
```

**Success criteria:**
- `crates/cjc-runtime/src/lib.rs` ≤ 80 lines
- All public types accessible via `cjc_runtime::TypeName` (no API change)
- `cargo test --workspace` passes with 0 failures

**Regression gate:** `cargo test --workspace`

---

### S3-P0-06: Proptest Infrastructure

**Goal:** Add property-based tests for parser and type checker in `tests/prop_tests/`.

**Files created:**
- `tests/prop_tests/mod.rs`
- `tests/prop_tests/parser_props.rs` — 11 properties
- `tests/prop_tests/type_checker_props.rs` — 7 properties
- `tests/prop_tests/round_trip_props.rs` — 5 properties

**Key invariants tested:**
- Parser never panics on any input (ASCII or UTF-8)
- Valid integer/float/bool/string expressions parse without errors
- Assigning to immutable binding always produces E0150
- Integer arithmetic is commutative at the value level
- Type checker never panics on any parseable program

**Cargo integration:**
```toml
[[test]]
name = "prop_tests"
path = "tests/prop_tests/mod.rs"

[dev-dependencies]
proptest = "1"
```

**Success criteria:** `cargo test prop_tests` passes (256 cases per property minimum)

---

### S3-P0-07: End-to-End Fixture Runner

**Goal:** Golden-file test runner that walks `tests/fixtures/` and compares program output against `.stdout`/`.stderr` files.

**New public API:** `cjc_mir_exec::run_program_capture(program, seed) -> Result<(Value, Vec<String>), MirExecError>`

**Fixture categories:**
```
tests/fixtures/
  basic/        — arithmetic, let binding, if-else, recursion, string ops
  closures/     — adder, counter
  match_patterns/ — option, result, int match
  tco/          — countdown
  for_loops/    — range sum, nested, collect
  fstring/      — interpolation
  const_expr/   — pi_approx
  ad/           — gradient (auto-diff)
  error_cases/  — E0150, E0300, E0130 expected errors
```

**Minimum fixture count:** 20 `.cjc` + `.stdout`/`.stderr` pairs

**Success criteria:** `cargo test fixtures` passes; new `.cjc` files are auto-discovered

---

## S3-P1: Performance

### S3-P1-01: Vec COW (ADR-0009)

**Goal:** `Value::Array(Rc<Vec<Value>>)` and `Value::Tuple(Rc<Vec<Value>>)` instead of plain `Vec`.

**Change surface:** ~98 match sites across 6 crates (mechanical refactoring).

**Key mutation sites:** Use `Rc::make_mut(&mut v)` for any `push`, index assignment, or `extend` on `Value::Array` contents.

**Validation:** Parity gate (`milestone_2_4/parity/`) must pass unchanged — COW is an implementation detail, not a behavioral change.

**New test:** `tests/audit_tests/test_audit_cow_array.rs`

**Success criteria:**
- `cargo test --workspace` 0 failures
- `cargo test milestone_2_4 -- parity` 0 failures
- `Value::Array` uses `Rc<Vec<Value>>`

---

### S3-P1-02: Parallel Matmul (ADR-0011)

**Goal:** Optional `rayon`-parallel matmul for matrices where `m*k*n >= 64^3`.

**Feature gate:** `cjc-runtime/parallel`

**Accumulator switch:**
- Serial path: `KahanAccumulatorF64` (deterministic for fixed order)
- Parallel path: `BinnedAccumulatorF64` (commutative, deterministic across thread orders)

**Threshold:** `m * k * n >= THRESHOLD^3` where `THRESHOLD = 64` (configurable via `CJC_MATMUL_THRESHOLD` env var).

**Updated test:** `test_audit_parallelism_absence.rs` gated on `#[cfg(not(feature = "cjc-runtime/parallel"))]`

**New test:** `tests/audit_tests/test_audit_parallel_matmul.rs` — 5 tests for correctness and determinism

**Success criteria:**
- `cargo test --workspace` passes (serial, no feature)
- `cargo test --workspace --features cjc-runtime/parallel` passes (parallel)
- Serial and parallel results for ≤64×64 are bit-identical

---

## S3-P1: Compiler Depth

### S3-P1-06: CFG Phi Nodes + Use-Def Chains + Dominator Tree (ADR-0012)

**Goal:** Extend `cjc-mir/src/cfg.rs` with SSA infrastructure (without changing the executor).

**New types in cfg.rs:**
```rust
pub struct PhiNode { pub result: TempId, pub incoming: Vec<(BlockId, TempId)> }
pub struct UseDefChain { pub defs: HashMap<TempId, (BlockId, usize)>, pub uses: ... }
pub struct DomTree { pub idom: Vec<Option<BlockId>>, pub frontier: Vec<Vec<BlockId>> }
```

**New methods on MirCfg:**
```rust
pub fn build_use_def(&self) -> UseDefChain
pub fn build_dom_tree(&self) -> DomTree
```

**Algorithm:** Cooper et al. iterative dominator algorithm (O(n log n), no external deps).

**Optimizer update:** `cjc-mir/src/optimize.rs` DCE uses `UseDefChain::is_dead()` instead of `HashSet<String>` liveness.

**New tests:**
- Extend `tests/hardening_tests/test_h4_mir_cfg.rs` with 6 new tests
- New `tests/audit_tests/test_audit_cfg_ssa.rs` with 5 tests

**Success criteria:**
- `PhiNode`, `UseDefChain`, `DomTree` are public types
- `build_use_def()` and `build_dom_tree()` implemented and tested
- `cargo test milestone_2_4 -- optimizer` passes
- `cargo test --workspace` 0 failures

---

### S3-P1-07: TCO Extension (Conditional Branches)

**Goal:** Extend tail-call detection to cover `if/else` branches where both arms are tail calls.

**Current coverage:** Direct tail calls (`return f(args)` and body-expression `f(args)`).

**Extended coverage:**
- `if c { f(n-1) } else { g(n-1) }` — both branches in tail position → mutual recursion TCO
- Enabling pattern: `exec_body` is called for each branch body; tail calls in branch result expressions propagate `MirExecError::TailCall` to the trampoline

**New tests in `test_phase2_tco.rs`:**
- `tco_mutual_recursion_even_odd` — `even(1_000_000)` / `odd(1_000_000)` without stack overflow
- `tco_conditional_branch_tail_call` — `count_down(500_000)` via if/else
- `tco_if_else_both_tail_calls` — Collatz sequence from 27 (111 steps)

**Success criteria:**
- `even(1_000_000)` completes without stack overflow
- All existing 5 TCO tests still pass
- `cargo test test_phase2_tco` 0 failures

---

### S3-P1-08: Shape Inference Pipeline (E0500, E0501, E0502)

**Goal:** Emit compile-time diagnostics for tensor shape mismatches detectable from type annotations.

**New error codes:**
| Code | Condition |
|------|-----------|
| E0500 | Shape mismatch in tensor op (e.g., matmul inner dimensions) |
| E0501 | Unknown symbolic shape dimension used before binding |
| E0502 | Rank mismatch (e.g., matmul on 3D tensor) |

**Implementation:** `TypeChecker::check_expr` for `Call { callee: Field { name: "matmul" }, .. }` nodes — infer output shape from input shapes, emit E0500/E0502 on mismatch.

**Shape inference for matmul:**
```
Tensor<[M, K]>.matmul(Tensor<[K2, N]>)
  → if K == K2 (concrete): Ok(Tensor<[M, N]>)
  → if K != K2 (concrete): E0500
  → if K2 != 2D: E0502
  → if shapes unknown: Tensor<unknown_shape> (no error)
```

**New test:** `tests/audit_tests/test_audit_shape_inference.rs` (7 tests)

**Success criteria:**
- `test_audit_shape_inference` passes
- No regressions in `tests/milestone_2_4/shape/`
- `cargo test --workspace` 0 failures

---

## S3-P2: Domain Capability

### S3-P2-01 through S3-P2-04: ML Types

**New types in cjc-runtime:**
- `DType` — runtime dtype enum (F64, F32, F16, Bf16, I64..U8, Bool)
- `QuantizedTensor` — INT8/INT4 inference tensors with `dequantize()` method
- `MaskTensor` — bit-packed boolean masks with `and()`, `or()`, `apply_additive_mask()`
- `SparseCsr` extensions — `matvec()`, `to_dense()`, `shape()` methods

**New Value variants:** `Value::DTypeVal(DType)`, `Value::QuantizedTensor(Rc<QuantizedTensor>)`, `Value::MaskTensor(Rc<RefCell<MaskTensor>>)`

**New test:** `tests/audit_tests/test_audit_ml_types.rs` (11 tests)

**Success criteria:**
- All 11 ML type tests pass
- `SparseTensor.matvec()` dispatches correctly in mir-exec
- `cargo test --workspace` 0 failures

---

## S3-P1: Language Completeness

### Numeric Type Extensions

**New Type variants (cjc-types):**
`I8`, `I16`, `I32`, `U16`, `U32`, `U64`, `I128`, `U128`, `F16`, `Complex`

**New Value variants (cjc-runtime):**
`I8(i8)`, `I16(i16)`, `I32(i32)`, `U16(u16)`, `U32(u32)`, `U64(u64)`, `I128(i128)`, `U128(u128)`

**New error codes:**
- `E0200`: implicit numeric cast not allowed
- `E0201`: bit operation requires integer type

**New test:** `tests/audit_tests/test_audit_numeric_types.rs` (11 tests)

---

### Collection Type Extensions

**New Type variants (cjc-types):**
`Option(Box<Type>)`, `Result { ok, err }`, `Range { elem }`, `Slice { elem }`, `Set(Box<Type>)`, `Queue(Box<Type>)`

**New runtime types (cjc-runtime):**
`CjcSet` (insertion-order dedup), `CjcQueue` (VecDeque-backed FIFO)

**New Value variants:** `Value::Set(Rc<RefCell<CjcSet>>)`, `Value::Queue(Rc<RefCell<CjcQueue>>)`

**Method dispatch:** Set: `insert`, `contains`, `len`, `is_empty`, `to_list`. Queue: `enqueue`, `dequeue`, `peek`, `len`, `is_empty`.

**New test:** `tests/audit_tests/test_audit_collections.rs` (8 tests)

---

## Regression Gates

Every task must pass the following gates before being marked complete:

```bash
# Primary gate
cargo test --workspace
# Expected: 0 failed

# Parity gate (AST-eval == MIR-exec)
cargo test milestone_2_4 -- parity

# NoGC gate
cargo test milestone_2_4 -- nogc

# Optimizer gate
cargo test milestone_2_4 -- optimizer

# Fixture gate (after S3-P0-07)
cargo test fixtures

# Proptest gate (after S3-P0-06)
cargo test prop_tests

# Parallel gate (after S3-P1-02)
cargo test --workspace --features cjc-runtime/parallel
```

**Baseline test count:** 535 (Stage 2.4). This count must only increase. Any reduction in the test count indicates a test was accidentally deleted.

---

## New Test Files Added in Stage 3

| File | Tests | Task |
|------|-------|------|
| `tests/prop_tests/parser_props.rs` | 11 properties | S3-P0-06 |
| `tests/prop_tests/type_checker_props.rs` | 7 properties | S3-P0-06 |
| `tests/prop_tests/round_trip_props.rs` | 5 properties | S3-P0-06 |
| `tests/fixtures/runner.rs` | 1 (#[test]) | S3-P0-07 |
| `tests/audit_tests/test_audit_cow_array.rs` | 5 | S3-P1-01 |
| `tests/audit_tests/test_audit_parallel_matmul.rs` | 5 | S3-P1-02 |
| `tests/audit_tests/test_audit_numeric_types.rs` | 11 | S3-P1-03 |
| `tests/audit_tests/test_audit_collections.rs` | 8 | S3-P1-04/05 |
| `tests/audit_tests/test_audit_cfg_ssa.rs` | 5 | S3-P1-06 |
| `tests/audit_tests/test_audit_shape_inference.rs` | 7 | S3-P1-08 |
| `tests/audit_tests/test_audit_ml_types.rs` | 11 | S3-P2-01..04 |
| Extended: `tests/audit_tests/test_phase2_tco.rs` | +3 | S3-P1-07 |
| Extended: `tests/hardening_tests/test_h4_mir_cfg.rs` | +6 | S3-P1-06 |
| **Total new tests** | **~90** | — |

---

## Anti-Patterns (Prohibited)

1. **Do not change public function signatures** in `cjc-mir-exec/src/lib.rs`.
2. **Do not add rayon to workspace dependencies** — optional feature in `cjc-runtime` only.
3. **Do not change KahanAccumulatorF64** in the serial matmul path.
4. **Do not break `Value::Enum` representation** — `Option<T>` and `Result<T,E>` at the language level remain as `Value::Enum { variant: "Some"/"None"/"Ok"/"Err" }` at runtime.
5. **Do not rename existing test files.**
6. **Do not remove `#[cfg(test)]` blocks** during module splits — move them to the target file.

---

## Stage 4 Preview (Post Stage 3)

Once Stage 3 is complete (grade target: A-/A), Stage 4 priorities are:

1. **LLVM/Cranelift native backend** — compile CJC programs to native code for a subset of the language
2. **LSP server** — editor integration (VS Code extension)
3. **Concurrent execution model** — `Channel<T>`, `Arc<T>`, `Mutex<T>` for data parallelism
4. **File I/O** — `File` type, `read_file()`, `write_file()` builtins
5. **JSON support** — `Json` type with parse/serialize
