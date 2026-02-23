# CJC Stage 2 — Progress Report

> **Date**: February 2026
> **Stage**: 2 (IR Pipeline, Closures, Pattern Matching, For-Loops)
> **Current Milestone**: 2.0 (IR Skeleton + MIR Executor MVP) — **COMPLETE**
> **Build Status**: Clean (zero errors, zero warnings)
> **Tests**: 340/340 passing (170 inline + 170 integration)

---

## 1. What Was Accomplished

### 1.1 Stage 2 Specification Written

**File**: `docs/CJC_STAGE2_SPEC.md` (~1,127 lines)

A complete, implementation-ready specification covering all 8 required sections:

| Section | Title | Key Decisions |
|---------|-------|---------------|
| 1 | MVP Scope | IN/OUT scope tables; 5 anti-feature-creep rules |
| 2 | IR-Exec Backend | **MIR Interpreter** chosen over bytecode VM (6-criteria comparison) |
| 3 | HIR/MIR/Kernel IR | Full Rust `enum`/`struct` definitions for all three IR levels |
| 4 | Closure Capture Rules | `Ref` (default) and `Clone` (nogc-forced) modes; 5-step capture analysis |
| 5 | Pattern Matching MVP | 8 pattern kinds; decision tree compilation to MIR Branch/Switch |
| 6 | For-Loop MVP | Range (`0..n`) and iterator forms; desugaring to `while` in HIR |
| 7 | Parity Gates | 10 gates (G-1 through G-10) required before Stage 3 |
| 8 | Milestone Plan | 6 milestones (2.0 through 2.5) with line/test estimates |

### 1.2 Milestone 2.0 — IR Skeleton + MIR Executor MVP (COMPLETE)

Three new crates created, implementing the full AST -> HIR -> MIR -> Execute pipeline:

#### cjc-hir (AST -> HIR Lowering)
- **File**: `crates/cjc-hir/src/lib.rs`
- HIR data structures: `HirProgram`, `HirFn`, `HirExpr` (17 expression kinds), `HirStmt` (6 statement kinds)
- `AstLowering` pass that converts AST to HIR, including:
  - **Pipe desugaring**: `a |> f(b)` lowered to `Call(f, [a, b])`
  - All expression/statement/declaration types preserved
  - `HirId` monotonic IDs for every HIR node
- **18 inline tests + 18 integration tests**

#### cjc-mir (HIR -> MIR Lowering)
- **File**: `crates/cjc-mir/src/lib.rs`
- MIR data structures: `MirProgram`, `MirFunction`, `MirBody`, `MirStmt`, `MirExpr`
- `HirToMir` pass that converts HIR to simplified MIR:
  - Top-level statements collected into synthetic `__main` function
  - Impl methods registered as `Target.method` qualified names
  - If/else chains flattened into nested MIR if-stmts
  - Struct/Class defs captured as `MirStructDef`
- **6 inline tests + 6 integration tests**

#### cjc-mir-exec (MIR Reference Interpreter)
- **File**: `crates/cjc-mir-exec/src/lib.rs`
- `MirExecutor` — full interpreter that operates on MIR instead of AST
- Feature-parity with `cjc-eval` tree-walk interpreter:
  - All 14 builtins: print, Tensor.zeros/ones/randn/from_vec, matmul, Buffer.alloc, len, assert, assert_eq, clock, gc_alloc, gc_collect, gc_live_count
  - All 10 tensor methods: sum, mean, shape, len, to_vec, matmul, add, sub, reshape, get
  - Binary/unary ops, short-circuit logic, type promotion (Int+Float -> Float)
  - Struct creation, field access/assignment, method dispatch
  - Array indexing, tensor multi-indexing
  - Scoped variables, early return, while loops
  - Lambda registration, GC heap integration, deterministic RNG
- Convenience functions: `run_program()` and `run_program_with_executor()` for full AST -> HIR -> MIR -> Execute pipeline
- **8 inline tests + 8 integration tests**

### 1.3 Integration Test Suite

**Location**: `tests/` directory (13 files, 170 tests)

All tests have been extracted into standalone integration test files in the `tests/` folder. These serve as a **regression safety net** — if any internal refactoring breaks public API behavior, these tests catch it independently from the inline tests.

| File | Crate | Tests | What It Covers |
|------|-------|-------|----------------|
| `test_ad.rs` | cjc-ad | 12 | Forward-mode dual numbers, reverse-mode gradients, finite diff |
| `test_data.rs` | cjc-data | 10 | DataFrame creation, filtering, group-by, aggregation, tensor bridge |
| `test_diag.rs` | cjc-diag | 3 | Span merging, diagnostic rendering, diagnostic bag |
| `test_dispatch.rs` | cjc-dispatch | 8 | Concrete/generic/constrained dispatch, specificity, errors |
| `test_eval.rs` | cjc-eval | 28 | Arithmetic, control flow, functions, structs, tensors, pipes, scopes |
| `test_hir.rs` | cjc-hir | 18 | AST-to-HIR lowering: literals, ops, pipes, fn decls, structs, lambdas |
| `test_lexer.rs` | cjc-lexer | 14 | Tokens, keywords, operators, comments, spans, strings, numbers |
| `test_mir.rs` | cjc-mir | 6 | HIR-to-MIR lowering: literals, ops, functions, programs, structs |
| `test_mir_exec.rs` | cjc-mir-exec | 8 | Full pipeline: arithmetic, calls, loops, pipes, tensors, recursion, print |
| `test_parser.rs` | cjc-parser | 33 | All declaration/statement/expression parsing, error recovery |
| `test_repro.rs` | cjc-repro | 6 | RNG determinism, Kahan summation (f32/f64), pairwise sum |
| `test_runtime.rs` | cjc-runtime | 17 | Buffer COW, tensor ops/matmul/reshape, GC alloc/collect/reuse |
| `test_types.rs` | cjc-types | 7 | Built-in types, trait satisfaction, scoping, type matching |
| **Total** | | **170** | |

### 1.4 Workspace Infrastructure

- Root `[package]` named `cjc` with dependencies on all 14 library crates
- `src/lib.rs` umbrella crate for integration test discovery
- Visibility fixes for 9 items across 4 crates to support integration testing

---

## 2. Current Test Summary

```
$ cargo test --workspace

340 tests total:
  170 inline unit tests    (inside each crate's #[cfg(test)] modules)
+ 170 integration tests    (tests/*.rs at workspace root)
= 340/340 passing
```

**Breakdown by crate (inline + integration):**

| Crate | Inline | Integration | Total |
|-------|--------|-------------|-------|
| cjc-ad | 12 | 12 | 24 |
| cjc-data | 10 | 10 | 20 |
| cjc-diag | 3 | 3 | 6 |
| cjc-dispatch | 8 | 8 | 16 |
| cjc-eval | 28 | 28 | 56 |
| cjc-hir | 18 | 18 | 36 |
| cjc-lexer | 14 | 14 | 28 |
| cjc-mir | 6 | 6 | 12 |
| cjc-mir-exec | 8 | 8 | 16 |
| cjc-parser | 33 | 33 | 66 |
| cjc-repro | 6 | 6 | 12 |
| cjc-runtime | 17 | 17 | 34 |
| cjc-types | 7 | 7 | 14 |
| **Total** | **170** | **170** | **340** |

---

## 3. Crate Inventory

| Crate | Purpose | Stage |
|-------|---------|-------|
| cjc-ast | AST data structures | 1 |
| cjc-diag | Diagnostics (spans, errors) | 1 |
| cjc-lexer | Tokenizer | 1 |
| cjc-parser | Recursive descent parser | 1 |
| cjc-types | Type system, scopes | 1 |
| cjc-dispatch | Multiple dispatch resolution | 1 |
| cjc-runtime | Buffer/Tensor/GC/Value | 1 |
| cjc-eval | Tree-walk AST interpreter (legacy) | 1 |
| cjc-ad | Automatic differentiation | 1 |
| cjc-data | DataFrame DSL | 1 |
| cjc-repro | Deterministic RNG, stable summation | 1 |
| cjc-cli | CLI binary (lex/parse/check/run) | 1 |
| **cjc-hir** | **HIR + AST-to-HIR lowering** | **2** |
| **cjc-mir** | **MIR + HIR-to-MIR lowering** | **2** |
| **cjc-mir-exec** | **MIR reference interpreter** | **2** |
| **Total** | | **15 crates** |

---

## 4. Stage 2 Roadmap Status

### Milestone Tracker

| Milestone | Goal | Status | Lines Added | Tests Added |
|-----------|------|--------|-------------|-------------|
| **Pre-2.0** | Specification + test infra | **COMPLETE** | 0 code, 1127 spec | 138 integration |
| **2.0** | IR Skeleton + MIR Executor MVP | **COMPLETE** | ~2,800 | 32 new (18+6+8) |
| **2.1** | Closures in MIR | NOT STARTED | ~800 est. | ~20 new |
| **2.2** | Match + Patterns | NOT STARTED | ~600 est. | ~15 new |
| **2.3** | For-loop + Desugar | NOT STARTED | ~400 est. | ~10 new |
| **2.4** | Optimizer + NoGC Verifier | NOT STARTED | ~500 est. | ~15 new |
| **2.5** | Parity Gate Sign-Off | NOT STARTED | ~100 est. | 0 new |

### Parity Gate Status

| Gate | Description | Status |
|------|-------------|--------|
| G-1 | 138 unit tests via new pipeline | IN PROGRESS (8 pipeline tests passing) |
| G-2 | 3 demos produce identical output | BLOCKED (needs CLI integration) |
| G-3 | Matmul benchmark parity | BLOCKED (needs CLI integration) |
| G-4 | Memory pressure parity | BLOCKED (needs CLI integration) |
| G-5 | 20+ closure tests | BLOCKED (needs 2.1) |
| G-6 | 15+ match tests | BLOCKED (needs 2.2) |
| G-7 | 10+ for-loop tests | BLOCKED (needs 2.3) |
| G-8 | NoGC verifier tests | BLOCKED (needs 2.4) |
| G-9 | AD benchmark parity | READY (AD engine unchanged) |
| G-10 | Optimizer soundness tests | BLOCKED (needs 2.4) |

---

## 5. Architecture Decisions Locked In

These decisions were made in the Stage 2 spec and are now **frozen**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IR execution model | **MIR Interpreter** (not bytecode VM) | Maps 1:1 to LLVM IR; easier debugging; tree-walk style proven |
| IR pipeline | **AST -> HIR -> MIR -> MIR-Exec** | Standard compiler architecture; each level has clear purpose |
| Closure capture | **Ref (default) + Clone (nogc)** | Safe by default; explicit copy semantics in performance zones |
| Pattern compilation | **Decision trees** | Standard approach (Rust/OCaml); compiles to MIR Branch/Switch |
| For-loop lowering | **Desugar to while in HIR** | Zero new MIR constructs needed; simple, correct |
| New crates | **cjc-hir, cjc-mir, cjc-mir-exec** | Clean separation of concerns; each crate independently testable |
| Exhaustiveness checking | **Deferred to Stage 3** | Needs enum support to be truly useful |

---

## 6. Files Changed/Created in Stage 2 Work

### New Files

| File | Purpose |
|------|---------|
| `docs/CJC_STAGE2_SPEC.md` | Complete Stage 2 specification (1,127 lines) |
| `docs/CJC_STAGE2_PROGRESS.md` | This document |
| `src/lib.rs` | Umbrella crate for integration test discovery |
| `crates/cjc-hir/Cargo.toml` | HIR crate manifest |
| `crates/cjc-hir/src/lib.rs` | HIR data structures + AST-to-HIR lowering (~680 lines) |
| `crates/cjc-mir/Cargo.toml` | MIR crate manifest |
| `crates/cjc-mir/src/lib.rs` | MIR data structures + HIR-to-MIR lowering (~530 lines) |
| `crates/cjc-mir-exec/Cargo.toml` | MIR executor crate manifest |
| `crates/cjc-mir-exec/src/lib.rs` | MIR reference interpreter (~1,200 lines) |
| `tests/test_ad.rs` | Integration tests for cjc-ad (12 tests) |
| `tests/test_data.rs` | Integration tests for cjc-data (10 tests) |
| `tests/test_diag.rs` | Integration tests for cjc-diag (3 tests) |
| `tests/test_dispatch.rs` | Integration tests for cjc-dispatch (8 tests) |
| `tests/test_eval.rs` | Integration tests for cjc-eval (28 tests) |
| `tests/test_hir.rs` | Integration tests for cjc-hir (18 tests) |
| `tests/test_lexer.rs` | Integration tests for cjc-lexer (14 tests) |
| `tests/test_mir.rs` | Integration tests for cjc-mir (6 tests) |
| `tests/test_mir_exec.rs` | Integration tests for cjc-mir-exec (8 tests) |
| `tests/test_parser.rs` | Integration tests for cjc-parser (33 tests) |
| `tests/test_repro.rs` | Integration tests for cjc-repro (6 tests) |
| `tests/test_runtime.rs` | Integration tests for cjc-runtime (17 tests) |
| `tests/test_types.rs` | Integration tests for cjc-types (7 tests) |

### Modified Files

| File | Change |
|------|--------|
| `Cargo.toml` | Added root `[package]`, 3 new crate members, deps for integration tests |
| `crates/cjc-eval/src/lib.rs` | Made `define()`, `exec_if()`, `eval_expr()` public |
| `crates/cjc-runtime/src/lib.rs` | Made `Tensor.buffer`, `GcHeap.free_list`, `GcRef.index` public |
| `crates/cjc-hir/src/lib.rs` | Made `lower_expr()`, `lower_stmt()`, `lower_fn_decl()`, `lower_if()` public |
| `crates/cjc-mir/src/lib.rs` | Made `lower_expr()`, `lower_fn()`, `lower_if_stmt()` public |

---

## 7. Next Steps

The immediate next task is **CLI integration + Parity Gate G-1/G-2**:

1. Update `cjc-cli` to add `--mir` flag that runs the new AST -> HIR -> MIR -> MIR-Exec pipeline
2. Keep `--legacy` flag (or default) for the old tree-walk interpreter
3. Run all 3 demos through both pipelines and verify identical output (Gate G-2)
4. Run benchmarks through the new pipeline (Gates G-3, G-4, G-9)

Then proceed to **Milestone 2.1 — Closures in MIR**:

1. Implement closure capture analysis (Ref/Clone modes)
2. Lambda-lifting in HIR -> MIR pass
3. 20+ closure-specific tests (Gate G-5)

---

*End of Stage 2 Progress Report*
