# CJC Codebase Audit & Report Card

**Date:** 2026-03-02
**Scope:** Full workspace (17 crates, ~54K LOC, 3,033 tests)
**Methodology:** Six-role stack audit — Language Design, Runtime & Tensors, Compiler Pipeline, Testing & Quality, Developer Experience, Architecture

---

## Executive Summary

CJC is an exceptionally well-engineered scientific computing language. It achieves **zero production external dependencies**, **deterministic execution** (RC memory, seeded RNG, Kahan summation), and **full eval/MIR-exec parity** across 200+ builtins. The codebase is clean (zero TODOs/FIXMEs), well-tested (3,033 tests, 0 failures), and architecturally sound (proper DAG dependency graph, no circular dependencies).

The main areas for improvement are: missing syntactic conveniences (bitwise ops, compound assignment, if-expressions), no CI/CD automation, monolithic execution engine files, and sparse inline documentation on public APIs.

---

## Overall Report Card

| Category | Grade | Summary |
|----------|-------|---------|
| **Language Design & Syntax** | **B+** | Strong parser/type system, missing operators & expression forms |
| **Runtime & Tensor System** | **A** | Excellent numerics, COW tensors, deterministic memory |
| **Compiler Pipeline** | **A-** | Clean HIR→MIR→Exec, good optimizer, AD needs Hessian |
| **Testing & Quality** | **B+** | 3,033 tests, fuzzing present, no CI/CD or benchmarks |
| **Developer Experience** | **A-** | Great docs & errors, no REPL, manual CLI arg parsing |
| **Architecture & Code Quality** | **A** | Clean layering, zero external deps, 4 safe `unsafe` blocks |
| **OVERALL** | **A-** | Production-quality scientific computing foundation |

---

## 1. Language Design & Syntax — Grade: B+

### What's Good

**Parser (Pratt-based) — A**
- Correct operator precedence across 10 levels with left/right associativity
- Excellent error recovery with synchronization points at semicolons and braces
- Handles ambiguous constructs well (struct literal vs block, division vs regex)
- Named + positional arguments, pipe operator (`|>`), format strings (`f"..."`)

**Type System — A-**
- 31 type variants covering primitives, tensors, collections, functions, ADTs
- Unification algorithm with occurs check and shape broadcasting (NumPy-compliant)
- Effect system with 8 bitflags: PURE, IO, ALLOC, GC, NONDET, MUTATES, ARENA_OK, CAPTURES
- Trait hierarchy: Numeric > {Int, Float > Differentiable}

**Pattern Matching — A**
- Wildcard, binding, literal, tuple, struct, and enum variant patterns
- Nested destructuring supported
- Implemented in both eval and MIR-exec with parity

**String Handling — A+**
- Regular strings with full escape sequences
- Raw strings (`r"..."`, `r#"..."#`)
- Byte strings and byte chars with hex escapes
- Format strings with interpolation (`f"hello {name}"`)

### What Needs Work

**Missing Operators — C**

| Category | Missing | Impact |
|----------|---------|--------|
| Bitwise | `&`, `\|`, `^`, `<<`, `>>`, `~` | Blocks low-level algorithms, hash functions |
| Power | `**` | Users must call `pow()` builtin instead |
| Compound assignment | `+=`, `-=`, `*=`, `/=`, `%=` | Verbose mutation patterns |
| Ternary | `cond ? a : b` | No inline conditional expression |

**Missing Expression Forms — C+**

| Feature | Status | Impact |
|---------|--------|--------|
| If-expression | `if` is statement only | Cannot write `let x = if c { a } else { b }` |
| Array slicing | Not supported | No `a[1:3]` syntax; must use manual loops |
| Range as value | Only in `for` loops | Cannot pass ranges to functions |
| List comprehension | Not supported | Must use for-loop + array_push pattern |
| Match guards | Not supported | No `pat if cond => ...` |

**Number Literals — B-**

| Literal | Status |
|---------|--------|
| Decimal | Supported (with `_` separators, type suffixes) |
| Scientific | Supported (`1e10`, `1.5e-3`) |
| Hexadecimal | **Missing** (`0xDEADBEEF`) |
| Binary | **Missing** (`0b10101010`) |
| Octal | **Missing** (`0o777`) |

**Other Gaps**
- No visibility modifiers (`pub`/`private`)
- No type aliases (`type Foo = Bar`)
- No associated types in traits
- No loop labels for targeted break/continue
- Unicode handling is byte-level (no explicit multi-byte support)

### Recommendations
1. **Priority 1:** Add `if`-as-expression (enables functional patterns)
2. **Priority 1:** Add compound assignments (`+=`, `-=`, etc.)
3. **Priority 2:** Add hex/binary/octal integer literals
4. **Priority 2:** Add bitwise operators
5. **Priority 3:** Add power operator (`**`)
6. **Priority 3:** Add array slicing syntax
7. **Priority 4:** Add match guards and alternative patterns

---

## 2. Runtime & Tensor System — Grade: A

### What's Good

**Builtin Library — A**
- 248+ registered builtins covering math, stats, ML, data, string ops
- Consistent error handling pattern across all builtins (arg count + type checks)
- Proper numeric precision: Kahan summation in matmul, binned accumulation for order-invariance, two-pass numerically-stable softmax

**Tensor Implementation — A**
- Dense f64 storage with stride-based views (zero-copy slicing, transposing)
- COW semantics: `clone()` is O(1) shallow; mutation triggers deep copy
- NumPy-compliant broadcasting with proper shape validation
- Tiled matmul for cache locality, parallel matmul via optional Rayon

**Value System — A**
- 23 value variants covering all CJC types
- Smart memory strategy: primitives on stack, collections via `Rc<T>`, mutable containers via `Rc<RefCell<T>>`
- Bf16 with correct truncation semantics and widen-compute-narrow arithmetic
- Deterministic map iteration via `DetMap`

**Memory Model — A+**
- Deterministic reference counting replaced mark-sweep GC
- LIFO slot reuse in ObjectSlab guarantees deterministic allocation order
- Per-frame bump arena for non-escaping function-local values
- 16-byte aligned pool for SIMD readiness
- Same seed + same program = bit-identical execution

**Effect System — A+**
- Single source of truth for all 248 builtin classifications
- Thread-local caching avoids rebuild overhead
- Conservative approach: unregistered builtins default to unsafe

### What Needs Work

**Sparse Tensors — D (Dead Code)**
- `Value::SparseTensor(SparseCsr)` registered in value enum
- Only `SparseCsr.to_dense()` builtin exposed
- No constructors, no operations, no tests
- **Recommendation:** Either complete sparse support or remove stubs

**Minor Issues**
- 6 `.unwrap()` calls in tensor activation functions (relu, sigmoid, tanh) — should use `.expect("shape invariant")`
- `HashMap` used for Struct fields — nondeterministic iteration order; should use `IndexMap`
- Tiled matmul uses naive accumulation (not Kahan) — intentional for cache, but undocumented
- No overflow guards on `int(NaN)` or `int(inf)` conversions
- `Map.insert` appears twice in effect registry (duplicate)

### Recommendations
1. **Decide on sparse tensors:** complete or remove (reduces dead code)
2. **Replace `.unwrap()` with `.expect("reason")`** in tensor activation functions
3. **Use `IndexMap` for struct fields** to preserve insertion order deterministically
4. **Document numeric precision trade-offs** (Kahan vs Binned vs naive paths)
5. **Add edge-case tests:** empty tensors, `int(NaN)`, `int(inf)`, all-NaN argmax

---

## 3. Compiler Pipeline — Grade: A-

### What's Good

**HIR (High-Level IR) — A**
- Clean desugaring: pipes to calls, multi-index to method chains
- Every node carries `HirId` for diagnostics
- Full language feature coverage (closures, patterns, enums)

**MIR (Mid-Level IR) — A**
- CFG-based representation with explicit control flow
- Lambda lifting: closures become top-level functions with captured params
- Escape analysis annotations via `AllocHint` (Stack, Arena, Rc)
- Linalg opcodes already exposed at MIR level (future optimization target)

**MIR Optimizer — B+**
- Constant folding respects IEEE 754 (doesn't fold div-by-zero)
- Dead code elimination with conservative purity checks
- Conditional branch elimination (`if true { a } else { b }` → `a`)
- Strength reduction pass present

**Eval Engine — A-**
- 200+ builtins, proper scope semantics, good error propagation
- Short-circuit evaluation for `&&` and `||`
- Tensor arithmetic with scalar broadcast
- CSV streaming support (Csv.parse, Csv.stream_sum)

**MIR-Exec Engine — A**
- Tail-call optimization via trampoline (enables efficient recursion)
- Arena allocation for non-escaping values (reduces GC pressure)
- Function caching via `Rc<MirFunction>` (avoids clones)
- Full parity with eval engine (verified by 50+ parity tests)

**NoGC Verifier — A+**
- Three-phase verification: call graph, effect registry, escape analysis
- Conservative fixpoint iteration for transitive GC propagation
- Rejects RC bindings in `@nogc` blocks; allows Arena bindings
- Sound analysis with good error messages (includes call chains)

**Automatic Differentiation — B+**
- Forward mode (dual numbers) with transcendentals (sin, cos, exp, ln, sqrt, pow)
- Reverse mode (computational graph) with 20+ GradOp kinds
- Correct gradient formulas (verified by tests): matmul, sigmoid, relu, tanh
- Both modes tested against known derivatives

### What Needs Work

**Optimizer Gaps — B-**
- LICM (loop-invariant code motion) incomplete — CFG infrastructure exists but pass not wired
- CSE (common subexpression elimination) had aliasing bug (fixed in commit 291c5ba) — needs stress testing
- No instruction scheduling, register allocation, or SIMD optimization
- No inlining pass

**AD Gaps — B**
- No double-backprop (Hessian computation)
- StructField and MapLookup gradients are placeholders (don't flow correctly)
- MatMul gradient not fused (could combine transpose operations)
- No vmap or batched AD
- No autodiff of loss functions from language level (API only)

**Other**
- NoGC verifier rejects all indirect calls (overly conservative)
- Monomorphization is preparation only (no actual code specialization benefit)
- No bytecode compilation or JIT (both engines are interpreters)

### Recommendations
1. **Complete LICM pass** on CFG for loop optimization
2. **Stress test CSE** with adversarial inputs after aliasing fix
3. **Add Hessian support** (double-backprop) for second-order optimization
4. **Fix StructField gradient placeholders** in AD
5. **Consider bytecode compilation** for 5-10x speedup over tree-walking
6. **Add inlining pass** for small functions

---

## 4. Testing & Quality — Grade: B+

### What's Good

**Test Coverage — A**
- 3,033 tests across 192 test files, 55 test binaries
- Zero failures, 20 ignored (intentional diagnostic tests)
- Well-organized suites: milestone (198), parity (50+), tidy (44), chess RL (49), math hardening (120)
- Edge cases covered: f16 precision, i4 quantized, NaN handling, u64 boundaries

**Parity Gates — A+**
- 50+ tests ensure AST-eval and MIR-exec produce identical results
- Covers all builtin categories: math, tensor, string, data, ML
- Determinism verified with double-run gates

**Property-Based Testing — B+**
- Proptest: parser robustness, complex arithmetic, round-trip invariants
- Bolero fuzzing: lexer, parser, MIR pipeline, optimizer parity
- 5 fuzz targets with libfuzzer-sys integration

**Diagnostic System — A**
- Span-aware error reporting with byte offset to line:col conversion
- Multi-label support for secondary locations
- Error codes (E0001, E0100, etc.) with hints for recovery
- Source context with underline carets

### What Needs Work

**CI/CD — F (Missing)**
- No `.github/workflows/`, no `Makefile`, no `justfile`
- All testing is manual (developer runs `cargo test` locally)
- No automated PR gates, no clippy enforcement, no coverage tracking
- No scheduled fuzzing runs

**Benchmarking — D**
- No criterion-based benchmarks anywhere in codebase
- Informal timing tests only (`Instant::now()` in a few test files)
- No matmul, parser, type-checker, or compilation time benchmarks

**Panic/Unwrap Usage — C+**
- ~540 panic/unwrap/expect instances across core crates
- Highest concentration: cjc-eval (195), cjc-mir-exec (123), cjc-runtime (65)
- Many are in hot interpreter paths (performance vs safety trade-off)
- Scope invariants (`scopes.last().unwrap()`) undocumented

**Negative Testing — C**
- Limited tests verifying that invalid programs produce correct error messages
- Only ~12 tests directly verify diagnostic behavior
- Module system has only 3 tests for cycle detection, symbol resolution

**Error Type Consistency — B-**
- 6 different error enums across crates; some use structured fields, others use `String` catch-all
- Break/Continue encoded as error variants (unconventional anti-pattern)
- No error codes in Eval and Data layers (only MirExec wraps diagnostics)

### Recommendations
1. **Critical: Set up GitHub Actions CI** — test, clippy, build gates
2. **Add criterion benchmarks** for matmul, parser, type checker, compilation
3. **Audit unwrap() in hot paths** — document safety invariants or replace with proper errors
4. **Expand negative testing** — 20+ tests for error messages, invalid programs
5. **Standardize error codes** across all layers
6. **Add scheduled fuzzing** in CI (nightly bolero runs)

---

## 5. Developer Experience — Grade: A-

### What's Good

**External Documentation — A+**
- 60+ markdown files in `docs/`
- Complete syntax reference (CJC_SYNTAX_AND_TYPES.md, 650+ lines)
- Full v1 specification (CJC_v1_MVP_SPEC.md)
- 12 Architecture Decision Records (docs/adr/)
- Phase changelogs with detailed changes
- Chess RL benchmark documentation (design, determinism proof, results)

**Error Messages — A**
- Span-aware with file:line:col locations
- Source context with visual caret underlines
- Hint system for actionable suggestions
- Error codes with consistent prefix (E0001, E0100, etc.)

**Zero External Dependencies — A+**
- Only Rayon (optional, for parallel matmul)
- All dev dependencies (bolero, proptest, tempfile) are test-only
- Workspace inheritance for consistent versioning

**Code Quality — A**
- Zero TODO/FIXME/HACK markers in source code
- Only 5 build warnings (all minor: unused imports, unreachable pattern)
- Clean `Result`-based error propagation throughout
- Consistent naming conventions

### What Needs Work

**CLI — B**
- Manual/ad-hoc argument parsing (no `clap` or structured parser)
- No REPL or interactive mode
- No `--help` flag (only `print_usage()`)
- No `--quiet`/`--verbose` flags
- No color-coded error output (ANSI colors missing)
- No file watching or incremental compilation

**Inline Documentation — C+**
- ~4.9% doc density (2,626 `///` comments + 596 `//!` blocks across 54K LOC)
- Public APIs in parser, lexer, types crates have minimal doc comments
- No inline code examples in public function signatures
- Builtins in runtime have no user-facing documentation

**Project Onboarding — C**
- No root README.md with quick start guide
- No CONTRIBUTING.md or development setup instructions
- No architecture overview document mapping crates to roles
- Newcomers must read 60+ docs files to understand the system

### Recommendations
1. **Create root README.md** with architecture diagram, quick start, examples
2. **Add REPL support** for interactive CJC exploration
3. **Add ANSI color output** to diagnostic renderer
4. **Add inline examples** to public API docs (especially builtins, parser, tensor)
5. **Consider structured CLI** (clap or custom) for extensibility
6. **Fix 5 build warnings** for a clean `cargo build` output

---

## 6. Architecture & Code Quality — Grade: A

### What's Good

**Dependency Graph — A+**
- Proper DAG with no circular dependencies
- Clean 5-layer architecture: Foundations → Lexing/Types → Parsing/Runtime → Evaluation → CLI
- 4 leaf crates with zero dependencies (cjc-ast, cjc-diag, cjc-repro, cjc-regex)

**Unsafe Code — A+**
- Only 4 `unsafe` blocks in entire 54K LOC codebase
- All 4 are in memory subsystem (aligned_pool, frame_arena, object_slab)
- All are well-documented with safety comments
- All perform sound lifetime extensions for type-erased storage

**Reproducibility — A+**
- SplitMix64 RNG with explicit seeding and deterministic forking
- Kahan + pairwise summation for stable floating-point reduction
- Deterministic memory allocation (LIFO slot reuse)
- Same seed + same program = bit-identical results

**Feature Flags — A**
- Single `parallel` feature (optional Rayon) — not enabled by default
- No platform-specific code (good for reproducibility)
- `#[cfg(test)]` modules properly isolated

**LOC Distribution — A-**
- cjc-runtime is largest (16.9K LOC) but well-modularized into 31 files
- Clean separation: each crate has single responsibility
- Total: ~54K LOC across 17 crates

### What Needs Work

**Monolithic Files — B-**
- cjc-eval: 4,105 LOC in single `lib.rs`
- cjc-mir-exec: 3,581 LOC in single `lib.rs`
- cjc-parser: 2,706 LOC in single `lib.rs`
- cjc-hir: 2,302 LOC in single `lib.rs`
- All are navigable but would benefit from splitting

**Coupling in Execution Tier — B**
- cjc-mir-exec has 14 dependencies (highest in workspace)
- Includes unused cjc-lexer and cjc-parser dependencies
- cjc-eval has 9 dependencies (second highest)
- Both pull in almost every other crate

**Testability — B-**
- Tight coupling in execution tier makes isolation testing harder
- No `Evaluator` trait to abstract eval vs mir-exec
- Module system spans all IR levels (hard to test individually)

### Recommendations
1. **Remove unused deps** from cjc-mir-exec (cjc-lexer, cjc-parser)
2. **Split monolithic files**: extract `eval_builtins.rs`, `mir_exec_dispatch.rs`, `parser_expr.rs`
3. **Add `Evaluator` trait** to abstract eval vs mir-exec for polymorphic CLI
4. **Create ARCHITECTURE.md** mapping each crate to its role and layer
5. **Consider module abstraction** to decouple module system from parser/HIR/MIR

---

## Cross-Cutting Findings

### Numeric Precision & Determinism — A+
CJC demonstrates exceptional discipline in numeric handling:
- **6 accumulation strategies**: Kahan, Binned, pairwise, naive (cache), parallel Kahan, two-pass softmax
- **IEEE 754 compliance**: optimizer refuses to fold div-by-zero, preserves NaN semantics
- **Deterministic everywhere**: RNG, memory, dispatch, tensor operations, training loops
- **Verified**: Chess RL benchmark produces bit-identical results across runs with same seed

### Memory Safety — A+
- 4 unsafe blocks total (all sound, all documented)
- RC-based memory model avoids GC pauses
- NoGC verifier catches allocation in performance-critical paths
- Escape analysis distinguishes Arena vs Rc allocation

### Parity Guarantee — A+
- Both eval and MIR-exec engines dispatch identical builtins
- 50+ parity tests verify output equivalence
- User-function priority fix applied to both engines simultaneously
- Three-layer wiring discipline prevents "works in eval, breaks in MIR" bugs

---

## Priority Recommendations Summary

### Tier 1 — High Impact, Low Effort
| # | Recommendation | Effort | Impact |
|---|---------------|--------|--------|
| 1 | Set up GitHub Actions CI (test, clippy, build) | 1 day | Prevents regressions |
| 2 | Create root README.md with quick start | 1 day | Developer onboarding |
| 3 | Remove unused deps from cjc-mir-exec | 30 min | Cleaner architecture |
| 4 | Fix 5 build warnings | 30 min | Clean builds |
| 5 | Replace `.unwrap()` with `.expect()` in tensor activations | 1 hour | Safer panics |

### Tier 2 — High Impact, Medium Effort
| # | Recommendation | Effort | Impact |
|---|---------------|--------|--------|
| 6 | Add compound assignments (`+=`, `-=`, etc.) | 2-3 days | Major ergonomic win |
| 7 | Make `if` an expression | 2 days | Enables functional patterns |
| 8 | Add criterion benchmarks (matmul, parser, type checker) | 2 days | Performance tracking |
| 9 | Add hex/binary/octal integer literals | 1-2 days | Low-level algorithm support |
| 10 | Decide on sparse tensors (complete or remove) | 1-2 days | Reduces dead code |

### Tier 3 — Medium Impact, Higher Effort
| # | Recommendation | Effort | Impact |
|---|---------------|--------|--------|
| 11 | Add bitwise operators | 3-4 days | Enables hashing, bit manipulation |
| 12 | Add REPL support | 3-5 days | Interactive exploration |
| 13 | Complete LICM optimizer pass | 3-5 days | Loop performance |
| 14 | Add Hessian support (double-backprop) | 3-5 days | Second-order optimization |
| 15 | Split monolithic engine files | 2-3 days | Maintainability |

### Tier 4 — Long-Term Vision
| # | Recommendation | Effort | Impact |
|---|---------------|--------|--------|
| 16 | Bytecode compilation / JIT | Weeks | 5-10x performance |
| 17 | Full type inference (HM-style) | Weeks | Reduced annotation burden |
| 18 | Array slicing syntax | 1 week | NumPy-like ergonomics |
| 19 | Evaluator trait abstraction | 1 week | Testability, extensibility |
| 20 | vmap / batched AD | 1-2 weeks | ML training performance |

---

## Appendix: Crate Map

```
crates/                          LOC    Deps  Layer
├── cjc-ast/                    1,258    0    Foundation (AST node definitions)
├── cjc-diag/                     294    0    Foundation (diagnostics + spans)
├── cjc-repro/                    360    0    Foundation (deterministic RNG + summation)
├── cjc-regex/                    960    0    Foundation (regex engine)
├── cjc-lexer/                  1,534    1    Lexing (tokenization)
├── cjc-types/                  4,153    2    Types (type system + effects)
├── cjc-parser/                 2,706    3    Parsing (Pratt parser)
├── cjc-dispatch/                 493    3    Dispatch (overload resolution)
├── cjc-runtime/               16,909    1    Runtime (tensors, builtins, GC, arena)
├── cjc-ad/                     1,123    1    AD (forward + reverse mode)
├── cjc-data/                   7,004    3    Data (DataFrame, CSV, tidy DSL)
├── cjc-hir/                    2,302    3    HIR (desugared AST)
├── cjc-mir/                    5,912    4    MIR (CFG, optimizer, NoGC verifier)
├── cjc-eval/                   4,105    9    Eval (tree-walk interpreter)
├── cjc-mir-exec/               3,581   14    MIR-Exec (MIR interpreter + TCO)
├── cjc-module/                   915    7    Module (multi-file orchestration)
└── cjc-cli/                      264    9    CLI (entry point)
                               ──────
                               ~54,001 LOC total
```

---

*Report generated by six-role stack audit: Language Design Analyst, Runtime Architect, Compiler Engineer, Quality Assurance Lead, DX Specialist, Architecture Reviewer*
