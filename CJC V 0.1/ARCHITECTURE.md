# Architecture

## Compilation Pipeline

```
Source Code
    |
    v
[Lexer] ──> Token Stream (59 token kinds)
    |
    v
[Parser] ──> AST (Abstract Syntax Tree)
    |
    v
[TypeChecker] ──> Diagnostics (E0xxx-E8xxx)
    |
    ├──> [Eval] ──> Result        (AST interpreter, default)
    |
    └──> [HIR Lowering] ──> HIR (High-level IR)
              |
              v
         [MIR Lowering] ──> MIR (Mid-level IR)
              |
              v
         [Optimizer] ──> Optimized MIR (CF + DCE)
              |
              v
         [MIR Executor] ──> Result  (--mir-opt flag)
```

## Crate Map

CJC is organized as a Cargo workspace with 15 crates:

| Crate | Purpose |
|-------|---------|
| `cjc-lexer` | Tokenizer — source text to token stream |
| `cjc-parser` | Parser — tokens to AST |
| `cjc-ast` | AST data structures and pretty-printer |
| `cjc-diag` | Diagnostic system — structured errors with spans |
| `cjc-types` | Type checker, trait system, effect checker, type environment |
| `cjc-dispatch` | Builtin function dispatch (300+ functions) |
| `cjc-runtime` | Value representation, tensor ops, statistics, linear algebra |
| `cjc-eval` | AST interpreter (tree-walk evaluator) |
| `cjc-hir` | HIR — high-level IR with pipe desugaring |
| `cjc-mir` | MIR — mid-level IR with CFG, SSA, optimizer |
| `cjc-mir-exec` | MIR executor — block-based interpreter |
| `cjc-ad` | Automatic differentiation (Dual numbers, GradGraph) |
| `cjc-data` | DataFrame, TidyView, tidy verb dispatch |
| `cjc-snap` | Content-addressable serialization (encode/decode/hash) |
| `cjc-repro` | Reproducibility utilities (SplitMix64 RNG) |
| `cjc-regex` | Regex engine adapter |
| `cjc-module` | Multi-file module resolution |
| `cjc-cli` | CLI entry point and REPL |

**Zero external dependencies** — everything is implemented from scratch in Rust.

## Key APIs

```rust
// Lex + Parse (convenience function)
let (program, diags) = cjc_parser::parse_source(src);

// Type check
let mut checker = cjc_types::TypeChecker::new();
checker.check_program(&program);

// Execute via AST interpreter
let mut interpreter = cjc_eval::Interpreter::new(seed);
let result = interpreter.exec(&program)?;

// Execute via MIR (optimized)
let (value, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)?;

// Execute via MIR (with optimizer enabled)
cjc_mir_exec::run_program_optimized(&program, seed)?;

// NoGC verification
cjc_mir_exec::verify_nogc(&program)?;
```

## Value Model

The runtime uses a `Value` enum with 16+ variants:

| Variant | Memory | GC |
|---------|--------|-----|
| `I64(i64)` | Stack | No |
| `F64(f64)` | Stack | No |
| `Bool(bool)` | Stack | No |
| `Str(String)` | Heap (owned) | No |
| `Array(Vec<Value>)` | Heap | No |
| `Tuple(Vec<Value>)` | Heap | No |
| `Struct { name, fields }` | Stack/heap | No |
| `Tensor(TensorValue)` | COW buffer | No |
| `Map(BTreeMap)` | Heap (ordered) | No |
| `ClassRef(GcRef)` | GC heap | Yes |
| `Void` | Zero-size | No |
| `Complex(f64, f64)` | Stack | No |
| `Enum { variant, payload }` | Stack/heap | No |
| `Bytes(Vec<u8>)` | Heap | No |

## Tensor Architecture

Tensors use copy-on-write (COW) storage:

```
TensorValue {
    buffer: Rc<RefCell<Vec<f64>>>,   // shared data
    shape: Vec<usize>,               // dimensions
    strides: Vec<usize>,             // for views
    offset: usize,                   // for slicing
}
```

Multiple tensors can share the same buffer. Deep copies only happen when
a shared buffer is mutated (Rc strong count > 1).

## MIR and Optimizer

The MIR is a control-flow graph (CFG) with basic blocks:

```
BasicBlock {
    instructions: Vec<MirInstr>,
    terminator: Terminator,          // Goto, Branch, Return
}
```

The optimizer applies:
- **Constant Folding** — evaluate constant expressions at compile time
- **Dead Code Elimination** — remove unreachable blocks and unused assignments
- **SSA Form** — single static assignment with phi nodes at join points

## Design Decisions

Key architectural choices documented in ADRs:

1. **Kahan Accumulator** — all floating-point summation uses compensated
   accumulation to minimize rounding errors
2. **SplitMix64 RNG** — fast, high-quality PRNG with reproducible sequences
3. **BTreeMap over HashMap** — ordered collections for deterministic iteration
4. **COW Tensors** — shared buffers with copy-on-write for memory efficiency
5. **CFG + Phi Nodes** — true SSA form for analysis (dominators, liveness)
6. **No External Dependencies** — everything from regex to linear algebra
   is implemented in Rust within the workspace

## Test Architecture

3,495+ tests across multiple suites:

| Suite | Tests | Purpose |
|-------|-------|---------|
| Language Hardening | 183 | Diagnostics, types, effects, SSA, optimizer |
| Role 9 Classes | 52 | Records, traits, value semantics, method effects |
| Mathematics | 120+ | Numerical accuracy, determinism, precision |
| Chess RL Benchmark | 49 | Full RL workload in pure CJC |
| Data Science | 100+ | Tidy verbs, joins, pivots, group-by |
| Parity Gates | 50+ | AST-eval == MIR-exec for all features |
| Audit | 30+ | Coverage gaps, regression tests |
