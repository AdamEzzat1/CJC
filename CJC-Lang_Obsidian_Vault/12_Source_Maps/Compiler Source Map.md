---
title: Compiler Source Map
tags: [source-map, compiler]
status: Grounded in crate layout
---

# Compiler Source Map

Pointers from concepts to the files that implement them in the compiler pipeline. File paths are relative to `C:\Users\adame\CJC`.

## Pipeline at a glance

```
source → Lexer → Parser → AST → TypeChecker → HIR → MIR → Optimizer → { cjc-eval | cjc-mir-exec }
```

| Stage | Crate | Hub note |
|---|---|---|
| Tokenization | `crates/cjc-lexer/` | [[Lexer]] |
| Parsing | `crates/cjc-parser/` | [[Parser]] |
| AST definitions | `crates/cjc-ast/` | [[AST]] |
| Type system | `crates/cjc-types/` | [[Type Checker]] |
| Diagnostics | `crates/cjc-diag/` | [[Diagnostics]], [[Error Codes]] |
| HIR lowering | `crates/cjc-hir/` | [[HIR]], [[Capture Analysis]] |
| MIR lowering + opt + NoGC | `crates/cjc-mir/` | [[MIR]], [[MIR Optimizer]], [[NoGC Verifier]], [[Escape Analysis]] |
| AST interpreter (v1) | `crates/cjc-eval/` | [[cjc-eval]] |
| MIR executor (v2) | `crates/cjc-mir-exec/` | [[cjc-mir-exec]] |
| Dispatch | `crates/cjc-dispatch/` | [[Dispatch Layer]] |

## Lexer (`cjc-lexer`)

- `crates/cjc-lexer/src/lib.rs` — `Lexer::new(src).tokenize() -> (Vec<Token>, DiagnosticBag)`.
- Token kinds are enumerated here; determinism is trivial at this stage.

## Parser (`cjc-parser`)

- `crates/cjc-parser/src/lib.rs` — `parse_source(src)` convenience that does lex + parse.
- `Parser::new(tokens).parse_program() -> (Program, DiagnosticBag)`.
- Pratt-style operator precedence; Rust-inspired statement syntax.

## AST (`cjc-ast`)

- `crates/cjc-ast/src/lib.rs` — `Expr`, `ExprKind`, `Stmt`, `Item`, `FieldDecl { default: Option<Expr>, .. }`.
- `FieldDecl.default` is load-bearing for struct literal defaults — don't forget in test helpers.

## Type checker (`cjc-types`)

- `crates/cjc-types/src/lib.rs` — the `Type` enum and inference engine.
- Shape inference (S3-P1-08) is planned here; error codes E0500/E0501/E0502 are reserved.
- `Any` type annotation used for dynamic/polymorphic values.

## Diagnostics (`cjc-diag`)

- `crates/cjc-diag/src/lib.rs` — `Diagnostic`, `DiagnosticBag`, `Severity`, error code ranges.
- Error codes organized as E0001..E8xxx; see [[Error Codes]].

## HIR (`cjc-hir`)

- `crates/cjc-hir/src/lib.rs` — `AstLowering`, capture analysis.
- Closure captures are resolved here before reaching MIR. See [[Capture Analysis]].

## MIR (`cjc-mir`)

- `crates/cjc-mir/src/lib.rs` — entry point and re-exports.
- `crates/cjc-mir/src/lower.rs` — `HirToMir` lowering.
- `crates/cjc-mir/src/cfg.rs` — `MirCfg`, dominator tree (S3-P1-06 extends this). See [[CFG]], [[Dominator Tree]].
- `crates/cjc-mir/src/optimize.rs` — CF, SR, DCE, CSE, LICM, SCCP, SSA-DCE, CFG-cleanup passes. See [[MIR Optimizer]].
- `crates/cjc-mir/src/nogc_verify.rs` — the [[NoGC Verifier]] call-graph fixpoint.
- Escape analysis lives alongside; classifies allocations as Stack / Arena / Rc. See [[Escape Analysis]].

## AST interpreter (`cjc-eval`)

- `crates/cjc-eval/src/lib.rs` — `Interpreter::new(seed).exec(&program) -> EvalResult`.
- This is the **reference semantics**. If anything disagrees with eval, eval wins until proven otherwise. See [[cjc-eval]].

## MIR executor (`cjc-mir-exec`)

- `crates/cjc-mir-exec/src/lib.rs` — `run_program_with_executor(program, seed)`, `run_program_optimized`, `verify_nogc`.
- Register machine walking MIR; tied for semantics with `cjc-eval` via [[Parity Gates]].

## Dispatch layer (`cjc-dispatch`)

- `crates/cjc-dispatch/src/lib.rs` — operator routing to typed kernels.
- Both executors go through this for consistency.

## Key cross-cutting files

| Concept | File |
|---|---|
| Add a new builtin | `cjc-runtime/src/builtins.rs` + `cjc-eval/src/lib.rs` + `cjc-mir-exec/src/lib.rs` ([[Wiring Pattern]]) |
| Add a new operator | update AST, Type checker, HIR, MIR lowering, both executors |
| Add a new Value variant | `cjc-runtime/src/lib.rs` (`Value` enum) + both executor match arms |
| Deterministic RNG | `cjc-repro/src/rng.rs` (SplitMix64) |
| Deterministic reduction | `cjc-repro/src/accumulator.rs` (Kahan / Binned) |
| Parity gate tests | `tests/milestone_2_4/parity/` |

## Related

- [[Compiler Architecture]]
- [[Wiring Pattern]]
- [[Compiler Concept Graph]]
