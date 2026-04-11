---
title: HIR
tags: [compiler, ir]
status: Implemented
---

# HIR

**Crate**: `cjc-hir` — `crates/cjc-hir/src/lib.rs` (~2,793 LOC).

## Summary

High-level Intermediate Representation. The AST is lowered into HIR by `AstLowering`, which desugars high-level constructs and runs [[Capture Analysis]]. HIR is still tree-shaped but simpler than AST: `for` is gone, pipes are gone, closures carry explicit captures.

## Key types

- `HirProgram`, `HirItem`, `HirFn`
- `HirExpr` / `HirExprKind`
- `HirStmt` / `HirStmtKind`
- `HirCapture` with a `CaptureMode` (at least `ByRef`, `ByClone`)

## Desugarings applied

| From | To |
|---|---|
| `x \|> f(y) \|> g()` | `g(f(x, y))` |
| `for i in 0..10 { ... }` | `let i = 0; while i < 10 { ... i += 1; }` (shape — exact form TBD in source) |
| Closures | `HirFn` items + `HirCapture` list |
| `if cond { ... }` block | If expression node (when `if` becomes expression — **Planned**) |

## Capture analysis

Each closure is analyzed to determine which locals it reads and writes. The result is a `HirCapture` list that both executors can use to build closure values without re-running analysis. See [[Capture Analysis]].

## Why HIR exists

HIR sits between AST and MIR to give a layer where:
- Desugarings can happen without touching the parser or AST shape
- Semantic analysis that crosses many nodes is easier (the tree is simpler)
- [[MIR]] lowering can focus on control-flow flattening and SSA, not on what `|>` means

## Related

- [[AST]]
- [[MIR]]
- [[Capture Analysis]]
- [[Closures]]
