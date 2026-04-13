---
title: AST
tags: [compiler, ir]
status: Implemented
---

# AST

**Crate**: `cjc-ast` — `crates/cjc-ast/src/lib.rs` (~4,103 LOC). This is a **leaf crate** with zero dependencies on other compiler crates. Everything downstream consumes it.

## Summary

Plain tree form of a CJC-Lang program, emitted by the [[Parser]] and consumed by [[Type Checker]], [[HIR]], and [[cjc-eval]].

## Key types

- `Program { decls: Vec<Decl> }`
- `Decl { kind: DeclKind, span: Span }` where `DeclKind` covers `Fn`, `Struct`, `Enum`, `Class`, `Trait`, `Mod`, `Import`
- `Stmt` / `StmtKind` — `Let`, `Assign`, `ExprStmt`, `While`, `For`, `If`, `Return`
- `Expr` / `ExprKind` — literals, binary, unary, call, method call, block, closure, match, pipe, index, field, tensor literal, ...
- `Pattern` — literal, identifier, tuple, struct, enum variant, wildcard
- `TypeExpr` — annotations in source form (resolved by [[Type Checker]])
- `Span` — byte range in source; also in [[cjc-diag]]
- `FieldDecl { name, ty, default: Option<Expr> }` — note the default field, used even though the surface form for using it is pending.

## Visitor pattern

`cjc-ast` provides a visitor infrastructure used by multiple passes (HIR lowering, structural validation, metrics). Visitors are explicit, not macro-generated, which keeps them debuggable.

## Metrics and validation

The crate includes:
- Node counting and feature detection (e.g., "does this program use closures?")
- Structural validation (e.g., no orphan statements)

These feed into CLI commands like `cjcl inspect` — see [[CLI Surfaces]].

## What is NOT here

- Type information — added by [[Type Checker]] (which annotates, it does not rewrite).
- Desugaring — done in [[HIR]].

## Related

- [[Parser]]
- [[Type Checker]]
- [[HIR]]
- [[cjc-eval]]
