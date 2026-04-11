---
title: Parser
tags: [compiler, frontend]
status: Implemented
---

# Parser

**Crate**: `cjc-parser` — `crates/cjc-parser/src/lib.rs` (~3,142 LOC).

## Summary

Pratt parser producing an `Ast::Program`. Operator precedence is expressed as binding powers per `TokenKind`, which is what makes adding operators cheap.

## Public API

```rust
// Explicit:
let (tokens, diags) = Lexer::new(src).tokenize();
let (program, diags) = Parser::new(tokens).parse_program();

// Convenience:
let (program, diags) = cjc_parser::parse_source(src);
```

Both return `(Program, DiagnosticBag)`.

## What it produces

The parser builds the [[AST]] defined in [[cjc-ast]]. Nodes include:
- `Program { decls: Vec<Decl> }`
- `Decl` / `DeclKind` — `fn`, `struct`, `enum`, `class`, `trait`, `mod`, `import`
- `Stmt` / `StmtKind`
- `Expr` / `ExprKind`
- `Pattern`, `TypeExpr`, `Span`

## Error recovery

The parser has synchronization points so that after a parse error it can skip forward to the next statement or declaration boundary and continue. This is what makes it possible for IDE scenarios (and [[cjc-analyzer]]) to show multiple errors at once.

## Not currently supported

- **Default parameters** — grammar does not accept `fn f(x: f64, tol: f64 = 1e-6)`.
- **Variadic parameters** — grammar does not accept `...args`.
- **Decorators** — no `@name fn ...` syntax yet.
- **`if` as expression** — expression position, e.g., `let x = if c { a } else { b }`, is not accepted.

All of these are on [[Roadmap]].

## Source

- `crates/cjc-parser/src/lib.rs` — main parser
- precedence module — Pratt binding powers

## Related

- [[Lexer]]
- [[AST]]
- [[Syntax]]
- [[Operators and Precedence]]
- [[Type Checker]]
