---
title: Default Parameters
tags: [language, implemented]
status: Implemented (v0.1.7, 2026-04-19)
---

# Default Parameters

Function parameters may carry default values. A call that omits a trailing argument uses the default expression, evaluated in the **caller's scope** at call time.

## Syntax

```cjcl
fn solve(x: f64, tol: f64 = 1e-6) -> f64 {
    // ...
}

solve(3.0);         // tol = 1e-6 (default used)
solve(3.0, 1e-9);   // tol = 1e-9 (overridden)
```

Any parameter may have a default; positional parameters with defaults must appear after positional parameters without defaults in the normal Python-style ordering.

## Rules

- The default expression is evaluated **once per call**, in the caller's scope (not captured at definition time).
- A variadic parameter **cannot have a default** — `fn bad(...args: f64 = 1.0)` is a parse error. This mirrors the existing [[Variadic Functions]] restriction.
- Type annotations on the parameter are still required — defaults do not relax that rule. See [[Syntax]].
- Defaults flow through both executors identically — see [[Parity Gates]].

## Implementation surface

Feature wired through the full pipeline ([[Wiring Pattern]]):

| Layer | File | Note |
|---|---|---|
| AST | `cjc-ast/src/lib.rs` | `Param::default: Option<Expr>` — field already existed; parser path newly accepts `=` after the type annotation |
| Parser | `cjc-parser/src/lib.rs` | Accepts `identifier : Type = Expr` in parameter position |
| HIR | `cjc-hir/src/lib.rs:945` | Default expressions lowered and stored alongside the parameter |
| MIR | `cjc-mir/src/lib.rs:671` | Call-site lowering inserts the default expression for omitted trailing arguments |
| Eval (v1) | `cjc-eval/src/lib.rs` | Evaluates the default in the caller's scope before binding |
| MIR-exec (v2) | `cjc-mir-exec/src/lib.rs` | Mirrors eval semantics — both executors produce identical bindings |

## Test coverage

`tests/test_defaults.rs` — **32 tests, all passing**.

Covers basic defaults, multiple defaults, interaction with positional args, type-checking of the default expression, variadic-rejection parser test, and parity between [[cjc-eval]] and [[cjc-mir-exec]].

## Related

- [[Syntax]]
- [[Variadic Functions]]
- [[Wiring Pattern]]
- [[Parity Gates]]
- [[Expressions and Statements]]
- [[Version History]]
