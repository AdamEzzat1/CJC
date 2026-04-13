---
title: Expressions and Statements
tags: [language, semantics]
status: Implemented
---

# Expressions and Statements

CJC-Lang distinguishes statements from expressions, but some constructs are expression-like: blocks implicitly return their last expression, so function bodies commonly end without `return`.

```cjcl
fn square(x: f64) -> f64 {
    x * x   // no `return`, no semicolon — this is the block's value
}
```

## Statement kinds

- `let` bindings
- Assignments
- `while` loops
- `for` loops (desugared to `while` in [[HIR]])
- `if` / `else` (currently a **statement**, planned to become an expression)
- Expression statements (an expression followed by `;`)
- `return`

## Expression kinds

- Literals (see [[Syntax]])
- Identifiers
- Function calls
- Method calls (`.sum()`, `.transpose()`)
- Unary and binary operators
- Closures
- Block expressions `{ ... }`
- `match` expressions (expression form — see [[Patterns and Match]])
- Pipe expressions: `data |> f() |> g()`
- Indexing: `arr[i]`, `tensor[i, j]`
- Field access: `p.x`

## The `if` expression question

Currently in CJC-Lang:

```cjcl
let x = if cond { a } else { b };   // NOT yet supported
```

is rejected. `if` is a statement. Making it an expression is one of the explicit priority features in the CLAUDE.md prompt and in [[Roadmap]]. When implemented, it will require:
- New `ExprKind::IfExpr` in [[AST]]
- Type unification of both arms in [[Type Checker]]
- HIR lowering update in [[HIR]]
- MIR lowering update in [[MIR]]
- Execution support in both [[cjc-eval]] and [[cjc-mir-exec]]
- Parity tests in [[Parity Gates]]

## Pipe operator

```cjcl
data |> filter(col("x") > 0) |> group_by("k") |> summarize(mean)
```

The pipe is desugared in [[HIR]] into nested function calls: `g(f(x))` becomes `x |> f |> g`. This keeps the runtime unchanged and makes data pipelines readable.

## `array_push` semantics

A quirk documented in CLAUDE.md: `array_push(arr, val)` returns a **new** array rather than mutating in place. Idiomatic use is:

```cjcl
arr = array_push(arr, val);
```

This is consistent with the COW buffer model — see [[Memory Model]] and [[COW Buffers]].

## Related

- [[Syntax]]
- [[Patterns and Match]]
- [[HIR]]
- [[Closures]]
- [[Memory Model]]
