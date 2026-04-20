---
title: Syntax
tags: [language, syntax]
status: Implemented
---

# Syntax

CJC-Lang's syntax is intentionally modest. The surface is small; the depth comes from the standard library and the determinism properties of the runtime. This note summarizes the current surface syntax grounded in `docs/SYNTAX.md`, `docs/CJC_Syntax_V0.1.md`, and the README.

## Source file basics

- **Extension**: `.cjcl` (was `.cjc` before v0.1.4 — see [[Version History]]).
- **Comments**: `//` line comments.
- **Keywords**: ~26 including `fn`, `let`, `mut`, `if`, `else`, `while`, `for`, `in`, `return`, `struct`, `enum`, `class`, `trait`, `match`, `true`, `false`, `nil`, `mod`, `import`, `pub`, `Any`.
- **Case**: identifiers are case-sensitive; snake_case by convention for functions and variables.

## Function declarations

Parameters **require** type annotations:

```cjcl
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

Without a return type, the function is implicitly `()` (unit). The last expression in a block is the return value — see [[Expressions and Statements]].

**Default parameters** (shipped v0.1.7):

```cjcl
fn solve(x: f64, tol: f64 = 1e-6) -> f64 { ... }

solve(3.0);        // tol defaults to 1e-6
solve(3.0, 1e-9);  // tol overridden
```

Defaults are evaluated in the caller's scope on each call and lower through AST → HIR → MIR into a default-insertion step at the call site. Variadic parameters **cannot** have a default value — `fn bad(...args: f64 = 1.0)` is a parse error. See [[Default Parameters]].

**Variadic parameters** (`fn f(...args: f64)`) — shipped. See [[Variadic Functions]].

**Not yet supported:**
- Decorators (`@log fn train(...)`) — [[Roadmap]].

## Variables

```cjcl
let x: i64 = 42;
let pi: f64 = 3.14159;
let name: str = "CJC-Lang";
let flag: bool = true;
let mut counter: i64 = 0;
counter += 1;
```

Immutable by default. `let mut` opts into rebinding/assignment.

## Structs and Enums

```cjcl
struct Point { x: f64, y: f64 }

enum Shape {
    Circle(f64),
    Rect(f64, f64),
}
```

Structs support `FieldDecl { default: Option<Expr> }` — the AST carries default field values, though the surface syntax for using them is part of ongoing work. Enums are tagged unions and are the target of [[Patterns and Match]].

## Control flow

```cjcl
if x > 0 { print("positive"); } else { print("non-positive"); }

while count < 100 { count += 1; }

for i in 0..10 { print(i); }
```

`if` works as **both a statement and an expression** (verified in both executors):

```cjcl
let x: i64 = if 1 < 2 { 10 } else { 20 };
```

See [[If as Expression]].

**Syntactic rule**: no semicolons after `while {}`, `if {}`, or `for {}` blocks inside function bodies.

## Literals

| Kind | Example |
|---|---|
| Integer | `42`, `-1`, `0xFF` |
| Float | `3.14`, `1e-6` |
| String | `"hello"` |
| Format string | `f"mean: {mean_val}"` |
| Raw string | `r"\d+"` |
| Byte string | `b"\x00\x01"` |
| Regex | `/\d+/gi` — see [[Regex Engine]] |
| Bool | `true`, `false` |
| Array | `[1, 2, 3]` |
| Tensor | `[| 1.0, 2.0; 3.0, 4.0 |]` |
| Tuple | `(1, "a", 3.14)` |

Tensor literals use `[| ... |]` with `;` separating rows.

## Operators

See [[Operators and Precedence]].

- Arithmetic: `+ - * / % **`
- Comparison: `== != < > <= >=`
- Logical: `&& || !`
- Bitwise: `& | ^ ~ << >>`
- Pipe: `data |> transform() |> output()`
- Regex match: `~=` and `!~`
- Assignment: `=` and compound assigns `+= -= *= /=`

## Closures

```cjcl
let double = |x: i64| x * 2;
apply(double, 21)
```

Closures capture by analysis — see [[Capture Analysis]]. The surface supports parameter types; return types are inferred. **Note**: CLAUDE.md records that inline anonymous function literals are not fully supported in all contexts — "must use named functions" in some call sites. **Needs verification** for exact limits.

## Modules

```cjcl
mod math;
import stats.linear
```

The module surface exists but is not the default path; see [[Module System]] and [[Current State of CJC-Lang]].

## Type annotations and `Any`

Use `Any` when you need dynamic/polymorphic typing:

```cjcl
fn dispatch(x: Any) -> Any { ... }
```

See [[Types]].

## Related

- [[Types]]
- [[Expressions and Statements]]
- [[Patterns and Match]]
- [[Closures]]
- [[Operators and Precedence]]
- [[Format Strings]]
- [[Lexer]]
- [[Parser]]
