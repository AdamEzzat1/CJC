---
title: Variadic Functions
tags: [language, implemented]
status: Implemented (verified 2026-04-09 in both executors)
---

# Variadic Functions

CJC-Lang supports variadic parameters using the `...` prefix. A variadic parameter collects any remaining positional arguments into a deterministic array that is bound to the parameter name inside the function body.

## Syntax

```cjcl
fn sum(...values: f64) -> f64 {
    let total = 0.0;
    let i = 0;
    while i < len(values) {
        total = total + values[i];
        i = i + 1;
    }
    total
}

print(sum(1.0, 2.0, 3.0, 4.0));   // 10
print(sum());                     // 0 — zero-arg variadic is legal
```

Variadics may follow regular positional parameters but **must be the last parameter**:

```cjcl
fn format_with(prefix: str, ...args: Any) -> str { prefix }
fn first_plus_rest(first: f64, ...rest: f64) -> f64 { ... }
```

## Rules

- The variadic parameter is always last. A non-variadic parameter after a variadic is a parse error.
- Variadic parameters **cannot have a default value** — `fn bad(...args: f64 = 1.0)` is rejected by the parser. This is enforced by `tests/test_variadic.rs::parse_variadic_default_rejected`.
- A variadic call with zero trailing arguments binds an empty array, not `null` — `len(args)` returns 0.
- Element type follows the annotation: `...values: f64` means every collected argument must be an `f64`. Use `...args: Any` for heterogeneous collection.
- Inside the function the variadic name behaves exactly like a normal array — indexable, `len(...)`-queryable, iterable via `for … in`.

## Implementation surface

The feature is wired through the full pipeline per the [[Wiring Pattern]]:

| Layer | File | Note |
|---|---|---|
| Lexer | `cjc-lexer/src/lib.rs` | `TokenKind::DotDotDot` distinguishes `...` from `..` (range operator) |
| Parser | `cjc-parser/src/lib.rs:717` | `let is_variadic = self.eat(TokenKind::DotDotDot).is_some();` inside parameter parsing |
| AST | `cjc-ast/src/lib.rs:296-298` | `Param::is_variadic: bool` field |
| HIR / MIR | `cjc-hir`, `cjc-mir` | Variadic params lower to an array parameter; the call-site materializes the trailing arguments into a fresh array value |
| Eval (v1) | `cjc-eval/src/lib.rs` | Reads `is_variadic` on the callee, packs the trailing argument list into a `Value::Array` before binding |
| MIR-exec (v2) | `cjc-mir-exec/src/lib.rs` | Mirrors eval: call-site gather, single-register bind |

## Test coverage

`tests/test_variadic.rs` — **11 tests, all passing** (`cargo test --test test_variadic --release`).

Covers:

- Lexer: `...` vs `..` disambiguation
- Parser: variadic alone, with leading params, default-rejection
- Eval + MIR-exec parity on: non-empty call, zero-arg call, single-arg call, leading-param + variadic
- Fixed output strings between the two executors

## Deterministic allocation

The collected array is built in argument order using `cjc-runtime`'s COW array machinery, so a call with the same arguments always produces a bit-identical array. There is no hidden sort, hash, or heap-ordering step — see [[Determinism Contract]].

## Related

- [[Expressions and Statements]]
- [[Syntax]]
- [[Wiring Pattern]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Parity Gates]]
