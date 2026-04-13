---
title: If as Expression
tags: [language, implemented]
status: Implemented (verified 2026-04-09 in both executors)
---

# If as Expression

`if` is a full expression in CJC-Lang — it returns a value that can be bound, passed as an argument, or used as a function return. Both execution backends ([[cjc-eval]] and [[cjc-mir-exec]]) produce identical results.

## Verified behavior

This program compiles and runs in both executors:

```cjcl
fn main() -> i64 {
    let x: i64 = if 1 < 2 { 10 } else { 20 };
    print(x);
    0
}
```

- `cjcl run if_test.cjcl`                — prints `10`
- `cjcl run if_test.cjcl --mir-opt`      — prints `10`

Verified 2026-04-09 by running the fixture through both backends. Parity holds.

## Semantics

- Both branches must unify to a compatible type (enforced by the type checker).
- `else` is **mandatory** when `if` is used in expression position — otherwise the value is undefined when the condition is false.
- The statement form still works: `if cond { x = a; } else { x = b; }` is valid where no value is needed.
- `if` expressions can appear anywhere an expression can: `let`, arguments, returns, array/tuple literals, struct field initializers.

## Previous confusion

Earlier drafts of `CLAUDE.md` contradicted themselves — the syntax-rules section said "`if` works as BOTH a statement AND an expression" while the feature-scope section listed "`if` AS AN EXPRESSION" as feature #1 still to implement. The feature is in fact already shipped; the feature-scope entry is stale and should be removed. See the historical [[Open Questions]] entry.

## Related

- [[Expressions and Statements]]
- [[Syntax]]
- [[Patterns and Match]]
- [[Parity Gates]]
