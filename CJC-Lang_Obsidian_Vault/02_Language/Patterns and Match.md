---
title: Patterns and Match
tags: [language, patterns]
status: Implemented
---

# Patterns and Match

`match` was delivered in **Stage 2.2** along with structural destructuring. It works with tuples, structs, and enum variants.

```cjcl
enum Shape {
    Circle(f64),
    Rect(f64, f64),
}

fn area(s: Shape) -> f64 {
    match s {
        Circle(r) => 3.14159 * r ** 2.0,
        Rect(w, h) => w * h,
        _ => 0.0,
    }
}
```

## Supported patterns

- Literal patterns (`1`, `"hello"`, `true`)
- Identifier binds (`x`)
- Wildcard (`_`)
- Tuple destructuring `(a, b, c)`
- Struct destructuring `Point { x, y }`
- Enum variant patterns `Circle(r)`, `Rect(w, h)`

## Exhaustiveness

The [[Type Checker]] is expected to detect non-exhaustive matches; **needs verification** that this is fully wired for user-defined enums in v0.1.4. The error-code registry in `docs/spec/error_codes.md` reserves space in E01xx for type-system diagnostics.

## Parity

Match expressions are executed by both [[cjc-eval]] and [[cjc-mir-exec]]. Destructuring is lowered in [[HIR]] into a sequence of checked binds, and then into register-level tests + jumps in [[MIR]]. The parity tests in [[Parity Gates]] cover both.

## Related

- [[Syntax]]
- [[Types]]
- [[Type Checker]]
- [[HIR]]
- [[MIR]]
- [[Parity Gates]]
