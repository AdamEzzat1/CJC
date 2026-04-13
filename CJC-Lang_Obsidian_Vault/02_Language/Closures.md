---
title: Closures
tags: [language, closures]
status: Implemented
---

# Closures

Closures were delivered in **Stage 2.1** with [[Capture Analysis]].

```cjcl
fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

fn main() -> i64 {
    let double = |x: i64| x * 2;
    apply(double, 21)   // 42
}
```

## Capture modes

`cjc-hir` defines `HirCapture` with a `CaptureMode` that is at least:
- `ByRef` — captured by reference (typical for numerics)
- `ByClone` — captured by value clone

The [[HIR]] pass runs [[Capture Analysis]] on every closure to determine which locals it reads and how they should be captured. This is what makes closures work in both the tree-walking interpreter and the MIR executor without reallocating captures on every call.

## Limitations observed in CLAUDE.md

> "CJC-Lang doesn't support inline anonymous function literals — must use named functions"

This guidance in CLAUDE.md suggests that in **some call sites** you must use a named `fn` rather than an inline `|x| ...`. Needs verification of the exact restriction. It is likely tied to parser ambiguity in specific positions (e.g., method call chains) rather than a blanket prohibition.

## Closures vs function pointers

CJC-Lang has both:
- `fn(i64) -> i64` — plain function pointer type (no captures)
- Closure values — can carry captured state

The type system distinguishes the two, and [[Dispatch Layer]] handles both.

## Related

- [[Capture Analysis]]
- [[HIR]]
- [[AST]]
- [[Syntax]]
