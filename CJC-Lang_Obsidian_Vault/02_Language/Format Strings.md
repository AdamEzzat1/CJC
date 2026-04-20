---
title: Format Strings
tags: [language, strings]
status: Implemented
---

# Format Strings

CJC-Lang supports Python-style f-strings:

```cjcl
let name = "Ada";
print(f"Hello, {name}!");

let mean_val: f64 = mean(data);
let std_val: f64 = sd(data);
print(f"mean: {mean_val}, std: {std_val}");
```

They are lexed as a distinct token form by [[cjc-lexer]] and expanded in the [[Parser]] or lowered in [[HIR]] into a call that concatenates with an interpolated value stream.

## Pipeline (v0.1.7)

| Layer | Representation |
|---|---|
| Lexer | `TokenKind::FStringLit` — the whole form arrives as one token whose payload is a segment list |
| Parser | `ExprKind::FStringLit(Vec<(String, Option<Box<Expr>>)>)` — each tuple is `(literal_chunk, optional_interpolation)` |
| HIR | Desugars into a `BinOp::Add(Str, Str)` chain, materializing each interpolated value via the string-conversion path |
| MIR / MIR-exec | No direct handler — the HIR desugaring means MIR sees only ordinary `Add` nodes. This is why parity is automatic. |

Interpolation expressions may be arbitrary, not just identifiers:

```cjcl
let x: i64 = 41;
print(f"x+1 is {x + 1}");
```

`tests/test_fstring.rs` — **21 tests, all passing**.

## Determinism implications

Float formatting is a source of non-determinism across platforms. [[Vizor]] documents in `docs/vizor/DETERMINISM.md` that it uses `{:.2}` fixed-width formatting to guarantee identical SVG output across backends and platforms. The same discipline applies anywhere format strings touch floats that must be reproducible. See [[Numerical Truth]].

## Related

- [[Syntax]]
- [[Lexer]]
- [[Numerical Truth]]
