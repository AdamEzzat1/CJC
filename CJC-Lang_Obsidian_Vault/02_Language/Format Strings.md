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

## Determinism implications

Float formatting is a source of non-determinism across platforms. [[Vizor]] documents in `docs/vizor/DETERMINISM.md` that it uses `{:.2}` fixed-width formatting to guarantee identical SVG output across backends and platforms. The same discipline applies anywhere format strings touch floats that must be reproducible. See [[Numerical Truth]].

## Related

- [[Syntax]]
- [[Lexer]]
- [[Numerical Truth]]
