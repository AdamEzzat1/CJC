---
title: Regex Engine
tags: [data, runtime]
status: Implemented
---

# Regex Engine

**Crate**: `cjc-regex` — `crates/cjc-regex/src/lib.rs` (~43K, single file).

## Summary

An NFA-based regex engine using Thompson's construction and epsilon-closure simulation. **No backtracking** — which means linear-time matching in the length of the input, and no catastrophic regex cases.

## Why NFA and not backtracking

- **Performance predictability** — O(nm) worst case is predictable; backtracking can be exponential.
- **Determinism** — the simulation is order-independent.
- **Simplicity** — no hash maps in the inner loop.

## Features

- Character classes: `\d`, `\w`, `\s` and their complements
- Custom classes `[abc]`, `[a-z]`
- Alternation `a|b`
- Quantifiers `*`, `+`, `?`, `{n}`, `{n,m}`
- Grouping `(...)`
- Flags: `i` (case-insensitive), `m` (multiline), `s` (dotall), `x` (extended), `g` (global)

## API

- `is_match(pattern, text)`
- `find(pattern, text)`
- `find_all(pattern, text)`
- `split(pattern, text)`

## Surface syntax integration

CJC-Lang has dedicated regex literal syntax: `/pattern/flags`. Plus operators `~=` and `!~`:

```cjcl
if name ~= /^[A-Z][a-z]+$/ { ... }
```

See [[Syntax]] and [[Operators and Precedence]].

## Related

- [[Builtins Catalog]]
- [[DataFrame DSL]]
- [[Syntax]]
