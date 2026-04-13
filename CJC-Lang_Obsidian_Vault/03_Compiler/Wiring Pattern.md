---
title: Wiring Pattern
tags: [compiler, contribution]
status: Implemented convention
---

# Wiring Pattern

A convention documented in CLAUDE.md for how new builtins and operators are added to CJC-Lang without breaking [[Parity Gates]].

## The three-place rule

Every new builtin function must be registered in **three** places:

1. **`cjc-runtime/src/builtins.rs`** — the shared stateless dispatch layer that both executors call.
2. **`cjc-eval/src/lib.rs`** — AST interpreter call handling.
3. **`cjc-mir-exec/src/lib.rs`** — MIR executor call handling.

Every new operator or expression kind must work in **both** executors with identical semantics. If a contributor forgets one of the three places, parity gates catch it the moment the test suite runs.

## Why three places and not one

Ideally you'd register a builtin once and both executors would pick it up. In practice:
- `cjc-eval` consumes AST and needs to know how to evaluate the call from a tree position.
- `cjc-mir-exec` consumes MIR and needs the call to have a stable register-machine lowering.
- `cjc-runtime::builtins` is where the actual semantics live (the body of `sum`, `sin`, `matmul`).

The runtime provides the *what*; each executor provides the *how to reach it*.

## Enforcement

There is no compile-time check that all three places are updated. Enforcement is through:
- [[Parity Gates]] — a missing registration usually breaks parity tests immediately.
- Code review — reviewers look for the three-place pattern.
- The language hardening test suite — stresses every builtin through both executors.

## Related

- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Dispatch Layer]]
- [[Builtins Catalog]]
- [[Parity Gates]]
