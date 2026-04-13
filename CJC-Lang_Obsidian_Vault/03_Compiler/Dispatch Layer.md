---
title: Dispatch Layer
tags: [compiler, runtime]
status: Implemented
---

# Dispatch Layer

**Crate**: `cjc-dispatch` — `crates/cjc-dispatch/src/lib.rs` (~694 LOC).

## Summary

Multi-dispatch by type for operators and builtin function calls. Given a call site and the runtime types of its arguments, pick the most specific registered implementation.

## Specificity ranking

```
None < Generic < Constrained < Concrete
```

Used to order candidate implementations. The dispatcher picks the highest-specificity match and errors on ambiguous ties via a `CoherenceChecker`.

## Why it exists

Both executors need to resolve calls like `a + b` or `sum(xs)` to a concrete implementation. Without a shared dispatch layer, [[cjc-eval]] and [[cjc-mir-exec]] could drift and break [[Parity Gates]]. The dispatcher is the single source of truth.

## How builtins plug in

The shared entry point is `cjc-runtime::builtins` — a stateless dispatch of builtin names to implementations. Both executors call into it. See [[Wiring Pattern]] and [[Builtins Catalog]].

## Related

- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Builtins Catalog]]
- [[Wiring Pattern]]
- [[Parity Gates]]
