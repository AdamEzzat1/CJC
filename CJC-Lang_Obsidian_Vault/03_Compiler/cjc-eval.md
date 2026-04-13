---
title: cjc-eval
tags: [compiler, runtime, executor]
status: Implemented
---

# cjc-eval

**Crate**: `cjc-eval` — `crates/cjc-eval/src/lib.rs` (~5,300 LOC).

## Summary

The **v1 tree-walking interpreter**. Takes a typed [[AST]] and walks it to produce a result. Simpler than [[cjc-mir-exec]], used as the reference semantics for [[Parity Gates]].

## Public API

```rust
let result = Interpreter::new(seed).exec(&program);
```

Returns an `EvalResult` with the final value or an error. The `seed` threads through [[SplitMix64]] so all RNG use is reproducible.

## Execution model

- Maintains a scope chain of variable bindings.
- Dispatches every operator and builtin call through [[cjc-dispatch]] and `crates/cjc-runtime/src/builtins.rs`.
- Closures are resolved against the `HirCapture` info created by [[Capture Analysis]] — even though this interpreter consumes AST, it reuses the same capture metadata as MIR-exec to preserve parity.

## Performance characteristics

Tree-walking is 10–100× slower than native compiled code. This is **accepted** — the interpreter exists for:
- Reference semantics (what the language *means*)
- Development convenience (fast REPL roundtrips)
- Parity verification against [[cjc-mir-exec]]

When you want speed, use `cjcl run --mir-opt`.

## Parity

Every test in [[Parity Gates]] runs the program through both cjc-eval and cjc-mir-exec and asserts byte-identical output. This is the strongest correctness invariant in the compiler.

## Related

- [[cjc-mir-exec]]
- [[Parity Gates]]
- [[Dispatch Layer]]
- [[Builtins Catalog]]
- [[AST]]
