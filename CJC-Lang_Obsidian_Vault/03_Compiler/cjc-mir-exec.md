---
title: cjc-mir-exec
tags: [compiler, runtime, executor]
status: Implemented
---

# cjc-mir-exec

**Crate**: `cjc-mir-exec` — `crates/cjc-mir-exec/src/lib.rs` (~4,754 LOC).

## Summary

The **v2 register-machine executor**. Takes a `MirProgram` and runs it directly, walking the [[CFG]] and executing SSA-form instructions. Designed for speed relative to [[cjc-eval]] while preserving bit-identical output.

## Public API

```rust
// Standard execution
let result = cjc_mir_exec::run_program_with_executor(&program, seed);

// With MIR optimizer enabled
let result = cjc_mir_exec::run_program_optimized(&program, seed);

// NoGC verification
let result = cjc_mir_exec::verify_nogc(&program);
```

All three take the typed `Program` (the AST), lower it through [[HIR]] → [[MIR]], then execute. The `seed` threads through [[SplitMix64]] RNG use.

## Execution model

- **Register machine**: every SSA variable maps to a register slot.
- **Frame arena**: per-call bump allocator for `Arena`-classified allocations.
- **Tail-call trampoline**: tail calls jump to a new frame without growing the call stack (observed by `crates/cjc-mir-exec/src/lib.rs`).
- **Shared dispatch**: all builtins route through [[cjc-dispatch]] and `cjc-runtime::builtins`, guaranteeing the same semantics as [[cjc-eval]].

## Feature crates

`cjc-mir-exec` depends on many "feature" crates to execute rich operations:
- [[cjc-runtime]] — tensors, numerics, ML
- [[cjc-data]] — DataFrame operations
- [[cjc-regex]] — regex
- [[cjc-module]] — multi-file (when wired)
- [[cjc-ad]] — AD (forward) — MIR integration for reverse AD is **Planned**
- [[cjc-quantum]] — quantum ops (via dispatch)

## Parity guarantee

Every feature must work *identically* in [[cjc-eval]] and cjc-mir-exec. When a new builtin is added, the CLAUDE.md wiring rule says: register it in **three** places — `cjc-runtime/src/builtins.rs`, `cjc-eval`, and `cjc-mir-exec`. See [[Wiring Pattern]].

## Related

- [[MIR]]
- [[MIR Optimizer]]
- [[Parity Gates]]
- [[Frame Arena]]
- [[Memory Model]]
- [[cjc-eval]]
- [[Wiring Pattern]]
