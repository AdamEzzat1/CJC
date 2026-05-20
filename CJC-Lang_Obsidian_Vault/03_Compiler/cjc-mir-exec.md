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

- **Tree-walker** (despite the "register-machine" framing above). `eval_expr`
  recursively walks `MirExpr` nodes via Rust's match dispatch; there is no
  opcode dispatch loop or bytecode. The "register" framing was aspirational.
  See [[ADR-0024 Tier-0 Slot Resolution]] for the architectural discussion
  and the perf programme stacked on this finding ([[Tier-0 Interpreter Perf]]).
- **Frame arena**: per-call bump allocator for `Arena`-classified allocations.
- **Tail-call trampoline**: tail calls jump to a new frame without growing the call stack (observed by `crates/cjc-mir-exec/src/lib.rs`).
- **Shared dispatch**: all builtins route through [[cjc-dispatch]] and `cjc-runtime::builtins`, guaranteeing the same semantics as [[cjc-eval]].
- **Variable resolution** (post Tier-0 T0-b Stage 2): `MirExprKind::VarLocal { name, slot }`
  carries a statically-resolved slot index for function-local references.
  Stage 2 routes both `Var` and `VarLocal` through the same name-lookup
  fallback (or-pattern in every dispatch site — `eval_expr`, `eval_call`,
  `exec_assign`, TCO checks). Stage 3 will switch `VarLocal` reads/writes
  to a flat `Vec<Value>` call frame.

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
- [[Tier-0 Interpreter Perf]]
- [[ADR-0024 Tier-0 Slot Resolution]]
