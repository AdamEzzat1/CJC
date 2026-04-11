---
title: TCO Extension
tags: [compiler, planned]
status: Planned (roadmap S3-P1-07)
---

# TCO Extension

Planned extension to [[cjc-mir-exec]]'s tail-call optimization to cover conditional branches and mutual recursion.

## Current coverage

The MIR executor currently supports **direct** tail calls: `return f(args)` and body-expression `f(args)`. This unrolls a simple recursion loop like `countdown(n) = if n == 0 then 0 else countdown(n - 1)` into a trampoline.

## Extended coverage (S3-P1-07)

- **Conditional branches in tail position**: `if c { f(n-1) } else { g(n-1) }` — both arms are tail calls.
- **Mutual recursion**: `even(n) -> odd(n-1)` and `odd(n) -> even(n-1)` — both functions must be TCO'd.

## Enabling pattern

`exec_body` is called for each branch body. Tail calls in branch result expressions propagate `MirExecError::TailCall` up to the trampoline rather than being executed as normal calls.

## Tests planned

From `docs/spec/stage3_roadmap.md` §S3-P1-07:

- `tco_mutual_recursion_even_odd` — `even(1_000_000)` / `odd(1_000_000)` without stack overflow.
- `tco_conditional_branch_tail_call` — `count_down(500_000)` via `if`/`else`.
- `tco_if_else_both_tail_calls` — Collatz sequence from 27 (111 steps).

## Success criteria

- `even(1_000_000)` completes without stack overflow.
- All existing 5 TCO tests still pass.
- `cargo test test_phase2_tco` — 0 failures.

## Related

- [[cjc-mir-exec]]
- [[Roadmap]]
- [[MIR]]
