---
title: ADR-0003 Backward-compatible run_program
tags: [adr, accepted, executor, api]
status: Accepted
date: 2024-01-20
source: docs/adr/ADR-0003-backward-compatible-run-program.md
---

# ADR-0003 — Backward-compatible `run_program` Entry Points

**Status:** Accepted · **Date:** 2024-01-20

## The decision

The base executor entry point `run_program(program, seed) -> Value` **never changes signature**. New capabilities (optimizer, type checker, monomorphization, NoGC verifier, modules) get **their own sibling functions** rather than adding parameters to the base signature.

Today's entry points in `cjc-mir-exec`:

- `run_program(program, seed)` — base, always stable
- `run_program_optimized(program, seed)` — with MIR optimizer
- `run_program_type_checked(program, seed)` — with type checker
- `run_program_monomorphized(program, seed)` — with monomorphization
- `verify_nogc(program)` — NoGC verifier
- `lower_to_mir(program)` — inspection / debugging
- `run_program_with_modules(entry_path, seed)` — multi-file programs (see [[Module System]])

## Why this matters

- **No signature churn** across 100+ test binaries, fixture runners, and benchmarks.
- **Opt-in capabilities.** Each caller picks exactly the features it wants. The CLI wires `--mir-opt`, `--multi-file`, etc., to the appropriate sibling.
- **Easy to read the call graph.** Grepping for which feature set a test depends on means grepping for the entry-point name, not decoding a bitflag.

## What this constrains

- Every new capability must either piggyback on an existing entry point or add a new sibling. **No silent parameter additions.**
- The base `run_program` must remain the minimal semantics for parity gates against [[cjc-eval]].

## Related

- [[cjc-mir-exec]]
- [[ADR-0001 Tree-form MIR]] — makes siblings cheap because the core walker is shared
- [[Parity Gates]]
- [[ADR Index]]
