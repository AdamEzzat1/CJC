---
title: CLI Surfaces
tags: [cli, tooling]
status: Implemented
---

# CLI Surfaces

**Crate**: `cjc-cli` — `crates/cjc-cli/src/lib.rs` (~1,298 LOC), with per-command modules under `crates/cjc-cli/src/commands/`.

## Binary name

`cjcl` (was `cjc` before v0.1.4). Install with `cargo install cjc-lang`.

## Core execution commands

| Command | What it does |
|---|---|
| `cjcl run <file.cjcl>` | Run a program |
| `cjcl repl` | Interactive REPL — see [[REPL]] |
| `cjcl lex <file.cjcl>` | Tokenize and print tokens |
| `cjcl parse <file.cjcl>` | Parse and pretty-print AST |
| `cjcl check <file.cjcl>` | Type-check without executing |

## Flags

| Flag | Description |
|---|---|
| `--seed N` | RNG seed (default: 42) |
| `--mir-opt` | Enable MIR optimizations |
| `--mir` | Use [[cjc-mir-exec]] backend |
| `--time` | Print execution time |
| `--color` / `--no-color` | Control color output |

## Extended commands (Phase 1–3, 26 command modules)

The `crates/cjc-cli/src/commands/` directory contains:

**Execution + data + reproducibility**
- `view` — display program output / values
- `proof` — deterministic execution proof
- `flow` — control flow analysis
- `patch` — source transformation
- `seek` — search / locate
- `drift` — differential testing between versions
- `forge` — generate test fixtures

**Inspection + observability + validation**
- `inspect` — runtime introspection
- `schema` — type schema extraction
- `check2` — alternate type check
- `trace` — execution tracing
- `mem` — memory usage
- `bench` — micro-benchmark
- `pack` — binary packaging (via [[Binary Serialization]])
- `doctor` — environment diagnostics

**Compiler visibility + numerical + CI**
- `emit` — MIR emission (calls `crates/cjc-mir/src/inspect.rs`)
- `explain` — error-code explanation (looks up [[Error Codes]])
- `gc`, `nogc` — GC inspection / NoGC mode validation
- `audit` — codebase audit runner
- `precision` — numerical precision analysis
- `lock` — determinism locking
- `parity` — [[Parity Gates]] verification runner
- `test_cmd` — test execution framework
- `ci` — CI/CD pipeline integration

## Entry point

`crates/cjc-cli/src/main.rs` is the binary entrypoint. `lib.rs` wires everything up via a uniform `parse_args → run → print_help` dispatch pattern.

## Related

- [[REPL]]
- [[Language Server]]
- [[Parity Gates]]
- [[Error Codes]]
- [[Binary Serialization]]
- [[MIR]]
