# Changelog

All notable changes to CJC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-03-27

### Added

- **Core language:** functions with type annotations, closures, while/for loops, if/else (statement and expression), pattern matching, structs, enums, traits, generics
- **Type system** with inference, 25+ value types, and generic support
- **Dual execution:** AST tree-walk interpreter (`cjc-eval`) and MIR register-machine executor (`cjc-mir-exec`) with full parity
- **MIR optimization pipeline:** constant folding, strength reduction, DCE, CSE, LICM
- **SSA construction** (Cytron minimal) with 6 SSA optimization passes
- **Loop analysis**, reduction analysis, and legality verification infrastructure
- **282+ builtin functions** across math, string, array, tensor, statistics, and I/O
- **Deterministic execution:** SplitMix64 RNG, Kahan/Binned accumulators, BTreeMap everywhere — same seed = bit-identical output
- **Automatic differentiation:** forward-mode dual numbers and reverse-mode tape with gradient graphs
- **DataFrame library** (tidyverse-inspired): filter, select, mutate, group_by, summarize, join
- **Tensor operations** with SIMD acceleration (no FMA for bit-reproducibility)
- **Sparse linear algebra:** CSR format, SpMV, basic iterative solvers
- **NFA-based regex engine** (zero external dependencies)
- **Binary serialization** (`cjc-snap`)
- **Grammar-of-graphics visualization** (`cjc-vizor`)
- **NoGC verification** pass for allocation-free code paths
- **Escape analysis** for automatic memory tier selection (Stack/Arena/Rc)
- **CLI** with run, repl, check, lex, parse, ast, hir, mir, inspect, schema, bench, and diagnostic commands
- **5,600+ tests** with determinism verification

### Known Limitations

- Single-file programs only (module system is incomplete)
- Visibility modifiers (`pub`/`priv`) parsed but not enforced
- Parser recovery limited to top-level declarations
- Range type exists in type system but no `Value::Range` at runtime
- Some CLI subcommands are stubs (marked experimental)
