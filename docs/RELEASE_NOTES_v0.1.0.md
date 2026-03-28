# CJC v0.1.0 — Release Notes

## What is CJC?

CJC is a deterministic numerical programming language designed for reproducible computation, statistical analysis, and ML pipelines. Written in Rust with zero external runtime dependencies, CJC guarantees that the same program with the same seed produces bit-identical output across runs and platforms.

## Key Features

- **Deterministic by design** — SplitMix64 RNG, Kahan/Binned summation, BTreeMap-only collections. No hidden non-determinism.
- **Dual execution engine** — AST tree-walk interpreter for rapid prototyping; MIR register-machine executor with SSA optimizations for performance.
- **282+ builtins** — Math, statistics, linear algebra, string processing, tensors, DataFrames, I/O, and more.
- **Automatic differentiation** — Forward-mode (dual numbers) and reverse-mode (tape) AD for ML training.
- **Tidyverse-inspired data library** — `filter()`, `select()`, `mutate()`, `group_by()`, `summarize()`, `join()` on DataFrames.
- **SIMD-accelerated tensors** — No FMA (fused multiply-add) to preserve bit-reproducibility.
- **Zero external runtime dependencies** — Regex, serialization, RNG, and linear algebra are all built from scratch.
- **5,600+ tests** — Including determinism verification and executor parity checks.

## Getting Started

1. **Install** (from crates.io):
   ```bash
   cargo install cjc-cli
   ```

2. **Write** a CJC program (`hello.cjc`):
   ```
   fn greet(name: String) {
       print("Hello, " + name + "!");
   }
   greet("world");
   ```

3. **Run** it:
   ```bash
   cjc run hello.cjc
   ```

   Or use the REPL:
   ```bash
   cjc repl
   ```

## Known Limitations

- **Single-file only** — The module system (`mod`, `import`) is parsed but not fully functional.
- **Visibility not enforced** — `pub`/`priv` are parsed but all items are treated as public.
- **Parser recovery** — Syntax errors mid-function lose the rest of that function body.
- **Some CLI subcommands are experimental** — `mem`, `trace` are stubs.

## What's Next (v0.2.0 Roadmap)

- Multi-file module system with proper visibility enforcement
- Package manager (ADR-0013: git-based pinning with deterministic resolution)
- Code formatter (`cjc fmt`)
- Statement-level parser error recovery
- Gradual `unwrap()` → `Result<>` migration in public APIs
