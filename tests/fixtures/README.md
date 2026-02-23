# CJC End-to-End Fixture Tests

## Overview

This directory contains end-to-end test fixtures for the CJC compiler.
Each fixture consists of a `.cjc` source file and corresponding golden output files.

## Directory Layout

```
tests/fixtures/
  <category>/
    <name>.cjc        — CJC source file (input)
    <name>.stdout     — Expected stdout output (golden)
    <name>.stderr     — Expected error substring (for error test cases)
    <name>.exitcode   — Expected exit code (optional, default: 0)
  runner.rs           — The test runner
  README.md           — This file
```

## Categories

| Directory      | Description                                   |
|---------------|-----------------------------------------------|
| `basic/`      | Hello world, arithmetic, basic expressions    |
| `numeric/`    | Float operations, numeric edge cases          |
| `for_loops/`  | Range iteration, loop constructs              |
| `closures/`   | Lambda expressions, higher-order functions    |
| `tco/`        | Tail call optimization stress tests           |
| `fstring/`    | String interpolation (f-strings)              |
| `enums/`      | Option/Result types, pattern matching         |
| `structs/`    | Struct definition and field access            |
| `error_cases/`| Programs expected to fail (type errors, etc.) |

## Running

```bash
# Run all fixtures
cargo test --test fixtures

# Run with output visible
cargo test --test fixtures -- --nocapture

# Update golden files with actual output
CJC_FIXTURE_UPDATE=1 cargo test --test fixtures
```

## How It Works

The runner (`runner.rs`):
1. Discovers all `.cjc` files recursively under `tests/fixtures/`
2. Parses and executes each via the MIR interpreter (seed=42 for determinism)
3. Compares captured stdout against `.stdout` golden files
4. For error fixtures (`.stderr` exists), runs with type-checking enabled and checks error messages
5. Reports a clear diff on mismatches

## Adding a New Fixture

1. Create `tests/fixtures/<category>/<name>.cjc` with your CJC source
2. Run `CJC_FIXTURE_UPDATE=1 cargo test --test fixtures` to auto-generate golden `.stdout`
3. Review the generated golden file
4. For error test cases, create a `.stderr` file with expected error substrings (one per line)
