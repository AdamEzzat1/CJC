# ADR-0003: Backward-Compatible run_program() API Surface

**Status:** Accepted
**Date:** 2024-01-20
**Deciders:** Systems Architect, Technical Lead
**Supersedes:** none

## Context

As the MIR executor (`cjc-mir-exec`) gained new capabilities (optimization, type checking, monomorphization, NoGC verification), a decision was needed: should each capability be a flag on a single `run_program(program, seed, flags)` function, or should each capability have its own entry point?

Options evaluated:
1. **Single entry with flags enum**: `run_program(program, seed, RunFlags { optimize: bool, type_check: bool, ... })`. Simple but requires callers to opt-in to each capability explicitly.
2. **Separate entry points per capability**: `run_program_optimized`, `run_program_type_checked`, `run_program_monomorphized`. Each is independently testable and composable.
3. **Builder pattern**: `ProgramRunner::new(program).with_optimization().with_type_checking().run(seed)`. Expressive but heavier API surface.

## Decision

Use **separate entry points per capability** with a **stable base function**:

```rust
// Stable base — never changes signature
pub fn run_program(program: &Program, seed: u64) -> MirExecResult;
pub fn run_program_with_executor(program: &Program, seed: u64) -> Result<(Value, Executor)>;

// Capability variants — each adds exactly one pipeline stage
pub fn run_program_optimized(program: &Program, seed: u64) -> MirExecResult;
pub fn run_program_type_checked(program: &Program, seed: u64) -> MirExecResult;
pub fn run_program_monomorphized(program: &Program, seed: u64) -> MirExecResult;
pub fn verify_nogc(program: &Program) -> Result<(), String>;
pub fn lower_to_mir(program: &Program) -> cjc_mir::MirProgram;
```

## Rationale

- **Test isolation**: Each capability can be tested independently. Optimizer bugs do not affect the base interpreter.
- **Parity gate**: The base `run_program` vs `cjc-eval` parity gate (G-1/G-2) remains clean — no flags to accidentally enable.
- **Additive**: New capabilities (e.g., `run_program_capture` for fixture testing) can be added without breaking existing callers.

## Consequences

**Positive:**
- All 1,692 tests that use `run_program` continue to work unchanged as new capabilities are added.
- Each entry point is individually benchmarkable.

**Known limitations:**
- Capability combinations (e.g., "optimize + type check + monomorphize") require a composed entry point or a new function.
- The function list grows linearly with capabilities.

## Implementation Notes

- Crates affected: `cjc-mir-exec`
- Files: `crates/cjc-mir-exec/src/lib.rs`
- New function to add: `run_program_capture(program: &Program, seed: u64) -> Result<(Value, Vec<String>), MirExecError>` for fixture test runner
- Regression gate: `cargo test --workspace` must pass with 0 failures
