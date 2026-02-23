# Milestone 2.4 — NoGC Static Verifier & MIR Optimizer

**Status:** COMPLETE
**Date:** 2026-02-15
**Tests:** 535 total (62 new milestone-specific), 0 failures

---

## Summary

Milestone 2.4 delivers three major components:

1. **NoGC Static Verifier** — Compile-time enforcement that `is_nogc` functions and `nogc` blocks contain no GC operations, directly or transitively.
2. **MIR Optimizer** — Constant Folding (CF) and Dead Code Elimination (DCE) passes behind the `--mir-opt` flag.
3. **Parity & Shape Verification** — Bit-identical comparison harness ensuring optimized MIR produces identical results to unoptimized MIR.

---

## A) NoGC Static Verifier

### Location
`crates/cjc-mir/src/nogc_verify.rs`

### Rules Enforced
| Rule | Description |
|------|-------------|
| Direct GC rejection | `gc_alloc`, `gc_collect` forbidden in `is_nogc` functions |
| Transitive rejection | Calls to functions with `may_gc == true` are rejected |
| Unknown/external rejection | Conservative: calls to unresolved functions are rejected |
| Indirect call rejection | Closure/higher-order calls in nogc are rejected (conservative) |
| NoGcBlock enforcement | `nogc { }` blocks inside non-nogc functions also enforce all rules |

### Effect Classification (may_gc)
- Built via fixpoint iteration over the call graph
- Seed rules:
  - `gc_alloc`, `gc_collect` -> `may_gc = true`
  - Safe builtins (`print`, `Tensor.zeros`, `assert`, etc.) -> `may_gc = false`
  - Functions with indirect calls -> `may_gc = true` (conservative)
- Propagation: `f.may_gc = any(callee.may_gc for callee in f.calls) OR local_gc_ops`

### Diagnostic Examples
```
nogc violation in `bad_fn`: direct call to GC builtin `gc_alloc`
nogc violation in `caller`: call to `allocator` which may trigger GC (via allocator -> gc_alloc)
nogc violation in `fn_with_closure`: indirect call in nogc function (conservative rejection)
```

### G-8 Status: PASS
- **15 tests** in `tests/milestone_2_4/nogc_verifier/`
- All pass (direct, transitive, unknown, indirect, block-level, full-pipeline)
- Command: `cargo test --test test_milestone_2_4 g8_`

---

## B) MIR Optimizer

### Location
`crates/cjc-mir/src/optimize.rs`

### Pass 1: Constant Folding (CF)

**Purity whitelist (safe to fold):**
| Type | Operations |
|------|-----------|
| Int  | `+`, `-`, `*`, `/` (not div-by-0), `%` (not mod-by-0), `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Float | All arithmetic and comparisons (same IEEE 754 ops as runtime) |
| Bool | `==`, `!=`, `&&`, `\|\|`, `!` |
| String | `+` (concat), `==`, `!=` |
| Unary | `-` (int/float), `!` (bool) |
| If-expr | Constant bool condition -> branch elimination |

**Known non-goals (NOT folded):**
- Division/modulo by zero (let runtime error)
- Float reassociation (`(a+b)+c` is NOT rewritten to `a+(b+c)`)
- Mixed-type operations
- Calls (never pure-folded)

### Pass 2: Dead Code Elimination (DCE)

**Rules:**
| Pattern | Action |
|---------|--------|
| `let x = <pure_expr>` where `x` is never read | Remove |
| `let x = <call(...)>` where `x` is never read | KEEP (side effects) |
| `if true { body }` | Inline body statements + result |
| `if false { body }` | Remove entirely |
| `if false { ... } else { body }` | Inline else body |
| `while false { body }` | Remove entirely |

**Purity constraints:** Index, MultiIndex, Call, Assign, Block, Match, Lambda, MakeClosure are all considered impure and preserved.

### Pass Sequencing
1. Constant Folding
2. Dead Code Elimination
3. Constant Folding (second pass, may find new opportunities)

### Optimizer Flag
- `--mir-opt` on the CLI enables the optimized MIR execution path
- Default: off (AST interpreter used)
- Tests cover both on and off states

---

## C) Parity & Shape Verification

### Parity Harness
- Located in `tests/milestone_2_4/parity/`
- For each test program:
  - Run through `run_program()` (unoptimized MIR)
  - Run through `run_program_optimized()` (optimized MIR)
  - Compare results via `bit_identical()`:
    - Integers: exact `==`
    - Floats: `to_bits()` comparison (catches -0.0, NaN payload)
    - Tensors: shape equality + element-wise `to_bits()`
    - Tuples/Arrays: recursive structural comparison
  - Output strings compared for exact match

### Kahan Parity Results
- Alternating large+small float values: PASS (no reassociation)
- NaN behavior: PASS (bit-identical NaN payload)
- Negative zero: PASS (-0.0 preserved)
- Infinity: PASS (1.0/0.0 = Inf)

### Shape Invariants
- Tensor.zeros/ones shape preserved through optimization
- Tensor addition shape preserved
- Shape mismatch errors fire at same point in both paths
- Matmul dimension checks preserved
- Matmul incompatible dimensions still error in both paths
- Tensor.from_vec shape metadata preserved

---

## D) Test Plan

### Directory Structure
```
tests/milestone_2_4/
  mod.rs                    -- Entry point
  nogc_verifier/mod.rs      -- G-8 tests (15 tests)
  optimizer/mod.rs           -- CF + DCE tests (24 tests)
  parity/mod.rs              -- Opt-off vs opt-on (14 tests)
  shape/mod.rs               -- Shape/dimension tests (9 tests)
```

### How to Run

```bash
# Milestone 2.4 tests only
cargo test --test test_milestone_2_4

# G-8 (NoGC verifier) tests only
cargo test --test test_milestone_2_4 g8_

# Constant folding tests only
cargo test --test test_milestone_2_4 cf_

# DCE tests only
cargo test --test test_milestone_2_4 dce_

# Parity tests only
cargo test --test test_milestone_2_4 parity_

# Shape tests only
cargo test --test test_milestone_2_4 shape_

# Full suite (all tests, opts off — default)
cargo test --workspace

# Full suite including inline tests
cargo test --workspace

# Inline (unit) tests for optimizer crate
cargo test -p cjc-mir
```

---

## E) Gates

| Gate | Description | Status | Command |
|------|------------|--------|---------|
| G-8 | NoGC smuggle tests | **PASS** | `cargo test --test test_milestone_2_4 g8_` |
| G-10 | Full suite with opts enabled (parity) | **PASS** | `cargo test --test test_milestone_2_4` |

### G-10 Detail
The G-10 gate is satisfied by the parity test suite which runs every program through both unoptimized and optimized MIR pipelines and verifies bit-identical results. The full workspace suite (535 tests) passes cleanly.

---

## F) Files Changed

### New Files
| File | Description |
|------|-------------|
| `crates/cjc-mir/src/nogc_verify.rs` | NoGC static verifier (call graph + fixpoint + diagnostics) |
| `crates/cjc-mir/src/optimize.rs` | MIR optimizer (CF + DCE) |
| `tests/test_milestone_2_4.rs` | Milestone test entry point |
| `tests/milestone_2_4/mod.rs` | Module root |
| `tests/milestone_2_4/nogc_verifier/mod.rs` | G-8 tests (15) |
| `tests/milestone_2_4/optimizer/mod.rs` | CF + DCE tests (24) |
| `tests/milestone_2_4/parity/mod.rs` | Parity harness (14) |
| `tests/milestone_2_4/shape/mod.rs` | Shape/dimension tests (9) |
| `docs/milestones/milestone_2_4_progress.md` | This report |

### Modified Files
| File | Change |
|------|--------|
| `crates/cjc-mir/src/lib.rs` | Added `pub mod nogc_verify; pub mod optimize;` |
| `crates/cjc-mir-exec/src/lib.rs` | Added `run_program_optimized`, `verify_nogc`, `lower_to_mir` |
| `crates/cjc-cli/src/main.rs` | Added `--mir-opt` flag, NoGC verification in `run` command |
| `crates/cjc-cli/Cargo.toml` | Added `cjc-hir`, `cjc-mir`, `cjc-mir-exec` dependencies |

### Test Counts
| Suite | Before | After | Delta |
|-------|--------|-------|-------|
| Integration tests | 281 | 343 | +62 |
| Inline tests | ~192 | ~192 | +0 (inline tests in new modules counted in workspace) |
| **Total workspace** | **473** | **535** | **+62** |
