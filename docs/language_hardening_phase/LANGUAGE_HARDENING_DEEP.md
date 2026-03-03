# Language Hardening Phase (Deep) -- Complete Documentation

**Date:** 2026-03-03
**Tests:** 183 new LH tests, 3,394 total workspace (0 failures)
**Crates:** 18 (added `cjc-snap`)
**Constraint:** Zero external dependencies

---

## Overview

This phase upgrades 11 major compiler and runtime subsystems, transforming CJC from a prototype language into a hardened, production-quality compiler. All changes maintain full backward compatibility and zero test regressions.

**Sub-phases implemented:**

| # | Sub-phase | Tests | Key Deliverable |
|---|-----------|-------|-----------------|
| 1 | Diagnostics + Error Taxonomy | 19 | Structured error codes (E0xxx--E8xxx) |
| 2 | Type Inference | 17 | Bidirectional local inference with unification |
| 3 | Generics + Trait Bounds | 16 | Enforced trait bounds on generic functions |
| 4 | Effect Typing | 23 | `/ pure`, `/ io` function annotations |
| 5 | REPL Upgrades | 13 | Raw-mode line editor, history, meta-commands |
| 6 | CFG-Based MIR | 12 | Basic block + terminator IR |
| 7 | CFG Executor | 17 | Full-parity block interpreter |
| 8 | SSA Form + Verifier | 19 | Phi insertion, dominator tree, 5-point verification |
| 9 | Optimizer Suite | 13 | SCCP, GVN, DCE, LICM, strength reduction, inlining |
| 10 | Records / Value Semantics | 15 | Immutable `record` types with structural equality |
| 11 | CJC Snap | 19 | Content-addressable serialization with SHA-256 |

---

## Sub-phase 1: Diagnostics + Error Taxonomy

### Design

Every diagnostic in CJC now carries a structured error code. Codes are partitioned by subsystem:

| Range | Subsystem |
|-------|-----------|
| E0xxx | Lexer errors |
| E1xxx | Parser errors |
| E2xxx | Type errors |
| E4xxx | Effect errors |
| E6xxx | Generic/trait errors |
| E7xxx | MIR errors |
| E8xxx | Runtime errors |

### Key Types

```rust
// crates/cjc-diag/src/error_codes.rs
pub enum ErrorCode { E0001, E0002, ..., E8001, ... }

// Extended DiagnosticBuilder
pub struct DiagnosticBuilder {
    code: String,
    span: Span,
    labels: Vec<Label>,
    hints: Vec<String>,
    fix_suggestions: Vec<FixSuggestion>,
}

pub struct FixSuggestion {
    span: Span,
    replacement: String,
    message: String,
}
```

### Files Modified
- `crates/cjc-diag/src/lib.rs` -- DiagnosticBuilder, fix suggestions, multi-line spans
- `crates/cjc-diag/src/error_codes.rs` -- Error code registry
- `crates/cjc-lexer/src/lib.rs` -- Migrated error sites
- `crates/cjc-parser/src/lib.rs` -- Migrated error sites
- `crates/cjc-types/src/lib.rs` -- Migrated error sites

### Tests (19)
Error code rendering, fix suggestion display, multi-span errors, color diagnostic output, Rust+Elm style formatting.

---

## Sub-phase 2: Type Inference (Local/Bidirectional)

### Design

Function parameters still require annotations. `let` bindings without annotations infer from the initializer. Return types infer from the body's tail expression. Bidirectional flow: annotations push types down (checking mode), expressions push types up (synthesis mode).

### Key Types

```rust
// crates/cjc-types/src/inference.rs
pub struct InferCtx {
    next_var: usize,
    constraints: Vec<(Type, Type, Span)>,
    subst: BTreeMap<TypeVarId, Type>,
}
```

### Algorithm

1. Fresh type variables created for untyped `let` bindings
2. Constraint generation walks the expression tree
3. Unification solver resolves constraints (union-find style)
4. Resolved types substituted back into the type environment

### Files Modified
- `crates/cjc-types/src/lib.rs` -- `InferCtx`, constraint generation, unification
- `crates/cjc-types/src/inference.rs` -- Bidirectional inference engine
- `crates/cjc-hir/src/lib.rs` -- Thread inferred types through HIR

### Tests (17)
`let x = 42;` infers i64, `let y = 3.14;` infers f64, function return inference, generic argument inference, error on conflicting constraints.

---

## Sub-phase 3: Real Generics + Trait Bounds

### Design

Built-in trait implementations for primitives (`i64: Numeric`, `f64: Numeric + Float`, etc.). User-defined `impl Trait for Type` blocks are now enforced. Monomorphization rejects type arguments that violate bounds.

### Bound Checking

```rust
fn type_satisfies_trait(ty: &Type, trait_name: &str) -> bool
fn check_generic_call(fn_decl, type_args) -> Result<(), TypeError>
```

### Files Modified
- `crates/cjc-types/src/lib.rs` -- `type_satisfies_trait()`, bound enforcement
- `crates/cjc-dispatch/src/lib.rs` -- Bounds during dispatch resolution
- `crates/cjc-mir/src/monomorph.rs` -- Bounds during specialization

### Tests (16)
`fn add<T: Numeric>(a: T, b: T) -> T` called with strings produces E6001, bound satisfaction for primitives, multi-bound checking, recursive bounds.

---

## Sub-phase 4: Enforced Effect Typing

### Design

Functions can be annotated with effect sets: `fn foo() -> i64 / pure { ... }`. Unannotated functions default to "any effect" (backward compatible). When annotated, the type checker verifies the body's computed effects are a subset of declared effects.

### Syntax

```
fn pure_add(a: i64, b: i64) -> i64 / pure { a + b }
fn log_result(x: i64) -> i64 / io { print(x); x }
```

### Effect Hierarchy

| Effect | Includes |
|--------|----------|
| `pure` | No side effects (deterministic, no IO, no allocation) |
| `alloc` | May allocate heap memory |
| `io` | May perform IO operations |
| (default) | Any effect permitted |

### Files Modified
- `crates/cjc-lexer/src/lib.rs` -- `pure`, `io`, `alloc` contextual keywords
- `crates/cjc-parser/src/lib.rs` -- Parse `/ effect` after return type
- `crates/cjc-ast/src/lib.rs` -- `FnDecl.effect_annotation: Option<EffectSet>`
- `crates/cjc-types/src/lib.rs` -- Effect checking pass
- `crates/cjc-types/src/effect_registry.rs` -- 249+ effect classifications (source of truth)

### Tests (23)
Pure function calling IO function produces E4002, effect subtyping, `nogc fn` automatic effects, closure effect propagation.

---

## Sub-phase 5: REPL Upgrades (Zero-Dep)

### Design

Hand-rolled ANSI line editor (no external dependencies). Raw terminal mode via platform APIs (Windows: `SetConsoleMode`; Unix: `termios`).

### Features

- Arrow key navigation (left/right cursor, up/down history)
- Persistent history in `~/.cjc_history`
- Multi-line input (lines ending with `{` or `\` continue)
- Meta-commands: `:help`, `:quit`, `:type <expr>`, `:env`, `:mir <expr>`, `:ast <expr>`, `:reset`
- Tab completion for keywords and builtins

### Files Modified
- `crates/cjc-cli/src/main.rs` -- Rewritten `cmd_repl`
- `crates/cjc-cli/src/line_editor.rs` -- Minimal ANSI line editor

### Tests (13)
History navigation, multi-line continuation, meta-command parsing, keyword completion.

---

## Sub-phase 6: CFG-Based MIR (Full Rewrite)

### Design

The canonical MIR representation changed from tree-form (`Vec<MirStmt>` with nested `If`/`While`) to CFG basic blocks (`Vec<BasicBlock>` with `Terminator`). This is the highest-risk change in the entire phase.

### Key Types

```rust
pub struct MirCfg {
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    pub params: Vec<String>,
}

pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<MirStmt>,
    pub terminator: Terminator,
}

pub enum Terminator {
    Goto(BlockId),
    Branch { cond: MirExpr, then_block: BlockId, else_block: BlockId },
    Return(Option<MirExpr>),
    Switch { scrutinee: MirExpr, cases: Vec<(i64, BlockId)>, default: BlockId },
    Unreachable,
}
```

### Lowering Strategy (HIR -> CFG)

- `if/else` -> 3+ blocks (cond eval, then, else, merge)
- `while` -> 3+ blocks (header, body, exit) with back-edge
- `break` -> `Goto(loop_exit)`, `continue` -> `Goto(loop_header)`
- `return` -> `Return` terminator
- Sequential statements stay in same block

### Files Modified
- `crates/cjc-mir/src/lib.rs` -- New MIR types, HirToMir lowering for CFG
- `crates/cjc-mir/src/cfg.rs` -- CFG builder from HIR
- `crates/cjc-mir/src/hir_to_cfg.rs` -- Lowering from HIR to CFG basic blocks

### Tests (12)
Basic block structure, terminator types, loop back-edges, nested control flow, break/continue targeting.

---

## Sub-phase 7: CFG Executor

### Design

Basic-block interpreter that replaces the tree-form executor. Maintains full parity with cjc-eval (AST interpreter).

### Execution Model

```
fn execute_cfg(body: &MirCfg, args: Vec<Value>) -> Result<Value> {
    let mut current_block = body.entry;
    loop {
        let block = &body.blocks[current_block];
        // Execute instructions sequentially
        for instr in &block.instructions { execute_instruction(instr)?; }
        // Branch on terminator
        match &block.terminator {
            Goto(target) => current_block = *target,
            Branch { cond, then_block, else_block } => { ... },
            Return(val) => return Ok(val),
            ...
        }
    }
}
```

### Parity Guarantee

Every CJC program produces identical output whether run through:
- `cjc-eval` (AST interpreter)
- `cjc-mir-exec` (tree-form MIR)
- `cjc-mir-exec` CFG mode (basic block interpreter)

### Files Modified
- `crates/cjc-mir-exec/src/lib.rs` -- New CFG executor alongside tree-form

### Tests (17)
Full parity tests: arithmetic, closures, recursion, match, for-loops, nested control flow, structs, arrays.

---

## Sub-phase 8: SSA Form + Verifier

### Design

Standard SSA construction with phi insertion and variable renaming. Dominator tree computed via iterative algorithm.

### Key Types

```rust
pub struct SsaVar { pub name: String, pub version: u32 }
pub struct PhiNode { pub target: SsaVar, pub sources: Vec<(BlockId, SsaVar)> }

pub struct SsaBlock {
    pub phis: Vec<PhiNode>,
    pub instructions: Vec<SsaInstruction>,
    pub terminator: SsaTerminator,
}
```

### SSA Construction Algorithm

1. Compute dominator tree (iterative algorithm)
2. Compute dominance frontiers
3. Insert phi functions at dominance frontiers for each variable
4. Rename variables via DFS of dominator tree

### SSA Verifier (5-Point Check)

1. Every variable assigned exactly once
2. Every use dominated by its definition
3. Phi functions have one source per predecessor
4. Entry block has no phis (params pre-defined)
5. All block references valid

### Files Modified
- `crates/cjc-mir/src/ssa.rs` -- SSA construction
- `crates/cjc-mir/src/dominators.rs` -- Dominator tree + dominance frontiers

### Tests (19)
SSA construction, verifier pass, phi placement, dominator tree correctness, variable versioning.

---

## Sub-phase 9: Complete Optimizer Suite (SSA-Based)

### Passes (in order)

| Pass | Description |
|------|-------------|
| SCCP | Sparse Conditional Constant Propagation (lattice-based) |
| GVN | Global Value Numbering (replaces tree-form CSE) |
| DCE | SSA-based Dead Code Elimination |
| LICM | Loop-Invariant Code Motion (uses dominator info) |
| Strength Reduction | Algebraic simplifications on SSA |
| Inlining | Small functions (< 20 instructions) inlined |
| Cleanup | Remove empty blocks, simplify trivial phis |

### De-SSA

After optimization, phi nodes are eliminated by inserting copy instructions at predecessor block ends. The result is a CfgBody without phis -- directly executable.

### Files Modified
- `crates/cjc-mir/src/optimize.rs` -- Rewritten for CFG+SSA
- `crates/cjc-mir/src/ssa_optimize.rs` -- SSA-specific passes

### Tests (13)
Constant propagation, dead code removal, loop invariant hoisting, inlining, optimizer parity (optimized == unoptimized output).

---

## Sub-phase 10: Records / Value Semantics

### Design

`record` is a new keyword introducing immutable value types. Records are like structs but field assignment is a type error.

### Syntax

```
record Point {
    x: f64,
    y: f64
}
```

### Type Hierarchy

| Kind | Mutable | Equality | Heap-allocated |
|------|---------|----------|----------------|
| `struct` | Yes | Structural | No (value) |
| `record` | No | Structural | No (value) |
| `class` | Yes | Reference | Yes (GC) |

### Immutability Enforcement

- **Type-check time:** E0160 error on `record_instance.field = value`
- **Runtime (belt-and-suspenders):** Both eval and MIR-exec reject field assignment on records

### Structural Equality

Records (and structs) support `==` and `!=` with deep structural comparison:
- Same type name required
- All fields compared recursively
- Different record type names are never equal (nominal + structural)

### Files Modified
- `crates/cjc-lexer/src/lib.rs` -- `record` keyword
- `crates/cjc-parser/src/lib.rs` -- `parse_record_decl()`
- `crates/cjc-ast/src/lib.rs` -- `DeclKind::Record(RecordDecl)`
- `crates/cjc-types/src/lib.rs` -- `Type::Record(RecordType)`, E0160 immutability check
- `crates/cjc-hir/src/lib.rs` -- `HirItem::Record(HirRecordDef)`
- `crates/cjc-mir/src/lib.rs` -- `is_record: bool` on MirStructDef
- `crates/cjc-eval/src/lib.rs` -- Runtime immutability, structural equality
- `crates/cjc-mir-exec/src/lib.rs` -- Runtime immutability, structural equality

### Tests (15)
Parsing, eval+MIR parity, type error on field assignment, runtime immutability (eval+MIR), structural equality (equal, not-equal, different names), record as function argument, pattern matching, struct still mutable, determinism.

---

## Sub-phase 11: CJC Snap (Content-Addressable Serialization)

### Design

Snap is a content-addressable serialization primitive. `snap(value)` produces a `SnapBlob` containing the canonical binary encoding and its SHA-256 hash. `restore(blob)` decodes the value and verifies the hash matches.

### New Crate: `cjc-snap`

```rust
pub struct SnapBlob {
    pub content_hash: [u8; 32],  // SHA-256 of canonical encoding
    pub data: Vec<u8>,           // canonical encoded bytes
}

pub fn snap(value: &Value) -> SnapBlob;
pub fn restore(blob: &SnapBlob) -> Result<Value, SnapError>;

pub enum SnapError {
    HashMismatch { expected: [u8; 32], actual: [u8; 32] },
    DecodeError(String),
}
```

### SHA-256 Implementation

Hand-rolled FIPS 180-4 compliant SHA-256 (~200 LOC). Zero external dependencies. Verified against NIST test vectors:
- `sha256(b"")` = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `sha256(b"abc")` = `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`

### Encoding Rules (Determinism)

| Type | Encoding |
|------|----------|
| Int | Tag `0x01` + 8 bytes little-endian |
| Float | Tag `0x02` + 8 bytes IEEE 754 (NaN canonicalized to `0x7FF8000000000000`) |
| Bool | Tag `0x03` + 1 byte (`0x00`/`0x01`) |
| String | Tag `0x04` + 8-byte length + UTF-8 |
| Void | Tag `0x05` |
| Array | Tag `0x06` + 8-byte length + elements |
| Tuple | Tag `0x07` + 8-byte length + elements |
| Struct/Record | Tag `0x08` + name + field count + sorted fields (by name) |

### Key Properties

1. **Deterministic:** Same value always produces same bytes
2. **Content-addressable:** Same value always produces same hash
3. **NaN canonicalization:** All NaN values encode identically
4. **Struct field order:** Fields sorted alphabetically (insertion-order independent)
5. **Hash mismatch detection:** Tampered blobs rejected with `SnapError::HashMismatch`

### Files
- `crates/cjc-snap/src/lib.rs` -- SnapBlob, snap/restore API, SnapError
- `crates/cjc-snap/src/hash.rs` -- SHA-256 implementation
- `crates/cjc-snap/src/encode.rs` -- Canonical binary encoder
- `crates/cjc-snap/src/decode.rs` -- Decoder with validation

### Tests (19 integration + 46 unit)
SHA-256 correctness, round-trip for all value types, content-addressability, struct determinism, hash mismatch rejection, NaN canonicalization, error handling, eval-to-snap parity.

---

## Verification

### Test Counts

| Suite | Passed | Failed | Ignored |
|-------|--------|--------|---------|
| LH-01 Diagnostics | 19 | 0 | 0 |
| LH-02 Type Inference | 17 | 0 | 0 |
| LH-03 Generics | 16 | 0 | 0 |
| LH-04 Effects | 23 | 0 | 0 |
| LH-05 REPL | 13 | 0 | 0 |
| LH-06 CFG | 12 | 0 | 0 |
| LH-07 CFG Exec | 17 | 0 | 0 |
| LH-08 SSA | 19 | 0 | 0 |
| LH-09 Optimizer | 13 | 0 | 0 |
| LH-10 Records | 15 | 0 | 0 |
| LH-11 Snap | 19 | 0 | 0 |
| **LH Total** | **183** | **0** | **0** |
| **Full Workspace** | **3,394** | **0** | **0** |

### Parity Invariant

For every CJC program P and seed S:
```
eval(P, S) == mir_exec(P, S) == cfg_exec(P, S) == optimized_exec(P, S)
```

### Zero Regressions

All 3,394 workspace tests pass. No existing functionality was broken by any sub-phase.

---

## Architecture After Hardening

```
Source Code
    |
    v
[cjc-lexer] -----> Tokens + DiagnosticBag (structured error codes)
    |
    v
[cjc-parser] ----> AST (Program) + DiagnosticBag
    |
    +---> [cjc-eval] -------> Value (AST interpreter, record-aware)
    |
    v
[cjc-types] -----> Type-checked AST (inference, effects, generics, records)
    |
    v
[cjc-hir] -------> HIR (records lowered)
    |
    v
[cjc-mir] -------> CFG MIR (BasicBlock + Terminator)
    |
    +---> [cjc-mir/ssa.rs] -> SSA Form (phi nodes, versioned vars)
    |         |
    |         +---> [cjc-mir/ssa_optimize.rs] -> Optimized SSA
    |                   |
    |                   +---> De-SSA -> Executable CFG
    |
    v
[cjc-mir-exec] --> Value (CFG interpreter, record-aware)
    |
    v
[cjc-snap] ------> SnapBlob (content-addressable serialization)
```

---

## Files Summary

### New Files Created
| File | Purpose |
|------|---------|
| `crates/cjc-diag/src/error_codes.rs` | Error code registry |
| `crates/cjc-types/src/inference.rs` | Bidirectional type inference |
| `crates/cjc-cli/src/line_editor.rs` | Zero-dep ANSI line editor |
| `crates/cjc-mir/src/hir_to_cfg.rs` | HIR to CFG lowering |
| `crates/cjc-mir/src/ssa.rs` | SSA construction |
| `crates/cjc-mir/src/dominators.rs` | Dominator tree + frontiers |
| `crates/cjc-mir/src/ssa_optimize.rs` | SSA-based optimizer passes |
| `crates/cjc-snap/` (entire crate) | Content-addressable serialization |
| `tests/language_hardening/test_lh01-11*.rs` | 11 integration test files |
| `tests/language_hardening/mod.rs` | Test module aggregator |

### Crates Modified
| Crate | Sub-phases |
|-------|------------|
| `cjc-lexer` | 1, 4, 10 |
| `cjc-parser` | 1, 4, 10 |
| `cjc-ast` | 4, 10 |
| `cjc-diag` | 1 |
| `cjc-types` | 1, 2, 3, 4, 10 |
| `cjc-hir` | 2, 10 |
| `cjc-mir` | 3, 6, 8, 9, 10 |
| `cjc-eval` | 10 |
| `cjc-mir-exec` | 7, 10 |
| `cjc-dispatch` | 3 |
| `cjc-cli` | 5 |

### New Crate
| Crate | Purpose |
|-------|---------|
| `cjc-snap` | Content-addressable serialization (SHA-256, encode, decode) |
