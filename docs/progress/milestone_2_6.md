# CJC Milestone 2.6 — Progress Report

**Status: COMPLETE**
**Date: 2026-02-16**
**Total Tests: 694 (62 new + 632 existing, 0 failures)**

---

## Overview

Milestone 2.6 delivers algebraic data types (enums with payloads), the standard prelude enums (`Option<T>`, `Result<T, E>`), the try operator (`?`), static exhaustiveness checking, error metadata propagation, brain float 16 (`bf16`) support, full MIR monomorphization, and deterministic map removal fixes. These features complete the deferred items from Milestone 2.5 and establish CJC's error handling and ADT foundations.

---

## Features Implemented

### A. Enum Declarations + ADTs with Payloads

**Syntax:**
```
enum Color { Red, Green, Blue }
enum Shape { Circle(f64), Rect(f64, f64) }
enum Option<T> { Some(T), None }
enum Result<T, E> { Ok(T), Err(E) }
```

**Implementation details:**
- Parser: `parse_enum_decl()` handles generic type parameters, comma-separated variants with optional tuple payloads
- Type system: `Type::Enum(EnumType)` with `EnumType { name, type_params, variants }` and `EnumVariant { name, fields }`
- Runtime: `Value::Enum { enum_name: String, variant: String, fields: Vec<Value> }` — tagged union, stack-allocatable, no boxing
- Pattern matching: `PatternKind::Variant { enum_name, variant, fields }` with recursive destructuring

**Variant resolution strategy:** `Some(42)` is parsed as `ExprKind::Call { callee: Ident("Some"), args: [42] }`. During HIR lowering, a `variant_names: HashMap<String, String>` map (variant -> enum) resolves calls to known variant names into `HirExprKind::VariantLit`. Unit variants like `None` used as expressions or in patterns are similarly resolved from `Ident`/`Binding` forms. This avoids requiring type information at parse time.

### B. Standard Prelude Enums (Option / Result)

`Option<T>` and `Result<T, E>` are auto-registered in `TypeEnv::new()` before any user code is processed:
- Enum types registered in `type_defs`
- Variant constructors registered as functions in `fn_sigs`: `Some: fn(T) -> Option<T>`, `None: fn() -> Option<T>`, `Ok: fn(T) -> Result<T, E>`, `Err: fn(E) -> Result<T, E>`
- Variant names pre-populated in HIR lowering's `variant_names` map
- Available globally without import in both eval (v1) and MIR-exec (v2) pipelines

### C. Try Operator `?`

**Parsing:** `?` is parsed as a postfix unary operator in the Pratt parser loop. `TokenKind::Question` at postfix binding power produces `ExprKind::Try(Box::new(inner))`.

**Desugaring at HIR level:** The `?` operator is fully eliminated during AST-to-HIR lowering. `ExprKind::Try(inner)` becomes:
```
HirExprKind::Match {
    scrutinee: lower(inner),
    arms: [
        { pattern: Variant("Result", "Ok", [Binding("__try_v")]), body: Var("__try_v") },
        { pattern: Variant("Result", "Err", [Binding("__try_e")]),
          body: Return(VariantLit("Result", "Err", [Var("__try_e")])) },
    ]
}
```

**Type checking:** The inner expression must type to `Result<T, E>`. The enclosing function's return type must be `Result<_, E>` (same error type). The expression's type resolves to `T`.

**Properties:** No exceptions, no hidden allocations, works in `nogc` functions, preserves error metadata through propagation.

### D. Static Exhaustiveness Checking

New `check_match_exhaustiveness()` method on TypeChecker:
- Activates when the match scrutinee types to `Type::Enum(e)`
- Collects covered variants from all match arm patterns (variant patterns by name, wildcard/binding patterns cover all)
- If not all variants are covered and no wildcard arm exists, emits error `E0130` listing the missing variants with source spans
- Example: matching on `Option<T>` with only `Some(x) => ...` produces: `"non-exhaustive match: missing variants: None"`

### E. Error Metadata

**Error struct in prelude:** `Error` is registered as a built-in struct with fields `{ span: i64, message: String }`.

**Integration:** Core failing operations (division by zero, index out of bounds) can return `Result<T, Error>` where the Error carries the source span. The `?` operator propagates Error values naturally through its match desugaring — no special machinery needed. The span is retained from the origin through any number of `?` propagations.

### F. bf16 (Brain Float 16) Support

**Runtime representation:**
```rust
pub struct Bf16(pub u16);  // raw bit storage

impl Bf16 {
    pub fn from_f32(v: f32) -> Self { Bf16((v.to_bits() >> 16) as u16) }
    pub fn to_f32(self) -> f32 { f32::from_bits((self.0 as u32) << 16) }
}
```

- `Value::Bf16(Bf16)` variant in the runtime
- Arithmetic: widen to f32, compute, narrow back to bf16 (add, sub, mul, div)
- Type system: `Type::Bf16` — classified as numeric and float
- Builtin conversions: `bf16_to_f32(bf16) -> f32` and `f32_to_bf16(f32) -> bf16`
- Deterministic: identical bit patterns across runs, no platform-dependent rounding

**Golden bit patterns verified in tests:**
- `bf16(1.0)` = `0x3F80`
- `bf16(-1.0)` = `0xBF80`
- `bf16(0.0)` = `0x0000`
- Round-trip: `bf16_to_f32(f32_to_bf16(x))` preserves bf16-representable values exactly

### G. DetMap Removal Fix

**Problem:** Robin Hood cleanup during `remove()` was appending displaced entries to the `order` vector at the END, breaking strict insertion-order preservation.

**Fix:** During Robin Hood cleanup, displaced entries are tracked with their original order positions. After re-inserting displaced entries, the `order` vector is reconstructed to preserve the original relative positions of surviving entries.

**Semantics (locked):**
- After `remove(key)`: the key is removed, remaining entries maintain their original insertion order
- Re-inserted keys get a new position at end
- Iteration order is deterministic and matches insertion history

### H. Full MIR Monomorphization

**New file:** `crates/cjc-mir/src/monomorph.rs` (~600 lines)

**Architecture:** A post-lowering MIR pass that specializes generic functions for concrete type arguments.

**Pipeline integration:** AST -> HIR -> MIR -> **Monomorph** -> Optimize -> Execute

**Phases:**
1. **Collection:** Walk all reachable function bodies starting from `__main`, identify calls to generic functions, infer concrete type arguments from argument expressions
2. **Specialization:** For each unique `(fn_name, Vec<ConcreteType>)` instantiation, clone the MIR function body and substitute type parameter strings with concrete type names throughout parameters, return type, and all nested expressions/patterns
3. **Rewriting:** Replace generic call sites with calls to mangled specialized names

**Name mangling:** Deterministic format `{fn_name}__M__{type1}_{type2}` (stable, no randomness, sorted canonical type names)

**Type inference heuristic:** `infer_type_from_expr()` maps MIR expression kinds to concrete types:
- `IntLit` -> `"i64"`, `FloatLit` -> `"f64"`, `BoolLit` -> `"bool"`, `StringLit` -> `"String"`
- `StructLit { name }` -> struct name, `VariantLit { enum_name }` -> enum name
- Variable lookups resolved from parameter type context

**Budget:** Hard limit of 1000 specializations per program. `MonomorphReport` tracks: specialization count, top-10 fanout functions, budget exceeded flag. Compilation fails if limit exceeded.

**CLI flag:** `--mir-mono` activates monomorphization + MIR execution via `run_program_monomorphized()`

**Note:** Since the CJC runtime is dynamically typed, monomorphization is primarily a correctness/performance preparation step for future static typing and codegen backends. Generic functions already work without it via dynamic dispatch.

---

## Pipeline Changes

### Before Milestone 2.6
```
Source -> Lexer -> Parser -> AST -> TypeChecker -> Eval (v1)
                                                -> HIR -> MIR -> Optimize -> MIR-exec (v2)
```

### After Milestone 2.6
```
Source -> Lexer -> Parser -> AST -> TypeChecker -> Eval (v1)
                                                -> HIR -> MIR -> Monomorph -> Optimize -> MIR-exec (v2)
```

Key change: Type parameters are now preserved through the entire pipeline (AST `FnDecl.type_params` -> `HirFn.type_params` -> `MirFunction.type_params`), enabling the monomorphization pass to operate on MIR.

---

## Design Decisions

1. **Variant resolution at HIR, not parse time:** Variant constructors like `Some(42)` are parsed as regular function calls. Resolution to `VariantLit` happens during HIR lowering using a `variant_names` map. This avoids coupling the parser to the type system and keeps parsing context-free.

2. **Try desugaring at HIR level:** The `?` operator is eliminated entirely during AST-to-HIR lowering. MIR and the runtime never see `Try` — they only see the equivalent match expression. This keeps downstream passes simple and means `?` inherits all match optimizations for free.

3. **Nominal enum unification:** Two `Type::Enum` values unify only if they share the same name, then variant fields are unified pairwise. This is nominal (not structural) typing, matching Rust/Swift semantics rather than OCaml/TypeScript structural approaches.

4. **Monomorphization as MIR transform:** Rather than monomorphizing at the type-checker level or during HIR lowering, the pass operates on MIR. This allows it to benefit from MIR's simplified representation and compose cleanly with optimization passes that follow.

5. **bf16 widen-compute-narrow:** All bf16 arithmetic widens to f32 before computing. This matches hardware bf16 behavior (no native bf16 ALU in most CPUs) and ensures numerical correctness. The truncation to bf16 happens only when storing the result.

6. **Unit variant pattern ambiguity:** `None` in a pattern is parsed as `PatternKind::Binding("None")`. During HIR lowering, known unit variant names are converted to `HirPatternKind::Variant`. This means user variable names that shadow variant names (`let None = ...`) are correctly handled — the variable binding takes precedence in non-pattern contexts.

---

## Test Coverage

| Feature | Test File | Count |
|---|---|---|
| Enum Parsing + Construction + Matching | milestone_2_6/enums.rs | 16 |
| Option/Result Prelude + ? Operator | milestone_2_6/option_result.rs | 10 |
| Exhaustiveness Checking | milestone_2_6/exhaustiveness.rs | 6 |
| bf16 Arithmetic + Bit Patterns | milestone_2_6/bf16.rs | 14 |
| MIR Monomorphization | milestone_2_6/monomorph.rs | 6 |
| DetMap Removal Order | milestone_2_6/detmap.rs | 5 |
| Eval vs MIR-Exec Parity | milestone_2_6/parity.rs | 5 |
| **Total New** | | **62** |
| **Total Workspace** | | **694** |

### Parity Discipline
Every enum/option/result/try test runs through BOTH the eval (v1) and MIR-exec (v2) interpreters, asserting identical output. This ensures the two execution backends remain functionally equivalent.

### Determinism Verification
Full test suite run twice with identical results — no non-determinism in enum construction, pattern matching, monomorphization ordering, or DetMap iteration.

---

## Files Modified

| File | Changes |
|---|---|
| `crates/cjc-types/src/lib.rs` | `Type::Enum(EnumType)`, `Type::Bf16`, `EnumType`/`EnumVariant` structs, `unify()` for enums+bf16, `check_match_exhaustiveness()`, Try/VariantLit type checking, prelude enum registration, bf16 builtins, `register_enum()` |
| `crates/cjc-runtime/src/lib.rs` | `Value::Enum`, `Value::Bf16(Bf16)`, `Bf16` struct with from/to f32 + arithmetic, `values_equal_static()` + `value_hash()` for enums, DetMap `remove()` order fix |
| `crates/cjc-parser/src/lib.rs` | `parse_enum_decl()`, variant patterns in `parse_pattern()`, `?` postfix operator in Pratt parser |
| `crates/cjc-hir/src/lib.rs` | `HirItem::Enum(HirEnumDef)`, `HirEnumDef`/`HirVariantDef`, `HirExprKind::VariantLit`, `HirPatternKind::Variant`, `type_params` on `HirFn`, `variant_names` map, `?` desugaring to match, variant resolution from Call/Ident/Binding |
| `crates/cjc-mir/src/lib.rs` | `MirEnumDef`/`MirVariantDef`, `enum_defs` on `MirProgram`, `type_params` on `MirFunction`, `MirExprKind::VariantLit`, `MirPattern::Variant`, enum/variant lowering |
| `crates/cjc-mir/src/nogc_verify.rs` | Walk `VariantLit` fields + `Variant` patterns (enum construction is NoGC-safe) |
| `crates/cjc-mir/src/optimize.rs` | `VariantLit` in constant_fold_expr (recurse into fields), is_pure_expr (pure if all fields pure), collect_used_vars_expr |
| `crates/cjc-eval/src/lib.rs` | `variant_to_enum` map, enum `register_decl`, Try inline desugaring, VariantLit construction, Variant pattern matching, unit variant resolution in Ident eval + Binding patterns, variant constructor interception in eval_call |
| `crates/cjc-mir-exec/src/lib.rs` | `VariantLit` evaluation, `Variant` pattern matching, `run_program_monomorphized()` entry point |
| `crates/cjc-cli/src/main.rs` | `--mir-mono` CLI flag, `mir_mono` parameter to `cmd_run`, monomorphized execution path |

## Files Created

| File | Purpose |
|---|---|
| `crates/cjc-mir/src/monomorph.rs` | Full MIR monomorphization pass (~600 lines): collection, specialization, rewriting, budget enforcement |
| `tests/test_milestone_2_6.rs` | Test harness entry point |
| `tests/milestone_2_6/mod.rs` | Module declarations (7 submodules) |
| `tests/milestone_2_6/enums.rs` | Enum parsing, construction, matching tests |
| `tests/milestone_2_6/option_result.rs` | Option/Result prelude + ? operator tests |
| `tests/milestone_2_6/exhaustiveness.rs` | Static exhaustiveness checking tests |
| `tests/milestone_2_6/bf16.rs` | bf16 arithmetic + golden bit pattern tests |
| `tests/milestone_2_6/monomorph.rs` | MIR monomorphization pass tests |
| `tests/milestone_2_6/detmap.rs` | DetMap removal order preservation tests |
| `tests/milestone_2_6/parity.rs` | Eval vs MIR-exec parity tests |
| `docs/progress/milestone_2_6.md` | This document |

---

## Regression Summary

- All 632 existing tests pass (0 regressions)
- All 62 new milestone tests pass
- NoGC verifier: no regressions, enum construction classified as safe
- Optimizer (CF + DCE): no regressions, VariantLit handled correctly
- Milestone 2.4 tests: all pass (including nogc_verifier and optimizer tests)
- Milestone 2.5 tests: all pass (including monomorph, maps, views, linalg)
- Determinism double-run: identical output

---

## Deferred Items from Milestone 2.5 — Now Complete

| Item | Status |
|---|---|
| Full MIR monomorphization pass with function cloning and name mangling | Done (Phase 8) |
| bf16 scalar type | Done (Phase 1 + 3) |
| DetMap remove order fix | Done (Phase 1) |

## Remaining Deferred Items

- GC Map variant (DetMap on GcHeap)
- Symbolic shape constraints through kernel boundaries
- Column pruning optimization for join nodes
- Linalg eigenvalue decomposition
- Compiled backend (LLVM/Cranelift codegen) — monomorphization infrastructure now ready
- Native bf16 hardware intrinsics (currently software emulation via f32 widening)
