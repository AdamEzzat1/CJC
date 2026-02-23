# CJC Diagnostic Error Code Registry

**Last Updated:** 2025-01-01

This document is the canonical source of truth for all CJC compiler diagnostic codes.
Error codes are emitted by `cjc-types`, `cjc-mir`, and `cjc-mir-exec`.

## Code Ranges

| Range | Category |
|-------|----------|
| E0001–E0099 | Parse errors (emitted by `cjc-parser`) |
| E0100–E0149 | Type system: basic type errors |
| E0150–E0199 | Type system: binding and mutability |
| E0200–E0249 | Type system: numeric types and casts |
| E0300–E0349 | Type system: traits and generics |
| E0400–E0449 | Type system: constants and statics |
| E0500–E0549 | Type system: shape inference (tensors) |
| W0001–W0099 | Warnings (non-fatal) |
| I0001–I0099 | Informational hints |

---

## Error Codes

### Parse Errors (E0001–E0099)

| Code | Message Template | Emitted By | Notes |
|------|-----------------|------------|-------|
| E0001 | unexpected token `{token}` | `Parser::parse_primary` | Catch-all for unexpected tokens |
| E0002 | expected `{expected}`, found `{found}` | `Parser::expect` | Missing required token |
| E0003 | unterminated string literal | `Lexer::lex_string` | EOF inside string |
| E0004 | invalid escape sequence `{seq}` | `Lexer::lex_string` | Unknown `\x` escape |

### Type Errors (E0100–E0149)

| Code | Message Template | Emitted By | Test File |
|------|-----------------|------------|-----------|
| E0100 | type mismatch: expected `{expected}`, found `{found}` | `TypeChecker::check_assign` | `tests/test_types.rs` |
| E0101 | cannot apply operator `{op}` to types `{lhs}` and `{rhs}` | `TypeChecker::check_binop` | `tests/test_types.rs` |
| E0102 | function `{name}` not found | `TypeChecker::check_call` | `tests/test_types.rs` |
| E0103 | wrong number of arguments: expected `{expected}`, found `{found}` | `TypeChecker::check_call` | `tests/test_types.rs` |
| E0110 | undefined variable `{name}` | `TypeChecker::check_var` | `tests/test_types.rs` |
| E0120 | field `{field}` not found on struct `{struct_name}` | `TypeChecker::check_field` | `tests/test_types.rs` |
| E0125 | index type must be `i64`, found `{found}` | `TypeChecker::check_index` | `tests/test_types.rs` |
| E0130 | non-exhaustive match on enum `{enum_name}` | `TypeChecker::check_match_exhaustiveness` | `tests/milestone_2_6/exhaustiveness.rs` |

### Binding and Mutability (E0150–E0199)

| Code | Message Template | Emitted By | Test File |
|------|-----------------|------------|-----------|
| E0150 | cannot assign to immutable variable `{name}` | `TypeChecker::check_assign` | `tests/audit_tests/test_phase2_mutable_binding.rs` |
| E0151 | cannot mutably borrow immutable variable `{name}` | `TypeChecker::check_borrow` | (future) |

### Numeric Types and Casts (E0200–E0249)

| Code | Message Template | Emitted By | Test File | Status |
|------|-----------------|------------|-----------|--------|
| E0200 | implicit numeric cast not allowed: `{from}` to `{to}` | `TypeChecker::check_binop` | `tests/audit_tests/test_audit_numeric_types.rs` | **New (Stage 3)** |
| E0201 | bit operation requires integer type, found `{type}` | `TypeChecker::check_binop` | `tests/audit_tests/test_audit_numeric_types.rs` | **New (Stage 3)** |

### Traits and Generics (E0300–E0349)

| Code | Message Template | Emitted By | Test File |
|------|-----------------|------------|-----------|
| E0300 | trait bound not satisfied: `{type}` does not implement `{trait}` | `TypeChecker::check_fn_call` | `tests/audit_tests/test_phase2_monomorphization.rs` |
| E0301 | trait `{trait}` not found | `TypeChecker::check_impl` | `tests/test_types.rs` |
| E0310 | duplicate implementation of trait `{trait}` for type `{type}` | `TypeChecker::check_impl` | (future) |

### Constants and Statics (E0400–E0449)

| Code | Message Template | Emitted By | Test File |
|------|-----------------|------------|-----------|
| E0400 | constant initializer is not a compile-time constant expression | `TypeChecker::check_const_decl` | `tests/audit_tests/test_phase2_const_exprs.rs` |
| E0401 | const type annotation `{annotation}` does not match initializer type `{actual}` | `TypeChecker::check_const_decl` | `tests/audit_tests/test_phase2_const_exprs.rs` |

### Shape Inference — Tensor Types (E0500–E0549)

| Code | Message Template | Emitted By | Test File | Status |
|------|-----------------|------------|-----------|--------|
| E0500 | shape mismatch in `{op}`: dimension `{dim1}` != `{dim2}` | `TypeChecker::check_tensor_op` | `tests/audit_tests/test_audit_shape_inference.rs` | **New (Stage 3)** |
| E0501 | unknown shape dimension `{name}` used before binding | `TypeChecker::check_tensor_shape` | `tests/audit_tests/test_audit_shape_inference.rs` | **New (Stage 3)** |
| E0502 | rank mismatch: `{op}` requires {expected}-D tensor, found {actual}-D | `TypeChecker::check_tensor_op` | `tests/audit_tests/test_audit_shape_inference.rs` | **New (Stage 3)** |

---

## Warning Codes

| Code | Message Template | Emitted By | Test File | Status |
|------|-----------------|------------|-----------|--------|
| W0010 | recursive call not in tail position — may cause stack overflow | MIR lint pass | `tests/audit_tests/test_phase2_tco.rs` | **Planned (Stage 3)** |
| W0020 | unused variable `{name}` | `TypeChecker` | `tests/test_types.rs` | Existing |
| W0030 | unreachable code after `return` | `MirExecutor` | — | Existing |

---

## Deprecated Codes

| Old Code | Replaced By | Reason |
|----------|-------------|--------|
| E0115 | E0300 | Trait bound error code renamed for consistency with Rust conventions |

---

## Diagnostic Code Lifecycle

### Adding a New Code

1. Choose a code number from the appropriate range (check this registry first).
2. Add the code to this registry with message template, emitting function, and test file.
3. Implement the diagnostic in `cjc-types/src/lib.rs` or `cjc-mir-exec/src/lib.rs`.
4. Write a test in the appropriate `tests/audit_tests/` file that triggers the code.
5. Update `docs/spec/error_codes.md` with the **Status: Active** flag.

### Retiring a Code

1. Mark the code as **Deprecated** in this registry.
2. Add a "Replaced By" entry.
3. Keep the deprecated code emitting for one major version, then remove.

---

## Error Message Style Guide

CJC diagnostic messages follow these conventions:

1. **Start with a verb** in imperative form: "cannot", "expected", "type mismatch"
2. **Quote identifiers** with backticks: `` `my_variable` ``, `` `i64` ``
3. **Include context** in the message: what was found vs. what was expected
4. **Suggest a fix** in the hints array when possible
5. **No trailing periods** in the main message
6. **Capitalize the first word only**

**Good:** `` cannot assign to immutable variable `x` ``
**Bad:** `Variable x is immutable and cannot be assigned.`
