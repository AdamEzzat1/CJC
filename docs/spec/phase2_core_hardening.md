# Phase 2: Core Hardening — Implementation Specification

**Status:** Completed
**Date:** 2025
**Scope:** Correctness fixes + interpreter performance for the CJC compiler pipeline

---

## Overview

Phase 2 hardening addressed critical correctness gaps and performance bottlenecks identified in the Phase 1 / Reality Audit. The work was organized into four priority tiers (P0–P3), with P0 being blocking correctness issues and P3 being documentation-only items.

### Summary

| Task | Priority | Status | Files Changed |
|------|----------|--------|---------------|
| P0-1: Generic monomorphization | P0 | ✅ Pre-existing (tests added) | `cjc-mir/src/monomorph.rs` |
| P0-2: Trait bound enforcement (E0300) | P0 | ✅ Implemented | `cjc-types/src/lib.rs` |
| P0-3: Mutable binding enforcement (E0150) | P0 | ✅ Implemented | `cjc-types/src/lib.rs` |
| P1-2: Tail call optimization | P1 | ✅ Implemented | `cjc-mir-exec/src/lib.rs` |
| P1-3: Vec COW for Array/Tuple | P1 | 🔶 Deferred (risk) | — |
| P1-4: Scope stack optimization | P1 | 🔶 Deferred (risk) | — |
| P1-5: Interned function bodies | P1 | ✅ Implemented | `cjc-mir-exec/src/lib.rs` |
| P1-6: Parallel matmul | P1 | 🔶 Deferred | — |
| P2-1: impl Trait for Type parser | P2 | ✅ Implemented | `cjc-parser/src/lib.rs` |
| P2-2: Dependent shape types | P2 | ✅ Pre-existing (tests added) | `cjc-types/src/lib.rs` |
| P2-3: Const expressions | P2 | ✅ Implemented | All pipeline crates |
| P2-5: String interpolation | P2 | ✅ Implemented | All pipeline crates |
| P2-6: Result/Option builtins | P2 | ✅ Implemented | `cjc-mir-exec/src/lib.rs` |

---

## P0 Items — Blocking Correctness

### P0-1: Generic Monomorphization (Pre-existing)

**Finding:** Monomorphization was already fully implemented in `cjc-mir/src/monomorph.rs` via a work-queue approach with proper name mangling and call-site rewriting.

**Implementation:** `fn__M__type1_type2` naming convention, budget limit of 1,000 specializations, `infer_type_from_expr` heuristic to determine concrete types at call sites.

**Tests added:** `tests/audit_tests/test_phase2_monomorphization.rs` — 6 tests covering:
- `identity<T>` with i64 and f64 specialization
- Multiple specializations in same program
- Nested generic calls
- Generic `max` function
- Generic function in loop

### P0-2: Trait Bound Enforcement — E0300

**Problem:** The trait bound checker was emitting `E0115` (incorrect code) instead of the canonical `E0300` code.

**Fix:** Updated `check_fn_call` in `cjc-types/src/lib.rs` to emit `E0300` with improved diagnostic message:
```
E0300: trait bound not satisfied: type `T` does not implement `Trait` (required by type parameter `U`)
hint: type parameter `U` requires: `Trait`
```

### P0-3: Mutable Binding Enforcement — E0150

**Problem:** `let` and `let mut` were treated identically — immutable bindings could be reassigned without error.

**Implementation:**
- Changed `TypeEnv::scopes` from `Vec<HashMap<String, Type>>` to `Vec<HashMap<String, (Type, bool)>>` where `bool` is `is_mutable`.
- Added `const_defs: HashMap<String, Type>` to `TypeEnv` for compile-time constants.
- Added methods: `define_var_mut()`, `lookup_var_entry()`, `is_var_mutable()`.
- `check_let`: calls `define_var_mut()` when `l.mutable == true`.
- `ExprKind::Assign { target, .. }`: checks `is_var_mutable(name)`, emits E0150 if false.

**Error format:**
```
E0150: cannot assign to immutable variable `x`
hint: consider making `x` mutable: `let mut x = ...`
```

**Tests:** `tests/audit_tests/test_phase2_mutable_binding.rs` — 6 tests.

---

## P1 Items — Interpreter Performance

### P1-2: Tail Call Optimization (Interpreter-Level)

**Problem:** Deep tail-recursive functions grew the Rust call stack and caused stack overflows.

**Implementation:** Added a trampoline loop in `call_function`:

```rust
fn call_function(&mut self, name: &str, args: &[Value]) -> MirExecResult {
    let mut current_name = name.to_string();
    let mut current_args: Vec<Value> = args.to_vec();
    loop {
        // ... bind params, execute body ...
        match self.exec_body(&func.body) {
            Err(MirExecError::TailCall { name, args }) => {
                current_name = name;
                current_args = args;
                continue;  // trampoline: reuse stack frame
            }
            result => return result,
        }
    }
}
```

TCO is triggered in two positions:
1. **`MirStmt::Return`**: If the returned expression is a direct call to a user function, emit `MirExecError::TailCall` instead of recursing.
2. **`exec_body` result expression**: If the body's tail expression is a direct call, emit `MirExecError::TailCall`.

**Coverage:** Only detects simple self-recursive and inter-function tail calls (`return f(...)` or body expression `f(...)`). Does not optimize tail calls through closures or through indirection.

**Tests:** `tests/audit_tests/test_phase2_tco.rs` — 5 tests including 100,000-iteration countdown.

### P1-3: Vec COW for Array/Tuple (Deferred)

**Rationale for deferral:** Changing `Value::Array(Vec<Value>)` to `Value::Array(Rc<Vec<Value>>)` requires touching 98+ match sites across 6 crates. The risk of regression outweighs the performance benefit at this stage. Documented as a future ADR.

**Future approach:** Wrap `Vec<Value>` in `Rc<RefCell<Vec<Value>>>` in the runtime crate, update all match sites, add a parity gate regression test.

### P1-4: Scope Stack Optimization (Deferred)

**Rationale for deferral:** The scope stack (`Vec<HashMap<String, Value>>`) is already fast for typical programs. The optimization would use `SmallVec<[HashMap<String, Value>; 8]>` to avoid heap allocation for shallow stacks. Deferred pending a profiling-driven decision.

### P1-5: Interned Function Bodies — `Rc<MirFunction>`

**Problem:** Every `call_function` call cloned the entire `MirFunction` (including body AST), causing O(body_size) allocation per call.

**Fix:** Changed `MirExecutor::functions` from `HashMap<String, MirFunction>` to `HashMap<String, Rc<MirFunction>>`. Functions are wrapped in `Rc::new()` once during `exec()` and shared by reference on every call.

**Lambda functions** (created at runtime via `MirExprKind::Lambda`) are also stored as `Rc<MirFunction>`.

### P1-6: Parallel Matmul (Deferred)

**Rationale for deferral:** The existing matmul uses single-threaded BLAS. Adding `rayon` parallelism requires careful benchmark validation to avoid overhead dominating for small matrices. Deferred to a dedicated performance phase.

---

## P2 Items — Language Completeness

### P2-1: `impl Trait for Type` Parser Fix

**Problem:** The parser only supported CJC-native `impl Type : Trait { }` syntax, not the Rust-style `impl Trait for Type { }` syntax.

**Fix:** Updated `parse_impl_decl` to detect the `for` keyword after the first type expression:

```rust
// (1) `impl Type : Trait { ... }` — original CJC syntax
// (2) `impl Trait for Type { ... }` — Rust-style syntax
let (target, trait_ref) = if self.eat(TokenKind::For).is_some() {
    let concrete = self.parse_type_expr()?;
    (concrete, Some(first_type))  // Rust style: trait is first_type, target is concrete
} else if self.eat(TokenKind::Colon).is_some() {
    (first_type, Some(self.parse_type_expr()?))  // CJC style
} else {
    (first_type, None)  // Bare impl
};
```

**Tests:** `tests/audit_tests/test_phase2_impl_trait_syntax.rs` — 5 tests.

### P2-2: Dependent Shape Types (Pre-existing)

**Finding:** Shape constraints (`Tensor<[M, N]>`) are already implemented via `ShapeSubst` and `unify_shapes`/`unify_shape_dim` in `cjc-types`. The feature is complete but untested.

**Tests added:** See `test_audit_datatype_inventory_smoke.rs` (prior phase).

### P2-3: Const Expressions

**New syntax:** `const NAME: Type = expr;`

**Constraints:** The initializer must be a *const expression* — a literal value or a negated literal. Runtime expressions (function calls, variable references) are rejected with E0400.

**Pipeline changes:**
- **Lexer:** Added `TokenKind::Const`
- **AST:** Added `DeclKind::Const(ConstDecl)` with struct `{ name, ty, value, span }`
- **Parser:** Added `parse_const_decl()`, dispatched from `parse_decl` on `TokenKind::Const`
- **Type checker:** Added `check_const_decl()` with E0400 (non-const-safe expr) and E0401 (type mismatch) diagnostics
- **HIR:** Lowered as immutable `LetStmt` (transparent to downstream)
- **Eval:** Added `DeclKind::Const` arm that evaluates the initializer and binds the name

**Error codes:**
- `E0400`: constant initializer is not a compile-time constant expression
- `E0401`: const type annotation does not match initializer type

**Tests:** `tests/audit_tests/test_phase2_const_exprs.rs` — 8 tests.

### P2-5: String Interpolation — `f"...{expr}..."`

**Syntax:** `f"text {expr} more text"` produces a `String` by evaluating each `{expr}` and concatenating.

**Escape sequences:**
- `{{` → literal `{`
- `}}` → literal `}`
- All standard string escapes (`\n`, `\t`, etc.) are supported

**Pipeline changes:**
- **Lexer:** Added `TokenKind::FStringLit`; `lex_fstring()` stores the raw content including `{...}` blocks verbatim in the token text
- **AST:** Added `ExprKind::FStringLit(Vec<(String, Option<Box<Expr>>)>)` — alternating (literal, optional expr) segments
- **Parser:** `parse_fstring_segments()` re-lexes and re-parses each `{...}` hole using a sub-parser
- **Type checker:** Type-checks each interpolated expression; result type is always `Str`
- **HIR:** Desugars into a chain of string `+` and `to_string(expr)` calls
- **Eval (AST):** Evaluates each segment directly, concatenates via `format!("{val}")`
- **MIR executor:** `to_string` added as a builtin free function; functions-as-values fallback added for `Var` lookup

**Runtime behavior:** `f"value = {x}"` is equivalent to `"value = " + to_string(x)`.

**Tests:** `tests/audit_tests/test_phase2_fstring.rs` — 7 tests.

### P2-6: Result/Option Builtin Methods

**New methods added** to `dispatch_method` in `cjc-mir-exec`:

#### Option<T> methods
| Method | Behavior |
|--------|----------|
| `opt.unwrap()` | Returns inner value; panics on `None` |
| `opt.unwrap_or(default)` | Returns inner or default |
| `opt.is_some()` | Returns `Bool` |
| `opt.is_none()` | Returns `Bool` |
| `opt.map(f)` | Applies `f` to inner value if `Some`, propagates `None` |
| `opt.and_then(f)` | FlatMap — `f` must return `Option<U>` |

#### Result<T,E> methods
| Method | Behavior |
|--------|----------|
| `r.unwrap()` | Returns inner value; errors with `Err` contents on `Err` |
| `r.unwrap_or(default)` | Returns inner or default |
| `r.is_ok()` | Returns `Bool` |
| `r.is_err()` | Returns `Bool` |
| `r.map(f)` | Applies `f` to `Ok` value, propagates `Err` |
| `r.and_then(f)` | FlatMap — `f` must return `Result<U,E>` |

**Also added:** String methods (`len`, `is_empty`, `to_upper`, `to_lower`, `trim`, `contains`, `starts_with`, `ends_with`, `split`) and Array methods (`len`, `is_empty`, `first`, `last`) to `dispatch_method`.

**First-class functions:** `MirExprKind::Var` now falls back to function table lookup if the name is not in scope, enabling `opt.map(my_fn)` syntax.

**Tests:** `tests/audit_tests/test_phase2_result_option.rs` — 8 tests.

---

## Regression Gate

All changes were validated against the full test suite with **zero regressions**:

```
Full workspace test run:
  All test results: ok  (0 FAILED)
  Audit tests (test_audit): 127 passed, 0 failed, 0 ignored
```

---

## Known Limitations and Future Work

### P1-3: Vec COW (Future ADR-9)
- Change `Value::Array(Vec<Value>)` to `Value::Array(Rc<Vec<Value>>)` once a migration guide and parity gate are prepared.

### P1-4: Scope Stack (Future ADR-10)
- Profile first; if scope-push dominates, switch to `SmallVec<[HashMap; 8]>`.

### P1-6: Parallel Matmul (Future ADR-11)
- Benchmark first. Add `rayon` for matrices >= 256×256.

### TCO Completeness
- The current TCO only detects **direct tail calls** (`return f(args)` or body-expression `f(args)`).
- Mutual recursion across different functions is trampolined correctly.
- Tail calls through closures, trait methods, or conditional branches are not detected.

### String Interpolation Limitations
- The `f"..."` sub-parser creates a fresh lexer+parser for each `{...}` hole at parse time.
- Complex nested expressions (lambdas, closures) inside holes may fail to parse.
- No support for format specifiers (e.g., `{x:.2f}`).

---

## Error Code Registry (Phase 2 Additions)

| Code | Message | When |
|------|---------|------|
| E0150 | Cannot assign to immutable variable `name` | Assignment to `let` (not `let mut`) binding |
| E0300 | Trait bound not satisfied | Generic call where type arg doesn't implement required trait |
| E0400 | Const expression is not compile-time constant | Non-literal in `const` initializer |
| E0401 | Type mismatch in const initializer | Inferred type ≠ declared type in `const` |
| E0450 | Unexpected `}` in format string | Unmatched `}` in `f"..."` |
| E0451 | Invalid expression in format string | Parse failure in `{...}` hole |
