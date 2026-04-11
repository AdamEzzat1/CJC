> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# Gap Closure: Parallel Matmul, Vec COW, F16, Range/Slice, Fixtures

**Date:** 2026-02-23
**Scope:** B2, B3, C7, C8, A1 (focused gap-closure sprint)

---

## Summary

This change closes all outstanding gaps identified in the honest audit of the
P1 Correctness & Types sprint. Two items were fully unimplemented (B2, B3),
two were partial (C7, C8), and one category of fixtures was missing (A1).

**Regression:** 2080 tests pass, 0 failures.

---

## B2: Parallel Deterministic Matmul (rayon)

### What changed
- **`crates/cjc-runtime/src/tensor.rs`**: Refactored `matmul()` into
  `matmul_sequential()` (always available) and `matmul_parallel_mode_a()`
  (behind `#[cfg(feature = "parallel")]`).
- **`crates/cjc-runtime/Cargo.toml`**: `rayon = "1"` as optional dependency,
  `parallel = ["rayon"]` feature gate.

### Architecture: Mode A
- Parallelizes over output **rows** using `rayon::par_chunks_mut(n)`.
- Each row performs a sequential k-reduction with `KahanAccumulatorF64`.
- Threshold: any dimension >= 256 triggers the parallel path.
- **Determinism guarantee:** Each output element `C[i,j]` is computed by
  exactly one thread with a fixed-order Kahan summation. No cross-thread
  reductions, no floating-point non-determinism.

### Audit tests
- `tests/audit_tests/test_parallel_matmul.rs` (6 tests):
  - Parallel == sequential (bit-identical)
  - 100-run determinism
  - Odd shapes (257x259) * (259x263)
  - Small matrices (sequential path)
  - 1x1 edge case
  - Threshold boundary (255, 256, 257)

---

## B3: Vec COW (Copy-on-Write Arrays and Tuples)

### What changed
- **`crates/cjc-runtime/src/value.rs`**:
  - `Value::Array(Vec<Value>)` -> `Value::Array(Rc<Vec<Value>>)`
  - `Value::Tuple(Vec<Value>)` -> `Value::Tuple(Rc<Vec<Value>>)`
- **`crates/cjc-mir-exec/src/lib.rs`**: ~20 sites updated:
  - Construction: wrap in `Rc::new(...)`
  - Mutation: `Rc::make_mut(&mut arr)[idx] = val`
  - Iteration: `arr.iter()` via Deref
- **`crates/cjc-eval/src/lib.rs`**: ~20 sites updated (same pattern)
- **`tests/test_eval.rs`**: Updated Array construction in tests
- **`tests/test_regression_gate.rs`**: Fixed iterator and clone patterns
- **`tests/test_reference_kernels.rs`**: Fixed iterator and clone patterns

### Semantics
- **O(1) clone:** `let b = a` only bumps the `Rc` refcount.
- **COW on mutation:** `Rc::make_mut()` triggers a deep copy only when
  the array has multiple owners (refcount > 1).
- **Structural equality preserved:** Array comparison uses `Deref` to
  `Vec<Value>`, which works transparently through `Rc`.

---

## C7: F16 End-to-End Type System Integration

### What changed
- **`crates/cjc-types/src/lib.rs`**:
  - Added `Type::F16` variant to the Type enum
  - Registered in `is_numeric()`, `is_float()`, `is_value_type()`, `is_nogc_safe()`
  - Added unification rule: `(Type::F16, Type::F16) => Ok(a.clone())`
  - Added `types_match` rule
  - Registered `"f16"` in type_defs
  - Registered Numeric + Float trait impls
  - Added resolve_type_expr case
  - Added conversion builtins: `f16_to_f64`, `f64_to_f16`, `f16_to_f32`, `f32_to_f16`
- **`crates/cjc-mir-exec/src/lib.rs`**:
  - F16 binary ops (add, sub, mul, div, comparisons)
  - F16 unary neg
  - F16 conversion builtins in dispatch_call + is_known_builtin
  - Bf16 conversion builtins (previously type-system only, now executable)
- **`crates/cjc-eval/src/lib.rs`**: Same additions as mir-exec

### Pre-existing (no change needed)
- `Value::F16(F16)` already existed in value.rs
- `F16.to_f64()` and `F16.to_f32()` method dispatch already existed
- `crates/cjc-runtime/src/f16.rs` complete IEEE 754 binary16 implementation

---

## C8: Range<T> + Slice<T> Type System

### What changed
- **`crates/cjc-types/src/lib.rs`**:
  - Added `Type::Range { elem: Box<Type> }` variant
  - Added `Type::Slice { elem: Box<Type> }` variant
  - Display: `Range<i64>`, `Slice<f64>`
  - Unification rules for both
  - `types_match` rules for both
  - `is_value_type` includes both
  - `resolve_type_expr` handles `Range<T>` and `Slice<T>` syntax

### Pre-existing (no change needed)
- `ForIter::Range { start, end }` in the AST handles `for x in 0..n`
- `Type::ByteSlice` handles byte-level slicing

---

## A1: Fixture Corrections (Regex + Byte-String Categories)

### New fixtures
- **`tests/fixtures/regex/basic_match.cjc`** + `.stdout`:
  Match/not-match operators, flags, digit matching, anchors, alternation
- **`tests/fixtures/regex/regex_variables.cjc`** + `.stdout`:
  Regex in let bindings, if conditions, function parameters, word boundaries
- **`tests/fixtures/bytes/byte_strings.cjc`** + `.stdout`:
  Byte string literals, len, is_empty, get, find_byte, split_byte

### Total fixture count: 13 (was 10)

---

## Files Modified

| File | Change |
|------|--------|
| `crates/cjc-runtime/src/value.rs` | Array/Tuple COW (`Rc<Vec<Value>>`) |
| `crates/cjc-runtime/src/tensor.rs` | Parallel matmul (Mode A) |
| `crates/cjc-runtime/Cargo.toml` | rayon optional dep |
| `crates/cjc-types/src/lib.rs` | Type::F16, Type::Range, Type::Slice |
| `crates/cjc-mir-exec/src/lib.rs` | COW migration + F16 ops + conversions |
| `crates/cjc-eval/src/lib.rs` | COW migration + F16 ops + conversions |
| `tests/test_eval.rs` | COW Array construction fix |
| `tests/test_regression_gate.rs` | COW iterator fixes |
| `tests/test_reference_kernels.rs` | COW iterator fixes |
| `tests/audit_tests/test_parallel_matmul.rs` | NEW: 6 parallel matmul tests |
| `tests/fixtures/regex/*.cjc` | NEW: 2 regex fixtures |
| `tests/fixtures/bytes/*.cjc` | NEW: 1 byte-string fixture |
