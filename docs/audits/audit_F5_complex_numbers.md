# Audit F5 — Complex Numbers

**Feature:** ComplexF64 runtime type with fixed-sequence arithmetic
**Status:** Complete
**Date:** 2026-02-23

---

## 1. What Shipped

### 1.1 ComplexF64 Runtime (`crates/cjc-runtime/src/complex.rs`)

- `ComplexF64` struct with `re: f64, im: f64` fields
- Constants: `ZERO`, `ONE`, `I`
- Constructors: `new(re, im)`, `real(re)`, `imag(im)`
- Arithmetic (fixed-sequence): `add`, `sub`, `mul_fixed`, `div_fixed`, `neg`, `scale`
- Queries: `norm_sq`, `abs`, `conj`, `is_nan`, `is_finite`
- BLAS operations: `complex_dot`, `complex_sum`, `complex_matmul` (all via BinnedAccumulator)
- `Display` implementation: `3+4i` / `3-4i` format

### 1.2 Executor Integration

Both `cjc-eval` (AST interpreter) and `cjc-mir-exec` (MIR executor) support:

- **Constructor:** `Complex(re, im)` — creates ComplexF64 from two f64 args
- **Binary ops:** `+`, `-`, `*`, `/`, `==`, `!=` on Complex values
- **Unary ops:** `-z` (negation)
- **Methods:** `.re()`, `.im()`, `.abs()`, `.conj()`, `.norm_sq()`, `.neg()`, `.scale(s)`, `.is_nan()`, `.is_finite()`, `.add(w)`, `.sub(w)`, `.mul(w)`, `.div(w)`

### 1.3 Value Integration

Complex numbers are stored as `Value::Complex(ComplexF64)` in the runtime value system, enabling:
- Let binding and variable assignment
- Function argument passing
- Return values
- Tuple/container storage
- Print output

---

## 2. Determinism Contract

### Fixed-Sequence Multiplication

```
t1 = a.re * b.re   (mul #1)
t2 = a.im * b.im   (mul #2)
t3 = a.re * b.im   (mul #3)
t4 = a.im * b.re   (mul #4)
re = t1 - t2        (sub #1)
im = t3 + t4        (add #1)
```

All intermediates are stored in local variables, preventing LLVM from fusing operations into FMA instructions. This ensures bit-identical results across x86 and ARM platforms.

### Fixed-Sequence Division

```
cc    = rhs.re * rhs.re   (mul #1)
dd    = rhs.im * rhs.im   (mul #2)
denom = cc + dd            (add #1)
ac    = self.re * rhs.re   (mul #3)
bd    = self.im * rhs.im   (mul #4)
re    = (ac + bd) / denom  (add #2, div #1)
bc    = self.im * rhs.re   (mul #5)
ad    = self.re * rhs.im   (mul #6)
im    = (bc - ad) / denom  (sub #1, div #2)
```

Division by `0+0i` produces NaN/Inf stably (no panic).

### BLAS Determinism

Complex reductions (`complex_dot`, `complex_sum`, `complex_matmul`) feed real and imaginary parts separately into `BinnedAccumulatorF64`, providing order-invariant deterministic summation.

---

## 3. Testing Documentation

### 3.1 Audit Tests (19 tests)

**File:** `tests/audit_tests/test_complex_f64_runtime.rs`

| Section | Tests | Coverage |
|---------|-------|----------|
| Correctness | 4 | add/sub, mul_exact, div_nontrivial, conj/norm_sq/abs |
| Determinism | 3 | mul 100x bit-identical, div 100x bit-identical, pipeline 100x |
| Edge Cases | 3 | div-by-zero no panic, NaN propagation (7 ops), large magnitude |
| Execution Semantics | 9 | let binding, binary ops pipeline, methods pipeline, unary neg, function passthrough, equality, eval-vs-mir parity |

### 3.2 Property Tests (17 tests)

**File:** `tests/prop_tests/complex_props.rs`

| Category | Tests | Cases | Coverage |
|----------|-------|-------|----------|
| Algebraic Identities | 10 | 200 each | add/mul commutativity, add/mul identity, mul-zero, additive inverse, conj involution, conj distributes over add, norm_sq=z*conj(z), abs>=0, sub roundtrip, div roundtrip |
| Determinism | 2 | 100 each | mul bit-identical, div bit-identical |
| Special Values | 3 | 100 each | all ops no-panic on NaN/Inf/0, NaN propagation, Display no-panic |
| **Total** | **17** | | Finite + special-value domains |

### 3.3 Fixture Tests (1 fixture, 14 assertions)

**File:** `tests/fixtures/complex/complex_basic.cjc`

Covers: construction, display, re/im accessors, +/-/*/÷, negation, abs, norm_sq, conj, equality/inequality. Golden output verified against MIR-exec pipeline with seed=42.

### 3.4 Inline Unit Tests (20 tests)

**File:** `crates/cjc-runtime/src/complex.rs` (module `tests`)

Covers: mul basic, mul commutative, mul identity, i^2=-1, conj, abs, dot basic, dot deterministic, sum deterministic, sum near-order-invariant, sum merge-order-invariant, matmul identity, matmul deterministic, div basic, div nontrivial, div by zero, div roundtrip, signed zero, NaN propagation, display.

### 3.5 Fuzz Targets (3 targets)

**Directory:** `fuzz/`

| Target | File | Corpus Seeds | Description |
|--------|------|-------------|-------------|
| `parser` | `fuzz/fuzz_targets/parser.rs` | 6 seeds | Arbitrary bytes → parse; must not panic |
| `lexer` | `fuzz/fuzz_targets/lexer.rs` | 5 seeds | Arbitrary bytes → lex; must not panic |
| `complex_eval` | `fuzz/fuzz_targets/complex_eval.rs` | 5 seeds | Parsed programs → MIR-exec; must not panic |

**Build status:** All 3 targets compile with `cargo +nightly fuzz build`.
**Smoke check:** Not runnable on Windows MSVC (libFuzzer requires Linux/macOS). Targets are CI-ready for Linux environments.

### 3.6 Regression Summary

```
cargo test --workspace
Total: 2118 passed, 0 failed, 9 ignored
```

---

## 4. Known Limitations

1. **No `Type::Complex` in the type system.** Complex is a runtime-only type; the type checker does not know about it. Function parameters typed as `Complex` will fail type checking. Workaround: use dynamic dispatch or untyped patterns.

2. **No mixed-type arithmetic.** `Complex + f64` is not supported; users must construct `Complex(x, 0.0)` explicitly.

3. **No complex literals.** There is no `3+4i` literal syntax; users must call `Complex(3.0, 4.0)`.

4. **Division overflow.** Division of large-magnitude complex numbers may overflow to Inf even when the mathematical result is finite (naive algorithm, not Smith's method).

5. **Fuzz testing on Windows.** libFuzzer-based fuzzing requires Linux/macOS. The fuzz infrastructure is ready but cannot be smoke-tested on Windows MSVC.

---

## 5. File Index

| File | Role |
|------|------|
| `crates/cjc-runtime/src/complex.rs` | ComplexF64 struct + fixed-sequence ops + BLAS |
| `crates/cjc-mir-exec/src/lib.rs` | MIR executor: Complex constructor, binops, methods |
| `crates/cjc-eval/src/lib.rs` | AST evaluator: Complex constructor, binops, methods |
| `tests/audit_tests/test_complex_f64_runtime.rs` | 19 audit tests (correctness, determinism, edge, semantics) |
| `tests/prop_tests/complex_props.rs` | 17 property tests (algebraic, determinism, special values) |
| `tests/fixtures/complex/complex_basic.cjc` | Fixture: Complex operations end-to-end |
| `tests/fixtures/complex/complex_basic.stdout` | Golden output for fixture |
| `fuzz/Cargo.toml` | Fuzz crate manifest |
| `fuzz/fuzz_targets/parser.rs` | Fuzz target: parser no-panic |
| `fuzz/fuzz_targets/lexer.rs` | Fuzz target: lexer no-panic |
| `fuzz/fuzz_targets/complex_eval.rs` | Fuzz target: eval no-panic |
| `fuzz/corpus/parser/*.cjc` | 6 parser corpus seeds |
| `fuzz/corpus/lexer/*.txt` | 5 lexer corpus seeds |
| `fuzz/corpus/complex_eval/*.cjc` | 5 complex eval corpus seeds |
| `docs/audits/audit_F5_complex_numbers.md` | This document |
