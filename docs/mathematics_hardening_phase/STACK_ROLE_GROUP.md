# Mathematics Hardening Phase — Stack Role Group Prompt

## Mission Statement

Harden CJC's foundational mathematical primitives so that stats, linear algebra, and machine learning libraries built on top of them are as fast, accurate, and ergonomic as possible. Every addition must be general-purpose, zero-dependency, and pass strict regression gates.

---

## Stacked Roles

### Role 1: Numerical Computing Architect
**Responsibility:** Design the mathematical primitive API surface.
- Every new builtin must accept both `Int` and `Float` inputs (promote Int→f64)
- Return type is always `Float` (f64) for transcendental/rounding functions
- Naming follows established conventions: lowercase, no prefix (e.g., `sin`, `cos`, `pow`)
- Constants follow UPPER_SNAKE convention when accessed as builtins (e.g., `PI`, `E`, `INF`, `NAN`, `TAU`)
- Tensor constructors follow `Tensor.name` convention (e.g., `Tensor.linspace`, `Tensor.eye`)

### Role 2: Precision Guardian
**Responsibility:** Ensure numerical accuracy.
- All implementations delegate to Rust's `f64` methods (IEEE 754 compliant)
- No approximations where exact computation exists
- `hypot(x,y)` must use `f64::hypot` (overflow-safe)
- `log1p(x)` and `expm1(x)` must use their dedicated methods (precision near zero)
- Verify edge cases: NaN propagation, infinity handling, domain errors
- Each function must have at least 3 unit tests including edge cases

### Role 3: Three-Layer Wiring Engineer
**Responsibility:** Maintain CJC's wiring discipline.
Every new builtin must be wired in exactly 3 places:
1. **`cjc-runtime/src/builtins.rs`** — Core implementation in `dispatch_builtin()`
2. **`cjc-types/src/effect_registry.rs`** — Effect classification (pure/alloc/nondet)
3. **`cjc-eval/src/lib.rs`** AND **`cjc-mir-exec/src/lib.rs`** — Add to `is_known_builtin()` in BOTH engines

Stateful builtins (using RNG) must also be wired in `dispatch_call()` in both engines.
Tensor methods must be wired in `dispatch_method()` in both engines.

### Role 4: Regression Gatekeeper
**Responsibility:** Zero regressions, complete test coverage.
- All 2186+ existing tests must continue to pass
- All 49 chess RL tests must continue to pass
- New test suite: `tests/mathematics_hardening_phase/`
- Each new builtin gets ≥3 tests: basic correctness, edge cases, type promotion
- Tensor constructors get shape validation tests
- Determinism tests: same inputs → identical outputs across eval and MIR-exec

### Role 5: Documentation Steward
**Responsibility:** Document every change.
- Update `docs/mathematics_hardening_phase/` with CHANGES.md
- Each new builtin documented with: signature, description, edge cases, examples
- Effect classification documented for each builtin
- MEMORY.md updated with new phase information

---

## Scope: What Gets Added

### Sprint 1: Core Scalar Math (21 builtins)

**Trigonometric (7):**
| Builtin | Signature | Rust Method | Effect |
|---------|-----------|-------------|--------|
| `sin` | `(Number) -> Float` | `f64::sin()` | PURE |
| `cos` | `(Number) -> Float` | `f64::cos()` | PURE |
| `tan` | `(Number) -> Float` | `f64::tan()` | PURE |
| `asin` | `(Number) -> Float` | `f64::asin()` | PURE |
| `acos` | `(Number) -> Float` | `f64::acos()` | PURE |
| `atan` | `(Number) -> Float` | `f64::atan()` | PURE |
| `atan2` | `(Number, Number) -> Float` | `f64::atan2()` | PURE |

**Hyperbolic (3):**
| Builtin | Signature | Rust Method | Effect |
|---------|-----------|-------------|--------|
| `sinh` | `(Number) -> Float` | `f64::sinh()` | PURE |
| `cosh` | `(Number) -> Float` | `f64::cosh()` | PURE |
| `tanh` | `(Number) -> Float` | `f64::tanh()` | PURE |

**Exponentiation & Logarithms (4):**
| Builtin | Signature | Rust Method | Effect |
|---------|-----------|-------------|--------|
| `pow` | `(Number, Number) -> Float` | `f64::powf()` | PURE |
| `log2` | `(Number) -> Float` | `f64::log2()` | PURE |
| `log10` | `(Number) -> Float` | `f64::log10()` | PURE |
| `log1p` | `(Number) -> Float` | `f64::ln_1p()` | PURE |

**Rounding (2):**
| Builtin | Signature | Rust Method | Effect |
|---------|-----------|-------------|--------|
| `ceil` | `(Number) -> Float` | `f64::ceil()` | PURE |
| `round` | `(Number) -> Float` | `f64::round()` | PURE |

**Comparison & Sign (3):**
| Builtin | Signature | Rust Method | Effect |
|---------|-----------|-------------|--------|
| `min` | `(Number, Number) -> Float` | `f64::min()` | PURE |
| `max` | `(Number, Number) -> Float` | `f64::max()` | PURE |
| `sign` | `(Number) -> Float` | `f64::signum()` | PURE |

**Precision Helpers (2):**
| Builtin | Signature | Rust Method | Effect |
|---------|-----------|-------------|--------|
| `hypot` | `(Number, Number) -> Float` | `f64::hypot()` | PURE |
| `expm1` | `(Number) -> Float` | `f64::exp_m1()` | PURE |

### Sprint 2: Mathematical Constants (5 builtins)

| Builtin | Value | Effect |
|---------|-------|--------|
| `PI` | `std::f64::consts::PI` | PURE |
| `E` | `std::f64::consts::E` | PURE |
| `TAU` | `std::f64::consts::TAU` | PURE |
| `INF` | `f64::INFINITY` | PURE |
| `NAN` | `f64::NAN` | PURE |

### Sprint 3: Tensor Constructors (6 builtins)

| Builtin | Signature | Effect |
|---------|-----------|--------|
| `Tensor.linspace` | `(start: Float, end: Float, n: Int) -> Tensor` | ALLOC |
| `Tensor.arange` | `(start: Float, end: Float, step: Float) -> Tensor` | ALLOC |
| `Tensor.eye` | `(n: Int) -> Tensor` | ALLOC |
| `Tensor.full` | `(shape: Array[Int], value: Float) -> Tensor` | ALLOC |
| `Tensor.diag` | `(Tensor 1D) -> Tensor 2D` or `(Tensor 2D) -> Tensor 1D` | ALLOC |
| `Tensor.uniform` | `(shape: Array[Int]) -> Tensor` | NONDET+ALLOC |

### Sprint 4: Tensor Reductions & Operations (8 methods)

| Method | Signature | Effect |
|--------|-----------|--------|
| `.max()` | `Tensor -> Float` | PURE |
| `.min()` | `Tensor -> Float` | PURE |
| `.var()` | `Tensor -> Float` | PURE |
| `.std()` | `Tensor -> Float` | PURE |
| `.abs()` | `Tensor -> Tensor` | ALLOC |
| `.mean_axis(axis)` | `Tensor -> Tensor` | ALLOC |
| `.max_axis(axis)` | `Tensor -> Tensor` | ALLOC |
| `.min_axis(axis)` | `Tensor -> Tensor` | ALLOC |

### Sprint 5: Vector/Matrix Operations (4 builtins)

| Builtin | Signature | Effect |
|---------|-----------|--------|
| `dot` | `(Tensor 1D, Tensor 1D) -> Float` | PURE |
| `outer` | `(Tensor 1D, Tensor 1D) -> Tensor 2D` | ALLOC |
| `cross` | `(Tensor 1D[3], Tensor 1D[3]) -> Tensor 1D[3]` | ALLOC |
| `norm` | `(Tensor, ord?: Int) -> Float` | PURE |

### Sprint 6: Parity Fixes

- Ensure `sort`, `sqrt`, `floor`, `abs`, `int`, `float`, `isnan`, `isinf` are in MIR-exec `is_known_builtin()`
- Ensure Tensor+Int Add/Sub operations work identically in both engines
- Add `to_string` to eval's `is_known_builtin()`

---

## Constraints

1. **Zero external dependencies** — All math uses Rust stdlib (`f64` methods, `std::f64::consts`)
2. **Zero regressions** — `cargo test --workspace` must pass with 0 failures
3. **General-purpose naming** — No domain-specific names; `sin` not `trig_sine`
4. **Three-layer wiring** — Every builtin in all 3 layers or it doesn't ship
5. **Both engines** — Every builtin works in eval AND MIR-exec identically
6. **Deterministic** — All pure builtins are deterministic; same input → same output
7. **IEEE 754 compliant** — All floating-point behavior follows the standard

---

## Test Structure

```
tests/mathematics_hardening_phase/
  mod.rs                              -- Module declarations
  helpers.rs                          -- Shared test helpers (run_mir, parse helpers)
  test_01_scalar_trig.rs              -- Trigonometric functions (sin, cos, tan, asin, acos, atan, atan2)
  test_02_scalar_hyperbolic.rs        -- Hyperbolic functions (sinh, cosh, tanh)
  test_03_scalar_exp_log.rs           -- Exponentiation & logarithms (pow, log2, log10, log1p, expm1)
  test_04_scalar_rounding.rs          -- Rounding functions (ceil, round)
  test_05_scalar_comparison.rs        -- Comparison & sign (min, max, sign, hypot)
  test_06_constants.rs                -- Mathematical constants (PI, E, TAU, INF, NAN)
  test_07_tensor_constructors.rs      -- Tensor constructors (linspace, arange, eye, full, diag, uniform)
  test_08_tensor_reductions.rs        -- Tensor reductions (max, min, var, std, abs, axis reductions)
  test_09_vector_ops.rs               -- Vector operations (dot, outer, cross, norm)
  test_10_parity.rs                   -- Eval vs MIR-exec parity verification
```

---

## Success Criteria

- [ ] All 21 scalar math builtins implemented and tested
- [ ] All 5 mathematical constants accessible
- [ ] All 6 tensor constructors implemented and tested
- [ ] All 8 tensor reduction methods implemented and tested
- [ ] All 4 vector operations implemented and tested
- [ ] Parity verified between eval and MIR-exec for all new additions
- [ ] 0 regressions across entire workspace
- [ ] Documentation complete in `docs/mathematics_hardening_phase/`
