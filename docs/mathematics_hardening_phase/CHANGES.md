# Mathematics Hardening Phase: Changes

## Summary

Added 44 new mathematical primitives to CJC's runtime, covering scalar math, mathematical constants, tensor constructors, tensor reductions, and vector/matrix operations. All additions are general-purpose, zero-dependency (Rust stdlib only), and maintain full parity between eval and MIR-exec engines.

## New Scalar Math Builtins (21)

### Trigonometric (7)
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `sin(x)` | Number -> Float | Sine function |
| `cos(x)` | Number -> Float | Cosine function |
| `tan(x)` | Number -> Float | Tangent function |
| `asin(x)` | Number -> Float | Inverse sine (returns NaN if |x| > 1) |
| `acos(x)` | Number -> Float | Inverse cosine (returns NaN if |x| > 1) |
| `atan(x)` | Number -> Float | Inverse tangent |
| `atan2(y, x)` | (Number, Number) -> Float | Two-argument arctangent |

### Hyperbolic (3)
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `sinh(x)` | Number -> Float | Hyperbolic sine |
| `cosh(x)` | Number -> Float | Hyperbolic cosine |
| `tanh_scalar(x)` | Number -> Float | Hyperbolic tangent (scalar; avoids conflict with tensor `tanh_activation`) |

### Exponentiation & Logarithms (4)
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `pow(base, exp)` | (Number, Number) -> Float | General exponentiation |
| `log2(x)` | Number -> Float | Base-2 logarithm |
| `log10(x)` | Number -> Float | Base-10 logarithm |
| `log1p(x)` | Number -> Float | ln(1+x), precise near zero |

### Rounding (2)
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `ceil(x)` | Number -> Float/Int | Ceiling (rounds up) |
| `round(x)` | Number -> Float/Int | Round to nearest (ties round away from zero) |

### Comparison & Sign (3)
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `min(a, b)` | (Number, Number) -> Float | Minimum of two numbers |
| `max(a, b)` | (Number, Number) -> Float | Maximum of two numbers |
| `sign(x)` | Number -> Float | Sign function (-1, 0, or 1) |

### Precision Helpers (2)
| Builtin | Signature | Description |
|---------|-----------|-------------|
| `hypot(x, y)` | (Number, Number) -> Float | sqrt(x^2 + y^2) without overflow |
| `expm1(x)` | Number -> Float | exp(x) - 1, precise near zero |

## Mathematical Constants (5)

| Builtin | Value | Description |
|---------|-------|-------------|
| `PI()` | 3.14159... | Ratio of circumference to diameter |
| `E()` | 2.71828... | Euler's number |
| `TAU()` | 6.28318... | 2*PI |
| `INF()` | +infinity | Positive infinity |
| `NAN_VAL()` | NaN | Not-a-number |

## Tensor Constructors (6)

| Builtin | Signature | Description |
|---------|-----------|-------------|
| `Tensor.linspace(start, end, n)` | (Float, Float, Int) -> Tensor | n evenly-spaced points from start to end |
| `Tensor.arange(start, end, step?)` | (Float, Float, Float?) -> Tensor | Range with step (default step=1.0) |
| `Tensor.eye(n)` | Int -> Tensor[n,n] | n x n identity matrix |
| `Tensor.full(shape, value)` | (Array[Int], Float) -> Tensor | Constant-filled tensor |
| `Tensor.diag(tensor)` | Tensor -> Tensor | 1D -> diagonal matrix, or 2D -> diagonal extraction |
| `Tensor.uniform(shape)` | Array[Int] -> Tensor | Uniform random [0, 1) (seeded RNG) |

## Tensor Reduction Methods (8)

| Method | Signature | Description |
|--------|-----------|-------------|
| `.max()` | Tensor -> Float | Maximum element value |
| `.min()` | Tensor -> Float | Minimum element value |
| `.var()` | Tensor -> Float | Population variance |
| `.std()` | Tensor -> Float | Population standard deviation |
| `.abs()` | Tensor -> Tensor | Element-wise absolute value |
| `.mean_axis(axis)` | (Tensor 2D, Int) -> Tensor | Mean along axis |
| `.max_axis(axis)` | (Tensor 2D, Int) -> Tensor | Max along axis |
| `.min_axis(axis)` | (Tensor 2D, Int) -> Tensor | Min along axis |

## Vector/Matrix Operations (4)

| Builtin | Signature | Description |
|---------|-----------|-------------|
| `dot(a, b)` | (Tensor 1D, Tensor 1D) -> Float | Dot product |
| `outer(a, b)` | (Tensor 1D, Tensor 1D) -> Tensor 2D | Outer product |
| `cross(a, b)` | (Tensor[3], Tensor[3]) -> Tensor[3] | Cross product (3D vectors only) |
| `norm(tensor, ord?)` | (Tensor, Int?) -> Float | Vector norm (default L2; supports L1, Lp) |

## Architecture Improvements

### User-Function Priority Over Builtins
Added a critical improvement: user-defined functions now take priority over builtins with the same name. This prevents naming collisions when users define functions like `outer()`, `min()`, `max()` in their CJC programs. Applied to both eval and MIR-exec engines.

### Three-Layer Wiring
All new builtins follow the established three-layer wiring discipline:
1. **`cjc-runtime/src/builtins.rs`** -- Implementation
2. **`cjc-types/src/effect_registry.rs`** -- Effect classification
3. **`cjc-eval/src/lib.rs`** + **`cjc-mir-exec/src/lib.rs`** -- Dispatch

## Effect Classifications

| Category | Effect | Builtins |
|----------|--------|----------|
| Pure (no side effects) | PURE | sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh_scalar, pow, log2, log10, log1p, expm1, ceil, round, min, max, sign, hypot, PI, E, TAU, INF, NAN_VAL, dot, norm |
| Allocating | ALLOC | outer, cross, Tensor.linspace, Tensor.arange, Tensor.eye, Tensor.full, Tensor.diag |
| Nondeterministic | NONDET+ALLOC | Tensor.uniform |

## Test Coverage

| Suite | Tests | Coverage |
|-------|-------|---------|
| 01: Scalar Trig | 14 | sin, cos, tan, asin, acos, atan, atan2 + Pythagorean identity |
| 02: Scalar Hyperbolic | 8 | sinh, cosh, tanh_scalar + cosh^2 - sinh^2 = 1 identity |
| 03: Scalar Exp/Log | 14 | pow, log2, log10, log1p, expm1 + inverse identities |
| 04: Scalar Rounding | 8 | ceil, round + floor/ceil bracket property |
| 05: Scalar Comparison | 14 | min, max, sign, hypot + overflow safety |
| 06: Constants | 8 | PI, E, TAU, INF, NAN_VAL + identity checks |
| 07: Tensor Constructors | 16 | linspace, arange, eye, full, diag, uniform |
| 08: Tensor Reductions | 12 | max, min, var, std, abs, mean_axis, max_axis, min_axis |
| 09: Vector Ops | 11 | dot, outer, cross, norm + properties |
| 10: Parity | 17 | Eval vs MIR-exec identical output for all categories |
| **Total** | **120** | |

## Regression Status

| Metric | Value |
|--------|-------|
| Total workspace tests passed | 3033 |
| Total workspace tests failed | 0 |
| Total workspace tests ignored | 20 |
| Chess RL tests passed | 66 |
| Chess RL tests failed | 0 |
| Math hardening tests passed | 120 |
| Math hardening tests failed | 0 |

## Files Modified

| File | Changes |
|------|---------|
| `crates/cjc-runtime/src/builtins.rs` | +300 lines: all new builtin implementations |
| `crates/cjc-types/src/effect_registry.rs` | +20 lines: effect registration for all new builtins |
| `crates/cjc-eval/src/lib.rs` | +80 lines: is_known_builtin + Tensor.uniform + tensor methods + user-fn priority |
| `crates/cjc-mir-exec/src/lib.rs` | +80 lines: is_known_builtin + Tensor.uniform + tensor methods + user-fn priority |
| `Cargo.toml` | +4 lines: test binary registration |

## Files Created

| File | Purpose |
|------|---------|
| `docs/mathematics_hardening_phase/STACK_ROLE_GROUP.md` | Stack role group prompt |
| `docs/mathematics_hardening_phase/CHANGES.md` | This file |
| `tests/test_math_hardening.rs` | Test harness |
| `tests/mathematics_hardening_phase/mod.rs` | Module declarations |
| `tests/mathematics_hardening_phase/helpers.rs` | Shared test helpers |
| `tests/mathematics_hardening_phase/test_01_scalar_trig.rs` | Trigonometric tests |
| `tests/mathematics_hardening_phase/test_02_scalar_hyperbolic.rs` | Hyperbolic tests |
| `tests/mathematics_hardening_phase/test_03_scalar_exp_log.rs` | Exp/log tests |
| `tests/mathematics_hardening_phase/test_04_scalar_rounding.rs` | Rounding tests |
| `tests/mathematics_hardening_phase/test_05_scalar_comparison.rs` | Comparison tests |
| `tests/mathematics_hardening_phase/test_06_constants.rs` | Constants tests |
| `tests/mathematics_hardening_phase/test_07_tensor_constructors.rs` | Constructor tests |
| `tests/mathematics_hardening_phase/test_08_tensor_reductions.rs` | Reduction tests |
| `tests/mathematics_hardening_phase/test_09_vector_ops.rs` | Vector ops tests |
| `tests/mathematics_hardening_phase/test_10_parity.rs` | Parity verification tests |
