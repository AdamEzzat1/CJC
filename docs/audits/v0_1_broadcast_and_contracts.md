# Audit: v0.1 Broadcasting Builtins + Contract Tests

**Date:** 2026-03-03
**Scope:** Broadcasting builtins, contract test suite, documentation updates

---

## Changes Summary

### Phase 1: Broadcasting Builtins (5 source files)

| File | Change |
|------|--------|
| `crates/cjc-runtime/src/tensor.rs` | Added 5 element-wise binary methods: `elem_pow`, `elem_min`, `elem_max`, `elem_atan2`, `elem_hypot` |
| `crates/cjc-runtime/src/builtins.rs` | Added `broadcast` (23 unary fns) and `broadcast2` (9 binary fns) dispatch |
| `crates/cjc-eval/src/lib.rs` | Registered `broadcast`/`broadcast2` in `is_known_builtin` |
| `crates/cjc-mir-exec/src/lib.rs` | Registered `broadcast`/`broadcast2` in `is_known_builtin` |
| `crates/cjc-types/src/effect_registry.rs` | Registered both as ALLOC effect |

### Phase 2: Contract Tests (8 test files, 45 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_v0_1.rs` | entry | Module entry point |
| `tests/v0_1/mod.rs` | index | Module index |
| `tests/v0_1/test_repl_constraints.rs` | 8 | Array parsing, multi-line braces, parse recovery, Map prelude, continuation, meta-commands, state accumulation, eval/MIR parity |
| `tests/v0_1/test_stats_contracts.rs` | 8 | mean() absent as free fn, tensor.mean() works, median, sd, variance, quantile, iqr, eval/MIR parity |
| `tests/v0_1/test_lm_contracts.rs` | 7 | Intercept auto-add, coefficient counts (p=1,p=2), R^2 range, residuals length, arg validation, rank deficiency, eval/MIR parity |
| `tests/v0_1/test_type_system_reality.rs` | 6 | Generics parse, bounded generics, Any polymorphism, type annotations required, deterministic dispatch, struct+Any parity |
| `tests/v0_1/test_broadcast_builtins.rs` | 10 | broadcast sin/sqrt/exp/relu, unknown fn error, broadcast2 add/pow, unknown fn error, eval/MIR parity x2 |
| `tests/v0_1/test_effects_enforcement.rs` | 6 | pure cannot alloc, pure cannot IO, alloc allows tensor, unannotated OK, broadcast ALLOC in registry, NoGC rejects gc_alloc |

### Phase 3: Regression Gate

- **Total tests:** 3,540 passed, 0 failed, 21 ignored
- **New tests:** 45 (all passing)
- **Existing tests:** No regressions

### Phase 4: Documentation Updates (6 files)

| File | Change |
|------|--------|
| `CJC V 0.1/BUILTIN_REFERENCE.md` | Added "Broadcasting Builtins" section (#44) with full function tables |
| `CJC V 0.1/KNOWN_LIMITATIONS.md` | Added: mean() is tensor-only, lm() intercept contract, no broadcast fusion |
| `CJC V 0.1/CLI_AND_REPL_GUIDE.md` | Added: multi-line input details, parse error recovery, prelude symbols |
| `CJC V 0.1/DATA_SCIENCE_GUIDE.md` | Updated: lm() contract with correct 4-arg signature, added broadcasting section |
| `CJC V 0.1/SYNTAX_REFERENCE.md` | Added: broadcasting builtins section (10.1) with syntax examples |
| `docs/audits/v0_1_broadcast_and_contracts.md` | This file |

---

## Determinism Notes

- All tests use seed 42
- `broadcast` and `broadcast2` are deterministic: `tensor.map(f)` applies f in index order
- All parity tests confirm eval and MIR produce identical output
- Broadcasting uses the existing `elementwise_binop` which iterates in row-major order
- No floating-point non-determinism: all operations use f64 with IEEE 754 semantics

## Design Decisions

1. **Explicit builtins over operator syntax**: Chose `broadcast("sin", t)` over `.+` dot-operator syntax. Rationale: no parser changes, clear allocation semantics, easy to extend.

2. **Effect classification as ALLOC (not GC)**: Broadcasting allocates new tensors via `Vec<f64>` (no GC pressure). Consistent with `Tensor.zeros`, `Tensor.ones`, and other tensor constructors.

3. **Dispatch through shared builtins**: Both `broadcast` and `broadcast2` are implemented in `cjc_runtime::builtins::dispatch_builtin`, which is called by both eval and MIR-exec. This ensures parity by construction.

4. **23 unary + 9 binary functions**: Covers all common mathematical operations. Unknown function names produce a clear runtime error.
