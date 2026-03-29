# CJC v0.2 Beta — Phase 1 Changelog

**Date**: 2026-03-29
**Constraint**: Determinism is the main contract — same seed = bit-identical output.

## Summary

Phase 1 of the v0.2 beta focused on hardening the compiler pipeline,
eliminating production panics, improving error diagnostics, and adding
test infrastructure for property-based and fuzz testing.

---

## Changes by Category

### 1. Production Panic Elimination (Phase 1A — from prior session)

- **cjc-eval**: Replaced 14 GradGraph `.unwrap()` calls with `ok_or_else` error returns
- **cjc-mir-exec**: Replaced 16 GradGraph `.unwrap()` calls with `ok_or_else` error returns
- **Both executors**: Replaced 4 `unreachable!()` with proper error returns
- **cjc-runtime/builtins**: Replaced window function `unreachable!()` with error return
- **cjc-runtime/value**: Improved Enum Display to include `enum_name`

### 2. Type System Improvements (Phase 1B — from prior session)

- **Generic type instantiation**: Added `substitute_type_params_concrete` and
  `collect_unresolved_bindings` for proper generic parameter substitution
  (e.g., `Some(42)` correctly infers `Option<i64>`)
- **Record immutability**: Closed CompoundAssign gap — records now correctly
  reject compound assignment via error code E0160
- **Nested assignment targets**: Full LValue chain resolution via `LvalueStep`
  enum and `walk_lvalue_path` helper methods

### 3. Non-Greedy Regex Quantifiers (New)

**File**: `crates/cjc-regex/src/lib.rs`

- Added `has_lazy` flag to NFA struct to track lazy quantifiers
- Parser now recognizes `*?`, `+?`, `??` syntax
- NFA simulation returns shortest match (first Accept) when `has_lazy` is true,
  instead of longest match (last Accept)
- All 35 regex tests pass including 7 new lazy quantifier tests

### 4. Effect Annotation Validation (New)

**File**: `crates/cjc-types/src/lib.rs`

- `effects_from_names` now returns `(EffectSet, Vec<String>)` — unknown effect
  names produce error messages instead of being silently ignored
- Levenshtein distance suggestion: `unknown effect 'prue', did you mean 'pure'?`
- All 3 call sites updated to emit diagnostics with error code E0410
- Valid effects: `pure`, `io`, `alloc`, `gc`, `nondet`, `mutates`, `arena_ok`, `captures`

### 5. Tensor Error Handling (New)

**File**: `crates/cjc-runtime/src/tensor.rs`

- Replaced `unwrap_or(0.0)` in broadcast binop with proper `ok_or_else` error
  returns, producing descriptive "index out of bounds" messages instead of
  silently using 0.0

### 6. NoGC False Positive Reduction (New)

**File**: `crates/cjc-mir/src/nogc_verify.rs`

- Added `is_known_safe_method` helper that checks method names against the
  effect registry across all type prefixes
- Method calls on computed objects (e.g., `some_fn().len()`) are no longer
  conservatively marked as indirect calls when the method name is a known
  safe builtin
- Unknown methods on computed objects remain conservatively rejected

### 7. Error Cascade Prevention (New)

**File**: `crates/cjc-diag/src/lib.rs`

- Added `error_limit: usize` field to `DiagnosticBag` (default: 50)
- `emit()` now suppresses error-severity diagnostics beyond the limit
- Prevents error storms where a single root cause produces hundreds of
  cascading diagnostics

### 8. Verified Pre-Existing Features

The following features from the v0.2 plan were already implemented:

- **Full outer join**: `TidyView::full_join` with BTreeMap-based deterministic
  key lookup and proper null-filling for unmatched rows
- **Higher-order AD**: `GradGraph::hessian`, `hessian_diag`, `double_backward`
  all implemented with deterministic tape traversal
- **Window functions**: 15+ window functions in builtins (`row_number`, `rank`,
  `dense_rank`, `lag`, `lead`, `cumsum`, `cummean`, `rolling_mean`, etc.)
- **Range/Slice types**: First-class type variants with proper unification,
  substitution, and display

---

## Test Infrastructure

### Property-Based Tests (`tests/beta_tests/prop/`)

| Test | Description |
|------|-------------|
| `prop_eval_determinism_arithmetic` | 10-run eval determinism for float arithmetic |
| `prop_eval_determinism_integer_ops` | 10-run eval determinism for integer ops |
| `prop_eval_determinism_nested_calls` | 10-run eval determinism for nested function calls |
| `prop_parity_simple_arithmetic` | Eval vs MIR-exec parity on arithmetic |
| `prop_parity_conditionals` | Eval vs MIR-exec parity on if/else |
| `prop_parity_functions` | Eval vs MIR-exec parity on function calls |
| `prop_parity_loops` | Eval vs MIR-exec parity on while loops |
| `prop_parity_recursion` | Eval vs MIR-exec parity on recursive functions |
| `prop_mir_exec_determinism_10_runs` | 10-run MIR-exec determinism for recursion |
| `prop_kahan_summation_stability` | Kahan accumulator order-independence |
| `prop_btreemap_deterministic_iteration` | BTreeMap iteration always sorted |
| `prop_splitmix64_deterministic` | SplitMix64 same-seed determinism |

### Fuzz Tests (`tests/beta_tests/fuzz/`)

| Test | Description |
|------|-------------|
| `fuzz_lexer_random_ascii` | 500 random ASCII inputs — lexer must not panic |
| `fuzz_lexer_random_bytes` | 500 random byte inputs — documents char boundary bug |
| `fuzz_parser_random_source` | 500 random sources — parser must not panic |
| `fuzz_parser_malformed_programs` | 30 malformed CJC fragments — parser must not panic |
| `fuzz_typechecker_random_valid_programs` | 6 valid programs — type checker must not panic |
| `fuzz_eval_doesnt_panic` | 5 edge-case programs (div/0, NaN, empty) — eval must not panic |
| `fuzz_regex_random_patterns` | 30+ regex patterns including invalid — engine must not panic |
| `fuzz_regex_random_ascii_patterns` | 200 random pattern+haystack pairs — must not panic |
| `fuzz_mir_exec_doesnt_panic` | 5 valid programs — MIR-exec must not panic |

### Test Results

- **Beta tests**: 21 passed, 0 failed
- **Workspace regression**: Pending (full `cargo test --workspace` run)

---

## Known Issues Discovered

1. **Lexer char boundary panic**: The lexer panics when `String::from_utf8_lossy`
   produces replacement characters (`\uFFFD`) that span multiple bytes. The lexer
   tries to index at byte offsets that fall inside multi-byte chars. Tracked by
   `fuzz_lexer_random_bytes` test.

---

## Determinism Audit

All changes in this phase maintain the determinism contract:

- No `HashMap`/`HashSet` introduced (BTreeMap/BTreeSet only)
- No floating-point reductions without Kahan summation
- No non-deterministic iteration ordering
- All RNG usage via SplitMix64 with explicit seed
- Both executors (eval + MIR-exec) tested for parity

---

## Files Modified

| File | Changes |
|------|---------|
| `crates/cjc-regex/src/lib.rs` | Non-greedy quantifiers + lazy NFA matching |
| `crates/cjc-types/src/lib.rs` | Effect validation + generic substitution |
| `crates/cjc-runtime/src/tensor.rs` | Broadcast binop error handling |
| `crates/cjc-mir/src/nogc_verify.rs` | Safe method recognition |
| `crates/cjc-diag/src/lib.rs` | Error limit |
| `tests/beta_tests/prop/test_prop_determinism.rs` | 12 property-based tests |
| `tests/beta_tests/fuzz/test_fuzz_parser.rs` | 9 fuzz tests |
| `tests/test_beta_tests.rs` | Test harness entry point |
