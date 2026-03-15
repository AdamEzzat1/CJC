# Final Phase Hardening Before VM

**Date:** 2026-03-15
**Test Count:** 4,824 passed | 0 failed | 48 ignored

This document covers all features implemented, bugs fixed, and tests added during the final hardening phase prior to VM development.

---

## 1. Module System Visibility Enforcement

**Files changed:** `crates/cjc-module/src/lib.rs`, `crates/cjc-cli/src/main.rs`

### What changed
- `check_visibility()` function added to `cjc-module` that inspects the module dependency graph and reports violations when a module imports a private symbol from another module.
- CLI (`--multi-file` mode) now calls `check_visibility()` before execution and exits with clear error messages if violations are found.
- `merge_programs()` aliases all functions (pub and private) during merging; enforcement is a separate concern handled at the CLI level.

### Visibility rules
- `pub fn`, `pub struct`, `pub record` are accessible from importing modules.
- Functions/structs/records without `pub` default to `Private`.
- In single-file mode, visibility is not enforced (backward compatible).
- In multi-file mode (`--multi-file`), importing a private symbol produces a compile error.

### Usage
```cjc
// math.cjc
pub fn add(a: f64, b: f64) -> f64 { a + b }
fn internal_helper() -> f64 { 0.0 }  // private

// main.cjc
import math
let x = add(1.0, 2.0);  // OK — add is pub
// internal_helper() would trigger a visibility violation in --multi-file mode
```

---

## 2. Cross-File Diagnostics

**Files changed:** `crates/cjc-types/src/lib.rs`

### What changed
- `TypeChecker` now accepts an optional filename via `new_with_filename(filename)`.
- When set, diagnostic messages include the originating filename for multi-file error reporting.
- Module system passes filenames through to the type checker during per-module checking.

---

## 3. Pattern Match Bool Exhaustiveness

**Files changed:** `crates/cjc-types/src/lib.rs`

### What changed
- Extended the existing enum exhaustiveness checker to handle `bool` types.
- `check_bool_exhaustiveness()` verifies that `match` expressions on booleans cover both `true` and `false` (or include a wildcard).
- Integrated into the main `check_match_exhaustiveness()` dispatcher.

### Usage
```cjc
let x = true;
match x {
    true => "yes",
    false => "no",
}
// Omitting one arm without a wildcard produces a warning
```

---

## 4. String Manipulation Builtins

**Files changed:** `crates/cjc-runtime/src/builtins.rs`

### New builtins (12 total)
All are registered in the shared builtin dispatch and work in both `cjc-eval` and `cjc-mir-exec`.

| Builtin | Signature | Description |
|---------|-----------|-------------|
| `str_upper(s)` | `str -> str` | Uppercase |
| `str_lower(s)` | `str -> str` | Lowercase |
| `str_trim(s)` | `str -> str` | Trim whitespace |
| `str_contains(s, needle)` | `(str, str) -> bool` | Substring check |
| `str_replace(s, from, to)` | `(str, str, str) -> str` | Replace **first** occurrence |
| `str_split(s, delim)` | `(str, str) -> [str]` | Split into array |
| `str_join(arr, sep)` | `([str], str) -> str` | Join array with separator |
| `str_starts_with(s, prefix)` | `(str, str) -> bool` | Prefix check |
| `str_ends_with(s, suffix)` | `(str, str) -> bool` | Suffix check |
| `str_repeat(s, n)` | `(str, i64) -> str` | Repeat N times |
| `str_chars(s)` | `str -> [str]` | Split into characters |
| `str_substr(s, start, len)` | `(str, i64, i64) -> str` | Byte-indexed substring |

### Important: `str_replace` semantics
`str_replace` replaces the **first** occurrence only, consistent with the existing tidyverse `str_replace` function. Use `str_replace_all` for global replacement.

---

## 5. AD Gradient Clipping

**Files changed:** `crates/cjc-ad/src/lib.rs`, `crates/cjc-ad/Cargo.toml`

### New methods on `GradGraph`
- `clip_grad(max_norm: f64)` — Element-wise clipping to `[-max_norm, max_norm]`.
- `clip_grad_norm(max_norm: f64) -> f64` — Global L2 norm clipping. Scales all gradients so total norm does not exceed `max_norm`. Returns actual norm before clipping.

### Determinism
Both methods use `KahanAccumulatorF64` from `cjc-repro` for norm computation, ensuring bit-identical results across runs.

---

## 6. Bug Fixes

### 6a. `str_replace` shadowing tidy dispatch (REGRESSION FIX)
The new `str_replace` builtin in `builtins.rs` used Rust's `.replace()` (all occurrences), but the existing tidy dispatch `str_replace` only replaces the first occurrence. Since `dispatch_builtin` runs before `tidy_dispatch`, the new builtin shadowed the old behavior. Fixed by using `.replacen(from, to, 1)`.

### 6b. Module merge skipping private functions (REGRESSION FIX)
The visibility enforcement initially skipped private functions during `merge_programs()`. This broke the `module_exec_two_files` test where non-pub functions are expected to work. Fixed: merge aliases all functions; visibility enforcement is handled separately via `check_visibility()`.

### 6c. Golden hash update
The primitive ABI golden hash was temporarily out of sync during the `str_replace` fix cycle. Verified that the hash returned to the original value after the fix, confirming no ABI drift.

---

## 7. Test Files Added

All new tests are in `tests/final_phase_hardening_before_vm/`:

| File | Tests | What it verifies |
|------|-------|-----------------|
| `test_module_visibility.rs` | 8 | pub/private aliasing, violations, cyclic deps, determinism |
| `test_trait_impl.rs` | 6 | Trait parsing, impl methods in eval & MIR, conformance, parity |
| `test_diagnostics.rs` | 5 | Filename on diagnostics, renderer, type checker |
| `test_exhaustiveness.rs` | 3 | Enum coverage, wildcard, bool exhaustiveness |
| `test_string_builtins.rs` | 16 | All 12 str_* builtins + eval integration + parity |
| `test_ad_clip_grad.rs` | 3 | clip_grad, clip_grad_norm, no-clip-needed |
| `test_parity.rs` | 14 | String ops, if-expr, variadic, defaults, struct methods, f-strings, RNG |

**Total new tests: 55**

---

## 8. Verification Summary

```
Workspace test suite: cargo test --workspace --no-fail-fast
  Passed:  4,824
  Failed:  0
  Ignored: 48
  Time:    ~15 minutes (including CNN/RL benchmarks)
```

All existing tests pass. Zero regressions introduced.

---

## 9. Architecture Invariants Preserved

- **Determinism:** All new code uses `BTreeMap`/`BTreeSet`. Gradient clipping uses `KahanAccumulatorF64`.
- **Dual executor parity:** All string builtins work identically in `cjc-eval` and `cjc-mir-exec` (verified by parity tests).
- **Pipeline integrity:** Lexer -> Parser -> AST -> HIR -> MIR -> Exec chain unmodified.
- **Backward compatibility:** No syntax changes. Visibility enforcement only activates in `--multi-file` mode.
- **No hidden allocations:** All new builtins are stateless functions in the shared dispatch layer.
