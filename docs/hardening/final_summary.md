# CJC v0.1 Hardening Audit — Final Summary

## Overview

Comprehensive hardening pass across the CJC compiler repository (20 crates, ~75K LOC).
All new tests live under `tests/cjc_v0_1_hardening/`.

## Test Counts

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Total passing | 2,976 | 3,287 | +311 |
| Total failing | 0 | 0 | 0 |
| Total ignored | 26 | 27 | +1 |

## New Test Breakdown (311 tests)

| Category | File | Count |
|----------|------|-------|
| Unit: Lexer | unit/test_lexer_hardening.rs | 15 |
| Unit: Parser | unit/test_parser_hardening.rs | 27 |
| Unit: Type Checker | unit/test_type_checker_hardening.rs | 13 |
| Unit: Runtime Builtins | unit/test_runtime_builtins_hardening.rs | ~25 |
| Unit: Eval | unit/test_eval_hardening.rs | 22 |
| Unit: MIR-exec | unit/test_mir_exec_hardening.rs | 24 |
| Unit: Dispatch | unit/test_dispatch_hardening.rs | 11 |
| Unit: Data (DataFrame) | unit/test_data_hardening.rs | 8 |
| Unit: Snap | unit/test_snap_hardening.rs | 8 |
| Unit: Regex | unit/test_regex_hardening.rs | 15 |
| Unit: Repro (RNG/Kahan) | unit/test_repro_hardening.rs | 11 |
| Prop: Lexer | prop/test_lexer_props.rs | 4 |
| Prop: Eval | prop/test_eval_props.rs | 4 |
| Prop: Snap | prop/test_snap_props.rs | 6 |
| Prop: Repro | prop/test_repro_props.rs | 5 |
| Prop: Dispatch | prop/test_dispatch_props.rs | 6 |
| Fuzz (Bolero) | fuzz/test_fuzz_hardening.rs | 8 |
| Integration: Parity | integration/test_wiring_parity.rs | 22 |
| Integration: Builtins | integration/test_wiring_builtins.rs | 15 |
| Integration: HIR/MIR | integration/test_wiring_hir_mir.rs | 16 |
| Determinism: Execution | determinism/test_execution_determinism.rs | 14 |
| Determinism: Numerical | determinism/test_numerical_determinism.rs | 14 |

## Ignored Tests (27 total)

All intentionally gated — no regressions or bugs:

| Group | Count | Reason |
|-------|-------|--------|
| Vizor snapshot generators | 11 | Produce reference SVG/BMP files, opt-in only |
| Chess RL debug tests | 3 | Known parser limitation: `};` after `while` in fn body |
| Perf benchmarks (H12) | 6 | Slow benchmarks, release-only |
| Memory model perf | 1 | Slow benchmark |
| Forcats capacity tests | 4 | Slow (65K+ unique strings) |
| Tidy speed gate | 1 | Perf gate, release-only |
| Perf tidy fixtures | 1 | Perf benchmark |

## Key Findings

### API Corrections Discovered During Testing
- Tensor `sum()`/`mean()` use method syntax: `t.sum()`, `t.mean()` — NOT `sum(t)`
- `Value::String` wraps `Rc<String>`, not bare `String`
- `KahanAccumulatorF64` uses `finalize()` not `sum()`
- `cjc-regex` API: `is_match(pattern, flags, haystack: &[u8])` — 3 args
- `Token` uses `text` field, not `lexeme`
- `DataFrame::new()` takes 0 args; use `from_columns()` for construction
- `cjc_hir::AstLowering::new().lower_program(&program)` for HIR lowering
- `cjc_runtime::dispatch` is for reduction strategies, NOT general operator dispatch

### Wiring Gap (Documented, Not Fixed)
- Runtime: ~230 builtin functions
- Eval whitelist: ~150 functions
- MIR-exec whitelist: ~140 functions
- Gap exists but is non-blocking for current use cases

### Parity Gaps (Documented)
- Closures stored in variables: MIR-exec supports calling from source, eval does not
- This is a known limitation of the AST tree-walk interpreter

### Snap Decoder Safety
- `snap_decode` trusts length prefixes — random bytes can trigger OOM
- Fuzz targets cap input size or use roundtrip strategy to avoid this

## Files Created

All under `tests/cjc_v0_1_hardening/`:
```
helpers.rs
mod.rs
unit/mod.rs
unit/test_lexer_hardening.rs
unit/test_parser_hardening.rs
unit/test_type_checker_hardening.rs
unit/test_runtime_builtins_hardening.rs
unit/test_eval_hardening.rs
unit/test_mir_exec_hardening.rs
unit/test_dispatch_hardening.rs
unit/test_data_hardening.rs
unit/test_snap_hardening.rs
unit/test_regex_hardening.rs
unit/test_repro_hardening.rs
prop/mod.rs
prop/test_lexer_props.rs
prop/test_eval_props.rs
prop/test_snap_props.rs
prop/test_repro_props.rs
prop/test_dispatch_props.rs
fuzz/mod.rs
fuzz/test_fuzz_hardening.rs
integration/mod.rs
integration/test_wiring_parity.rs
integration/test_wiring_builtins.rs
integration/test_wiring_hir_mir.rs
determinism/mod.rs
determinism/test_execution_determinism.rs
determinism/test_numerical_determinism.rs
```

Entry point: `tests/test_cjc_v0_1_hardening.rs`

## Open Risks

1. **Snap decoder OOM**: `snap_decode` does not validate length prefixes against available input — untrusted input can trigger massive allocations
2. **Eval closure parity**: Eval cannot call closures stored in variables from source-parsed programs
3. **Builtin wiring gap**: ~80-90 runtime builtins not wired into one or both executors
4. **HashMap usage**: Found in 9+ files — potential determinism risk if iteration order matters
