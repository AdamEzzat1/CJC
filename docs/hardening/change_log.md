# CJC v0.1 Hardening — Change Log

## Files Created

### Test Infrastructure
- `tests/test_cjc_v0_1_hardening.rs` — Master test entry point
- `tests/cjc_v0_1_hardening/mod.rs` — Module declarations
- `tests/cjc_v0_1_hardening/helpers.rs` — Shared helpers (run_mir, run_eval, assert_parity, etc.)

### Unit Tests (11 files)
- `tests/cjc_v0_1_hardening/unit/mod.rs`
- `tests/cjc_v0_1_hardening/unit/test_lexer_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_parser_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_type_checker_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_runtime_builtins_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_eval_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_mir_exec_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_dispatch_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_data_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_snap_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_regex_hardening.rs`
- `tests/cjc_v0_1_hardening/unit/test_repro_hardening.rs`

### Property Tests (5 files)
- `tests/cjc_v0_1_hardening/prop/mod.rs`
- `tests/cjc_v0_1_hardening/prop/test_lexer_props.rs`
- `tests/cjc_v0_1_hardening/prop/test_eval_props.rs`
- `tests/cjc_v0_1_hardening/prop/test_snap_props.rs`
- `tests/cjc_v0_1_hardening/prop/test_repro_props.rs`
- `tests/cjc_v0_1_hardening/prop/test_dispatch_props.rs`

### Fuzz Tests (1 file)
- `tests/cjc_v0_1_hardening/fuzz/mod.rs`
- `tests/cjc_v0_1_hardening/fuzz/test_fuzz_hardening.rs`

### Integration Tests (3 files)
- `tests/cjc_v0_1_hardening/integration/mod.rs`
- `tests/cjc_v0_1_hardening/integration/test_wiring_parity.rs`
- `tests/cjc_v0_1_hardening/integration/test_wiring_builtins.rs`
- `tests/cjc_v0_1_hardening/integration/test_wiring_hir_mir.rs`

### Determinism Tests (2 files)
- `tests/cjc_v0_1_hardening/determinism/mod.rs`
- `tests/cjc_v0_1_hardening/determinism/test_execution_determinism.rs`
- `tests/cjc_v0_1_hardening/determinism/test_numerical_determinism.rs`

### Documentation (4 files)
- `docs/hardening/final_summary.md`
- `docs/hardening/test_matrix.md`
- `docs/hardening/open_risks.md`
- `docs/hardening/change_log.md`

## Files Modified

None. No existing source code was modified. All changes are additive (new test files and documentation).

## Semantic Changes

None. No language semantics, compiler behavior, or runtime behavior was modified.
